"""
Flood Risk Prediction API v3
=============================
POST /predict        -> single location
POST /predict/batch  -> up to 20 locations concurrently
GET  /health         -> service status
GET  /debug/wris     -> raw WRIS probe for debugging
GET  /docs           -> Swagger UI

Pipeline:
  1. fetch_all()   WRIS -> Open-Meteo -> climate normals
  2. predict()     ML with proper scaler from pkl
  3. validate()    completeness-based confidence cap + physical rules
  4. insights()    Groq with data quality context
"""

import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
)
from app.services.data_fetcher import DataFetcher
from app.services.predictor import FloodPredictor
from app.services.validator import validate
from app.services.groq_insights import GroqInsights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

predictor    = FloodPredictor()
data_fetcher = DataFetcher()
groq_service = GroqInsights()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Flood Risk Prediction API v3...")
    logger.info("Model will load on first prediction (lazy loading)")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Flood Risk Prediction API",
    description=(
        "Real-time flood risk for India.\n\n"
        "**Data pipeline (priority per field):**\n"
        "1. India WRIS POST API\n"
        "2. Open-Meteo archive / forecast\n"
        "3. Climate / hydro normals (last resort)\n\n"
        "**Confidence calibration:**\n"
        "`confidence` always equals `class_probabilities[risk_level]`.\n"
        "Max confidence = 0.40 + (realtime_fields/8) * 0.55. "
        "All fallback -> max 0.40.\n\n"
        "**Response transparency:**\n"
        "`data_quality` shows what was real vs estimated.\n"
        "`raw_ml_output` shows exact model output before adjustments.\n"
        "`validation` shows what physical rules fired and why."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Meta"])
async def health():
    return {
        "status":         "ok",
        "model_loaded":   predictor.is_loaded,
        "model_name":     predictor.model_name,
        "groq_available": groq_service.is_available,
    }


@app.get("/debug/wris", tags=["Meta"])
async def debug_wris(
    stateName:    str = Query(..., example="Assam"),
    districtName: str = Query(..., example="Dhubri"),
    endpoint:     str = Query("RainFall"),
    agency:       str = Query("CWC"),
    startdate:    str = Query("2024-07-01"),
    enddate:      str = Query("2024-07-05"),
):
    """Raw WRIS POST probe — inspect exact API response for debugging."""
    url = f"https://indiawris.gov.in/Dataset/{endpoint}"
    params = {
        "stateName": stateName, "districtName": districtName,
        "agencyName": agency, "startdate": startdate,
        "enddate": enddate, "download": "false", "page": 0, "size": 10,
    }
    headers = {
        "accept": "application/json",
        "User-Agent": "Mozilla/5.0 (compatible; FloodRiskAPI/3.0)",
        "Referer": "https://indiawris.gov.in/wris/",
        "Origin": "https://indiawris.gov.in",
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0),
                                     follow_redirects=True) as client:
            resp = await client.post(url, params=params, content=b"", headers=headers)
            try:
                body = resp.json()
            except Exception:
                body = resp.text[:2000]
        return {"http_status": resp.status_code, "request_url": str(resp.url),
                "wris_response": body}
    except Exception as e:
        return {"error": str(e), "params": params}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Predict flood risk for a single location.

    Response includes three transparency layers:
    - **data_quality**: real-time vs fallback breakdown, max allowed confidence
    - **raw_ml_output**: exact model probabilities before any adjustment
    - **validation**: which physical rules fired, what changed and why
    """
    return await _run_prediction(request)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    """Predict for up to 20 locations concurrently."""
    if len(request.locations) > 20:
        raise HTTPException(status_code=422, detail="Batch limit is 20 locations.")

    results_raw = await asyncio.gather(
        *[_run_prediction(loc) for loc in request.locations],
        return_exceptions=True,
    )
    results = []
    for loc, res in zip(request.locations, results_raw):
        if isinstance(res, Exception):
            logger.error("Batch error %s, %s: %s", loc.districtName, loc.stateName, res)
            results.append({"error": str(res), "location": loc.model_dump()})
        else:
            results.append(res.model_dump())
    return BatchPredictResponse(results=results, count=len(results))


async def _run_prediction(request: PredictRequest) -> PredictResponse:
    logger.info(
        "Request: (%.4f, %.4f) | %s, %s | %s",
        request.latitude, request.longitude,
        request.districtName, request.stateName, request.date,
    )

    # 1. Fetch
    try:
        fetched = await data_fetcher.fetch_all(
            lat=request.latitude, lon=request.longitude,
            date=str(request.date), state=request.stateName,
            district=request.districtName,
        )
    except Exception as e:
        logger.error("Fetch error: %s", e)
        raise HTTPException(status_code=502, detail=f"External API error: {e}")

    # 2. ML predict
    try:
        prediction = predictor.predict(
            lat=request.latitude, lon=request.longitude,
            date=str(request.date), state=request.stateName,
            district=request.districtName, flood_zone=fetched.flood_zone,
            features=fetched.features,
        )
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # 3. Validate + calibrate
    validated = validate(
        prediction=prediction,
        features=fetched.features,
        missing_fields=fetched.missing_fields,
        data_sources=fetched.sources,
    )

    # 4. Groq insights with full context
    insights = await groq_service.generate_insights(
        risk_level=validated.risk_level,
        confidence=validated.confidence,
        features=fetched.features,
        location=f"{request.districtName}, {request.stateName}",
        date=str(request.date),
        missing_fields=fetched.missing_fields,
        adjustments=validated.validation.adjustments,
    )

    # 5. Respond
    return PredictResponse(
        risk_level=validated.risk_level,
        confidence=validated.confidence,
        risk_score=validated.risk_score,
        class_probabilities=validated.class_probabilities,
        raw_data=fetched.features,
        data_quality=validated.data_quality,
        raw_ml_output=validated.raw_ml_output,
        validation=validated.validation,
        ai_insights=insights,
    )
