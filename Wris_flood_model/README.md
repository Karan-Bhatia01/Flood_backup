# GeoAware — Real-Time Flood Risk Prediction System for India

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

**A production-grade FastAPI backend that predicts flood risk (Low / Moderate / High) for any district in India by fetching real-time hydro-meteorological data, running a trained ML model, applying physical consistency validation, and returning AI-generated insights via Groq LLM.**

---

## Table of Contents

- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [API Response Structure](#api-response-structure)
- [Data Pipeline](#data-pipeline)
- [ML Model](#ml-model)
- [Confidence Calibration](#confidence-calibration)
- [Physical Validation Rules](#physical-validation-rules)
- [Setup Instructions](#setup-instructions)
- [Environment Variables](#environment-variables)
- [Known Limitations](#known-limitations)

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI 0.115 + Uvicorn |
| **ML Model** | Scikit-learn StackingClassifier (GradientBoosting + RandomForest + KNN) |
| **Data Source 1** | India WRIS POST API (indiawris.gov.in) |
| **Data Source 2** | Open-Meteo Historical/Forecast API (free, no key) |
| **Data Source 3** | India climate normals (last-resort fallback) |
| **AI Insights** | Groq LLM (llama3-8b-8192) |
| **Config** | pydantic-settings + .env |
| **HTTP Client** | httpx (async) |
| **Language** | Python 3.8+ |

---

## Project Structure

```
GeoAware_flood/
├── app/
│   ├── main.py                    # FastAPI routes + pipeline orchestrator
│   ├── config.py                  # Settings loaded from .env
│   ├── models/
│   │   └── schemas.py             # Pydantic request/response models
│   └── services/
│       ├── predictor.py           # ML inference with proper scaling
│       ├── data_fetcher.py        # WRIS + Open-Meteo + fallback fetching
│       ├── validator.py           # Physical consistency + confidence calibration
│       └── groq_insights.py       # Groq LLM AI analysis
├── model/
│   └── flood_model.pkl            # Trained model + transformers (from notebooks/)
├── notebooks/
│   ├── flood_prediction_final.ipynb   # Full training pipeline
│   └── models/
│       └── flood_model_stacking_ensemble.pkl   # Correct pkl with all transformers
├── src/
│   └── components/
│       └── data_collection_new.py # India WRIS dataset builder script
├── requirements.txt
├── .env.example
└── run.py
```

---

## API Endpoints

### 🔹 POST `/predict`

**Single location flood risk prediction**

**Request:**
```json
{
  "latitude": 23.1815,
  "longitude": 79.9864,
  "date": "2024-12-15",
  "stateName": "Madhya Pradesh",
  "districtName": "Indore"
}
```

**Response:** Full `PredictResponse` (see [API Response Structure](#api-response-structure))

---

### 🔹 POST `/predict/batch`

**Up to 20 concurrent predictions with isolated per-item failure handling**

**Request:**
```json
{
  "locations": [
    { "latitude": 23.1815, "longitude": 79.9864, "date": "2024-12-15", "stateName": "Madhya Pradesh", "districtName": "Indore" },
    { "latitude": 28.7041, "longitude": 77.1025, "date": "2024-12-15", "stateName": "Delhi", "districtName": "New Delhi" }
  ]
}
```

**Response:**
```json
{
  "results": [
    { "success": true, "data": {...} },
    { "success": false, "error": "District not found", "location": {...} }
  ],
  "count": 2
}
```

---

### 🔹 GET `/health`

**System health and readiness check**

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "StackingClassifier",
  "groq_available": true
}
```

---

### 🔹 GET `/debug/wris`

**Debug tool to verify WRIS data availability for any district**

**Query Parameters:**
- `stateName` (required): State name
- `districtName` (required): District name
- `endpoint` (optional): Specific WRIS endpoint
- `agency` (optional): CWC, NRSC, state agency
- `startdate` (optional): YYYY-MM-DD
- `enddate` (optional): YYYY-MM-DD

**Response:** Raw WRIS API JSON response

---

## API Response Structure

The `/predict` endpoint returns a comprehensive response with 6 top-level sections:

```json
{
  "risk_level": "Low",
  "confidence": 0.85,
  "risk_score": 0.32,
  "class_probabilities": {
    "Low": 0.85,
    "Moderate": 0.12,
    "High": 0.03
  },

  "raw_data": {
    "rainfall_mm": 12.5,
    "temperature_c": 28.3,
    "humidity_pct": 65,
    "river_discharge_m3_s": 150.8,
    "water_level_m": 2.1,
    "soil_moisture": 22.5,
    "atmospheric_pressure": 1013.25,
    "evapotranspiration": 3.2
  },

  "data_quality": {
    "total_fields": 8,
    "realtime_count": 3,
    "fallback_count": 5,
    "completeness_pct": 100.0,
    "sources": {
      "rainfall_mm": "open-meteo",
      "temperature_c": "open-meteo",
      "humidity_pct": "open-meteo",
      "river_discharge_m3_s": "climate-normal-fallback",
      "water_level_m": "climate-normal-fallback",
      "soil_moisture": "open-meteo",
      "atmospheric_pressure": "open-meteo",
      "evapotranspiration": "open-meteo"
    },
    "missing_fields": [],
    "max_allowed_confidence": 0.675
  },

  "raw_ml_output": {
    "risk_level": "Low",
    "confidence": 0.92,
    "class_probabilities": {
      "Low": 0.92,
      "Moderate": 0.07,
      "High": 0.01
    },
    "model_name": "StackingClassifier",
    "note": "Raw probabilities before physical validation and confidence calibration"
  },

  "validation": {
    "rules_checked": 4,
    "rules_fired": 1,
    "risk_adjusted": false,
    "confidence_adjusted": true,
    "adjustments": [
      "Confidence capped by data quality: 3/8 real-time fields (max_allowed: 0.675)"
    ]
  },

  "ai_insights": {
    "explanation": "Low flood risk detected. Current hydrological conditions are stable with moderate rainfall and below-average river discharge.",
    "action_advice": "Continue routine monitoring. No immediate precautions required.",
    "severity_note": "Confidence level reflects moderate data availability; 5 of 8 fields are based on climatological normals.",
    "generated_by": "groq",
    "model_used": "llama3-8b-8192",
    "fallback": false
  }
}
```

### Response Sections Explained

| Section | Purpose |
|---|---|
| **risk_level, confidence, risk_score, class_probabilities** | Final prediction with probabilities |
| **raw_data** | The 8 input features used by the ML model |
| **data_quality** | Metadata: field count, real-time vs. fallback, source tracking, confidence cap |
| **raw_ml_output** | Unmodified model output before validation adjustments |
| **validation** | Physical consistency rules applied; which rules fired and adjustments made |
| **ai_insights** | Groq LLM-generated natural language analysis and action recommendations |

**Key Invariant:** `confidence == class_probabilities[risk_level]` — these two values are always identical.

---

## Data Pipeline

The system fetches 8 features per prediction in this priority order:

### Priority 1: India WRIS POST API

- Attempts to fetch all 8 features from India WRIS (indiawris.gov.in)
- Fires all 8 endpoint calls concurrently for speed
- For each feature, tries confirmed agency list (CWC, NRSC VIC MODEL, state agency)
- Returns mean of available records in a **±3-day window** around the requested date
- Features: rainfall, temperature, humidity, river discharge, water level, soil moisture, atmospheric pressure, evapotranspiration

### Priority 2: Open-Meteo API

- Falls back to Open-Meteo **if WRIS returns no data**
- Fetches: rainfall, temperature, humidity, atmospheric pressure, ET₀, soil moisture
- Uses **historical archive API** for past dates
- Uses **forecast API** for near-future dates (up to 16 days ahead)
- Free service; no API key required
- River discharge and water level are **not available** via Open-Meteo

### Priority 3: Climate & Hydro Normals

- **Last resort fallback** when both source 1 and 2 fail
- Monthly averages for weather fields (rainfall, temperature, etc.)
- Flood-zone hydro normals for river discharge and water level
- Coverage: all 85 districts included in model training

### Data Quality Tracking

The `data_quality.sources` field in every response tells you **exactly which source** provided each value:

```json
"sources": {
  "rainfall_mm": "india-wris",
  "temperature_c": "open-meteo",
  "humidity_pct": "climate-normal-fallback",
  ...
}
```

This transparency allows users to assess prediction reliability based on data provenance.

---

## ML Model

- **Algorithm:** StackingClassifier with ensemble base learners:
  - GradientBoosting
  - RandomForest
  - KNN (k=5)
  - Meta-learner: Logistic Regression

- **Training Data:** 266,614 rows across 85 districts in India (2024–2026)

- **Feature Set:** 135 features total
  - 18 numeric features (rainfall, discharge, etc.)
  - 117 one-hot encoded categorical features:
    - State name (27 states + territories)
    - District name (85 districts)
    - Flood zone (4 zones)

- **Target Variable:** Engineered composite flood risk score → `Low / Moderate / High`
  - Percentile-based bins guarantee ~33% per class (perfectly balanced)

- **Model Performance:** ~0.9963 macro F1-score on test set

- **Saved Transformers:** Model pickle includes:
  - `num_imputer` (SimpleImputer for numeric fields)
  - `scaler` (RobustScaler for numeric scaling)
  - `cat_imputer` (SimpleImputer for categorical fields)
  - `clip_bounds` (feature clipping parameters)
  
  This ensures inference uses the exact same preprocessing pipeline as training.

---

## Confidence Calibration

Confidence reflects both model output probability **and** data quality:

```
max_confidence = 0.40 + (realtime_fields / 8) × 0.55

Final Confidence = min(raw_model_prob, max_confidence)
```

### Examples

| Real-Time Fields | Max Allowed Confidence |
|---|---|
| 0/8 | 0.40 |
| 2/8 | 0.54 |
| 4/8 | 0.675 |
| 6/8 | 0.81 |
| 8/8 | 0.95 |

**Note:** In the response, `confidence == class_probabilities[risk_level]` — they are always identical. This is the final, calibrated confidence value returned to the user.

---

## Physical Validation Rules

The validator applies 4 consistency rules post-prediction:

### 1. Completeness Cap

**Logic:** Proportionally reduce confidence based on estimated (non-real-time) data

**Implementation:** See [Confidence Calibration](#confidence-calibration) — confidence capped by the formula above

**Justification:** Predictions relying on climatological normals are less reliable than those using live data

---

### 2. Dry Condition Check

**Logic:** If rainfall < 1 mm AND soil moisture < 15% AND humidity < 50% AND ET > 4 mm/day, the environment is physically dry

- **Action:** Downgrade `High` risk → `Moderate`
- **Exception:** If river discharge ≥ 200 m³/s, skip downgrade (indicates upstream flooding event)

**Justification:** High flood risk contradicts drought-like conditions unless a major upstream event is occurring

---

### 3. ET Conflict

**Logic:** High evapotranspiration (ET > 4 mm/day) + low rainfall (< 5 mm) contradicts High risk

**Action:** Reduce confidence to max 0.70

**Justification:** High ET implies active water cycling and vegetation; this contradicts imminent flood conditions

---

### 4. Confidence Floor

**Logic:** Never return confidence below 0.30

**Action:** Clamp final confidence to [0.30, max_allowed_confidence]

**Justification:** Even with poor data quality, the ensemble model has learned patterns; confidence should not drop to near-zero

---

## Setup Instructions

Follow these steps to set up and run GeoAware:

### 1. Clone and enter project
```bash
cd GeoAware_flood
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. **IMPORTANT:** Copy the correct model file

The `notebooks/models/` pickle has all transformers loaded; the `model/` pickle alone will fail. Copy the correct one:

```bash
copy notebooks\models\flood_model_stacking_ensemble.pkl model\flood_model.pkl  # Windows
cp notebooks/models/flood_model_stacking_ensemble.pkl model/flood_model.pkl    # Linux/Mac
```

### 5. Configure environment
```bash
copy .env.example .env  # Windows
cp .env.example .env     # Linux/Mac
```

Edit `.env` and add your Groq API key (free tier at [console.groq.com](https://console.groq.com)):
```
GROQ_API_KEY=your-key-here
```

### 6. Run the server
```bash
uvicorn app.main:app --reload --port 8000
```

### 7. Open API documentation
Visit [http://localhost:8000/docs](http://localhost:8000/docs) in your browser for interactive Swagger UI

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `MODEL_PATH` | No | `model/flood_model.pkl` | Path to trained model pickle |
| `GROQ_API_KEY` | Recommended | `""` | Groq LLM API key (free tier available at console.groq.com) |
| `GROQ_MODEL` | No | `llama3-8b-8192` | Groq model name to use for insights |
| `APP_ENV` | No | `development` | Environment mode: `development` or `production` |

### Example `.env`
```
MODEL_PATH=model/flood_model.pkl
GROQ_API_KEY=gsk_your_api_key_here
GROQ_MODEL=llama3-8b-8192
APP_ENV=development
```

---

## Known Limitations

1. **River Discharge & Water Level Unavailable via Open-Meteo**
   - These parameters always fall back to flood-zone hydro normals when WRIS has no data for a district
   - This reduces data quality for predictions in remote or unmeasured districts

2. **WRIS Coverage Varies by District**
   - Urban and industrialized districts have dense station networks
   - Rural and remote districts may have no or sparse WRIS station data
   - Check `/debug/wris` endpoint to verify data availability for your district

3. **Model Generalization**
   - ML model trained only on 85 districts
   - Predictions for unknown districts use baseline category (treated as Andhra Pradesh / Ahmedabad / Central India zone in one-hot encoding)
   - Out-of-distribution districts should be interpreted with caution

4. **Future Dates Beyond 16 Days**
   - Open-Meteo forecast API covers only ±16 days
   - Dates beyond this window default to **climate normals for all weather fields**
   - River discharge and water level already use normals, so older fallback behavior persists

---

## Contributing

Contributions are welcome! Please follow these guidelines:

- Submit issues for bugs or feature requests
- Create pull requests with descriptive commit messages
- Ensure all tests pass before submitting

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) file for details.

---

## Support

For issues, questions, or feedback, please open an issue on the project repository.

---

**Last Updated:** April 2026
