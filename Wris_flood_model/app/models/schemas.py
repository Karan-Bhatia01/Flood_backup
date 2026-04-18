from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import date as DateType


# ── Request ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    latitude:     float    = Field(..., ge=-90,  le=90)
    longitude:    float    = Field(..., ge=-180, le=180)
    date:         DateType = Field(...)
    stateName:    str      = Field(...)
    districtName: str      = Field(...)

    @field_validator("stateName", "districtName")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Must not be empty")
        return v.strip().title()

    model_config = {"json_schema_extra": {"example": {
        "latitude": 26.02, "longitude": 89.97,
        "date": "2025-07-15", "stateName": "Assam", "districtName": "Dhubri",
    }}}


class BatchPredictRequest(BaseModel):
    locations: List[PredictRequest] = Field(..., description="Up to 20 locations")


# ── Feature data ──────────────────────────────────────────────────────────────

class CollectedFeatures(BaseModel):
    rainfall_mm:          Optional[float] = None
    temperature_c:        Optional[float] = None
    humidity_pct:         Optional[float] = None
    river_discharge_m3_s: Optional[float] = None
    water_level_m:        Optional[float] = None
    soil_moisture:        Optional[float] = None
    atmospheric_pressure: Optional[float] = None
    evapotranspiration:   Optional[float] = None


# ── Transparency sub-models ───────────────────────────────────────────────────

class DataQuality(BaseModel):
    """How much of the data is real vs estimated."""
    total_fields:       int
    realtime_count:     int
    fallback_count:     int
    completeness_pct:   float          # 0–100
    sources:            Dict[str, str] # field -> "india-wris" | "open-meteo" | "climate-normal-fallback"
    missing_fields:     List[str]
    max_allowed_confidence: float      # derived from completeness


class RawMLOutput(BaseModel):
    """Exact model output before any post-processing."""
    risk_level:          str
    confidence:          float
    class_probabilities: Dict[str, float]
    model_name:          str
    note:                str = "Raw probabilities before physical validation and confidence calibration"


class ValidationDetail(BaseModel):
    """What the physical validator did and why."""
    rules_checked:       int
    rules_fired:         int
    risk_adjusted:       bool
    confidence_adjusted: bool
    adjustments:         List[str]


# ── AI Insights ───────────────────────────────────────────────────────────────

class AIInsights(BaseModel):
    explanation:   str
    action_advice: str
    severity_note: str
    generated_by:  str           = "groq"
    model_used:    Optional[str] = None
    fallback:      bool          = False


# ── Final response ────────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    # Final calibrated decision
    risk_level:          str
    confidence:          float          # always == class_probabilities[risk_level]
    risk_score:          float          # 0=Low, 0.5=Moderate, 1=High weighted
    class_probabilities: Dict[str, float]

    # Input data as collected
    raw_data: CollectedFeatures

    # Transparency layers
    data_quality:      DataQuality
    raw_ml_output:     RawMLOutput
    validation:        ValidationDetail

    # AI analysis
    ai_insights: AIInsights


class BatchPredictResponse(BaseModel):
    results: List[Dict[str, Any]]
    count:   int
