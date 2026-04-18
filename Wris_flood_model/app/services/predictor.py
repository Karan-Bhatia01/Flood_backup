"""
FloodPredictor v4
=================
Loads num_imputer, scaler, cat_imputer, clip_bounds from pkl.
No hardcoded scaler values. Exact mirror of notebook build_features().
"""

import logging
import joblib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from app.models.schemas import CollectedFeatures
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    risk_level:          str
    confidence:          float
    risk_score:          float
    class_probabilities: Dict[str, float]
    model_name:          str


class FloodPredictor:

    def __init__(self):
        self._model       = None
        self._le          = None
        self._feat_names  = None
        self._num_imp     = None
        self._scaler      = None
        self._num_cols    = None
        self._cat_imp     = None
        self._cat_cols    = None
        self._clip_bounds = None
        self._is_loaded   = False
        self.model_name   = "not loaded"

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def _ensure_model_loaded(self, path: Optional[str] = None):
        """Lazy load model only when needed (on first prediction)."""
        if self._is_loaded:
            return
        
        model_path = Path(path or settings.MODEL_PATH)
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}.\n"
                "Make sure the model file exists in your repository at:\n"
                "notebooks/models/flood_model_stacking_ensemble.pkl"
            )
        
        # Load model using joblib (handles compressed models)
        logger.info("Loading model from %s...", model_path)
        saved = joblib.load(model_path)

        required = ["model", "label_encoder", "feature_names",
                    "num_imputer", "scaler", "num_cols",
                    "cat_imputer", "cat_cols", "clip_bounds"]
        missing = [k for k in required if k not in saved]
        if missing:
            raise ValueError(
                f"pkl is missing keys: {missing}. "
                "Re-run the notebook final save cell to regenerate the pkl with transformers."
            )

        self._model       = saved["model"]
        self._le          = saved["label_encoder"]
        self._feat_names  = saved["feature_names"]
        self._num_imp     = saved["num_imputer"]
        self._scaler      = saved["scaler"]
        self._num_cols    = saved["num_cols"]
        self._cat_imp     = saved["cat_imputer"]
        self._cat_cols    = saved["cat_cols"]
        self._clip_bounds = saved["clip_bounds"]
        self.model_name   = type(self._model).__name__
        self._is_loaded   = True

        logger.info(
            "Model loaded: %s | features: %d | classes: %s | "
            "scaler center[:3]: %s",
            self.model_name, len(self._feat_names),
            list(self._le.classes_),
            self._scaler.center_[:3].round(3)
        )

    def predict(
        self,
        lat: float,
        lon: float,
        date: str,
        state: str,
        district: str,
        flood_zone: str,
        features: CollectedFeatures,
    ) -> PredictionResult:
        # Lazy load model on first prediction
        self._ensure_model_loaded()

        X        = self._build_feature_vector(lat, lon, date, state, district, flood_zone, features)
        proba    = self._model.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        classes  = list(self._le.classes_)

        class_probs = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
        risk_level  = self._le.inverse_transform([pred_idx])[0]
        confidence  = round(float(proba[pred_idx]), 4)

        weights    = {"Low": 0.0, "Moderate": 0.5, "High": 1.0}
        risk_score = round(sum(weights.get(c, 0.5) * p for c, p in class_probs.items()), 4)

        logger.info("ML raw: %s %.1f%% | %s", risk_level, confidence * 100, class_probs)

        return PredictionResult(
            risk_level=risk_level,
            confidence=confidence,
            risk_score=risk_score,
            class_probabilities=class_probs,
            model_name=self.model_name,
        )

    def _build_feature_vector(
        self,
        lat: float,
        lon: float,
        date: str,
        state: str,
        district: str,
        flood_zone: str,
        features: CollectedFeatures,
    ) -> np.ndarray:
        """
        Exact mirror of notebook build_features():
          1. Clip outliers with saved training bounds
          2. Log-transform hydro features
          3. Engineer interaction + time features
          4. Scale with saved RobustScaler (via saved num_imputer)
          5. One-hot encode with saved cat_imputer
          6. Align to feat_names order
        """
        month = int(date[5:7])

        # Step 1: safe defaults
        raw = {
            "rainfall_mm":          float(features.rainfall_mm          or 0.0),
            "temperature_c":        float(features.temperature_c        or 25.0),
            "humidity_pct":         float(features.humidity_pct         or 65.0),
            "river_discharge_m3_s": float(features.river_discharge_m3_s or 50.0),
            "water_level_m":        float(features.water_level_m        or 100.0),
            "soil_moisture":        float(features.soil_moisture        or 15.0),
            "atmospheric_pressure": float(features.atmospheric_pressure or 1010.0),
            "evapotranspiration":   float(features.evapotranspiration   or 2.0),
        }

        # Step 2: clip to training bounds
        for col, (lo, hi) in self._clip_bounds.items():
            raw[col] = float(np.clip(raw[col], lo, hi))

        # Step 3: log-transform
        raw["rainfall_mm"]          = float(np.log1p(max(raw["rainfall_mm"], 0)))
        raw["river_discharge_m3_s"] = float(np.log1p(max(raw["river_discharge_m3_s"], 0)))
        raw["water_level_m"]        = float(np.log1p(max(raw["water_level_m"], 0)))

        # Step 4: engineered features
        hum = float(np.clip(raw["humidity_pct"], 0, 100))
        raw["rain_x_humidity"]    = raw["rainfall_mm"] * hum / 100.0
        raw["discharge_x_wlevel"] = raw["river_discharge_m3_s"] * raw["water_level_m"]
        raw["moisture_deficit"]   = max(raw["evapotranspiration"], 0) - max(raw["soil_moisture"], 0)
        raw["heat_index"]         = raw["temperature_c"] * hum / 100.0
        raw["month_sin"]          = float(np.sin(2 * np.pi * month / 12))
        raw["month_cos"]          = float(np.cos(2 * np.pi * month / 12))
        raw["is_monsoon"]         = float(month in [6, 7, 8, 9])
        raw["dist_from_center"]   = float(np.sqrt((lat - 20.5) ** 2 + (lon - 78.9) ** 2))
        raw["latitude"]           = lat
        raw["longitude"]          = lon

        # Step 5: scale numerics using saved imputer + scaler
        num_arr    = np.array([[raw.get(col, 0.0) for col in self._num_cols]])
        num_scaled = self._scaler.transform(self._num_imp.transform(num_arr))
        num_df     = pd.DataFrame(num_scaled, columns=self._num_cols)

        # Step 6: one-hot encode categoricals using saved cat_imputer
        cat_input = pd.DataFrame([{
            "stateName":    state.strip().title(),
            "districtName": district.strip().title(),
            "flood_zone":   flood_zone.strip(),
        }])[self._cat_cols]
        cat_df  = pd.DataFrame(self._cat_imp.transform(cat_input), columns=self._cat_cols)
        cat_ohe = pd.get_dummies(cat_df, drop_first=True)

        # Step 7: align to exact training column order
        row_df = pd.concat([num_df, cat_ohe], axis=1)
        row_df = row_df.reindex(columns=self._feat_names, fill_value=0.0)

        return row_df.values.astype(np.float32)
