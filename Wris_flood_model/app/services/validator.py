"""
PredictionValidator v3
======================
Fixes every issue from the technical audit:

1. confidence ALWAYS == class_probabilities[predicted_class]
   No more 0.9999 probability but 0.7 confidence.

2. Confidence cap based on data completeness (not binary flags):
     max_confidence = 0.40 + (realtime_fields / 8) * 0.55
     0/8 real  -> max 0.40
     4/8 real  -> max 0.675
     8/8 real  -> max 0.95

3. Dry-condition physical check:
   rain<1mm AND soil<15% AND humidity<50% AND ET>4mm/day -> cannot be High
   unless discharge >= 200 m3/s (upstream event).

4. All adjustments fully documented in ValidationDetail.

5. Python 3.8 compatible (List/Dict from typing, no list[str] syntax).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from app.models.schemas import (
    CollectedFeatures, DataQuality, RawMLOutput, ValidationDetail
)
from app.services.predictor import PredictionResult

logger = logging.getLogger(__name__)

# ── Physical thresholds (not hardcoded numerics — all named constants) ────────
TOTAL_FEATURE_FIELDS    = 8
CONF_BASE               = 0.40   # confidence when 0% data is real
CONF_SCALE              = 0.55   # additional confidence per 100% real data
CONF_MAX                = 0.95   # cap even with 100% real data (model uncertainty)
CONF_FLOOR              = 0.30   # never below this

DRY_RAIN_THRESHOLD      = 1.0    # mm — below = no meaningful rain
DRY_SOIL_THRESHOLD      = 15.0   # % — below = dry soil
DRY_HUMIDITY_THRESHOLD  = 50.0   # % — below = dry air
DRY_ET_THRESHOLD        = 4.0    # mm/day — above = hot/dry evaporation
UPSTREAM_DISCHARGE_MIN  = 200.0  # m3/s — above = likely upstream event
ET_CONFLICT_RAIN_MAX    = 5.0    # mm — below this + high ET = inconsistency

WEIGHTS = {"Low": 0.0, "Moderate": 0.5, "High": 1.0}


@dataclass
class ValidatedPrediction:
    risk_level:          str
    confidence:          float
    risk_score:          float
    class_probabilities: Dict[str, float]
    data_quality:        DataQuality
    raw_ml_output:       RawMLOutput
    validation:          ValidationDetail


def validate(
    prediction:     PredictionResult,
    features:       CollectedFeatures,
    missing_fields: List[str],
    data_sources:   Dict[str, str],
) -> ValidatedPrediction:

    risk_level  = prediction.risk_level
    class_probs = dict(prediction.class_probabilities)
    adjustments: List[str] = []
    risk_adjusted       = False
    confidence_adjusted = False
    rules_checked       = 0

    # ─────────────────────────────────────────────────────────────────────
    # 1. Data quality assessment
    # ─────────────────────────────────────────────────────────────────────
    fallback_fields  = [f for f, src in data_sources.items() if "fallback" in src]
    realtime_fields  = [f for f in data_sources if f not in fallback_fields]
    fallback_count   = len(fallback_fields)
    realtime_count   = len(realtime_fields)
    completeness     = realtime_count / TOTAL_FEATURE_FIELDS   # 0.0–1.0

    max_conf = round(min(CONF_BASE + completeness * CONF_SCALE, CONF_MAX), 3)

    dq = DataQuality(
        total_fields=TOTAL_FEATURE_FIELDS,
        realtime_count=realtime_count,
        fallback_count=fallback_count,
        completeness_pct=round(completeness * 100, 1),
        sources=data_sources,
        missing_fields=missing_fields,
        max_allowed_confidence=max_conf,
    )

    raw_ml = RawMLOutput(
        risk_level=prediction.risk_level,
        confidence=prediction.confidence,
        class_probabilities=prediction.class_probabilities,
        model_name=prediction.model_name,
    )

    # ─────────────────────────────────────────────────────────────────────
    # 2. Confidence cap from data completeness
    #    Scale class_probs so predicted class == max_conf, renormalize.
    # ─────────────────────────────────────────────────────────────────────
    rules_checked += 1
    raw_conf = prediction.confidence
    if raw_conf > max_conf:
        confidence_adjusted = True
        ratio = max_conf / max(raw_conf, 1e-9)
        other_sum = sum(p for cls, p in class_probs.items() if cls != risk_level)
        new_other  = 1.0 - max_conf
        for cls in class_probs:
            if cls == risk_level:
                class_probs[cls] = max_conf
            else:
                class_probs[cls] = round(
                    (class_probs[cls] / max(other_sum, 1e-9)) * new_other, 4
                )
        # Renormalize to exactly 1.0
        total = sum(class_probs.values())
        class_probs = {k: round(v / total, 4) for k, v in class_probs.items()}
        adjustments.append(
            f"Confidence reduced {raw_conf:.2f} -> {max_conf:.2f}: "
            f"only {realtime_count}/{TOTAL_FEATURE_FIELDS} fields from real-time sources "
            f"({', '.join(fallback_fields) if fallback_fields else 'none'} used climate normals). "
            f"Formula: 0.40 + {completeness:.2f} * 0.55 = {max_conf:.2f}."
        )

    # ─────────────────────────────────────────────────────────────────────
    # 3. Physical consistency: dry conditions vs High risk
    # ─────────────────────────────────────────────────────────────────────
    rules_checked += 1
    rain      = features.rainfall_mm          or 0.0
    soil      = features.soil_moisture        or 20.0
    humidity  = features.humidity_pct         or 65.0
    et0       = features.evapotranspiration   or 2.0
    discharge = features.river_discharge_m3_s or 0.0

    all_dry = (
        rain      < DRY_RAIN_THRESHOLD
        and soil     < DRY_SOIL_THRESHOLD
        and humidity < DRY_HUMIDITY_THRESHOLD
        and et0      > DRY_ET_THRESHOLD
    )

    if risk_level == "High" and all_dry:
        if discharge >= UPSTREAM_DISCHARGE_MIN:
            # Keep High — upstream event is plausible — but cap confidence further
            new_conf = min(class_probs["High"], 0.65)
            high_p   = class_probs["High"]
            diff     = high_p - new_conf
            class_probs["High"]     = round(new_conf, 4)
            class_probs["Moderate"] = round(class_probs.get("Moderate", 0) + diff * 0.7, 4)
            class_probs["Low"]      = round(1.0 - class_probs["High"] - class_probs["Moderate"], 4)
            total = sum(class_probs.values())
            class_probs = {k: round(v / total, 4) for k, v in class_probs.items()}
            confidence_adjusted = True
            adjustments.append(
                f"High risk kept but confidence reduced to {new_conf:.2f}: "
                f"locally dry (rain={rain:.0f}mm, soil={soil:.0f}%, "
                f"humidity={humidity:.0f}%, ET={et0:.1f}mm/day) "
                f"but discharge={discharge:.0f} m3/s >= {UPSTREAM_DISCHARGE_MIN:.0f} "
                "suggests upstream flooding or dam release. Verify with upstream gauges."
            )
        else:
            # Downgrade High -> Moderate
            risk_adjusted = True
            high_p = class_probs.get("High", 0.0)
            class_probs["High"]     = round(high_p * 0.15, 4)
            class_probs["Moderate"] = round(high_p * 0.55 + class_probs.get("Moderate", 0.0), 4)
            class_probs["Low"]      = round(max(1.0 - class_probs["High"] - class_probs["Moderate"], 0.0), 4)
            total = sum(class_probs.values())
            class_probs = {k: round(v / total, 4) for k, v in class_probs.items()}
            risk_level = "Moderate"
            confidence_adjusted = True
            adjustments.append(
                "Risk DOWNGRADED High -> Moderate: "
                f"rain={rain:.0f}mm (<{DRY_RAIN_THRESHOLD}), "
                f"soil={soil:.0f}% (<{DRY_SOIL_THRESHOLD}), "
                f"humidity={humidity:.0f}% (<{DRY_HUMIDITY_THRESHOLD}), "
                f"ET={et0:.1f}mm/day (>{DRY_ET_THRESHOLD}) — "
                f"all dry signals present. "
                f"Discharge={discharge:.0f} m3/s below upstream threshold "
                f"({UPSTREAM_DISCHARGE_MIN} m3/s) so upstream event unlikely."
            )

    # ─────────────────────────────────────────────────────────────────────
    # 4. ET conflict: high ET + low rain but still High
    # ─────────────────────────────────────────────────────────────────────
    rules_checked += 1
    if risk_level == "High" and et0 > DRY_ET_THRESHOLD and rain < ET_CONFLICT_RAIN_MAX:
        current_conf = class_probs.get("High", 0.0)
        new_conf     = min(current_conf, 0.70)
        if new_conf < current_conf:
            diff = current_conf - new_conf
            class_probs["High"]     = round(new_conf, 4)
            class_probs["Moderate"] = round(class_probs.get("Moderate", 0) + diff * 0.6, 4)
            class_probs["Low"]      = round(max(1.0 - class_probs["High"] - class_probs["Moderate"], 0.0), 4)
            total = sum(class_probs.values())
            class_probs = {k: round(v / total, 4) for k, v in class_probs.items()}
            confidence_adjusted = True
            adjustments.append(
                f"Confidence further reduced: ET={et0:.1f}mm/day (>{DRY_ET_THRESHOLD}) "
                f"and rain={rain:.0f}mm (<{ET_CONFLICT_RAIN_MAX}mm) conflict with High flood risk."
            )

    # ─────────────────────────────────────────────────────────────────────
    # 5. Floor — never below CONF_FLOOR
    # ─────────────────────────────────────────────────────────────────────
    rules_checked += 1
    class_probs[risk_level] = max(class_probs.get(risk_level, 0.0), CONF_FLOOR)
    total = sum(class_probs.values())
    class_probs = {k: round(v / total, 4) for k, v in class_probs.items()}

    # Final sync: confidence MUST equal class_probs[risk_level]
    confidence = round(class_probs[risk_level], 4)
    risk_score = round(sum(WEIGHTS.get(c, 0.5) * p for c, p in class_probs.items()), 4)

    vd = ValidationDetail(
        rules_checked=rules_checked,
        rules_fired=len(adjustments),
        risk_adjusted=risk_adjusted,
        confidence_adjusted=confidence_adjusted,
        adjustments=adjustments,
    )

    if adjustments:
        logger.info("Validator: %d rules fired for final risk=%s conf=%.2f",
                    len(adjustments), risk_level, confidence)

    return ValidatedPrediction(
        risk_level=risk_level,
        confidence=confidence,
        risk_score=risk_score,
        class_probabilities=class_probs,
        data_quality=dq,
        raw_ml_output=raw_ml,
        validation=vd,
    )
