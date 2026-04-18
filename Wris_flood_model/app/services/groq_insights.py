"""
GroqInsights v3
===============
- Compact prompt (no 413 errors)
- Passes data quality and validation context to Groq
- Handles dict/non-string values in parsed JSON
- Rule-based fallback if Groq unavailable
"""

import json
import logging
import re
from typing import List, Optional

import httpx

from app.models.schemas import AIInsights, CollectedFeatures
from app.config import settings

logger      = logging.getLogger(__name__)
GROQ_URL    = "https://api.groq.com/openai/v1/chat/completions"


class GroqInsights:

    def __init__(self):
        self.is_available = bool(settings.GROQ_API_KEY)
        if not self.is_available:
            logger.warning("GROQ_API_KEY not set — using rule-based fallback")

    async def generate_insights(
        self,
        risk_level:     str,
        confidence:     float,
        features:       CollectedFeatures,
        location:       str,
        date:           str,
        missing_fields: Optional[List[str]] = None,
        adjustments:    Optional[List[str]] = None,
    ) -> AIInsights:
        if self.is_available:
            try:
                return await self._call_groq(
                    risk_level, confidence, features, location, date,
                    missing_fields or [], adjustments or []
                )
            except Exception as e:
                logger.warning("Groq failed (%s) — using fallback", e)
        return self._fallback(risk_level, features, location)

    async def _call_groq(
        self,
        risk_level: str, confidence: float, features: CollectedFeatures,
        location: str, date: str, missing_fields: List[str], adjustments: List[str],
    ) -> AIInsights:
        # Build compact prompt
        data_note = (
            f" NOTE: {len(missing_fields)}/8 fields are climate-normal estimates, not real-time."
            if missing_fields else ""
        )
        adj_note = (
            f" VALIDATOR: {adjustments[0][:120]}"
            if adjustments else ""
        )
        prompt = (
            f"Location:{location} Date:{date} Risk:{risk_level} conf:{confidence:.0%}"
            f"{data_note}{adj_note}\n"
            f"Rain:{features.rainfall_mm:.0f}mm T:{features.temperature_c:.0f}C "
            f"Hum:{features.humidity_pct:.0f}% Q:{features.river_discharge_m3_s:.0f}m3/s "
            f"WL:{features.water_level_m:.0f}m SM:{features.soil_moisture:.0f}% "
            f"P:{features.atmospheric_pressure:.0f}hPa ET:{features.evapotranspiration:.1f}mm\n"
            'Respond ONLY in JSON: {"explanation":"...","action_advice":"...","severity_note":"..."}'
        )

        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
            resp = await client.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}",
                         "Content-Type": "application/json"},
                json={
                    "model": settings.GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": (
                            "You are a senior hydrologist for India. "
                            "Respond ONLY in JSON with string keys: "
                            "explanation, action_advice, severity_note. "
                            "Each value must be a plain string, max 80 words. "
                            "Reflect data confidence in your tone."
                        )},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 350,
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            content = re.sub(r"```json|```", "", content).strip()
            parsed  = json.loads(content)

        def to_str(v: object) -> str:
            if isinstance(v, str):
                return v
            if isinstance(v, dict):
                return "; ".join(f"{k}: {val}" for k, val in v.items())
            return str(v)

        return AIInsights(
            explanation=to_str(parsed.get("explanation", "")),
            action_advice=to_str(parsed.get("action_advice", "")),
            severity_note=to_str(parsed.get("severity_note", "")),
            generated_by="groq",
            model_used=settings.GROQ_MODEL,
            fallback=False,
        )

    def _fallback(self, risk_level: str, features: CollectedFeatures,
                  location: str) -> AIInsights:
        templates = {
            "High": {
                "explanation": (
                    f"{location} has HIGH flood risk. River discharge "
                    f"({features.river_discharge_m3_s:.0f} m3/s) and water level "
                    f"({features.water_level_m:.0f}m) are critically elevated."
                ),
                "action_advice": (
                    "Evacuate low-lying areas immediately. Activate emergency flood response. "
                    "Avoid flooded roads and riverbanks. Monitor NDRF/SDRF alerts."
                ),
                "severity_note": "Reassess every hour — conditions may worsen rapidly.",
            },
            "Moderate": {
                "explanation": (
                    f"{location} has MODERATE flood risk. Rainfall ({features.rainfall_mm:.0f}mm) "
                    f"and river levels ({features.water_level_m:.0f}m) are elevated but not critical."
                ),
                "action_advice": (
                    "Stay alert and monitor weather updates. Avoid riverbanks. "
                    "Prepare emergency kit. Pre-position rescue equipment."
                ),
                "severity_note": "Reassess every 3 hours — could escalate with more rain.",
            },
            "Low": {
                "explanation": (
                    f"{location} has LOW flood risk. "
                    "All hydro-meteorological indicators are within normal range."
                ),
                "action_advice": "No immediate action needed. Monitor IMD forecasts.",
                "severity_note": "Reassess if rainfall forecast exceeds 30mm/hr.",
            },
        }
        t = templates.get(risk_level, templates["Moderate"])
        return AIInsights(
            explanation=t["explanation"], action_advice=t["action_advice"],
            severity_note=t["severity_note"], generated_by="rule-based-fallback",
            model_used=None, fallback=True,
        )
