"""
DataFetcher v2
==============
Priority per field: WRIS -> Open-Meteo -> climate/hydro normals.
All 8 WRIS calls fire concurrently with Open-Meteo.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import mean
from typing import Dict, List, Optional

import httpx

from app.models.schemas import CollectedFeatures
from app.config import settings

logger = logging.getLogger(__name__)

CLIMATE_NORMALS = {
    1:  (8,   18, 55, 1013, 2.1, 14),
    2:  (10,  21, 50, 1012, 2.8, 13),
    3:  (12,  26, 45, 1010, 3.5, 12),
    4:  (20,  30, 40, 1008, 4.2, 11),
    5:  (45,  33, 45, 1006, 4.5, 12),
    6:  (130, 30, 70, 1003, 3.8, 20),
    7:  (250, 28, 82, 1002, 3.2, 26),
    8:  (230, 28, 83, 1002, 3.0, 27),
    9:  (150, 28, 78, 1004, 3.1, 24),
    10: (70,  27, 70, 1007, 2.9, 19),
    11: (25,  23, 62, 1010, 2.4, 16),
    12: (12,  19, 58, 1012, 2.0, 15),
}

HYDRO_NORMALS = {
    "West Coast":             (85,  320),
    "NE India":               (420, 890),
    "Ganga Plain":            (210, 440),
    "Ganga Delta":            (380, 560),
    "Central India":          (95,  180),
    "Deccan":                 (40,  95),
    "East Coast":             (130, 260),
    "Krishna-Godavari Delta": (175, 310),
    "West India":             (55,  120),
    "Himalayas":              (310, 680),
}

STATE_ZONE_MAP = {
    "Kerala": "West Coast",           "Karnataka": "West Coast",
    "Goa": "West Coast",              "Maharashtra": "West Coast",
    "Gujarat": "West India",          "Rajasthan": "West India",
    "Assam": "NE India",              "Meghalaya": "NE India",
    "Nagaland": "NE India",           "Manipur": "NE India",
    "Mizoram": "NE India",            "Tripura": "NE India",
    "Arunachal Pradesh": "NE India",  "Sikkim": "Himalayas",
    "Himachal Pradesh": "Himalayas",  "Uttarakhand": "Himalayas",
    "Jammu And Kashmir": "Himalayas", "Ladakh": "Himalayas",
    "Uttar Pradesh": "Ganga Plain",   "Bihar": "Ganga Plain",
    "Jharkhand": "Ganga Plain",       "West Bengal": "Ganga Delta",
    "Odisha": "East Coast",           "Andhra Pradesh": "East Coast",
    "Tamil Nadu": "East Coast",       "Telangana": "Krishna-Godavari Delta",
    "Chhattisgarh": "Central India",  "Madhya Pradesh": "Central India",
    "Haryana": "Ganga Plain",         "Punjab": "Ganga Plain",
    "Delhi": "Ganga Plain",
}

WRIS_ENDPOINTS = {
    "rainfall_mm":          {"path": "RainFall",              "agencies": ["CWC", "{state}"]},
    "temperature_c":        {"path": "Temperature",           "agencies": ["{state}", "CWC"]},
    "humidity_pct":         {"path": "Relative Humidity",     "agencies": ["CWC", "{state}"]},
    "river_discharge_m3_s": {"path": "River Water Discharge", "agencies": ["CWC"]},
    "water_level_m":        {"path": "River Water Level",     "agencies": ["CWC"]},
    "soil_moisture":        {"path": "Soil Moisture",         "agencies": ["NRSC VIC MODEL"]},
    "atmospheric_pressure": {"path": "Atmospheric Pressure",  "agencies": ["{state} SW", "{state}", "CWC"]},
    "evapotranspiration":   {"path": "Evapo Transpiration",   "agencies": ["NRSC VIC MODEL"]},
}

WRIS_BASE    = "https://indiawris.gov.in/Dataset"
WRIS_HEADERS = {
    "accept":     "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; FloodRiskAPI/3.0)",
    "Referer":    "https://indiawris.gov.in/wris/",
    "Origin":     "https://indiawris.gov.in",
}
WRIS_VALUE_COLS = [
    "dataValue", "value", "Value", "data_value",
    "ObservedValue", "reading", "discharge", "waterLevel",
    "rainfall", "temperature", "humidity", "pressure",
]


@dataclass
class FetchedData:
    features:       CollectedFeatures
    flood_zone:     str
    sources:        Dict[str, str] = field(default_factory=dict)
    missing_fields: List[str]      = field(default_factory=list)


class DataFetcher:

    OPENMETEO_ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
    OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self):
        self._timeout = httpx.Timeout(connect=10.0, read=25.0, write=5.0, pool=5.0)

    async def fetch_all(
        self, lat: float, lon: float, date: str, state: str, district: str,
    ) -> FetchedData:
        month      = int(date[5:7])
        flood_zone = STATE_ZONE_MAP.get(state.strip().title(), "Central India")
        target_dt  = datetime.strptime(date, "%Y-%m-%d")
        start_date = (target_dt - timedelta(days=3)).strftime("%Y-%m-%d")
        end_date   = (target_dt + timedelta(days=1)).strftime("%Y-%m-%d")

        feat_keys  = list(WRIS_ENDPOINTS.keys())
        all_coros  = [
            self._fetch_wris_feature(k, WRIS_ENDPOINTS[k], state, district, start_date, end_date)
            for k in feat_keys
        ] + [self._fetch_openmeteo(lat, lon, date)]

        all_results = await asyncio.gather(*all_coros, return_exceptions=True)

        wris_results: Dict[str, Optional[float]] = {}
        for i, key in enumerate(feat_keys):
            res = all_results[i]
            wris_results[key] = None if isinstance(res, Exception) else res

        om_result = all_results[-1]
        if isinstance(om_result, Exception):
            logger.warning("Open-Meteo error: %s", om_result)
            om_result = {}

        sources: Dict[str, str] = {}
        missing: List[str]      = []
        normals = CLIMATE_NORMALS[month]
        hydro   = HYDRO_NORMALS.get(flood_zone, (100, 200))

        def pick(feat: str, wris_val: Optional[float], om_key: Optional[str],
                 fallback_val: float) -> float:
            if wris_val is not None:
                sources[feat] = "india-wris"
                return float(wris_val)
            om_val = (om_result or {}).get(om_key) if om_key else None
            if om_val is not None:
                sources[feat] = "open-meteo"
                return float(om_val)
            missing.append(feat)
            sources[feat] = "climate-normal-fallback"
            return float(fallback_val)

        features = CollectedFeatures(
            rainfall_mm=          pick("rainfall_mm",          wris_results["rainfall_mm"],          "rainfall_mm",          normals[0]),
            temperature_c=        pick("temperature_c",        wris_results["temperature_c"],        "temperature_c",        normals[1]),
            humidity_pct=         pick("humidity_pct",         wris_results["humidity_pct"],         "humidity_pct",         normals[2]),
            atmospheric_pressure= pick("atmospheric_pressure", wris_results["atmospheric_pressure"], "atmospheric_pressure", normals[3]),
            evapotranspiration=   pick("evapotranspiration",   wris_results["evapotranspiration"],   "evapotranspiration",   normals[4]),
            soil_moisture=        pick("soil_moisture",        wris_results["soil_moisture"],        "soil_moisture",        normals[5]),
            river_discharge_m3_s= pick("river_discharge_m3_s",wris_results["river_discharge_m3_s"], None,                   hydro[0]),
            water_level_m=        pick("water_level_m",        wris_results["water_level_m"],        None,                   hydro[1]),
        )

        if missing:
            logger.info("Fallback for: %s", missing)
        return FetchedData(features=features, flood_zone=flood_zone,
                           sources=sources, missing_fields=missing)

    async def _fetch_wris_feature(self, feature_name: str, cfg: dict, state: str,
                                   district: str, start_date: str, end_date: str
                                   ) -> Optional[float]:
        for agency in self._resolve_agencies(cfg["agencies"], state):
            val = await self._wris_post(cfg["path"], state, district, agency,
                                        start_date, end_date)
            if val is not None:
                return val
        return None

    async def _wris_post(self, endpoint_path: str, state: str, district: str,
                          agency: str, start_date: str, end_date: str
                          ) -> Optional[float]:
        url        = f"{WRIS_BASE}/{endpoint_path}"
        all_values: List[float] = []
        async with httpx.AsyncClient(headers=WRIS_HEADERS, timeout=self._timeout,
                                     follow_redirects=True) as client:
            for page in range(10):
                params = {
                    "stateName": state, "districtName": district,
                    "agencyName": agency, "startdate": start_date,
                    "enddate": end_date, "download": "false",
                    "page": page, "size": 500,
                }
                try:
                    resp = await client.post(url, params=params, content=b"")
                    resp.raise_for_status()
                    payload = resp.json()
                except Exception as e:
                    logger.warning("WRIS error [%s/%s]: %s", endpoint_path, agency, e)
                    break

                if page == 0:
                    sc = payload.get("statusCode")
                    if sc and sc != 200:
                        break

                records = payload.get("data", [])
                if isinstance(records, dict):
                    records = (records.get("content") or records.get("data")
                               or records.get("records") or [])
                if not records:
                    break

                for rec in records:
                    if not isinstance(rec, dict):
                        continue
                    col = next((c for c in WRIS_VALUE_COLS if c in rec), None)
                    if col:
                        try:
                            all_values.append(float(rec[col]))
                        except (TypeError, ValueError):
                            pass

                if len(records) < 500:
                    break

        return mean(all_values) if all_values else None

    async def _fetch_openmeteo(self, lat: float, lon: float, date: str) -> dict:
        variables = ["precipitation_sum", "temperature_2m_mean",
                     "relative_humidity_2m_mean", "surface_pressure_mean",
                     "et0_fao_evapotranspiration", "soil_moisture_0_to_10cm_mean"]
        col_map = {
            "precipitation_sum": "rainfall_mm", "temperature_2m_mean": "temperature_c",
            "relative_humidity_2m_mean": "humidity_pct", "surface_pressure_mean": "atmospheric_pressure",
            "et0_fao_evapotranspiration": "evapotranspiration",
            "soil_moisture_0_to_10cm_mean": "soil_moisture",
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                r = await client.get(self.OPENMETEO_ARCHIVE_URL, params={
                    "latitude": lat, "longitude": lon, "start_date": date,
                    "end_date": date, "daily": ",".join(variables), "timezone": "Asia/Kolkata",
                })
                r.raise_for_status()
                result = self._parse_om(r.json(), col_map, date)
                if result:
                    return result
            except Exception as e:
                logger.warning("Open-Meteo archive failed: %s", e)
            try:
                fvars = ["precipitation_sum", "temperature_2m_mean", "relative_humidity_2m_max",
                         "surface_pressure_msl", "et0_fao_evapotranspiration", "soil_moisture_0_to_1cm_sum"]
                fmap  = {
                    "precipitation_sum": "rainfall_mm", "temperature_2m_mean": "temperature_c",
                    "relative_humidity_2m_max": "humidity_pct", "surface_pressure_msl": "atmospheric_pressure",
                    "et0_fao_evapotranspiration": "evapotranspiration",
                    "soil_moisture_0_to_1cm_sum": "soil_moisture",
                }
                r = await client.get(self.OPENMETEO_FORECAST_URL, params={
                    "latitude": lat, "longitude": lon, "daily": ",".join(fvars),
                    "timezone": "Asia/Kolkata", "forecast_days": 7,
                })
                r.raise_for_status()
                return self._parse_om(r.json(), fmap, date)
            except Exception as e:
                logger.error("Open-Meteo forecast failed: %s", e)
                return {}

    @staticmethod
    def _parse_om(data: dict, col_map: dict, target_date: str) -> dict:
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        try:
            idx = dates.index(target_date)
        except ValueError:
            idx = 0 if dates else None
        if idx is None:
            return {}
        result = {}
        for api_col, our_col in col_map.items():
            vals = daily.get(api_col, [])
            val  = vals[idx] if idx < len(vals) else None
            result[our_col] = float(val) if val is not None else None
        return result

    @staticmethod
    def _resolve_agencies(templates: list, state: str) -> list:
        seen: set  = set()
        resolved: List[str] = []
        for t in templates:
            a = t.replace("{state}", state)
            if a not in seen:
                resolved.append(a)
                seen.add(a)
        return resolved
