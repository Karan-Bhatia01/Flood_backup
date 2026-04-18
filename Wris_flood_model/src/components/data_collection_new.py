"""
India-WRIS Flood Feature Dataset Builder  (v6 — correct agency names)
======================================================================
API spec (confirmed from Swagger):
  POST https://indiawris.gov.in/Dataset/<DatasetName>
       ?stateName=...&districtName=...&agencyName=...
       &startdate=YYYY-MM-DD&enddate=YYYY-MM-DD
       &download=false&page=0&size=500
  Body: empty  (-d '')
  Response: { "statusCode": 200, "message": "...", "data": [...] }

CONFIRMED AGENCY NAMES (from Swagger examples):
  Temperature          → state name  e.g. "Jharkhand"
  River Water Level    → "CWC"
  River Water Discharge→ "CWC"
  Relative Humidity    → "CWC"
  RainFall             → "CWC"
  Evapo Transpiration  → "NRSC VIC MODEL"   (satellite model — all districts)
  Soil Moisture        → "NRSC VIC MODEL"   (satellite model — all districts)
  Atmospheric Pressure → state + " SW"  e.g. "Telangana SW"  OR  state name

  Each endpoint has its own agency list tried in order (first hit wins).

Usage:
    pip install requests pandas tqdm python-dateutil
    python wris_dataset_builder.py
"""

import sys, io, requests, pandas as pd, time, json, logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
from tqdm import tqdm

# Windows UTF-8 fix
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ================================================================
#  CONFIGURATION
# ================================================================

BASE_URL = "https://indiawris.gov.in/Dataset"

END_DATE   = datetime.today()
START_DATE = END_DATE - relativedelta(years=2)

PAGE_SIZE  = 500    # API max is 1000
MAX_PAGES  = 40     # 40 × 500 = 20,000 records per endpoint per district

OUTPUT_DIR = Path("wris_output")
FINAL_FILE = OUTPUT_DIR / "flood_features_india.csv"
LOG_FILE   = OUTPUT_DIR / "wris_fetch.log"

DELAY_BETWEEN_CALLS = 0.6   # seconds between requests

# ================================================================
#  ENDPOINT REGISTRY
#  Each endpoint now carries its own agency list (tried in order).
#
#  Key insight from Swagger:
#    - Station-based data  → state agency or "CWC"
#    - Model-based data    → "NRSC VIC MODEL"  (covers ALL districts)
#    - Atmospheric pressure→ "<State> SW" variant first, then state, then CWC
# ================================================================

# These agencies are endpoint-level defaults — overridden per-endpoint below
STATION_AGENCIES = ["CWC"]          # river/weather station data
MODEL_AGENCIES   = ["NRSC VIC MODEL"]  # NRSC satellite-model data

# Endpoint path → list of agencies to try (in order)
# State-name agency is added dynamically per location at runtime
ENDPOINT_AGENCIES = {
    "rainfall_mm":          {"path": "RainFall",              "agencies": ["CWC", "{state}"]},
    "temperature_c":        {"path": "Temperature",           "agencies": ["{state}", "CWC"]},
    "humidity_pct":         {"path": "Relative Humidity",     "agencies": ["CWC", "{state}"]},
    "river_discharge_m3_s": {"path": "River Water Discharge", "agencies": ["CWC"]},
    "water_level_m":        {"path": "River Water Level",     "agencies": ["CWC"]},
    "soil_moisture":        {"path": "Soil Moisture",         "agencies": ["NRSC VIC MODEL"]},
    "atmospheric_pressure": {"path": "Atmospheric Pressure",  "agencies": ["{state} SW", "{state}", "CWC", "Telangana SW"]},
    "evapotranspiration":   {"path": "Evapo Transpiration",   "agencies": ["NRSC VIC MODEL"]},
}

# ================================================================
#  FLOOD-PRONE LOCATIONS  (80 districts, 22 states)
#  Sources: CWC Flood Forecasting Network, NDMA atlas, CWC 2024 report
# ================================================================

LOCATIONS = [

    # ── ASSAM  (Brahmaputra + Barak — highest flood frequency in India)
    {"stateName": "Assam", "districtName": "Dhubri",    "latitude": 26.0200, "longitude": 89.9700, "river": "Brahmaputra", "flood_zone": "NE India"},
    {"stateName": "Assam", "districtName": "Dhemaji",   "latitude": 27.4800, "longitude": 94.5700, "river": "Brahmaputra", "flood_zone": "NE India"},
    {"stateName": "Assam", "districtName": "Lakhimpur", "latitude": 27.2300, "longitude": 94.1000, "river": "Brahmaputra", "flood_zone": "NE India"},
    {"stateName": "Assam", "districtName": "Morigaon",  "latitude": 26.2500, "longitude": 92.3300, "river": "Brahmaputra", "flood_zone": "NE India"},
    {"stateName": "Assam", "districtName": "Nagaon",    "latitude": 26.3500, "longitude": 92.6800, "river": "Brahmaputra", "flood_zone": "NE India"},
    {"stateName": "Assam", "districtName": "Cachar",    "latitude": 24.8300, "longitude": 92.7500, "river": "Barak",       "flood_zone": "NE India"},
    {"stateName": "Assam", "districtName": "Majuli",    "latitude": 26.9500, "longitude": 94.1700, "river": "Brahmaputra", "flood_zone": "NE India"},

    # ── BIHAR  (most flood-affected state by population)
    {"stateName": "Bihar", "districtName": "Darbhanga",      "latitude": 26.1542, "longitude": 85.8918, "river": "Bagmati",  "flood_zone": "Ganga Plain"},
    {"stateName": "Bihar", "districtName": "Muzaffarpur",    "latitude": 26.1197, "longitude": 85.3910, "river": "Gandak",   "flood_zone": "Ganga Plain"},
    {"stateName": "Bihar", "districtName": "Saharsa",        "latitude": 25.8800, "longitude": 86.5900, "river": "Kosi",     "flood_zone": "Ganga Plain"},
    {"stateName": "Bihar", "districtName": "Supaul",         "latitude": 26.1200, "longitude": 86.6000, "river": "Kosi",     "flood_zone": "Ganga Plain"},
    {"stateName": "Bihar", "districtName": "Gopalganj",      "latitude": 26.4700, "longitude": 84.4400, "river": "Gandak",   "flood_zone": "Ganga Plain"},
    {"stateName": "Bihar", "districtName": "Sitamarhi",      "latitude": 26.5900, "longitude": 85.4800, "river": "Bagmati",  "flood_zone": "Ganga Plain"},
    {"stateName": "Bihar", "districtName": "East Champaran", "latitude": 26.6500, "longitude": 84.9100, "river": "Gandak",   "flood_zone": "Ganga Plain"},
    {"stateName": "Bihar", "districtName": "Patna",          "latitude": 25.5941, "longitude": 85.1376, "river": "Ganga",    "flood_zone": "Ganga Plain"},

    # ── JHARKHAND  (confirmed working in WRIS Swagger)
    {"stateName": "Jharkhand", "districtName": "Godda",     "latitude": 24.8300, "longitude": 87.2100, "river": "Ganga",      "flood_zone": "Ganga Plain"},
    {"stateName": "Jharkhand", "districtName": "Sahibganj", "latitude": 25.2400, "longitude": 87.6400, "river": "Ganga",      "flood_zone": "Ganga Plain"},
    {"stateName": "Jharkhand", "districtName": "Dumka",     "latitude": 24.2700, "longitude": 87.2500, "river": "Mayurakshi", "flood_zone": "Ganga Plain"},
    {"stateName": "Jharkhand", "districtName": "Deoghar",   "latitude": 24.4800, "longitude": 86.7000, "river": "Ajay",       "flood_zone": "Ganga Plain"},

    # ── WEST BENGAL  (Ganga delta + DVC floods)
    {"stateName": "West Bengal", "districtName": "Murshidabad",       "latitude": 24.1800, "longitude": 88.2700, "river": "Ganga",    "flood_zone": "Ganga Delta"},
    {"stateName": "West Bengal", "districtName": "Malda",             "latitude": 25.0100, "longitude": 88.1400, "river": "Ganga",    "flood_zone": "Ganga Delta"},
    {"stateName": "West Bengal", "districtName": "Hooghly",           "latitude": 22.9000, "longitude": 88.3900, "river": "Hooghly",  "flood_zone": "Ganga Delta"},
    {"stateName": "West Bengal", "districtName": "Bardhaman",         "latitude": 23.2300, "longitude": 87.8600, "river": "Damodar",  "flood_zone": "Ganga Delta"},
    {"stateName": "West Bengal", "districtName": "North 24 Parganas", "latitude": 22.6600, "longitude": 88.4200, "river": "Ichamati", "flood_zone": "Ganga Delta"},

    # ── UTTAR PRADESH  (Ganga + Ghaghra + Rapti)
    {"stateName": "Uttar Pradesh", "districtName": "Gorakhpur",       "latitude": 26.7606, "longitude": 83.3732, "river": "Rapti",   "flood_zone": "Ganga Plain"},
    {"stateName": "Uttar Pradesh", "districtName": "Bahraich",        "latitude": 27.5700, "longitude": 81.5900, "river": "Ghaghra", "flood_zone": "Ganga Plain"},
    {"stateName": "Uttar Pradesh", "districtName": "Gonda",           "latitude": 27.1300, "longitude": 81.9600, "river": "Ghaghra", "flood_zone": "Ganga Plain"},
    {"stateName": "Uttar Pradesh", "districtName": "Ballia",          "latitude": 25.7500, "longitude": 84.1400, "river": "Ganga",   "flood_zone": "Ganga Plain"},
    {"stateName": "Uttar Pradesh", "districtName": "Varanasi",        "latitude": 25.3176, "longitude": 82.9739, "river": "Ganga",   "flood_zone": "Ganga Plain"},
    {"stateName": "Uttar Pradesh", "districtName": "Lakhimpur Kheri", "latitude": 27.9500, "longitude": 80.7800, "river": "Ghaghra", "flood_zone": "Ganga Plain"},

    # ── ODISHA  (Mahanadi + Brahmani + Baitarani)
    {"stateName": "Odisha", "districtName": "Cuttack",       "latitude": 20.4625, "longitude": 85.8830, "river": "Mahanadi",     "flood_zone": "East Coast"},
    {"stateName": "Odisha", "districtName": "Kendrapara",    "latitude": 20.5000, "longitude": 86.4200, "river": "Mahanadi",     "flood_zone": "East Coast"},
    {"stateName": "Odisha", "districtName": "Jajpur",        "latitude": 20.8400, "longitude": 86.3300, "river": "Brahmani",     "flood_zone": "East Coast"},
    {"stateName": "Odisha", "districtName": "Bhadrak",       "latitude": 21.0600, "longitude": 86.5000, "river": "Baitarani",    "flood_zone": "East Coast"},
    {"stateName": "Odisha", "districtName": "Balasore",      "latitude": 21.4900, "longitude": 86.9300, "river": "Subarnarekha", "flood_zone": "East Coast"},
    {"stateName": "Odisha", "districtName": "Puri",          "latitude": 19.8100, "longitude": 85.8300, "river": "Daya",         "flood_zone": "East Coast"},
    {"stateName": "Odisha", "districtName": "Jagatsinghpur", "latitude": 20.2600, "longitude": 86.1700, "river": "Mahanadi",     "flood_zone": "East Coast"},

    # ── ANDHRA PRADESH  (Krishna + Godavari delta)
    {"stateName": "Andhra Pradesh", "districtName": "West Godavari", "latitude": 16.9200, "longitude": 81.3300, "river": "Godavari", "flood_zone": "Krishna-Godavari Delta"},
    {"stateName": "Andhra Pradesh", "districtName": "East Godavari", "latitude": 17.0000, "longitude": 82.2300, "river": "Godavari", "flood_zone": "Krishna-Godavari Delta"},
    {"stateName": "Andhra Pradesh", "districtName": "Krishna",       "latitude": 16.6100, "longitude": 80.8000, "river": "Krishna",  "flood_zone": "Krishna-Godavari Delta"},
    {"stateName": "Andhra Pradesh", "districtName": "Guntur",        "latitude": 16.3007, "longitude": 80.4428, "river": "Krishna",  "flood_zone": "Krishna-Godavari Delta"},
    {"stateName": "Andhra Pradesh", "districtName": "Eluru",         "latitude": 16.7100, "longitude": 81.1000, "river": "Godavari", "flood_zone": "Krishna-Godavari Delta"},

    # ── KERALA  (extreme rainfall + backwater floods — 2018 century flood)
    {"stateName": "Kerala", "districtName": "Ernakulam",      "latitude": 10.0000, "longitude": 76.3200, "river": "Periyar",   "flood_zone": "West Coast"},
    {"stateName": "Kerala", "districtName": "Thrissur",       "latitude": 10.5276, "longitude": 76.2144, "river": "Chalakudy", "flood_zone": "West Coast"},
    {"stateName": "Kerala", "districtName": "Alappuzha",      "latitude":  9.4981, "longitude": 76.3388, "river": "Pamba",     "flood_zone": "West Coast"},
    {"stateName": "Kerala", "districtName": "Pathanamthitta", "latitude":  9.2648, "longitude": 76.7870, "river": "Pamba",     "flood_zone": "West Coast"},
    {"stateName": "Kerala", "districtName": "Wayanad",        "latitude": 11.6854, "longitude": 76.1320, "river": "Kabani",    "flood_zone": "West Coast"},
    {"stateName": "Kerala", "districtName": "Idukki",         "latitude":  9.9189, "longitude": 76.9700, "river": "Periyar",   "flood_zone": "West Coast"},

    # ── TELANGANA
    {"stateName": "Telangana", "districtName": "Bhadradri Kothagudem", "latitude": 17.5500, "longitude": 80.6200, "river": "Godavari", "flood_zone": "Deccan"},
    {"stateName": "Telangana", "districtName": "Suryapet",             "latitude": 17.1400, "longitude": 79.6300, "river": "Krishna",  "flood_zone": "Deccan"},
    {"stateName": "Telangana", "districtName": "Nalgonda",             "latitude": 17.0500, "longitude": 79.2700, "river": "Krishna",  "flood_zone": "Deccan"},
    {"stateName": "Telangana", "districtName": "Nirmal",               "latitude": 19.0900, "longitude": 78.3400, "river": "Godavari", "flood_zone": "Deccan"},

    # ── MADHYA PRADESH  (Narmada + Chambal)
    {"stateName": "Madhya Pradesh", "districtName": "Harda",       "latitude": 22.3400, "longitude": 77.0900, "river": "Narmada", "flood_zone": "Central India"},
    {"stateName": "Madhya Pradesh", "districtName": "Hoshangabad", "latitude": 22.7500, "longitude": 77.7300, "river": "Narmada", "flood_zone": "Central India"},
    {"stateName": "Madhya Pradesh", "districtName": "Gwalior",     "latitude": 26.2183, "longitude": 78.1828, "river": "Chambal", "flood_zone": "Central India"},
    {"stateName": "Madhya Pradesh", "districtName": "Morena",      "latitude": 26.5000, "longitude": 77.9900, "river": "Chambal", "flood_zone": "Central India"},

    # ── GUJARAT  (Tapi + Sabarmati + Mahi)
    {"stateName": "Gujarat", "districtName": "Vadodara",  "latitude": 22.3072, "longitude": 73.1812, "river": "Vishwamitri", "flood_zone": "West India"},
    {"stateName": "Gujarat", "districtName": "Surat",     "latitude": 21.1702, "longitude": 72.8311, "river": "Tapi",        "flood_zone": "West India"},
    {"stateName": "Gujarat", "districtName": "Bharuch",   "latitude": 21.7051, "longitude": 72.9959, "river": "Narmada",     "flood_zone": "West India"},
    {"stateName": "Gujarat", "districtName": "Ahmedabad", "latitude": 23.0225, "longitude": 72.5714, "river": "Sabarmati",   "flood_zone": "West India"},

    # ── MAHARASHTRA  (Godavari headwaters + urban flooding)
    {"stateName": "Maharashtra", "districtName": "Nashik",   "latitude": 19.9975, "longitude": 73.7898, "river": "Godavari",   "flood_zone": "West Coast"},
    {"stateName": "Maharashtra", "districtName": "Kolhapur", "latitude": 16.7050, "longitude": 74.2433, "river": "Panchganga", "flood_zone": "West Coast"},
    {"stateName": "Maharashtra", "districtName": "Sangli",   "latitude": 16.8524, "longitude": 74.5815, "river": "Krishna",    "flood_zone": "West Coast"},
    {"stateName": "Maharashtra", "districtName": "Mumbai",   "latitude": 19.0760, "longitude": 72.8777, "river": "Mithi",      "flood_zone": "West Coast"},

    # ── UTTARAKHAND  (Himalayan flash floods)
    {"stateName": "Uttarakhand", "districtName": "Haridwar",      "latitude": 29.9457, "longitude": 78.1642, "river": "Ganga",      "flood_zone": "Himalayas"},
    {"stateName": "Uttarakhand", "districtName": "Chamoli",       "latitude": 30.4000, "longitude": 79.3200, "river": "Alaknanda",  "flood_zone": "Himalayas"},
    {"stateName": "Uttarakhand", "districtName": "Tehri Garhwal", "latitude": 30.3800, "longitude": 78.4800, "river": "Bhagirathi", "flood_zone": "Himalayas"},
    {"stateName": "Uttarakhand", "districtName": "Dehradun",      "latitude": 30.3165, "longitude": 78.0322, "river": "Tons",       "flood_zone": "Himalayas"},

    # ── HIMACHAL PRADESH  (Beas / Sutlej flash floods)
    {"stateName": "Himachal Pradesh", "districtName": "Mandi",  "latitude": 31.7100, "longitude": 76.9200, "river": "Beas", "flood_zone": "Himalayas"},
    {"stateName": "Himachal Pradesh", "districtName": "Kullu",  "latitude": 31.9600, "longitude": 77.1100, "river": "Beas", "flood_zone": "Himalayas"},
    {"stateName": "Himachal Pradesh", "districtName": "Kangra", "latitude": 32.1000, "longitude": 76.2700, "river": "Beas", "flood_zone": "Himalayas"},

    # ── JAMMU & KASHMIR  (Jhelum / Chenab)
    {"stateName": "Jammu And Kashmir", "districtName": "Srinagar", "latitude": 34.0837, "longitude": 74.7973, "river": "Jhelum", "flood_zone": "Himalayas"},
    {"stateName": "Jammu And Kashmir", "districtName": "Anantnag", "latitude": 33.7300, "longitude": 75.1500, "river": "Jhelum", "flood_zone": "Himalayas"},

    # ── PUNJAB  (Sutlej + Ravi)
    {"stateName": "Punjab", "districtName": "Ferozepur", "latitude": 30.9200, "longitude": 74.6200, "river": "Sutlej", "flood_zone": "Ganga Plain"},
    {"stateName": "Punjab", "districtName": "Rupnagar",  "latitude": 30.9700, "longitude": 76.5300, "river": "Sutlej", "flood_zone": "Ganga Plain"},
    {"stateName": "Punjab", "districtName": "Pathankot", "latitude": 32.2700, "longitude": 75.6500, "river": "Ravi",   "flood_zone": "Ganga Plain"},

    # ── HARYANA  (Yamuna)
    {"stateName": "Haryana", "districtName": "Yamunanagar", "latitude": 30.1290, "longitude": 77.2674, "river": "Yamuna", "flood_zone": "Ganga Plain"},
    {"stateName": "Haryana", "districtName": "Karnal",      "latitude": 29.6857, "longitude": 76.9905, "river": "Yamuna", "flood_zone": "Ganga Plain"},
    {"stateName": "Haryana", "districtName": "Panipat",     "latitude": 29.3909, "longitude": 76.9635, "river": "Yamuna", "flood_zone": "Ganga Plain"},

    # ── DELHI
    {"stateName": "Delhi", "districtName": "New Delhi", "latitude": 28.6139, "longitude": 77.2090, "river": "Yamuna", "flood_zone": "Ganga Plain"},

    # ── RAJASTHAN  (Chambal / Mahi)
    {"stateName": "Rajasthan", "districtName": "Kota",           "latitude": 25.2138, "longitude": 75.8648, "river": "Chambal", "flood_zone": "Central India"},
    {"stateName": "Rajasthan", "districtName": "Sawai Madhopur", "latitude": 25.9900, "longitude": 76.3500, "river": "Chambal", "flood_zone": "Central India"},
    {"stateName": "Rajasthan", "districtName": "Banswara",       "latitude": 23.5500, "longitude": 74.4400, "river": "Mahi",    "flood_zone": "Central India"},

    # ── KARNATAKA  (Krishna + West Coast rivers)
    {"stateName": "Karnataka", "districtName": "Uttara Kannada", "latitude": 14.7900, "longitude": 74.6900, "river": "Sharavathi", "flood_zone": "West Coast"},
    {"stateName": "Karnataka", "districtName": "Shivamogga",     "latitude": 13.9299, "longitude": 75.5681, "river": "Tunga",      "flood_zone": "West Coast"},
    {"stateName": "Karnataka", "districtName": "Kodagu",         "latitude": 12.4200, "longitude": 75.7400, "river": "Cauvery",    "flood_zone": "West Coast"},
    {"stateName": "Karnataka", "districtName": "Belagavi",       "latitude": 15.8497, "longitude": 74.4977, "river": "Krishna",    "flood_zone": "West Coast"},
    {"stateName": "Karnataka", "districtName": "Bagalkot",       "latitude": 16.1800, "longitude": 75.7000, "river": "Krishna",    "flood_zone": "West Coast"},

    # ── CHHATTISGARH  (Mahanadi headwaters)
    {"stateName": "Chhattisgarh", "districtName": "Raipur", "latitude": 21.2514, "longitude": 81.6296, "river": "Mahanadi",  "flood_zone": "Central India"},
    {"stateName": "Chhattisgarh", "districtName": "Durg",   "latitude": 21.1900, "longitude": 81.2800, "river": "Sheonath",  "flood_zone": "Central India"},
    {"stateName": "Chhattisgarh", "districtName": "Bastar", "latitude": 19.1200, "longitude": 81.9500, "river": "Indravati", "flood_zone": "Central India"},

    # ── MEGHALAYA  (highest rainfall — Barak headwaters)
    {"stateName": "Meghalaya", "districtName": "East Khasi Hills", "latitude": 25.5700, "longitude": 91.8800, "river": "Umiam",  "flood_zone": "NE India"},
    {"stateName": "Meghalaya", "districtName": "South Garo Hills", "latitude": 25.3300, "longitude": 90.4800, "river": "Simsang", "flood_zone": "NE India"},

    # ── ARUNACHAL PRADESH  (Brahmaputra tributaries)
    {"stateName": "Arunachal Pradesh", "districtName": "Papum Pare", "latitude": 27.1000, "longitude": 93.6000, "river": "Brahmaputra", "flood_zone": "NE India"},
    {"stateName": "Arunachal Pradesh", "districtName": "Lohit",      "latitude": 28.0000, "longitude": 96.0000, "river": "Lohit",       "flood_zone": "NE India"},

    # ── MANIPUR / NAGALAND / TRIPURA
    {"stateName": "Manipur",  "districtName": "Imphal West", "latitude": 24.8170, "longitude": 93.9368, "river": "Imphal",   "flood_zone": "NE India"},
    {"stateName": "Nagaland", "districtName": "Dimapur",     "latitude": 25.9000, "longitude": 93.7200, "river": "Dhansiri", "flood_zone": "NE India"},
    {"stateName": "Tripura",  "districtName": "West Tripura","latitude": 23.8315, "longitude": 91.2868, "river": "Haora",    "flood_zone": "NE India"},
]


# ================================================================
#  LOGGING
# ================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("wris")
log.setLevel(logging.DEBUG)

fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(fh)


class AsciiSafeHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            self.stream.write(
                msg.encode("ascii", errors="replace").decode("ascii") + self.terminator
            )
            self.flush()
        except Exception:
            self.handleError(record)


ch = AsciiSafeHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(ch)


# ================================================================
#  HTTP SESSION
# ================================================================

SESSION = requests.Session()
SESSION.headers.update({
    "accept":     "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer":    "https://indiawris.gov.in/wris/",
    "Origin":     "https://indiawris.gov.in",
})


# ================================================================
#  AGENCY RESOLVER
#  Replaces "{state}" placeholder with the actual state name
# ================================================================

def resolve_agencies(agency_template_list, state):
    resolved = []
    seen = set()
    for a in agency_template_list:
        resolved_a = a.replace("{state}", state)
        if resolved_a not in seen:
            resolved.append(resolved_a)
            seen.add(resolved_a)
    return resolved


# ================================================================
#  CORE FETCH — tries each agency in order, stops at first hit
# ================================================================

def fetch_wris_data(endpoint_path, state, district, agency_templates, start_date, end_date):
    """Try each resolved agency in order; return (records, working_agency)."""
    agencies = resolve_agencies(agency_templates, state)
    for agency in agencies:
        records = _fetch_one_agency(
            endpoint_path, state, district, agency, start_date, end_date
        )
        if records:
            return records, agency
        log.debug("    Agency '%s' -> no data.", agency)
    return [], None


def _fetch_one_agency(endpoint_path, state, district, agency, start_date, end_date):
    url = f"{BASE_URL}/{endpoint_path}"
    all_records = []

    for page in range(0, MAX_PAGES):
        params = {
            "stateName":    state,
            "districtName": district,
            "agencyName":   agency,
            "startdate":    start_date.strftime("%Y-%m-%d"),
            "enddate":      end_date.strftime("%Y-%m-%d"),
            "download":     "false",
            "page":         page,
            "size":         PAGE_SIZE,
        }

        try:
            response = SESSION.post(url, params=params, data="", timeout=30)
            log.debug("      POST %s  status=%s", response.url, response.status_code)
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            log.warning("      HTTP %s  url=%s  agency=%s",
                        response.status_code, url, agency)
            break
        except requests.exceptions.RequestException as e:
            log.error("      Request error: %s", e)
            break

        try:
            payload = response.json()
        except json.JSONDecodeError:
            log.warning("      Non-JSON response: %s", response.text[:200])
            break

        # statusCode 500 = "no data" — stop trying more pages for this agency
        body_status = payload.get("statusCode")
        if page == 0 and body_status and body_status != 200:
            log.debug("      API statusCode=%s: %s",
                      body_status, payload.get("message", ""))
            break

        records = payload.get("data", [])
        if isinstance(records, dict):
            records = (records.get("content") or records.get("data")
                       or records.get("records") or [])

        if not records:
            break

        all_records.extend(records)
        log.debug("      Page %d: +%d (total: %d)", page, len(records), len(all_records))

        if len(records) < PAGE_SIZE:
            break   # partial page = last page

        time.sleep(DELAY_BETWEEN_CALLS)

    return all_records


# ================================================================
#  NORMALISE
# ================================================================

def normalise_records(raw_records, feature_name, location):
    if not raw_records:
        return pd.DataFrame()

    df = pd.DataFrame(raw_records)

    # Date — station data uses "dataTime"; model data (NRSC) uses "date"
    date_candidates = ["dataTime", "date", "Date", "observation_date",
                       "obsDate", "recordDate", "Dated"]
    date_col = next((c for c in date_candidates if c in df.columns), None)
    df["date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT

    # Value — confirmed: "dataValue" for both station and NRSC model data
    value_candidates = ["dataValue", "value", "Value", "data_value",
                        "ObservedValue", "reading", "discharge", "waterLevel",
                        "rainfall", "temperature", "humidity", "pressure"]
    val_col = next((c for c in value_candidates if c in df.columns), None)
    df[feature_name] = pd.to_numeric(df[val_col], errors="coerce") if val_col else float("nan")

    df["station_code"] = df.get("stationCode", "")
    df["station_name"] = df.get("stationName", "")
    df["unit"]         = df.get("unit", "")
    df["stateName"]    = location["stateName"]
    df["districtName"] = location["districtName"]
    df["latitude"]     = location["latitude"]
    df["longitude"]    = location["longitude"]
    df["river"]        = location.get("river", "")
    df["flood_zone"]   = location.get("flood_zone", "")

    keep = ["date", "stateName", "districtName", "latitude", "longitude",
            "river", "flood_zone", "station_code", "station_name", "unit",
            feature_name]
    return df[[c for c in keep if c in df.columns]]


# ================================================================
#  AGGREGATE TO DAILY
# ================================================================

def aggregate_to_daily(df, feature_name):
    if df.empty:
        return df

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    group_cols = ["date", "stateName", "districtName", "latitude", "longitude",
                  "river", "flood_zone", "station_code", "station_name", "unit"]
    # Rainfall → sum (accumulate); everything else → mean
    agg_func = "sum" if feature_name == "rainfall_mm" else "mean"
    return df.groupby(group_cols, as_index=False)[feature_name].agg(agg_func)


# ================================================================
#  MAIN FETCH LOOP
# ================================================================

def build_flood_dataset():
    log.info("India-WRIS Flood Feature Dataset Builder  v6")
    log.info("Start date : %s", START_DATE.date())
    log.info("End date   : %s", END_DATE.date())
    log.info("Locations  : %d  (%d states)",
             len(LOCATIONS), len({l["stateName"] for l in LOCATIONS}))
    log.info("")
    log.info("Endpoint → agency mapping:")
    for feat, cfg in ENDPOINT_AGENCIES.items():
        log.info("  %-28s  agencies=%s", feat, cfg["agencies"])
    log.info("=" * 65)

    all_features   = []
    agency_hit_log = {}   # district -> {feature: agency_that_worked}

    for feature_key, cfg in ENDPOINT_AGENCIES.items():
        endpoint_path    = cfg["path"]
        agency_templates = cfg["agencies"]

        log.info("=== Feature: %-28s  POST /Dataset/%s ===",
                 feature_key, endpoint_path)
        feature_dfs = []

        for loc in tqdm(LOCATIONS, desc=f"  {feature_key}", ncols=90):
            state    = loc["stateName"]
            district = loc["districtName"]

            records, agency_used = fetch_wris_data(
                endpoint_path, state, district, agency_templates,
                START_DATE, END_DATE
            )

            if not records:
                log.debug("  No data: %s / %s / %s", state, district, feature_key)
                continue

            agency_hit_log.setdefault(district, {})[feature_key] = agency_used
            log.info("  ✓ %-25s  n=%-5d  agency=%s",
                     district, len(records), agency_used)

            df_norm  = normalise_records(records, feature_key, loc)
            df_daily = aggregate_to_daily(df_norm, feature_key)
            feature_dfs.append(df_daily)

        if feature_dfs:
            combined = pd.concat(feature_dfs, ignore_index=True)
            all_features.append((feature_key, combined))
            log.info("  Feature %s: %d daily rows, %d districts",
                     feature_key, len(combined), combined["districtName"].nunique())
        else:
            log.warning("  Feature %s: NO data from any location.", feature_key)

    if not all_features:
        log.error("No features fetched at all. Check agency names and WRIS portal.")
        return

    # Merge all features on shared columns
    log.info("")
    log.info("Merging %d features into final dataset...", len(all_features))

    merge_cols = ["date", "stateName", "districtName", "latitude", "longitude",
                  "river", "flood_zone", "station_code", "station_name", "unit"]

    final_df = all_features[0][1]
    for feat_key, df_feat in all_features[1:]:
        final_df = pd.merge(final_df, df_feat, on=merge_cols, how="outer")

    # Ensure all feature columns exist
    for col in ENDPOINT_AGENCIES.keys():
        if col not in final_df.columns:
            final_df[col] = float("nan")

    final_df.sort_values(["flood_zone", "stateName", "districtName", "date"],
                         inplace=True)
    final_df.reset_index(drop=True, inplace=True)

    # ── Save ──────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(FINAL_FILE, index=False, encoding="utf-8-sig")
    log.info("Saved master CSV -> %s  (%d rows)", FINAL_FILE, len(final_df))

    for district, grp in final_df.groupby("districtName"):
        safe = district.lower().replace(" ", "_")
        grp.to_csv(OUTPUT_DIR / f"{safe}.csv", index=False, encoding="utf-8-sig")

    for zone, grp in final_df.groupby("flood_zone"):
        safe = zone.lower().replace(" ", "_").replace("-", "_")
        grp.to_csv(OUTPUT_DIR / f"zone_{safe}.csv", index=False, encoding="utf-8-sig")

    # ── Summary ───────────────────────────────────────────────
    log.info("")
    log.info("=" * 65)
    log.info("SUMMARY")
    log.info("  Total rows  : %d", len(final_df))
    log.info("  Date range  : %s  ->  %s",
             final_df["date"].min().date(), final_df["date"].max().date())
    log.info("  Districts   : %d", final_df["districtName"].nunique())
    log.info("  States      : %s", sorted(final_df["stateName"].unique()))
    log.info("")
    log.info("  Feature coverage (missing %%):")
    for col in ENDPOINT_AGENCIES.keys():
        if col in final_df.columns:
            pct = final_df[col].isna().mean() * 100
            bar = "█" * int((100 - pct) / 5) + "░" * int(pct / 5)
            log.info("    %-28s %5.1f%%  %s", col, pct, bar)
    log.info("")
    log.info("  Agencies that returned data:")
    for dist, feats in agency_hit_log.items():
        log.info("    %-25s  %s", dist, set(feats.values()))
    log.info("=" * 65)


# ================================================================
#  ENTRY POINT
# ================================================================

if __name__ == "__main__":
    log.info("Starting WRIS Flood Feature Dataset Builder")
    build_flood_dataset()
    log.info("All done!")