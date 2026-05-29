"""
Surfcasting Analytics API – v3.0 (Final Production)
Tunisian dialect, moon phases (حمل/فساد), seasonal tips, pressure bonus, scan, etc.
"""
import os, math, asyncio, logging, traceback, zoneinfo, re, time
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional
from collections import defaultdict

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("surfcasting")

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Surfcasting Analytics", version="3.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY مفقود")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "google/gemini-2.5-flash"

MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

cache = {}
CACHE_TTL = 3600

class ReportRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    beach_orientation: int = Field(..., ge=0, le=360)
    beach_type: str = Field(..., pattern="^(sandy|rocky)$")
    target_date: str = Field(..., pattern="^(today|tomorrow|day_after)$")

class AutoOrientationRequest(BaseModel):
    latitude: float
    longitude: float

class ScanRequest(BaseModel):
    governorates: List[str]
    target_date: str = "today"
    target_species: Optional[str] = None

class SurfError(Exception): pass

@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})

@app.get("/health")
def health():
    return {"status": "ok", "openrouter": bool(OPENROUTER_API_KEY), "model": MODEL_NAME}

# --------------- دوال مساعدة ---------------
async def fetch_timezone_info(lat, lon):
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(MARINE_URL, params={
                "latitude": lat, "longitude": lon,
                "hourly": "wave_height", "timezone": "auto", "forecast_days": 1
            }, timeout=10)
            r.raise_for_status()
            tz = r.json().get("timezone", "UTC")
            zoneinfo.ZoneInfo(tz)
            return tz
    except: return "UTC"

def extract_real_date_from_times(times, tz_name):
    if not times:
        return datetime.now(zoneinfo.ZoneInfo(tz_name)).date()
    dt = datetime.fromisoformat(times[0])
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=zoneinfo.ZoneInfo(tz_name))
    return dt.date()

def resolve_target_date(txt, real_today):
    if txt == "today": return real_today
    elif txt == "tomorrow": return real_today + timedelta(days=1)
    return real_today + timedelta(days=2)

async def fetch_marine(lat, lon, start, end):
    async with httpx.AsyncClient() as c:
        r = await c.get(MARINE_URL, params={
            "latitude": lat, "longitude": lon,
            "hourly": ["wave_height","wave_period","wave_direction",
                      "swell_wave_height","swell_wave_period","swell_wave_direction",
                      "sea_surface_temperature"],
            "timezone": "auto", "start_date": start, "end_date": end
        }, timeout=20)
        r.raise_for_status()
        return r.json()

async def fetch_weather(lat, lon, start, end):
    async with httpx.AsyncClient() as c:
        r = await c.get(WEATHER_URL, params={
            "latitude": lat, "longitude": lon,
            "hourly": ["wind_speed_10m","wind_direction_10m","wind_gusts_10m",
                      "pressure_msl","temperature_2m","precipitation","weather_code"],
            "daily": ["sunrise","sunset"],
            "timezone": "auto", "start_date": start, "end_date": end
        }, timeout=20)
        r.raise_for_status()
        return r.json()

async def get_bottom_type(lat: float, lon: float) -> str:
    query = f"""
    [out:json];
    (
      node(around:500,{lat},{lon})["surface"="sand"];
      node(around:500,{lat},{lon})["natural"="beach"];
      node(around:500,{lat},{lon})["surface"="gravel"];
      node(around:500,{lat},{lon})["surface"="rock"];
    );
    out body;
    """
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(OVERPASS_URL, params={"data": query}, timeout=15.0)
            resp.raise_for_status()
            data = resp.json()
            elements = data.get("elements", [])
            if not elements:
                return "sandy"
            for el in elements:
                tags = el.get("tags", {})
                surface = tags.get("surface", "").lower()
                if "rock" in surface or "gravel" in surface or "pebbles" in surface:
                    return "rocky"
            return "sandy"
    except Exception as e:
        logger.error(f"Bottom type detection error: {e}")
        return "sandy"

@app.post("/detect-bottom-type")
@limiter.limit("10/minute")
async def detect_bottom_type(request: Request, req: AutoOrientationRequest):
    bottom = await get_bottom_type(req.latitude, req.longitude)
    return {"bottom_type": bottom}

async def get_auto_orientation(lat, lon):
    query = f"""
    [out:json];
    (way(around:5000,{lat},{lon})["natural"="coastline"];);
    out geom;
    """
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(OVERPASS_URL, params={"data": query}, timeout=20)
            r.raise_for_status()
            data = r.json()
            els = data.get("elements", [])
            if not els: return 0
            for el in els:
                geom = el.get("geometry", [])
                if geom and len(geom) >= 2:
                    p1, p2 = geom[0], geom[-1]
                    dx = p2["lon"] - p1["lon"]
                    dy = p2["lat"] - p1["lat"]
                    angle = (math.degrees(math.atan2(dx, dy)) + 360) % 360
                    return int(round((angle + 90) % 360))
            return 0
    except: return 0

@app.post("/auto-orientation")
@limiter.limit("5/minute")
async def auto_orientation(request: Request, req: AutoOrientationRequest):
    return {"orientation": await get_auto_orientation(req.latitude, req.longitude)}

def safe_float(v):
    try: return 0.0 if math.isnan(float(v)) else float(v)
    except: return 0.0

def angle_diff(w, b):
    d = abs(w - b) % 360
    return 360 - d if d > 180 else d

def wind_class_detailed(diff):
    if diff < 30: return "بحرية مباشرة"
    elif diff < 45: return "بحرية خفيفة"
    elif diff < 60: return "جانبية مائلة للبحر"
    elif diff <= 120: return "جانبية"
    elif diff < 150: return "جانبية مائلة للبر"
    elif diff < 165: return "برية خفيفة"
    else: return "برية مباشرة"

def weather_desc(code):
    if code <= 1: return "صافية"
    elif code == 2: return "غائمة جزئياً"
    elif code == 3: return "غائمة"
    elif code <= 48: return "ضباب"
    elif code <= 55: return "رذاذ"
    elif code <= 65: return "مطر"
    elif code <= 77: return "ثلج"
    elif code <= 82: return "زخات مطر"
    elif code <= 99: return "عواصف"
    return "غير معروف"

# --------------- تفضيلات الأسماك ---------------
SPECIES_PREFERENCES = {
    "قاروص": {
        "ideal_sst": (14, 18),
        "preferred_wind": ["بحرية مباشرة", "بحرية خفيفة", "جانبية مائلة للبحر"],
        "ideal_wave_range": (0.5, 1.5),
        "ideal_power_range": (0.5, 2.0),
        "bottom_type": "sandy"
    },
    "دوراد": {
        "ideal_sst": (19, 26),
        "preferred_wind": ["برية مباشرة", "برية خفيفة", "جانبية مائلة للبر"],
        "ideal_wave_range": (0.3, 0.8),
        "ideal_power_range": (0.1, 1.0),
        "bottom_type": "sandy"
    },
    "سارغ": {
        "ideal_sst": (16, 22),
        "preferred_wind": ["بحرية مباشرة", "بحرية خفيفة", "جانبية"],
        "ideal_wave_range": (0.5, 1.2),
        "ideal_power_range": (0.3, 1.5),
        "bottom_type": "rocky"
    },
    "بوري": {
        "ideal_sst": (15, 28),
        "preferred_wind": ["جانبية", "برية خفيفة", "بحرية خفيفة"],
        "ideal_wave_range": (0.1, 0.6),
        "ideal_power_range": (0.0, 0.5),
        "bottom_type": "sandy"
    },
    "ماربري": {
        "ideal_sst": (18, 24),
        "preferred_wind": ["برية مباشرة", "برية خفيفة", "جانبية"],
        "ideal_wave_range": (0.3, 0.8),
        "ideal_power_range": (0.1, 0.8),
        "bottom_type": "sandy"
    }
}

# --------------- مراحل القمر وأيام الحمل/الفساد ---------------
def moon_phase_detail(d: date) -> dict:
    y, m, day = d.year, d.month, d.day
    if m < 3:
        y -= 1
        m += 12
    a = int(y / 100)
    b = 2 - a + int(a / 4)
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + day + b - 1524.5
    new_moon_jd = 2451550.1
    days_since_new = jd - new_moon_jd
    lunar_month = 29.53058867
    phase = (days_since_new % lunar_month) / lunar_month
    idx = int(phase * 8) % 8
    phases = {
        0: "محاق", 1: "هلال أول", 2: "تربيع أول", 3: "أحدب متزايد",
        4: "بدر", 5: "أحدب متناقص", 6: "تربيع ثاني", 7: "هلال آخر"
    }
    name = phases.get(idx, "محاق")
    if 0.0 <= phase < 0.125 or 0.875 <= phase <= 1.0:
        status = "أيام متوسطة (محاق / هلال آخر)"
        activity = "ينشط القاع ليلاً بشكل خاص، والنهار يكون متوسط"
    elif 0.125 <= phase < 0.5:
        status = "أيام حمل"
        activity = "الأسماك نشيطة طوال اليوم، والصيد ممتاز ليلاً ونهاراً"
    else:
        status = "أيام فساد"
        activity = "الأسماك أقل نشاطاً، يُفضّل الصيد في الساعات الذهبية (الشروق والغروب)"
    return {"name": name, "status": status, "activity": activity, "phase": round(phase, 3)}

def moon_fishing_guidance(d: date) -> str:
    detail = moon_phase_detail(d)
    name = detail["name"]
    status = detail["status"]
    if "محاق" in name:
        return f"{status}. ركز على الصيد الليلي للقاروص والسارغ باستعمال طعوم بروائح قوية."
    elif "هلال أول" in name or "تربيع أول" in name:
        return f"{status}. فرصة ممتازة لكل الأنواع، استهدف القاروص نهاراً والدوراد عند الغروب."
    elif "أحدب متزايد" in name:
        return f"{status}. البوري والدوراد نشيطين نهاراً، القاروص ينشط ليلاً."
    elif "بدر" in name:
        return f"{status}. الأسماك السطحية (الدوراد، البوري) نشيطة نهاراً، والصيد الليلي ضعيف."
    elif "أحدب متناقص" in name or "تربيع ثاني" in name:
        return f"{status}. الصيد متوسط، استعمل الطعوم القوية (الشريب، دود الكف) في الفجر والغروب."
    else:
        return f"{status}. الصيد مقبول، لكن الأفضل في الساعات الذهبية."

def evaluate_spot(marine, weather, orient, sunrise_str, sunset_str, beach_type, target_species=None):
    mh = marine.get("hourly", {})
    wh = weather.get("hourly", {})
    times = mh.get("time", [])
    if not times:
        return 0.0, {}

    wave_h = [safe_float(x) for x in mh.get("wave_height", [])]
    wave_p = [safe_float(x) for x in mh.get("wave_period", [])]
    wind_speed = [safe_float(x) for x in wh.get("wind_speed_10m", [])]
    wind_dir_vals = [safe_float(x) for x in wh.get("wind_direction_10m", [])]
    wind_gust = [safe_float(x) for x in wh.get("wind_gusts_10m", [])]
    pressure = [safe_float(x) for x in wh.get("pressure_msl", [])]
    sst = [safe_float(x) for x in mh.get("sea_surface_temperature", [])]

    wind_classes_detailed = [wind_class_detailed(angle_diff(wd, orient)) for wd in wind_dir_vals]

    N = len(times)
    score = 0.0
    red_hours = 0
    green_hours = 0

    try:
        sr_h = int(sunrise_str.split(":")[0])
        ss_h = int(sunset_str.split(":")[0])
    except:
        sr_h, ss_h = 6, 18

    for i in range(N):
        power = 0.49 * (wave_h[i] ** 2) * wave_p[i]
        if power > 3.0 or wave_h[i] > 1.8 or wind_gust[i] > 50 or pressure[i] < 1005:
            red_hours += 1
            score -= 15
        elif 0.3 <= wave_h[i] <= 1.0 and 0.1 <= power <= 1.5 and wind_speed[i] < 27.8:
            green_hours += 1
            score += 10
            dt = datetime.fromisoformat(times[i])
            h = dt.hour
            if abs(h - sr_h) <= 2 or abs(h - ss_h) <= 2:
                score += 5
        else:
            if 0.2 <= wave_h[i] <= 1.2: score += 3
            elif wave_h[i] < 0.2: score += 1
            else: score -= 2
            if wind_speed[i] < 15: score += 4
            elif wind_speed[i] < 25: score += 2
            else: score -= 1

    avg_wave = sum(wave_h) / N
    avg_power = sum(0.49 * (wave_h[i]**2) * wave_p[i] for i in range(N)) / N
    avg_wind = sum(wind_speed) / N
    avg_sst = sum(sst) / N
    avg_press = sum(pressure) / N
    dominant_wind = max(set(wind_classes_detailed), key=wind_classes_detailed.count)

    # مكافأة الضغط المستقر
    if 1015 <= avg_press <= 1025:
        score += 8

    if target_species and target_species in SPECIES_PREFERENCES:
        prefs = SPECIES_PREFERENCES[target_species]
        match_score = 0
        lo, hi = prefs["ideal_sst"]
        if lo <= avg_sst <= hi:
            match_score += 30
        elif abs(avg_sst - lo) <= 2 or abs(avg_sst - hi) <= 2:
            match_score += 15
        if prefs["bottom_type"] == beach_type:
            match_score += 20
        if dominant_wind in prefs["preferred_wind"]:
            match_score += 25
        elif any(w in prefs["preferred_wind"] for w in wind_classes_detailed):
            match_score += 10
        if prefs["ideal_wave_range"][0] <= avg_wave <= prefs["ideal_wave_range"][1]:
            match_score += 15
        if prefs["ideal_power_range"][0] <= avg_power <= prefs["ideal_power_range"][1]:
            match_score += 10
        factor = 0.5 + (match_score / 100.0)
    else:
        factor = 1.0

    normalized = max(0.0, min(100.0, (score / 200.0) * 100.0 * factor))

    summary = {
        "avg_wave": round(avg_wave, 2),
        "avg_power": round(avg_power, 2),
        "avg_wind": round(avg_wind, 1),
        "avg_sst": round(avg_sst, 1),
        "dominant_wind": dominant_wind,
        "green_hours": green_hours,
        "red_hours": red_hours,
    }
    return round(normalized, 1), summary

# --------------- قاعدة بيانات الشواطئ ---------------
TUNISIAN_BEACHES = {
    "بنزرت": [
        {"name": "شاطئ الكورنيش (بنزرت)", "lat": 37.2744, "lon": 9.8739, "orientation": 45, "type": "sandy"},
        {"name": "شاطئ سيدي سالم", "lat": 37.2800, "lon": 9.8800, "orientation": 45, "type": "sandy"},
        {"name": "شاطئ الحسيان", "lat": 37.2600, "lon": 9.8600, "orientation": 0, "type": "sandy"},
        {"name": "شاطئ الكاب سيرات", "lat": 37.3500, "lon": 9.7500, "orientation": 315, "type": "rocky"},
        {"name": "شاطئ سيدي عياد", "lat": 37.3300, "lon": 9.7800, "orientation": 0, "type": "sandy"},
        {"name": "شاطئ غار الملح", "lat": 37.1667, "lon": 10.1833, "orientation": 315, "type": "sandy"},
        {"name": "شاطئ سيدي علي المكي", "lat": 37.1500, "lon": 10.2000, "orientation": 0, "type": "sandy"},
        {"name": "شاطئ البطاح", "lat": 37.1300, "lon": 10.2200, "orientation": 45, "type": "sandy"},
        {"name": "شاطئ أوتيك (الشواية)", "lat": 37.0800, "lon": 10.1000, "orientation": 90, "type": "sandy"},
        {"name": "شاطئ رفراف", "lat": 37.2167, "lon": 10.0833, "orientation": 0, "type": "sandy"},
        {"name": "شاطئ رأس الجبل", "lat": 37.2500, "lon": 10.0500, "orientation": 315, "type": "sandy"},
        {"name": "شاطئ الزوارع", "lat": 37.2700, "lon": 10.0200, "orientation": 0, "type": "sandy"},
        {"name": "شاطئ لالة مريم", "lat": 37.2000, "lon": 10.0500, "orientation": 45, "type": "sandy"},
    ],
    "نابل": [
        {"name": "شاطئ نابل المدينة", "lat": 36.4500, "lon": 10.7333, "orientation": 90, "type": "sandy"},
        {"name": "شاطئ الحمامات", "lat": 36.4000, "lon": 10.6167, "orientation": 90, "type": "sandy"},
        {"name": "شاطئ ياسمين الحمامات", "lat": 36.3800, "lon": 10.5500, "orientation": 90, "type": "sandy"},
        {"name": "شاطئ الحمامات الجنوبي", "lat": 36.3500, "lon": 10.5500, "orientation": 90, "type": "sandy"},
        {"name": "شاطئ قليبية", "lat": 36.8500, "lon": 11.1000, "orientation": 45, "type": "sandy"},
        {"name": "شاطئ منزل حر", "lat": 36.8300, "lon": 11.1200, "orientation": 0, "type": "sandy"},
        {"name": "شاطئ الهوارية", "lat": 37.0333, "lon": 11.0167, "orientation": 315, "type": "rocky"},
        {"name": "شاطئ وادي الخف", "lat": 37.0200, "lon": 11.0300, "orientation": 0, "type": "rocky"},
        {"name": "شاطئ بني خيار", "lat": 36.4833, "lon": 10.7833, "orientation": 90, "type": "sandy"},
        {"name": "شاطئ دار شعبان", "lat": 36.4700, "lon": 10.7500, "orientation": 90, "type": "sandy"},
        {"name": "شاطئ قربة", "lat": 36.5500, "lon": 10.8500, "orientation": 90, "type": "sandy"},
        {"name": "شاطئ شط قربة", "lat": 36.5600, "lon": 10.8700, "orientation": 0, "type": "sandy"},
        {"name": "شاطئ منزل تميم", "lat": 36.7000, "lon": 10.9500, "orientation": 0, "type": "sandy"},
        {"name": "شاطئ سيدي الجديدي", "lat": 36.7200, "lon": 10.9800, "orientation": 45, "type": "sandy"},
        {"name": "شاطئ سليمان", "lat": 36.6333, "lon": 10.5000, "orientation": 90, "type": "sandy"},
        {"name": "شاطئ شط مريم", "lat": 36.6500, "lon": 10.4500, "orientation": 90, "type": "sandy"},
        {"name": "شاطئ تاكلسة", "lat": 36.7500, "lon": 10.6500, "orientation": 45, "type": "sandy"},
        {"name": "شاطئ المعمورة", "lat": 36.5500, "lon": 10.6000, "orientation": 90, "type": "sandy"},
        {"name": "شاطئ سيدي بوسعيد", "lat": 36.8700, "lon": 10.3500, "orientation": 0, "type": "sandy"},
    ],
    "تونس": [
        {"name": "شاطئ حلق الوادي", "lat": 36.8167, "lon": 10.3167, "orientation": 0, "type": "sandy"},
        {"name": "شاطئ الكرم", "lat": 36.8500, "lon": 10.3200, "orientation": 45, "type": "sandy"},
        {"name": "شاطئ قرطاج", "lat": 36.8528, "lon": 10.3264, "orientation": 90, "type": "sandy"},
        {"name": "شاطئ المرسى", "lat": 36.8764, "lon": 10.3253, "orientation": 45, "type": "sandy"},
        {"name": "شاطئ روّاد", "lat": 36.9667, "lon": 10.1833, "orientation": 45, "type": "sandy"},
        {"name": "شاطئ قلعة الأندلس", "lat": 36.9167, "lon": 10.1667, "orientation": 0, "type": "sandy"},
    ],
    "سوسة": [
        {"name": "شاطئ بوجعفر", "lat": 35.8333, "lon": 10.6333, "orientation": 90, "type": "sandy"},
        {"name": "شاطئ القنطاوي", "lat": 35.8833, "lon": 10.6000, "orientation": 90, "type": "sandy"},
        {"name": "شاطئ حمام سوسة", "lat": 35.8500, "lon": 10.6000, "orientation": 90, "type": "sandy"},
        {"name": "شاطئ شط الرمال", "lat": 35.9000, "lon": 10.5500, "orientation": 45, "type": "sandy"},
        {"name": "شاطئ هرقلة", "lat": 36.0000, "lon": 10.4500, "orientation": 0, "type": "sandy"},
        {"name": "شاطئ سيدي بوعلي", "lat": 35.8500, "lon": 10.4500, "orientation": 45, "type": "sandy"},
    ],
    "أريانة": [
        {"name": "شاطئ روّاد (الغدير)", "lat": 36.9833, "lon": 10.1833, "orientation": 45, "type": "sandy"},
        {"name": "شاطئ حي النصر", "lat": 36.9500, "lon": 10.2000, "orientation": 0, "type": "sandy"},
        {"name": "شاطئ قلعة الأندلس", "lat": 36.9167, "lon": 10.1667, "orientation": 0, "type": "sandy"},
        {"name": "شاطئ شط مروان", "lat": 36.9000, "lon": 10.1500, "orientation": 45, "type": "sandy"},
    ],
    "بن عروس": [
        {"name": "شاطئ رادس", "lat": 36.7500, "lon": 10.2833, "orientation": 0, "type": "sandy"},
        {"name": "شاطئ الزهراء", "lat": 36.7333, "lon": 10.3000, "orientation": 45, "type": "sandy"},
        {"name": "شاطئ حمام الأنف", "lat": 36.7167, "lon": 10.3333, "orientation": 0, "type": "sandy"},
        {"name": "شاطئ برج السدرية", "lat": 36.7000, "lon": 10.3667, "orientation": 45, "type": "sandy"},
        {"name": "شاطئ حمام الشط", "lat": 36.6833, "lon": 10.3833, "orientation": 90, "type": "sandy"},
    ],
}

SYSTEM_PROMPT = """أنت صياد سرفكاستينغ تونسي محترف ومحلل بحري قدير. اكتب تقريراً بحرياً كاملاً باللغة العربية والمصطلحات التونسية الدارجة (المرصاص، اللدونة، التيارات الجارفة، القفلة، دود الكف، القمبري، الشريب، القرابين، الصابونة...). التقرير نص واحد متصل بلا نقاط أو رموز، يغطي:
1. تحليل عام: حالة السماء، حرارة الهواء والماء، القمر وتأثيره (أيام الحمل/الفساد، المحاق، البدر...)، الشروق والغروب. **استخدم معلومات حالة القمر للصيد والتوجيه لتحدد بالضبط متى يكون النشاط أفضل (نهار/ليل) وما هي الأسماك المستهدفة في كل فترة.**
2. الأمواج والتيارات: نطاق الموج و swell، الطاقة. بناءً عليها:
   - وزن ونوع الرصاصة (صابونة، هرم، قرابين) مع تعليل. إذا الموج أقل من 0.5م استعمل صابونة 80-100غ وارم بعيداً، وإذا بين 0.5-1م استعمل هرم 100-120غ وارم أقرب. للصخور استعمل قرابين (مخالب) حتى لا تعلق.
   - اقترح تركيبة (Montage): "بكرة بمخالب" للصخور، "صابونة جارية" للرمل، "كراتين" للبوري.
   - خطة طوارئ: إذا ارتفع الموج فجأة، غيّر الرصاصة إلى مخالب فوراً وقصّر مسافة الرمي.
3. الرياح والأعشاب: سرعة واتجاه الرياح (بحرية مباشرة، برية، جانبية...)، تأثيرها على الأعشاب ونقاء الماء، الأمطار. تكتيك الرمي: إذا الرياح بحرية، ارمِ بزاوية 45° عكس الريح. وإذا برية، ارمِ مع الريح لمسافة أطول. إذا جانبية، ارمِ بعكس التيار.
4. التقييم والتوصيات:
   - أفضل الأوقات بدقة مع تكتيك متقدم: قبل الشروق استعمل طعوم رائحة قوية (دود الكف، الشريب)، بعد الشروق طعوم بصرية (القمبري، السردين)، في الليل استعمل طعوم لامعة ومسافات أقرب مع مصباح رأس للسلامة.
   - استراتيجية خاصة للبوري: إذا كنت تستهدف البوري، استعمل تركيب الكراتين بعجينة الخبز ودود الكف، وارمِ لمسافة 80-100م.
   - الطعوم حسب القاع (دود الكف للرمل، القمبري للصخور، الشريب للماء البارد).
   - إذا أمطار، نبه لمصبات الوديان.
   - تحذير سلامة إذا تجاوزت الرياح 30 كم/س أو الموج 1.5م. أضف نصائح سلامة ليلية (مصباح رأس، إخبار أحد بمكانك).
   - علامات الصياد المحترف: راقب تغير لون الماء، تجمع الطيور، القفزات الصغيرة.
   - نصيحة موسمية حسب الشهر الحالي (الشتاء، الربيع، الصيف، الخريف) واذكر الطعوم المناسبة.
   - قارن سريعاً مع الأيام الماضية (هل البحر يهدأ أم يضطرب؟).
   - اختم بسؤال للصياد: "أي نوع سمك تستهدف اليوم؟" أو "هل جربت الشريب هذا الموسم؟"
   - تذكير بمراجعة جدول المد المحلي (آخر ساعتين من المد هي الأفضل).

اكتب بلغة خبير ميداني، موجز ومفيد. لا تذكر أنك تلقيت بيانات أو أنك ذكاء اصطناعي."""

async def call_openrouter(ctx):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ctx}
        ],
        "max_tokens": 7000,
        "temperature": 0.3
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(OPENROUTER_URL, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        raise SurfError("استجابة OpenRouter فارغة")

@app.post("/scan-best")
@limiter.limit("5/minute")
async def scan_best_spots(request: Request, req: ScanRequest):
    beaches = []
    for gov in req.governorates:
        if gov in TUNISIAN_BEACHES:
            for b in TUNISIAN_BEACHES[gov]:
                beaches.append({**b, "governorate": gov})
    if not beaches:
        raise HTTPException(400, "لا توجد شواطئ للولايات المحددة")

    tz_name = "Africa/Tunis"
    now = datetime.now(zoneinfo.ZoneInfo(tz_name))
    if req.target_date == "today":
        target_dt = now.date()
    elif req.target_date == "tomorrow":
        target_dt = now.date() + timedelta(days=1)
    else:
        target_dt = now.date() + timedelta(days=2)

    start = target_dt - timedelta(days=2)
    end = target_dt + timedelta(days=1)
    sem = asyncio.Semaphore(5)

    async def process(beach):
        async with sem:
            try:
                m, w = await asyncio.gather(
                    fetch_marine(beach["lat"], beach["lon"], start.isoformat(), end.isoformat()),
                    fetch_weather(beach["lat"], beach["lon"], start.isoformat(), end.isoformat())
                )
                sunrise = w.get("daily", {}).get("sunrise", ["06:00"])[0]
                sunset = w.get("daily", {}).get("sunset", ["18:00"])[0]
                score, summary = evaluate_spot(
                    m, w, beach["orientation"], sunrise, sunset,
                    beach["type"], target_species=req.target_species
                )
                return {
                    "name": beach["name"],
                    "governorate": beach["governorate"],
                    "score": score,
                    "summary": summary,
                    "type": beach["type"]
                }
            except Exception as e:
                logger.error(f"فشل تقييم {beach['name']}: {e}")
                return None

    tasks = [process(b) for b in beaches]
    results = await asyncio.gather(*tasks)
    valid = [r for r in results if r is not None]
    valid.sort(key=lambda x: x["score"], reverse=True)
    return {"target_date": target_dt.isoformat(), "top10": valid[:10]}

def aggregate_physics(marine, weather, beach_orient, beach_type, target_date_obj, tz_name):
    tz = zoneinfo.ZoneInfo(tz_name) if tz_name else zoneinfo.ZoneInfo("UTC")
    mh = marine.get("hourly", {})
    wh = weather.get("hourly", {})
    dw = weather.get("daily", {})

    times = mh.get("time", [])
    wave_h = [safe_float(x) for x in mh.get("wave_height", [])]
    wave_p = [safe_float(x) for x in mh.get("wave_period", [])]
    swell_h = [safe_float(x) for x in mh.get("swell_wave_height", [])]
    swell_p = [safe_float(x) for x in mh.get("swell_wave_period", [])]
    sst = [safe_float(x) for x in mh.get("sea_surface_temperature", [])]

    wind_speed = [safe_float(x) for x in wh.get("wind_speed_10m", [])]
    wind_dir = [safe_float(x) for x in wh.get("wind_direction_10m", [])]
    wind_gust = [safe_float(x) for x in wh.get("wind_gusts_10m", [])]
    pressure = [safe_float(x) for x in wh.get("pressure_msl", [])]
    temp_air = [safe_float(x) for x in wh.get("temperature_2m", [])]
    precip = [safe_float(x) for x in wh.get("precipitation", [])]
    weather_code = [int(safe_float(x)) for x in wh.get("weather_code", [])]

    wind_kph = wind_speed
    gust_kph = wind_gust
    wave_power = [0.49 * (h**2) * p for h, p in zip(wave_h, wave_p)]

    wind_classes_detailed = [wind_class_detailed(angle_diff(wd, beach_orient)) for wd in wind_dir]

    dtimes = []
    for t_str in times:
        dt = datetime.fromisoformat(t_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        dtimes.append(dt)

    target_start = datetime.combine(target_date_obj, datetime.min.time(), tzinfo=tz)
    target_end = target_start + timedelta(days=1)
    past_start = target_start - timedelta(hours=48)

    past_idx = [i for i, dt in enumerate(dtimes) if past_start <= dt < target_start]
    target_idx = [i for i, dt in enumerate(dtimes) if target_start <= dt < target_end]

    if not target_idx:
        return {"past_avg_power":0, "dominant_wind":"غير معروف", "blocks":[], "red_flags":[], "green_flags":[], "weed_risk":False, "bio":{}, "avg_sst":0, "extra_info":{}}

    past_avg_power_val = sum(wave_power[i] for i in past_idx) / max(len(past_idx), 1)
    dominant = max(set(wind_classes_detailed), key=wind_classes_detailed.count)
    sustained_hrs = sum(1 for i in past_idx if wind_kph[i] > 18.5)

    press_target = [pressure[i] for i in target_idx]
    avg_press = sum(press_target) / len(press_target)
    press_trend = "مستقر"
    if len(press_target) >= 6:
        half = len(press_target)//2
        first = sum(press_target[:half])/half
        second = sum(press_target[half:])/(len(press_target)-half)
        diff = second - first
        if diff > 1: press_trend = "في ارتفاع"
        elif diff < -1: press_trend = "في انخفاض"

    periods = defaultdict(lambda: {"indices":[], "temps":[], "precip":[], "codes":[]})
    for i in target_idx:
        h = dtimes[i].hour
        if 4 <= h <= 11: key = "morning"
        elif 12 <= h <= 17: key = "afternoon"
        else: key = "night"
        periods[key]["indices"].append(i)
        periods[key]["temps"].append(temp_air[i])
        periods[key]["precip"].append(precip[i])
        periods[key]["codes"].append(weather_code[i])

    block_list = []
    for key, pd in periods.items():
        idxs = pd["indices"]
        if not idxs: continue
        avg_h = sum(wave_h[i] for i in idxs)/len(idxs)
        min_h = min(wave_h[i] for i in idxs)
        max_h = max(wave_h[i] for i in idxs)
        avg_p = sum(wave_power[i] for i in idxs)/len(idxs)
        avg_w = sum(wind_kph[i] for i in idxs)/len(idxs)
        min_w = min(wind_kph[i] for i in idxs)
        max_w = max(wind_kph[i] for i in idxs)
        wc = max(set(wind_classes_detailed[i] for i in idxs), key=wind_classes_detailed.count)
        avg_sw_h = sum(swell_h[i] for i in idxs)/len(idxs)
        avg_sw_p = sum(swell_p[i] for i in idxs)/len(idxs)
        avg_air = sum(pd["temps"])/len(pd["temps"]) if pd["temps"] else None
        total_precip = sum(pd["precip"]) if pd["precip"] else 0
        most_common_code = max(set(pd["codes"]), key=pd["codes"].count) if pd["codes"] else 0
        block_list.append({
            "name": {"morning":"الصباح","afternoon":"الظهر","night":"الليل"}[key],
            "wave_h_range": f"{min_h:.2f}-{max_h:.2f}",
            "power": round(avg_p,2),
            "wind_kph_range": f"{min_w:.1f}-{max_w:.1f}",
            "wind_dir": wc,
            "swell_h_range": f"{min(swell_h[i] for i in idxs):.2f}-{max(swell_h[i] for i in idxs):.2f}",
            "swell_period": round(avg_sw_p*2)/2,
            "air_temp": round(avg_air,1) if avg_air else "غير متوفر",
            "precip_mm": round(total_precip,1),
            "weather": weather_desc(most_common_code)
        })

    sunrise = dw.get("sunrise", ["غير معروف"])[0] if dw.get("sunrise") else "غير معروف"
    sunset = dw.get("sunset", ["غير معروف"])[0] if dw.get("sunset") else "غير معروف"

    reds, greens = [], []
    for i in target_idx:
        hh = dtimes[i].strftime("%H:%M")
        if wave_power[i] > 3 or wave_h[i] > 1.8 or wind_gust[i] > 50 or pressure[i] < 1005:
            reds.append(hh)
        if 0.3 <= wave_h[i] <= 1 and 0.1 <= wave_power[i] <= 1.5 and wind_kph[i] < 27.8:
            greens.append(hh)

    past_wp_avg = sum(wave_p[i] for i in past_idx)/max(len(past_idx),1)
    past_sh_avg = sum(swell_h[i] for i in past_idx)/max(len(past_idx),1)
    weed = (past_wp_avg >= 8.0 and past_sh_avg > 1.0 and wind_classes_detailed[target_idx[0]].startswith("بحرية"))

    avg_sst = sum(sst[i] for i in target_idx)/len(target_idx)
    bio = {}
    if avg_sst < 16: bio["high"] = ["قاروص", "سارغ"]
    elif avg_sst > 19: bio["high"] = ["دوراد", "ماربري"]
    else: bio["high"] = []
    if beach_type == "rocky": bio.setdefault("additional", []).append("سارغ")
    elif beach_type == "sandy": bio.setdefault("additional", []).append("بوري")

    moon_detail = moon_phase_detail(target_date_obj)
    moon_guidance = moon_fishing_guidance(target_date_obj)

    swell_day = [swell_h[i] for i in target_idx]
    swell_range = f"{min(swell_day):.2f}-{max(swell_day):.2f}"

    press_vals = [pressure[i] for i in target_idx]
    p3 = press_vals[-1] - press_vals[0] if len(press_vals) > 1 else 0
    p6 = press_vals[-1] - press_vals[0] if len(press_vals) > 2 else 0

    return {
        "past_avg_power": round(past_avg_power_val,2),
        "dominant_wind": dominant,
        "sustained_hrs": sustained_hrs,
        "blocks": block_list,
        "red_flags": reds[:5],
        "green_flags": greens[:5],
        "weed_risk": weed,
        "bio": bio,
        "avg_sst": round(avg_sst,1),
        "extra_info": {
            "pressure_avg": round(avg_press,1),
            "pressure_change_3h": round(p3,1),
            "pressure_change_6h": round(p6,1),
            "sunrise": sunrise,
            "sunset": sunset,
            "moon_phase": moon_detail["name"],
            "moon_status": moon_detail["status"],
            "moon_activity": moon_detail["activity"],
            "moon_guidance": moon_guidance,
            "swell_range_day": swell_range
        }
    }

def build_context(req, agg, tz_name):
    beach = "رملي" if req.beach_type=="sandy" else "صخري"
    lines = [
        f"الموقع: {req.latitude:.2f},{req.longitude:.2f}، شاطئ {beach}، اتجاه البحر {req.beach_orientation}°",
        f"التاريخ: {req.target_date} (توقيت {tz_name})",
        f"حرارة الماء: {agg.get('avg_sst','غير معروف')}°م",
        f"متوسط طاقة الموج 48س: {agg.get('past_avg_power',0)} kW/m",
        f"الرياح السائدة: {agg.get('dominant_wind','')}",
        f"خطر الأعشاب: {'نعم' if agg.get('weed_risk') else 'لا'}"
    ]
    extra = agg.get("extra_info", {})
    if extra:
        lines.append(f"الضغط الجوي: {extra.get('pressure_avg','')} hPa (تغير 3س: {extra.get('pressure_change_3h','')}, 6س: {extra.get('pressure_change_6h','')})")
        lines.append(f"الشروق: {extra.get('sunrise','')} | الغروب: {extra.get('sunset','')}")
        lines.append(f"حالة القمر للصيد: {extra.get('moon_status','')} - {extra.get('moon_activity','')}")
        lines.append(f"توجيه: {extra.get('moon_guidance','')}")
        lines.append(f"Swell اليومي: {extra.get('swell_range_day','')} م")

    if agg.get("blocks"):
        lines.append("تفاصيل الفترات:")
        for b in agg["blocks"]:
            lines.append(
                f"- {b['name']}: {b['weather']}, الموج {b['wave_h_range']}م, "
                f"swell {b['swell_h_range']}م/{b['swell_period']}ث, طاقة {b['power']}kW/m, "
                f"الرياح {b['wind_kph_range']} كم/س ({b['wind_dir']}), حرارة ~{b['air_temp']}°م, أمطار {b['precip_mm']}مم"
            )
    if agg.get("red_flags"): lines.append("ساعات الخطر: " + ", ".join(agg["red_flags"]))
    if agg.get("green_flags"): lines.append("ساعات مثالية: " + ", ".join(agg["green_flags"]))
    if agg.get("bio"):
        bio = agg["bio"]
        lines.append("الأسماك المتوقعة: " + ", ".join(bio.get("high",[]) + bio.get("additional",[])))
    # نصيحة موسمية
    month = req.target_date if isinstance(req.target_date, date) else date.today().month
    try:
        if isinstance(req.target_date, str):
            # محاولة استخراج الشهر إذا كان النص 'today' أو ما شابه
            month = date.today().month
        else:
            month = req.target_date.month
    except:
        month = date.today().month
    if month in [12, 1, 2]:
        lines.append("نصيحة موسمية: الشتاء ممتاز للقاروص والسارغ، استعمل الشريب والسردين المجمد.")
    elif month in [3, 4, 5]:
        lines.append("نصيحة موسمية: الربيع مثالي للبوري والدوراد، استعمل دود الكف والقمبري.")
    elif month in [6, 7, 8]:
        lines.append("نصيحة موسمية: الصيف تكثر فيه الأسماك السطحية، استعمل السردين الطازج والطعم الحي.")
    else:
        lines.append("نصيحة موسمية: الخريف يعود القاروص بقوة، جرب الشريب المخمّر.")

    lines.append("تذكير: راجع جدول المد المحلي – آخر ساعتين من المد هما الأفضل.")
    return "\n".join(lines)

@app.post("/generate-report")
@limiter.limit("10/minute")
async def generate_report(request: Request, req: ReportRequest):
    cache_key = f"{req.latitude:.4f}_{req.longitude:.4f}_{req.beach_orientation}_{req.beach_type}_{req.target_date}"
    if cache_key in cache and time.time() - cache[cache_key]["ts"] < CACHE_TTL:
        return cache[cache_key]["data"]
    try:
        tz_name = await fetch_timezone_info(req.latitude, req.longitude)
        target_dt = date.today()
        start = target_dt - timedelta(days=2)
        end = target_dt + timedelta(days=1)
        marine, weather = await asyncio.gather(
            fetch_marine(req.latitude, req.longitude, start.isoformat(), end.isoformat()),
            fetch_weather(req.latitude, req.longitude, start.isoformat(), end.isoformat())
        )
        real_today = extract_real_date_from_times(marine["hourly"]["time"], tz_name)
        target_dt = resolve_target_date(req.target_date, real_today)
        agg = aggregate_physics(marine, weather, req.beach_orientation, req.beach_type, target_dt, tz_name)
        ctx = build_context(req, agg, tz_name)
        report = await call_openrouter(ctx)
        result = {"report": report, "meta": {"timezone": tz_name, "target_date": target_dt.isoformat()}}
        cache[cache_key] = {"ts": time.time(), "data": result}
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
