"""
Surfcasting Analytics API – v5.3 (Production‑ready: مسح مع أسباب + تقرير حي مفصل)
"""
import os, math, asyncio, logging, traceback, zoneinfo, time
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("surfcasting")

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Surfcasting Analytics", version="5.3.0")
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
cache_lock = asyncio.Lock()
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

@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": "خطأ داخلي"})

@app.get("/health")
def health():
    return {"status": "ok", "openrouter": bool(OPENROUTER_API_KEY), "model": MODEL_NAME}

# ========== أدوات الشبكة ==========
async def fetch_with_retry(url: str, params: dict, max_retries=3, timeout=20.0) -> dict:
    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, params=params, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries:
                await asyncio.sleep(5 * attempt)
                continue
            raise
        except (httpx.ConnectError, httpx.TimeoutException):
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
                continue
            raise

async def post_with_retry(url: str, json: dict, headers: dict, max_retries=3, timeout=120.0) -> dict:
    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=json, headers=headers, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries:
                await asyncio.sleep(5 * attempt)
                continue
            raise
        except (httpx.ConnectError, httpx.TimeoutException):
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
                continue
            raise

# ========== دوال مساعدة ==========
def safe_float(v):
    try:
        return 0.0 if math.isnan(float(v)) else float(v)
    except:
        return 0.0

def angle_diff(w, b):
    d = abs(w - b) % 360
    return 360 - d if d > 180 else d

def wind_class_detailed(diff):
    if diff < 30: return "بحرية مباشرة"
    if diff < 45: return "بحرية خفيفة"
    if diff < 60: return "جانبية مائلة للبحر"
    if diff <= 120: return "جانبية"
    if diff < 150: return "جانبية مائلة للبر"
    if diff < 165: return "برية خفيفة"
    return "برية مباشرة"

def weather_desc(code):
    if code <= 1: return "صافية"
    if code == 2: return "غائمة جزئياً"
    if code == 3: return "غائمة"
    if code <= 48: return "ضباب"
    if code <= 55: return "رذاذ"
    if code <= 65: return "مطر"
    if code <= 77: return "ثلج"
    if code <= 82: return "زخات مطر"
    if code <= 99: return "عواصف"
    return "غير معروف"

def moon_phase_detail(d: date):
    y, m, day = d.year, d.month, d.day
    if m < 3:
        y -= 1
        m += 12
    a = int(y / 100)
    b = 2 - a + int(a / 4)
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + day + b - 1524.5
    days_since_new = jd - 2451550.1
    phase = (days_since_new % 29.53058867) / 29.53058867
    idx = int(phase * 8) % 8
    phases = {0: "محاق", 1: "هلال أول", 2: "تربيع أول", 3: "أحدب متزايد",
              4: "بدر", 5: "أحدب متناقص", 6: "تربيع ثاني", 7: "هلال آخر"}
    name = phases.get(idx, "محاق")
    if phase < 0.125 or phase > 0.875:
        status = "أيام متوسطة (محاق / هلال آخر)"
        activity = "ينشط القاع ليلاً بشكل خاص، والنهار متوسط"
    elif phase < 0.5:
        status = "أيام حمل"
        activity = "الأسماك نشيطة طوال اليوم، والصيد ممتاز ليلاً ونهاراً"
    else:
        status = "أيام فساد"
        activity = "الأسماك أقل نشاطاً، يُفضّل الصيد في الساعات الذهبية"
    return {"name": name, "status": status, "activity": activity, "phase": round(phase, 3)}

def moon_fishing_guidance(d: date):
    detail = moon_phase_detail(d)
    name = detail["name"]
    status = detail["status"]
    if "محاق" in name:
        return f"{status}. ركز على الصيد الليلي للقاروص والسارغ."
    elif "هلال أول" in name or "تربيع أول" in name:
        return f"{status}. فرصة ممتازة، استهدف القاروص نهاراً والدوراد عند الغروب."
    elif "أحدب متزايد" in name:
        return f"{status}. البوري والدوراد نشيطين نهاراً، القاروص ليلاً."
    elif "بدر" in name:
        return f"{status}. الأسماك السطحية نشيطة نهاراً، الليل ضعيف."
    elif "أحدب متناقص" in name or "تربيع ثاني" in name:
        return f"{status}. استعمل الطعوم القوية في الفجر والغروب."
    else:
        return f"{status}. الصيد مقبول في الساعات الذهبية."

async def fetch_timezone_info(lat, lon):
    try:
        data = await fetch_with_retry(MARINE_URL, {
            "latitude": lat, "longitude": lon,
            "hourly": "wave_height", "timezone": "auto", "forecast_days": 1
        }, timeout=10)
        tz = data.get("timezone", "UTC")
        zoneinfo.ZoneInfo(tz)
        return tz
    except:
        return "UTC"

def resolve_target_date(txt, real_today):
    if txt == "today": return real_today
    if txt == "tomorrow": return real_today + timedelta(days=1)
    return real_today + timedelta(days=2)

def align_hourly_data(marine_hourly, weather_hourly, tz):
    m_times = marine_hourly.get("time", [])
    w_times = weather_hourly.get("time", [])
    if not m_times or not w_times:
        return [], {}

    m_map = {}
    for i, t in enumerate(m_times):
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=tz)
        m_map[dt] = i

    w_map = {}
    for i, t in enumerate(w_times):
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=tz)
        w_map[dt] = i

    common = sorted(set(m_map) & set(w_map))
    if not common:
        return [], {}

    def extract(key, src, idx_map):
        arr = src.get(key, [])
        return [arr[idx_map[t]] if arr and idx_map[t] < len(arr) else 0.0 for t in common]

    weather_codes = [int(safe_float(x)) for x in extract("weather_code", weather_hourly, w_map)]

    aligned = {
        "wave_height": extract("wave_height", marine_hourly, m_map),
        "wave_period": extract("wave_period", marine_hourly, m_map),
        "wave_direction": extract("wave_direction", marine_hourly, m_map),
        "swell_wave_height": extract("swell_wave_height", marine_hourly, m_map),
        "swell_wave_period": extract("swell_wave_period", marine_hourly, m_map),
        "swell_wave_direction": extract("swell_wave_direction", marine_hourly, m_map),
        "sea_surface_temperature": extract("sea_surface_temperature", marine_hourly, m_map),
        "wind_speed_10m": extract("wind_speed_10m", weather_hourly, w_map),
        "wind_direction_10m": extract("wind_direction_10m", weather_hourly, w_map),
        "wind_gusts_10m": extract("wind_gusts_10m", weather_hourly, w_map),
        "pressure_msl": extract("pressure_msl", weather_hourly, w_map),
        "temperature_2m": extract("temperature_2m", weather_hourly, w_map),
        "precipitation": extract("precipitation", weather_hourly, w_map),
        "weather_code": weather_codes
    }
    return common, aligned

def get_daily_value(daily_data, target_date, key, default="غير معروف"):
    times = daily_data.get("time", [])
    vals = daily_data.get(key, [])
    for i, t in enumerate(times):
        if date.fromisoformat(t) == target_date:
            return vals[i] if i < len(vals) else default
    return default

# ========== Overpass ==========
async def get_bottom_type(lat, lon):
    query = f"""
    [out:json];
    (node(around:500,{lat},{lon})["surface"="sand"];
     node(around:500,{lat},{lon})["natural"="beach"];
     node(around:500,{lat},{lon})["surface"="gravel"];
     node(around:500,{lat},{lon})["surface"="rock"];);
    out body;"""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(OVERPASS_URL, params={"data": query}, timeout=15)
            r.raise_for_status()
            els = r.json().get("elements", [])
            if not els: return "sandy"
            for el in els:
                tags = el.get("tags", {})
                s = tags.get("surface", "").lower()
                if "rock" in s or "gravel" in s or "pebbles" in s:
                    return "rocky"
            return "sandy"
    except: return "sandy"

async def get_auto_orientation(lat, lon):
    query = f"""
    [out:json];
    (way(around:5000,{lat},{lon})["natural"="coastline"];);
    out geom;"""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(OVERPASS_URL, params={"data": query}, timeout=20)
            r.raise_for_status()
            els = r.json().get("elements", [])
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

@app.post("/detect-bottom-type")
@limiter.limit("10/minute")
async def detect_bottom_type(request: Request, req: AutoOrientationRequest):
    return {"bottom_type": await get_bottom_type(req.latitude, req.longitude)}

@app.post("/auto-orientation")
@limiter.limit("5/minute")
async def auto_orientation(request: Request, req: AutoOrientationRequest):
    return {"orientation": await get_auto_orientation(req.latitude, req.longitude)}

# ========== تفضيلات الأسماك ==========
SPECIES_PREFERENCES = {
    "قاروص": {"ideal_sst":(14,18),"preferred_wind":["بحرية مباشرة","بحرية خفيفة","جانبية مائلة للبحر"],"ideal_wave_range":(0.5,1.5),"ideal_power_range":(0.5,2.0),"bottom_type":"sandy"},
    "دوراد": {"ideal_sst":(19,26),"preferred_wind":["برية مباشرة","برية خفيفة","جانبية مائلة للبر"],"ideal_wave_range":(0.3,0.8),"ideal_power_range":(0.1,1.0),"bottom_type":"sandy"},
    "سارغ": {"ideal_sst":(16,22),"preferred_wind":["بحرية مباشرة","بحرية خفيفة","جانبية"],"ideal_wave_range":(0.5,1.2),"ideal_power_range":(0.3,1.5),"bottom_type":"rocky"},
    "بوري": {"ideal_sst":(15,28),"preferred_wind":["جانبية","برية خفيفة","بحرية خفيفة"],"ideal_wave_range":(0.1,0.6),"ideal_power_range":(0.0,0.5),"bottom_type":"sandy"},
    "ماربري": {"ideal_sst":(18,24),"preferred_wind":["برية مباشرة","برية خفيفة","جانبية"],"ideal_wave_range":(0.3,0.8),"ideal_power_range":(0.1,0.8),"bottom_type":"sandy"}
}

# ========== تقييم بقعة (للمسح) ==========
def evaluate_spot(times, aligned, orient, sunrise_str, sunset_str, beach_type, target_species=None):
    if not times: return 0.0, {}
    wh = aligned["wave_height"]
    wp = aligned["wave_period"]
    ws = aligned["wind_speed_10m"]
    wd = aligned["wind_direction_10m"]
    wg = aligned["wind_gusts_10m"]
    pr = aligned["pressure_msl"]
    sst = aligned["sea_surface_temperature"]
    wcls = [wind_class_detailed(angle_diff(d, orient)) for d in wd]
    N = len(times)
    score = 0.0
    red = 0
    green = 0
    try:
        sr_h = int(sunrise_str.split(":")[0])
        ss_h = int(sunset_str.split(":")[0])
    except:
        sr_h, ss_h = 6, 18
    for i in range(N):
        power = 0.49 * (wh[i] ** 2) * wp[i]
        if power > 3 or wh[i] > 1.8 or wg[i] > 50 or pr[i] < 1005:
            red += 1
            score -= 15
        elif 0.3 <= wh[i] <= 1 and 0.1 <= power <= 1.5 and ws[i] < 27.8:
            green += 1
            score += 10
            if abs(times[i].hour - sr_h) <= 2 or abs(times[i].hour - ss_h) <= 2:
                score += 5
        else:
            if 0.2 <= wh[i] <= 1.2:
                score += 3
            elif wh[i] < 0.2:
                score += 1
            else:
                score -= 2
            if ws[i] < 15:
                score += 4
            elif ws[i] < 25:
                score += 2
            else:
                score -= 1
    avg_wave = sum(wh) / N
    avg_power = sum(0.49 * (wh[i]**2) * wp[i] for i in range(N)) / N
    avg_wind = sum(ws) / N
    avg_sst = sum(sst) / N
    avg_press = sum(pr) / N
    dominant = max(set(wcls), key=wcls.count)
    if 1015 <= avg_press <= 1025:
        score += 8
    factor = 1.0
    if target_species and target_species in SPECIES_PREFERENCES:
        pref = SPECIES_PREFERENCES[target_species]
        match = 0
        lo, hi = pref["ideal_sst"]
        if lo <= avg_sst <= hi:
            match += 30
        elif abs(avg_sst - lo) <= 2 or abs(avg_sst - hi) <= 2:
            match += 15
        if pref["bottom_type"] == beach_type:
            match += 20
        if dominant in pref["preferred_wind"]:
            match += 25
        elif any(w in pref["preferred_wind"] for w in wcls):
            match += 10
        if pref["ideal_wave_range"][0] <= avg_wave <= pref["ideal_wave_range"][1]:
            match += 15
        if pref["ideal_power_range"][0] <= avg_power <= pref["ideal_power_range"][1]:
            match += 10
        factor = 0.5 + match / 100.0
    normalized = max(0, min(100, (score / 200) * 100 * factor))
    summary = {
        "avg_wave": round(avg_wave, 2),
        "avg_power": round(avg_power, 2),
        "avg_wind": round(avg_wind, 1),
        "avg_sst": round(avg_sst, 1),
        "dominant_wind": dominant,
        "green_hours": green,
        "red_hours": red,
    }
    return round(normalized, 1), summary

# ========== قاعدة الشواطئ الكاملة ==========
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

# ========== توليد أسباب الاختيار ==========
def generate_spot_reason(summary, beach_type, target_species):
    reasons = []
    wave = summary["avg_wave"]
    power = summary["avg_power"]
    wind = summary["avg_wind"]
    sst = summary["avg_sst"]
    dominant_wind = summary["dominant_wind"]
    green = summary["green_hours"]
    red = summary["red_hours"]

    # حالة الموج
    if 0.3 <= wave <= 1.0:
        reasons.append("ارتفاع الموج مثالي للصيد")
    elif wave < 0.3:
        reasons.append("موج منخفض جداً، قد يقلل النشاط")
    else:
        reasons.append("موج مرتفع نسبياً، يحتاج حذر")

    if power <= 1.5:
        reasons.append("طاقة الموج مناسبة للرمي")
    else:
        reasons.append("طاقة الموج عالية، ينصح برصاص ثقيل")

    # الرياح
    if wind < 15:
        reasons.append("رياح هادئة تساعد على الرمي البعيد")
    elif wind < 25:
        reasons.append("رياح متوسطة، مناسبة مع الحذر")
    else:
        reasons.append("رياح قوية، تصعّب التحكم في الخيط")

    reasons.append(f"الرياح السائدة {dominant_wind}")

    # حرارة الماء
    reasons.append(f"حرارة الماء {sst}°م")

    # ساعات خضراء / حمراء
    if green > 0:
        reasons.append(f"توجد {green} ساعات خضراء مثالية")
    if red > 0:
        reasons.append(f"توجد {red} ساعات حمراء خطرة")

    # التوافق مع النوع المستهدف
    if target_species and target_species in SPECIES_PREFERENCES:
        pref = SPECIES_PREFERENCES[target_species]
        if pref["bottom_type"] == beach_type:
            reasons.append("نوع القاع مناسب جداً للسمك المستهدف")
        if pref["ideal_sst"][0] <= sst <= pref["ideal_sst"][1]:
            reasons.append("حرارة الماء مثالية لنشاط السمك")
        if dominant_wind in pref["preferred_wind"]:
            reasons.append("اتجاه الرياح مفضل لهذا النوع")
        wave_lo, wave_hi = pref["ideal_wave_range"]
        power_lo, power_hi = pref["ideal_power_range"]
        if wave_lo <= wave <= wave_hi:
            reasons.append("ارتفاع الموج ضمن النطاق المفضل")
        if power_lo <= power <= power_hi:
            reasons.append("طاقة الموج ضمن النطاق المفضل")

    return "؛ ".join(reasons)

# ========== مسح أفضل الشواطئ مع الأسباب ==========
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

    tz = zoneinfo.ZoneInfo("Africa/Tunis")
    today = datetime.now(tz).date()
    target_dt = today if req.target_date == "today" else (today + timedelta(days=1) if req.target_date == "tomorrow" else today + timedelta(days=2))
    start = target_dt - timedelta(days=1)
    end = target_dt + timedelta(days=1)
    sem = asyncio.Semaphore(5)

    async def process(b):
        async with sem:
            try:
                m, w = await asyncio.gather(
                    fetch_with_retry(MARINE_URL, {
                        "latitude": b["lat"], "longitude": b["lon"],
                        "hourly": "wave_height,wave_period,wave_direction,swell_wave_height,swell_wave_period,swell_wave_direction,sea_surface_temperature",
                        "timezone": "Africa/Tunis", "start_date": start.isoformat(), "end_date": end.isoformat()
                    }),
                    fetch_with_retry(WEATHER_URL, {
                        "latitude": b["lat"], "longitude": b["lon"],
                        "hourly": "wind_speed_10m,wind_direction_10m,wind_gusts_10m,pressure_msl,temperature_2m,precipitation,weather_code",
                        "daily": "sunrise,sunset",
                        "timezone": "Africa/Tunis", "start_date": start.isoformat(), "end_date": end.isoformat()
                    })
                )
                all_t, aligned = align_hourly_data(m["hourly"], w["hourly"], tz)
                target_t = [t for t in all_t if t.date() == target_dt] or all_t
                indices = [all_t.index(t) for t in target_t]
                filtered = {k: [v[i] for i in indices] for k, v in aligned.items()}
                sunrise = get_daily_value(w.get("daily", {}), target_dt, "sunrise", "06:00")
                sunset = get_daily_value(w.get("daily", {}), target_dt, "sunset", "18:00")
                score, summ = evaluate_spot(target_t, filtered, b["orientation"], sunrise, sunset, b["type"], req.target_species)
                reason = generate_spot_reason(summ, b["type"], req.target_species)
                return {
                    "name": b["name"],
                    "governorate": b["governorate"],
                    "score": score,
                    "summary": summ,
                    "type": b["type"],
                    "reason": reason
                }
            except Exception as e:
                logger.error(f"فشل تقييم {b['name']}: {e}")
                return None

    res = await asyncio.gather(*[process(b) for b in beaches], return_exceptions=True)
    valid = [r for r in res if isinstance(r, dict) and r is not None]
    valid.sort(key=lambda x: x["score"], reverse=True)
    return {"target_date": target_dt.isoformat(), "top10": valid[:10]}

# ========== التجميع الفيزيائي المفصّل ==========
def aggregate_physics(all_times, aligned, orient, target_date_obj, sunrise, sunset):
    tz = all_times[0].tzinfo if all_times else zoneinfo.ZoneInfo("UTC")
    target_start = datetime.combine(target_date_obj, datetime.min.time(), tzinfo=tz)
    target_end = target_start + timedelta(days=1)
    past_start = target_start - timedelta(hours=48)
    past_idx = [i for i, t in enumerate(all_times) if past_start <= t < target_start]
    target_idx = [i for i, t in enumerate(all_times) if target_start <= t < target_end]
    if not target_idx:
        return {
            "past_avg_power": 0, "dominant_wind": "غير معروف", "blocks": [],
            "red_flags": [], "green_flags": [], "weed_risk": False,
            "bio": {}, "avg_sst": 0, "extra_info": {}
        }

    def pick(key):
        return [aligned[key][i] for i in target_idx]

    wh = pick("wave_height")
    wp = pick("wave_period")
    swh = pick("swell_wave_height")
    swp = pick("swell_wave_period")
    sst = pick("sea_surface_temperature")
    ws = pick("wind_speed_10m")
    wd = pick("wind_direction_10m")
    wg = pick("wind_gusts_10m")
    pr = pick("pressure_msl")
    ta = pick("temperature_2m")
    precip = pick("precipitation")
    wcode = [int(v) if v else 0 for v in pick("weather_code")]

    wave_power = [0.49 * (h**2) * p for h, p in zip(wh, wp)]
    wind_cls = [wind_class_detailed(angle_diff(d, orient)) for d in wd]

    past_avg_power = 0.0
    past_sh_avg = 0.0
    if past_idx:
        past_avg_power = sum(0.49 * (aligned["wave_height"][i]**2) * aligned["wave_period"][i] for i in past_idx) / len(past_idx)
        past_sh_avg = sum(aligned["swell_wave_height"][i] for i in past_idx) / len(past_idx)

    weed_risk = False
    if target_idx and wind_cls:
        first_wind = wind_cls[0]
        if first_wind.startswith("بحرية"):
            if past_sh_avg > 0.8 or past_avg_power > 5.0:
                weed_risk = True

    peak_gust = max(wg) if wg else 0.0
    dominant = max(set(wind_cls), key=wind_cls.count) if wind_cls else "غير معروف"
    sustained = sum(1 for i in past_idx if aligned["wind_speed_10m"][i] > 18.5) if past_idx else 0

    periods = defaultdict(lambda: {"indices": [], "start_idx": None, "end_idx": None})
    for idx, i in enumerate(target_idx):
        h = all_times[i].hour
        if 4 <= h <= 11:
            key = "morning"
        elif 12 <= h <= 17:
            key = "afternoon"
        else:
            key = "night"
        periods[key]["indices"].append(idx)
        if periods[key]["start_idx"] is None:
            periods[key]["start_idx"] = idx
        periods[key]["end_idx"] = idx

    blocks = []
    for key in ["morning", "afternoon", "night"]:
        pd = periods[key]
        idxs = pd["indices"]
        if not idxs:
            continue
        avg_h = sum(wh[i] for i in idxs) / len(idxs)
        min_h, max_h = min(wh[i] for i in idxs), max(wh[i] for i in idxs)
        avg_pow = sum(wave_power[i] for i in idxs) / len(idxs)
        avg_w = sum(ws[i] for i in idxs) / len(idxs)
        min_w, max_w = min(ws[i] for i in idxs), max(ws[i] for i in idxs)
        wc_dominant = max(set(wind_cls[i] for i in idxs), key=wind_cls.count)
        avg_swh = sum(swh[i] for i in idxs) / len(idxs)
        avg_swp = sum(swp[i] for i in idxs) / len(idxs)
        avg_air = sum(ta[i] for i in idxs) / len(idxs) if ta else 0
        total_precip = sum(precip[i] for i in idxs)
        most_code = max(set(wcode[i] for i in idxs), key=wcode[i].count) if idxs else 0

        swell_dominance = "مختلط"
        if avg_swh > 0.7 * avg_h:
            swell_dominance = "الطاقة أساساً قادمة من بعيد (swell قوي)"
        elif avg_h - avg_swh > 0.2:
            swell_dominance = "الموج ناتج عن الرياح المحلية (wind sea)"

        wind_start = wind_cls[idxs[0]]
        wind_end = wind_cls[idxs[-1]]
        if wind_start != wind_end:
            wind_trend = f"الرياح تتحول من {wind_start} إلى {wind_end} تدريجياً خلال هذه الفترة"
        else:
            wind_trend = f"الرياح ثابتة {wind_start} طوال الفترة"

        sea_state = "هادئ" if avg_h < 0.3 else "متوسط الهيجان" if avg_h < 0.8 else "هائج"

        blocks.append({
            "name": {"morning": "الصباح", "afternoon": "الظهر", "night": "الليل"}[key],
            "time_range": f"{all_times[target_idx[idxs[0]]].strftime('%H:%M')} - {all_times[target_idx[idxs[-1]]].strftime('%H:%M')}",
            "sea_state": sea_state,
            "wave_height": f"{min_h:.2f}-{max_h:.2f}",
            "wave_power": round(avg_pow, 2),
            "swell_height": f"{min(swh[i] for i in idxs):.2f}-{max(swh[i] for i in idxs):.2f}",
            "swell_period": round(avg_swp, 1),
            "swell_dominance": swell_dominance,
            "wind_speed": f"{min_w:.1f}-{max_w:.1f}",
            "wind_gust_peak": round(max(wg[i] for i in idxs), 1) if wg else 0,
            "wind_dir_dominant": wc_dominant,
            "wind_trend": wind_trend,
            "air_temp": round(avg_air, 1),
            "precip": round(total_precip, 1),
            "weather": weather_desc(most_code)
        })

    reds, greens = [], []
    for i in range(len(wh)):
        hh = all_times[target_idx[i]].strftime("%H:%M")
        if wave_power[i] > 3 or wh[i] > 1.8 or wg[i] > 50 or pr[i] < 1005:
            reds.append(hh)
        if 0.3 <= wh[i] <= 1 and 0.1 <= wave_power[i] <= 1.5 and ws[i] < 27.8:
            greens.append(hh)

    avg_sst = sum(sst) / len(sst) if sst else 0.0
    avg_press = sum(pr) / len(pr) if pr else 0.0
    p3 = pr[-1] - pr[0] if len(pr) > 1 else 0.0
    p6 = pr[-1] - pr[0] if len(pr) > 2 else 0.0

    bio = {}
    if avg_sst < 16:
        bio["high"] = ["قاروص", "سارغ"]
    elif avg_sst > 19:
        bio["high"] = ["دوراد", "ماربري"]
    else:
        bio["high"] = []

    moon_d = moon_phase_detail(target_date_obj)
    moon_g = moon_fishing_guidance(target_date_obj)

    extra_info = {
        "pressure_avg": round(avg_press, 1),
        "pressure_change_3h": round(p3, 1),
        "pressure_change_6h": round(p6, 1),
        "sunrise": sunrise,
        "sunset": sunset,
        "moon_phase": moon_d["name"],
        "moon_status": moon_d["status"],
        "moon_activity": moon_d["activity"],
        "moon_guidance": moon_g,
        "peak_gust_today": round(peak_gust, 1)
    }

    return {
        "past_avg_power": round(past_avg_power, 2),
        "dominant_wind": dominant,
        "sustained_hrs": sustained,
        "blocks": blocks,
        "red_flags": reds[:5],
        "green_flags": greens[:5],
        "weed_risk": weed_risk,
        "bio": bio,
        "avg_sst": round(avg_sst, 1),
        "extra_info": extra_info
    }

def build_context(req, agg, tz_name):
    beach = "رملي" if req.beach_type == "sandy" else "صخري"
    moon = agg["extra_info"]
    lines = []

    lines.append(f"الموقع: شاطئ {beach} اتجاهه {req.beach_orientation}° شمال.")
    lines.append(f"التاريخ: {req.target_date}، توقيت {tz_name}.")
    lines.append(f"درجة حرارة الماء: {agg['avg_sst']}°م.")
    lines.append(f"القمر: {moon['moon_status']} ({moon['moon_phase']}). {moon['moon_guidance']}")
    lines.append(f"الشروق {moon['sunrise']}، الغروب {moon['sunset']}.")

    lines.append(f"الرياح السائدة اليوم: {agg['dominant_wind']}، مع هبات تصل إلى {moon['peak_gust_today']} كم/س.")
    if moon['peak_gust_today'] > 30:
        lines.append("تحذير: هبات الرياح القوية تشكل خطراً على القصبة وتجعل الرمي بعيداً صعباً.")
    lines.append(f"خطر الأعشاب: {'نعم، الأعشاب متوقعة قرب الشاطئ' if agg['weed_risk'] else 'منخفض'}.")

    lines.append("\nتفاصيل الفترات الزمنية:")
    for b in agg["blocks"]:
        lines.append(f"\n【{b['name']} ({b['time_range']})】")
        lines.append(f"حالة البحر: {b['sea_state']}.")
        lines.append(f"ارتفاع الموج: {b['wave_height']} متر. swell: {b['swell_height']} متر، دورته {b['swell_period']} ثانية.")
        lines.append(f"تحليل الموج: {b['swell_dominance']}.")
        lines.append(f"طاقة الموج: {b['wave_power']} kW/m.")
        lines.append(f"الرياح: سرعة {b['wind_speed']} كم/س، اتجاه سائد {b['wind_dir_dominant']}. {b['wind_trend']}.")
        if b['wind_gust_peak'] > 25:
            lines.append(f"انتبه: هبات رياح عاتية تصل إلى {b['wind_gust_peak']} كم/س في هذه الفترة.")
        lines.append(f"حرارة الهواء: {b['air_temp']}°م، السماء: {b['weather']}، أمطار: {b['precip']} مم.")

    if agg["red_flags"]:
        lines.append(f"\nساعات الخطر (تجنب الصيد): {', '.join(agg['red_flags'])}")
    if agg["green_flags"]:
        lines.append(f"أفضل الساعات للصيد: {', '.join(agg['green_flags'])}")

    if agg["bio"].get("high"):
        lines.append(f"\nالأسماك المتوقعة بناءً على الحرارة: {', '.join(agg['bio']['high'])}.")

    lines.append("\nبناءً على هذا الوصف الدقيق، قدم تحليلك الاحترافي وتوصياتك النهائية (الرصاصة، التركيبة، الطعم، خطة الطوارئ، السلامة).")
    return "\n".join(lines)

SYSTEM_PROMPT = """أنت صياد سرفكاستينغ تونسي محترف ومحلل بحري واقعي. ستتلقى وصفاً تفصيلياً لحالة البحر والطقس والأمواج والرياح. مهمتك: أعد كتابة هذا الوصف بأسلوب الصياد الخبير المحترف بالدارجة التونسية، مع إضافة التوصيات التالية:
- أفضل توقيت للصيد ولماذا.
- نوع ووزن الرصاصة المناسبة مع تعليل تقني.
- التركيبة (montage) المثالية للقاع.
- الطعم الأنسب حسب الموسم والظروف.
- خطة طوارئ إذا ساءت الأحوال.
- نصائح سلامة واقعية.
- قارن مع الأيام الماضية بناءً على طاقة الموج الماضية المذكورة.
لا تتفاءل كذباً، وإذا كانت الظروف سيئة قل ذلك بوضوح. النص النهائي متصل بدون رموز أو نقاط نهاية، بالعامية التونسية المفهومة."""

async def call_openrouter(ctx):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ctx}],
        "max_tokens": 7000,
        "temperature": 0.3
    }
    data = await post_with_retry(OPENROUTER_URL, payload, headers)
    if "choices" in data and len(data["choices"]) > 0:
        return data["choices"][0]["message"]["content"]
    raise HTTPException(502, "OpenRouter فارغ")

@app.post("/generate-report")
@limiter.limit("10/minute")
async def generate_report(request: Request, req: ReportRequest):
    cache_key = f"{req.latitude:.4f}_{req.longitude:.4f}_{req.beach_orientation}_{req.beach_type}_{req.target_date}"
    async with cache_lock:
        if cache_key in cache and time.time() - cache[cache_key]["ts"] < CACHE_TTL:
            return cache[cache_key]["data"]

    try:
        tz_name = await fetch_timezone_info(req.latitude, req.longitude)
        tz = zoneinfo.ZoneInfo(tz_name)
        real_today = datetime.now(tz).date()
        target_dt = resolve_target_date(req.target_date, real_today)
        start = target_dt - timedelta(days=2)
        end = target_dt + timedelta(days=1)

        marine, weather = await asyncio.gather(
            fetch_with_retry(MARINE_URL, {
                "latitude": req.latitude, "longitude": req.longitude,
                "hourly": ["wave_height", "wave_period", "wave_direction",
                          "swell_wave_height", "swell_wave_period", "swell_wave_direction",
                          "sea_surface_temperature"],
                "timezone": tz_name, "start_date": start.isoformat(), "end_date": end.isoformat()
            }),
            fetch_with_retry(WEATHER_URL, {
                "latitude": req.latitude, "longitude": req.longitude,
                "hourly": ["wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
                          "pressure_msl", "temperature_2m", "precipitation", "weather_code"],
                "daily": ["sunrise", "sunset"],
                "timezone": tz_name, "start_date": start.isoformat(), "end_date": end.isoformat()
            })
        )

        all_t, aligned = align_hourly_data(marine["hourly"], weather["hourly"], tz)
        if not all_t:
            raise HTTPException(500, "لا بيانات")

        sunrise = get_daily_value(weather.get("daily", {}), target_dt, "sunrise", "06:00")
        sunset = get_daily_value(weather.get("daily", {}), target_dt, "sunset", "18:00")

        agg = aggregate_physics(all_t, aligned, req.beach_orientation, target_dt, sunrise, sunset)
        ctx = build_context(req, agg, tz_name)
        report = await call_openrouter(ctx)

        result = {"report": report, "meta": {"timezone": tz_name, "target_date": target_dt.isoformat()}}

        async with cache_lock:
            cache[cache_key] = {"ts": time.time(), "data": result}
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/generate-report failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="فشل إنشاء التقرير")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
