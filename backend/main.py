"""
Surfcasting Analytics API – v4.0 (Production‑ready, all fixes + enhancements)
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
app = FastAPI(title="Surfcasting Analytics", version="4.0.0")
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

# ========== نماذج البيانات ==========
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

class SurfError(Exception):
    pass

@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})

@app.get("/health")
def health():
    return {"status": "ok", "openrouter": bool(OPENROUTER_API_KEY), "model": MODEL_NAME}

# ========== أدوات الشبكة مع إعادة المحاولة ==========
async def fetch_with_retry(url: str, params: dict, max_retries=3, timeout=20.0) -> dict:
    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, params=params, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries:
                wait = 5 * attempt
                logger.warning(f"429 - retry in {wait}s")
                await asyncio.sleep(wait)
                continue
            raise
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(f"Connection error: {e} - retry in {wait}s")
                await asyncio.sleep(wait)
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
                wait = 5 * attempt
                logger.warning(f"429 on POST - retry in {wait}s")
                await asyncio.sleep(wait)
                continue
            raise
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(f"POST connection error: {e} - retry in {wait}s")
                await asyncio.sleep(wait)
                continue
            raise

# ========== الدوال المساعدة (رياضيات، رياح، طقس، قمر) ==========
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

SPECIES_PREFERENCES = {
    "قاروص": {"ideal_sst":(14,18),"preferred_wind":["بحرية مباشرة","بحرية خفيفة","جانبية مائلة للبحر"],"ideal_wave_range":(0.5,1.5),"ideal_power_range":(0.5,2.0),"bottom_type":"sandy"},
    "دوراد": {"ideal_sst":(19,26),"preferred_wind":["برية مباشرة","برية خفيفة","جانبية مائلة للبر"],"ideal_wave_range":(0.3,0.8),"ideal_power_range":(0.1,1.0),"bottom_type":"sandy"},
    "سارغ": {"ideal_sst":(16,22),"preferred_wind":["بحرية مباشرة","بحرية خفيفة","جانبية"],"ideal_wave_range":(0.5,1.2),"ideal_power_range":(0.3,1.5),"bottom_type":"rocky"},
    "بوري": {"ideal_sst":(15,28),"preferred_wind":["جانبية","برية خفيفة","بحرية خفيفة"],"ideal_wave_range":(0.1,0.6),"ideal_power_range":(0.0,0.5),"bottom_type":"sandy"},
    "ماربري": {"ideal_sst":(18,24),"preferred_wind":["برية مباشرة","برية خفيفة","جانبية"],"ideal_wave_range":(0.3,0.8),"ideal_power_range":(0.1,0.8),"bottom_type":"sandy"}
}

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

def extract_real_date_from_times(times, tz_name):
    if not times:
        return datetime.now(zoneinfo.ZoneInfo(tz_name)).date()
    dt = datetime.fromisoformat(times[0])
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=zoneinfo.ZoneInfo(tz_name))
    return dt.date()

def resolve_target_date(txt, real_today):
    if txt == "today": return real_today
    if txt == "tomorrow": return real_today + timedelta(days=1)
    return real_today + timedelta(days=2)

# ========== مزامنة البيانات الساعية ==========
def align_hourly_data(marine_hourly, weather_hourly, tz):
    m_times = marine_hourly.get("time", [])
    w_times = weather_hourly.get("time", [])
    if not m_times or not w_times:
        return [], {}
    m_map = {}
    for i, t in enumerate(m_times):
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        m_map[dt] = i
    w_map = {}
    for i, t in enumerate(w_times):
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        w_map[dt] = i
    common = sorted(set(m_map) & set(w_map))
    if not common:
        return [], {}
    def extract(key, src, idx_map):
        arr = src.get(key, [])
        return [arr[idx_map[t]] if arr and idx_map[t] < len(arr) else 0.0 for t in common]
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
        "weather_code": [int(safe_float(x)) for x in extract("weather_code", weather_hourly, w_map)]
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

# ========== تقييم البقعة (يُستخدم للمسح) ==========
def evaluate_spot(times, aligned, orient, sunrise_str, sunset_str, beach_type, target_species=None):
    if not times: return 0.0, {}
    wave_h = aligned["wave_height"]
    wave_p = aligned["wave_period"]
    wind_speed = aligned["wind_speed_10m"]
    wind_dir_vals = aligned["wind_direction_10m"]
    wind_gust = aligned["wind_gusts_10m"]
    pressure = aligned["pressure_msl"]
    sst = aligned["sea_surface_temperature"]

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
            h = times[i].hour
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

    if 1015 <= avg_press <= 1025:
        score += 8

    factor = 1.0
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

# ========== توليد أسباب التقييم للمسح ==========
def generate_spot_reason(summary, beach_type, target_species):
    reasons = []
    wave = summary["avg_wave"]
    power = summary["avg_power"]
    wind = summary["avg_wind"]
    sst = summary["avg_sst"]
    dominant_wind = summary["dominant_wind"]
    green = summary["green_hours"]
    red = summary["red_hours"]
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
    if wind < 15:
        reasons.append("رياح هادئة تساعد على الرمي البعيد")
    elif wind < 25:
        reasons.append("رياح متوسطة، مناسبة مع الحذر")
    else:
        reasons.append("رياح قوية، تصعّب التحكم في الخيط")
    reasons.append(f"الرياح السائدة {dominant_wind}")
    reasons.append(f"حرارة الماء {sst}°م")
    if green > 0:
        reasons.append(f"توجد {green} ساعات خضراء مثالية")
    if red > 0:
        reasons.append(f"توجد {red} ساعات حمراء خطرة")
    if target_species and target_species in SPECIES_PREFERENCES:
        pref = SPECIES_PREFERENCES[target_species]
        if pref["bottom_type"] == beach_type:
            reasons.append("نوع القاع مناسب جداً للسمك المستهدف")
        if pref["ideal_sst"][0] <= sst <= pref["ideal_sst"][1]:
            reasons.append("حرارة الماء مثالية لنشاط السمك")
        if dominant_wind in pref["preferred_wind"]:
            reasons.append("اتجاه الرياح مفضل لهذا النوع")
        if pref["ideal_wave_range"][0] <= wave <= pref["ideal_wave_range"][1]:
            reasons.append("ارتفاع الموج ضمن النطاق المفضل")
        if pref["ideal_power_range"][0] <= power <= pref["ideal_power_range"][1]:
            reasons.append("طاقة الموج ضمن النطاق المفضل")
    return "؛ ".join(reasons)

# ========== قاعدة بيانات الشواطئ ==========
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

# ========== مسح أفضل الشواطئ (مع أسباب) ==========
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

    start = target_dt - timedelta(days=1)
    end = target_dt + timedelta(days=1)
    sem = asyncio.Semaphore(5)

    async def process(beach):
        async with sem:
            try:
                m, w = await asyncio.gather(
                    fetch_with_retry(MARINE_URL, {
                        "latitude": beach["lat"], "longitude": beach["lon"],
                        "hourly": ["wave_height","wave_period","wave_direction","swell_wave_height","swell_wave_period","swell_wave_direction","sea_surface_temperature"],
                        "timezone": tz_name, "start_date": start.isoformat(), "end_date": end.isoformat()
                    }),
                    fetch_with_retry(WEATHER_URL, {
                        "latitude": beach["lat"], "longitude": beach["lon"],
                        "hourly": ["wind_speed_10m","wind_direction_10m","wind_gusts_10m","pressure_msl","temperature_2m","precipitation","weather_code"],
                        "daily": ["sunrise","sunset"],
                        "timezone": tz_name, "start_date": start.isoformat(), "end_date": end.isoformat()
                    })
                )
                tz = zoneinfo.ZoneInfo(tz_name)
                all_times, aligned = align_hourly_data(m["hourly"], w["hourly"], tz)
                target_times = [t for t in all_times if t.date() == target_dt]
                if not target_times:
                    target_times = all_times  # fallback
                indices = [all_times.index(t) for t in target_times]
                filtered = {k: [v[i] for i in indices] for k, v in aligned.items()}
                sunrise = get_daily_value(w.get("daily", {}), target_dt, "sunrise", "06:00")
                sunset = get_daily_value(w.get("daily", {}), target_dt, "sunset", "18:00")
                score, summary = evaluate_spot(target_times, filtered, beach["orientation"], sunrise, sunset, beach["type"], target_species=req.target_species)
                reason = generate_spot_reason(summary, beach["type"], req.target_species)
                return {
                    "name": beach["name"],
                    "governorate": beach["governorate"],
                    "score": score,
                    "summary": summary,
                    "type": beach["type"],
                    "reason": reason
                }
            except Exception as e:
                logger.error(f"فشل تقييم {beach['name']}: {e}")
                return None

    tasks = [process(b) for b in beaches]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    valid = [r for r in results if isinstance(r, dict) and r is not None]
    valid.sort(key=lambda x: x["score"], reverse=True)
    return {"target_date": target_dt.isoformat(), "top10": valid[:10]}

# ========== تجميع فيزيائي مفصل ==========
def aggregate_physics(all_times, aligned, beach_orient, target_date_obj, tz_name):
    tz = zoneinfo.ZoneInfo(tz_name)
    target_start = datetime.combine(target_date_obj, datetime.min.time(), tzinfo=tz)
    target_end = target_start + timedelta(days=1)
    past_start = target_start - timedelta(hours=48)

    past_idx = [i for i, t in enumerate(all_times) if past_start <= t < target_start]
    target_idx = [i for i, t in enumerate(all_times) if target_start <= t < target_end]

    if not target_idx:
        return {
            "past_avg_power":0, "dominant_wind":"غير معروف", "blocks":[], "red_flags":[], "green_flags":[],
            "weed_risk":False, "bio":{}, "avg_sst":0, "extra_info":{}, "target_date_obj": target_date_obj
        }

    def pick(key): return [aligned[key][i] for i in target_idx]
    wave_h = pick("wave_height")
    wave_p = pick("wave_period")
    swell_h = pick("swell_wave_height")
    swell_p = pick("swell_wave_period")
    sst = pick("sea_surface_temperature")
    wind_speed = pick("wind_speed_10m")
    wind_dir = pick("wind_direction_10m")
    wind_gust = pick("wind_gusts_10m")
    pressure = pick("pressure_msl")
    temp_air = pick("temperature_2m")
    precip = pick("precipitation")
    weather_code = [int(v) if v else 0 for v in pick("weather_code")]

    wave_power = [0.49 * (h**2) * p for h, p in zip(wave_h, wave_p)]
    wind_classes_detailed = [wind_class_detailed(angle_diff(wd, beach_orient)) for wd in wind_dir]

    # الأيام السابقة
    past_avg_power_val = 0.0
    past_sh_avg = 0.0
    if past_idx:
        past_avg_power_val = sum(0.49 * (aligned["wave_height"][i]**2) * aligned["wave_period"][i] for i in past_idx) / len(past_idx)
        past_sh_avg = sum(aligned["swell_wave_height"][i] for i in past_idx) / len(past_idx)

    # خطر الأعشاب
    weed = False
    if target_idx and wind_classes_detailed:
        first_wind = wind_classes_detailed[0]
        if first_wind.startswith("بحرية") and (past_sh_avg > 0.8 or past_avg_power_val > 5.0):
            weed = True

    dominant = max(set(wind_classes_detailed), key=wind_classes_detailed.count) if wind_classes_detailed else "غير معروف"
    sustained_hrs = sum(1 for i in past_idx if wind_speed[i] > 18.5) if past_idx else 0
    peak_gust_today = max(wind_gust) if wind_gust else 0.0

    # بناء الفترات
    periods = defaultdict(lambda: {"indices":[], "start_idx":None, "end_idx":None})
    for idx, i in enumerate(target_idx):
        h = all_times[i].hour
        if 4 <= h <= 11: key = "morning"
        elif 12 <= h <= 17: key = "afternoon"
        else: key = "night"
        periods[key]["indices"].append(idx)
        if periods[key]["start_idx"] is None: periods[key]["start_idx"] = idx
        periods[key]["end_idx"] = idx

    blocks = []
    for key in ["morning", "afternoon", "night"]:
        pd = periods[key]
        idxs = pd["indices"]
        if not idxs: continue
        avg_h = sum(wave_h[i] for i in idxs)/len(idxs)
        min_h, max_h = min(wave_h[i] for i in idxs), max(wave_h[i] for i in idxs)
        avg_pow = sum(wave_power[i] for i in idxs)/len(idxs)
        avg_w = sum(wind_speed[i] for i in idxs)/len(idxs)
        min_w, max_w = min(wind_speed[i] for i in idxs), max(wind_speed[i] for i in idxs)
        wc_dominant = max(set(wind_classes_detailed[i] for i in idxs), key=wind_classes_detailed.count)
        avg_swh = sum(swell_h[i] for i in idxs)/len(idxs)
        avg_swp = sum(swell_p[i] for i in idxs)/len(idxs)
        avg_air = sum(temp_air[i] for i in idxs)/len(idxs) if temp_air else 0
        total_precip = sum(precip[i] for i in idxs)
        most_code = max(set(weather_code[i] for i in idxs), key=weather_code[i].count) if idxs else 0

        swell_dominance = "مختلط"
        if avg_swh > 0.7 * avg_h:
            swell_dominance = "الطاقة أساساً قادمة من بعيد (swell قوي)"
        elif avg_h - avg_swh > 0.2:
            swell_dominance = "الموج ناتج عن الرياح المحلية (wind sea)"

        wind_start = wind_classes_detailed[idxs[0]]
        wind_end = wind_classes_detailed[idxs[-1]]
        wind_trend = f"الرياح تتحول من {wind_start} إلى {wind_end} تدريجياً" if wind_start != wind_end else f"الرياح ثابتة {wind_start} طوال الفترة"

        sea_state = "هادئ" if avg_h < 0.3 else "متوسط الهيجان" if avg_h < 0.8 else "هائج"

        blocks.append({
            "name": {"morning":"الصباح","afternoon":"الظهر","night":"الليل"}[key],
            "time_range": f"{all_times[target_idx[idxs[0]]].strftime('%H:%M')} - {all_times[target_idx[idxs[-1]]].strftime('%H:%M')}",
            "sea_state": sea_state,
            "wave_height": f"{min_h:.2f}-{max_h:.2f}",
            "wave_power": round(avg_pow,2),
            "swell_height": f"{min(swell_h[i] for i in idxs):.2f}-{max(swell_h[i] for i in idxs):.2f}",
            "swell_period": round(avg_swp,1),
            "swell_dominance": swell_dominance,
            "wind_speed": f"{min_w:.1f}-{max_w:.1f}",
            "wind_gust_peak": round(max(wind_gust[i] for i in idxs),1) if wind_gust else 0,
            "wind_dir": wc_dominant,
            "wind_trend": wind_trend,
            "air_temp": round(avg_air,1),
            "precip": round(total_precip,1),
            "weather": weather_desc(most_code)
        })

    reds, greens = [], []
    for i in range(len(wave_h)):
        hh = all_times[target_idx[i]].strftime("%H:%M")
        if wave_power[i] > 3 or wave_h[i] > 1.8 or wind_gust[i] > 50 or pressure[i] < 1005:
            reds.append(hh)
        if 0.3 <= wave_h[i] <= 1 and 0.1 <= wave_power[i] <= 1.5 and wind_speed[i] < 27.8:
            greens.append(hh)

    avg_sst = sum(sst)/len(sst) if sst else 0.0
    avg_press = sum(pressure)/len(pressure) if pressure else 0.0
    p3 = pressure[-1]-pressure[0] if len(pressure)>1 else 0.0
    p6 = pressure[-1]-pressure[0] if len(pressure)>2 else 0.0

    bio = {}
    if avg_sst < 16: bio["high"] = ["قاروص", "سارغ"]
    elif avg_sst > 19: bio["high"] = ["دوراد", "ماربري"]
    else: bio["high"] = []

    moon_detail = moon_phase_detail(target_date_obj)
    moon_guidance = moon_fishing_guidance(target_date_obj)
    sunrise = get_daily_value({"time":[target_date_obj.isoformat()],"sunrise":["06:00"]}, target_date_obj, "sunrise", "06:00")
    sunset = get_daily_value({"time":[target_date_obj.isoformat()],"sunset":["18:00"]}, target_date_obj, "sunset", "18:00")
    # لكننا سنستخدم sunrise/sunset من خارج الدالة، لذا نضبطها لاحقاً

    extra_info = {
        "pressure_avg": round(avg_press,1),
        "pressure_change_3h": round(p3,1),
        "pressure_change_6h": round(p6,1),
        "moon_phase": moon_detail["name"],
        "moon_status": moon_detail["status"],
        "moon_activity": moon_detail["activity"],
        "moon_guidance": moon_guidance,
        "peak_gust_today": round(peak_gust_today,1)
    }

    return {
        "past_avg_power": round(past_avg_power_val,2),
        "dominant_wind": dominant,
        "sustained_hrs": sustained_hrs,
        "blocks": blocks,
        "red_flags": reds[:5],
        "green_flags": greens[:5],
        "weed_risk": weed,
        "bio": bio,
        "avg_sst": round(avg_sst,1),
        "extra_info": extra_info,
        "target_date_obj": target_date_obj
    }

# ========== بناء السياق المفصل ==========
def build_context(req, agg, tz_name):
    beach = "رملي" if req.beach_type=="sandy" else "صخري"
    target_date_obj = agg.get("target_date_obj")
    moon = agg["extra_info"]

    lines = [
        f"الموقع: شاطئ {beach} اتجاهه {req.beach_orientation}° شمال.",
        f"التاريخ: {req.target_date} (توقيت {tz_name})",
        f"حرارة الماء: {agg['avg_sst']}°م",
        f"القمر: {moon['moon_status']} ({moon['moon_phase']}). {moon['moon_guidance']}",
        f"الرياح السائدة اليوم: {agg['dominant_wind']}، مع هبات تصل إلى {moon['peak_gust_today']} كم/س",
        f"خطر الأعشاب: {'نعم، الأعشاب متوقعة قرب الشاطئ' if agg['weed_risk'] else 'منخفض'}",
        f"متوسط طاقة الموج 48س الماضية: {agg['past_avg_power']} kW/m"
    ]
    if moon['peak_gust_today'] > 30:
        lines.append("تحذير: هبات الرياح القوية تشكل خطراً على القصبة وتجعل الرمي بعيداً صعباً.")

    lines.append("\nتفاصيل الفترات الزمنية:")
    for b in agg["blocks"]:
        lines.append(f"\n【{b['name']} ({b['time_range']})】")
        lines.append(f"حالة البحر: {b['sea_state']}.")
        lines.append(f"ارتفاع الموج: {b['wave_height']} متر. swell: {b['swell_height']} متر، دورته {b['swell_period']} ثانية.")
        lines.append(f"تحليل الموج: {b['swell_dominance']}.")
        lines.append(f"طاقة الموج: {b['wave_power']} kW/m.")
        lines.append(f"الرياح: سرعة {b['wind_speed']} كم/س، اتجاه سائد {b['wind_dir']}. {b['wind_trend']}.")
        if b['wind_gust_peak'] > 25:
            lines.append(f"انتبه: هبات رياح عاتية تصل إلى {b['wind_gust_peak']} كم/س في هذه الفترة.")
        lines.append(f"حرارة الهواء: {b['air_temp']}°م، السماء: {b['weather']}، أمطار: {b['precip']} مم.")

    if agg["red_flags"]: lines.append(f"\nساعات الخطر (تجنب الصيد): {', '.join(agg['red_flags'])}")
    if agg["green_flags"]: lines.append(f"أفضل الساعات للصيد: {', '.join(agg['green_flags'])}")
    if agg["bio"].get("high"): lines.append(f"\nالأسماك المتوقعة بناءً على الحرارة: {', '.join(agg['bio']['high'])}.")

    if target_date_obj:
        month = target_date_obj.month
        if month in [12,1,2]: lines.append("نصيحة موسمية: الشتاء ممتاز للقاروص والسارغ.")
        elif month in [3,4,5]: lines.append("نصيحة موسمية: الربيع مثالي للبوري والدوراد.")
        elif month in [6,7,8]: lines.append("نصيحة موسمية: الصيف تكثر فيه الأسماك السطحية.")
        else: lines.append("نصيحة موسمية: الخريف يعود القاروص بقوة.")

    lines.append("تذكير: راجع جدول المد المحلي – آخر ساعتين من المد هما الأفضل.")
    lines.append("\nبناءً على هذا الوصف الدقيق، قدم تحليلك الاحترافي وتوصياتك النهائية (الرصاصة، التركيبة، الطعم، خطة الطوارئ، السلامة).")
    return "\n".join(lines)

SYSTEM_PROMPT = """أنت صياد سرفكاستينغ تونسي محترف ومحلل بحري واقعي. اكتب تقريراً بحرياً كاملاً باللغة العربية والمصطلحات التونسية الدارجة. التقرير نص واحد متصل بلا نقاط أو رموز، يغطي:
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
   - نصيحة موسمية حسب الشهر الحالي.
   - قارن سريعاً مع الأيام الماضية (هل البحر يهدأ أم يضطرب؟).
   - اختم بسؤال للصياد: "أي نوع سمك تستهدف اليوم؟" أو "هل جربت الشريب هذا الموسم؟"
   - تذكير بمراجعة جدول المد المحلي (آخر ساعتين من المد هي الأفضل).
اكتب بلغة خبير ميداني، موجز ومفيد. لا تذكر أنك تلقيت بيانات أو أنك ذكاء اصطناعي. إذا كانت الظروف سيئة، قل ذلك بوضوح ولا تتفاءل كذباً."""

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
    data = await post_with_retry(OPENROUTER_URL, payload, headers)
    if "choices" in data and len(data["choices"]) > 0:
        return data["choices"][0]["message"]["content"]
    raise SurfError("استجابة OpenRouter فارغة")

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
        # تحديد التاريخ الحقيقي اليوم في تونس
        now_tn = datetime.now(zoneinfo.ZoneInfo("Africa/Tunis"))
        real_today = now_tn.date()
        target_dt = resolve_target_date(req.target_date, real_today)

        start = target_dt - timedelta(days=2)
        end = target_dt + timedelta(days=1)

        # جلب البيانات
        marine = await fetch_with_retry(MARINE_URL, {
            "latitude": req.latitude, "longitude": req.longitude,
            "hourly": ["wave_height","wave_period","wave_direction","swell_wave_height","swell_wave_period","swell_wave_direction","sea_surface_temperature"],
            "timezone": tz_name, "start_date": start.isoformat(), "end_date": end.isoformat()
        })
        weather = await fetch_with_retry(WEATHER_URL, {
            "latitude": req.latitude, "longitude": req.longitude,
            "hourly": ["wind_speed_10m","wind_direction_10m","wind_gusts_10m","pressure_msl","temperature_2m","precipitation","weather_code"],
            "daily": ["sunrise","sunset"],
            "timezone": tz_name, "start_date": start.isoformat(), "end_date": end.isoformat()
        })

        # مزامنة
        all_times, aligned = align_hourly_data(marine["hourly"], weather["hourly"], tz)
        if not all_times:
            raise HTTPException(500, "لا توجد بيانات ساعية")

        sunrise = get_daily_value(weather.get("daily", {}), target_dt, "sunrise", "06:00")
        sunset = get_daily_value(weather.get("daily", {}), target_dt, "sunset", "18:00")

        agg = aggregate_physics(all_times, aligned, req.beach_orientation, target_dt, tz_name)
        # نضيف sunrise/sunset للـ extra_info
        agg["extra_info"]["sunrise"] = sunrise
        agg["extra_info"]["sunset"] = sunset

        ctx = build_context(req, agg, tz_name)
        report = await call_openrouter(ctx)

        result = {
            "report": report,
            "meta": {"timezone": tz_name, "target_date": target_dt.isoformat()}
        }

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
