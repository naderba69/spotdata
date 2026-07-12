"""
Surfcasting Analytics API – v18.0.20 (Perfect Line Separation & Icon Formatting)
- حل جذري لتلاصق الأسطر: دالة enforce_line_breaks.
- SYSTEM_PROMPT صارم لمنع السلاسل المتصلة.
- تحسين add_paragraph_spacing.
- جميع ميزات الإصدارات السابقة.
"""
import os, math, asyncio, logging, traceback, zoneinfo, json, time, re
from datetime import datetime, timedelta, date
from typing import Dict, Optional, List, Tuple, Set
from collections import defaultdict
from contextlib import asynccontextmanager

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

http_client = httpx.AsyncClient(timeout=120.0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await http_client.aclose()

app = FastAPI(title="Surfcasting Analytics", version="18.0.20", lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
if ALLOWED_ORIGINS == "*":
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
else:
    origins = [o.strip() for o in ALLOWED_ORIGINS.split(",")]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY مفقود")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "google/gemini-2.5-flash-lite"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "SurfcastingAnalytics/1.0 (naderba69@gmail.com)"

class AutoOrientationRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

class RawDataReportRequest(BaseModel):
    beach_orientation: int = Field(..., ge=0, le=360)
    beach_type: str = Field(..., pattern="^(sandy|rocky)$")
    target_date: str = Field(..., pattern="^(today|tomorrow|day_after)$")
    marine_data: Optional[dict] = None
    weather_data: Optional[dict] = None
    latitude: Optional[float] = 36.8
    longitude: Optional[float] = 10.1

class DetectBottomRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": "خطأ داخلي في الخادم"})

@app.get("/health")
def health():
    return {"status": "ok", "version": "18.0.20"}

# ==================== دوال مساعدة ====================
async def post_with_retry(url, json_data, headers, max_retries=3):
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            r = await http_client.post(url, json=json_data, headers=headers)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPStatusError, json.JSONDecodeError, httpx.DecodingError) as e:
            last_exc = e
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in [429, 500, 502, 503] and attempt < max_retries:
                await asyncio.sleep(5 * attempt)
                continue
            if attempt < max_retries and not isinstance(e, httpx.HTTPStatusError):
                await asyncio.sleep(2 ** attempt)
                continue
            raise
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            last_exc = e
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
                continue
            raise
    raise last_exc or Exception("فشل الاتصال")

def safe_float(v):
    try:
        return 0.0 if math.isnan(float(v)) else float(v)
    except (TypeError, ValueError):
        return 0.0

def angle_diff(w, b):
    d = abs(w - b) % 360
    return 360 - d if d > 180 else d

def signed_angle_diff(w, b):
    d = (w - b + 180) % 360 - 180
    return d

def circular_diff(a: float, b: float) -> float:
    diff = abs(a - b) % 24
    return min(diff, 24 - diff)

def is_close(t1: float, t2: float, margin: float = 1.5) -> bool:
    return circular_diff(t1, t2) <= margin

def calc_bearing(lat1, lon1, lat2, lon2):
    lat1_r, lon1_r, lat2_r, lon2_r = math.radians(lat1), math.radians(lon1), math.radians(lat2), math.radians(lon2)
    dlon = lon2_r - lon1_r
    x = math.sin(dlon) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def calc_distance(lat1, lon1, lat2, lon2):
    dlat = (lat2 - lat1) * 111320
    dlon = (lon2 - lon1) * 111320 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.sqrt(dlat**2 + dlon**2)

def format_time(h: float) -> str:
    h = h % 24
    hh = int(h)
    mm = int((h - hh) * 60)
    return f"{hh:02d}:{mm:02d}"

def format_time_gap(hours_decimal: float) -> str:
    if hours_decimal <= 0:
        return "0 دقيقة"
    total_minutes = round(hours_decimal * 60)
    h = total_minutes // 60
    m = total_minutes % 60
    parts = []
    if h > 0:
        parts.append(f"{h} ساعة" if h == 1 else f"{h} ساعات")
    if m > 0:
        parts.append(f"{m} دقيقة" if m == 1 else f"{m} دقائق")
    return " و ".join(parts) if parts else "0 دقيقة"

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
    if code <= 82: return "زخات مطر"
    if code <= 99: return "عواصف"
    return "غير معروف"

def deg_to_compass(deg):
    val = int((deg / 22.5) + 0.5) % 16
    arr = ["شمال","شمال شمال شرق","شمال شرق","شرق شمال شرق","شرق","شرق جنوب شرق","جنوب شرق","جنوب جنوب شرق",
           "جنوب","جنوب جنوب غرب","جنوب غرب","غرب جنوب غرب","غرب","غرب شمال غرب","شمال غرب","شمال شمال غرب"]
    return arr[val]

def resolve_target_date(txt, real_today):
    if txt == "today": return real_today
    if txt == "tomorrow": return real_today + timedelta(days=1)
    return real_today + timedelta(days=2)

def get_moon_and_tide_analysis(d: date):
    y, m, day = d.year, d.month, d.day
    if m < 3: y -= 1; m += 12
    a = int(y / 100)
    b = 2 - a + int(a / 4)
    jd = 365.25 * (y + 4716) + 30.6001 * (m + 1) + day + b - 1524.5
    days_since_new = jd - 2451550.1
    phase = (days_since_new % 29.53058867) / 29.53058867
    idx = int(phase * 8) % 8
    names = {0:"محاق",1:"هلال أول",2:"تربيع أول",3:"أحدب متزايد",4:"بدر",5:"أحدب متناقص",6:"تربيع ثاني",7:"هلال آخر"}
    if idx in [0, 4]: tide_strength = "مد وجزر قوي جداً (مد ربيعي)"
    elif idx in [2, 6]: tide_strength = "مد وجزر ضعيف جداً (مد محاقي)"
    else: tide_strength = "مد وجزر متوسط"
    return {"name": names[idx], "phase_decimal": phase, "tide_strength": tide_strength, "idx": idx}

def safe_parse_time(time_str: str) -> float:
    try:
        parts = time_str.split(":")
        h = float(parts[0])
        m = float(parts[1]) if len(parts) > 1 else 0.0
        return h + m / 60.0
    except (ValueError, IndexError):
        return 6.0

def get_moon_age_days(d: date) -> float:
    y, m, day = d.year, d.month, d.day
    if m < 3: y -= 1; m += 12
    a = int(y / 100)
    b = 2 - a + int(a / 4)
    jd = 365.25 * (y + 4716) + 30.6001 * (m + 1) + day + b - 1524.5
    days_since_new = jd - 2451550.1
    age = days_since_new % 29.53058867
    return age

def get_haml_mat_status(age_days: float) -> dict:
    if 13 <= age_days <= 16:
        day_in = int(age_days - 13) + 1
        return {
            "status": "أيام الحياء",
            "phase": "حمل البدر",
            "description": f"اليوم {day_in} في حمل البدر. البحر حايي، التيارات قوية، الصيد في ذروته من الشاطئ.",
            "score_delta": 15
        }
    elif 28 <= age_days or age_days <= 1:
        if age_days >= 28:
            day_in = int(age_days - 28) + 1
        else:
            day_in = int(age_days + 29.53 - 28) + 1
            day_in = min(day_in, 4)
        return {
            "status": "أيام الحياء",
            "phase": "حمل المحاق",
            "description": f"اليوم {day_in} في حمل المحاق. البحر حايي، التيارات قوية، الصيد في ذروته من الشاطئ.",
            "score_delta": 15
        }
    elif 7 <= age_days <= 9 or 21 <= age_days <= 23:
        if 7 <= age_days <= 9:
            phase_name = "التربيع الأول"
            day_in = int(age_days - 7) + 1
        else:
            phase_name = "التربيع الثاني"
            day_in = int(age_days - 21) + 1
        return {
            "status": "أيام المات",
            "phase": phase_name,
            "description": f"اليوم {day_in} في {phase_name}. البحر مْيِّت، الماء راكد، الصيد أصعب من الشاطئ.",
            "score_delta": -15
        }
    else:
        return {
            "status": "أيام عادية",
            "phase": "",
            "description": "لا توجد مؤشرات حيائية أو مات قوية. الصيد من الشاطئ ممكن.",
            "score_delta": 0
        }

def get_fishing_platform_advice(haml_status: str) -> str:
    if "الحياء" in haml_status:
        return "الصيد من الشاطئ ممتاز اليوم. التيارات قوية تجلب الأسماك."
    elif "المات" in haml_status:
        return "الصيد من الشاطئ صعب اليوم بسبب ركود الماء."
    else:
        return "الصيد من الشاطئ ممكن اليوم."

def estimate_tidal_windows(target_date_obj, moon_analysis, sunrise_str, sunset_str, latitude):
    sr_h = safe_parse_time(sunrise_str)
    ss_h = safe_parse_time(sunset_str)
    moon_age_hours = moon_analysis["phase_decimal"] * 29.53 * 24
    lunitidal_correction = (latitude - 10) * 2.5 / 60.0
    base_hw_hour = (moon_age_hours * 0.04) % 12 + 6 + lunitidal_correction
    base_hw_hour = base_hw_hour % 24
    hw1 = base_hw_hour
    lw1 = (hw1 + 6.2) % 24
    hw2 = (hw1 + 12.4) % 24
    lw2 = (lw1 + 12.4) % 24

    windows = {"HW1": format_time(hw1), "LW1": format_time(lw1), "HW2": format_time(hw2), "LW2": format_time(lw2)}

    golden_windows = []
    lw1_minus2 = (lw1 - 2) % 24
    lw2_minus2 = (lw2 - 2) % 24

    if is_close(hw1, sr_h, 1.5):
        golden_windows.append(f"ساعة ذهبية صباحية: تزامن المد العالي الأول ({windows['HW1']}) مع الفجر ({sunrise_str}).")
    if is_close(hw2, ss_h, 1.5):
        golden_windows.append(f"ساعة ذهبية مسائية: تزامن المد العالي الثاني ({windows['HW2']}) مع الغروب ({sunset_str}).")
    if is_close(lw1_minus2, sr_h, 1.5) or is_close(lw2_minus2, sr_h, 1.5):
        golden_windows.append("نافذة الجزر الممتازة: تزامن بداية جزر قوي مع الفجر.")
    if is_close(lw1_minus2, ss_h, 1.5) or is_close(lw2_minus2, ss_h, 1.5):
        golden_windows.append("نافذة الجزر الممتازة: تزامن بداية جزر قوي مع الغروب.")

    if not golden_windows:
        hw1_gap = circular_diff(hw1, sr_h)
        hw2_gap = circular_diff(hw2, ss_h)
        golden_windows.append(f"لا توجد ساعة ذهبية. المد العالي الأول ({windows['HW1']}) يبعد {format_time_gap(hw1_gap)} عن الفجر. المد العالي الثاني ({windows['HW2']}) يبعد {format_time_gap(hw2_gap)} عن الغروب.")

    return windows, golden_windows

def calculate_solunar(d: date, lat: float):
    y, m, day = d.year, d.month, d.day
    if m < 3: y -= 1; m += 12
    a = int(y / 100)
    b = 2 - a + int(a / 4)
    jd = 365.25 * (y + 4716) + 30.6001 * (m + 1) + day + b - 1524.5
    days_since_new = jd - 2451550.1
    moon_phase = (days_since_new % 29.53058867) / 29.53058867
    moon_transit = (moon_phase * 24 + 12) % 24
    major1 = moon_transit
    major2 = (moon_transit + 12) % 24
    minor1 = (major1 + 6) % 24
    minor2 = (major2 + 6) % 24
    return {"major1": format_time(major1), "major2": format_time(major2),
            "minor1": format_time(minor1), "minor2": format_time(minor2)}

def align_hourly_data(marine_hourly, weather_hourly, tz_name):
    tz = zoneinfo.ZoneInfo(tz_name)
    m_times = marine_hourly.get("time", [])
    w_times = weather_hourly.get("time", [])
    if not m_times or not w_times:
        return [], {}
    m_map, w_map = {}, {}
    for i, t in enumerate(m_times):
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        m_map[dt.replace(minute=0, second=0, microsecond=0)] = i
    for i, t in enumerate(w_times):
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        w_map[dt.replace(minute=0, second=0, microsecond=0)] = i
    common = sorted(set(m_map) & set(w_map))
    if not common:
        return [], {}
    def extract(key, src, idx_map):
        arr = src.get(key, [])
        return [arr[idx_map[t]] if arr and idx_map[t] < len(arr) else 0.0 for t in common]
    return common, {
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
        "visibility": extract("visibility", weather_hourly, w_map),
        "weather_code": [int(safe_float(x)) for x in extract("weather_code", weather_hourly, w_map)]
    }

# ==================== 50+ شاطئ تونسي ====================
TUNISIAN_BEACHES = [
    {"name":"شاطئ طبرقة", "lat":36.9544, "lon":8.7581, "orientation":315, "type":"sandy"},
    {"name":"شاطئ عين دراهم", "lat":36.9580, "lon":8.7540, "orientation":315, "type":"sandy"},
    {"name":"شاطئ بنزرت", "lat":37.2744, "lon":9.8739, "orientation":90, "type":"sandy"},
    {"name":"شاطئ رفراف", "lat":37.1911, "lon":10.0392, "orientation":45, "type":"sandy"},
    {"name":"شاطئ غار الملح", "lat":37.1750, "lon":10.1792, "orientation":90, "type":"sandy"},
    {"name":"شاطئ رأس الجبل", "lat":37.2169, "lon":10.1228, "orientation":45, "type":"sandy"},
    {"name":"شاطئ قليبية", "lat":36.8500, "lon":11.1000, "orientation":45, "type":"sandy"},
    {"name":"شاطئ الهوارية", "lat":37.0575, "lon":11.0153, "orientation":0, "type":"rocky"},
    {"name":"شاطئ سيدي علي المكي", "lat":37.1611, "lon":10.2564, "orientation":45, "type":"sandy"},
    {"name":"شاطئ قرطاج", "lat":36.8528, "lon":10.3264, "orientation":90, "type":"sandy"},
    {"name":"شاطئ المرسى", "lat":36.8794, "lon":10.3244, "orientation":90, "type":"sandy"},
    {"name":"شاطئ حلق الوادي", "lat":36.8167, "lon":10.3047, "orientation":90, "type":"sandy"},
    {"name":"شاطئ رادس", "lat":36.7500, "lon":10.2833, "orientation":90, "type":"sandy"},
    {"name":"شاطئ الزهراء", "lat":36.7222, "lon":10.3000, "orientation":90, "type":"sandy"},
    {"name":"شاطئ حمام الأنف", "lat":36.7183, "lon":10.3342, "orientation":90, "type":"sandy"},
    {"name":"شاطئ سليمان", "lat":36.6950, "lon":10.4939, "orientation":90, "type":"sandy"},
    {"name":"شاطئ نابل", "lat":36.4561, "lon":10.7389, "orientation":90, "type":"sandy"},
    {"name":"شاطئ الحمامات", "lat":36.4000, "lon":10.6167, "orientation":90, "type":"sandy"},
    {"name":"شاطئ ياسمين الحمامات", "lat":36.3667, "lon":10.5333, "orientation":90, "type":"sandy"},
    {"name":"شاطئ هرقلة", "lat":36.0333, "lon":10.5000, "orientation":90, "type":"sandy"},
    {"name":"شاطئ الشابة", "lat":35.9039, "lon":10.5739, "orientation":90, "type":"sandy"},
    {"name":"شاطئ سوسة", "lat":35.8250, "lon":10.6400, "orientation":90, "type":"sandy"},
    {"name":"شاطئ القنطاوي", "lat":35.8750, "lon":10.5950, "orientation":90, "type":"sandy"},
    {"name":"شاطئ المنستير", "lat":35.7667, "lon":10.8167, "orientation":90, "type":"sandy"},
    {"name":"شاطئ سقانص", "lat":35.7583, "lon":10.8028, "orientation":90, "type":"sandy"},
    {"name":"شاطئ المهدية", "lat":35.5047, "lon":11.0622, "orientation":90, "type":"sandy"},
    {"name":"شاطئ قصور الساف", "lat":35.6167, "lon":10.8833, "orientation":90, "type":"sandy"},
    {"name":"شاطئ صفاقس", "lat":34.7400, "lon":10.7600, "orientation":90, "type":"sandy"},
    {"name":"شاطئ قرقنة", "lat":34.7042, "lon":11.2389, "orientation":90, "type":"sandy"},
    {"name":"شاطئ اللوزة", "lat":34.5833, "lon":10.4167, "orientation":90, "type":"sandy"},
    {"name":"شاطئ قابس", "lat":33.8881, "lon":10.0981, "orientation":90, "type":"sandy"},
    {"name":"شاطئ جرجيس", "lat":33.5000, "lon":11.1167, "orientation":90, "type":"sandy"},
    {"name":"شاطئ جربة (ميدون)", "lat":33.8075, "lon":10.9931, "orientation":90, "type":"sandy"},
    {"name":"شاطئ جربة (حومة السوق)", "lat":33.8833, "lon":10.8667, "orientation":90, "type":"sandy"},
    {"name":"شاطئ جربة (أغير)", "lat":33.8167, "lon":11.0500, "orientation":90, "type":"sandy"},
    {"name":"شاطئ الزارات", "lat":33.6833, "lon":10.3500, "orientation":90, "type":"sandy"},
    {"name":"شاطئ بنقردان", "lat":33.1381, "lon":11.2167, "orientation":90, "type":"sandy"},
    {"name":"شاطئ طبرقة 2", "lat":36.9600, "lon":8.7600, "orientation":315, "type":"sandy"},
    {"name":"شاطئ ماطر", "lat":37.0600, "lon":9.6600, "orientation":45, "type":"sandy"},
    {"name":"شاطئ أوتيك", "lat":37.1481, "lon":10.0617, "orientation":45, "type":"sandy"},
    {"name":"شاطئ منزل بورقيبة", "lat":37.0683, "lon":9.8258, "orientation":45, "type":"sandy"},
    {"name":"شاطئ سجنان", "lat":37.1700, "lon":9.3600, "orientation":315, "type":"sandy"},
    {"name":"شاطئ الكرم", "lat":36.8467, "lon":10.3167, "orientation":90, "type":"sandy"},
    {"name":"شاطئ أريانة", "lat":36.8750, "lon":10.2083, "orientation":90, "type":"sandy"},
    {"name":"شاطئ المحمدية", "lat":36.6667, "lon":10.1500, "orientation":90, "type":"sandy"},
    {"name":"شاطئ مرناق", "lat":36.6833, "lon":10.2833, "orientation":90, "type":"sandy"},
    {"name":"شاطئ بومهل", "lat":36.7264, "lon":10.2917, "orientation":90, "type":"sandy"},
    {"name":"شاطئ البطان", "lat":36.7100, "lon":10.2700, "orientation":90, "type":"sandy"},
    {"name":"شاطئ خلاص", "lat":36.7972, "lon":10.2750, "orientation":90, "type":"sandy"},
]

def find_nearest_beach_orientation(lat: float, lon: float) -> Optional[int]:
    min_dist = float('inf')
    nearest_orient = None
    for b in TUNISIAN_BEACHES:
        dist = calc_distance(b["lat"], b["lon"], lat, lon)
        if dist < min_dist and dist < 20000:
            min_dist = dist
            nearest_orient = b["orientation"]
    return nearest_orient

def find_nearest_beach_type(lat: float, lon: float) -> Optional[str]:
    min_dist = float('inf')
    nearest_type = None
    for b in TUNISIAN_BEACHES:
        dist = calc_distance(b["lat"], b["lon"], lat, lon)
        if dist < min_dist and dist < 20000:
            min_dist = dist
            nearest_type = b["type"]
    return nearest_type

async def get_auto_orientation_overpass(lat, lon):
    for radius in [3000, 5000, 10000]:
        query = f"""[out:json];(way(around:{radius},{lat},{lon})["natural"="coastline"];);out geom;"""
        try:
            async with httpx.AsyncClient() as c:
                r = await c.get(OVERPASS_URL, params={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=20)
                r.raise_for_status()
                els = r.json().get("elements", [])
                if not els: continue
                best_dist, best_tangent, best_point = float('inf'), None, None
                for el in els:
                    geom = el.get("geometry", [])
                    if len(geom) < 2: continue
                    closest_idx = min(range(len(geom)), key=lambda i: calc_distance(lat, lon, geom[i]["lat"], geom[i]["lon"]))
                    p = geom[closest_idx]
                    d = calc_distance(lat, lon, p["lat"], p["lon"])
                    if d < best_dist:
                        best_dist, best_point = d, p
                        prev_i = closest_idx - 1 if closest_idx > 0 else 0
                        next_i = closest_idx + 1 if closest_idx < len(geom) - 1 else len(geom) - 1
                        if prev_i != next_i:
                            best_tangent = calc_bearing(geom[prev_i]["lat"], geom[prev_i]["lon"], geom[next_i]["lat"], geom[next_i]["lon"])
                if not best_tangent or not best_point: continue
                n_a, n_b = (best_tangent + 90) % 360, (best_tangent - 90) % 360
                c2u = calc_bearing(best_point["lat"], best_point["lon"], lat, lon)
                d_a = abs(c2u - n_a); d_a = 360 - d_a if d_a > 180 else d_a
                d_b = abs(c2u - n_b); d_b = 360 - d_b if d_b > 180 else d_b
                return int(round(((n_a if d_a < d_b else n_b) + 180) % 360))
        except Exception: continue
    return 0

@app.post("/auto-orientation")
@limiter.limit("5/minute")
async def auto_orientation(request: Request, req: AutoOrientationRequest):
    orientation = await get_auto_orientation_overpass(req.latitude, req.longitude)
    if orientation != 0: return {"orientation": orientation, "source": "overpass"}
    orientation = find_nearest_beach_orientation(req.latitude, req.longitude)
    if orientation is not None: return {"orientation": orientation, "source": "nearest_beach"}
    return {"orientation": -1, "source": "none", "message": "تعذر التحديد التلقائي."}

@app.post("/detect-bottom-type")
@limiter.limit("10/minute")
async def detect_bottom_type(request: Request, req: DetectBottomRequest):
    nearest_type = find_nearest_beach_type(req.latitude, req.longitude)
    if nearest_type: return {"bottom_type": nearest_type, "source": "nearby_beach", "confidence": "medium"}
    query = f"""[out:json];(
        way(around:2000,{req.latitude},{req.longitude})["natural"="sand"];
        way(around:2000,{req.latitude},{req.longitude})["natural"="shingle"];
        way(around:2000,{req.latitude},{req.longitude})["natural"="bare_rock"];
    );out body;"""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(OVERPASS_URL, params={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=15)
            r.raise_for_status()
            data = r.json()
            elements = data.get("elements", [])
            if elements:
                tags = elements[0].get("tags", {})
                nat = tags.get("natural", "")
                if nat == "sand": return {"bottom_type": "sandy", "source": "overpass", "confidence": "high"}
                if nat in ["shingle", "bare_rock"]: return {"bottom_type": "rocky", "source": "overpass", "confidence": "high"}
    except Exception: pass
    return {"bottom_type": "unknown", "source": "none", "confidence": "low"}

# ==================== جلب بيانات الطقس والبحر ====================
async def fetch_marine_data_from_openmeteo(lat: float, lon: float):
    url = "https://marine-api.open-meteo.com/v1/marine"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "wave_height,wave_period,wave_direction,swell_wave_height,swell_wave_period,swell_wave_direction,sea_surface_temperature",
        "timezone": "Africa/Tunis", "forecast_days": 3
    }
    try:
        r = await http_client.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"Open-Meteo marine fetch failed: {e}")
        return None

async def fetch_weather_data_from_openmeteo(lat: float, lon: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "wind_speed_10m,wind_direction_10m,wind_gusts_10m,pressure_msl,temperature_2m,precipitation,visibility,weather_code",
        "daily": "sunrise,sunset",
        "timezone": "Africa/Tunis", "forecast_days": 3
    }
    try:
        r = await http_client.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"Open-Meteo weather fetch failed: {e}")
        return None

# ==================== التحليلات الإضافية ====================
def analyze_weed_risk(sea_memory, wave_height, wind_direction, orient):
    risk = "منخفض"; advice = ""
    has_weed = "صوفة" in sea_memory or "أعشاب" in sea_memory
    if has_weed:
        risk = "مرتفع" if "تحذير صوفة" in sea_memory else "متوسط"
        advice = "استعمل صائدات مضادة للأعشاب وارمِ بزاوية حادة نحو البحر لتجنب الالتفاف."
    return {"risk": risk, "advice": advice}

def analyze_backwash(wind_speed: float, wind_dir: float, orient: float, wave_height: float) -> dict:
    wind_diff = angle_diff(wind_dir, orient)
    is_onshore = wind_diff < 30
    is_strong = wind_speed > 30
    severity = "منخفض"; effect = ""
    if is_onshore and is_strong and wave_height > 0.8:
        severity = "مرتفع"
        effect = f"رياح بحرية قوية ({wind_speed:.0f} كم/س) تضرب الموج نحو الشاطئ، ثم يرتد الموج بقوة نحو البحر. هذا يخلق تياراً عكسياً قوياً يسحب الرصاصة للشاطئ باستمرار وقد يدفن الخيط. حتى مع وزن ثقيل، يصعب تثبيت الطعم."
    elif is_onshore and wind_speed > 15:
        severity = "متوسط"
        effect = f"رياح بحرية ({wind_speed:.0f} كم/س) تخلق تياراً عكسياً خفيفاً. الرصاصة قد تتحرك قليلاً نحو الشاطئ لكن يمكن التحكم بها بوزن أثقل."
    return {"severity": severity, "effect": effect}

def analyze_debris_risk(sea_memory: str, past_rain: float, wind_speed: float) -> dict:
    has_floods = "سيول" in sea_memory
    has_weed = "صوفة" in sea_memory or "أعشاب" in sea_memory
    is_windy = wind_speed > 20
    risk = "منخفض"; effect = ""
    if has_floods and has_weed and is_windy:
        risk = "مرتفع"
        effect = "السيول الأخيرة حملت كميات كبيرة من الأعشاب والأغصان والمواد العضوية إلى البحر. هذه المواد تطفو الآن وتتجمع على الخيط والرصاصة، مما يزيد الوزن ويغير شكل الطعم. الأوساخ تسد العقد وتجعل الخيط مرئياً للسمك. يجب تنظيف الخيط كل بضع رميات، والصيد شبه مستحيل."
    elif has_weed and is_windy:
        risk = "متوسط"
        effect = "الأعشاب البحرية والصوفة تطفو بكثافة وتتجمع على الخيط. استعمل صائدات مضادة للأعشاب وتجنب الرمي في التيارات الجانبية."
    elif has_floods:
        risk = "متوسط"
        effect = "بعد السيول، قد تكون هناك مواد عالقة في الماء تتراكم على الخيط. انتبه لنظافة الخيط."
    return {"risk": risk, "effect": effect}

def get_seasonal_bait(month: int, water_temp: float) -> str:
    if month in [12,1,2]: bait = "السردين أو القمبري"
    elif month in [3,4,5]: bait = "الحبار أو الدود البحري"
    elif month in [6,7,8]: bait = "القمبري (الأفضل) أو الحبار"
    elif month in [9,10,11]: bait = "السردين المهاجر أو الحبار"
    else: bait = "القمبري (طوال السنة)"
    if water_temp > 22: bait += " (يفضل الطعم الحي أو المتحرك)"
    return bait

def calculate_confidence_index(period_flags: dict, is_mirror_sea: bool, has_golden: bool, nogo_count: int, warning_count: int,
                                block_wind_ok: bool, block_wave_ok: bool, is_night_with_tide: bool) -> int:
    base = 70
    if is_mirror_sea: base -= 40
    if not has_golden: base -= 20
    if nogo_count > 0: base -= 30
    if warning_count > 0: base -= 10
    if block_wind_ok: base += 10
    if block_wave_ok: base += 10
    if is_night_with_tide: base += 15
    base += period_flags.get("is_spring_tide", 0) * 15
    base += period_flags.get("is_pressure_dropping", 0) * 15
    return max(0, min(100, base))

# ==================== نظام الأوزان ====================
def apply_scoring(agg: dict) -> int:
    score = 50
    flags = agg["flags"]
    extra = agg["extra_info"]
    blocks = agg["blocks"]

    if not flags["has_golden_window"]:
        score -= 15
    if agg["avg_sst"] > 27.0:
        score -= 15
    elif agg["avg_sst"] < 13.0:
        score -= 15
    for b in blocks:
        if b.get("backwash", {}).get("severity") == "متوسط":
            score -= 10
            break
    press_change = abs(extra.get("pressure_change", 0))
    if 1.5 <= press_change <= 4.0:
        score -= 10
    if flags["is_lateral_strong"]:
        score -= 10
    for b in blocks:
        wp_val = b["_raw"]["wave_period"]
        if 0.1 < wp_val < 4.0:
            score -= 10
            break
    month = agg.get("target_month", datetime.now().month)
    if month in [3,4]:
        score -= 20

    for b in blocks:
        if b["wind_dir"].startswith("برية"):
            if b["_raw"]["avg_wind"] > 20:
                score -= 10
                break
            elif b["_raw"]["avg_wind"] <= 15:
                score += 5
                break

    if any(b["wind_dir"].startswith("بحرية") and b["_raw"]["avg_wind"] < 15 for b in blocks):
        score += 15
    if any(0.6 <= b["_raw"]["avg_wave_h"] <= 1.2 for b in blocks):
        score += 20
    if any(b.get("wave_angle_diff") and b["wave_angle_diff"] < 30 and b["_raw"]["avg_wave_h"] >= 0.6 for b in blocks):
        score += 15
    if flags["has_golden_window"]:
        score += 25
    if press_change < 1.0:
        score += 10
    if any(b.get("wave_angle_diff") and b["wave_angle_diff"] < 30 for b in blocks):
        score += 10
    moon_idx = agg["tide_analysis"]["idx"]
    if moon_idx == 4 and any(b["name"] == "الليل" for b in blocks):
        score += 15

    haml_delta = extra.get("haml_score_delta", 0)
    score += haml_delta

    score = max(0, min(100, score))
    return score

# ==================== محرك التجميع الفيزيائي ====================
def aggregate_physics(all_times, aligned, orient, target_date_obj, sunrise, sunset, latitude):
    tz = all_times[0].tzinfo if all_times else zoneinfo.ZoneInfo("UTC")
    target_start = datetime.combine(target_date_obj, datetime.min.time(), tzinfo=tz)
    target_end = target_start + timedelta(days=1)
    past_start = target_start - timedelta(hours=48)
    past_idx = [i for i, t in enumerate(all_times) if past_start <= t < target_start]
    target_idx = [i for i, t in enumerate(all_times) if target_start <= t < target_end]

    nogo_reasons = []
    warnings = []

    empty_res = {
        "sea_memory":"غير معروف","lateral_current":"غير معروف","pressure_state":"مستقر","tide_analysis":{},"sst_stability":"مستقر","bio_matrix":{},"avg_sst":0,"hidden_factors":{},"blocks":[],"red_flags":[],"green_flags":[],"extra_info":{}, "transitions":[], "flags":{}, "nogo_reasons":[], "warnings":[], "final_verdict": "غير مناسب", "score": 0
    }
    if not target_idx:
        return empty_res

    def pick(k): 
        arr = aligned.get(k, [])
        return [arr[i] if i < len(arr) else 0.0 for i in target_idx]

    wh = pick("wave_height"); wp = pick("wave_period"); swh = pick("swell_wave_height")
    swp = pick("swell_wave_period"); swd = pick("swell_wave_direction"); wd_wave = pick("wave_direction")
    sst = pick("sea_surface_temperature"); ws = pick("wind_speed_10m"); wd = pick("wind_direction_10m")
    wg = pick("wind_gusts_10m"); pr = pick("pressure_msl"); ta = pick("temperature_2m"); prec = pick("precipitation")
    vis = pick("visibility"); wcode = [int(v) if v else 0 for v in pick("weather_code")]

    wave_power = [0.49*(h**2)*p for h,p in zip(wh,wp)]
    wind_cls = [wind_class_detailed(angle_diff(d, orient)) for d in wd]

    has_swell_data = len(swh) > 0 and not all(v == 0.0 for v in swh)
    actual_swell_exists = has_swell_data and max(swh) > 0.05

    sea_memory = "بحر صافي وهادئ"
    past_rain_total = 0.0
    if past_idx:
        p_wh = aligned.get("wave_height", []); p_wp = aligned.get("wave_period", [])
        p_swh = aligned.get("swell_wave_height", []); p_swp = aligned.get("swell_wave_period", [])
        p_ws = aligned.get("wind_speed_10m", []); p_wd = aligned.get("wind_direction_10m", [])
        p_prec = aligned.get("precipitation", [])
        valid_past = [i for i in past_idx if i < len(p_wh) and i < len(p_wp) and i < len(p_ws) and i < len(p_wd)]
        if valid_past:
            weighted_past_power = 0.0; weighted_past_swh = 0.0; total_weight = 0.0; past_onshore_hours = 0.0; past_rain = 0.0
            for pos, i in enumerate(valid_past):
                decay_weight = 0.8 ** (len(valid_past) - 1 - pos)
                weighted_past_power += decay_weight * 0.49*(p_wh[i]**2)*p_wp[i]
                weighted_past_swh += decay_weight * p_swh[i] if i < len(p_swh) else 0
                total_weight += decay_weight
                if wind_class_detailed(angle_diff(p_wd[i], orient)).startswith("بحرية"): past_onshore_hours += decay_weight
                past_rain += p_prec[i] if i < len(p_prec) else 0
            past_avg = weighted_past_power / total_weight if total_weight > 0 else 0
            past_sh = weighted_past_swh / total_weight if total_weight > 0 else 0
            past_onshore_ratio = past_onshore_hours / total_weight
            past_rain_total = past_rain / len(valid_past)
            if past_avg > 6.0 and past_onshore_ratio > 0.4: sea_memory = "بحر خامر وعكر جداً."
            elif past_avg > 4.0 and past_onshore_ratio > 0.3: sea_memory = "بحر يعكر ببطء."
            if past_sh > 0.8 and past_avg > 4.0: sea_memory += " | تحذير صوفة."
            if past_rain_total > 10.0: sea_memory += " | سيول."

    lateral_fx = 0.0; lateral_fy = 0.0; max_wh = max(wh) if wh else 0.0
    for i in range(len(wh)):
        w_dir = wd_wave[i] if i < len(wd_wave) else 0.0
        if w_dir != 0.0:
            signed_angle = math.radians(signed_angle_diff(w_dir, orient))
            force = wh[i] * wh[i] 
            lateral_fx += force * math.sin(signed_angle)
            lateral_fy += force * math.cos(signed_angle)

    total_force = math.sqrt(lateral_fx**2 + lateral_fy**2)
    lateral_force_ratio = abs(lateral_fx) / total_force if total_force > 0 else 0
    avg_wave_h = sum(wh) / len(wh) if wh else 0
    is_mirror_sea = max_wh < 0.3
    is_lateral_strong = lateral_force_ratio > 0.7 and avg_wave_h > 0.6

    if is_mirror_sea: nogo_reasons.append("بحر مرآوي تام (أقل من 0.3م): لا تيارات ولا حركة سطحية، الأسماك لا تقترب.")

    lateral_dir_text = "لليمين" if lateral_fx > 0 else "لليسار" if lateral_fx < 0 else ""
    lateral_current = "تيار جانبي معدوم (بحر مرآوي)" if is_mirror_sea else f"تيار جانبي قوي {lateral_dir_text}" if is_lateral_strong else f"تيار جانبي متوسط {lateral_dir_text}" if (lateral_force_ratio > 0.4 and avg_wave_h > 0.4) else "تيار جانبي ضعيف"

    cross_angles = []
    if has_swell_data:
        cross_angles = [angle_diff(swd[i], wd_wave[i]) for i in range(len(swd)) if swd[i] != 0.0 and i < len(wd_wave) and wd_wave[i] != 0.0]
    is_cross_sea_dangerous = False
    cross_sea_risk = "منخفض"
    if cross_angles and not is_mirror_sea:
        avg_cross, max_cross = sum(cross_angles) / len(cross_angles), max(cross_angles)
        if max_cross > 60 and avg_cross > 40: cross_sea_risk = "بحر مختلط وخطير"; is_cross_sea_dangerous = True
        elif max_cross > 45: cross_sea_risk = "بحر مختلط متوسط"
    if is_cross_sea_dangerous: nogo_reasons.append("بحر مختلط خطير: السويل والموج المحلي يتقاطعان بزاوية كبيرة.")

    steepness_vals = [h / (1.56 * (p**2)) for h, p in zip(wh, wp) if p and p > 0.1]
    avg_steepness = sum(steepness_vals) / len(steepness_vals) if steepness_vals else 0
    if avg_steepness > 0.06:
        steepness_desc = "موج حاد وقصير"
    elif avg_steepness < 0.03:
        steepness_desc = "موج منخفض الانحدار (سلس)"
    else:
        steepness_desc = "موج متوسط الانحدار"

    tide_analysis = get_moon_and_tide_analysis(target_date_obj)
    tidal_windows, golden_windows = estimate_tidal_windows(target_date_obj, tide_analysis, sunrise, sunset, latitude)
    solunar = calculate_solunar(target_date_obj, latitude)

    moon_age = get_moon_age_days(target_date_obj)
    haml_info = get_haml_mat_status(moon_age)
    platform_advice = get_fishing_platform_advice(haml_info["status"])

    is_neap_tide = tide_analysis["idx"] in [2, 6]
    is_spring_tide = tide_analysis["idx"] in [0, 4]
    if is_neap_tide: warnings.append(f"مد ضعيف (مد محاقي - {tide_analysis['name']}): تيارات غذائية ضعيفة، الأسماك أقل تجمعاً.")
    has_golden_window = any("تزامن" in g for g in golden_windows)
    if not has_golden_window: warnings.append("لا توجد ساعة ذهبية: قد يقل نشاط الأسماك.")

    avg_sst = sum(sst)/len(sst) if sst else 0
    sst_diff = max(sst) - min(sst) if len(sst) > 1 else 0
    sst_stability = "صدمة حرارية" if sst_diff > 2.0 else "تغير بطيء" if sst_diff > 1.0 else "مستقر تماماً"
    is_murky = "عكر" in sea_memory or "خامر" in sea_memory
    is_weedy = "صوفة" in sea_memory
    max_air_temp = max(ta) if ta else 0
    month = target_date_obj.month
    seabass_sst_limit = 20.0 if month in [6,7,8,9] else 18.0
    if avg_sst > seabass_sst_limit: warnings.append(f"حرارة ماء عالية ({avg_sst:.1f}°م): تتجاوز حد القاروص.")

    avg_press = sum(pr)/len(pr) if pr else 0
    press_change = pr[-1] - pr[-4] if len(pr) >= 4 else (pr[-1] - pr[0] if len(pr) > 1 else 0)
    is_pressure_rising_fast = press_change > 1.5
    is_pressure_dropping_fast = press_change < -2.0
    is_pressure_shock = abs(press_change) > 4.0
    if is_pressure_shock:
        nogo_reasons.append(f"اضطراب حاد في الضغط الجوي (> 4 hPa/3h): الأسماك تتوقف عن الأكل تماماً.")
    elif is_pressure_rising_fast:
        warnings.append(f"ضغط مرتفع ({press_change:+.1f} hPa/3h): قد يبطئ النشاط.")
    elif is_pressure_dropping_fast:
        warnings.append(f"ضغط منخفض ({press_change:.1f} hPa/3h): قد تبدأ الأسماك بالتغذية قبل العاصفة.")

    pressure_note = "ينخفض (إيجابي)" if is_pressure_dropping_fast else "مستقر (محايد)" if not is_pressure_rising_fast else "يرتفع (سلبي)"
    pressure_state = f"انخفاض ({press_change:.1f} hPa)" if is_pressure_dropping_fast else f"ارتفاع ({press_change:+.1f} hPa)" if is_pressure_rising_fast else f"مستقر ({press_change:+.1f} hPa)"

    bio_matrix = {
        "قاروص": {
            "status": "نشط جداً" if (avg_sst < seabass_sst_limit and is_murky) else "نشط" if avg_sst < seabass_sst_limit else "غائب تقريباً",
            "reason": f"يحتاج عكراً ودرجة أقل من {seabass_sst_limit}°م. الضغط: {pressure_note}.",
            "preferences": "14-20°م | متوسط الهيجان | تيار معتدل | عكارة خفيفة-متوسطة | الفجر، الغروب، الليل | السردين، القمبري، الحبار"
        },
        "دنيس": {
            "status": "نشط" if (avg_sst > 18 and not is_mirror_sea) else "خامل",
            "reason": "يحب الماء الدافئ قرب الصخور، الموج والعكارة الخفيفة تخرجه للبحث عن القشريات.",
            "preferences": "18-24°م | هادئ-متوسط | تيار ضعيف-متوسط | عكارة خفيفة | الفجر، الغروب | الحبار، الدود، القمبري"
        },
        "بوري": {
            "status": "نشط" if (not is_murky and not is_weedy and not is_mirror_sea) else "خامل",
            "reason": "يحتاج حركة سطحية. في بحر مرآوي يختفي تماماً." if is_mirror_sea else "يحتاج حركة سطحية.",
            "preferences": "16-26°م | أي حالة | تيار سطحي للعوالق | ماء نظيف | النهار كله | عجينة، دود، خبز"
        },
        "سارغ": {
            "status": "ضعيف" if avg_sst > 22 else "نشط",
            "reason": "يتأثر بالحرارة." if avg_sst > 22 else "درجة الحرارة مناسبة.",
            "preferences": "16-22°م | هادئ-متوسط | أي تيار | ماء نظيف | الفجر والغروب | القمبري، الحبار، الدود"
        },
        "مرمار": {
            "status": "نشط" if (avg_sst > 18 and not is_mirror_sea and lateral_force_ratio > 0.2) else "خامل",
            "reason": "يتجمع في أسراب نهاراً، يحتاج تياراً يجلب العوالق.",
            "preferences": "18-26°م | هادئ-متوسط | تيار معتدل | ماء نظيف-قليل العكارة | النهار | دود، قمبري صغير، عجينة"
        },
        "شلبة": {
            "status": "نشط" if (avg_sst > 17 and not is_mirror_sea) else "خامل",
            "reason": "يفضل المياه الدافئة قرب الصخور والأعشاب.",
            "preferences": "17-25°م | هادئ-متوسط | تيار معتدل | ماء نظيف | الصباح الباكر | القمبري، الحبار"
        },
        "تريلية": {
            "status": "نشط" if (avg_sst > 18 and not is_murky) else "خامل",
            "reason": "تحب المياه الدافئة والواضحة، تتجمع في أسراب نهاراً.",
            "preferences": "18-26°م | هادئ | أي تيار | ماء نظيف | النهار | السردين، العجينة"
        },
        "بغبغان": {
            "status": "نشط" if (avg_sst > 19 and is_murky) else "خامل",
            "reason": "يحب العكارة والمياه الدافئة، ينشط قرب القاع.",
            "preferences": "19-27°م | متوسط الهيجان | تيار معتدل | عكارة | الليل | السردين، القمبري"
        },
        "سوبيا": {
            "status": "نشط" if (avg_sst > 18 and not is_mirror_sea) else "خامل",
            "reason": "يظهر ليلاً مع المد العالي، يحب الأضواء والقاع الرملي.",
            "preferences": "18-24°م | هادئ-متوسط | تيار قوي | ماء نظيف | الليل مع المد العالي | السردين، الحبار"
        }
    }

    peak_gust = max(wg) if wg else 0.0
    if peak_gust > 60:
        nogo_reasons.append(f"رياح عاتية (هبات {peak_gust:.0f} > 60 كم/س): تمنع الرمي الآمن وقد تقطع الخيط.")

    dominant = max(set(wind_cls), key=wind_cls.count) if wind_cls else "غير معروف"
    periods = defaultdict(list)
    for idx, i in enumerate(target_idx):
        h = all_times[i].hour
        if 4 <= h <= 11: periods["morning"].append(idx)
        elif 12 <= h <= 17: periods["afternoon"].append(idx)
        else: periods["night"].append(idx)

    def parse_tidal_time(t_str):
        parts = t_str.split(":")
        return int(parts[0]) + int(parts[1])/60

    slack_info = ""
    for hw_key, lw_key in [("HW1","LW1"), ("HW2","LW2")]:
        hw_t = parse_tidal_time(tidal_windows[hw_key])
        lw_t = parse_tidal_time(tidal_windows[lw_key])
        slack_info += f"المد العالي {tidal_windows[hw_key]} مياه ميتة: {format_time(hw_t-0.75)}-{format_time(hw_t+0.75)}; الجزر المنخفض {tidal_windows[lw_key]} مياه ميتة: {format_time(lw_t-0.75)}-{format_time(lw_t+0.75)}; "
    slack_info = slack_info.rstrip("; ")

    blocks = []
    for key in ["morning", "afternoon", "night"]:
        idxs = periods[key]
        if not idxs: continue
        avg_h = sum(wh[i] for i in idxs)/len(idxs)
        max_h = max(wh[i] for i in idxs)
        avg_w = sum(ws[i] for i in idxs)/len(idxs)
        max_w = max(ws[i] for i in idxs)
        wc_dom = max(set(wind_cls[i] for i in idxs), key=wind_cls.count)
        avg_swh_b = sum(swh[i] for i in idxs)/len(idxs)
        avg_swp_b = sum(swp[i] for i in idxs)/len(idxs)
        avg_swd_b = sum(swd[i] for i in idxs)/len(idxs) if swd else 0
        avg_wave_dir = sum(wd_wave[i] for i in idxs)/len(idxs) if wd_wave else 0
        avg_air = sum(ta[i] for i in idxs)/len(idxs) if ta else 0
        max_gust_b = max(wg[i] for i in idxs)
        most_code = max(set(wcode[i] for i in idxs), key=wcode.count) if idxs else 0
        avg_press_b = sum(pr[i] for i in idxs)/len(idxs) if pr else 0
        avg_vis_b = sum(vis[i] for i in idxs)/len(idxs) if vis else 0
        avg_wp_b = sum(wp[i] for i in idxs)/len(idxs) if wp else 0
        avg_wd_b = sum(wd[i] for i in idxs)/len(idxs) if wd else 0

        if max_h < 0.4:
            sea = "بحر مرآوي"
        elif max_h < 0.9:
            sea = "هادئ"
        elif max_h < 1.3:
            sea = "متوسط الهيجان"
        else:
            sea = "هائج"

        if max_h > 2.0 and not any("بحر هائج" in r for r in nogo_reasons):
            nogo_reasons.append(f"بحر هائج (أمواج > 2.0م في {key}): الرمي مستحيل والخطر كبير.")
        if avg_wp_b > 10.0 and avg_h > 0.8 and not any("أمواج أرضية" in r for r in nogo_reasons):
            nogo_reasons.append(f"أمواج أرضية عالية الطاقة (فترة {avg_wp_b:.1f} > 10 ثوانٍ مع ارتفاع {avg_h:.2f}م): تجرف الرصاص وتدفن الخيوط.")
        if most_code in [95, 96, 99] and not any("الصواعق" in r for r in nogo_reasons):
            nogo_reasons.append("خطر الصواعق والبرق: قصبة الكاربون تجذب البرق، لا ترفعها.")

        backwash = analyze_backwash(avg_w, avg_wd_b, orient, avg_h)
        if backwash["severity"] == "مرتفع" and not any("يرجع الرصاص" in r for r in nogo_reasons):
            nogo_reasons.append(f"تيار راجع عنيف: {backwash['effect']}")

        debris = analyze_debris_risk(sea_memory, past_rain_total, avg_w)
        if debris["risk"] == "مرتفع" and not any("أوساخ" in r for r in nogo_reasons):
            nogo_reasons.append(f"أوساخ وصوفة كثيفة: {debris['effect']}")

        final_swd = None if avg_swd_b == 0.0 else avg_swd_b
        final_wd = None if avg_wave_dir == 0.0 else avg_wave_dir
        swell_angle = angle_diff(final_swd, orient) if final_swd else None
        wave_angle = angle_diff(final_wd, orient) if final_wd else None

        swell_wave_interaction = "متوافقان"
        if not is_mirror_sea and swell_angle is not None and wave_angle is not None and final_swd and final_wd:
            diff_sw = angle_diff(final_swd, final_wd)
            if diff_sw > 40: swell_wave_interaction = "متقاطعان بشدة"
            elif diff_sw > 25: swell_wave_interaction = "متقاطعان بسيط"

        wind_effect_dist = 0
        if "بحرية" in wc_dom: wind_effect_dist = avg_w * 1.2
        elif "برية" in wc_dom: wind_effect_dist = -avg_w * 1.0

        block_wind_ok = (avg_w < 20 and wc_dom.startswith("بحرية")) or (wc_dom.startswith("برية") and avg_w <= 15)
        block_wave_ok = 0.6 <= avg_h <= 1.2
        is_night = (key == "night")
        is_night_with_tide = is_night and is_close(parse_tidal_time(tidal_windows["HW2"]), safe_parse_time(sunset), 1.5)
        period_flags = {"is_spring_tide": 1 if is_spring_tide else 0, "is_pressure_dropping": 1 if is_pressure_dropping_fast else 0}
        confidence = calculate_confidence_index(period_flags, is_mirror_sea, has_golden_window, len(nogo_reasons), len(warnings),
                                                block_wind_ok, block_wave_ok, is_night_with_tide)

        base_dist = 50
        if avg_h > 0.8: base_dist = 40
        elif avg_h > 0.5: base_dist = 50
        else: base_dist = 60
        recommended_dist = max(20, min(80, base_dist + wind_effect_dist * 2))

        block_data = {
            "name":{"morning":"الصباح","afternoon":"الظهيرة","night":"الليل"}[key],
            "time_range":f"{all_times[target_idx[idxs[0]]].strftime('%H:%M')}-{all_times[target_idx[idxs[-1]]].strftime('%H:%M')}",
            "sea_state":sea,"wave_height":f"أقصى {max_h:.2f}م",
            "swell_dir": deg_to_compass(final_swd) if final_swd else ("معدوم" if not actual_swell_exists else "غير معروف"),
            "wave_dir": deg_to_compass(final_wd) if final_wd else "غير معروف",
            "swell_angle_diff": round(swell_angle,0) if swell_angle is not None else None,
            "wave_angle_diff": round(wave_angle,0) if wave_angle is not None else None,
            "swell_wave_interaction": swell_wave_interaction,
            "wind_speed":f"متوسط {avg_w:.1f} - أقصى {max_w:.1f} كم/س",
            "wind_gust_peak":round(max_gust_b,1),
            "wind_dir":wc_dom, "air_temp":round(avg_air,1), "weather":weather_desc(most_code),
            "confidence": confidence,
            "recommended_cast_distance": round(recommended_dist, 0),
            "backwash": backwash,
            "debris": debris,
            "_raw": {
                "avg_wave_h": round(avg_h, 3), "max_wave_h": round(max_h, 3),
                "avg_wind": round(avg_w, 1), "max_wind": round(max_w, 1),
                "max_gust": round(max_gust_b, 1),
                "swell_h": round(avg_swh_b, 3), "swell_p": round(avg_swp_b, 1),
                "air_temp": round(avg_air, 1), "pressure": round(avg_press_b, 1),
                "visibility": round(avg_vis_b, 0),
                "has_swell": actual_swell_exists,
                "wave_period": round(avg_wp_b, 1),
                "wind_effect_dist": round(wind_effect_dist, 0),
                "recommended_cast_distance": round(recommended_dist, 0)
            }
        }
        blocks.append(block_data)

    reds, greens = [], []
    for i in range(len(wh)):
        hh = all_times[target_idx[i]].strftime("%H:%M")
        if wave_power[i] > 3 or wh[i] > 1.8 or wg[i] > 50 or pr[i] < 1005: reds.append(hh)
        if 0.3 <= wh[i] <= 1 and 0.1 <= wave_power[i] <= 1.5 and ws[i] < 27.8: greens.append(hh)

    weed_analysis = analyze_weed_risk(sea_memory, wh, wd, orient)
    seasonal_bait = get_seasonal_bait(month, avg_sst)

    extra = {
        "pressure_avg":round(avg_press,1), "peak_gust_today":round(peak_gust,1),
        "sunrise":sunrise, "sunset":sunset, "max_air_temp": round(max_air_temp, 1),
        "is_mirror_sea": is_mirror_sea, "tidal_windows": tidal_windows, "golden_windows": golden_windows,
        "has_swell_data": has_swell_data, "actual_swell_exists": actual_swell_exists,
        "solunar": solunar, "slack_times": slack_info,
        "weed_risk": weed_analysis, "seasonal_bait": seasonal_bait,
        "past_rain_total": round(past_rain_total, 1),
        "pressure_change": round(press_change, 1),
        "moon_age_days": round(moon_age, 1),
        "haml_status": haml_info["status"],
        "haml_phase": haml_info["phase"],
        "haml_description": haml_info["description"],
        "haml_score_delta": haml_info["score_delta"],
        "platform_advice": platform_advice
    }

    flags = {
        "is_mirror_sea": is_mirror_sea, "is_lateral_strong": is_lateral_strong,
        "is_pressure_rising_fast": is_pressure_rising_fast, "is_pressure_dropping_fast": is_pressure_dropping_fast,
        "is_cross_sea_dangerous": is_cross_sea_dangerous, "is_murky": is_murky, "is_weedy": is_weedy,
        "has_golden_window": has_golden_window, "is_neap_tide": is_neap_tide, "is_spring_tide": is_spring_tide
    }

    critical_nogo = [r for r in nogo_reasons if any(kw in r for kw in [
        "مرآوي", "مختلط خطير", "أوساخ", "صوفة", "يرجع الرصاص", "رياح عاتية",
        "فترة موجة عالية", "صواعق", "برق", "اضطراب حاد في الضغط", "هائج"
    ])]
    if critical_nogo:
        final_verdict = "غير مناسب"
    elif warnings:
        final_verdict = "فرصة مع تحفظات"
    else:
        final_verdict = "مناسب"

    agg_result = {
        "dominant_wind":dominant, "blocks":blocks, "red_flags":reds[:5], "green_flags":greens[:5],
        "sea_memory":sea_memory, "lateral_current":lateral_current, "pressure_state":pressure_state,
        "tide_analysis":tide_analysis, "sst_stability":sst_stability,
        "hidden_factors": {"cross_sea_risk": cross_sea_risk, "wave_steepness": steepness_desc, "golden_lock": "مد قوي" if is_spring_tide else "مد ضعيف" if is_neap_tide else "متوسط"},
        "bio_matrix":bio_matrix, "avg_sst":round(avg_sst,1), "extra_info":extra,
        "transitions": [], "flags": flags,
        "nogo_reasons": nogo_reasons, "warnings": warnings,
        "final_verdict": final_verdict,
        "target_month": target_date_obj.month
    }
    score = apply_scoring(agg_result)
    agg_result["score"] = score
    return agg_result

# ==================== التفكيك الديناميكي ====================
def format_tidal_flow_periods(tidal_windows: dict) -> str:
    flows = []
    for key, time_str in tidal_windows.items():
        h = safe_parse_time(time_str)
        start = format_time((h - 1) % 24)
        end = format_time((h + 1) % 24)
        if key.startswith("HW"):
            flows.append(f"المد العالي ({time_str}): {start} - {end}")
        else:
            flows.append(f"الجزر المنخفض ({time_str}): {start} - {end}")
    return flows

def calculate_interactions(agg: dict) -> List[str]:
    interactions = []
    flags = agg.get("flags", {})
    extra = agg.get("extra_info", {})
    blocks = agg.get("blocks", [])
    is_mirror_sea = flags.get("is_mirror_sea", False)
    is_pressure_rising_fast = flags.get("is_pressure_rising_fast", False)
    is_pressure_dropping_fast = flags.get("is_pressure_dropping_fast", False)
    has_golden_window = flags.get("has_golden_window", False)
    is_neap_tide = flags.get("is_neap_tide", False)
    is_spring_tide = flags.get("is_spring_tide", False)

    golden_windows = extra.get("golden_windows", [])
    tidal_windows = extra.get("tidal_windows", {})
    tide_analysis = agg.get("tide_analysis", {})
    bio_matrix = agg.get("bio_matrix", {})
    sea_memory = agg.get("sea_memory", "")
    avg_sst = agg.get("avg_sst", 0)
    pressure_state = agg.get("pressure_state", "")
    nogo_reasons = agg.get("nogo_reasons", [])
    warnings = agg.get("warnings", [])
    final_verdict = agg.get("final_verdict", "غير مناسب")
    solunar = extra.get("solunar", {})
    weed_risk = extra.get("weed_risk", {})
    seasonal_bait = extra.get("seasonal_bait", "غير محدد")
    slack_info = extra.get("slack_times", "غير محدد")
    hidden_factors = agg.get("hidden_factors", {})
    sunrise = extra.get("sunrise", "06:00")
    sunset = extra.get("sunset", "18:00")

    interactions.append(f"[التوقيت الأساسي] المد العالي الأول: {tidal_windows.get('HW1')} | الجزر المنخفض الأول: {tidal_windows.get('LW1')} | المد العالي الثاني: {tidal_windows.get('HW2')} | الجزر المنخفض الثاني: {tidal_windows.get('LW2')}")
    interactions.append(f"[القمر والمد] القمر: {tide_analysis.get('name')}. قوة المد: {tide_analysis.get('tide_strength')}")
    
    haml_status = extra.get("haml_status", "")
    haml_phase = extra.get("haml_phase", "")
    haml_desc = extra.get("haml_description", "")
    interactions.append(f"[مؤشر الشاطئ] {haml_status} ({haml_phase}). {haml_desc}")
    interactions.append(f"[انحدار الموج] {hidden_factors.get('wave_steepness', 'متوسط')}")

    interactions.append(f"[فترات سولونار] رئيسي: {solunar.get('major1')} و {solunar.get('major2')} | ثانوي: {solunar.get('minor1')} و {solunar.get('minor2')}")

    flows = format_tidal_flow_periods(tidal_windows)
    flow_text = " | ".join(flows)
    interactions.append(f"[فترات الجريان] {flow_text}")
    interactions.append(f"[المياه الميتة] {slack_info}")

    platform_advice = extra.get("platform_advice", "")
    if platform_advice:
        interactions.append(f"[نصيحة الصيد] {platform_advice}")

    if is_neap_tide:
        interactions.append(f"[تأثير المد] مد محاقي ضعيف، تيارات غذائية ضعيفة.")
    elif not is_spring_tide:
        interactions.append(f"[تأثير المد] مد متوسط، تيارات معتدلة.")

    current_sea_state = blocks[0]["sea_state"] if blocks else "غير معروف"
    interactions.append(f"[حالة البحر العامة] {current_sea_state}")

    for b in blocks:
        name = b['name']; time_range = b['time_range']; raw = b.get("_raw", {})
        sea_state = b['sea_state']; wind_cls = b['wind_dir']
        avg_wind = raw.get("avg_wind", 0); max_gust = raw.get("max_gust", 0)
        wind_effect = raw.get("wind_effect_dist", 0)
        wave_p = raw.get("wave_period", 0)
        recommended_dist = raw.get("recommended_cast_distance", 50)

        interactions.append(f"[{name} ({time_range})]")
        interactions.append(f"  البحر: {sea_state} | الثقة: {b.get('confidence',0)}%")
        sign = '+' if wind_effect >= 0 else ''
        interactions.append(f"  الرياح: {wind_cls} {avg_wind:.1f} كم/س (هبات {max_gust} كم/س) | تأثير الرمي: {sign}{wind_effect:.0f}م")
        interactions.append(f"  الموج: مباشر وقصير ({wave_p} ث) | المسافة: {recommended_dist:.0f}م")

        active_fish = []
        inactive_fish = []
        for fish, data in bio_matrix.items():
            if "نشط" in data['status']:
                active_fish.append(fish)
            else:
                inactive_fish.append(fish)
        interactions.append(f"  نشط: {', '.join(active_fish) if active_fish else 'لا يوجد'}")
        interactions.append(f"  خامل: {', '.join(inactive_fish) if inactive_fish else 'لا يوجد'}")

    interactions.append(f"[حرارة الماء] {avg_sst}°م. الاستقرار: {agg.get('sst_stability')}")
    interactions.append(f"[ذاكرة البحر] {sea_memory}")
    interactions.append(f"[الضغط] {pressure_state}")
    interactions.append(f"[الطعم الموسمي] {seasonal_bait}")
    interactions.append(f"[نسبة النجاح] {agg.get('score', 0)}%")

    if final_verdict == "فرصة مع تحفظات":
        interactions.append(f"[الحسم النهائي] فرصة مع تحفظات: {', '.join(warnings) if warnings else 'ظروف متغيرة'}")
    elif final_verdict == "غير مناسب":
        reasons = " | ".join(nogo_reasons) if nogo_reasons else "ظروف غير كافية"
        interactions.append(f"[الحسم النهائي] غير مناسب: {reasons}")
    else:
        interactions.append(f"[الحسم النهائي] مناسب، الظروف ممتازة.")
    return interactions

def build_context(req, agg, tz_name):
    beach = "رملي" if req.beach_type == "sandy" else "صخري"
    extra = agg["extra_info"]
    chain_interactions = calculate_interactions(agg)

    facts = [
        f"شاطئ {beach} (اتجاه {req.beach_orientation}°).",
        f"ذاكرة البحر: {agg['sea_memory']}",
        f"الضغط: {agg['pressure_state']}",
        f"القمر: {agg['tide_analysis']['tide_strength']}.",
        f"حرارة الماء: {agg['avg_sst']}°م. الهواء: {extra.get('max_air_temp', 'N/A')}°م.",
        f"مؤشر الشاطئ: {extra.get('haml_status', 'غير معروف')} ({extra.get('haml_phase', '')}).",
        f"نصيحة الصيد: {extra.get('platform_advice', '')}",
    ]
    final_verdict = agg["final_verdict"]
    score = agg.get("score", 0)
    seasonal_bait = extra.get("seasonal_bait", "الطعم الموسمي")

    if final_verdict == "مناسب":
        summary = f"✅ مناسب ({score}%) – الظروف ممتازة. الطعم: {seasonal_bait}."
    elif final_verdict == "فرصة مع تحفظات":
        summary = f"⚠️ فرصة مع تحفظات ({score}%) – توجد فرصة. الطعم: {seasonal_bait}."
    else:
        summary = f"❌ غير مناسب ({score}%) – لا توجد فرصة."

    lines = [
        f"[الملخص التنفيذي] {summary}",
        "",
        "=== معلومات أساسية ===", "\n".join(facts), "\n",
        "=== التفاعلات ===", *chain_interactions
    ]
    return "\n".join(lines)

SYSTEM_PROMPT = """أنت خبير سيرفكاستينغ تونسي. تكتب تقارير احترافية فخمة.
القرار النهائي موجود في [الحسم النهائي]. لا تغيره.
نسبة النجاح في [نسبة النجاح].

استخدم الهيكل التالي مع أيقونات:

🎯 0. الملخص التنفيذي
> نسبة النجاح: (أدخل النسبة). القرار: (أدخل القرار). الطعم: (أدخل الطعم).

⏱️ 1. التوقيت المدوي وحركة المياه
🌊 مواقيت المد والجزر: اذكر الأوقات (المد العالي الأول، الجزر المنخفض الأول...).
🏖️ مؤشر الشاطئ (أيام الحياء والمات): (أدخل حالة الحياء/المات). (أدخل وصف التأثير).
⏳ فترات سولونار: رئيسي وثانوي.

🏃‍♂️ 2. فترات الحركة مقابل المياه الميتة
لكل مد/جزر: 🟢 فترة الجريان، 🔴 المياه الميتة. استخدم النطاقات الزمنية الصحيحة (من الأصغر إلى الأكبر).

🕒 3. التفكيك الديناميكي الزمني
لكل فترة (صباح، ظهيرة، ليل):
- حالة البحر والثقة.
- الرياح وتأثيرها على الرمي (أشر إلى + إذا كانت تزيد المسافة).
- الموج والمسافة المقترحة.
- الأسماك النشطة والخاملة (بدون تكرار).
- ربط العوامل.

⚖️ 4. ميزان العوامل الميدانية
🔴 العوامل الحمراء: ...
🟢 العوامل الإيجابية: ...

🏹 5. التكتيك الميداني والسلامة
- الرصاص، التوقيت، المسافة، الطعم، المكان (الشاطئ فقط).
- ⚠️ إرشادات السلامة.

📊 6. الأرقام المرجعية
(أدرج الأرقام المرجعية لكل فترة).

قواعد:
- لا تكسر أي وقت أو رقم. الوقت كامل في سطر واحد.
- استخدم دائمًا النطاقات الزمنية من الأصغر إلى الأكبر.
- لا تذكر المركب.
- لا تستخدم حروفًا لاتينية.
- اكتب بالدارجة التونسية.
- تأكد من أن تأثير الرياح البحري يظهر بإشارة '+' لزيادة مسافة الرمي.
- كل نقطة بيانات (مد، جزر، تيار، مؤشر...) يجب أن تكون في سطر منفصل. استخدم دائماً علامة * أو - في بداية السطر، ثم أيقونة، ثم النص. مثال صحيح: * 🔹 المد العالي الأول: الساعة 10:11. مثال خاطئ: 🌊 مواقيت المد والجزر: - المد العالي الأول: 10:11 - .... لا تكتب سلاسل متصلة بشرطات.
"""

async def call_openrouter(ctx):
    headers = {"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json"}
    payload = {"model":MODEL_NAME,"messages":[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":ctx}],"max_tokens":12000,"temperature":0.1}
    data = await post_with_retry(OPENROUTER_URL, payload, headers)
    if "choices" in data and data["choices"]:
        return data["choices"][0]["message"]["content"]
    raise Exception("OpenRouter استجابة فارغة")

# ==================== دوال معالجة النص ====================
def fix_time_ranges(text: str) -> str:
    pattern = r'(\d{2}:\d{2})\s*[-–]\s*(\d{2}:\d{2})'
    def repl(m):
        t1, t2 = m.group(1), m.group(2)
        to_min = lambda s: int(s.split(':')[0])*60 + int(s.split(':')[1])
        if to_min(t1) > to_min(t2):
            return f"{t2} - {t1}"
        return f"{t1} - {t2}"
    return re.sub(pattern, repl, text)

def enforce_line_breaks(text: str) -> str:
    """
    يضمن أن كل نقطة تبدأ بـ * أو - أو أيقونة تكون في سطر منفصل.
    يفصل الجمل المتلاصقة التي تستخدم شرطات متتالية.
    """
    lines = text.split('\n')
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            new_lines.append('')
            continue
        # إذا كان السطر يحتوي على " - " متعددة، نفصلها إلى أسطر مستقلة
        if ' - ' in stripped:
            # نبحث عن نمط مثل: "المد العالي الأول: 10:11 - الجزر المنخفض الأول: 16:23 - ..."
            parts = re.split(r'\s+-\s+', stripped)
            # إذا كان هناك أكثر من جزأين، نفصل
            if len(parts) > 1:
                # الجزء الأول يبقى كما هو
                first = parts[0].strip()
                if first:
                    new_lines.append(first)
                # الأجزاء التالية تحصل على شرطة في البداية
                for part in parts[1:]:
                    part = part.strip()
                    if part:
                        new_lines.append(f"   * {part}")
                continue
        # إذا كان السطر يبدأ بـ * أو - أو أيقونة شائعة، نضمن أنه في سطر منفصل
        if re.match(r'^[🔹🔸🌊🟢🔴🎯⏱️🏖️⏳🏃‍♂️🕒⚖️🏹📊🌅☀️🌃🐟💤🔄💨📊📐🌡️🛠️⏱️🎯🦐⚠️📌]', stripped) or re.match(r'^[*-] ', stripped):
            if new_lines and new_lines[-1] != '':
                new_lines.append('')
            new_lines.append(stripped)
        else:
            new_lines.append(stripped)
    return '\n'.join(new_lines)

def add_paragraph_spacing(text: str) -> str:
    lines = text.split('\n')
    new_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            new_lines.append('')
            continue
        # سطر فارغ قبل العناوين الرئيسية
        if re.match(r'^(🎯|⏱️|🏃‍♂️|🕒|⚖️|🏹|📊|\d\.)', stripped):
            if new_lines and new_lines[-1] != '':
                new_lines.append('')
        # نضمن أن النقاط تبدأ بسطر جديد
        if re.match(r'^[*-] ', stripped) or re.match(r'^[🔹🔸🌊🟢🔴]', stripped):
            if new_lines and new_lines[-1] != '' and not re.match(r'^[*-] ', new_lines[-1]):
                pass
        new_lines.append(stripped)
    return '\n'.join(new_lines)

def clean_report_text(text: str) -> str:
    text = re.sub(r'(المد العالي|الجزر المنخفض):(\d{2}:\d{2})', r'\1: \2', text)
    text = re.sub(r'(\w)\s+:\s+', r'\1: ', text)
    text = re.sub(r'(\d{2}:\d{2})\s+(\d{2}:\d{2})', r'\1 و \2', text)
    text = re.sub(r'\*\*\s*([^*]+)\s*\*\*', r'\1', text)
    text = re.sub(r'(\d+\.\d+)\s*°\s*م', r'\1°م', text)
    text = fix_time_ranges(text)
    return text.strip()

def fix_broken_number_lines(text: str) -> str:
    lines = text.split('\n')
    fixed = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if re.search(r'\d+:\s*$', line) and i+1 < len(lines):
            next_line = lines[i+1].strip()
            if re.match(r'^\d+', next_line):
                merged = line.rstrip() + next_line
                merged = re.sub(r'(\d+):\s+(\d+)', r'\1:\2', merged)
                fixed.append(merged)
                i += 2
                continue
        if re.match(r'^-\s', line) and fixed:
            fixed[-1] = fixed[-1].rstrip() + ' ' + line
            i += 1
            continue
        line = re.sub(r'(\d+):\s+(\d+)', r'\1:\2', line)
        fixed.append(line)
        i += 1
    return '\n'.join(fixed)

def extract_numbers_from_text(text: str) -> List[float]:
    return [float(m) for m in re.findall(r'-?\d+\.?\d*', text)]

def get_allowed_numbers(agg: dict) -> Set[float]:
    allowed = set()
    for b in agg.get("blocks", []):
        raw = b.get("_raw", {})
        for v in raw.values():
            if isinstance(v, (int, float)) and v > 0:
                allowed.add(v); allowed.add(round(v,1)); allowed.add(round(v,0))
        if "confidence" in b: allowed.add(round(b["confidence"],1)); allowed.add(round(b["confidence"],0))
        if "recommended_cast_distance" in b: allowed.add(round(b["recommended_cast_distance"],1))
        if "wind_gust_peak" in b: allowed.add(round(b["wind_gust_peak"],1))
        if b.get("wave_angle_diff") is not None: allowed.add(round(b["wave_angle_diff"],1))
        if b.get("swell_angle_diff") is not None: allowed.add(round(b["swell_angle_diff"],1))
    extra = agg.get("extra_info", {})
    for v in extra.values():
        if isinstance(v, (int, float)): allowed.add(round(v,1))
    for v in agg.get("hidden_factors", {}).values():
        if isinstance(v, (int, float)): allowed.add(round(v,1))
    allowed.add(round(agg.get("avg_sst",0),1))
    allowed.add(round(extra.get("max_air_temp",0),1))
    allowed.add(round(extra.get("peak_gust_today",0),1))
    allowed.add(round(extra.get("pressure_avg",0),1))
    allowed.add(float(agg.get("score",0)))
    return {x for x in allowed if x > 0.5}

@app.post("/generate-report")
@limiter.limit("10/minute")
async def generate_report(request: Request, req: RawDataReportRequest):
    try:
        if req.marine_data and req.weather_data:
            marine_data, weather_data = req.marine_data, req.weather_data
        else:
            lat = req.latitude or 36.8; lon = req.longitude or 10.1
            marine_data = await fetch_marine_data_from_openmeteo(lat, lon)
            weather_data = await fetch_weather_data_from_openmeteo(lat, lon)
            if not marine_data or not weather_data:
                raise HTTPException(502, "تعذر جلب البيانات")

        marine_hourly = marine_data.get("hourly", marine_data)
        weather_hourly = weather_data.get("hourly", {})
        daily = weather_data.get("daily", {})
        tz_name = marine_data.get("timezone", "Africa/Tunis")
        now_tn = datetime.now(zoneinfo.ZoneInfo("Africa/Tunis"))
        target_dt = resolve_target_date(req.target_date, now_tn.date())
        raw_sr = daily.get("sunrise", ["06:00"])[0]
        raw_ss = daily.get("sunset", ["18:00"])[0]
        sunrise = re.search(r'\d{2}:\d{2}', raw_sr).group() if re.search(r'\d{2}:\d{2}', raw_sr) else "06:00"
        sunset = re.search(r'\d{2}:\d{2}', raw_ss).group() if re.search(r'\d{2}:\d{2}', raw_ss) else "18:00"
        latitude = req.latitude or 36.8

        all_times, aligned = align_hourly_data(marine_hourly, weather_hourly, tz_name)
        if not all_times:
            raise HTTPException(500, "لا توجد بيانات ساعية متزامنة")

        agg = aggregate_physics(all_times, aligned, req.beach_orientation, target_dt, sunrise, sunset, latitude)

        if agg["final_verdict"] == "غير مناسب" and any("هائج" in r or "صواعق" in r for r in agg["nogo_reasons"]):
            return {
                "report": f"❌ غير مناسب ({agg['score']}%) – لا توجد فرصة.",
                "meta": {"score": agg['score'], "hard_nogo": True}
            }

        ctx = build_context(req, agg, tz_name)
        report = await call_openrouter(ctx)

        report = clean_report_text(report)
        report = fix_broken_number_lines(report)
        report = fix_time_ranges(report)
        report = enforce_line_breaks(report)   # الجديد: يفصل النقاط المتلاصقة
        report = add_paragraph_spacing(report)

        computed_text = "\n\n━━━━━━━━━━━━━━━━━━━━━━━━\n📊 الأرقام المرجعية (للتحقق)\n━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        computed_text += f"🔹 حرارة الماء: {agg['avg_sst']}°م | حرارة الهواء: {agg['extra_info']['max_air_temp']}°م\n"
        computed_text += f"🔹 الرياح: أقصى هبات {agg['extra_info']['peak_gust_today']} كم/س | الضغط: {agg['extra_info']['pressure_avg']} hPa\n"
        for b in agg['blocks']:
            r = b['_raw']
            sign = '+' if r['wind_effect_dist'] >= 0 else ''
            computed_text += f"🔸 {b['name']} ({b['time_range']}): ثقة {b['confidence']}% | مسافة {b['recommended_cast_distance']}م | موج {r['avg_wave_h']}-{r['max_wave_h']}م | رياح {r['avg_wind']} كم/س | تأثير الرياح {sign}{r['wind_effect_dist']}م\n"
        computed_text += "━━━━━━━━━━━━━━━━━━━━━━━━\n"

        report += computed_text

        allowed = get_allowed_numbers(agg)
        found_nums = extract_numbers_from_text(report)
        suspicious = [n for n in found_nums if n > 0.5 and n not in allowed and not (n.is_integer() and 0 <= n <= 59)]
        if suspicious:
            logger.warning(f"أرقام مشبوهة (تم تجاهلها): {suspicious}")

        clean_blocks = [{k:v for k,v in b.items() if k != "_raw"} for b in agg["blocks"]]
        meta = {
            "timezone": tz_name, "target_date": target_dt.isoformat(), "hard_nogo": False,
            "score": agg["score"],
            "tidal_estimation": agg["extra_info"]["tidal_windows"],
            "golden_windows": agg["extra_info"]["golden_windows"],
            "solunar": agg["extra_info"]["solunar"],
            "final_verdict": agg["final_verdict"],
            "nogo_reasons": agg["nogo_reasons"],
            "warnings": agg.get("warnings", []),
            "blocks": clean_blocks
        }
        return {"report": report, "meta": meta}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"generate-report error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, detail="فشل إنشاء التقرير")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
