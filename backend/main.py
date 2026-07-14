"""
Surfcasting Analytics API – v20.3.13 (OpenRouter, Zero‑Error)
- استخدام OpenRouter بدلاً من Gemini المباشر لحل مشكلة Rate Limit.
- جميع تحسينات v20.3.12 مضمنة (Overpass حصري، أمان، scoring محسّن، CORS، health، ضغط، راحة).
- يحتفظ بنفس SYSTEM_PROMPT وسياق البيانات المرسلة.
- جاهز للإنتاج بمجرد تعيين OPENROUTER_API_KEY.
"""
import os, math, asyncio, logging, traceback, zoneinfo, re, random
from datetime import datetime, timedelta, date
from typing import Optional, List
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

# إخفاء سجلات httpx لمنع تسرب المفتاح
logging.getLogger("httpx").setLevel(logging.WARNING)

# ========== مفاتيح API والروابط ==========
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY مفقود. ضعه في متغيرات البيئة.")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "google/gemini-2.5-flash"

# قائمة خوادم Overpass
OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]

USER_AGENT = "SurfcastingAnalytics/1.0 (naderba69@gmail.com)"
_TIME_RE = re.compile(r'\d{2}:\d{2}')

# ========== نماذج البيانات ==========
class AutoOrientationRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

class RawDataReportRequest(BaseModel):
    beach_orientation: int = Field(..., ge=0, le=360)
    beach_type: Optional[str] = Field(None, pattern="^(sandy|rocky)$")
    target_date: str = Field(..., pattern="^(today|tomorrow|day_after)$")
    marine_data: Optional[dict] = None
    weather_data: Optional[dict] = None
    latitude: Optional[float] = 36.8
    longitude: Optional[float] = 10.1

class DetectBottomRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="Surfcasting Analytics", version="20.3.13", lifespan=lifespan)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
if ALLOWED_ORIGINS == "*":
    logger.warning("CORS is open to all origins. Restrict in production.")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True, max_age=600)
else:
    origins = [o.strip() for o in ALLOWED_ORIGINS.split(",")]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"], allow_credentials=True, max_age=600)

@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": "خطأ داخلي في الخادم"})

@app.get("/health")
async def health():
    overpass_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(OVERPASS_SERVERS[0] + "?data=[out:json];node(1);out;")
            overpass_ok = r.status_code == 200
    except Exception:
        pass
    return {
        "status": "ok",
        "version": "20.3.13",
        "openrouter_configured": bool(OPENROUTER_API_KEY),
        "overpass_reachable": overpass_ok,
        "timestamp": datetime.now(zoneinfo.ZoneInfo("Africa/Tunis")).isoformat()
    }

# ---------- دوال مساعدة (بدون تغيير عن v20.3.12) ----------
def safe_float(v) -> float:
    try:
        result = float(v)
        return 0.0 if (math.isnan(result) or math.isinf(result)) else result
    except (TypeError, ValueError):
        return 0.0

def angle_diff(w, b):
    if w is None or b is None:
        return 180.0
    d = abs(w - b) % 360
    return 360 - d if d > 180 else d

def signed_angle_diff(w, b):
    if w is None or b is None:
        return 0.0
    d = (w - b + 180) % 360 - 180
    return d

def circular_diff(a: float, b: float) -> float:
    a, b = a % 24, b % 24
    diff = abs(a - b)
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
    try:
        if math.isnan(h) or math.isinf(h):
            return "00:00"
        h = h % 24
        hh = int(h)
        mm = int(round((h - hh) * 60))
        if mm >= 60:
            hh = (hh + 1) % 24
            mm = 0
        return f"{hh % 24:02d}:{mm:02d}"
    except (TypeError, ValueError):
        return "00:00"

def wind_class_detailed(diff):
    if diff < 30: return "بحرية مباشرة"
    if diff < 45: return "بحرية خفيفة"
    if diff < 60: return "جانبية مائلة للبحر"
    if diff <= 120: return "جانبية"
    if diff < 150: return "جانبية مائلة للبر"
    if diff < 165: return "برية خفيفة"
    return "برية مباشرة"

def weather_desc(code):
    if code == 0: return "صافية تماماً"
    if code == 1: return "صافية مع غيوم خفيفة"
    if code == 2: return "غائمة جزئياً"
    if code == 3: return "غائمة"
    if code <= 48: return "ضباب"
    if code <= 55: return "رذاذ"
    if code <= 65: return "مطر"
    if code <= 82: return "زخات مطر"
    if code <= 99: return "عواصف"
    return "غير معروف"

def deg_to_compass(deg):
    if deg is None:
        return "غير معروف"
    val = int((deg / 22.5) + 0.5) % 16
    arr = ["شمال","شمال شمال شرق","شمال شرق","شرق شمال شرق","شرق","شرق جنوب شرق","جنوب شرق","جنوب جنوب شرق",
           "جنوب","جنوب جنوب غرب","جنوب غرب","غرب جنوب غرب","غرب","غرب شمال غرب","شمال غرب","شمال شمال غرب"]
    return arr[val]

def circular_mean(angles_deg: list) -> Optional[float]:
    valid = [a for a in angles_deg if a is not None]
    if not valid:
        return None
    radians_vals = [math.radians(a) for a in valid]
    sin_sum = sum(math.sin(r) for r in radians_vals)
    cos_sum = sum(math.cos(r) for r in radians_vals)
    if abs(sin_sum) < 1e-10 and abs(cos_sum) < 1e-10:
        return None
    mean_rad = math.atan2(sin_sum, cos_sum)
    return (math.degrees(mean_rad) + 360) % 360

def resolve_target_date(txt, real_today):
    if txt == "today": return real_today
    if txt == "tomorrow": return real_today + timedelta(days=1)
    return real_today + timedelta(days=2)

def _julian_day(d: date) -> float:
    y, m, day = d.year, d.month, d.day
    if m < 3:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + day + b - 1524.5

def get_moon_and_tide_analysis(d: date):
    jd = _julian_day(d)
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

def safe_parse_iso(ts: str, tz: zoneinfo.ZoneInfo) -> Optional[datetime]:
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        return dt
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse ISO time '{ts}': {e}")
        return None

def get_moon_age_days(d: date) -> float:
    jd = _julian_day(d)
    days_since_new = jd - 2451550.1
    return days_since_new % 29.53058867

def get_haml_mat_status(age_days: float) -> dict:
    if 13 <= age_days <= 16:
        day_in = int(age_days - 13) + 1
        return {"status":"أيام الحياء","phase":"حمل البدر","description":f"اليوم {day_in} في حمل البدر. البحر حايي، التيارات قوية، الصيد في ذروته من الشاطئ (Surfcasting).","score_delta":15}
    elif age_days >= 28 or age_days <= 2:
        raw_day = age_days - 28 if age_days >= 28 else age_days
        day_in = max(1, min(3, int(raw_day) + 1))
        return {"status":"أيام الحياء","phase":"حمل المحاق","description":f"اليوم {day_in} في حمل المحاق. البحر حايي، التيارات قوية، الصيد في ذروته من الشاطئ (Surfcasting).","score_delta":15}
    elif 7 <= age_days <= 9 or 21 <= age_days <= 23:
        phase_name = "التربيع الأول" if age_days <= 9 else "التربيع الثاني"
        day_in = int(age_days - 7) + 1 if age_days <= 9 else int(age_days - 21) + 1
        return {"status":"أيام المات","phase":phase_name,"description":f"اليوم {day_in} في {phase_name}. البحر مْيِّت، الماء راكد، الصيد أصعب من الشاطئ.","score_delta":-15}
    else:
        return {"status":"أيام عادية","phase":"","description":"لا توجد مؤشرات حيائية أو مات قوية. الصيد من الشاطئ ممكن.","score_delta":0}

def get_fishing_platform_advice(haml_status: str) -> str:
    if "الحياء" in haml_status: return "الصيد من الشاطئ ممتاز اليوم. التيارات قوية تجلب الأسماك."
    elif "المات" in haml_status: return "الصيد من الشاطئ صعب اليوم بسبب ركود الماء."
    else: return "الصيد من الشاطئ ممكن اليوم."

LUNAR_DAY_HOURS = 24.84101
MEDITERRANEAN_LUNITIDAL_CORRECTION = 0.5

def estimate_tidal_windows(target_date_obj, moon_analysis, sunrise_str, sunset_str, latitude, longitude):
    sr_h = safe_parse_time(sunrise_str)
    ss_h = safe_parse_time(sunset_str)
    lunar_transit_hour = (moon_analysis["phase_decimal"] * LUNAR_DAY_HOURS + longitude / 15.0) % 24
    base_hw_hour = (lunar_transit_hour + 1.2 + MEDITERRANEAN_LUNITIDAL_CORRECTION) % 24
    hw1 = base_hw_hour
    lw1 = (hw1 + 6.2) % 24
    hw2 = (hw1 + 12.4) % 24
    lw2 = (lw1 + 12.4) % 24
    windows = {"HW1": format_time(hw1), "LW1": format_time(lw1), "HW2": format_time(hw2), "LW2": format_time(lw2)}
    golden_windows = []
    lw1_minus2 = (lw1 - 2) % 24
    lw2_minus2 = (lw2 - 2) % 24
    if is_close(hw1, sr_h, 1.5): golden_windows.append(f"ساعة ذهبية صباحية: تزامن المد العالي الأول ({windows['HW1']}) مع الفجر ({sunrise_str}).")
    if is_close(hw2, ss_h, 1.5): golden_windows.append(f"ساعة ذهبية مسائية: تزامن المد العالي الثاني ({windows['HW2']}) مع الغروب ({sunset_str}).")
    if is_close(lw1_minus2, sr_h, 1.5) or is_close(lw2_minus2, sr_h, 1.5): golden_windows.append("نافذة الجزر الممتازة: تزامن بداية جزر قوي مع الفجر.")
    if is_close(lw1_minus2, ss_h, 1.5) or is_close(lw2_minus2, ss_h, 1.5): golden_windows.append("نافذة الجزر الممتازة: تزامن بداية جزر قوي مع الغروب.")
    if not golden_windows:
        hw1_gap = circular_diff(hw1, sr_h)
        hw2_gap = circular_diff(hw2, ss_h)
        golden_windows.append(f"لا توجد ساعة ذهبية. المد العالي الأول ({windows['HW1']}) يبعد {format_time_gap(hw1_gap)} عن الفجر. المد العالي الثاني ({windows['HW2']}) يبعد {format_time_gap(hw2_gap)} عن الغروب.")
    return windows, golden_windows

def _arabic_count(n: int, singular: str, dual: str, plural: str) -> str:
    if n == 1: return f"{singular}"
    elif n == 2: return f"{dual}"
    elif 3 <= n <= 10: return f"{n} {plural}"
    else: return f"{n} {singular}"

def format_time_gap(hours_decimal: float) -> str:
    if hours_decimal <= 0: return "0 دقيقة"
    total_minutes = round(hours_decimal * 60)
    h = total_minutes // 60
    m = total_minutes % 60
    parts = []
    if h > 0: parts.append(_arabic_count(h, "ساعة", "ساعتين", "ساعات"))
    if m > 0: parts.append(_arabic_count(m, "دقيقة", "دقيقتين", "دقائق"))
    return " و ".join(parts) if parts else "0 دقيقة"

def calculate_solunar(d: date, lat: float, lon: float):
    jd = _julian_day(d)
    days_since_new = jd - 2451550.1
    moon_phase = (days_since_new % 29.53058867) / 29.53058867
    lon_correction = lon / 15.0
    moon_transit = (moon_phase * LUNAR_DAY_HOURS + lon_correction) % 24
    major1 = moon_transit
    major2 = (moon_transit + 12.42) % 24
    minor1 = (major1 + 6.21) % 24
    minor2 = (major2 + 6.21) % 24
    return {"major1": format_time(major1), "major2": format_time(major2), "minor1": format_time(minor1), "minor2": format_time(minor2)}

def align_hourly_data(marine_hourly, weather_hourly, tz_name):
    tz = zoneinfo.ZoneInfo(tz_name)
    m_times = marine_hourly.get("time", [])
    w_times = weather_hourly.get("time", [])
    if not m_times or not w_times:
        return [], {}
    m_map, w_map = {}, {}
    for i, t in enumerate(m_times):
        dt = safe_parse_iso(t, tz)
        if dt is not None:
            m_map[dt.replace(minute=0, second=0, microsecond=0)] = i
    for i, t in enumerate(w_times):
        dt = safe_parse_iso(t, tz)
        if dt is not None:
            w_map[dt.replace(minute=0, second=0, microsecond=0)] = i
    common = sorted(set(m_map) & set(w_map))
    if not common:
        logger.warning(f"No common timestamps! Marine times: {m_times[:3]}, Weather times: {w_times[:3]}")
        return [], {}
    def extract(key, src, idx_map, default=0.0):
        arr = src.get(key, [])
        result = []
        for t in common:
            if t in idx_map and arr and idx_map[t] < len(arr):
                result.append(arr[idx_map[t]])
            else:
                result.append(default)
        return result
    return common, {
        "wave_height": extract("wave_height", marine_hourly, m_map, 0.0),
        "wave_period": extract("wave_period", marine_hourly, m_map, 0.0),
        "wave_direction": extract("wave_direction", marine_hourly, m_map, None),
        "swell_wave_height": extract("swell_wave_height", marine_hourly, m_map, 0.0),
        "swell_wave_period": extract("swell_wave_period", marine_hourly, m_map, 0.0),
        "swell_wave_direction": extract("swell_wave_direction", marine_hourly, m_map, None),
        "sea_surface_temperature": extract("sea_surface_temperature", marine_hourly, m_map, None),
        "wind_speed_10m": extract("wind_speed_10m", weather_hourly, w_map, 0.0),
        "wind_direction_10m": extract("wind_direction_10m", weather_hourly, w_map, None),
        "wind_gusts_10m": extract("wind_gusts_10m", weather_hourly, w_map, 0.0),
        "pressure_msl": extract("pressure_msl", weather_hourly, w_map, 1013.0),
        "temperature_2m": extract("temperature_2m", weather_hourly, w_map, None),
        "relative_humidity_2m": extract("relative_humidity_2m", weather_hourly, w_map, 65.0),
        "precipitation": extract("precipitation", weather_hourly, w_map, 0.0),
        "visibility": extract("visibility", weather_hourly, w_map, 10000.0),
        "weather_code": extract("weather_code", weather_hourly, w_map, 0.0)
    }

# ---------- قاعدة الشواطئ ----------
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

def find_nearest_beach_info(lat: float, lon: float, max_dist: float = 20000) -> Optional[dict]:
    min_dist = float('inf')
    nearest = None
    for b in TUNISIAN_BEACHES:
        dist = calc_distance(b["lat"], b["lon"], lat, lon)
        if dist < min_dist and dist < max_dist:
            min_dist = dist
            nearest = {"orientation": b["orientation"], "type": b["type"], "distance": round(dist, 0)}
    return nearest

# ---------- Overpass orientation (حصراً) ----------
async def _overpass_orientation_inner(lat, lon):
    async with httpx.AsyncClient(timeout=8.0, headers={"User-Agent": USER_AGENT}) as client:
        for radius in [3000, 5000, 10000]:
            query = f"""[out:json];(way(around:{radius},{lat},{lon})["natural"="coastline"];);out geom;"""
            for attempt_idx, server in enumerate(OVERPASS_SERVERS):
                try:
                    if attempt_idx > 0:
                        await asyncio.sleep(1.0)
                    r = await client.get(server, params={"data": query})
                    r.raise_for_status()
                    els = r.json().get("elements", [])
                    if not els:
                        continue
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
                    perp1 = (best_tangent + 90) % 360
                    perp2 = (best_tangent - 90) % 360
                    to_user = calc_bearing(best_point["lat"], best_point["lon"], lat, lon)
                    diff1 = angle_diff(perp1, to_user)
                    diff2 = angle_diff(perp2, to_user)
                    seaward = perp1 if diff1 > diff2 else perp2
                    return int(round(seaward))
                except Exception as e:
                    logger.warning(f"Overpass failed ({server}, radius={radius}): {e}")
                    continue
    return None

async def get_auto_orientation_overpass(lat, lon):
    try:
        return await asyncio.wait_for(_overpass_orientation_inner(lat, lon), timeout=25.0)
    except asyncio.TimeoutError:
        logger.warning("Overpass orientation global timeout (25s)")
        return None

@app.post("/auto-orientation")
@limiter.limit("5/minute")
async def auto_orientation(request: Request, req: AutoOrientationRequest):
    orientation = await get_auto_orientation_overpass(req.latitude, req.longitude)
    if orientation is None:
        raise HTTPException(status_code=502, detail="تعذر تحديد اتجاه الشاطئ من الخريطة. يرجى المحاولة لاحقاً أو إدخال الاتجاه يدوياً.")
    return {"orientation": orientation, "source": "overpass"}

@app.post("/detect-bottom-type")
@limiter.limit("10/minute")
async def detect_bottom_type(request: Request, req: DetectBottomRequest):
    info = find_nearest_beach_info(req.latitude, req.longitude)
    if info: return {"bottom_type": info["type"], "source": "nearby_beach", "confidence": "medium"}
    query = f"""[out:json];(
        way(around:2000,{req.latitude},{req.longitude})["natural"="sand"];
        way(around:2000,{req.latitude},{req.longitude})["natural"="shingle"];
        way(around:2000,{req.latitude},{req.longitude})["natural"="bare_rock"];
    );out body;"""
    async with httpx.AsyncClient(timeout=15, headers={"User-Agent": USER_AGENT}) as client:
        for server in OVERPASS_SERVERS[:2]:
            try:
                r = await client.get(server, params={"data": query})
                r.raise_for_status()
                data = r.json()
                elements = data.get("elements", [])
                if elements:
                    tags = elements[0].get("tags", {})
                    nat = tags.get("natural", "")
                    if nat == "sand": return {"bottom_type": "sandy", "source": "overpass", "confidence": "high"}
                    if nat in ["shingle", "bare_rock"]: return {"bottom_type": "rocky", "source": "overpass", "confidence": "high"}
                break
            except Exception as e:
                logger.warning(f"Overpass bottom detection failed ({server}): {e}")
                continue
    return {"bottom_type": "unknown", "source": "none", "confidence": "low"}

# ---------- جلب البيانات ----------
async def fetch_marine_data_from_openmeteo(client: httpx.AsyncClient, lat: float, lon: float):
    url = "https://marine-api.open-meteo.com/v1/marine"
    params = {"latitude": lat, "longitude": lon, "hourly": "wave_height,wave_period,wave_direction,swell_wave_height,swell_wave_period,swell_wave_direction,sea_surface_temperature", "timezone": "Africa/Tunis", "forecast_days": 3}
    try:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"Open-Meteo marine fetch failed: {e}")
        return None

async def fetch_weather_data_from_openmeteo(client: httpx.AsyncClient, lat: float, lon: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "hourly": "wind_speed_10m,wind_direction_10m,wind_gusts_10m,pressure_msl,temperature_2m,relative_humidity_2m,precipitation,visibility,weather_code", "daily": "sunrise,sunset", "timezone": "Africa/Tunis", "forecast_days": 3}
    try:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"Open-Meteo weather fetch failed: {e}")
        return None

# ---------- أدوات التحليل التكتيكي ----------
def get_water_clarity(wind_speed, wave_height, is_murky, is_weedy, haml_status, avg_vis_b=10000):
    if avg_vis_b < 200: return "ضباب كثيف (رؤية معدومة)"
    if is_weedy: return "عكر جداً (أعشاب وصوفة)"
    if is_murky: return "عكر (بحر خامر)"
    if "الحياء" in haml_status and (wind_speed > 15 or wave_height > 0.5): return "عكر/مخلوط (التيارات تقلب القاع)"
    if wind_speed > 20: return "متوسط العكارة (رياح قوية)"
    if wave_height < 0.3 and wind_speed < 10: return "صافي جداً"
    return "صافي"

def suggest_rig(haml_status: str, is_lateral_strong: bool, wind_speed: float, is_mirror_sea: bool) -> str:
    strong_current = "الحياء" in haml_status or is_lateral_strong
    if strong_current or wind_speed > 20: return "مونتاج باتير نوستر قصير (فروع 50سم، ثقيل) – يمنع التشابك في التيار"
    if is_mirror_sea or (not is_lateral_strong and wind_speed < 10): return "مونتاج بسنود طويل (فرع سفلي 150سم، خفيف) – حركة طبيعية للطعم"
    return "مونتاج عادي (فرع 80-100سم) – مرن للظروف المتوسطة"

def casting_angle_correction(wind_dir, orient):
    if wind_dir is None or orient is None: return 0
    diff = signed_angle_diff(wind_dir, orient)
    if abs(diff) < 30: return 0
    raw = round(-diff * 0.12)
    return max(-15, min(15, raw))

def calculate_comfort_index(temp, wind_speed, humidity=None):
    if temp is None: return 50
    humidity = max(0.0, min(100.0, float(humidity))) if humidity is not None else 50.0
    temp = max(-10.0, min(50.0, float(temp)))
    if humidity > 40:
        apparent = temp + 0.33 * (humidity / 100 * 6.105 * math.exp(17.27 * temp / (237.7 + temp))) - 4.0
    else:
        apparent = temp
    ideal_temp = 21.0
    temp_penalty = abs(ideal_temp - apparent) * 2.5
    wind_penalty = (wind_speed / 10) ** 1.5 * 5
    humidity_penalty = abs(humidity - 55) * 0.15
    base = 100 - temp_penalty - wind_penalty - humidity_penalty
    return max(0, min(100, int(base)))

def get_confidence_label(confidence):
    if confidence >= 90: return "ذروة ملكية"
    if confidence >= 80: return "ممتازة"
    if confidence >= 70: return "جيدة جداً"
    if confidence >= 60: return "جيدة"
    if confidence >= 50: return "مقبولة"
    return "ضعيفة"

# ---------- تحليلات إضافية ----------
def analyze_weed_risk(sea_memory):
    risk = "منخفض"; advice = ""
    has_weed = "صوفة" in sea_memory or "أعشاب" in sea_memory
    if has_weed:
        risk = "مرتفع" if "تحذير صوفة" in sea_memory else "متوسط"
        advice = "استعمل صائدات مضادة للأعشاب وارمِ بزاوية حادة نحو البحر لتجنب الالتفاف."
    return {"risk": risk, "advice": advice}

def analyze_backwash(wind_speed: float, wind_dir, orient: float, wave_height: float) -> dict:
    if wind_dir is None: return {"severity": "منخفض", "effect": ""}
    wind_diff = angle_diff(wind_dir, orient)
    is_onshore = wind_diff < 30
    severity = "منخفض"; effect = ""
    if is_onshore and wind_speed > 30 and wave_height > 0.8:
        severity = "مرتفع"
        effect = f"رياح بحرية قوية ({wind_speed:.0f} كم/س) تضرب الموج نحو الشاطئ، ثم يرتد الموج بقوة نحو البحر. هذا يخلق تياراً عكسياً قوياً يسحب الرصاصة للشاطئ باستمرار وقد يدفن الخيط. حتى مع وزن ثقيل، يصعب تثبيت الطعم."
    elif is_onshore and wind_speed > 15:
        severity = "متوسط"
        effect = f"رياح بحرية ({wind_speed:.0f} كم/س) تخلق تياراً عكسياً خفيفاً. الرصاصة قد تتحرك قليلاً نحو الشاطئ لكن يمكن التحكم بها بوزن أثقل."
    return {"severity": severity, "effect": effect}

def analyze_debris_risk(sea_memory: str, wind_speed: float) -> dict:
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

def calculate_confidence_index(period_flags, is_mirror_sea, has_golden, nogo_count, period_warning_count,
                                block_wind_ok, block_wave_ok, is_night_with_tide):
    base = 70
    if is_mirror_sea: base -= 40
    if not has_golden: base -= 20
    if nogo_count > 0: base -= 30
    if period_warning_count > 0: base -= 10
    if block_wind_ok: base += 10
    if block_wave_ok: base += 10
    if is_night_with_tide: base += 15
    base += period_flags.get("is_spring_tide", 0) * 15
    base += period_flags.get("is_pressure_dropping", 0) * 15
    return max(0, min(100, base))

def apply_scoring(agg: dict) -> int:
    score = 50
    flags = agg["flags"]
    extra = agg["extra_info"]
    blocks = agg["blocks"]

    if not flags["has_golden_window"]: score -= 15
    if agg["avg_sst"] is not None:
        if agg["avg_sst"] > 27.0: score -= 15
        elif agg["avg_sst"] < 13.0: score -= 15
    if extra.get("sst_stability", "") == "صدمة حرارية": score -= 10
    for b in blocks:
        if b.get("backwash", {}).get("severity") == "متوسط":
            score -= 10
            break
    press_change = abs(extra.get("pressure_change", 0))
    if 3.0 <= press_change <= 6.0: score -= 10
    elif press_change > 6.0: score -= 20
    if flags["is_lateral_strong"]: score -= 10
    for b in blocks:
        wp_val = b["_raw"]["wave_period"]
        if 0.1 < wp_val < 4.0:
            score -= 10
            break
    has_strong_offshore = any(b["wind_dir"].startswith("برية") and b["_raw"]["avg_wind"] > 20 for b in blocks)
    has_light_offshore = any(b["wind_dir"].startswith("برية") and b["_raw"]["avg_wind"] <= 15 for b in blocks)
    if has_strong_offshore: score -= 10
    elif has_light_offshore: score += 5
    if any(b["wind_dir"].startswith("بحرية") and b["_raw"]["avg_wind"] < 15 for b in blocks): score += 15
    if any(0.6 <= b["_raw"]["avg_wave_h"] <= 1.2 for b in blocks): score += 20
    if any(b.get("wave_angle_diff") is not None and b["wave_angle_diff"] < 30 and b["_raw"]["avg_wave_h"] >= 0.6 for b in blocks): score += 15
    if flags["has_golden_window"]: score += 25
    if press_change < 2.0: score += 10
    moon_idx = agg["tide_analysis"]["idx"]
    if moon_idx == 4 and any(b["name"] in ("الغسق","السحر") for b in blocks): score += 15
    haml_delta = extra.get("haml_score_delta", 0)
    score += haml_delta
    negative_wind_effect = any(b["_raw"].get("wind_effect_dist", 0) < -15 for b in blocks)
    if negative_wind_effect: score -= 5

    score = max(0, min(100, score))
    lethal_blocks = sum(1 for b in blocks if b.get("has_lethal_nogo", False))
    total_blocks = len(blocks)
    if total_blocks > 0:
        lethal_ratio = lethal_blocks / total_blocks
        if lethal_ratio == 1.0: score = min(score, 30)
        elif lethal_ratio >= 0.5: score = min(score, 65)
        elif lethal_ratio > 0: score = min(score, 80)
        elif agg.get("warnings"): score = min(score, 95)
    return score

# ---------- التجميع الفيزيائي ----------
def get_period_fish_status(avg_sst, is_night, is_murky, is_weedy, is_mirror_sea, lateral_force_ratio, water_clarity, seabass_sst_limit):
    if avg_sst is None: avg_sst = 20.0
    active, inactive = [], []
    seabass_active = (is_night or is_murky or lateral_force_ratio > 0.5) and avg_sst < seabass_sst_limit and not is_mirror_sea
    if seabass_active: active.append("قاروص")
    else: inactive.append("قاروص")
    if avg_sst > 18 and not is_mirror_sea: active.append("دنيس")
    else: inactive.append("دنيس")
    if not is_murky and not is_weedy and not is_mirror_sea: active.append("بوري")
    else: inactive.append("بوري")
    if avg_sst <= 22: active.append("سارغ")
    else: inactive.append("سارغ")
    if avg_sst > 18 and not is_mirror_sea and lateral_force_ratio > 0.2: active.append("مرمار")
    else: inactive.append("مرمار")
    if avg_sst > 17 and not is_mirror_sea: active.append("شلبة")
    else: inactive.append("شلبة")
    if avg_sst > 18 and not is_murky: active.append("تريلية")
    else: inactive.append("تريلية")
    if avg_sst > 19 and is_murky: active.append("بغبغان")
    else: inactive.append("بغبغان")
    if avg_sst > 18 and not is_mirror_sea and (is_night or "عكر" in water_clarity): active.append("سوبيا")
    else: inactive.append("سوبيا")
    return active, inactive

def aggregate_physics(all_times, aligned, orient, target_date_obj, sunrise, sunset, latitude, longitude):
    tz = all_times[0].tzinfo if all_times else zoneinfo.ZoneInfo("UTC")
    target_start = datetime.combine(target_date_obj, datetime.min.time(), tzinfo=tz)
    target_end = target_start + timedelta(days=1)
    past_start = target_start - timedelta(hours=48)
    past_idx = [i for i, t in enumerate(all_times) if past_start <= t < target_start]
    target_idx = [i for i, t in enumerate(all_times) if target_start <= t < target_end]
    warnings = []
    empty_res = {"sea_memory":"غير معروف","lateral_current":"غير معروف","pressure_state":"مستقر","tide_analysis":{},"sst_stability":"مستقر","avg_sst":None,"hidden_factors":{},"blocks":[],"red_flags":[],"green_flags":[],"extra_info":{},"transitions":[],"flags":{},"nogo_reasons":[],"warnings":[],"final_verdict":"غير مناسب","score":0}
    if not target_idx: return empty_res

    def pick(k, default=0.0):
        arr = aligned.get(k, [])
        return [(arr[i] if arr[i] is not None else default) if i < len(arr) else default for i in target_idx]

    wh = pick("wave_height"); wp = pick("wave_period"); swh = pick("swell_wave_height")
    swp = pick("swell_wave_period"); swd = pick("swell_wave_direction", None); wd_wave = pick("wave_direction", None)
    sst = pick("sea_surface_temperature", None); ws = pick("wind_speed_10m"); wd = pick("wind_direction_10m", None)
    wg = pick("wind_gusts_10m"); pr = pick("pressure_msl", 1013.0); ta = pick("temperature_2m", None)
    rh = pick("relative_humidity_2m", 65.0); prec = pick("precipitation"); vis = pick("visibility", 10000.0)
    wcode = [int(v) if isinstance(v, (int, float)) else 0 for v in pick("weather_code")]

    wave_power = [0.49*(safe_float(h)**2)*safe_float(p) for h,p in zip(wh,wp)]
    wind_cls = [wind_class_detailed(angle_diff(d, orient)) if d is not None else "غير معروف" for d in wd]
    has_swell_data = len(swh) > 0 and not all(v == 0.0 for v in swh)
    actual_swell_exists = has_swell_data and max(swh) > 0.05

    sea_memory = "بحر صافي وهادئ"
    accumulated_rain_48h = 0.0; max_rain_hourly = 0.0
    if past_idx:
        p_wh = aligned.get("wave_height", []); p_wp = aligned.get("wave_period", [])
        p_swh = aligned.get("swell_wave_height", []); p_swp = aligned.get("swell_wave_period", [])
        p_ws = aligned.get("wind_speed_10m", []); p_wd = aligned.get("wind_direction_10m", [])
        p_prec = aligned.get("precipitation", [])
        valid_past = [i for i in past_idx if i < len(p_wh) and i < len(p_wp) and i < len(p_ws) and i < len(p_wd)]
        if valid_past:
            weighted_past_power = 0.0; weighted_past_swh = 0.0; total_weight = 0.0; past_onshore_hours = 0.0
            total_rain_past = 0.0; hourly_rain_values = []
            for pos, i in enumerate(valid_past):
                decay_weight = 0.8 ** (len(valid_past) - 1 - pos)
                weighted_past_power += decay_weight * 0.49*(safe_float(p_wh[i])**2)*safe_float(p_wp[i])
                swh_contribution = (decay_weight * safe_float(p_swh[i])) if i < len(p_swh) else 0.0
                weighted_past_swh += swh_contribution
                total_weight += decay_weight
                if p_wd[i] is not None and wind_class_detailed(angle_diff(p_wd[i], orient)).startswith("بحرية"):
                    past_onshore_hours += decay_weight
                val = safe_float(p_prec[i]) if i < len(p_prec) else 0
                total_rain_past += val
                hourly_rain_values.append(val)
            past_avg = weighted_past_power / total_weight if total_weight > 0 else 0
            past_sh = weighted_past_swh / total_weight if total_weight > 0 else 0
            past_onshore_ratio = past_onshore_hours / total_weight
            accumulated_rain_48h = total_rain_past
            max_rain_hourly = max(hourly_rain_values) if hourly_rain_values else 0.0
            if past_avg > 6.0 and past_onshore_ratio > 0.4: sea_memory = "بحر خامر وعكر جداً."
            elif past_avg > 4.0 and past_onshore_ratio > 0.3: sea_memory = "بحر يعكر ببطء."
            if past_sh > 0.8 and past_avg > 4.0: sea_memory += " | تحذير صوفة."
            if accumulated_rain_48h > 45.0 or max_rain_hourly > 15.0: sea_memory += " | سيول."

    day_lateral_fx, day_lateral_fy = 0.0, 0.0
    for i in range(len(wh)):
        if wd_wave[i] is not None:
            w_dir = wd_wave[i]
            signed_angle = math.radians(signed_angle_diff(w_dir, orient))
            force = safe_float(wh[i]) ** 2
            day_lateral_fx += force * math.sin(signed_angle)
            day_lateral_fy += force * math.cos(signed_angle)
    day_total = math.sqrt(day_lateral_fx**2 + day_lateral_fy**2)
    day_lateral_ratio = abs(day_lateral_fx) / day_total if day_total > 1e-9 else 0
    avg_wave_h_day = sum(wh) / len(wh) if wh else 0

    tide_analysis = get_moon_and_tide_analysis(target_date_obj)
    tidal_windows, golden_windows = estimate_tidal_windows(target_date_obj, tide_analysis, sunrise, sunset, latitude, longitude)
    solunar = calculate_solunar(target_date_obj, latitude, longitude)
    moon_age = get_moon_age_days(target_date_obj)
    haml_info = get_haml_mat_status(moon_age)
    platform_advice = get_fishing_platform_advice(haml_info["status"])

    is_neap_tide = tide_analysis["idx"] in [2, 6]; is_spring_tide = tide_analysis["idx"] in [0, 4]
    if is_neap_tide: warnings.append(f"مد ضعيف (مد محاقي - {tide_analysis['name']}): تيارات غذائية ضعيفة، الأسماك أقل تجمعاً.")
    has_golden_window = any("تزامن" in g for g in golden_windows)
    if not has_golden_window: warnings.append("لا توجد ساعة ذهبية: قد يقل نشاط الأسماك.")

    valid_sst = [v for v in sst if v is not None and 8.0 <= v <= 35.0]
    avg_sst = sum(valid_sst) / len(valid_sst) if valid_sst else None
    sst_diff = max(valid_sst) - min(valid_sst) if len(valid_sst) > 1 else 0
    sst_stability = "صدمة حرارية" if sst_diff > 2.0 else "تغير بطيء" if sst_diff > 1.0 else "مستقر تماماً"
    is_murky = "عكر" in sea_memory or "خامر" in sea_memory
    is_weedy = "صوفة" in sea_memory
    valid_ta = [v for v in ta if v is not None]
    max_air_temp = max(valid_ta) if valid_ta else 20.0
    month = target_date_obj.month
    seabass_sst_limit = 20.0 if month in [6,7,8,9] else 18.0
    if avg_sst is not None and avg_sst > seabass_sst_limit: warnings.append(f"حرارة ماء عالية ({avg_sst:.1f}°م): تتجاوز حد القاروص.")
    if is_weedy: warnings.append("صوفة محتملة: استعمل مونتاجاً مضاداً للأعشاب.")

    avg_press = sum(safe_float(v) for v in pr) / len(pr) if pr else 0
    press_change_day = safe_float(pr[-1]) - safe_float(pr[0]) if len(pr) > 1 else 0.0
    if abs(press_change_day) > 8.0: pressure_note = "مضطرب بشدة (سلبي جداً)"
    elif press_change_day > 3.0: pressure_note = "يرتفع خلال اليوم (سلبي)"
    elif press_change_day < -3.0: pressure_note = "ينخفض خلال اليوم (إيجابي)"
    else: pressure_note = "مستقر (محايد)"
    pressure_state = f"{pressure_note} (تغير يومي: {press_change_day:+.1f} hPa)"

    peak_gust_day = max(safe_float(v) for v in wg) if wg else 0.0
    dominant = max(set(wind_cls), key=wind_cls.count) if wind_cls else "غير معروف"
    periods = defaultdict(list)
    for idx, i in enumerate(target_idx):
        h = all_times[i].hour
        if 4 <= h <= 11: periods["morning"].append(idx)
        elif 12 <= h <= 17: periods["afternoon"].append(idx)
        else:
            if h >= 18: periods["evening"].append(idx)
            else: periods["late_night"].append(idx)

    def parse_tidal_time(t_str: str) -> float:
        try:
            parts = t_str.split(":")
            if len(parts) < 2: return 0.0
            return int(parts[0]) + int(parts[1]) / 60.0
        except: return 0.0

    slack_info = ""
    for hw_key, lw_key in [("HW1","LW1"), ("HW2","LW2")]:
        hw_t = parse_tidal_time(tidal_windows[hw_key])
        lw_t = parse_tidal_time(tidal_windows[lw_key])
        slack_info += f"المد العالي {tidal_windows[hw_key]} مياه ميتة: {format_time(hw_t-0.75)}-{format_time(hw_t+0.75)}; الجزر المنخفض {tidal_windows[lw_key]} مياه ميتة: {format_time(lw_t-0.75)}-{format_time(lw_t+0.75)}; "
    slack_info = slack_info.rstrip("; ")

    steepness_vals = [safe_float(h) / (1.56 * safe_float(p)**2) for h, p in zip(wh, wp) if safe_float(p) > 0.1 and h is not None]
    avg_steepness = sum(steepness_vals) / len(steepness_vals) if steepness_vals else 0
    steepness_desc = ("موج حاد وقصير" if avg_steepness > 0.06 else "موج منخفض الانحدار (سلس)" if avg_steepness < 0.03 else "موج متوسط الانحدار")

    cross_angles = []
    for i in range(min(len(swd), len(wd_wave))):
        if swd[i] is not None and wd_wave[i] is not None and swd[i] != 0.0 and wd_wave[i] != 0.0:
            cross_angles.append(angle_diff(swd[i], wd_wave[i]))
    is_cross_sea_dangerous = bool(cross_angles and max(cross_angles) > 60 and sum(cross_angles)/len(cross_angles) > 40)
    cross_sea_risk = "بحر مختلط وخطير" if is_cross_sea_dangerous else "منخفض"

    blocks = []
    populated_periods = 0; periods_with_lethal = 0; mirror_with_gusts = []; mirror_without_gusts = []
    period_names_arabic = {"morning":"الصباح","afternoon":"الظهيرة","evening":"الغسق","late_night":"السحر"}

    for key in ["morning", "afternoon", "evening", "late_night"]:
        idxs = periods[key]
        if not idxs: continue
        populated_periods += 1
        avg_h = sum(safe_float(wh[i]) for i in idxs)/len(idxs)
        max_h = max(safe_float(wh[i]) for i in idxs)
        avg_w = sum(safe_float(ws[i]) for i in idxs)/len(idxs)
        max_w = max(safe_float(ws[i]) for i in idxs)
        sub_wind_cls = [wind_cls[i] for i in idxs]
        wc_dom = max(set(sub_wind_cls), key=sub_wind_cls.count) if sub_wind_cls else "غير معروف"
        sub_wcode = [wcode[i] for i in idxs]
        most_code = max(set(sub_wcode), key=sub_wcode.count) if sub_wcode else 0
        swh_vals = [safe_float(swh[i]) for i in idxs if safe_float(swh[i]) > 0.05]
        avg_swh_b = sum(swh_vals) / len(swh_vals) if swh_vals else 0.0
        swp_vals = [safe_float(swp[i]) for i in idxs if safe_float(swp[i]) > 0.1]
        avg_swp_b = sum(swp_vals) / len(swp_vals) if swp_vals else 0.0
        angles_swd = [swd[i] for i in idxs if i < len(swd)]; avg_swd_b = circular_mean(angles_swd)
        angles_wave_dir = [wd_wave[i] for i in idxs if i < len(wd_wave)]; avg_wave_dir = circular_mean(angles_wave_dir)
        angles_wd_b = [wd[i] for i in idxs if i < len(wd)]; avg_wd_b = circular_mean(angles_wd_b)
        avg_air = sum(safe_float(ta[i]) for i in idxs if ta[i] is not None); air_count = sum(1 for i in idxs if ta[i] is not None); avg_air = avg_air / air_count if air_count > 0 else 20.0
        rh_vals = [safe_float(rh[i]) for i in idxs if safe_float(rh[i]) > 5]; avg_rh = sum(rh_vals) / len(rh_vals) if rh_vals else 65.0
        gust_vals = [safe_float(wg[i]) for i in idxs if i < len(wg)]; max_gust_b = max(gust_vals) if gust_vals else 0.0
        avg_press_b = sum(safe_float(pr[i]) for i in idxs)/len(idxs) if pr else 0
        vis_vals = [safe_float(vis[i]) for i in idxs if safe_float(vis[i]) > 0]; avg_vis_b = sum(vis_vals) / len(vis_vals) if vis_vals else 10000.0
        wp_vals = [safe_float(wp[i]) for i in idxs if safe_float(wp[i]) > 0.5]; avg_wp_b = sum(wp_vals) / len(wp_vals) if wp_vals else 0.0
        pr_period = [safe_float(pr[i]) for i in idxs]; press_change_3h_period = pr_period[-1] - pr_period[0] if len(pr_period) >= 2 else 0.0
        period_hours = len(pr_period); press_rate = press_change_3h_period * (3.0 / max(1, period_hours)) if period_hours > 0 else 0.0
        period_lateral_fx, period_lateral_fy = 0.0, 0.0
        for i in idxs:
            if wd_wave[i] is not None:
                w_dir = wd_wave[i]; signed_angle = math.radians(signed_angle_diff(w_dir, orient))
                force = safe_float(wh[i]) ** 2; period_lateral_fx += force * math.sin(signed_angle); period_lateral_fy += force * math.cos(signed_angle)
        period_total_force = math.sqrt(period_lateral_fx**2 + period_lateral_fy**2)
        period_lateral_force_ratio = abs(period_lateral_fx) / period_total_force if period_total_force > 1e-9 else 0.0
        period_is_mirror_sea = max_h < 0.3
        period_is_lateral_strong = period_lateral_force_ratio > 0.7 and avg_h > 0.6
        if period_is_mirror_sea:
            sea = "بحر مرآوي"
            if max_gust_b >= 15: mirror_with_gusts.append(key)
            else: mirror_without_gusts.append(key)
        elif max_h < 0.9: sea = "هادئ"
        elif max_h < 1.3: sea = "متوسط الهيجان"
        else: sea = "هائج"

        period_nogo = []
        if period_is_mirror_sea: period_nogo.append("بحر مرآوي تام (أقل من 0.3م): لا تيارات ولا حركة سطحية، الأسماك لا تقترب.")
        if max_h > 2.0: period_nogo.append(f"بحر هائج (أمواج > 2.0م): الرمي مستحيل والخطر كبير.")
        if avg_wp_b > 10.0 and avg_h > 0.8: period_nogo.append(f"أمواج أرضية عالية الطاقة (فترة {avg_wp_b:.1f} > 10 ثوانٍ مع ارتفاع {avg_h:.2f}م): تجرف الرصاص وتدفن الخيوط.")
        if most_code in [95, 96, 99]: period_nogo.append("خطر الصواعق والبرق: قصبة الكاربون تجذب البرق، لا ترفعها.")
        if max_gust_b > 60: period_nogo.append(f"رياح عاتية في هذه الفترة (هبات {max_gust_b:.0f} > 60 كم/س): تمنع الرمي الآمن وقد تقطع الخيط.")
        if accumulated_rain_48h > 50.0 or max_rain_hourly > 20.0: period_nogo.append("عكارة طينية شديدة (سيول غزيرة): انعدام الرؤية تحت الماء، الأسماك تختفي.")
        if avg_vis_b < 200: period_nogo.append("ضباب كثيف جداً (رؤية < 200م): خطر على السلامة وصيد مستحيل.")
        if period_is_lateral_strong and avg_h > 0.8: period_nogo.append("تيار جانبي عنيف: يجرف الرصاصة فوراً ولا يمكن تثبيت الطعم.")

        backwash = analyze_backwash(avg_w, avg_wd_b, orient, avg_h)
        if backwash["severity"] == "مرتفع": period_nogo.append(f"تيار راجع عنيف: {backwash['effect']}")
        debris = analyze_debris_risk(sea_memory, avg_w)
        if debris["risk"] == "مرتفع": period_nogo.append(f"أوساخ وصوفة كثيفة: {debris['effect']}")

        period_warnings = []
        if period_is_mirror_sea and max_gust_b >= 15: period_warnings.append("بحر مرآوي مع تموجات سطحية خفيفة بسبب الرياح.")
        if is_weedy: period_warnings.append("صوفة محتملة في هذه الفترة.")
        if abs(press_rate) > 4.0: period_warnings.append(f"اضطراب ضغط في هذه الفترة ({press_rate:+.1f} hPa/3h)")
        elif press_rate > 1.5: period_warnings.append(f"ضغط مرتفع في هذه الفترة ({press_rate:+.1f} hPa/3h)")
        elif press_rate < -2.0: period_warnings.append(f"ضغط منخفض في هذه الفترة ({press_rate:.1f} hPa/3h)")

        has_lethal_nogo = len(period_nogo) > 0
        if has_lethal_nogo: periods_with_lethal += 1

        has_swell_dir = actual_swell_exists and (avg_swd_b is not None)
        final_swd = avg_swd_b if has_swell_dir else None
        has_wave_dir = avg_wave_dir is not None; final_wd = avg_wave_dir if has_wave_dir else None
        swell_angle = angle_diff(final_swd, orient) if final_swd is not None else None
        wave_angle = angle_diff(final_wd, orient) if final_wd is not None else None
        swell_wave_interaction = "متوافقان"
        if not period_is_mirror_sea and swell_angle is not None and wave_angle is not None and final_swd and final_wd:
            diff_sw = angle_diff(final_swd, final_wd)
            if diff_sw > 40: swell_wave_interaction = "متقاطعان بشدة"
            elif diff_sw > 25: swell_wave_interaction = "متقاطعان بسيط"

        if avg_wd_b is not None:
            wind_dir_rad = math.radians(avg_wd_b); orient_rad = math.radians(orient)
            frontal = math.cos(wind_dir_rad - orient_rad); lateral = abs(math.sin(wind_dir_rad - orient_rad))
            wind_effect_dist = avg_w * (-frontal) * (1 - 0.5 * lateral)
        else: wind_effect_dist = 0.0

        block_wind_ok = (avg_w < 20 and wc_dom.startswith("بحرية")) or (wc_dom.startswith("برية") and avg_w <= 15)
        block_wave_ok = 0.6 <= avg_h <= 1.2
        is_night = key in ("evening", "late_night")
        hw1_t = parse_tidal_time(tidal_windows["HW1"]); hw2_t = parse_tidal_time(tidal_windows["HW2"])
        sunset_t = safe_parse_time(sunset); sunrise_t = safe_parse_time(sunrise)
        if key == "late_night": is_night_with_tide = is_night and (is_close(hw1_t, sunrise_t, 1.5) or is_close(hw2_t, sunrise_t, 1.5))
        elif key == "evening": is_night_with_tide = is_night and (is_close(hw1_t, sunset_t, 1.5) or is_close(hw2_t, sunset_t, 1.5))
        else: is_night_with_tide = False

        period_flags_dict = {"is_spring_tide": 1 if is_spring_tide else 0, "is_pressure_dropping": 1 if press_rate < -2.0 else 0}
        confidence = calculate_confidence_index(period_flags_dict, period_is_mirror_sea, has_golden_window,
                                                len(period_nogo), len(period_warnings), block_wind_ok, block_wave_ok, is_night_with_tide)
        base_dist = 40 if avg_h > 0.8 else (50 if avg_h > 0.5 else 60)
        recommended_dist = max(30, min(100, round(base_dist + wind_effect_dist * 1.5)))
        corr_angle = 0 if avg_wd_b is None else casting_angle_correction(avg_wd_b, orient)
        water_clarity = get_water_clarity(avg_w, avg_h, is_murky, is_weedy, haml_info["status"], avg_vis_b)
        rig = suggest_rig(haml_info["status"], period_is_lateral_strong, avg_w, period_is_mirror_sea)
        comfort = calculate_comfort_index(avg_air, avg_w, avg_rh)
        sst_period = [sst[i] for i in idxs if i < len(sst) and sst[i] is not None and 8.0 <= sst[i] <= 35.0]
        avg_sst_period = sum(sst_period) / len(sst_period) if sst_period else (avg_sst if avg_sst is not None else 20.0)
        active_fish, inactive_fish = get_period_fish_status(avg_sst_period, is_night, is_murky, is_weedy, period_is_mirror_sea, period_lateral_force_ratio, water_clarity, seabass_sst_limit)

        start_time = all_times[target_idx[idxs[0]]].strftime('%H:%M')
        end_time = all_times[target_idx[idxs[-1]]].strftime('%H:%M')
        time_range = f"{start_time} - {end_time}"

        block_data = {
            "name": period_names_arabic[key], "time_range": time_range,
            "sea_state":sea,"wave_height":f"أقصى {max_h:.2f}م",
            "swell_dir": deg_to_compass(final_swd) if final_swd is not None else ("معدوم" if not actual_swell_exists else "غير معروف"),
            "wave_dir": deg_to_compass(final_wd) if final_wd is not None else "غير معروف",
            "swell_angle_diff": round(swell_angle,0) if swell_angle is not None else None,
            "wave_angle_diff": round(wave_angle,0) if wave_angle is not None else None,
            "swell_wave_interaction": swell_wave_interaction,
            "wind_speed":f"متوسط {avg_w:.1f} - أقصى {max_w:.1f} كم/س","wind_gust_peak":round(max_gust_b,1),
            "wind_dir":wc_dom, "air_temp":round(avg_air,1), "weather":weather_desc(most_code),
            "confidence": confidence, "confidence_label": get_confidence_label(confidence),
            "recommended_cast_distance": round(recommended_dist, 0), "casting_angle_correction": corr_angle,
            "water_clarity": water_clarity, "suggested_rig": rig, "comfort_index": comfort,
            "backwash": backwash, "debris": debris, "active_fish": active_fish, "inactive_fish": inactive_fish,
            "period_warnings": period_warnings, "nogo_reasons": period_nogo, "has_lethal_nogo": has_lethal_nogo,
            "_raw": {
                "avg_wave_h": round(avg_h, 3), "max_wave_h": round(max_h, 3),
                "avg_wind": round(avg_w, 1), "max_wind": round(max_w, 1),
                "max_gust": round(max_gust_b, 1), "swell_h": round(avg_swh_b, 3), "swell_p": round(avg_swp_b, 1),
                "air_temp": round(avg_air, 1), "pressure": round(avg_press_b, 1),
                "visibility": round(avg_vis_b, 0), "has_swell": actual_swell_exists,
                "wave_period": round(avg_wp_b, 1), "wind_dir_deg": round(avg_wd_b, 0) if avg_wd_b is not None else 0,
                "wind_effect_dist": round(wind_effect_dist, 0), "recommended_cast_distance": round(recommended_dist, 0),
                "press_rate": round(press_rate, 2)
            }
        }
        blocks.append(block_data)

    if mirror_with_gusts:
        names = [period_names_arabic[p] for p in mirror_with_gusts]; warnings.append(f"بحر مرآوي مع تموجات سطحية خفيفة بسبب الرياح في: {', '.join(names)}.")
    if mirror_without_gusts:
        names = [period_names_arabic[p] for p in mirror_without_gusts]; warnings.append(f"بحر مرآوي تام (سطح أملس) في: {', '.join(names)}.")

    any_pressure_rising = any(b["_raw"].get("press_rate", 0) > 1.5 for b in blocks)
    any_pressure_dropping = any(b["_raw"].get("press_rate", 0) < -2.0 for b in blocks)

    all_periods_have_lethal = (populated_periods > 0) and (periods_with_lethal == populated_periods)
    any_period_has_lethal = periods_with_lethal > 0

    if all_periods_have_lethal: final_verdict = "غير مناسب"
    elif any_period_has_lethal: final_verdict = "فرصة مع تحفظات"
    elif warnings: final_verdict = "فرصة مع تحفظات"
    else: final_verdict = "مناسب"

    all_nogo_reasons = [r for b in blocks for r in b["nogo_reasons"]]
    reds, greens = [], []
    for i in range(len(wh)):
        hh = all_times[target_idx[i]].strftime("%H:%M")
        wg_i = safe_float(wg[i]) if i < len(wg) else 0.0
        pr_i = safe_float(pr[i]) if i < len(pr) else 1013.0
        ws_i = safe_float(ws[i]) if i < len(ws) else 0.0
        if wave_power[i] > 3 or safe_float(wh[i]) > 1.8 or wg_i > 50 or pr_i < 1005: reds.append(hh)
        if 0.3 <= safe_float(wh[i]) <= 1 and 0.1 <= wave_power[i] <= 1.5 and ws_i < 27.8: greens.append(hh)

    weed_analysis = analyze_weed_risk(sea_memory)
    seasonal_bait = get_seasonal_bait(month, avg_sst if avg_sst is not None else 20.0)

    extra = {
        "pressure_avg":round(avg_press,1), "peak_gust_today":round(peak_gust_day,1),
        "sunrise":sunrise, "sunset":sunset, "max_air_temp": round(max_air_temp, 1),
        "tidal_windows": tidal_windows, "golden_windows": golden_windows,
        "has_swell_data": has_swell_data, "actual_swell_exists": actual_swell_exists,
        "solunar": solunar, "slack_times": slack_info,
        "weed_risk": weed_analysis, "seasonal_bait": seasonal_bait,
        "past_rain_accumulated_48h": round(accumulated_rain_48h, 1),
        "max_rain_hourly": round(max_rain_hourly, 1),
        "pressure_change": round(press_change_day, 1), "pressure_note": pressure_note,
        "moon_age_days": round(moon_age, 1),
        "haml_status": haml_info["status"], "haml_phase": haml_info["phase"],
        "haml_description": haml_info["description"], "haml_score_delta": haml_info["score_delta"],
        "platform_advice": platform_advice, "sst_stability": sst_stability
    }

    flags = {
        "is_mirror_sea": any(b["sea_state"] == "بحر مرآوي" for b in blocks),
        "is_lateral_strong": day_lateral_ratio > 0.7 and avg_wave_h_day > 0.6,
        "is_pressure_rising_fast": any_pressure_rising, "is_pressure_dropping_fast": any_pressure_dropping,
        "is_cross_sea_dangerous": is_cross_sea_dangerous, "is_murky": is_murky, "is_weedy": is_weedy,
        "has_golden_window": has_golden_window, "is_neap_tide": is_neap_tide, "is_spring_tide": is_spring_tide
    }

    agg_result = {
        "dominant_wind":dominant, "blocks":blocks, "red_flags":reds[:5], "green_flags":greens[:5],
        "sea_memory":sea_memory, "lateral_current":"تيار جانبي متغير", "pressure_state":pressure_state,
        "tide_analysis":tide_analysis, "sst_stability":sst_stability,
        "hidden_factors": {"cross_sea_risk": cross_sea_risk, "wave_steepness": steepness_desc,
                           "golden_lock": "مد قوي" if is_spring_tide else "مد ضعيف" if is_neap_tide else "متوسط"},
        "avg_sst": avg_sst, "extra_info":extra, "transitions": [], "flags": flags,
        "nogo_reasons": all_nogo_reasons, "warnings": warnings, "final_verdict": final_verdict,
        "target_month": target_date_obj.month
    }
    score = apply_scoring(agg_result)
    agg_result["score"] = score
    return agg_result

# ---------- بناء التقرير ----------
def format_tidal_flow_periods(tidal_windows: dict) -> dict:
    periods = {}
    for key, time_str in tidal_windows.items():
        h = safe_parse_time(time_str)
        slack_start = format_time((h - 0.5) % 24); slack_end = format_time((h + 0.5) % 24)
        is_high = key.startswith("HW")
        early_end = slack_start
        if is_high: early_label = "آخر مد دافع"
        else: early_label = "آخر جزر ساحب"
        late_start = slack_end
        if is_high: late_label = "بداية جزر ساحب"
        else: late_label = "بداية مد دافع"
        name = "المد العالي" if is_high else "الجزر المنخفض"
        periods[key] = {"name": name, "time": time_str, "early_label": early_label, "early_time": early_end,
                       "slack": f"{slack_start} - {slack_end}", "late_label": late_label, "late_time": late_start}
    return periods

def build_flow_section(tidal_windows: dict) -> str:
    periods = format_tidal_flow_periods(tidal_windows)
    lines = ["🏃‍♂️ 2. فترات الحركة مقابل المياه الميتة (تقديرية)"]
    lines.append("↳ تنويه: أوقات المد والجزر تقديرية بهامش خطأ ±30 دقيقة.")
    for key, data in periods.items():
        lines.append(f" * 🌊 {data['name']} ({data['time']}):")
        lines.append(f"   * 🔵 {data['early_label']}: حتى {data['early_time']}")
        lines.append(f"   * 🔴 مياه ميتة: {data['slack']}")
        lines.append(f"   * 🟢 {data['late_label']}: من {data['late_time']}")
    lines.append("↳ نصيحة: أفضل صيد في نهاية المد الدافع أو بداية الجزر الساحب (حسب الظاهرة). تجنب مركز المياه الميتة.")
    return "\n".join(lines)

def calculate_interactions(agg: dict) -> List[str]:
    interactions = []
    extra = agg.get("extra_info", {}); blocks = agg.get("blocks", [])
    hidden_factors = agg.get("hidden_factors", {}); solunar = extra.get("solunar", {})
    warnings = agg.get("warnings", []); nogo_reasons = agg.get("nogo_reasons", [])
    final_verdict = agg.get("final_verdict", "غير مناسب"); pressure_note = extra.get("pressure_note", "مستقر")
    sunrise = extra.get("sunrise", "غير متوفر"); sunset = extra.get("sunset", "غير متوفر")
    interactions.append(f"[الشروق والغروب] 🌅 الشروق: {sunrise} | 🌇 الغروب: {sunset}")
    interactions.append(f"[الضغط الجوي] {pressure_note}")
    haml_status = extra.get("haml_status", ""); haml_phase = extra.get("haml_phase", "")
    haml_desc = extra.get("haml_description", ""); platform_advice = extra.get("platform_advice", "")
    interactions.append(f"[مؤشر الشاطئ] {haml_status} ({haml_phase}). {haml_desc} {platform_advice}")
    interactions.append(f"[انحدار الموج] {hidden_factors.get('wave_steepness', 'متوسط')}")
    interactions.append(f"[فترات سولونار] 🎯 الفترات الرئيسية: {solunar.get('major1')} | {solunar.get('major2')} 🎯 الفترات الثانوية: {solunar.get('minor1')} | {solunar.get('minor2')}")
    for b in blocks:
        name = b['name']; time_range = b['time_range']; raw = b.get("_raw", {})
        interactions.append(f"[{name} ({time_range})]")
        interactions.append(f"  📊 حالة البحر والثقة: {b['sea_state']} | نسبة الثقة: {b.get('confidence',0)}% ({b.get('confidence_label','')})")
        wind_eff = raw.get('wind_effect_dist', 0); sign = '+' if wind_eff > 0 else '' if wind_eff < 0 else ''
        interactions.append(f"  💨 الرياح والرمي: {b['wind_dir']} {raw.get('avg_wind',0):.1f} كم/س (هبات {raw.get('max_gust',0)} كم/س) | تأثير: {sign}{wind_eff:.0f}م")
        wp_val = raw.get('wave_period', 0)
        interactions.append(f"  🌊 الموج والمسافة: {wp_val:.1f} ث | المسافة: {raw.get('recommended_cast_distance',0):.0f}م" if wp_val > 0 else "  🌊 الموج والمسافة: غير متوفر")
        if b.get("nogo_reasons"): interactions.append(f"  ⛔ موانع: {'; '.join(b['nogo_reasons'])}")
        if b.get("period_warnings"): interactions.append(f"  ⚠️ تحذيرات: {'; '.join(b['period_warnings'])}")
    interactions.append(f"[نسبة النجاح] {agg.get('score', 0)}%")
    if final_verdict == "فرصة مع تحفظات":
        interactions.append(f"[الحسم النهائي] فرصة مع تحفظات: {', '.join(warnings) if warnings else 'ظروف متغيرة'}")
    elif final_verdict == "غير مناسب":
        interactions.append(f"[الحسم النهائي] غير مناسب: {', '.join(nogo_reasons)}")
    else: interactions.append(f"[الحسم النهائي] مناسب، الظروف ممتازة.")
    return interactions

def build_context(req, agg, tz_name):
    extra = agg["extra_info"]; chain_interactions = calculate_interactions(agg)
    target_date = resolve_target_date(req.target_date, datetime.now(zoneinfo.ZoneInfo(tz_name)).date())
    date_str = target_date.strftime("%d/%m/%Y")
    if agg["final_verdict"] == "غير مناسب":
        main_reason = "بحر غير مناسب للصيد في جميع الفترات"
        if agg["nogo_reasons"]: main_reason = agg["nogo_reasons"][0]
    elif agg["final_verdict"] == "فرصة مع تحفظات":
        good_periods = [b["name"] for b in agg["blocks"] if not b["has_lethal_nogo"]]
        bad_periods = [b["name"] for b in agg["blocks"] if b["has_lethal_nogo"]]
        if bad_periods: main_reason = f"فترات غير مناسبة: {', '.join(bad_periods)}. فترات مناسبة: {', '.join(good_periods) if good_periods else 'لا يوجد'}"
        else: main_reason = "توجد تحفظات لكن يمكن التكيف معها"
    else: main_reason = "ظروف ممتازة للصيد"
    facts = [
        f"🎯 0. الملخص التنفيذي ليوم {date_str}",
        f"> نسبة النجاح: {agg['score']}%",
        f">  * القرار النهائي: {agg['final_verdict']}",
        f">  * السبب الرئيسي: {main_reason}",
        f">  * الطعم المستهدف: {extra.get('seasonal_bait', '')}",
    ]
    facts.append("⏱️ 1. التوقيت المدوي وحركة المياه")
    facts.append("🌊 مواقيت المد والجزر (تقديرية)")
    for k, v in extra.get("tidal_windows", {}).items():
        name = "المد العالي" if k.startswith("HW") else "الجزر المنخفض"
        facts.append(f" * 🔹 {name}: الساعة {v}")
    facts.append(f" * 🌅 الشروق: {extra.get('sunrise', '')} | 🌇 الغروب: {extra.get('sunset', '')}")
    facts.append("🏖️ مؤشر الشاطئ (أيام الحياء والمات)")
    facts.append(f" * 🌊 الوضعية: {extra.get('haml_status', '')} ({extra.get('haml_phase', '')}).")
    facts.append(f" * 📌 حالة البحر: {extra.get('haml_description', '')}")
    facts.append("🌡️ الضغط الجوي")
    facts.append(f" * الوضع: {extra.get('pressure_note', 'مستقر')}")
    lines = ["\n".join(facts), "", "=== التفاعلات ===", *chain_interactions]
    return "\n".join(lines)

# ---------- SYSTEM PROMPT (مختصر) ----------
SYSTEM_PROMPT = """أنت خبير سيرفكاستينغ تونسي. اكتب تقريرًا بالدارجة التونسية باستخدام البيانات التالية.
القرار النهائي ونسبة النجاح موجودان في [الحسم النهائي] و[نسبة النجاح]. لا تغيرهما.

استخدم التنسيق التالي بالضبط:
🎯 0. الملخص التنفيذي ليوم (التاريخ) – النسبة، القرار، السبب، الطعم.
⏱️ 1. التوقيت المدوي وحركة المياه – المد والجزر، الشروق/الغروب، مؤشر الشاطئ، الضغط، السولونار.
🏃‍♂️ 2. فترات الحركة مقابل المياه الميتة (مضمّن تلقائياً).
🕒 3. التفكيك الديناميكي الزمني – لكل فترة: الحالة، الرياح، الموج، الموانع والتحذيرات (إن وجدت).
⚖️ 4. ميزان العوامل – العوامل الحمراء (المعوقات) والخضراء (الإيجابيات).
🏹 5. التكتيك الميداني والسلامة – الرصاص، التوقيت، المسافة.

قواعد مهمة:
- اكتب بالدارجة التونسية فقط.
- لا تخترع موانع غير موجودة في البيانات.
- استخدم الأوقات والنطاقات كما هي في البيانات.
- لا تذكر المركب ولا تستخدم حروفًا لاتينية.
"""

# ---------- استدعاء OpenRouter ----------
async def call_openrouter(ctx: str) -> str:
    MAX_CONTEXT_CHARS = 30000
    if len(ctx) > MAX_CONTEXT_CHARS:
        ctx = ctx[:MAX_CONTEXT_CHARS]
        logger.warning(f"Context truncated to {MAX_CONTEXT_CHARS} chars")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://spotdata.onrender.com",
        "X-Title": "Surfcasting Analytics"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ctx}
        ],
        "temperature": 0.1,
        "max_tokens": 15000
    }

    RETRY_WAITS = [20, 30, 40]
    max_retries = 3
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(1, max_retries + 1):
            try:
                r = await client.post(OPENROUTER_URL, json=payload, headers=headers)
                if r.status_code == 429:
                    wait = RETRY_WAITS[min(attempt - 1, len(RETRY_WAITS) - 1)] + random.uniform(0, 5)
                    logger.warning(f"OpenRouter rate limit, retrying in {wait:.1f}s...")
                    await asyncio.sleep(wait)
                    continue
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except httpx.TimeoutException as e:
                logger.warning(f"OpenRouter timeout (attempt {attempt}/{max_retries}): {e}")
                if attempt == max_retries:
                    raise HTTPException(504, "انتهت مهلة الاتصال بـ OpenRouter")
                await asyncio.sleep(RETRY_WAITS[min(attempt - 1, len(RETRY_WAITS) - 1)])
            except (httpx.HTTPStatusError, KeyError, IndexError) as e:
                logger.error(f"OpenRouter call failed: {e}")
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code not in [429, 500, 502, 503]:
                    raise
                if attempt == max_retries:
                    raise Exception("فشل الاتصال بـ OpenRouter بعد عدة محاولات")
                await asyncio.sleep(RETRY_WAITS[min(attempt - 1, len(RETRY_WAITS) - 1)])
            except Exception as e:
                logger.error(f"OpenRouter unexpected error: {e}")
                if attempt == max_retries:
                    raise Exception(f"خطأ غير متوقع في الاتصال بـ OpenRouter: {type(e).__name__}")
                await asyncio.sleep(RETRY_WAITS[min(attempt - 1, len(RETRY_WAITS) - 1)])
    raise Exception("فشل الاتصال بـ OpenRouter")

# ---------- معالجة النصوص ----------
def fix_time_ranges(text: str) -> str:
    pattern = r'(\d{2}:\d{2})\s*[-–]\s*(\d{2}:\d{2})'
    def repl(m):
        t1, t2 = m.group(1), m.group(2)
        to_min = lambda s: int(s.split(':')[0])*60 + int(s.split(':')[1])
        m1, m2 = to_min(t1), to_min(t2)
        diff = m2 - m1
        is_swapped_same_day = diff < 0 and abs(diff) < 720
        if is_swapped_same_day:
            return f"{t2} - {t1}"
        return f"{t1} - {t2}"
    return re.sub(pattern, repl, text)

def fix_broken_time_in_headers(text: str) -> str:
    text = re.sub(r'(\*\s+[^*(]+)\((\d{2}:\d{2})\s*\n\s*\*\s+(\d{2}:\d{2})\)', r'\1(\2 - \3)', text)
    return text

def replace_english_commas(text: str) -> str:
    text = re.sub(r'(?<=[\u0600-\u06FF\s\d]),(?=[\u0600-\u06FF\s\d])', '،', text)
    return text

def enforce_line_breaks(text: str) -> str:
    lines = text.split('\n')
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            new_lines.append('')
            continue
        if re.match(r'^[*-] ', stripped) or re.match(r'^[🔹🔸🌊🟢🔴🎯⏱️🏖️⏳🏃‍♂️🕒⚖️🏹📊🌅☀️🌃🌆🌙🐟💤🔄💨📊📐🌡️🛠️⏱️🎯🦐⚠️📌💧😌⛔]', stripped):
            if new_lines and new_lines[-1] != '' and not re.match(r'^[*-] ', new_lines[-1]):
                new_lines.append('')
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
        if re.match(r'^(🎯|⏱️|🏃‍♂️|🕒|⚖️|🏹|📊|\d\.)', stripped):
            if new_lines and new_lines[-1] != '':
                new_lines.append('')
        new_lines.append(stripped)
    return '\n'.join(new_lines)

def clean_report_text(text: str) -> str:
    text = re.sub(r'(المد العالي|الجزر المنخفض):(\d{2}:\d{2})', r'\1: \2', text)
    text = re.sub(r'(\w)\s+:\s+', r'\1: ', text)
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
        if re.search(r'\b\d{2}:\s*$', line) and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if re.match(r'^\d{2}\b', next_line):
                merged = line.rstrip() + next_line
                merged = re.sub(r'(\d+):\s+(\d+)', r'\1:\2', merged)
                fixed.append(merged)
                i += 2
                continue
        line = re.sub(r'(\d+):\s+(\d+)', r'\1:\2', line)
        fixed.append(line)
        i += 1
    return '\n'.join(fixed)

# ========== نقطة نهاية التقرير الرئيسية ==========
@app.post("/generate-report")
@limiter.limit("2/minute")   # يمكن خفضها إلى 1/minute حسب الحاجة
async def generate_report(request: Request, req: RawDataReportRequest):
    try:
        return await asyncio.wait_for(_generate_report_inner(req), timeout=150.0)
    except asyncio.TimeoutError:
        logger.error("generate-report global timeout")
        raise HTTPException(504, detail="انتهت مهلة إنشاء التقرير. حاول مرة أخرى.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"generate-report error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, detail="فشل إنشاء التقرير")

async def _generate_report_inner(req: RawDataReportRequest):
    async with httpx.AsyncClient(timeout=30.0) as client:
        if req.marine_data and req.weather_data:
            marine_data, weather_data = req.marine_data, req.weather_data
        else:
            lat = req.latitude or 36.8; lon = req.longitude or 10.1
            marine_data, weather_data = await asyncio.gather(
                fetch_marine_data_from_openmeteo(client, lat, lon),
                fetch_weather_data_from_openmeteo(client, lat, lon),
                return_exceptions=True
            )
            if isinstance(marine_data, BaseException):
                logger.error(f"Marine data fetch exception: {marine_data}")
                raise HTTPException(502, "تعذر جلب بيانات البحر من المصدر")
            if isinstance(weather_data, BaseException):
                logger.error(f"Weather data fetch exception: {weather_data}")
                raise HTTPException(502, "تعذر جلب بيانات الطقس من المصدر")
            if not marine_data:
                raise HTTPException(502, "بيانات البحر فارغة")
            if not weather_data:
                raise HTTPException(502, "بيانات الطقس فارغة")

    marine_hourly = marine_data.get("hourly", marine_data)
    weather_hourly = weather_data.get("hourly", {})
    daily = weather_data.get("daily", {})
    tz_name = marine_data.get("timezone", "Africa/Tunis")
    now_tn = datetime.now(zoneinfo.ZoneInfo("Africa/Tunis"))
    target_dt = resolve_target_date(req.target_date, now_tn.date())
    date_index_map = {"today": 0, "tomorrow": 1, "day_after": 2}
    day_idx = date_index_map.get(req.target_date, 0)

    sunrise_list = daily.get("sunrise", ["06:00"] * 3)
    sunset_list  = daily.get("sunset", ["18:00"] * 3)
    raw_sr = sunrise_list[min(day_idx, len(sunrise_list)-1)] if sunrise_list else "06:00"
    raw_ss = sunset_list[min(day_idx, len(sunset_list)-1)] if sunset_list else "18:00"
    m_sr = _TIME_RE.search(raw_sr)
    sunrise = m_sr.group() if m_sr else "06:00"
    m_ss = _TIME_RE.search(raw_ss)
    sunset = m_ss.group() if m_ss else "18:00"
    latitude = req.latitude or 36.8
    longitude = req.longitude or 10.1

    all_times, aligned = align_hourly_data(marine_hourly, weather_hourly, tz_name)
    if not all_times:
        raise HTTPException(500, "لا توجد بيانات ساعية متزامنة")

    agg = aggregate_physics(all_times, aligned, req.beach_orientation, target_dt, sunrise, sunset, latitude, longitude)

    HARD_NOGO_KEYWORDS = ["هائج", "صواعق", "ضباب كثيف", "عكارة طينية", "رياح عاتية", "أمواج أرضية", "تيار جانبي عنيف", "تيار راجع عنيف"]
    if (agg["final_verdict"] == "غير مناسب" and
        all(b.get("has_lethal_nogo", False) for b in agg["blocks"]) and
        any(any(kw in r for kw in HARD_NOGO_KEYWORDS) for r in agg["nogo_reasons"])):
        return {
            "report": f"❌ غير مناسب ({agg['score']}%) – {agg['nogo_reasons'][0]}",
            "meta": {"score": agg['score'], "hard_nogo": True}
        }

    ctx = build_context(req, agg, tz_name)
    report = await call_openrouter(ctx)

    report = clean_report_text(report)
    report = fix_broken_number_lines(report)
    report = fix_broken_time_in_headers(report)
    report = replace_english_commas(report)
    report = enforce_line_breaks(report)
    report = add_paragraph_spacing(report)

    flow_section = build_flow_section(agg["extra_info"]["tidal_windows"])
    pattern = r'🏃‍♂️\s*2\.\s*فترات الحركة.*?(?=🕒\s*3\.|⚖️\s*4\.|$)'
    report = re.sub(pattern, flow_section + '\n\n', report, flags=re.DOTALL)

    report = re.sub(r'6\.\s*الأرقام المرجعية.*$', '', report, flags=re.DOTALL).strip()

    raw_numbers = "\n\n━━━━━━━━━━━━━━━━━━━━━━━━\n📊 الأرقام المرجعية (للتحقق)\n━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    avg_sst_display = round(agg['avg_sst'], 1) if agg['avg_sst'] is not None else "غير متوفر"
    raw_numbers += f"🔹 حرارة الماء: {avg_sst_display}°م | حرارة الهواء: {agg['extra_info']['max_air_temp']}°م\n"
    raw_numbers += f"🔹 الرياح: أقصى هبات {agg['extra_info']['peak_gust_today']} كم/س | الضغط: {agg['extra_info']['pressure_avg']} hPa\n"
    for b in agg['blocks']:
        r = b['_raw']
        wind_eff = r['wind_effect_dist']
        sign = '+' if wind_eff > 0 else ''
        raw_numbers += (f"🔸 {b['name']} ({b['time_range']}): "
                        f"ثقة {b['confidence']}% ({b.get('confidence_label','')}) | مسافة {b['recommended_cast_distance']}م | "
                        f"موج {r['avg_wave_h']}-{r['max_wave_h']}م | رياح {r['avg_wind']} كم/س ({r.get('wind_dir_deg','')}°) | "
                        f"تأثير الرياح {sign}{wind_eff:.0f}م | "
                        f"عكارة: {b.get('water_clarity','')} | مونتاج: {b.get('suggested_rig','')} | راحة: {b.get('comfort_index','')}%")
        if b.get("nogo_reasons"):
            raw_numbers += f" | ⛔: {'; '.join(b['nogo_reasons'])}"
        raw_numbers += "\n"
    raw_numbers += "━━━━━━━━━━━━━━━━━━━━━━━━\n"
    report += raw_numbers

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
