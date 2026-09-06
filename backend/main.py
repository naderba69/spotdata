"""
Surfcasting Analytics API – v22.0.0 (Enterprise‑Grade – Production Ready)

الإرث (v21.0.1):
- Dead‑zone in pressure scoring eliminated (continuous logic).
- wind_dir_deg is None when data missing (no false north).
- Sea‑state classification keeps steepness for all wave heights.
- Lateral current physics corrected (short waves penalised).
- Solunar minor times fixed (midpoint between majors).
- Flow section injected at correct position (before section 3).
- Division‑by‑zero protection (steepness, period).
- Index‑safe sunrise/sunset lookup.
- Graceful degradation on empty data.
- Lunar transit naming replaces misleading "tide" labels.
- API key hidden in logs.
- Period order starts with late_night (00:00‑04:00).
- Beach orientation displayed in report.
- Scoring balanced (mat penalty reduced, not‑suitable threshold lowered).

جديد v22.0.0 (الأداء والجودة):
- إعدادات مركزية من متغيّرات البيئة (backend/config.py).
- كاش ذاكري TTL + Single‑flight (backend/cache.py): تقارير، Open‑Meteo، Overpass،
  وتذكّر الفشل (negative caching) مع تقديم قيمة قديمة عند تعذّر المصدر.
- Gemini: إيقاف «التفكير العميق» افتراضياً (thinking budget = 0) لتقليل الزمن،
  مهلة قراءة قابلة للضبط، إعادة محاولة أقصر وأذكى (لا إعادة محاولة على 4xx).
- معرّف طلب (X‑Request‑ID) في كل ردّ وسجل، وقياس زمن المعالجة (X‑Response‑Time).
- تحقّق من حمولة البيانات قبل المعالجة: رسالة 400 واضحة بدل خطأ 500 مبهم.
- /health يعرض حالة الخدمات وإحصاءات الكاش (مخبّأ 30 ثانية).
- إصلاحات دقة: max(air temp) مع قيم None، فحص None لحرارة الماء،
  واستقرار المنطقة الزمنية عند اسم غير معروف.
"""
import os, math, asyncio, logging, traceback, zoneinfo, re, random, time, uuid, hashlib, json
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any, Tuple
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

try:  # يعمل عند التشغيل من داخل مجلد backend (uvicorn main:app)
    from config import settings
    from cache import TTLCache
except ImportError:  # أو كحزمة (uvicorn backend.main:app)
    from backend.config import settings
    from backend.cache import TTLCache

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("surfcasting")
logging.getLogger("httpx").setLevel(logging.WARNING)

GEMINI_API_KEY = settings.GEMINI_API_KEY
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY غير مضبوط: التطبيق يعمل لكن توليد التقارير سيفشل حتى يُضبط المفتاح.")

# ---------- API Key Protection in Logs ----------
class SensitiveDataFilter(logging.Filter):
    def __init__(self, secret: str):
        super().__init__()
        self.secret = secret

    def filter(self, record):
        if self.secret and hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = record.msg.replace(self.secret, "[GEMINI_KEY_HIDDEN]")
        return True

logger.addFilter(SensitiveDataFilter(GEMINI_API_KEY))

GEMINI_URL = settings.GEMINI_URL
GEMINI_RETRY_WAITS = settings.GEMINI_RETRY_WAITS

# ---------- In-memory caches (TTL + single-flight) ----------
CACHE_ENABLED = settings.CACHE_ENABLED
report_cache = TTLCache("reports", ttl=settings.CACHE_TTL_REPORT_S,
                        max_entries=max(16, settings.CACHE_MAX_ENTRIES // 4),
                        enabled=CACHE_ENABLED)
upstream_cache = TTLCache("openmeteo", ttl=settings.CACHE_TTL_UPSTREAM_S,
                          max_entries=settings.CACHE_MAX_ENTRIES, enabled=CACHE_ENABLED)
overpass_cache = TTLCache("overpass", ttl=settings.CACHE_TTL_OVERPASS_S,
                          max_entries=settings.CACHE_MAX_ENTRIES, enabled=CACHE_ENABLED)
failure_cache = TTLCache("failures", ttl=settings.CACHE_TTL_FAILURE_S,
                         max_entries=settings.CACHE_MAX_ENTRIES, enabled=CACHE_ENABLED)
health_cache = TTLCache("health", ttl=30.0, max_entries=4, enabled=CACHE_ENABLED)

START_MONOTONIC = time.monotonic()

def _hash_payload(payload: Any) -> str:
    """بصمة خفيفة لحمولة البيانات (تُستخدم في مفتاح كاش التقارير)."""
    try:
        raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        raw = repr(payload)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]

USER_AGENT = "SurfcastingAnalytics/2.0 (production)"
_TIME_RE = re.compile(r'\d{2}:\d{2}')

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

app = FastAPI(title=settings.APP_NAME, version=settings.VERSION, lifespan=lifespan,
              description="تحليل فيزيائي لحالات البحر والصيد الساحلي (سيرفكاستينغ) في تونس.")
limiter = Limiter(key_func=get_remote_address, enabled=settings.RATE_LIMIT_ENABLED)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = settings.ALLOWED_ORIGINS
if ALLOWED_ORIGINS == "*":
    logger.warning("CORS is open to all origins. Restrict in production with ALLOWED_ORIGINS.")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=False, max_age=600)
else:
    origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"], allow_credentials=True, max_age=600)

# ---------- Request context: معرّف الطلب + زمن المعالجة ----------
@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.error(f"[{request_id}] Unhandled middleware error\n{traceback.format_exc()}")
        response = JSONResponse(status_code=500,
                                content={"detail": "خطأ داخلي في الخادم", "request_id": request_id})
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{elapsed_ms:.0f}ms"
    return response

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """توحيد شكل أخطاء HTTP مع إرفاق معرّف الطلب لتسهيل التشخيص."""
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": request_id},
        headers=exc.headers,
    )

@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", None)
    logger.error(f"[{request_id}] Unhandled: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500,
                        content={"detail": "خطأ داخلي في الخادم", "request_id": request_id})

@app.get("/health")
async def health():
    """فحص صحة الخدمة + حالة الاعتماديات (مخبّأ 30 ثانية لتجنّب ضرب Overpass)."""

    async def _probe() -> dict:
        overpass_ok = False
        try:
            async with httpx.AsyncClient(timeout=5.0, headers={"User-Agent": USER_AGENT}) as c:
                r = await c.get(OVERPASS_SERVERS[0], params={"data": "[out:json];node(1);out;"})
                overpass_ok = r.status_code == 200
        except Exception as e:
            logger.warning(f"Overpass health probe failed: {e}")
        return {
            "status": "ok",
            "version": settings.VERSION,
            "gemini_configured": settings.gemini_configured,
            "gemini_model": settings.GEMINI_MODEL,
            "overpass_reachable": overpass_ok,
            "rate_limit_enabled": settings.RATE_LIMIT_ENABLED,
            "uptime_seconds": round(time.monotonic() - START_MONOTONIC, 1),
            "caches": {c.name: c.stats() for c in (report_cache, upstream_cache, overpass_cache, failure_cache)},
            "timestamp": datetime.now(resolve_timezone(settings.DEFAULT_TZ)).isoformat(),
        }

    value, _ = await health_cache.get_or_set("health", _probe, ttl=30.0)
    return value

# ---------- Utility Helpers ----------
def safe_float(v) -> float:
    try:
        result = float(v)
        return 0.0 if (math.isnan(result) or math.isinf(result)) else result
    except (TypeError, ValueError):
        return 0.0

def angle_diff(w, b):
    if w is None or b is None: return 180.0
    d = abs(w - b) % 360
    return 360 - d if d > 180 else d

def signed_angle_diff(w, b):
    if w is None or b is None: return 0.0
    return (w - b + 180) % 360 - 180

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
        if math.isnan(h) or math.isinf(h): return "00:00"
        h = h % 24
        hh = int(h)
        mm = int(round((h - hh) * 60))
        if mm >= 60: hh = (hh + 1) % 24; mm = 0
        return f"{hh % 24:02d}:{mm:02d}"
    except (TypeError, ValueError): return "00:00"

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
    if deg is None: return "غير معروف"
    val = int((deg / 22.5) + 0.5) % 16
    arr = ["شمال","شمال شمال شرق","شمال شرق","شرق شمال شرق","شرق","شرق جنوب شرق","جنوب شرق","جنوب جنوب شرق",
           "جنوب","جنوب جنوب غرب","جنوب غرب","غرب جنوب غرب","غرب","غرب شمال غرب","شمال غرب","شمال شمال غرب"]
    return arr[val]

def circular_mean(angles_deg: list) -> Optional[float]:
    valid = [a for a in angles_deg if a is not None]
    if not valid: return None
    radians_vals = [math.radians(a) for a in valid]
    sin_sum = sum(math.sin(r) for r in radians_vals)
    cos_sum = sum(math.cos(r) for r in radians_vals)
    if abs(sin_sum) < 1e-10 and abs(cos_sum) < 1e-10: return None
    return (math.degrees(math.atan2(sin_sum, cos_sum)) + 360) % 360

def resolve_target_date(txt, real_today):
    if txt == "today": return real_today
    if txt == "tomorrow": return real_today + timedelta(days=1)
    return real_today + timedelta(days=2)

def pick_daily_value(daily: dict, key: str, target_date: date, today: date, fallback: str) -> str:
    """
    يختار قيمة يومية (شروق/غروب) بالاعتماد على تواريخ daily.time وليس على موقع الفهرس.
    (الفرونتند يرسل نطاقاً يبدأ قبل اليوم بيومين، لذا day_idx وحده يعطي يوماً خاطئاً.)
    """
    arr = daily.get(key) or []
    if not arr: return fallback
    times = [t[:10] if isinstance(t, str) else "" for t in (daily.get("time") or [])]
    target_str = target_date.isoformat()
    if target_str in times:
        i = times.index(target_str)
        if i < len(arr) and arr[i]:
            return arr[i]
    today_str = today.isoformat()
    if today_str in times:
        i = times.index(today_str) + (target_date - today).days
        if 0 <= i < len(arr) and arr[i]:
            return arr[i]
    i = min(len(arr) - 1, max(0, (target_date - today).days))
    return arr[i] if arr[i] else fallback

def _julian_day(d: date) -> float:
    y, m, day = d.year, d.month, d.day
    if m < 3: y -= 1; m += 12
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
    except (ValueError, IndexError): return 6.0

def resolve_timezone(tz_name: Optional[str]) -> zoneinfo.ZoneInfo:
    """يرجع كائن المنطقة الزمنية، ويعود للمنطقة الافتراضية إن كان الاسم غير معروف."""
    fallback = settings.DEFAULT_TZ
    for candidate in (tz_name, fallback, "UTC"):
        if not candidate:
            continue
        try:
            return zoneinfo.ZoneInfo(candidate)
        except (zoneinfo.ZoneInfoNotFoundError, ValueError, TypeError) as e:
            logger.warning(f"Unknown timezone '{candidate}' ({e}); trying next fallback.")
    return zoneinfo.ZoneInfo("UTC")

def safe_parse_iso(ts: str, tz: zoneinfo.ZoneInfo) -> Optional[datetime]:
    try:
        if ts.endswith("Z"): ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=tz)
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
        return {"status":"أيام الحياء","phase":"حمل البدر","days":day_in,"description":f"اليوم {day_in} في حمل البدر. البحر حايي، التيارات قوية، الصيد في ذروته من الشاطئ.","score_delta":15}
    elif age_days >= 28 or age_days <= 2:
        raw_day = age_days - 28 if age_days >= 28 else age_days
        day_in = max(1, min(3, int(raw_day) + 1))
        return {"status":"أيام الحياء","phase":"حمل المحاق","days":day_in,"description":f"اليوم {day_in} في حمل المحاق. البحر حايي، التيارات قوية، الصيد في ذروته من الشاطئ.","score_delta":15}
    elif 7 <= age_days <= 9 or 21 <= age_days <= 23:
        phase_name = "التربيع الأول" if age_days <= 9 else "التربيع الثاني"
        day_in = int(age_days - 7) + 1 if age_days <= 9 else int(age_days - 21) + 1
        return {"status":"أيام المات","phase":phase_name,"days":day_in,"description":f"اليوم {day_in} في {phase_name}. البحر مْيِّت، الماء راكد، الصيد أصعب من الشاطئ.","score_delta":-8}
    else:
        return {"status":"أيام عادية","phase":"","days":0,"description":"لا توجد مؤشرات حيائية أو مات قوية. الصيد من الشاطئ ممكن.","score_delta":0}

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
    if is_close(hw1, sr_h, 1.5): golden_windows.append(f"ساعة ذهبية صباحية: تزامن العبور القمري الأول ({windows['HW1']}) مع الفجر ({sunrise_str}).")
    if is_close(hw2, ss_h, 1.5): golden_windows.append(f"ساعة ذهبية مسائية: تزامن العبور القمري الثاني ({windows['HW2']}) مع الغروب ({sunset_str}).")
    if is_close(lw1_minus2, sr_h, 1.5) or is_close(lw2_minus2, sr_h, 1.5): golden_windows.append("نافذة الجزر الممتازة: تزامن بداية جزر قوي مع الفجر.")
    if is_close(lw1_minus2, ss_h, 1.5) or is_close(lw2_minus2, ss_h, 1.5): golden_windows.append("نافذة الجزر الممتازة: تزامن بداية جزر قوي مع الغروب.")
    if not golden_windows:
        hw1_gap = circular_diff(hw1, sr_h)
        hw2_gap = circular_diff(hw2, ss_h)
        golden_windows.append(f"لا توجد ساعة ذهبية. العبور القمري الأول ({windows['HW1']}) يبعد {format_time_gap(hw1_gap)} عن الفجر. العبور القمري الثاني ({windows['HW2']}) يبعد {format_time_gap(hw2_gap)} عن الغروب.")
    golden_windows.insert(0, "⚠️ تنبيه: المد والجزر في تونس ضعيف جداً، لا تعتمد عليه كلياً في تخطيط الصيد. هذه الأوقات تعكس العبور القمري التقريبي فقط.")
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
    major1 = (12.0 + moon_phase * 24.0 + lon_correction) % 24
    major2 = (major1 + 12.42) % 24
    minor1 = (major1 + 6.21) % 24
    minor2 = (major2 + 6.21) % 24
    return {"major1": format_time(major1), "major2": format_time(major2), "minor1": format_time(minor1), "minor2": format_time(minor2)}

def align_hourly_data(marine_hourly, weather_hourly, tz_name):
    tz = resolve_timezone(tz_name)
    m_times = marine_hourly.get("time", [])
    w_times = weather_hourly.get("time", [])
    if not m_times or not w_times: return [], {}
    m_map, w_map = {}, {}
    for i, t in enumerate(m_times):
        dt = safe_parse_iso(t, tz)
        if dt is not None: m_map[dt.replace(minute=0, second=0, microsecond=0)] = i
    for i, t in enumerate(w_times):
        dt = safe_parse_iso(t, tz)
        if dt is not None: w_map[dt.replace(minute=0, second=0, microsecond=0)] = i
    common = sorted(set(m_map) & set(w_map))
    if not common:
        logger.warning(f"No common timestamps! Marine times: {m_times[:3]}, Weather times: {w_times[:3]}")
        return [], {}
    def extract(key, src, idx_map, default=None):
        arr = src.get(key, [])
        result = []
        for t in common:
            if t in idx_map and arr and idx_map[t] < len(arr) and arr[idx_map[t]] is not None:
                result.append(arr[idx_map[t]])
            else:
                result.append(default)
        return result
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
        "relative_humidity_2m": extract("relative_humidity_2m", weather_hourly, w_map),
        "precipitation": extract("precipitation", weather_hourly, w_map),
        "visibility": extract("visibility", weather_hourly, w_map),
        "weather_code": extract("weather_code", weather_hourly, w_map)
    }

# ---------- Beaches ----------
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
    min_dist = float('inf'); nearest = None
    for b in TUNISIAN_BEACHES:
        dist = calc_distance(b["lat"], b["lon"], lat, lon)
        if dist < min_dist and dist < max_dist:
            min_dist = dist
            nearest = {"orientation": b["orientation"], "type": b["type"], "distance": round(dist, 0)}
    return nearest

# ---------- Overpass orientation (improved for bays) ----------
async def _overpass_orientation_inner(lat, lon):
    async with httpx.AsyncClient(timeout=8.0, headers={"User-Agent": USER_AGENT}) as client:
        for radius in [3000, 5000, 10000]:
            query = f"""[out:json];(way(around:{radius},{lat},{lon})["natural"="coastline"];);out geom;"""
            for attempt_idx, server in enumerate(OVERPASS_SERVERS):
                try:
                    if attempt_idx > 0: await asyncio.sleep(1.0)
                    r = await client.get(server, params={"data": query})
                    r.raise_for_status()
                    els = r.json().get("elements", [])
                    if not els: continue
                    best_dist, best_tangent, best_point = float('inf'), None, None
                    all_perpendiculars = []
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
                        if len(geom) >= 3 and abs(closest_idx - (len(geom)//2)) < 5:
                            for offset in [-2, -1, 0, 1, 2]:
                                idx = max(0, min(len(geom)-1, closest_idx + offset))
                                if idx > 0 and idx < len(geom)-1:
                                    tangent = calc_bearing(geom[idx-1]["lat"], geom[idx-1]["lon"], geom[idx+1]["lat"], geom[idx+1]["lon"])
                                    perp = (tangent + 90) % 360
                                    all_perpendiculars.append(perp)
                    if not best_tangent or not best_point: continue
                    perp1 = (best_tangent + 90) % 360
                    perp2 = (best_tangent - 90) % 360
                    to_user = calc_bearing(best_point["lat"], best_point["lon"], lat, lon)
                    diff1 = angle_diff(perp1, to_user)
                    diff2 = angle_diff(perp2, to_user)
                    seaward = perp1 if diff1 > diff2 else perp2
                    if all_perpendiculars:
                        all_perpendiculars.append(seaward)
                        seaward = circular_mean(all_perpendiculars) or seaward
                    return int(round(seaward))
                except Exception as e:
                    logger.warning(f"Overpass failed ({server}, radius={radius}): {e}")
                    continue
    return None

async def get_auto_orientation_overpass(lat, lon):
    """اتجاه الشاطئ من Overpass مع كاش (24 ساعة) وتذكّر الفشل (60 ثانية)."""
    key = ("orientation", round(lat, 3), round(lon, 3))
    hit, value = await overpass_cache.get(key)
    if hit:
        return value
    failed, _ = await failure_cache.get(key)
    if failed:
        logger.info(f"Skipping Overpass orientation (recent failure) for {key}")
        return None
    try:
        value = await asyncio.wait_for(_overpass_orientation_inner(lat, lon), timeout=settings.OVERPASS_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning(f"Overpass orientation global timeout ({settings.OVERPASS_TIMEOUT_S}s)")
        value = None
    if value is None:
        await failure_cache.set(key, True)
        return None
    await overpass_cache.set(key, value)
    return value

@app.post("/auto-orientation")
@limiter.limit(settings.RATE_LIMIT_ORIENTATION)
async def auto_orientation(request: Request, req: AutoOrientationRequest):
    orientation = await get_auto_orientation_overpass(req.latitude, req.longitude)
    if orientation is None:
        return {"orientation": -1, "source": "none", "message": "تعذر تحديد اتجاه الشاطئ من الخريطة. يرجى المحاولة لاحقاً أو إدخال الاتجاه يدوياً."}
    return {"orientation": orientation, "source": "overpass"}

async def _detect_bottom_type_uncached(lat: float, lon: float) -> dict:
    info = find_nearest_beach_info(lat, lon)
    if info: return {"bottom_type": info["type"], "source": "nearby_beach", "confidence": "medium"}
    query = f"""[out:json];(way(around:2000,{lat},{lon})["natural"="sand"];way(around:2000,{lat},{lon})["natural"="shingle"];way(around:2000,{lat},{lon})["natural"="bare_rock"];);out body;"""
    async with httpx.AsyncClient(timeout=15, headers={"User-Agent": USER_AGENT}) as client:
        for server in OVERPASS_SERVERS[:2]:
            try:
                r = await client.get(server, params={"data": query})
                r.raise_for_status()
                elements = r.json().get("elements", [])
                if elements:
                    nat = elements[0].get("tags", {}).get("natural", "")
                    if nat == "sand": return {"bottom_type": "sandy", "source": "overpass", "confidence": "high"}
                    if nat in ["shingle", "bare_rock"]: return {"bottom_type": "rocky", "source": "overpass", "confidence": "high"}
                break
            except Exception as e:
                logger.warning(f"Overpass bottom detection failed ({server}): {e}")
                continue
    return {"bottom_type": "unknown", "source": "none", "confidence": "low"}

async def get_bottom_type_cached(lat: float, lon: float) -> dict:
    """نوع القاع مع كاش طويل، وتذكّر قصير للحالات التي تفشل (unknown)."""
    key = ("bottom", round(lat, 3), round(lon, 3))
    hit, value = await overpass_cache.get(key)
    if hit:
        return value
    failed, _ = await failure_cache.get(key)
    if failed:
        return {"bottom_type": "unknown", "source": "cached_failure", "confidence": "low"}
    value = await _detect_bottom_type_uncached(lat, lon)
    if value.get("bottom_type") == "unknown":
        await failure_cache.set(key, True)
        return value
    await overpass_cache.set(key, value)
    return value

@app.post("/detect-bottom-type")
@limiter.limit(settings.RATE_LIMIT_BOTTOM)
async def detect_bottom_type(request: Request, req: DetectBottomRequest):
    return await get_bottom_type_cached(req.latitude, req.longitude)

# ---------- Fetch data ----------
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_HOURLY_VARS = "wave_height,wave_period,wave_direction,swell_wave_height,swell_wave_period,swell_wave_direction,sea_surface_temperature"
WEATHER_HOURLY_VARS = "wind_speed_10m,wind_direction_10m,wind_gusts_10m,pressure_msl,temperature_2m,relative_humidity_2m,precipitation,visibility,weather_code"
# نحتاج 48 ساعة ماضية (ذاكرة البحر) + يوم الهدف + يوم بعده → past_days=2 و forecast_days=4
PAST_DAYS = 2
FORECAST_DAYS = 4

async def fetch_marine_data_from_openmeteo(client: httpx.AsyncClient, lat: float, lon: float):
    params = {"latitude": lat, "longitude": lon, "hourly": MARINE_HOURLY_VARS,
              "timezone": settings.DEFAULT_TZ, "past_days": PAST_DAYS, "forecast_days": FORECAST_DAYS}
    try:
        r = await client.get(MARINE_URL, params=params, timeout=settings.UPSTREAM_TIMEOUT_S)
        r.raise_for_status(); return r.json()
    except Exception as e:
        logger.error(f"Marine fetch failed: {e}"); return None

async def fetch_weather_data_from_openmeteo(client: httpx.AsyncClient, lat: float, lon: float):
    params = {"latitude": lat, "longitude": lon, "hourly": WEATHER_HOURLY_VARS, "daily": "sunrise,sunset",
              "timezone": settings.DEFAULT_TZ, "past_days": PAST_DAYS, "forecast_days": FORECAST_DAYS}
    try:
        r = await client.get(WEATHER_URL, params=params, timeout=settings.UPSTREAM_TIMEOUT_S)
        r.raise_for_status(); return r.json()
    except Exception as e:
        logger.error(f"Weather fetch failed: {e}"); return None

def _upstream_key(kind: str, lat: float, lon: float):
    """مفتاح كاش للإحداثيات (تقريب ~1 كم لأن بيانات النماذج شبكية أصلاً)."""
    return (kind, round(lat, 2), round(lon, 2))

async def _fetch_upstream_cached(kind: str, client: httpx.AsyncClient, lat: float, lon: float, fn) -> Optional[dict]:
    """جلب مع كاش + تذكّر الفشل + رحلة واحدة للطلبات المتزامنة."""
    key = _upstream_key(kind, lat, lon)
    failed, _ = await failure_cache.get(key)
    if failed:
        logger.warning(f"Upstream {kind} marked as recently failed for {key[1:]}; serving degraded response.")
        return None
    value, _from_cache = await upstream_cache.get_or_set(key, lambda: fn(client, lat, lon))
    if not value:
        await failure_cache.set(key, True)
        return None
    return value

async def fetch_marine_data_cached(client: httpx.AsyncClient, lat: float, lon: float):
    return await _fetch_upstream_cached("marine", client, lat, lon, fetch_marine_data_from_openmeteo)

async def fetch_weather_data_cached(client: httpx.AsyncClient, lat: float, lon: float):
    return await _fetch_upstream_cached("weather", client, lat, lon, fetch_weather_data_from_openmeteo)

# ---------- Tactical Helpers ----------
def get_water_clarity(wind_speed, wave_height, is_murky, is_weedy, haml_status, avg_vis_b=10000):
    if avg_vis_b < 200: return "ضباب كثيف (رؤية معدومة)"
    if is_weedy: return "عكر جداً (أعشاب وصوفة)"
    if is_murky: return "عكر (بحر خامر)"
    if "الحياء" in haml_status and (wind_speed > 15 or wave_height > 0.5): return "عكر/مخلوط (التيارات تقلب القاع)"
    if wind_speed > 20: return "متوسط العكارة (رياح قوية)"
    if wave_height < 0.3 and wind_speed < 10: return "صافي جداً"
    return "صافي"

def suggest_rig(haml_status: str, is_lateral_strong: bool, wind_speed: float, is_mirror_sea: bool, beach_type: str = "sandy") -> str:
    strong_current = "الحياء" in haml_status or is_lateral_strong
    if strong_current and wind_speed > 20:
        return "مونتاج باتير نوستر قصير (فروع 50سم، ثقيل) – يمنع التشابك في التيار"
    if is_mirror_sea or (not is_lateral_strong and wind_speed < 10):
        if beach_type == "sandy": return "مونتاج بسنود طويل (فرع سفلي 150سم، خفيف) – حركة طبيعية للطعم"
        else: return "مونتاج كليب داون خفيف – لتجنب التشابك في القاع الصخري"
    if wind_speed > 20 and not strong_current:
        return "مونتاج عادي برصاص ثقيل نسبياً – لمواجهة الرياح"
    return "مونتاج عادي (فرع 80-100سم) – مرن للظروف المتوسطة"

def casting_angle_correction(wind_dir, orient):
    if wind_dir is None or orient is None: return 0
    diff = signed_angle_diff(wind_dir, orient)
    if abs(diff) < 20: return 0
    raw = round(-diff * 0.25)
    return max(-30, min(30, raw))

def calculate_comfort_index(temp, wind_speed, humidity=None):
    if temp is None: return 50
    humidity = max(0.0, min(100.0, float(humidity))) if humidity is not None else 50.0
    temp = max(-10.0, min(50.0, float(temp)))
    if temp < 10:
        wind_chill = 13.12 + 0.6215 * temp - 11.37 * (wind_speed ** 0.16) + 0.3965 * temp * (wind_speed ** 0.16)
        apparent = wind_chill
    else:
        apparent = temp + 0.33 * (humidity / 100 * 6.105 * math.exp(17.27 * temp / (237.7 + temp))) - 4.0
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

# ---------- Analysis Helpers ----------
def analyze_weed_risk(sea_memory):
    risk = "منخفض"; advice = ""
    has_weed = "صوفة" in sea_memory or "أعشاب" in sea_memory
    if has_weed:
        risk = "مرتفع" if "تحذير صوفة" in sea_memory else "متوسط"
        advice = "استعمل صائدات مضادة للأعشاب وارمِ بزاوية حادة نحو البحر لتجنب الالتفاف."
    return {"risk": risk, "advice": advice}

def analyze_backwash(wind_speed: float, wind_dir, orient: float, wave_height: float) -> dict:
    if wind_dir is None: return {"severity": "منخفض", "effect": ""}
    wind_speed = safe_float(wind_speed)
    wave_height = safe_float(wave_height)
    wind_diff = angle_diff(wind_dir, orient)
    is_onshore = wind_diff < 30
    severity = "منخفض"; effect = ""
    if is_onshore and wind_speed > 30 and wave_height > 0.8:
        severity = "مرتفع"
        effect = f"رياح بحرية قوية ({wind_speed:.0f} كم/س) تضرب الموج نحو الشاطئ، ثم يرتد الموج بقوة نحو البحر. هذا يخلق تياراً عكسياً قوياً يسحب الرصاصة للشاطئ باستمرار وقد يدفن الخيط."
    elif is_onshore and wind_speed > 15:
        severity = "متوسط"
        effect = f"رياح بحرية ({wind_speed:.0f} كم/س) تخلق تياراً عكسياً خفيفاً. الرصاصة قد تتحرك قليلاً نحو الشاطئ لكن يمكن التحكم بها بوزن أثقل."
    return {"severity": severity, "effect": effect}

def analyze_debris_risk(sea_memory: str, wind_speed: float) -> dict:
    wind_speed = safe_float(wind_speed)
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
    if is_mirror_sea: base -= 25
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
    press_change = extra.get("pressure_change", 0)
    if press_change > 6.0: score -= 20
    elif press_change > 3.0: score -= 10
    elif press_change < -6.0: score += 15
    elif press_change < -3.0: score += 5
    elif abs(press_change) <= 2.0: score += 5

    if flags["is_lateral_strong"]: score -= 10
    for b in blocks:
        wp_val = b["_raw"]["wave_period"]
        if 0.1 < wp_val < 4.0:
            score -= 10
            break
    has_strong_offshore = any(b["wind_dir"].startswith("برية") and b["_raw"]["avg_wind"] > 20 for b in blocks)
    has_light_offshore = any(b["wind_dir"].startswith("برية") and b["_raw"]["avg_wind"] <= 15 for b in blocks)
    if has_strong_offshore: score -= 5
    elif has_light_offshore: score += 5
    if any(b["wind_dir"].startswith("بحرية") and b["_raw"]["avg_wind"] < 15 for b in blocks): score += 15
    if any(0.6 <= b["_raw"]["avg_wave_h"] <= 1.2 for b in blocks): score += 20
    if any(b.get("wave_angle_diff") is not None and b["wave_angle_diff"] < 30 and b["_raw"]["avg_wave_h"] >= 0.6 for b in blocks): score += 15
    if flags["has_golden_window"]: score += 25
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
    return score

# ---------- Fish Activity ----------
def get_period_fish_status(avg_sst, is_night, is_murky, is_weedy, is_mirror_sea, lateral_force_ratio, water_clarity, seabass_sst_limit):
    if avg_sst is None: avg_sst = 20.0
    active, inactive = [], []
    seabass_likes_current = 0.15 < lateral_force_ratio < 0.45
    seabass_active = (is_night or is_murky) and avg_sst < seabass_sst_limit and not is_mirror_sea and seabass_likes_current
    if seabass_active: active.append("قاروص")
    else: inactive.append("قاروص")
    if avg_sst > 18 and not is_mirror_sea: active.append("دنيس")
    else: inactive.append("دنيس")
    if not is_murky and not is_weedy and not is_mirror_sea: active.append("بوري")
    else: inactive.append("بوري")
    if avg_sst <= 24: active.append("سارغ")
    else: inactive.append("سارغ")
    if avg_sst > 18 and lateral_force_ratio > 0.15: active.append("مرمار")
    else: inactive.append("مرمار")
    if avg_sst > 17 and not is_mirror_sea: active.append("شلبة")
    else: inactive.append("شلبة")
    if avg_sst > 18 and not is_murky: active.append("تريلية")
    else: inactive.append("تريلية")
    if avg_sst > 19 and is_murky: active.append("بغبغان")
    else: inactive.append("بغبغان")
    if avg_sst > 18 and (is_night or "عكر" in water_clarity): active.append("سوبيا")
    else: inactive.append("سوبيا")
    return active, inactive

# ---------- Core Aggregation ----------
def aggregate_physics(all_times, aligned, orient, target_date_obj, sunrise, sunset, latitude, longitude, beach_type="sandy"):
    tz = all_times[0].tzinfo if all_times else zoneinfo.ZoneInfo("UTC")
    target_start = datetime.combine(target_date_obj, datetime.min.time(), tzinfo=tz)
    target_end = target_start + timedelta(days=1)
    past_start = target_start - timedelta(hours=48)
    past_idx = [i for i, t in enumerate(all_times) if past_start <= t < target_start]
    target_idx = [i for i, t in enumerate(all_times) if target_start <= t < target_end]
    warnings = []
    empty_res = {"sea_memory":"غير معروف","lateral_current":"غير معروف","pressure_state":"مستقر","tide_analysis":{},"sst_stability":"مستقر","avg_sst":None,"hidden_factors":{},"blocks":[],"red_flags":[],"green_flags":[],"extra_info":{},"transitions":[],"flags":{},"nogo_reasons":[],"warnings":["لا توجد بيانات ساعية ليوم الهدف."],"final_verdict":"بيانات غير كافية","score":0}
    if not target_idx: return empty_res

    def pick(k, default=None):
        arr = aligned.get(k, [])
        return [(arr[i] if arr[i] is not None else default) if i < len(arr) else default for i in target_idx]

    wh = pick("wave_height"); wp = pick("wave_period"); swh = pick("swell_wave_height")
    swp = pick("swell_wave_period"); swd = pick("swell_wave_direction", None); wd_wave = pick("wave_direction", None)
    sst = pick("sea_surface_temperature", None); ws = pick("wind_speed_10m"); wd = pick("wind_direction_10m", None)
    wg = pick("wind_gusts_10m"); pr = pick("pressure_msl", None); ta = pick("temperature_2m", None)
    rh = pick("relative_humidity_2m", None); prec = pick("precipitation"); vis = pick("visibility", None)
    wcode = [int(v) if isinstance(v, (int, float)) else 0 for v in pick("weather_code")]

    def safe_mean(arr, default=0.0):
        vals = [safe_float(v) for v in arr if v is not None]
        return sum(vals)/len(vals) if vals else default

    wave_power = [0.49*(safe_float(h)**2)*safe_float(p) for h,p in zip(wh,wp)]
    wind_cls = [wind_class_detailed(angle_diff(d, orient)) if d is not None else "غير معروف" for d in wd]
    has_swell_data = len(swh) > 0 and not all(v == 0.0 for v in swh)
    actual_swell_exists = has_swell_data and max(swh) > 0.05

    sea_memory = "بحر صافي وهادئ"
    accumulated_rain_48h = 0.0; max_rain_hourly = 0.0
    if past_idx:
        p_wh = aligned.get("wave_height", []); p_wp = aligned.get("wave_period", [])
        p_swh = aligned.get("swell_wave_height", []); p_ws = aligned.get("wind_speed_10m", [])
        p_wd = aligned.get("wind_direction_10m", []); p_prec = aligned.get("precipitation", [])
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
            if accumulated_rain_48h > 50.0 or max_rain_hourly > 20.0: sea_memory += " | سيول."

    # lateral current with corrected physics (short waves penalised)
    day_lateral_fx, day_lateral_fy = 0.0, 0.0
    for i in range(len(wh)):
        if wd_wave[i] is not None:
            w_dir = wd_wave[i]
            signed_angle = math.radians(signed_angle_diff(w_dir, orient))
            wave_h = safe_float(wh[i])
            wave_p = safe_float(wp[i])
            force = (wave_h ** 2) * (1.0 / max(0.5, wave_p)) * 5.0
            day_lateral_fx += force * math.sin(signed_angle)
            day_lateral_fy += force * math.cos(signed_angle)
    day_total = math.sqrt(day_lateral_fx**2 + day_lateral_fy**2)
    day_lateral_ratio = abs(day_lateral_fx) / day_total if day_total > 1e-9 else 0
    avg_wave_h_day = safe_mean(wh)

    tide_analysis = get_moon_and_tide_analysis(target_date_obj)
    tidal_windows, golden_windows = estimate_tidal_windows(target_date_obj, tide_analysis, sunrise, sunset, latitude, longitude)
    solunar = calculate_solunar(target_date_obj, latitude, longitude)
    moon_age = get_moon_age_days(target_date_obj)
    haml_info = get_haml_mat_status(moon_age)
    platform_advice = get_fishing_platform_advice(haml_info["status"])

    is_neap_tide = tide_analysis["idx"] in [2, 6]; is_spring_tide = tide_analysis["idx"] in [0, 4]
    if is_neap_tide: warnings.append(f"مد ضعيف (مد محاقي - {tide_analysis['name']}): تيارات غذائية ضعيفة.")
    has_golden_window = any("تزامن" in g for g in golden_windows)
    if not has_golden_window: warnings.append("لا توجد ساعة ذهبية: قد يقل نشاط الأسماك.")

    valid_sst = [v for v in sst if v is not None and 8.0 <= v <= 35.0]
    avg_sst = sum(valid_sst) / len(valid_sst) if valid_sst else None
    if avg_sst is not None and avg_sst > 30: warnings.append(f"حرارة ماء مرتفعة جداً ({avg_sst:.1f}°م): قد تكون البيانات خاطئة.")
    sst_diff = max(valid_sst) - min(valid_sst) if len(valid_sst) > 1 else 0
    sst_stability = "صدمة حرارية" if sst_diff > 2.0 else "تغير بطيء" if sst_diff > 1.0 else "مستقر تماماً"
    is_murky = "عكر" in sea_memory or "خامر" in sea_memory
    is_weedy = "صوفة" in sea_memory
    valid_air_temps = [t for t in ta if t is not None]
    max_air_temp = max(valid_air_temps) if valid_air_temps else 20.0
    seabass_sst_limit = 22.0 if target_date_obj.month in [6,7,8,9] else 20.0
    if avg_sst is not None and avg_sst > seabass_sst_limit: warnings.append(f"حرارة ماء عالية ({avg_sst:.1f}°م): تتجاوز حد القاروص.")
    if is_weedy: warnings.append("صوفة محتملة.")

    avg_press = safe_mean(pr, 1013.0)
    valid_pr = [safe_float(v) for v in pr if v is not None]
    press_change_day = valid_pr[-1] - valid_pr[0] if len(valid_pr) >= 2 else 0.0
    if press_change_day > 6.0: pressure_note = "يرتفع بشدة (سلبي جداً)"
    elif press_change_day > 3.0: pressure_note = "يرتفع خلال اليوم (سلبي)"
    elif press_change_day < -6.0: pressure_note = "ينخفض بشدة (إيجابي جداً: تحفيز التغذية)"
    elif press_change_day < -3.0: pressure_note = "ينخفض خلال اليوم (إيجابي)"
    else: pressure_note = "مستقر (محايد)"
    pressure_state = f"{pressure_note} (تغير يومي: {press_change_day:+.1f} hPa)"

    peak_gust_day = max(safe_float(v) for v in wg if v is not None) if any(v is not None for v in wg) else 0.0
    dominant = max(set(wind_cls), key=wind_cls.count) if wind_cls else "غير معروف"

    periods = defaultdict(list)
    for idx, i in enumerate(target_idx):
        h = all_times[i].hour
        if 0 <= h <= 3: periods["late_night"].append(idx)
        elif 4 <= h <= 11: periods["morning"].append(idx)
        elif 12 <= h <= 17: periods["afternoon"].append(idx)
        else: periods["evening"].append(idx)

    def parse_tidal_time(t_str: str) -> float:
        try:
            parts = t_str.split(":")
            return int(parts[0]) + int(parts[1]) / 60.0 if len(parts) >= 2 else 0.0
        except: return 0.0

    slack_info = ""
    for hw_key, lw_key in [("HW1","LW1"), ("HW2","LW2")]:
        hw_t = parse_tidal_time(tidal_windows[hw_key])
        lw_t = parse_tidal_time(tidal_windows[lw_key])
        slack_info += f"العبور القمري {tidal_windows[hw_key]} مياه ميتة: {format_time(hw_t-0.75)}-{format_time(hw_t+0.75)}; الجزر المحاقي {tidal_windows[lw_key]} مياه ميتة: {format_time(lw_t-0.75)}-{format_time(lw_t+0.75)}; "
    slack_info = slack_info.rstrip("; ")

    steepness_vals = [safe_float(h) / (1.56 * safe_float(p)**2) for h, p in zip(wh, wp) if safe_float(p) > 0.1 and h is not None]
    avg_steepness = sum(steepness_vals) / len(steepness_vals) if steepness_vals else 0
    steepness_desc = ("موج حاد وقصير" if avg_steepness > 0.06 else "موج منخفض الانحدار" if avg_steepness < 0.03 else "موج متوسط الانحدار")

    cross_angles = []
    for i in range(min(len(swd), len(wd_wave))):
        if swd[i] is not None and wd_wave[i] is not None and swd[i] != 0.0 and wd_wave[i] != 0.0:
            cross_angles.append(angle_diff(swd[i], wd_wave[i]))
    is_cross_sea_dangerous = bool(cross_angles and max(cross_angles) > 60 and sum(cross_angles)/len(cross_angles) > 40)
    cross_sea_risk = "بحر مختلط وخطير" if is_cross_sea_dangerous else "منخفض"

    blocks = []
    populated_periods = 0; periods_with_lethal = 0; mirror_with_gusts = []; mirror_without_gusts = []
    period_names_arabic = {"late_night":"السحر","morning":"الصباح","afternoon":"الظهيرة","evening":"الغسق"}

    for key in ["late_night", "morning", "afternoon", "evening"]:
        idxs = periods[key]
        if not idxs: continue
        populated_periods += 1

        sub_wh = [safe_float(wh[i]) for i in idxs if wh[i] is not None]
        sub_wp = [safe_float(wp[i]) for i in idxs if wp[i] is not None]
        sub_ws = [safe_float(ws[i]) for i in idxs if ws[i] is not None]
        sub_wd = [wd[i] for i in idxs if wd[i] is not None]
        sub_wg = [safe_float(wg[i]) for i in idxs if wg[i] is not None]
        sub_pr = [safe_float(pr[i]) for i in idxs if pr[i] is not None]
        sub_ta = [safe_float(ta[i]) for i in idxs if ta[i] is not None]
        sub_rh = [safe_float(rh[i]) for i in idxs if rh[i] is not None]
        sub_vis = [safe_float(vis[i]) for i in idxs if vis[i] is not None]
        sub_wcode = [wcode[i] for i in idxs]
        sub_swh = [safe_float(swh[i]) for i in idxs if swh[i] is not None]
        sub_swp = [safe_float(swp[i]) for i in idxs if swp[i] is not None]
        sub_swd = [swd[i] for i in idxs if swd[i] is not None]
        sub_wave_dir = [wd_wave[i] for i in idxs if wd_wave[i] is not None]
        sub_sst = [sst[i] for i in idxs if sst[i] is not None and 8.0 <= sst[i] <= 35.0]

        avg_h = sum(sub_wh)/len(sub_wh) if sub_wh else 0.0
        max_h = max(sub_wh) if sub_wh else 0.0
        avg_w = sum(sub_ws)/len(sub_ws) if sub_ws else 0.0
        max_w = max(sub_ws) if sub_ws else 0.0
        wc_dom = max(set([wind_class_detailed(angle_diff(d, orient)) for d in sub_wd]), key=lambda c: [wind_class_detailed(angle_diff(d, orient)) for d in sub_wd].count(c)) if sub_wd else "غير معروف"
        most_code = max(set(sub_wcode), key=sub_wcode.count) if sub_wcode else 0
        avg_swh_b = sum(sub_swh)/len(sub_swh) if sub_swh else 0.0
        avg_swp_b = sum(sub_swp)/len(sub_swp) if sub_swp else 0.0
        avg_swd_b = circular_mean(sub_swd)
        avg_wave_dir = circular_mean(sub_wave_dir)
        avg_wd_b = circular_mean(sub_wd)
        avg_air = sum(sub_ta)/len(sub_ta) if sub_ta else 20.0
        avg_rh = sum(sub_rh)/len(sub_rh) if sub_rh else 65.0
        max_gust_b = max(sub_wg) if sub_wg else 0.0
        avg_press_b = sum(sub_pr)/len(sub_pr) if sub_pr else 1013.0
        avg_vis_b = sum(sub_vis)/len(sub_vis) if sub_vis else 10000.0
        avg_wp_b = sum(p for p in sub_wp if p > 0.5)/max(1, sum(1 for p in sub_wp if p > 0.5))
        press_rate = (sub_pr[-1] - sub_pr[0]) * (3.0 / max(1, len(sub_pr))) if len(sub_pr)>=2 else 0.0

        # period lateral force with corrected physics
        period_lateral_fx, period_lateral_fy = 0.0, 0.0
        for i in idxs:
            if wd_wave[i] is not None and safe_float(wh[i]) > 0.1 and safe_float(wp[i]) > 0.1:
                w_dir = wd_wave[i]; signed_angle = math.radians(signed_angle_diff(w_dir, orient))
                wave_h = safe_float(wh[i]); wave_p = safe_float(wp[i])
                force = (wave_h ** 2) * (1.0 / max(0.5, wave_p)) * 5.0
                period_lateral_fx += force * math.sin(signed_angle)
                period_lateral_fy += force * math.cos(signed_angle)
        period_total_force = math.sqrt(period_lateral_fx**2 + period_lateral_fy**2)
        period_lateral_force_ratio = abs(period_lateral_fx) / period_total_force if period_total_force > 1e-9 else 0.0
        period_is_mirror_sea = max_h < 0.3
        period_is_lateral_strong = period_lateral_force_ratio > 0.7 and avg_h > 0.6

        # sea state with steepness (corrected for all heights)
        period_steepness_vals = []
        for i in idxs:
            p = safe_float(wp[i])
            h = safe_float(wh[i])
            if p > 0.5 and h is not None and h > 0:
                period_steepness_vals.append(h / (1.56 * p**2))
        period_steepness = sum(period_steepness_vals)/len(period_steepness_vals) if period_steepness_vals else 0.0

        if period_is_mirror_sea: sea = "بحر مرآوي"
        elif max_h < 0.6: sea = "هادئ"
        elif max_h < 0.9: sea = "متموج خفيف"
        elif max_h < 1.3: sea = "متوسط الهيجان" if period_steepness > 0.06 else "متموج بقوة"
        elif max_h <= 2.0: sea = "بحر غسالة خطير" if period_steepness > 0.08 else "هائج خفيف"
        else: sea = "هائج"

        if period_is_mirror_sea:
            if max_gust_b >= 15: mirror_with_gusts.append(key)
            else: mirror_without_gusts.append(key)

        period_nogo = []
        if max_h > 2.0: period_nogo.append(f"بحر هائج (أمواج > 2.0م): الرمي مستحيل والخطر كبير.")
        if avg_wp_b > 12.0 and avg_h > 0.8: period_nogo.append(f"أمواج أرضية عالية الطاقة")
        if most_code in [95, 96, 99]: period_nogo.append("خطر الصواعق والبرق")
        if wc_dom.startswith("بحرية") and max_gust_b > 55: period_nogo.append("رياح بحرية عاتية")
        elif wc_dom.startswith("برية") and max_gust_b > 65: period_nogo.append("رياح برية عاتية جداً")
        elif max_gust_b > 70: period_nogo.append("رياح عاتية جداً")
        if accumulated_rain_48h > 50.0 or max_rain_hourly > 20.0: period_nogo.append("عكارة طينية شديدة (سيول)")
        if avg_vis_b < 200: period_nogo.append("ضباب كثيف جداً")
        if period_is_lateral_strong and avg_h > 1.0: period_nogo.append("تيار جانبي عنيف: يجرف الرصاصة فوراً ولا يمكن تثبيت الطعم.")
        if period_steepness > 0.08 and avg_h > 0.6:
            period_nogo.append("⛔ بحر غسالة عنيف (موج قصير جداً ومتلاطم): الرصاصة لا تثبت والخيط يتشابك باستمرار. الصيد شبه مستحيل.")

        backwash = analyze_backwash(avg_w, avg_wd_b, orient, avg_h)
        if backwash["severity"] == "مرتفع": period_nogo.append(f"تيار راجع عنيف")
        debris = analyze_debris_risk(sea_memory, avg_w)
        if debris["risk"] == "مرتفع": period_nogo.append("أوساخ وصوفة كثيفة")

        period_warnings = []
        if period_is_mirror_sea: period_warnings.append("بحر مرآوي: قد تخاف الأسماك من ظلال القصبة والخيط نهاراً، لكنه ممتاز ليلاً.")
        if is_weedy: period_warnings.append("صوفة محتملة في هذه الفترة.")
        if abs(press_rate) > 4.0: period_warnings.append(f"اضطراب ضغط ({press_rate:+.1f} hPa/3h)")
        elif press_rate > 1.5: period_warnings.append(f"ضغط مرتفع ({press_rate:+.1f} hPa/3h)")
        elif press_rate < -2.0: period_warnings.append(f"ضغط منخفض ({press_rate:.1f} hPa/3h)")
        if period_steepness > 0.06 and avg_h > 0.4:
            if period_steepness > 0.08:
                period_warnings.append("⚠️ بحر غسالة عنيف (موج قصير جداً): استعمل رصاصاً ثقيلاً مفلطحاً ومونتاجاً قصيراً جداً (40-50سم).")
            else:
                period_warnings.append("⚠️ بحر غسالة (موج قصير): الرصاصة تقفز وقد تتشابك. استعمل رصاصاً ثقيلاً مفلطحاً (Grip Lead) ومونتاجاً قصيراً.")

        has_lethal_nogo = len(period_nogo) > 0
        if has_lethal_nogo: periods_with_lethal += 1

        has_swell_dir = actual_swell_exists and (avg_swd_b is not None)
        final_swd = avg_swd_b if has_swell_dir else None
        has_wave_dir = avg_wave_dir is not None; final_wd = avg_wave_dir if has_wave_dir else None
        swell_angle = angle_diff(final_swd, orient) if final_swd is not None else None
        wave_angle = angle_diff(final_wd, orient) if final_wd is not None else None

        if avg_wd_b is not None:
            wind_dir_rad = math.radians(avg_wd_b); orient_rad = math.radians(orient)
            frontal = math.cos(wind_dir_rad - orient_rad); lateral = abs(math.sin(wind_dir_rad - orient_rad))
            wind_effect_dist = avg_w * (-frontal * 0.4 + (0.2 if frontal < 0 else 0) * abs(frontal) - lateral * 0.1)
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
        rig = suggest_rig(haml_info["status"], period_is_lateral_strong, avg_w, period_is_mirror_sea, beach_type)
        comfort = calculate_comfort_index(avg_air, avg_w, avg_rh)
        avg_sst_period = sum(sub_sst)/len(sub_sst) if sub_sst else (avg_sst if avg_sst else 20.0)
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
                "wave_period": round(avg_wp_b, 1),
                "wind_dir_deg": round(avg_wd_b, 0) if avg_wd_b is not None else None,
                "wind_effect_dist": round(wind_effect_dist, 0), "recommended_cast_distance": round(recommended_dist, 0),
                "press_rate": round(press_rate, 2)
            }
        }
        blocks.append(block_data)

    if mirror_with_gusts:
        names = [period_names_arabic[p] for p in mirror_with_gusts]; warnings.append(f"بحر مرآوي مع تموجات خفيفة في: {', '.join(names)}.")
    if mirror_without_gusts:
        names = [period_names_arabic[p] for p in mirror_without_gusts]; warnings.append(f"بحر مرآوي تام في: {', '.join(names)}.")

    any_pressure_rising = any(b["_raw"].get("press_rate", 0) > 1.5 for b in blocks)
    any_pressure_dropping = any(b["_raw"].get("press_rate", 0) < -2.0 for b in blocks)

    all_nogo_reasons = [r for b in blocks for r in b["nogo_reasons"]]
    reds, greens = [], []
    for i in range(len(wh)):
        hh = all_times[target_idx[i]].strftime("%H:%M")
        wg_i = safe_float(wg[i]) if i < len(wg) else 0.0
        pr_i = safe_float(pr[i]) if i < len(pr) else 1013.0
        ws_i = safe_float(ws[i]) if i < len(ws) else 0.0
        if wave_power[i] > 3 or safe_float(wh[i]) > 1.8 or wg_i > 50 or pr_i < 1005: reds.append(hh)
        if 0.3 <= safe_float(wh[i]) <= 1 and 0.1 <= wave_power[i] <= 1.5 and ws_i < 27.8: greens.append(hh)

    extra = {
        "pressure_avg":round(avg_press,1), "peak_gust_today":round(peak_gust_day,1),
        "sunrise":sunrise, "sunset":sunset, "max_air_temp": round(max_air_temp, 1),
        "tidal_windows": tidal_windows, "golden_windows": golden_windows,
        "has_swell_data": has_swell_data, "actual_swell_exists": actual_swell_exists,
        "solunar": solunar, "slack_times": slack_info,
        "weed_risk": analyze_weed_risk(sea_memory), "seasonal_bait": get_seasonal_bait(target_date_obj.month, avg_sst if avg_sst else 20.0),
        "past_rain_accumulated_48h": round(accumulated_rain_48h, 1),
        "max_rain_hourly": round(max_rain_hourly, 1),
        "pressure_change": round(press_change_day, 1), "pressure_note": pressure_note,
        "moon_age_days": round(moon_age, 1),
        "haml_status": haml_info["status"], "haml_phase": haml_info["phase"],
        "haml_description": haml_info["description"], "haml_score_delta": haml_info["score_delta"],
        "platform_advice": platform_advice, "sst_stability": sst_stability,
        "beach_orientation": orient
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
        "nogo_reasons": all_nogo_reasons, "warnings": warnings,
        "target_month": target_date_obj.month
    }
    if not blocks:
        agg_result["warnings"].append("بيانات الأرصاد غير كافية أو مفقودة لإنشاء تحليل زمني.")
        agg_result["final_verdict"] = "بيانات غير كافية"
        agg_result["score"] = 0
        return agg_result

    score = apply_scoring(agg_result)
    agg_result["score"] = score

    lethal_blocks = sum(1 for b in blocks if b.get("has_lethal_nogo", False))
    total_blocks = len(blocks)
    if total_blocks == 0:
        final_verdict = "بيانات غير كافية"
    elif lethal_blocks == total_blocks:
        final_verdict = "غير مناسب"
    elif score < 15:
        final_verdict = "غير مناسب"
    elif score < 40:
        final_verdict = "فرصة مع تحفظات"
    elif lethal_blocks > 0:
        final_verdict = "فرصة مع تحفظات"
    elif warnings:
        final_verdict = "فرصة مع تحفظات"
    else:
        final_verdict = "مناسب"
    agg_result["final_verdict"] = final_verdict
    return agg_result

# ---------- Report builders ----------
def format_tidal_flow_periods(tidal_windows: dict) -> dict:
    periods = {}
    for key, time_str in tidal_windows.items():
        h = safe_parse_time(time_str)
        slack_start = format_time((h - 0.5) % 24); slack_end = format_time((h + 0.5) % 24)
        is_high = key.startswith("HW")
        early_end = slack_start
        if is_high: early_label = "آخر حركة المد الدافق (تقديري)"
        else: early_label = "آخر حركة الجزر الساحب (تقديري)"
        late_start = slack_end
        if is_high: late_label = "بداية حركة المد الدافق (تقديري)"
        else: late_label = "بداية حركة الجزر الساحب (تقديري)"
        name = "العبور القمري الأول (تقدير)" if is_high else "الجزر المحاقي (تقدير)"
        periods[key] = {"name": name, "time": time_str, "early_label": early_label, "early_time": early_end,
                       "slack": f"{slack_start} - {slack_end}", "late_label": late_label, "late_time": late_start}
    return periods

def build_flow_section(tidal_windows: dict) -> str:
    periods = format_tidal_flow_periods(tidal_windows)
    lines = ["🏃‍♂️ 2. فترات الحركة (الأوقات الخضراء)"]
    for key, data in periods.items():
        lines.append(f" * 🌊 {data['name']} ({data['time']}): 🟢 {data['late_label']} من {data['late_time']}")
    lines.append("↳ نصيحة: أفضل صيد في بداية المد الدافق أو بداية الجزر الساحب. تجنب مركز المياه الميتة. (تذكر: المد والجزر في تونس ضعيف جداً)")
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
    interactions.append(f"[فترات سولونار] 🎯 الرئيسية: {solunar.get('major1')} | {solunar.get('major2')} 🎯 الثانوية: {solunar.get('minor1')} | {solunar.get('minor2')}")
    for b in blocks:
        name = b['name']; time_range = b['time_range']; raw = b.get("_raw", {})
        interactions.append(f"[{name} ({time_range})]")
        interactions.append(f"  📊 حالة البحر والثقة: {b['sea_state']} | نسبة الثقة: {b.get('confidence',0)}% ({b.get('confidence_label','')})")
        wind_eff = raw.get('wind_effect_dist', 0); sign = '+' if wind_eff > 0 else ''
        interactions.append(f"  💨 الرياح والرمي: {b['wind_dir']} {raw.get('avg_wind',0):.1f} كم/س (هبات {raw.get('max_gust',0)} كم/س) | تأثير: {sign}{wind_eff:.0f}م")
        interactions.append(f"  🌊 الموج والمسافة: {raw.get('wave_period',0):.1f} ث | المسافة: {raw.get('recommended_cast_distance',0):.0f}م")
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
    target_date = resolve_target_date(req.target_date, datetime.now(resolve_timezone(tz_name)).date())
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
        f">  * اتجاه الشاطئ: {extra.get('beach_orientation', 'غير محدد')}°",
    ]
    facts.append("⏱️ 1. التوقيت المدوي وحركة المياه")
    facts.append("🌊 مواقيت العبور القمري (تقديرية)")
    for k, v in extra.get("tidal_windows", {}).items():
        name = "العبور القمري" if k.startswith("HW") else "الجزر المحاقي"
        facts.append(f" * 🔹 {name}: الساعة {v}")
    facts.append(f" * 🌅 الشروق: {extra.get('sunrise', '')} | 🌇 الغروب: {extra.get('sunset', '')}")
    facts.append("🏖️ مؤشر الشاطئ (أيام الحياء والمات)")
    moon_age = extra.get("moon_age_days", 0)
    haml_status = extra.get("haml_status", "")
    haml_phase = extra.get("haml_phase", "")
    haml_desc = extra.get("haml_description", "")
    if haml_status != "أيام عادية" and moon_age:
        moon_text = f" * 🌊 الوضعية: {haml_status} ({haml_phase}). {haml_desc} (عمر القمر {moon_age:.1f} يوم)"
    else: moon_text = f" * 🌊 الوضعية: {haml_status}. {haml_desc}"
    facts.append(moon_text)
    facts.append("🌡️ الضغط الجوي")
    facts.append(f" * الوضع: {extra.get('pressure_note', 'مستقر')}")
    lines = ["\n".join(facts), "", "=== التفاعلات ===", *chain_interactions]
    return "\n".join(lines)

SYSTEM_PROMPT = """أنت خبير سيرفكاستينغ تونسي. اكتب تقريرًا بالدارجة التونسية باستخدام البيانات التالية.
القرار النهائي ونسبة النجاح موجودان في [الحسم النهائي] و[نسبة النجاح]. لا تغيرهما.

استخدم التنسيق التالي بالضبط:
🎯 0. الملخص التنفيذي ليوم (التاريخ) – النسبة، القرار، السبب، الطعم، اتجاه الشاطئ.
⏱️ 1. التوقيت المدوي وحركة المياه – العبور القمري، الشروق/الغروب، مؤشر الشاطئ، الضغط، السولونار.
🏃‍♂️ 2. فترات الحركة (الأوقات الخضراء) – لا تكتب هذا القسم مطلقاً. سيتم إضافته تلقائياً.
🕒 3. التفكيك الديناميكي الزمني – لكل فترة (بالترتيب: السحر، الصباح، الظهيرة، الغسق): الحالة، الرياح، الموج، الموانع والتحذيرات (إن وجدت).
⚖️ 4. ميزان العوامل – العوامل الحمراء (المعوقات) والخضراء (الإيجابيات).
🏹 5. التكتيك الميداني والسلامة – الرصاص، التوقيت، المسافة.

قواعد مهمة:
- اكتب بالدارجة التونسية فقط.
- لا تخترع موانع غير موجودة في البيانات.
- استخدم الأوقات والنطاقات كما هي في البيانات.
- لا تذكر المركب ولا تستخدم حروفًا لاتينية.
"""

# ---------- Gemini caller with validation ----------
class GeminiNotConfigured(RuntimeError):
    """يرمى عندما لا يكون مفتاح Gemini مضبوطاً على الخادم."""


def build_gemini_payload(ctx: str, with_thinking_config: bool = True) -> dict:
    """يبني حمولة الطلب. تعطيل «التفكير العميق» يقلّل زمن الرد بشكل ملحوظ."""
    generation: Dict[str, Any] = {
        "temperature": settings.GEMINI_TEMPERATURE,
        "maxOutputTokens": settings.GEMINI_MAX_OUTPUT_TOKENS,
        "candidateCount": 1,
    }
    if with_thinking_config and settings.GEMINI_THINKING_BUDGET >= 0:
        generation["thinkingConfig"] = {"thinkingBudget": settings.GEMINI_THINKING_BUDGET}
    return {"contents": [{"parts": [{"text": SYSTEM_PROMPT + "\n\n" + ctx}]}], "generationConfig": generation}


def parse_retry_after(response: httpx.Response) -> Optional[float]:
    """يقرأ ترويسة Retry-After إن وجدت (ثوانٍ)."""
    raw = response.headers.get("Retry-After") or response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


async def call_gemini(ctx):
    if not GEMINI_API_KEY:
        raise GeminiNotConfigured("GEMINI_API_KEY غير مضبوط على الخادم.")
    if len(ctx) > settings.GEMINI_MAX_CONTEXT_CHARS:
        ctx = ctx[: settings.GEMINI_MAX_CONTEXT_CHARS]
        logger.warning(f"Context truncated to {settings.GEMINI_MAX_CONTEXT_CHARS} chars")
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    timeout = httpx.Timeout(connect=settings.GEMINI_CONNECT_TIMEOUT_S,
                            read=settings.GEMINI_READ_TIMEOUT_S,
                            write=10.0, pool=10.0)
    max_attempts = max(1, settings.GEMINI_MAX_ATTEMPTS)
    use_thinking_config = settings.GEMINI_THINKING_BUDGET >= 0
    last_error: Optional[Exception] = None

    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(1, max_attempts + 1):
            payload = build_gemini_payload(ctx, with_thinking_config=use_thinking_config)
            try:
                r = await client.post(GEMINI_URL, json=payload, headers=headers)
                if r.status_code == 429:
                    wait = parse_retry_after(r)
                    if wait is None:
                        base = GEMINI_RETRY_WAITS[min(attempt - 1, len(GEMINI_RETRY_WAITS) - 1)]
                        wait = base + random.uniform(0, 2)
                    logger.warning(f"Gemini 429 – retrying in {wait:.1f}s (attempt {attempt}/{max_attempts})")
                    await asyncio.sleep(min(wait, 30.0))
                    continue
                if r.status_code == 400 and use_thinking_config:
                    # بعض نسخ النموذج لا تقبل thinkingConfig → نعيد بدونها فوراً
                    logger.warning("Gemini rejected thinkingConfig; retrying without it.")
                    use_thinking_config = False
                    continue
                r.raise_for_status()
                data = r.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise RuntimeError("Gemini أرجع استجابة فارغة (ربما فلتر أمان)")
                first = candidates[0]
                if first.get("finishReason") == "SAFETY":
                    raise RuntimeError("Gemini حظر الاستجابة لأسباب أمان")
                parts = first.get("content", {}).get("parts", [])
                if not parts or "text" not in parts[0]:
                    raise RuntimeError("بنية استجابة Gemini غير متوقعة")
                return parts[0]["text"]
            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"Gemini timeout (attempt {attempt}/{max_attempts}): {e}")
            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code if e.response is not None else 0
                logger.error(f"Gemini HTTP {status}: {e}")
                if status in (400, 401, 403, 404) and not (status == 400 and use_thinking_config):
                    raise  # خطأ دائم: لا فائدة من إعادة المحاولة
            except (KeyError, IndexError, ValueError) as e:
                last_error = e
                logger.error(f"Gemini malformed response: {e}")
            except Exception as e:
                last_error = e
                logger.error(f"Gemini unexpected error: {e}")

            if attempt < max_attempts:
                wait = GEMINI_RETRY_WAITS[min(attempt - 1, len(GEMINI_RETRY_WAITS) - 1)]
                await asyncio.sleep(wait)
    raise RuntimeError("فشل الاتصال بـ Gemini") from last_error

def generate_offline_report(req: RawDataReportRequest, agg: dict, tz_name: str) -> str:
    """
    تقرير احتياطي يُبنى محلياً من نتائج التحليل عند تعذّر الوصول إلى Gemini
    (ضغط API، نفاد الحصة، أو غياب المفتاح). بنفس بنية الأقسام المتوقعة.
    """
    extra = agg.get("extra_info", {})
    blocks = agg.get("blocks", [])
    target_date = resolve_target_date(req.target_date, datetime.now(resolve_timezone(tz_name)).date())
    lines = []

    lines.append(f"🎯 0. الملخص التنفيذي ليوم {target_date.strftime('%d/%m/%Y')}")
    lines.append(f"> نسبة النجاح: {agg.get('score', 0)}%")
    lines.append(f">  * القرار النهائي: {agg.get('final_verdict', '')}")
    reason = agg.get("nogo_reasons", []) or agg.get("warnings", []) or ["ظروف متوسطة"]
    lines.append(f">  * السبب الرئيسي: {reason[0]}")
    lines.append(f">  * الطعم المستهدف: {extra.get('seasonal_bait', '')}")
    lines.append(f">  * اتجاه الشاطئ: {extra.get('beach_orientation', req.beach_orientation)}° "
                 f"({deg_to_compass(extra.get('beach_orientation', req.beach_orientation))})")
    lines.append("")

    lines.append("⏱️ 1. التوقيت المدوي وحركة المياه")
    lines.append("🌊 مواقيت العبور القمري (تقديرية)")
    for key, value in (extra.get("tidal_windows") or {}).items():
        label = "العبور القمري" if key.startswith("HW") else "الجزر المحاقي"
        lines.append(f" * 🔹 {label}: الساعة {value}")
    lines.append(f" * 🌅 الشروق: {extra.get('sunrise', 'غير متوفر')} | 🌇 الغروب: {extra.get('sunset', 'غير متوفر')}")
    lines.append("🏖️ مؤشر الشاطئ (أيام الحياء والمات)")
    haml_status = extra.get("haml_status", "")
    haml_phase = extra.get("haml_phase", "")
    haml_title = f"{haml_status} ({haml_phase})" if haml_phase else haml_status
    lines.append(f" * 🌊 الوضعية: {haml_title}. {extra.get('haml_description', '')} "
                 f"{extra.get('platform_advice', '')}")
    sol = extra.get("solunar") or {}
    lines.append(f"🎯 أوقات سولونار — الرئيسية: {sol.get('major1', '-')} و {sol.get('major2', '-')} | "
                 f"الثانوية: {sol.get('minor1', '-')} و {sol.get('minor2', '-')}")
    lines.append("🌡️ الضغط الجوي")
    press_change = extra.get("pressure_change", 0) or 0
    press_text = "0.0" if abs(press_change) < 0.05 else f"{press_change:+.1f}"
    lines.append(f" * الوضع: {extra.get('pressure_note', 'مستقر')} (تغير يومي {press_text} hPa)")
    lines.append(f"🌡️ حرارة الماء: {round(agg['avg_sst'], 1) if agg.get('avg_sst') is not None else 'غير متوفر'}°م | "
                 f"حرارة الهواء العظمى: {extra.get('max_air_temp', 'غير متوفر')}°م")
    lines.append("")

    lines.append(build_flow_section(extra.get("tidal_windows") or {}))
    lines.append("")

    lines.append("🕒 3. التفكيك الديناميكي الزمني")
    for b in blocks:
        raw = b.get("_raw", {})
        lines.append(f" * {b.get('name', '')} ({b.get('time_range', '')}):")
        lines.append(f"   - حالة البحر: {b.get('sea_state', '')} | الثقة: {b.get('confidence', 0)}% "
                     f"({b.get('confidence_label', '')})")
        wind_eff = raw.get("wind_effect_dist", 0)
        sign = "+" if wind_eff > 0 else ""
        lines.append(f"   - الرياح: {b.get('wind_dir', '')} متوسط {raw.get('avg_wind', 0)} كم/س "
                     f"(هبات {raw.get('max_gust', 0)} كم/س) | تأثير على الرمية: {sign}{wind_eff:.0f}م")
        lines.append(f"   - الموج: أقصى {raw.get('max_wave_h', 0)}م | الفترة: {raw.get('wave_period', 0)}ث | "
                     f"المسافة المقترحة: {raw.get('recommended_cast_distance', 0):.0f}م")
        lines.append(f"   - عكارة الماء: {b.get('water_clarity', '')} | المونتاج: {b.get('suggested_rig', '')} | "
                     f"راحة: {b.get('comfort_index', 0)}%")
        active = "، ".join(b.get("active_fish", [])) or "لا شيء مؤكد"
        lines.append(f"   - الأسماك الأكثر نشاطاً: {active}")
        for item in b.get("nogo_reasons", []):
            lines.append(f"   - ⛔ مانع: {item}")
        for item in b.get("period_warnings", []):
            lines.append(f"   - ⚠️ تحذير: {item}")
        lines.append("")

    lines.append("⚖️ 4. ميزان العوامل")
    reds = agg.get("nogo_reasons", []) + agg.get("warnings", [])
    lines.append("🔴 العوامل الحمراء (المعوقات):")
    if reds:
        for item in reds[:8]:
            lines.append(f" * {item}")
    else:
        lines.append(" * لا توجد معوقات مهمة.")
    lines.append("🟢 العوامل الخضراء (الإيجابيات):")
    greens = []
    if agg.get("flags", {}).get("has_golden_window"):
        greens += [g for g in (extra.get("golden_windows") or []) if "تنبيه" not in g]
    if extra.get("haml_score_delta", 0) > 0:
        greens.append("البحر في حمل حيوي (تيارات غذائية قوية).")
    if any(0.6 <= b.get("_raw", {}).get("avg_wave_h", 0) <= 1.2 for b in blocks):
        greens.append("ارتفاع موج مناسب (0.6 - 1.2م) في بعض الفترات.")
    if extra.get("pressure_change", 0) <= -3:
        greens.append("الضغط ينخفض: نشاط تغذية أفضل.")
    best = max(blocks, key=lambda b: b.get("confidence", 0)) if blocks else None
    if best:
        greens.append(f"أفضل فترة: {best.get('name', '')} ({best.get('time_range', '')}) بثقة {best.get('confidence', 0)}%.")
    for item in greens[:8]:
        lines.append(f" * {item}")
    lines.append("")

    lines.append("🏹 5. التكتيك الميداني والسلامة")
    best_block = best or (blocks[0] if blocks else None)
    if best_block:
        raw = best_block.get("_raw", {})
        lines.append(f" * أفضل نافذة: {best_block.get('name', '')} ({best_block.get('time_range', '')})")
        lines.append(f" * الرمي: مسافة {raw.get('recommended_cast_distance', 0):.0f}م، "
                     f"تصحيح زاوية {best_block.get('casting_angle_correction', 0)}° حسب الرياح.")
        lines.append(f" * المونتاج: {best_block.get('suggested_rig', '')}")
        lines.append(f" * الطعم: {extra.get('seasonal_bait', '')}")
    if agg.get("flags", {}).get("is_mirror_sea"):
        lines.append(" * بحر مرآوي: خفّض سماكة الخيط والقماش (قلادة رفيعة) واستعمل خيطاً شفافاً.")
    if agg.get("flags", {}).get("is_lateral_strong"):
        lines.append(" * تيار جانبي قوي: زد وزن الرصاصة واستعمل رصاصاً مفلطحاً (Grip Lead).")
    if agg.get("flags", {}).get("is_weedy"):
        lines.append(" * صوفة متوقعة: استعمل صائدات مضادة للأعشاب ونظّف الخيط كل رميات قليلة.")
    lines.append(" * السلامة: لا ترمِ في البحر الهائج، ولا تصعد على الصخور المبللة، ولا تصيد وحيداً ليلاً.")
    lines.append("")
    lines.append("ملاحظة: هذا التقرير أُنشئ آلياً من الحسابات المحلية (وضع احتياطي دون نموذج لغوي).")
    return "\n".join(lines)


def generate_manual_context(req: RawDataReportRequest, agg: dict, tz_name: str) -> str:
    extra = agg["extra_info"]
    target_date = resolve_target_date(req.target_date, datetime.now(resolve_timezone(tz_name)).date())
    date_str = target_date.strftime("%d/%m/%Y")
    moon_age = extra.get("moon_age_days", 0)
    haml_status = extra.get("haml_status", ""); haml_phase = extra.get("haml_phase", ""); haml_desc = extra.get("haml_description", "")
    if haml_status != "أيام عادية": moon_line = f"{haml_status} ({haml_phase}). {haml_desc} (عمر القمر {moon_age:.1f} يوم)"
    else: moon_line = f"{haml_status}. {haml_desc}"
    sol = extra.get("solunar", {})
    sol_text = f"الرئيسية: {sol.get('major1')} و {sol.get('major2')} | الثانوية: {sol.get('minor1')} و {sol.get('minor2')}"
    flows = []
    for k, v in (extra.get('tidal_windows') or {}).items():
        h = safe_parse_time(v)
        if k.startswith("HW"): flows.append(f"بداية جزر ساحب {format_time(h+0.5)}")
        else: flows.append(f"بداية مد دافق {format_time(h+0.5)}")
    if agg["final_verdict"] == "غير مناسب":
        main_reason = "بحر غير مناسب للصيد في جميع الفترات"
        if agg["nogo_reasons"]: main_reason = agg["nogo_reasons"][0]
    elif agg["final_verdict"] == "فرصة مع تحفظات":
        good_periods = [b["name"] for b in agg["blocks"] if not b["has_lethal_nogo"]]
        bad_periods = [b["name"] for b in agg["blocks"] if b["has_lethal_nogo"]]
        if bad_periods: main_reason = f"فترات غير مناسبة: {', '.join(bad_periods)}. فترات مناسبة: {', '.join(good_periods) if good_periods else 'لا يوجد'}"
        else: main_reason = "توجد تحفظات لكن يمكن التكيف معها"
    else: main_reason = "ظروف ممتازة للصيد"
    lines = []
    lines.append("اكتب تقرير صيد سيرفكاستينغ تونسي بالدارجة التونسية باستخدام جميع المعطيات التالية:")
    lines.append(f"التاريخ: {date_str}")
    lines.append(f"اتجاه الشاطئ: {req.beach_orientation}° | نوع القاع: {req.beach_type or 'غير محدد'}")
    lines.append(f"القرار النهائي: {agg['final_verdict']} | نسبة النجاح: {agg['score']}%")
    lines.append(f"السبب الرئيسي: {main_reason}")
    lines.append(f"الطعم المستهدف: {extra.get('seasonal_bait', '')}")
    lines.append(f"الشروق: {extra.get('sunrise', 'غير متوفر')} | الغروب: {extra.get('sunset', 'غير متوفر')}")
    lines.append(f"الضغط الجوي: {extra.get('pressure_note', 'مستقر')} (تغير يومي {extra.get('pressure_change', 0):+.1f} hPa)")
    lines.append(f"حرارة الماء: {round(agg['avg_sst'],1) if agg['avg_sst'] is not None else 'غير متوفر'}°م | حرارة الهواء العظمى: {extra.get('max_air_temp', 'غير متوفر')}°م")
    lines.append(f"الرياح اليومية السائدة: {agg.get('dominant_wind', 'غير معروف')} | أقصى هبات: {extra.get('peak_gust_today', 'غير متوفر')} كم/س")
    tw = extra.get('tidal_windows') or {}
    lines.append(f"العبور القمري: HW1={tw.get('HW1', '-')}, LW1={tw.get('LW1', '-')}, HW2={tw.get('HW2', '-')}, LW2={tw.get('LW2', '-')}")
    lines.append(f"أوقات السولونار: {sol_text}")
    lines.append(f"مؤشر الشاطئ: {moon_line}")
    lines.append("فترات الحركة (الخضراء): " + " | ".join(flows))
    lines.append("")
    lines.append("تفاصيل الفترات:")
    for b in agg['blocks']:
        r = b["_raw"]
        conf = b.get("confidence",0); conf_label = b.get("confidence_label","")
        nogo = f"موانع: {'; '.join(b['nogo_reasons'])}" if b.get("nogo_reasons") else "لا موانع"
        warn = f"تحذير: {'; '.join(b['period_warnings'])}" if b.get("period_warnings") else ""
        active = ', '.join(b.get('active_fish', [])) if b.get('active_fish') else 'لا يوجد'
        inactive = ', '.join(b.get('inactive_fish', [])) if b.get('inactive_fish') else 'لا يوجد'
        wind_deg = r.get('wind_dir_deg')
        wind_deg_str = f"{wind_deg:.0f}°" if wind_deg is not None else "غير معروف"
        lines.append(f"{b['name']} ({b['time_range']}):")
        lines.append(f"  الثقة: {conf}% ({conf_label}) | البحر: {b['sea_state']} | أقصى موج: {r['max_wave_h']}م | فترة الموج: {r['wave_period']}ث")
        lines.append(f"  الرياح: {b['wind_dir']} متوسط {r['avg_wind']} كم/س (هبات {r['max_gust']} كم/س) | تأثير الرياح: {r['wind_effect_dist']:+.0f}م | اتجاه: {wind_deg_str}")
        lines.append(f"  المسافة: {r['recommended_cast_distance']}م | تصحيح الزاوية: {b.get('casting_angle_correction',0)}°")
        lines.append(f"  عكارة الماء: {b.get('water_clarity','')} | المونتاج: {b.get('suggested_rig','')} | مؤشر الراحة: {b.get('comfort_index',50)}%")
        lines.append(f"  الأسماك النشطة: {active} | الخاملة: {inactive}")
        if nogo != "لا موانع": lines.append(f"  ⛔ {nogo}")
        if warn: lines.append(f"  ⚠️ {warn}")
        lines.append("")
    all_active = set()
    for b in agg['blocks']:
        for f in b.get('active_fish', []): all_active.add(f)
    lines.append(f"قائمة الأسماك النشطة إجمالاً: {', '.join(sorted(all_active)) if all_active else 'لا يوجد'}")
    return "\n".join(lines)

# ---------- Text helpers ----------
def fix_time_ranges(text: str) -> str:
    pattern = r'(\d{2}:\d{2})\s*[-–]\s*(\d{2}:\d{2})'
    def repl(m):
        t1, t2 = m.group(1), m.group(2)
        to_min = lambda s: int(s.split(':')[0])*60 + int(s.split(':')[1])
        m1, m2 = to_min(t1), to_min(t2)
        if m2 - m1 < 0 and abs(m2 - m1) < 720: return f"{t2} - {t1}"
        return f"{t1} - {t2}"
    return re.sub(pattern, repl, text)

def fix_broken_time_in_headers(text: str) -> str:
    return re.sub(r'(\*\s+[^*(]+)\((\d{2}:\d{2})\s*\n\s*\*\s+(\d{2}:\d{2})\)', r'\1(\2 - \3)', text)

def replace_english_commas(text: str) -> str:
    text = re.sub(r'(?<=[\u0600-\u06FF\s]),(?=[\u0600-\u06FF\s])', '،', text)
    return text

def _keep_indent(line: str) -> str:
    """يحافظ على المسافة البادئة للأسطر الفرعية وينظّف نهايتها فقط."""
    return line.rstrip() if line[:1].isspace() else line.strip()


def enforce_line_breaks(text: str) -> str:
    lines = text.split('\n')
    new_lines = []
    for line in lines:
        stripped = _keep_indent(line)
        if not stripped: new_lines.append(''); continue
        if re.match(r'^[*-] ', stripped) or re.match(r'^[🔹🔸🌊🟢🔴🎯⏱️🏖️⏳🏃‍♂️🕒⚖️🏹📊🌅☀️🌃🌆🌙🐟💤🔄💨📊📐🌡️🛠️⏱️🎯🦐⚠️📌💧😌⛔]', stripped):
            if new_lines and new_lines[-1] != '' and not re.match(r'^[*-] ', new_lines[-1]): new_lines.append('')
        new_lines.append(stripped)
    return '\n'.join(new_lines)

def add_paragraph_spacing(text: str) -> str:
    lines = text.split('\n')
    new_lines = []
    for line in lines:
        stripped = _keep_indent(line)
        if not stripped: new_lines.append(''); continue
        if re.match(r'^(🎯|⏱️|🏃‍♂️|🕒|⚖️|🏹|📊|\d\.)', stripped):
            if new_lines and new_lines[-1] != '': new_lines.append('')
        new_lines.append(stripped)
    return '\n'.join(new_lines)

def clean_report_text(text: str) -> str:
    text = re.sub(r'(العبور القمري|الجزر المحاقي):(\d{2}:\d{2})', r'\1: \2', text)
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
                fixed.append(merged); i += 2; continue
        line = re.sub(r'(\d+):\s+(\d+)', r'\1:\2', line)
        fixed.append(line); i += 1
    return '\n'.join(fixed)

# ---------- Main Endpoint ----------
def validate_report_payload(req: RawDataReportRequest) -> None:
    """يتحقّق من الحمولة قبل المعالجة: رسالة 400 واضحة بدل خطأ 500 مبهم."""
    if req.marine_data is not None:
        if not isinstance(req.marine_data, dict):
            raise HTTPException(400, detail="حقل marine_data يجب أن يكون كائناً JSON.")
        hourly = req.marine_data.get("hourly") or {}
        if not isinstance(hourly, dict) or not hourly.get("time"):
            raise HTTPException(400, detail="بيانات البحر (marine_data) لا تحتوي على سلسلة زمنية صالحة (hourly.time).")
    if req.weather_data is not None:
        if not isinstance(req.weather_data, dict):
            raise HTTPException(400, detail="حقل weather_data يجب أن يكون كائناً JSON.")
        hourly = req.weather_data.get("hourly") or {}
        if not isinstance(hourly, dict) or not hourly.get("time"):
            raise HTTPException(400, detail="بيانات الطقس (weather_data) لا تحتوي على سلسلة زمنية صالحة (hourly.time).")
        daily = req.weather_data.get("daily") or {}
        if not isinstance(daily, dict) or not daily.get("sunrise") or not daily.get("sunset"):
            raise HTTPException(400, detail="بيانات الطقس (weather_data) لا تحتوي على أوقات الشروق والغروب (daily).")

def report_cache_key(req: RawDataReportRequest) -> Tuple:
    """مفتاح الكاش: الموقع + الإعدادات + بصمة البيانات حتى لا نقدّم تقريراً قديماً لبيانات جديدة."""
    lat = round(req.latitude or 0.0, 3)
    lon = round(req.longitude or 0.0, 3)
    return (lat, lon, req.beach_orientation, req.beach_type or "", req.target_date,
            _hash_payload(req.marine_data), _hash_payload(req.weather_data))

@app.post("/generate-report")
@limiter.limit(settings.RATE_LIMIT_REPORT)
async def generate_report(request: Request, req: RawDataReportRequest):
    validate_report_payload(req)
    key = report_cache_key(req)
    hit, cached = await report_cache.get(key)
    if hit and isinstance(cached, dict) and cached.get("report"):
        result = dict(cached)
        meta = dict(result.get("meta") or {})
        meta["cached"] = True
        result["meta"] = meta
        response = JSONResponse(content=result)
        response.headers["X-Cache"] = "HIT"
        return response

    try:
        result = await asyncio.wait_for(_generate_report_inner(req), timeout=settings.REPORT_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise HTTPException(504, detail="انتهت مهلة إنشاء التقرير. حاول مرة أخرى.")
    except HTTPException: raise
    except Exception as e:
        logger.error(f"generate-report error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, detail="فشل إنشاء التقرير")

    if isinstance(result, dict):
        meta = dict(result.get("meta") or {})
        meta["cached"] = False
        result["meta"] = meta
        await report_cache.set(key, result)
    response = JSONResponse(content=result)
    response.headers["X-Cache"] = "MISS"
    return response

async def _generate_report_inner(req: RawDataReportRequest):
    async with httpx.AsyncClient(timeout=settings.UPSTREAM_TIMEOUT_S) as client:
        if req.marine_data and req.weather_data:
            marine_data, weather_data = req.marine_data, req.weather_data
        else:
            lat = req.latitude or 36.8; lon = req.longitude or 10.1
            marine_data, weather_data = await asyncio.gather(
                fetch_marine_data_cached(client, lat, lon),
                fetch_weather_data_cached(client, lat, lon),
                return_exceptions=True
            )
            if isinstance(marine_data, Exception): raise HTTPException(502, "تعذر جلب بيانات البحر من المصدر")
            if isinstance(weather_data, Exception): raise HTTPException(502, "تعذر جلب بيانات الطقس من المصدر")
            if not marine_data: raise HTTPException(502, "بيانات البحر فارغة")
            if not weather_data: raise HTTPException(502, "بيانات الطقس فارغة")

    marine_hourly = marine_data.get("hourly", marine_data)
    weather_hourly = weather_data.get("hourly", {})
    daily = weather_data.get("daily", {})
    tz_name = marine_data.get("timezone", settings.DEFAULT_TZ)
    tz = resolve_timezone(tz_name)
    now_tn = datetime.now(tz)
    target_dt = resolve_target_date(req.target_date, now_tn.date())

    raw_sr = pick_daily_value(daily, "sunrise", target_dt, now_tn.date(), "06:00")
    raw_ss = pick_daily_value(daily, "sunset", target_dt, now_tn.date(), "18:00")
    sr_match = _TIME_RE.search(str(raw_sr)); ss_match = _TIME_RE.search(str(raw_ss))
    sunrise = sr_match.group() if sr_match else "06:00"
    sunset = ss_match.group() if ss_match else "18:00"
    latitude = req.latitude or 36.8; longitude = req.longitude or 10.1
    beach_type = req.beach_type or "sandy"

    all_times, aligned = align_hourly_data(marine_hourly, weather_hourly, tz_name)
    if not all_times:
        detail = ("تعذّر مزامنة البيانات الساعية بين البحر والطقس (لا توجد ساعات مشتركة). "
                  "تحقّق من أنّ marine_data و weather_data يغطيان نفس الفترة.")
        raise HTTPException(400 if (req.marine_data and req.weather_data) else 502, detail=detail)

    agg = aggregate_physics(all_times, aligned, req.beach_orientation, target_dt, sunrise, sunset, latitude, longitude, beach_type)

    if not agg.get("blocks") or agg.get("final_verdict") == "بيانات غير كافية":
        raise HTTPException(422, detail=("لا توجد بيانات أرصاد كافية لليوم المطلوب. "
                                         "جرّب تاريخاً أقرب (اليوم أو الغد) أو بقعة ساحلية أخرى."))

    HARD_NOGO_KEYWORDS = ["هائج", "صواعق", "ضباب كثيف", "عكارة طينية", "رياح عاتية", "أمواج أرضية", "تيار جانبي عنيف", "تيار راجع عنيف"]
    if (agg["final_verdict"] == "غير مناسب" and
        all(b.get("has_lethal_nogo", False) for b in agg["blocks"]) and
        any(any(kw in r for kw in HARD_NOGO_KEYWORDS) for r in agg["nogo_reasons"])):
        return {
            "report": f"❌ غير مناسب ({agg['score']}%) – {agg['nogo_reasons'][0]}",
            "meta": {"score": agg['score'], "hard_nogo": True, "generated_by": "rules"}
        }

    def offline_response(error: Exception) -> dict:
        """يردّ بتقرير محلي مكتمل عند تعذّر النموذج اللغوي."""
        logger.error(f"Falling back to offline report: {error}")
        manual = generate_manual_context(req, agg, tz_name)
        report_text = generate_offline_report(req, agg, tz_name)
        for cleanup in (clean_report_text, fix_broken_number_lines, fix_broken_time_in_headers,
                        replace_english_commas, enforce_line_breaks, add_paragraph_spacing):
            report_text = cleanup(report_text)
        extra_info = agg.get("extra_info", {})
        return {
            "report": report_text,
            "manual_context": manual,
            "meta": {
                "score": agg["score"],
                "final_verdict": agg["final_verdict"],
                "generated_by": "offline",
                "gemini_configured": bool(GEMINI_API_KEY),
                "gemini_error": str(error)[:200],
                "tidal_windows": extra_info.get("tidal_windows", {}),
                "solunar": extra_info.get("solunar", {}),
                "blocks": [{k: v for k, v in b.items() if k != "_raw"} for b in agg["blocks"]],
            },
        }

    try:
        flow_section_text = build_flow_section(agg["extra_info"]["tidal_windows"])
        ctx = build_context(req, agg, tz_name)
        try:
            report = await call_gemini(ctx)
        except GeminiNotConfigured as e:
            logger.error(f"Gemini not configured: {e}")
            if not settings.OFFLINE_FALLBACK:
                raise HTTPException(503, detail="خدمة توليد التقارير غير مضبوطة على الخادم (مفتاح Gemini مفقود).")
            return offline_response(e)
        report = clean_report_text(report)
        report = fix_broken_number_lines(report)
        report = fix_broken_time_in_headers(report)
        report = replace_english_commas(report)
        report = enforce_line_breaks(report)
        report = add_paragraph_spacing(report)

        # Inject flow section at the correct position (before section 3)
        report = re.sub(r'(🕒\s*3\.|⚖️\s*4\.)', f"{flow_section_text}\n\n\\g<0>", report, count=1)

        raw_numbers = "\n\n━━━━━━━━━━━━━━━━━━━━━━━━\n📊 الأرقام المرجعية (للتحقق)\n━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        avg_sst_display = round(agg['avg_sst'], 1) if agg['avg_sst'] is not None else "غير متوفر"
        raw_numbers += f"🔹 حرارة الماء: {avg_sst_display}°م | حرارة الهواء: {agg['extra_info']['max_air_temp']}°م\n"
        raw_numbers += f"🔹 الرياح: أقصى هبات {agg['extra_info']['peak_gust_today']} كم/س | الضغط: {agg['extra_info']['pressure_avg']} hPa\n"
        for b in agg['blocks']:
            r = b['_raw']
            wind_eff = r['wind_effect_dist']; sign = '+' if wind_eff > 0 else ''
            wind_deg = r.get('wind_dir_deg')
            wind_deg_str = f"{wind_deg:.0f}°" if wind_deg is not None else "غير معروف"
            raw_numbers += (f"🔸 {b['name']} ({b['time_range']}): "
                            f"ثقة {b['confidence']}% ({b.get('confidence_label','')}) | مسافة {b['recommended_cast_distance']}م | "
                            f"موج {r['avg_wave_h']}-{r['max_wave_h']}م | رياح {r['avg_wind']} كم/س ({wind_deg_str}) | "
                            f"تأثير الرياح {sign}{wind_eff:.0f}م | "
                            f"عكارة: {b.get('water_clarity','')} | مونتاج: {b.get('suggested_rig','')} | راحة: {b.get('comfort_index','')}%")
            if b.get("nogo_reasons"): raw_numbers += f" | ⛔: {'; '.join(b['nogo_reasons'])}"
            raw_numbers += "\n"
        raw_numbers += "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        report += raw_numbers

        clean_blocks = [{k:v for k,v in b.items() if k != "_raw"} for b in agg["blocks"]]
        meta = {
            "timezone": tz_name, "target_date": target_dt.isoformat(), "hard_nogo": False, "generated_by": "gemini",
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
        raise  # أخطاء HTTP مقصودة (503 مثلًا) لا تُستبدل بسياق يدوي
    except HTTPException:
        raise  # أخطاء HTTP مقصودة لا تُستبدل بتقرير احتياطي
    except Exception as e:
        logger.error(f"Gemini failed: {e}")
        if not settings.OFFLINE_FALLBACK:
            raise HTTPException(502, detail="تعذّر توليد التقرير من نموذج اللغة. حاول مجدداً.")
        return offline_response(e)

# ---------- خدمة الواجهة الثابتة (نشر موحّد: واجهة + API على نفس الأصل) ----------
try:
    from fastapi.staticfiles import StaticFiles
    from pathlib import Path as _Path

    FRONTEND_DIR = _Path(__file__).resolve().parent.parent / "frontend"
    if FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
        logger.info(f"Serving frontend from {FRONTEND_DIR}")
except Exception as e:  # لا يمنع تشغيل الـ API إن تعذّر تركيب الملفات الثابتة
    logger.warning(f"Static frontend not mounted: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
