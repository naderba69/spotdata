"""
Surfcasting Analytics API – v9.0 (Completely rebuilt orientation – coastline search method)
"""
import os, math, asyncio, logging, traceback, zoneinfo, time
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Tuple
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
app = FastAPI(title="Surfcasting Analytics", version="9.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY مفقود")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "google/gemini-2.5-flash-lite"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "SurfcastingAnalytics/1.0 (naderba69@gmail.com)"

cache = {}
cache_lock = asyncio.Lock()
CACHE_TTL = 3600

class AutoOrientationRequest(BaseModel):
    latitude: float
    longitude: float

class RawDataReportRequest(BaseModel):
    beach_orientation: int = Field(..., ge=0, le=360)
    beach_type: str = Field(..., pattern="^(sandy|rocky)$")
    target_date: str = Field(..., pattern="^(today|tomorrow|day_after)$")
    marine_data: dict
    weather_data: dict

@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})

@app.get("/health")
def health():
    return {"status": "ok", "openrouter": bool(OPENROUTER_API_KEY), "model": MODEL_NAME, "mode": "client-side"}

async def post_with_retry(url, json_data, headers, max_retries=3, timeout=120):
    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient() as c:
                r = await c.post(url, json=json_data, headers=headers, timeout=timeout)
                r.raise_for_status()
                return r.json()
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

# ---------- دوال مساعدة ----------
def safe_float(v):
    try: return 0.0 if math.isnan(float(v)) else float(v)
    except: return 0.0

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

def moon_phase_detail(d: date):
    y, m, day = d.year, d.month, d.day
    if m < 3: y-=1; m+=12
    a = int(y/100); b = 2-a+int(a/4)
    jd = int(365.25*(y+4716)) + int(30.6001*(m+1)) + day + b - 1524.5
    days_since_new = jd - 2451550.1
    phase = (days_since_new % 29.53058867) / 29.53058867
    idx = int(phase * 8) % 8
    phases = {0:"محاق",1:"هلال أول",2:"تربيع أول",3:"أحدب متزايد",4:"بدر",5:"أحدب متناقص",6:"تربيع ثاني",7:"هلال آخر"}
    name = phases.get(idx, "محاق")
    if phase < 0.125 or phase > 0.875:
        status, activity = "أيام متوسطة (محاق / هلال آخر)", "ينشط القاع ليلاً بشكل خاص"
    elif phase < 0.5:
        status, activity = "أيام حمل", "الأسماك نشيطة طوال اليوم"
    else:
        status, activity = "أيام فساد", "الأسماك أقل نشاطاً"
    return {"name":name, "status":status, "activity":activity}

def moon_fishing_guidance(d: date):
    detail = moon_phase_detail(d)
    n = detail["name"]
    if "محاق" in n: return f"{detail['status']}. ركز على الصيد الليلي للقاروص والسارغ."
    if "هلال أول" in n or "تربيع أول" in n: return f"{detail['status']}. فرصة ممتازة."
    if "أحدب متزايد" in n: return f"{detail['status']}. البوري والدوراد نهاراً."
    if "بدر" in n: return f"{detail['status']}. الأسماك السطحية نشيطة نهاراً."
    return f"{detail['status']}. الصيد مقبول."

def resolve_target_date(txt, real_today):
    if txt == "today": return real_today
    if txt == "tomorrow": return real_today + timedelta(days=1)
    return real_today + timedelta(days=2)

def align_hourly_data(marine_hourly, weather_hourly, tz_name):
    tz = zoneinfo.ZoneInfo(tz_name)
    m_times = marine_hourly.get("time", [])
    w_times = weather_hourly.get("time", [])
    if not m_times or not w_times: return [], {}
    m_map = {}
    for i,t in enumerate(m_times):
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=tz)
        m_map[dt] = i
    w_map = {}
    for i,t in enumerate(w_times):
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=tz)
        w_map[dt] = i
    common = sorted(set(m_map) & set(w_map))
    if not common: return [], {}
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

# ---------- قاعدة الشواطئ لم تعد تُستخدم للتوجيه التلقائي ----------
TUNISIAN_BEACHES = {
    "بنزرت": [
        {"name":"شاطئ الكورنيش (بنزرت)","lat":37.2744,"lon":9.8739,"orientation":45,"type":"sandy"},
        {"name":"شاطئ سيدي سالم","lat":37.2800,"lon":9.8800,"orientation":45,"type":"sandy"},
        {"name":"شاطئ الحسيان","lat":37.2600,"lon":9.8600,"orientation":0,"type":"sandy"},
        {"name":"شاطئ الكاب سيرات","lat":37.3500,"lon":9.7500,"orientation":315,"type":"rocky"},
        {"name":"شاطئ سيدي عياد","lat":37.3300,"lon":9.7800,"orientation":0,"type":"sandy"},
        {"name":"شاطئ غار الملح","lat":37.1667,"lon":10.1833,"orientation":315,"type":"sandy"},
        {"name":"شاطئ سيدي علي المكي","lat":37.1500,"lon":10.2000,"orientation":0,"type":"sandy"},
        {"name":"شاطئ البطاح","lat":37.1300,"lon":10.2200,"orientation":45,"type":"sandy"},
        {"name":"شاطئ أوتيك (الشواية)","lat":37.0800,"lon":10.1000,"orientation":90,"type":"sandy"},
        {"name":"شاطئ رفراف","lat":37.2167,"lon":10.0833,"orientation":0,"type":"sandy"},
        {"name":"شاطئ رأس الجبل","lat":37.2500,"lon":10.0500,"orientation":315,"type":"sandy"},
        {"name":"شاطئ الزوارع","lat":37.2700,"lon":10.0200,"orientation":0,"type":"sandy"},
        {"name":"شاطئ لالة مريم","lat":37.2000,"lon":10.0500,"orientation":45,"type":"sandy"},
    ],
    "نابل": [
        {"name":"شاطئ نابل المدينة","lat":36.4500,"lon":10.7333,"orientation":90,"type":"sandy"},
        {"name":"شاطئ الحمامات","lat":36.4000,"lon":10.6167,"orientation":90,"type":"sandy"},
        {"name":"شاطئ ياسمين الحمامات","lat":36.3800,"lon":10.5500,"orientation":90,"type":"sandy"},
        {"name":"شاطئ الحمامات الجنوبي","lat":36.3500,"lon":10.5500,"orientation":90,"type":"sandy"},
        {"name":"شاطئ قليبية","lat":36.8500,"lon":11.1000,"orientation":45,"type":"sandy"},
        {"name":"شاطئ منزل حر","lat":36.8300,"lon":11.1200,"orientation":0,"type":"sandy"},
        {"name":"شاطئ الهوارية","lat":37.0333,"lon":11.0167,"orientation":315,"type":"rocky"},
        {"name":"شاطئ وادي الخف","lat":37.0200,"lon":11.0300,"orientation":0,"type":"rocky"},
        {"name":"شاطئ بني خيار","lat":36.4833,"lon":10.7833,"orientation":90,"type":"sandy"},
        {"name":"شاطئ دار شعبان","lat":36.4700,"lon":10.7500,"orientation":90,"type":"sandy"},
        {"name":"شاطئ قربة","lat":36.5500,"lon":10.8500,"orientation":90,"type":"sandy"},
        {"name":"شاطئ شط قربة","lat":36.5600,"lon":10.8700,"orientation":0,"type":"sandy"},
        {"name":"شاطئ منزل تميم","lat":36.7000,"lon":10.9500,"orientation":0,"type":"sandy"},
        {"name":"شاطئ سيدي الجديدي","lat":36.7200,"lon":10.9800,"orientation":45,"type":"sandy"},
        {"name":"شاطئ سليمان","lat":36.6333,"lon":10.5000,"orientation":90,"type":"sandy"},
        {"name":"شاطئ شط مريم","lat":36.6500,"lon":10.4500,"orientation":90,"type":"sandy"},
        {"name":"شاطئ تاكلسة","lat":36.7500,"lon":10.6500,"orientation":45,"type":"sandy"},
        {"name":"شاطئ المعمورة","lat":36.5500,"lon":10.6000,"orientation":90,"type":"sandy"},
        {"name":"شاطئ سيدي بوسعيد","lat":36.8700,"lon":10.3500,"orientation":0,"type":"sandy"},
    ],
    "تونس": [
        {"name":"شاطئ حلق الوادي","lat":36.8167,"lon":10.3167,"orientation":0,"type":"sandy"},
        {"name":"شاطئ الكرم","lat":36.8500,"lon":10.3200,"orientation":45,"type":"sandy"},
        {"name":"شاطئ قرطاج","lat":36.8528,"lon":10.3264,"orientation":90,"type":"sandy"},
        {"name":"شاطئ المرسى","lat":36.8764,"lon":10.3253,"orientation":45,"type":"sandy"},
        {"name":"شاطئ روّاد","lat":36.9667,"lon":10.1833,"orientation":45,"type":"sandy"},
    ],
    "سوسة": [
        {"name":"شاطئ بوجعفر","lat":35.8333,"lon":10.6333,"orientation":90,"type":"sandy"},
        {"name":"شاطئ القنطاوي","lat":35.8833,"lon":10.6000,"orientation":90,"type":"sandy"},
        {"name":"شاطئ حمام سوسة","lat":35.8500,"lon":10.6000,"orientation":90,"type":"sandy"},
        {"name":"شاطئ شط الرمال","lat":35.9000,"lon":10.5500,"orientation":45,"type":"sandy"},
        {"name":"شاطئ هرقلة","lat":36.0000,"lon":10.4500,"orientation":0,"type":"sandy"},
        {"name":"شاطئ سيدي بوعلي","lat":35.8500,"lon":10.4500,"orientation":45,"type":"sandy"},
    ],
    "أريانة": [
        {"name":"شاطئ روّاد (الغدير)","lat":36.9833,"lon":10.1833,"orientation":45,"type":"sandy"},
        {"name":"شاطئ حي النصر","lat":36.9500,"lon":10.2000,"orientation":0,"type":"sandy"},
        {"name":"شاطئ قلعة الأندلس","lat":36.9167,"lon":10.1667,"orientation":0,"type":"sandy"},
        {"name":"شاطئ شط مروان","lat":36.9000,"lon":10.1500,"orientation":45,"type":"sandy"},
    ],
    "بن عروس": [
        {"name":"شاطئ رادس","lat":36.7500,"lon":10.2833,"orientation":0,"type":"sandy"},
        {"name":"شاطئ الزهراء","lat":36.7333,"lon":10.3000,"orientation":45,"type":"sandy"},
        {"name":"شاطئ حمام الأنف","lat":36.7167,"lon":10.3333,"orientation":0,"type":"sandy"},
        {"name":"شاطئ برج السدرية","lat":36.7000,"lon":10.3667,"orientation":45,"type":"sandy"},
        {"name":"شاطئ حمام الشط","lat":36.6833,"lon":10.3833,"orientation":90,"type":"sandy"},
    ],
}

# ---------- الطريقة الجديدة: البحث عن خط ساحلي في الاتجاه الأمامي ----------
async def is_direction_sea(lat, lon, angle_deg, distance_m=300):
    """
    تختبر إن كان الاتجاه angle_deg يشير إلى البحر،
    وذلك بالتحقق من عدم وجود خط ساحلي في نقطة تبعد distance_m متر.
    """
    # تحويل المسافة إلى درجات تقريبية
    dlat = distance_m / 111320.0  # متر لكل درجة عرض
    dlon = distance_m / (111320.0 * math.cos(math.radians(lat)))  # متر لكل درجة طول
    rad = math.radians(angle_deg)
    test_lat = lat + dlat * math.cos(rad)
    test_lon = lon + dlon * math.sin(rad)

    # استعلام عن خط ساحلي حول نقطة الاختبار في نطاق 200 متر
    query = f"""[out:json];(way(around:200,{test_lat},{test_lon})["natural"="coastline"];);out count;"""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(OVERPASS_URL, params={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=10)
            r.raise_for_status()
            data = r.json()
            count = len(data.get("elements", []))
            return count == 0  # True إذا لم يوجد خط ساحلي (بحر)
    except:
        return False  # في حالة الفشل، نعتبره ليس بحرًا (أمان)

# ---------- OSM مع الفحص الجديد ----------
async def get_osm_orientation(lat, lon):
    """
    تحدد اتجاه الشاطئ نحو البحر باستخدام أقرب خط ساحلي،
    مع اختبار كلا الاتجاهين (±90°) باستخدام is_direction_sea.
    """
    for radius in [200, 500, 1000, 2000, 5000]:
        query = f"""[out:json];(way(around:{radius},{lat},{lon})["natural"="coastline"];);out geom;"""
        try:
            async with httpx.AsyncClient() as c:
                r = await c.get(OVERPASS_URL, params={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=20)
                r.raise_for_status()
                data = r.json()
                elements = data.get("elements", [])
                if not elements: continue
                best_dist = float('inf')
                best_shore = None
                for el in elements:
                    geom = el.get("geometry", [])
                    if len(geom) < 2: continue
                    for i, p in enumerate(geom):
                        d = math.sqrt((p["lat"]-lat)**2 + (p["lon"]-lon)**2)
                        if d < best_dist:
                            best_dist = d
                            j = i+1 if i+1 < len(geom) else i-1
                            dx = geom[j]["lon"] - p["lon"]
                            dy = geom[j]["lat"] - p["lat"]
                            dlon = math.radians(dx) * math.cos(math.radians((p["lat"]+geom[j]["lat"])/2))
                            dlat = math.radians(dy)
                            shore_angle = (math.degrees(math.atan2(dlon, dlat)) + 360) % 360
                            best_shore = shore_angle
                if best_shore is None: continue

                # الاتجاهان المحتملان نحو البحر
                cand1 = (best_shore + 90) % 360
                cand2 = (best_shore - 90) % 360

                # اختبار كلا الاتجاهين
                sea1 = await is_direction_sea(lat, lon, cand1)
                sea2 = await is_direction_sea(lat, lon, cand2)

                if sea1 and not sea2:
                    return int(round(cand1))
                elif sea2 and not sea1:
                    return int(round(cand2))
                elif sea1 and sea2:
                    # كلاهما بحر – نأخذ cand1 (قاعدة OSM)
                    return int(round(cand1))
                else:
                    # لا هذا ولا ذاك – نأخذ cand1 مع تحذير
                    logger.warning("كلا الاتجاهين لم يُظهر بحرًا – استخدام cand1.")
                    return int(round(cand1))

        except Exception as e:
            logger.warning(f"محاولة Overpass بنطاق {radius}m فشلت: {e}")
            continue
    return None

async def get_bottom_type(lat, lon):
    query = f"""[out:json];(node(around:500,{lat},{lon})["surface"="sand"];node(around:500,{lat},{lon})["natural"="beach"];node(around:500,{lat},{lon})["surface"="gravel"];node(around:500,{lat},{lon})["surface"="rock"];);out body;"""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(OVERPASS_URL, params={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=15)
            r.raise_for_status()
            els = r.json().get("elements", [])
            if not els: return "sandy"
            for el in els:
                if "rock" in el.get("tags", {}).get("surface", "").lower(): return "rocky"
            return "sandy"
    except: return "sandy"

@app.post("/detect-bottom-type")
@limiter.limit("10/minute")
async def detect_bottom_type(request: Request, req: AutoOrientationRequest):
    return {"bottom_type": await get_bottom_type(req.latitude, req.longitude)}

@app.post("/auto-orientation")
@limiter.limit("5/minute")
async def auto_orientation(request: Request, req: AutoOrientationRequest):
    osm = await get_osm_orientation(req.latitude, req.longitude)
    if osm is not None:
        return {"orientation": osm, "source": "osm_verified"}

    # إذا فشل OSM تمامًا، لا نلجأ إلى قاعدة الشواطئ، بل نطلب الإدخال اليدوي
    return {"orientation": -1, "source": "none", "message": "تعذر التحديد التلقائي. الرجاء إدخال اتجاه الشاطئ يدويًا."}

# ---------- التجميع الفيزيائي (دون تغيير) ----------
def aggregate_physics(all_times, aligned, orient, target_date_obj, sunrise, sunset):
    tz = all_times[0].tzinfo if all_times else zoneinfo.ZoneInfo("UTC")
    target_start = datetime.combine(target_date_obj, datetime.min.time(), tzinfo=tz)
    target_end = target_start + timedelta(days=1)
    past_start = target_start - timedelta(hours=48)
    past_idx = [i for i, t in enumerate(all_times) if past_start <= t < target_start]
    target_idx = [i for i, t in enumerate(all_times) if target_start <= t < target_end]
    if not target_idx: return {"past_avg_power":0,"dominant_wind":"غير معروف","blocks":[],"red_flags":[],"green_flags":[],"weed_risk":False,"bio":{},"avg_sst":0,"extra_info":{}}
    def pick(k): return [aligned[k][i] for i in target_idx]
    wh = pick("wave_height"); wp = pick("wave_period"); swh = pick("swell_wave_height"); swp = pick("swell_wave_period"); swd = pick("swell_wave_direction")
    wd_wave = pick("wave_direction")
    sst = pick("sea_surface_temperature"); ws = pick("wind_speed_10m"); wd = pick("wind_direction_10m")
    wg = pick("wind_gusts_10m"); pr = pick("pressure_msl"); ta = pick("temperature_2m"); prec = pick("precipitation")
    wcode = [int(v) if v else 0 for v in pick("weather_code")]
    wave_power = [0.49*(h**2)*p for h,p in zip(wh,wp)]
    wind_cls = [wind_class_detailed(angle_diff(d, orient)) for d in wd]
    past_avg = 0.0; past_sh = 0.0
    if past_idx:
        past_avg = sum(0.49*(aligned["wave_height"][i]**2)*aligned["wave_period"][i] for i in past_idx)/len(past_idx)
        past_sh = sum(aligned["swell_wave_height"][i] for i in past_idx)/len(past_idx)
    weed = wind_cls[0].startswith("بحرية") and (past_sh > 0.8 or past_avg > 5.0) if target_idx else False
    peak_gust = max(wg) if wg else 0.0
    dominant = max(set(wind_cls), key=wind_cls.count) if wind_cls else "غير معروف"
    periods = defaultdict(list)
    for idx, i in enumerate(target_idx):
        h = all_times[i].hour
        if 4 <= h <= 11: periods["morning"].append(idx)
        elif 12 <= h <= 17: periods["afternoon"].append(idx)
        else: periods["night"].append(idx)
    blocks = []
    for key in ["morning", "afternoon", "night"]:
        idxs = periods[key]
        if not idxs: continue
        avg_h = sum(wh[i] for i in idxs)/len(idxs)
        min_h, max_h = min(wh[i] for i in idxs), max(wh[i] for i in idxs)
        avg_pow = sum(wave_power[i] for i in idxs)/len(idxs)
        avg_w = sum(ws[i] for i in idxs)/len(idxs)
        min_w, max_w = min(ws[i] for i in idxs), max(ws[i] for i in idxs)
        wc_dom = max(set(wind_cls[i] for i in idxs), key=wind_cls.count)
        avg_swh = sum(swh[i] for i in idxs)/len(idxs)
        avg_swp = sum(swp[i] for i in idxs)/len(idxs)
        avg_swd = sum(swd[i] for i in idxs)/len(idxs) if swd else 0
        avg_wave_dir = sum(wd_wave[i] for i in idxs)/len(idxs) if wd_wave else 0
        avg_air = sum(ta[i] for i in idxs)/len(idxs) if ta else 0
        total_precip = sum(prec[i] for i in idxs)
        most_code = max(set(wcode[i] for i in idxs), key=wcode.count) if idxs else 0
        swell_dom = "مختلط"
        if avg_swh > 0.7 * avg_h: swell_dom = "الطاقة أساساً قادمة من بعيد (swell قوي)"
        elif avg_h - avg_swh > 0.2: swell_dom = "الموج ناتج عن الرياح المحلية (wind sea)"
        wind_start = wind_cls[idxs[0]]; wind_end = wind_cls[idxs[-1]]
        wind_trend = f"تتحول من {wind_start} إلى {wind_end}" if wind_start != wind_end else f"ثابتة {wind_start}"
        sea = "هادئ" if avg_h < 0.3 else "متوسط الهيجان" if avg_h < 0.8 else "هائج"
        swell_dir_desc = deg_to_compass(avg_swd) if avg_swd else "غير معروف"
        wave_dir_desc = deg_to_compass(avg_wave_dir) if avg_wave_dir else "غير معروف"
        swell_angle = angle_diff(avg_swd, orient) if avg_swd else None
        wave_angle = angle_diff(avg_wave_dir, orient) if avg_wave_dir else None
        blocks.append({
            "name":{"morning":"الصباح","afternoon":"الظهر","night":"الليل"}[key],
            "time_range":f"{all_times[target_idx[idxs[0]]].strftime('%H:%M')}-{all_times[target_idx[idxs[-1]]].strftime('%H:%M')}",
            "sea_state":sea,"wave_height":f"{min_h:.2f}-{max_h:.2f}","wave_power":round(avg_pow,2),
            "swell_height":f"{min(swh[i] for i in idxs):.2f}-{max(swh[i] for i in idxs):.2f}",
            "swell_period":round(avg_swp,1),
            "swell_dir": swell_dir_desc,
            "swell_angle_diff": round(swell_angle,0) if swell_angle is not None else None,
            "wave_dir": wave_dir_desc,
            "wave_angle_diff": round(wave_angle,0) if wave_angle is not None else None,
            "swell_dominance":swell_dom,"wind_speed":f"{min_w:.1f}-{max_w:.1f}","wind_gust_peak":round(max(wg[i] for i in idxs),1),
            "wind_dir":wc_dom,"wind_trend":wind_trend,"air_temp":round(avg_air,1),"precip":round(total_precip,1),
            "weather":weather_desc(most_code)
        })
    reds, greens = [], []
    for i in range(len(wh)):
        hh = all_times[target_idx[i]].strftime("%H:%M")
        if wave_power[i] > 3 or wh[i] > 1.8 or wg[i] > 50 or pr[i] < 1005: reds.append(hh)
        if 0.3 <= wh[i] <= 1 and 0.1 <= wave_power[i] <= 1.5 and ws[i] < 27.8: greens.append(hh)
    avg_sst = sum(sst)/len(sst) if sst else 0
    avg_press = sum(pr)/len(pr) if pr else 0
    press_3h_change = pr[-1] - pr[0] if len(pr) > 1 else 0
    bio = {}
    if avg_sst < 16: bio["high"] = ["قاروص", "سارغ"]
    elif avg_sst > 19: bio["high"] = ["دوراد", "ماربري"]
    moon = moon_phase_detail(target_date_obj)
    moon_g = moon_fishing_guidance(target_date_obj)
    extra = {
        "pressure_avg":round(avg_press,1),
        "pressure_change_3h":round(press_3h_change,1),
        "sunrise":sunrise,"sunset":sunset,
        "moon_phase":moon["name"],"moon_status":moon["status"],"moon_guidance":moon_g,
        "peak_gust_today":round(peak_gust,1)
    }
    return {"past_avg_power":round(past_avg,2),"dominant_wind":dominant,"blocks":blocks,"red_flags":reds[:5],"green_flags":greens[:5],"weed_risk":weed,"bio":bio,"avg_sst":round(avg_sst,1),"extra_info":extra}

def deg_to_compass(deg):
    val = int((deg/22.5)+0.5)
    arr = ["شمال","شمال شمال شرق","شمال شرق","شرق شمال شرق","شرق","شرق جنوب شرق","جنوب شرق","جنوب جنوب شرق","جنوب","جنوب جنوب غرب","جنوب غرب","غرب جنوب غرب","غرب","غرب شمال غرب","شمال غرب","شمال شمال غرب"]
    return arr[val % 16]

def build_context(req, agg, tz_name):
    beach = "رملي" if req.beach_type == "sandy" else "صخري"
    moon = agg["extra_info"]
    orient = req.beach_orientation
    angle_summary = []
    for b in agg["blocks"]:
        sa = b.get("swell_angle_diff")
        wa = b.get("wave_angle_diff")
        sa_desc = "عمودي" if sa is not None and 70 <= sa <= 110 else "مائل"
        wa_desc = "عمودي" if wa is not None and 70 <= wa <= 110 else "مائل"
        angle_summary.append(f"{b['name']}: Swell {b['swell_dir']} ({sa_desc})، موج محلي {b['wave_dir']} ({wa_desc})")
    period_summary = []
    for b in agg["blocks"]:
        p = b.get("swell_period")
        if p:
            if p <= 4: desc = "سطحي (لا يقتلع حشيشاً)"
            elif p >= 7: desc = "عميق (خطر اقتلاع البوسيدونيا)"
            else: desc = "متوسط"
            period_summary.append(f"{b['name']}: {p}s ({desc})")
        else:
            period_summary.append(f"{b['name']}: دورة غير معروفة")
    wind_summary = []
    for b in agg["blocks"]:
        wind_summary.append(f"{b['name']}: {b['wind_dir']} {b['wind_speed']} كم/س (هبات {b['wind_gust_peak']})، {b['wind_trend']}")
    press_avg = moon.get("pressure_avg", 1013)
    press_change = moon.get("pressure_change_3h", 0)
    if press_change < -0.5:
        press_trend = "في انخفاض (الساعة الذهبية للصيد)"
    elif press_change > 0.5:
        press_trend = "في ارتفاع (الأسماك قد تكون خاملة)"
    else:
        press_trend = "مستقر"
    swell_ok = all(b.get("swell_height") and float(b["swell_height"].split("-")[1]) <= 1.3 for b in agg["blocks"])
    period_ok = all(b.get("swell_period") and b["swell_period"] >= 7 for b in agg["blocks"])
    wind_ok = all(b.get("wind_speed") and float(b["wind_speed"].split("-")[1]) <= 33 for b in agg["blocks"])
    pressure_ok = press_change <= 0
    score = sum([swell_ok, period_ok, wind_ok, pressure_ok])
    if score == 4:
        verdict = "مثالية (Go)"
    elif score == 3:
        verdict = "جيدة مع استعدادات (Go)"
    elif score == 2:
        verdict = "مقبولة بشروط (Go بحذر)"
    else:
        verdict = "سيئة (No-Go)"
    lines = [
        f"شاطئ {beach} (اتجاه {orient}°) - {req.target_date} - توقيت {tz_name}",
        f"حرارة الماء: {agg['avg_sst']}°م | القمر: {moon['moon_status']} ({moon['moon_phase']})",
        f"الشروق {moon['sunrise']} | الغروب {moon['sunset']}",
        f"الضغط الجوي: {press_avg} hPa ({press_trend})",
        f"خطر الأعشاب: {'مرتفع' if agg['weed_risk'] else 'منخفض'} | طاقة الموج الماضية: {agg['past_avg_power']} kW/m",
        "",
        "زوايا الموج:",
        *angle_summary,
        "",
        "دورة الموج:",
        *period_summary,
        "",
        "الرياح:",
        *wind_summary,
        "",
        f"أفضل ساعات الصيد: {', '.join(agg['green_flags']) if agg['green_flags'] else 'لا يوجد'}",
        f"ساعات الخطر: {', '.join(agg['red_flags']) if agg['red_flags'] else 'لا يوجد'}",
        f"الأسماك المتوقعة: {', '.join(agg['bio'].get('high', []))}",
        "",
        f"تقييم آلي أولي: {verdict} (الموج {'✓' if swell_ok else '✗'} | الدورة {'✓' if period_ok else '✗'} | الرياح {'✓' if wind_ok else '✗'} | الضغط {'✓' if pressure_ok else '✗'})",
        "",
        "المطلوب: تقرير تحليلي واحد يدمج هذه المعطيات، مع توصيات محددة (وزن الرصاصة، زاوية الرمي، الطعم، خطة طوارئ، سلامة) وقرار نهائي (Go/No‑Go). لا تسرد البيانات بل حللها واستنتج."
    ]
    return "\n".join(lines)

SYSTEM_PROMPT = """أنت محلل بحري تونسي محترف لصيد السرفكاستينغ. اكتب تقريراً تحليلياً واحداً بالدارجة التونسية. التقرير نص متصل بدون رموز.

ستتلقى معلومات موجزة عن زوايا الموج، دورته، الرياح، الضغط الجوي، وأفضل ساعات الصيد. مهمتك تحليل هذه المعلومات وربطها ببعضها لاستخلاص توصيات وقرار.

اتبع هذا الهيكل بدقة:

1. تحليل زوايا الموج وتأثيرها على التيار الجانبي وثبات الرصاص. اشرح كيف ستؤثر الزوايا المائلة أو العمودية على الحمل الجانبي وانجراف الطعم.

2. تحليل دورة الموج وخطر الأعشاب. اربط الدورة (بالثواني) بخطر اقتلاع الحشيش أو نظافة الماء.

3. تحليل الرياح: سرعتها، هباتها، اتجاهها. كيف ستؤثر على مسافة الرمي، دقته، وسلامة الصياد.

4. تحليل الضغط الجوي: اذكر قيمته وتغيره وتأثير ذلك على نشاط الأسماك.

5. توصيات محددة:
   - وزن ونوع الرصاصة (صابونة، هرم، قرابين) مع تعليل.
   - زاوية الرمي المثالية.
   - الطعم المناسب.
   - تحذيرات السلامة (خاصة إذا تجاوزت الهبات 30 كم/س أو الموج 1.5م).
   - خطة طوارئ.

6. قرار نهائي (Go/No‑Go) مع ذكر الأسباب بوضوح. إذا كان القرار Go، اذكر أفضل توقيت للخروج. إذا كان No‑Go، اذكر الأسباب بصراحة.

اكتب بلغة صياد خبير، واقعي وقاسٍ. لا تسرد المعلومات بل حللها. كل جملة تحمل معلومة جديدة. لا تذكر أنك تلقيت بيانات أو أنك ذكاء اصطناعي."""

async def call_openrouter(ctx):
    headers = {"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json"}
    payload = {"model":MODEL_NAME,"messages":[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":ctx}],"max_tokens":7000,"temperature":0.3}
    data = await post_with_retry(OPENROUTER_URL, payload, headers)
    if "choices" in data and data["choices"]: return data["choices"][0]["message"]["content"]
    raise Exception("OpenRouter فارغ")

@app.post("/generate-report")
@limiter.limit("10/minute")
async def generate_report(request: Request, req: RawDataReportRequest):
    try:
        marine_data = req.marine_data
        weather_data = req.weather_data
        marine_hourly = marine_data.get("hourly", marine_data)
        weather_hourly = weather_data.get("hourly", {})
        daily = weather_data.get("daily", {})
        tz_name = marine_data.get("timezone", "Africa/Tunis")
        now_tn = datetime.now(zoneinfo.ZoneInfo("Africa/Tunis"))
        target_dt = resolve_target_date(req.target_date, now_tn.date())
        sunrise = daily.get("sunrise", ["06:00"])[0] if daily.get("sunrise") else "06:00"
        sunset = daily.get("sunset", ["18:00"])[0] if daily.get("sunset") else "18:00"
        all_times, aligned = align_hourly_data(marine_hourly, weather_hourly, tz_name)
        if not all_times: raise HTTPException(500, "لا توجد بيانات ساعية متزامنة")
        agg = aggregate_physics(all_times, aligned, req.beach_orientation, target_dt, sunrise, sunset)
        ctx = build_context(req, agg, tz_name)
        report = await call_openrouter(ctx)
        return {"report": report, "meta": {"timezone": tz_name, "target_date": target_dt.isoformat()}}
    except HTTPException: raise
    except Exception as e:
        logger.error(f"generate-report error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, detail="فشل إنشاء التقرير")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
