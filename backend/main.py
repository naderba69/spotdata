"""
Surfcasting Analytics API – v9.7.1 (Hotfix: Endpoints & Fallbacks Restored)
"""
import os, math, asyncio, logging, traceback, zoneinfo
from datetime import datetime, timedelta, date
from typing import Dict, Optional
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
app = FastAPI(title="Surfcasting Analytics", version="9.7.1")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY: raise RuntimeError("OPENROUTER_API_KEY مفقود")
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
    marine_data: dict
    weather_data: dict

@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": "خطأ داخلي في الخادم"})

@app.get("/health")
def health(): return {"status": "ok", "version": "9.7.1"}

async def post_with_retry(url, json_data, headers, max_retries=3, timeout=120.0):
    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient() as c:
                r = await c.post(url, json=json_data, headers=headers, timeout=timeout)
                r.raise_for_status()
                return r.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries: await asyncio.sleep(5 * attempt); continue
            raise
        except (httpx.ConnectError, httpx.TimeoutException):
            if attempt < max_retries: await asyncio.sleep(2 ** attempt); continue
            raise

def safe_float(v):
    try: return 0.0 if math.isnan(float(v)) else float(v)
    except: return 0.0

def angle_diff(w, b):
    d = abs(w - b) % 360; return 360 - d if d > 180 else d

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

def get_max_val(range_str, fallback=0.0):
    if not range_str: return fallback
    try: return float(str(range_str).split("-")[-1])
    except: return fallback

def wind_class_detailed(diff):
    if diff < 30: return "بحرية مباشرة"
    if diff < 45: return "بحرية خفيفة"
    if diff < 60: return "جانبية مائلة للبحر"
    if diff <= 120: return "جانبية"
    if diff < 150: return "جانبية مائلة للبر"
    if diff < 165: return "برية خفيفة"
    return "برية مباشرة"

def deg_to_compass(deg):
    val = int((deg / 22.5) + 0.5) % 16
    arr = ["شمال","شمال شرق","شمال شرق","شرق","شرق","جنوب شرق","جنوب شرق","جنوب","جنوب","جنوب غرب","جنوب غرب","غرب","غرب","شمال غرب","شمال غرب","شمال"]
    return arr[val]

def resolve_target_date(txt, real_today):
    if txt == "today": return real_today
    if txt == "tomorrow": return real_today + timedelta(days=1)
    return real_today + timedelta(days=2)

def calc_wave_steepness(h, p):
    if not p or p <= 0: return 0.0
    return h / (1.56 * (p ** 2))

def check_freshwater_risk(past_idx, precip_array):
    if not past_idx: return "منخفض"
    valid = [i for i in past_idx if i < len(precip_array)]
    if not valid: return "منخفض"
    total_rain = sum(precip_array[i] for i in valid)
    if total_rain > 15.0: return "مرتفع جداً (سيول ومياه عذبة طافية تغطي السطح وتمنع الأكسجين والرؤية)"
    if total_rain > 5.0: return "مرتفع (طبقة مياه عذبة خفيفة على السطح قد تربك الأسماك السطحية)"
    return "منخفض"

def check_stratification_risk(past_idx, past_swh_array, past_ws_array):
    if not past_idx: return "منخفض"
    valid = [i for i in past_idx if i < len(past_swh_array) and i < len(past_ws_array)]
    if not valid: return "منخفض"
    avg_swh = sum(past_swh_array[i] for i in valid) / len(valid)
    avg_ws = sum(past_ws_array[i] for i in valid) / len(valid)
    if avg_swh < 0.2 and avg_ws < 10.0: return "مرتفع (بحر مسطح لفترة طويلة، غياب اختلاط المياه يجعل القاع يفقد الأكسجين والسمك يصبح خاملاً)"
    return "منخفض"

def get_moon_and_tide_analysis(d: date):
    y, m, day = d.year, d.month, d.day
    if m < 3: y-=1; m+=12
    a = int(y/100); b = 2-a+int(a/4)
    jd = int(365.25*(y+4716)) + int(30.6001*(m+1)) + day + b - 1524.5
    days_since_new = jd - 2451550.1
    phase = (days_since_new % 29.53058867) / 29.53058867
    idx = int(phase * 8) % 8
    names = {0:"محاق",1:"هلال أول",2:"تربيع أول",3:"أحدب متزايد",4:"بدر",5:"أحدب متناقص",6:"تربيع ثاني",7:"هلال آخر"}
    if idx in [0, 4]: tide_strength = "مد وجزر قوي جداً (Spring Tides)"
    elif idx in [2, 6]: tide_strength = "مد وجزر ضعيف (Neap Tides)"
    else: tide_strength = "مد وجزر متوسط"
    return {"name": names[idx], "tide_strength": tide_strength, "idx": idx}

def align_hourly_data(marine_hourly, weather_hourly, tz_name):
    tz = zoneinfo.ZoneInfo(tz_name)
    m_times = marine_hourly.get("time", [])
    w_times = weather_hourly.get("time", [])
    if not m_times or not w_times: return [], {}
    m_map, w_map = {}, {}
    for i, t in enumerate(m_times):
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=tz)
        m_map[dt.replace(minute=0, second=0, microsecond=0)] = i
    for i, t in enumerate(w_times):
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=tz)
        w_map[dt.replace(minute=0, second=0, microsecond=0)] = i
    common = sorted(set(m_map) & set(w_map))
    if not common: return [], {}
    def extract(key, src, idx_map):
        arr = src.get(key, [])
        return [arr[idx_map[t]] if arr and idx_map[t] < len(arr) else 0.0 for t in common]
    return common, {
        "wave_height": extract("wave_height", marine_hourly, m_map), "wave_period": extract("wave_period", marine_hourly, m_map),
        "wave_direction": extract("wave_direction", marine_hourly, m_map), "swell_wave_height": extract("swell_wave_height", marine_hourly, m_map),
        "swell_wave_period": extract("swell_wave_period", marine_hourly, m_map), "swell_wave_direction": extract("swell_wave_direction", marine_hourly, m_map),
        "sea_surface_temperature": extract("sea_surface_temperature", marine_hourly, m_map), "wind_speed_10m": extract("wind_speed_10m", weather_hourly, w_map),
        "wind_direction_10m": extract("wind_direction_10m", weather_hourly, w_map), "wind_gusts_10m": extract("wind_gusts_10m", weather_hourly, w_map),
        "pressure_msl": extract("pressure_msl", weather_hourly, w_map), "temperature_2m": extract("temperature_2m", weather_hourly, w_map),
        "precipitation": extract("precipitation", weather_hourly, w_map), "weather_code": [int(safe_float(x)) for x in extract("weather_code", weather_hourly, w_map)]
    }

# ==================== قاعدة بيانات الشواطئ الكاملة (Fallback) ====================
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

def find_nearest_beach_orientation(lat: float, lon: float) -> Optional[int]:
    min_dist = float('inf')
    nearest_orient = None
    for gov, beaches in TUNISIAN_BEACHES.items():
        for b in beaches:
            dist = calc_distance(b["lat"], b["lon"], lat, lon)
            if dist < min_dist:
                min_dist = dist
                nearest_orient = b["orientation"]
    return nearest_orient

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

async def get_auto_orientation_overpass(lat, lon):
    query = f"""[out:json];(way(around:3000,{lat},{lon})["natural"="coastline"];);out geom;"""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(OVERPASS_URL, params={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=20)
            r.raise_for_status()
            els = r.json().get("elements", [])
            if not els: return 0
            best_dist, best_tangent, best_point = float('inf'), None, None
            for el in els:
                geom = el.get("geometry", [])
                if len(geom) < 2: continue
                for i in range(len(geom)):
                    p = geom[i]
                    d = calc_distance(lat, lon, p["lat"], p["lon"])
                    if d < best_dist:
                        best_dist, best_point = d, p
                        prev_i, next_i = max(0, i - 1), min(len(geom) - 1, i + 1)
                        if prev_i != next_i:
                            best_tangent = calc_bearing(geom[prev_i]["lat"], geom[prev_i]["lon"], geom[next_i]["lat"], geom[next_i]["lon"])
            if not best_tangent or not best_point: return 0
            n_a, n_b = (best_tangent + 90) % 360, (best_tangent - 90) % 360
            c2u = calc_bearing(best_point["lat"], best_point["lon"], lat, lon)
            d_a = abs(c2u - n_a); d_a = 360 - d_a if d_a > 180 else d_a
            d_b = abs(c2u - n_b); d_b = 360 - d_b if d_b > 180 else d_b
            return int(round(((n_a if d_a < d_b else n_b) + 180) % 360))
    except Exception as e:
        logger.warning(f"Overpass API failed (Render might be blocked): {e}")
        return 0

# ==================== Endpoints الخاصة بالشاطئ (مُصلحة) ====================
@app.post("/detect-bottom-type")
@limiter.limit("10/minute")
async def detect_bottom_type(request: Request, req: AutoOrientationRequest):
    return {"bottom_type": await get_bottom_type(req.latitude, req.longitude)}

@app.post("/auto-orientation")
@limiter.limit("5/minute")
async def auto_orientation(request: Request, req: AutoOrientationRequest):
    # 1. محاولة الحساب الدقيق عبر OSM
    orientation = await get_auto_orientation_overpass(req.latitude, req.longitude)
    if orientation != 0:
        return {"orientation": orientation, "source": "osm_verified"}
    
    # 2. في حالة فشل OSM (شائع على السحابة)، نلجأ لقاعدة البيانات المحلية
    logger.info("OSM failed, falling back to local beach database...")
    orientation = find_nearest_beach_orientation(req.latitude, req.longitude)
    if orientation is not None:
        return {"orientation": orientation, "source": "nearest_beach_db"}
        
    # 3. فشل كامل
    return {"orientation": -1, "source": "none", "message": "تعذر التحديد التلقائي. الرجاء إدخال الاتجاه يدوياً."}

# ==================== محرك التوليف الفيزيائي والبيئي الشامل ====================
def aggregate_physics(all_times, aligned, orient, beach_type, target_date_obj, sunrise, sunset):
    tz = all_times[0].tzinfo if all_times else zoneinfo.ZoneInfo("UTC")
    target_start = datetime.combine(target_date_obj, datetime.min.time(), tzinfo=tz)
    target_end = target_start + timedelta(days=1)
    past_start = target_start - timedelta(hours=48)
    past_idx = [i for i, t in enumerate(all_times) if past_start <= t < target_start]
    target_idx = [i for i, t in enumerate(all_times) if target_start <= t < target_end]
    
    empty_res = {"sea_memory":"غير معروف","lateral_current":"غير معروف","bottom_energy":"منخفض","pressure_state":"مستقر","tide_analysis":{},"sst_stability":"مستقر","bio_matrix":{},"avg_sst":0,"hidden_factors":{},"blocks":[],"red_flags":[],"green_flags":[],"extra_info":{}}
    if not target_idx: return empty_res
    
    def pick(k): 
        arr = aligned.get(k, [])
        return [arr[i] if i < len(arr) else 0.0 for i in target_idx]
        
    wh = pick("wave_height"); wp = pick("wave_period"); swh = pick("swell_wave_height")
    swp = pick("swell_wave_period"); swd = pick("swell_wave_direction"); wd_wave = pick("wave_direction")
    sst = pick("sea_surface_temperature"); ws = pick("wind_speed_10m"); wd = pick("wind_direction_10m")
    wg = pick("wind_gusts_10m"); pr = pick("pressure_msl"); ta = pick("temperature_2m"); prec = pick("precipitation")
    wcode = [int(v) if v else 0 for v in pick("weather_code")]
    
    wind_cls = [wind_class_detailed(angle_diff(d, orient)) for d in wd]
    wave_power = [0.49*(h**2)*p for h,p in zip(wh,wp)]
    
    sea_memory = "بحر صافي وهادئ (لا توجد عوامل تعكير سابقة)"
    if past_idx:
        p_wh = aligned.get("wave_height", []); p_wp = aligned.get("wave_period", [])
        p_swh = aligned.get("swell_wave_height", []); p_swp = aligned.get("swell_wave_period", [])
        p_ws = aligned.get("wind_speed_10m", []); p_wd = aligned.get("wind_direction_10m", [])
        valid_past = [i for i in past_idx if i < len(p_wh) and i < len(p_wp) and i < len(p_ws) and i < len(p_wd)]
        
        if valid_past:
            past_power = sum(0.49*(p_wh[i]**2)*p_wp[i] for i in valid_past) / len(valid_past)
            past_onshore_hours = sum(1 for i in valid_past if wind_class_detailed(angle_diff(p_wd[i], orient)).startswith("بحرية"))
            past_onshore_ratio = past_onshore_hours / len(valid_past)
            past_swp_avg = sum(p_swp[i] for i in valid_past if i < len(p_swp)) / len(valid_past)
            past_swh_avg = sum(p_swh[i] for i in valid_past if i < len(p_swh)) / len(valid_past)
            
            if past_power > 6.0 and past_onshore_ratio > 0.4: sea_memory = "بحر خامر وعكر جداً (طونى): رياح بحرية قوية في اليومين الماضيين قلبت القاع وخلطت المياه بالرمل والطين. الرؤية شبه معدومة."
            elif past_power > 4.0 and past_onshore_ratio > 0.3: sea_memory = "بحر يعكر ببطء: رياح بحرية متوسطة خلقت طبقة عكرة قريبة من الشاطئ."
            
            if beach_type == "rocky" and past_swp_avg > 7.0 and past_swh_avg > 0.8: sea_memory += " | تحذير صوفة: الأمواج الطويلة الماضية اقتلعت أعشاب البوسيدونيا من القاع الصخري."
            elif beach_type == "sandy" and past_power > 5.0: sea_memory += " | توقع طحالب رملية محمولة جواً بسبب هيجان الموج الماضي."

    valid_wd = [angle_diff(w, orient) for w in wd_wave if w != 0]
    avg_wave_angle = sum(valid_wd) / len(valid_wd) if valid_wd else 90
    lateral_force = math.sin(math.radians(avg_wave_angle))
    avg_wave_h = sum(wh) / len(wh) if wh else 0
    
    if lateral_force > 0.8 and avg_wave_h > 0.6: lateral_current = "تيار جارف قوي جداً (موازي للشاطئ): الرصاصة ستنجرف بسرعة كبيرة، والخط سيصبح قوساً. يتطلب رصاص ثقيل جداً أو تغيير زاوية الرمي عكس التيار بـ 30-40 درجة."
    elif lateral_force > 0.5 and avg_wave_h > 0.4: lateral_current = "تيار جانبي متوسط: سيحدث انجراف تدريجي للطعم. يجب مراقبة خيط الخط وتعديل الثقل."
    else: lateral_current = "تيار جانبي ضعيف أو معدوم: الموج يدفع للخلف وللأمام (عمودي)، الرصاصة ستثبت جيداً في القاع دون انجراف عرضي."

    max_swp = max(swp) if swp else 0
    max_wh = max(wh) if wh else 0
    if max_swp >= 8 and max_wh > 0.8: bottom_energy = "قوي جداً: مواج طويلة ستضرب القاع بقوة وتقتلع أي أعشاب متبقية وتجرفها نحو الشاطئ (خطر مادام للصيد)."
    elif max_swp >= 6 and max_wh > 0.6: bottom_energy = "متوسط: احتمال وجود بعض الأعشاب المحمولة في أعماق مختلفة."
    else: bottom_energy = "ضعيف: الطاقة سطحية، القاع مستقر."

    avg_press = sum(pr)/len(pr) if pr else 1013
    press_change = (pr[-1] if pr else 1013) - (pr[0] if pr else 1013)
    if press_change < -2.0: pressure_state = "انخفاض حاد ومستمر: الأسماك تدرك العاصفة وتتغذى بشراهة. فرصة ذهبية لكن السلامة مهددة."
    elif press_change < -0.5: pressure_state = "انخفاض بطيء: نشاط جيد ومستقر للأسماك."
    elif press_change > 1.5: pressure_state = "ارتفاع حاد: الأسماك تمتلئ هواء وتتوقف عن الأكل لفترة."
    else: pressure_state = "مستقر أو شبه مستقر: لا تأثير مباشر، العوامل الأخرى حاكمة."

    sst_diff = max(sst) - min(sst) if len(sst) > 1 else 0
    sst_stability = "صدمة حرارية (تغير > 2 درجة)" if sst_diff > 2.0 else "تغير ملحوظ" if sst_diff > 1.0 else "مستقر تماماً"
    avg_sst = sum(sst)/len(sst) if sst else 0

    hidden_factors = {
        "freshwater_risk": check_freshwater_risk(past_idx, aligned.get("precipitation", [])),
        "stratification_risk": check_stratification_risk(past_idx, aligned.get("swell_wave_height", []), aligned.get("wind_speed_10m", []))
    }
    steepness_values = [calc_wave_steepness(h, p) for h, p in zip(wh, wp) if p > 0]
    avg_steepness = sum(steepness_values) / len(steepness_values) if steepness_values else 0
    if avg_steepness > 0.06: hidden_factors["wave_steepness"] = "موج حاد وقصير (Steep). ينكسر بقوة، يخلق ماءاً أبيض كثيفاً. سيء للرمي البعيد."
    elif avg_steepness < 0.03: hidden_factors["wave_steepness"] = "موج طويل وهادئ (Swell). ينكسر بعيداً ويخلق خنادق طبيعية. ممتاز."
    else: hidden_factors["wave_steepness"] = "موج متوسط الانحدار."

    tide_analysis = get_moon_and_tide_analysis(target_date_obj)
    if tide_analysis["idx"] in [0, 4]: hidden_factors["golden_lock"] = "مد قوي (Spring). احتمال تزامن الشروق مع مد قوي."
    elif tide_analysis["idx"] in [2, 6]: hidden_factors["golden_lock"] = "مد ضعيف (Neap). مستوى الماء عند الشروق سيكون متوسطاً."
    else: hidden_factors["golden_lock"] = "متوسط."

    is_murky = "عكر" in sea_memory or "خامر" in sea_memory
    is_weedy = "صوفة" in sea_memory or "قوي جداً" in bottom_energy
    is_freshwater_risk = "مرتفع" in hidden_factors["freshwater_risk"]
    is_stratified = "مرتفع" in hidden_factors["stratification_risk"]

    bio_matrix = {
        "قاروص": {"viability": "معدومة" if is_stratified else "ممتازة" if (avg_sst < 18 and is_murky and not is_weedy) else "جيدة" if avg_sst < 19 else "ضعيفة", "reason": "يكره البحر المسطح الميت. يستفيد من العكر لكن الأعشاب تدمر خطوطه."},
        "دوراد": {"viability": "شبه معدومة" if (is_murky or is_freshwater_risk) else "ممتازة" if (avg_sst > 18) else "مقبولة", "reason": "يهرب من المياه العذبة والعكرة تماماً. أصعب سمكة في أيام الأمطار."},
        "بوري": {"viability": "معدومة" if (is_weedy or is_freshwater_risk) else "جيدة جداً" if (not is_murky and avg_sst > 15) else "ضعيفة", "reason": "الأوساخ العالقة تفقده قدرته على رؤية الطعم السطحي."},
        "سارغ": {"viability": "معدومة" if is_stratified else "ممكنة" if (avg_sst < 22 and not is_weedy) else "ضعيفة", "reason": "يتأثر بالحرارة والأعشاب العالقة قرب الصخور."}
    }

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
        
        blocks.append({
            "name":{"morning":"الصباح","afternoon":"الظهر","night":"الليل"}[key],
            "time_range":f"{all_times[target_idx[idxs[0]]].strftime('%H:%M')}-{all_times[target_idx[idxs[-1]]].strftime('%H:%M')}",
            "wave_height":f"{min_h:.2f}-{max_h:.2f}", "wave_power":round(avg_pow,2),
            "swell_height":f"{min(swh[i] for i in idxs):.2f}-{max(swh[i] for i in idxs):.2f}",
            "swell_period":round(avg_swp,1), "swell_dir": deg_to_compass(avg_swd),
            "swell_angle_diff": round(angle_diff(avg_swd, orient),0) if avg_swd else None,
            "wave_dir": deg_to_compass(avg_wave_dir),
            "wave_angle_diff": round(angle_diff(avg_wave_dir, orient),0) if avg_wave_dir else None,
            "wind_speed":f"{min_w:.1f}-{max_w:.1f}", "wind_gust_peak":round(max(wg[i] for i in idxs),1), "wind_dir":wc_dom
        })
        
    reds, greens = [], []
    for i in range(len(wh)):
        hh = all_times[target_idx[i]].strftime("%H:%M")
        if wh[i] > 1.8 or wg[i] > 50 or pr[i] < 1000: reds.append(hh)
        if 0.3 <= wh[i] <= 1.0 and wave_power[i] <= 1.5 and ws[i] < 27.8: greens.append(hh)
        
    return {
        "dominant_wind":dominant, "blocks":blocks, "red_flags":reds[:5], "green_flags":greens[:5], 
        "sea_memory":sea_memory, "lateral_current":lateral_current, "bottom_energy":bottom_energy,
        "pressure_state":pressure_state, "tide_analysis":tide_analysis, "sst_stability":sst_stability,
        "bio_matrix":bio_matrix, "avg_sst":round(avg_sst,1), "hidden_factors":hidden_factors,
        "extra_info": {"pressure_avg":round(avg_press,1), "pressure_change":round(press_change,1), "sunrise":sunrise, "sunset":sunset, "peak_gust_today":round(peak_gust,1)}
    }

def build_context(req, agg, tz_name):
    beach = "رملي" if req.beach_type == "sandy" else "صخري"
    bio_text = "\n".join([f"- {fish}: {data['viability']} ({data['reason']})" for fish, data in agg["bio_matrix"].items()])
    blocks_text = "\n".join([f"  * {b['name']} ({b['time_range']}): موج {b['wave_height']}م (دورة {b['swell_period']}ث)، رياح {b['wind_dir']} {b['wind_speed']}كم/س (هبات {b['wind_gust_peak']})" for b in agg["blocks"]])

    lines = [
        "أنت محلل فيزيائي وبيئي للسيرفكاستينغ. قم بتوليف المعطيات التالية في تقرير تحليلي واحد.",
        "",
        f"=== 1. ذاكرة البحر (حالة المياه نتيجة الـ 48 ساعة الماضية) ===",
        agg["sea_memory"],
        "",
        f"=== 2. ميكانيكا الموائع وتحرك المعدات ===",
        f"التيار الجانبي وانجراف الرصاصة: {agg['lateral_current']}",
        f"طاقة القاع والأعشاب: {agg['bottom_energy']}",
        "",
        f"=== 3. ديناميكيات الضغط الجوي والتمثيل الغذائي ===",
        agg["pressure_state"],
        "",
        f"=== 4. المعطيات الفيزيائية لفترات اليوم ===",
        blocks_text,
        f"القمر والمد: {agg['tide_analysis']['name']} ({agg['tide_analysis']['tide_strength']}).",
        f"حرارة الماء: {agg['avg_sst']}°م (الاستقرار: {agg['sst_stability']}).",
        "",
        f"=== 5. الجدوى البيولوجية للأنواع المستهدفة ===",
        bio_text,
        "",
        f"=== 6. العوامل الخفية (The Hidden Killers) ===",
        f"خطر السيول والمياه العذبة: {agg['hidden_factors'].get('freshwater_risk', 'غير معروف')}",
        f"خطر انعدام التمازج (البحر الميت تحت السطح): {agg['hidden_factors'].get('stratification_risk', 'غير معروف')}",
        f"ميكانيكا الموج وانحداره: {agg['hidden_factors'].get('wave_steepness', 'غير معروف')}",
        f"تزامن الشروق مع المد: {agg['hidden_factors'].get('golden_lock', 'غير معروف')}",
        "",
        f"=== 7. التوقيتات الحرجة ===",
        f"أفضل ساعات (خضراء): {', '.join(agg['green_flags']) if agg['green_flags'] else 'لا يوجد'}",
        f"ساعات الخطر (حمراء): {', '.join(agg['red_flags']) if agg['red_flags'] else 'لا يوجد'}",
    ]
    return "\n".join(lines)

SYSTEM_PROMPT = """أنت عالم أحياء بحرية ومهندس ميكانيكا موائع، ومحلل خبرة تونسي في السيرفكاستينغ. 
مطلوب منك كتابة تقرير تحليلي مطول، مفصل، ومترابط جداً بالدارجة التونسية.

قواعد كتابة التقرير (يجب اتباعها بحذافير):
1. التفصيل المعمق: لا تكتب جمل قصيرة. لكل ظاهرة فيزيائية، اكتب فقرة كاملة تشرح "لماذا" و"كيف" تؤثر على ميكانيكة الصيد وسلوك السمك.
2. الترابط الحتمي (The Chain Reaction): لا تناقش أي عامل بشكل منفصل. يجب أن تربط ذاكرة البحر بالضغط، والضغط بالرياح، والرياح بالتيار الجانبي، والتيار بانجراف الرصاصة، وانجراف الرصاصة بفشل أو نجاح الرمية.
3. ممنوع النقل الحرفي: لا تنقل الأرقام من المعطيات. حولها لوصف فيزيائي (مثلاً: بدلاً من "الضغط 1008"، قل "الضغط المنخفض الذي يسبق الجبهة الهوائية").
4. الهيكلة الإلزامية للتقرير (اكتب بهذا الترتيب بالضبط):
   - مقدمة سلوكية: ماذا يفعل البحر اليوم ككائن حي؟ (اربط الذاكرة بالعوامل الخفية).
   - التحليل الميكانيكي للموج والتيار: اشرح كيف ستتحرك الرصاصة وكيف سيتشكل القاع بناءً على انحدار الموج والتيار الجانبي.
   - التحليل البيئي للأنواع: بناءً على الحالة الفيزيائية، اشرح أين تختبئ الأسماك ولماذا.
   - التوصيات التقنية الدقيقة: وزن الرصاصة (مع التعليل الفيزيائي)، نوعها، زاوية الرمي (مع التعليل)، والطعم (مع التعليل بناءً على رؤية الماء).
   - الاستنتاج النهائي (القرار): ابدأ الفقرة بـ "القرار النهائي:" ثم اكتب (Go / No-Go / Conditional Go) متبوعاً بملخص جملتين يربط كل شيء ببعض."""

async def call_openrouter(ctx):
    headers = {"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json"}
    payload = {"model":MODEL_NAME,"messages":[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":ctx}],"max_tokens":8000,"temperature":0.65}
    data = await post_with_retry(OPENROUTER_URL, payload, headers)
    if "choices" in data and data["choices"]: return data["choices"][0]["message"]["content"]
    raise Exception("OpenRouter استجابة فارغة")

@app.post("/generate-report")
@limiter.limit("10/minute")
async def generate_report(request: Request, req: RawDataReportRequest):
    try:
        marine_hourly = req.marine_data.get("hourly", req.marine_data)
        weather_hourly = req.weather_data.get("hourly", {})
        daily = req.weather_data.get("daily", {})
        tz_name = req.marine_data.get("timezone", "Africa/Tunis")
        now_tn = datetime.now(zoneinfo.ZoneInfo("Africa/Tunis"))
        target_dt = resolve_target_date(req.target_date, now_tn.date())
        sunrise = daily.get("sunrise", ["06:00"])[0] if daily.get("sunrise") else "06:00"
        sunset = daily.get("sunset", ["18:00"])[0] if daily.get("sunset") else "18:00"
        
        all_times, aligned = align_hourly_data(marine_hourly, weather_hourly, tz_name)
        if not all_times: raise HTTPException(500, "لا توجد بيانات ساعية متزامنة")
        
        agg = aggregate_physics(all_times, aligned, req.beach_orientation, req.beach_type, target_dt, sunrise, sunset)
        
        if agg["extra_info"]["peak_gust_today"] > 60 or any(get_max_val(b.get("wave_height")) > 2.5 for b in agg["blocks"]):
            return {"report": "قرار نهائي: No-Go مطلق. ظروف بحرية خطرة تهدد حياتك مباشرة (هبات قاسية أو موج متطرف). لا تحاول المخاطرة.", "meta": {"hard_nogo": True}}

        ctx = build_context(req, agg, tz_name)
        report = await call_openrouter(ctx)
        return {"report": report, "meta": {"timezone": tz_name, "target_date": target_dt.isoformat(), "hard_nogo": False}}
    except HTTPException: raise
    except Exception as e:
        logger.error(f"generate-report error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, detail="فشل إنشاء التقرير")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
