"""
Surfcasting Analytics API – v8.1-Final (Stable & Mathematically Correct, Improved Report)
"""
import os, math, asyncio, logging, traceback, zoneinfo
from datetime import datetime, timedelta, date
from typing import Dict, Optional, Tuple
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
app = FastAPI(title="Surfcasting Analytics", version="8.1.0")
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

# ==================== النماذج (Pydantic) ====================
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
def health():
    return {"status": "ok", "openrouter": bool(OPENROUTER_API_KEY), "model": MODEL_NAME, "mode": "client-side"}

# ==================== أدوات الشبكة ====================
async def post_with_retry(url: str, json_data: dict, headers: dict, max_retries: int = 3, timeout: float = 120.0) -> dict:
    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=json_data, headers=headers, timeout=timeout)
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

# ==================== دوال مساعدة رياضية وجغرافية ====================
def safe_float(v):
    try: return 0.0 if math.isnan(float(v)) else float(v)
    except: return 0.0

def angle_diff(w, b):
    d = abs(w - b) % 360
    return 360 - d if d > 180 else d

def calc_bearing(lat1, lon1, lat2, lon2) -> float:
    """حساب الاتجاه الصحيح من نقطة لأخرى (0=شمال، 90=شرق)"""
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlon = lon2_r - lon1_r
    x = math.sin(dlon) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def calc_distance(lat1, lon1, lat2, lon2) -> float:
    """حساب المسافة التقريبية بالمتر"""
    dlat = (lat2 - lat1) * 111320
    dlon = (lon2 - lon1) * 111320 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.sqrt(dlat**2 + dlon**2)

def get_max_val(range_str, fallback=0.0) -> float:
    """استخراج القيمة العظمى من نص مثل '0.5-1.2' بأمان تام"""
    if not range_str: return fallback
    try:
        return float(str(range_str).split("-")[-1])
    except: return fallback

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

def deg_to_compass(deg):
    val = int((deg / 22.5) + 0.5) % 16
    arr = ["شمال","شمال شمال شرق","شمال شرق","شرق شمال شرق","شرق","شرق جنوب شرق","جنوب شرق","جنوب جنوب شرق","جنوب","جنوب جنوب غرب","جنوب غرب","غرب جنوب غرب","غرب","غرب شمال غرب","شمال غرب","شمال شمال غرب"]
    return arr[val]

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
    if phase < 0.125 or phase > 0.875: status, activity = "أيام متوسطة (محاق / هلال آخر)", "ينشط القاع ليلاً بشكل خاص"
    elif phase < 0.5: status, activity = "أيام حمل", "الأسماك نشيطة طوال اليوم"
    else: status, activity = "أيام فساد", "الأسماك أقل نشاطاً"
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

# ==================== مزامنة بيانات الطقس ====================
def align_hourly_data(marine_hourly, weather_hourly, tz_name):
    tz = zoneinfo.ZoneInfo(tz_name)
    m_times = marine_hourly.get("time", [])
    w_times = weather_hourly.get("time", [])
    if not m_times or not w_times: return [], {}
    
    m_map, w_map = {}, {}
    for i, t in enumerate(m_times):
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=tz)
        dt = dt.replace(minute=0, second=0, microsecond=0) # إصلاح عدم تطابق الثواني
        m_map[dt] = i
        
    for i, t in enumerate(w_times):
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=tz)
        dt = dt.replace(minute=0, second=0, microsecond=0) # إصلاح عدم تطابق الثواني
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

# ==================== قاعدة بيانات الشواطئ (Fallback) ====================
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

# ==================== OSM: التحديد التلقائي الصحيح ====================
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
    """حساب دقيق لاتجاه الشاطئ نحو البحر بناءً على المماس المحلي"""
    query = f"""[out:json];(way(around:3000,{lat},{lon})["natural"="coastline"];);out geom;"""
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(OVERPASS_URL, params={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=20)
            r.raise_for_status()
            els = r.json().get("elements", [])
            
            if not els: return 0
            
            best_dist = float('inf')
            best_tangent = None
            best_point = None
            
            for el in els:
                geom = el.get("geometry", [])
                if len(geom) < 2: continue
                
                for i in range(len(geom)):
                    p = geom[i]
                    d = calc_distance(lat, lon, p["lat"], p["lon"])
                    
                    if d < best_dist:
                        best_dist = d
                        best_point = p
                        
                        prev_i = max(0, i - 1)
                        next_i = min(len(geom) - 1, i + 1)
                        
                        if prev_i != next_i:
                            p_prev = geom[prev_i]
                            p_next = geom[next_i]
                            best_tangent = calc_bearing(p_prev["lat"], p_prev["lon"], p_next["lat"], p_next["lon"])
            
            if best_tangent is None or best_point is None:
                return 0
                
            normal_a = (best_tangent + 90) % 360
            normal_b = (best_tangent - 90) % 360
            
            coast_to_user = calc_bearing(best_point["lat"], best_point["lon"], lat, lon)
            
            diff_a = abs(coast_to_user - normal_a)
            diff_a = 360 - diff_a if diff_a > 180 else diff_a
            diff_b = abs(coast_to_user - normal_b)
            diff_b = 360 - diff_b if diff_b > 180 else diff_b
            
            if diff_a < diff_b:
                sea_direction = (normal_a + 180) % 360
            else:
                sea_direction = (normal_b + 180) % 360
                
            return int(round(sea_direction))
            
    except Exception as e:
        logger.error(f"Overpass error: {e}")
        return 0

# ==================== Endpoints ====================
@app.post("/detect-bottom-type")
@limiter.limit("10/minute")
async def detect_bottom_type(request: Request, req: AutoOrientationRequest):
    return {"bottom_type": await get_bottom_type(req.latitude, req.longitude)}

@app.post("/auto-orientation")
@limiter.limit("5/minute")
async def auto_orientation(request: Request, req: AutoOrientationRequest):
    orientation = await get_auto_orientation_overpass(req.latitude, req.longitude)
    if orientation != 0:
        return {"orientation": orientation, "source": "overpass"}
    
    orientation = find_nearest_beach_orientation(req.latitude, req.longitude)
    if orientation is not None:
        return {"orientation": orientation, "source": "nearest_beach"}
        
    return {"orientation": -1, "source": "none", "message": "لم نتمكن من تحديد الاتجاه تلقائياً."}

# ==================== التجميع الفيزيائي الآمن ====================
def aggregate_physics(all_times, aligned, orient, target_date_obj, sunrise, sunset):
    tz = all_times[0].tzinfo if all_times else zoneinfo.ZoneInfo("UTC")
    target_start = datetime.combine(target_date_obj, datetime.min.time(), tzinfo=tz)
    target_end = target_start + timedelta(days=1)
    past_start = target_start - timedelta(hours=48)
    
    past_idx = [i for i, t in enumerate(all_times) if past_start <= t < target_start]
    target_idx = [i for i, t in enumerate(all_times) if target_start <= t < target_end]
    
    empty_res = {"past_avg_power":0,"dominant_wind":"غير معروف","blocks":[],"red_flags":[],"green_flags":[],"weed_risk":False,"bio":{"high":[]},"avg_sst":0,"extra_info":{}}
    if not target_idx: return empty_res
    
    # إصلاح انهيار الفهرس خارج الحدود
    def pick(k): 
        arr = aligned.get(k, [])
        return [arr[i] if i < len(arr) else 0.0 for i in target_idx]
        
    wh = pick("wave_height"); wp = pick("wave_period"); swh = pick("swell_wave_height")
    swp = pick("swell_wave_period"); swd = pick("swell_wave_direction"); wd_wave = pick("wave_direction")
    sst = pick("sea_surface_temperature"); ws = pick("wind_speed_10m"); wd = pick("wind_direction_10m")
    wg = pick("wind_gusts_10m"); pr = pick("pressure_msl"); ta = pick("temperature_2m"); prec = pick("precipitation")
    wcode = [int(v) if v else 0 for v in pick("weather_code")]
    
    wave_power = [0.49*(h**2)*p for h,p in zip(wh,wp)]
    wind_cls = [wind_class_detailed(angle_diff(d, orient)) for d in wd]
    
    # إصلاح حساب الماضي بأمان
    past_avg, past_sh = 0.0, 0.0
    if past_idx:
        p_wh = aligned.get("wave_height", [])
        p_wp = aligned.get("wave_period", [])
        p_swh = aligned.get("swell_wave_height", [])
        valid_past = [i for i in past_idx if i < len(p_wh) and i < len(p_wp) and i < len(p_swh)]
        if valid_past:
            past_avg = sum(0.49*(p_wh[i]**2)*p_wp[i] for i in valid_past) / len(valid_past)
            past_sh = sum(p_swh[i] for i in valid_past) / len(valid_past)
            
    # إصلاح حساب الأعشاب (بناءً على نسبة الساعات بدل الساعة الأولى فقط)
    weed = False
    if target_idx and wind_cls:
        onshore_ratio = sum(1 for w in wind_cls if w.startswith("بحرية")) / len(wind_cls)
        weed = onshore_ratio > 0.5 and (past_sh > 0.8 or past_avg > 5.0)
        
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
        if avg_h > 0 and avg_swh > 0.7 * avg_h: swell_dom = "الطاقة أساساً قادمة من بعيد (swell قوي)"
        elif avg_h - avg_swh > 0.2: swell_dom = "الموج ناتج عن الرياح المحلية (wind sea)"
        
        wind_start = wind_cls[idxs[0]]; wind_end = wind_cls[idxs[-1]]
        wind_trend = f"تتحول من {wind_start} إلى {wind_end}" if wind_start != wind_end else f"ثابتة {wind_start}"
        sea = "هادئ" if avg_h < 0.3 else "متوسط الهيجان" if avg_h < 0.8 else "هائج"
        
        swell_angle = angle_diff(avg_swd, orient) if avg_swd else None
        wave_angle = angle_diff(avg_wave_dir, orient) if avg_wave_dir else None
        
        blocks.append({
            "name":{"morning":"الصباح","afternoon":"الظهر","night":"الليل"}[key],
            "time_range":f"{all_times[target_idx[idxs[0]]].strftime('%H:%M')}-{all_times[target_idx[idxs[-1]]].strftime('%H:%M')}",
            "sea_state":sea,"wave_height":f"{min_h:.2f}-{max_h:.2f}","wave_power":round(avg_pow,2),
            "swell_height":f"{min(swh[i] for i in idxs):.2f}-{max(swh[i] for i in idxs):.2f}",
            "swell_period":round(avg_swp,1),
            "swell_dir": deg_to_compass(avg_swd) if avg_swd else "غير معروف",
            "swell_angle_diff": round(swell_angle,0) if swell_angle is not None else None,
            "wave_dir": deg_to_compass(avg_wave_dir) if avg_wave_dir else "غير معروف",
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
    
    # إصلاح حساب تغير الضغط (آخر 3 ساعات فعلياً)
    press_3h_change = pr[-1] - pr[-4] if len(pr) >= 4 else (pr[-1] - pr[0] if len(pr) > 1 else 0)
    
    # إصلاح القائمة الفارغة للأسماك
    bio = {"high": []}
    if avg_sst < 16: bio["high"] = ["قاروص", "سارغ"]
    elif avg_sst < 19: bio["high"] = ["قاروص", "بوري", "سارغ"] # منطقة انتقالية
    else: bio["high"] = ["دوراد", "ماربري", "بوري"]
    
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

# ==================== بناء سياق الذكاء الاصطناعي (تم تحسينه لتقرير تحليلي متكامل) ====================
def build_context(req, agg, tz_name):
    beach = "رملي" if req.beach_type == "sandy" else "صخري"
    moon = agg["extra_info"]
    orient = req.beach_orientation

    # ملخص موحد للزوايا
    angle_summary = []
    for b in agg["blocks"]:
        sa = b.get("swell_angle_diff")
        wa = b.get("wave_angle_diff")
        sa_desc = "عمودي" if sa is not None and 70 <= sa <= 110 else "مائل"
        wa_desc = "عمودي" if wa is not None and 70 <= wa <= 110 else "مائل"
        angle_summary.append(f"{b['name']}: Swell {b['swell_dir']} ({sa_desc})، موج محلي {b['wave_dir']} ({wa_desc})")

    # ملخص موحد للدورة
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

    # ملخص موحد للرياح
    wind_summary = []
    for b in agg["blocks"]:
        wind_summary.append(f"{b['name']}: {b['wind_dir']} {b['wind_speed']} كم/س (هبات {b['wind_gust_peak']})، {b['wind_trend']}")

    # الضغط الجوي وتغيره
    press_avg = moon.get("pressure_avg", 1013)
    press_change = moon.get("pressure_change_3h", 0)
    if press_change < -0.5: press_trend = "في انخفاض (الساعة الذهبية للصيد)"
    elif press_change > 0.5: press_trend = "في ارتفاع (الأسماك قد تكون خاملة)"
    else: press_trend = "مستقر"

    # تقييم أولي آلي
    has_blocks = bool(agg["blocks"])
    swell_ok = all(get_max_val(b.get("swell_height")) <= 1.3 for b in agg["blocks"]) if has_blocks else False
    period_ok = all(b.get("swell_period", 0) >= 7 for b in agg["blocks"]) if has_blocks else False
    wind_ok = all(get_max_val(b.get("wind_speed")) <= 33 for b in agg["blocks"]) if has_blocks else False
    pressure_ok = press_change <= 0

    score = sum([swell_ok, period_ok, wind_ok, pressure_ok])
    if score == 4: verdict = "مثالية (Go)"
    elif score == 3: verdict = "جيدة مع استعدادات (Go)"
    elif score == 2: verdict = "مقبولة بشروط (Go بحذر)"
    else: verdict = "سيئة (No-Go)"

    # تجميع الرسالة
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
