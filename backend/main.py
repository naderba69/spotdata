"""
Surfcasting Analytics API – v10.3 (Precision Mirror – Windy Image Analysis Integrated)
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
app = FastAPI(title="Surfcasting Analytics", version="10.3.0")
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
def health(): return {"status": "ok", "version": "10.3.0"}

# ==================== أدوات الشبكة والرياضيات ====================
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
    if m < 3: y-=1; m+=12
    a = int(y/100); b = 2-a+int(a/4)
    jd = int(365.25*(y+4716)) + int(30.6001*(m+1)) + day + b - 1524.5
    days_since_new = jd - 2451550.1
    phase = (days_since_new % 29.53058867) / 29.53058867
    idx = int(phase * 8) % 8
    names = {0:"محاق",1:"هلال أول",2:"تربيع أول",3:"أحدب متزايد",4:"بدر",5:"أحدب متناقص",6:"تربيع ثاني",7:"هلال آخر"}
    if idx in [0, 4]: tide_strength = "مد وجزر قوي جداً (Spring Tides)"
    elif idx in [2, 6]: tide_strength = "مد وجزر ضعيف جداً (Neap Tides)"
    else: tide_strength = "مد وجزر متوسط"
    return {"name": names[idx], "phase_decimal": phase, "tide_strength": tide_strength, "idx": idx}

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
        "precipitation": extract("precipitation", weather_hourly, w_map), 
        "visibility": extract("visibility", weather_hourly, w_map),
        "weather_code": [int(safe_float(x)) for x in extract("weather_code", weather_hourly, w_map)]
    }

# ==================== قاعدة الشواطئ الكاملة ====================
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

# ==================== OSM مع تكرار نصف القطر ====================
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
                    for i in range(len(geom)):
                        p = geom[i]
                        d = calc_distance(lat, lon, p["lat"], p["lon"])
                        if d < best_dist:
                            best_dist, best_point = d, p
                            prev_i, next_i = max(0, i - 1), min(len(geom) - 1, i + 1)
                            if prev_i != next_i:
                                best_tangent = calc_bearing(geom[prev_i]["lat"], geom[prev_i]["lon"], geom[next_i]["lat"], geom[next_i]["lon"])
                if not best_tangent or not best_point: continue
                n_a, n_b = (best_tangent + 90) % 360, (best_tangent - 90) % 360
                c2u = calc_bearing(best_point["lat"], best_point["lon"], lat, lon)
                d_a = abs(c2u - n_a); d_a = 360 - d_a if d_a > 180 else d_a
                d_b = abs(c2u - n_b); d_b = 360 - d_b if d_b > 180 else d_b
                return int(round(((n_a if d_a < d_b else n_b) + 180) % 360))
        except: continue
    return 0

@app.post("/auto-orientation")
@limiter.limit("5/minute")
async def auto_orientation(request: Request, req: AutoOrientationRequest):
    orientation = await get_auto_orientation_overpass(req.latitude, req.longitude)
    if orientation != 0: return {"orientation": orientation, "source": "overpass"}
    orientation = find_nearest_beach_orientation(req.latitude, req.longitude)
    if orientation is not None: return {"orientation": orientation, "source": "nearest_beach"}
    return {"orientation": -1, "source": "none", "message": "تعذر التحديد التلقائي."}

# ==================== محرك التجميع الفيزيائي المتقدم (v10.3) ====================
def aggregate_physics(all_times, aligned, orient, target_date_obj, sunrise, sunset):
    tz = all_times[0].tzinfo if all_times else zoneinfo.ZoneInfo("UTC")
    target_start = datetime.combine(target_date_obj, datetime.min.time(), tzinfo=tz)
    target_end = target_start + timedelta(days=1)
    past_start = target_start - timedelta(hours=48)
    past_idx = [i for i, t in enumerate(all_times) if past_start <= t < target_start]
    target_idx = [i for i, t in enumerate(all_times) if target_start <= t < target_end]
    
    empty_res = {"sea_memory":"غير معروف","lateral_current":"غير معروف","pressure_state":"مستقر","tide_analysis":{},"sst_stability":"مستقر","bio_matrix":{},"avg_sst":0,"hidden_factors":{},"blocks":[],"red_flags":[],"green_flags":[],"extra_info":{}, "transitions":[]}
    if not target_idx: return empty_res
    
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
    
    # 1. ذاكرة البحر
    sea_memory = "بحر صافي وهادئ (لا توجد عوامل تعكير سابقة)"
    past_avg, past_sh = 0.0, 0.0
    sudden_wind_shift = False
    sst_trend = "مستقر"
    if past_idx:
        p_wh = aligned.get("wave_height", []); p_wp = aligned.get("wave_period", [])
        p_swh = aligned.get("swell_wave_height", []); p_swp = aligned.get("swell_wave_period", [])
        p_ws = aligned.get("wind_speed_10m", []); p_wd = aligned.get("wind_direction_10m", [])
        p_prec = aligned.get("precipitation", []); p_sst = aligned.get("sea_surface_temperature", [])
        valid_past = [i for i in past_idx if i < len(p_wh) and i < len(p_wp) and i < len(p_ws) and i < len(p_wd)]
        
        if valid_past:
            past_avg = sum(0.49*(p_wh[i]**2)*p_wp[i] for i in valid_past) / len(valid_past)
            past_sh = sum(p_swh[i] for i in valid_past if i < len(p_swh)) / len(valid_past)
            past_onshore_hours = sum(1 for i in valid_past if wind_class_detailed(angle_diff(p_wd[i], orient)).startswith("بحرية"))
            past_onshore_ratio = past_onshore_hours / len(valid_past)
            past_swp_avg = sum(p_swp[i] for i in valid_past if i < len(p_swp)) / len(valid_past)
            past_rain = sum(p_prec[i] for i in valid_past if i < len(p_prec))
            
            if past_avg > 6.0 and past_onshore_ratio > 0.4:
                sea_memory = "بحر خامر وعكر جداً (طونى): رياح بحرية قوية في اليومين الماضيين قلبت القاع وخلطت المياه بالرمل والطين. الرؤية شبه معدومة."
            elif past_avg > 4.0 and past_onshore_ratio > 0.3:
                sea_memory = "بحر يعكر ببطء: رياح بحرية متوسطة خلقت طبقة عكرة قريبة من الشاطئ."
                
            if past_swp_avg > 7.0 and past_sh > 0.8:
                sea_memory += " | تحذير صوفة: الأمواج الطويلة الماضية اقتلعت أعشاب البوسيدونيا من القاع."
            if past_rain > 10.0:
                sea_memory += " | سيول: أمطار غزيرة سابقة أدخلت مياه عذبة وطينية للساحل."

            if len(valid_past) >= 6:
                first_half_wd = [p_wd[i] for i in valid_past[:len(valid_past)//2]]
                second_half_wd = [p_wd[i] for i in valid_past[len(valid_past)//2:]]
                if angle_diff(sum(first_half_wd)/len(first_half_wd), sum(second_half_wd)/len(second_half_wd)) > 90:
                    sudden_wind_shift = True

            past_sst_vals = [p_sst[i] for i in valid_past if i < len(p_sst)]
            if len(past_sst_vals) >= 12:
                half = len(past_sst_vals)//2
                older_half = past_sst_vals[:half]
                newer_half = past_sst_vals[half:]
                if older_half and newer_half:
                    older_avg = sum(older_half)/len(older_half)
                    newer_avg = sum(newer_half)/len(newer_half)
                    diff_sst_past = newer_avg - older_avg
                    if diff_sst_past < -1.5:
                        sst_trend = f"انخفاض حاد في حرارة الماء خلال الساعات الماضية ({abs(diff_sst_past):.1f}°م). الأسماك قد تكون في حالة صدمة باردة."
                    elif diff_sst_past > 1.5:
                        sst_trend = f"ارتفاع حاد في حرارة الماء ({diff_sst_past:.1f}°م). قد ينشط الأسماك السطحية."

    # 2. التيار الجانبي
    valid_wd_wave = [angle_diff(w, orient) for w in wd_wave if w != 0]
    avg_wave_angle = sum(valid_wd_wave) / len(valid_wd_wave) if valid_wd_wave else 90
    lateral_force = math.sin(math.radians(avg_wave_angle))
    avg_wave_h = sum(wh) / len(wh) if wh else 0
    
    if lateral_force > 0.8 and avg_wave_h > 0.6: lateral_current = "تيار جارف قوي جداً (موازي للشاطئ): الرصاصة ستنجرف بسرعة كبيرة، والخط سيصبح قوساً. يتطلب رصاص ثقيل جداً أو تغيير زاوية الرمي عكس التيار بـ 30-40 درجة."
    elif lateral_force > 0.5 and avg_wave_h > 0.4: lateral_current = "تيار جانبي متوسط: سيحدث انجراف تدريجي للطعم. يجب مراقبة خيط الخط وتعديل الثقل."
    else: lateral_current = "تيار جانبي ضعيف أو معدوم: الموج يدفع للخلف وللأمام (عمودي)، الرصاصة ستثبت جيداً في القاع دون انجراف عرضي."

    # 3. العوامل الخفية (مُطوَّرة)
    freshwater_risk = "منخفض"
    stratification_risk = "منخفض"
    cross_sea_risk = "منخفض"
    visibility_status = "رؤية ممتازة (>10 كم)"
    mirror_sea_risk = False  # (جديد) خطر البحر المرآة نهاراً
    
    if past_idx:
        if any(i < len(aligned.get("precipitation", [])) for i in past_idx):
            if sum(aligned["precipitation"][i] for i in past_idx if i < len(aligned["precipitation"])) > 10.0:
                freshwater_risk = "مرتفع جداً (سيول ومياه عذبة طافية تغطي السطح وتمنع الأكسجين)"
                
        valid_strat = [i for i in past_idx if i < len(aligned.get("swell_wave_height", [])) and i < len(aligned.get("wind_speed_10m", []))]
        if valid_strat:
            avg_p_swh = sum(aligned["swell_wave_height"][i] for i in valid_strat)/len(valid_strat)
            avg_p_ws = sum(aligned["wind_speed_10m"][i] for i in valid_strat)/len(valid_strat)
            if avg_p_swh < 0.2 and avg_p_ws < 10.0:
                stratification_risk = "مرتفع (بحر مسطح لفترة طويلة، القاع يفقد الأكسجين والسمك يصبح خاملاً)"

    # (جديد) تقييم خطر البحر المرآة
    avg_wave_h_day = sum(wh) / len(wh) if wh else 0
    if avg_wave_h_day < 0.3 and max(swp) < 6:
        mirror_sea_risk = True

    cross_angles = []
    for i in range(len(swd)):
        if swd[i] != 0 and wd_wave[i] != 0:
            cross_angles.append(angle_diff(swd[i], wd_wave[i]))
    if cross_angles:
        avg_cross = sum(cross_angles) / len(cross_angles)
        max_cross = max(cross_angles)
        if max_cross > 60 and avg_cross > 40:
            cross_sea_risk = "بحر مختلط وخطير (Cross Sea): اتجاه السويل القادم يتعارض بشدة مع اتجاه الموج المحلي. هذا يخلق أمواجاً قصيرة وعشوائية ومتداخلة، تصطاد بشكل عشوائي وتجعل التحكم في الرصاصة شبه مستحيل."
        elif max_cross > 45:
            cross_sea_risk = "بحر مختلط متوسط: بعض التضارب بين السويل والموج المحلي. طاقة مشوهشة ومتقلل ثبات الرمية."

    valid_vis = [v for v in vis if v > 0]
    if valid_vis:
        min_vis = min(valid_vis)
        if min_vis < 1000: visibility_status = "ضباب كثيف جداً (أقل من 1 كم). الرؤية شبه معدومة، خطورة الانزلاق والغرق. يُنصح بتقليل مسافة الرمية والصيد الدقيق جداً."
        elif min_vis < 5000: visibility_status = "ضباب خفيف (أقل من 5 كم). يجب الحذر."
        elif min_vis > 15000: visibility_status = "رؤية ممتازة."
    else:
        visibility_status = "بيانات الرؤية غير متوفرة."

    steepness_vals = [h / (1.56 * (p**2)) for h, p in zip(wh, wp) if p > 0]
    avg_steepness = sum(steepness_vals) / len(steepness_vals) if steepness_vals else 0
    if avg_steepness > 0.06: steepness_desc = "موج حاد وقصير (Steep). ينكسر بقوة، يخلق ماءاً أبيض كثيفاً. سيء للرمي البعيد."
    elif avg_steepness < 0.03: steepness_desc = "موج طويل وهادئ (Swell). ينكسر بعيداً ويخلق خنادق طبيعية. ممتاز."
    else: steepness_desc = "موج متوسط الانحدار."

    tide_analysis = get_moon_and_tide_analysis(target_date_obj)
    golden_lock = "مد قوي (Spring)" if tide_analysis["idx"] in [0,4] else "مد ضعيف (Neap)" if tide_analysis["idx"] in [2,6] else "متوسط"

    # 4. باقي الحسابات
    if len(sst) > 1:
        sst_diff = max(sst) - min(sst)
        sst_stability = "صدمة حرارية (انخفاض/ارتفاع حاد)" if sst_diff > 2.0 else "تغير بطيء" if sst_diff > 1.0 else "مستقر تماماً"
    else: sst_stability = "بيانات غير كافية"
    
    max_swp = max(swp) if swp else 0
    onshore_hours = sum(1 for w in wind_cls if w.startswith("بحرية"))
    
    is_past_murky = "عكر" in sea_memory or "خامر" in sea_memory
    clarity_risk = is_past_murky or (max_swp >= 8 and onshore_hours > len(wind_cls)*0.3)
    
    weed = onshore_hours > len(wind_cls)*0.5 and (past_sh > 0.8 or past_avg > 5.0)
    
    avg_sst = sum(sst)/len(sst) if sst else 0
    
    is_murky = "عكر" in sea_memory or "خامر" in sea_memory
    is_weedy = "صوفة" in sea_memory
    is_fresh = "مرتفع" in freshwater_risk
    is_dead = "مرتفع" in stratification_risk

    bio_matrix = {
        "قاروص": {"status": "معدوم" if is_dead else "نشط جداً" if (avg_sst < 18 and is_murky and not is_weedy) else "نشط" if avg_sst < 18 else "غائب تقريباً", "reason": "يكره البحر المسطح الميت. يستفيد من العكر لكن الأعشاب تدمر خطوطه."},
        "دوراد": {"status": "شبه معدوم" if (is_murky or is_fresh) else "نشط" if (avg_sst > 18 and not is_murky) else "خامل", "reason": "يهرب من المياه العذبة والعكرة تماماً."},
        "بوري": {"status": "معدوم" if (is_weedy or is_fresh) else "نشط" if (not is_murky and tide_analysis["idx"] in [2, 6]) else "مقبول", "reason": "الأوساخ العالقة تفقده قدرته على رؤية الطعم السطحي."},
        "سارغ": {"status": "معدوم" if is_dead else "ممكن" if avg_sst < 22 else "ضعيف", "reason": "يتأثر بالحرارة المرتفعة والأعشاب العالقة قرب الصخور."}
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
    raw_blocks_meta = []
    
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
        
        # (جديد) وصف تفاعل السويل والموج المحلي
        swell_wave_interaction = ""
        if swell_angle is not None and wave_angle is not None:
            diff_sw = angle_diff(avg_swd, avg_wave_dir) if avg_swd and avg_wave_dir else 0
            if diff_sw > 40:
                swell_wave_interaction = f"السويل والموج المحلي يتقاطعان بزاوية ({diff_sw:.0f}°) مما يخلق بحراً مختلطاً يضاعف التيار الجانبي."
            else:
                swell_wave_interaction = "السويل والموج المحلي متوافقان في الاتجاه، البحر منتظم."
        else:
            swell_wave_interaction = "لا توجد بيانات كافية لتقييم تفاعل السويل والموج المحلي."
        
        block_data = {
            "name":{"morning":"الصباح","afternoon":"الظهر","night":"الليل"}[key],
            "time_range":f"{all_times[target_idx[idxs[0]]].strftime('%H:%M')}-{all_times[target_idx[idxs[-1]]].strftime('%H:%M')}",
            "sea_state":sea,"wave_height":f"{min_h:.2f}-{max_h:.2f}","wave_power":round(avg_pow,2),
            "swell_height":f"{min(swh[i] for i in idxs):.2f}-{max(swh[i] for i in idxs):.2f}",
            "swell_period":round(avg_swp,1), "swell_dir": deg_to_compass(avg_swd) if avg_swd else "غير معروف",
            "swell_angle_diff": round(swell_angle,0) if swell_angle is not None else None,
            "wave_dir": deg_to_compass(avg_wave_dir) if avg_wave_dir else "غير معروف",
            "wave_angle_diff": round(wave_angle,0) if wave_angle is not None else None,
            "swell_wave_interaction": swell_wave_interaction,
            "swell_dominance":swell_dom,"wind_speed":f"{min_w:.1f}-{max_w:.1f}","wind_gust_peak":round(max(wg[i] for i in idxs),1),
            "wind_dir":wc_dom,"wind_trend":wind_trend,"air_temp":round(avg_air,1),"precip":round(total_precip,1),
            "weather":weather_desc(most_code)
        }
        blocks.append(block_data)
        raw_blocks_meta.append({"name": block_data["name"], "max_h": max_h, "avg_swp": avg_swp, "wind_cls": wc_dom})

    transitions = []
    for i in range(len(raw_blocks_meta) - 1):
        b1, b2 = raw_blocks_meta[i], raw_blocks_meta[i+1]
        changes = []
        if b1["wind_cls"] != b2["wind_cls"]:
            if "بحرية" in b2["wind_cls"] and "برية" in b1["wind_cls"]:
                changes.append("انقلاب رياح خطير من برية إلى بحرية (ستضرب الشاطئ بقوة فجأة وتجعل الماء عكراً ومضطرباً، الراحة في خطر والصيد صعب جداً).")
            elif "برية" in b2["wind_cls"] and "بحرية" in b1["wind_cls"]:
                changes.append("تحول ممتاز للرياح من بحرية إلى برية (ستهدأ سطح البحر تدريجياً وتنظف أي عكر أو طحالب عالقة، فرصة ذهبية قد تفتح).")
        h_diff = b2["max_h"] - b1["max_h"]
        if h_diff < -0.3: changes.append(f"تهدأ واضح للموج (انخفاض أقصى بنسبة {abs(h_diff):.2f}م)، الماء سيبدأ في الترسبح والنقاء.")
        elif h_diff > 0.3: changes.append(f"تصاعد مفاجئ في هيجان البحر (ارتفاع أقصى بنسبة {h_diff:.2f}م)، احذر من ارتطام الأمواج المفاجئ.")
        swp_diff = b2["avg_swp"] - b1["avg_swp"]
        if swp_diff > 1.5 and b1["avg_swp"] < 8: changes.append("دورة السويل تتطاول بوضوح (تشير إلى وصول موج بعيد وقادم من أعماق المحيط، قد يضرب الشاطئ بقوة لاحقاً ويغير قاع الشاطئ).")
        elif swp_diff < -1.5 and b1["avg_swp"] > 7: changes.append("دورة السويل تقصر (الموج يتحول من بعيد إلى محلي، علامة على استقرار تام ونهاية اضطراب القاع).")
        if changes: transitions.append(f"التطور من {b1['name']} إلى {b2['name']}: " + " | ".join(changes))

    reds, greens = [], []
    for i in range(len(wh)):
        hh = all_times[target_idx[i]].strftime("%H:%M")
        if wave_power[i] > 3 or wh[i] > 1.8 or wg[i] > 50 or pr[i] < 1005: reds.append(hh)
        if 0.3 <= wh[i] <= 1 and 0.1 <= wave_power[i] <= 1.5 and ws[i] < 27.8: greens.append(hh)
        
    avg_press = sum(pr)/len(pr) if pr else 0
    press_change = pr[-1] - pr[-4] if len(pr) >= 4 else (pr[-1] - pr[0] if len(pr) > 1 else 0)

    if avg_press > 1025: press_abs_desc = "مرتفع جداً"; press_abs_effect = "الأسماك خاملة وكسولة بغض النظر عن التغير"
    elif avg_press < 1008: press_abs_desc = "منخفض جداً"; press_abs_effect = "إشارة خطر للأسماك، تدفعها للتغذية حتى لو كان التغير طفيفاً"
    else: press_abs_desc = "معتدل"; press_abs_effect = "لا تأثير سلبي مباشر"

    if press_change < -2.0: pressure_state = f"ضغط {press_abs_desc} ({avg_press:.0f} hPa) في انخفاض حاد ({press_change:.1f}). الساعة الذهبية للصيد. {press_abs_effect}."
    elif press_change < -0.5: pressure_state = f"ضغط {press_abs_desc} ({avg_press:.0f} hPa) في انخفاض بطيء ({press_change:.1f}). نشاط جيد. {press_abs_effect}."
    elif press_change > 1.5: pressure_state = f"ضغط {press_abs_desc} ({avg_press:.0f} hPa) في ارتفاع حاد ({press_change:+.1f}). الأسماك تتوقف عن الأكل. {press_abs_effect}."
    else: pressure_state = f"ضغط {press_abs_desc} ومستقر ({avg_press:.0f} hPa، تغير {press_change:+.1f}). {press_abs_effect}."

    extra = {
        "pressure_avg":round(avg_press,1), "pressure_change_3h":round(press_change,1),
        "sunrise":sunrise, "sunset":sunset, "peak_gust_today":round(peak_gust,1)
    }
    return {
        "dominant_wind":dominant, "blocks":blocks, "red_flags":reds[:5], "green_flags":greens[:5],
        "sea_memory":sea_memory, "lateral_current":lateral_current, "pressure_state":pressure_state,
        "tide_analysis":tide_analysis, "sst_stability":sst_stability,
        "hidden_factors": {
            "freshwater_risk": freshwater_risk, "stratification_risk": stratification_risk,
            "wave_steepness": steepness_desc, "golden_lock": golden_lock,
            "cross_sea_risk": cross_sea_risk, "sst_trend": sst_trend,
            "sudden_wind_shift": sudden_wind_shift, "visibility_status": visibility_status,
            "weed_risk": weed, "clarity_risk": clarity_risk,
            "mirror_sea_risk": mirror_sea_risk  # (جديد)
        },
        "bio_matrix":bio_matrix, "avg_sst":round(avg_sst,1), "extra_info":extra,
        "transitions": transitions
    }

# ==================== بناء سياق صارم (v10.3) ====================
def build_context(req, agg, tz_name):
    beach = "رملي" if req.beach_type == "sandy" else "صخري"
    orient = req.beach_orientation
    moon = agg["tide_analysis"]
    extra = agg["extra_info"]
    hf = agg["hidden_factors"]

    interactions = [
        f"حرارة الماء السطحية: {agg['avg_sst']}°م ({agg['sst_stability']}).",
        f"ذاكرة البحر وحالته الراهنة: {agg['sea_memory']}",
        f"ميكانيكا الموج والتيار الجانبي: {agg['lateral_current']}",
        f"حالة الضغط الجوي: {agg['pressure_state']}"
    ]
    # تنبيه صريح عن التيار
    if "ضعيف" in agg["lateral_current"]:
        interactions.append("✅ لا يوجد تيار جانبي مؤثر اليوم – الرصاصة ستثبت بأوزان عادية (80-120 غرام).")
    else:
        interactions.append("⚠️ يوجد تيار جانبي يتطلب رصاصاً ثقيلاً.")
    
    # تنبيه البحر المرآة
    if hf.get("mirror_sea_risk"):
        interactions.append("🪞 تحذير البحر المرآة: المياه شديدة الهدوء والصفاء. الأسماك ترى الخيوط بوضوح في النهار. يجب استخدام خيوط رفيعة (Fluorocarbon) والصيد ليلاً.")
    
    if hf.get("weed_risk"): interactions.append("🚨 خطر الأعشاب: البوسيدونيا مقتلعة وتتجه للشاطئ.")
    if hf.get("clarity_risk"): interactions.append("⚠️ الماء عكر وغير صافٍ.")
    if hf.get("sudden_wind_shift"): interactions.append("⚠️ تغير مفاجئ في اتجاه الرياح خلال الساعات الماضية، الأسماك متوترة.")
    if "انخفاض حاد" in hf.get("sst_trend", ""): interactions.append(f"⚠️ {hf['sst_trend']}")
    if "مرتفع" in hf["freshwater_risk"]: interactions.append(f"⚠️ خطر السيول والمياه العذبة: {hf['freshwater_risk']}")
    if "مرتفع" in hf["stratification_risk"]: interactions.append(f"⚠️ خطر انعدام التمازج: {hf['stratification_risk']}")
    if "خطير" in hf.get("cross_sea_risk", ""): interactions.append(f"⚠️ {hf['cross_sea_risk']}")
    if "خفيف" in hf.get("visibility_status", ""): interactions.append(f"⚠️ الرؤية: {hf['visibility_status']}")

    if moon["idx"] in [0, 4]: interactions.append(f"تأثير المد: {moon['tide_strength']}. تيارات قوية جداً.")
    elif moon["idx"] in [2, 6]: interactions.append(f"تأثير المد: {moon['tide_strength']}. تيارات ضعيفة.")

    for b in agg["blocks"]:
        sa = b.get("swell_angle_diff"); wa = b.get("wave_angle_diff")
        sa_desc = "عمودي" if sa is not None and 70 <= sa <= 110 else "مائل"
        wa_desc = "عمودي" if wa is not None and 70 <= wa <= 110 else "مائل"
        interactions.append(f"في {b['name']}: البحر {b['sea_state']}، السويل {b['swell_dir']} ({sa_desc})، الموج المحلي {b['wave_dir']} ({wa_desc}) – {b['swell_wave_interaction']} الرياح {b['wind_dir']} {b['wind_speed']}كم/س ({b['wind_trend']}). حرارة الهواء {b['air_temp']}°م، السماء {b['weather']}.")

    if agg.get("transitions"):
        interactions.extend(["", "=== التطور الديناميكي للظروف خلال اليوم ==="])
        interactions.extend(agg["transitions"])

    bio_text = "\n".join([f"- {fish}: {data['status']} ({data['reason']})" for fish, data in agg["bio_matrix"].items()])

    lines = [
        f"المهمة: تحليل ظروف السيرفكاستينغ لشاطئ {beach} (اتجاه {orient}°) - توقيت {tz_name}.",
        "البيانات الخام ممنوعة من الإعادة في التقرير. حلل التفاعلات التالية واستنتج:",
        "", "=== سلسلة التفاعلات الحرجة والمتسلسلة ===", *interactions, "",
        "=== العوامل الخفية ===", f"انحدار الموج: {hf['wave_steepness']}", f"تزامن المد: {hf['golden_lock']}",
        f"البحر المختلط: {hf['cross_sea_risk']}", f"صدمة الحرارة: {hf['sst_trend']}",
        f"خطر الأعشاب: {'نعم' if hf.get('weed_risk') else 'لا'} | عكر الماء: {'نعم' if hf.get('clarity_risk') else 'لا'}",
        f"خطر البحر المرآة: {'نعم (المياه صافية جداً نهاراً)' if hf.get('mirror_sea_risk') else 'لا'}",
        "", f"=== تقييم الأنواع المستهدفة ===", bio_text, "",
        f"=== التوقيتات ===", f"أفضل ساعات (خضراء): {', '.join(agg['green_flags']) if agg['green_flags'] else 'لا يوجد'}",
        f"ساعات الخطر (حمراء): {', '.join(agg['red_flags']) if agg['red_flags'] else 'لا يوجد'}",
        f"الشروق {extra['sunrise']} | الغروب {extra['sunset']} | هبات قصوى {extra['peak_gust_today']} كم/س",
        "", "المطلوب: تقرير تحليلي مركب ومفصل وطويل (ليس سردياً). ابدأ بالنتيجة النهائية (Go/No-Go) ثم فكك الأسباب. اربط كل ظاهرة بتأثيرها الميكانيكي على الخطاف والرصاصة والسمكة."
    ]
    return "\n".join(lines)

SYSTEM_PROMPT = """أنت عالم أحياء بحرية ومحلل فيزيائي متخصص حصرياً في صيد السرفكاستينغ (Surfcasting) في البحر المتوسط (تونس).
مهمتك: تحويل التفاعلات الفيزيائية المعطاة إلى تقرير استنتاجي طويل ومفصل بالدارجة التونسية.

قواعد صارمة جداً:
1.  **الربط الإجباري:** كل جملة يجب أن تربط بين ظاهرة فيزيائية وتأثيرها المباشر على ميكانيكية الصيد (الرمي، ثبات الرصاص، انجراف الطعم، رؤية السمك للطعم، سلوك السمك). لا تكتب جملة منعزلة.
2.  **تحليل الذاكرة البحرية:** ابدأ بتحليل حالة الماء اليوم بناءً على الأيام السابقة (العكر، الأعشاب، الأمطار) واشرح كيف ستؤثر على اختيارك للطعم ومكان الرمي.
3.  **تحليل الزوايا والتيار:** اذكر زوايا الموج (عمودي/مائل) في كل فترة، واشرح بالتفصيل كيف ستخلق تياراً جانبياً (جرار) يجرف الرصاص. اربط ذلك مباشرة بوزن الرصاصة المطلوبة. إذا كان التيار الجانبي "ضعيف أو معدوم"، يجب أن تبدأ التوصية بـ "بما أنه لا يوجد تيار جانبي (جرار) اليوم..." ثم تستنتج الوزن المناسب (80-120 غرام).
4.  **تحليل الضغط الجوي:** استخدم وصف الضغط الجوي المُعطى (الذي يحوي القيمة المطلقة والتغير) لتحديد مستوى نشاط الأسماك. لا تهمل القيمة المطلقة.
5.  **القرار النهائي:** يجب أن يكون في السطر الأول: "Go" أو "No-Go". إذا كان "Go"، حدد متى بالضبط. إذا كان "No-Go"، اذكر السبب الرئيسي بوضوح.
6.  **الأسلوب:** واقعي، قاسٍ، لا يجامل. استخدم الدارجة التونسية الاحترافية. لا تذكر أبداً عبارات مثل "حسب المعطيات" أو "بناءً على البيانات".
7.  **تفاعل السويل والموج المحلي:** إذا ذكر التحليل أن "السويل والموج المحلي يتقاطعان بزاوية..."، يجب أن تشرح كيف سيؤثر ذلك على شكل البحر واستقرار الطعم.
8.  **لا تخمن أوزاناً ثقيلة دون مبرر:** إذا كان التيار الجانبي ضعيفاً، فالوزن المثالي هو 80-120 غرام. لا تقترح 150 غرام أو أكثر إلا إذا كان هناك تيار جانبي قوي أو أمواج عالية.
9.  **خطر البحر المرآة:** إذا ذكر التحليل "تحذير البحر المرآة"، يجب أن تشرح أن الماء الصافي جداً نهاراً يجعل الأسماك ترى الخيوط، وتوصي باستخدام خيوط Fluorocarbon رفيعة (0.25-0.30 مم) والصيد ليلاً حصراً.

اكتب تقريراً واحداً متصلاً، طويلاً، عميقاً، يشرح "لماذا" و"كيف" سيحدث كل شيء. كن دقيقاً، لا مجال للخطأ."""

async def call_openrouter(ctx):
    headers = {"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json"}
    payload = {"model":MODEL_NAME,"messages":[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":ctx}],"max_tokens":8000,"temperature":0.5}
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
        
        agg = aggregate_physics(all_times, aligned, req.beach_orientation, target_dt, sunrise, sunset)
        
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
