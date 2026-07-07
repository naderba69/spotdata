"""
Surfcasting Analytics API – v12.0.0 (The Hardened Production Build)
- Fixed 0-Degree Directional Trap
- Replaced String-Based Logic with Boolean Flags
- Implemented Exponential Decay for Sea Memory
- Dynamic SST Biological Thresholds
"""
import os, math, asyncio, logging, traceback, zoneinfo
from datetime import datetime, timedelta, date
from typing import Dict, Optional, List
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
app = FastAPI(title="Surfcasting Analytics", version="12.0.0")
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
def health(): return {"status": "ok", "version": "12.0.0"}

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

# ==================== قاعدة الشواطئ ====================
TUNISIAN_BEACHES = [
    {"name":"شاطئ الحمامات","lat":36.4000,"lon":10.6167,"orientation":90,"type":"sandy"},
    {"name":"شاطئ ياسمين الحمامات","lat":36.3800,"lon":10.5500,"orientation":90,"type":"sandy"},
    {"name":"شاطئ قليبية","lat":36.8500,"lon":11.1000,"orientation":45,"type":"sandy"},
    {"name":"شاطئ المعمورة","lat":36.5500,"lon":10.6000,"orientation":90,"type":"sandy"},
    {"name":"شاطئ قرطاج","lat":36.8528,"lon":10.3264,"orientation":90,"type":"sandy"},
    {"name":"شاطئ المرسى","lat":36.8764,"lon":10.3253,"orientation":45,"type":"sandy"},
    {"name":"شاطئ بوجعفر","lat":35.8333,"lon":10.6333,"orientation":90,"type":"sandy"},
    {"name":"شاطئ القنطاوي","lat":35.8833,"lon":10.6000,"orientation":90,"type":"sandy"},
    {"name":"شاطئ رادس","lat":36.7500,"lon":10.2833,"orientation":0,"type":"sandy"},
    {"name":"شاطئ حلق الوادي","lat":36.8167,"lon":10.3167,"orientation":0,"type":"sandy"},
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

# ==================== محرك التجميع الفيزيائي (v12.0.0) ====================
def aggregate_physics(all_times, aligned, orient, target_date_obj, sunrise, sunset):
    tz = all_times[0].tzinfo if all_times else zoneinfo.ZoneInfo("UTC")
    target_start = datetime.combine(target_date_obj, datetime.min.time(), tzinfo=tz)
    target_end = target_start + timedelta(days=1)
    past_start = target_start - timedelta(hours=48)
    past_idx = [i for i, t in enumerate(all_times) if past_start <= t < target_start]
    target_idx = [i for i, t in enumerate(all_times) if target_start <= t < target_end]
    
    empty_res = {"sea_memory":"غير معروف","lateral_current":"غير معروف","pressure_state":"مستقر","tide_analysis":{},"sst_stability":"مستقر","bio_matrix":{},"avg_sst":0,"hidden_factors":{},"blocks":[],"red_flags":[],"green_flags":[],"extra_info":{}, "transitions":[], "flags":{}}
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
    
    # 1. ذاكرة البحر (مع Exponential Decay)
    sea_memory = "بحر صافي وهادئ (لا توجد عوامل تعكير سابقة)"
    past_sh = 0.0
    sudden_wind_shift = "لا يوجد"
    sst_trend = "مستقر"
    
    if past_idx:
        p_wh = aligned.get("wave_height", []); p_wp = aligned.get("wave_period", [])
        p_swh = aligned.get("swell_wave_height", []); p_swp = aligned.get("swell_wave_period", [])
        p_ws = aligned.get("wind_speed_10m", []); p_wd = aligned.get("wind_direction_10m", [])
        p_prec = aligned.get("precipitation", []); p_sst = aligned.get("sea_surface_temperature", [])
        valid_past = [i for i in past_idx if i < len(p_wh) and i < len(p_wp) and i < len(p_ws) and i < len(p_wd)]
        
        if valid_past:
            # Exponential Decay: الساعات الأخيرة لها وزن 1.0، والأولى 0.2 (لأن الذاكرة تتبخر)
            weighted_past_power = 0.0
            weighted_past_swh = 0.0
            weighted_past_swp = 0.0
            total_weight = 0.0
            
            past_onshore_hours = 0.0
            past_rain = 0.0
            
            for pos, i in enumerate(valid_past):
                decay_weight = 0.8 ** (len(valid_past) - 1 - pos)
                weighted_past_power += decay_weight * 0.49*(p_wh[i]**2)*p_wp[i]
                weighted_past_swh += decay_weight * p_swh[i] if i < len(p_swh) else 0
                weighted_past_swp += decay_weight * p_swp[i] if i < len(p_swp) else 0
                total_weight += decay_weight
                
                if wind_class_detailed(angle_diff(p_wd[i], orient)).startswith("بحرية"): past_onshore_hours += decay_weight
                past_rain += p_prec[i] if i < len(p_prec) else 0

            past_avg = weighted_past_power / total_weight if total_weight > 0 else 0
            past_sh = weighted_past_swh / total_weight if total_weight > 0 else 0
            past_swp_avg = weighted_past_swp / total_weight if total_weight > 0 else 0
            past_onshore_ratio = past_onshore_hours / total_weight
            past_rain = past_rain / len(valid_past) # متوسط المطر عادي
            
            if past_avg > 6.0 and past_onshore_ratio > 0.4: sea_memory = "بحر خامر وعكر جداً (طونى)."
            elif past_avg > 4.0 and past_onshore_ratio > 0.3: sea_memory = "بحر يعكر ببطء."
            if past_swp_avg > 7.0 and past_sh > 0.8: sea_memory += " | تحذير صوفة (أعشاب)."
            if past_rain > 10.0: sea_memory += " | سيول."

            if len(valid_past) >= 6:
                first_half_wd = [p_wd[i] for i in valid_past[:len(valid_past)//2]]
                second_half_wd = [p_wd[i] for i in valid_past[len(valid_past)//2:]]
                if angle_diff(sum(first_half_wd)/len(first_half_wd), sum(second_half_wd)/len(second_half_wd)) > 90:
                    cls_first = wind_class_detailed(angle_diff(sum(first_half_wd)/len(first_half_wd), orient))
                    cls_second = wind_class_detailed(angle_diff(sum(second_half_wd)/len(second_half_wd), orient))
                    sudden_wind_shift = f"تحول مفاجئ من {cls_first} إلى {cls_second}."

            past_sst_vals = [p_sst[i] for i in valid_past if i < len(p_sst)]
            if len(past_sst_vals) >= 12:
                half = len(past_sst_vals)//2
                diff_sst_past = (sum(past_sst_vals[half:])/len(past_sst_vals[half:])) - (sum(past_sst_vals[:half])/len(past_sst_vals[:half]))
                if diff_sst_past < -1.5: sst_trend = f"انخفاض حاد ({abs(diff_sst_past):.1f}°م)."
                elif diff_sst_past > 1.5: sst_trend = f"ارتفاع حاد ({diff_sst_past:.1f}°م)."

    # 2. حساب التيار الجانبي كمتجهات (مع تجاهل 0.0)
    lateral_fx = 0.0
    lateral_fy = 0.0
    max_wh = max(wh) if wh else 0.0
    
    for i in range(len(wh)):
        w_dir = wd_wave[i] if i < len(wd_wave) else 0.0
        # Trap Fix: 0.0 تعني لا بيانات (Calm) وليست اتجاه الشمال
        if w_dir != 0.0:
            angle = math.radians(angle_diff(w_dir, orient))
            force = wh[i] * wh[i] 
            lateral_fx += force * math.sin(angle)
            lateral_fy += force * math.cos(angle)
            
    total_force = math.sqrt(lateral_fx**2 + lateral_fy**2)
    lateral_force_ratio = abs(lateral_fx) / total_force if total_force > 0 else 0
    avg_wave_h = sum(wh) / len(wh) if wh else 0

    is_mirror_sea = max_wh < 0.4
    is_lateral_strong = False
    
    if is_mirror_sea:
        lateral_current = "تيار جانبي معدوم (بحر مرآوي / ميت)"
    elif lateral_force_ratio > 0.7 and avg_wave_h > 0.6: 
        lateral_current = "تيار جارف قوي جداً"
        is_lateral_strong = True
    elif lateral_force_ratio > 0.4 and avg_wave_h > 0.4: 
        lateral_current = "تيار جانبي متوسط"
    else: 
        lateral_current = "تيار جانبي ضعيف"

    freshwater_risk = "منخفض"
    stratification_risk = "منخفض"
    cross_sea_risk = "منخفض"
    visibility_status = "رؤية ممتازة"
    
    if past_idx:
        if any(i < len(aligned.get("precipitation", [])) for i in past_idx):
            if sum(aligned["precipitation"][i] for i in past_idx if i < len(aligned["precipitation"])) > 10.0:
                freshwater_risk = "مرتفع جداً"
        valid_strat = [i for i in past_idx if i < len(aligned.get("swell_wave_height", [])) and i < len(aligned.get("wind_speed_10m", []))]
        if valid_strat:
            if sum(aligned["swell_wave_height"][i] for i in valid_strat)/len(valid_strat) < 0.2 and sum(aligned["wind_speed_10m"][i] for i in valid_strat)/len(valid_strat) < 10.0:
                stratification_risk = "مرتفع (بحر مسطح)"

    # Trap Fix: استبعاد 0.0 من تقاطع الأمواج
    cross_angles = [angle_diff(swd[i], wd_wave[i]) for i in range(len(swd)) if swd[i] != 0.0 and i < len(wd_wave) and wd_wave[i] != 0.0]
    is_cross_sea_dangerous = False
    if cross_angles and not is_mirror_sea:
        avg_cross, max_cross = sum(cross_angles) / len(cross_angles), max(cross_angles)
        if max_cross > 60 and avg_cross > 40: 
            cross_sea_risk = "بحر مختلط وخطير"
            is_cross_sea_dangerous = True
        elif max_cross > 45: 
            cross_sea_risk = "بحر مختلط متوسط"

    valid_vis = [v for v in vis if v > 0]
    if valid_vis:
        min_vis = min(valid_vis)
        if min_vis < 1000: visibility_status = "ضباب كثيف جداً"
        elif min_vis < 5000: visibility_status = "ضباب خفيف"

    steepness_vals = [h / (1.56 * (p**2)) for h, p in zip(wh, wp) if p > 0]
    avg_steepness = sum(steepness_vals) / len(steepness_vals) if steepness_vals else 0
    steepness_desc = "موج حاد وقصير" if avg_steepness > 0.06 else "موج طويل" if avg_steepness < 0.03 else "موج متوسط"

    tide_analysis = get_moon_and_tide_analysis(target_date_obj)
    golden_lock = "مد قوي" if tide_analysis["idx"] in [0,4] else "مد ضعيف" if tide_analysis["idx"] in [2,6] else "متوسط"

    sst_diff = max(sst) - min(sst) if len(sst) > 1 else 0
    sst_stability = "صدمة حرارية" if sst_diff > 2.0 else "تغير بطيء" if sst_diff > 1.0 else "مستقر تماماً"
    
    is_murky = "عكر" in sea_memory or "خامر" in sea_memory
    is_weedy = "صوفة" in sea_memory
    is_fresh = "مرتفع" in freshwater_risk
    is_dead = "مرتفع" in stratification_risk
    
    clarity_risk = is_murky
    weed_risk = is_weedy
    
    avg_sst = sum(sst)/len(sst) if sst else 0
    max_air_temp = max(ta) if ta else 0
    
    # Dynamic SST: تحمل الأسماك لحرارة أعلى في الصيف
    month = target_date_obj.month
    seabass_sst_limit = 20.0 if month in [6, 7, 8, 9] else 18.0

    bio_matrix = {
        "قاروص": {"status": "معدوم" if is_dead else "غائب تقريباً" if (not is_murky and is_mirror_sea) else "نشط جداً" if (avg_sst < seabass_sst_limit and is_murky and not is_weedy) else "نشط" if avg_sst < seabass_sst_limit else "غائب تقريباً", "reason": f"يكره البحر المسطح والصافي (المرآة) نهاراً. يحتاج عكراً ودرجة أقل من {seabass_sst_limit}°م."},
        "دوراد": {"status": "نشط" if (avg_sst > 18 and not is_murky) else "خامل", "reason": "يحب النظافة لكن الحرارة المرتفعة والبحر الساكن تخلق مشكلة بصرية له نهاراً."},
        "بوري": {"status": "نشط" if (not is_murky and not is_weedy and not is_mirror_sea) else "خامل", "reason": "البحر المرآوي يجلعه حذر جداً، يفضل حركة خفيفة على السطح."},
        "سارغ": {"status": "ضعيف", "reason": "يتأثر بالحرارة المرتفعة."}
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
        
        sea = "بحر مرآوي (ميت)" if max_h < 0.4 else "هادئ" if max_h < 0.8 else "متوسط الهيجان" if max_h < 1.2 else "هائج"
        
        # Trap Fix: تحويل 0.0 إلى None لمنع رسمها كـ "شمال"
        final_swd = None if avg_swd == 0.0 else avg_swd
        final_wd = None if avg_wave_dir == 0.0 else avg_wave_dir
        
        swell_angle = angle_diff(final_swd, orient) if final_swd else None
        wave_angle = angle_diff(final_wd, orient) if final_wd else None
        
        swell_wave_interaction = "لا يوجد تفاعل (قوى شبه معدومة)."
        if not is_mirror_sea and swell_angle is not None and wave_angle is not None:
            diff_sw = angle_diff(final_swd, final_wd)
            if diff_sw > 40: swell_wave_interaction = f"تقاطع بزاوية ({diff_sw:.0f}°) يخلق فوضى."
            else: swell_wave_interaction = "متوافقان، بحر منتظم."
        
        block_data = {
            "name":{"morning":"الصباح","afternoon":"الظهر","night":"الليل"}[key],
            "time_range":f"{all_times[target_idx[idxs[0]]].strftime('%H:%M')}-{all_times[target_idx[idxs[-1]]].strftime('%H:%M')}",
            "sea_state":sea,"wave_height":f"{min_h:.2f}-{max_h:.2f}","wave_power":round(avg_pow,2),
            "swell_height":f"{min(swh[i] for i in idxs):.2f}-{max(swh[i] for i in idxs):.2f}",
            "swell_period":round(avg_swp,1), "swell_dir": deg_to_compass(final_swd) if final_swd else "غير معروف",
            "swell_angle_diff": round(swell_angle,0) if swell_angle is not None else None,
            "wave_dir": deg_to_compass(final_wd) if final_wd else "غير معروف",
            "wave_angle_diff": round(wave_angle,0) if wave_angle is not None else None,
            "swell_wave_interaction": swell_wave_interaction,
            "wind_speed":f"{min_w:.1f}-{max_w:.1f}","wind_gust_peak":round(max(wg[i] for i in idxs),1),
            "wind_dir":wc_dom, "air_temp":round(avg_air,1),"precip":round(total_precip,1),
            "weather":weather_desc(most_code)
        }
        blocks.append(block_data)
        raw_blocks_meta.append({"name": block_data["name"], "max_h": max_h, "wind_cls": wc_dom})

    transitions = []
    for i in range(len(raw_blocks_meta) - 1):
        b1, b2 = raw_blocks_meta[i], raw_blocks_meta[i+1]
        changes = []
        if b1["wind_cls"] != b2["wind_cls"]:
            if "بحرية" in b2["wind_cls"] and "برية" in b1["wind_cls"]: changes.append("انقلاب رياح خطير من برية إلى بحرية.")
            elif "برية" in b2["wind_cls"] and "بحرية" in b1["wind_cls"]: changes.append("تحول ممتاز للرياح من بحرية إلى برية.")
        h_diff = b2["max_h"] - b1["max_h"]
        if h_diff < -0.3: changes.append(f"تهدأ واضح للموج (انخفاض أقصى {abs(h_diff):.2f}م).")
        elif h_diff > 0.3: changes.append(f"تصعيد في هيجان البحر (ارتفاع أقصى {h_diff:.2f}م).")
        if changes: transitions.append(f"من {b1['name']} إلى {b2['name']}: " + " | ".join(changes))

    reds, greens = [], []
    for i in range(len(wh)):
        hh = all_times[target_idx[i]].strftime("%H:%M")
        if wave_power[i] > 3 or wh[i] > 1.8 or wg[i] > 50 or pr[i] < 1005: reds.append(hh)
        is_night = all_times[target_idx[i]].hour < 6 or all_times[target_idx[i]].hour > 19
        if 0.3 <= wh[i] <= 1 and 0.1 <= wave_power[i] <= 1.5 and ws[i] < 27.8:
            if is_mirror_sea and not is_night: continue 
            greens.append(hh)
        
    # حساب الضغط (all_times مرتبة زمنياً، لذا الفهرس الأخير هو آخر 3 ساعات)
    avg_press = sum(pr)/len(pr) if pr else 0
    press_change = pr[-1] - pr[-4] if len(pr) >= 4 else (pr[-1] - pr[0] if len(pr) > 1 else 0)

    is_pressure_rising_fast = False
    is_pressure_dropping_fast = False
    
    if avg_press > 1025: press_abs_desc = "مرتفع جداً"; press_abs_effect = "خمول تام بسبب امتلاء المثانة الهوائية"
    elif avg_press < 1008: press_abs_desc = "منخفض جداً"; press_abs_effect = "تغذية عنيفة قبل العاصفة"
    else: press_abs_desc = "معتدل"; press_abs_effect = "لا تأثير مباشر"

    if press_change < -2.0: 
        pressure_state = f"ضغط {press_abs_desc} ({avg_press:.0f} hPa) في انخفاض حاد ({press_change:.1f}). {press_abs_effect}."
        is_pressure_dropping_fast = True
    elif press_change < -0.5: 
        pressure_state = f"ضغط {press_abs_desc} ({avg_press:.0f} hPa) في انخفاض بطيء ({press_change:.1f}). {press_abs_effect}."
    elif press_change > 1.5: 
        pressure_state = f"ضغط {press_abs_desc} ({avg_press:.0f} hPa) في ارتفاع حاد ({press_change:+.1f}). توقف فوري للتغذية."
        is_pressure_rising_fast = True
    else: pressure_state = f"ضغط {press_abs_desc} ومستقر ({avg_press:.0f} hPa، تغير {press_change:+.1f}). {press_abs_effect}."

    extra = {
        "pressure_avg":round(avg_press,1), "pressure_change_3h":round(press_change,1),
        "sunrise":sunrise, "sunset":sunset, "peak_gust_today":round(peak_gust,1),
        "is_mirror_sea": is_mirror_sea, "max_air_temp": round(max_air_temp, 1)
    }
    
    # الإضافة الأهم: قاموس الأعلام المنطقية (Boolean Flags) لمنع الاعتماد على النصوص
    flags = {
        "is_mirror_sea": is_mirror_sea,
        "is_lateral_strong": is_lateral_strong,
        "is_pressure_rising_fast": is_pressure_rising_fast,
        "is_pressure_dropping_fast": is_pressure_dropping_fast,
        "is_cross_sea_dangerous": is_cross_sea_dangerous,
        "is_murky": is_murky,
        "is_weedy": is_weedy
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
            "weed_risk": weed_risk, "clarity_risk": clarity_risk
        },
        "bio_matrix":bio_matrix, "avg_sst":round(avg_sst,1), "extra_info":extra,
        "transitions": transitions,
        "flags": flags # تمرير الأعلام بأمان
    }

# ==================== محرك التفاعلات (v12.0.0 - Boolean Logic) ====================
def calculate_interactions(agg: dict) -> List[str]:
    interactions = []
    hf = agg["hidden_factors"]
    blocks = agg["blocks"]
    extra = agg["extra_info"]
    
    # استخدام الأعلام المنطقية المضغوطة بدل مطابقة النصوص
    flags = agg.get("flags", {})
    is_mirror_sea = flags.get("is_mirror_sea", False)
    is_lateral_strong = flags.get("is_lateral_strong", False)
    is_pressure_rising_fast = flags.get("is_pressure_rising_fast", False)
    is_pressure_dropping_fast = flags.get("is_pressure_dropping_fast", False)
    is_cross_sea_dangerous = flags.get("is_cross_sea_dangerous", False)
    
    if is_mirror_sea:
        interactions.append("[تفاعل البحر المرآوي] الموج أقل من 0.4م. قوة الدفع المائي شبه معدومة. التيار الجانبي مستحيل فيزيائياً. الرؤية تحت الماء في أعلى مستوياتها.")
        
        if extra.get("max_air_temp", 0) > 28:
            interactions.append("[مفارقة البحر المرآوي النهاري] الأوضاع فيزيائياً مثالية للرمي (لا تيار، رصاصة 80 غرام ستثبت تماماً). لكن بيولوجياً: الماء صافي كالمرآة + شمس حارقة (+28°م) = الأسماك تشعر بالانكشاف التام وتختبئ في أعماق تتجاوز 50 متراً. الرصاصة والطعم سيكونان مرئيين بشكل ساطع على الرمل الأبيض في القاع الضحل، ولن تجرؤ أي سمكة على عبور 'المنطقة الميتة' القريبة من الشاطئ للوصول للطعم.")
            interactions.append("[الحسم النهائي - No-Go نهاراً (فخ بيولوجي)] الصيد النهاري مستحيل رغم جمال البحر وهدوئه. لا تذهب نهاراً.")
        else:
            interactions.append("[تفاعل التخفي الليلي] غياب الموج يعني غياب 'الغسيل' (Wash) الذي يخفي الخيط عادة. لذلك، يمنع منعاً باتاً استعمال اللونص (الرصاصة الطويلة) لأنها ستكون مكشوفة كجسم غريب على الرمل الأبيض وستنفر السمك. يجب استعمال رصاصة قصيرة (مستقيمة) لتندمج مع القاع، وطعم صغير جداً، وخيط Flurocarbon رفيع يتلاشى بصرياً تحت الماء.")
            interactions.append("[الحسم النهائي - Go مشروط بشروط التخفي] يمكن الصيد ليلاً فقط بشرط استعمال رصاصة قصيرة (وليس لونص طويل)، وطعم دقيق جداً، والرمي موجه نحو الخنادق العميقة قليلاً لتجنب المنطقة الضحلية الخالية من السمك.")
        return interactions

    for b in blocks:
        wind_is_onshore = "بحرية" in b["wind_dir"]
        wave_is_straight = b.get("wave_angle_diff") is not None and b["wave_angle_diff"] < 60
        
        if wind_is_onshore and wave_is_straight:
            interactions.append(f"[تفاعل ميكانيكي - {b['name']}] رياح بحرية + موج عمودي = تضخيم الهيجان المباشر، دفع قوي للرصاصة للخلف.")
        elif "برية" in b["wind_dir"]:
            interactions.append(f"[تفاعل ميكانيكي - {b['name']}] رياح برية = كبس الموج وتقليل المسافة، لكن عدم وجود تيار جانبي.")
            
        if "تقاطع" in b.get("swell_wave_interaction", ""):
            interactions.append(f"[تفاعل ميكانيكي - {b['name']}] {b['swell_wave_interaction']} فوضى عشوائية في حركة الماء.")

    if is_lateral_strong:
        interactions.append("[تفاعل الثبات] تيار جانبي قوي (تم حسابه رياضياً عبر المتجهات). استنتاج ميكانيكي: رصاصة أقل من 150 غرام ستنجرف خارج نطاق الرؤية.")
    else:
        interactions.append("[تفاعل الثبات] غياب تيار جانبي. استنتاج ميكانيكي: رصاصة 80-120 غرام كافية تماماً، الثقل الزائد سيضر بالمسافة دون فائدة.")

    if is_pressure_rising_fast:
        interactions.append("[تفاعل الفسيولوجيا] ارتفاع حاد في الضغط = توقف فوري للتغذية (المثانة الهوائية ممتلئة).")
    elif is_pressure_dropping_fast:
        interactions.append("[تفاعل الفسيولوجيا] انخفاض حاد = نافذة ذهبية للتغذية العنيفة.")

    if is_cross_sea_dangerous: interactions.append("[الحسم النهائي - No-Go] بحر مختلط خطير يمنع السيطرة.")
    elif len(agg["red_flags"]) >= 4: interactions.append(f"[الحسم النهائي - No-Go] هيجان متواصل ({len(agg['red_flags'])} ساعات خطر).")
    elif len(agg["green_flags"]) >= 3 and not is_lateral_strong and not is_pressure_rising_fast: interactions.append("[الحسم النهائي - Go] ظروف ميكانيكية مثالية (ثبات + نشاط).")
    else: interactions.append("[الحسم النهائي - No-Go] عدم توفر ظروف ميكانيكية أو بيولوجية كافية.")

    return interactions

# ==================== بناء السياق (v12.0.0) ====================
def build_context(req, agg, tz_name):
    beach = "رملي" if req.beach_type == "sandy" else "صخري"
    orient = req.beach_orientation
    extra = agg["extra_info"]
    hf = agg["hidden_factors"]
    
    chain_interactions = calculate_interactions(agg)

    facts = [
        f"[الأساسيات] شاطئ {beach} (اتجاه {orient}°) - توقيت {tz_name}.",
        f"[الذاكرة البحرية] {agg['sea_memory']}",
        f"[الضغط الجوي] {agg['pressure_state']}",
        f"[المد والقمر] {agg['tide_analysis']['tide_strength']}.",
        f"[حرارة الماء] {agg['avg_sst']}°م. حرارة الهواء القصوى: {extra.get('max_air_temp', 'N/A')}°م.",
    ]
    if hf["sudden_wind_shift"] != "لا يوجد": facts.append(f"[ديناميكا الرياح] {hf['sudden_wind_shift']}")

    for b in agg["blocks"]:
        sa_desc = "عمودي" if b.get("swell_angle_diff") is not None and 70 <= b["swell_angle_diff"] <= 110 else "مائل"
        wa_desc = "عمودي" if b.get("wave_angle_diff") is not None and 70 <= b["wave_angle_diff"] <= 110 else "مائل"
        facts.append(f"[فترة {b['name']} ({b['time_range']})] البحر {b['sea_state']}. السويل {b['swell_dir']} ({sa_desc})، الموج {b['wave_dir']} ({wa_desc}). الرياح {b['wind_dir']} {b['wind_speed']}كم/س.")

    if agg.get("transitions"): facts.append(f"[التحولات] " + " | ".join(agg["transitions"]))
    
    facts.append(f"[التوقيتات] خضراء: {', '.join(agg['green_flags']) if agg['green_flags'] else 'لا يوجد'} | حمراء: {', '.join(agg['red_flags']) if agg['red_flags'] else 'لا يوجد'}.")

    bio_text = "\n".join([f"- {fish}: {data['status']} ({data['reason']})" for fish, data in agg["bio_matrix"].items()])

    lines = [
        "=== ملف القضية الفيزيائي ===", "\n".join(facts), "\n",
        "=== مصفوفة الكائنات ===", bio_text, "\n",
        "=== سلسلة التفاعلات المترابطة (أساس القرار الحصري) ===", 
        *chain_interactions, "\n",
        "[مهمة النظام]: فسر سلسلة التفاعلات حرفياً. لا تبتكر."
    ]
    return "\n".join(lines)

SYSTEM_PROMPT = """أنت محلل فيزيائي بحري صارم. مهمتك هي تفسير سلسلة التفاعلات المرفقة فقط.

قاعدة ذهبية: القرار موجود مسبقاً في "سلسلة التفاعلات" تحت عنوان "[الحسم النهائي]". دورك هو كتابة تقرير يبرر هذا الحسم.

هيكل التقرير الإجباري:
1. تفكيك الحالة الراهنة.
2. تفسير سلسلة التفاعلات (تحويل كل معادلة فيزيائية لفقرة توضح تأثيرها على الرصاصة والطعم).
3. التكتيك الميداني الصارم (فقط في حالة Go أو Go مشروط):
   - الرصاصة: يجب أن تستنتج وزنها حصرياً من "تفاعل الثبات". (مثال: إذا قال التفاعل "غياب تيار جانبي"، قل "رصاصة 80-120 غرام"). لا تقترح أبداً أكثر من 120 غرام إذا لم يذكر التفاعل وجود تيار قوي.
   - الطعم: يستنتج من تفاعل البيئة (عكر/نظيف/مرآوي).
4. صياغة القرار النهائي: "القرار النهائي: [انسخ النص من الحسم النهائي]".

قواعد ممنوعة تحت طائلة الفشل:
- لا تقترح رصاصة ثقيلة (أكثر من 120 غرام) إذا كان البحر "مرآوياً" أو إذا قال التفاعل "غياب تيار جانبي".
- لا تقل أن الارتفاع الحاد في الضغط يجعل السمك يأكل (هذا خطأ بيولوجي، الارتفاع يوقف الأكل).
- لا تبدأ بالقرار. ابدأ بالتفكيك.
- لا توجد عبارة "يمكن تجربة" في حالة No-Go.

اكتب بالدارجة التونسية الجافة، فقرات مترابطة، بدون مجاملات."""

async def call_openrouter(ctx):
    headers = {"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json"}
    payload = {"model":MODEL_NAME,"messages":[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":ctx}],"max_tokens":4000,"temperature":0.1}
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
            return {"report": "قرار نهائي: No-Go مطلق. ظروف بحرية خطرة تهدد حياتك مباشرة.", "meta": {"hard_nogo": True}}

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
