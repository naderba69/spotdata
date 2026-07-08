"""
Surfcasting Analytics API – v14.0.0 (The Rich-Dynamics Engine)
- إصلاح التفكيك الديناميكي: ينتج تحليلاً غنياً لكل فترة حتى في البحر المرآوي.
- تجميع أسباب No-Go/Go بشكل صريح.
- التمييز بين "لا بيانات سويل" و"سويل معدوم فعلاً".
- تمرير أوقات المد إجبارياً في القسم الأول.
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
app = FastAPI(title="Surfcasting Analytics", version="14.0.0")
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
def health(): return {"status": "ok", "version": "14.0.0"}

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

def estimate_tidal_windows(target_date_obj, moon_analysis, sunrise_str, sunset_str):
    try:
        sr_h = int(sunrise_str.split(":")[0])
        ss_h = int(sunset_str.split(":")[0])
    except:
        sr_h, ss_h = 6, 18

    moon_age_hours = moon_analysis["phase_decimal"] * 29.53 * 24
    base_hw_hour = (moon_age_hours * 0.04) % 12 + 6
    
    hw1 = base_hw_hour
    lw1 = (hw1 + 6.2) % 24
    hw2 = (hw1 + 12.4) % 24
    lw2 = (lw1 + 12.4) % 24

    def format_time(h):
        hh = int(h)
        mm = int((h - hh) * 60)
        return f"{hh:02d}:{mm:02d}"

    windows = {"HW1": format_time(hw1), "LW1": format_time(lw1), "HW2": format_time(hw2), "LW2": format_time(lw2)}

    golden_windows = []
    def is_close(t1, t2, margin=1.5):
        return abs(t1 - t2) <= margin or abs(t1 - t2) >= (24 - margin)

    if is_close(hw1, sr_h, 1.5):
        golden_windows.append(f"ساعة ذهبية صباحية: تزامن المد العالي (HW1={windows['HW1']}) مع الفجر ({sunrise_str}).")
    if is_close(hw2, ss_h, 1.5):
        golden_windows.append(f"ساعة ذهبية مسائية: تزامن المد العالي (HW2={windows['HW2']}) مع الغروب ({sunset_str}).")
    if is_close(lw1 - 2, sr_h, 1.5) or is_close(lw2 - 2, sr_h, 1.5):
        golden_windows.append(f"نافذة الجزر الممتازة: تزامن بداية جزر قوي مع الفجر.")
    if is_close(lw1 - 2, ss_h, 1.5) or is_close(lw2 - 2, ss_h, 1.5):
        golden_windows.append(f"نافذة الجزر الممتازة: تزامن بداية جزر قوي مع الغروب.")
        
    if not golden_windows:
        hw1_gap = abs(hw1 - sr_h) if abs(hw1 - sr_h) < 12 else 24 - abs(hw1 - sr_h)
        hw2_gap = abs(hw2 - ss_h) if abs(hw2 - ss_h) < 12 else 24 - abs(hw2 - ss_h)
        golden_windows.append(f"لا توجد ساعة ذهبية. HW1 ({windows['HW1']}) يبعد {hw1_gap:.1f} ساعة عن الفجر. HW2 ({windows['HW2']}) يبعد {hw2_gap:.1f} ساعة عن الغروب.")

    return windows, golden_windows

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

TUNISIAN_BEACHES = [
    {"name":"شاطئ الحمامات","lat":36.4000,"lon":10.6167,"orientation":90,"type":"sandy"},
    {"name":"شاطئ قليبية","lat":36.8500,"lon":11.1000,"orientation":45,"type":"sandy"},
    {"name":"شاطئ قرطاج","lat":36.8528,"lon":10.3264,"orientation":90,"type":"sandy"},
    {"name":"شاطئ بوجعفر","lat":35.8333,"lon":10.6333,"orientation":90,"type":"sandy"},
    {"name":"شاطئ رادس","lat":36.7500,"lon":10.2833,"orientation":0,"type":"sandy"},
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

# ==================== محرك التجميع الفيزيائي ====================
def aggregate_physics(all_times, aligned, orient, target_date_obj, sunrise, sunset):
    tz = all_times[0].tzinfo if all_times else zoneinfo.ZoneInfo("UTC")
    target_start = datetime.combine(target_date_obj, datetime.min.time(), tzinfo=tz)
    target_end = target_start + timedelta(days=1)
    past_start = target_start - timedelta(hours=48)
    past_idx = [i for i, t in enumerate(all_times) if past_start <= t < target_start]
    target_idx = [i for i, t in enumerate(all_times) if target_start <= t < target_end]
    
    nogo_reasons = []  # ← NEW: تجميع أسباب الرفض
    
    empty_res = {"sea_memory":"غير معروف","lateral_current":"غير معروف","pressure_state":"مستقر","tide_analysis":{},"sst_stability":"مستقر","bio_matrix":{},"avg_sst":0,"hidden_factors":{},"blocks":[],"red_flags":[],"green_flags":[],"extra_info":{}, "transitions":[], "flags":{}, "nogo_reasons":[]}
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
    
    # ← NEW: التمييز بين "لا بيانات سويل" و"سويل فعلاً معدوم"
    has_swell_data = len(swh) > 0 and not all(v == 0.0 for v in swh)
    actual_swell_exists = has_swell_data and max(swh) > 0.05
    
    sea_memory = "بحر صافي وهادئ"
    past_sh = 0.0
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
            past_rain = past_rain / len(valid_past)
            if past_avg > 6.0 and past_onshore_ratio > 0.4: sea_memory = "بحر خامر وعكر جداً."
            elif past_avg > 4.0 and past_onshore_ratio > 0.3: sea_memory = "بحر يعكر ببطء."
            if past_sh > 0.8 and past_avg > 4.0: sea_memory += " | تحذير صوفة."
            if past_rain > 10.0: sea_memory += " | سيول."

    lateral_fx = 0.0; lateral_fy = 0.0; max_wh = max(wh) if wh else 0.0
    for i in range(len(wh)):
        w_dir = wd_wave[i] if i < len(wd_wave) else 0.0
        if w_dir != 0.0:
            angle = math.radians(angle_diff(w_dir, orient))
            force = wh[i] * wh[i] 
            lateral_fx += force * math.sin(angle)
            lateral_fy += force * math.cos(angle)
            
    total_force = math.sqrt(lateral_fx**2 + lateral_fy**2)
    lateral_force_ratio = abs(lateral_fx) / total_force if total_force > 0 else 0
    avg_wave_h = sum(wh) / len(wh) if wh else 0
    is_mirror_sea = max_wh < 0.4
    is_lateral_strong = not is_mirror_sea and lateral_force_ratio > 0.7 and avg_wave_h > 0.6
    
    if is_mirror_sea:
        nogo_reasons.append("بحر مرآوي: الموج أقل من 0.4م، لا تيارات ولا حركة سطحية لتجذب الأسماك.")
    
    lateral_current = "تيار جانبي معدوم (بحر مرآوي)" if is_mirror_sea else "تيار جارف قوي جداً" if is_lateral_strong else "تيار جانبي متوسط" if (lateral_force_ratio > 0.4 and avg_wave_h > 0.4) else "تيار جانبي ضعيف"

    cross_angles = [angle_diff(swd[i], wd_wave[i]) for i in range(len(swd)) if swd[i] != 0.0 and i < len(wd_wave) and wd_wave[i] != 0.0]
    is_cross_sea_dangerous = False
    cross_sea_risk = "منخفض"
    if cross_angles and not is_mirror_sea:
        avg_cross, max_cross = sum(cross_angles) / len(cross_angles), max(cross_angles)
        if max_cross > 60 and avg_cross > 40: cross_sea_risk = "بحر مختلط وخطير"; is_cross_sea_dangerous = True
        elif max_cross > 45: cross_sea_risk = "بحر مختلط متوسط"

    if is_cross_sea_dangerous:
        nogo_reasons.append("بحر مختلط خطير: السويل والموج المحلي يتقاطعان بزاوية كبيرة مما يخلق فوضى دوامية.")

    steepness_vals = [h / (1.56 * (p**2)) for h, p in zip(wh, wp) if p > 0]
    avg_steepness = sum(steepness_vals) / len(steepness_vals) if steepness_vals else 0
    steepness_desc = "موج حاد وقصير" if avg_steepness > 0.06 else "موج طويل" if avg_steepness < 0.03 else "موج متوسط"

    tide_analysis = get_moon_and_tide_analysis(target_date_obj)
    tidal_windows, golden_windows = estimate_tidal_windows(target_date_obj, tide_analysis, sunrise, sunset)
    
    # ← NEW: تحقق من قوة المد
    is_neap_tide = tide_analysis["idx"] in [2, 6]
    is_spring_tide = tide_analysis["idx"] in [0, 4]
    if is_neap_tide:
        nogo_reasons.append(f"مد ضعيف (Neap Tides - {tide_analysis['name']}): الفرق بين المد والجزر ضئيل، لا تيارات غذائية قوية.")
    
    has_golden_window = any("تزامن" in g for g in golden_windows)
    if not has_golden_window:
        nogo_reasons.append("لا توجد ساعة ذهبية: لا تزامن بين أوقات المد والفجر/الغروب.")

    sst_diff = max(sst) - min(sst) if len(sst) > 1 else 0
    sst_stability = "صدمة حرارية" if sst_diff > 2.0 else "تغير بطيء" if sst_diff > 1.0 else "مستقر تماماً"
    is_murky = "عكر" in sea_memory or "خامر" in sea_memory
    is_weedy = "صوفة" in sea_memory
    avg_sst = sum(sst)/len(sst) if sst else 0
    max_air_temp = max(ta) if ta else 0
    month = target_date_obj.month
    seabass_sst_limit = 20.0 if month in [6, 7, 8, 9] else 18.0
    
    if avg_sst > seabass_sst_limit:
        nogo_reasons.append(f"حرارة ماء عالية ({avg_sst:.1f}°م): تتجاوز حد القاروص ({seabass_sst_limit}°م في {month}).")

    bio_matrix = {
        "قاروص": {"status": "نشط جداً" if (avg_sst < seabass_sst_limit and is_murky) else "نشط" if avg_sst < seabass_sst_limit else "غائب تقريباً", "reason": f"يحتاج عكراً ودرجة أقل من {seabass_sst_limit}°م."},
        "دوراد": {"status": "نشط" if (avg_sst > 18 and not is_murky) else "خامل", "reason": "يحب النظافة ويأتي مع المد العالي ليلاً."},
        "بوري": {"status": "نشط" if (not is_murky and not is_weedy and not is_mirror_sea) else "خامل", "reason": "يحتاج حركة سطحية. في بحر مرآوي يختفي تماماً." if is_mirror_sea else "يحتاج حركة سطحية."},
        "سارغ": {"status": "ضعيف", "reason": "يتأثر بالحرارة."}
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
        
        sea = "بحر مرآوي" if max_h < 0.4 else "هادئ" if max_h < 0.8 else "متوسط الهيجان" if max_h < 1.2 else "هائج"
        final_swd = None if avg_swd_b == 0.0 else avg_swd_b
        final_wd = None if avg_wave_dir == 0.0 else avg_wave_dir
        swell_angle = angle_diff(final_swd, orient) if final_swd else None
        wave_angle = angle_diff(final_wd, orient) if final_wd else None
        
        swell_wave_interaction = "متوافقان"
        if not is_mirror_sea and swell_angle is not None and wave_angle is not None and final_swd and final_wd:
            diff_sw = angle_diff(final_swd, final_wd)
            if diff_sw > 40: swell_wave_interaction = "متقاطعان بشدة"
            elif diff_sw > 25: swell_wave_interaction = "متقاطعان بسيط"
        
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
            # ← NEW: قيم خام للسياق
            "_raw": {
                "avg_wave_h": round(avg_h, 3), "max_wave_h": round(max_h, 3),
                "avg_wind": round(avg_w, 1), "max_wind": round(max_w, 1),
                "max_gust": round(max_gust_b, 1),
                "swell_h": round(avg_swh_b, 3), "swell_p": round(avg_swp_b, 1),
                "air_temp": round(avg_air, 1), "pressure": round(avg_press_b, 1),
                "visibility": round(avg_vis_b, 0),
                "has_swell": actual_swell_exists
            }
        }
        blocks.append(block_data)

    reds, greens = [], []
    for i in range(len(wh)):
        hh = all_times[target_idx[i]].strftime("%H:%M")
        if wave_power[i] > 3 or wh[i] > 1.8 or wg[i] > 50 or pr[i] < 1005: reds.append(hh)
        if 0.3 <= wh[i] <= 1 and 0.1 <= wave_power[i] <= 1.5 and ws[i] < 27.8: greens.append(hh)
        
    avg_press = sum(pr)/len(pr) if pr else 0
    press_change = pr[-1] - pr[-4] if len(pr) >= 4 else (pr[-1] - pr[0] if len(pr) > 1 else 0)
    is_pressure_rising_fast = press_change > 1.5
    is_pressure_dropping_fast = press_change < -2.0
    
    if is_pressure_rising_fast:
        nogo_reasons.append(f"ضغط مرتفع حاد ({press_change:+.1f} هكتوباسكال): يبطئ النشاط البيولوجي ويوقف الأكل.")
    
    if is_pressure_dropping_fast: pressure_state = f"انخفاض حاد ({press_change:.1f}). تغذية عنيفة."
    elif is_pressure_rising_fast: pressure_state = f"ارتفاع حاد ({press_change:+.1f}). توقف فوري."
    else: pressure_state = f"مستقر ({press_change:+.1f})."

    extra = {
        "pressure_avg":round(avg_press,1), "peak_gust_today":round(peak_gust,1),
        "sunrise":sunrise, "sunset":sunset, "max_air_temp": round(max_air_temp, 1),
        "is_mirror_sea": is_mirror_sea, "tidal_windows": tidal_windows, "golden_windows": golden_windows,
        "has_swell_data": has_swell_data, "actual_swell_exists": actual_swell_exists
    }
    
    flags = {
        "is_mirror_sea": is_mirror_sea, "is_lateral_strong": is_lateral_strong,
        "is_pressure_rising_fast": is_pressure_rising_fast, "is_pressure_dropping_fast": is_pressure_dropping_fast,
        "is_cross_sea_dangerous": is_cross_sea_dangerous, "is_murky": is_murky, "is_weedy": is_weedy,
        "has_golden_window": has_golden_window, "is_neap_tide": is_neap_tide, "is_spring_tide": is_spring_tide
    }

    # ← NEW: الحسم النهائي
    is_go = has_golden_window and not is_pressure_rising_fast and not is_cross_sea_dangerous and not is_mirror_sea
    final_verdict = "Go" if is_go else "No-Go"

    return {
        "dominant_wind":dominant, "blocks":blocks, "red_flags":reds[:5], "green_flags":greens[:5],
        "sea_memory":sea_memory, "lateral_current":lateral_current, "pressure_state":pressure_state,
        "tide_analysis":tide_analysis, "sst_stability":sst_stability,
        "hidden_factors": {"cross_sea_risk": cross_sea_risk, "wave_steepness": steepness_desc, "golden_lock": "مد قوي" if is_spring_tide else "مد ضعيف" if is_neap_tide else "متوسط"},
        "bio_matrix":bio_matrix, "avg_sst":round(avg_sst,1), "extra_info":extra,
        "transitions": [], "flags": flags,
        "nogo_reasons": nogo_reasons,  # ← NEW
        "final_verdict": final_verdict  # ← NEW
    }

# ==================== محرك التفكيك الديناميكي (v14.0.0) ====================
def calculate_interactions(agg: dict) -> List[str]:
    interactions = []
    flags = agg.get("flags", {})
    extra = agg.get("extra_info", {})
    blocks = agg.get("blocks", [])
    is_mirror_sea = flags.get("is_mirror_sea", False)
    is_lateral_strong = flags.get("is_lateral_strong", False)
    is_pressure_rising_fast = flags.get("is_pressure_rising_fast", False)
    is_pressure_dropping_fast = flags.get("is_pressure_dropping_fast", False)
    is_cross_sea_dangerous = flags.get("is_cross_sea_dangerous", False)
    has_golden_window = flags.get("has_golden_window", False)
    is_neap_tide = flags.get("is_neap_tide", False)
    is_spring_tide = flags.get("is_spring_tide", False)
    actual_swell_exists = extra.get("actual_swell_exists", False)
    
    golden_windows = extra.get("golden_windows", [])
    tidal_windows = extra.get("tidal_windows", {})
    tide_analysis = agg.get("tide_analysis", {})
    bio_matrix = agg.get("bio_matrix", {})
    sea_memory = agg.get("sea_memory", "")
    avg_sst = agg.get("avg_sst", 0)
    pressure_state = agg.get("pressure_state", "")
    nogo_reasons = agg.get("nogo_reasons", [])
    final_verdict = agg.get("final_verdict", "No-Go")

    # ===== 1. التوقيت المدوي (دائماً) =====
    hw1 = tidal_windows.get("HW1", "?")
    lw1 = tidal_windows.get("LW1", "?")
    hw2 = tidal_windows.get("HW2", "?")
    lw2 = tidal_windows.get("LW2", "?")
    tide_name = tide_analysis.get("name", "?")
    tide_str = tide_analysis.get("tide_strength", "?")
    
    interactions.append(f"[التوقيت المدوي] HW1: {hw1} | LW1: {lw1} | HW2: {hw2} | LW2: {lw2}. القمر: {tide_name}. القوة: {tide_str}.")
    
    if has_golden_window:
        for g in golden_windows:
            if "تزامن" in g:
                interactions.append(f"[ساعة ذهبية] {g}")
    else:
        interactions.append(f"[ساعات ذهبية] {golden_windows[0] if golden_windows else 'لا توجد معلومات كافية.'}")

    # ===== 2. التفكيك الديناميكي الزمني (لكل فترة) =====
    for b in blocks:
        name = b['name']
        time_range = b['time_range']
        raw = b.get("_raw", {})
        sea_state = b['sea_state']
        wind_cls = b['wind_dir']
        avg_wind = raw.get("avg_wind", 0)
        max_gust = raw.get("max_gust", 0)
        wave_interaction = b.get("swell_wave_interaction", "")
        has_swell = raw.get("has_swell", False)
        air_temp = raw.get("air_temp", 0)
        block_swell_h = raw.get("swell_h", 0)
        
        interactions.append(f"[ديناميكية {name} ({time_range})] حالة البحر: {sea_state}.")
        
        if is_mirror_sea:
            # تحليل غني للبحر المرآوي لكل فترة
            interactions.append(f"  → الرياح: {wind_cls} بمتوسط {avg_wind} كم/س، هبات قصوى {max_gust} كم/س.")
            
            if avg_wind < 5:
                interactions.append(f"  → تأثير الرياح: شبه معدومة. لا دفع للطعم، مسافة الرمي تعتمد كلياً على القذف اليدوي.")
            elif avg_wind < 15:
                if "بحرية" in wind_cls:
                    interactions.append(f"  → تأثير الرياح: بحرية خفيفة قد تضيف بضع أمتار للرمي لكنها لا تخلق تيارات.")
                elif "برية" in wind_cls:
                    interactions.append(f"  → تأثير الرياح: برية خفيفة تكبس السطح وتسطحه أكثر.")
                else:
                    interactions.append(f"  → تأثير الرياح: جانبية خفيفة، تأثيرها محدود على مسافة الرمي.")
            
            if max_gust > 35:
                interactions.append(f"  → تحذير الهبات: {max_gust} كم/س قد تقطع الخط رغم هدوء البحر العام.")
            else:
                interactions.append(f"  → الهبات: ضمن الحدود الآمنة ({max_gust} كم/س). لا تأثير مفاجئ.")
            
            # السويل
            if has_swell:
                interactions.append(f"  → السويل: موجود بارتفاع {block_swell_h:.2f}م لكنه غير كافٍ لخلق حركة قاعية.")
            else:
                interactions.append(f"  → السويل: {b['swell_dir']}. لا مساهمة في الحركة.")
            
            # القاع والاستقرار
            interactions.append(f"  → القاع: ثابت تماماً كالمرآة. لا تحريك للرمال ولا الأعشاب. الطعم يسقط ويبقى في مكانه بلا حركة.")
            
            # تقنيات مستحيلة
            interactions.append(f"  → تقنيات ممنوعة: اللونص (يعتمد على التيار لحمل الطعم). الرمي الطويل بلا فائدة.")
            
            # تحليل خاص بكل فترة
            if name == "الصباح":
                interactions.append(f"  → حكم الصباح: منطقة ميتة. لا حركة سطحية = لا بوري. لا عكارة = لا قاروص يبحث عن غطاء. درجة الحرارة {air_temp}°م.")
            elif name == "الظهيرة":
                interactions.append(f"  → حكم الظهيرة: أسوأ فترة. الحرارة ترتفع ({air_temp}°م)، أي نشاط متبقي من الصباح يتلاشى. ضوء قوي يخفي الطعم. ممنوع تماماً.")
            elif name == "الليل":
                seabass_status = bio_matrix.get("قاروص", {}).get("status", "غير معروف")
                seabass_reason = bio_matrix.get("قاروص", {}).get("reason", "")
                if "نشط" in seabass_status:
                    interactions.append(f"  → حكم الليل: الفرصة الوحيدة. القاروص ({seabass_status}: {seabass_reason}) يتحول لمفترس في الظلام. استخدم رصاصة قصيرة للتخفي قرب الصخور أو الكسرات.")
                else:
                    interactions.append(f"  → حكم الليل: حتى الظلام لا ينقذ الوضع. القاروص {seabass_status} ({seabass_reason}).")
                    
        else:
            # تحليل البحر غير المرآوي
            interactions.append(f"  → الرياح: {wind_cls} بمتوسط {avg_wind} كم/س، هبات قصوى {max_gust} كم/س.")
            
            wind_is_onshore = "بحرية" in wind_cls
            wave_angle = b.get("wave_angle_diff")
            wave_is_straight = wave_angle is not None and wave_angle < 60
            
            if wind_is_onshore and wave_is_straight:
                interactions.append(f"  → تطابق الرياح والموج: رياح بحرية تتقابل موجاً عمودياً (من {b['wave_dir']} بزاوية {wave_angle}°). يضخم الهيجان ويخلق دفعاً قوياً للخلف (Backwash) يعيق الرمي.")
            elif wind_is_onshore:
                interactions.append(f"  → رياح بحرية مع موج مائل (من {b['wave_dir']} بزاوية {wave_angle}°). تشويش على السطح وعدم استقرار.")
            elif "برية" in wind_cls:
                interactions.append(f"  → رياح برية ({wind_cls}) تكبس الموج وتسطح البحر. تقلل المسافة لكن تمنع الموج من الوصول للشاطئ.")
            
            if "بشدة" in wave_interaction:
                interactions.append(f"  → تقاطع السويل والموج: {wave_interaction}. يخلق فوضى عشوائية وتيارات دوامية خطيرة.")
            elif "بسيط" in wave_interaction:
                interactions.append(f"  → تقاطع السويل والموج: {wave_interaction}. عدم استقرار معتدل.")
            
            if max_gust > 35:
                interactions.append(f"  → تحذير الهبات: {max_gust} كم/س قد تقطع الخط أو تسبب صدمة مفاجئة.")
            
            # تأثير القاع
            if "هائج" in sea_state:
                interactions.append(f"  → القاع: مضطرب بشدة. الرمال تتحرك والأعشاب تطفو. الطعم يتنقل بلا توقف.")
            elif "متوسط" in sea_state:
                interactions.append(f"  → القاع: حركة معتدلة. مناطق الكسرات تكون نشطة غذائياً.")
            else:
                interactions.append(f"  → القاع: مستقر نسبياً.")

    # ===== 3. تفاعلات إضافية =====
    # حرارة الماء
    month_hint = ""
    if avg_sst > 22: month_hint = " حرارة مرتفعة تبعث السمك للعمق."
    elif avg_sst < 14: month_hint = " حرارة منخفضة تبطئ الأيض."
    interactions.append(f"[حرارة الماء] {avg_sst}°م. الاستقرار: {agg.get('sst_stability', '?')}.{month_hint}")
    
    # ذاكرة البحر
    interactions.append(f"[ذاكرة البحر] {sea_memory}")
    
    # الضغط
    if is_pressure_dropping_fast:
        interactions.append(f"[الضغط] {pressure_state} نافذة ذهبية إضافية: الأسماك تتغذى بعنف قبل العاصفة.")
    elif is_pressure_rising_fast:
        interactions.append(f"[الضغط] {pressure_state} يبطئ ويوقف النشاط البيولوجي.")
    else:
        interactions.append(f"[الضغط] {pressure_state} لا تحفيز ولا تثبيط من الضغط.")

    # ===== 4. الحسم النهائي =====
    if final_verdict == "Go":
        interactions.append(f"[الحسم النهائي - Go] الظروف مؤاتية. الساعة الذهبية موجودة والبحر يوفر حركة كافية.")
    else:
        if nogo_reasons:
            reasons_text = " | ".join(nogo_reasons)
            interactions.append(f"[الحسم النهائي - No-Go] الأسباب التراكمية: {reasons_text}")
        else:
            interactions.append(f"[الحسم النهائي - No-Go] الظروف غير كافية لصيد مجدٍ.")

    return interactions

def build_context(req, agg, tz_name):
    beach = "رملي" if req.beach_type == "sandy" else "صخري"
    extra = agg["extra_info"]
    hf = agg["hidden_factors"]
    chain_interactions = calculate_interactions(agg)

    facts = [
        f"شاطئ {beach} (اتجاه {req.beach_orientation}°).",
        f"الذاكرة: {agg['sea_memory']}",
        f"الضغط: {agg['pressure_state']}",
        f"القمر: {agg['tide_analysis']['tide_strength']}.",
        f"حرارة الماء: {agg['avg_sst']}°م. الهواء القصوى: {extra.get('max_air_temp', 'N/A')}°م.",
    ]
    facts.append(f"خضراء: {', '.join(agg['green_flags']) if agg['green_flags'] else 'لا يوجد'} | حمراء: {', '.join(agg['red_flags']) if agg['red_flags'] else 'لا يوجد'}.")

    bio_text = "\n".join([f"- {fish}: {data['status']} ({data['reason']})" for fish, data in agg["bio_matrix"].items()])

    # ← NEW: بيانات Blocks الخام للسياق
    blocks_raw_text = []
    for b in agg["blocks"]:
        r = b.get("_raw", {})
        blocks_raw_text.append(
            f"  {b['name']} ({b['time_range']}): بحر={b['sea_state']}, "
            f"موج أقصى={r.get('max_wave_h',0):.2f}م، رياح={b['wind_dir']} ({r.get('avg_wind',0)} كم/س), "
            f"هبات={r.get('max_gust',0)} كم/س، سويل={'موجود' if r.get('has_swell') else 'معدوم'} "
            f"({r.get('swell_h',0):.2f}م)، حرارة={r.get('air_temp',0)}°م، "
            f"تقاطع سويل/موج={b.get('swell_wave_interaction','?')}"
        )

    lines = [
        "=== ملف القضية ===", "\n".join(facts), "\n",
        "=== الكائنات ===", bio_text, "\n",
        "=== بيانات الفترات الخام ===", *blocks_raw_text, "\n",
        "=== التفكيك الديناميكي والتفاعلات ===", *chain_interactions
    ]
    return "\n".join(lines)

SYSTEM_PROMPT = """أنت خبير سيرفكاستينغ تونسي. تفهم لغة المد والجزر والساعات الذهبية.
القرار النهائي محدد سلفاً في "الحسم النهائي" داخل التفاعلات. لا تغيره أبداً.

هيكل التقرير الإجباري:

1. التوقيت المدوي:
- اذكر أوقات HW1, LW1, HW2, LW2 بالأرقام.
- اذكر اسم القمر وقوة المد.
- اذكر هل توجد ساعة ذهبية (تزامن مد مع فجر/غروب) مع شرح تأثيرها.
- إذا كان المد ضعيفاً (Neap)، اشرح ما يعنيه عملياً: تيارات غذائية ضعيفة، فرق مستوى ضئيل.

2. التفكيك الديناميكي الزمني:
- لكل فترة (صباح، ظهيرة، ليل)، خذ التحليل الميكانيكي من "ديناميكية الفترة" واكتبه بلغة تونسية احترافية.
- ربط السبب بالنتيجة: الرياح الفلانية تسبب كذا → مما يعني كذا للصيد.
- لا تكرر نفس الجملة مرتين. كل فترة必须有 تحليل مختلف.
- لا تسرد أرقاماً خاماً بلا شرح، بل فسر تأثيرها الفيزيائي.
- إذا كان البحر مرآوياً، اشرح لماذا كل تقنية غير ممكنة (لا لونص، لا بوري، إلخ).

3. التكتيك الميداني (قاعدة حفرية):
- اكتبه حصرياً وإذا بدأ الحسم بـ "Go".
- يجب أن يراعي: نوع السمك النشط، حالة البحر، اتجاه الرياح، التقنية المناسبة.
- إذا كان الحسم "No-Go"، امسح هذا القسم بالكامل. لا تكتب "لا يوجد تكتيك" - فقط لا تذكره.

قواعد صارمة:
- لا تقل "كما ذكرنا سابقاً" أو "بالنسبة لما سبق". كل قسم مستقل.
- لا تخلق معلومات ليست في السياق.
- اكتب بالدارجة التونسية الاحترافية المفصلة."""

async def call_openrouter(ctx):
    headers = {"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json"}
    payload = {"model":MODEL_NAME,"messages":[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":ctx}],"max_tokens":4500,"temperature":0.1}
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
            return {"report": "قرار نهائي: No-Go مطلق. ظروف بحرية خطرة تهدد حياتك.", "meta": {"hard_nogo": True}}

        ctx = build_context(req, agg, tz_name)
        report = await call_openrouter(ctx)
        
        # ← NEW: تنظيف _raw من الـ blocks قبل الإرسال
        clean_blocks = []
        for b in agg["blocks"]:
            clean_b = {k: v for k, v in b.items() if k != "_raw"}
            clean_blocks.append(clean_b)
        
        meta = {
            "timezone": tz_name, "target_date": target_dt.isoformat(), "hard_nogo": False,
            "tidal_estimation": agg["extra_info"]["tidal_windows"],
            "golden_windows_detected": agg["extra_info"]["golden_windows"],
            "final_verdict": agg["final_verdict"],
            "nogo_reasons": agg["nogo_reasons"],
            "blocks": clean_blocks
        }
        return {"report": report, "meta": meta}
    except HTTPException: raise
    except Exception as e:
        logger.error(f"generate-report error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, detail="فشل إنشاء التقرير")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
