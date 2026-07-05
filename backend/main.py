"""
Surfcasting Analytics API – v9.1.1 (Precision Fix)
تم إصلاح أخطاء الفهرسة وتخيل الذكاء الاصطناعي دون المساس بالمنطق الأساسي.
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
app = FastAPI(title="Surfcasting Analytics", version="9.1.1")
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
def health(): return {"status": "ok", "version": "9.1.1"}

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
    arr = ["شمال","شمال شرق","شمال شرق","شرق","شرق","جنوب شرق","جنوب شرق","جنوب","جنوب","جنوب غرب","جنوب غرب","غرب","غرب","شمال غرب","شمال غرب","شمال"]
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
        "precipitation": extract("precipitation", weather_hourly, w_map), "weather_code": [int(safe_float(x)) for x in extract("weather_code", weather_hourly, w_map)]
    }

TUNISIAN_BEACHES = { 
    "بنزرت": [{"name":"شاطئ الكورنيش","lat":37.2744,"lon":9.8739,"orientation":45,"type":"sandy"}],
    "نابل": [{"name":"شاطئ الحمامات","lat":36.4000,"lon":10.6167,"orientation":90,"type":"sandy"}],
}

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
    except: return 0

@app.post("/auto-orientation")
@limiter.limit("5/minute")
async def auto_orientation(request: Request, req: AutoOrientationRequest):
    orientation = await get_auto_orientation_overpass(req.latitude, req.longitude)
    if orientation != 0: return {"orientation": orientation, "source": "overpass"}
    return {"orientation": -1, "source": "none", "message": "تعذر التحديد التلقائي."}

# ==================== محرك التجميع الفيزيائي المتقدم ====================
def aggregate_physics(all_times, aligned, orient, target_date_obj, sunrise, sunset):
    tz = all_times[0].tzinfo if all_times else zoneinfo.ZoneInfo("UTC")
    target_start = datetime.combine(target_date_obj, datetime.min.time(), tzinfo=tz)
    target_end = target_start + timedelta(days=1)
    past_start = target_start - timedelta(hours=48)
    past_idx = [i for i, t in enumerate(all_times) if past_start <= t < target_start]
    target_idx = [i for i, t in enumerate(all_times) if target_start <= t < target_end]
    
    empty_res = {"past_avg_power":0,"dominant_wind":"غير معروف","blocks":[],"red_flags":[],"green_flags":[],"weed_risk":False,"tide_analysis":{},"clarity_risk":False,"sst_stability":"مستقر","bio_matrix":{},"avg_sst":0,"extra_info":{}, "sea_memory":"غير معروف", "lateral_current":"غير معروف", "pressure_state":"مستقر", "hidden_factors":{}}
    if not target_idx: return empty_res
    
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
    
    # 1. ذاكرة البحر وحالته الراهنة
    sea_memory = "بحر صافي وهادئ (لا توجد عوامل تعكير سابقة)"
    past_avg, past_sh = 0.0, 0.0
    if past_idx:
        p_wh = aligned.get("wave_height", []); p_wp = aligned.get("wave_period", [])
        p_swh = aligned.get("swell_wave_height", []); p_swp = aligned.get("swell_wave_period", [])
        p_ws = aligned.get("wind_speed_10m", []); p_wd = aligned.get("wind_direction_10m", [])
        p_prec = aligned.get("precipitation", [])
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

    # 2. ميكانيكا التيار الجانبي (تم الإصلاح الجذري هنا: المرور المباشر على القيم)
    valid_wd_wave = [angle_diff(w, orient) for w in wd_wave if w != 0] # إصلاح IndexError
    avg_wave_angle = sum(valid_wd_wave) / len(valid_wd_wave) if valid_wd_wave else 90
    lateral_force = math.sin(math.radians(avg_wave_angle))
    avg_wave_h = sum(wh) / len(wh) if wh else 0
    
    if lateral_force > 0.8 and avg_wave_h > 0.6: lateral_current = "تيار جارف قوي جداً (موازي للشاطئ): الرصاصة ستنجرف بسرعة كبيرة، والخط سيصبح قوساً. يتطلب رصاص ثقيل جداً أو تغيير زاوية الرمي عكس التيار بـ 30-40 درجة."
    elif lateral_force > 0.5 and avg_wave_h > 0.4: lateral_current = "تيار جانبي متوسط: سيحدث انجراف تدريجي للطعم. يجب مراقبة خيط الخط وتعديل الثقل."
    else: lateral_current = "تيار جانبي ضعيف أو معدوم: الموج يدفع للخلف وللأمام (عمودي)، الرصاصة ستثبت جيداً في القاع دون انجراف عرضي."

    # 3. العوامل الخفية
    freshwater_risk = "منخفض"
    stratification_risk = "منخفض"
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

    steepness_vals = [h / (1.56 * (p**2)) for h, p in zip(wh, wp) if p > 0]
    avg_steepness = sum(steepness_vals) / len(steepness_vals) if steepness_vals else 0
    if avg_steepness > 0.06: steepness_desc = "موج حاد وقصير (Steep). ينكسر بقوة، يخلق ماءاً أبيض كثيفاً. سيء للرمي البعيد."
    elif avg_steepness < 0.03: steepness_desc = "موج طويل وهادئ (Swell). ينكسر بعيداً ويخلق خنادق طبيعية. ممتاز."
    else: steepness_desc = "موج متوسط الانحدار."

    tide_analysis = get_moon_and_tide_analysis(target_date_obj)
    golden_lock = "مد قوي (Spring)" if tide_analysis["idx"] in [0,4] else "مد ضعيف (Neap)" if tide_analysis["idx"] in [2,6] else "متوسط"

    # 4. باقي الحسابات الأصلية (مع إصلاح تناقض الوضوح)
    if len(sst) > 1:
        sst_diff = max(sst) - min(sst)
        sst_stability = "صدمة حرارية (انخفاض/ارتفاع حاد)" if sst_diff > 2.0 else "تغير بطيء" if sst_diff > 1.0 else "مستقر تماماً"
    else: sst_stability = "بيانات غير كافية"
    
    max_swp = max(swp) if swp else 0
    onshore_hours = sum(1 for w in wind_cls if w.startswith("بحرية"))
    
    # إصلاح المنطق: إذا كانت الذاكرة تقول عكر، نتجاوز حساب اليوم ليكون منطقياً
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
        "دوراد": {"status": "شبه معدوم" if (is_murky or is_fresh) else "نشط" if (avg_sst > 18 and not is_murky) else "خامل", "reason": "يهرب من المياه العذبة والعكرة تماماً. أصعب سمكة في أيام الأمطار."},
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
        
        swell_angle = angle_diff(avg_swd, orient) if avg_swd else None
        wave_angle = angle_diff(avg_wave_dir, orient) if avg_wave_dir else None
        
        blocks.append({
            "name":{"morning":"الصباح","afternoon":"الظهر","night":"الليل"}[key],
            "time_range":f"{all_times[target_idx[idxs[0]]].strftime('%H:%M')}-{all_times[target_idx[idxs[-1]]].strftime('%H:%M')}",
            "wave_height":f"{min_h:.2f}-{max_h:.2f}", "wave_power":round(avg_pow,2),
            "swell_height":f"{min(swh[i] for i in idxs):.2f}-{max(swh[i] for i in idxs):.2f}",
            "swell_period":round(avg_swp,1), "swell_dir": deg_to_compass(avg_swd) if avg_swd else "غير معروف",
            "swell_angle_diff": round(swell_angle,0) if swell_angle is not None else None,
            "wave_dir": deg_to_compass(avg_wave_dir) if avg_wave_dir else "غير معروف",
            "wave_angle_diff": round(wave_angle,0) if wave_angle is not None else None,
            "wind_speed":f"{min_w:.1f}-{max_w:.1f}", "wind_gust_peak":round(max(wg[i] for i in idxs),1),
            "wind_dir":wc_dom
        })
        
    reds, greens = [], []
    for i in range(len(wh)):
        hh = all_times[target_idx[i]].strftime("%H:%M")
        if wave_power[i] > 3 or wh[i] > 1.8 or wg[i] > 50 or pr[i] < 1005: reds.append(hh)
        if 0.3 <= wh[i] <= 1 and 0.1 <= wave_power[i] <= 1.5 and ws[i] < 27.8: greens.append(hh)
        
    avg_press = sum(pr)/len(pr) if pr else 0
    press_change = pr[-1] - pr[-4] if len(pr) >= 4 else (pr[-1] - pr[0] if len(pr) > 1 else 0)
    
    # إصلاح تخيل الذكاء الاصطناعي: إرفاق القيمة الحقيقية لمنع الانقلاب
    if press_change < -2.0: pressure_state = f"انخفاض حاد ومستمر ({press_change:.1f} hPa): الأسماك تتسابق على الأكل قبل العاصفة. فرصة ذهبية."
    elif press_change < -0.5: pressure_state = f"انخفاض بطيء ({press_change:.1f} hPa): نشاط جيد ومستمر للأسماك."
    elif press_change > 1.5: pressure_state = f"ارتفاع حاد ({press_change:+.1f} hPa): الأسماك تمتلئ هواء وتتوقف عن الأكل لفترة."
    else: pressure_state = f"مستقر أو شبه مستقر ({press_change:+.1f} hPa): لا تأثير مباشر، العوامل الأخرى حاكمة."

    extra = {
        "pressure_avg":round(avg_press,1), "pressure_change_3h":round(press_change,1),
        "sunrise":sunrise, "sunset":sunset, "peak_gust_today":round(peak_gust,1)
    }
    return {
        "past_avg_power":round(past_avg,2), "dominant_wind":dominant, "blocks":blocks, 
        "red_flags":reds[:5], "green_flags":greens[:5], "weed_risk":weed, 
        "tide_analysis":tide_analysis, "clarity_risk":clarity_risk, "sst_stability":sst_stability,
        "sea_memory": sea_memory, "lateral_current": lateral_current, "pressure_state": pressure_state,
        "hidden_factors": {
            "freshwater_risk": freshwater_risk,
            "stratification_risk": stratification_risk,
            "wave_steepness": steepness_desc,
            "golden_lock": golden_lock
        },
        "bio_matrix":bio_matrix, "avg_sst":round(avg_sst,1), "extra_info":extra
    }

# ==================== بناء سياق التفاعلات ====================
def build_context(req, agg, tz_name):
    beach = "رملي" if req.beach_type == "sandy" else "صخري"
    orient = req.beach_orientation
    moon = agg["tide_analysis"]
    extra = agg["extra_info"]
    hf = agg["hidden_factors"]

    interactions = []
    
    interactions.append(f"ذاكرة البحر وحالته الراهنة: {agg['sea_memory']}")
    interactions.append(f"ميكانيكا الموج والتيار الجانبي: {agg['lateral_current']}")
    
    # إصلاح تخيل الضغط: إرسال المتوسط الحقيقي للسياق
    interactions.append(f"حالة الضغط الجوي (المتوسط {extra['pressure_avg']} hPa): {agg['pressure_state']}")
    
    if "مرتفع" in hf["freshwater_risk"]: interactions.append(f"⚠️ خطر السيول والمياه العذبة: {hf['freshwater_risk']}")
    if "مرتفع" in hf["stratification_risk"]: interactions.append(f"⚠️ خطر انعدام التمازج (البحر الميت تحت السطح): {hf['stratification_risk']}")

    if moon["idx"] in [0, 4]:
        interactions.append(f"🌊 تأثير المد: {moon['tide_strength']}. التيارات الجانبية ستكون قوية جداً. إذا كان الشاطئ رملياً ستنجرف الرصاصات بسرعة، تتطلب أوزاناً ثقيلة. ممتاز للقاروص الليلي، كارثي للدوراد.")
    elif moon["idx"] in [2, 6]:
        interactions.append(f"🌊 تأثير المد: {moon['tide_strength']}. التيارات ضعيفة. فرصة ذهبية للبوري والدوراد لعدم انجراف الطعم، لكن القاروص قد يكون بطيئاً.")

    for b in agg["blocks"]:
        sa = b.get("swell_angle_diff")
        wa = b.get("wave_angle_diff")
        sa_desc = "عمودي" if sa is not None and 70 <= sa <= 110 else "مائل"
        wa_desc = "عمودي" if wa is not None and 70 <= wa <= 110 else "مائل"
        interactions.append(f"في {b['name']}: السويل {b['swell_dir']} ({sa_desc})، الموج المحلي {b['wave_dir']} ({wa_desc})، الرياح {b['wind_dir']} {b['wind_speed']}كم/س.")

    if agg["weed_risk"]:
        interactions.append("🚨 خطر قاتل: الأعشاب البحرية (البوسيدونيا) مقتلعة من القاع بسبب طاقة الأمواج الماضية وتتجه نحو الشاطئ. ستغلف الخطافات وتجعل الصيد مستحيلاً.")

    bio_text = "\n".join([f"- {fish}: {data['status']} ({data['reason']})" for fish, data in agg["bio_matrix"].items()])

    lines = [
        f"المهمة: تحليل ظروف السيرفكاستينغ لشاطئ {beach} (اتجاه {orient}°) - توقيت {tz_name}.",
        f"البيانات الخام ممنوعة من الإعادة في التقرير. حلل التفاعلات التالية واستنتج:",
        "",
        "=== سلسلة التفاعلات الحرجة والمتسلسلة ===",
        *interactions,
        "",
        "=== العوامل الخفية المؤثرة ===",
        f"انحدار الموج (Steepness): {hf['wave_steepness']}",
        f"تزامن المد مع الشروق: {hf['golden_lock']}",
        "",
        f"=== تقييم الأنواع المستهدفة ===",
        bio_text,
        "",
        f"=== التوقيتات ===",
        f"أفضل ساعات (خضراء): {', '.join(agg['green_flags']) if agg['green_flags'] else 'لا يوجد'}",
        f"ساعات الخطر (حمراء): {', '.join(agg['red_flags']) if agg['red_flags'] else 'لا يوجد'}",
        f"الشروق {extra['sunrise']} | الغروب {extra['sunset']} | هبات قصوى {extra['peak_gust_today']} كم/س",
        "",
        "المطلوب: اكتب تقريراً تحليلياً مركباً ومفصلاً وطويلاً (ليس سردياً). ابدأ بالنتيجة النهائية (Go/No-Go) ثم فكك الأسباب. اربط كل ظاهرة بتأثيرها الميكانيكي على الخطاف والرصاصة والسمكة. اكتب فقرات طويلة ومترابطة تشرح سلسلة ردود الأفعال."
    ]
    return "\n".join(lines)

# ==================== برومبت التحليل الاستنتاجي المطور ====================
SYSTEM_PROMPT = """أنت عالم أحياء بحرية ومحلل فيزيائي متخصص حصرياً في صيد السرفكاستينغ (Surfcasting) في البحر المتوسط (تونس).
ممنوع تماماً سرد البيانات العددية التي أرسلتها لك. دورك هو "التوليد الاستنتاجي" (Generative Inference).

قواعد صارمة:
1. الالتزام الحرفي بالأرقام: إذا ذكر السياق أن الضغط "مستقر تماماً (+0.2 hPa)"، ممنوع تماماً القول في التقرير "الضغط مرتفع" أو "هناك ضغط عالٍ". يجب أن تقول "الضغط مستقر ولا يؤثر".
2. لا تقل أبداً "بناءً على البيانات" أو "الموج يبلغ" أو "الرياح سرعتها". بدلاً من ذلك قل "الموج الطويل يضرب الساحل بعمود تام مما يخلق...".
3. كل فقرة يجب أن تربط ظاهرة فيزيائية (موج/رياح/ضغط) بتأثيرها المباشر على سلوك السمكة أو ميكانيكية الصيد (انجراف، ثبات، رؤية).
4. القرار (Go/No-Go) يجب أن يكون في أول سطرين من التقرير، واضحاًWithout ambiguity.
5. إذا كان القرار No-Go، اشرح لماذا باستخدام سلسلة التفاعلات (مثلاً: لا نذهب ليس لأن الموج مرتفع، بل لأن الموج المرتفع مقترن بدورة طويلة قتلت الأعشاب وأصبح الماء عكراً والضغط مرتفع).
6. التوصيات الفنية (الرصاصة، الطعم، الزاوية) يجب أن تكون نتيجة مباشرة للتحليل الفيزيائي (مثلاً: "بسبب التيار الجانبي الناتج عن الموج المائل، سنستخدم رصاصة هرمية 150غ لضمان الثبات").
7. تحدث بالتفصيل عن: الرياح وتأثيرها على الموج وعلى مسافة الرمي. الموج والسويل والتيارات الجارفة وعلاقتها بالأعشاب وتحريك الرصاصة. الضغط وتغيراته وتأثيره على نشاط الأسماك. حالة البحر (خامر، صوافة، نظيف) مستندة للأيام السابقة.

اكتب بأسلوب ضيق، علمي، دارجة تونسية خالصة، لا تسامح ولا جمل تعبيرية فارغة."""

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
