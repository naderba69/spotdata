"""
Surfcasting Analytics API – v4.2 (Stable upgrade from v3.0, with detailed analysis & spot reasons)
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
app = FastAPI(title="Surfcasting Analytics", version="4.2.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY مفقود")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "google/gemini-2.5-flash"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

cache = {}
cache_lock = asyncio.Lock()
CACHE_TTL = 3600

class ReportRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    beach_orientation: int = Field(..., ge=0, le=360)
    beach_type: str = Field(..., pattern="^(sandy|rocky)$")
    target_date: str = Field(..., pattern="^(today|tomorrow|day_after)$")

class AutoOrientationRequest(BaseModel):
    latitude: float
    longitude: float

class ScanRequest(BaseModel):
    governorates: List[str]
    target_date: str = "today"
    target_species: Optional[str] = None

@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})

@app.get("/health")
def health():
    return {"status": "ok", "openrouter": bool(OPENROUTER_API_KEY), "model": MODEL_NAME}

# --------------- دوال الشبكة (نفس الأصل) ---------------
async def fetch_with_retry(url, params, max_retries=3, timeout=20):
    for attempt in range(1, max_retries+1):
        try:
            async with httpx.AsyncClient() as c:
                r = await c.get(url, params=params, timeout=timeout)
                r.raise_for_status()
                return r.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries:
                await asyncio.sleep(5*attempt)
                continue
            raise
        except (httpx.ConnectError, httpx.TimeoutException):
            if attempt < max_retries:
                await asyncio.sleep(2**attempt)
                continue
            raise

async def fetch_marine(lat, lon, start, end):
    return await fetch_with_retry(MARINE_URL, {
        "latitude": lat, "longitude": lon,
        "hourly": ["wave_height","wave_period","wave_direction",
                  "swell_wave_height","swell_wave_period","swell_wave_direction",
                  "sea_surface_temperature"],
        "timezone": "auto", "start_date": start, "end_date": end
    })

async def fetch_weather(lat, lon, start, end):
    return await fetch_with_retry(WEATHER_URL, {
        "latitude": lat, "longitude": lon,
        "hourly": ["wind_speed_10m","wind_direction_10m","wind_gusts_10m",
                  "pressure_msl","temperature_2m","precipitation","weather_code"],
        "daily": ["sunrise","sunset"],
        "timezone": "auto", "start_date": start, "end_date": end
    })

# --------------- دوال مساعدة ---------------
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
    phase = (days_since_new % 29.53058867)/29.53058867
    idx = int(phase*8)%8
    phases = {0:"محاق",1:"هلال أول",2:"تربيع أول",3:"أحدب متزايد",4:"بدر",5:"أحدب متناقص",6:"تربيع ثاني",7:"هلال آخر"}
    name = phases.get(idx,"محاق")
    if phase<0.125 or phase>0.875: status, activity = "أيام متوسطة", "القاع ينشط ليلاً"
    elif phase<0.5: status, activity = "أيام حمل", "الأسماك نشيطة"
    else: status, activity = "أيام فساد", "الأسماك أقل نشاطاً"
    return {"name":name,"status":status,"activity":activity}

def moon_fishing_guidance(d:date):
    detail = moon_phase_detail(d)
    n = detail["name"]
    if "محاق" in n: return f"{detail['status']}. ركز على الصيد الليلي."
    if "هلال أول" in n or "تربيع أول" in n: return f"{detail['status']}. فرصة ممتازة."
    if "أحدب متزايد" in n: return f"{detail['status']}. البوري والدوراد نهاراً."
    if "بدر" in n: return f"{detail['status']}. الأسماك السطحية نهاراً."
    return f"{detail['status']}. الصيد مقبول."

async def fetch_timezone_info(lat, lon):
    try:
        data = await fetch_with_retry(MARINE_URL, {"latitude":lat,"longitude":lon,"hourly":"wave_height","timezone":"auto","forecast_days":1}, timeout=10)
        return data.get("timezone","UTC")
    except: return "UTC"

def resolve_target_date(txt, real_today):
    if txt == "today": return real_today
    if txt == "tomorrow": return real_today + timedelta(days=1)
    return real_today + timedelta(days=2)

# --------------- تقييم بقعة (للمسح) ---------------
def evaluate_spot(marine, weather, orient, sunrise_str, sunset_str, beach_type, target_species=None):
    mh = marine.get("hourly", {})
    wh = weather.get("hourly", {})
    times = mh.get("time", [])
    if not times: return 0.0, {}
    wave_h = [safe_float(x) for x in mh.get("wave_height",[])]
    wave_p = [safe_float(x) for x in mh.get("wave_period",[])]
    wind_speed = [safe_float(x) for x in wh.get("wind_speed_10m",[])]
    wind_dir_vals = [safe_float(x) for x in wh.get("wind_direction_10m",[])]
    wind_gust = [safe_float(x) for x in wh.get("wind_gusts_10m",[])]
    pressure = [safe_float(x) for x in wh.get("pressure_msl",[])]
    sst = [safe_float(x) for x in mh.get("sea_surface_temperature",[])]
    wind_classes = [wind_class_detailed(angle_diff(wd, orient)) for wd in wind_dir_vals]
    N = len(times)
    score = 0.0; red = 0; green = 0
    try:
        sr_h = int(sunrise_str.split(":")[0]); ss_h = int(sunset_str.split(":")[0])
    except: sr_h, ss_h = 6, 18
    for i in range(N):
        power = 0.49 * (wave_h[i]**2) * wave_p[i]
        if power>3 or wave_h[i]>1.8 or wind_gust[i]>50 or pressure[i]<1005:
            red+=1; score-=15
        elif 0.3<=wave_h[i]<=1 and 0.1<=power<=1.5 and wind_speed[i]<27.8:
            green+=1; score+=10
            if abs(datetime.fromisoformat(times[i]).hour-sr_h)<=2 or abs(datetime.fromisoformat(times[i]).hour-ss_h)<=2: score+=5
        else:
            if 0.2<=wave_h[i]<=1.2: score+=3
            elif wave_h[i]<0.2: score+=1
            else: score-=2
            if wind_speed[i]<15: score+=4
            elif wind_speed[i]<25: score+=2
            else: score-=1
    avg_wave = sum(wave_h)/N; avg_power = sum(0.49*(wave_h[i]**2)*wave_p[i] for i in range(N))/N
    avg_wind = sum(wind_speed)/N; avg_sst = sum(sst)/N; avg_press = sum(pressure)/N
    dominant = max(set(wind_classes), key=wind_classes.count)
    if 1015<=avg_press<=1025: score+=8
    factor = 1.0
    if target_species and target_species in SPECIES_PREFERENCES:
        pref = SPECIES_PREFERENCES[target_species]
        match = 0
        lo,hi = pref["ideal_sst"]
        if lo<=avg_sst<=hi: match+=30
        elif abs(avg_sst-lo)<=2 or abs(avg_sst-hi)<=2: match+=15
        if pref["bottom_type"]==beach_type: match+=20
        if dominant in pref["preferred_wind"]: match+=25
        elif any(w in pref["preferred_wind"] for w in wind_classes): match+=10
        if pref["ideal_wave_range"][0]<=avg_wave<=pref["ideal_wave_range"][1]: match+=15
        if pref["ideal_power_range"][0]<=avg_power<=pref["ideal_power_range"][1]: match+=10
        factor = 0.5 + match/100.0
    normalized = max(0, min(100, (score/200)*100*factor))
    summary = {"avg_wave":round(avg_wave,2),"avg_power":round(avg_power,2),"avg_wind":round(avg_wind,1),
               "avg_sst":round(avg_sst,1),"dominant_wind":dominant,"green_hours":green,"red_hours":red}
    return round(normalized,1), summary

def spot_reason(summary, beach_type, target_species):
    reasons = []
    wave, power, wind, sst = summary["avg_wave"], summary["avg_power"], summary["avg_wind"], summary["avg_sst"]
    dom = summary["dominant_wind"]
    if 0.3<=wave<=1: reasons.append("موج مثالي")
    elif wave<0.3: reasons.append("موج منخفض")
    else: reasons.append("موج مرتفع")
    if power<=1.5: reasons.append("طاقة مناسبة")
    else: reasons.append("طاقة عالية")
    if wind<15: reasons.append("رياح هادئة")
    elif wind<25: reasons.append("رياح متوسطة")
    else: reasons.append("رياح قوية")
    reasons.append(f"{dom}")
    reasons.append(f"ماء {sst}°")
    if summary["green_hours"]: reasons.append(f"{summary['green_hours']} ساعة خضراء")
    if summary["red_hours"]: reasons.append(f"{summary['red_hours']} ساعة حمراء")
    if target_species in SPECIES_PREFERENCES:
        pref = SPECIES_PREFERENCES[target_species]
        if pref["bottom_type"]==beach_type: reasons.append("قاع مناسب")
        if pref["ideal_sst"][0]<=sst<=pref["ideal_sst"][1]: reasons.append("حرارة مثالية")
        if dom in pref["preferred_wind"]: reasons.append("رياح مفضلة")
    return "؛ ".join(reasons)

# --------------- الشواطئ (نفس الأصل) ---------------
TUNISIAN_BEACHES = {  # نفس القائمة الكبيرة بدون تغيير
    "بنزرت": [
        {"name":"شاطئ الكورنيش (بنزرت)","lat":37.2744,"lon":9.8739,"orientation":45,"type":"sandy"},
        # ... اختصار لطول الرسالة، لكن في الكود الفعلي ستكون كاملة
    ],
    "نابل": [{"name":"شاطئ الحمامات","lat":36.4000,"lon":10.6167,"orientation":90,"type":"sandy"}],
    # ... إلخ، يجب وضع القائمة الكاملة كما في الأصل
}

@app.post("/scan-best")
@limiter.limit("5/minute")
async def scan_best_spots(request: Request, req: ScanRequest):
    beaches = []
    for gov in req.governorates:
        if gov in TUNISIAN_BEACHES:
            for b in TUNISIAN_BEACHES[gov]:
                beaches.append({**b, "governorate": gov})
    if not beaches:
        raise HTTPException(400, "لا توجد شواطئ")
    tz_name = "Africa/Tunis"
    now = datetime.now(zoneinfo.ZoneInfo(tz_name))
    target_dt = now.date() if req.target_date=="today" else (now.date()+timedelta(days=1) if req.target_date=="tomorrow" else now.date()+timedelta(days=2))
    start = target_dt - timedelta(days=2)
    end = target_dt + timedelta(days=1)
    sem = asyncio.Semaphore(5)
    async def process(b):
        async with sem:
            try:
                m, w = await asyncio.gather(
                    fetch_marine(b["lat"], b["lon"], start.isoformat(), end.isoformat()),
                    fetch_weather(b["lat"], b["lon"], start.isoformat(), end.isoformat())
                )
                sunrise = w.get("daily",{}).get("sunrise",["06:00"])[0]
                sunset = w.get("daily",{}).get("sunset",["18:00"])[0]
                score, summary = evaluate_spot(m, w, b["orientation"], sunrise, sunset, b["type"], req.target_species)
                reason = spot_reason(summary, b["type"], req.target_species)
                return {"name":b["name"],"governorate":b["governorate"],"score":score,"summary":summary,"type":b["type"],"reason":reason}
            except Exception as e:
                logger.error(f"فشل {b['name']}: {e}")
                return None
    tasks = [process(b) for b in beaches]
    results = await asyncio.gather(*tasks)
    valid = [r for r in results if r is not None]
    valid.sort(key=lambda x: x["score"], reverse=True)
    return {"target_date":target_dt.isoformat(),"top10":valid[:10]}

# --------------- تجميع التقرير المفصل (مُحسَّن) ---------------
def aggregate_physics(marine, weather, orient, target_date_obj, tz_name, sunrise_str, sunset_str):
    tz = zoneinfo.ZoneInfo(tz_name) if tz_name else zoneinfo.ZoneInfo("UTC")
    mh = marine.get("hourly",{}); wh = weather.get("hourly",{})
    times = mh.get("time",[])
    if not times: return {"past_avg_power":0,"dominant_wind":"غير معروف","blocks":[],"red_flags":[],"green_flags":[],"weed_risk":False,"bio":{},"avg_sst":0,"extra_info":{}}
    wave_h = [safe_float(x) for x in mh.get("wave_height",[])]
    wave_p = [safe_float(x) for x in mh.get("wave_period",[])]
    swell_h = [safe_float(x) for x in mh.get("swell_wave_height",[])]
    swell_p = [safe_float(x) for x in mh.get("swell_wave_period",[])]
    sst = [safe_float(x) for x in mh.get("sea_surface_temperature",[])]
    wind_speed = [safe_float(x) for x in wh.get("wind_speed_10m",[])]
    wind_dir = [safe_float(x) for x in wh.get("wind_direction_10m",[])]
    wind_gust = [safe_float(x) for x in wh.get("wind_gusts_10m",[])]
    pressure = [safe_float(x) for x in wh.get("pressure_msl",[])]
    temp_air = [safe_float(x) for x in wh.get("temperature_2m",[])]
    precip = [safe_float(x) for x in wh.get("precipitation",[])]
    weather_code = [int(safe_float(x)) for x in wh.get("weather_code",[])]

    wave_power = [0.49*(h**2)*p for h,p in zip(wave_h, wave_p)]
    wind_classes = [wind_class_detailed(angle_diff(wd, orient)) for wd in wind_dir]

    dtimes = [datetime.fromisoformat(t).replace(tzinfo=tz) if datetime.fromisoformat(t).tzinfo is None else datetime.fromisoformat(t) for t in times]
    target_start = datetime.combine(target_date_obj, datetime.min.time(), tzinfo=tz)
    target_end = target_start + timedelta(days=1)
    past_start = target_start - timedelta(hours=48)
    past_idx = [i for i,dt in enumerate(dtimes) if past_start<=dt<target_start]
    target_idx = [i for i,dt in enumerate(dtimes) if target_start<=dt<target_end]
    if not target_idx: return {"past_avg_power":0,"dominant_wind":"غير معروف","blocks":[],"red_flags":[],"green_flags":[],"weed_risk":False,"bio":{},"avg_sst":0,"extra_info":{}}

    past_avg_power = sum(wave_power[i] for i in past_idx)/max(len(past_idx),1) if past_idx else 0.0
    past_sh_avg = sum(swell_h[i] for i in past_idx)/max(len(past_idx),1) if past_idx else 0.0
    dominant = max(set(wind_classes), key=wind_classes.count) if wind_classes else "غير معروف"

    # خطر الأعشاب (نفس منطقك)
    weed = False
    if target_idx and wind_classes:
        if wind_classes[target_idx[0]].startswith("بحرية") and (past_sh_avg>0.8 or past_avg_power>5.0):
            weed = True

    # بناء كتل زمنية مفصلة
    periods = defaultdict(list)
    for idx,i in enumerate(target_idx):
        h = dtimes[i].hour
        if 4<=h<=11: periods["morning"].append(idx)
        elif 12<=h<=17: periods["afternoon"].append(idx)
        else: periods["night"].append(idx)

    blocks = []
    for key in ["morning","afternoon","night"]:
        idxs = periods[key]
        if not idxs: continue
        avg_h = sum(wave_h[i] for i in idxs)/len(idxs)
        min_h,max_h = min(wave_h[i] for i in idxs), max(wave_h[i] for i in idxs)
        avg_pow = sum(wave_power[i] for i in idxs)/len(idxs)
        avg_w = sum(wind_speed[i] for i in idxs)/len(idxs)
        min_w,max_w = min(wind_speed[i] for i in idxs), max(wind_speed[i] for i in idxs)
        wc_dom = max(set(wind_classes[i] for i in idxs), key=wind_classes.count)
        avg_swh = sum(swell_h[i] for i in idxs)/len(idxs)
        avg_swp = sum(swell_p[i] for i in idxs)/len(idxs)
        avg_air = sum(temp_air[i] for i in idxs)/len(idxs) if temp_air else 0
        total_precip = sum(precip[i] for i in idxs)
        most_code = max(set(weather_code[i] for i in idxs), key=weather_code[i].count) if idxs else 0

        # تحليل swell
        swell_dom = "مختلط"
        if avg_swh > 0.7*avg_h: swell_dom = "طاقة قادمة من بعيد (swell)"
        elif avg_h - avg_swh > 0.2: swell_dom = "موج محلي (wind sea)"

        wind_start = wind_classes[idxs[0]]
        wind_end = wind_classes[idxs[-1]]
        wind_trend = f"تتحول من {wind_start} إلى {wind_end}" if wind_start!=wind_end else f"ثابتة {wind_start}"
        sea_state = "هادئ" if avg_h<0.3 else "متوسط الهيجان" if avg_h<0.8 else "هائج"

        blocks.append({
            "name":{"morning":"الصباح","afternoon":"الظهر","night":"الليل"}[key],
            "time_range":f"{dtimes[target_idx[idxs[0]]].strftime('%H:%M')}-{dtimes[target_idx[idxs[-1]]].strftime('%H:%M')}",
            "sea_state":sea_state,
            "wave_height":f"{min_h:.2f}-{max_h:.2f}",
            "wave_power":round(avg_pow,2),
            "swell_height":f"{min(swell_h[i] for i in idxs):.2f}-{max(swell_h[i] for i in idxs):.2f}",
            "swell_period":round(avg_swp,1),
            "swell_dominance":swell_dom,
            "wind_speed":f"{min_w:.1f}-{max_w:.1f}",
            "wind_gust_peak":round(max(wind_gust[i] for i in idxs),1) if wind_gust else 0,
            "wind_dir":wc_dom,
            "wind_trend":wind_trend,
            "air_temp":round(avg_air,1),
            "precip":round(total_precip,1),
            "weather":weather_desc(most_code)
        })

    reds, greens = [], []
    for i in target_idx:
        hh = dtimes[i].strftime("%H:%M")
        if wave_power[i]>3 or wave_h[i]>1.8 or wind_gust[i]>50 or pressure[i]<1005: reds.append(hh)
        if 0.3<=wave_h[i]<=1 and 0.1<=wave_power[i]<=1.5 and wind_speed[i]<27.8: greens.append(hh)

    avg_sst = sum(sst[i] for i in target_idx)/len(target_idx)
    avg_press = sum(pressure[i] for i in target_idx)/len(target_idx)
    peak_gust = max(wind_gust[i] for i in target_idx) if target_idx else 0
    bio = {}
    if avg_sst<16: bio["high"]=["قاروص","سارغ"]
    elif avg_sst>19: bio["high"]=["دوراد","ماربري"]
    moon = moon_phase_detail(target_date_obj)
    moon_g = moon_fishing_guidance(target_date_obj)
    extra = {
        "pressure_avg":round(avg_press,1),
        "sunrise":sunrise_str,"sunset":sunset_str,
        "moon_phase":moon["name"],"moon_status":moon["status"],"moon_activity":moon["activity"],"moon_guidance":moon_g,
        "peak_gust_today":round(peak_gust,1)
    }
    return {"past_avg_power":round(past_avg_power,2),"dominant_wind":dominant,"blocks":blocks,"red_flags":reds[:5],"green_flags":greens[:5],"weed_risk":weed,"bio":bio,"avg_sst":round(avg_sst,1),"extra_info":extra}

def build_context(req, agg, tz_name):
    beach = "رملي" if req.beach_type=="sandy" else "صخري"
    moon = agg["extra_info"]
    lines = [
        f"الموقع: شاطئ {beach} اتجاهه {req.beach_orientation}° شمال.",
        f"التاريخ: {req.target_date} (توقيت {tz_name})",
        f"حرارة الماء: {agg['avg_sst']}°م",
        f"القمر: {moon['moon_status']} ({moon['moon_phase']}). {moon['moon_guidance']}",
        f"الرياح السائدة: {agg['dominant_wind']}، هبات تصل {moon['peak_gust_today']} كم/س",
        f"خطر الأعشاب: {'نعم' if agg['weed_risk'] else 'منخفض'}",
        f"متوسط طاقة الموج 48س: {agg['past_avg_power']} kW/m"
    ]
    if moon['peak_gust_today']>30: lines.append("تحذير: هبات رياح قوية!")
    lines.append("\nتفاصيل الفترات:")
    for b in agg["blocks"]:
        lines.append(f"\n【{b['name']} ({b['time_range']})】")
        lines.append(f"حالة البحر: {b['sea_state']}. الموج {b['wave_height']}م, swell {b['swell_height']}م/{b['swell_period']}ث, طاقة {b['wave_power']}kW/m")
        lines.append(f"الرياح: {b['wind_speed']} كم/س, {b['wind_dir']}. {b['wind_trend']}. هبات {b['wind_gust_peak']} كم/س")
        lines.append(f"حرارة الهواء: {b['air_temp']}°م, {b['weather']}, أمطار {b['precip']}مم")
    if agg["red_flags"]: lines.append(f"\nساعات الخطر: {', '.join(agg['red_flags'])}")
    if agg["green_flags"]: lines.append(f"أفضل الساعات: {', '.join(agg['green_flags'])}")
    if agg["bio"].get("high"): lines.append(f"\nالأسماك المتوقعة: {', '.join(agg['bio']['high'])}")
    lines.append("\nقدم تحليلك الاحترافي وتوصياتك النهائية (الرصاصة، التركيبة، الطعم، السلامة).")
    return "\n".join(lines)

SYSTEM_PROMPT = """أنت صياد سرفكاستينغ تونسي محترف. اكتب تقريراً بالدارجة التونسية، نص واحد متصل، يشمل تحليل البحر والموج والرياح والأعشاب والقمر، مع توصيات الرصاصة والتركيبة والطعم وخطة الطوارئ والسلامة. كن واقعياً ولا تبالغ في التفاؤل، وإذا الظروف سيئة قل ذلك."""

async def call_openrouter(ctx):
    headers = {"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json"}
    payload = {"model":MODEL_NAME,"messages":[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":ctx}],"max_tokens":7000,"temperature":0.3}
    async with httpx.AsyncClient() as client:
        resp = await client.post(OPENROUTER_URL, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"]
        raise Exception("OpenRouter فارغ")

@app.post("/generate-report")
@limiter.limit("10/minute")
async def generate_report(request: Request, req: ReportRequest):
    cache_key = f"{req.latitude:.4f}_{req.longitude:.4f}_{req.beach_orientation}_{req.beach_type}_{req.target_date}"
    async with cache_lock:
        if cache_key in cache and time.time() - cache[cache_key]["ts"] < CACHE_TTL:
            return cache[cache_key]["data"]
    try:
        tz_name = await fetch_timezone_info(req.latitude, req.longitude)
        now_tn = datetime.now(zoneinfo.ZoneInfo("Africa/Tunis"))
        target_dt = resolve_target_date(req.target_date, now_tn.date())
        start = target_dt - timedelta(days=2)
        end = target_dt + timedelta(days=1)
        marine, weather = await asyncio.gather(
            fetch_marine(req.latitude, req.longitude, start.isoformat(), end.isoformat()),
            fetch_weather(req.latitude, req.longitude, start.isoformat(), end.isoformat())
        )
        sunrise = weather.get("daily",{}).get("sunrise",["06:00"])[0]
        sunset = weather.get("daily",{}).get("sunset",["18:00"])[0]
        agg = aggregate_physics(marine, weather, req.beach_orientation, target_dt, tz_name, sunrise, sunset)
        ctx = build_context(req, agg, tz_name)
        report = await call_openrouter(ctx)
        result = {"report":report, "meta":{"timezone":tz_name,"target_date":target_dt.isoformat()}}
        async with cache_lock:
            cache[cache_key] = {"ts":time.time(), "data":result}
        return result
    except Exception as e:
        logger.error(f"generate-report failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, detail="فشل إنشاء التقرير")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
