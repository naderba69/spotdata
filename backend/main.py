"""
Surfcasting Analytics API – Final v2.3
Single Gemini call, max_tokens=8000, optimized concise prompt.
Avoids Render 60s timeout. Full report without truncation.
"""
import os, math, asyncio, logging, traceback, zoneinfo, re
from datetime import datetime, timedelta, date
from typing import List, Dict

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("surfcasting")

app = FastAPI(title="Surfcasting Analytics", version="2.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise RuntimeError("GEMINI_API_KEY مفقود")
genai.configure(api_key=GEMINI_KEY)

MODEL_NAME = "models/gemini-2.5-flash"

MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# --------------- النماذج ---------------
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

class SurfError(Exception): pass

@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})

@app.get("/health")
def health():
    return {"status": "ok", "gemini": bool(GEMINI_KEY), "model": MODEL_NAME}

# --------------- الدوال المساعدة (المنطقة الزمنية) ---------------
async def fetch_timezone_info(lat, lon):
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(MARINE_URL, params={
                "latitude": lat, "longitude": lon,
                "hourly": "wave_height", "timezone": "auto", "forecast_days": 1
            }, timeout=10)
            r.raise_for_status()
            data = r.json()
            tz_name = data.get("timezone", "UTC")
            zoneinfo.ZoneInfo(tz_name)
            return tz_name
    except:
        return "UTC"

def extract_real_date_from_times(times: list, tz_name: str) -> date:
    if not times:
        tz = zoneinfo.ZoneInfo(tz_name)
        return datetime.now(tz).date()
    try:
        dt = datetime.fromisoformat(times[0])
        if dt.tzinfo is None:
            tz = zoneinfo.ZoneInfo(tz_name)
            dt = dt.replace(tzinfo=tz)
        return dt.date()
    except:
        tz = zoneinfo.ZoneInfo(tz_name)
        return datetime.now(tz).date()

def resolve_target_date_from_real(txt: str, real_today: date) -> date:
    if txt == "today": return real_today
    elif txt == "tomorrow": return real_today + timedelta(days=1)
    return real_today + timedelta(days=2)

# --------------- جلب البيانات ---------------
async def fetch_marine(lat, lon, start, end):
    async with httpx.AsyncClient() as c:
        r = await c.get(MARINE_URL, params={
            "latitude": lat, "longitude": lon,
            "hourly": ["wave_height","wave_period","wave_direction",
                      "swell_wave_height","swell_wave_period","swell_wave_direction",
                      "sea_surface_temperature"],
            "timezone": "auto", "start_date": start, "end_date": end
        }, timeout=20)
        r.raise_for_status()
        return r.json()

async def fetch_weather(lat, lon, start, end):
    async with httpx.AsyncClient() as c:
        r = await c.get(WEATHER_URL, params={
            "latitude": lat, "longitude": lon,
            "hourly": ["wind_speed_10m","wind_direction_10m","wind_gusts_10m",
                      "pressure_msl","temperature_2m","precipitation","weather_code"],
            "daily": ["sunrise","sunset"],
            "timezone": "auto", "start_date": start, "end_date": end
        }, timeout=20)
        r.raise_for_status()
        return r.json()

# --------------- الاتجاه التلقائي ---------------
async def get_auto_orientation(lat: float, lon: float) -> int:
    query = f"""
    [out:json];
    (
      way(around:5000,{lat},{lon})["natural"="coastline"];
      relation(around:5000,{lat},{lon})["natural"="coastline"];
    );
    out geom;
    """
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(OVERPASS_URL, params={"data": query}, timeout=20.0)
            resp.raise_for_status()
            data = resp.json()
            elements = data.get("elements", [])
            if not elements: return 0
            for el in elements:
                geom = None
                if el["type"] == "way":
                    geom = el.get("geometry", [])
                elif el["type"] == "relation":
                    for member in el.get("members", []):
                        if member.get("type") == "way" and member.get("geometry"):
                            geom = member["geometry"]
                            break
                if geom and len(geom) >= 2:
                    p1, p2 = geom[0], geom[-1]
                    dx = p2["lon"] - p1["lon"]
                    dy = p2["lat"] - p1["lat"]
                    angle_rad = math.atan2(dx, dy)
                    angle_deg = (math.degrees(angle_rad) + 360) % 360
                    return int(round((angle_deg + 90) % 360))
            return 0
    except Exception as e:
        logger.error(f"Orientation error: {e}")
        return 0

@app.post("/auto-orientation")
async def auto_orientation(req: AutoOrientationRequest):
    angle = await get_auto_orientation(req.latitude, req.longitude)
    return {"orientation": angle}

# --------------- مرحلة القمر ---------------
def moon_phase(d: date) -> str:
    y, m, day = d.year, d.month, d.day
    if m < 3:
        y -= 1
        m += 12
    a = int(y / 100)
    b = 2 - a + int(a / 4)
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + day + b - 1524.5
    new_moon_jd = 2451550.1
    days_since_new = jd - new_moon_jd
    lunar_month = 29.53058867
    phase = (days_since_new % lunar_month) / lunar_month
    idx = int(phase * 8) % 8
    phases = {
        0: "محاق", 1: "هلال أول", 2: "تربيع أول", 3: "أحدب متزايد",
        4: "بدر", 5: "أحدب متناقص", 6: "تربيع ثاني", 7: "هلال آخر"
    }
    return phases.get(idx, "محاق")

# --------------- دوال فيزيائية مساعدة ---------------
def safe_float(v):
    if v is None: return 0.0
    try:
        f = float(v)
        return 0.0 if math.isnan(f) else f
    except: return 0.0

def angle_diff(w, b):
    d = abs(w - b) % 360
    return 360 - d if d > 180 else d

def wind_class_detailed(diff):
    if diff < 30: return "بحرية مباشرة"
    elif diff < 45: return "بحرية خفيفة"
    elif diff < 60: return "جانبية مائلة للبحر"
    elif diff <= 120: return "جانبية"
    elif diff < 150: return "جانبية مائلة للبر"
    elif diff < 165: return "برية خفيفة"
    else: return "برية مباشرة"

def weather_desc(code):
    if code == 0: return "صافية"
    elif code == 1: return "مشمس غالباً"
    elif code == 2: return "غائم جزئي"
    elif code == 3: return "غائم"
    elif 45 <= code <= 48: return "ضباب"
    elif 51 <= code <= 55: return "رذاذ"
    elif 61 <= code <= 65: return "مطر"
    elif 80 <= code <= 82: return "زخات مطر"
    elif 95 <= code <= 99: return "عواصف"
    return "غير معروف"

def aggregate_physics(marine, weather, beach_orient, beach_type, target_date_obj, tz_name):
    tz = zoneinfo.ZoneInfo(tz_name) if tz_name else zoneinfo.ZoneInfo("UTC")
    mh = marine.get("hourly", {})
    wh = weather.get("hourly", {})
    dw = weather.get("daily", {})

    times = mh.get("time", [])
    wave_h = [safe_float(x) for x in mh.get("wave_height", [])]
    wave_p = [safe_float(x) for x in mh.get("wave_period", [])]
    swell_h = [safe_float(x) for x in mh.get("swell_wave_height", [])]
    swell_p = [safe_float(x) for x in mh.get("swell_wave_period", [])]
    sst = [safe_float(x) for x in mh.get("sea_surface_temperature", [])]

    wind_speed = [safe_float(x) for x in wh.get("wind_speed_10m", [])]
    wind_dir = [safe_float(x) for x in wh.get("wind_direction_10m", [])]
    wind_gust = [safe_float(x) for x in wh.get("wind_gusts_10m", [])]
    pressure = [safe_float(x) for x in wh.get("pressure_msl", [])]
    temp_air = [safe_float(x) for x in wh.get("temperature_2m", [])]
    precip = [safe_float(x) for x in wh.get("precipitation", [])]
    weather_code = [int(safe_float(x)) for x in wh.get("weather_code", [])]

    wind_kph = wind_speed
    gust_kph = wind_gust
    wave_power = [0.49 * (h**2) * p for h, p in zip(wave_h, wave_p)]

    wind_classes_detailed = []
    for wd in wind_dir:
        d = angle_diff(wd, beach_orient)
        wind_classes_detailed.append(wind_class_detailed(d))

    dtimes = []
    for t_str in times:
        dt = datetime.fromisoformat(t_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        dtimes.append(dt)

    target_start = datetime.combine(target_date_obj, datetime.min.time(), tzinfo=tz)
    target_end = target_start + timedelta(days=1)
    past_start = target_start - timedelta(hours=48)

    past_idx = [i for i, dt in enumerate(dtimes) if past_start <= dt < target_start]
    target_idx = [i for i, dt in enumerate(dtimes) if target_start <= dt < target_end]

    if not target_idx:
        return {"past_avg_power":0, "dominant_wind":"غير معروف", "sustained_hrs":0,
                "blocks":[], "red_flags":[], "green_flags":[], "weed_risk":False,
                "bio":{}, "avg_sst":0, "extra_info":{}}

    past_avg_power_val = sum(wave_power[i] for i in past_idx) / max(len(past_idx),1)
    dominant = max(set(wind_classes_detailed), key=wind_classes_detailed.count)
    sustained_hrs = sum(1 for i in past_idx if wind_kph[i] > 18.5)

    press_target = [pressure[i] for i in target_idx]
    avg_press = sum(press_target) / len(press_target)
    press_trend = "مستقر"
    if len(press_target) >= 6:
        half = len(press_target)//2
        first = sum(press_target[:half])/half
        second = sum(press_target[half:])/(len(press_target)-half)
        diff = second - first
        if diff > 1: press_trend = "في ارتفاع"
        elif diff < -1: press_trend = "في انخفاض"

    periods = {
        "morning": {"indices": [], "label": "الصباح", "temps": [], "precip": [], "codes": []},
        "afternoon": {"indices": [], "label": "الظهر", "temps": [], "precip": [], "codes": []},
        "night": {"indices": [], "label": "الليل", "temps": [], "precip": [], "codes": []}
    }
    for i in target_idx:
        h = dtimes[i].hour
        if 4 <= h <= 11: key = "morning"
        elif 12 <= h <= 17: key = "afternoon"
        else: key = "night"
        periods[key]["indices"].append(i)
        periods[key]["temps"].append(temp_air[i])
        periods[key]["precip"].append(precip[i])
        periods[key]["codes"].append(weather_code[i])

    block_list = []
    for key, pd in periods.items():
        idxs = pd["indices"]
        if not idxs: continue
        avg_h = sum(wave_h[i] for i in idxs)/len(idxs)
        min_h = min(wave_h[i] for i in idxs)
        max_h = max(wave_h[i] for i in idxs)
        avg_p = sum(wave_power[i] for i in idxs)/len(idxs)
        avg_w = sum(wind_kph[i] for i in idxs)/len(idxs)
        min_w = min(wind_kph[i] for i in idxs)
        max_w = max(wind_kph[i] for i in idxs)
        wc = max(set(wind_classes_detailed[i] for i in idxs), key=wind_classes_detailed.count)
        avg_sw_h = sum(swell_h[i] for i in idxs)/len(idxs)
        avg_sw_p = sum(swell_p[i] for i in idxs)/len(idxs)
        avg_air = sum(pd["temps"])/len(pd["temps"]) if pd["temps"] else None
        total_precip = sum(pd["precip"]) if pd["precip"] else 0
        most_common_code = max(set(pd["codes"]), key=pd["codes"].count) if pd["codes"] else 0
        block_list.append({
            "name": pd["label"],
            "wave_h_range": f"{min_h:.2f}-{max_h:.2f}",
            "avg_wave_h": round(avg_h,2),
            "power": round(avg_p,2),
            "wind_kph_range": f"{min_w:.1f}-{max_w:.1f}",
            "avg_wind_kph": round(avg_w,1),
            "wind_dir": wc,
            "swell_h": round(avg_sw_h,2),
            "swell_period": round(avg_sw_p * 2) / 2,
            "air_temp": round(avg_air,1) if avg_air is not None else "غير متوفر",
            "precip_mm": round(total_precip,1),
            "weather": weather_desc(most_common_code)
        })

    sunrise = dw.get("sunrise", ["غير معروف"])[0] if dw.get("sunrise") else "غير معروف"
    sunset = dw.get("sunset", ["غير معروف"])[0] if dw.get("sunset") else "غير معروف"

    reds, greens = [], []
    for i in target_idx:
        hh = dtimes[i].strftime("%H:%M")
        if (wave_power[i] > 3.0 or wave_h[i] > 1.8 or gust_kph[i] > 50 or pressure[i] < 1005):
            reds.append(hh)
        if (0.3 <= wave_h[i] <= 1.0 and 0.1 <= wave_power[i] <= 1.5 and wind_kph[i] < 27.8):
            greens.append(hh)

    past_wp_avg = sum(wave_p[i] for i in past_idx) / max(len(past_idx),1)
    past_sh_avg = sum(swell_h[i] for i in past_idx) / max(len(past_idx),1)
    weed = (past_wp_avg >= 8.0 and past_sh_avg > 1.0 and wind_classes_detailed[target_idx[0]].startswith("بحرية"))

    avg_sst = sum(sst[i] for i in target_idx) / max(len(target_idx),1)
    bio = {}
    if avg_sst < 16: bio["high"] = ["قاروص", "سارغ كبير"]
    elif avg_sst > 19: bio["high"] = ["دوراد", "ماربري"]
    else: bio["high"] = []
    if beach_type == "rocky": bio.setdefault("additional", []).append("سارغ")
    if beach_type == "sandy": bio.setdefault("additional", []).append("بوري")

    return {
        "past_avg_power": round(past_avg_power_val,2),
        "dominant_wind": dominant,
        "sustained_hrs": sustained_hrs,
        "blocks": block_list,
        "red_flags": reds[:5],
        "green_flags": greens[:5],
        "weed_risk": weed,
        "bio": bio,
        "avg_sst": round(avg_sst,1),
        "extra_info": {
            "pressure_avg": round(avg_press,1),
            "pressure_trend": press_trend,
            "sunrise": sunrise,
            "sunset": sunset,
            "moon_phase": moon_phase(target_date_obj)
        }
    }

def build_context(req, agg, tz_name):
    beach = "رملي" if req.beach_type == "sandy" else "صخري"
    lines = [
        f"الموقع: {req.latitude:.2f}, {req.longitude:.2f}، شاطئ {beach}، اتجاه البحر {req.beach_orientation}°",
        f"التاريخ: {req.target_date} (توقيت {tz_name})",
        f"حرارة الماء: {agg['avg_sst']}°م",
        f"متوسط طاقة الموج (48 ساعة): {agg['past_avg_power']} kW/m",
        f"الرياح السائدة: {agg['dominant_wind']}",
        f"ساعات الرياح > 18.5 كم/س: {agg['sustained_hrs']} ساعة",
        f"خطر الأعشاب: {'نعم' if agg['weed_risk'] else 'لا'}"
    ]
    extra = agg.get("extra_info", {})
    if extra:
        lines.append(f"الضغط الجوي: {extra['pressure_avg']} hPa ({extra['pressure_trend']})")
        lines.append(f"الشروق: {extra['sunrise']} | الغروب: {extra['sunset']}")
        lines.append(f"القمر: {extra['moon_phase']}")

    if agg["blocks"]:
        lines.append("تفاصيل الفترات:")
        for b in agg["blocks"]:
            lines.append(
                f"- {b['name']}: {b['weather']}, الموج {b['wave_h_range']}م, "
                f"swell {b['swell_h']}م/{b['swell_period']}ث, طاقة {b['power']} kW/m, "
                f"الرياح {b['wind_kph_range']} كم/س ({b['wind_dir']}), "
                f"حرارة الهواء ~{b['air_temp']}°م, أمطار {b['precip_mm']}مم"
            )
    if agg["red_flags"]:
        lines.append("ساعات الخطر: " + "، ".join(agg["red_flags"]))
    if agg["green_flags"]:
        lines.append("ساعات مثالية: " + "، ".join(agg["green_flags"]))
    if agg["bio"].get("high") or agg["bio"].get("additional"):
        lines.append("الأسماك المتوقعة: " + ", ".join(agg["bio"].get("high", []) + agg["bio"].get("additional", [])))
    return "\n".join(lines)

# --------------- SYSTEM PROMPT (مركز، مختصر، يضمن اكتمال التقرير في طلب واحد) ---------------
SYSTEM_PROMPT = """أنت صياد سرفكاستينغ تونسي خبير. اكتب تقريراً بحرياً كاملاً بالعربية والدارجة التونسية (المرصاص، اللدونة، القفلة، دود الكف، إلخ) بناءً على البيانات التالية. التقرير يجب أن يكون نصاً واحداً متصلاً بدون رموز أو نقاط، ويغطي هذه النقاط باختصار دقيق:
1. تحليل عام: حالة السماء، حرارة الهواء والماء، القمر، الشروق/الغروب.
2. الأمواج والتيارات: نطاق الموج، swell، الطاقة، ثبات الرصاصة، وزن ونوع الرصاصة المناسبة.
3. الرياح والأعشاب: سرعة واتجاه الرياح، تأثيرها على الأعشاب ونقاء الماء، الأمطار.
4. تقييم الصيد: أفضل الأوقات، تكتيك الرمي، الطعوم، الأسماك المتوقعة. نبه لمراجعة جدول المد المحلي.

اكتب بلغة خبير ميداني موجز ومفيد. لا تذكر أنك تلقيت بيانات."""

# --------------- استدعاء Gemini (طلب واحد مع إعادة المحاولة عند 429) ---------------
async def call_gemini(ctx):
    model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_PROMPT)
    gen_cfg = {"temperature": 0.3, "max_output_tokens": 8000}

    async def _attempt():
        resp = await asyncio.to_thread(model.generate_content, contents=ctx, generation_config=gen_cfg)
        if resp.candidates and resp.candidates[0].content.parts:
            return resp.candidates[0].content.parts[0].text.strip()
        raise SurfError("رد فارغ")

    try:
        return await _attempt()
    except Exception as e:
        err_str = str(e)
        if "429" in err_str and "retry" in err_str.lower():
            delay = 45
            match = re.search(r'retry in (\d+\.?\d*)s', err_str)
            if match: delay = float(match.group(1))
            logger.warning(f"Quota exceeded, waiting {delay}s")
            await asyncio.sleep(delay + 2)
            return await _attempt()
        raise SurfError(f"فشل Gemini: {err_str}")

# --------------- نقطة نهاية التقرير الفردي ---------------
@app.post("/generate-report")
async def generate_report(req: ReportRequest):
    try:
        tz_name = await fetch_timezone_info(req.latitude, req.longitude)
        target_dt = date.today()
        start_dt = target_dt - timedelta(days=2)
        end_dt = target_dt + timedelta(days=1)

        marine, weather = await asyncio.gather(
            fetch_marine(req.latitude, req.longitude, start_dt.isoformat(), end_dt.isoformat()),
            fetch_weather(req.latitude, req.longitude, start_dt.isoformat(), end_dt.isoformat())
        )
        real_today = extract_real_date_from_times(marine.get("hourly", {}).get("time", []), tz_name)
        target_dt = resolve_target_date_from_real(req.target_date, real_today)

        agg = aggregate_physics(marine, weather, req.beach_orientation, req.beach_type, target_dt, tz_name)
        ctx = build_context(req, agg, tz_name)
        report = await call_gemini(ctx)
        return {"report": report, "meta": {"timezone": tz_name, "target_date": target_dt.isoformat()}}
    except Exception as e:
        logger.exception("خطأ")
        raise HTTPException(status_code=500, detail=str(e))

# --------------- قاعدة بيانات الشواطئ (للمسح) ---------------
TUNISIAN_BEACHES: Dict[str, List[Dict]] = {
    "بنزرت": [
        {"name": "شاطئ الكورنيش", "lat": 37.2744, "lon": 9.8739, "orientation": 45, "type": "sandy"},
        {"name": "شاطئ رفراف", "lat": 37.2167, "lon": 10.0833, "orientation": 0, "type": "sandy"},
    ],
    "نابل": [
        {"name": "شاطئ الحمامات", "lat": 36.4000, "lon": 10.6167, "orientation": 90, "type": "sandy"},
        {"name": "شاطئ قليبية", "lat": 36.8500, "lon": 11.1000, "orientation": 45, "type": "sandy"},
    ],
    # ... أكمل باقي الولايات كما في النسخة السابقة
}

def evaluate_spot(marine, weather, beach_orient):
    # نفس دالة التقييم السريع
    pass  # اختصار للطبع، استخدم الكود السابق

@app.post("/scan-best")
async def scan_best_spots(req: ScanRequest):
    # نفس الكود السابق
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
