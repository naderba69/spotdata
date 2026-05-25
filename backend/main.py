"""
Surfcasting Analytics API - Final Production v10.0
Enhanced with Pressure, Rain, Weather, Sunrise/Sunset, Moon Phase, Tides Reminder
Tunisian Arabic Prompt – 0% Physics Error
"""
import os, math, asyncio, logging, traceback, zoneinfo
from datetime import datetime, timedelta, date

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("surfcasting")

app = FastAPI(title="Surfcasting Analytics", version="10.0.0")
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

class SurfError(Exception): pass

@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})

@app.get("/health")
def health():
    return {"status": "ok", "gemini": bool(GEMINI_KEY), "model": MODEL_NAME}

# --------------- دوال المنطقة الزمنية ---------------
async def fetch_timezone(lat, lon):
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(MARINE_URL, params={
                "latitude": lat, "longitude": lon,
                "hourly": "wave_height", "timezone": "auto", "forecast_days": 1
            }, timeout=10)
            r.raise_for_status()
            tz = r.json().get("timezone", "UTC")
            zoneinfo.ZoneInfo(tz)
            return tz
    except:
        return "UTC"

def target_date_from_str(txt, tz_name):
    try:
        tz = zoneinfo.ZoneInfo(tz_name)
    except:
        tz = zoneinfo.ZoneInfo("UTC")
    now = datetime.now(tz)
    if txt == "today": return now.date()
    elif txt == "tomorrow": return now.date() + timedelta(days=1)
    return now.date() + timedelta(days=2)

# --------------- جلب البيانات البحرية والطقس ---------------
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
        # نجلب بيانات ساعية ويومية معاً
        r = await c.get(WEATHER_URL, params={
            "latitude": lat, "longitude": lon,
            "hourly": ["wind_speed_10m","wind_direction_10m","wind_gusts_10m",
                      "pressure_msl","temperature_2m","precipitation","weather_code"],
            "daily": ["sunrise","sunset","precipitation_sum"],
            "timezone": "auto",
            "start_date": start,
            "end_date": end
        }, timeout=20)
        r.raise_for_status()
        return r.json()

# --------------- الاتجاه التلقائي للشاطئ ---------------
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

# --------------- دالة مرحلة القمر ---------------
def moon_phase(d: date) -> str:
    """حساب تقريبي لمرحلة القمر (0-7) وإرجاع اسم المرحلة بالعربية"""
    y, m, day = d.year, d.month, d.day
    if m < 3:
        y -= 1
        m += 12
    a = int(y / 100)
    b = 2 - a + int(a / 4)
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + day + b - 1524.5
    # عدد الأيام منذ القمر الجديد في 6 يناير 2000
    new_moon_jd = 2451550.1
    days_since_new = jd - new_moon_jd
    # مدة الشهر القمري 29.53 يوم
    lunar_month = 29.53058867
    phase = (days_since_new % lunar_month) / lunar_month
    idx = int(phase * 8) % 8
    phases = {
        0: "محاق (قمر جديد)",
        1: "هلال أول",
        2: "تربيع أول",
        3: "أحدب متزايد",
        4: "بدر (قمر كامل)",
        5: "أحدب متناقص",
        6: "تربيع ثاني",
        7: "هلال آخر"
    }
    return phases.get(idx, "محاق")

# --------------- المحرك الفيزيائي الموسع ---------------
def safe_float(v):
    if v is None: return 0.0
    try:
        f = float(v)
        return 0.0 if math.isnan(f) else f
    except: return 0.0

def angle_diff(w, b):
    d = abs(w - b) % 360
    return 360 - d if d > 180 else d

def wind_class(d):
    if d < 45: return "Onshore"
    if d > 135: return "Offshore"
    return "Sideshore"

def weather_desc(code: int) -> str:
    """تحويل كود الطقس إلى وصف عربي مبسط"""
    if code <= 1: return "مشمس"
    elif code == 2: return "غائم جزئياً"
    elif code == 3: return "غائم"
    elif 45 <= code <= 48: return "ضباب"
    elif 51 <= code <= 55: return "رذاذ"
    elif 61 <= code <= 65: return "مطر"
    elif 71 <= code <= 77: return "ثلج"
    elif 80 <= code <= 82: return "زخات مطر"
    elif 95 <= code <= 99: return "عواصف رعدية"
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

    wind_classes = []
    for wd in wind_dir:
        d = angle_diff(wd, beach_orient)
        wind_classes.append(wind_class(d))

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
                "bio":{}, "avg_sst":0, "air_temps":{}, "extra_info":{}}

    past_avg_power = sum(wave_power[i] for i in past_idx) / max(len(past_idx),1)
    past_wclass = [wind_classes[i] for i in past_idx]
    dominant = max(set(past_wclass), key=past_wclass.count) if past_wclass else "Offshore"
    sustained_hrs = sum(1 for i in past_idx if wind_kph[i] > 18.5)

    # ضغط جوي: قيمة واتجاه
    press_target = [pressure[i] for i in target_idx]
    avg_press = sum(press_target) / len(press_target)
    press_trend = "مستقر"
    if len(press_target) >= 6:
        first_half = sum(press_target[:len(press_target)//2]) / (len(press_target)//2)
        second_half = sum(press_target[len(press_target)//2:]) / (len(press_target) - len(press_target)//2)
        diff = second_half - first_half
        if diff > 1: press_trend = "في ارتفاع"
        elif diff < -1: press_trend = "في انخفاض"

    # تجميع الفترات
    periods = {
        "morning": {"indices": [], "label": "الصباح (04-11)", "temps": [], "precip": [], "codes": []},
        "afternoon": {"indices": [], "label": "الظهر (12-17)", "temps": [], "precip": [], "codes": []},
        "night": {"indices": [], "label": "الليل (18-03)", "temps": [], "precip": [], "codes": []}
    }
    for i in target_idx:
        h = dtimes[i].hour
        if 4 <= h <= 11:
            key = "morning"
        elif 12 <= h <= 17:
            key = "afternoon"
        else:
            key = "night"
        periods[key]["indices"].append(i)
        periods[key]["temps"].append(temp_air[i])
        periods[key]["precip"].append(precip[i])
        periods[key]["codes"].append(weather_code[i])

    block_list = []
    for key, pd in periods.items():
        idxs = pd["indices"]
        if not idxs: continue
        avg_h = sum(wave_h[i] for i in idxs) / len(idxs)
        avg_p = sum(wave_power[i] for i in idxs) / len(idxs)
        avg_w = sum(wind_kph[i] for i in idxs) / len(idxs)
        wc = max(set(wind_classes[i] for i in idxs), key=wind_classes.count)
        avg_sw_h = sum(swell_h[i] for i in idxs) / len(idxs)
        avg_sw_p = sum(swell_p[i] for i in idxs) / len(idxs)
        avg_air = sum(pd["temps"])/len(pd["temps"]) if pd["temps"] else None
        total_precip = sum(pd["precip"]) if pd["precip"] else 0
        weather_codes = pd["codes"] if pd["codes"] else [0]
        most_common_code = max(set(weather_codes), key=weather_codes.count)
        block_list.append({
            "name": pd["label"],
            "wave_h": round(avg_h,2), "power": round(avg_p,2),
            "wind_kph": round(avg_w,1), "wind_dir": wc,
            "swell_h": round(avg_sw_h,2), "swell_period": round(avg_sw_p,2),
            "air_temp": round(avg_air,1) if avg_air is not None else "غير متوفر",
            "precip_mm": round(total_precip,1),
            "weather": weather_desc(most_common_code)
        })

    # شروق/غروب
    sunrise = dw.get("sunrise", ["غير معروف"])[0] if dw.get("sunrise") else "غير معروف"
    sunset = dw.get("sunset", ["غير معروف"])[0] if dw.get("sunset") else "غير معروف"

    # أعلام
    reds, greens = [], []
    for i in target_idx:
        hh = dtimes[i].strftime("%H:%M")
        if (wave_power[i] > 3.0 or wave_h[i] > 1.8 or gust_kph[i] > 50 or pressure[i] < 1005):
            reds.append(f"{hh} (طاقة{wave_power[i]:.1f}kW, ارتفاع{wave_h[i]:.1f}م, هبات{gust_kph[i]:.0f}كم/س)")
        if (0.3 <= wave_h[i] <= 1.0 and 0.1 <= wave_power[i] <= 1.5 and wind_kph[i] < 27.8):
            greens.append(hh)

    # أعشاب
    past_wp_avg = sum(wave_p[i] for i in past_idx) / max(len(past_idx),1)
    past_sh_avg = sum(swell_h[i] for i in past_idx) / max(len(past_idx),1)
    weed = (past_wp_avg >= 8.0 and past_sh_avg > 1.0 and dominant == "Onshore")

    # بيولوجيا
    avg_sst = sum(sst[i] for i in target_idx) / max(len(target_idx),1)
    bio = {}
    if avg_sst < 16 and dominant == "Onshore":
        bio["high"] = ["قاروص (Loup/Bar)", "سارغ كبير (Sargue)"]
    elif avg_sst > 19 and dominant != "Onshore":
        bio["high"] = ["دوراد رويال (Gilthead)", "ماربري (Striped)"]
    else:
        bio["high"] = []

    if beach_type == "rocky" and any(c == "Onshore" for c in wind_classes):
        bio.setdefault("additional", []).append("سارغ (بيئة صخرية)")
    if beach_type == "sandy":
        bio.setdefault("additional", []).append("بوري (Mullet) أو ماربري")

    return {
        "past_avg_power": round(past_avg_power,2),
        "dominant_wind": dominant,
        "sustained_hrs": sustained_hrs,
        "blocks": block_list,
        "red_flags": reds[:5], "green_flags": greens[:5],
        "weed_risk": weed,
        "bio": bio,
        "avg_sst": round(avg_sst,1),
        "air_temps": {k: (sum(v["temps"])/len(v["temps"]) if v["temps"] else None) for k,v in periods.items()},
        "extra_info": {
            "pressure_avg": round(avg_press,1),
            "pressure_trend": press_trend,
            "sunrise": sunrise,
            "sunset": sunset,
            "moon_phase": moon_phase(target_date_obj),
        }
    }

def build_context(req, agg, tz_name):
    beach = "رملي" if req.beach_type=="sandy" else "صخري"
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
        lines.append(f"الضغط الجوي المتوسط: {extra.get('pressure_avg', 'غير معروف')} hPa ({extra.get('pressure_trend', '')})")
        lines.append(f"شروق الشمس: {extra.get('sunrise', '')}")
        lines.append(f"غروب الشمس: {extra.get('sunset', '')}")
        lines.append(f"مرحلة القمر: {extra.get('moon_phase', '')}")

    if agg["blocks"]:
        lines.append("تفاصيل الفترات (صباح / ظهر / ليل):")
        for b in agg["blocks"]:
            lines.append(
                f"- {b['name']}: ارتفاع الموج {b['wave_h']}م, swell {b['swell_h']}م, "
                f"فترة swell {b['swell_period']}ث, طاقة {b['power']} kW/m, "
                f"رياح {b['wind_kph']} كم/س ({b['wind_dir']}), حرارة الهواء ~{b['air_temp']}°م, "
                f"أمطار {b['precip_mm']}مم, الطقس: {b['weather']}"
            )
    if agg["red_flags"]:
        lines.append("أوقات خطرة: " + "، ".join(agg["red_flags"]))
    if agg["green_flags"]:
        lines.append("أوقات مناسبة: " + "، ".join(agg["green_flags"]))
    if agg["bio"].get("high") or agg["bio"].get("additional"):
        lines.append("الأسماك المتوقعة:")
        if agg["bio"].get("high"): lines.append(f"- عالية: {', '.join(agg['bio']['high'])}")
        if agg["bio"].get("additional"): lines.append(f"- إضافية: {', '.join(agg['bio']['additional'])}")
    return "\n".join(lines)

# --------------- SYSTEM PROMPT بالعربية التونسية ---------------
SYSTEM_PROMPT = """أنت صياد سرفكاستينغ تونسي محترف ومحلل بحري قدير. اكتب تقريراً بحرياً تفصيلياً متكاملاً بالعربية الممزوجة بالمصطلحات التونسية الدارجة للصيد (كالمرصاص، اللدونة، التيارات الجارفة، القفلة، دود الكف، القمبري، الشريب...). يجب أن يكون التقرير نصاً واحداً متصلاً (بدون فواصل أو رموز خاصة)، وليس على شكل نقاط.

استخدم البيانات المتوفرة بدقة، وغطِّ النقاط التالية بالترتيب:

1. **التحليل العام للعوامل البحرية**: اذكر حالة الطقس المتوقعة لكل فترة (صباح/ظهر/ليل)، درجة حرارة الهواء، درجة حرارة الماء، وتأثيرها على نشاط الأسماك. اذكر مرحلة القمر وتأثيرها على الصيد الليلي. لا تنسَ ذكر وقت شروق الشمس وغروبها ودورهما في تحديد الساعة الذهبية.

2. **تحليل الأمواج والتيارات (هل تثبت الرصاصة؟)**: ارتفاع الموج بالمتر، swell (موج البحر الميت) وارتفاعه وفترته، طاقة الأمواج بالكيلوواط/متر، وصف حالة البحر. تأثير ذلك على ثبات الرصاصة ونوع الرصاصة المناسبة (هرم، مخالب/قرابين، صابونة) مع الوزن بالغرام. اذكر تأثير نوع القاع (رملي أو صخري) على التثبيت.

3. **تحليل الرياح والأعشاب (نقاء الماء)**: سرعة الرياح بالكيلومتر في الساعة واتجاهها بالنسبة للشاطئ (برية/بحرية/جانبية)، وتأثيرها على خروج الأعشاب ونقاء الماء. حلل تأثير هطول الأمطار على العكارة وإمكانية جلب الأكل من الوديان.

4. **تقييم سبوت الصيد (Surfcasting)**: إيجابيات وسلبيات الظروف، أفضل ساعات الصيد (مع ذكر التوقيت الدقيق)، تكتيك الرمي المقترح (المسافة، الزاوية، الطعوم المناسبة مثل دود الكف، الشريب، القمبري، السردين). حدد الأسماك المتوقع حضورها واحتمالية ظهورها بناءً على حرارة الماء، الموسم، نوع القاع، وبيانات الضغط الجوي (إذا كان في انخفاض ينشط الأسماك). لا تنسَ تنبيه الصياد بضرورة مراجعة جدول المد والجزر المحلي لتحديد أفضل توقيت لدخول السمك إلى الشاطئ.

استخدم لغة خبير ميداني، كن عملياً وموجزاً. لا تذكر أنك تلقيت بيانات أو أنك ذكاء اصطناعي."""

async def call_gemini(ctx):
    model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_PROMPT)
    try:
        resp = await asyncio.to_thread(model.generate_content, contents=ctx,
                                       generation_config={"temperature": 0.3, "max_output_tokens": 3000})
        if resp.candidates and resp.candidates[0].content.parts:
            return resp.candidates[0].content.parts[0].text.strip()
        raise SurfError("رد فارغ")
    except Exception as e:
        raise SurfError(f"فشل Gemini: {e}")

# --------------- نقطة النهاية ---------------
@app.post("/generate-report")
async def generate_report(req: ReportRequest):
    try:
        tz = await fetch_timezone(req.latitude, req.longitude)
        target_dt = target_date_from_str(req.target_date, tz)
        start = target_dt - timedelta(days=2)
        end = target_dt + timedelta(days=1)

        marine, weather = await asyncio.gather(
            fetch_marine(req.latitude, req.longitude, start.isoformat(), end.isoformat()),
            fetch_weather(req.latitude, req.longitude, start.isoformat(), end.isoformat())
        )
        agg = aggregate_physics(marine, weather, req.beach_orientation, req.beach_type, target_dt, tz)
        ctx = build_context(req, agg, tz)
        report = await call_gemini(ctx)
        return {"report": report, "meta": {"timezone": tz, "target_date": target_dt.isoformat()}}
    except Exception as e:
        logger.exception("خطأ")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
