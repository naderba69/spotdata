"""
Surfcasting Analytics API - Final Production v9.0
FastAPI Backend (Render) – 0% Physics Error, Unified Report, Tunisian Details
"""
import os, math, asyncio, logging, traceback, zoneinfo
from datetime import datetime, timedelta, date

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import google.generativeai as genai

# ------------------------- إعداد السجلات -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("surfcasting")

# ------------------------- التهيئة -------------------------
app = FastAPI(title="Surfcasting Analytics", version="9.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise RuntimeError("GEMINI_API_KEY مفقود من متغيرات البيئة")
genai.configure(api_key=GEMINI_KEY)

# النموذج الأحدث من Google
MODEL_NAME = "models/gemini-2.5-flash"

MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# ------------------------- النماذج -------------------------
class ReportRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    beach_orientation: int = Field(..., ge=0, le=360, description="اتجاه الشاطئ نحو البحر بالدرجات (0-360)")
    beach_type: str = Field(..., pattern="^(sandy|rocky)$")
    target_date: str = Field(..., pattern="^(today|tomorrow|day_after)$")

class AutoOrientationRequest(BaseModel):
    latitude: float
    longitude: float

class SurfError(Exception):
    pass

# ------------------------- معالجة الأخطاء المركزية -------------------------
@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})

@app.get("/health")
def health():
    return {"status": "ok", "gemini": bool(GEMINI_KEY), "model": MODEL_NAME}

# ------------------------- دوال المنطقة الزمنية -------------------------
async def fetch_timezone(lat: float, lon: float) -> str:
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(MARINE_URL, params={
                "latitude": lat, "longitude": lon,
                "hourly": "wave_height", "timezone": "auto", "forecast_days": 1
            }, timeout=10)
            r.raise_for_status()
            tz = r.json().get("timezone", "UTC")
            zoneinfo.ZoneInfo(tz)  # تحقق من الصلاحية
            return tz
    except Exception:
        return "UTC"

def target_date_from_str(txt: str, tz_name: str) -> date:
    try:
        tz = zoneinfo.ZoneInfo(tz_name)
    except Exception:
        tz = zoneinfo.ZoneInfo("UTC")
    now = datetime.now(tz)
    if txt == "today":
        return now.date()
    elif txt == "tomorrow":
        return now.date() + timedelta(days=1)
    return now.date() + timedelta(days=2)

# ------------------------- جلب البيانات البحرية والطقس -------------------------
async def fetch_marine(lat, lon, start, end):
    async with httpx.AsyncClient() as c:
        r = await c.get(MARINE_URL, params={
            "latitude": lat, "longitude": lon,
            "hourly": [
                "wave_height", "wave_period", "wave_direction",
                "swell_wave_height", "swell_wave_period", "swell_wave_direction",
                "sea_surface_temperature"
            ],
            "timezone": "auto",
            "start_date": start,
            "end_date": end
        }, timeout=20)
        r.raise_for_status()
        return r.json()

async def fetch_weather(lat, lon, start, end):
    async with httpx.AsyncClient() as c:
        r = await c.get(WEATHER_URL, params={
            "latitude": lat, "longitude": lon,
            "hourly": [
                "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
                "pressure_msl", "temperature_2m"
            ],
            "timezone": "auto",
            "start_date": start,
            "end_date": end
        }, timeout=20)
        r.raise_for_status()
        return r.json()

# ------------------------- الاتجاه التلقائي للشاطئ (OpenStreetMap) -------------------------
async def get_auto_orientation(lat: float, lon: float) -> int:
    """
    تحدد اتجاه الشاطئ نحو البحر باستخدام أقرب خط ساحلي في OpenStreetMap.
    تستخدم نصف قطر 5 كم وتدعم العلاقات (relations) والطرق (ways).
    """
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
            if not elements:
                logger.warning("لم يُعثر على خط ساحلي قريب")
                return 0

            # البحث عن أول عنصر به إحداثيات
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
                    p1 = geom[0]
                    p2 = geom[-1]
                    dx = p2["lon"] - p1["lon"]
                    dy = p2["lat"] - p1["lat"]
                    angle_rad = math.atan2(dx, dy)
                    angle_deg = (math.degrees(angle_rad) + 360) % 360
                    sea_angle = (angle_deg + 90) % 360  # عمودي على الخط نحو البحر
                    return int(round(sea_angle))
            return 0
    except Exception as e:
        logger.error(f"فشل تحديد الاتجاه: {e}")
        return 0

@app.post("/auto-orientation")
async def auto_orientation(req: AutoOrientationRequest):
    angle = await get_auto_orientation(req.latitude, req.longitude)
    return {"orientation": angle}

# ------------------------- المحرك الفيزيائي -------------------------
def safe_float(v):
    if v is None: return 0.0
    try:
        f = float(v)
        return 0.0 if math.isnan(f) else f
    except:
        return 0.0

def angle_diff(wdir, beach_orient):
    d = abs(wdir - beach_orient) % 360
    return 360 - d if d > 180 else d

def wind_class(diff):
    if diff < 45:
        return "Onshore"
    elif diff > 135:
        return "Offshore"
    return "Sideshore"

def aggregate_physics(marine, weather, beach_orient, beach_type, target_date_obj, tz_name):
    tz = zoneinfo.ZoneInfo(tz_name) if tz_name else zoneinfo.ZoneInfo("UTC")
    mh = marine.get("hourly", {})
    wh = weather.get("hourly", {})

    times = mh.get("time", [])
    wave_h = [safe_float(x) for x in mh.get("wave_height", [])]
    wave_p = [safe_float(x) for x in mh.get("wave_period", [])]
    swell_h = [safe_float(x) for x in mh.get("swell_wave_height", [])]
    swell_p = [safe_float(x) for x in mh.get("swell_wave_period", [])]
    sst = [safe_float(x) for x in mh.get("sea_surface_temperature", [])]

    wind_speed = [safe_float(x) for x in wh.get("wind_speed_10m", [])]     # km/h
    wind_dir = [safe_float(x) for x in wh.get("wind_direction_10m", [])]
    wind_gust = [safe_float(x) for x in wh.get("wind_gusts_10m", [])]     # km/h
    pressure = [safe_float(x) for x in wh.get("pressure_msl", [])]
    temp_air = [safe_float(x) for x in wh.get("temperature_2m", [])]

    wind_kph = wind_speed
    gust_kph = wind_gust

    wave_power = [0.49 * (h**2) * p for h, p in zip(wave_h, wave_p)]
    swell_power = [0.49 * (h**2) * p for h, p in zip(swell_h, swell_p)]

    wind_classes = []
    for wd in wind_dir:
        d = angle_diff(wd, beach_orient)
        wind_classes.append(wind_class(d))

    # تحويل الأوقات إلى كائنات datetime واعية بالمنطقة الزمنية
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
        return {
            "past_avg_power": 0.0,
            "dominant_wind": "غير معروف",
            "sustained_hrs": 0,
            "blocks": [],
            "red_flags": [],
            "green_flags": [],
            "weed_risk": False,
            "bio": {},
            "avg_sst": 0.0,
            "air_temps": {}
        }

    # ملخص 48 ساعة
    past_avg_power = sum(wave_power[i] for i in past_idx) / max(len(past_idx), 1)
    past_wclass = [wind_classes[i] for i in past_idx]
    dominant = max(set(past_wclass), key=past_wclass.count) if past_wclass else "Offshore"
    sustained_hrs = sum(1 for i in past_idx if wind_kph[i] > 18.5)

    # تجميع الفترات اليومية مع حرارة الهواء
    blocks_map = {
        "morning": {"indices": [], "label": "الصباح (04-11)", "temps": []},
        "afternoon": {"indices": [], "label": "الظهر (12-17)", "temps": []},
        "night": {"indices": [], "label": "الليل (18-03)", "temps": []}
    }
    for i in target_idx:
        h = dtimes[i].hour
        if 4 <= h <= 11:
            blocks_map["morning"]["indices"].append(i)
            blocks_map["morning"]["temps"].append(temp_air[i])
        elif 12 <= h <= 17:
            blocks_map["afternoon"]["indices"].append(i)
            blocks_map["afternoon"]["temps"].append(temp_air[i])
        else:
            blocks_map["night"]["indices"].append(i)
            blocks_map["night"]["temps"].append(temp_air[i])

    block_list = []
    for k, blk in blocks_map.items():
        idxs = blk["indices"]
        if not idxs:
            continue
        avg_h = sum(wave_h[i] for i in idxs) / len(idxs)
        avg_p = sum(wave_power[i] for i in idxs) / len(idxs)
        avg_w = sum(wind_kph[i] for i in idxs) / len(idxs)
        wc = max(set(wind_classes[i] for i in idxs), key=wind_classes.count)
        avg_sw_h = sum(swell_h[i] for i in idxs) / len(idxs)
        avg_sw_p = sum(swell_p[i] for i in idxs) / len(idxs)
        avg_air = sum(blk["temps"]) / len(blk["temps"]) if blk["temps"] else None
        block_list.append({
            "name": blk["label"],
            "wave_h": round(avg_h, 2),
            "power": round(avg_p, 2),
            "wind_kph": round(avg_w, 1),
            "wind_dir": wc,
            "swell_h": round(avg_sw_h, 2),
            "swell_period": round(avg_sw_p, 2),
            "air_temp": round(avg_air, 1) if avg_air is not None else "غير متوفر"
        })

    # أعلام حمراء / خضراء
    red_flags, green_flags = [], []
    for i in target_idx:
        hh = dtimes[i].strftime("%H:%M")
        if (wave_power[i] > 3.0 or wave_h[i] > 1.8 or gust_kph[i] > 50 or pressure[i] < 1005):
            red_flags.append(f"{hh} (طاقة{wave_power[i]:.1f}kW, ارتفاع{wave_h[i]:.1f}م, هبات{gust_kph[i]:.0f}كم/س)")
        if (0.3 <= wave_h[i] <= 1.0 and 0.1 <= wave_power[i] <= 1.5 and wind_kph[i] < 27.8):
            green_flags.append(hh)

    # خطر الأعشاب
    past_wp_avg = sum(wave_p[i] for i in past_idx) / max(len(past_idx), 1)
    past_sh_avg = sum(swell_h[i] for i in past_idx) / max(len(past_idx), 1)
    weed = (past_wp_avg >= 8.0 and past_sh_avg > 1.0 and dominant == "Onshore")

    # بيولوجيا
    avg_sst = sum(sst[i] for i in target_idx) / max(len(target_idx), 1)
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
        "past_avg_power": round(past_avg_power, 2),
        "dominant_wind": dominant,
        "sustained_hrs": sustained_hrs,
        "blocks": block_list,
        "red_flags": red_flags[:5],
        "green_flags": green_flags[:5],
        "weed_risk": weed,
        "bio": bio,
        "avg_sst": round(avg_sst, 1),
        "air_temps": {k: (sum(v)/len(v) if v else None) for k, v in blocks_map.items()}
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
    if agg["blocks"]:
        lines.append("تفاصيل الفترات:")
        for b in agg["blocks"]:
            lines.append(
                f"- {b['name']}: ارتفاع الموج {b['wave_h']}م, swell {b['swell_h']}م, "
                f"فترة swell {b['swell_period']}ث, طاقة {b['power']} kW/m, "
                f"رياح {b['wind_kph']} كم/س ({b['wind_dir']}), حرارة الهواء ~{b['air_temp']}°م"
            )
    if agg["red_flags"]:
        lines.append("أوقات خطرة: " + "، ".join(agg["red_flags"]))
    if agg["green_flags"]:
        lines.append("أوقات مناسبة: " + "، ".join(agg["green_flags"]))
    if agg["bio"].get("high") or agg["bio"].get("additional"):
        lines.append("الأسماك المتوقعة:")
        if agg["bio"].get("high"):
            lines.append(f"- عالية: {', '.join(agg['bio']['high'])}")
        if agg["bio"].get("additional"):
            lines.append(f"- إضافية: {', '.join(agg['bio']['additional'])}")
    return "\n".join(lines)

# ------------------------- نظام التعليمات (Prompt) -------------------------
SYSTEM_PROMPT = """أنت صياد سرفكاستينغ تونسي محترف ومحلل بحري. اكتب تقريراً بحرياً تفصيلياً متكاملاً بالعربية بناءً على البيانات المحسوبة. التقرير يجب أن يكون نصاً واحداً متصلاً (بدون فواصل أو رموز خاصة)، وليس على شكل نقاط أو أقسام منفصلة.

قم بتغطية النقاط التالية بالترتيب:

1. **التحليل العام للعوامل البحرية**: اذكر حالة الطقس المتوقعة، درجة حرارة الهواء صباحاً وظهراً ومساءً، درجة حرارة الماء وتأثيرها على نشاط الأسماك.

2. **تحليل الأمواج والتيارات (هل تثبت الرصاصة؟)**: ارتفاع الموج بالمتر، swell (موج البحر الميت) وارتفاعه وفترته، طاقة الأمواج بالكيلوواط/متر، وصف حالة البحر. تأثير ذلك على ثبات الرصاصة. أوصِ بوزن ونوع الرصاص المناسب (هرم، مخالب، صابونة) مع تعليل. اذكر تأثير نوع القاع (رملي/صخري).

3. **تحليل الرياح والأعشاب (نقاء الماء)**: سرعة الرياح بالكيلومتر في الساعة، اتجاهها بالنسبة للشاطئ (برية/بحرية/جانبية)، وتأثير ذلك على نقاء الماء وخروج الأعشاب.

4. **تقييم سبوت الصيد (Surfcasting)**: إيجابيات وسلبيات الظروف، أفضل ساعات الصيد (مع ذكر التوقيت)، تكتيك الرمي المقترح (المسافة، الزاوية)، الطعوم المناسبة (مثل دود الكف، الشريب، القشريات)، والأسماك المتوقع حضورها مع احتمالاتها بناءً على حرارة الماء والموسم ونوع القاع.

استخدم المصطلحات التونسية الدارجة: المرصاص، اللدونة، التيارات الجارفة، القفلة، دود الكف... كن دقيقاً، عملياً، وموجزاً. لا تذكر أنك تلقيت بيانات."""

async def call_gemini(context_text: str) -> str:
    model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_PROMPT)
    gen_cfg = {"temperature": 0.3, "max_output_tokens": 2500}
    try:
        resp = await asyncio.to_thread(model.generate_content, contents=context_text, generation_config=gen_cfg)
        if resp.candidates and resp.candidates[0].content.parts:
            return resp.candidates[0].content.parts[0].text.strip()
        raise SurfError("استجابة Gemini فارغة")
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        raise SurfError(f"فشل Gemini: {str(e)}")

# ------------------------- نقطة النهاية الرئيسية -------------------------
@app.post("/generate-report")
async def generate_report(req: ReportRequest):
    try:
        # 1. المنطقة الزمنية
        tz = await fetch_timezone(req.latitude, req.longitude)
        # 2. التاريخ
        target_dt = target_date_from_str(req.target_date, tz)
        # 3. نطاق البيانات
        start_dt = target_dt - timedelta(days=2)
        end_dt = target_dt + timedelta(days=1)
        start_str = start_dt.isoformat()
        end_str = end_dt.isoformat()

        # 4. جلب البيانات
        marine_data, weather_data = await asyncio.gather(
            fetch_marine(req.latitude, req.longitude, start_str, end_str),
            fetch_weather(req.latitude, req.longitude, start_str, end_str)
        )
        # 5. الحسابات الفيزيائية
        agg = aggregate_physics(
            marine_data, weather_data,
            req.beach_orientation, req.beach_type,
            target_dt, tz
        )
        # 6. بناء سياق Gemini
        context = build_context(req, agg, tz)
        # 7. استدعاء Gemini
        report = await call_gemini(context)

        return {
            "report": report,
            "meta": {
                "timezone": tz,
                "target_date": target_dt.isoformat(),
                "orientation_used": req.beach_orientation
            }
        }

    except SurfError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"خطأ Open-Meteo: {e.response.status_code}")
    except Exception as e:
        logger.exception("خطأ غير متوقع")
        raise HTTPException(status_code=500, detail=f"خطأ داخلي: {str(e)}")

# ------------------------- تشغيل التطبيق -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
