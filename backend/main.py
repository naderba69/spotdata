"""
Surfcasting Analytics API - Final Production v4.0
FastAPI Backend with Gemini 2.5 Flash & Auto Beach Orientation
"""
import os, math, asyncio, logging, traceback, re, zoneinfo
from datetime import datetime, timedelta, date

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import google.generativeai as genai

# ------------------------- إعداد التسجيل -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("surfcasting")

# ------------------------- التهيئة -------------------------
app = FastAPI(title="Surfcasting Analytics", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise RuntimeError("GEMINI_API_KEY غير موجود في متغيرات البيئة")
genai.configure(api_key=GEMINI_KEY)

MODEL_NAME = "models/gemini-2.5-flash"

MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# ------------------------- النماذج -------------------------
class ReportRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    beach_orientation: int = Field(..., ge=0, le=360)
    beach_type: str = Field(..., pattern="^(sandy|rocky)$")
    target_date: str = Field(..., pattern="^(today|tomorrow|day_after)$")

class AutoOrientationRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

# ------------------------- معالجة الأخطاء -------------------------
class SurfError(Exception):
    pass

@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": str(exc), "type": type(exc).__name__})

@app.get("/health")
def health():
    return {"status": "ok", "gemini": bool(GEMINI_KEY), "model": MODEL_NAME}

# ------------------------- دوال المنطقة الزمنية -------------------------
async def fetch_timezone(lat: float, lon: float) -> str:
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(MARINE_URL, params={
                "latitude": lat, "longitude": lon,
                "hourly": "wave_height", "timezone": "auto", "forecast_days": 1
            }, timeout=10)
            r.raise_for_status()
            tz = r.json().get("timezone", "UTC")
            zoneinfo.ZoneInfo(tz)
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

# ------------------------- جلب البيانات -------------------------
async def fetch_marine(lat, lon, start, end):
    async with httpx.AsyncClient() as client:
        r = await client.get(MARINE_URL, params={
            "latitude": lat, "longitude": lon,
            "hourly": ["wave_height","wave_period","wave_direction",
                      "swell_wave_height","swell_wave_period","swell_wave_direction",
                      "sea_surface_temperature"],
            "timezone": "auto", "start_date": start, "end_date": end
        }, timeout=20)
        r.raise_for_status()
        return r.json()

async def fetch_weather(lat, lon, start, end):
    async with httpx.AsyncClient() as client:
        r = await client.get(WEATHER_URL, params={
            "latitude": lat, "longitude": lon,
            "hourly": ["wind_speed_10m","wind_direction_10m","wind_gusts_10m","pressure_msl"],
            "timezone": "auto", "start_date": start, "end_date": end
        }, timeout=20)
        r.raise_for_status()
        return r.json()

# ------------------------- الاتجاه التلقائي للشاطئ -------------------------
async def get_auto_orientation(lat: float, lon: float) -> int:
    """
    يستخدم OpenStreetMap Overpass API لإيجاد أقرب خط ساحلي
    ويعيد الزاوية العمودية باتجاه البحر (0-360).
    """
    radius = 1000  # متر
    query = f"""
    [out:json];
    (
      way(around:{radius},{lat},{lon})["natural"="coastline"];
    );
    out body;
    """
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(OVERPASS_URL, params={"data": query}, timeout=15.0)
            resp.raise_for_status()
            data = resp.json()

            elements = data.get("elements", [])
            if not elements:
                logger.warning("لم يتم العثور على خط ساحلي، نستخدم 0°")
                return 0

            # نأخذ أول قطعة خط ساحلي
            way = elements[0]
            nodes = way.get("geometry", [])
            if len(nodes) < 2:
                return 0

            # نأخذ أقرب نقطتين لنا لحساب الاتجاه
            p1 = nodes[0]
            p2 = nodes[-1]
            dx = p2["lon"] - p1["lon"]
            dy = p2["lat"] - p1["lat"]
            angle_rad = math.atan2(dx, dy)  # اتجاه الخط
            angle_deg = (math.degrees(angle_rad) + 360) % 360

            # الاتجاه نحو البحر هو العمودي +90 درجة (يمكن توجيهه حسب الموقع)
            sea_angle = (angle_deg + 90) % 360
            return int(round(sea_angle))

    except Exception as e:
        logger.error(f"فشل في حساب الاتجاه التلقائي: {e}")
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
    except: return 0.0

def angle_diff(wdir, beach_orient):
    d = abs(wdir - beach_orient) % 360
    return 360 - d if d > 180 else d

def wind_class(diff):
    if diff < 45: return "Onshore"
    if diff > 135: return "Offshore"
    return "Sideshore"

def aggregate_physics(marine, weather, beach_orient, beach_type, target_date_obj, tz_name):
    tz = zoneinfo.ZoneInfo(tz_name) if tz_name else zoneinfo.ZoneInfo("UTC")
    mh = marine["hourly"]
    wh = weather["hourly"]

    times = mh["time"]
    wave_h = [safe_float(x) for x in mh["wave_height"]]
    wave_p = [safe_float(x) for x in mh["wave_period"]]
    swell_h = [safe_float(x) for x in mh["swell_wave_height"]]
    swell_p = [safe_float(x) for x in mh["swell_wave_period"]]
    sst = [safe_float(x) for x in mh["sea_surface_temperature"]]

    wind_speed = [safe_float(x) for x in wh["wind_speed_10m"]]
    wind_dir = [safe_float(x) for x in wh["wind_direction_10m"]]
    wind_gust = [safe_float(x) for x in wh["wind_gusts_10m"]]
    pressure = [safe_float(x) for x in wh["pressure_msl"]]

    wind_knots = [v * 0.5399568 for v in wind_speed]
    gust_kph = wind_gust

    wave_power = [0.49 * (h**2) * p for h, p in zip(wave_h, wave_p)]
    swell_power = [0.49 * (h**2) * p for h, p in zip(swell_h, swell_p)]
    swell_speed = [p * 5.6 for p in swell_p]

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
        return {
            "past_avg_power": 0.0, "dominant_wind": "غير معروف", "sustained_hrs": 0,
            "blocks": [], "red_flags": [], "green_flags": [], "weed_risk": False,
            "bio": {}, "avg_sst": 0.0
        }

    past_avg_power = sum(wave_power[i] for i in past_idx) / max(len(past_idx), 1)
    past_wclass = [wind_classes[i] for i in past_idx]
    dominant = max(set(past_wclass), key=past_wclass.count) if past_wclass else "Offshore"
    sustained_hrs = sum(1 for i in past_idx if wind_knots[i] > 10)

    blocks = {
        "morning": {"indices": [], "label": "الصباح (04-11)"},
        "afternoon": {"indices": [], "label": "الظهر (12-17)"},
        "night": {"indices": [], "label": "الليل (18-03)"}
    }
    for i in target_idx:
        h = dtimes[i].hour
        if 4 <= h <= 11: blocks["morning"]["indices"].append(i)
        elif 12 <= h <= 17: blocks["afternoon"]["indices"].append(i)
        else: blocks["night"]["indices"].append(i)

    block_list = []
    for k, blk in blocks.items():
        idxs = blk["indices"]
        if not idxs: continue
        avg_h = sum(wave_h[i] for i in idxs)/len(idxs)
        avg_p = sum(wave_power[i] for i in idxs)/len(idxs)
        avg_w = sum(wind_knots[i] for i in idxs)/len(idxs)
        wc = max(set(wind_classes[i] for i in idxs), key=wind_classes.count)
        block_list.append({
            "name": blk["label"], "wave_h": round(avg_h,2),
            "power": round(avg_p,2), "wind_knots": round(avg_w,1),
            "wind_dir": wc
        })

    reds, greens = [], []
    for i in target_idx:
        hh = dtimes[i].strftime("%H:%M")
        if (wave_power[i] > 3.0 or wave_h[i] > 1.8 or gust_kph[i] > 50 or pressure[i] < 1005):
            reds.append(f"{hh} (طاقة{wave_power[i]:.1f}kW, ارتفاع{wave_h[i]:.1f}م, هبات{gust_kph[i]:.0f}كم/س)")
        if (0.3 <= wave_h[i] <= 1.0 and 0.1 <= wave_power[i] <= 1.5 and wind_knots[i] < 15):
            greens.append(hh)

    past_wp_avg = sum(wave_p[i] for i in past_idx)/max(len(past_idx),1)
    past_sh_avg = sum(swell_h[i] for i in past_idx)/max(len(past_idx),1)
    weed = (past_wp_avg >= 8.0 and past_sh_avg > 1.0 and dominant == "Onshore")

    avg_sst = sum(sst[i] for i in target_idx)/max(len(target_idx),1)
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
        "red_flags": reds[:5],
        "green_flags": greens[:5],
        "weed_risk": weed,
        "bio": bio,
        "avg_sst": round(avg_sst,1)
    }

# ------------------------- بناء سياق Gemini -------------------------
def build_gemini_text(req: ReportRequest, agg: dict, tz_name: str) -> str:
    beach = "رملي" if req.beach_type == "sandy" else "صخري"
    lines = [
        f"الموقع: خط عرض {req.latitude:.2f}, خط طول {req.longitude:.2f}, شاطئ {beach}, اتجاه الشاطئ {req.beach_orientation}°",
        f"التاريخ المستهدف: {req.target_date} (المنطقة الزمنية {tz_name})",
        "",
        "الملخص التاريخي (48 ساعة الماضية):",
        f"- متوسط طاقة الموج: {agg['past_avg_power']} kW/m",
        f"- الرياح السائدة: {agg['dominant_wind']}",
        f"- مدة الرياح > 10 عقدة: {agg['sustained_hrs']} ساعة",
        "",
        "كتل الفترات:"
    ]
    for b in agg["blocks"]:
        lines.append(f"- {b['name']}: ارتفاع {b['wave_h']}م, طاقة {b['power']} kW/m, رياح {b['wind_knots']} عقدة ({b['wind_dir']})")

    lines.append("\nالأعلام الحمراء:")
    lines.extend([f"- {f}" for f in agg["red_flags"]] if agg["red_flags"] else ["- لا يوجد"])
    lines.append("\nالأعلام الخضراء:")
    lines.extend([f"- {f}" for f in agg["green_flags"]] if agg["green_flags"] else ["- لا يوجد"])
    lines.append(f"\nخطر الأعشاب: {'موجود' if agg['weed_risk'] else 'غير موجود'}")
    lines.append(f"\nحرارة الماء: {agg['avg_sst']}°م")
    lines.append("الاحتمال البيولوجي:")
    if agg["bio"].get("high"):
        lines.append(f"- مرتفع: {', '.join(agg['bio']['high'])}")
    if agg["bio"].get("additional"):
        lines.append(f"- إضافي: {', '.join(agg['bio']['additional'])}")
    return "\n".join(lines)

SYSTEM_PROMPT = """أنت صياد سرفكاستينغ محترف وخبير في التحليلات البحرية. مهمتك الوحيدة هي ترجمة المتغيرات الفيزيائية المُحتسبة مسبقاً والخطوط التاريخية والاحتمالات البيولوجية إلى تقرير صيد تكتيكي عملي مكتوب بالكامل باللغة العربية.

تعليمات صارمة للتنسيق:
- يجب أن تكتب التقرير مكوناً من 4 أقسام بالضبط.
- كل قسم يجب أن يُفصل عن الآخر بالفاصل '---' (ثلاث شرطات) في سطر مستقل.
- لا تضيف أي عناوين جانبية أو أرقام أو نصوص قبل أو بعد الفواصل.
- لا تكتب أي شيء قبل القسم الأول ولا بعد القسم الرابع.
- مثال للهيكل المطلوب:
نص القسم الأول
---
نص القسم الثاني
---
نص القسم الثالث
---
نص القسم الرابع

الآن اكتب التقرير وفق البيانات المرفقة."""

async def call_gemini(text: str) -> str:
    model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_PROMPT)
    gen_cfg = {"temperature": 0.2, "max_output_tokens": 2000}
    try:
        resp = await asyncio.to_thread(model.generate_content, contents=text, generation_config=gen_cfg)
        if not resp.candidates or not resp.candidates[0].content.parts:
            raise SurfError("رد فارغ من Gemini")
        report = resp.candidates[0].content.parts[0].text
        # تنظيف وتقسيم صارم
        sections = re.split(r'\n?---\n?', report.strip())
        if len(sections) != 4:
            # محاولة إصلاح: ربما أضاف عناوين، نحاول تجاهل الأسطر التي تبدأ بـ "القسم"
            clean_sections = []
            for s in sections:
                # إزالة عنوان "القسم X:" إذا وجد
                s = re.sub(r'^القسم\s*\d+[:\-]?\s*', '', s.strip())
                clean_sections.append(s)
            if len(clean_sections) == 4:
                return "---".join(clean_sections)
            else:
                # فشل الإصلاح، نعيد المحاولة
                retry_text = text + "\n\nتنبيه مهم جداً: أعد كتابة التقرير بstrict 4 أقسام مفصولة بـ --- فقط، بدون أي نصوص إضافية."
                resp2 = await asyncio.to_thread(model.generate_content, contents=retry_text, generation_config=gen_cfg)
                if resp2.candidates and resp2.candidates[0].content.parts:
                    report2 = resp2.candidates[0].content.parts[0].text
                    sections2 = re.split(r'\n?---\n?', report2.strip())
                    if len(sections2) == 4:
                        return "---".join(sections2)
                # إذا فشل كل شيء، نعيد التقرير الأصلي مع فواصل واضحة
                return report
        return "---".join(sections)
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        raise SurfError(f"فشل Gemini: {str(e)}")

# ------------------------- نقطة النهاية -------------------------
@app.post("/generate-report")
async def generate_report(req: ReportRequest):
    try:
        logger.info(f"تقرير لـ {req.latitude},{req.longitude} اتجاه {req.beach_orientation}")
        tz = await fetch_timezone(req.latitude, req.longitude)
        target_dt = target_date_from_str(req.target_date, tz)
        start_dt = target_dt - timedelta(days=2)
        end_dt = target_dt + timedelta(days=1)
        start_str = start_dt.isoformat()
        end_str = end_dt.isoformat()

        marine, weather = await asyncio.gather(
            fetch_marine(req.latitude, req.longitude, start_str, end_str),
            fetch_weather(req.latitude, req.longitude, start_str, end_str)
        )
        agg = aggregate_physics(marine, weather, req.beach_orientation, req.beach_type, target_dt, tz)
        context = build_gemini_text(req, agg, tz)
        report = await call_gemini(context)
        return {
            "report": report,
            "meta": {"timezone": tz, "target_date": target_dt.isoformat()}
        }
    except SurfError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"خطأ Open-Meteo: {e.response.status_code}")
    except Exception as e:
        logger.exception("خطأ غير متوقع")
        raise HTTPException(status_code=500, detail=f"خطأ داخلي: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
