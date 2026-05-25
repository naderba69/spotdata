"""
High-Precision Surfcasting Analytics Backend
FastAPI application - Deploy on Render as Web Service
"""
import os
import math
import asyncio
from datetime import datetime, timedelta, date
from typing import Optional, List, Tuple

import pytz
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import google.generativeai as genai

# ================== التهيئة ==================
app = FastAPI(title="Surfcasting Analytics API", version="1.0.0")

# CORS للسماح بالاتصال من أي واجهة أمامية (Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# مفتاح Gemini (يُخزَّن في متغيرات بيئة Render)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY غير موجود في متغيرات البيئة")

genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-1.5-flash"

# Open-Meteo API endpoints
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# ================== النماذج (Pydantic) ==================
class ReportRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    beach_orientation: int = Field(..., ge=0, le=360, description="اتجاه الشاطئ نحو البحر (درجات)")
    beach_type: str = Field(..., pattern="^(sandy|rocky)$")
    target_date: str = Field(..., pattern="^(today|tomorrow|day_after)$")

# ================== دوال مساعدة ==================
async def get_timezone(lat: float, lon: float) -> str:
    """
    جلب اسم المنطقة الزمنية من Open-Meteo باستخدام استدعاء بسيط.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wave_height",   # أي متغير، فقط لنحصل على timezone في الرد
        "timezone": "auto",
        "forecast_days": 1,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(MARINE_URL, params=params, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
        return data.get("timezone", "UTC")  # احتياط UTC

def resolve_target_date(target_str: str, tz_name: str) -> date:
    """
    تحويل today / tomorrow / day_after إلى تاريخ حقيقي بالمنطقة الزمنية المعطاة.
    """
    tz = pytz.timezone(tz_name)
    now = datetime.now(tz)
    if target_str == "today":
        return now.date()
    elif target_str == "tomorrow":
        return (now + timedelta(days=1)).date()
    else:  # day_after
        return (now + timedelta(days=2)).date()

async def fetch_marine_data(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """
    جلب بيانات الموج والارتفاع والطاقة (Open-Meteo Marine API).
    المخرجات تحتوي على مصفوفة ساعية لكل متغير.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "wave_height",
            "wave_period",
            "wave_direction",
            "swell_wave_height",
            "swell_wave_period",
            "swell_wave_direction",
            "sea_surface_temperature",
        ],
        "timezone": "auto",
        "start_date": start_date,
        "end_date": end_date,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(MARINE_URL, params=params, timeout=20.0)
        resp.raise_for_status()
        return resp.json()

async def fetch_weather_data(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """
    جلب بيانات الرياح والضغط (Open-Meteo Weather API).
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "pressure_msl",
        ],
        "timezone": "auto",
        "start_date": start_date,
        "end_date": end_date,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(WEATHER_URL, params=params, timeout=20.0)
        resp.raise_for_status()
        return resp.json()

def safe_float(val) -> float:
    """استبدال NaN / None بصفر لتجنب انهيار الحسابات."""
    if val is None:
        return 0.0
    try:
        if math.isnan(float(val)):
            return 0.0
        return float(val)
    except (ValueError, TypeError):
        return 0.0

def compute_angle_diff(wind_dir, beach_orientation):
    """حساب الفرق الزاوي بين الرياح والشاطئ مع القيم بين 0 و 180."""
    diff = abs(wind_dir - beach_orientation) % 360
    if diff > 180:
        diff = 360 - diff
    return diff

def classify_wind(angle_diff):
    if angle_diff < 45:
        return "Onshore"
    elif angle_diff > 135:
        return "Offshore"
    else:
        return "Sideshore"

# ================== محرك الفيزياء والتجميع ==================
def calculate_physics_and_aggregate(
    marine: dict,
    weather: dict,
    beach_orientation: int,
    beach_type: str,
    target_date: str,
    target_date_obj: date,
    tz_name: str,
) -> dict:
    """
    تنفيذ جميع الحسابات الفيزيائية، تجميع الكتل، استخراج الأعلام،
    وإنتاج الكائن المُضغوط الذي سيُرسل إلى Gemini.
    """
    # استخراج الأعمدة الزمنية والمتغيرات
    times_marine = marine["hourly"]["time"]
    times_weather = weather["hourly"]["time"]

    wave_heights = [safe_float(h) for h in marine["hourly"]["wave_height"]]
    wave_periods = [safe_float(p) for p in marine["hourly"]["wave_period"]]
    wave_dirs = [safe_float(d) for d in marine["hourly"]["wave_direction"]]
    swell_heights = [safe_float(h) for h in marine["hourly"]["swell_wave_height"]]
    swell_periods = [safe_float(p) for p in marine["hourly"]["swell_wave_period"]]
    swell_dirs = [safe_float(d) for d in marine["hourly"]["swell_wave_direction"]]
    sst = [safe_float(t) for t in marine["hourly"]["sea_surface_temperature"]]

    wind_speeds = [safe_float(v) for v in weather["hourly"]["wind_speed_10m"]]   # km/h
    wind_dirs = [safe_float(d) for d in weather["hourly"]["wind_direction_10m"]]
    wind_gusts = [safe_float(g) for g in weather["hourly"]["wind_gusts_10m"]]   # km/h
    pressures = [safe_float(p) for p in weather["hourly"]["pressure_msl"]]

    # ========== التحويلات والتصحيحات ==========
    # Open-Meteo يقدم سرعة الرياح بوحدة km/h -> نحول إلى عُقد
    wind_knots = [v * 0.5399568 for v in wind_speeds]
    # الهبات أصلًا بـ km/h نتركها كما هي
    wind_gusts_kmh = wind_gusts  # km/h

    # طاقة الموج الكلية (kW/m)
    wave_powers = [0.49 * (h ** 2) * p for h, p in zip(wave_heights, wave_periods)]
    # طاقة الطفح
    swell_powers = [0.49 * (h ** 2) * p for h, p in zip(swell_heights, swell_periods)]
    # سرعة الطفح (km/h)
    swell_speeds = [p * 5.6 for p in swell_periods]

    # تصنيف اتجاه الرياح لكل ساعة
    wind_classes = []
    for wd in wind_dirs:
        diff = compute_angle_diff(wd, beach_orientation)
        wind_classes.append(classify_wind(diff))

    # ========== تحويل الأوقات إلى كائنات datetime واعية بالمنطقة الزمنية ==========
    tz = pytz.timezone(tz_name)
    dtimes = []
    for t_str in times_marine:
        # Open-Meteo يعيد الأوقات بصيغة ISO مع الإزاحة (مثال: 2025-03-10T14:00+01:00)
        dt = datetime.fromisoformat(t_str)
        if dt.tzinfo is None:
            dt = tz.localize(dt)
        dtimes.append(dt)

    # ========== تقسيم الفترات: ماضي (48 ساعة) والهدف (24 ساعة) ==========
    target_start = tz.localize(datetime.combine(target_date_obj, datetime.min.time()))
    target_end = target_start + timedelta(days=1)  # حتى منتصف ليل اليوم التالي
    past_start = target_start - timedelta(hours=48)

    past_indices = [i for i, dt in enumerate(dtimes) if past_start <= dt < target_start]
    target_indices = [i for i, dt in enumerate(dtimes) if target_start <= dt < target_end]

    # التعامل مع البيانات المفقودة
    if not target_indices:
        # إذا لم تتوفر بيانات، نملأ بمتوسطات تقديرية
        past_avg_power = sum(wave_powers[i] for i in past_indices) / max(len(past_indices), 1)
        dominant_wind = max(set(wind_classes), key=wind_classes.count) if wind_classes else "Offshore"
        sustained_wind_hours = 0
        return {
            "past_avg_wave_power": round(past_avg_power, 2),
            "dominant_wind_48h": dominant_wind,
            "sustained_wind_hours": sustained_wind_hours,
            "blocks": [],
            "red_flags": [],
            "green_flags": [],
            "weed_risk": False,
            "bio_prediction": {},
            "error": "لا توجد بيانات كافية ليوم الهدف"
        }

    # ========== ملخص الـ 48 ساعة الماضية ==========
    past_wave_powers = [wave_powers[i] for i in past_indices]
    past_avg_power = sum(past_wave_powers) / max(len(past_wave_powers), 1)
    past_wind_classes = [wind_classes[i] for i in past_indices]
    dominant_wind = max(set(past_wind_classes), key=past_wind_classes.count) if past_wind_classes else "Offshore"
    sustained_wind_hours = sum(1 for i in past_indices if wind_knots[i] > 10)

    # ========== الكتل اليومية (صباح / ظهر / ليل) ==========
    blocks = {
        "morning": {"indices": [], "label": "الصباح (04-11)"},
        "afternoon": {"indices": [], "label": "الظهر (12-17)"},
        "night": {"indices": [], "label": "الليل (18-03)"}
    }
    for i in target_indices:
        hour = dtimes[i].hour
        if 4 <= hour <= 11:
            blocks["morning"]["indices"].append(i)
        elif 12 <= hour <= 17:
            blocks["afternoon"]["indices"].append(i)
        else:
            blocks["night"]["indices"].append(i)

    block_summaries = []
    for key, block in blocks.items():
        idxs = block["indices"]
        if not idxs:
            continue
        avg_h = sum(wave_heights[i] for i in idxs) / len(idxs)
        avg_p = sum(wave_powers[i] for i in idxs) / len(idxs)
        avg_wind = sum(wind_knots[i] for i in idxs) / len(idxs)
        wind_cat = max(set(wind_classes[i] for i in idxs), key=wind_classes.count)
        block_summaries.append({
            "name": block["label"],
            "avg_wave_height": round(avg_h, 2),
            "avg_wave_power": round(avg_p, 2),
            "avg_wind_knots": round(avg_wind, 1),
            "wind_direction": wind_cat,
        })

    # ========== الأعلام الحرجة والساعة ==========
    red_flags = []
    green_flags = []
    for i in target_indices:
        hour_str = dtimes[i].strftime("%H:%M")
        # أعلام حمراء
        if (wave_powers[i] > 3.0 or wave_heights[i] > 1.8 or
            wind_gusts_kmh[i] > 50 or pressures[i] < 1005):
            red_flags.append(f"{hour_str} (طاقة {wave_powers[i]:.1f}, ارتفاع {wave_heights[i]:.1f}م, هبات {wind_gusts_kmh[i]:.0f})")

        # أعلام خضراء
        if (0.3 <= wave_heights[i] <= 1.0 and
            0.1 <= wave_powers[i] <= 1.5 and
            wind_knots[i] < 15):
            green_flags.append(hour_str)

    # ========== خطر الأعشاب ==========
    past_wave_period_avg = sum(wave_periods[i] for i in past_indices) / max(len(past_indices), 1)
    past_swell_height_avg = sum(swell_heights[i] for i in past_indices) / max(len(past_indices), 1)
    weed_risk = (past_wave_period_avg >= 8.0 and past_swell_height_avg > 1.0 and dominant_wind == "Onshore")

    # ========== الاحتمالية البيولوجية ==========
    avg_sst = sum(sst[i] for i in target_indices) / max(len(target_indices), 1)
    bio = {}
    if avg_sst < 16 and dominant_wind == "Onshore":
        bio["high"] = ["قاروص (Loup/Bar)", "سارغ كبير (Sargue)"]
    elif avg_sst > 19 and (dominant_wind != "Onshore"):
        bio["high"] = ["دوراد رويال (Gilthead)", "ماربري (Striped)"]
    else:
        bio["high"] = []

    if beach_type == "rocky" and any(wc == "Onshore" for wc in wind_classes):
        bio.setdefault("additional", []).append("سارغ (بيئة صخرية)")

    if beach_type == "sandy":
        bio.setdefault("additional", []).append("بوري (Mullet) أو ماربري")

    return {
        "past_avg_wave_power": round(past_avg_power, 2),
        "dominant_wind_48h": dominant_wind,
        "sustained_wind_hours": sustained_wind_hours,
        "blocks": block_summaries,
        "red_flags": red_flags[:5],   # أبرز 5 فقط
        "green_flags": green_flags[:5],
        "weed_risk": weed_risk,
        "bio_prediction": bio,
        "avg_sst": round(avg_sst, 1),
    }

# ================== بناء النص المُرسل إلى Gemini ==================
def build_gemini_context(req: ReportRequest, aggregated: dict, tz_name: str) -> str:
    """
    تحويل البيانات المُجمعة إلى النص المُنسق الذي سيفهمه Gemini.
    """
    lat, lon = req.latitude, req.longitude
    orient = req.beach_orientation
    beach = "رملي" if req.beach_type == "sandy" else "صخري"

    text = f"""الموقع: خط عرض {lat:.2f}, خط طول {lon:.2f}, شاطئ {beach}, اتجاه الشاطئ {orient}°
التاريخ المستهدف: {req.target_date} (المنطقة الزمنية {tz_name})

الملخص التاريخي (48 ساعة الماضية):
- متوسط طاقة الموج: {aggregated['past_avg_wave_power']} kW/m
- الرياح السائدة: {aggregated['dominant_wind_48h']}
- مدة الرياح المستمرة > 10 عقدة: {aggregated['sustained_wind_hours']} ساعة

كتل فترات اليوم المستهدف:
"""
    for block in aggregated["blocks"]:
        text += f"- {block['name']}: ارتفاع {block['avg_wave_height']}م, طاقة {block['avg_wave_power']} kW/m, رياح {block['avg_wind_knots']} عقدة ({block['wind_direction']})\n"

    text += "\nالأعلام الحرجة (الخطر):\n"
    if aggregated["red_flags"]:
        for flag in aggregated["red_flags"]:
            text += f"- {flag}\n"
    else:
        text += "- لا يوجد\n"

    text += "\nالأعلام الخضراء (فترات مناسبة):\n"
    if aggregated["green_flags"]:
        for flag in aggregated["green_flags"]:
            text += f"- {flag}\n"
    else:
        text += "- لا يوجد\n"

    text += f"\nخطر الأعشاب: {'موجود' if aggregated['weed_risk'] else 'غير موجود'}\n"

    text += f"\nالاحتمال البيولوجي (حرارة الماء {aggregated.get('avg_sst', 'غير معروف')}°م):\n"
    if aggregated["bio_prediction"].get("high"):
        text += f"- احتمال مرتفع: {', '.join(aggregated['bio_prediction']['high'])}\n"
    if aggregated["bio_prediction"].get("additional"):
        text += f"- احتمال إضافي: {', '.join(aggregated['bio_prediction']['additional'])}\n"

    return text

# ================== استدعاء Gemini ==================
SYSTEM_PROMPT = """أنت صياد سرفكاستينغ محترف وخبير في التحليلات البحرية. مهمتك الوحيدة هي ترجمة المتغيرات الفيزيائية المُحتسبة مسبقاً والخطوط التاريخية والاحتمالات البيولوجية إلى تقرير صيد تكتيكي عملي مكتوب بالكامل باللغة العربية. لا تقم بأي عمليات حسابية. يجب تنسيق المخرجات بالضبط في 4 أقسام منفصلة بفاصل صارم '---' (بدون رموز إضافية) بحيث تستطيع الواجهة الأمامية تحليلها إلى بطاقات منفصلة:
القسم 1: التحليل التاريخي وحالة الماء ونقائه وعلاقته بآخر 48 ساعة.
---
القسم 2: تقلبات فترات اليوم وحركة التغيرات الفيزيائية بين الصباح والظهر والليل.
---
القسم 3: الجدول الساعي واستراتيجية المرصاص (اللدونة)، هل الرصاصة ستثبت في القاع بناءً على قيمة الطاقة المحسوبة أم ستخرج؟ وما هو الوزن ونوع الرصاص الأنسب (الهرم، المخالب Grappin، الصابونة)؟
---
القسم 4: التكهن الاحتمالي للأسماك المتوقع حضورها بناءً على حرارة الماء والموسم الحالي وقاع البحر مع تحديد الطعوم المناسبة لها وتكتيك الرمي.
حافظ على نبرة خبير، ند للند، عالية الاحترافية، ومرتكزة على الفيزياء باستخدام مصطلحات الصيد العربية الشائعة."""

async def call_gemini(prompt: str) -> str:
    """
    إرسال السياق إلى Gemini واستلام التقرير العربي.
    مع إعادة المحاولة مرة واحدة إذا لم تكن الأقسام 4.
    """
    model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_PROMPT)
    # الإعدادات: درجة حرارة منخفضة لنتائج متسقة
    generation_config = {
        "temperature": 0.2,
        "max_output_tokens": 2000,
    }
    try:
        response = await asyncio.to_thread(
            model.generate_content,
            contents=prompt,
            generation_config=generation_config,
        )
        # استخراج النص
        if response.candidates and response.candidates[0].content.parts:
            report = response.candidates[0].content.parts[0].text
        else:
            raise ValueError("استجابة Gemini فارغة")
        # التحقق من عدد الأقسام
        sections = report.split("---")
        if len(sections) != 4:
            # إعادة المحاولة مرة واحدة مع طلب أكثر صرامة
            retry_prompt = prompt + "\n\nتنبيه: يجب أن تحتوي إجابتك على 4 أقسام بالضبط، تفصلها '---' بدون أي نصوص أخرى."
            response2 = await asyncio.to_thread(
                model.generate_content,
                contents=retry_prompt,
                generation_config=generation_config,
            )
            if response2.candidates and response2.candidates[0].content.parts:
                report2 = response2.candidates[0].content.parts[0].text
                if len(report2.split("---")) == 4:
                    return report2
            # إذا فشلت المحاولة الثانية، نعيد الأول مع تنبيه للواجهة
            return report  # ستقوم الواجهة بعرض ما أمكن
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشل الاتصال بـ Gemini: {str(e)}")

# ================== نقطة النهاية الرئيسية ==================
@app.post("/generate-report", response_model=dict)
async def generate_report(req: ReportRequest):
    try:
        # 1. تحديد المنطقة الزمنية
        tz_name = await get_timezone(req.latitude, req.longitude)

        # 2. تحويل target_date إلى تاريخ حقيقي
        target_date_obj = resolve_target_date(req.target_date, tz_name)

        # 3. حساب تواريخ البداية والنهاية لفتح Open-Meteo (48 ساعة قبل + 24 ساعة الهدف)
        start_dt = target_date_obj - timedelta(days=2)  # 48 ساعة قبل
        end_dt = target_date_obj + timedelta(days=1)    # يوم الهدف + يوم بعده لضمان التغطية
        start_str = start_dt.isoformat()
        end_str = end_dt.isoformat()

        # 4. جلب البيانات البحرية والطقس بالتوازي
        marine_data, weather_data = await asyncio.gather(
            fetch_marine_data(req.latitude, req.longitude, start_str, end_str),
            fetch_weather_data(req.latitude, req.longitude, start_str, end_str),
        )

        # 5. تشغيل المحرك الفيزيائي والتجميع
        aggregated = calculate_physics_and_aggregate(
            marine_data,
            weather_data,
            req.beach_orientation,
            req.beach_type,
            req.target_date,
            target_date_obj,
            tz_name,
        )

        # 6. بناء سياق Gemini
        context = build_gemini_context(req, aggregated, tz_name)

        # 7. استدعاء Gemini والحصول على التقرير العربي
        report_arabic = await call_gemini(context)

        return {
            "report": report_arabic,
            "meta": {
                "timezone": tz_name,
                "target_date": target_date_obj.isoformat(),
                "blocks": aggregated["blocks"],
                "red_flags": aggregated["red_flags"],
                "green_flags": aggregated["green_flags"],
                "weed_risk": aggregated["weed_risk"],
            }
        }

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"خطأ في جلب البيانات: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ داخلي: {str(e)}")

# ================== تشغيل التطبيق ==================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
