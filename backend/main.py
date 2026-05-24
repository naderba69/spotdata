# =============================================================================
# SurfCast Analytics — FastAPI Backend
# Python: 3.11.9 | Deployment: Render Free Tier
# =============================================================================

import os, math, statistics, asyncio
import httpx
import google.generativeai as genai

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from datetime import datetime, timedelta, timezone

# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="SurfCast Analytics API", version="2.0.0")

# ─────────────────────────────────────────────────────────────────────────────
# CORS — الإصلاح الجذري
# allow_credentials=True + allow_origins="*" محظور بمعيار CORS
# الحل: credentials=False دائماً مع origins="*"
# ─────────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,          # ← الإصلاح الحاسم
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
    max_age=86400,
)

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────────────────────
class ForecastRequest(BaseModel):
    latitude:          float
    longitude:         float
    beach_orientation: int
    beach_type:        str
    target_date:       str

    @field_validator("beach_type")
    @classmethod
    def v_beach(cls, v):
        if v not in ("sandy", "rocky"):
            raise ValueError("beach_type must be 'sandy' or 'rocky'")
        return v

    @field_validator("target_date")
    @classmethod
    def v_date(cls, v):
        if v not in ("today", "tomorrow", "day_after"):
            raise ValueError("target_date must be 'today', 'tomorrow', or 'day_after'")
        return v

    @field_validator("beach_orientation")
    @classmethod
    def v_orient(cls, v):
        if not (0 <= v <= 360):
            raise ValueError("beach_orientation must be 0–360")
        return v

    @field_validator("latitude")
    @classmethod
    def v_lat(cls, v):
        if not (-90 <= v <= 90):
            raise ValueError("latitude must be -90 to 90")
        return v

    @field_validator("longitude")
    @classmethod
    def v_lng(cls, v):
        if not (-180 <= v <= 180):
            raise ValueError("longitude must be -180 to 180")
        return v

# ─────────────────────────────────────────────────────────────────────────────
# SAFE VALUE
# ─────────────────────────────────────────────────────────────────────────────
def _s(v, fb=0.0):
    if v is None: return fb
    try:
        f = float(v)
        return fb if math.isnan(f) else f
    except: return fb

def _get(lst, i, fb=0.0):
    try: return _s(lst[i], fb)
    except: return fb

# ─────────────────────────────────────────────────────────────────────────────
# FETCH OPEN-METEO
# ─────────────────────────────────────────────────────────────────────────────
MARINE_URL  = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

MARINE_VARS  = "wave_height,wave_period,swell_wave_height,swell_wave_period,swell_wave_direction,wind_wave_height,wind_wave_period"
WEATHER_VARS = "wind_speed_10m,wind_direction_10m,wind_gusts_10m,pressure_msl,temperature_2m"

async def fetch_data(req: ForecastRequest) -> dict:
    fd = {"today":1,"tomorrow":2,"day_after":3}[req.target_date]
    params = {
        "latitude":  req.latitude,
        "longitude": req.longitude,
        "past_days": 2,
        "forecast_days": fd,
        "timezone": "auto",
    }
    async with httpx.AsyncClient(timeout=30.0) as c:
        mr, wr = await asyncio.gather(
            c.get(MARINE_URL,  params={**params, "hourly": MARINE_VARS}),
            c.get(WEATHER_URL, params={**params, "hourly": WEATHER_VARS}),
        )
    if mr.status_code != 200:
        raise HTTPException(502, f"Marine API error {mr.status_code}")
    if wr.status_code != 200:
        raise HTTPException(502, f"Weather API error {wr.status_code}")

    mh = mr.json().get("hourly", {})
    wh = wr.json().get("hourly", {})
    ts = mh.get("time", [])

    merged = {}
    for i, t in enumerate(ts):
        merged[t] = {
            "wave_height":          _get(mh.get("wave_height",[]),          i),
            "wave_period":          _get(mh.get("wave_period",[]),          i),
            "swell_wave_height":    _get(mh.get("swell_wave_height",[]),    i),
            "swell_wave_period":    _get(mh.get("swell_wave_period",[]),    i),
            "swell_wave_direction": _get(mh.get("swell_wave_direction",[]), i),
            "wind_wave_height":     _get(mh.get("wind_wave_height",[]),     i),
            "wind_wave_period":     _get(mh.get("wind_wave_period",[]),     i),
            "wind_speed_10m":       _get(wh.get("wind_speed_10m",[]),       i),
            "wind_direction_10m":   _get(wh.get("wind_direction_10m",[]),   i),
            "wind_gusts_10m":       _get(wh.get("wind_gusts_10m",[]),       i),
            "pressure_msl":         _get(wh.get("pressure_msl",[]),         i, 1013.25),
            "temperature_2m":       _get(wh.get("temperature_2m",[]),       i, 15.0),
        }
    return merged

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def wind_class(wd, bo):
    diff = abs(wd - bo)
    if diff > 180: diff = 360 - diff
    if diff < 45:  return "Onshore"
    if diff > 135: return "Offshore"
    return "Sideshore"

def compute(row, bo):
    wh, wp   = row["wave_height"],       row["wave_period"]
    swh, swp = row["swell_wave_height"], row["swell_wave_period"]
    ws, wg   = row["wind_speed_10m"],    row["wind_gusts_10m"]
    wd       = row["wind_direction_10m"]
    return {
        **row,
        "wind_knots":  round(ws  * 1.943844, 3),
        "gust_kmh":    round(wg  * 3.6,      3),
        "wave_power":  round(0.49 * wh**2 * wp,  4),
        "swell_speed": round(swp * 5.6,          3),
        "swell_power": round(0.49 * swh**2 * swp,4),
        "wind_type":   wind_class(wd, bo),
    }

def flag(m):
    red = (
        m["wave_power"]   > 3.0  or
        m["wave_height"]  > 1.8  or
        m["gust_kmh"]     > 50.0 or
        m["pressure_msl"] < 1005.0
    )
    green = (
        0.3 <= m["wave_height"] <= 1.0 and
        0.1 <= m["wave_power"]  <= 1.5 and
        m["wind_knots"] < 15.0
    )
    return "RED" if red else ("GREEN" if green else "YELLOW")

def cold_front(temps):
    for i in range(3, len(temps)):
        if temps[i] - temps[i-3] <= -2.5:
            return True
    return False

# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────
BLOCKS = {
    "Morning":      range(4,12),
    "Afternoon":    range(12,18),
    "Evening_Night": list(range(18,24)) + list(range(0,4)),
}

def aggregate(merged, bo, beach_type, target_date):
    today   = datetime.now(timezone.utc).date()
    offsets = {"today":0,"tomorrow":1,"day_after":2}
    tday    = (today + timedelta(days=offsets[target_date])).isoformat()

    all_ts  = sorted(merged.keys())
    past_ts = [t for t in all_ts if t[:10] < tday]
    tday_ts = [t for t in all_ts if t[:10] == tday]

    comp = {t: compute(merged[t], bo) for t in all_ts}

    # 48h baseline
    past = [comp[t] for t in past_ts]
    def avg(lst, key, fb=0.0):
        vals = [h[key] for h in lst if h.get(key) is not None]
        return round(statistics.mean(vals), 3) if vals else fb

    awp48  = avg(past, "wave_power")
    ash48  = avg(past, "swell_wave_height")
    awper48= avg(past, "wave_period")
    wts48  = [h["wind_type"] for h in past] or ["Unknown"]
    dom_w  = max(set(wts48), key=wts48.count)
    sus_h  = sum(1 for h in past if h["wind_knots"] > 15)

    # diurnal
    tday_data = [comp[t] for t in tday_ts]
    def block_avg(hr_range):
        blk = [comp[t] for t in tday_ts if int(t[11:13]) in hr_range]
        if not blk: return None
        keys = ["wave_height","wave_power","wind_knots","gust_kmh",
                "swell_power","pressure_msl","temperature_2m"]
        out = {k: round(statistics.mean(h[k] for h in blk), 3) for k in keys}
        wts  = [h["wind_type"] for h in blk]
        out["dominant_wind_type"] = max(set(wts), key=wts.count)
        return out

    diurnal = {b: block_avg(h) for b, h in BLOCKS.items()}

    # flags
    flagged = {"RED":[],"GREEN":[],"YELLOW":[]}
    for t in tday_ts:
        m = comp[t]
        f = flag(m)
        flagged[f].append({
            "time":        t[11:16],
            "wave_height": m["wave_height"],
            "wave_power":  m["wave_power"],
            "wind_knots":  m["wind_knots"],
            "gust_kmh":    m["gust_kmh"],
            "pressure_msl":m["pressure_msl"],
            "wind_type":   m["wind_type"],
        })

    temps       = [comp[t]["temperature_2m"] for t in tday_ts]
    cf_detected = cold_front(temps) if temps else False
    if cf_detected:
        flagged["RED"].append({"time":"FULL_DAY","warning":"ΔT ≤ -2.5°C / 3h — موجة برد"})

    weed = awper48 >= 8.0 and ash48 > 1.0 and dom_w == "Onshore"

    # sinker
    max_wp = max((h["wave_power"] for h in tday_data), default=0.0)
    avg_wp = statistics.mean(h["wave_power"] for h in tday_data) if tday_data else 0.0
    if max_wp < 0.5:   sinker = "60–80g (ظروف خفيفة، سحب محدود)"
    elif max_wp < 1.5: sinker = "80–120g (توتر متوسط، تثبيت جيد)"
    elif max_wp < 3.0: sinker = "120–175g (توتر قوي، رصاصة مشوكة موصى بها)"
    else:              sinker = "175g+ أو إلغاء الجلسة — الرصاصة لن تتثبت"

    # species
    avg_t   = statistics.mean(temps) if temps else 15.0
    month   = today.month
    species = []

    if avg_t < 16.0 and dom_w == "Onshore":
        species += [
            {"species":"European Sea Bass (Loup / Bar)","probability":"HIGH",
             "trigger":"مياه باردة + موج بحري = ضغط على أسماك الطعم",
             "bait":"أنقليس حي، حبار طازج، أو سرطان محاري"},
            {"species":"Large White Seabream (Grand Sarg)","probability":"HIGH",
             "trigger":"مياه باردة + عكارة = تفعيل الصيد القاعي",
             "bait":"دود البحر، بلح البحر، أو البطل الطازج"},
        ]
    if avg_t > 19.0 and dom_w in ("Offshore","Sideshore"):
        species += [
            {"species":"Gilthead Seabream (Daurade Royale)","probability":"HIGH",
             "trigger":"بحر دافئ صافٍ هادئ = دوراد يدخل المياه الضحلة",
             "bait":"سرطان الشاطئ، حبار صغير، أو جمبري"},
            {"species":"Striped Seabream (Marbré)","probability":"HIGH",
             "trigger":"بحر دافئ هادئ + قاع رملي/صخري مختلط",
             "bait":"دود الرمل، جمبري صغير، أو بلح البحر"},
        ]
    if beach_type == "rocky" and any(h["wave_height"] > 0.4 for h in tday_data):
        if not any(s["species"]=="Large White Seabream (Grand Sarg)" for s in species):
            species.append({"species":"White Seabream (Sarg)","probability":"MEDIUM-HIGH",
                            "trigger":"قاع صخري + خط رغوة = ممر كمين السرق",
                            "bait":"قنفذ البحر، بلح البحر، أو دود على طعم قاعي"})
    if beach_type == "sandy" and dom_w in ("Offshore","Sideshore"):
        species.append({"species":"Flathead Mullet (Bouri) / Striped Seabream (Marbré)",
                        "probability":"MEDIUM",
                        "trigger":"شرف رملي هادئ = ممرات تغذية ضحلة",
                        "bait":"عجينة خبز (بوري)، دود رمل أو جمبري (مربو)"})
    if month in (10,11,12,1,2) and not any(s["species"]=="European Sea Bass (Loup / Bar)" for s in species):
        species.append({"species":"European Sea Bass (Loup / Bar)",
                        "probability":"MEDIUM (ذروة موسمية)",
                        "trigger":"شهور الشتاء تُنشّط عدوانية التغذية قبل الهجرة",
                        "bait":"سردينة طازجة، حبار، أو دود كبير"})
    if not species:
        species.append({"species":"Opportunistic Mixed Bag (Bogue, Mullet, Sarg)",
                        "probability":"LOW-MEDIUM",
                        "trigger":"ظروف انتقالية — لا محفز نوعي نشط",
                        "bait":"دود رمل أو قطع سرطان صغيرة"})

    return {
        "meta": {"latitude":None,"longitude":None,"target_date":tday,
                 "beach_orientation":bo,"beach_type":beach_type},
        "past_48h_baseline": {
            "avg_wave_power_48h":   awp48,
            "avg_swell_height_48h": ash48,
            "avg_wave_period_48h":  awper48,
            "dominant_wind_type":   dom_w,
            "sustained_wind_hours": sus_h,
        },
        "weed_risk_active":     weed,
        "diurnal_blocks":       diurnal,
        "flagged_hours":        flagged,
        "cold_front_detected":  cf_detected,
        "sinker_recommendation":sinker,
        "max_wave_power_today": round(max_wp, 3),
        "avg_wave_power_today": round(avg_wp, 3),
        "avg_temp_today":       round(avg_t,  1),
        "species_predictions":  species,
    }

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """أنت خبير صيد سيرف وتحليل بحري من المستوى الاحترافي. مهمتك الوحيدة هي تحويل البيانات الفيزيائية المسبقة الحساب إلى تقرير تكتيكي عملي.
لا تُجرِ أي حسابات. جميع الأرقام نهائية.

نسّق إجابتك في 4 أقسام حرفياً:

## 1. 🌊 السياق البحري والتاريخي
ربط حالة البحر خلال الـ 48 ساعة الماضية بالوضوح المتوقع للمياه وتواجد الأعشاب.

## 2. 🕐 التحولات اليومية
اليوم الكامل: الصباح vs الظهيرة vs الليل — مع التنبيه للانتقالات الحرجة.

## 3. ⚡ نوافذ الصيد والتجهيز
الساعات الخضراء: نوافذ الصيد المثلى. الساعات الحمراء: خطر أو غير منتج.
هل ستتثبت الرصاصة؟ حجم الخطاف وقوة الخيط والزاوية المناسبة.

## 4. 🎯 توقعات الصيد الاحتمالية
الأنواع مرتبة حسب الاحتمالية، مع السبب البيئي والطعم الأمثل والوقت الأنسب.

الأسلوب: مهني، مباشر، مبني على الفيزياء، كصياد يخاطب صياداً آخر. بدون حشو."""

def build_context(p, req):
    b = p["past_48h_baseline"]
    d = p["diurnal_blocks"]
    f = p["flagged_hours"]
    lines = [
        "=== حزمة البيانات التحليلية ===",
        f"الموقع: {req.latitude:.5f}°N, {req.longitude:.5f}°E",
        f"التاريخ المستهدف: {p['meta']['target_date']}",
        f"اتجاه الشاطئ: {req.beach_orientation}° | النوع: {req.beach_type.upper()}",
        "",
        "--- خط الأساس 48 ساعة ---",
        f"متوسط قوة الموج: {b['avg_wave_power_48h']} kW/m",
        f"متوسط ارتفاع الموج: {b['avg_swell_height_48h']} م",
        f"متوسط دورة الموج: {b['avg_wave_period_48h']} ث",
        f"نوع الريح السائد: {b['dominant_wind_type']}",
        f"ساعات رياح مستمرة (>15 عقدة): {b['sustained_wind_hours']}h",
        f"خطر الأعشاب: {'نعم — حمل أعشاب مرتفع' if p['weed_risk_active'] else 'لا — خطر منخفض'}",
        "",
        "--- نظرة عامة على اليوم ---",
        f"متوسط الحرارة: {p['avg_temp_today']}°م",
        f"موجة برد: {'نعم' if p['cold_front_detected'] else 'لا'}",
        f"أقصى قوة موج: {p['max_wave_power_today']} kW/m",
        f"متوسط قوة موج: {p['avg_wave_power_today']} kW/m",
        f"توصية الرصاصة: {p['sinker_recommendation']}",
        "",
        "--- متوسطات الكتل اليومية ---",
    ]
    for bn, bd in d.items():
        lines.append(f"  [{bn}]:")
        if bd:
            for k, v in bd.items():
                lines.append(f"    {k}: {v}")
        else:
            lines.append("    لا توجد بيانات")

    lines += ["", "--- الساعات المُعلَّمة ---"]
    for col in ("RED","GREEN","YELLOW"):
        hrs = f[col]
        if hrs:
            lines.append(f"  {col} ({len(hrs)}):")
            for h in hrs[:8]:
                if "warning" in h:
                    lines.append(f"    {h['time']}: {h['warning']}")
                else:
                    lines.append(
                        f"    {h['time']} → H={h.get('wave_height')}م | "
                        f"P={h.get('wave_power')}kW/m | "
                        f"W={h.get('wind_knots')}kts | "
                        f"G={h.get('gust_kmh')}km/h | "
                        f"Pr={h.get('pressure_msl')}hPa | "
                        f"T={h.get('wind_type')}"
                    )

    lines += ["", "--- المصفوفة البيولوجية ---"]
    for sp in p["species_predictions"]:
        lines += [
            f"  [{sp['species']}]",
            f"    الاحتمال: {sp['probability']}",
            f"    السبب: {sp['trigger']}",
            f"    الطعم: {sp['bait']}",
        ]
    lines += ["", "=== نهاية الحزمة ===", "أنشئ التقرير التكتيكي الكامل."]
    return "\n".join(lines)

async def call_gemini(ctx):
    if not GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY غير مُعيَّن في متغيرات البيئة")
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            temperature=0.6, max_output_tokens=2048, top_p=0.9
        ),
    )
    resp = await model.generate_content_async(ctx)
    return resp.text

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "SurfCast API v2.0"}

@app.get("/ping")
async def ping():
    """Lightweight keep-alive endpoint for UptimeRobot"""
    return {"pong": True}

@app.post("/analyze")
async def analyze(req: ForecastRequest):
    merged  = await fetch_data(req)
    if not merged:
        raise HTTPException(502, "لا توجد بيانات من Open-Meteo")
    payload = aggregate(merged, req.beach_orientation, req.beach_type, req.target_date)
    payload["meta"]["latitude"]  = req.latitude
    payload["meta"]["longitude"] = req.longitude
    ctx     = build_context(payload, req)
    report  = await call_gemini(ctx)
    return {
        "analytics":            payload,
        "ai_report":            report,
        "context_tokens_approx":len(ctx.split()),
    }
