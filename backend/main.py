"""
Surfcasting Analytics API – v11.3.0 (Seasonal awareness + Lunar phase + Target species)
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
app = FastAPI(title="Surfcasting Analytics", version="11.3.0")
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
def health():
    return {"status": "ok", "version": "11.3.0"}


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
    return math.sqrt(dlat ** 2 + dlon ** 2)


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


def moon_phase_detail(d: date):
    """تفاصيل طور القمر مع حالة أيام الحمل والفساد."""
    y, m, day = d.year, d.month, d.day
    if m < 3: y -= 1; m += 12
    a = int(y / 100); b = 2 - a + int(a / 4)
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + day + b - 1524.5
    days_since_new = jd - 2451550.1
    phase = (days_since_new % 29.53058867) / 29.53058867
    idx = int(phase * 8) % 8
    names = {0: "محاق", 1: "هلال أول", 2: "تربيع أول", 3: "أحدب متزايد", 4: "بدر", 5: "أحدب متناقص", 6: "تربيع ثاني", 7: "هلال آخر"}
    name = names.get(idx, "محاق")
    if phase < 0.125 or phase > 0.875:
        lunar_status = "أيام متوسطة (محاق / هلال آخر)"
    elif phase < 0.5:
        lunar_status = "أيام حمل – الأسماك نشيطة طوال اليوم، الصيد ممتاز ليلاً ونهاراً"
    else:
        lunar_status = "أيام فساد – الأسماك أقل نشاطاً، يُفضّل الصيد في الساعات الذهبية (الشروق والغروب)"
    return {"name": name, "status": lunar_status, "phase": phase, "idx": idx}


def get_moon_and_tide_analysis(d: date):
    moon = moon_phase_detail(d)
    idx = moon["idx"]
    if idx in [0, 4]: tide_strength = "مد وجزر قوي جداً (Spring Tides)"
    elif idx in [2, 6]: tide_strength = "مد وجزر ضعيف جداً (Neap Tides)"
    else: tide_strength = "مد وجزر متوسط"
    return {"name": moon["name"], "status": moon["status"], "tide_strength": tide_strength, "idx": idx}


def get_seasonal_species_and_tips(month: int) -> dict:
    """الأسماك الموسمية والنصائح حسب الشهر في تونس."""
    if month in [12, 1, 2]:
        return {
            "target": "القاروص، السارغ (الشرغ)، الأنقليس (القنجة)",
            "bait": "الشريب (الدود البحري)، السردين المجمد، الحبار (الكالامار)",
            "tip": "الشتاء ممتاز للأسماك الكبيرة. ركز على الصيد الليلي أو في المياه العميقة. استعمل طعوم غنية بالبروتين."
        }
    elif month in [3, 4, 5]:
        return {
            "target": "البوري، الدوراد (الدنيس)، القاروص الصغير (المنكوس)",
            "bait": "دود الكف (الرمل)، القمبري (الجمبري) الطازج، المحار",
            "tip": "الربيع موسم التكاثر. الأسماك تقترب من الشاطئ. مثالي للصيد النهاري بالعجين أو الدود."
        }
    elif month in [6, 7, 8]:
        return {
            "target": "الدوراد (الدنيس)، المارمري (المرمار)، البوري الكبير",
            "bait": "السردين الطازج، القمبري الحي، الطعم الصناعي (الماكيت)",
            "tip": "الصيف حرارة عالية. الأسماك في العمق نهاراً وتقترب ليلاً. الصيد الليلي أو في الصباح الباكر جداً هو الأفضل."
        }
    else:  # 9, 10, 11
        return {
            "target": "القاروص الكبير، السارغ، الدوراد",
            "bait": "الشريب المخمّر، الحبار، السردين المجمد",
            "tip": "الخريف موسم الذروة. الأسماك تبدأ التغذية استعداداً للشتاء. أفضل أوقات السنة للصيد."
        }


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
    {"name":"شاطئ الكورنيش (بنزرت)","lat":37.2744,"lon":9.8739,"orientation":45,"type":"sandy"},
    # ... (القائمة الكاملة كما في v11.2)
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


# ==================== محرك التجميع الفيزيائي (بدون تغيير عن 11.2) ====================
def aggregate_physics(all_times, aligned, orient, target_date_obj, sunrise, sunset):
    # ... (نفس الكود الموجود في v11.2 بدون أي تغيير) ...
    pass  # حذف للاختصار، استخدم نفس دالة aggregate_physics من v11.2


# ==================== محرك التفاعلات (يُعيد best_tide_phase) ====================
def calculate_interactions(agg: dict) -> Tuple[List[str], str]:
    interactions = []
    hf = agg["hidden_factors"]
    blocks = agg["blocks"]
    lateral = agg["lateral_current"]
    extra = agg["extra_info"]
    is_mirror_sea = extra.get("is_mirror_sea", False)
    tide = agg["tide_analysis"]
    best_tide_phase = "طوال اليوم"

    if tide.get("tide_strength", "").startswith("مد وجزر قوي"):
        interactions.append("[تأثير المد] مد قوي (Spring Tide) – تيارات جانبية قوية متوقعة خاصة في الساعتين قبل وبعد الذروة. أفضل صيد يكون مع بداية الجزر أو نهاية المد.")
        best_tide_phase = "بداية الجزر أو نهاية المد"
    elif tide.get("tide_strength", "").startswith("مد وجزر ضعيف"):
        interactions.append("[تأثير المد] مد ضعيف (Neap Tide) – تيارات بطيئة، الماء يتحرك قليلاً. مناسب للصيد الهادئ طوال اليوم.")
    else:
        interactions.append("[تأثير المد] مد متوسط – التيارات معتدلة. أفضل فترتين هما آخر ساعتين من المد وأول ساعتين من الجزر.")
        best_tide_phase = "آخر ساعتين من المد وأول ساعتين من الجزر"

    sunrise = extra.get("sunrise", "06:00")
    sunset = extra.get("sunset", "18:00")
    interactions.append(f"[الساعة الذهبية] الشروق {sunrise} والغروب {sunset}. أفضل أوقات الصيد تكون قبل الشروق بساعة وقبل الغروب بساعتين.")

    if is_mirror_sea:
        interactions.append("[تفاعل البحر المرآوي] الموج أقل من 0.4م. قوة الدفع المائي شبه معدومة. التيار الجانبي مستحيل فيزيائياً.")
        night_green = [g for g in agg["green_flags"] if any(h in g for h in ["00:", "01:", "02:", "03:", "04:", "05:"])]
        if extra.get("max_air_temp", 0) > 28:
            interactions.append("[قاعدة المنطقة الميتة] الماء صافٍ كالمرآة + حرارة هواء عالية = المنطقة الضحلة القريبة من الشاطئ أصبحت 'منطقة ميتة بيولوجياً' نهاراً.")
            if night_green:
                interactions.append("[تفاعل الليل/العمق] لكن الظلام يكسر حاجز الخوف البصري. الأسماك ستعود للتغذية في المياه الضحلة ليلاً. يجب الرمي لمسافات بعيدة للوصول إلى أول حفرة أو منحدر.")
                interactions.append("[الحسم النهائي - Go ليلاً فقط] الصيد مقتصر على الليل مع رصاص انسيابي خفيف (80-120غ) ورمي بعيد جداً. لا فائدة من الصيد النهاري.")
                interactions.append("[تكتيك المكان] ارمِ خلف الحفرة الأولى (80-100 متر) أو عند المنحدر. الماء الضحل فارغ نهاراً.")
            else:
                interactions.append("[الحسم النهائي - No-Go مطلق (منطقة ميتة)] لا توجد ساعات ليلية مناسبة. لا تذهب للصيد.")
            return interactions, best_tide_phase
        else:
            interactions.append("[تفاعل الليل/بارد] الظلام يكسر حاجز الخوف البصري، والأسماك قد تقترب من الشاطئ للبحث عن غذاء في المياه الضحلة الهادئة.")
            interactions.append("[الحسم النهائي - Go ليلاً فقط] الصيد مقتصر على الليل. الأسماك ستكون حذرة جداً لذلك يجب إخفاء المعدات قدر الإمكان (خيوط Fluorocarbon رفيعة).")
            return interactions, best_tide_phase

    for b in blocks:
        wind_is_onshore = "بحرية" in b["wind_dir"]
        wave_is_straight = b.get("wave_angle_diff") is not None and b["wave_angle_diff"] < 60
        if wind_is_onshore and wave_is_straight:
            interactions.append(f"[تفاعل ميكانيكي - {b['name']}] رياح بحرية + موج عمودي = تضخيم الهيجان المباشر، دفع قوي للرصاصة للخلف. ارمِ بزاوية 45° عكس الريح.")
        elif "برية" in b["wind_dir"]:
            interactions.append(f"[تفاعل ميكانيكي - {b['name']}] رياح برية = تساعد على الرمي لمسافات أطول لكن لا توجد تيارات جانبية.")
        if "تقاطع" in b.get("swell_wave_interaction", ""):
            interactions.append(f"[تفاعل ميكانيكي - {b['name']}] {b['swell_wave_interaction']} فوضى عشوائية في حركة الماء.")

    if "جارف قوي" in lateral:
        interactions.append("[تفاعل الثبات] تيار جانبي قوي. استنتاج ميكانيكي: رصاصة هرمية أو مخالب لا تقل عن 150 غرام.")
    elif "ضعيف" in lateral or "معدوم" in lateral:
        interactions.append("[تفاعل الثبات] غياب تيار جانبي. استنتاج ميكانيكي: رصاصة 80-120 غرام كافية تماماً، الثقل الزائد يضر بالمسافة دون فائدة.")

    if "ارتفاع حاد" in agg["pressure_state"]:
        interactions.append("[تفاعل الفسيولوجيا] ارتفاع حاد في الضغط = توقف فوري للتغذية (المثانة الهوائية ممتلئة).")
    elif "انخفاض حاد" in agg["pressure_state"]:
        interactions.append("[تفاعل الفسيولوجيا] انخفاض حاد = نافذة ذهبية للتغذية العنيفة.")

    if "ضباب" in hf.get("visibility_status", ""):
        interactions.append("[تحذير الضباب] الرؤية منخفضة. تجنب الصيد الليلي إذا كان الضباب كثيفاً. استعمل مصباح رأس واحترس من الانزلاق.")

    if "خطير" in hf.get("cross_sea_risk", ""):
        interactions.append("[الحسم النهائي - No-Go] بحر مختلط خطير يمنع السيطرة.")
    elif len(agg["red_flags"]) >= 4:
        interactions.append(f"[الحسم النهائي - No-Go] هيجان متواصل ({len(agg['red_flags'])} ساعات خطر).")
    elif len(agg["green_flags"]) >= 3 and "معدوم" in lateral and "توقف" not in agg["pressure_state"]:
        interactions.append("[الحسم النهائي - Go] ظروف ميكانيكية مثالية (ثبات + نشاط).")
    else:
        interactions.append("[الحسم النهائي - No-Go] عدم توفر ظروف ميكانيكية أو بيولوجية كافية.")

    return interactions, best_tide_phase


# ==================== بناء السياق الموسمي ====================
def build_context(req, agg, tz_name):
    beach = "رملي" if req.beach_type == "sandy" else "صخري"
    orient = req.beach_orientation
    extra = agg["extra_info"]
    hf = agg["hidden_factors"]
    moon = agg["tide_analysis"]
    now = datetime.now(zoneinfo.ZoneInfo("Africa/Tunis"))
    seasonal = get_seasonal_species_and_tips(now.month)

    periods_detail = []
    for b in agg["blocks"]:
        detail = (
            f"【{b['name']} ({b['time_range']})】\n"
            f"• البحر: {b['sea_state']} | الموج: {b['wave_height']}م (دورة {b['swell_period']}ث)\n"
            f"• Swell: {b['swell_height']}م من {b['swell_dir']} (زاوية {b.get('swell_angle_diff','?')}°)\n"
            f"• الموج المحلي: {b.get('wave_dir','?')} (زاوية {b.get('wave_angle_diff','?')}°)\n"
            f"• الرياح: {b['wind_dir']} {b['wind_speed']} كم/س (هبات {b['wind_gust_peak']} كم/س) – {b.get('wind_trend','')}\n"
            f"• حرارة الهواء: {b['air_temp']}°م | السماء: {b['weather']} | أمطار: {b['precip']}مم\n"
            f"• طاقة الموج: {b['wave_power']} kW/m | مصدر الطاقة: {b.get('swell_dominance','مختلط')}\n"
            f"• تفاعل السويل والموج: {b['swell_wave_interaction']}\n"
        )
        periods_detail.append(detail)

    general_conditions = [
        f"حرارة الماء: {agg['avg_sst']}°م ({agg['sst_stability']}). اتجاه الشاطئ: {orient}°.",
        f"الضغط: {extra['pressure_avg']} hPa، تغير 3س: {extra.get('pressure_change_3h',0):+.1f} hPa.",
        f"القمر: {moon['name']} ({moon['status']}).",
        f"المد: {moon['tide_strength']}.",
        f"الذاكرة البحرية: {agg['sea_memory']}",
        f"التيار الجانبي: {agg['lateral_current']}",
        f"خطر البحر المرآة: {'نعم' if extra.get('is_mirror_sea') else 'لا'}",
        f"حرارة الهواء القصوى: {extra.get('max_air_temp', 'N/A')}°م",
        f"الرؤية: {hf.get('visibility_status','غير متوفرة')}",
        f"خطر الأعشاب: {'نعم' if hf.get('weed_risk') else 'لا'} | عكر الماء: {'نعم' if hf.get('clarity_risk') else 'لا'}",
        f"تغير مفاجئ للرياح: {hf.get('sudden_wind_shift','لا')}",
        f"صدمة حرارية: {hf.get('sst_trend','مستقر')}",
        f"البحر المختلط: {hf.get('cross_sea_risk','منخفض')}",
        f"تزامن المد: {hf.get('golden_lock','غير معروف')}",
        f"انحدار الموج: {hf.get('wave_steepness','غير معروف')}",
    ]

    timing = f"ساعات خضراء: {', '.join(agg['green_flags']) if agg['green_flags'] else 'لا يوجد'}\nساعات حمراء: {', '.join(agg['red_flags']) if agg['red_flags'] else 'لا يوجد'}"
    bio_text = "\n".join([f"- {fish}: {data['status']} ({data['reason']})" for fish, data in agg["bio_matrix"].items()])

    chain_interactions, best_tide_phase = calculate_interactions(agg)

    lines = [
        "=== قراءة رقمية للخريطة (Windy) ===",
        *periods_detail,
        "",
        "=== الظروف العامة ===",
        *general_conditions,
        "",
        f"=== الموسم وأيام القمر ===",
        f"نحن الآن في {moon['status']}. أفضل توقيت للمد: {best_tide_phase}.",
        f"الأسماك الموسمية في تونس ({now.strftime('%B')}): {seasonal['target']}.",
        f"الطعم الموسمي المقترح: {seasonal['bait']}.",
        f"نصيحة موسمية: {seasonal['tip']}",
        "",
        "=== الأسماك المتوقعة اليوم ===",
        bio_text,
        timing,
        "",
        "=== سلسلة التفاعلات والحسم ===",
        *chain_interactions,
        "",
        "[المهمة] اكتب تقريراً واحداً متصلاً (بدون تكرار). ابدأ بوصف كل فترة زمنية مع الأرقام، ثم حلل الأسباب والتأثيرات الميكانيكية، ثم أعط توصيات محددة (وزن الرصاص، الطعم، التوقيت، مسافة الرمي). اذكر أيام الحمل/الفساد والسمك الموسمي في تونس. اختم بالقرار النهائي (Go/No‑Go). لا تبتكر أرقاماً، استخدم الأرقام المعطاة فقط."
    ]
    return "\n".join(lines)


SYSTEM_PROMPT = """أنت قارئ خرائط Windy محترف ومحلل فيزيائي لصيد السرفكاستينغ في تونس. ستصلك قراءة رقمية كاملة للبحر والطقس في ثلاث فترات (صباح، ظهر، ليل). مهمتك: كتابة تقرير واحد متصل بالدارجة التونسية، بدون تكرار.

**هيكل التقرير الإجباري (مرة واحدة):**

1. **وصف الفترات (قراءة الصورة)**
   لكل فترة، اكتب فقرة قصيرة تصف ما "تراه" على الخريطة مباشرة، مع ذكر **الأرقام الأساسية** (سرعة الرياح، ارتفاع الموج، اتجاه السويل، حرارة الهواء).

2. **التحليل الفيزيائي**
   اربط الظواهر ببعضها دون تكرار: كيف تؤثر زوايا الموج على التيار؟ كيف تؤثر الرياح على الرمي؟ ماذا يعني الضغط؟ استخدم "الذاكرة البحرية". مرة واحدة فقط.

3. **السياق الموسمي والقمري**
   اذكر أننا في "أيام حمل" أو "أيام فساد" حسب القمر. اذكر السمك الموسمي في تونس والطعم الموسمي. اربط ذلك بالأسماك المتوقعة اليوم.

4. **التوصيات الميدانية**
   - الرصاصة (وزن ونوع).
   - الطعم (الموسمي + المناسب لصفاء الماء).
   - التوقيت (متى بالضبط، مع المد والجزر).
   - مسافة الرمي (قريب/بعيد).
   - تحذيرات (هبات، ضباب، بحر مرآوي).

5. **القرار النهائي**
   "Go" أو "No-Go". إذا "Go"، اذكر الوقت. إذا "No-Go"، اذكر السبب.

**قواعد صارمة:**
- لا تكرر أي جملة. كل فقرة تحمل معلومة جديدة.
- لا تخمن أرقاماً. استخدم الأرقام المعطاة فقط.
- إذا ذكر "Go ليلاً فقط"، لا تنصح بالصيد النهاري أبداً.
- اكتب بلغة صياد تونسي خبير، مباشر، دون مجاملات.
- لا تستخدم عبارات "بناءً على المعطيات" أو "حسب البيانات"."""


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
