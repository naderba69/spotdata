import os, math, asyncio, logging, traceback, zoneinfo, time, json
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional
from collections import defaultdict

import httpx
import streamlit as st
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("surfcasting")

# ================== الإعدادات ==================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "google/gemini-2.5-flash"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

cache = {}
CACHE_TTL = 3600

# ================== دوال مساعدة ==================
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
    if phase<0.125 or phase>0.875: status, activity = "أيام متوسطة", "ينشط القاع ليلاً"
    elif phase<0.5: status, activity = "أيام حمل", "الأسماك نشيطة"
    else: status, activity = "أيام فساد", "الأسماك أقل نشاطاً"
    return {"name":name,"status":status,"activity":activity}

def moon_fishing_guidance(d:date):
    d_ = moon_phase_detail(d)
    n = d_["name"]
    if "محاق" in n: return f"{d_['status']}. ركز على الصيد الليلي."
    if "هلال أول" in n or "تربيع أول" in n: return f"{d_['status']}. فرصة ممتازة."
    if "أحدب متزايد" in n: return f"{d_['status']}. البوري والدوراد نهاراً."
    if "بدر" in n: return f"{d_['status']}. الأسماك السطحية نهاراً."
    return f"{d_['status']}. الصيد مقبول."

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
        except: continue

async def fetch_timezone_info(lat, lon):
    try:
        data = await fetch_with_retry(MARINE_URL, {"latitude":lat,"longitude":lon,"hourly":"wave_height","timezone":"auto","forecast_days":1}, timeout=10)
        return data.get("timezone","UTC")
    except: return "UTC"

def resolve_target_date(txt, real_today):
    if txt == "today": return real_today
    if txt == "tomorrow": return real_today + timedelta(days=1)
    return real_today + timedelta(days=2)

def align_hourly_data(marine_hourly, weather_hourly, tz_name):
    tz = zoneinfo.ZoneInfo(tz_name)
    m_times = marine_hourly.get("time", [])
    w_times = weather_hourly.get("time", [])
    if not m_times or not w_times: return [], {}
    m_map, w_map = {}, {}
    for i,t in enumerate(m_times):
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=tz)
        m_map[dt] = i
    for i,t in enumerate(w_times):
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=tz)
        w_map[dt] = i
    common = sorted(set(m_map)&set(w_map))
    if not common: return [], {}
    def extract(key, src, idx_map):
        arr = src.get(key,[])
        return [arr[idx_map[t]] if arr and idx_map[t]<len(arr) else 0.0 for t in common]
    aligned = {
        "wave_height": extract("wave_height", marine_hourly, m_map),
        "wave_period": extract("wave_period", marine_hourly, m_map),
        "swell_wave_height": extract("swell_wave_height", marine_hourly, m_map),
        "swell_wave_period": extract("swell_wave_period", marine_hourly, m_map),
        "sea_surface_temperature": extract("sea_surface_temperature", marine_hourly, m_map),
        "wind_speed_10m": extract("wind_speed_10m", weather_hourly, w_map),
        "wind_direction_10m": extract("wind_direction_10m", weather_hourly, w_map),
        "wind_gusts_10m": extract("wind_gusts_10m", weather_hourly, w_map),
        "pressure_msl": extract("pressure_msl", weather_hourly, w_map),
        "temperature_2m": extract("temperature_2m", weather_hourly, w_map),
        "precipitation": extract("precipitation", weather_hourly, w_map),
        "weather_code": [int(safe_float(x)) for x in extract("weather_code", weather_hourly, w_map)]
    }
    return common, aligned

# ================== تجميع البيانات ==================
def aggregate_physics(all_times, aligned, orient, target_date_obj, sunrise, sunset):
    tz = all_times[0].tzinfo if all_times else zoneinfo.ZoneInfo("UTC")
    target_start = datetime.combine(target_date_obj, datetime.min.time(), tzinfo=tz)
    target_end = target_start + timedelta(days=1)
    past_start = target_start - timedelta(hours=48)
    past_idx = [i for i,t in enumerate(all_times) if past_start<=t<target_start]
    target_idx = [i for i,t in enumerate(all_times) if target_start<=t<target_end]
    if not target_idx: return {"past_avg_power":0,"dominant_wind":"غير معروف","blocks":[],"red_flags":[],"green_flags":[],"weed_risk":False,"bio":{},"avg_sst":0,"extra_info":{}}
    def pick(k): return [aligned[k][i] for i in target_idx]
    wh = pick("wave_height"); wp = pick("wave_period"); swh = pick("swell_wave_height"); swp = pick("swell_wave_period")
    sst = pick("sea_surface_temperature"); ws = pick("wind_speed_10m"); wd = pick("wind_direction_10m")
    wg = pick("wind_gusts_10m"); pr = pick("pressure_msl"); ta = pick("temperature_2m"); prec = pick("precipitation")
    wcode = [int(v) if v else 0 for v in pick("weather_code")]
    wave_power = [0.49*(h**2)*p for h,p in zip(wh,wp)]
    wind_cls = [wind_class_detailed(angle_diff(d,orient)) for d in wd]
    past_avg = 0.0; past_sh = 0.0
    if past_idx:
        past_avg = sum(0.49*(aligned["wave_height"][i]**2)*aligned["wave_period"][i] for i in past_idx)/len(past_idx)
        past_sh = sum(aligned["swell_wave_height"][i] for i in past_idx)/len(past_idx)
    weed = wind_cls[0].startswith("بحرية") and (past_sh>0.8 or past_avg>5.0) if target_idx else False
    peak_gust = max(wg) if wg else 0.0
    dominant = max(set(wind_cls), key=wind_cls.count) if wind_cls else "غير معروف"
    periods = defaultdict(list)
    for idx,i in enumerate(target_idx):
        h = all_times[i].hour
        if 4<=h<=11: periods["morning"].append(idx)
        elif 12<=h<=17: periods["afternoon"].append(idx)
        else: periods["night"].append(idx)
    blocks = []
    for key in ["morning","afternoon","night"]:
        idxs = periods[key]
        if not idxs: continue
        avg_h = sum(wh[i] for i in idxs)/len(idxs)
        min_h,max_h = min(wh[i] for i in idxs), max(wh[i] for i in idxs)
        avg_pow = sum(wave_power[i] for i in idxs)/len(idxs)
        avg_w = sum(ws[i] for i in idxs)/len(idxs)
        min_w,max_w = min(ws[i] for i in idxs), max(ws[i] for i in idxs)
        wc_dom = max(set(wind_cls[i] for i in idxs), key=wind_cls.count)
        avg_swh = sum(swh[i] for i in idxs)/len(idxs)
        avg_swp = sum(swp[i] for i in idxs)/len(idxs)
        avg_air = sum(ta[i] for i in idxs)/len(idxs) if ta else 0
        total_precip = sum(prec[i] for i in idxs)
        most_code = max(set(wcode[i] for i in idxs), key=wcode[i].count) if idxs else 0
        swell_dom = "مختلط"
        if avg_swh>0.7*avg_h: swell_dom = "طاقة قادمة من بعيد"
        elif avg_h - avg_swh >0.2: swell_dom = "موج محلي"
        wind_start = wind_cls[idxs[0]]; wind_end = wind_cls[idxs[-1]]
        wind_trend = f"تتحول من {wind_start} إلى {wind_end}" if wind_start!=wind_end else f"ثابتة {wind_start}"
        sea = "هادئ" if avg_h<0.3 else "متوسط" if avg_h<0.8 else "هائج"
        blocks.append({
            "name":{"morning":"الصباح","afternoon":"الظهر","night":"الليل"}[key],
            "time_range":f"{all_times[target_idx[idxs[0]]].strftime('%H:%M')}-{all_times[target_idx[idxs[-1]]].strftime('%H:%M')}",
            "sea_state":sea,"wave_height":f"{min_h:.2f}-{max_h:.2f}","wave_power":round(avg_pow,2),
            "swell_height":f"{min(swh[i] for i in idxs):.2f}-{max(swh[i] for i in idxs):.2f}","swell_period":round(avg_swp,1),
            "swell_dominance":swell_dom,"wind_speed":f"{min_w:.1f}-{max_w:.1f}","wind_gust_peak":round(max(wg[i] for i in idxs),1),
            "wind_dir":wc_dom,"wind_trend":wind_trend,"air_temp":round(avg_air,1),"precip":round(total_precip,1),
            "weather":weather_desc(most_code)
        })
    reds, greens = [], []
    for i in range(len(wh)):
        hh = all_times[target_idx[i]].strftime("%H:%M")
        if wave_power[i]>3 or wh[i]>1.8 or wg[i]>50 or pr[i]<1005: reds.append(hh)
        if 0.3<=wh[i]<=1 and 0.1<=wave_power[i]<=1.5 and ws[i]<27.8: greens.append(hh)
    avg_sst = sum(sst)/len(sst) if sst else 0
    avg_press = sum(pr)/len(pr) if pr else 0
    bio = {}
    if avg_sst<16: bio["high"]=["قاروص","سارغ"]
    elif avg_sst>19: bio["high"]=["دوراد","ماربري"]
    moon = moon_phase_detail(target_date_obj)
    moon_g = moon_fishing_guidance(target_date_obj)
    extra = {
        "pressure_avg":round(avg_press,1),"sunrise":sunrise,"sunset":sunset,
        "moon_phase":moon["name"],"moon_status":moon["status"],"moon_guidance":moon_g,
        "peak_gust_today":round(peak_gust,1)
    }
    return {"past_avg_power":round(past_avg,2),"dominant_wind":dominant,"blocks":blocks,"red_flags":reds[:5],"green_flags":greens[:5],"weed_risk":weed,"bio":bio,"avg_sst":round(avg_sst,1),"extra_info":extra}

def build_context(lat, lon, orient, beach_type, target_date, agg, tz_name):
    beach = "رملي" if beach_type=="sandy" else "صخري"
    moon = agg["extra_info"]
    lines = [
        f"الموقع: شاطئ {beach} اتجاهه {orient}° شمال.",
        f"التاريخ: {target_date} (توقيت {tz_name})",
        f"حرارة الماء: {agg['avg_sst']}°م",
        f"القمر: {moon['moon_status']} ({moon['moon_phase']}). {moon['moon_guidance']}",
        f"الرياح السائدة: {agg['dominant_wind']}، هبات {moon['peak_gust_today']} كم/س",
        f"خطر الأعشاب: {'نعم' if agg['weed_risk'] else 'منخفض'}",
        f"طاقة الموج الماضية: {agg['past_avg_power']} kW/m"
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

SYSTEM_PROMPT = """أنت صياد سرفكاستينغ تونسي محترف. اكتب تقريراً بالدارجة التونسية، نص واحد متصل، يشمل تحليل البحر والموج والرياح والأعشاب والقمر، مع توصيات الرصاصة والتركيبة والطعم والسلامة. كن واقعياً ولا تبالغ."""

async def call_openrouter(ctx):
    headers = {"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json"}
    payload = {"model":MODEL_NAME,"messages":[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":ctx}],"max_tokens":7000,"temperature":0.3}
    async with httpx.AsyncClient() as c:
        r = await c.post(OPENROUTER_URL, json=payload, headers=headers, timeout=120)
        r.raise_for_status()
        data = r.json()
        if "choices" in data and data["choices"]: return data["choices"][0]["message"]["content"]
        raise Exception("OpenRouter فارغ")

async def generate_full_report(lat, lon, orient, beach_type, target_date_text):
    tz_name = await fetch_timezone_info(lat, lon)
    tz = zoneinfo.ZoneInfo(tz_name)
    now_tn = datetime.now(zoneinfo.ZoneInfo("Africa/Tunis"))
    target_dt = resolve_target_date(target_date_text, now_tn.date())
    start = target_dt - timedelta(days=2); end = target_dt + timedelta(days=1)
    marine = await fetch_with_retry(MARINE_URL, {
        "latitude":lat,"longitude":lon,
        "hourly":["wave_height","wave_period","wave_direction","swell_wave_height","swell_wave_period","swell_wave_direction","sea_surface_temperature"],
        "timezone":tz_name,"start_date":start.isoformat(),"end_date":end.isoformat()
    })
    weather = await fetch_with_retry(WEATHER_URL, {
        "latitude":lat,"longitude":lon,
        "hourly":["wind_speed_10m","wind_direction_10m","wind_gusts_10m","pressure_msl","temperature_2m","precipitation","weather_code"],
        "daily":["sunrise","sunset"],
        "timezone":tz_name,"start_date":start.isoformat(),"end_date":end.isoformat()
    })
    all_times, aligned = align_hourly_data(marine["hourly"], weather["hourly"], tz_name)
    if not all_times: raise Exception("لا بيانات ساعية")
    sunrise = weather["daily"]["sunrise"][0] if "sunrise" in weather.get("daily",{}) else "06:00"
    sunset = weather["daily"]["sunset"][0] if "sunset" in weather.get("daily",{}) else "18:00"
    agg = aggregate_physics(all_times, aligned, orient, target_dt, sunrise, sunset)
    ctx = build_context(lat, lon, orient, beach_type, target_date_text, agg, tz_name)
    report = await call_openrouter(ctx)
    return report, agg, tz_name, target_dt

# ================== واجهة Streamlit ==================
st.set_page_config(page_title="Surfcasting Analytics", page_icon="🎣", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0b132b; color: #e0e7ff; }
    .stButton>button { background-color: #00e5ff; color: #0b132b; font-weight: bold; border-radius: 10px; padding: 10px 24px; }
    .stButton>button:hover { background-color: #00b8d4; }
    .report-box { background-color: #1a2a4a; border-radius: 15px; padding: 20px; border: 1px solid #2a3a5a; }
    .metric-box { background-color: #0f1a2e; border-radius: 10px; padding: 15px; text-align: center; border: 1px solid #2a3a5a; }
</style>
""", unsafe_allow_html=True)

st.title("🎣 Surfcasting Analytics – تونس")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    lat = st.number_input("خط العرض", value=37.2000, format="%.4f")
with col2:
    lon = st.number_input("خط الطول", value=10.0500, format="%.4f")
with col3:
    beach_type = st.selectbox("نوع الشاطئ", ["sandy", "rocky"], format_func=lambda x: "رملي" if x=="sandy" else "صخري")

col4, col5 = st.columns(2)
with col4:
    orient = st.slider("اتجاه الشاطئ نحو البحر (°)", 0, 360, 45)
with col5:
    target_date_text = st.selectbox("التاريخ المستهدف", ["today", "tomorrow", "day_after"], format_func=lambda x: {"today":"اليوم","tomorrow":"غداً","day_after":"بعد غد"}[x])

if st.button("📊 توليد التقرير البحري", use_container_width=True):
    with st.spinner("جاري تحليل البيانات البحرية... قد يستغرق 30-50 ثانية"):
        try:
            report, agg, tz_name, target_dt = asyncio.run(generate_full_report(lat, lon, orient, beach_type, target_date_text))
            
            st.markdown("### 📋 التقرير البحري")
            st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 📊 ملخص البيانات")
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.markdown(f'<div class="metric-box">🌊 الموج<br><b>{agg["avg_sst"]}°م</b></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="metric-box">💨 الرياح<br><b>{agg["dominant_wind"]}</b></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="metric-box">⚠️ الخطر<br><b>{"نعم" if agg["weed_risk"] else "لا"}</b></div>', unsafe_allow_html=True)
            with c4: st.markdown(f'<div class="metric-box">🌙 القمر<br><b>{agg["extra_info"]["moon_phase"]}</b></div>', unsafe_allow_html=True)
            
            st.download_button("📥 تحميل التقرير", report, file_name=f"report_{target_dt.isoformat()}.txt")
        except Exception as e:
            st.error(f"❌ فشل إنشاء التقرير: {str(e)}")
