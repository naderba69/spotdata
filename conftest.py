"""
conftest.py – تهيئة بيئة الاختبارات.

يضبط المتغيّرات قبل استيراد backend/main.py (لأن config.py يقرأ البيئة مرة واحدة)،
ويضيف مجلد backend إلى مسار الاستيراد، ويوفّر بيانات اصطناعية شبيهة
بردود Open-Meteo لاختبار خط التحليل دون الحاجة إلى شبكة.
"""

import math
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# تُقرأ هذه القيم عند استيراد config.py، لذا تُضبط قبل استيراد main
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")
os.environ.setdefault("RATE_LIMIT_ENABLED", "0")
os.environ.setdefault("CACHE_ENABLED", "0")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("ALLOWED_ORIGINS", "*")

import pytest  # noqa: E402

TZ = "Africa/Tunis"


# ---------------------------------------------------------------- بيانات اصطناعية
def _times(start: datetime, hours: int):
    return [(start + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(hours)]


def _wave(start: datetime, hours: int = 96, base: float = 0.6, amp: float = 0.3):
    return [round(base + amp * math.sin(i / 6.0), 3) for i in range(hours)]


def build_marine(start=None, hours: int = 96, tz: str = TZ, wave_dir: float = 90.0,
                 sst: float = 22.0, wave_base: float = 0.6, wave_amp: float = 0.3,
                 period: float = 6.0, none_every: int = 0):
    """حمولة بحرية شبيهة بردّ Open-Meteo Marine."""
    start = start or default_start()
    times = _times(start, hours)
    wave_h = _wave(start, hours, wave_base, wave_amp)

    def maybe(values):
        if not none_every:
            return values
        return [None if (i % none_every == 0) else v for i, v in enumerate(values)]

    return {
        "latitude": 36.8, "longitude": 10.1, "timezone": tz,
        "hourly": {
            "time": times,
            "wave_height": maybe(wave_h),
            "wave_period": [round(period + math.sin(i / 8.0), 2) for i in range(hours)],
            "wave_direction": [wave_dir] * hours,
            "swell_wave_height": [round(v * 0.8, 3) for v in wave_h],
            "swell_wave_period": [round(period + 1.0, 2)] * hours,
            "swell_wave_direction": [wave_dir] * hours,
            "sea_surface_temperature": [round(sst + math.sin(i / 12.0), 2) for i in range(hours)],
        },
    }


def build_weather(start=None, hours: int = 96, tz: str = TZ, wind_speed: float = 12.0,
                  wind_dir: float = 270.0, gusts: float = 22.0, pressure: float = 1013.0,
                  air_temp: float = 25.0, code: int = 0, visibility: float = 10000.0,
                  precipitation: float = 0.0, humidity: float = 60.0, days: int = 4):
    """حمولة طقس شبيهة بردّ Open-Meteo Forecast."""
    start = start or default_start()
    times = _times(start, hours)
    day0 = start.date()
    daily_time = [(day0 + timedelta(days=i)).isoformat() for i in range(days)]
    return {
        "latitude": 36.8, "longitude": 10.1, "timezone": tz,
        "hourly": {
            "time": times,
            "wind_speed_10m": [round(wind_speed + 3 * math.sin(i / 7.0), 2) for i in range(hours)],
            "wind_direction_10m": [wind_dir] * hours,
            "wind_gusts_10m": [round(gusts + 4 * math.sin(i / 5.0), 2) for i in range(hours)],
            "pressure_msl": [round(pressure + 0.05 * math.sin(i / 10.0), 2) for i in range(hours)],
            "temperature_2m": [round(air_temp + 4 * math.sin((i - 6) / 24.0 * math.pi), 2) for i in range(hours)],
            "relative_humidity_2m": [humidity] * hours,
            "precipitation": [precipitation] * hours,
            "visibility": [visibility] * hours,
            "weather_code": [code] * hours,
        },
        "daily": {
            "time": daily_time,
            "sunrise": [f"{d}T06:15" for d in daily_time],
            "sunset": [f"{d}T19:05" for d in daily_time],
        },
    }


def default_start():
    """يبدأ النطاق قبل اليوم بيومين كما يفعل الفرونتند (يومان ماضيان + اليوم + غداً)."""
    return datetime.combine(date.today() - timedelta(days=2), datetime.min.time())


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def tz_name():
    return TZ


@pytest.fixture()
def sample_marine():
    return build_marine()


@pytest.fixture()
def sample_weather():
    return build_weather()


@pytest.fixture()
def report_payload(sample_marine, sample_weather):
    return {
        "beach_orientation": 90,
        "beach_type": "sandy",
        "target_date": "tomorrow",
        "latitude": 36.8,
        "longitude": 10.1,
        "marine_data": sample_marine,
        "weather_data": sample_weather,
    }


@pytest.fixture(autouse=True)
def clean_caches():
    """يفرّغ الكاشات بين الاختبارات حتى لا تتسرّب نتائج اختبار إلى آخر."""
    import asyncio
    import main
    for cache in (main.report_cache, main.upstream_cache, main.overpass_cache,
                  main.failure_cache, main.health_cache):
        try:
            asyncio.run(cache.clear())
        except RuntimeError:  # داخل حلقة أحداث قائمة (اختبار غير متزامن)
            pass
    yield
    try:
        asyncio.run(main.report_cache.clear())
    except RuntimeError:
        pass


@pytest.fixture()
def client():
    from starlette.testclient import TestClient
    import main
    with TestClient(main.app) as c:
        yield c


@pytest.fixture()
def fake_gemini(monkeypatch):
    """يستبدل نداء Gemini الحقيقي بردّ ثابت ويحسب عدد النداءات."""
    import main
    calls = {"count": 0, "last_ctx": None}

    async def _fake(ctx):
        calls["count"] += 1
        calls["last_ctx"] = ctx
        return (
            "🎯 0. الملخص التنفيذي ليوم 07/09/2026\n"
            "> نسبة النجاح: 62%\n"
            ">  * القرار النهائي: فرصة مع تحفظات\n"
            "🕒 3. التفكيك الديناميكي الزمني\n"
            " * السحر (00:00 - 03:00): بحر هادئ\n"
            "⚖️ 4. ميزان العوامل\n"
            "🏹 5. التكتيك الميداني والسلامة\n"
        )

    monkeypatch.setattr(main, "call_gemini", _fake)
    return calls
