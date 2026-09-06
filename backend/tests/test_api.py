"""اختبارات نقاط النهاية (API) باستخدام TestClient وبدون أي اتصال شبكي فعلي."""

import pytest

import main as m


# ---------------------------------------------------------------- الصحة
def test_health_reports_status_and_dependencies(client, monkeypatch):
    async def _probe_orientation(*args, **kwargs):
        return None

    monkeypatch.setattr(m, "OVERPASS_SERVERS", ["http://127.0.0.1:9/api/interpreter"], raising=False)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == m.settings.VERSION
    assert "gemini_configured" in body and "caches" in body
    assert "reports" in body["caches"]


def test_request_id_is_echoed(client):
    response = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"
    assert response.headers["X-Response-Time"].endswith("ms")


# ---------------------------------------------------------------- التقرير
def test_generate_report_success(client, report_payload, fake_gemini):
    response = client.post("/generate-report", json=report_payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert "تقرير" in body["report"] or "الملخص" in body["report"]
    assert body["meta"]["score"] is not None
    assert body["meta"]["cached"] is False
    assert body["meta"]["final_verdict"]
    assert response.headers["X-Cache"] == "MISS"
    assert fake_gemini["count"] == 1
    assert "[الشروق والغروب]" in fake_gemini["last_ctx"]


def test_generate_report_uses_cache_on_second_call(client, report_payload, fake_gemini, monkeypatch):
    monkeypatch.setattr(m.report_cache, "enabled", True)
    first = client.post("/generate-report", json=report_payload)
    second = client.post("/generate-report", json=report_payload)
    assert first.headers["X-Cache"] == "MISS"
    assert second.headers["X-Cache"] == "HIT"
    assert second.json()["meta"]["cached"] is True
    assert fake_gemini["count"] == 1  # نداء Gemini واحد فقط
    assert first.json()["report"] == second.json()["report"]


def test_cache_key_changes_when_inputs_change(client, report_payload, fake_gemini, monkeypatch):
    """تغيير الإعدادات يولّد مفتاح كاش مختلفاً (طلب جديد وليس نسخة مخزّنة)."""
    monkeypatch.setattr(m.report_cache, "enabled", True)
    first = client.post("/generate-report", json=report_payload)
    other = dict(report_payload, beach_type="rocky")
    response = client.post("/generate-report", json=other)
    assert first.headers["X-Cache"] == "MISS"
    assert response.headers["X-Cache"] == "MISS"
    assert response.json()["report"]
    assert response.json()["meta"]["cached"] is False
    assert fake_gemini["count"] == 2


def test_missing_key_falls_back_to_offline_report(client, report_payload, monkeypatch):
    """غياب المفتاح لا يعطّل المستخدم: نردّ بتقرير محلي موسوم بأصل التوليد."""
    monkeypatch.setattr(m, "GEMINI_API_KEY", "")
    response = client.post("/generate-report", json=report_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["generated_by"] == "offline"
    assert body["meta"]["gemini_configured"] is False
    assert "🎯 0. الملخص التنفيذي" in body["report"]


def test_missing_key_returns_503_when_offline_disabled(client, report_payload, monkeypatch):
    monkeypatch.setattr(m, "GEMINI_API_KEY", "")
    monkeypatch.setattr(m.settings, "OFFLINE_FALLBACK", False)
    response = client.post("/generate-report", json=report_payload)
    assert response.status_code == 503
    assert "Gemini" in response.json()["detail"]


def test_gemini_failure_falls_back_to_manual_context(client, report_payload, monkeypatch):
    async def broken(ctx):
        raise RuntimeError("ضغط API")

    monkeypatch.setattr(m, "call_gemini", broken)
    response = client.post("/generate-report", json=report_payload)
    assert response.status_code == 200
    body = response.json()
    assert "manual_context" in body
    assert "اكتب تقرير صيد" in body["manual_context"]


def test_invalid_marine_payload_returns_400(client, report_payload):
    payload = dict(report_payload, marine_data={"hourly": {}})
    response = client.post("/generate-report", json=payload)
    assert response.status_code == 400
    assert "marine_data" in response.json()["detail"]


def test_invalid_weather_payload_returns_400(client, report_payload):
    payload = dict(report_payload, weather_data={"hourly": {"time": ["2026-09-06T00:00"]}})
    response = client.post("/generate-report", json=payload)
    assert response.status_code == 400
    assert "weather_data" in response.json()["detail"]


def test_upstream_failure_returns_502(client, monkeypatch):
    async def no_marine(*args, **kwargs):
        return None

    async def no_weather(*args, **kwargs):
        return None

    monkeypatch.setattr(m, "fetch_marine_data_cached", no_marine)
    monkeypatch.setattr(m, "fetch_weather_data_cached", no_weather)
    response = client.post("/generate-report", json={
        "beach_orientation": 90, "target_date": "today", "latitude": 36.8, "longitude": 10.1
    })
    assert response.status_code == 502


def test_hard_nogo_returns_short_report(client, monkeypatch, fake_gemini):
    from conftest import build_marine, build_weather
    payload = {
        "beach_orientation": 90,
        "beach_type": "sandy",
        "target_date": "tomorrow",
        "latitude": 36.8,
        "longitude": 10.1,
        "marine_data": build_marine(wave_base=3.2, wave_amp=0.5, period=4.0),
        "weather_data": build_weather(wind_speed=55, gusts=95, wind_dir=90, code=95),
    }
    response = client.post("/generate-report", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["hard_nogo"] is True
    assert body["report"].startswith("❌")
    assert fake_gemini["count"] == 0  # لا داعي لنداء Gemini


def test_validation_error_shape(client):
    response = client.post("/generate-report", json={"beach_orientation": 500, "target_date": "today"})
    assert response.status_code == 422
    assert "request_id" in response.json() or "detail" in response.json()


# ---------------------------------------------------------------- الاتجاه والقاع
def test_auto_orientation_endpoint(client, monkeypatch):
    async def fake_orientation(lat, lon):
        return 135

    monkeypatch.setattr(m, "get_auto_orientation_overpass", fake_orientation)
    response = client.post("/auto-orientation", json={"latitude": 36.8, "longitude": 10.1})
    assert response.status_code == 200
    body = response.json()
    assert body["orientation"] == 135 and body["source"] == "overpass"


def test_auto_orientation_endpoint_when_unknown(client, monkeypatch):
    async def fake_orientation(lat, lon):
        return None

    monkeypatch.setattr(m, "get_auto_orientation_overpass", fake_orientation)
    response = client.post("/auto-orientation", json={"latitude": 36.8, "longitude": 10.1})
    assert response.status_code == 200
    assert response.json()["orientation"] == -1


def test_detect_bottom_type_endpoint(client, monkeypatch):
    async def fake_bottom(lat, lon):
        return {"bottom_type": "rocky", "source": "nearby_beach", "confidence": "medium"}

    monkeypatch.setattr(m, "get_bottom_type_cached", fake_bottom)
    response = client.post("/detect-bottom-type", json={"latitude": 37.0575, "longitude": 11.0153})
    assert response.status_code == 200
    assert response.json()["bottom_type"] == "rocky"


# ---------------------------------------------------------------- كاش Overpass والفشل
@pytest.mark.asyncio
async def test_orientation_cache_avoids_repeated_overpass_calls(monkeypatch):
    calls = {"n": 0}

    async def fake_inner(lat, lon):
        calls["n"] += 1
        return 90

    monkeypatch.setattr(m, "_overpass_orientation_inner", fake_inner)
    monkeypatch.setattr(m.overpass_cache, "enabled", True)
    monkeypatch.setattr(m.failure_cache, "enabled", True)
    await m.overpass_cache.clear()
    await m.failure_cache.clear()

    first = await m.get_auto_orientation_overpass(36.8001, 10.1001)
    second = await m.get_auto_orientation_overpass(36.8002, 10.1002)  # نفس المفتاح بعد التقريب
    assert first == second == 90
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_failed_overpass_is_remembered_then_retried(monkeypatch):
    calls = {"n": 0}

    async def failing(lat, lon):
        calls["n"] += 1
        return None

    monkeypatch.setattr(m, "_overpass_orientation_inner", failing)
    monkeypatch.setattr(m.overpass_cache, "enabled", True)
    monkeypatch.setattr(m.failure_cache, "enabled", True)
    monkeypatch.setattr(m.failure_cache, "ttl", 0.05)
    await m.overpass_cache.clear()
    await m.failure_cache.clear()

    assert await m.get_auto_orientation_overpass(36.9, 10.9) is None
    assert await m.get_auto_orientation_overpass(36.9, 10.9) is None
    assert calls["n"] == 1  # المحاولة الثانية استُبدلت بتذكّر الفشل

    import asyncio
    await asyncio.sleep(0.08)
    assert await m.get_auto_orientation_overpass(36.9, 10.9) is None
    assert calls["n"] == 2  # بعد انتهاء مدة تذكّر الفشل نعيد المحاولة


def test_insufficient_data_for_target_day_returns_422(client, report_payload):
    """اليوم المطلوب خارج نطاق البيانات المتاحة → رسالة 422 واضحة بدل تقرير فارغ."""
    payload = dict(report_payload, target_date="day_after")
    response = client.post("/generate-report", json=payload)
    assert response.status_code == 422
    assert "بيانات" in response.json()["detail"]


def test_offline_report_is_returned_when_gemini_fails(client, report_payload, monkeypatch):
    """عند تعذّر Gemini نحصل على تقرير محلي مكتمل الأقسام (لا مجرد سياق يدوي)."""
    async def broken(ctx):
        raise RuntimeError("ضغط API")

    monkeypatch.setattr(m, "call_gemini", broken)
    response = client.post("/generate-report", json=report_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["generated_by"] == "offline"
    assert "manual_context" in body  # يبقى متاحاً لمن أراد اللصق في Gemini يدوياً
    report = body["report"]
    for header in ("🎯 0. الملخص التنفيذي", "⏱️ 1. التوقيت المدوي",
                   "🏃", "🕒 3. التفكيك الديناميكي الزمني",
                   "⚖️ 4. ميزان العوامل", "🏹 5. التكتيك الميداني والسلامة"):
        assert header in report, f"القسم {header} مفقود من التقرير الاحتياطي"
    assert "السحر" in report and "الغسق" in report


def test_offline_report_has_no_empty_parentheses(client, report_payload, monkeypatch):
    async def broken(ctx):
        raise RuntimeError("ضغط API")

    monkeypatch.setattr(m, "call_gemini", broken)
    report = client.post("/generate-report", json=report_payload).json()["report"]
    assert "()" not in report
