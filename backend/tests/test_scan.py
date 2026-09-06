"""اختبارات مسح الشواطئ (بدون شبكة: مصادر البيانات تُستبدل ببيانات اصطناعية)."""

import asyncio
from datetime import date

import pytest

import main as m
from conftest import build_marine, build_weather


# ---------------------------------------------------------------- اختيار المرشحين
def test_candidate_beaches_sorted_by_distance_and_radius():
    found = m.candidate_beaches(36.85, 10.33, 40)  # حول قرطاج/تونس
    assert found, "يجب أن توجد شواطئ معروفة حول تونس العاصمة"
    distances = [b["distance_km"] for b in found]
    assert distances == sorted(distances)
    assert max(distances) <= 40


def test_candidate_beaches_respects_type_and_empty_result():
    rocky = m.candidate_beaches(37.05, 11.01, 60, beach_type="rocky")
    assert all(b["type"] == "rocky" for b in rocky)
    # وسط الصحراء: لا شواطئ قريبة
    assert m.candidate_beaches(33.0, 8.0, 5) == []


# ---------------------------------------------------------------- تنقيط بقعة واحدة
@pytest.mark.asyncio
async def test_score_spot_returns_score_and_best_period(monkeypatch):
    async def fake_marine(client, lat, lon):
        return build_marine()

    async def fake_weather(client, lat, lon):
        return build_weather()

    monkeypatch.setattr(m, "fetch_marine_data_cached", fake_marine)
    monkeypatch.setattr(m, "fetch_weather_data_cached", fake_weather)

    beach = {"name": "شاطئ اختبار", "lat": 36.8, "lon": 10.1, "orientation": 90,
             "type": "sandy", "distance_km": 3.2}
    today = date.today()
    result = await m.score_spot(None, beach, today + __import__("datetime").timedelta(days=1),
                                today, "Africa/Tunis", asyncio.Semaphore(2))
    assert "error" not in result, result.get("error")
    assert 0 <= result["score"] <= 100
    assert result["final_verdict"]
    assert result["best_period"]["name"]
    assert len(result["blocks"]) == 4
    assert result["distance_km"] == 3.2


@pytest.mark.asyncio
async def test_score_spot_reports_error_when_upstream_fails(monkeypatch):
    async def broken(client, lat, lon):
        return None

    monkeypatch.setattr(m, "fetch_marine_data_cached", broken)
    monkeypatch.setattr(m, "fetch_weather_data_cached", broken)
    beach = {"name": "بقعة", "lat": 36.8, "lon": 10.1, "orientation": 90, "type": "sandy"}
    result = await m.score_spot(None, beach, date.today(), date.today(),
                                "Africa/Tunis", asyncio.Semaphore(1))
    assert "error" in result and "score" not in result


# ---------------------------------------------------------------- نقطة النهاية
def _patch_upstream(monkeypatch):
    async def fake_marine(client, lat, lon):
        return build_marine()

    async def fake_weather(client, lat, lon):
        return build_weather()

    monkeypatch.setattr(m, "fetch_marine_data_cached", fake_marine)
    monkeypatch.setattr(m, "fetch_weather_data_cached", fake_weather)


def test_scan_spots_returns_ranked_list(client, monkeypatch):
    _patch_upstream(monkeypatch)
    response = client.post("/scan-spots", json={
        "latitude": 36.85, "longitude": 10.33, "radius_km": 60,
        "target_date": "tomorrow", "max_spots": 5,
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] >= 1
    assert body["count"] <= 5
    scores = [s["score"] for s in body["spots"]]
    assert scores == sorted(scores, reverse=True)
    assert body["best"]["name"] == body["spots"][0]["name"]
    assert body["failed"] == 0


def test_scan_spots_cache(client, monkeypatch):
    _patch_upstream(monkeypatch)
    monkeypatch.setattr(m.scan_cache, "enabled", True)
    payload = {"latitude": 36.85, "longitude": 10.33, "radius_km": 60,
               "target_date": "tomorrow", "max_spots": 4}
    first = client.post("/scan-spots", json=payload).json()
    second = client.post("/scan-spots", json=payload).json()
    assert first["cached"] is False
    assert second["cached"] is True
    assert first["spots"] == second["spots"]


def test_scan_spots_no_candidates(client):
    response = client.post("/scan-spots", json={
        "latitude": 33.0, "longitude": 8.0, "radius_km": 5,
        "target_date": "tomorrow", "max_spots": 5,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 0 and body["spots"] == []
    assert "message" in body and body["best"] is None


def test_scan_spots_validates_input(client):
    assert client.post("/scan-spots", json={"latitude": 36.8, "longitude": 10.1,
                                            "target_date": "later"}).status_code == 422


def test_scan_summary_offline(client, monkeypatch):
    async def broken(ctx):
        raise RuntimeError("ضغط API")

    monkeypatch.setattr(m, "call_gemini_scan", broken)
    spots = [
        {"name": "قرطاج", "score": 78, "final_verdict": "مناسب", "distance_km": 4.1,
         "orientation": 90, "beach_type": "sandy",
         "best_period": {"name": "الغسق", "time_range": "18:00 - 23:00", "confidence": 72,
                         "sea_state": "متموج خفيف", "wind_dir": "برية خفيفة",
                         "recommended_cast_distance": 60, "suggested_rig": "مونتاج عادي"}},
        {"name": "رادس", "score": 41, "final_verdict": "فرصة مع تحفظات", "distance_km": 12.0,
         "orientation": 90, "beach_type": "sandy",
         "best_period": {"name": "الصباح", "time_range": "04:00 - 11:00", "confidence": 55,
                         "sea_state": "هادئ", "wind_dir": "جانبية",
                         "recommended_cast_distance": 55, "suggested_rig": "مونتاج عادي"}},
    ]
    response = client.post("/scan-summary", json={"spots": spots, "target_date": "tomorrow"})
    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["generated_by"] == "offline"
    assert "قرطاج" in body["summary"] and "ملخّص مسح الشواطئ" in body["summary"]


def test_scan_summary_with_gemini(client, monkeypatch):
    async def fake(ctx):
        assert "عدد البقاع" in ctx
        return "🏆 أفضل البقاع:\n 1. قرطاج — بحر متموج ورياح برية."

    monkeypatch.setattr(m, "call_gemini_scan", fake)
    spots = [{"name": "قرطاج", "score": 78, "final_verdict": "مناسب", "distance_km": 4.1,
              "best_period": {"name": "الغسق", "time_range": "18:00 - 23:00", "confidence": 72}}]
    response = client.post("/scan-summary", json={"spots": spots, "target_date": "tomorrow"})
    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["generated_by"] == "gemini"
    assert "قرطاج" in body["summary"]


def test_scan_summary_requires_spots(client):
    response = client.post("/scan-summary", json={"spots": [], "target_date": "tomorrow"})
    assert response.status_code == 400


def test_build_scan_summary_context_lists_spots():
    ctx = m.build_scan_summary_context(m.ScanSummaryRequest(
        spots=[{"name": "مرسى", "score": 60, "final_verdict": "فرصة مع تحفظات",
                "distance_km": 9.0, "orientation": 45, "beach_type": "rocky",
                "best_period": {"name": "السحر", "time_range": "00:00 - 03:00", "confidence": 66,
                                "sea_state": "هادئ", "wind_dir": "برية مباشرة",
                                "recommended_cast_distance": 60},
                "solunar": {"major1": "08:00", "major2": "20:00"}, "avg_sst": 23.4,
                "sunrise": "06:15", "sunset": "19:05", "warnings": ["مد ضعيف"]}],
        target_date="tomorrow"))
    assert "مرسى" in ctx and "مد ضعيف" in ctx and "سولونار" in ctx
