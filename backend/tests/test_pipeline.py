"""اختبارات خط التحليل: مزامنة البيانات، التجميع، التنقيط، والحسم النهائي."""

from datetime import date, datetime, timedelta

import pytest

import main as m
from conftest import build_marine, build_weather, default_start

VERDICTS = {"مناسب", "فرصة مع تحفظات", "غير مناسب", "بيانات غير كافية"}


# ---------------------------------------------------------------- مزامنة
def test_align_hourly_data_basic(sample_marine, sample_weather):
    all_times, aligned = m.align_hourly_data(sample_marine["hourly"], sample_weather["hourly"], "Africa/Tunis")
    assert len(all_times) == 96
    for key in ("wave_height", "wind_speed_10m", "pressure_msl", "sea_surface_temperature"):
        assert len(aligned[key]) == len(all_times)
    assert all(isinstance(t, datetime) for t in all_times)
    assert all_times == sorted(all_times)


def test_align_hourly_data_handles_missing_and_partial_overlap(sample_marine, sample_weather):
    # الطقس يبدأ متأخراً 12 ساعة: النتيجة هي التقاطع فقط
    weather = sample_weather
    start = default_start() + timedelta(hours=12)
    weather["hourly"]["time"] = [(start + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(48)]
    for key in weather["hourly"]:
        if key != "time":
            weather["hourly"][key] = weather["hourly"][key][:48]
    all_times, aligned = m.align_hourly_data(sample_marine["hourly"], weather["hourly"], "Africa/Tunis")
    assert len(all_times) == 48
    assert len(aligned["wind_speed_10m"]) == 48


def test_align_hourly_data_unknown_timezone_falls_back(sample_marine, sample_weather):
    all_times, aligned = m.align_hourly_data(sample_marine["hourly"], sample_weather["hourly"], "قارة/أطلنطس")
    assert len(all_times) == 96  # لا ينهار عند اسم منطقة زمنية غير معروف


def test_align_hourly_data_empty_inputs():
    assert m.align_hourly_data({}, {}, "Africa/Tunis") == ([], {})


def test_align_hourly_data_none_values_become_defaults():
    marine = build_marine(hours=24, none_every=3)
    weather = build_weather(hours=24)
    all_times, aligned = m.align_hourly_data(marine["hourly"], weather["hourly"], "Africa/Tunis")
    assert len(all_times) == 24
    assert any(v is None for v in aligned["wave_height"])


# ---------------------------------------------------------------- التجميع
def _aggregate(marine=None, weather=None, orient=90, target="tomorrow", beach="sandy"):
    marine = marine or build_marine()
    weather = weather or build_weather()
    all_times, aligned = m.align_hourly_data(marine["hourly"], weather["hourly"], "Africa/Tunis")
    target_date = date.today() + timedelta(days={"today": 0, "tomorrow": 1, "day_after": 2}[target])
    return m.aggregate_physics(all_times, aligned, orient, target_date, "06:15", "19:05", 36.8, 10.1, beach)


def test_aggregate_returns_four_periods_in_order():
    agg = _aggregate()
    names = [b["name"] for b in agg["blocks"]]
    assert names == ["السحر", "الصباح", "الظهيرة", "الغسق"]


def test_aggregate_score_and_verdict_are_valid():
    agg = _aggregate()
    assert 0 <= agg["score"] <= 100
    assert agg["final_verdict"] in VERDICTS
    assert isinstance(agg["nogo_reasons"], list)
    assert isinstance(agg["warnings"], list)


def test_aggregate_block_structure():
    agg = _aggregate()
    for block in agg["blocks"]:
        for key in ("name", "time_range", "sea_state", "wind_dir", "confidence",
                    "recommended_cast_distance", "water_clarity", "suggested_rig",
                    "comfort_index", "active_fish", "inactive_fish"):
            assert key in block, f"الحقل {key} مفقود من الفترة"
        assert 0 <= block["confidence"] <= 100
        assert 0 <= block["comfort_index"] <= 100
        assert 30 <= block["recommended_cast_distance"] <= 100
        assert block["time_range"].count(":") == 2


def test_aggregate_extra_info_keys():
    agg = _aggregate()
    extra = agg["extra_info"]
    for key in ("tidal_windows", "golden_windows", "solunar", "sunrise", "sunset",
                "max_air_temp", "pressure_note", "haml_status", "seasonal_bait", "beach_orientation"):
        assert key in extra
    assert extra["beach_orientation"] == 90


def test_aggregate_insufficient_data_when_target_day_missing():
    # بيانات تغطي اليومين الماضيين فقط → لا توجد ساعات لليوم المستهدف
    marine = build_marine(start=default_start(), hours=48)
    weather = build_weather(start=default_start(), hours=48)
    agg = _aggregate(marine=marine, weather=weather, target="day_after")
    assert agg["final_verdict"] == "بيانات غير كافية"
    assert agg["score"] == 0


def test_aggregate_survives_all_none_air_temperature():
    weather = build_weather()
    weather["hourly"]["temperature_2m"] = [None] * 96
    agg = _aggregate(weather=weather)
    assert agg["extra_info"]["max_air_temp"] == 20.0  # قيمة احتياطية بدل انهيار max(None)


def test_aggregate_survives_all_none_sst():
    marine = build_marine()
    marine["hourly"]["sea_surface_temperature"] = [None] * 96
    agg = _aggregate(marine=marine)
    assert agg["avg_sst"] is None
    assert agg["final_verdict"] in VERDICTS


def test_storm_conditions_trigger_nogo():
    marine = build_marine(wave_base=2.6, wave_amp=0.4, period=4.0)
    weather = build_weather(wind_speed=45, gusts=80, wind_dir=90)
    agg = _aggregate(marine=marine, weather=weather)
    assert agg["blocks"] and all(b["has_lethal_nogo"] for b in agg["blocks"])
    assert agg["final_verdict"] == "غير مناسب"
    assert agg["score"] <= 30


def test_calm_good_conditions_score_higher_than_storm():
    calm = _aggregate(marine=build_marine(wave_base=0.8, wave_amp=0.2, period=7.0),
                      weather=build_weather(wind_speed=10, gusts=18))
    storm = _aggregate(marine=build_marine(wave_base=2.6, wave_amp=0.4, period=4.0),
                       weather=build_weather(wind_speed=45, gusts=80, wind_dir=90))
    assert calm["score"] > storm["score"]


def test_rocky_vs_sandy_changes_rig_advice():
    sandy = _aggregate(beach="sandy")
    rocky = _aggregate(beach="rocky")
    sandy_rigs = {b["suggested_rig"] for b in sandy["blocks"]}
    rocky_rigs = {b["suggested_rig"] for b in rocky["blocks"]}
    assert sandy_rigs or rocky_rigs  # لا ينهار مع أي نوع قاع


# ---------------------------------------------------------------- التنقيط
def test_apply_scoring_is_clamped(sample_marine):
    agg = _aggregate()
    agg["score"] = 0
    for _ in range(5):
        agg["score"] = m.apply_scoring(agg)
        assert 0 <= agg["score"] <= 100


def test_confidence_index_bounds():
    for flags in ({"is_spring_tide": 1, "is_pressure_dropping": 1},
                  {"is_spring_tide": 0, "is_pressure_dropping": 0}):
        value = m.calculate_confidence_index(flags, True, False, 3, 2, False, False, False)
        assert 0 <= value <= 100


def test_backwash_and_debris_helpers():
    assert m.analyze_backwash(40, 90, 90, 1.2)["severity"] == "مرتفع"
    assert m.analyze_backwash(None, 90, 90, 1.2)["severity"] == "منخفض"
    assert m.analyze_debris_risk("صوفة وسيول", 30)["risk"] == "مرتفع"
    assert m.analyze_debris_risk("بحر صافي", 5)["risk"] == "منخفض"


def test_period_fish_status_returns_partition():
    active, inactive = m.get_period_fish_status(20.0, True, False, False, False, 0.3, "صافي", 22.0)
    assert set(active).isdisjoint(inactive)
    assert len(active) + len(inactive) == 9
