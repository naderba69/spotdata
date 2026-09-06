"""اختبارات الوحدات المساعدة والحسابات الفلكية/الفيزيائية."""

import math
from datetime import date, timedelta

import pytest

import main as m


# ---------------------------------------------------------------- أرقام ووقت
@pytest.mark.parametrize("value,expected", [
    (6.5, "06:30"),
    (24.0, "00:00"),
    (-0.25, "23:45"),
    (23.999, "00:00"),
    (float("nan"), "00:00"),
    (None, "00:00"),
])
def test_format_time(value, expected):
    assert m.format_time(value) == expected


@pytest.mark.parametrize("raw,expected", [
    (None, 0.0), ("abc", 0.0), (float("nan"), 0.0), (float("inf"), 0.0),
    ("3.5", 3.5), (7, 7.0),
])
def test_safe_float(raw, expected):
    assert m.safe_float(raw) == expected


def test_safe_parse_time():
    assert m.safe_parse_time("06:30") == pytest.approx(6.5)
    assert m.safe_parse_time("19") == pytest.approx(19.0)
    assert m.safe_parse_time("غير صالح") == pytest.approx(6.0)  # قيمة احتياطية آمنة


# ---------------------------------------------------------------- زوايا
@pytest.mark.parametrize("w,b,expected", [(350, 10, 20), (0, 180, 180), (10, 10, 0), (0, 270, 90)])
def test_angle_diff(w, b, expected):
    assert m.angle_diff(w, b) == pytest.approx(expected)


def test_signed_angle_diff_sign():
    assert m.signed_angle_diff(10, 350) == pytest.approx(20)
    assert m.signed_angle_diff(350, 10) == pytest.approx(-20)
    assert m.signed_angle_diff(None, 10) == 0.0


def test_circular_diff_wraps_midnight():
    assert m.circular_diff(23, 1) == pytest.approx(2)
    assert m.circular_diff(1, 23) == pytest.approx(2)


def test_circular_mean_of_opposite_angles():
    # 350° و 10° متوسطهما 0° (وليس 180°)
    result = m.circular_mean([350, 10])
    assert result is not None
    assert min(result % 360, 360 - result % 360) == pytest.approx(0, abs=1e-6)
    assert m.circular_mean([]) is None
    assert m.circular_mean([None, None]) is None


def test_deg_to_compass():
    assert m.deg_to_compass(0) == "شمال"
    assert m.deg_to_compass(90) == "شرق"
    assert m.deg_to_compass(180) == "جنوب"
    assert m.deg_to_compass(None) == "غير معروف"


def test_bearing_and_distance():
    assert m.calc_bearing(36.8, 10.1, 36.9, 10.1) == pytest.approx(0.0, abs=0.5)   # شمالاً
    assert m.calc_bearing(36.8, 10.1, 36.8, 10.2) == pytest.approx(90.0, abs=0.5)  # شرقاً
    assert m.calc_distance(36.8, 10.1, 37.8, 10.1) == pytest.approx(111_320, rel=0.01)


# ---------------------------------------------------------------- رياح وطقس
@pytest.mark.parametrize("diff,expected", [
    (0, "بحرية مباشرة"), (40, "بحرية خفيفة"), (90, "جانبية"),
    (160, "برية خفيفة"), (180, "برية مباشرة"),
])
def test_wind_class_detailed(diff, expected):
    assert m.wind_class_detailed(diff) == expected


def test_weather_desc():
    assert m.weather_desc(0) == "صافية تماماً"
    assert m.weather_desc(61) == "مطر"
    assert m.weather_desc(999) == "غير معروف"


# ---------------------------------------------------------------- قمر ومد
def test_moon_phase_index_is_valid():
    for day in range(1, 29):
        info = m.get_moon_and_tide_analysis(date(2026, 9, day))
        assert 0 <= info["idx"] <= 7
        assert 0.0 <= info["phase_decimal"] < 1.0
        assert info["name"]


def test_moon_age_wraps_within_lunar_month():
    age = m.get_moon_age_days(date(2026, 9, 6))
    assert 0.0 <= age < 29.53058867


def test_haml_and_mat_windows():
    # أيام الحياء: حمل البدر (13-16) وحمل المحاق (28-2)
    assert m.get_haml_mat_status(14.0)["status"] == "أيام الحياء"
    assert m.get_haml_mat_status(29.0)["status"] == "أيام الحياء"
    # أيام المات: التربيع الأول (7-9) والتربيع الثاني (21-23)
    assert m.get_haml_mat_status(8.0)["status"] == "أيام المات"
    assert m.get_haml_mat_status(22.0)["status"] == "أيام المات"
    assert m.get_haml_mat_status(5.0)["status"] == "أيام عادية"
    assert "الشاطئ" in m.get_fishing_platform_advice("أيام الحياء")


def test_solunar_major_periods_are_half_lunar_day_apart():
    sol = m.calculate_solunar(date(2026, 9, 6), 36.8, 10.1)
    assert set(sol) == {"major1", "major2", "minor1", "minor2"}
    for value in sol.values():
        assert len(value) == 5 and value[2] == ":"
    t1 = m.safe_parse_time(sol["major1"])
    t2 = m.safe_parse_time(sol["major2"])
    # الفترة الرئيسية الثانية تبعد نصف يوم قمري (12.42 ساعة) عن الأولى
    assert m.format_time(t1 + 12.42) == m.format_time(t2)


def test_estimate_tidal_windows_shape():
    analysis = m.get_moon_and_tide_analysis(date(2026, 9, 6))
    windows, golden = m.estimate_tidal_windows(date(2026, 9, 6), analysis, "06:15", "19:05", 36.8, 10.1)
    assert set(windows) == {"HW1", "LW1", "HW2", "LW2"}
    for value in windows.values():
        assert len(value) == 5
    assert golden and "المد والجزر في تونس ضعيف" in golden[0]


def test_format_time_gap_arabic_plural():
    assert m.format_time_gap(1.0) == "ساعة"
    assert m.format_time_gap(2.0) == "ساعتين"
    assert m.format_time_gap(1.5) == "ساعة و 30 دقيقة"
    assert m.format_time_gap(0.0) == "0 دقيقة"


# ---------------------------------------------------------------- منطق مساعد
def test_comfort_index_is_bounded():
    for temp in (-10, 0, 15, 25, 40, 50):
        for wind in (0, 15, 40, 80):
            value = m.calculate_comfort_index(temp, wind, 60)
            assert 0 <= value <= 100
    assert m.calculate_comfort_index(None, 10, 60) == 50


def test_confidence_labels_monotonic():
    labels = [m.get_confidence_label(v) for v in (95, 85, 75, 65, 55, 20)]
    assert len(set(labels)) == 6
    assert m.get_confidence_label(95) == "ذروة ملكية"
    assert m.get_confidence_label(20) == "ضعيفة"


def test_seasonal_bait_by_month():
    assert "سردين" in m.get_seasonal_bait(1, 15).lower() or "قمبري" in m.get_seasonal_bait(1, 15)
    assert "يفضل الطعم الحي" in m.get_seasonal_bait(7, 26)
    assert m.get_seasonal_bait(7, 20).endswith("الحبار") or "قمبري" in m.get_seasonal_bait(7, 20)


def test_resolve_timezone_fallback():
    assert str(m.resolve_timezone("Africa/Tunis")) == "Africa/Tunis"
    assert str(m.resolve_timezone("Not/AZone")) == m.settings.DEFAULT_TZ


def test_pick_daily_value_uses_dates_not_index():
    from conftest import default_start
    today = date.today()
    day0 = default_start().date()
    times = [(day0 + timedelta(days=i)).isoformat() for i in range(4)]
    daily = {"time": times, "sunrise": [f"{t}T06:15" for t in times]}
    # الفرونتند يرسل نطاقاً يبدأ قبل اليوم بيومين، الفهرس 0 ليس اليوم
    value = m.pick_daily_value(daily, "sunrise", today, today, "06:00")
    assert value.startswith(today.isoformat())
    assert m.pick_daily_value({"sunrise": []}, "sunrise", today, today, "06:00") == "06:00"


def test_water_clarity_and_rig_are_deterministic():
    assert "صافي" in m.get_water_clarity(5, 0.2, False, False, "أيام عادية")
    assert "عكر" in m.get_water_clarity(5, 0.2, True, False, "أيام عادية")
    assert m.suggest_rig("أيام عادية", False, 8, False, "sandy") != ""


def test_casting_angle_correction_bounded():
    assert m.casting_angle_correction(None, 90) == 0
    assert -30 <= m.casting_angle_correction(10, 200) <= 30
