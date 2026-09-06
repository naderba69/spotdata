"""اختبارات تنظيف نص التقرير قبل إرساله للمستخدم."""

import main as m


def test_fix_time_ranges_swaps_reversed_range():
    assert m.fix_time_ranges("الفترة 13:00 - 11:00") == "الفترة 11:00 - 13:00"


def test_fix_time_ranges_keeps_valid_range():
    assert m.fix_time_ranges("الفترة 06:00 - 18:00") == "الفترة 06:00 - 18:00"


def test_clean_report_text_fixes_glued_times_and_bold():
    raw = "العبور القمري:06:00 **عنوان** 22.5° م"
    out = m.clean_report_text(raw)
    assert "العبور القمري: 06:00" in out
    assert "**" not in out
    assert "22.5°م" in out


def test_fix_broken_number_lines_merges_split_time():
    text = "الساعة 06:\n30 تماماً"
    assert "06:30" in m.fix_broken_number_lines(text)


def test_fix_broken_time_in_headers():
    text = "🕒 3. التفكيك\n* السحر (00:00\n* 03:00)"
    assert "(00:00 - 03:00)" in m.fix_broken_time_in_headers(text)


def test_replace_english_commas_between_arabic():
    assert "،" in m.replace_english_commas("بحر هادئ, رياح خفيفة")


def test_enforce_line_breaks_and_spacing_are_idempotent():
    text = "🎯 0. الملخص\nنسبة النجاح 62%\n⚖️ 4. ميزان العوامل\nبحر هادئ"
    once = m.add_paragraph_spacing(m.enforce_line_breaks(text))
    twice = m.add_paragraph_spacing(m.enforce_line_breaks(once))
    assert once == twice
    assert once.count("\n\n") >= 1


def test_full_cleanup_pipeline_on_sample_report():
    report = (
        "🎯 0. الملخص التنفيذي ليوم 07/09/2026\n"
        "> نسبة النجاح: 62%\n"
        "🕒 3. التفكيك الديناميكي الزمني\n"
        " * السحر (00:00\n"
        " * 03:00): بحر هادئ, رياح خفيفة\n"
        "⚖️ 4. ميزان العوامل\n"
    )
    out = m.clean_report_text(m.fix_broken_number_lines(m.fix_broken_time_in_headers(report)))
    out = m.add_paragraph_spacing(m.enforce_line_breaks(m.replace_english_commas(out)))
    assert "00:00" in out and "الملخص التنفيذي" in out
    assert out == out.strip()
