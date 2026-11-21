from gemma_ocr.verify import checks, edit_distance, format_warnings, norm, numbers, snap_to_layer, words


def test_norm_digits_letters_presentation_forms():
    assert norm("۱۴۰۳") == "1403"
    assert norm("١٢٣") == "123"
    assert norm("كيك") == "کیک"
    assert norm("ﭘﺎﯾﺎﻧﻪ") == "پایانه"          # Arabic presentation forms from PDF text layers
    assert norm("a‌b‏") == "ab"


def test_numbers_dates_and_separators():
    t = "مبلغ ۵۶,۳۹۲,۵۲۰ تومان تاریخ ۱۴۰۳/۱۰/۲۳ کد 123 و 4567"
    assert numbers(t) == ["56392520", "1403/10/23", "4567"]   # 3-digit numbers ignored


def test_words_ignores_html_tags_but_not_delimiters():
    assert words("<b>سلام</b> دنیا") == ["سلام", "دنیا"]
    assert words("<<<\nسلام\n>>>") == ["سلام"]


def test_edit_distance():
    assert edit_distance("1403", "1400") == 1
    assert edit_distance("", "12") == 2


def test_snap_number_one_digit_off_keeps_script():
    layer = set(numbers("شماره 140368100071539323"))
    text, fixes = snap_to_layer("شماره ۱۴۰۳۶۸۱۰۰۰۷۱۵۳۹۳۲۸", layer)
    assert text == "شماره ۱۴۰۳۶۸۱۰۰۰۷۱۵۳۹۳۲۳"
    assert fixes == [{"ocr": "140368100071539328", "text_layer": "140368100071539323"}]


def test_snap_keeps_thousands_separators():
    text, _ = snap_to_layer("640,001 ریال", set(numbers("640000")))
    assert text == "640,000 ریال"


def test_snap_dates():
    layer = set(numbers("1403/10/09 و 1403/7/24"))
    text, fixes = snap_to_layer("1400/10/09 و ۱۴۰۰/۷/۲۴ و 1403/10/09", layer)
    assert text == "1403/10/09 و ۱۴۰۳/۷/۲۴ و 1403/10/09"
    assert len(fixes) == 2


def test_snap_leaves_ambiguous_and_far_numbers():
    layer = set(numbers("12345 12346"))
    assert snap_to_layer("12347", layer)[0] == "12347"         # two candidates at distance 1
    assert snap_to_layer("99999", set(numbers("12345")))[0] == "99999"


def test_snap_does_not_touch_date_parts():
    text, fixes = snap_to_layer("1402/10/23", set(numbers("1403")))
    assert text == "1402/10/23" and not fixes


def test_format_warnings():
    assert format_warnings("موبایل ۰۹۱۷۹۸۸۴۱۵") == ["mobile with 10 digits: 0917988415"]
    assert format_warnings("موبایل 09179884150") == []
    assert format_warnings("کد ملی: 123456789") == ["national ID with 9 digits: 123456789"]


def test_checks_cross_read_and_layer():
    c = checks("شماره 1234567 و 7654321", "شماره 1234567", "شماره 1234567 و 7654321 متن کافی برای لایه متنی")
    assert c["unconfirmed_by_cross_read"] == ["7654321"]
    assert c["text_layer"]["layer_numbers_found"] == 2
    assert checks("1234567", None, "")["unconfirmed_by_cross_read"] == []
