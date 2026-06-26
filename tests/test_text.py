"""Tests for the text normalization module."""

from wiki_tts.text import clean_spoken_text, init_nemo


def test_removes_citation_brackets():
    result = clean_spoken_text("Earth is the third planet[1].")
    assert result == "Earth is the third planet."


def test_removes_phonetic_guides():
    result = clean_spoken_text("Earth (/ˈɜːrθ/) is a planet.")
    assert result == "Earth is a planet."


def test_normalizes_percent():
    result = clean_spoken_text("Covering 70.8% of Earth.")
    assert "seventy point eight percent" in result


def test_normalizes_integer():
    result = clean_spoken_text("42 is the answer.")
    assert "forty-two" in result


# ── Unit expansion tests ──────────────────────────────────────────────────


def test_unit_expansion_metric():
    result = clean_spoken_text("The mountain is 2060 m tall.")
    assert "meters" in result


def test_unit_expansion_imperial():
    result = clean_spoken_text("The canyon is 6758 ft deep.")
    assert "feet" in result


def test_unit_expansion_speed():
    result = clean_spoken_text("Winds reached 120 km/h.")
    assert "kilometers per hour" in result


def test_unit_expansion_temperature():
    result = clean_spoken_text("Water boils at 100 °C.")
    assert "degrees Celsius" in result


def test_unit_expansion_decimal():
    result = clean_spoken_text("The trail is 3.5 km long.")
    assert "kilometers" in result


def test_unit_expansion_no_false_positive():
    """Standalone 'm' without a preceding number must not be expanded."""
    result = clean_spoken_text("The m in theorem is silent.")
    assert "meters" not in result


def test_unit_expansion_mm_before_m():
    """'mm' must not be partially matched by the 'm' rule."""
    result = clean_spoken_text("The pipe is 50 mm wide.")
    assert "millimeters" in result


def test_unit_expansion_kmh_before_km():
    """'km/h' must not be partially matched by the 'km' rule."""
    result = clean_spoken_text("Speed limit is 80 km/h.")
    assert "kilometers per hour" in result


def test_unit_expansion_no_digit_no_match():
    """Unit abbreviation without preceding number must not expand."""
    result = clean_spoken_text("The length in ft was measured.")
    assert "feet" not in result


def test_numbers_still_work():
    """Existing number normalization must not break."""
    result = clean_spoken_text("The result is 42.")
    assert "forty-two" in result


def test_percent_still_works():
    """Existing percent normalization must not break."""
    result = clean_spoken_text("Success rate is 99%.")
    assert "percent" in result


# ── NeMo-specific tests (require nemo_text_processing) ─────────────────────


def test_nemo_date_normalization():
    init_nemo()
    result = clean_spoken_text("The date is 2025-01-15.")
    assert "january" in result or "January" in result


def test_nemo_currency_normalization():
    init_nemo()
    result = clean_spoken_text("It costs $99.99.")
    assert "dollars" in result


def test_nemo_ordinal_normalization():
    init_nemo()
    result = clean_spoken_text("He came in 1st place.")
    assert "first" in result


def test_nemo_abbreviation_context():
    init_nemo()
    result = clean_spoken_text("123 Main St.")
    assert "Street" in result or "street" in result


def test_nemo_whitelist_protection():
    init_nemo()
    result = clean_spoken_text("NASA launched a mission.")
    assert "NASA" in result or "Nasa" in result


def test_nemo_measurement_singular_plural():
    init_nemo()
    result = clean_spoken_text("It weighs 1 kg.")
    assert "kilogram" in result


# ── Edge-case tests (work without NeMo) ───────────────────────────────────


def test_strips_html_tags():
    result = clean_spoken_text("CO<sub>2</sub> concentration.")
    assert "CO" in result and "concentration" in result
    assert "<sub>" not in result and "</sub>" not in result


def test_en_dash_to_word():
    result = clean_spoken_text("100–900 million years.")
    assert "to" in result


def test_en_dash_no_number_no_match():
    """En-dash without surrounding numbers must not be replaced."""
    result = clean_spoken_text("New York–London flight.")
    assert "to" not in result


def test_compound_unit_m_s():
    result = clean_spoken_text("Wind speed is 10 m/s.")
    assert "meters per second" in result or "metres per second" in result


def test_compound_unit_m_s2_with_superscript():
    result = clean_spoken_text("Acceleration is 9.8 m/s².")
    assert "meters per second" in result or "metres per second" in result
    assert "squared" in result


def test_large_number_with_unit():
    """2 million km must expand the unit even without a digit right before 'km'."""
    result = clean_spoken_text("The distance is 2 million km.")
    assert "kilometers" in result or "kilometres" in result


def test_handles_empty():
    assert clean_spoken_text("") == ""
    assert clean_spoken_text(None) == ""


def test_strips_extra_whitespace():
    result = clean_spoken_text("Hello    world.\n\nNext paragraph.")
    assert result == "Hello world. Next paragraph."
