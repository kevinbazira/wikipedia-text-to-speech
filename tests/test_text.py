"""Tests for the text normalization module."""

from wiki_tts.text import clean_spoken_text


def test_removes_citation_brackets():
    result = clean_spoken_text("Earth is the third planet[1].")
    assert result == "Earth is the third planet."


def test_removes_phonetic_guides():
    result = clean_spoken_text("Earth (English: /ˈɜːrθ/) is a planet.")
    assert result == "Earth is a planet."


def test_normalizes_percent():
    result = clean_spoken_text("Covering 70.8% of Earth.")
    assert "seventy point eight percent" in result


def test_normalizes_integer():
    result = clean_spoken_text("42 is the answer.")
    assert "forty-two" in result


def test_handles_empty():
    assert clean_spoken_text("") == ""
    assert clean_spoken_text(None) == ""


def test_strips_extra_whitespace():
    result = clean_spoken_text("Hello    world.\n\nNext paragraph.")
    assert result == "Hello world. Next paragraph."
