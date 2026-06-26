"""Unit tests for pure Wikipedia section traversal helpers."""

from dataclasses import dataclass, field

from wiki_tts.wikipedia_utils import find_section_by_title, get_valid_sections


@dataclass
class DummySection:
    title: str
    text: str = ""
    sections: list["DummySection"] = field(default_factory=list)


def test_get_valid_sections_recurses_and_skips_blocklisted_branches():
    sections = [
        DummySection("History", sections=[DummySection("Early life")]),
        DummySection("References", sections=[DummySection("Should be skipped")]),
        DummySection("Culture", sections=[DummySection("Notes", sections=[DummySection("Also skipped")])]),
    ]

    result = get_valid_sections(sections)

    assert [section.title for section in result] == ["History", "Early life", "Culture"]


def test_get_valid_sections_blocklist_is_case_insensitive():
    sections = [DummySection("SEE ALSO"), DummySection("Further Reading"), DummySection("Geography")]

    result = get_valid_sections(sections)

    assert [section.title for section in result] == ["Geography"]


def test_find_section_by_title_finds_nested_section():
    target = DummySection("Target")
    sections = [DummySection("Parent", sections=[DummySection("Sibling"), target])]

    assert find_section_by_title(sections, "Target") is target


def test_find_section_by_title_returns_none_for_missing_section():
    sections = [DummySection("Parent", sections=[DummySection("Child")])]

    assert find_section_by_title(sections, "Missing") is None
