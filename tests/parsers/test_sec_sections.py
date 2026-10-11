from finsight.parsers.sec_html import ParsedFiling
from finsight.parsers.sec_sections import (
    FilingSection,
    SecSectionDetector,
    SectionHeading,
)


def test_find_candidates_supports_inline_and_split_headings() -> None:
    filing = ParsedFiling(
        title="Example filing",
        text=(
            "Introduction\n"
            "Item 1. Financial Statements\n"
            "Financial content\n"
            "Item 1a.\n"
            "Risk Factors\n"
            "Risk content"
        ),
    )

    headings = SecSectionDetector().find_candidates(filing)

    assert headings == [
        SectionHeading(
            item="1",
            title="Financial Statements",
            start_line=1,
            heading_end_line=1,
        ),
        SectionHeading(
            item="1A",
            title="Risk Factors",
            start_line=3,
            heading_end_line=4,
        ),
    ]


def test_extract_sections_uses_last_duplicate_heading() -> None:
    filing = ParsedFiling(
        title="Example filing",
        text=(
            "Table of Contents\n"
            "Item 1. Financial Statements\n"
            "3\n"
            "Item 1A.\n"
            "Risk Factors\n"
            "12\n"
            "Filing body\n"
            "Item 1. Financial Statements\n"
            "Actual financial section text.\n"
            "Item 1A. Risk Factors\n"
            "Actual risk section text."
        ),
    )

    sections = SecSectionDetector().extract_sections(filing)

    assert sections == [
        FilingSection(
            item="1",
            title="Financial Statements",
            text="Actual financial section text.",
            start_line=7,
            end_line=9,
        ),
        FilingSection(
            item="1A",
            title="Risk Factors",
            text="Actual risk section text.",
            start_line=9,
            end_line=11,
        ),
    ]


def test_extract_sections_keeps_same_item_with_different_titles() -> None:
    filing = ParsedFiling(
        title="Example 10-Q",
        text=(
            "Item 1. Financial Statements\n"
            "Financial statement content.\n"
            "Item 1. Legal Proceedings\n"
            "Legal proceeding content."
        ),
    )

    sections = SecSectionDetector().extract_sections(filing)

    assert [(section.item, section.title) for section in sections] == [
        ("1", "Financial Statements"),
        ("1", "Legal Proceedings"),
    ]


def test_extract_sections_returns_empty_list_without_headings() -> None:
    filing = ParsedFiling(
        title=None,
        text="A filing document without recognizable item headings.",
    )

    assert SecSectionDetector().extract_sections(filing) == []
