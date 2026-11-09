from datetime import date

from finsight.ui.catalog import (
    FilingCatalogItem,
    company_names,
    excerpt,
    filings_for_company,
)


def filing(
    filing_id: int,
    company_name: str,
    filing_date: date,
    report_period: date | None = None,
) -> FilingCatalogItem:
    return FilingCatalogItem(
        id=filing_id,
        company_name=company_name,
        form_type="10-K",
        filing_date=filing_date,
        report_period=report_period,
        source_url="https://www.sec.gov/example",
    )


def test_catalog_groups_companies_and_filings() -> None:
    filings = [
        filing(1, "NVIDIA CORP", date(2026, 2, 25)),
        filing(2, "APPLE INC", date(2025, 10, 31)),
        filing(3, "NVIDIA CORP", date(2025, 2, 26)),
    ]

    assert company_names(filings) == ["APPLE INC", "NVIDIA CORP"]
    assert [item.id for item in filings_for_company(filings, "NVIDIA CORP")] == [1, 3]


def test_filing_label_uses_report_period_year() -> None:
    item = filing(
        1,
        "NVIDIA CORP",
        date(2026, 2, 25),
        report_period=date(2026, 1, 25),
    )

    assert item.label == "10-K · FY 2026 · filed Feb 25, 2026"


def test_quarterly_filing_label_shows_report_period() -> None:
    item = FilingCatalogItem(
        id=2,
        company_name="NVIDIA CORP",
        form_type="10-Q",
        filing_date=date(2026, 8, 26),
        report_period=date(2026, 7, 26),
        source_url="https://www.sec.gov/example",
    )

    assert item.label == (
        "10-Q · period ended Jul 26, 2026 · filed Aug 26, 2026"
    )


def test_excerpt_normalizes_whitespace_and_truncates() -> None:
    assert excerpt("one\n\n two   three", limit=12) == "one two thr…"
