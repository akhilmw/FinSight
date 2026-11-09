from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class FilingCatalogItem:
    id: int
    company_name: str
    form_type: str
    filing_date: date
    report_period: date | None
    source_url: str

    @property
    def label(self) -> str:
        period = self.report_period or self.filing_date
        if self.form_type == "10-Q":
            return (
                f"10-Q · period ended {period:%b %d, %Y} · "
                f"filed {self.filing_date:%b %d, %Y}"
            )
        return (
            f"{self.form_type} · FY {period.year} · "
            f"filed {self.filing_date:%b %d, %Y}"
        )


def company_names(filings: list[FilingCatalogItem]) -> list[str]:
    return sorted({filing.company_name for filing in filings})


def filings_for_company(
    filings: list[FilingCatalogItem],
    company_name: str,
) -> list[FilingCatalogItem]:
    return [
        filing
        for filing in filings
        if filing.company_name == company_name
    ]


def excerpt(text: str, limit: int = 700) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"
