from dataclasses import dataclass
from datetime import date
from types import TracebackType
from typing import Self

import httpx

from finsight.core.config import get_settings


@dataclass(frozen=True)
class SecCompany:
    ticker: str
    cik: str
    name: str


@dataclass(frozen=True)
class SecFiling:
    accession_number: str
    cik: str
    company_name: str
    form_type: str
    filing_date: date
    report_period: date | None
    primary_document: str
    source_url: str


class CompanyNotFoundError(Exception):
    pass


class SecClient:
    TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        settings = get_settings()
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": settings.sec_user_agent,
                "Accept": "application/json",
            },
            timeout=settings.sec_request_timeout_seconds,
            follow_redirects=True,
            transport=transport,
        )

    async def get_company(self, ticker: str) -> SecCompany:
        normalized_ticker = ticker.strip().upper()
        if not normalized_ticker:
            raise ValueError("Ticker must not be empty")

        response = await self._client.get(self.TICKER_URL)
        response.raise_for_status()
        data = response.json()

        for company in data.values():
            if company["ticker"].upper() == normalized_ticker:
                return SecCompany(
                    ticker=normalized_ticker,
                    cik=str(company["cik_str"]).zfill(10),
                    name=str(company["title"]),
                )

        raise CompanyNotFoundError(f"SEC company not found for ticker: {normalized_ticker}")

    async def get_recent_filings(
        self,
        company: SecCompany,
        form_types: tuple[str, ...] = ("10-K", "10-Q"),
        limit: int = 10,
    ) -> list[SecFiling]:
        if limit < 1:
            raise ValueError("Limit must be at least 1")

        url = self.SUBMISSIONS_URL.format(cik=company.cik)
        response = await self._client.get(url)
        response.raise_for_status()
        data = response.json()

        recent = data["filings"]["recent"]

        rows = zip(
            recent["accessionNumber"],
            recent["form"],
            recent["filingDate"],
            recent["reportDate"],
            recent["primaryDocument"],
            strict=True,
        )

        archive_cik = str(int(company.cik))
        result_sec_filings: list[SecFiling] = []

        for (
            accession_number,
            form_type,
            filing_date,
            report_date,
            primary_document,
        ) in rows:
            if form_type not in form_types:
                continue

            accession_path = accession_number.replace("-", "")
            source_url = (
                f"{self.ARCHIVES_BASE_URL}/{archive_cik}/{accession_path}/{primary_document}"
            )

            sec_filing = SecFiling(
                accession_number=accession_number,
                cik=company.cik,
                company_name=company.name,
                form_type=form_type,
                filing_date=date.fromisoformat(filing_date),
                report_period=(date.fromisoformat(report_date) if report_date else None),
                primary_document=primary_document,
                source_url=source_url,
            )

            result_sec_filings.append(sec_filing)

            if len(result_sec_filings) == limit:
                break

        return result_sec_filings

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self._client.aclose()
