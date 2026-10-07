import asyncio
from datetime import date

import httpx
import pytest

from finsight.clients.sec import (
    CompanyNotFoundError,
    SecClient,
    SecCompany,
    SecFiling,
)


def test_get_company_normalizes_ticker_and_cik() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/files/company_tickers.json"

        return httpx.Response(
            status_code=200,
            json={
                "0": {
                    "cik_str": 1045810,
                    "ticker": "NVDA",
                    "title": "NVIDIA CORP",
                }
            },
        )

    async def run_test() -> None:
        transport = httpx.MockTransport(handler)

        async with SecClient(transport=transport) as client:
            company = await client.get_company(" nvda ")

        assert company == SecCompany(
            ticker="NVDA",
            cik="0001045810",
            name="NVIDIA CORP",
        )

    asyncio.run(run_test())


def test_get_company_raises_for_unknown_ticker() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, json={})

    async def run_test() -> None:
        transport = httpx.MockTransport(handler)

        async with SecClient(transport=transport) as client:
            with pytest.raises(
                CompanyNotFoundError,
                match="NOTREAL",
            ):
                await client.get_company("NOTREAL")

    asyncio.run(run_test())


def test_get_recent_filings_filters_normalizes_and_limits_results() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/submissions/CIK0001045810.json"

        return httpx.Response(
            status_code=200,
            json={
                "filings": {
                    "recent": {
                        "accessionNumber": [
                            "0001045810-26-000030",
                            "0001045810-26-000021",
                            "0001045810-25-000099",
                        ],
                        "form": ["8-K", "10-K", "10-Q"],
                        "filingDate": [
                            "2026-03-01",
                            "2026-02-25",
                            "2025-11-19",
                        ],
                        "reportDate": ["", "2026-01-25", ""],
                        "primaryDocument": [
                            "nvda-8k.htm",
                            "nvda-20260125.htm",
                            "nvda-20251026.htm",
                        ],
                    }
                }
            },
        )

    async def run_test() -> None:
        company = SecCompany(
            ticker="NVDA",
            cik="0001045810",
            name="NVIDIA CORP",
        )
        transport = httpx.MockTransport(handler)

        async with SecClient(transport=transport) as client:
            filings = await client.get_recent_filings(company, limit=2)

        assert filings == [
            SecFiling(
                accession_number="0001045810-26-000021",
                cik="0001045810",
                company_name="NVIDIA CORP",
                form_type="10-K",
                filing_date=date(2026, 2, 25),
                report_period=date(2026, 1, 25),
                primary_document="nvda-20260125.htm",
                source_url=(
                    "https://www.sec.gov/Archives/edgar/data/1045810/"
                    "000104581026000021/nvda-20260125.htm"
                ),
            ),
            SecFiling(
                accession_number="0001045810-25-000099",
                cik="0001045810",
                company_name="NVIDIA CORP",
                form_type="10-Q",
                filing_date=date(2025, 11, 19),
                report_period=None,
                primary_document="nvda-20251026.htm",
                source_url=(
                    "https://www.sec.gov/Archives/edgar/data/1045810/"
                    "000104581025000099/nvda-20251026.htm"
                ),
            ),
        ]

    asyncio.run(run_test())
