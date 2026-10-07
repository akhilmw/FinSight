import asyncio

from finsight.clients.sec import SecClient


async def main() -> None:
    async with SecClient() as client:
        company = await client.get_company("NVDA")
        filings = await client.get_recent_filings(
            company,
            form_types=("10-K", "10-Q"),
            limit=5,
        )

    print(company)

    for filing in filings:
        print(
            filing.form_type,
            filing.filing_date,
            filing.report_period,
            filing.source_url,
        )


if __name__ == "__main__":
    asyncio.run(main())