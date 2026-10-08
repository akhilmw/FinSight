import asyncio

from finsight.clients.sec import SecClient
from finsight.db.session import SessionLocal
from finsight.services.filing_ingestion import FilingIngestionService


async def main() -> None:
    with SessionLocal() as session:
        async with SecClient() as sec_client:
            service = FilingIngestionService(
                session=session,
                sec_client=sec_client,
            )

            result = await service.ingest(
                ticker="NVDA",
                limit=5,
            )

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
