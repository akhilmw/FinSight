import asyncio
from dataclasses import dataclass

from openai import AsyncOpenAI

from finsight.chunking.section import SectionChunker
from finsight.clients.sec import SecClient
from finsight.core.config import get_settings
from finsight.db.session import SessionLocal
from finsight.embeddings import OpenAIEmbeddingClient
from finsight.parsers.sec_html import SecHtmlParser
from finsight.parsers.sec_sections import SecSectionDetector
from finsight.repositories.filing import FilingRepository
from finsight.services.filing_chunking import FilingChunkingService
from finsight.services.filing_download import FilingDownloadService
from finsight.services.filing_embedding import FilingEmbeddingService
from finsight.services.filing_ingestion import FilingIngestionService
from finsight.services.filing_pipeline import FilingPipelineService
from finsight.services.filing_processing import FilingProcessingService
from finsight.storage.local import LocalDocumentStore

PORTFOLIO_TICKERS = (
    "NVDA",
    "MSFT",
    "AAPL",
    "AMZN",
    "GOOGL",
    "META",
    "NFLX",
    "AMD",
    "PLTR",
    "CSCO",
)
PORTFOLIO_FORM_TYPES = ("10-K", "10-Q")


@dataclass(frozen=True)
class PortfolioIndexResult:
    ticker: str
    form_type: str
    filing_id: int | None
    status: str
    chunks_embedded: int = 0
    error: str | None = None


async def main() -> None:
    settings = get_settings()
    if settings.openai_api_key is None:
        raise RuntimeError("FINSIGHT_OPENAI_API_KEY is not configured")

    results: list[PortfolioIndexResult] = []
    async with (
        SecClient() as sec_client,
        AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
        ) as sdk_client,
    ):
        for ticker in PORTFOLIO_TICKERS:
            for form_type in PORTFOLIO_FORM_TYPES:
                print(
                    f"[{ticker} {form_type}] discovering latest filing",
                    flush=True,
                )
                try:
                    result = await _index_filing(
                        ticker=ticker,
                        form_type=form_type,
                        sec_client=sec_client,
                        sdk_client=sdk_client,
                    )
                except Exception as exc:
                    result = PortfolioIndexResult(
                        ticker=ticker,
                        form_type=form_type,
                        filing_id=None,
                        status="failed",
                        error=str(exc),
                    )
                results.append(result)
                print(
                    f"[{ticker} {form_type}] {result.status}",
                    flush=True,
                )

    print("\nPortfolio indexing summary", flush=True)
    for result in results:
        detail = f"filing_id={result.filing_id}"
        if result.chunks_embedded:
            detail += f", embedded={result.chunks_embedded}"
        if result.error:
            detail += f", error={result.error}"
        print(
            f"- {result.ticker} {result.form_type}: "
            f"{result.status} ({detail})",
            flush=True,
        )

    failures = [result for result in results if result.status == "failed"]
    if failures:
        raise RuntimeError(f"{len(failures)} portfolio filings failed to index")


async def _index_filing(
    *,
    ticker: str,
    form_type: str,
    sec_client: SecClient,
    sdk_client: AsyncOpenAI,
) -> PortfolioIndexResult:
    settings = get_settings()

    with SessionLocal() as session:
        ingestion = await FilingIngestionService(
            session=session,
            sec_client=sec_client,
        ).ingest(
            ticker=ticker,
            form_types=(form_type,),
            limit=1,
        )
        if not ingestion.filing_ids:
            return PortfolioIndexResult(
                ticker=ticker,
                form_type=form_type,
                filing_id=None,
                status=f"no {form_type} found",
            )

        filing_id = ingestion.filing_ids[0]
        filing = FilingRepository(session).get_by_id(filing_id)
        if filing is None:
            raise RuntimeError(f"Filing {filing_id} was not stored")
        if filing.status == "embedded":
            return PortfolioIndexResult(
                ticker=ticker,
                form_type=form_type,
                filing_id=filing_id,
                status="already embedded",
            )

        print(
            f"[{ticker} {form_type}] processing filing {filing_id} "
            f"filed {filing.filing_date}",
            flush=True,
        )
        pipeline = FilingPipelineService(
            download_service=FilingDownloadService(
                session=session,
                sec_client=sec_client,
                document_store=LocalDocumentStore(
                    settings.raw_document_directory
                ),
            ),
            processing_service=FilingProcessingService(
                session=session,
                html_parser=SecHtmlParser(),
                section_detector=SecSectionDetector(),
            ),
            chunking_service=FilingChunkingService(
                session=session,
                chunker=SectionChunker(
                    max_tokens=600,
                    overlap_tokens=100,
                ),
            ),
            embedding_service=FilingEmbeddingService(
                session=session,
                embedding_client=OpenAIEmbeddingClient(
                    client=sdk_client,
                    model=settings.embedding_model,
                    dimensions=settings.embedding_dimensions,
                ),
                batch_size=settings.embedding_batch_size,
            ),
        )
        pipeline_result = await pipeline.run(filing_id)

    return PortfolioIndexResult(
        ticker=ticker,
        form_type=form_type,
        filing_id=filing_id,
        status="embedded",
        chunks_embedded=pipeline_result.chunks_embedded,
    )


if __name__ == "__main__":
    asyncio.run(main())
