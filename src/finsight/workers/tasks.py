import asyncio

from openai import AsyncOpenAI

from finsight.chunking.section import SectionChunker
from finsight.clients.sec import SecClient
from finsight.core.config import get_settings
from finsight.db.session import SessionLocal
from finsight.embeddings import OpenAIEmbeddingClient
from finsight.parsers.sec_html import SecHtmlParser
from finsight.parsers.sec_sections import SecSectionDetector
from finsight.services.filing_chunking import FilingChunkingService
from finsight.services.filing_download import FilingDownloadService
from finsight.services.filing_embedding import FilingEmbeddingService
from finsight.services.filing_pipeline import FilingPipelineService
from finsight.services.filing_processing import FilingProcessingService
from finsight.storage.local import LocalDocumentStore
from finsight.workers.celery_app import celery_app


@celery_app.task(name="finsight.health_check")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@celery_app.task(
    name="finsight.process_filing",
    acks_late=True,
    reject_on_worker_lost=True,
)
def process_filing_task(
    filing_id: int,
) -> dict[str, int | str]:
    return asyncio.run(run_filing_pipeline(filing_id))


async def run_filing_pipeline(
    filing_id: int,
) -> dict[str, int | str]:
    settings = get_settings()

    if settings.openai_api_key is None:
        raise RuntimeError("The embedding service is not configured")

    with SessionLocal() as session:
        async with (
            SecClient() as sec_client,
            AsyncOpenAI(
                api_key=settings.openai_api_key.get_secret_value()
            ) as sdk_client,
        ):
            embedding_client = OpenAIEmbeddingClient(
                client=sdk_client,
                model=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
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
                    embedding_client=embedding_client,
                    batch_size=settings.embedding_batch_size,
                ),
            )

            result = await pipeline.run(filing_id)

    return {
        "filing_id": result.filing_id,
        "document_path": result.document_path,
        "sections_created": result.sections_created,
        "chunks_created": result.chunks_created,
        "chunks_embedded": result.chunks_embedded,
    }
