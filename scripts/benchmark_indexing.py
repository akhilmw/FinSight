import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from openai import AsyncOpenAI

from finsight.chunking.section import SectionChunker
from finsight.core.config import get_settings
from finsight.db.models import Filing
from finsight.db.session import SessionLocal
from finsight.embeddings import OpenAIEmbeddingClient
from finsight.parsers.sec_html import SecHtmlParser
from finsight.parsers.sec_sections import SecSectionDetector
from finsight.services.filing_chunking import FilingChunkingService
from finsight.services.filing_embedding import FilingEmbeddingService
from finsight.services.filing_processing import FilingProcessingService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark FinSight filing indexing.",
    )
    parser.add_argument(
        "filing_id",
        type=int,
        help="Downloaded source filing to benchmark.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evals/results/indexing_benchmark.json"),
        help="Path where the benchmark result will be written.",
    )
    return parser.parse_args()


def clone_filing(source: Filing) -> Filing:
    return Filing(
        accession_number=f"benchmark-{uuid4().hex[:20]}",
        cik=source.cik,
        company_name=source.company_name,
        form_type=source.form_type,
        filing_date=source.filing_date,
        report_period=source.report_period,
        source_url=source.source_url,
        status="downloaded",
        raw_document_path=source.raw_document_path,
        content_sha256=source.content_sha256,
        downloaded_at=source.downloaded_at,
    )


async def benchmark_indexing(
    source_filing_id: int,
    output_path: Path,
) -> None:
    settings = get_settings()

    if settings.openai_api_key is None:
        raise RuntimeError("The embedding service is not configured")

    with SessionLocal() as session:
        source_filing = session.get(Filing, source_filing_id)

        if source_filing is None:
            raise ValueError(f"Filing {source_filing_id} was not found")

        if source_filing.raw_document_path is None:
            raise ValueError(
                f"Filing {source_filing_id} has not been downloaded"
            )

        benchmark_filing = clone_filing(source_filing)
        session.add(benchmark_filing)
        session.commit()
        session.refresh(benchmark_filing)
        benchmark_filing_id = benchmark_filing.id

        try:
            async with AsyncOpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
            ) as sdk_client:
                embedding_client = OpenAIEmbeddingClient(
                    client=sdk_client,
                    model=settings.embedding_model,
                    dimensions=settings.embedding_dimensions,
                )

                processing_service = FilingProcessingService(
                    session=session,
                    html_parser=SecHtmlParser(),
                    section_detector=SecSectionDetector(),
                )
                chunking_service = FilingChunkingService(
                    session=session,
                    chunker=SectionChunker(
                        max_tokens=600,
                        overlap_tokens=100,
                    ),
                )
                embedding_service = FilingEmbeddingService(
                    session=session,
                    embedding_client=embedding_client,
                    batch_size=settings.embedding_batch_size,
                )

                total_started_at = perf_counter()

                processing_started_at = perf_counter()
                processing_result = await processing_service.process(
                    benchmark_filing_id
                )
                processing_ms = (
                    perf_counter() - processing_started_at
                ) * 1000

                chunking_started_at = perf_counter()
                chunking_result = chunking_service.chunk_filing(
                    benchmark_filing_id
                )
                chunking_ms = (
                    perf_counter() - chunking_started_at
                ) * 1000

                embedding_started_at = perf_counter()
                embedding_result = await embedding_service.embed_filing(
                    benchmark_filing_id
                )
                embedding_ms = (
                    perf_counter() - embedding_started_at
                ) * 1000

                total_ms = (
                    perf_counter() - total_started_at
                ) * 1000

            result = {
                "generated_at": datetime.now(UTC).isoformat(),
                "source_filing_id": source_filing_id,
                "embedding_model": settings.embedding_model,
                "embedding_dimensions": settings.embedding_dimensions,
                "embedding_batch_size": settings.embedding_batch_size,
                "sections": processing_result.created,
                "chunks": chunking_result.created,
                "embedded_chunks": embedding_result.embedded,
                "embedding_batches": embedding_result.batches,
                "processing_ms": processing_ms,
                "chunking_ms": chunking_ms,
                "embedding_and_storage_ms": embedding_ms,
                "total_indexing_ms": total_ms,
                "chunks_per_second": (
                    chunking_result.created / (total_ms / 1000)
                ),
            }

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2) + "\n")

            print(json.dumps(result, indent=2))
            print()
            print(f"Benchmark result: {output_path}")
        finally:
            temporary_filing = session.get(Filing, benchmark_filing_id)

            if temporary_filing is not None:
                session.delete(temporary_filing)
                session.commit()


def main() -> None:
    args = parse_args()
    asyncio.run(
        benchmark_indexing(
            source_filing_id=args.filing_id,
            output_path=args.output,
        )
    )


if __name__ == "__main__":
    main()
