import asyncio
from unittest.mock import AsyncMock, MagicMock, call

from finsight.services.filing_chunking import FilingChunkingResult
from finsight.services.filing_download import FilingDownloadResult
from finsight.services.filing_embedding import FilingEmbeddingResult
from finsight.services.filing_pipeline import FilingPipelineService
from finsight.services.filing_processing import FilingProcessingResult


def test_pipeline_runs_all_stages_in_order_and_returns_summary() -> None:
    download_service = MagicMock()
    download_service.download = AsyncMock(
        return_value=FilingDownloadResult(
            filing_id=3,
            path="data/raw/filing.html",
            sha256="abc123",
        )
    )
    processing_service = MagicMock()
    processing_service.process = AsyncMock(
        return_value=FilingProcessingResult(
            filing_id=3,
            created=12,
            replaced=0,
        )
    )
    chunking_service = MagicMock()
    chunking_service.chunk_filing.return_value = FilingChunkingResult(
        filing_id=3,
        sections_processed=12,
        created=80,
        replaced=0,
    )
    embedding_service = MagicMock()
    embedding_service.embed_filing = AsyncMock(
        return_value=FilingEmbeddingResult(
            filing_id=3,
            total_chunks=80,
            embedded=80,
            skipped=0,
            batches=2,
        )
    )

    calls = MagicMock()
    calls.attach_mock(download_service.download, "download")
    calls.attach_mock(processing_service.process, "process")
    calls.attach_mock(chunking_service.chunk_filing, "chunk")
    calls.attach_mock(embedding_service.embed_filing, "embed")

    service = FilingPipelineService(
        download_service=download_service,
        processing_service=processing_service,
        chunking_service=chunking_service,
        embedding_service=embedding_service,
    )

    result = asyncio.run(service.run(filing_id=3))

    assert calls.mock_calls == [
        call.download(3),
        call.process(3),
        call.chunk(3),
        call.embed(3),
    ]
    assert result.filing_id == 3
    assert result.document_path == "data/raw/filing.html"
    assert result.sections_created == 12
    assert result.chunks_created == 80
    assert result.chunks_embedded == 80
