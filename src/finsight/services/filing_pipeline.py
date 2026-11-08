from dataclasses import dataclass

from finsight.services.filing_chunking import FilingChunkingService
from finsight.services.filing_download import FilingDownloadService
from finsight.services.filing_embedding import FilingEmbeddingService
from finsight.services.filing_processing import FilingProcessingService


@dataclass(frozen=True)
class FilingPipelineResult:
    filing_id: int
    document_path: str
    sections_created: int
    chunks_created: int
    chunks_embedded: int


class FilingPipelineService:
    def __init__(
        self,
        download_service: FilingDownloadService,
        processing_service: FilingProcessingService,
        chunking_service: FilingChunkingService,
        embedding_service: FilingEmbeddingService,
    ) -> None:
        self._download_service = download_service
        self._processing_service = processing_service
        self._chunking_service = chunking_service
        self._embedding_service = embedding_service

    async def run(
        self,
        filing_id: int,
    ) -> FilingPipelineResult:
        download_result = await self._download_service.download(filing_id)

        processing_result = await self._processing_service.process(filing_id)

        chunking_result = self._chunking_service.chunk_filing(filing_id)

        embedding_result = await self._embedding_service.embed_filing(
            filing_id
        )

        return FilingPipelineResult(
            filing_id=filing_id,
            document_path=download_result.path,
            sections_created=processing_result.created,
            chunks_created=chunking_result.created,
            chunks_embedded=embedding_result.embedded,
        )