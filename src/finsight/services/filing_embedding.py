from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import batched

from sqlalchemy.orm import Session

from finsight.embeddings import EmbeddingClient
from finsight.repositories.filing import FilingRepository
from finsight.repositories.filing_chunk import FilingChunkRepository


class FilingNotFoundError(Exception):
    pass


class FilingHasNoChunksError(Exception):
    pass


@dataclass(frozen=True)
class FilingEmbeddingResult:
    filing_id: int
    total_chunks: int
    embedded: int
    skipped: int
    batches: int


class FilingEmbeddingService:
    def __init__(
        self,
        session: Session,
        embedding_client: EmbeddingClient,
        batch_size: int = 64,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self._session = session
        self._embedding_client = embedding_client
        self._batch_size = batch_size
        self._filing_repository = FilingRepository(session)
        self._chunk_repository = FilingChunkRepository(session)

    async def embed_filing(
        self,
        filing_id: int,
    ) -> FilingEmbeddingResult:
        filing = self._filing_repository.get_by_id(filing_id)

        if filing is None:
            raise FilingNotFoundError(f"Filing {filing_id} was not found")

        all_chunks = self._chunk_repository.list_by_filing_id(filing_id)

        if not all_chunks:
            raise FilingHasNoChunksError(f"Filing {filing_id} has no stored chunks")

        pending_chunks = self._chunk_repository.list_requiring_embedding(
            filing_id=filing_id,
            embedding_model=self._embedding_client.model,
        )

        embedded = 0
        batch_count = 0

        try:
            for chunk_batch in batched(
                pending_chunks,
                self._batch_size,
            ):
                embeddings = await self._embedding_client.embed(
                    [chunk.text for chunk in chunk_batch]
                )
                embedded_at = datetime.now(UTC)

                for chunk, embedding in zip(
                    chunk_batch,
                    embeddings,
                    strict=True,
                ):
                    chunk.embedding = embedding
                    chunk.embedding_model = self._embedding_client.model
                    chunk.embedded_at = embedded_at

                # Commit each completed batch so a failed retry can resume.
                self._session.commit()

                embedded += len(chunk_batch)
                batch_count += 1

            filing.status = "embedded"
            self._session.commit()
        except Exception:
            self._session.rollback()
            raise

        return FilingEmbeddingResult(
            filing_id=filing_id,
            total_chunks=len(all_chunks),
            embedded=embedded,
            skipped=len(all_chunks) - len(pending_chunks),
            batches=batch_count,
        )
