from dataclasses import dataclass
from datetime import date

from sqlalchemy.orm import Session

from finsight.embeddings import EmbeddingClient
from finsight.repositories.filing_chunk import (
    FilingChunkRepository,
)


@dataclass(frozen=True)
class FilingSearchMatch:
    chunk_id: int
    filing_id: int
    accession_number: str
    company_name: str
    form_type: str
    filing_date: date
    source_url: str
    item: str
    title: str
    text: str
    distance: float
    similarity: float


@dataclass(frozen=True)
class FilingSearchResult:
    query: str
    matches: list[FilingSearchMatch]


class FilingSearchService:
    def __init__(
        self,
        session: Session,
        embedding_client: EmbeddingClient,
    ) -> None:
        self._embedding_client = embedding_client
        self._chunk_repository = FilingChunkRepository(session)

    async def search(
        self,
        query: str,
        limit: int = 5,
        filing_id: int | None = None,
    ) -> FilingSearchResult:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("query must not be blank")

        if limit < 1:
            raise ValueError("limit must be at least 1")

        query_embeddings = await self._embedding_client.embed([normalized_query])
        query_embedding = query_embeddings[0]

        results = self._chunk_repository.search_similar(
            query_embedding=query_embedding,
            embedding_model=self._embedding_client.model,
            limit=limit,
            filing_id=filing_id,
        )

        matches = [
            FilingSearchMatch(
                chunk_id=result.chunk.id,
                filing_id=result.filing.id,
                accession_number=result.filing.accession_number,
                company_name=result.filing.company_name,
                form_type=result.filing.form_type,
                filing_date=result.filing.filing_date,
                source_url=result.filing.source_url,
                item=result.section.item,
                title=result.section.title,
                text=result.chunk.text,
                distance=result.distance,
                similarity=1.0 - result.distance,
            )
            for result in results
        ]

        return FilingSearchResult(
            query=normalized_query,
            matches=matches,
        )
