import asyncio
from dataclasses import dataclass, replace
from datetime import date

from sqlalchemy.orm import Session

from finsight.embeddings import EmbeddingClient
from finsight.repositories.filing_chunk import (
    FilingChunkRepository,
)
from finsight.reranking import Reranker
from finsight.retrieval.fusion import reciprocal_rank_fusion
from finsight.schemas import SearchMode


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
    score: float
    retrieval_score: float | None = None
    rerank_score: float | None = None
    distance: float | None = None
    similarity: float | None = None


@dataclass(frozen=True)
class FilingSearchResult:
    query: str
    mode: SearchMode
    matches: list[FilingSearchMatch]


class FilingSearchService:
    def __init__(
        self,
        session: Session,
        embedding_client: EmbeddingClient,
        reranker: Reranker | None = None,
    ) -> None:
        self._embedding_client = embedding_client
        self._reranker = reranker
        self._chunk_repository = FilingChunkRepository(session)

    async def _rerank_matches(
        self,
        query: str,
        matches: list[FilingSearchMatch],
        limit: int,
    ) -> list[FilingSearchMatch]:
        if self._reranker is None:
            raise RuntimeError("reranker is not configured")

        scores = await asyncio.to_thread(
            self._reranker.rerank,
            query,
            [match.text for match in matches],
        )

        if len(scores) != len(matches):
            raise RuntimeError(
                "reranker returned a different number of scores than documents"
            )

        ranked = sorted(
            zip(matches, scores, strict=True),
            key=lambda result: (-result[1], result[0].chunk_id),
        )

        return [
            replace(
                match,
                score=rerank_score,
                retrieval_score=match.score,
                rerank_score=rerank_score,
            )
            for match, rerank_score in ranked[:limit]
        ]

    async def _search_hybrid(
        self,
        query: str,
        limit: int,
        filing_id: int | None,
        company_name: str | None,
        year: int | None,
        form_type: str | None,
        candidate_limit: int | None = None,
    ) -> list[FilingSearchMatch]:
        retrieval_limit = candidate_limit or max(limit * 4, 20)

        query_embeddings = await self._embedding_client.embed([query])
        query_embedding = query_embeddings[0]

        vector_results = self._chunk_repository.search_similar(
            query_embedding=query_embedding,
            embedding_model=self._embedding_client.model,
            limit=retrieval_limit,
            filing_id=filing_id,
            company_name=company_name,
            year=year,
            form_type=form_type,
        )
        keyword_results = self._chunk_repository.search_keyword(
            query=query,
            limit=retrieval_limit,
            filing_id=filing_id,
            company_name=company_name,
            year=year,
            form_type=form_type,
        )

        vector_by_id = {
            result.chunk.id: result
            for result in vector_results
        }
        keyword_by_id = {
            result.chunk.id: result
            for result in keyword_results
        }

        scores = reciprocal_rank_fusion(
            ranked_lists=[
                [result.chunk.id for result in vector_results],
                [result.chunk.id for result in keyword_results],
            ],
        )

        ranked_chunk_ids = sorted(
            scores,
            key=lambda chunk_id: (-scores[chunk_id], chunk_id),
        )[:limit]

        matches: list[FilingSearchMatch] = []

        for chunk_id in ranked_chunk_ids:
            vector_result = vector_by_id.get(chunk_id)
            result = vector_result or keyword_by_id[chunk_id]

            distance = (
                vector_result.distance
                if vector_result is not None
                else None
            )
            similarity = (
                1.0 - distance
                if distance is not None
                else None
            )

            matches.append(
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
                    score=scores[chunk_id],
                    retrieval_score=scores[chunk_id],
                    distance=distance,
                    similarity=similarity,
                )
            )

        return matches

    async def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.VECTOR,
        limit: int = 5,
        filing_id: int | None = None,
        company_name: str | None = None,
        year: int | None = None,
        form_type: str | None = None,
    ) -> FilingSearchResult:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("query must not be blank")

        if limit < 1:
            raise ValueError("limit must be at least 1")

        if mode is SearchMode.KEYWORD:
            keyword_results = self._chunk_repository.search_keyword(
                query=normalized_query,
                limit=limit,
                filing_id=filing_id,
                company_name=company_name,
                year=year,
                form_type=form_type,
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
                    score=result.rank,
                )
                for result in keyword_results
            ]
        elif mode is SearchMode.HYBRID_RERANK:
            candidate_limit = max(limit * 4, 20)
            candidates = await self._search_hybrid(
                query=normalized_query,
                limit=candidate_limit,
                filing_id=filing_id,
                company_name=company_name,
                year=year,
                form_type=form_type,
                candidate_limit=candidate_limit,
            )
            matches = await self._rerank_matches(
                query=normalized_query,
                matches=candidates,
                limit=limit,
            )
        elif mode is SearchMode.HYBRID:
            matches = await self._search_hybrid(
                query=normalized_query,
                limit=limit,
                filing_id=filing_id,
                company_name=company_name,
                year=year,
                form_type=form_type,
            )
        else:
            query_embeddings = await self._embedding_client.embed([normalized_query])
            query_embedding = query_embeddings[0]

            vector_results = self._chunk_repository.search_similar(
                query_embedding=query_embedding,
                embedding_model=self._embedding_client.model,
                limit=limit,
                filing_id=filing_id,
                company_name=company_name,
                year=year,
                form_type=form_type,
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
                    score=1.0 - result.distance,
                    distance=result.distance,
                    similarity=1.0 - result.distance,
                )
                for result in vector_results
            ]

        return FilingSearchResult(
            query=normalized_query,
            mode=mode,
            matches=matches,
        )
