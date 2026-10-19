from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from finsight.evaluation.dataset import RetrievalEvalCase
from finsight.evaluation.metrics import recall_at_k, reciprocal_rank
from finsight.schemas import SearchMode
from finsight.services.filing_search import FilingSearchResult


class RetrievalSearchService(Protocol):
    async def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.VECTOR,
        limit: int = 5,
        filing_id: int | None = None,
        company_name: str | None = None,
        year: int | None = None,
        form_type: str | None = None,
    ) -> FilingSearchResult: ...


@dataclass(frozen=True)
class RetrievalEvalResult:
    evaluation_id: str
    query_type: str
    mode: SearchMode
    retrieved_chunk_ids: list[int]
    relevant_chunk_ids: set[int]
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank: float
    latency_ms: float


RETRIEVAL_MODES = (
    SearchMode.VECTOR,
    SearchMode.KEYWORD,
    SearchMode.HYBRID,
    SearchMode.HYBRID_RERANK,
)


async def run_retrieval_evaluation(
    cases: list[RetrievalEvalCase],
    search_service: RetrievalSearchService,
) -> list[RetrievalEvalResult]:
    if not cases:
        raise ValueError("cases must not be empty")

    evaluation_results: list[RetrievalEvalResult] = []

    for case in cases:
        for mode in RETRIEVAL_MODES:
            started_at = perf_counter()

            search_result = await search_service.search(
                query=case.question,
                mode=mode,
                limit=10,
                company_name=case.filters.company_name,
                year=case.filters.year,
                form_type=case.filters.form_type,
            )

            latency_ms = (
                perf_counter() - started_at
            ) * 1000

            retrieved_chunk_ids = [
                match.chunk_id
                for match in search_result.matches
            ]

            evaluation_results.append(
                RetrievalEvalResult(
                    evaluation_id=case.id,
                    query_type=case.query_type,
                    mode=mode,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    relevant_chunk_ids=case.relevant_chunk_ids,
                    recall_at_5=recall_at_k(
                        retrieved_chunk_ids,
                        case.relevant_chunk_ids,
                        k=5,
                    ),
                    recall_at_10=recall_at_k(
                        retrieved_chunk_ids,
                        case.relevant_chunk_ids,
                        k=10,
                    ),
                    reciprocal_rank=reciprocal_rank(
                        retrieved_chunk_ids,
                        case.relevant_chunk_ids,
                    ),
                    latency_ms=latency_ms,
                )
            )

    return evaluation_results