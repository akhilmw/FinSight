import asyncio
from datetime import date

import pytest

from finsight.evaluation.dataset import RetrievalEvalCase
from finsight.evaluation.runner import run_retrieval_evaluation
from finsight.schemas import SearchMode
from finsight.services.filing_search import (
    FilingSearchMatch,
    FilingSearchResult,
)


def create_case(index: int) -> RetrievalEvalCase:
    return RetrievalEvalCase.model_validate(
        {
            "id": f"fin-{index:03d}",
            "question": f"Evaluation question {index}",
            "query_type": "mixed",
            "filters": {
                "company_name": "NVIDIA",
                "year": 2026,
                "form_type": "10-K",
            },
            "relevant_chunks": [
                {
                    "accession_number": "0001045810-26-000021",
                    "item": "1",
                    "section_title": "Business",
                    "chunk_index": 0,
                    "current_chunk_id": 10,
                },
                {
                    "accession_number": "0001045810-26-000021",
                    "item": "1",
                    "section_title": "Business",
                    "chunk_index": 1,
                    "current_chunk_id": 20,
                },
            ],
            "answer_hint": "Expected answer.",
        }
    )


def create_match(chunk_id: int) -> FilingSearchMatch:
    return FilingSearchMatch(
        chunk_id=chunk_id,
        filing_id=3,
        accession_number="0001045810-26-000021",
        company_name="NVIDIA CORP",
        form_type="10-K",
        filing_date=date(2026, 2, 25),
        source_url="https://www.sec.gov/example.htm",
        item="1",
        title="Business",
        text=f"Chunk {chunk_id}",
        score=1.0,
    )


class FakeSearchService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

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
        self.calls.append(
            {
                "query": query,
                "mode": mode,
                "limit": limit,
                "filing_id": filing_id,
                "company_name": company_name,
                "year": year,
                "form_type": form_type,
            }
        )

        chunk_ids_by_mode = {
            SearchMode.VECTOR: [100, 101, 102, 103, 104, 10, 20],
            SearchMode.KEYWORD: [10],
            SearchMode.HYBRID: [20, 10],
            SearchMode.HYBRID_RERANK: [10, 20],
        }

        return FilingSearchResult(
            query=query,
            mode=mode,
            matches=[
                create_match(chunk_id)
                for chunk_id in chunk_ids_by_mode[mode]
            ],
        )


def test_runner_executes_all_cases_in_every_retrieval_mode() -> None:
    cases = [create_case(index) for index in range(1, 26)]
    service = FakeSearchService()

    results = asyncio.run(
        run_retrieval_evaluation(
            cases=cases,
            search_service=service,
        )
    )

    assert len(results) == 100
    assert len(service.calls) == 100
    assert [result.mode for result in results[:4]] == [
        SearchMode.VECTOR,
        SearchMode.KEYWORD,
        SearchMode.HYBRID,
        SearchMode.HYBRID_RERANK,
    ]
    assert all(call["limit"] == 10 for call in service.calls)
    assert all(call["company_name"] == "NVIDIA" for call in service.calls)
    assert all(call["year"] == 2026 for call in service.calls)
    assert all(call["form_type"] == "10-K" for call in service.calls)


def test_runner_calculates_metrics_for_each_mode() -> None:
    service = FakeSearchService()

    results = asyncio.run(
        run_retrieval_evaluation(
            cases=[create_case(1)],
            search_service=service,
        )
    )

    vector, keyword, hybrid, hybrid_rerank = results

    assert vector.recall_at_5 == pytest.approx(0.0)
    assert vector.recall_at_10 == pytest.approx(1.0)
    assert vector.reciprocal_rank == pytest.approx(1 / 6)

    assert keyword.recall_at_5 == pytest.approx(0.5)
    assert keyword.recall_at_10 == pytest.approx(0.5)
    assert keyword.reciprocal_rank == pytest.approx(1.0)

    assert hybrid.recall_at_5 == pytest.approx(1.0)
    assert hybrid.recall_at_10 == pytest.approx(1.0)
    assert hybrid.reciprocal_rank == pytest.approx(1.0)

    assert hybrid_rerank.recall_at_5 == pytest.approx(1.0)
    assert hybrid_rerank.recall_at_10 == pytest.approx(1.0)
    assert hybrid_rerank.reciprocal_rank == pytest.approx(1.0)

    assert all(result.latency_ms >= 0 for result in results)


def test_runner_rejects_empty_cases() -> None:
    with pytest.raises(ValueError, match="cases must not be empty"):
        asyncio.run(
            run_retrieval_evaluation(
                cases=[],
                search_service=FakeSearchService(),
            )
        )
