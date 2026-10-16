import pytest

from finsight.evaluation.report import summarize_retrieval_results
from finsight.evaluation.runner import RetrievalEvalResult
from finsight.schemas import SearchMode


def create_result(
    evaluation_id: str,
    mode: SearchMode,
    recall_at_5: float,
    recall_at_10: float,
    reciprocal_rank: float,
    latency_ms: float,
) -> RetrievalEvalResult:
    return RetrievalEvalResult(
        evaluation_id=evaluation_id,
        query_type="mixed",
        mode=mode,
        retrieved_chunk_ids=[487],
        relevant_chunk_ids={487},
        recall_at_5=recall_at_5,
        recall_at_10=recall_at_10,
        reciprocal_rank=reciprocal_rank,
        latency_ms=latency_ms,
    )


def test_summarizes_results_by_retrieval_mode() -> None:
    summaries = summarize_retrieval_results(
        [
            create_result(
                "fin-001",
                SearchMode.VECTOR,
                recall_at_5=0.5,
                recall_at_10=1.0,
                reciprocal_rank=0.5,
                latency_ms=80.0,
            ),
            create_result(
                "fin-002",
                SearchMode.VECTOR,
                recall_at_5=1.0,
                recall_at_10=1.0,
                reciprocal_rank=1.0,
                latency_ms=100.0,
            ),
            create_result(
                "fin-001",
                SearchMode.KEYWORD,
                recall_at_5=0.0,
                recall_at_10=0.5,
                reciprocal_rank=0.25,
                latency_ms=10.0,
            ),
        ]
    )

    assert [summary.mode for summary in summaries] == [
        SearchMode.VECTOR,
        SearchMode.KEYWORD,
    ]

    vector, keyword = summaries
    assert vector.question_count == 2
    assert vector.mean_recall_at_5 == pytest.approx(0.75)
    assert vector.mean_recall_at_10 == pytest.approx(1.0)
    assert vector.mean_reciprocal_rank == pytest.approx(0.75)
    assert vector.mean_latency_ms == pytest.approx(90.0)

    assert keyword.question_count == 1
    assert keyword.mean_recall_at_5 == pytest.approx(0.0)
    assert keyword.mean_recall_at_10 == pytest.approx(0.5)
    assert keyword.mean_reciprocal_rank == pytest.approx(0.25)
    assert keyword.mean_latency_ms == pytest.approx(10.0)


def test_summary_uses_vector_keyword_hybrid_order() -> None:
    summaries = summarize_retrieval_results(
        [
            create_result("fin-001", SearchMode.HYBRID, 1.0, 1.0, 1.0, 90.0),
            create_result("fin-001", SearchMode.KEYWORD, 1.0, 1.0, 1.0, 10.0),
            create_result("fin-001", SearchMode.VECTOR, 1.0, 1.0, 1.0, 80.0),
        ]
    )

    assert [summary.mode for summary in summaries] == [
        SearchMode.VECTOR,
        SearchMode.KEYWORD,
        SearchMode.HYBRID,
    ]


def test_summary_rejects_empty_results() -> None:
    with pytest.raises(ValueError, match="results must not be empty"):
        summarize_retrieval_results([])
