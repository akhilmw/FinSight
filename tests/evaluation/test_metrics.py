import pytest

from finsight.evaluation.metrics import recall_at_k, reciprocal_rank


def test_recall_at_k_returns_full_recall_when_all_relevant_chunks_are_found() -> None:
    result = recall_at_k(
        retrieved_chunk_ids=[487, 600, 488],
        relevant_chunk_ids={487, 488},
        k=3,
    )

    assert result == pytest.approx(1.0)


def test_recall_at_k_returns_partial_recall() -> None:
    result = recall_at_k(
        retrieved_chunk_ids=[600, 487, 700, 488],
        relevant_chunk_ids={487, 488},
        k=3,
    )

    assert result == pytest.approx(0.5)


def test_recall_at_k_returns_zero_when_nothing_relevant_is_found() -> None:
    result = recall_at_k(
        retrieved_chunk_ids=[600, 700, 800],
        relevant_chunk_ids={487, 488},
        k=3,
    )

    assert result == pytest.approx(0.0)


def test_recall_at_k_does_not_count_duplicate_results_twice() -> None:
    result = recall_at_k(
        retrieved_chunk_ids=[487, 487, 600],
        relevant_chunk_ids={487, 488},
        k=3,
    )

    assert result == pytest.approx(0.5)


@pytest.mark.parametrize("k", [0, -1])
def test_recall_at_k_rejects_invalid_k(k: int) -> None:
    with pytest.raises(ValueError, match="k must be at least 1"):
        recall_at_k(
            retrieved_chunk_ids=[487],
            relevant_chunk_ids={487},
            k=k,
        )


def test_recall_at_k_rejects_empty_relevance_labels() -> None:
    with pytest.raises(
        ValueError,
        match="relevant_chunk_ids must not be empty",
    ):
        recall_at_k(
            retrieved_chunk_ids=[487],
            relevant_chunk_ids=set(),
            k=1,
        )


@pytest.mark.parametrize(
    ("retrieved_chunk_ids", "expected"),
    [
        ([487, 600, 700], 1.0),
        ([600, 487, 700], 0.5),
        ([600, 700, 487], 1 / 3),
        ([600, 700, 800], 0.0),
    ],
)
def test_reciprocal_rank_uses_first_relevant_result(
    retrieved_chunk_ids: list[int],
    expected: float,
) -> None:
    result = reciprocal_rank(
        retrieved_chunk_ids=retrieved_chunk_ids,
        relevant_chunk_ids={487, 488},
    )

    assert result == pytest.approx(expected)


def test_reciprocal_rank_rejects_empty_relevance_labels() -> None:
    with pytest.raises(
        ValueError,
        match="relevant_chunk_ids must not be empty",
    ):
        reciprocal_rank(
            retrieved_chunk_ids=[487],
            relevant_chunk_ids=set(),
        )
