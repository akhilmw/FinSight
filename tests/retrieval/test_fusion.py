import pytest

from finsight.retrieval.fusion import reciprocal_rank_fusion


def test_fusion_rewards_chunks_ranked_highly_in_both_lists() -> None:
    scores = reciprocal_rank_fusion(
        ranked_lists=[
            [10, 20, 30],
            [20, 40, 10],
        ],
    )

    ranked_chunk_ids = sorted(
        scores,
        key=lambda chunk_id: scores[chunk_id],
        reverse=True,
    )

    assert ranked_chunk_ids[0] == 20
    assert scores[20] == pytest.approx((1 / 62) + (1 / 61))


def test_fusion_includes_chunks_found_by_only_one_retriever() -> None:
    scores = reciprocal_rank_fusion(
        ranked_lists=[
            [10, 20],
            [30, 40],
        ],
    )

    assert set(scores) == {10, 20, 30, 40}


def test_fusion_does_not_score_duplicate_chunk_twice_per_list() -> None:
    scores = reciprocal_rank_fusion(
        ranked_lists=[[10, 10]],
    )

    assert scores[10] == pytest.approx(1 / 61)


def test_fusion_returns_empty_scores_for_empty_rankings() -> None:
    assert reciprocal_rank_fusion([]) == {}


def test_fusion_rejects_invalid_rank_constant() -> None:
    with pytest.raises(
        ValueError,
        match="rank_constant must be at least 1",
    ):
        reciprocal_rank_fusion(
            ranked_lists=[[10]],
            rank_constant=0,
        )
