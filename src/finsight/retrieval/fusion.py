from collections.abc import Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]],
    rank_constant: int = 60,
) -> dict[int, float]:
    if rank_constant < 1:
        raise ValueError("rank_constant must be at least 1")

    scores: dict[int, float] = {}

    for ranked_list in ranked_lists:
        seen: set[int] = set()

        for rank, chunk_id in enumerate(ranked_list, start=1):
            if chunk_id in seen:
                continue

            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (
                1.0 / (rank_constant + rank)
            )

    return scores