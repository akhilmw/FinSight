from collections.abc import Sequence


def recall_at_k(
    retrieved_chunk_ids: Sequence[int],
    relevant_chunk_ids: set[int],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")

    if not relevant_chunk_ids:
        raise ValueError("relevant_chunk_ids must not be empty")

    retrieved_at_k = set(retrieved_chunk_ids[:k])
    relevant_found = retrieved_at_k & relevant_chunk_ids

    return len(relevant_found) / len(relevant_chunk_ids)


def reciprocal_rank(
    retrieved_chunk_ids: Sequence[int],
    relevant_chunk_ids: set[int],
) -> float:
    if not relevant_chunk_ids:
        raise ValueError("relevant_chunk_ids must not be empty")

    for rank, chunk_id in enumerate(
        retrieved_chunk_ids,
        start=1,
    ):
        if chunk_id in relevant_chunk_ids:
            return 1.0 / rank

    return 0.0