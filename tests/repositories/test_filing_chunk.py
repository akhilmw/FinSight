import pytest

from finsight.repositories.filing_chunk import (
    _build_disjunctive_websearch_query,
)


def test_builds_or_query_from_natural_language() -> None:
    result = _build_disjunctive_websearch_query(
        "What workloads does NVIDIA's CUDA-based stack accelerate?"
    )

    assert result == (
        '"What" OR "workloads" OR "does" OR "NVIDIA\'s" OR '
        '"CUDA-based" OR "stack" OR "accelerate"'
    )


def test_preserves_financial_numbers_and_form_names() -> None:
    result = _build_disjunctive_websearch_query(
        "Why was the $4.5 billion H20 charge in the 10-K?"
    )

    assert '"4.5"' in result
    assert '"H20"' in result
    assert '"10-K"' in result


@pytest.mark.parametrize("query", ["", "   ", "?! $ %"])
def test_rejects_query_without_searchable_terms(query: str) -> None:
    with pytest.raises(
        ValueError,
        match="query must contain at least one searchable term",
    ):
        _build_disjunctive_websearch_query(query)
