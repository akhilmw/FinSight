from typing import Any

import pytest

import finsight.reranking.cross_encoder as cross_encoder_module
from finsight.reranking.cross_encoder import CrossEncoderReranker


def test_rerank_builds_query_document_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeCrossEncoder:
        def __init__(self, model_name: str) -> None:
            captured["model_name"] = model_name

        def predict(
            self,
            pairs: list[tuple[str, str]],
            show_progress_bar: bool,
        ) -> list[float]:
            captured["pairs"] = pairs
            captured["show_progress_bar"] = show_progress_bar
            return [1.5, 7.2]

    monkeypatch.setattr(
        cross_encoder_module,
        "CrossEncoder",
        FakeCrossEncoder,
    )

    reranker = CrossEncoderReranker(model_name="test-model")

    scores = reranker.rerank(
        query="What threatens NVIDIA revenue?",
        documents=[
            "Demand for gaming GPUs increased.",
            "Export controls may restrict GPU sales.",
        ],
    )

    assert scores == [1.5, 7.2]
    assert captured == {
        "model_name": "test-model",
        "pairs": [
            (
                "What threatens NVIDIA revenue?",
                "Demand for gaming GPUs increased.",
            ),
            (
                "What threatens NVIDIA revenue?",
                "Export controls may restrict GPU sales.",
            ),
        ],
        "show_progress_bar": False,
    }


def test_rerank_returns_empty_list_for_no_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCrossEncoder:
        def __init__(self, model_name: str) -> None:
            pass

    monkeypatch.setattr(
        cross_encoder_module,
        "CrossEncoder",
        FakeCrossEncoder,
    )

    reranker = CrossEncoderReranker(model_name="test-model")

    assert reranker.rerank("question", []) == []