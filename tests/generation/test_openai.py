import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import AsyncOpenAI

from finsight.generation import AnswerEvidence
from finsight.generation.openai import (
    AnswerFieldDeltaExtractor,
    InvalidAnswerResponseError,
    OpenAIAnswerGenerator,
    StructuredAnswer,
)


def test_answer_field_delta_extractor_decodes_only_new_answer_text() -> None:
    extractor = AnswerFieldDeltaExtractor()

    assert extractor.feed('{"answer":"Revenue') == "Revenue"
    assert extractor.feed('{"answer":"Revenue increased\\nby') == " increased\nby"
    assert extractor.feed('{"answer":"Revenue increased\\nby \\u20') == ""
    assert extractor.feed('{"answer":"Revenue increased\\nby \\u20ac"') == " €"


def test_answer_field_delta_extractor_ignores_other_fields() -> None:
    extractor = AnswerFieldDeltaExtractor()

    assert extractor.feed('{"status":"working"') == ""


def create_sdk_client(
    parsed: StructuredAnswer | None,
) -> tuple[AsyncOpenAI, AsyncMock]:
    parse = AsyncMock(
        return_value=SimpleNamespace(output_parsed=parsed),
    )
    sdk_client = MagicMock()
    sdk_client.responses.parse = parse

    return cast(AsyncOpenAI, sdk_client), parse


def create_evidence() -> list[AnswerEvidence]:
    return [
        AnswerEvidence(
            chunk_id=487,
            item="1A",
            title="Risk Factors",
            text="Export controls may restrict sales in certain markets.",
        ),
        AnswerEvidence(
            chunk_id=488,
            item="7",
            title="Management's Discussion and Analysis",
            text="Revenue increased during the reporting period.",
        ),
    ]


def test_generate_returns_validated_answer_and_deduplicates_citations() -> None:
    sdk_client, parse = create_sdk_client(
        StructuredAnswer(
            answer=" Export controls may restrict sales. ",
            sufficient_evidence=True,
            cited_chunk_ids=[487, 488, 487],
        )
    )
    generator = OpenAIAnswerGenerator(
        client=sdk_client,
        model="test-answer-model",
    )

    result = asyncio.run(
        generator.generate(
            question="  What could restrict sales?  ",
            evidence=create_evidence(),
        )
    )

    assert result.answer == "Export controls may restrict sales."
    assert result.sufficient_evidence is True
    assert result.cited_chunk_ids == [487, 488]
    assert generator.model == "test-answer-model"

    parse.assert_awaited_once()
    await_args = parse.await_args
    assert await_args is not None
    request = await_args.kwargs
    assert request["model"] == "test-answer-model"
    assert request["text_format"] is StructuredAnswer
    assert request["max_output_tokens"] == 800
    assert request["store"] is False
    assert "Question:\nWhat could restrict sales?" in request["input"]
    assert "[CHUNK 487]" in request["input"]
    assert "Item 1A" not in request["input"]
    assert "Section: 1A — Risk Factors" in request["input"]


def test_generate_returns_insufficient_answer_without_evidence() -> None:
    sdk_client, parse = create_sdk_client(None)
    generator = OpenAIAnswerGenerator(
        client=sdk_client,
        model="test-answer-model",
    )

    result = asyncio.run(
        generator.generate(
            question="What was the revenue?",
            evidence=[],
        )
    )

    assert result.sufficient_evidence is False
    assert result.cited_chunk_ids == []
    assert "insufficient" in result.answer
    parse.assert_not_awaited()


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_generate_rejects_blank_question(question: str) -> None:
    sdk_client, parse = create_sdk_client(None)
    generator = OpenAIAnswerGenerator(
        client=sdk_client,
        model="test-answer-model",
    )

    with pytest.raises(ValueError, match="question must not be blank"):
        asyncio.run(generator.generate(question, create_evidence()))

    parse.assert_not_awaited()


def test_generate_clears_citations_for_insufficient_evidence() -> None:
    sdk_client, _ = create_sdk_client(
        StructuredAnswer(
            answer="The evidence does not contain that information.",
            sufficient_evidence=False,
            cited_chunk_ids=[999],
        )
    )
    generator = OpenAIAnswerGenerator(
        client=sdk_client,
        model="test-answer-model",
    )

    result = asyncio.run(
        generator.generate("What is the answer?", create_evidence())
    )

    assert result.sufficient_evidence is False
    assert result.cited_chunk_ids == []


def test_generate_rejects_missing_structured_output() -> None:
    sdk_client, _ = create_sdk_client(None)
    generator = OpenAIAnswerGenerator(
        client=sdk_client,
        model="test-answer-model",
    )

    with pytest.raises(
        InvalidAnswerResponseError,
        match="did not return a structured answer",
    ):
        asyncio.run(
            generator.generate("What is the answer?", create_evidence())
        )


def test_generate_rejects_unknown_citation() -> None:
    sdk_client, _ = create_sdk_client(
        StructuredAnswer(
            answer="An unsupported answer.",
            sufficient_evidence=True,
            cited_chunk_ids=[999],
        )
    )
    generator = OpenAIAnswerGenerator(
        client=sdk_client,
        model="test-answer-model",
    )

    with pytest.raises(
        InvalidAnswerResponseError,
        match="cited chunks that were not supplied",
    ):
        asyncio.run(
            generator.generate("What is the answer?", create_evidence())
        )


def test_generate_requires_citation_for_sufficient_answer() -> None:
    sdk_client, _ = create_sdk_client(
        StructuredAnswer(
            answer="An uncited answer.",
            sufficient_evidence=True,
            cited_chunk_ids=[],
        )
    )
    generator = OpenAIAnswerGenerator(
        client=sdk_client,
        model="test-answer-model",
    )

    with pytest.raises(
        InvalidAnswerResponseError,
        match="must cite at least one chunk",
    ):
        asyncio.run(
            generator.generate("What is the answer?", create_evidence())
        )
