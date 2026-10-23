import asyncio
from datetime import date
from typing import cast

import pytest

from finsight.generation import (
    AnswerEvidence,
    AnswerStreamCompleted,
    AnswerTextDelta,
    GeneratedAnswer,
)
from finsight.schemas import SearchMode
from finsight.services.filing_answer import (
    FilingAnswerCompleted,
    FilingAnswerRetrieval,
    FilingAnswerService,
    FilingAnswerTextDelta,
    InvalidGeneratedCitationError,
)
from finsight.services.filing_search import (
    FilingSearchMatch,
    FilingSearchResult,
    FilingSearchService,
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
        item="1A",
        title="Risk Factors",
        text=f"Evidence from chunk {chunk_id}.",
        score=0.9,
        retrieval_score=0.03,
        rerank_score=0.9,
    )


class FakeSearchService:
    def __init__(self, matches: list[FilingSearchMatch]) -> None:
        self._matches = matches
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
        return FilingSearchResult(
            query=query,
            mode=mode,
            matches=self._matches,
        )


class FakeAnswerGenerator:
    def __init__(self, result: GeneratedAnswer) -> None:
        self._result = result
        self.calls: list[tuple[str, list[AnswerEvidence]]] = []

    async def generate(
        self,
        question: str,
        evidence: list[AnswerEvidence],
    ) -> GeneratedAnswer:
        self.calls.append((question, evidence))
        return self._result

    async def stream(
        self,
        question: str,
        evidence: list[AnswerEvidence],
    ):
        self.calls.append((question, evidence))
        yield AnswerTextDelta(self._result.answer)
        yield AnswerStreamCompleted(self._result)


def create_service(
    matches: list[FilingSearchMatch],
    generated: GeneratedAnswer,
) -> tuple[FilingAnswerService, FakeSearchService, FakeAnswerGenerator]:
    search_service = FakeSearchService(matches)
    answer_generator = FakeAnswerGenerator(generated)
    service = FilingAnswerService(
        search_service=cast(FilingSearchService, search_service),
        answer_generator=answer_generator,
    )
    return service, search_service, answer_generator


def test_answer_retrieves_reranked_evidence_and_builds_trusted_citations() -> None:
    service, search_service, answer_generator = create_service(
        matches=[create_match(487), create_match(488)],
        generated=GeneratedAnswer(
            answer="Export controls may restrict sales.",
            sufficient_evidence=True,
            cited_chunk_ids=[488, 487],
        ),
    )

    result = asyncio.run(
        service.answer(
            question="  What could restrict NVIDIA sales?  ",
            evidence_limit=7,
            filing_id=3,
            company_name="NVIDIA",
            year=2026,
            form_type="10-K",
        )
    )

    assert search_service.calls == [
        {
            "query": "What could restrict NVIDIA sales?",
            "mode": SearchMode.HYBRID_RERANK,
            "limit": 7,
            "filing_id": 3,
            "company_name": "NVIDIA",
            "year": 2026,
            "form_type": "10-K",
        }
    ]
    assert answer_generator.calls == [
        (
            "What could restrict NVIDIA sales?",
            [
                AnswerEvidence(
                    chunk_id=487,
                    item="1A",
                    title="Risk Factors",
                    text="Evidence from chunk 487.",
                ),
                AnswerEvidence(
                    chunk_id=488,
                    item="1A",
                    title="Risk Factors",
                    text="Evidence from chunk 488.",
                ),
            ],
        )
    ]
    assert result.question == "What could restrict NVIDIA sales?"
    assert result.answer == "Export controls may restrict sales."
    assert result.sufficient_evidence is True
    assert [citation.chunk_id for citation in result.citations] == [488, 487]
    assert result.citations[0].source_url == "https://www.sec.gov/example.htm"
    assert result.citations[0].item == "1A"
    assert result.citations[0].page_number is None


def test_answer_returns_no_citations_when_evidence_is_insufficient() -> None:
    service, search_service, answer_generator = create_service(
        matches=[],
        generated=GeneratedAnswer(
            answer="The available evidence is insufficient.",
            sufficient_evidence=False,
            cited_chunk_ids=[],
        ),
    )

    result = asyncio.run(service.answer("What is the dividend policy?"))

    assert len(search_service.calls) == 1
    assert answer_generator.calls == [("What is the dividend policy?", [])]
    assert result.sufficient_evidence is False
    assert result.citations == []


def test_answer_rejects_citation_outside_retrieved_evidence() -> None:
    service, _, _ = create_service(
        matches=[create_match(487)],
        generated=GeneratedAnswer(
            answer="An unsupported answer.",
            sufficient_evidence=True,
            cited_chunk_ids=[999],
        ),
    )

    with pytest.raises(
        InvalidGeneratedCitationError,
        match="outside the retrieved evidence",
    ):
        asyncio.run(service.answer("What is the answer?"))


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_answer_rejects_blank_question_before_search(question: str) -> None:
    service, search_service, answer_generator = create_service(
        matches=[],
        generated=GeneratedAnswer("unused", False, []),
    )

    with pytest.raises(ValueError, match="question must not be blank"):
        asyncio.run(service.answer(question))

    assert search_service.calls == []
    assert answer_generator.calls == []


def test_answer_rejects_invalid_evidence_limit_before_search() -> None:
    service, search_service, answer_generator = create_service(
        matches=[],
        generated=GeneratedAnswer("unused", False, []),
    )

    with pytest.raises(
        ValueError,
        match="evidence_limit must be at least 1",
    ):
        asyncio.run(service.answer("What is the answer?", evidence_limit=0))

    assert search_service.calls == []
    assert answer_generator.calls == []


def test_stream_answer_emits_retrieval_text_and_validated_completion() -> None:
    service, _, _ = create_service(
        matches=[create_match(487)],
        generated=GeneratedAnswer(
            answer="Export controls may restrict sales.",
            sufficient_evidence=True,
            cited_chunk_ids=[487],
        ),
    )

    async def collect_events():
        return [
            event
            async for event in service.stream_answer(
                "What could restrict sales?"
            )
        ]

    events = asyncio.run(collect_events())

    assert isinstance(events[0], FilingAnswerRetrieval)
    assert isinstance(events[1], FilingAnswerTextDelta)
    assert isinstance(events[2], FilingAnswerCompleted)
    assert events[0].chunk_ids == [487]
    assert events[1].text == "Export controls may restrict sales."
    assert events[2].result.citations[0].chunk_id == 487
