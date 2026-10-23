from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import date

from finsight.generation import (
    AnswerEvidence,
    AnswerGenerator,
    AnswerStreamCompleted,
    AnswerTextDelta,
    GeneratedAnswer,
)
from finsight.schemas import SearchMode
from finsight.services.filing_search import FilingSearchMatch, FilingSearchService


@dataclass(frozen=True)
class AnswerCitation:
    chunk_id: int
    filing_id: int
    accession_number: str
    company_name: str
    form_type: str
    filing_date: date
    source_url: str
    item: str
    title: str
    page_number: int | None = None


@dataclass(frozen=True)
class FilingAnswerResult:
    question: str
    answer: str
    sufficient_evidence: bool
    citations: list[AnswerCitation]


@dataclass(frozen=True)
class FilingAnswerRetrieval:
    chunk_ids: list[int]


@dataclass(frozen=True)
class FilingAnswerTextDelta:
    text: str


@dataclass(frozen=True)
class FilingAnswerCompleted:
    result: FilingAnswerResult


FilingAnswerStreamEvent = (
    FilingAnswerRetrieval
    | FilingAnswerTextDelta
    | FilingAnswerCompleted
)


class InvalidGeneratedCitationError(RuntimeError):
    pass


class FilingAnswerService:
    def __init__(
        self,
        search_service: FilingSearchService,
        answer_generator: AnswerGenerator,
    ) -> None:
        self._search_service = search_service
        self._answer_generator = answer_generator

    async def answer(
        self,
        question: str,
        evidence_limit: int = 5,
        filing_id: int | None = None,
        company_name: str | None = None,
        year: int | None = None,
        form_type: str | None = None,
    ) -> FilingAnswerResult:
        normalized_question, matches, evidence = await self._retrieve(
            question=question,
            evidence_limit=evidence_limit,
            filing_id=filing_id,
            company_name=company_name,
            year=year,
            form_type=form_type,
        )

        generated = await self._answer_generator.generate(
            question=normalized_question,
            evidence=evidence,
        )

        return self._build_result(normalized_question, generated, matches)

    async def stream_answer(
        self,
        question: str,
        evidence_limit: int = 5,
        filing_id: int | None = None,
        company_name: str | None = None,
        year: int | None = None,
        form_type: str | None = None,
    ) -> AsyncIterator[FilingAnswerStreamEvent]:
        normalized_question, matches, evidence = await self._retrieve(
            question=question,
            evidence_limit=evidence_limit,
            filing_id=filing_id,
            company_name=company_name,
            year=year,
            form_type=form_type,
        )

        yield FilingAnswerRetrieval(
            chunk_ids=[match.chunk_id for match in matches],
        )

        async for event in self._answer_generator.stream(
            question=normalized_question,
            evidence=evidence,
        ):
            if isinstance(event, AnswerTextDelta):
                yield FilingAnswerTextDelta(event.text)
                continue

            if isinstance(event, AnswerStreamCompleted):
                yield FilingAnswerCompleted(
                    self._build_result(
                        normalized_question,
                        event.answer,
                        matches,
                    )
                )

    async def _retrieve(
        self,
        question: str,
        evidence_limit: int,
        filing_id: int | None,
        company_name: str | None,
        year: int | None,
        form_type: str | None,
    ) -> tuple[str, list[FilingSearchMatch], list[AnswerEvidence]]:
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError("question must not be blank")

        if evidence_limit < 1:
            raise ValueError("evidence_limit must be at least 1")

        search_result = await self._search_service.search(
            query=normalized_question,
            mode=SearchMode.HYBRID_RERANK,
            limit=evidence_limit,
            filing_id=filing_id,
            company_name=company_name,
            year=year,
            form_type=form_type,
        )

        evidence = [
            AnswerEvidence(
                chunk_id=match.chunk_id,
                item=match.item,
                title=match.title,
                text=match.text,
            )
            for match in search_result.matches
        ]
        return normalized_question, search_result.matches, evidence

    @staticmethod
    def _build_result(
        question: str,
        generated: GeneratedAnswer,
        matches: list[FilingSearchMatch],
    ) -> FilingAnswerResult:
        matches_by_chunk_id = {match.chunk_id: match for match in matches}

        invalid_chunk_ids = (
            set(generated.cited_chunk_ids)
            - matches_by_chunk_id.keys()
        )

        if invalid_chunk_ids:
            raise InvalidGeneratedCitationError(
                "The generated answer cited chunks outside the retrieved evidence"
            )

        citations = [
            AnswerCitation(
                chunk_id=match.chunk_id,
                filing_id=match.filing_id,
                accession_number=match.accession_number,
                company_name=match.company_name,
                form_type=match.form_type,
                filing_date=match.filing_date,
                source_url=match.source_url,
                item=match.item,
                title=match.title,
            )
            for chunk_id in generated.cited_chunk_ids
            if (match := matches_by_chunk_id.get(chunk_id)) is not None
        ]

        return FilingAnswerResult(
            question=question,
            answer=generated.answer,
            sufficient_evidence=generated.sufficient_evidence,
            citations=citations,
        )
