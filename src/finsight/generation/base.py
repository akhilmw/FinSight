from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class AnswerEvidence:
    chunk_id: int
    item: str
    title: str
    text: str


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    sufficient_evidence: bool
    cited_chunk_ids: list[int]


@dataclass(frozen=True)
class AnswerTextDelta:
    text: str


@dataclass(frozen=True)
class AnswerStreamCompleted:
    answer: GeneratedAnswer


AnswerStreamEvent = AnswerTextDelta | AnswerStreamCompleted


class AnswerGenerator(Protocol):
    async def generate(
        self,
        question: str,
        evidence: list[AnswerEvidence],
    ) -> GeneratedAnswer:
        ...

    def stream(
        self,
        question: str,
        evidence: list[AnswerEvidence],
    ) -> AsyncIterator[AnswerStreamEvent]:
        ...
