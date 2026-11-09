from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from finsight.db.models import Filing, FilingChunkRecord, FilingSectionRecord


class RetrievalEvalFilters(BaseModel):
    company_name: str | None = None
    year: int | None = Field(default=None, ge=1900, le=2100)
    form_type: Literal["10-K", "10-Q"] | None = None


class RelevantChunkLabel(BaseModel):
    accession_number: str
    item: str
    section_title: str
    chunk_index: int = Field(ge=0)
    current_chunk_id: int = Field(ge=1)


class RetrievalEvalCase(BaseModel):
    id: str = Field(pattern=r"^fin-\d{3}$")
    question: str = Field(min_length=1)
    query_type: Literal["exact", "semantic", "mixed"]
    filters: RetrievalEvalFilters
    relevant_chunks: tuple[RelevantChunkLabel, ...] = Field(
        min_length=1
    )
    answer_hint: str = Field(min_length=1)

    @property
    def relevant_chunk_ids(self) -> set[int]:
        return {
            label.current_chunk_id
            for label in self.relevant_chunks
        }


def load_retrieval_eval_cases(
    path: Path,
) -> list[RetrievalEvalCase]:
    cases: list[RetrievalEvalCase] = []
    seen_ids: set[str] = set()

    for line_number, line in enumerate(
        path.read_text().splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            case = RetrievalEvalCase.model_validate_json(line)
        except ValidationError as error:
            raise ValueError(
                f"Invalid evaluation case on line {line_number}"
            ) from error

        if case.id in seen_ids:
            raise ValueError(
                f"Duplicate evaluation case ID: {case.id}"
            )

        seen_ids.add(case.id)
        cases.append(case)

    if not cases:
        raise ValueError("Evaluation dataset must not be empty")

    return cases


def resolve_retrieval_eval_cases(
    cases: list[RetrievalEvalCase],
    session: Session,
) -> list[RetrievalEvalCase]:
    """Resolve durable labels to IDs in the current database.

    Chunk IDs can change when a filing is processed again. The accession
    number, section metadata, and chunk index are the stable evaluation label.
    """
    resolved_cases: list[RetrievalEvalCase] = []

    for case in cases:
        resolved_labels: list[RelevantChunkLabel] = []

        for label in case.relevant_chunks:
            statement = (
                select(FilingChunkRecord.id)
                .join(FilingSectionRecord)
                .join(Filing)
                .where(
                    Filing.accession_number == label.accession_number,
                    FilingSectionRecord.item == label.item,
                    FilingSectionRecord.title == label.section_title,
                    FilingChunkRecord.chunk_index == label.chunk_index,
                )
            )
            chunk_id = session.scalar(statement)
            if chunk_id is None:
                raise ValueError(
                    "Could not resolve evaluation label "
                    f"{case.id}: {label.accession_number} / "
                    f"{label.item} / {label.chunk_index}"
                )

            resolved_labels.append(
                label.model_copy(update={"current_chunk_id": chunk_id})
            )

        resolved_cases.append(
            case.model_copy(
                update={"relevant_chunks": tuple(resolved_labels)}
            )
        )

    return resolved_cases
