from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError


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