from datetime import date
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

SearchQuery = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=2000,
    ),
]


class SearchMode(StrEnum):
    VECTOR = "vector"
    KEYWORD = "keyword"


class FilingSearchRequest(BaseModel):
    query: SearchQuery
    mode: SearchMode = SearchMode.VECTOR
    filing_id: int | None = Field(default=None, ge=1)
    limit: int = Field(default=5, ge=1, le=20)


class FilingSearchMatchResponse(BaseModel):
    chunk_id: int
    filing_id: int
    accession_number: str
    company_name: str
    form_type: str
    filing_date: date
    source_url: str
    item: str
    title: str
    text: str
    score: float
    distance: float | None = None
    similarity: float | None = None


class FilingSearchResponse(BaseModel):
    query: str
    mode: SearchMode
    matches: list[FilingSearchMatchResponse]
