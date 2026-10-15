from datetime import date
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


class FilingSearchRequest(BaseModel):
    query: SearchQuery
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
    distance: float
    similarity: float


class FilingSearchResponse(BaseModel):
    query: str
    matches: list[FilingSearchMatchResponse]
