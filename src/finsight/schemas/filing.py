from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Ticker = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        to_upper=True,
        min_length=1,
        max_length=10,
    ),
]

FormType = Literal["10-K", "10-Q"]


class FilingIngestRequest(BaseModel):
    ticker: Ticker
    form_types: tuple[FormType, ...] = ("10-K", "10-Q")
    limit: int = Field(default=10, ge=1, le=50)


class FilingIngestResponse(BaseModel):
    ticker: str
    discovered: int
    created: int
    skipped: int


class FilingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    accession_number: str
    cik: str
    company_name: str
    form_type: str
    filing_date: date
    report_period: date | None
    source_url: str
    status: str
    created_at: datetime
    updated_at: datetime
