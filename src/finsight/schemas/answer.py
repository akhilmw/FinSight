from datetime import date

from pydantic import BaseModel, Field

from finsight.schemas.filing import FormType
from finsight.schemas.search import CompanyName, SearchQuery


class FilingAnswerRequest(BaseModel):
    question: SearchQuery
    evidence_limit: int = Field(default=5, ge=1, le=10)
    filing_id: int | None = Field(default=None, ge=1)
    company_name: CompanyName | None = None
    year: int | None = Field(default=None, ge=1900, le=2100)
    form_type: FormType | None = None


class AnswerCitationResponse(BaseModel):
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


class FilingAnswerResponse(BaseModel):
    question: str
    answer: str
    sufficient_evidence: bool
    citations: list[AnswerCitationResponse]
