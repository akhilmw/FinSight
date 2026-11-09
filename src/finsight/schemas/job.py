from typing import Any

from pydantic import BaseModel


class FilingPipelineJobResponse(BaseModel):
    task_id: str
    filing_id: int
    status: str


class JobStatusResponse(BaseModel):
    task_id: str
    status: str
    ready: bool
    successful: bool | None
    result: dict[str, Any] | None = None
    error: str | None = None
