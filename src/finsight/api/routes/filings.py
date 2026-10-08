from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from finsight.clients.sec import CompanyNotFoundError, SecClient
from finsight.db.session import get_db
from finsight.repositories.filing import FilingRepository
from finsight.schemas import (
    FilingIngestRequest,
    FilingIngestResponse,
    FilingResponse,
)
from finsight.services.filing_ingestion import FilingIngestionService

router = APIRouter(prefix="/filings", tags=["filings"])

DatabaseSession = Annotated[Session, Depends(get_db)]


@router.post(
    "/ingest",
    response_model=FilingIngestResponse,
    status_code=status.HTTP_200_OK,
)
async def ingest_filings(
    request: FilingIngestRequest,
    session: DatabaseSession,
) -> FilingIngestResponse:
    try:
        async with SecClient() as sec_client:
            service = FilingIngestionService(
                session=session,
                sec_client=sec_client,
            )

            result = await service.ingest(
                ticker=request.ticker,
                form_types=request.form_types,
                limit=request.limit,
            )
    except CompanyNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The SEC service could not be reached",
        ) from exc

    return FilingIngestResponse(
        ticker=result.ticker,
        discovered=result.discovered,
        created=result.created,
        skipped=result.skipped,
    )


@router.get(
    "",
    response_model=list[FilingResponse],
)
def list_filings(
    session: DatabaseSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[FilingResponse]:
    repository = FilingRepository(session)
    filings = repository.list_filings(
        limit=limit,
        offset=offset,
    )

    return [FilingResponse.model_validate(filing) for filing in filings]


@router.get(
    "/{filing_id}",
    response_model=FilingResponse,
)
def get_filing(
    filing_id: int,
    session: DatabaseSession,
) -> FilingResponse:
    repository = FilingRepository(session)
    filing = repository.get_by_id(filing_id)

    if filing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Filing {filing_id} was not found",
        )

    return FilingResponse.model_validate(filing)
