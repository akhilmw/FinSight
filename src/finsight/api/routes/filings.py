from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from finsight.api.dependencies import get_document_store, get_sec_client
from finsight.clients.sec import CompanyNotFoundError, InvalidFilingDocumentError, SecClient
from finsight.db.session import get_db
from finsight.repositories.filing import FilingRepository
from finsight.schemas import (
    FilingDownloadResponse,
    FilingIngestRequest,
    FilingIngestResponse,
    FilingResponse,
)
from finsight.services.filing_download import (
    FilingDownloadService,
    FilingNotFoundError,
)
from finsight.services.filing_ingestion import FilingIngestionService
from finsight.storage.local import LocalDocumentStore

router = APIRouter(prefix="/filings", tags=["filings"])

DatabaseSession = Annotated[Session, Depends(get_db)]
SecClientDependency = Annotated[
    SecClient,
    Depends(get_sec_client),
]


@router.post(
    "/ingest",
    response_model=FilingIngestResponse,
    status_code=status.HTTP_200_OK,
)
async def ingest_filings(
    request: FilingIngestRequest,
    session: DatabaseSession,
    sec_client: SecClientDependency,
) -> FilingIngestResponse:
    service = FilingIngestionService(
        session=session,
        sec_client=sec_client,
    )
    try:
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


@router.post(
    "/{filing_id}/download",
    response_model=FilingDownloadResponse,
)
async def download_filing(
    filing_id: int,
    session: DatabaseSession,
    sec_client: SecClientDependency,
    document_store: Annotated[
        LocalDocumentStore,
        Depends(get_document_store),
    ],
) -> FilingDownloadResponse:
    service = FilingDownloadService(
        session=session,
        sec_client=sec_client,
        document_store=document_store,
    )

    try:
        result = await service.download(filing_id)
    except FilingNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except (httpx.HTTPError, InvalidFilingDocumentError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The SEC filing could not be downloaded",
        ) from exc

    return FilingDownloadResponse(
        filing_id=result.filing_id,
        path=result.path,
        sha256=result.sha256,
    )
