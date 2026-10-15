from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from finsight.api.dependencies import get_embedding_client
from finsight.db.session import get_db
from finsight.embeddings import EmbeddingClient
from finsight.schemas import (
    FilingSearchMatchResponse,
    FilingSearchRequest,
    FilingSearchResponse,
)
from finsight.services.filing_search import FilingSearchService

router = APIRouter(prefix="/search", tags=["search"])

DatabaseSession = Annotated[Session, Depends(get_db)]
EmbeddingClientDependency = Annotated[
    EmbeddingClient,
    Depends(get_embedding_client),
]


@router.post(
    "",
    response_model=FilingSearchResponse,
)
async def search_filings(
    request: FilingSearchRequest,
    session: DatabaseSession,
    embedding_client: EmbeddingClientDependency,
) -> FilingSearchResponse:
    service = FilingSearchService(
        session=session,
        embedding_client=embedding_client,
    )

    result = await service.search(
        query=request.query,
        filing_id=request.filing_id,
        limit=request.limit,
    )

    return FilingSearchResponse(
        query=result.query,
        matches=[
            FilingSearchMatchResponse(
                chunk_id=match.chunk_id,
                filing_id=match.filing_id,
                accession_number=match.accession_number,
                company_name=match.company_name,
                form_type=match.form_type,
                filing_date=match.filing_date,
                source_url=match.source_url,
                item=match.item,
                title=match.title,
                text=match.text,
                distance=match.distance,
                similarity=match.similarity,
            )
            for match in result.matches
        ],
    )
