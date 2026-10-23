import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from finsight.api.dependencies import (
    get_answer_generator,
    get_embedding_client,
    get_reranker,
)
from finsight.db.session import get_db
from finsight.embeddings import EmbeddingClient
from finsight.generation import (
    AnswerGenerator,
    InvalidAnswerResponseError,
)
from finsight.reranking import Reranker
from finsight.schemas import (
    AnswerCitationResponse,
    FilingAnswerRequest,
    FilingAnswerResponse,
)
from finsight.services.filing_answer import (
    FilingAnswerCompleted,
    FilingAnswerRetrieval,
    FilingAnswerService,
    FilingAnswerTextDelta,
    InvalidGeneratedCitationError,
)
from finsight.services.filing_search import FilingSearchService

router = APIRouter(prefix="/answers", tags=["answers"])
logger = logging.getLogger(__name__)

DatabaseSession = Annotated[Session, Depends(get_db)]
EmbeddingClientDependency = Annotated[
    EmbeddingClient,
    Depends(get_embedding_client),
]
RerankerDependency = Annotated[
    Reranker,
    Depends(get_reranker),
]
AnswerGeneratorDependency = Annotated[
    AnswerGenerator,
    Depends(get_answer_generator),
]


@router.post(
    "",
    response_model=FilingAnswerResponse,
)
async def answer_filing_question(
    request: FilingAnswerRequest,
    session: DatabaseSession,
    embedding_client: EmbeddingClientDependency,
    reranker: RerankerDependency,
    answer_generator: AnswerGeneratorDependency,
) -> FilingAnswerResponse:
    search_service = FilingSearchService(
        session=session,
        embedding_client=embedding_client,
        reranker=reranker,
    )
    answer_service = FilingAnswerService(
        search_service=search_service,
        answer_generator=answer_generator,
    )

    try:
        result = await answer_service.answer(
            question=request.question,
            evidence_limit=request.evidence_limit,
            filing_id=request.filing_id,
            company_name=request.company_name,
            year=request.year,
            form_type=request.form_type,
        )
    except (InvalidAnswerResponseError, InvalidGeneratedCitationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return FilingAnswerResponse(
        question=result.question,
        answer=result.answer,
        sufficient_evidence=result.sufficient_evidence,
        citations=[
            AnswerCitationResponse(
                chunk_id=citation.chunk_id,
                filing_id=citation.filing_id,
                accession_number=citation.accession_number,
                company_name=citation.company_name,
                form_type=citation.form_type,
                filing_date=citation.filing_date,
                source_url=citation.source_url,
                item=citation.item,
                title=citation.title,
                page_number=citation.page_number,
            )
            for citation in result.citations
        ],
    )


@router.post("/stream")
async def stream_filing_answer(
    request: FilingAnswerRequest,
    http_request: Request,
    session: DatabaseSession,
    embedding_client: EmbeddingClientDependency,
    reranker: RerankerDependency,
    answer_generator: AnswerGeneratorDependency,
) -> StreamingResponse:
    search_service = FilingSearchService(
        session=session,
        embedding_client=embedding_client,
        reranker=reranker,
    )
    answer_service = FilingAnswerService(
        search_service=search_service,
        answer_generator=answer_generator,
    )

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for event in answer_service.stream_answer(
                question=request.question,
                evidence_limit=request.evidence_limit,
                filing_id=request.filing_id,
                company_name=request.company_name,
                year=request.year,
                form_type=request.form_type,
            ):
                if await http_request.is_disconnected():
                    return

                if isinstance(event, FilingAnswerRetrieval):
                    yield encode_sse(
                        "retrieval",
                        {
                            "evidence_count": len(event.chunk_ids),
                            "chunk_ids": event.chunk_ids,
                        },
                    )
                elif isinstance(event, FilingAnswerTextDelta):
                    yield encode_sse(
                        "answer_delta",
                        {"text": event.text},
                    )
                elif isinstance(event, FilingAnswerCompleted):
                    citations = [
                        AnswerCitationResponse(
                            chunk_id=citation.chunk_id,
                            filing_id=citation.filing_id,
                            accession_number=citation.accession_number,
                            company_name=citation.company_name,
                            form_type=citation.form_type,
                            filing_date=citation.filing_date,
                            source_url=citation.source_url,
                            item=citation.item,
                            title=citation.title,
                            page_number=citation.page_number,
                        ).model_dump(mode="json")
                        for citation in event.result.citations
                    ]
                    yield encode_sse(
                        "citations",
                        {
                            "sufficient_evidence": event.result.sufficient_evidence,
                            "citations": citations,
                        },
                    )
                    yield encode_sse(
                        "done",
                        {
                            "question": event.result.question,
                            "sufficient_evidence": event.result.sufficient_evidence,
                        },
                    )
        except Exception:
            logger.exception("Streaming answer generation failed")
            yield encode_sse(
                "error",
                {"message": "Answer generation failed"},
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def encode_sse(event: str, data: dict[str, object]) -> str:
    payload = json.dumps(data, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"
