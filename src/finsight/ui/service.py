import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from openai import AsyncOpenAI
from redis.asyncio import Redis
from redis.exceptions import RedisError

from finsight.cache import RedisAnswerCache
from finsight.core.config import get_settings
from finsight.db.session import SessionLocal
from finsight.embeddings import OpenAIEmbeddingClient
from finsight.generation import OpenAIAnswerGenerator
from finsight.repositories.filing import FilingRepository
from finsight.repositories.filing_chunk import FilingChunkRepository
from finsight.reranking.base import Reranker
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
)
from finsight.services.filing_search import FilingSearchService
from finsight.ui.catalog import FilingCatalogItem, excerpt

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_reranker() -> Reranker:
    from finsight.reranking.cross_encoder import CrossEncoderReranker

    return CrossEncoderReranker(model_name=get_settings().reranker_model)


@dataclass(frozen=True)
class CitationView:
    chunk_id: int
    company_name: str
    form_type: str
    filing_date: str
    item: str
    title: str
    source_url: str
    excerpt: str


@dataclass(frozen=True)
class AnswerView:
    answer: str
    sufficient_evidence: bool
    citations: list[CitationView]
    evidence_count: int
    cached: bool = False


def list_indexed_filings() -> list[FilingCatalogItem]:
    with SessionLocal() as session:
        records = FilingRepository(session).list_filings(limit=100)
        return [
            FilingCatalogItem(
                id=record.id,
                company_name=record.company_name,
                form_type=record.form_type,
                filing_date=record.filing_date,
                report_period=record.report_period,
                source_url=record.source_url,
            )
            for record in records
            if record.status == "embedded"
        ]


async def stream_filing_answer(
    *,
    question: str,
    filing_id: int,
    evidence_limit: int,
    on_delta: Callable[[str], None],
    on_retrieval: Callable[[int], None] | None = None,
) -> AnswerView:
    settings = get_settings()
    if settings.openai_api_key is None:
        raise RuntimeError("FINSIGHT_OPENAI_API_KEY is not configured")

    request = FilingAnswerRequest(
        question=question,
        filing_id=filing_id,
        evidence_limit=evidence_limit,
    )
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    cache = RedisAnswerCache(
        client=redis,
        ttl_seconds=settings.answer_cache_ttl_seconds,
        answer_model=settings.answer_model,
    )

    try:
        cached_response = await cache.get(request)
    except RedisError:
        logger.warning("Streamlit cache read failed", exc_info=True)
        cached_response = None

    try:
        if cached_response is not None:
            on_delta(cached_response.answer)
            with SessionLocal() as session:
                chunks = FilingChunkRepository(session).list_by_ids(
                    [citation.chunk_id for citation in cached_response.citations]
                )
                excerpts = {
                    chunk.id: excerpt(chunk.text)
                    for chunk in chunks
                }
            return _build_answer_view(
                cached_response,
                excerpts=excerpts,
                cached=True,
            )

        async with AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
        ) as sdk_client:
            with SessionLocal() as session:
                answer_service = FilingAnswerService(
                    search_service=FilingSearchService(
                        session=session,
                        embedding_client=OpenAIEmbeddingClient(
                            client=sdk_client,
                            model=settings.embedding_model,
                            dimensions=settings.embedding_dimensions,
                        ),
                        reranker=_get_reranker(),
                    ),
                    answer_generator=OpenAIAnswerGenerator(
                        client=sdk_client,
                        model=settings.answer_model,
                    ),
                )

                evidence_count = 0
                completed: FilingAnswerCompleted | None = None
                async for event in answer_service.stream_answer(
                    question=request.question,
                    evidence_limit=request.evidence_limit,
                    filing_id=request.filing_id,
                ):
                    if isinstance(event, FilingAnswerRetrieval):
                        evidence_count = len(event.chunk_ids)
                        if on_retrieval is not None:
                            on_retrieval(evidence_count)
                    elif isinstance(event, FilingAnswerTextDelta):
                        on_delta(event.text)
                    elif isinstance(event, FilingAnswerCompleted):
                        completed = event

                if completed is None:
                    raise RuntimeError("The answer stream ended unexpectedly")

                response = _response_from_result(completed)
                chunks = FilingChunkRepository(session).list_by_ids(
                    [citation.chunk_id for citation in response.citations]
                )
                excerpts = {
                    chunk.id: excerpt(chunk.text)
                    for chunk in chunks
                }

        try:
            await cache.set(request, response)
        except RedisError:
            logger.warning("Streamlit cache write failed", exc_info=True)

        return _build_answer_view(
            response,
            evidence_count=evidence_count,
            excerpts=excerpts,
        )
    finally:
        await redis.aclose()


def _response_from_result(
    completed: FilingAnswerCompleted,
) -> FilingAnswerResponse:
    result = completed.result
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


def _build_answer_view(
    response: FilingAnswerResponse,
    *,
    evidence_count: int | None = None,
    excerpts: dict[int, str] | None = None,
    cached: bool = False,
) -> AnswerView:
    excerpt_by_chunk = excerpts or {}
    return AnswerView(
        answer=response.answer,
        sufficient_evidence=response.sufficient_evidence,
        evidence_count=evidence_count or len(response.citations),
        cached=cached,
        citations=[
            CitationView(
                chunk_id=citation.chunk_id,
                company_name=citation.company_name,
                form_type=citation.form_type,
                filing_date=citation.filing_date.isoformat(),
                item=citation.item,
                title=citation.title,
                source_url=citation.source_url,
                excerpt=excerpt_by_chunk.get(citation.chunk_id, ""),
            )
            for citation in response.citations
        ],
    )
