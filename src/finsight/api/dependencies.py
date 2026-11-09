from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Annotated, cast

from celery import Celery
from fastapi import Depends, HTTPException, Request, status
from openai import AsyncOpenAI
from redis.asyncio import Redis

from finsight.cache import RedisAnswerCache
from finsight.clients.sec import SecClient
from finsight.core.config import get_settings
from finsight.embeddings import (
    EmbeddingClient,
    OpenAIEmbeddingClient,
)
from finsight.generation import AnswerGenerator, OpenAIAnswerGenerator
from finsight.reranking.base import Reranker
from finsight.reranking.cross_encoder import CrossEncoderReranker
from finsight.storage.local import LocalDocumentStore
from finsight.workers.celery_app import celery_app


def get_document_store() -> LocalDocumentStore:
    settings = get_settings()
    return LocalDocumentStore(settings.raw_document_directory)


async def get_sec_client() -> AsyncIterator[SecClient]:
    async with SecClient() as client:
        yield client


async def get_embedding_client() -> AsyncIterator[EmbeddingClient]:
    settings = get_settings()

    if settings.openai_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The embedding service is not configured",
        )

    async with AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value()) as sdk_client:
        yield OpenAIEmbeddingClient(
            client=sdk_client,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )


async def get_answer_generator() -> AsyncIterator[AnswerGenerator]:
    settings = get_settings()

    if settings.openai_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The answer generation service is not configured",
        )

    async with AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value()) as sdk_client:
        yield OpenAIAnswerGenerator(
            client=sdk_client,
            model=settings.answer_model,
        )


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    settings = get_settings()
    return CrossEncoderReranker(model_name=settings.reranker_model)


def get_redis_client(request: Request) -> Redis:
    return cast(Redis, request.app.state.redis_client)


def get_answer_cache(
    redis_client: Annotated[
        Redis,
        Depends(get_redis_client),
    ],
) -> RedisAnswerCache:
    settings = get_settings()

    return RedisAnswerCache(
        client=redis_client,
        ttl_seconds=settings.answer_cache_ttl_seconds,
        answer_model=settings.answer_model,
    )


def get_celery_app() -> Celery:
    return celery_app
