from collections.abc import AsyncIterator

from fastapi import HTTPException, status
from openai import AsyncOpenAI

from finsight.clients.sec import SecClient
from finsight.core.config import get_settings
from finsight.embeddings import (
    EmbeddingClient,
    OpenAIEmbeddingClient,
)
from finsight.storage.local import LocalDocumentStore


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
