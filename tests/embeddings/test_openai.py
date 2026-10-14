import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import AsyncOpenAI

from finsight.embeddings.openai import (
    InvalidEmbeddingResponseError,
    OpenAIEmbeddingClient,
)


def create_sdk_client(
    data: list[SimpleNamespace],
) -> tuple[AsyncOpenAI, AsyncMock]:
    create = AsyncMock(
        return_value=SimpleNamespace(data=data),
    )
    sdk_client = MagicMock()
    sdk_client.embeddings.create = create

    return cast(AsyncOpenAI, sdk_client), create


def test_embed_returns_vectors_in_input_order() -> None:
    sdk_client, create = create_sdk_client(
        [
            SimpleNamespace(index=1, embedding=[0.3, 0.4]),
            SimpleNamespace(index=0, embedding=[0.1, 0.2]),
        ]
    )
    client = OpenAIEmbeddingClient(
        client=sdk_client,
        model="text-embedding-3-small",
        dimensions=2,
    )

    embeddings = asyncio.run(client.embed(["first", "second"]))

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert client.model == "text-embedding-3-small"
    assert client.dimensions == 2
    create.assert_awaited_once_with(
        input=["first", "second"],
        model="text-embedding-3-small",
        dimensions=2,
        encoding_format="float",
    )


def test_embed_returns_empty_list_without_api_request() -> None:
    sdk_client, create = create_sdk_client([])
    client = OpenAIEmbeddingClient(
        client=sdk_client,
        model="text-embedding-3-small",
        dimensions=2,
    )

    assert asyncio.run(client.embed([])) == []
    create.assert_not_awaited()


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_embed_rejects_blank_input(text: str) -> None:
    sdk_client, create = create_sdk_client([])
    client = OpenAIEmbeddingClient(
        client=sdk_client,
        model="text-embedding-3-small",
        dimensions=2,
    )

    with pytest.raises(
        ValueError,
        match="embedding input must not be blank",
    ):
        asyncio.run(client.embed([text]))

    create.assert_not_awaited()


def test_client_rejects_invalid_dimensions() -> None:
    sdk_client, _ = create_sdk_client([])

    with pytest.raises(
        ValueError,
        match="dimensions must be at least 1",
    ):
        OpenAIEmbeddingClient(
            client=sdk_client,
            model="text-embedding-3-small",
            dimensions=0,
        )


def test_embed_rejects_response_count_mismatch() -> None:
    sdk_client, _ = create_sdk_client([SimpleNamespace(index=0, embedding=[0.1, 0.2])])
    client = OpenAIEmbeddingClient(
        client=sdk_client,
        model="text-embedding-3-small",
        dimensions=2,
    )

    with pytest.raises(
        InvalidEmbeddingResponseError,
        match="response count did not match input count",
    ):
        asyncio.run(client.embed(["first", "second"]))


def test_embed_rejects_unexpected_dimensions() -> None:
    sdk_client, _ = create_sdk_client([SimpleNamespace(index=0, embedding=[0.1])])
    client = OpenAIEmbeddingClient(
        client=sdk_client,
        model="text-embedding-3-small",
        dimensions=2,
    )

    with pytest.raises(
        InvalidEmbeddingResponseError,
        match="unexpected dimension",
    ):
        asyncio.run(client.embed(["first"]))
