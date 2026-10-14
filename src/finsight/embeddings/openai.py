from collections.abc import Sequence

from openai import AsyncOpenAI

from finsight.embeddings.base import EmbeddingClient


class InvalidEmbeddingResponseError(RuntimeError):
    pass


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        dimensions: int,
    ) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be at least 1")

        self._client = client
        self._model = model
        self._dimensions = dimensions

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError("embedding input must not be blank")

        response = await self._client.embeddings.create(
            input=list(texts),
            model=self._model,
            dimensions=self._dimensions,
            encoding_format="float",
        )

        ordered_data = sorted(
            response.data,
            key=lambda item: item.index,
        )
        embeddings = [item.embedding for item in ordered_data]

        if len(embeddings) != len(texts):
            raise InvalidEmbeddingResponseError(
                "Embedding response count did not match input count"
            )

        if any(len(embedding) != self._dimensions for embedding in embeddings):
            raise InvalidEmbeddingResponseError(
                "Embedding response contained an unexpected dimension"
            )

        return embeddings
