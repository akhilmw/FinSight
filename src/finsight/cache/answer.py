import hashlib
import json

from redis.asyncio import Redis

from finsight.schemas import FilingAnswerRequest, FilingAnswerResponse


class RedisAnswerCache:
    def __init__(
        self,
        client: Redis,
        ttl_seconds: int,
        answer_model: str,
    ) -> None:
        self._client = client
        self._ttl_seconds = ttl_seconds
        self._answer_model = answer_model

    def build_key(self, request: FilingAnswerRequest) -> str:
        request_data = {
            "model": self._answer_model,
            "request": request.model_dump(
                mode="json",
                exclude_none=False,
            ),
        }

        serialized = json.dumps(
            request_data,
            sort_keys=True,
            separators=(",", ":"),
        )

        request_hash = hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

        return f"finsight:answers:v1:{request_hash}"

    async def get(
        self,
        request: FilingAnswerRequest,
    ) -> FilingAnswerResponse | None:
        key = self.build_key(request)
        cached_value = await self._client.get(key)

        if cached_value is None:
            return None

        return FilingAnswerResponse.model_validate_json(cached_value)

    async def set(
        self,
        request: FilingAnswerRequest,
        response: FilingAnswerResponse,
    ) -> None:
        key = self.build_key(request)

        await self._client.set(
            key,
            response.model_dump_json(),
            ex=self._ttl_seconds,
        )