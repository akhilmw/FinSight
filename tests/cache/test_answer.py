import asyncio
from datetime import date
from typing import cast
from unittest.mock import AsyncMock, MagicMock

from redis.asyncio import Redis

from finsight.cache import RedisAnswerCache
from finsight.schemas import (
    AnswerCitationResponse,
    FilingAnswerRequest,
    FilingAnswerResponse,
)


def create_cache(
    cached_value: str | None = None,
) -> tuple[RedisAnswerCache, AsyncMock, AsyncMock]:
    get = AsyncMock(return_value=cached_value)
    set_value = AsyncMock()

    client = MagicMock()
    client.get = get
    client.set = set_value

    cache = RedisAnswerCache(
        client=cast(Redis, client),
        ttl_seconds=86_400,
        answer_model="test-answer-model",
    )

    return cache, get, set_value


def create_request() -> FilingAnswerRequest:
    return FilingAnswerRequest(
        question="What could negatively affect revenue?",
        filing_id=3,
        evidence_limit=5,
    )


def create_response() -> FilingAnswerResponse:
    return FilingAnswerResponse(
        question="What could negatively affect revenue?",
        answer="Competition and supply constraints could affect revenue.",
        sufficient_evidence=True,
        citations=[
            AnswerCitationResponse(
                chunk_id=512,
                filing_id=3,
                accession_number="0001045810-26-000021",
                company_name="NVIDIA CORP",
                form_type="10-K",
                filing_date=date(2026, 2, 25),
                source_url="https://www.sec.gov/example.htm",
                item="1A",
                title="Risk Factors",
            )
        ],
    )


def test_build_key_is_deterministic() -> None:
    cache, _, _ = create_cache()
    request = create_request()

    first_key = cache.build_key(request)
    second_key = cache.build_key(request)

    assert first_key == second_key
    assert first_key.startswith("finsight:answers:v1:")


def test_build_key_changes_when_request_changes() -> None:
    cache, _, _ = create_cache()

    first_request = create_request()
    second_request = FilingAnswerRequest(
        question="What could negatively affect revenue?",
        filing_id=4,
        evidence_limit=5,
    )

    assert cache.build_key(first_request) != cache.build_key(second_request)


def test_get_returns_none_on_cache_miss() -> None:
    cache, get, _ = create_cache(cached_value=None)
    request = create_request()

    result = asyncio.run(cache.get(request))

    assert result is None
    get.assert_awaited_once_with(cache.build_key(request))


def test_get_deserializes_cached_response() -> None:
    expected = create_response()
    cache, get, _ = create_cache(
        cached_value=expected.model_dump_json(),
    )
    request = create_request()

    result = asyncio.run(cache.get(request))

    assert result == expected
    get.assert_awaited_once_with(cache.build_key(request))


def test_set_serializes_response_with_ttl() -> None:
    cache, _, set_value = create_cache()
    request = create_request()
    response = create_response()

    asyncio.run(cache.set(request, response))

    set_value.assert_awaited_once_with(
        cache.build_key(request),
        response.model_dump_json(),
        ex=86_400,
    )
