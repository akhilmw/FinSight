import asyncio
from collections.abc import Sequence
from datetime import date

import pytest
from sqlalchemy.orm import Session

from finsight.db.models import Filing, FilingChunkRecord, FilingSectionRecord
from finsight.repositories.filing_chunk import (
    FilingChunkKeywordSearchResult,
    FilingChunkRepository,
    FilingChunkSearchResult,
)
from finsight.schemas import SearchMode
from finsight.services.filing_search import FilingSearchService


class FakeEmbeddingClient:
    model = "text-embedding-3-small"
    dimensions = 3

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]


def create_repository_result() -> FilingChunkSearchResult:
    filing = Filing(
        id=3,
        accession_number="0001045810-26-000021",
        cik="0001045810",
        company_name="NVIDIA CORP",
        form_type="10-K",
        filing_date=date(2026, 2, 25),
        report_period=date(2026, 1, 25),
        source_url="https://www.sec.gov/example.htm",
        status="embedded",
    )
    section = FilingSectionRecord(
        id=25,
        filing_id=filing.id,
        item="1A",
        title="Risk Factors",
        text="Risk factor section.",
        start_line=10,
        end_line=20,
    )
    chunk = FilingChunkRecord(
        id=487,
        section_id=section.id,
        chunk_index=0,
        text="Competition may adversely affect revenue.",
        token_count=7,
        start_token=0,
        end_token=7,
    )

    return FilingChunkSearchResult(
        chunk=chunk,
        section=section,
        filing=filing,
        distance=0.2,
    )


def create_keyword_repository_result() -> FilingChunkKeywordSearchResult:
    vector_result = create_repository_result()
    return FilingChunkKeywordSearchResult(
        chunk=vector_result.chunk,
        section=vector_result.section,
        filing=vector_result.filing,
        rank=0.42,
    )


def test_search_embeds_query_and_maps_ranked_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_search_similar(
        repository: FilingChunkRepository,
        query_embedding: list[float],
        embedding_model: str,
        limit: int = 5,
        filing_id: int | None = None,
    ) -> list[FilingChunkSearchResult]:
        captured.update(
            query_embedding=query_embedding,
            embedding_model=embedding_model,
            limit=limit,
            filing_id=filing_id,
        )
        return [create_repository_result()]

    monkeypatch.setattr(
        FilingChunkRepository,
        "search_similar",
        fake_search_similar,
    )

    embedding_client = FakeEmbeddingClient()
    service = FilingSearchService(
        session=Session(),
        embedding_client=embedding_client,
    )

    result = asyncio.run(
        service.search(
            query="  What threatens NVIDIA revenue?  ",
            limit=3,
            filing_id=3,
        )
    )

    assert embedding_client.calls == [["What threatens NVIDIA revenue?"]]
    assert captured == {
        "query_embedding": [0.1, 0.2, 0.3],
        "embedding_model": "text-embedding-3-small",
        "limit": 3,
        "filing_id": 3,
    }
    assert result.query == "What threatens NVIDIA revenue?"
    assert len(result.matches) == 1

    match = result.matches[0]
    assert match.chunk_id == 487
    assert match.filing_id == 3
    assert match.accession_number == "0001045810-26-000021"
    assert match.company_name == "NVIDIA CORP"
    assert match.form_type == "10-K"
    assert match.item == "1A"
    assert match.title == "Risk Factors"
    assert match.text == "Competition may adversely affect revenue."
    assert match.score == pytest.approx(0.8)
    assert match.distance == pytest.approx(0.2)
    assert match.similarity == pytest.approx(0.8)


def test_keyword_search_skips_embedding_and_maps_ranked_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_search_keyword(
        repository: FilingChunkRepository,
        query: str,
        limit: int = 5,
        filing_id: int | None = None,
    ) -> list[FilingChunkKeywordSearchResult]:
        captured.update(
            query=query,
            limit=limit,
            filing_id=filing_id,
        )
        return [create_keyword_repository_result()]

    monkeypatch.setattr(
        FilingChunkRepository,
        "search_keyword",
        fake_search_keyword,
    )

    embedding_client = FakeEmbeddingClient()
    service = FilingSearchService(
        session=Session(),
        embedding_client=embedding_client,
    )

    result = asyncio.run(
        service.search(
            query="  gaming revenue growth  ",
            mode=SearchMode.KEYWORD,
            limit=3,
            filing_id=3,
        )
    )

    assert embedding_client.calls == []
    assert captured == {
        "query": "gaming revenue growth",
        "limit": 3,
        "filing_id": 3,
    }
    assert result.mode is SearchMode.KEYWORD
    assert len(result.matches) == 1
    assert result.matches[0].score == pytest.approx(0.42)
    assert result.matches[0].distance is None
    assert result.matches[0].similarity is None


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_search_rejects_blank_query(query: str) -> None:
    embedding_client = FakeEmbeddingClient()
    service = FilingSearchService(
        session=Session(),
        embedding_client=embedding_client,
    )

    with pytest.raises(ValueError, match="query must not be blank"):
        asyncio.run(service.search(query))

    assert embedding_client.calls == []


def test_search_rejects_invalid_limit() -> None:
    embedding_client = FakeEmbeddingClient()
    service = FilingSearchService(
        session=Session(),
        embedding_client=embedding_client,
    )

    with pytest.raises(ValueError, match="limit must be at least 1"):
        asyncio.run(service.search("revenue risks", limit=0))

    assert embedding_client.calls == []
