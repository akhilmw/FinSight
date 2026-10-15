from collections.abc import Iterator, Sequence
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from finsight.api.dependencies import (
    get_document_store,
    get_embedding_client,
    get_sec_client,
)
from finsight.clients.sec import SecClient, SecCompany, SecFiling
from finsight.db.base import Base
from finsight.db.session import get_db
from finsight.embeddings import EmbeddingClient
from finsight.main import app
from finsight.schemas import SearchMode
from finsight.services.filing_search import (
    FilingSearchMatch,
    FilingSearchResult,
    FilingSearchService,
)
from finsight.storage.local import LocalDocumentStore

FAKE_FILING_HTML = b"""
<html><body>
  <h1>Item 1. Business</h1>
  <p>We design accelerated computing platforms.</p>
  <h1>Item 1A. Risk Factors</h1>
  <p>Competition may adversely affect our business.</p>
</body></html>
"""


class FakeEmbeddingClient:
    model = "text-embedding-3-small"
    dimensions = 1536

    async def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return [[0.1] * self.dimensions for _ in texts]


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    testing_session = sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )

    Base.metadata.create_all(bind=engine)

    fake_sec_client = AsyncMock(spec=SecClient)
    fake_sec_client.get_company.return_value = SecCompany(
        ticker="NVDA",
        cik="0001045810",
        name="NVIDIA CORP",
    )
    fake_sec_client.get_recent_filings.return_value = [
        SecFiling(
            accession_number="0001045810-26-000021",
            cik="0001045810",
            company_name="NVIDIA CORP",
            form_type="10-K",
            filing_date=date(2026, 2, 25),
            report_period=date(2026, 1, 25),
            primary_document="nvda-20260125.htm",
            source_url=(
                "https://www.sec.gov/Archives/edgar/data/"
                "1045810/000104581026000021/nvda-20260125.htm"
            ),
        )
    ]
    fake_sec_client.download_filing_html.return_value = FAKE_FILING_HTML
    fake_embedding_client = FakeEmbeddingClient()

    def override_get_db() -> Iterator[Session]:
        with testing_session() as session:
            yield session

    async def override_get_sec_client() -> SecClient:
        return cast(SecClient, fake_sec_client)

    def override_get_document_store() -> LocalDocumentStore:
        return LocalDocumentStore(tmp_path)

    async def override_get_embedding_client() -> EmbeddingClient:
        return fake_embedding_client

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_sec_client] = override_get_sec_client
    app.dependency_overrides[get_document_store] = override_get_document_store
    app.dependency_overrides[get_embedding_client] = override_get_embedding_client

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def test_ingest_list_and_retrieve_filing(client: TestClient) -> None:
    first_response = client.post(
        "/filings/ingest",
        json={
            "ticker": " nvda ",
            "form_types": ["10-K", "10-Q"],
            "limit": 5,
        },
    )

    assert first_response.status_code == 200
    assert first_response.json() == {
        "ticker": "NVDA",
        "discovered": 1,
        "created": 1,
        "skipped": 0,
    }

    second_response = client.post(
        "/filings/ingest",
        json={
            "ticker": "NVDA",
            "form_types": ["10-K", "10-Q"],
            "limit": 5,
        },
    )

    assert second_response.status_code == 200
    assert second_response.json() == {
        "ticker": "NVDA",
        "discovered": 1,
        "created": 0,
        "skipped": 1,
    }

    list_response = client.get("/filings")

    assert list_response.status_code == 200

    filings = list_response.json()
    assert len(filings) == 1
    assert filings[0]["accession_number"] == "0001045810-26-000021"
    assert filings[0]["cik"] == "0001045810"
    assert filings[0]["status"] == "discovered"

    filing_id = filings[0]["id"]
    detail_response = client.get(f"/filings/{filing_id}")

    assert detail_response.status_code == 200
    assert detail_response.json()["id"] == filing_id


def test_get_filing_returns_404_when_missing(client: TestClient) -> None:
    response = client.get("/filings/999")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Filing 999 was not found",
    }


def test_download_filing_updates_api_resource(client: TestClient) -> None:
    ingest_response = client.post(
        "/filings/ingest",
        json={
            "ticker": "NVDA",
            "form_types": ["10-K", "10-Q"],
            "limit": 5,
        },
    )
    assert ingest_response.status_code == 200

    filings = client.get("/filings").json()
    filing_id = filings[0]["id"]

    download_response = client.post(f"/filings/{filing_id}/download")

    assert download_response.status_code == 200
    download_result = download_response.json()
    expected_content = FAKE_FILING_HTML

    assert download_result["filing_id"] == filing_id
    assert download_result["sha256"] == sha256(expected_content).hexdigest()
    assert Path(download_result["path"]).read_bytes() == expected_content

    detail_response = client.get(f"/filings/{filing_id}")
    assert detail_response.status_code == 200

    filing = detail_response.json()
    assert filing["status"] == "downloaded"
    assert filing["raw_document_path"] == download_result["path"]
    assert filing["content_sha256"] == download_result["sha256"]
    assert filing["downloaded_at"] is not None


def test_download_filing_returns_404_when_missing(client: TestClient) -> None:
    response = client.post("/filings/999/download")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Filing 999 was not found",
    }


def test_process_chunk_and_embed_filing_end_to_end(
    client: TestClient,
) -> None:
    ingest_response = client.post(
        "/filings/ingest",
        json={
            "ticker": "NVDA",
            "form_types": ["10-K"],
            "limit": 1,
        },
    )
    assert ingest_response.status_code == 200

    filing_id = client.get("/filings").json()[0]["id"]

    assert client.post(f"/filings/{filing_id}/download").status_code == 200
    assert client.post(f"/filings/{filing_id}/process").json() == {
        "filing_id": filing_id,
        "created": 2,
        "replaced": 0,
    }

    first_chunk_response = client.post(f"/filings/{filing_id}/chunk")

    assert first_chunk_response.status_code == 200
    assert first_chunk_response.json() == {
        "filing_id": filing_id,
        "sections_processed": 2,
        "created": 2,
        "replaced": 0,
    }

    second_chunk_response = client.post(f"/filings/{filing_id}/chunk")

    assert second_chunk_response.status_code == 200
    assert second_chunk_response.json() == {
        "filing_id": filing_id,
        "sections_processed": 2,
        "created": 2,
        "replaced": 2,
    }
    first_embed_response = client.post(f"/filings/{filing_id}/embed")

    assert first_embed_response.status_code == 200
    assert first_embed_response.json() == {
        "filing_id": filing_id,
        "total_chunks": 2,
        "embedded": 2,
        "skipped": 0,
        "batches": 1,
    }

    second_embed_response = client.post(f"/filings/{filing_id}/embed")

    assert second_embed_response.status_code == 200
    assert second_embed_response.json() == {
        "filing_id": filing_id,
        "total_chunks": 2,
        "embedded": 0,
        "skipped": 2,
        "batches": 0,
    }
    assert client.get(f"/filings/{filing_id}").json()["status"] == "embedded"


def test_chunk_filing_returns_404_when_missing(client: TestClient) -> None:
    response = client.post("/filings/999/chunk")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Filing 999 was not found",
    }


def test_chunk_filing_returns_409_without_sections(
    client: TestClient,
) -> None:
    client.post(
        "/filings/ingest",
        json={
            "ticker": "NVDA",
            "form_types": ["10-K"],
            "limit": 1,
        },
    )
    filing_id = client.get("/filings").json()[0]["id"]

    response = client.post(f"/filings/{filing_id}/chunk")

    assert response.status_code == 409
    assert response.json() == {
        "detail": f"Filing {filing_id} has no processed sections",
    }


def test_embed_filing_returns_404_when_missing(client: TestClient) -> None:
    response = client.post("/filings/999/embed")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Filing 999 was not found",
    }


def test_embed_filing_returns_409_without_chunks(
    client: TestClient,
) -> None:
    client.post(
        "/filings/ingest",
        json={
            "ticker": "NVDA",
            "form_types": ["10-K"],
            "limit": 1,
        },
    )
    filing_id = client.get("/filings").json()[0]["id"]

    response = client.post(f"/filings/{filing_id}/embed")

    assert response.status_code == 409
    assert response.json() == {
        "detail": f"Filing {filing_id} has no stored chunks",
    }


@pytest.mark.parametrize("filing_id", [None, 3])
def test_search_returns_ranked_evidence_with_optional_filing_filter(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    filing_id: int | None,
) -> None:
    captured: dict[str, object] = {}

    async def fake_search(
        service: FilingSearchService,
        query: str,
        mode: SearchMode = SearchMode.VECTOR,
        limit: int = 5,
        filing_id: int | None = None,
    ) -> FilingSearchResult:
        captured.update(
            query=query,
            mode=mode,
            limit=limit,
            filing_id=filing_id,
        )
        return FilingSearchResult(
            query=query,
            mode=mode,
            matches=[
                FilingSearchMatch(
                    chunk_id=487,
                    filing_id=3,
                    accession_number="0001045810-26-000021",
                    company_name="NVIDIA CORP",
                    form_type="10-K",
                    filing_date=date(2026, 2, 25),
                    source_url="https://www.sec.gov/example.htm",
                    item="1A",
                    title="Risk Factors",
                    text="Competition may adversely affect revenue.",
                    score=0.8,
                    distance=0.2,
                    similarity=0.8,
                )
            ],
        )

    monkeypatch.setattr(FilingSearchService, "search", fake_search)

    payload: dict[str, object] = {
        "query": "revenue risks",
        "mode": SearchMode.VECTOR,
        "limit": 3,
    }
    if filing_id is not None:
        payload["filing_id"] = filing_id

    response = client.post("/search", json=payload)

    assert response.status_code == 200
    assert captured == {
        "query": "revenue risks",
        "mode": SearchMode.VECTOR,
        "limit": 3,
        "filing_id": filing_id,
    }
    assert response.json() == {
        "query": "revenue risks",
        "mode": "vector",
        "matches": [
            {
                "chunk_id": 487,
                "filing_id": 3,
                "accession_number": "0001045810-26-000021",
                "company_name": "NVIDIA CORP",
                "form_type": "10-K",
                "filing_date": "2026-02-25",
                "source_url": "https://www.sec.gov/example.htm",
                "item": "1A",
                "title": "Risk Factors",
                "text": "Competition may adversely affect revenue.",
                "score": 0.8,
                "distance": 0.2,
                "similarity": 0.8,
            }
        ],
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "   "},
        {"query": "revenue", "limit": 0},
        {"query": "revenue", "limit": 21},
        {"query": "revenue", "filing_id": 0},
    ],
)
def test_search_rejects_invalid_request(
    client: TestClient,
    payload: dict[str, object],
) -> None:
    response = client.post("/search", json=payload)

    assert response.status_code == 422
