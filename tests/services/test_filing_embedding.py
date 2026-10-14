import asyncio
from collections.abc import Iterator, Sequence
from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from finsight.db.base import Base
from finsight.db.models import (
    Filing,
    FilingChunkRecord,
    FilingSectionRecord,
)
from finsight.repositories.filing_chunk import FilingChunkRepository
from finsight.services.filing_embedding import (
    FilingEmbeddingService,
    FilingHasNoChunksError,
    FilingNotFoundError,
)


class FakeEmbeddingClient:
    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        fail_on_call: int | None = None,
    ) -> None:
        self.model = model
        self.dimensions = 1536
        self.fail_on_call = fail_on_call
        self.calls: list[list[str]] = []

    async def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        self.calls.append(list(texts))

        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("embedding provider failed")

        return [[float(index + 1)] * self.dimensions for index, _ in enumerate(texts)]


@pytest.fixture
def session() -> Iterator[Session]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)

    with Session(engine) as database_session:
        yield database_session

    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def create_filing_with_chunks(
    session: Session,
    *,
    chunk_count: int = 5,
) -> Filing:
    filing = Filing(
        accession_number="0001045810-26-000021",
        cik="0001045810",
        company_name="NVIDIA CORP",
        form_type="10-K",
        filing_date=date(2026, 2, 25),
        report_period=date(2026, 1, 25),
        source_url="https://www.sec.gov/example.htm",
        status="chunked",
    )
    session.add(filing)
    session.flush()

    section = FilingSectionRecord(
        filing_id=filing.id,
        item="1A",
        title="Risk Factors",
        text="Risk factor text.",
        start_line=1,
        end_line=10,
    )
    session.add(section)
    session.flush()

    session.add_all(
        [
            FilingChunkRecord(
                section_id=section.id,
                chunk_index=index,
                text=f"Risk factor chunk {index}.",
                token_count=5,
                start_token=index * 4,
                end_token=index * 4 + 5,
            )
            for index in range(chunk_count)
        ]
    )
    session.commit()
    session.refresh(filing)

    return filing


def test_embed_filing_batches_persists_and_skips_completed_chunks(
    session: Session,
) -> None:
    filing = create_filing_with_chunks(session)
    embedding_client = FakeEmbeddingClient()
    service = FilingEmbeddingService(
        session=session,
        embedding_client=embedding_client,
        batch_size=2,
    )

    first_result = asyncio.run(service.embed_filing(filing.id))
    stored_chunks = FilingChunkRepository(session).list_by_filing_id(filing.id)

    session.refresh(filing)

    assert first_result.total_chunks == 5
    assert first_result.embedded == 5
    assert first_result.skipped == 0
    assert first_result.batches == 3
    assert [len(call) for call in embedding_client.calls] == [2, 2, 1]
    assert filing.status == "embedded"
    assert all(chunk.embedding is not None for chunk in stored_chunks)
    assert all(chunk.embedding_model == "text-embedding-3-small" for chunk in stored_chunks)
    assert all(chunk.embedded_at is not None for chunk in stored_chunks)

    second_result = asyncio.run(service.embed_filing(filing.id))

    assert second_result.total_chunks == 5
    assert second_result.embedded == 0
    assert second_result.skipped == 5
    assert second_result.batches == 0
    assert [len(call) for call in embedding_client.calls] == [2, 2, 1]


def test_embed_filing_resumes_after_partial_failure(
    session: Session,
) -> None:
    filing = create_filing_with_chunks(session)
    failing_client = FakeEmbeddingClient(fail_on_call=2)
    failing_service = FilingEmbeddingService(
        session=session,
        embedding_client=failing_client,
        batch_size=2,
    )

    with pytest.raises(RuntimeError, match="embedding provider failed"):
        asyncio.run(failing_service.embed_filing(filing.id))

    partially_embedded = FilingChunkRepository(session).list_requiring_embedding(
        filing_id=filing.id,
        embedding_model=failing_client.model,
    )
    assert len(partially_embedded) == 3

    retry_client = FakeEmbeddingClient()
    retry_service = FilingEmbeddingService(
        session=session,
        embedding_client=retry_client,
        batch_size=2,
    )

    retry_result = asyncio.run(retry_service.embed_filing(filing.id))

    assert retry_result.total_chunks == 5
    assert retry_result.embedded == 3
    assert retry_result.skipped == 2
    assert retry_result.batches == 2
    assert [len(call) for call in retry_client.calls] == [2, 1]


def test_embed_filing_rejects_missing_filing(session: Session) -> None:
    service = FilingEmbeddingService(
        session=session,
        embedding_client=FakeEmbeddingClient(),
    )

    with pytest.raises(
        FilingNotFoundError,
        match="Filing 999 was not found",
    ):
        asyncio.run(service.embed_filing(999))


def test_embed_filing_rejects_filing_without_chunks(
    session: Session,
) -> None:
    filing = create_filing_with_chunks(session, chunk_count=0)
    service = FilingEmbeddingService(
        session=session,
        embedding_client=FakeEmbeddingClient(),
    )

    with pytest.raises(
        FilingHasNoChunksError,
        match=f"Filing {filing.id} has no stored chunks",
    ):
        asyncio.run(service.embed_filing(filing.id))


def test_embedding_service_rejects_invalid_batch_size(
    session: Session,
) -> None:
    with pytest.raises(
        ValueError,
        match="batch_size must be at least 1",
    ):
        FilingEmbeddingService(
            session=session,
            embedding_client=FakeEmbeddingClient(),
            batch_size=0,
        )
