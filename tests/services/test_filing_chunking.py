from collections.abc import Iterator
from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from finsight.chunking.section import SectionChunker
from finsight.db.base import Base
from finsight.db.models import Filing, FilingSectionRecord
from finsight.repositories.filing_chunk import FilingChunkRepository
from finsight.services.filing_chunking import (
    FilingChunkingService,
    FilingHasNoSectionsError,
    FilingNotFoundError,
)


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


def create_filing(session: Session) -> Filing:
    filing = Filing(
        accession_number="0001045810-26-000021",
        cik="0001045810",
        company_name="NVIDIA CORP",
        form_type="10-K",
        filing_date=date(2026, 2, 25),
        report_period=date(2026, 1, 25),
        source_url=(
            "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm"
        ),
        status="parsed",
    )
    session.add(filing)
    session.flush()

    session.add_all(
        [
            FilingSectionRecord(
                filing_id=filing.id,
                item="1",
                title="Business",
                text="Revenue increased significantly. " * 20,
                start_line=1,
                end_line=20,
            ),
            FilingSectionRecord(
                filing_id=filing.id,
                item="1A",
                title="Risk Factors",
                text="Competition remains intense. " * 20,
                start_line=21,
                end_line=40,
            ),
        ]
    )
    session.commit()
    session.refresh(filing)

    return filing


def create_service(session: Session) -> FilingChunkingService:
    return FilingChunkingService(
        session=session,
        chunker=SectionChunker(
            max_tokens=12,
            overlap_tokens=3,
        ),
    )


def test_chunk_filing_persists_ordered_chunks_and_supports_rechunking(
    session: Session,
) -> None:
    filing = create_filing(session)
    service = create_service(session)
    repository = FilingChunkRepository(session)

    first_result = service.chunk_filing(filing.id)
    stored_chunks = repository.list_by_filing_id(filing.id)

    session.refresh(filing)

    assert first_result.filing_id == filing.id
    assert first_result.sections_processed == 2
    assert first_result.created == len(stored_chunks)
    assert first_result.created > 2
    assert first_result.replaced == 0
    assert filing.status == "chunked"
    assert all(chunk.token_count <= 12 for chunk in stored_chunks)

    section_indexes: dict[int, list[int]] = {}
    for chunk in stored_chunks:
        section_indexes.setdefault(chunk.section_id, []).append(chunk.chunk_index)

    assert all(indexes == list(range(len(indexes))) for indexes in section_indexes.values())

    second_result = service.chunk_filing(filing.id)
    chunks_after_rechunking = repository.list_by_filing_id(filing.id)

    assert second_result.created == first_result.created
    assert second_result.replaced == first_result.created
    assert len(chunks_after_rechunking) == first_result.created


def test_chunk_filing_rejects_missing_filing(session: Session) -> None:
    with pytest.raises(
        FilingNotFoundError,
        match="Filing 999 was not found",
    ):
        create_service(session).chunk_filing(999)


def test_chunk_filing_rejects_filing_without_sections(
    session: Session,
) -> None:
    filing = Filing(
        accession_number="0001045810-26-000052",
        cik="0001045810",
        company_name="NVIDIA CORP",
        form_type="10-Q",
        filing_date=date(2026, 5, 20),
        report_period=date(2026, 4, 26),
        source_url="https://www.sec.gov/example.htm",
        status="downloaded",
    )
    session.add(filing)
    session.commit()
    session.refresh(filing)

    with pytest.raises(
        FilingHasNoSectionsError,
        match=f"Filing {filing.id} has no processed sections",
    ):
        create_service(session).chunk_filing(filing.id)
