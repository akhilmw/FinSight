import asyncio
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from finsight.db.base import Base
from finsight.db.models import Filing
from finsight.parsers.sec_html import SecHtmlParser
from finsight.parsers.sec_sections import SecSectionDetector
from finsight.repositories.filing_section import FilingSectionRepository
from finsight.services.filing_processing import (
    FilingNotReadyError,
    FilingProcessingService,
    NoSectionsFoundError,
    RawDocumentNotFoundError,
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


def create_filing(
    session: Session,
    raw_document_path: str | None,
) -> Filing:
    filing = Filing(
        accession_number="0001045810-26-000021",
        cik="0001045810",
        company_name="NVIDIA CORP",
        form_type="10-K",
        filing_date=date(2026, 2, 25),
        report_period=date(2026, 1, 25),
        source_url=(
            "https://www.sec.gov/Archives/edgar/data/"
            "1045810/000104581026000021/nvda-20260125.htm"
        ),
        status="downloaded" if raw_document_path is not None else "discovered",
        raw_document_path=raw_document_path,
    )
    session.add(filing)
    session.commit()
    session.refresh(filing)

    return filing


def create_service(session: Session) -> FilingProcessingService:
    return FilingProcessingService(
        session=session,
        html_parser=SecHtmlParser(),
        section_detector=SecSectionDetector(),
    )


def test_process_persists_sections_and_supports_reprocessing(
    session: Session,
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "filing.html"
    document_path.write_bytes(
        b"""
        <html><body>
          <h2>Table of Contents</h2>
          <p>Item 1. Business</p>
          <p>3</p>
          <p>Item 1A. Risk Factors</p>
          <p>12</p>
          <h1>Item 1. Business</h1>
          <p>We design accelerated computing platforms.</p>
          <h1>Item 1A. Risk Factors</h1>
          <p>Competition may adversely affect our business.</p>
        </body></html>
        """
    )
    filing = create_filing(session, document_path.as_posix())
    service = create_service(session)

    first_result = asyncio.run(service.process(filing.id))

    session.refresh(filing)
    stored_sections = FilingSectionRepository(session).list_by_filing_id(
        filing.id
    )

    assert first_result.filing_id == filing.id
    assert first_result.created == 2
    assert first_result.replaced == 0
    assert filing.status == "parsed"
    assert [(section.item, section.title) for section in stored_sections] == [
        ("1", "Business"),
        ("1A", "Risk Factors"),
    ]
    assert stored_sections[0].text == (
        "We design accelerated computing platforms."
    )
    assert stored_sections[1].text == (
        "Competition may adversely affect our business."
    )

    second_result = asyncio.run(service.process(filing.id))
    sections_after_reprocessing = (
        FilingSectionRepository(session).list_by_filing_id(filing.id)
    )

    assert second_result.created == 2
    assert second_result.replaced == 2
    assert len(sections_after_reprocessing) == 2


def test_process_rejects_missing_filing(session: Session) -> None:
    with pytest.raises(
        FilingNotReadyError,
        match="Filing 999 was not found",
    ):
        asyncio.run(create_service(session).process(999))


def test_process_rejects_undownloaded_filing(session: Session) -> None:
    filing = create_filing(session, raw_document_path=None)

    with pytest.raises(
        FilingNotReadyError,
        match=f"Filing {filing.id} has not been downloaded",
    ):
        asyncio.run(create_service(session).process(filing.id))


def test_process_rejects_missing_raw_document(
    session: Session,
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.html"
    filing = create_filing(session, missing_path.as_posix())

    with pytest.raises(
        RawDocumentNotFoundError,
        match="Raw document does not exist",
    ):
        asyncio.run(create_service(session).process(filing.id))


def test_process_rejects_document_without_sections(
    session: Session,
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "filing.html"
    document_path.write_bytes(
        b"<html><body><p>No item headings here.</p></body></html>"
    )
    filing = create_filing(session, document_path.as_posix())

    with pytest.raises(
        NoSectionsFoundError,
        match=f"No sections were detected for filing {filing.id}",
    ):
        asyncio.run(create_service(session).process(filing.id))
