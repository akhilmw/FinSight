import asyncio
from collections.abc import Iterator
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from finsight.clients.sec import SecClient
from finsight.db.base import Base
from finsight.db.models import Filing
from finsight.services.filing_download import (
    FilingDownloadService,
    FilingNotFoundError,
)
from finsight.storage.local import LocalDocumentStore


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
            "https://www.sec.gov/Archives/edgar/data/"
            "1045810/000104581026000021/nvda-20260125.htm"
        ),
        status="discovered",
    )
    session.add(filing)
    session.commit()
    session.refresh(filing)

    return filing


def test_download_saves_html_and_updates_filing(
    session: Session,
    tmp_path: Path,
) -> None:
    filing = create_filing(session)
    fake_html = b"<html><body>NVIDIA filing</body></html>"

    fake_sec_client = AsyncMock(spec=SecClient)
    fake_sec_client.download_filing_html.return_value = fake_html

    service = FilingDownloadService(
        session=session,
        sec_client=cast(SecClient, fake_sec_client),
        document_store=LocalDocumentStore(tmp_path),
    )

    result = asyncio.run(service.download(filing.id))

    assert result.filing_id == filing.id
    assert result.path == filing.raw_document_path
    assert result.sha256 == filing.content_sha256
    assert result.sha256 == sha256(fake_html).hexdigest()
    assert filing.status == "downloaded"
    assert filing.downloaded_at is not None
    assert Path(result.path).read_bytes() == fake_html
    fake_sec_client.download_filing_html.assert_awaited_once_with(
        filing.source_url
    )


def test_download_raises_when_filing_does_not_exist(
    session: Session,
    tmp_path: Path,
) -> None:
    fake_sec_client = AsyncMock(spec=SecClient)
    service = FilingDownloadService(
        session=session,
        sec_client=cast(SecClient, fake_sec_client),
        document_store=LocalDocumentStore(tmp_path),
    )

    with pytest.raises(FilingNotFoundError, match="Filing 999 was not found"):
        asyncio.run(service.download(999))

    fake_sec_client.download_filing_html.assert_not_awaited()
