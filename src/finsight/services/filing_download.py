import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from finsight.clients.sec import SecClient
from finsight.repositories.filing import FilingRepository
from finsight.storage.local import LocalDocumentStore


class FilingNotFoundError(Exception):
    pass


@dataclass(frozen=True)
class FilingDownloadResult:
    filing_id: int
    path: str
    sha256: str


class FilingDownloadService:
    def __init__(
        self,
        session: Session,
        sec_client: SecClient,
        document_store: LocalDocumentStore,
    ) -> None:
        self._session = session
        self._sec_client = sec_client
        self._document_store = document_store
        self._repository = FilingRepository(session)

    async def download(self, filing_id: int) -> FilingDownloadResult:
        filing = self._repository.get_by_id(filing_id)

        if filing is None:
            raise FilingNotFoundError(f"Filing {filing_id} was not found")

        try:
            content = await self._sec_client.download_filing_html(
                filing.source_url
            )

            stored_document = await asyncio.to_thread(
                self._document_store.save,
                cik=filing.cik,
                accession_number=filing.accession_number,
                content=content,
            )

            filing.raw_document_path = stored_document.path
            filing.content_sha256 = stored_document.sha256
            filing.downloaded_at = datetime.now(UTC)
            filing.status = "downloaded"

            self._session.commit()
            self._session.refresh(filing)

        except Exception:
            self._session.rollback()
            raise

        return FilingDownloadResult(
            filing_id=filing.id,
            path=stored_document.path,
            sha256=stored_document.sha256,
        )