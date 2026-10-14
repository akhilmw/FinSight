import asyncio
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from finsight.db.models import FilingSectionRecord
from finsight.parsers.sec_html import SecHtmlParser
from finsight.parsers.sec_sections import FilingSection, SecSectionDetector
from finsight.repositories.filing import FilingRepository
from finsight.repositories.filing_section import FilingSectionRepository


class FilingNotReadyError(Exception):
    pass


class FilingNotFoundError(FilingNotReadyError):
    pass


class FilingNotDownloadedError(FilingNotReadyError):
    pass


class RawDocumentNotFoundError(Exception):
    pass


class NoSectionsFoundError(Exception):
    pass


@dataclass(frozen=True)
class FilingProcessingResult:
    filing_id: int
    created: int
    replaced: int


class FilingProcessingService:
    def __init__(
        self,
        session: Session,
        html_parser: SecHtmlParser,
        section_detector: SecSectionDetector,
    ) -> None:
        self._session = session
        self._html_parser = html_parser
        self._section_detector = section_detector
        self._filing_repository = FilingRepository(session)
        self._section_repository = FilingSectionRepository(session)

    async def process(
        self,
        filing_id: int,
    ) -> FilingProcessingResult:
        filing = self._filing_repository.get_by_id(filing_id)

        if filing is None:
            raise FilingNotFoundError(f"Filing {filing_id} was not found")

        if filing.raw_document_path is None:
            raise FilingNotDownloadedError(f"Filing {filing_id} has not been downloaded")

        document_path = Path(filing.raw_document_path)

        if not document_path.is_file():
            raise RawDocumentNotFoundError(f"Raw document does not exist: {document_path}")

        content = await asyncio.to_thread(document_path.read_bytes)
        extracted_sections = await asyncio.to_thread(
            self._extract_sections,
            content,
        )

        if not extracted_sections:
            raise NoSectionsFoundError(f"No sections were detected for filing {filing_id}")

        existing_sections = self._section_repository.list_by_filing_id(filing_id)

        records = [
            FilingSectionRecord(
                filing_id=filing_id,
                item=section.item,
                title=section.title,
                text=section.text,
                start_line=section.start_line,
                end_line=section.end_line,
            )
            for section in extracted_sections
        ]

        try:
            self._section_repository.delete_by_filing_id(filing_id)
            self._section_repository.add_all(records)

            filing.status = "parsed"

            self._session.commit()
        except Exception:
            self._session.rollback()
            raise

        return FilingProcessingResult(
            filing_id=filing_id,
            created=len(records),
            replaced=len(existing_sections),
        )

    def _extract_sections(
        self,
        content: bytes,
    ) -> list[FilingSection]:
        parsed_filing = self._html_parser.parse(content)
        return self._section_detector.extract_sections(parsed_filing)
