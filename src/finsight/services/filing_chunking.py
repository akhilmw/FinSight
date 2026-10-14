from dataclasses import dataclass

from sqlalchemy.orm import Session

from finsight.chunking.section import SectionChunker
from finsight.db.models import FilingChunkRecord
from finsight.repositories.filing import FilingRepository
from finsight.repositories.filing_chunk import FilingChunkRepository
from finsight.repositories.filing_section import FilingSectionRepository


class FilingNotFoundError(Exception):
    pass


class FilingHasNoSectionsError(Exception):
    pass


@dataclass(frozen=True)
class FilingChunkingResult:
    filing_id: int
    sections_processed: int
    created: int
    replaced: int


class FilingChunkingService:
    def __init__(
        self,
        session: Session,
        chunker: SectionChunker,
    ) -> None:
        self._session = session
        self._chunker = chunker
        self._filing_repository = FilingRepository(session)
        self._section_repository = FilingSectionRepository(session)
        self._chunk_repository = FilingChunkRepository(session)

    def chunk_filing(
        self,
        filing_id: int,
    ) -> FilingChunkingResult:
        filing = self._filing_repository.get_by_id(filing_id)

        if filing is None:
            raise FilingNotFoundError(f"Filing {filing_id} was not found")

        sections = self._section_repository.list_by_filing_id(filing_id)

        if not sections:
            raise FilingHasNoSectionsError(f"Filing {filing_id} has no processed sections")

        existing_chunks = self._chunk_repository.list_by_filing_id(filing_id)

        records: list[FilingChunkRecord] = []

        for section in sections:
            chunks = self._chunker.chunk(section.text)

            records.extend(
                FilingChunkRecord(
                    section_id=section.id,
                    chunk_index=chunk.index,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    start_token=chunk.start_token,
                    end_token=chunk.end_token,
                )
                for chunk in chunks
            )

        try:
            self._chunk_repository.delete_by_filing_id(filing_id)
            self._chunk_repository.add_all(records)

            filing.status = "chunked"

            self._session.commit()
        except Exception:
            self._session.rollback()
            raise

        return FilingChunkingResult(
            filing_id=filing_id,
            sections_processed=len(sections),
            created=len(records),
            replaced=len(existing_chunks),
        )
