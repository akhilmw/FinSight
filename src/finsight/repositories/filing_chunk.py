from collections.abc import Sequence

from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session

from finsight.db.models import (
    FilingChunkRecord,
    FilingSectionRecord,
)


class FilingChunkRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def list_by_filing_id(
        self,
        filing_id: int,
    ) -> Sequence[FilingChunkRecord]:
        statement = (
            select(FilingChunkRecord)
            .join(FilingSectionRecord)
            .where(FilingSectionRecord.filing_id == filing_id)
            .order_by(
                FilingSectionRecord.start_line,
                FilingChunkRecord.chunk_index,
            )
        )

        return self._session.scalars(statement).all()

    def delete_by_filing_id(self, filing_id: int) -> None:
        section_ids = select(FilingSectionRecord.id).where(
            FilingSectionRecord.filing_id == filing_id
        )

        statement = delete(FilingChunkRecord).where(FilingChunkRecord.section_id.in_(section_ids))

        self._session.execute(statement)

    def list_requiring_embedding(
        self,
        filing_id: int,
        embedding_model: str,
    ) -> Sequence[FilingChunkRecord]:
        statement = (
            select(FilingChunkRecord)
            .join(FilingSectionRecord)
            .where(
                FilingSectionRecord.filing_id == filing_id,
                or_(
                    FilingChunkRecord.embedding.is_(None),
                    FilingChunkRecord.embedding_model.is_(None),
                    FilingChunkRecord.embedding_model != embedding_model,
                ),
            )
            .order_by(
                FilingSectionRecord.start_line,
                FilingChunkRecord.chunk_index,
            )
        )

        return self._session.scalars(statement).all()

    def add_all(
        self,
        chunks: list[FilingChunkRecord],
    ) -> None:
        self._session.add_all(chunks)
