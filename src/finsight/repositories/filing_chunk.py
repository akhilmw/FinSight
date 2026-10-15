from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session

from finsight.db.models import (
    Filing,
    FilingChunkRecord,
    FilingSectionRecord,
)


@dataclass(frozen=True)
class FilingChunkSearchResult:
    chunk: FilingChunkRecord
    section: FilingSectionRecord
    filing: Filing
    distance: float


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

    def search_similar(
        self,
        query_embedding: list[float],
        embedding_model: str,
        limit: int = 5,
        filing_id: int | None = None,
    ) -> list[FilingChunkSearchResult]:
        if limit < 1:
            raise ValueError("limit must be at least 1")

        distance = FilingChunkRecord.embedding.cosine_distance(query_embedding)

        statement = (
            select(
                FilingChunkRecord,
                FilingSectionRecord,
                Filing,
                distance.label("distance"),
            )
            .join(
                FilingSectionRecord,
                FilingChunkRecord.section_id == FilingSectionRecord.id,
            )
            .join(
                Filing,
                FilingSectionRecord.filing_id == Filing.id,
            )
            .where(
                FilingChunkRecord.embedding.is_not(None),
                FilingChunkRecord.embedding_model == embedding_model,
            )
        )

        if filing_id is not None:
            statement = statement.where(Filing.id == filing_id)

        statement = statement.order_by(distance).limit(limit)
        rows = self._session.execute(statement)

        return [
            FilingChunkSearchResult(
                chunk=chunk,
                section=section,
                filing=filing,
                distance=float(distance_value),
            )
            for chunk, section, filing, distance_value in rows
        ]
