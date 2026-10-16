from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

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

@dataclass(frozen=True)
class FilingChunkKeywordSearchResult:
    chunk: FilingChunkRecord
    section: FilingSectionRecord
    filing: Filing
    rank: float


def _filing_filter_conditions(
    filing_id: int | None = None,
    company_name: str | None = None,
    year: int | None = None,
    form_type: str | None = None,
) -> list[ColumnElement[bool]]:
    conditions: list[ColumnElement[bool]] = []

    if filing_id is not None:
        conditions.append(Filing.id == filing_id)

    if company_name is not None:
        normalized_company = company_name.strip()

        if normalized_company:
            conditions.append(
                Filing.company_name.ilike(f"%{normalized_company}%")
            )

    if year is not None:
        if not 1900 <= year <= 2100:
            raise ValueError("year must be between 1900 and 2100")

        conditions.extend(
            [
                Filing.filing_date >= date(year, 1, 1),
                Filing.filing_date < date(year + 1, 1, 1),
            ]
        )

    if form_type is not None:
        conditions.append(Filing.form_type == form_type)

    return conditions


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
        company_name: str | None = None,
        year: int | None = None,
        form_type: str | None = None,
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
        statement = statement.where(
            *_filing_filter_conditions(
                filing_id=filing_id,
                company_name=company_name,
                year=year,
                form_type=form_type,
            )
        )

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

    def search_keyword(
        self,
        query: str,
        limit: int = 5,
        filing_id: int | None = None,
        company_name: str | None = None,
        year: int | None = None,
        form_type: str | None = None,
    ) -> list[FilingChunkKeywordSearchResult]:
        if limit < 1:
            raise ValueError("limit must be at least 1")

        search_vector = func.to_tsvector(
            "english",
            FilingChunkRecord.text,
        )
        search_query = func.websearch_to_tsquery(
            "english",
            query,
        )
        rank = func.ts_rank_cd(search_vector, search_query)

        statement = (
            select(
                FilingChunkRecord,
                FilingSectionRecord,
                Filing,
                rank.label("rank"),
            )
            .join(
                FilingSectionRecord,
                FilingChunkRecord.section_id == FilingSectionRecord.id,
            )
            .join(
                Filing,
                FilingSectionRecord.filing_id == Filing.id,
            )
            .where(search_vector.op("@@")(search_query))
        )
        statement = statement.where(
            *_filing_filter_conditions(
                filing_id=filing_id,
                company_name=company_name,
                year=year,
                form_type=form_type,
            )
        )

        statement = statement.order_by(
            rank.desc(),
            FilingChunkRecord.id,
        ).limit(limit)

        rows = self._session.execute(statement)

        return [
            FilingChunkKeywordSearchResult(
                chunk=chunk,
                section=section,
                filing=filing,
                rank=float(rank_value),
            )
            for chunk, section, filing, rank_value in rows
        ]
