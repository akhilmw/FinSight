from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from finsight.db.models import Filing


class FilingRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_accession_number(
        self,
        accession_number: str,
    ) -> Filing | None:
        statement = select(Filing).where(Filing.accession_number == accession_number)

        return self._session.scalar(statement)

    def add(self, filing: Filing) -> Filing:
        self._session.add(filing)
        return filing

    def get_by_id(self, filing_id: int) -> Filing | None:
        return self._session.get(Filing, filing_id)

    def list_filings(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Filing]:
        statement = (
            select(Filing)
            .order_by(Filing.filing_date.desc(), Filing.id.desc())
            .offset(offset)
            .limit(limit)
        )

        return self._session.scalars(statement).all()
