from collections.abc import Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from finsight.db.models import FilingSectionRecord


class FilingSectionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def list_by_filing_id(
        self,
        filing_id: int,
    ) -> Sequence[FilingSectionRecord]:
        statement = (
            select(FilingSectionRecord)
            .where(FilingSectionRecord.filing_id == filing_id)
            .order_by(FilingSectionRecord.start_line)
        )

        return self._session.scalars(statement).all()

    def delete_by_filing_id(self, filing_id: int) -> None:
        statement = delete(FilingSectionRecord).where(FilingSectionRecord.filing_id == filing_id)
        self._session.execute(statement)

    def add_all(
        self,
        sections: list[FilingSectionRecord],
    ) -> None:
        self._session.add_all(sections)
