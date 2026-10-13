from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finsight.db.base import Base

if TYPE_CHECKING:
    from finsight.db.models.filing import Filing


class FilingSectionRecord(Base):
    __tablename__ = "filing_sections"

    __table_args__ = (
        UniqueConstraint(
            "filing_id",
            "item",
            "title",
            name="uq_filing_sections_filing_item_title",
        ),
        CheckConstraint(
            "start_line >= 0",
            name="ck_filing_sections_start_line",
        ),
        CheckConstraint(
            "end_line > start_line",
            name="ck_filing_sections_line_range",
        ),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    filing_id: Mapped[int] = mapped_column(
        ForeignKey("filings.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    item: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )

    title: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    start_line: Mapped[int] = mapped_column(nullable=False)
    end_line: Mapped[int] = mapped_column(nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    filing: Mapped["Filing"] = relationship(
        back_populates="sections",
    )