from datetime import date, datetime

from sqlalchemy import Date, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from finsight.db.base import Base


class Filing(Base):
    __tablename__ = "filings"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    accession_number: Mapped[str] = mapped_column(
        String(32),
        unique=True,
        nullable=False,
    )

    cik: Mapped[str] = mapped_column(
        String(10),
        index=True,
        nullable=False,
    )

    company_name: Mapped[str] = mapped_column(
        String(255),
        index=True,
        nullable=False,
    )

    form_type: Mapped[str] = mapped_column(
        String(16),
        index=True,
        nullable=False,
    )

    filing_date: Mapped[date] = mapped_column(
        Date,
        index=True,
        nullable=False,
    )

    report_period: Mapped[date | None] = mapped_column(
        Date,
        nullable=True,
    )

    source_url: Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(24),
        default="discovered",
        server_default="discovered",
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    raw_document_path: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
    )

    content_sha256: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    downloaded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
