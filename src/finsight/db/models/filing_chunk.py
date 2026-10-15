from datetime import datetime
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finsight.db.base import Base

if TYPE_CHECKING:
    from finsight.db.models.filing_section import FilingSectionRecord


class FilingChunkRecord(Base):
    __tablename__ = "filing_chunks"

    __table_args__ = (
        UniqueConstraint(
            "section_id",
            "chunk_index",
            name="uq_filing_chunks_section_index",
        ),
        CheckConstraint(
            "chunk_index >= 0",
            name="ck_filing_chunks_chunk_index",
        ),
        CheckConstraint(
            "token_count > 0",
            name="ck_filing_chunks_token_count",
        ),
        CheckConstraint(
            "start_token >= 0",
            name="ck_filing_chunks_start_token",
        ),
        CheckConstraint(
            "end_token > start_token",
            name="ck_filing_chunks_token_range",
        ),
        Index(
            "ix_filing_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={
                "embedding": "vector_cosine_ops",
            },
        ),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    section_id: Mapped[int] = mapped_column(
        ForeignKey("filing_sections.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    chunk_index: Mapped[int] = mapped_column(nullable=False)

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(1536),
        nullable=True,
    )

    embedding_model: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    embedded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    token_count: Mapped[int] = mapped_column(nullable=False)
    start_token: Mapped[int] = mapped_column(nullable=False)
    end_token: Mapped[int] = mapped_column(nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    section: Mapped["FilingSectionRecord"] = relationship(
        back_populates="chunks",
    )
