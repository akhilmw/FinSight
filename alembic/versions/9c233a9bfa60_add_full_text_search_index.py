"""add full text search index

Revision ID: 9c233a9bfa60
Revises: 17088099b54b
Create Date: 2026-10-14 20:41:57.414428

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '9c233a9bfa60'
down_revision: str | Sequence[str] | None = '17088099b54b'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the full-text-search GIN index."""
    op.execute(
        """
        CREATE INDEX ix_filing_chunks_text_fts
        ON filing_chunks
        USING gin (to_tsvector('english', text))
        """
    )


def downgrade() -> None:
    """Remove the full-text-search GIN index."""
    op.drop_index(
        "ix_filing_chunks_text_fts",
        table_name="filing_chunks",
    )
