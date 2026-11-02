"""add keyword search indexes

Revision ID: e7f3a9c2b1d5
Revises: d4e8b2a1c6f0
Create Date: 2026-11-02 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'e7f3a9c2b1d5'
down_revision: Union[str, Sequence[str], None] = 'd4e8b2a1c6f0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Expression indexes, so the tables are not rewritten. The search query
    # must use exactly the same expressions for PostgreSQL to use them.
    # Titles use the english configuration (stemming, stop words).
    # Author names use simple, because names must not be stemmed.
    op.execute(
        """
        CREATE INDEX books_title_fts_idx
        ON books
        USING gin (to_tsvector('english', title))
        """
    )
    op.execute(
        """
        CREATE INDEX authors_name_fts_idx
        ON authors
        USING gin (to_tsvector('simple', name))
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS authors_name_fts_idx")
    op.execute("DROP INDEX IF EXISTS books_title_fts_idx")
