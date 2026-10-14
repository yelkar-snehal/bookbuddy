"""enable pgvector extension

Revision ID: bb4f1c2d8e9a
Revises: a3204c0a87fc
Create Date: 2026-10-14 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'bb4f1c2d8e9a'
down_revision: Union[str, Sequence[str], None] = 'a3204c0a87fc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP EXTENSION IF EXISTS vector")
