"""add embedding dimension

Revision ID: d4e8b2a1c6f0
Revises: c9d7a6e5f4b3
Create Date: 2026-10-14 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4e8b2a1c6f0'
down_revision: Union[str, Sequence[str], None] = 'c9d7a6e5f4b3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'book_embeddings',
        sa.Column(
            'embedding_dimension',
            sa.Integer(),
            nullable=False,
            server_default='384',
        ),
    )
    op.alter_column('book_embeddings', 'embedding_dimension', server_default=None)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('book_embeddings', 'embedding_dimension')
