"""embedding text data

Revision ID: 3d5d5f752450
Revises: e45a460919ef
Create Date: 2025-01-19 16:27:41.851361

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3d5d5f752450'
down_revision: Union[str, None] = 'e45a460919ef'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("rs_amazon_reviews", sa.Column("title_and_text_embedding", sa.ARRAY(), autoincrement=False, nullable=True))
    op.add_column("rs_amazon_products", sa.Column("name_embedding", sa.ARRAY(), autoincrement=False, nullable=True))
    op.add_column("rs_amazon_products", sa.Column("description_embedding", sa.ARRAY(), autoincrement=False, nullable=True))


def downgrade() -> None:
    op.drop_column("rs_amazon_reviews", "title_and_text_embedding")
    op.drop_column("rs_amazon_products", "name_embedding")
    op.drop_column("rs_amazon_products", "description_embedding")
