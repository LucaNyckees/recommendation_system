"""first version

Revision ID: 164e51c92234
Revises: 
Create Date: 2024-11-12 11:28:01.087372

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY


# revision identifiers, used by Alembic.
revision: str = '164e51c92234'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "rs_products",
        # sa.Column('id', sa.INTEGER(),
        #                       sa.Identity(always=True, start=1, increment=1, minvalue=1, maxvalue=2147483647,
        #                                   cycle=False, cache=1), autoincrement=True, nullable=False),
        sa.Column('name', sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column('average_rating', sa.REAL(), autoincrement=False, nullable=True),
        sa.Column('rating_number', sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column('features', ARRAY(sa.VARCHAR()), autoincrement=False, nullable=True),
        sa.Column('descriptions', ARRAY(sa.VARCHAR()), autoincrement=False, nullable=True),
        sa.Column('price', sa.REAL(), autoincrement=False, nullable=True),
        sa.Column('image_urls', ARRAY(sa.VARCHAR()), autoincrement=False, nullable=True),
        sa.Column('store', sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column('categories', ARRAY(sa.VARCHAR()), autoincrement=False, nullable=True),
        sa.Column('parent_asin', ARRAY(sa.VARCHAR()), autoincrement=False, nullable=True),
        sa.Column('details', sa.JSON(), autoincrement=False, nullable=True),
    )


def downgrade() -> None:
    op.drop_table("rs_products")
