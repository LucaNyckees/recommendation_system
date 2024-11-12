"""first version

Revision ID: 164e51c92234
Revises: 
Create Date: 2024-11-12 11:28:01.087372

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY
from alembic_utils.pg_extension import PGExtension


# revision identifiers, used by Alembic.
revision: str = '164e51c92234'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


unaccent = PGExtension("public", "unaccent")
intarray = PGExtension("public", "intarray")

# NOTE: the 'rs' prefix stands for 'recommendation system'.


def upgrade() -> None:

    # creating pg extensions
    op.create_entity(unaccent)
    op.create_entity(intarray)

    # creating tables
    op.create_table(
        "rs_amazon_products",
        sa.Column('parent_asin', sa.BIGINT(), autoincrement=False, nullable=True),  # ID of the product
        sa.Column('name', sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column('main_category', sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column('average_rating', sa.REAL(), autoincrement=False, nullable=True),
        sa.Column('rating_number', sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column('features', ARRAY(sa.VARCHAR()), autoincrement=False, nullable=True),
        sa.Column('description', ARRAY(sa.VARCHAR()), autoincrement=False, nullable=True),
        sa.Column('price', sa.REAL(), autoincrement=False, nullable=True),
        sa.Column('image_urls', ARRAY(sa.VARCHAR()), autoincrement=False, nullable=True),
        sa.Column('store', sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column('categories', ARRAY(sa.VARCHAR()), autoincrement=False, nullable=True),
        sa.Column('details', sa.JSON(), autoincrement=False, nullable=True),
        sa.PrimaryKeyConstraint('parent_asin', name='rs_amazon_products_pkey'),
    )

    op.create_table(
        "rs_amazon_reviews",
        sa.Column('id', sa.INTEGER(),
                              sa.Identity(always=True, start=1, increment=1, minvalue=1, maxvalue=2147483647,
                                          cycle=False, cache=1), autoincrement=True, nullable=False),
        sa.Column('asin', sa.BIGINT(), autoincrement=False, nullable=True),
        sa.Column('parent_asin', sa.BIGINT(), autoincrement=False, nullable=True),
        sa.Column('title', sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column('rating', sa.REAL(), autoincrement=False, nullable=True),
        sa.Column('text', sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column('user_id', sa.BIGINT(), autoincrement=False, nullable=True),
        sa.Column('date', sa.DATE(), autoincrement=False, nullable=True),
        sa.Column('helpful_vote', sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column('verified_purchase', sa.BOOLEAN(), autoincrement=False, nullable=True),
        sa.PrimaryKeyConstraint('id', name='rs_amazon_reviews_pkey'),
        sa.ForeignKeyConstraint(['parent_asin'], ['rs_amazon_products.parent_asin'], name='rs_amazon_reviews_parent_asin_fkey'),
    )


def downgrade() -> None:
    op.drop_table("rs_amazon_reviews")
    op.drop_table("rs_amazon_products")
    op.drop_entity(unaccent)
    op.drop_entity(intarray)
