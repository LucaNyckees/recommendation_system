"""adding users table

Revision ID: e45a460919ef
Revises: 164e51c92234
Create Date: 2024-11-26 12:55:12.367172

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e45a460919ef"
down_revision: Union[str, None] = "164e51c92234"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "rs_amazon_users",
        sa.Column("user_id", sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column("age", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("gender", sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column("city", sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.PrimaryKeyConstraint("user_id", name="rs_amazon_users_pkey"),
        sa.ForeignKeyConstraint(["user_id"], ["rs_amazon_reviews.user_id"], name="rs_amazon_reviews_user_id_fkey"),
    )


def downgrade() -> None:
    op.drop_table("rs_amazon_users")
