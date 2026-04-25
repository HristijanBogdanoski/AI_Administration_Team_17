"""add embg to users

Revision ID: 20260425_0005
Revises: 20260425_0004
Create Date: 2026-04-25 12:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20260425_0005"
down_revision: Union[str, None] = "20260425_0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("embg", sa.String(length=13), nullable=True))
    op.create_index(op.f("ix_users_embg"), "users", ["embg"], unique=True)


def downgrade() -> None:
    op.drop_index(op.f("ix_users_embg"), table_name="users")
    op.drop_column("users", "embg")
