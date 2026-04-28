"""add address, phone number, and gender to users

Revision ID: 20260428_0009
Revises: 20260428_0008
Create Date: 2026-04-28 20:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20260428_0009"
down_revision: Union[str, None] = "20260428_0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("address", sa.String(length=255), nullable=True))
    op.add_column("users", sa.Column("phone_number", sa.String(length=50), nullable=True))
    op.add_column("users", sa.Column("gender", sa.String(length=20), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "gender")
    op.drop_column("users", "phone_number")
    op.drop_column("users", "address")