"""drop location column from services

Revision ID: 20260428_0007
Revises: 20260428_0006
Create Date: 2026-04-28 18:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20260428_0007"
down_revision: Union[str, None] = "20260428_0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("services", "location")


def downgrade() -> None:
    op.add_column("services", sa.Column("location", sa.String(length=255), nullable=True))
