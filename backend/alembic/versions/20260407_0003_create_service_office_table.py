"""create service_offices table

Revision ID: 20260407_0003
Revises:
Create Date: 2026-04-07
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision = "20260407_0003"
down_revision: Union[str, None] = "20260401_0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "service_offices",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("service_name", sa.String(length=255), nullable=False),
        sa.Column("service_id", sa.String(length=100), nullable=False),
        sa.Column("office_name", sa.String(length=255), nullable=False),
        sa.Column("address", sa.String(length=500), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("working_hours", sa.String(length=255), nullable=False),
        sa.Column("contact_email", sa.String(length=255), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("service_id"),
    )
    op.create_index("ix_service_offices_id", "service_offices", ["id"])
    op.create_index("ix_service_offices_service_name", "service_offices", ["service_name"])
    op.create_index("ix_service_offices_service_id", "service_offices", ["service_id"])


def downgrade() -> None:
    op.drop_index("ix_service_offices_service_id", table_name="service_offices")
    op.drop_index("ix_service_offices_service_name", table_name="service_offices")
    op.drop_index("ix_service_offices_id", table_name="service_offices")
    op.drop_table("service_offices")