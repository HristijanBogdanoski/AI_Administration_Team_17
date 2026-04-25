"""add service_id column to services

Revision ID: 20260425_0004
Revises: 6a3c0f332bc9
Create Date: 2026-04-25 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20260425_0004"
down_revision: Union[str, None] = "6a3c0f332bc9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("services", sa.Column("service_id", sa.String(length=100), nullable=True))
    op.create_index(op.f("ix_services_service_id"), "services", ["service_id"], unique=False)
    op.create_foreign_key(
        "fk_services_service_id_service_offices",
        "services",
        "service_offices",
        ["service_id"],
        ["service_id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_services_service_id_service_offices", "services", type_="foreignkey")
    op.drop_index(op.f("ix_services_service_id"), table_name="services")
    op.drop_column("services", "service_id")
