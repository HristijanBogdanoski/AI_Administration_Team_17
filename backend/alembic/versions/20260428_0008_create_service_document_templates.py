"""create service document templates

Revision ID: 20260428_0008
Revises: 20260428_0007
Create Date: 2026-04-28 19:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20260428_0008"
down_revision: Union[str, None] = "20260428_0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "service_document_templates",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("service_id", sa.String(length=100), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("template_type", sa.String(length=20), nullable=False, server_default="json"),
        sa.Column("template_body", sa.JSON(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.ForeignKeyConstraint(["service_id"], ["services.service_id"], ondelete="CASCADE"),
        sa.UniqueConstraint("service_id"),
    )
    op.create_index(
        op.f("ix_service_document_templates_id"),
        "service_document_templates",
        ["id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_service_document_templates_service_id"),
        "service_document_templates",
        ["service_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_service_document_templates_service_id"), table_name="service_document_templates")
    op.drop_index(op.f("ix_service_document_templates_id"), table_name="service_document_templates")
    op.drop_table("service_document_templates")