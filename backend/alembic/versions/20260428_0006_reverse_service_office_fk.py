"""reverse service and service_office foreign key direction

Revision ID: 20260428_0006
Revises: 20260425_0005
Create Date: 2026-04-28 18:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20260428_0006"
down_revision: Union[str, None] = "20260425_0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _find_foreign_key_name(table_name: str, constrained_column: str) -> str | None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    for fk in inspector.get_foreign_keys(table_name):
        cols = fk.get("constrained_columns") or []
        if constrained_column in cols:
            return fk.get("name")
    return None


def _find_unique_constraint_name(table_name: str, column_name: str) -> str | None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    for uq in inspector.get_unique_constraints(table_name):
        cols = uq.get("column_names") or []
        if cols == [column_name]:
            return uq.get("name")
    return None


def upgrade() -> None:
    bind = op.get_bind()

    duplicate_service_id = bind.execute(
        sa.text(
            """
            SELECT service_id
            FROM services
            WHERE service_id IS NOT NULL
            GROUP BY service_id
            HAVING COUNT(*) > 1
            LIMIT 1
            """
        )
    ).scalar_one_or_none()
    if duplicate_service_id is not None:
        raise RuntimeError(
            "Cannot apply migration: duplicate values found in services.service_id. "
            f"Example duplicate: '{duplicate_service_id}'"
        )

    null_service_id_count = bind.execute(
        sa.text("SELECT COUNT(*) FROM services WHERE service_id IS NULL")
    ).scalar_one()
    if null_service_id_count:
        raise RuntimeError(
            "Cannot apply migration: services.service_id contains NULL values. "
            "Populate service_id for all services before running this migration."
        )

    missing_service_id = bind.execute(
        sa.text(
            """
            SELECT so.service_id
            FROM service_offices so
            LEFT JOIN services s ON s.service_id = so.service_id
            WHERE s.service_id IS NULL
            LIMIT 1
            """
        )
    ).scalar_one_or_none()
    if missing_service_id is not None:
        raise RuntimeError(
            "Cannot apply migration: some service_offices.service_id values do not exist in services.service_id. "
            f"Example missing value: '{missing_service_id}'"
        )

    op.alter_column("services", "service_id", existing_type=sa.String(length=100), nullable=False)
    op.create_unique_constraint("uq_services_service_id", "services", ["service_id"])

    old_fk_name = _find_foreign_key_name("services", "service_id")
    if old_fk_name is not None:
        op.drop_constraint(old_fk_name, "services", type_="foreignkey")

    old_service_offices_uq = _find_unique_constraint_name("service_offices", "service_id")
    if old_service_offices_uq is not None:
        op.drop_constraint(old_service_offices_uq, "service_offices", type_="unique")

    op.create_foreign_key(
        "fk_service_offices_service_id_services",
        "service_offices",
        "services",
        ["service_id"],
        ["service_id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint("fk_service_offices_service_id_services", "service_offices", type_="foreignkey")
    op.create_unique_constraint("uq_service_offices_service_id", "service_offices", ["service_id"])

    op.create_foreign_key(
        "fk_services_service_id_service_offices",
        "services",
        "service_offices",
        ["service_id"],
        ["service_id"],
        ondelete="SET NULL",
    )

    op.drop_constraint("uq_services_service_id", "services", type_="unique")
    op.alter_column("services", "service_id", existing_type=sa.String(length=100), nullable=True)
