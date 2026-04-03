"""add user role

Revision ID: 20260401_0002
Revises: 20260326_0001
Create Date: 2026-04-01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20260401_0002"
down_revision: Union[str, None] = "20260326_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    columns = {col["name"] for col in inspector.get_columns("users")}
    if "role" not in columns:
        op.add_column(
            "users",
            sa.Column("role", sa.String(length=20), nullable=False, server_default="user"),
        )

    indexes = {idx["name"] for idx in inspector.get_indexes("users")}
    if op.f("ix_users_role") not in indexes:
        op.create_index(op.f("ix_users_role"), "users", ["role"], unique=False)

    checks = {chk["name"] for chk in inspector.get_check_constraints("users")}
    if "ck_users_role" not in checks:
        with op.batch_alter_table("users") as batch_op:
            batch_op.create_check_constraint("ck_users_role", "role IN ('user', 'admin')")


def downgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_constraint("ck_users_role", type_="check")
        batch_op.drop_index(op.f("ix_users_role"))
        batch_op.drop_column("role")
