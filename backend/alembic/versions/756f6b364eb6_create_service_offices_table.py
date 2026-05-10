"""obsolete duplicate service_offices branch (kept as merge placeholder)

Revision ID: 756f6b364eb6
Revises: 20260428_0009
Create Date: 2026-05-08 16:23:12.996447

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '756f6b364eb6'
down_revision: Union[str, None] = '20260428_0009'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # This revision used to duplicate the service_offices create branch.
    # The real schema is established by 6be9b5c3b043, so this revision is now a no-op.
    pass


def downgrade() -> None:
    pass
