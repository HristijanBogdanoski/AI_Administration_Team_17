"""merge heads

Revision ID: 81ef9fa18bcd
Revises: 6be9b5c3b043, 756f6b364eb6
Create Date: 2026-05-08 19:41:21.643486

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '81ef9fa18bcd'
down_revision: Union[str, None] = ('6be9b5c3b043', '756f6b364eb6')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
