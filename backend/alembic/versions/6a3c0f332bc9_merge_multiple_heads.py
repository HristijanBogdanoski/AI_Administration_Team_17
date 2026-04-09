"""merge multiple heads

Revision ID: 6a3c0f332bc9
Revises: 20260407_0003, 8a2ee1a27381
Create Date: 2026-04-09 18:59:02.844295

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6a3c0f332bc9'
down_revision: Union[str, None] = ('20260407_0003', '8a2ee1a27381')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
