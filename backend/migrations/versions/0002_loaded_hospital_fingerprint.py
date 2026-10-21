"""Remember which files the loaded hospital came from, so a changed file means a reload.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-20
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # A required column can't be added while a row has no value for it. This row is only a note of what was loaded,
    # so forget it: the next load fills the database again and records the fingerprint.
    op.execute("DELETE FROM loaded_hospital")
    op.add_column("loaded_hospital", sa.Column("fingerprint", sa.String(), nullable=False))


def downgrade() -> None:
    op.drop_column("loaded_hospital", "fingerprint")