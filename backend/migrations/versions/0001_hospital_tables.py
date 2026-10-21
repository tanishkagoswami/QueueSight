"""The hospital tables, with visits, tests and predictions as TimescaleDB hypertables.

Revision ID: 0001
Revises:
Create Date: 2026-10-20
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# table -> its time column. Each is stored in 30-day chunks, so a question about last week only reads the newest chunk
HYPERTABLES = {"encounters": "arrival_time", "diagnostic_tests": "requested_time", "predictions": "arrival_time"}


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")

    op.create_table(
        "departments",
        sa.Column("department_id", sa.String(), primary_key=True),
        sa.Column("department_name", sa.String(), nullable=False),
        sa.Column("staff_count", sa.Integer(), nullable=False),
        sa.Column("bed_count", sa.Integer(), nullable=False),
        sa.Column("daily_capacity", sa.Integer(), nullable=False),
        sa.Column("service_type", sa.String(), nullable=False),
    )
    op.create_table(
        "patients",
        sa.Column("patient_id", sa.String(), primary_key=True),
        sa.Column("age", sa.Integer()),
        sa.Column("sex", sa.String()),
        sa.Column("chronic_condition", sa.String()),
        sa.Column("home_area", sa.String()),
        sa.Column("insurance_type", sa.String()),
    )
    op.create_table(
        "encounters",
        sa.Column("encounter_id", sa.String(), nullable=False),
        sa.Column("patient_id", sa.String(), sa.ForeignKey("patients.patient_id"), nullable=False),
        sa.Column("arrival_time", sa.DateTime(), nullable=False),
        sa.Column("department_id", sa.String(), sa.ForeignKey("departments.department_id"), nullable=False),
        sa.Column("visit_type", sa.String()),
        sa.Column("triage_level", sa.String()),
        sa.Column("outcome", sa.String()),
        sa.Column("service_start_time", sa.DateTime()),
        sa.Column("service_end_time", sa.DateTime()),
        sa.Column("wait_minutes", sa.Integer()),
        sa.Column("service_minutes", sa.Integer()),
        sa.Column("discharge_time", sa.DateTime()),
        sa.PrimaryKeyConstraint("encounter_id", "arrival_time"),
    )
    op.create_index("ix_encounters_patient_id", "encounters", ["patient_id"])
    op.create_index("ix_encounters_department_id", "encounters", ["department_id"])
    op.create_table(
        "diagnostic_tests",
        sa.Column("test_id", sa.String(), nullable=False),
        sa.Column("encounter_id", sa.String(), nullable=False),
        sa.Column("test_type", sa.String()),
        sa.Column("requested_time", sa.DateTime(), nullable=False),
        sa.Column("completion_time", sa.DateTime()),
        sa.Column("priority", sa.String()),
        sa.PrimaryKeyConstraint("test_id", "requested_time"),
    )
    op.create_index("ix_diagnostic_tests_encounter_id", "diagnostic_tests", ["encounter_id"])
    op.create_table(
        "predictions",
        sa.Column("encounter_id", sa.String(), nullable=False),
        sa.Column("arrival_time", sa.DateTime(), nullable=False),
        sa.Column("department_id", sa.String(), nullable=False),
        sa.Column("pred_p10", sa.Float(), nullable=False),
        sa.Column("pred_p50", sa.Float(), nullable=False),
        sa.Column("pred_p90", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("encounter_id", "arrival_time"),
    )
    op.create_index("ix_predictions_department_id", "predictions", ["department_id"])
    op.create_table(
        "loaded_hospital",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("loaded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("encounters", sa.Integer(), nullable=False),
    )

    for table, time_column in HYPERTABLES.items():
        op.execute(f"SELECT create_hypertable('{table}', by_range('{time_column}', INTERVAL '30 days'))")


def downgrade() -> None:
    # dropping a hypertable also drops its chunks; the TimescaleDB extension stays, other databases may use it
    for table in ["loaded_hospital", "predictions", "diagnostic_tests", "encounters", "patients", "departments"]:
        op.drop_table(table)