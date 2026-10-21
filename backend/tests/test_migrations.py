"""The migrations build exactly the tables our models describe, as TimescaleDB hypertables, and can be undone.

They need a real PostgreSQL + TimescaleDB, so they only run when TEST_DATABASE_URL points at a SPARE database
(they empty it, so never point it at the real one):
    docker compose exec db createdb -U queuesight queuesight_test
    $env:TEST_DATABASE_URL="postgresql+psycopg://queuesight:queuesight@localhost:5433/queuesight_test"
"""
import os
from datetime import timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from app.db import HYPERTABLES, SQLModel, is_ours
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

TEST_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_URL, reason="needs TEST_DATABASE_URL (a spare database)")

BACKEND = Path(__file__).resolve().parents[1]
OUR_TABLES = set(SQLModel.metadata.tables)


@pytest.fixture(scope="module")
def alembic_cfg():
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", TEST_URL.replace("%", "%%"))
    command.downgrade(cfg, "base")   # start from an empty database
    command.upgrade(cfg, "head")
    yield cfg
    command.downgrade(cfg, "base")   # and leave it empty


@pytest.fixture(scope="module")
def engine(alembic_cfg):
    engine = create_engine(TEST_URL)
    yield engine
    engine.dispose()


def test_upgrade_creates_every_table(engine):
    assert OUR_TABLES <= set(inspect(engine).get_table_names())


def test_visits_tests_and_predictions_become_hypertables_in_30_day_chunks(engine):
    with engine.connect() as c:
        rows = c.execute(text("SELECT hypertable_name, column_name, time_interval "
                              "FROM timescaledb_information.dimensions")).all()
    assert {r.hypertable_name: r.column_name for r in rows} == HYPERTABLES
    assert {r.time_interval for r in rows} == {timedelta(days=30)}


def test_the_migrations_build_exactly_what_the_models_describe(engine):
    """Fails if someone changes a table in app/db.py but forgets to write a migration for it."""
    with engine.connect() as c:
        context = MigrationContext.configure(c, opts={"include_object": is_ours, "compare_type": True})
        assert compare_metadata(context, SQLModel.metadata) == []


def test_a_visit_must_belong_to_a_real_patient_and_department(engine):
    with engine.connect() as c, pytest.raises(IntegrityError):
        c.execute(text("INSERT INTO encounters (encounter_id, patient_id, arrival_time, department_id) "
                       "VALUES ('E1', 'nobody', '2025-03-01 10:00', 'nowhere')"))


def test_downgrade_removes_every_table_and_upgrade_brings_them_back(alembic_cfg, engine):
    command.downgrade(alembic_cfg, "base")
    assert not OUR_TABLES & set(inspect(engine).get_table_names())
    command.upgrade(alembic_cfg, "head")
    assert OUR_TABLES <= set(inspect(engine).get_table_names())