"""With DATABASE_URL set, the API reads the hospital from the database and gives exactly the same answers as from
the CSV files. Needs TEST_DATABASE_URL (a spare database, see test_migrations.py)."""
import os
from pathlib import Path

import pandas as pd
import pytest
from alembic import command
from alembic.config import Config
from app.config import ARTIFACTS_DIR, DATA_SOURCE, RAW_DIR
from app.state import State
from scripts.load_db import load
from sqlalchemy import create_engine, text

TEST_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_URL, reason="needs TEST_DATABASE_URL (a spare database)")
BACKEND = Path(__file__).resolve().parents[1]
OTHER = "sim" if DATA_SOURCE == "datathon" else "datathon"


@pytest.fixture(scope="module")
def engine():
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", TEST_URL.replace("%", "%%"))
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    engine = create_engine(TEST_URL)
    yield engine
    engine.dispose()
    command.downgrade(cfg, "base")


def load_state(monkeypatch, url):
    monkeypatch.setattr("app.state.DATABASE_URL", url)
    state = State()
    state.load()
    if state.engine is not None:
        state.engine.dispose()
    return state


def test_the_api_gives_the_same_answers_from_the_database_as_from_the_files(engine, monkeypatch):
    load(engine, DATA_SOURCE, RAW_DIR, ARTIFACTS_DIR, force=True)
    from_files, from_db = load_state(monkeypatch, None), load_state(monkeypatch, TEST_URL)
    assert (from_files.stored_in, from_db.stored_in) == ("CSV files", "PostgreSQL + TimescaleDB")
    pd.testing.assert_frame_equal(from_db.df, from_files.df)   # every visit, feature and predicted wait range
    assert from_db.default_time == from_files.default_time
    assert from_db.normal_wait == from_files.normal_wait


def test_a_database_holding_another_hospital_stops_with_a_clear_message(engine, monkeypatch):
    load(engine, OTHER, RAW_DIR, ARTIFACTS_DIR, force=True)   # the same files, labelled as the other hospital
    with pytest.raises(RuntimeError, match=f"holds the {OTHER} hospital, not {DATA_SOURCE}"):
        load_state(monkeypatch, TEST_URL)


def test_an_empty_database_stops_with_a_clear_message(engine, monkeypatch):
    with engine.begin() as c:
        c.execute(text("TRUNCATE loaded_hospital"))
    with pytest.raises(RuntimeError, match="holds no hospital yet"):
        load_state(monkeypatch, TEST_URL)