"""Loading a hospital into the database, and reading it back, gives exactly the cleaned hospital the files give.

The database tests need TEST_DATABASE_URL (a spare database, see test_migrations.py); the others always run.
"""
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
from alembic import command
from alembic.config import Config
from app.config import DATA_SOURCES, ROOT_DIR
from app.data import clean, load_clean_from_db, load_raw
from scripts.load_db import PREDICTIONS, fingerprint, load
from sqlalchemy import create_engine, text

TEST_URL = os.getenv("TEST_DATABASE_URL")
BACKEND = Path(__file__).resolve().parents[1]
RAW = DATA_SOURCES["datathon"]
ARTIFACTS = ROOT_DIR / "backend" / "artifacts" / "datathon"


@pytest.fixture(scope="module")
def engine():
    if not TEST_URL:
        pytest.skip("needs TEST_DATABASE_URL (a spare database)")
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", TEST_URL.replace("%", "%%"))
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    engine = create_engine(TEST_URL)
    yield engine
    engine.dispose()
    command.downgrade(cfg, "base")


@pytest.fixture
def datathon(engine):
    load(engine, "datathon", RAW, ARTIFACTS, force=True)
    return engine


def count(engine, table):
    with engine.connect() as c:
        return c.execute(text(f"SELECT count(*) FROM {table}")).scalar()


def copy_hospital(tmp_path, keep_visits=None):
    """A copy of the datathon files to play with, optionally keeping only the first few visits."""
    raw, artifacts = tmp_path / "raw", tmp_path / "artifacts"
    shutil.copytree(RAW, raw)
    artifacts.mkdir()
    shutil.copy(ARTIFACTS / PREDICTIONS, artifacts)
    if keep_visits:
        visits = pd.read_csv(raw / "encounters.csv").head(keep_visits)
        visits.to_csv(raw / "encounters.csv", index=False)
        preds = pd.read_csv(artifacts / PREDICTIONS)
        preds[preds["encounter_id"].isin(visits["encounter_id"])].to_csv(artifacts / PREDICTIONS, index=False)
    return raw, artifacts


def test_the_fingerprint_changes_when_any_file_or_the_hospital_changes(tmp_path):
    raw, artifacts = copy_hospital(tmp_path)
    before = fingerprint("datathon", raw, artifacts)
    assert fingerprint("datathon", raw, artifacts) == before
    assert fingerprint("sim", raw, artifacts) != before
    with open(raw / "patients.csv", "a") as f:
        f.write("\n")
    assert fingerprint("datathon", raw, artifacts) != before


def test_no_predictions_yet_stops_with_a_clear_message(tmp_path):
    with pytest.raises(SystemExit, match="train first"):
        load(None, "sim", tmp_path, tmp_path)


def test_reading_the_database_back_gives_exactly_the_cleaned_hospital(datathon):
    from_db, from_files = load_clean_from_db(datathon), clean(load_raw(RAW))
    for name in from_files:
        pd.testing.assert_frame_equal(from_db[name], from_files[name])   # same rows, values, order and types


def test_every_visit_gets_its_predicted_wait_range(datathon):
    assert count(datathon, "predictions") == count(datathon, "encounters") == len(pd.read_csv(ARTIFACTS / PREDICTIONS))


def test_loading_the_same_files_again_is_skipped(datathon):
    assert load(datathon, "datathon", RAW, ARTIFACTS) is None


def test_another_hospital_fully_replaces_the_old_one(datathon, tmp_path):
    raw, artifacts = copy_hospital(tmp_path, keep_visits=500)
    result = load(datathon, "sim", raw, artifacts)
    assert result["visits"] == count(datathon, "encounters") == count(datathon, "predictions") == 500
    with datathon.connect() as c:
        assert c.execute(text("SELECT source, encounters FROM loaded_hospital")).all() == [("sim", 500)]