"""The database tables match the dataset files column for column, and are ready to become TimescaleDB hypertables."""
import pandas as pd
import pytest
from app.config import ROOT_DIR
from app.db import CSV_TABLES, HYPERTABLES, SQLModel, get_engine
from sqlalchemy import DateTime


@pytest.mark.parametrize("name", list(CSV_TABLES))
def test_each_table_has_exactly_the_csv_columns_in_the_same_order(name):
    csv_columns = list(pd.read_csv(ROOT_DIR / "data" / "raw" / f"{name}.csv", nrows=0).columns)
    assert list(CSV_TABLES[name].__table__.columns.keys()) == csv_columns


@pytest.mark.parametrize("table, time_column", HYPERTABLES.items())
def test_hypertables_include_their_time_column_in_the_primary_key(table, time_column):
    primary_key = [c.name for c in SQLModel.metadata.tables[table].primary_key.columns]
    assert time_column in primary_key


def test_walkouts_may_have_no_service_times_but_every_visit_has_an_arrival_time():
    cols = CSV_TABLES["encounters"].__table__.columns
    assert cols["service_start_time"].nullable and cols["service_end_time"].nullable
    assert not cols["arrival_time"].nullable

def test_hospital_times_are_stored_as_the_hospital_clock_shows_them_never_shifted_to_utc():
    for name, table in SQLModel.metadata.tables.items():
        for col in table.columns:
            if isinstance(col.type, DateTime):
                assert col.type.timezone is (name == "loaded_hospital"), f"{name}.{col.name}"
                
def test_a_clear_message_when_no_database_is_configured(monkeypatch):
    monkeypatch.setattr("app.db.DATABASE_URL", None)
    with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
        get_engine()
# ... (existing code in test_db_models.py) ...

