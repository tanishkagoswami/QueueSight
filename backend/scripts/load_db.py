"""
QueueSight - put the chosen hospital into the database: its cleaned records, and the wait range our model predicted
for every visit. Skipped when the database already holds exactly these files.

Run from inside backend/, with DATABASE_URL set:
    python -m scripts.load_db              bring the tables up to date, then load (if anything changed)
    python -m scripts.load_db --force      load again even if nothing changed
"""
import hashlib
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from app.config import ARTIFACTS_DIR, DATA_SOURCE, RAW_DIR
from app.data import TABLES, clean, load_raw
from app.db import CSV_TABLES, LoadedHospital, Prediction, get_engine
from sqlalchemy import Engine, Integer, insert, select, text

BACKEND = Path(__file__).resolve().parents[1]
PREDICTIONS = "encounter_predictions.csv"   # written by scripts.train_models
TABLE_KEYS = {name: key for key, (name, _, _) in TABLES.items()}   # database table -> its name in load_raw()


def fingerprint(source: str, raw_dir: Path, artifacts_dir: Path) -> str:
    """One short code for these exact files: if any byte of the data or the predictions changes, so does the code."""
    h = hashlib.sha256(source.encode())
    for path in [raw_dir / f"{name}.csv" for name in CSV_TABLES] + [artifacts_dir / PREDICTIONS]:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _copy(cur, table, frame: pd.DataFrame):
    """Stream a table into PostgreSQL with COPY: far faster than one INSERT per row."""
    frame = frame.copy()
    for col in table.__table__.columns:
        if isinstance(col.type, Integer):
            frame[col.name] = frame[col.name].astype("Int64")   # 12.0 -> 12, so PostgreSQL accepts it as a whole number
    columns = list(table.__table__.columns.keys())
    with cur.copy(f"COPY {table.__tablename__} ({', '.join(columns)}) FROM STDIN (FORMAT csv)") as copy:
        copy.write(frame[columns].to_csv(index=False, header=False))


def load(engine: Engine, source: str, raw_dir: Path, artifacts_dir: Path, force: bool = False) -> dict | None:
    """Replace whatever hospital the database holds with this one, all in one transaction: the API sees either the
    old hospital or the new one, never half of each. Returns what was loaded, or None if it was already there."""
    if not (artifacts_dir / PREDICTIONS).exists():
        raise SystemExit(f"{artifacts_dir / PREDICTIONS} not found - train first:  python -m scripts.train_models")
    code = fingerprint(source, raw_dir, artifacts_dir)
    with engine.connect() as conn:
        loaded = conn.execute(select(LoadedHospital.__table__)).first()
    if loaded and loaded.fingerprint == code and not force:
        return None

    cleaned = clean(load_raw(raw_dir))
    predictions = pd.read_csv(artifacts_dir / PREDICTIONS, parse_dates=["arrival_time"])
    with engine.begin() as conn, conn.connection.driver_connection.cursor() as cur:
        conn.execute(text("TRUNCATE loaded_hospital, predictions, diagnostic_tests, encounters, patients, departments"))
        for name, table in CSV_TABLES.items():   # departments and patients first: visits point at them
            _copy(cur, table, cleaned[TABLE_KEYS[name]])
        _copy(cur, Prediction, predictions)
        conn.execute(insert(LoadedHospital.__table__).values(
            id=1, source=source, loaded_at=datetime.now(UTC),
            encounters=len(cleaned["encounters"]), fingerprint=code))
    return {"visits": len(cleaned["encounters"]), "tests": len(cleaned["tests"]),
            "patients": len(cleaned["patients"]), "predictions": len(predictions)}


def migrate():
    """Bring the database's tables up to the latest migration (does nothing if they already are)."""
    from alembic import command
    from alembic.config import Config
    command.upgrade(Config(str(BACKEND / "alembic.ini")), "head")


def main():
    t0 = time.time()
    migrate()
    result = load(get_engine(), DATA_SOURCE, RAW_DIR, ARTIFACTS_DIR, force="--force" in sys.argv)
    if result is None:
        print(f"[load_db] the database already holds these {DATA_SOURCE} files - nothing to do")
        return
    counts = ", ".join(f"{n:,} {what}" for what, n in result.items())
    print(f"[load_db] {DATA_SOURCE} hospital loaded: {counts} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()