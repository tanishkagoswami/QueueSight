"""
QueueSight - run the hospital simulator and save the result in the dataset's CSV format.

Run from inside backend/ :   python -m scripts.simulate                (1 year, seed 42 -> data/sim/)
                            python -m scripts.simulate --days 30 --seed 7
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from app.config import ROOT_DIR
from sim.config import SimConfig
from sim.engine import run
from sim.export import to_tables, write_csvs


def report(tables: dict[str, pd.DataFrame], util: pd.DataFrame, days: int, seconds: float) -> None:
    enc = tables["encounters"]
    lbc = enc["outcome"].eq("Left before care")
    print(f"\nSimulated {days} days in {seconds:.1f}s: {len(enc):,} visits ({len(enc) / days:.0f}/day), "
          f"{enc['patient_id'].nunique():,} patients, {len(tables['diagnostic_tests']):,} tests")

    print("\nMedian wait by triage (min):",
          enc.groupby("triage_level")["wait_minutes"].median().reindex(["Critical", "High", "Medium", "Low"]).to_dict())

    print(f"\nLeft before care: {lbc.mean():.1%} of visits")
    band = pd.cut(enc["wait_minutes"], [0, 15, 30, 60, 120, 10_000], right=False,
                  labels=["<15 min", "15-30", "30-60", "60-120", "120+"])
    print("Left before care by how long they waited (should rise):",
          lbc.groupby(band, observed=True).mean().round(3).to_dict())

    hour = pd.to_datetime(enc["arrival_time"]).dt.hour
    print("\nArrivals by hour (share):", (hour.value_counts(normalize=True).sort_index() * 100).round(1).to_dict())
    print("\nClinician utilisation (time with patients / time on duty):")
    print(util.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="Run the QueueSight hospital simulator")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--out", default=str(ROOT_DIR / "data" / "sim"))
    args = ap.parse_args()

    t0 = time.time()
    result = run(SimConfig(start_date=args.start, days=args.days, seed=args.seed))
    tables = to_tables(result)
    write_csvs(tables, Path(args.out))
    report(tables, result.utilisation, args.days, time.time() - t0)
    print(f"\nSaved 4 CSVs to {args.out}")


if __name__ == "__main__":
    main()