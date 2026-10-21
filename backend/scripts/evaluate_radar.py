"""
QueueSight - does adding walk-out risk to the radar's ranking put more of the patients who are about to leave at the top?

Replays every hour from 08:00 to 22:00, ranks the patients waiting at that moment, and checks the top 10 against
what really happened next. The weight is chosen on October (validation) and confirmed on Nov-Dec (test, never used
for choosing), the same split the models use. In a quiet hospital (like the datathon one) rarely more than 10 people
wait at once, so the top 10 is everyone and every weight scores the same.

Run from inside backend/:   $env:QUEUESIGHT_DATA="sim"; python -m scripts.evaluate_radar
"""
import time

import numpy as np
import pandas as pd
from app.snapshot import WALKOUT_WEIGHT, priority, queue_state
from app.state import STATE

WEIGHTS = [0, 25, 50, 100, 200, 400]   # 0 = minutes past the predicted range only
TOP = 10                               # how many patients a busy nurse realistically checks


def evaluate(start, end):
    df = STATE.df
    found = {w: {"left": 0, "overdue": 0, "slots": 0} for w in WEIGHTS}
    all_leavers = 0
    for t in pd.date_range(start, end, freq="h"):
        if not 8 <= t.hour <= 22:
            continue
        waiting = df[queue_state(df, t)["waiting"]]
        if waiting.empty:
            continue
        elapsed = ((t - waiting["arrival_time"]).dt.total_seconds() / 60).values
        risk = STATE.lbc_risk(waiting, elapsed)
        will_leave = waiting["is_lbc"].eq(1).values                              # what really happened next
        will_be_overdue = (waiting["wait_minutes"] > waiting["pred_p90"]).values
        all_leavers += will_leave.sum()
        for w in WEIGHTS:
            score = priority(elapsed - waiting["pred_p90"].values, risk, use_risk=True, weight=w)
            top = np.argsort(-score)[:TOP]
            found[w]["left"] += will_leave[top].sum()
            found[w]["overdue"] += will_be_overdue[top].sum()
            found[w]["slots"] += len(top)
    for w, f in found.items():
        tag = "  <- used" if w == WALKOUT_WEIGHT else ("  (overdue only)" if w == 0 else "")
        print(f"  weight {w:>3}: top {TOP} include {f['left'] / max(all_leavers, 1):6.1%} of those who later walked out, "
              f"{f['overdue'] / f['slots']:6.1%} of the top {TOP} end up overdue{tag}")


def main():
    t0 = time.time()
    STATE.load()
    if not STATE.walkout_model["trusted"]:
        print(f"Note: the walk-out model is NOT trusted for this hospital (AUC {STATE.walkout_model['auc']}), "
              "so the live radar ignores walk-out risk. The numbers below show what would happen if it didn't.\n")
    print("VALIDATION, October 2025 (used to choose the weight)")
    evaluate("2025-10-01", "2025-10-31 23:00")
    print("TEST, November-December 2025 (never used for choosing)")
    evaluate("2025-11-01", "2025-12-31 23:00")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()