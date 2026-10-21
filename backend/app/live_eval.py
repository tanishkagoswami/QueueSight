"""Judge the walk-out model the way the radar uses it: at real moments, knowing only how long each patient has waited SO FAR.

Training scores every visit with its FINAL wait, which nobody knows while the patient is still in the queue. So the
model looks better in hindsight than it really is on the radar. Here we replay real moments to measure its honest
"live" quality, and calibrate its percentages so that "30%" really means about 3 in 10 such patients walk out.
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from app.config import TRAIN_END, VALID_END
from app.ml import predict_lbc

EVERY = "30min"                       # how often we stop the clock during the replay
RISK_BANDS = [0, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0]
MIN_PER_STEP = 200                    # every calibration step rests on at least this many patient-moments
MAX_STEPS = 20


def replay(df, start, end, risk_fn, every=EVERY):
    """At every moment from start to end: each patient still waiting, their risk right now, and whether they later left.
    The same patient appears at every moment they were in the queue, just like on the live radar."""
    df = df[(df["arrival_time"] >= start - pd.Timedelta(days=2)) & (df["arrival_time"] <= end)]
    arrival, wait_end = df["arrival_time"].to_numpy(), df["wait_end_time"].to_numpy()
    rows, waited = [], []
    for t in pd.date_range(start, end, freq=every).to_numpy():
        idx = np.flatnonzero((arrival <= t) & (wait_end > t))   # arrived, not called in, not gone
        rows.append(idx)
        waited.append((t - arrival[idx]) / np.timedelta64(1, "m"))
    idx, waited = np.concatenate(rows), np.concatenate(waited)
    w = df.iloc[idx]
    return pd.DataFrame({"risk": risk_fn(w, waited) if len(w) else np.array([]), "waited": waited,
                         "triage_level": w["triage_level"].to_numpy(), "left": w["is_lbc"].to_numpy()})


def fit_calibrator(snaps, min_per_step=MIN_PER_STEP):
    """Map a raw risk score to the share of such patients who really walked out.
    Patients are sorted by raw score and put into equal-size groups of at least min_per_step, so no step of the
    staircase rests on a handful of people. Isotonic regression then makes sure it never goes down."""
    raw = snaps["risk"].to_numpy(dtype=float)
    left = snaps["left"].to_numpy(dtype=float)
    groups = np.array_split(np.argsort(raw, kind="stable"), max(1, min(MAX_STEPS, len(raw) // min_per_step)))
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit([raw[g].mean() for g in groups], [left[g].mean() for g in groups], sample_weight=[len(g) for g in groups])
    return iso


def calibration_table(shown, left):
    """For each band of shown risk: what we showed on average vs how many really left."""
    labels = [f"{a:.0%}-{b:.0%}" for a, b in itertools.pairwise(RISK_BANDS)]
    t = pd.DataFrame({"shown": shown, "left": left})
    t["band"] = pd.cut(t["shown"], RISK_BANDS, labels=labels, include_lowest=True)
    return [{"band": str(b), "shown": round(float(g["shown"].mean()), 3),
             "really_left": round(float(g["left"].mean()), 3), "n": len(g)}
            for b, g in t.groupby("band", observed=True)]


def _scores(y, s):
    if len(np.unique(y)) < 2:
        return {"auc": None, "pr_auc": None}
    return {"auc": round(roc_auc_score(y, s), 3), "pr_auc": round(average_precision_score(y, s), 3)}


def evaluate_live(df, lbc_bundle):
    """Fit the calibrator on October replays, then report honest live quality on Nov-Dec (test, never used for fitting)."""
    def risk_now(rows, waited):
        return predict_lbc(lbc_bundle, rows, wait_override=waited)

    valid = replay(df, TRAIN_END, VALID_END - pd.Timedelta(EVERY), risk_now)
    test = replay(df, VALID_END, df["arrival_time"].max(), risk_now)
    calibrator = fit_calibrator(valid) if len(valid) else None

    y = test["left"].to_numpy().astype(int)
    raw = test["risk"].to_numpy().astype(float)
    shown = calibrator.predict(raw) if calibrator is not None else raw
    triage_rate = df[df["arrival_time"] < TRAIN_END].groupby("triage_level")["is_lbc"].mean()
    tri = test["triage_level"].map(triage_rate).fillna(triage_rate.mean()).to_numpy()

    metrics = {
        "how": f"every {EVERY} of Nov-Dec 2025: each waiting patient's risk right now vs whether they later walked out",
        "patient_moments": len(test),
        "walked_out_pct": round(100 * y.mean(), 1) if len(y) else None,
        "model": {**_scores(y, shown), "brier": round(brier_score_loss(y, shown), 4) if len(y) else None},
        "rules": {"minutes waited so far": _scores(y, test["waited"].to_numpy()), "triage rate": _scores(y, tri)},
        "calibration_before": calibration_table(raw, y) if len(y) else [],
        "calibration_after": calibration_table(shown, y) if len(y) else [],
    }
    return calibrator, metrics