"""
QueueSight - data health check + quick "is there real signal?" test.

Run from inside backend/ :   python -m scripts.profile_data
Paste the whole output back to Claude.
"""
import sys

import numpy as np
import pandas as pd
from app.config import RANDOM_STATE, RAW_DIR, VALID_END
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

ENC_TIMES = ["arrival_time", "service_start_time", "service_end_time", "discharge_time"]
TEST_TIMES = ["requested_time", "completion_time"]


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def load():
    files = {
        "patients": ("patients.csv", []),
        "encounters": ("encounters.csv", ENC_TIMES),
        "departments": ("departments.csv", []),
        "tests": ("diagnostic_tests.csv", TEST_TIMES),
    }
    out = {}
    for name, (fname, date_cols) in files.items():
        path = RAW_DIR / fname
        if not path.exists():
            sys.exit(f"Missing {path}  ->  put the 4 CSVs inside data/raw/")
        # keep_default_na=False: otherwise pandas turns the real value "None" (chronic_condition) into NaN
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
        for c in date_cols:
            df[c] = pd.to_datetime(df[c], errors="coerce")  # bad timestamps become NaT
        out[name] = df
    return out["patients"], out["encounters"], out["departments"], out["tests"]


def basic_profile(name, df, id_col):
    print(f"\n--- {name}: {df.shape[0]} rows x {df.shape[1]} cols")
    print("columns:", list(df.columns))
    nulls = df.isna().sum()
    print("nulls:", nulls[nulls > 0].to_dict() or "none")
    print(f"duplicate {id_col}:", int(df[id_col].duplicated().sum()))
    for c in df.select_dtypes(include=["object", "string"]).columns:
        if c != id_col and df[c].nunique() <= 25:
            print(f"  {c}: {df[c].value_counts(dropna=False).to_dict()}")


def minutes(a, b):
    return (b - a).dt.total_seconds() / 60


def add_live_features(enc, pat, dep):
    """Features that would be KNOWN at the moment a patient arrives (no leakage)."""
    enc = enc.sort_values("arrival_time").reset_index(drop=True)
    n = len(enc)
    waiting_now = np.zeros(n, int)     # earlier arrivals in same dept not yet started
    in_service_now = np.zeros(n, int)  # earlier arrivals in same dept currently being served
    dept_prev_24h = np.zeros(n, int)
    for _, g in enc.groupby("department_id"):
        idx = g.index.values
        arr = g["arrival_time"].values
        st = g["service_start_time"].values
        en = g["service_end_time"].values
        for k, i in enumerate(idx):
            t = arr[k]
            # "has this earlier patient started yet?" is a fact known at time t
            waiting_now[i] = np.sum(st[:k] > t)
            in_service_now[i] = np.sum((st[:k] <= t) & (en[:k] > t))
            dept_prev_24h[i] = np.sum(arr[:k] > t - np.timedelta64(24, "h"))
    all_arr = enc["arrival_time"].values
    left = np.searchsorted(all_arr, all_arr - np.timedelta64(3, "h"), side="left")
    enc["hosp_prev_3h"] = np.arange(n) - left
    enc["waiting_now"] = waiting_now
    enc["in_service_now"] = in_service_now
    enc["dept_prev_24h"] = dept_prev_24h
    enc["hour"] = enc["arrival_time"].dt.hour
    enc["dow"] = enc["arrival_time"].dt.dayofweek
    enc["month"] = enc["arrival_time"].dt.month
    enc["prior_visits"] = enc.groupby("patient_id").cumcount()
    enc = enc.merge(pat, on="patient_id", how="left").merge(dep, on="department_id", how="left")
    return enc


def main():
    pat, enc, dep, tst = load()

    section("1. TABLE PROFILES")
    basic_profile("patients", pat, "patient_id")
    basic_profile("encounters", enc, "encounter_id")
    basic_profile("departments", dep, "department_id")
    basic_profile("diagnostic_tests", tst, "test_id")
    print("\nnumeric summary (encounters):")
    print(enc[["wait_minutes", "service_minutes"]].describe().round(1).T)
    print("age summary:", pat["age"].describe().round(1).to_dict())

    section("2. JOIN INTEGRITY")
    print("encounters with unknown patient   :", int((~enc.patient_id.isin(pat.patient_id)).sum()))
    print("encounters with unknown department:", int((~enc.department_id.isin(dep.department_id)).sum()))
    print("tests with unknown encounter      :", int((~tst.encounter_id.isin(enc.encounter_id)).sum()))
    print("patients with zero encounters     :", int((~pat.patient_id.isin(enc.patient_id)).sum()))
    print("encounters with >=1 test          :", int(enc.encounter_id.isin(tst.encounter_id).sum()))

    section("3. TIMESTAMP LOGIC CHECKS (counts of bad rows)")
    calc_wait = minutes(enc.arrival_time, enc.service_start_time)
    calc_serv = minutes(enc.service_start_time, enc.service_end_time)
    checks = {
        "arrival range": f"{enc.arrival_time.min()} -> {enc.arrival_time.max()}",
        "service_start < arrival": int((enc.service_start_time < enc.arrival_time).sum()),
        "service_end < service_start": int((enc.service_end_time < enc.service_start_time).sum()),
        "discharge < service_end": int((enc.discharge_time < enc.service_end_time).sum()),
        "wait_minutes != computed (>1 min)": int(((calc_wait - enc.wait_minutes).abs() > 1).sum()),
        "service_minutes != computed (>1 min)": int(((calc_serv - enc.service_minutes).abs() > 1).sum()),
        "negative wait/service": int(((enc.wait_minutes < 0) | (enc.service_minutes < 0)).sum()),
    }
    for k, v in checks.items():
        print(f"  {k}: {v}")
    gap = minutes(enc.service_end_time, enc.discharge_time)
    print("  service_end -> discharge gap (min):", gap.describe().round(1).to_dict())
    has_service = enc.service_start_time.notna() & enc.service_end_time.notna()
    print("\noutcome vs has-service-times:")
    print(pd.crosstab(enc.outcome, has_service, margins=True))

    t = tst.merge(enc[["encounter_id", "arrival_time", "discharge_time", "department_id"]], on="encounter_id", how="left")
    tat = minutes(t.requested_time, t.completion_time)
    print("\ntests: completion < requested      :", int((tat < 0).sum()))
    print("tests: requested before arrival      :", int((t.requested_time < t.arrival_time).sum()))
    print("tests: requested after discharge     :", int((t.requested_time > t.discharge_time).sum()))
    print("tests per encounter:", tst.groupby("encounter_id").size().value_counts().sort_index().to_dict())

    section("4. VOLUME vs CAPACITY")
    days = enc.arrival_time.dt.normalize().nunique()
    per_day = enc.groupby(enc.arrival_time.dt.date).size()
    print(f"days covered: {days} | visits/day mean {per_day.mean():.1f} min {per_day.min()} max {per_day.max()}")
    vol = enc.groupby("department_id").size().rename("visits").to_frame()
    vol["visits_per_day"] = (vol.visits / days).round(2)
    print(dep.set_index("department_id").join(vol).to_string())

    section("5. WHAT DRIVES WAITING? (median / mean wait in minutes)")
    for col in ["triage_level", "visit_type", "department_id", "outcome"]:
        g = enc.groupby(col)["wait_minutes"].agg(["count", "median", "mean"]).round(1)
        print(f"\nby {col}:\n{g.to_string()}")
    hb = pd.cut(enc.arrival_time.dt.hour, [0, 6, 10, 14, 18, 22, 24], right=False,
                labels=["00-06", "06-10", "10-14", "14-18", "18-22", "22-24"])
    print("\nby arrival hour:\n", enc.groupby(hb, observed=True)["wait_minutes"].agg(["count", "median", "mean"]).round(1).to_string())
    print("\nby weekday (0=Mon):\n", enc.groupby(enc.arrival_time.dt.dayofweek)["wait_minutes"].median().to_dict())
    print("\nby month:\n", enc.groupby(enc.arrival_time.dt.month)["wait_minutes"].median().to_dict())
    print("\nservice_minutes by department (median):", enc.groupby("department_id")["service_minutes"].median().to_dict())

    section("6. WALK-OUTS ('Left before care') & RETURN VISITS")
    lbc = enc.outcome.eq("Left before care")
    print(f"left-before-care rate: {lbc.mean():.3%} ({lbc.sum()} visits)")
    wq = pd.qcut(enc.wait_minutes, 5, duplicates="drop")
    print("LBC rate by wait quintile:", enc.groupby(wq, observed=True).apply(lambda g: round(g.outcome.eq('Left before care').mean(), 3), include_groups=False).to_dict())
    print("LBC rate by triage:", enc.groupby("triage_level").apply(lambda g: round(g.outcome.eq('Left before care').mean(), 3), include_groups=False).to_dict())
    e = enc.sort_values(["patient_id", "arrival_time"])
    nxt = e.groupby("patient_id")["arrival_time"].shift(-1)
    gap_days = (nxt - e.discharge_time).dt.total_seconds() / 86400
    print("visits per patient:", e.groupby("patient_id").size().describe().round(2).to_dict())
    for d in [3, 7, 30]:
        print(f"return within {d} days of discharge: {(gap_days <= d).mean():.2%}")
    rv = (gap_days <= 7)
    print("7-day return rate by chronic_condition:", e.merge(pat, on="patient_id").assign(r=rv.values).groupby("chronic_condition")["r"].mean().round(3).to_dict())

    section("7. DIAGNOSTIC TURNAROUND (minutes)")
    t["tat"] = tat
    print(t.groupby(["test_type", "priority"])["tat"].agg(["count", "median", "mean"]).round(1).to_string())
    print("\nmedian TAT by request hour bucket:",
          t.groupby(pd.cut(t.requested_time.dt.hour, [0, 6, 12, 18, 24], right=False), observed=True)["tat"].median().to_dict())

    section("8. SIGNAL CHECK - can a model beat simple baselines? (train Jan-Oct, test Nov-Dec)")
    df = add_live_features(enc, pat, dep)
    cats = ["department_id", "triage_level", "visit_type", "sex", "chronic_condition", "insurance_type", "home_area"]
    nums = ["hour", "dow", "month", "age", "prior_visits", "waiting_now", "in_service_now",
            "dept_prev_24h", "hosp_prev_3h", "staff_count", "bed_count", "daily_capacity"]
    X = df[cats + nums].copy()
    for c in cats:
        X[c] = X[c].astype("category").cat.codes
    tr = df.arrival_time < VALID_END
    te = ~tr
    y = df["wait_minutes"]
    print(f"train rows {tr.sum()} | test rows {te.sum()}")

    # Baselines
    glob = y[tr].median()
    df["_hb"] = df.hour // 4
    dh = df[tr].groupby(["department_id", "_hb"])["wait_minutes"].median()
    base_dh = df[te].apply(lambda r: dh.get((r.department_id, r._hb), glob), axis=1)
    tri = df[tr].groupby("triage_level")["wait_minutes"].median()
    base_tri = df.loc[te, "triage_level"].map(tri).fillna(glob)
    model = HistGradientBoostingRegressor(categorical_features=[X.columns.get_loc(c) for c in cats],
                                          max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    print(f"WAIT  MAE  global-median baseline : {mean_absolute_error(y[te], np.full(te.sum(), glob)):.2f}")
    print(f"WAIT  MAE  dept x 4h baseline     : {mean_absolute_error(y[te], base_dh):.2f}")
    print(f"WAIT  MAE  triage baseline        : {mean_absolute_error(y[te], base_tri):.2f}")
    print(f"WAIT  MAE  gradient boosting      : {mean_absolute_error(y[te], pred):.2f}   (R2 {r2_score(y[te], pred):.3f})")
    imp = permutation_importance(model, X[te], y[te], n_repeats=3, random_state=RANDOM_STATE, scoring="neg_mean_absolute_error")
    print("top wait drivers (MAE increase when shuffled):",
          dict(sorted(zip(X.columns, imp.importances_mean.round(2)), key=lambda kv: -kv[1])[:8]))

    yl = df.outcome.eq("Left before care").astype(int)
    if yl[tr].nunique() == 2 and yl[te].nunique() == 2:
        clf = HistGradientBoostingClassifier(categorical_features=[X.columns.get_loc(c) for c in cats],
                                             max_iter=200, learning_rate=0.05, random_state=RANDOM_STATE)
        clf.fit(X[tr], yl[tr])
        print(f"\nLBC   AUC  model at arrival (no wait info) : {roc_auc_score(yl[te], clf.predict_proba(X[te])[:, 1]):.3f}")
        print(f"LBC   AUC  actual wait_minutes alone       : {roc_auc_score(yl[te], df.loc[te, 'wait_minutes']):.3f}")
        print(f"LBC   positive rate test: {yl[te].mean():.3%}")

    # Diagnostic TAT signal
    tt = t.merge(df[["encounter_id", "triage_level", "hour"]], on="encounter_id", how="left").dropna(subset=["tat"])
    tt["req_hour"] = tt.requested_time.dt.hour
    trt = tt.requested_time < VALID_END
    base = tt[trt].groupby(["test_type", "priority"])["tat"].median()
    b = tt[~trt].apply(lambda r: base.get((r.test_type, r.priority), tt.tat[trt].median()), axis=1)
    Xt = tt[["test_type", "priority", "department_id", "triage_level", "req_hour"]].copy()
    for c in ["test_type", "priority", "department_id", "triage_level"]:
        Xt[c] = Xt[c].astype("category").cat.codes
    m2 = HistGradientBoostingRegressor(categorical_features=[0, 1, 2, 3], max_iter=300, learning_rate=0.05,
                                       random_state=RANDOM_STATE).fit(Xt[trt], tt.tat[trt])
    print(f"\nTAT   MAE  type x priority baseline : {mean_absolute_error(tt.tat[~trt], b):.2f}")
    print(f"TAT   MAE  gradient boosting        : {mean_absolute_error(tt.tat[~trt], m2.predict(Xt[~trt])):.2f}")

    print("\nDONE - paste everything above back to Claude.")


if __name__ == "__main__":
    main()