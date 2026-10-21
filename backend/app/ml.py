"""Our own models, trained from scratch on the provided data, with honest time-based evaluation."""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
)

from app.config import RANDOM_STATE, TRAIN_END, VALID_END
from app.features import (
    LBC_FEATURES,
    WAIT_CATS,
    WAIT_FEATURES,
    category_map,
    to_model_frame,
)

warnings.filterwarnings("ignore", message=".*eval_set.*")  # harmless notice in newer LightGBM

PARAMS = {"n_estimators": 3000, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 30,
              "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0,
              "random_state": RANDOM_STATE, "verbose": -1}


def split(ts):
    """train = Jan-Sep, valid = Oct (early stopping / choosing baselines), test = Nov-Dec (final report only)."""
    return ts < TRAIN_END, (ts >= TRAIN_END) & (ts < VALID_END), ts >= VALID_END


# ---------------------------------------------------------------- baselines
def _group_median(train, test, keys, target, fallback):
    med = train.groupby(keys)[target].median()
    idx = pd.MultiIndex.from_frame(test[keys]) if len(keys) > 1 else test[keys[0]]
    return pd.Series(med.reindex(idx).values, index=test.index).fillna(fallback)


def wait_baselines(train, other):
    glob = train["wait_minutes"].median()
    tr = train.assign(hb=train["hour"] // 4)
    ot = other.assign(hb=other["hour"] // 4)
    return {
        "global median": pd.Series(glob, index=other.index),
        "triage median": _group_median(tr, ot, ["triage_level"], "wait_minutes", glob),
        "dept x 4h median": _group_median(tr, ot, ["department_id", "hb"], "wait_minutes", glob),
        "dept x triage median": _group_median(tr, ot, ["department_id", "triage_level"], "wait_minutes", glob),
    }


# ---------------------------------------------------------------- wait-time model (P10 / P50 / P90)
def train_wait(df):
    tr, va, te = split(df["arrival_time"])
    cats = category_map(df, WAIT_CATS)
    X = to_model_frame(df, WAIT_FEATURES, cats)
    y = df["wait_minutes"].astype(float)

    models = {}
    for q in (0.1, 0.5, 0.9):
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **PARAMS)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="quantile",
              callbacks=[lgb.early_stopping(150, verbose=False)])
        models[f"p{int(q * 100)}"] = m

    p10, p50, p90 = (models[k].predict(X[te]) for k in ("p10", "p50", "p90"))
    p10, p90 = np.minimum(p10, p50), np.maximum(p90, p50)  # keep the band ordered

    # Pick the strongest baseline on VALIDATION (fair), then report everything on TEST
    base_va = wait_baselines(df[tr], df[va])
    best_name = min(base_va, key=lambda k: mean_absolute_error(y[va], base_va[k]))
    base_te = wait_baselines(df[tr], df[te])
    yt = y[te].values

    model_mae = mean_absolute_error(yt, p50)
    best_mae = mean_absolute_error(yt, base_te[best_name])
    by_dept = (pd.DataFrame({"dept": df.loc[te, "department_name"].values, "err": np.abs(yt - p50)})
               .groupby("dept")["err"].mean().round(2).to_dict())
    imp = pd.Series(models["p50"].booster_.feature_importance("gain"), index=WAIT_FEATURES)
    metrics = {
        "task": "Predict a patient's wait (minutes) at the moment they arrive",
        "rows": {"train": int(tr.sum()), "valid": int(va.sum()), "test": int(te.sum())},
        "split": "time-based: train Jan-Sep 2025, early-stop on Oct, report on Nov-Dec 2025",
        "baselines_test_mae": {k: round(mean_absolute_error(yt, v), 2) for k, v in base_te.items()},
        "best_baseline": best_name,
        "model_test_mae": round(model_mae, 2),
        "improvement_vs_best_baseline_pct": round(100 * (best_mae - model_mae) / best_mae, 1),
        "interval_p10_p90_coverage_pct": round(100 * np.mean((yt >= p10) & (yt <= p90)), 1),
        "interval_mean_width_min": round(float(np.mean(p90 - p10)), 1),
        "mae_by_department": by_dept,
        "best_iterations": {k: int(m.best_iteration_ or PARAMS["n_estimators"]) for k, m in models.items()},
        "feature_importance_gain": (imp / imp.sum()).sort_values(ascending=False).round(4).to_dict(),
    }
    return {"models": models, "features": WAIT_FEATURES, "categories": cats}, metrics


def predict_wait(bundle, df):
    X = to_model_frame(df, bundle["features"], bundle["categories"])
    p = {k: bundle["models"][k].predict(X) for k in ("p10", "p50", "p90")}
    p["p10"] = np.clip(np.minimum(p["p10"], p["p50"]), 0, None)
    p["p90"] = np.maximum(p["p90"], p["p50"])
    return pd.DataFrame(p, index=df.index)


# ---------------------------------------------------------------- walk-out ("left before care") risk
def train_lbc(df):
    tr, va, te = split(df["arrival_time"])
    cats = category_map(df, WAIT_CATS)
    X = to_model_frame(df, LBC_FEATURES, cats)
    y = df["is_lbc"].astype(int)

    params = {**PARAMS, "n_estimators": 2000, "min_child_samples": 40, "num_leaves": 7}
    m = lgb.LGBMClassifier(objective="binary", **params)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(150, verbose=False)])
    pt = m.predict_proba(X[te])[:, 1]
    yt = y[te].values

    def recall_at_top(score, frac=0.10):
        k = max(int(len(score) * frac), 1)
        top = np.argsort(-np.asarray(score))[:k]
        return float(yt[top].sum() / max(yt.sum(), 1))

    base_wait = df.loc[te, "wait_minutes"].values  # "longer wait = higher risk" rule
    tri_rate = df[tr].groupby("triage_level")["is_lbc"].mean()
    base_tri = df.loc[te, "triage_level"].map(tri_rate).fillna(y[tr].mean()).values
    has_both = len(np.unique(yt)) == 2

    def auc(s):
        return round(roc_auc_score(yt, s), 3) if has_both else None

    def ap(s):
        return round(average_precision_score(yt, s), 3) if has_both else None

    imp = pd.Series(m.booster_.feature_importance("gain"), index=LBC_FEATURES)
    metrics = {
        "task": "Probability a waiting patient leaves before being seen, given how long they have waited",
        "rows": {"train": int(tr.sum()), "valid": int(va.sum()), "test": int(te.sum())},
        "positive_rate_test_pct": round(100 * yt.mean(), 2),
        "baselines": {
            "wait-only rule": {"auc": auc(base_wait), "pr_auc": ap(base_wait),
                               "recall_top10pct": round(recall_at_top(base_wait), 3)},
            "triage rate": {"auc": auc(base_tri), "pr_auc": ap(base_tri),
                            "recall_top10pct": round(recall_at_top(base_tri), 3)},
        },
        "model": {"auc": auc(pt), "pr_auc": ap(pt), "recall_top10pct": round(recall_at_top(pt), 3),
                  "brier": round(brier_score_loss(yt, pt), 4)},
        "best_iteration": int(m.best_iteration_ or params["n_estimators"]),
        "feature_importance_gain": (imp / max(imp.sum(), 1e-9)).sort_values(ascending=False).round(4).to_dict(),
    }
    return {"model": m, "features": LBC_FEATURES, "categories": cats}, metrics


def predict_lbc(bundle, df, wait_override=None):
    d = df.copy()
    if wait_override is not None:
        d["wait_minutes"] = wait_override
    X = to_model_frame(d, bundle["features"], bundle["categories"])
    return bundle["model"].predict_proba(X)[:, 1]