"""
QueueSight - build the dataset, run quality checks, train + evaluate models, save everything to backend/artifacts/.

Run from inside backend/ :   python -m scripts.train_models
"""
import json
import time

import joblib
from app.config import ARTIFACTS_DIR
from app.data import load_clean
from app.features import build_encounter_features
from app.live_eval import evaluate_live
from app.ml import predict_lbc, predict_wait, train_lbc, train_wait
from app.quality import run_checks


def save_json(obj, name):
    path = ARTIFACTS_DIR / name
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


def main():
    t0 = time.time()
    (ARTIFACTS_DIR / "models").mkdir(parents=True, exist_ok=True)

    print("1/6 loading + cleaning data ...")
    raw, clean = load_clean()
    quality = run_checks(raw, clean)
    save_json(quality, "quality.json")
    print(f"    {quality['summary']['clean_rows']} | {quality['summary']['issues_found']} data issues logged")

    print("2/6 building leakage-safe features ...")
    df = build_encounter_features(clean)

    print("3/6 training wait-time model (P10/P50/P90) ...")
    wait_bundle, wait_m = train_wait(df)
    joblib.dump(wait_bundle, ARTIFACTS_DIR / "models" / "wait_model.joblib")

    print("4/6 training walk-out risk model ...")
    lbc_bundle, lbc_m = train_lbc(df)

    print("5/6 replaying real moments to judge walk-out risk live + calibrating it ...")
    lbc_bundle["live_calibrator"], lbc_m["live"] = evaluate_live(df, lbc_bundle)
    joblib.dump(lbc_bundle, ARTIFACTS_DIR / "models" / "lbc_model.joblib")

    print("6/6 scoring every visit + saving ...")
    preds = predict_wait(wait_bundle, df)
    out = df[["encounter_id", "patient_id", "department_id", "department_name", "arrival_time",
              "service_start_time", "triage_level", "visit_type", "outcome", "wait_minutes"]].copy()
    out[["pred_p10", "pred_p50", "pred_p90"]] = preds.round(1).values
    out["lbc_risk_actual_wait"] = predict_lbc(lbc_bundle, df).round(4)
    out["overdue_vs_p90"] = (out["wait_minutes"] > out["pred_p90"]).astype(int)
    out.to_csv(ARTIFACTS_DIR / "encounter_predictions.csv", index=False)
    save_json({"wait_time": wait_m, "walkout_risk": lbc_m}, "metrics.json")

    print("\n================ RESULTS (test = Nov-Dec 2025, never seen in training) ================")
    print(f"WAIT  baselines MAE : {wait_m['baselines_test_mae']}")
    print(f"WAIT  our model MAE : {wait_m['model_test_mae']}  "
          f"({wait_m['improvement_vs_best_baseline_pct']}% better than '{wait_m['best_baseline']}')")
    print(f"WAIT  P10-P90 band covers {wait_m['interval_p10_p90_coverage_pct']}% of real waits "
          f"(avg width {wait_m['interval_mean_width_min']} min)")
    print(f"WAIT  top drivers   : {list(wait_m['feature_importance_gain'])[:6]}")
    print(f"LBC   positive rate : {lbc_m['positive_rate_test_pct']}%")
    print(f"LBC   baselines     : {lbc_m['baselines']}")
    print(f"LBC   our model     : {lbc_m['model']}   <- in hindsight (uses each visit's final wait)")
    live = lbc_m["live"]
    print(f"LIVE  walk-out risk : {live['model']}   rules: {live['rules']}")
    top = [b for b in live["calibration_before"] if b["n"] >= 20][-1:]
    for b in top:
        print(f"LIVE  calibration   : highest band shown {b['shown']:.0%} before calibration, really left {b['really_left']:.0%}")
    print(f"LBC   top drivers   : {list(lbc_m['feature_importance_gain'])[:6]}")
    print(f"\nsaved to {ARTIFACTS_DIR}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()