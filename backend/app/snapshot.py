"""What the hospital looks like at any moment T: department board, KPIs and the Forgotten Patient Radar.
T is a 'replay clock' over the 2025 data - in a real deployment T is simply 'now' on a live feed."""
import numpy as np
import pandas as pd

from app.explain import contributions
from app.state import STATE


def _iso(x):
    return None if pd.isna(x) else pd.Timestamp(x).isoformat()


def _num(x, nd=1):
    return None if x is None or pd.isna(x) else round(float(x), nd)


def _status(ratio):
    if ratio is None:
        return "quiet"
    return "critical" if ratio >= 1.4 else "elevated" if ratio >= 1.1 else "normal"

# Each 1% of walk-out risk counts like 1 minute past the predicted range. Chosen on October 2025 and
# confirmed on Nov-Dec with scripts/evaluate_radar.py. Only used when the walk-out model is trusted.
WALKOUT_WEIGHT = 100
LEFT_BONUS = 30  # someone who already left unseen ranks above an equally overdue patient


def priority(minutes_over, risk=None, left=False, use_risk=False, weight=WALKOUT_WEIGHT):
    """How urgently staff should look at a patient: minutes past the predicted range,
    plus walk-out risk when the model is trusted for this hospital, plus a bump for anyone who already left."""
    score = np.maximum(minutes_over, 0)
    if use_risk and risk is not None:
        score = score + weight * np.asarray(risk)
    return score + (LEFT_BONUS if left else 0)

def queue_state(df, t):
    """Where every visit stands at time t. A patient stops waiting when they are called in
    or when they walk out, so a walk-out never stays on the board as "waiting" forever."""
    arrived = df["arrival_time"] <= t
    ended = df["wait_end_time"] <= t                  # called in, or gave up and left
    started = df["service_start_time"] <= t           # actually called in (empty for walk-outs -> False)
    return {
        "arrived": arrived,
        "waiting": arrived & ~ended,
        "ended": ended,
        "started": started,
        "left": ended & (df["is_lbc"] == 1),
        "in_service": started & (df["service_end_time"] > t),
    }


def board(t):
    df = STATE.df
    day0 = t.normalize()
    q = queue_state(df, t)
    waiting, started, in_service = q["waiting"], q["started"], q["in_service"]
    today = q["arrived"] & (df["arrival_time"] >= day0)
    seen_today = today & started          # median waits use patients who were called in
    ended_today = today & q["ended"]
    elapsed = (t - df["arrival_time"]).dt.total_seconds() / 60

    # A visit is "overdue" when its (finished or so-far) wait exceeds our P90 prediction
    over_done = ended_today & (df["wait_minutes"] > df["pred_p90"])
    over_wait = waiting & (elapsed > df["pred_p90"])
    left_today = today & q["left"]        # counted the moment the patient leaves

    week0 = day0 - pd.Timedelta(days=6)
    week = df[(df["arrival_time"] >= week0) & (df["arrival_time"] <= t) & started]

    rows = []
    for dep_id, name in STATE.dept_names.items():
        m = df["department_id"] == dep_id
        st = df[seen_today & m]
        normal = STATE.normal_wait.get(dep_id)
        med_today = st["wait_minutes"].median() if len(st) else None
        ratio = (med_today / normal) if (med_today is not None and normal) else None
        trend = (week[week["department_id"] == dep_id]
                 .groupby(week["arrival_time"].dt.normalize())["wait_minutes"].median()
                 .reindex(pd.date_range(week0, day0, freq="D")))
        rows.append({
            "department_id": dep_id, "name": name,
            "waiting_now": int((waiting & m).sum()), "in_service_now": int((in_service & m).sum()),
            "arrivals_today": int((today & m).sum()), "seen_today": len(st),
            "median_wait_today": _num(med_today), "normal_wait": _num(normal),
            "predicted_median_today": _num(df.loc[today & m, "pred_p50"].median()),
            "ratio_vs_normal": _num(ratio, 2), "status": _status(ratio),
            "overdue_today": int(((over_done | over_wait) & m).sum()),
            "walkouts_today": int((left_today & m).sum()),
            "trend_7d": [_num(v) for v in trend.values],
        })
    rank = {"critical": 0, "elevated": 1, "normal": 2, "quiet": 3}
    rows.sort(key=lambda r: (rank[r["status"]], -(r["ratio_vs_normal"] or 0), -r["waiting_now"]))

    kpis = {
        "waiting_now": int(waiting.sum()),
        "in_service_now": int(in_service.sum()),
        "arrivals_today": int(today.sum()),
        "median_wait_today": _num(df.loc[seen_today, "wait_minutes"].median()),
        "overdue_today": int((over_done | over_wait).sum()),
        "walkouts_today": int(left_today.sum()),
        "departments_under_pressure": sum(r["status"] in ("critical", "elevated") for r in rows),
    }
    return kpis, rows


def radar(t, hours=24, limit=40):
    """Patients at risk of being forgotten: waiting past their predicted range, likely to walk out, or already gone."""
    df = STATE.df
    lo = t - pd.Timedelta(hours=hours)
    q = queue_state(df, t)
    elapsed = (t - df["arrival_time"]).dt.total_seconds() / 60

    items = []
    w = df[q["waiting"]]
    if len(w):
        risk = STATE.lbc_risk(w, elapsed[w.index].values)
        for (i, r), rk in zip(w.iterrows(), risk):
            el = float(elapsed[i])
            items.append(_item(r, "waiting", el, rk))
    recent = df[q["ended"] & (df["arrival_time"] >= lo)]
    for i, r in recent[recent["is_lbc"] == 1].iterrows():
        items.append(_item(r, "left_unseen", float(r["wait_minutes"]), None))
    for i, r in recent[(recent["is_lbc"] == 0) & (recent["wait_minutes"] > recent["pred_p90"])].iterrows():
        items.append(_item(r, "was_overdue", float(r["wait_minutes"]), None))

    order = {"waiting": 0, "left_unseen": 1, "was_overdue": 2}
    items.sort(key=lambda x: (order[x["state"]], -(x["priority_score"])))
    return items[:limit]


def _item(r, state, waited, risk_now):
    over = waited - r["pred_p90"]
    score = priority(over, risk_now, left=state == "left_unseen", use_risk=STATE.walkout_model["trusted"])
    return {
        "encounter_id": r["encounter_id"], "patient_id": r["patient_id"],
        "department": STATE.dept_names.get(r["department_id"]), "department_id": r["department_id"],
        "triage_level": r["triage_level"], "visit_type": r["visit_type"], "age": _num(r["age"], 0),
        "arrival_time": _iso(r["arrival_time"]), "state": state,
        "waited_min": _num(waited), "pred_p10": _num(r["pred_p10"]), "pred_p50": _num(r["pred_p50"]),
        "pred_p90": _num(r["pred_p90"]), "minutes_over_p90": _num(over),
        "overdue": bool(over > 0), "walkout_risk_now": _num(risk_now, 3) if risk_now is not None else None,
        "priority_score": round(float(score), 1),
    }


def encounter_detail(encounter_id, t):
    df = STATE.df
    hit = df[df["encounter_id"] == encounter_id]
    if hit.empty:
        return None
    row = hit.iloc[[0]]
    r = row.iloc[0]
    ended = r["wait_end_time"] <= t               # called in or walked out by time t
    left = ended and r["is_lbc"] == 1
    waited = float(r["wait_minutes"]) if ended else max(float((t - r["arrival_time"]).total_seconds() / 60), 0)

    wb, lb = STATE.wait_bundle, STATE.lbc_bundle
    why_wait = contributions(wb["models"]["p50"], row, wb["features"], wb["categories"], STATE.dept_names, unit="min")
    row_now = row.assign(wait_minutes=waited)
    why_risk = contributions(lb["model"], row_now, lb["features"], lb["categories"], STATE.dept_names, unit="logodds")
    grid = np.arange(0, 181, 10)
    curve = STATE.lbc_risk(pd.concat([row] * len(grid), ignore_index=True), grid)

    tests = STATE.tests[STATE.tests["encounter_id"] == encounter_id]
    history = df[(df["patient_id"] == r["patient_id"]) & (df["arrival_time"] < r["arrival_time"])]
    return {
        "encounter_id": r["encounter_id"], "patient_id": r["patient_id"],
        "patient": {"age": _num(r["age"], 0), "sex": r["sex"], "chronic_condition": r["chronic_condition"],
                    "insurance_type": r["insurance_type"], "home_area": r["home_area"]},
        "department": STATE.dept_names.get(r["department_id"]), "triage_level": r["triage_level"],
        "visit_type": r["visit_type"], "arrival_time": _iso(r["arrival_time"]),
        "status": "left" if left else "seen" if ended else "waiting", "waited_min": _num(waited),
        "outcome": r["outcome"] if ended else None,
        "prediction": {"p10": _num(r["pred_p10"]), "p50": _num(r["pred_p50"]), "p90": _num(r["pred_p90"])},
        "why_wait": why_wait, "why_risk": why_risk,
        "walkout_risk_now": _num(STATE.lbc_risk(row_now, [waited])[0], 3),
        "risk_curve": [{"wait_min": int(g), "risk": round(float(p), 4)} for g, p in zip(grid, curve)],
        "tests": [{"test_id": x.test_id, "type": x.test_type, "priority": x.priority,
                   "requested": _iso(x.requested_time), "completed": _iso(x.completion_time),
                   "tat_min": _num(x.tat_minutes)} for x in tests.itertuples()],
        "history": [{"encounter_id": h.encounter_id, "arrival_time": _iso(h.arrival_time),
                     "department": STATE.dept_names.get(h.department_id), "outcome": h.outcome,
                     "wait_min": _num(h.wait_minutes)} for h in history.tail(10).itertuples()],
    }