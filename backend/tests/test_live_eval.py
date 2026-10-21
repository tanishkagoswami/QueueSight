"""The walk-out model is judged the way the radar uses it: live, knowing only how long each patient has waited so far."""
import pandas as pd
import pytest
from app.live_eval import fit_calibrator, replay
from app.state import walkout_verdict

T0 = pd.Timestamp("2025-03-01 10:00")


def at(minute):
    return T0 + pd.Timedelta(minutes=minute)


@pytest.fixture
def visits():
    """A walks out unseen at 10:30, B waits 10:10-10:50, C waits 10:40-11:00."""
    return pd.DataFrame({
        "encounter_id": ["A", "B", "C"],
        "arrival_time": [at(0), at(10), at(40)],
        "wait_end_time": [at(30), at(50), at(60)],
        "triage_level": ["Low", "Medium", "Low"],
        "is_lbc": [1, 0, 0],
    })


def test_replay_sees_only_patients_still_waiting_and_how_long_so_far(visits):
    snaps = replay(visits, at(20), at(50), risk_fn=lambda rows, waited: waited / 100, every="10min")
    # 10:20 A,B   10:30 B (A just left)   10:40 B,C   10:50 C (B just called in)
    assert list(snaps["waited"]) == [20, 10, 20, 30, 0, 10]
    assert list(snaps["left"]) == [1, 0, 0, 0, 0, 0]          # only A's moment counts as a later walk-out
    assert list(snaps["risk"]) == pytest.approx([0.2, 0.1, 0.2, 0.3, 0.0, 0.1])


def test_calibration_turns_overconfident_scores_into_real_rates():
    shown_90 = pd.DataFrame({"risk": [0.9] * 10, "left": [1] * 5 + [0] * 5})   # model says 90%, really 50%
    shown_20 = pd.DataFrame({"risk": [0.2] * 10, "left": [1] * 2 + [0] * 8})   # model says 20%, really 20%
    cal = fit_calibrator(pd.concat([shown_90, shown_20]), min_per_step=10)
    assert cal.predict([0.9, 0.2]) == pytest.approx([0.5, 0.2])
    assert cal.predict([0.99])[0] == pytest.approx(0.5)                          # never above what really happened

def test_a_handful_of_extreme_cases_cannot_create_an_extreme_step():
    usual = pd.DataFrame({"risk": [0.1] * 400, "left": [1] * 40 + [0] * 360})    # 10% really leave
    three_extremes = pd.DataFrame({"risk": [0.99] * 3, "left": [1] * 3})         # 3 of 3 left: too few to trust
    cal = fit_calibrator(pd.concat([usual, three_extremes]), min_per_step=200)
    assert cal.predict([0.99])[0] < 0.2                                          # not "100% will walk out"

def metrics(hindsight_auc, live_auc, *rule_aucs):
    m = {"walkout_risk": {"model": {"auc": hindsight_auc}}}
    if live_auc is not None:
        m["walkout_risk"]["live"] = {"model": {"auc": live_auc},
                                     "rules": {f"rule {i}": {"auc": a} for i, a in enumerate(rule_aucs)}}
    return m


@pytest.mark.parametrize("hindsight, live, rules, trusted", [
    (0.949, 0.790, (0.710, 0.713), True),    # simulated hospital: good live, beats the rules
    (0.533, 0.496, (0.505, 0.489), False),   # datathon: a coin flip live too
    (0.950, 0.600, (0.550,), False),         # great in hindsight, weak live -> NOT trusted
    (0.900, 0.800, (0.850,), False),         # a simple rule does better live -> use the rule
    (0.949, None, (), False),                # never measured live -> not trusted
])
def test_trust_is_judged_live_not_in_hindsight(hindsight, live, rules, trusted):
    assert walkout_verdict(metrics(hindsight, live, *rules))["trusted"] is trusted