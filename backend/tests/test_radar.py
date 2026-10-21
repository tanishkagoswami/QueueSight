"""The radar ranks by minutes past the predicted range, and adds walk-out risk only when that model is trusted live."""
import pytest
from app.snapshot import LEFT_BONUS, WALKOUT_WEIGHT, priority


def test_priority_is_minutes_overdue_when_risk_is_not_used():
    assert priority(20, risk=0.9) == 20          # risk ignored unless use_risk=True
    assert priority(-15, risk=0.9) == 0          # still within the predicted range
    assert priority(20, left=True) == 20 + LEFT_BONUS


def test_each_percent_of_risk_counts_like_a_minute_when_trusted():
    assert WALKOUT_WEIGHT == 100
    assert priority(-15, risk=0.60, use_risk=True) == pytest.approx(60)
    assert priority(10, risk=0.05, use_risk=True) == pytest.approx(15)


def test_a_likely_walkout_outranks_a_slightly_overdue_patient_only_when_trusted():
    slightly_overdue = {"minutes_over": 10, "risk": 0.02}
    likely_to_leave = {"minutes_over": -5, "risk": 0.60}   # still within range, but 60% likely to walk out
    for trusted, first in [(True, "leaver"), (False, "overdue")]:
        a = priority(**slightly_overdue, use_risk=trusted)
        b = priority(**likely_to_leave, use_risk=trusted)
        assert (b > a) == (first == "leaver")