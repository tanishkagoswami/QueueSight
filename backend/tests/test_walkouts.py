"""A patient who walks out stops "waiting" the moment they leave: in the data, the model features and the board."""
import pandas as pd
import pytest
from app.data import ENC_TIMES, TEST_TIMES, clean
from app.features import _live_load
from app.snapshot import queue_state
from sim.config import SimConfig
from sim.engine import run
from sim.export import to_tables

T0 = pd.Timestamp("2025-03-01 10:00")


def at(minute):
    return T0 + pd.Timedelta(minutes=minute)


@pytest.fixture
def visits():
    """Three patients in one department:
    A arrives 10:00 and walks out unseen at 10:30
    B arrives 10:10, is seen 10:50-11:05
    C arrives 10:40, is seen 11:00-11:20"""
    df = pd.DataFrame({
        "encounter_id": ["A", "B", "C"],
        "department_id": ["D01"] * 3,
        "arrival_time": [at(0), at(10), at(40)],
        "service_start_time": [pd.NaT, at(50), at(60)],
        "service_end_time": [pd.NaT, at(65), at(80)],
        "wait_minutes": [30.0, 40.0, 20.0],
        "is_lbc": [1, 0, 0],
    })
    df["wait_end_time"] = [at(30), at(50), at(60)]
    return df


def test_wait_end_time_is_when_called_in_or_when_a_walkout_left():
    t = to_tables(run(SimConfig(days=7, seed=2)))
    raw = {"patients": t["patients"], "departments": t["departments"],
           "encounters": t["encounters"], "tests": t["diagnostic_tests"]}
    for name, cols in [("encounters", ENC_TIMES), ("tests", TEST_TIMES)]:
        raw[name] = raw[name].assign(**{c: pd.to_datetime(raw[name][c]) for c in cols})
    enc = clean(raw)["encounters"]
    left = enc["is_lbc"].eq(1)
    assert left.any()
    seen = enc[~left]
    assert (seen["wait_end_time"] == seen["service_start_time"]).all()
    gone = enc[left]  # the simulator records the moment a walk-out left as their discharge time
    assert (gone["wait_end_time"] == gone["discharge_time"]).all()


def test_a_walkout_counts_as_waiting_until_they_leave(visits):
    waiting_now, *_ = _live_load(visits)
    # B arrives while A is still waiting; C arrives after A has left, while B is still waiting
    assert list(waiting_now) == [0, 1, 1]


@pytest.mark.parametrize("minute, waiting, left, in_service", [
    (20, {"A", "B"}, set(), set()),     # 10:20  A and B are both waiting
    (45, {"B", "C"}, {"A"}, set()),     # 10:45  A has walked out, C has arrived
    (55, {"C"}, {"A"}, {"B"}),          # 10:55  B is being seen
    (90, set(), {"A"}, set()),          # 11:30  everyone is done; A still counts as left
])
def test_where_each_patient_stands(visits, minute, waiting, left, in_service):
    q = queue_state(visits, at(minute))
    ids = visits["encounter_id"]
    assert set(ids[q["waiting"]]) == waiting
    assert set(ids[q["left"]]) == left
    assert set(ids[q["in_service"]]) == in_service