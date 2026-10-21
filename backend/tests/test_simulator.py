"""The simulator must be reproducible, match the dataset format exactly, and behave like a real hospital."""
import pandas as pd
import pytest
from app.config import ROOT_DIR
from sim.config import SimConfig
from sim.engine import run
from sim.export import to_tables

CLINICS = ["D02", "D04", "D05", "D08", "D09", "D10", "D11"]  # clinic-style departments (close at night)


def simulate(days=30, seed=1):
    return to_tables(run(SimConfig(days=days, seed=seed)))


@pytest.fixture(scope="module")
def hosp():
    return simulate()


@pytest.fixture(scope="module")
def enc(hosp):
    e = hosp["encounters"].copy()
    for c in ["arrival_time", "service_start_time", "service_end_time", "discharge_time"]:
        e[c] = pd.to_datetime(e[c])
    return e


def minutes(a, b):
    return (b - a).dt.total_seconds() / 60


# ---------------------------------------------------------------- reproducible
def test_same_seed_gives_identical_hospital():
    a, b = simulate(days=7, seed=3), simulate(days=7, seed=3)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_different_seed_gives_different_hospital():
    assert not simulate(days=7, seed=3)["encounters"].equals(simulate(days=7, seed=4)["encounters"])


# ---------------------------------------------------------------- same format as the DATATHON dataset
@pytest.mark.parametrize("name", ["patients", "encounters", "departments", "diagnostic_tests"])
def test_columns_match_the_real_dataset(hosp, name):
    real_columns = list(pd.read_csv(ROOT_DIR / "data" / "raw" / f"{name}.csv", nrows=0).columns)
    assert list(hosp[name].columns) == real_columns


def test_ids_are_unique_and_linked(hosp):
    e, t, p = hosp["encounters"], hosp["diagnostic_tests"], hosp["patients"]
    assert e["encounter_id"].is_unique and t["test_id"].is_unique and p["patient_id"].is_unique
    assert e["patient_id"].isin(p["patient_id"]).all()
    assert t["encounter_id"].isin(e["encounter_id"]).all()
    assert e["department_id"].isin(hosp["departments"]["department_id"]).all()


# ---------------------------------------------------------------- internally consistent
def test_walkouts_have_no_service_times_and_everyone_else_does(enc):
    left = enc["outcome"].eq("Left before care")
    assert left.any()
    assert enc.loc[left, ["service_start_time", "service_end_time"]].isna().all().all()
    assert enc.loc[~left, ["service_start_time", "service_end_time"]].notna().all().all()


def test_wait_minutes_match_the_timestamps(enc):
    left = enc["outcome"].eq("Left before care")
    seen = enc[~left]
    assert (minutes(seen["arrival_time"], seen["service_start_time"]) == seen["wait_minutes"]).all()
    gone = enc[left]  # for walk-outs, discharge_time is when they left
    assert (minutes(gone["arrival_time"], gone["discharge_time"]) == gone["wait_minutes"]).all()


def test_times_are_in_order(enc, hosp):
    seen = enc[enc["service_start_time"].notna()]
    assert (seen["service_start_time"] >= seen["arrival_time"]).all()
    assert (seen["service_end_time"] >= seen["service_start_time"]).all()
    assert (seen["discharge_time"] >= seen["service_end_time"]).all()
    t = hosp["diagnostic_tests"].merge(enc, on="encounter_id")
    req, done = pd.to_datetime(t["requested_time"]), pd.to_datetime(t["completion_time"])
    assert (done > req).all()
    assert ((req >= t["service_start_time"]) & (req <= t["service_end_time"])).all()


# ---------------------------------------------------------------- behaves like a hospital
def test_sicker_patients_are_seen_sooner(enc):
    med = enc.groupby("triage_level")["wait_minutes"].median()
    assert med["Critical"] < med["High"] < med["Medium"] < med["Low"]


def test_critical_patients_never_walk_out(enc):
    assert not enc.loc[enc["triage_level"].eq("Critical"), "outcome"].eq("Left before care").any()


def test_walkouts_rise_with_waiting(enc):
    left = enc["outcome"].eq("Left before care")
    short = left[enc["wait_minutes"].between(15, 30)].mean()
    long = left[enc["wait_minutes"].between(60, 120)].mean()
    assert long > 5 * short


def test_walkout_rate_is_realistic(enc):
    assert 0.01 < enc["outcome"].eq("Left before care").mean() < 0.10


def test_clinics_take_no_new_patients_at_night(enc):
    hour = enc.loc[enc["department_id"].isin(CLINICS), "arrival_time"].dt.hour
    assert hour.between(7, 18).all()


def test_pediatrics_only_sees_children(hosp):
    e, p = hosp["encounters"], hosp["patients"]
    ages = e[e["department_id"].eq("D03")].merge(p, on="patient_id")["age"]
    assert (ages < 16).all()


def test_urgent_tests_come_back_faster(hosp):
    t = hosp["diagnostic_tests"]
    tat = minutes(pd.to_datetime(t["requested_time"]), pd.to_datetime(t["completion_time"]))
    med = tat.groupby(t["priority"]).median()
    assert med["Urgent"] < med["Routine"]