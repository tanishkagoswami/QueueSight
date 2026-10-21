"""Discrete-event simulation of patient flow (SimPy).

arrive -> registration & triage -> queue for a clinician (sickest first) -> consultation -> outcome
A waiting patient whose patience runs out leaves before being seen. Clinicians come and go with their shifts.
All randomness is drawn up front from one seeded generator, so the same seed always gives the same hospital.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import simpy

from sim.config import (
    HOURLY_PROFILE,
    TEST_TYPES,
    VISIT_TYPE_MIX,
    WEEKDAY_FACTOR,
    Department,
    SimConfig,
)
from sim.population import make_population, pick_patients

MIN_PER_DAY = 1440
HANDOVER_PRIORITY = -1  # a clinician going off shift finishes their current patient, then leaves
CLOSING_PRIORITY = 99   # at closing time (0 on duty) clinicians stay until everyone already waiting is seen


@dataclass
class SimResult:
    config: SimConfig
    population: pd.DataFrame
    visits: pd.DataFrame          # one row per visit, times in minutes from the start date
    tests: pd.DataFrame
    utilisation: pd.DataFrame     # per department: clinician minutes on duty vs minutes spent with patients


def _arrivals(cfg: SimConfig, dept: Department, day_noise: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Non-homogeneous Poisson arrivals: expected count per hour = daily average x hour share x weekday x season x day noise."""
    start = pd.Timestamp(cfg.start_date)
    days = pd.date_range(start, periods=cfg.days, freq="D")
    hour_share = np.asarray(HOURLY_PROFILE[dept.profile], float)
    hour_share /= hour_share.sum()
    weekday = np.asarray(WEEKDAY_FACTOR[dept.profile])[days.dayofweek]
    season = 1 + cfg.seasonal_amplitude * np.cos(2 * np.pi * (days.dayofyear.to_numpy() - 1) / 365.25)
    expected = dept.daily_arrivals * (weekday * season * day_noise)[:, None] * hour_share[None, :]  # days x 24
    counts = rng.poisson(expected).ravel()
    hour_starts = np.arange(cfg.days * 24) * 60.0
    return np.sort(np.repeat(hour_starts, counts) + rng.uniform(0, 60, counts.sum()))


def _shift_changes(cfg: SimConfig, dept: Department) -> list[tuple[float, int]]:
    """(minute, clinicians on duty) for every shift boundary in the horizon, starting at minute 0."""
    start = pd.Timestamp(cfg.start_date)
    hours = sorted(dept.shifts)
    changes = []
    for d in range(cfg.days + 2):  # +2 days so the queue can drain after the last arrival
        weekend = (start + pd.Timedelta(days=d)).dayofweek >= 5
        for h in hours:
            staff = dept.shifts[h]
            if weekend and dept.profile == "clinic" and staff > 0:
                staff = max(1, round(staff * cfg.weekend_staff_factor))
            changes.append((d * MIN_PER_DAY + h * 60.0, staff))
    night_staff = changes[-1][1]  # the shift running overnight before the first boundary of day 0
    return [(0.0, night_staff)] + changes


def _shift_manager(env, res, capacity, changes, duty_log):
    """Keeps (capacity - clinicians on duty) servers blocked by 'off shift' requests."""
    blockers = []
    for i, (t, staff) in enumerate(changes):
        if t > env.now:
            yield env.timeout(t - env.now)
        target = capacity - staff
        while len(blockers) > target:
            req = blockers.pop()
            if req.triggered:
                res.release(req)
            else:
                req.cancel()
        priority = CLOSING_PRIORITY if staff == 0 else HANDOVER_PRIORITY
        while len(blockers) < target:
            blockers.append(res.request(priority=priority))
        end = changes[i + 1][0] if i + 1 < len(changes) else t
        duty_log.append((t, end, staff))


def _visit(env, i, res, triage, intake, patience, service, start_out, left_out):
    # Registration and triage happen for everyone, however quiet the department is
    if patience[i] <= intake[i]:
        yield env.timeout(patience[i])
        left_out[i] = env.now
        return
    yield env.timeout(intake[i])
    req = res.request(priority=int(triage[i]))
    if np.isinf(patience[i]):
        yield req
    else:
        yield req | env.timeout(patience[i] - intake[i])
        if not req.triggered:  # patience ran out before a clinician was free
            req.cancel()
            left_out[i] = env.now
            return
    start_out[i] = env.now
    yield env.timeout(service[i])
    res.release(req)


def _arrival_driver(env, order, arrival, procs):
    for i in order:
        if arrival[i] > env.now:
            yield env.timeout(arrival[i] - env.now)
        env.process(procs(i))


def run(cfg: SimConfig | None = None) -> SimResult:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    pop = make_population(cfg.n_patients, rng)
    shape = 1 / cfg.daily_noise_cv ** 2
    day_noise = rng.gamma(shape, 1 / shape, cfg.days)  # mean 1, shared by all departments (busy days hit everyone)

    # ---- draw every visit and all its random attributes up front
    blocks = []
    for d_idx, dept in enumerate(cfg.departments):
        t = _arrivals(cfg, dept, day_noise, rng)
        n = len(t)
        mix = dept.visit_mix or VISIT_TYPE_MIX[dept.profile]
        blocks.append(pd.DataFrame({
            "dept_idx": d_idx,
            "arrival": t,
            "patient_idx": pick_patients(pop, dept.patients, n, rng),
            "triage": rng.choice(4, n, p=dept.triage_mix),
            "visit_type": rng.choice(list(mix), n, p=list(mix.values())),
        }))
    v = pd.concat(blocks, ignore_index=True)
    n = len(v)
    tri = v["triage"].to_numpy()
    med = np.array([cfg.departments[d].service_median_min for d in v["dept_idx"]])
    v["service"] = np.maximum(3.0, rng.lognormal(np.log(med * np.asarray(cfg.triage_service_factor)[tri]), cfg.service_sigma))
    pat_med = np.asarray(cfg.patience_median_min)[tri]
    v["patience"] = np.where(np.isinf(pat_med), np.inf,
                             rng.lognormal(np.log(np.where(np.isinf(pat_med), 1, pat_med)), cfg.patience_sigma))
    v["intake"] = rng.lognormal(np.log(np.asarray(cfg.intake_median_min)[tri]), cfg.intake_sigma)
    v["u_outcome"] = rng.random(n)
    v["after"] = rng.lognormal(np.log(cfg.after_service_median_min), 0.6, n)
    v["boarding"] = rng.lognormal(np.log(cfg.boarding_median_min), 0.7, n)

    # ---- run the queues
    env = simpy.Environment()
    arrival = v["arrival"].to_numpy()
    triage = tri
    intake = v["intake"].to_numpy()
    patience = v["patience"].to_numpy()
    service = v["service"].to_numpy()
    start = np.full(n, np.nan)
    left = np.full(n, np.nan)
    resources, duty = [], []
    for dept in cfg.departments:
        changes = _shift_changes(cfg, dept)
        cap = max(s for _, s in changes)
        res = simpy.PriorityResource(env, capacity=cap)
        log = []
        env.process(_shift_manager(env, res, cap, changes, log))  # created first so staffing is set at minute 0
        resources.append(res)
        duty.append(log)
    dept_of = v["dept_idx"].to_numpy()

    def procs(i):
        return _visit(env, i, resources[dept_of[i]], triage, intake, patience, service, start, left)

    env.process(_arrival_driver(env, np.argsort(arrival, kind="stable"), arrival, procs))
    horizon = cfg.days * MIN_PER_DAY
    env.run(until=horizon + 2 * MIN_PER_DAY)

    v["start"], v["left"] = start, left
    v = v[v["start"].notna() | v["left"].notna()].reset_index(drop=True)  # drop anyone unresolved (should be none)

    # ---- outcomes
    seen = v["start"].notna()
    adm = np.asarray(cfg.admit_prob)[v["triage"]]
    trf = np.asarray(cfg.transfer_prob)[v["triage"]]
    u = v["u_outcome"].to_numpy()
    v["outcome"] = np.where(~seen, "Left before care",
                            np.where(u < adm, "Admitted", np.where(u < adm + trf, "Transferred", "Discharged")))
    v["end"] = np.where(seen, v["start"] + v["service"], np.nan)
    v["discharge"] = np.where(seen, v["end"] + v["after"] + np.where(v["outcome"] == "Admitted", v["boarding"], 0),
                              v["left"])

    tests = _make_tests(cfg, v[seen], rng)

    util = []
    for d_idx, dept in enumerate(cfg.departments):
        on_duty = sum((min(e, horizon) - s) * staff for s, e, staff in duty[d_idx] if s < horizon)
        busy = v.loc[seen & (v["dept_idx"] == d_idx), "service"].sum()
        util.append({"department": dept.name, "clinician_hours": round(on_duty / 60), "utilisation": round(busy / on_duty, 2)})

    return SimResult(cfg, pop, v, tests, pd.DataFrame(util))


def _make_tests(cfg: SimConfig, seen: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Tests are ordered during the consultation; urgent ones come back faster."""
    rows = []
    for d_idx, dept in enumerate(cfg.departments):
        if not dept.test_mix or dept.test_rate <= 0:
            continue
        g = seen[seen["dept_idx"] == d_idx]
        p = np.clip(dept.test_rate * np.asarray(cfg.test_triage_factor)[g["triage"]], 0, 0.98)
        n_tests = (rng.random(len(g)) < p) * rng.choice([1, 2, 3], len(g), p=[0.65, 0.27, 0.08])
        idx = np.repeat(g.index.to_numpy(), n_tests)
        if len(idx) == 0:
            continue
        k = len(idx)
        sub = seen.loc[idx]
        types = rng.choice(list(dept.test_mix), k, p=list(dept.test_mix.values()))
        urgent = rng.random(k) < np.asarray(cfg.urgent_prob)[sub["triage"]]
        requested = sub["start"].to_numpy() + rng.uniform(0.1, 0.7, k) * sub["service"].to_numpy()
        tat_med = np.array([cfg.test_tat_median_min[t] for t in types]) * np.where(urgent, cfg.urgent_tat_factor, 1.0)
        tat = np.maximum(3.0, rng.lognormal(np.log(tat_med), 0.5))
        rows.append(pd.DataFrame({"visit_idx": idx, "test_type": types, "priority": np.where(urgent, "Urgent", "Routine"),
                                  "requested": requested, "completed": requested + tat}))
    if not rows:
        return pd.DataFrame(columns=["visit_idx", "test_type", "priority", "requested", "completed"])
    return pd.concat(rows, ignore_index=True)


assert set(TEST_TYPES) >= {t for d in SimConfig().departments for t in d.test_mix}, "unknown test type in config"