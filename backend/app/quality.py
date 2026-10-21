"""Data Quality Center: every problem we find in the raw data, how big it is, and how we handled it."""

from app.data import LBC, days_covered


def _issue(id_, table, severity, title, count, total, handling):
    return {
        "id": id_, "table": table, "severity": severity, "title": title,
        "count": int(count), "total": int(total),
        "pct": round(100 * count / total, 2) if total else 0.0, "handling": handling,
    }


def run_checks(raw, clean):
    enc, pat, dep, tst = raw["encounters"], raw["patients"], raw["departments"], raw["tests"]
    n = len(enc)
    out = []

    def add(*a):
        if a[4] > 0:  # only report problems that actually exist in this data
            out.append(_issue(*a))

    add("dup_ids", "encounters", "high", "Duplicate encounter IDs",
        enc["encounter_id"].duplicated().sum(), n, "Kept the first occurrence")
    add("bad_arrival", "encounters", "high", "Missing or unreadable arrival time",
        enc["arrival_time"].isna().sum(), n, "Dropped - a visit cannot be placed in time")
    add("orphan_patient", "encounters", "high", "Visit refers to unknown patient",
        (~enc["patient_id"].isin(pat["patient_id"])).sum(), n, "Dropped from joined views")
    add("start_before_arrival", "encounters", "high", "Service starts before arrival",
        (enc["service_start_time"] < enc["arrival_time"]).sum(), n, "Excluded from wait-time training")
    wait_calc = (enc["service_start_time"] - enc["arrival_time"]).dt.total_seconds() / 60
    add("wait_mismatch", "encounters", "medium", "wait_minutes disagrees with timestamps (>1 min)",
        ((wait_calc - enc["wait_minutes"]).abs() > 1).sum(), n, "Timestamps treated as the source of truth")
    add("end_before_start", "encounters", "high", "Service ends before it starts",
        (enc["service_end_time"] < enc["service_start_time"]).sum(), n, "Service duration ignored for these rows")
    add("discharge_before_end", "encounters", "medium", "Discharge before service ends",
        (enc["discharge_time"] < enc["service_end_time"]).sum(), n, "Flagged, not used for length-of-stay")
    lbc_served = enc["outcome"].eq(LBC) & enc["service_start_time"].notna()
    add("lbc_with_service", "encounters", "medium",
        "'Left before care' visits still have service start/end times",
        lbc_served.sum(), int(enc["outcome"].eq(LBC).sum()) or 1,
        "Outcome label trusted; service times for these visits treated as unreliable")
    adm = enc["outcome"].isin(["Admitted", "Transferred"]) & enc["discharge_time"].notna()
    add("admit_with_discharge", "encounters", "low",
        "Admitted/Transferred visits have a 'discharge_time'",
        adm.sum(), int(enc["outcome"].isin(["Admitted", "Transferred"]).sum()) or 1,
        "Interpreted as 'left this department', not hospital discharge")

    # Patients
    p = len(pat)
    add("sex_unknown", "patients", "low", "Sex recorded as Unknown/Other",
        pat["sex"].isin(["Unknown", "Other"]).sum(), p, "Kept as its own category")
    add("insurance_unknown", "patients", "low", "Insurance type Unknown",
        pat["insurance_type"].eq("Unknown").sum(), p, "Kept as its own category")
    add("none_parsing", "patients", "medium",
        "chronic_condition value 'None' is read as missing by default CSV readers",
        pat["chronic_condition"].eq("None").sum(), p, "Loaded with NA-parsing disabled so 'None' stays a real value")

    # Tests
    t = tst.merge(enc[["encounter_id", "arrival_time", "discharge_time"]], on="encounter_id", how="left")
    k = len(t)
    add("test_orphan", "tests", "high", "Test refers to unknown encounter", t["arrival_time"].isna().sum(), k,
        "Dropped")
    add("test_negative_tat", "tests", "high", "Test completed before it was requested",
        (t["completion_time"] < t["requested_time"]).sum(), k, "Dropped from turnaround analysis")
    add("test_before_arrival", "tests", "medium", "Test requested before the patient arrived",
        (t["requested_time"] < t["arrival_time"]).sum(), k, "Flagged; synthetic timestamps not perfectly linked")
    add("test_after_discharge", "tests", "medium", "Test requested after the visit ended",
        (t["requested_time"] > t["discharge_time"]).sum(), k, "Flagged; synthetic timestamps not perfectly linked")

    # Departments: capacity vs real volume
    days = days_covered(clean["encounters"])
    vol = clean["encounters"].groupby("department_id").size() / days
    d = dep.set_index("department_id").join(vol.rename("visits_per_day"))
    over = d["daily_capacity"] > 10 * d["visits_per_day"].fillna(0)
    add("capacity_vs_volume", "departments", "medium",
        "daily_capacity is far above actual visits/day (>10x)", over.sum(), len(d),
        "Capacity not used literally; we model relative load from observed arrivals")
    beds_odd = d["service_type"].isin(["Diagnostics", "Support", "Admin"]) & (d["bed_count"] > 0)
    add("beds_non_clinical", "departments", "low", "Non-clinical departments list beds (e.g. Pharmacy)",
        beds_odd.sum(), len(d), "bed_count not used as a model feature")

    sev_rank = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda i: (sev_rank[i["severity"]], -i["count"]))
    summary = {
        "raw_rows": {k_: len(v) for k_, v in raw.items()},
        "clean_rows": {k_: len(v) for k_, v in clean.items()},
        "issues_found": len(out),
        "days_covered": days,
        "date_range": [str(clean["encounters"]["arrival_time"].min()), str(clean["encounters"]["arrival_time"].max())],
        "visits_per_day": round(len(clean["encounters"]) / days, 1),
    }
    return {"summary": summary, "issues": out}