"""Independently reconstruct published Table I, panel A, column 1 from the public GPS release."""

import argparse
import hashlib
import json
import tempfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ARCHIVE = "https://zenodo.org/records/10637043/files/replication_package_v7.zip"
ARCHIVE_BYTES = 9_361_827_603
ARMS = ["high", "low", "info", "ctrl"]
KEY = ["uidp", "date"]


def fetch_inputs(cache: Path) -> None:
    """Fetch the six deflated members from the versioned ZIP and verify their uncompressed bytes."""
    members = json.loads((HERE / "source-members.json").read_text())

    def fetch_one(item):
        name, metadata = item
        path = cache / name
        if path.exists():
            with path.open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() == metadata["sha256"]:
                    return
            raise ValueError(f"Cached file failed SHA-256 verification: {name}")
        start = metadata["start"]
        end = start + metadata["length"] - 1
        request = Request(f"{ARCHIVE}?start={start}", headers={"Range": f"bytes={start}-{end}"})
        with urlopen(request, timeout=120) as response:
            if response.headers.get("Content-Range") != f"bytes {start}-{end}/{ARCHIVE_BYTES}":
                raise ValueError(f"Unexpected byte range for {name}")
            payload = zlib.decompress(response.read(), -15)
        if hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
            raise ValueError(f"Download failed SHA-256 verification: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
            output.write(payload)
        Path(output.name).replace(path)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch_one, members.items()))


def study_dates(cache):
    logs = pd.read_csv(cache / "treatment/full_daily_logs.csv")
    logs = logs.loc[logs.type.isin(["report", "special"])].copy()
    logs["date"] = pd.to_datetime("2017-" + logs.reference_date.str[1:])
    special = logs.date.isin(
        pd.to_datetime(["2017-03-29", "2017-06-26", "2017-08-15", "2017-08-25"])
    )
    away = (
        logs.outstation.astype(str)
        .str.lower()
        .map({"true": True, "false": False, "0": False, "1": True})
    )
    if away.isna().any():
        raise ValueError("Unknown travel status in the daily reports")
    experiment = (
        ~special & ~away & logs.cycle_type.isin(["control_info", "control_qual", "dt", "area"])
    )
    logs["experiment"] = experiment
    starts = logs.loc[experiment].groupby("uidp").date.min().rename("first_experiment")
    return logs[KEY + ["study_cycle", "experiment"]], starts


def trip_outcomes(cache):
    cols = [
        "inb_frac",
        "t_start_bg_",
        "t_start_prec_max",
        "t_stop_prec_max",
        "dur_trips",
        "plb",
        "trips_dur_jump",
        "trips_plb_jump",
        "chain_diameter",
        "center_line_dist",
    ]
    trip_key = KEY + ["chain"]
    trips = pd.read_csv(
        cache / "coded_gps/sample_exp_chains_15 no PII.csv", usecols=trip_key + cols
    )
    # Match the precision of Stata's import / generated ratio variables at cut points.
    t = trips[cols].astype(np.float32)
    dur_jump = (t.trips_dur_jump / t.dur_trips).between(0.3, 5, inclusive="left")
    dist_jump = (t.trips_plb_jump / t.plb).between(0.3, 5, inclusive="left")
    keep = (
        (t.inb_frac.astype(float) >= 0.7)
        & (t.t_start_bg_ / 3600).between(7, 22)
        & (t.t_start_prec_max <= 15)
        & (t.t_stop_prec_max <= 15)
        & t.dur_trips.between(5, 180)
        & t.plb.between(500, 35000)
        & (t.trips_dur_jump < 30)
        & (t.trips_plb_jump < 2000)
        & ~dur_jump
        & ~dist_jump
        & ~((t.chain_diameter / t.plb).astype(float) < 0.3)
        & ~(((t.center_line_dist / t.plb).astype(float) < 0.3) & (t.plb < 2000))
    )
    rates = pd.read_csv(
        cache / "coded_gps/charges_trips_dt_15.csv", usecols=KEY + ["id_new", "rate"]
    )
    keep_trips = trips.loc[keep, trip_key].merge(
        rates.rename(columns={"id_new": "chain"}), on=trip_key, how="left", validate="one_to_one"
    )
    if keep_trips.rate.isna().any():
        raise ValueError("Missing hypothetical charge for a retained trip")
    keep_trips["y"] = (keep_trips.rate.astype(np.float32) / 24 * 100).astype(float)
    keep_trips["date"] = pd.to_datetime(keep_trips.date)
    return keep_trips.groupby(KEY).agg(y=("y", "sum"), trips=("y", "size")).reset_index()


def analysis_calendar(cache):
    roster = pd.read_csv(cache / "treatment/treatment roster noPII.csv")
    roster = roster.loc[roster.meeting.eq("done")].copy()
    assignment = pd.read_csv(cache / "treatment/prerandomized_assignment.csv")
    roster = roster.merge(
        assignment[["idx", "strat_cell", "dt_late", "set"] + ["dt_" + a for a in ARMS]],
        on=["idx", "strat_cell"],
        validate="many_to_one",
    )
    if not roster[["dt_" + a for a in ARMS]].sum(axis=1).eq(1).all():
        raise ValueError("Assignment is not one arm per commuter")
    roster["arm"] = roster[["dt_" + a for a in ARMS]].idxmax(axis=1).str[3:]
    roster["dt_late"] = roster.dt_late.str[0].astype(int)
    roster["assignment_date"] = pd.to_datetime("2017-" + roster.dateadded.str[1:])
    roster["stratum"] = roster.strat_cell + 10 * (roster["set"] == 1)
    design = assignment.groupby("strat_cell")[["dt_high", "dt_ctrl"]].sum()
    probability = design.dt_high / (design.dt_high + design.dt_ctrl)
    roster["p_high"] = roster.strat_cell.map(probability)
    logs, starts = study_dates(cache)
    if set(starts.index) != set(roster.uidp):
        raise ValueError("Completed roster and daily-report participants differ")

    q = pd.read_csv(
        cache / "coded_gps/quality_all.csv", usecols=KEY + ["data_quality"], parse_dates=["date"]
    )
    q = q.merge(
        roster[["uidp", "arm", "dt_late", "assignment_date", "stratum", "p_high"]],
        on="uidp",
        validate="many_to_one",
    )
    q = q.loc[
        (q.date.dt.dayofweek < 5)
        & (q.date != "2017-04-26")
        & ~((q.date <= "2017-04-03") & q.uidp.isin(["L0314124709", "L0221092234"]))
    ].copy()
    # Some late experiment reports follow the end of the GPS release; retain them for the loss denominator.
    q = q.merge(logs, on=KEY, how="outer", validate="one_to_one")
    q = q.loc[(q.date != "2017-04-26") & (q.date.dt.dayofweek < 5)]
    missing_roster = q.arm.isna()
    roster_index = roster.set_index("uidp")
    for col in ["arm", "dt_late", "assignment_date", "stratum", "p_high"]:
        q.loc[missing_roster, col] = q.loc[missing_roster, "uidp"].map(roster_index[col])
    q["post"] = q.experiment.eq(True).astype(int)
    # Special, trial and outstation reports are excluded, including those before the first experimental day.
    base = (q.date < q.uidp.map(starts)) & q.study_cycle.isna()
    q.loc[base, "study_cycle"] = 0
    active = q.post.eq(1) & (q.study_cycle - q.dt_late).between(1, 3)
    q = q.loc[base | active].copy()
    q["observed"] = q.data_quality.isin(["good", "bad"]) & (q.date < "2017-09-11")
    q = q.merge(trip_outcomes(cache), on=KEY, how="left", validate="one_to_one")
    q.loc[q.observed, ["y", "trips"]] = q.loc[q.observed, ["y", "trips"]].fillna(0)
    q.loc[~q.observed, ["y", "trips"]] = np.nan
    return q.sort_values(KEY).reset_index(drop=True)


def fixed_effects(panel, adjust=False):
    """Within-commuter regression with full-rank cycle dummies and Stata areg CR1 degrees of freedom."""
    p = panel.loc[panel.observed].copy()
    X = pd.DataFrame({a: (p.arm.eq(a) * p.post).astype(float) for a in ARMS[:-1]}, index=p.index)
    X = X.join(
        pd.get_dummies(p.study_cycle.astype(int), prefix="cycle", drop_first=True, dtype=float)
    )
    if adjust:
        # Fixed pretreatment characteristics are absorbed by commuter indicators.
        # Their post interactions permit differing baseline-to-post changes.
        for col in FEATURES:
            X["post_" + col] = p.post * p[col]
    x = (X - X.groupby(p.uidp).transform("mean")).to_numpy()
    y = (p.y - p.groupby("uidp").y.transform("mean")).to_numpy()
    b, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != x.shape[1]:
        raise ValueError("Within-commuter regressors are collinear")
    groups = p.uidp.nunique()
    n = len(y)
    scores = pd.DataFrame(x * (y - x @ b)[:, None]).groupby(p.uidp.to_numpy()).sum().to_numpy()
    bread = np.linalg.inv(x.T @ x)
    # areg includes absorbed commuter intercepts in residual degrees of freedom.
    variance = (
        (bread @ (scores.T @ scores) @ bread)
        * groups
        / (groups - 1)
        * (n - 1)
        / (n - groups - rank)
    )
    return {
        "n": n,
        "commuters": groups,
        "arm_coefficients": dict(zip(ARMS[:-1], b[:3], strict=True)),
        "cluster_se": dict(zip(ARMS[:-1], np.sqrt(np.diag(variance))[:3], strict=True)),
        "control_mean": p.loc[p.arm.eq("ctrl") & p.post.eq(1), "y"].mean(),
    }


FEATURES = ["mean_rate", "sd_rate", "mean_trips", "pre_days"]


def baseline_histories(panel):
    pre = panel.loc[panel.observed & panel.post.eq(0) & (panel.date < panel.assignment_date)]
    features = pre.groupby("uidp").agg(
        mean_rate=("y", "mean"),
        sd_rate=("y", "std"),
        mean_trips=("trips", "mean"),
        pre_days=("y", "size"),
    )
    if features.isna().any().any() or set(features.index) != set(panel.uidp):
        raise ValueError("Incomplete pre-assignment histories")
    return panel.join(features, on="uidp", validate="many_to_one")


def commuter_period(days):
    """Each commuter contributes one observed average within the specified dates."""
    p = days.groupby("uidp").agg(
        arm=("arm", "first"),
        p_high=("p_high", "first"),
        rate=("y", "mean"),
        **{col: (col, "first") for col in FEATURES},
    )
    p["change"] = p.rate - p.mean_rate
    return p


def weighted_contrast(persons, outcome):
    """Hájek high minus control contrast; one cluster per commuter within this window."""
    means = {}
    variance = 0.0
    for arm in ["high", "ctrl"]:
        p = persons.loc[persons.arm.eq(arm)]
        if len(p) < 2:
            return {"effect": None, "se": None}
        weights = 1 / (p.p_high if arm == "high" else 1 - p.p_high)
        normalized = weights / weights.sum()
        means[arm] = float((normalized * p[outcome]).sum())
        variance += len(p) / (len(p) - 1) * ((p[outcome] - means[arm]) * normalized).pow(2).sum()
    return {"effect": means["high"] - means["ctrl"], "se": float(np.sqrt(variance))}


def fold_for_person(uid):
    return int(hashlib.sha256(str(uid).encode()).hexdigest()[:8], 16) % 5


def rolling(panel):
    post = panel.loc[panel.post.eq(1) & panel.arm.isin(["high", "ctrl"]) & panel.observed]
    rows = []
    for month in pd.period_range("2017-03", "2017-09", freq="M"):
        start = month.start_time
        test = commuter_period(
            post.loc[
                (post.date >= start)
                & (post.date < (month + 1).start_time)
                & (post.assignment_date < start)
            ]
        )
        earlier = post.loc[post.date < start]
        train = commuter_period(earlier)
        row = {
            "month": str(month),
            "train_days": len(earlier),
            "train_people": len(train),
            "last_training_date": None if earlier.empty else earlier.date.max().date().isoformat(),
            "test_high": int(test.arm.eq("high").sum()),
            "test_ctrl": int(test.arm.eq("ctrl").sum()),
            "prior": weighted_contrast(train, "change"),
            "unadjusted": weighted_contrast(test, "change"),
        }
        if min(row["test_high"], row["test_ctrl"]) < 2:
            row["status"] = "insufficient scored arms"
            rows.append(row)
            continue
        # Known randomization probabilities and a single common prognostic model per fold.
        # Entire commuters are held out even when they were also active in the previous month.
        train = train.loc[train.arm.eq("ctrl")]
        heldout = test.index.map(fold_for_person)
        folds = train.index.map(fold_for_person)
        test["prediction"] = np.nan
        for fold in np.unique(heldout):
            history = train.loc[folds != fold]
            if len(history) < 2:
                continue
            model = make_pipeline(StandardScaler(), Ridge(alpha=10))
            model.fit(history[FEATURES], history.change)
            use = heldout == fold
            test.loc[use, "prediction"] = model.predict(test.loc[use, FEATURES])
        if test.prediction.isna().any():
            row["status"] = "insufficient earlier control histories"
        else:
            row["status"] = "scored"
            test["adjusted"] = test.change - test.prediction
            row["adjusted"] = weighted_contrast(test, "adjusted")
            control = test.loc[test.arm.eq("ctrl")]
            row["heldout_control_rmse"] = float(np.sqrt(control.adjusted.pow(2).mean()))
            row["zero_change_control_rmse"] = float(np.sqrt(control.change.pow(2).mean()))
        rows.append(row)
    return rows


def run(cache):
    fetch_inputs(cache)
    panel = baseline_histories(analysis_calendar(cache))
    published = fixed_effects(panel)
    printed = (
        round(published["arm_coefficients"]["high"], 2),
        round(published["cluster_se"]["high"], 2),
        published["n"],
        round(published["control_mean"], 2),
    )
    if printed != (-12.94, 6.04, 15_574, 96.88):
        raise ValueError(f"Published Table I reconstruction failed: {printed}")
    schedule = panel.loc[panel.post.eq(1)].copy()
    schedule["month"] = schedule.date.dt.to_period("M").astype(str)
    support = schedule.groupby(["month", "arm"]).agg(
        assigned_people=("uidp", "nunique"),
        scheduled_days=("uidp", "size"),
        observed_days=("observed", "sum"),
    )
    observed = schedule.loc[schedule.observed].groupby(["month", "arm"]).uidp.nunique()
    support["observed_people"] = observed.reindex(support.index, fill_value=0)
    support["lost_days"] = support.scheduled_days - support.observed_days
    return {
        "published_specification": published,
        "history_adjusted_specification": fixed_effects(panel, True),
        "baseline_days_range": [int(panel.pre_days.min()), int(panel.pre_days.max())],
        "monthly_support": support.reset_index().to_dict("records"),
        "rolling": rolling(panel),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/dml-ts/external")
    p.add_argument("--output", type=Path, help="Write measured aggregate JSON to this path.")
    args = p.parse_args()
    results = run(args.cache_dir / "bangalore-v1")
    data = json.dumps(results, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(data)
    print(data, end="")


if __name__ == "__main__":
    main()
