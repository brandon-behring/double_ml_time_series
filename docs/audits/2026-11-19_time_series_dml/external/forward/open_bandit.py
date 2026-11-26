"""Freeze an early retail policy effect and validate later off-policy estimates."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from temporalcv import TimeSeriesCrossValidator, newey_west_se

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_io import fetch  # noqa: E402

REVISION = "90a1283aecd19026caf15a0840af1176a41f6a76"
SOURCE = f"https://huggingface.co/datasets/zozonext/open-bandit/resolve/{REVISION}"
DIGESTS = {
    "random": "c4b6f65e62bf2c683914703ab6b875cc3e1b4ef0403a5779f548f5578cc34d6d",
    "bts": "f116bb11bd0b18f02ca69d313c790c63c006728b3fae28a562d7309e20a0e4b5",
}
ITEMS = 34
HOURS = pd.date_range("2019-11-24", periods=168, freq="h", tz="UTC", name="hour")
CUTOFF = pd.Timestamp("2019-11-28", tz="UTC")
KEYS = ["hour", "position", "item_id"]
SUMS = [
    "n",
    "clicks",
    "weight",
    "weight_sq",
    "weight_click",
    "weight_above_10",
    "rows_above_1000",
    "weight_above_1000",
]


def summarize_rows(rows, arm):
    """Sufficient totals for direct, inverse-probability and augmented scores."""
    p = rows.propensity_score
    if not p.between(0, 1, inclusive="right").all():
        raise ValueError("Expected positive logged item probabilities no larger than one")
    if arm == "random" and not np.allclose(p, 1 / ITEMS, rtol=0, atol=1e-12):
        raise ValueError("The random arm must choose uniformly from 34 items per position")
    if not rows.item_id.isin(range(ITEMS)).all() or not rows.position.isin([1, 2, 3]).all():
        raise ValueError("Item or position is outside the campaign design")
    if not rows.click.isin([0, 1]).all():
        raise ValueError("Invalid click indicator")
    weight = (1 / ITEMS) / p
    hour = pd.to_datetime(rows.timestamp, format="ISO8601", utc=True).dt.floor("h")
    if not hour.isin(HOURS).all():
        raise ValueError("Unexpected or missing campaign hour")
    cells = rows.assign(
        hour=hour,
        n=1,
        clicks=rows.click,
        weight=weight,
        weight_sq=weight**2,
        weight_click=weight * rows.click,
        weight_above_10=weight.where(weight > 10, 0),
        rows_above_1000=weight.gt(1000),
        weight_above_1000=weight.where(weight > 1000, 0),
        max_weight=weight,
    )
    return cells.groupby(KEYS).agg({**dict.fromkeys(SUMS, "sum"), "max_weight": "max"})


def load(cache, arm):
    path = fetch(
        f"{SOURCE}/{arm}/men/men.csv?download=true",
        cache,
        f"openbandit-mens-{arm}.csv",
        DIGESTS[arm],
    )
    cols = ["timestamp", "position", "item_id", "propensity_score", "click"]
    parts = [
        summarize_rows(rows, arm) for rows in pd.read_csv(path, usecols=cols, chunksize=150_000)
    ]
    data = (
        pd.concat(parts)
        .groupby(level=KEYS)
        .agg({**dict.fromkeys(SUMS, "sum"), "max_weight": "max"})
    )
    if not data.index.unique("hour").sort_values().equals(HOURS):
        raise ValueError(f"Incomplete calendar in {arm} arm")
    return data.reset_index()


def periods(lag):
    n_late = int((HOURS >= CUTOFF).sum())
    splitter = TimeSeriesCrossValidator(
        n_splits=1, gap=lag, test_size=n_late, min_train_size=len(HOURS) - n_late - lag
    )
    early, late = next(splitter.split(HOURS))
    return HOURS[early], HOURS[late]


def hourly_mean(hourly, column):
    """Ratio and hourly contributions to its linearization, in percentage points."""
    n = hourly.n.sum()
    if n == 0:
        raise ValueError("Missing observations for mean")
    rate = hourly[column].sum() / n
    return 100 * rate, 100 * (hourly[column] - rate * hourly.n) / n


def interval(point, contributions):
    # Restore zero contributions inside an embargo without compressing calendar time.
    index = pd.date_range(contributions.index.min(), contributions.index.max(), freq="h")
    score = contributions.reindex(index, fill_value=0).to_numpy() * len(index)
    out = {}
    for span in (1, 6, 12, 24):
        se = newey_west_se(score, bandwidth=span - 1).se
        out[str(span)] = [float(point - 1.96 * se), float(point + 1.96 * se)]
    return {"pp": float(point), "ci95_pp_by_span_hours": out}


def direct_effect(by_arm, hours):
    values = [hourly_mean(by_arm[arm].loc[hours], "clicks") for arm in ("bts", "random")]
    return values[0][0] - values[1][0], values[0][1] - values[1][1]


def fit_click_model(early):
    """Partially pool early Thompson item-position CTRs toward each position's rate."""
    counts = early.groupby(["position", "item_id"])[["n", "clicks"]].sum()
    totals = counts.groupby("position").sum().reindex([1, 2, 3])
    if totals.n.isna().any() or (totals.n <= 0).any():
        raise ValueError("Earlier Thompson data must cover each position")
    cells = pd.MultiIndex.from_product([[1, 2, 3], range(ITEMS)], names=["position", "item_id"])
    counts = counts.reindex(cells, fill_value=0)
    prior = counts.index.get_level_values("position").map(totals.clicks / totals.n)
    return ((counts.clicks + 100 * prior) / (counts.n + 100)).rename("fitted")


def off_policy(bts, by_arm, early, late):
    model = fit_click_model(bts[bts.hour.isin(early)])
    data = bts[bts.hour.isin(late)].join(model, on=["position", "item_id"]).copy()
    # Integrate over ALL target items at that slot, including unobserved actions.
    target_mean = data.position.map(model.groupby("position").mean())
    data["target_fitted"] = data.n * target_mean
    data["weighted_fitted"] = data.weight * data.fitted
    data["dr"] = data.target_fitted + data.weight_click - data.weighted_fitted
    hourly = data.groupby("hour")[["n", "weight_click", "dr", "weight"]].sum()
    benchmark, bench_score = hourly_mean(by_arm["random"].loc[late], "clicks")
    results = {"uniform_arm_ctr_pct": float(benchmark)}
    for method, column in (("ipw", "weight_click"), ("doubly_robust", "dr")):
        value, score = hourly_mean(hourly, column)
        results[method] = {
            "ctr_pct": float(value),
            "minus_uniform": interval(value - benchmark, score - bench_score),
        }
    n = int(data.n.sum())
    w = data.weight.sum()
    results["augmentation_terms_pct"] = {
        "uniform_model_mean": float(100 * data.target_fitted.sum() / n),
        "weighted_model_mean": float(100 * data.weighted_fitted.sum() / n),
    }
    results["weights"] = {
        "rows": n,
        "mean": float(w / n),
        "max": float(data.max_weight.max()),
        "effective_rows": float(w**2 / data.weight_sq.sum()),
        "weight_mass_above_10_pct": float(100 * data.weight_above_10.sum() / w),
        "rows_above_1000": int(data.rows_above_1000.sum()),
        "weight_mass_above_1000_pct": float(100 * data.weight_above_1000.sum() / w),
        "observed_items_per_position": {
            str(pos): int(count)
            for pos, count in data.groupby("position").item_id.nunique().items()
        },
    }
    peak = hourly.weight.idxmax()
    same_hours = late[late != peak]
    results["largest_weight_block"] = {
        "hour_utc": peak.isoformat(),
        "weight_mass_pct": float(100 * hourly.loc[peak, "weight"] / w),
        "dr_excluding_hour_pct": hourly_mean(hourly.loc[same_hours], "dr")[0],
        "uniform_excluding_hour_pct": hourly_mean(by_arm["random"].loc[same_hours], "clicks")[0],
    }
    return results


def compare(cells):
    by_arm = {key: val.groupby("hour")[["n", "clicks"]].sum() for key, val in cells.items()}
    table = {}
    for arm, counts in by_arm.items():
        n, clicks = counts[["n", "clicks"]].sum()
        table[arm] = {"rows": int(n), "clicks": int(clicks), "ctr_pct": float(100 * clicks / n)}
    sensitivity = {}
    for lag in (0, 24, 48):
        early, late = periods(lag)
        frozen, early_score = direct_effect(by_arm, early)
        measured, late_score = direct_effect(by_arm, late)
        error = frozen - measured
        sensitivity[str(lag)] = {
            "last_train_hour_utc": early[-1].isoformat(),
            "early_pp": float(frozen),
            "later_pp": float(measured),
            "signed_error": interval(error, early_score.subtract(late_score, fill_value=0)),
            "absolute_error_pp": float(abs(error)),
            "zero_absolute_error_pp": float(abs(measured)),
        }
    early, late = periods(24)
    days = {
        str(day.date()): float(direct_effect(by_arm, late[late.normalize() == day])[0])
        for day in late.normalize().unique()
    }
    return {
        "full_sample": table,
        "cutoff_utc": CUTOFF.isoformat(),
        "reporting_lags_hours": sensitivity,
        "later_by_day_pp": days,
        "later_off_policy": off_policy(cells["bts"], by_arm, early, late),
    }


def run(cache):
    result = compare({arm: load(cache, arm) for arm in DIGESTS})
    # NeurIPS Table 1 gives row counts and CTRs rounded to two decimals.
    for arm, n, pct in [("random", 452_949, 0.51), ("bts", 4_077_727, 0.67)]:
        record = result["full_sample"][arm]
        if record["rows"] != n or round(record["ctr_pct"], 2) != pct:
            raise ValueError(f"Published Table 1 mismatch for {arm}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().cache_dir), indent=2, allow_nan=False))
