"""Within-hour ordering checks at the policy and item levels of the public retail experiment."""

import pandas as pd
from benchmark_io import fetch

from . import open_bandit as bandit


def repetition(frame, label):
    """Same-label probability for neighboring records and its within-hour permutation expectation."""
    ordered = frame.sort_values("timestamp", kind="stable")
    hours = ordered.timestamp.dt.floor("h")
    same_hour = hours.eq(hours.shift())
    within = int(same_hour.sum())
    if not within:
        raise ValueError("No adjacent records share an hour")
    counts = ordered.groupby([hours, label]).size()
    hour_n = counts.groupby(level=0).sum()
    expected_pairs = ((counts * (counts - 1)).groupby(level=0).sum() / hour_n).sum()
    adjacent = ordered[label].eq(ordered[label].shift())
    ties = same_hour & ordered.timestamp.eq(ordered.timestamp.shift())
    distinct = same_hour & ~ties
    return {
        "pairs": within,
        "repeats": int((adjacent & same_hour).sum()),
        "observed": float(adjacent[same_hour].mean()),
        "permuted_expectation": float(expected_pairs / within),
        "tied_timestamp_pairs": int(ties.sum()),
        "observed_excluding_ties": float(adjacent[distinct].mean()) if distinct.any() else None,
    }


def retail_replay(cache):
    frames = []
    cells = {}
    extreme_rows = []
    for arm, digest in bandit.DIGESTS.items():
        path = fetch(
            f"{bandit.SOURCE}/{arm}/men/men.csv?download=true",
            cache,
            f"openbandit-mens-{arm}.csv",
            digest,
        )
        columns = ["position", "item_id", "timestamp", "propensity_score", "click"]
        chunks = []
        for rows in pd.read_csv(
            path,
            usecols=columns,
            chunksize=150_000,
            dtype={"position": "int8", "item_id": "int8", "click": "int8"},
        ):
            rows["timestamp"] = pd.to_datetime(rows.timestamp, utc=True, format="ISO8601")
            chunks.append(bandit.summarize_rows(rows, arm))
            frames.append(
                rows[["position", "item_id", "timestamp"]].assign(uniform=(arm == "random"))
            )
            if arm == "bts":
                late_extreme = (rows.timestamp >= bandit.CUTOFF) & (
                    (1 / bandit.ITEMS) / rows.propensity_score > 1000
                )
                if late_extreme.any():
                    extreme_rows.append(rows.loc[late_extreme])
        cells[arm] = (
            pd.concat(chunks)
            .groupby(level=bandit.KEYS)
            .agg({**dict.fromkeys(bandit.SUMS, "sum"), "max_weight": "max"})
            .reset_index()
        )
    logs = pd.concat(frames, ignore_index=True)
    checks = {}
    for position, rows in logs.groupby("position", sort=True):
        checks[str(position)] = {
            "policy_arm": repetition(rows, "uniform"),
            "uniform_item": repetition(rows[rows.uniform], "item_id"),
            "thompson_item": repetition(rows[~rows.uniform], "item_id"),
        }
    rare = []
    if extreme_rows:
        extremes = pd.concat(extreme_rows)
        for (hour, pos, item), rows in extremes.groupby(
            [extremes.timestamp.dt.floor("h"), "position", "item_id"]
        ):
            rare.append(
                {
                    "hour_utc": hour.isoformat(),
                    "position": int(pos),
                    "item": int(item),
                    "rows": len(rows),
                    "seconds": (rows.timestamp.max() - rows.timestamp.min()).total_seconds(),
                    "propensity_range": [
                        float(rows.propensity_score.min()),
                        float(rows.propensity_score.max()),
                    ],
                    "clicks": int(rows.click.sum()),
                }
            )
    # Use the same hour definitions and estimator functions as the original retail benchmark.
    old = bandit.compare(cells)
    summary = old["later_off_policy"]
    weights = summary["weights"]
    if sum(cell["rows"] for cell in rare) != weights["rows_above_1000"]:
        raise ValueError("Raw extreme-event count disagrees with aggregate weights")
    return {
        "repetition": checks,
        "late_weight_over_1000_cells": rare,
        "policy_effect": old["reporting_lags_hours"]["24"],
        "uniform_ctr_pct": summary["uniform_arm_ctr_pct"],
        "dr_ctr_pct": summary["doubly_robust"]["ctr_pct"],
        "weights": weights,
        "largest_weight_block": summary["largest_weight_block"],
    }
