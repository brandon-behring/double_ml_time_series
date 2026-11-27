"""Predict the effect of a Pennsylvania reemployment bonus at the July 1989 cutoff."""

import argparse
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[4])]
from benchmark_io import fetch  # noqa: E402

from dml_ts.dml.fwl import fwl_estimate  # noqa: E402

ARCHIVE = "https://www.upjohn.org/sites/default/files/2019-02/PA_ReempBonus.zip"
SHA256 = "9036c9a82a5ab69b580b6646a3749019924442b7c6d10e4224ab0b910a95ef53"
CUTOFF = pd.Timestamp("1989-07-01")
BASELINE = [
    "wba",
    "bp1000",
    "shdur",
    "female",
    "agelt35",
    "agegt54",
    "black",
    "hispanic",
    "othrace",
    "recall",
]


def historical(data):
    """Table VIII.3, pooled workshop arms and the report's restricted analysis sample."""
    sample = data.loc[data.revasamp.eq(1)].dropna(subset=BASELINE)
    pooled = sample.tg.replace(6, 4).astype(int)
    arms = pd.get_dummies(pooled, prefix="arm", drop_first=True)
    office = pd.get_dummies(sample.lo, prefix="office", drop_first=True)
    cohort = pd.get_dummies(sample.selt_dt.dt.to_period("Q"), prefix="quarter").iloc[:, :-1]
    x = pd.concat([arms, office, cohort, sample[BASELINE]], axis=1).astype(float)
    fit = sm.OLS(sample.wrwage1, sm.add_constant(x)).fit()
    partial = fwl_estimate(
        sample.wrwage1.to_numpy(),
        x.arm_4.to_numpy(),
        x.drop(columns="arm_4").to_numpy(),
        method="ols",
    )
    np.testing.assert_allclose(partial.theta, fit.params.arm_4, atol=1e-7, rtol=0)
    columns = [f"arm_{i}" for i in range(1, 6)]
    np.testing.assert_array_equal(np.rint(fit.params[columns]), [12, 83, 116, 74, -62])
    np.testing.assert_array_equal(np.rint(fit.bse[columns]), [83, 69, 74, 65, 75])
    np.testing.assert_equal(len(sample), 13913)
    return {
        "n": len(sample),
        "pooled": float(fit.params.arm_4),
        "se": float(fit.bse.arm_4),
        "control_mean": float(sample.loc[sample.tg.eq(0), "wrwage1"].mean()),
        "all_five_coefficients": fit.params[columns].tolist(),
        "all_five_ses": fit.bse[columns].tolist(),
    }


def early_records(data, lag, endpoint):
    before = data[data.selt_dt.lt(CUTOFF)]
    if endpoint == "earnings":
        # First full quarter after the benefit application; observable at a reconstructed reporting date.
        maturity = (before.ab_dt.dt.to_period("Q") + 1 + lag).dt.end_time
    elif endpoint == "ui":
        if lag:
            raise ValueError("UI duration uses a nine-week availability rule")
        maturity = before.ab_dt + pd.Timedelta(weeks=9)
    else:
        raise ValueError("unknown endpoint")
    return before[maturity.lt(CUTOFF)]


def interval(point, se):
    spread = norm.ppf(0.975) * se
    return [float(point - spread), float(point + spread)]


def contrast(data, outcome):
    """Weight arm differences by the share of claimants in each office and allocation period."""
    cells = data.groupby(["lo", "allocation_period"], observed=True)
    mass = 0
    point = variance = 0.0
    batch_influence = []
    for _, subset in cells:
        treated = subset.loc[subset.tg.eq(4), outcome].to_numpy(dtype=float)
        control = subset.loc[subset.tg.eq(0), outcome].to_numpy(dtype=float)
        if min(len(treated), len(control)) < 2:
            raise ValueError("Each comparison cell requires at least two claimants per arm")
        if not np.isfinite(treated).all() or not np.isfinite(control).all():
            raise ValueError("Missing outcome within an allocation cell")
        n = len(subset)
        mass += n
        point += n * (treated.mean() - control.mean())
        variance += n**2 * (treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))
        for arm, x, sign in [(4, treated, 1), (0, control, -1)]:
            batch_influence.append(
                pd.Series(
                    sign * n * (x - x.mean()) / len(x),
                    index=subset.loc[subset.tg.eq(arm), "selt_dt"],
                )
            )
    point /= mass
    se = np.sqrt(variance) / mass
    by_batch = pd.concat(batch_influence).groupby(level=0).sum() / mass
    batch_se = np.sqrt((by_batch @ by_batch) * len(by_batch) / (len(by_batch) - 1))
    return {
        "n": mass,
        "offices": int(data.lo.nunique()),
        "batches": len(by_batch),
        "cells": cells.ngroups,
        "treated": int(data.tg.eq(4).sum()),
        "estimate": float(point),
        "se": float(se),
        "interval": interval(point, se),
        "batch_se": float(batch_se),
    }


def assess(early, late, outcome):
    pred = contrast(early, outcome)
    observed = contrast(late, outcome)
    # Eligible claimants are observed once; cohorts and their calendar batches are disjoint.
    gap = pred["estimate"] - observed["estimate"]
    se = np.hypot(pred["se"], observed["se"])
    batch_se = np.hypot(pred["batch_se"], observed["batch_se"])
    return {
        "early": pred,
        "later": observed,
        "prediction_minus_later": gap,
        "interval": interval(gap, se),
        "batch_interval": interval(gap, batch_se),
        "zero_absolute_error": abs(observed["estimate"]),
        "prediction_absolute_error": abs(gap),
    }


def run(cache_dir):
    path = fetch(ARCHIVE, cache_dir, "pa-bonus.zip", SHA256)
    with (
        ZipFile(path) as book,
        book.open("cd/data/Upjohn Institute Created files/Recsfile.dta") as f,
    ):
        data = pd.read_stata(f, convert_categoricals=False)
    original = historical(data)
    panel = data[data.tg.isin([0, 4])].copy()
    if panel.inuidur1.isna().any():
        raise ValueError("Missing initial spell length in archive")
    panel["past_six"] = panel.inuidur1.gt(6).astype(float)
    # October 1988 expanded intake beyond Pittston. July and October 1989 changed allocation.
    panel["allocation_period"] = np.searchsorted(
        np.array(["1988-10-26", "1989-07-01", "1989-10-01"], dtype="datetime64[D]"),
        panel.selt_dt.to_numpy(dtype="datetime64[D]"),
        side="right",
    )
    later = panel[panel.selt_dt.ge(CUTOFF)]
    estimates = {}
    for lag in (0, 1, 2):
        early = early_records(panel, lag, "earnings")
        estimates[str(lag)] = assess(early, later, "wrwage1")
        if lag == 2:
            common = later[later.lo.isin(early.lo.unique())]
            pittston = assess(early, common, "wrwage1")
    duration = assess(early_records(panel, 0, "ui"), later, "past_six")
    counts = [estimates[str(lag)]["early"]["n"] for lag in (0, 1, 2)]
    np.testing.assert_array_equal(counts, [2492, 1227, 71])
    np.testing.assert_equal(len(later), 1775)
    return {
        "historical_table_VIII_3": original,
        "cutoff": "1989-07-01",
        "earnings": estimates,
        "lag_two_common_office": pittston,
        "ui_past_week_six": duration,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().cache_dir), indent=2, allow_nan=False))
