"""Did a current-week orange-juice price cut precede a subsequent decline in purchases?"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from temporalcv import newey_west_se

GROUP = ["store", "brand"]
SERIES = ["own", "rivals", "category"]
HORIZONS = ["prior_change", "current", "next", "second", "three_week_total"]


def calendar_shift(frame, column, offset):
    """Read the exact calendar week for the same store and brand."""
    indexed = frame.set_index([*GROUP, "week"])[column]
    keys = pd.MultiIndex.from_arrays(
        [frame.store, frame.brand, frame.week + offset], names=indexed.index.names
    )
    return indexed.reindex(keys).to_numpy()


def prepare(raw):
    frame = raw.sort_values(["week", *GROUP]).reset_index(drop=True).copy()
    groups = frame.groupby(["store", "week"])
    if frame.duplicated([*GROUP, "week"]).any() or not groups.brand.nunique().eq(3).all():
        raise ValueError("Expected three unique brands per store-week")
    frame["own"] = np.exp(frame.logmove)
    frame["category"] = groups.own.transform("sum")
    frame["rivals"] = frame.category - frame.own
    prior_price = calendar_shift(frame, "price", -1)
    # Work in displayed-price units and include the exact 10% boundary.
    frame["sale"] = np.where(
        np.isfinite(prior_price), frame.price <= 0.9 * prior_price + 1e-12, np.nan
    )
    frame["other_cuts"] = groups.sale.transform("sum") - frame.sale
    frame["log_prior_price"] = np.log(prior_price)
    frame["next_price"] = calendar_shift(frame, "price", 1)
    frame["prior_feature"] = calendar_shift(frame, "feat", -1)
    frame["early_log_sales"] = calendar_shift(frame, "logmove", -3)
    frame["season_sin"] = np.sin(2 * np.pi * frame.week / 52)
    frame["season_cos"] = np.cos(2 * np.pi * frame.week / 52)
    for series in SERIES:
        before = calendar_shift(frame, series, -1)
        frame[f"{series}_prior_change"] = before - calendar_shift(frame, series, -2)
        frame[f"{series}_current"] = frame[series] - before
        frame[f"{series}_next"] = calendar_shift(frame, series, 1) - before
        frame[f"{series}_second"] = calendar_shift(frame, series, 2) - before
        frame[f"{series}_three_week_total"] = frame[
            [f"{series}_current", f"{series}_next", f"{series}_second"]
        ].sum(axis=1, min_count=3)
    return frame


def forward_splits(weeks, dates=None):
    if dates is None:
        dates = np.unique(weeks)
    for train, test in TimeSeriesSplit(n_splits=5, gap=2).split(dates):
        train_idx = np.flatnonzero(np.isin(weeks, dates[train]))
        test_idx = np.flatnonzero(np.isin(weeks, dates[test]))
        if weeks[train_idx].max() + 2 >= weeks[test_idx].min():
            raise ValueError("Training label overlaps the held-out origin week")
        yield train_idx, test_idx


def retrospective(frame):
    """Describe completed sale spells whose next price is already known."""
    past_price = calendar_shift(frame, "price", -1)
    complete = np.isfinite(calendar_shift(frame, "price", -2)) & np.isfinite(
        calendar_shift(frame, "price", 2)
    )
    selected = frame.sale.eq(1) & (frame.next_price >= 0.98 * past_price)
    # The exposure and return rule require weeks -1 and +1. Together with
    # complete, this keeps every week from -2 through +2 for the same brand.
    events = selected & complete
    by_horizon = {}
    for series in ["own", "category"]:
        baseline = np.log(calendar_shift(frame, series, -1)[events])
        by_horizon[series] = {
            str(h): float((np.log(calendar_shift(frame, series, h)[events]) - baseline).mean())
            for h in [-2, 0, 1, 2]
        }
    return {
        "completed_spells": int(selected.sum()),
        "complete_five_week_spells": int(events.sum()),
        "mean_log_change_from_prior_week": by_horizon,
    }


def estimate(frame, exclude_rival_cuts=False):
    from dml_ts.dml._utils import theta_via_fwl
    from dml_ts.dml.double_ml import _influence_function_se

    outcomes = [f"{series}_{h}" for series in SERIES for h in HORIZONS]
    features = ["log_prior_price", "prior_feature", "early_log_sales", "season_sin", "season_cos"]
    data = frame.dropna(subset=["sale", *features, *outcomes])
    dates = np.unique(data.week)
    if exclude_rival_cuts:
        data = data.loc[data.other_cuts.eq(0)]
    x = pd.concat(
        [data[features], pd.get_dummies(data[["store", "brand"]].astype(str), dtype=float)], axis=1
    )
    values = data[["sale", *outcomes]].to_numpy(dtype=float)
    weeks = data.week.to_numpy()
    predictions = np.full_like(values, np.nan)
    folds = []
    for tr, te in forward_splits(weeks, dates):
        if min(np.unique(values[tr, 0]).size, np.unique(values[te, 0]).size) != 2:
            raise ValueError("Missing treatment variation in a training or test fold")
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0, solver="cholesky"))
        predictions[te] = model.fit(x.iloc[tr], values[tr]).predict(x.iloc[te])
        folds.append(
            {
                "train_through": int(weeks[tr].max()),
                "score_from": int(weeks[te].min()),
                "score_to": int(weeks[te].max()),
                "cuts": int(values[te, 0].sum()),
                "rows": len(te),
            }
        )
    scored = np.isfinite(predictions[:, 0])
    keys = data.loc[scored]
    calendar_offsets = weeks[scored] - weeks[scored].min()
    calendar_weeks = int(calendar_offsets.max() + 1)
    actual = values[scored]
    ridge_residuals = actual - predictions[scored]
    report = {
        "rows": len(keys),
        "cuts": int(keys.sale.sum()),
        "stores": int(keys.store.nunique()),
        "weeks": int(keys.week.nunique()),
        "calendar_weeks": calendar_weeks,
        "folds": folds,
        "simultaneous": int(((keys.sale == 1) & (keys.other_cuts > 0)).sum()),
        "treatment_oof_r2": float(1 - np.mean(ridge_residuals[:, 0] ** 2) / np.var(actual[:, 0])),
        "methods": {},
    }
    for method, resid in [
        ("difference", actual - actual.mean(axis=0)),
        ("forward_ridge_PLR", ridge_residuals),
    ]:
        t = resid[:, 0]
        v = np.mean(t**2)
        estimates = {}
        for j, name in enumerate(outcomes):
            y = resid[:, j + 1]
            theta, _ = theta_via_fwl(
                y, t, treatment=actual[:, 0], no_variation_msg="No residual price-cut variation"
            )
            _, scores = _influence_function_se(y, t, theta)
            if method == "difference":
                cut = actual[:, 0] == 1
                np.testing.assert_allclose(
                    theta, actual[cut, j + 1].mean() - actual[~cut, j + 1].mean()
                )
            # Zero score mass for calendar weeks with no eligible origin. The
            # sample mean over this calendar then equals the row-score mean.
            weekly = np.bincount(calendar_offsets, weights=scores)
            hac = newey_west_se(weekly * calendar_weeks / len(t), bandwidth="auto")
            se, bw = float(hac.se), int(hac.bandwidth)
            estimates[name] = {
                "slope": theta,
                "se": se,
                "ci95": [theta - 1.96 * se, theta + 1.96 * se],
            }
        for h in HORIZONS:
            np.testing.assert_allclose(
                estimates[f"category_{h}"]["slope"],
                estimates[f"own_{h}"]["slope"] + estimates[f"rivals_{h}"]["slope"],
                atol=1e-7,
            )
        report["methods"][method] = {
            "residual_treatment_second_moment": v,
            "hac_bandwidth": bw,
            "outcomes": estimates,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve()
    sys.path[:0] = [str(here.parents[5]), str(here.parents[1])]
    from benchmark_io import fetch

    from dml_ts.data.oj_loader import OJ_DATASET_HASH, OJ_DATASET_URL

    source = fetch(OJ_DATASET_URL, args.cache_dir, "oj_large.csv", OJ_DATASET_HASH)
    raw = pd.read_csv(source)
    if len(raw) != 28947 or raw.store.nunique() != 83 or raw.week.nunique() != 121:
        raise ValueError("Unexpected teaching extract")
    frame = prepare(raw)
    print(
        json.dumps(
            {
                "all": estimate(frame),
                "no_simultaneous_rival_cut": estimate(frame, exclude_rival_cuts=True),
                "retrospective_one_week_spells": retrospective(frame),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
