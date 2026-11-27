"""Reproduce the Dominick's tables in Chapter 4, using the pinned extract."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

from dml_ts import double_ml
from dml_ts.data.oj_loader import OJ_DATASET_HASH, OJDataLoader
from dml_ts.dml._panel import panel_standard_error
from dml_ts.dml._utils import compute_r2, cross_fit_predictions, get_nuisance_model, theta_via_fwl
from dml_ts.dml.double_ml import _influence_function_se

BASE = OJDataLoader.DEFAULT_FEATURES
BRANDS = ["dominicks", "minute.maid", "tropicana"]


def forward_folds(week):
    """Lift expanding splits of unique weeks to row indices."""
    dates = np.unique(week)
    for train, test in TimeSeriesSplit(n_splits=5).split(dates):
        yield (
            np.flatnonzero(np.isin(week, dates[train])),
            np.flatnonzero(np.isin(week, dates[test])),
        )


def retail_fit(spec, frame):
    sample, controls, folds = spec
    if sample in BRANDS:
        frame = frame.loc[frame["brand"] == sample]
    if sample in ("low_income", "high_income"):
        below = frame["INCOME"] <= frame["INCOME"].median()
        frame = frame.loc[below if sample == "low_income" else ~below]
    if folds == "forward":
        frame = frame.sort_values(["week", "store", "brand"])
    x = frame[BASE].to_numpy(dtype=float)
    if controls == "brand":
        indicators = frame["brand"].to_numpy()[:, None] == np.array(BRANDS[1:])
        x = np.column_stack([x, indicators])
    y = frame["logmove"].to_numpy()
    t = np.log(frame["price"].to_numpy())
    # Each process fits one specification. Limit the forest's inner workers
    # even when the invoking shell leaves DML_N_JOBS at its all-cores default.
    model = get_nuisance_model("random_forest", random_state=42).set_params(n_jobs=1)
    if folds == "shuffled":
        fit = double_ml(y, t, x, n_folds=5, model=model, random_state=42)
        u, v, theta, iid_se = fit.Y_residual, fit.T_residual, fit.theta, fit.se
        scores = fit.influence_scores
        r2y, r2t = fit.outcome_r2_cv, fit.treatment_r2_cv
    else:
        yhat, that = cross_fit_predictions(x, y, t, forward_folds(frame["week"]), model, model)
        scored = np.isfinite(yhat) & np.isfinite(that)
        frame = frame.iloc[np.flatnonzero(scored)]
        u, v = y[scored] - yhat[scored], t[scored] - that[scored]
        theta, _ = theta_via_fwl(u, v, no_variation_msg="No residual price variation.")
        iid_se, scores = _influence_function_se(u, v, theta)
        r2y, r2t = compute_r2(y[scored], yhat[scored]), compute_r2(t[scored], that[scored])

    # A one-coefficient no-intercept final regression has the same score and
    # Jacobian as PLR. Its CR1 correction per component reduces to G/(G-1).
    residual_fit = sm.OLS(u, v, hasconst=False).fit()
    np.testing.assert_allclose(residual_fit.params[0], theta, rtol=1e-12)
    st = frame["store"].to_numpy()
    wk = frame["week"].to_numpy()
    store_se, _ = panel_standard_error(scores, st, wk, "unit_cluster")
    week_se, _ = panel_standard_error(scores, wk, st, "unit_cluster")
    both_se, _ = panel_standard_error(scores, st, wk, "two_way_cluster")
    both, store, week = cov_cluster_2groups(residual_fit, st, wk)
    np.testing.assert_allclose(
        [both_se, store_se, week_se], np.sqrt([both.item(), store.item(), week.item()]), rtol=1e-12
    )
    v_by_store = pd.Series(v, index=frame.index).groupby(frame["store"]).std()
    v_by_week = pd.Series(v, index=frame.index).groupby(frame["week"]).std()
    return {
        "sample": sample,
        "controls": controls,
        "folds": folds,
        "rows": len(frame),
        "stores": frame["store"].nunique(),
        "weeks": frame["week"].nunique(),
        "store_weeks": len(frame[["store", "week"]].drop_duplicates()),
        "first_week": frame["week"].min(),
        "last_week": frame["week"].max(),
        "theta": theta,
        "se_row": iid_se,
        "se_store": store_se,
        "se_week": week_se,
        "se_store_week": both_se,
        "r2_y": r2y,
        "r2_t": r2t,
        "store_residual_sd_min": v_by_store.min(),
        "store_residual_sd_max": v_by_store.max(),
        "week_residual_sd_min": v_by_week.min(),
        "week_residual_sd_max": v_by_week.max(),
    }


def gasoline_comparison():
    """Chapter's known-answer experiment, preserving its legacy RNG draw order."""
    rng = np.random.RandomState(42)
    n = 5000
    income = rng.normal(50, 15, n)
    urban = rng.normal(0.6, 0.2, n)
    oil = rng.normal(80, 20, n)
    t = np.log(2.50 + 0.015 * oil + 0.005 * income + rng.normal(0, 0.15, n))
    y = 3.0 - 0.35 * t + 0.008 * income - 0.3 * urban - 0.004 * oil + rng.normal(0, 0.3, n)
    results = {}
    for name, features in (
        ("oil_omitted", (income, urban)),
        ("oil_observed", (income, urban, oil)),
    ):
        fit = double_ml(
            y, t, np.column_stack(features), n_folds=5, model="random_forest", random_state=42
        )
        results[name] = {"theta": fit.theta, "se": fit.se, "lo": fit.ci_lower, "hi": fit.ci_upper}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, required=True, help="Directory for the pinned oj_large.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for aggregate tables"
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    data = OJDataLoader(cache_dir=args.cache_dir).load()
    source = args.cache_dir / "oj_large.csv"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != OJ_DATASET_HASH:
        raise ValueError(
            "The cached CSV differs from the pinned extract. Use a fresh cache directory."
        )
    frame = data.raw_df
    if frame.duplicated(["store", "week", "brand"]).any():
        raise ValueError("Duplicate store-week-brand observations.")
    cell_sizes = frame.groupby(["store", "week"])["brand"].nunique()
    if not cell_sizes.eq(3).all() or sorted(frame["brand"].unique()) != BRANDS:
        raise ValueError("Unexpected brand coverage in the pinned extract.")

    specs = [
        ("all", controls, folds)
        for folds in ("shuffled", "forward")
        for controls in ("base", "brand")
    ]
    specs += [(sample, "base", "shuffled") for sample in [*BRANDS, "low_income", "high_income"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        table = pd.DataFrame(pool.map(partial(retail_fit, frame=frame), specs))
    folds = []
    for fold, (train, test) in enumerate(forward_folds(frame["week"]), 1):
        tr, te = frame.iloc[train], frame.iloc[test]
        assert tr["week"].max() < te["week"].min()
        folds.append(
            {
                "fold": fold,
                "train_rows": len(tr),
                "test_rows": len(te),
                "train_weeks": tr["week"].nunique(),
                "test_weeks": te["week"].nunique(),
                "last_train_week": tr["week"].max(),
                "first_test_week": te["week"].min(),
                "last_test_week": te["week"].max(),
            }
        )
    profile = {
        "source_sha256": digest,
        "ols_unadjusted": float(sm.OLS(data.Y, sm.add_constant(data.T)).fit().params[1]),
        "ols_adjusted": float(
            sm.OLS(data.Y, sm.add_constant(np.column_stack([data.T, data.X]))).fit().params[1]
        ),
        "gasoline": gasoline_comparison(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "retail-results.csv", index=False, float_format="%.12g")
    pd.DataFrame(folds).to_csv(args.output_dir / "retail-folds.csv", index=False)
    (args.output_dir / "retail-baselines.json").write_text(json.dumps(profile, indent=2) + "\n")


if __name__ == "__main__":
    main()
