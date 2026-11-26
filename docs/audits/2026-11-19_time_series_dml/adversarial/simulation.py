"""Paired known-effect comparison of parametric, spline and tree-adjusted PLR methods."""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import SplineTransformer
from temporalcv import TimeSeriesCrossValidator, newey_west_se
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from dml_ts.dml._utils import cross_fit_predictions, theta_via_fwl  # noqa: E402
from dml_ts.dml.double_ml import _influence_function_se  # noqa: E402

N = 1200
SEED = 20261121
TRUTH = 1.0
Z95 = norm.ppf(0.975)
SCENARIOS = ("linear", "additive", "interaction")
REGIMES = ("iid", "serial")
METHODS = ("ols", "spline", "crossfit_linear", "crossfit_spline", "dml_trees")


def generate(rep, scenario, regime):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, rep]))
    innovations = rng.normal(size=(N, 6))
    states = innovations.copy()
    if regime == "serial":
        # Stationary from the first draw. The six innovation streams are independent.
        for t in range(1, N):
            states[t] = 0.6 * states[t - 1] + 0.8 * innovations[t]
    elif regime != "iid":
        raise ValueError(f"Unknown regime {regime}")
    x = 4 * ndtr(states[:, :4]) - 2
    if scenario == "linear":
        q = 0.65 * x[:, 0] + 0.45 * x[:, 1]
    elif scenario in ("additive", "interaction"):
        q = np.sin(2 * x[:, 0]) + 0.6 * x[:, 1] ** 2
        if scenario == "interaction":
            q += 2 * ((x[:, 0] > 0) & (x[:, 1] > 0))
    else:
        raise ValueError(f"Unknown confounding function {scenario}")
    d = q + states[:, 4]
    y = TRUTH * d + q + states[:, 5]
    return x, d, y


def design(regime, rep, n=N):
    splitter = (
        KFold(n_splits=3, shuffle=True, random_state=rep)
        if regime == "iid"
        else TimeSeriesCrossValidator(n_splits=3, gap=5)
    )
    splits = list(splitter.split(np.arange(n)))
    used = np.sort(np.concatenate([test for _, test in splits]))
    if len(used) != len(np.unique(used)):
        raise ValueError("Overlapping evaluation blocks")
    return splits, used


def spline_features(x):
    # Knots come from the declared DGP support. No target-period moments set their locations.
    knots = np.tile(np.linspace(-2, 2, 5)[:, None], (1, x.shape[1]))
    return SplineTransformer(knots=knots, degree=3, include_bias=False).fit_transform(x)


def tree_model():
    return HistGradientBoostingRegressor(
        max_iter=48,
        max_leaf_nodes=12,
        min_samples_leaf=12,
        max_bins=32,
        learning_rate=0.12,
        l2_regularization=1,
        early_stopping=False,
        random_state=42,
    )


def estimate(method, x, d, y, regime, splits, used):
    if method not in (*METHODS, "oracle_spline"):
        raise ValueError(f"Unknown estimator {method}")
    features = spline_features(x) if method in ("spline", "crossfit_spline", "oracle_spline") else x
    if method == "oracle_spline":
        features = np.column_stack([features, (x[:, 0] > 0) & (x[:, 1] > 0)])
    if method in ("ols", "spline", "oracle_spline"):
        # FWL exactly equals joint regression on D, intercept, and the chosen X basis.
        w = np.column_stack([np.ones(len(used)), features[used]])
        response = np.column_stack([y[used], d[used]])
        residuals = response - w @ np.linalg.lstsq(w, response, rcond=None)[0]
        yr, dr = residuals.T
    else:
        model = tree_model() if method == "dml_trees" else LinearRegression()
        yh, dh = cross_fit_predictions(features, y, d, splits, model, model)
        yr, dr = y[used] - yh[used], d[used] - dh[used]
    theta, _ = theta_via_fwl(yr, dr, treatment=d[used], no_variation_msg="No treatment variation")
    se, psi = _influence_function_se(yr, dr, theta)
    if regime == "serial":
        se = float(newey_west_se(psi, bandwidth="auto").se)
    if not np.isfinite([theta, se]).all() or se <= 0:
        raise ValueError("Nonfinite fit or nonpositive standard error")
    return theta, se


def run_cell(args, methods=METHODS):
    rep, scenario, regime = args
    x, d, y = generate(rep, scenario, regime)
    splits, used = design(regime, rep)
    rows = []
    for method in methods:
        start = perf_counter()
        theta = se = None
        failure = ""
        try:
            theta, se = estimate(method, x, d, y, regime, splits, used)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "rep": rep,
                "scenario": scenario,
                "regime": regime,
                "method": method,
                "n_scored": len(used),
                "theta": theta,
                "se": se,
                "failure": failure,
                "seconds": perf_counter() - start,
            }
        )
    return rows


def mcse(values):
    x = np.asarray(values)
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def summaries(draws, methods=METHODS):
    rows, pairs = [], []
    for (scenario, regime), group in draws.groupby(["scenario", "regime"], sort=False):
        successful = group.theta.notna() & group.se.notna()
        complete = group.loc[successful].copy()
        complete["error"] = complete.theta - TRUTH
        complete["sq"] = complete.error**2
        complete["cover"] = (abs(complete.error) <= Z95 * complete.se).astype(float)
        for method in methods:
            all_runs = group.loc[group.method == method]
            good = complete.loc[complete.method == method]
            n = len(good)
            p, rmse = good.cover.mean(), np.sqrt(good.sq.mean())
            # Wilson limits remain informative even with zero observed misses or hits.
            scale = 1 + Z95**2 / n if n else np.nan
            center = (p + Z95**2 / (2 * n)) / scale if n else np.nan
            width = Z95 * np.sqrt(p * (1 - p) / n + Z95**2 / (4 * n**2)) / scale if n else np.nan
            rows.append(
                {
                    "scenario": scenario,
                    "regime": regime,
                    "method": method,
                    "attempted": len(all_runs),
                    "successful": n,
                    "failed": len(all_runs) - n,
                    "n_scored": all_runs.n_scored.iloc[0],
                    "bias": good.error.mean(),
                    "bias_mcse": mcse(good.error),
                    "rmse": rmse,
                    "rmse_mcse": mcse(good.sq) / (2 * rmse) if rmse > 0 else np.nan,
                    "coverage": p,
                    "coverage_mcse": mcse(good.cover),
                    "coverage_wilson_lo": center - width,
                    "coverage_wilson_hi": center + width,
                    "empirical_sd": good.theta.std(),
                    "mean_se": good.se.mean(),
                    "median_seconds": all_runs.seconds.median(),
                }
            )
        tree = complete.loc[complete.method == "dml_trees"].set_index("rep")
        for method in methods:
            if method == "dml_trees":
                continue
            base = complete.loc[complete.method == method].set_index("rep")
            pair = tree.join(base, how="inner", lsuffix="_tree", rsuffix="_baseline")
            n = len(pair)
            rt, rb = np.sqrt(pair.sq_tree.mean()), np.sqrt(pair.sq_baseline.mean())
            row = {"scenario": scenario, "regime": regime, "baseline": method, "paired_n": n}
            rmse_se = (
                mcse(pair.sq_tree / (2 * rt) - pair.sq_baseline / (2 * rb))
                if rt > 0 and rb > 0
                else np.nan
            )
            for name, delta, delta_se in [
                ("rmse", rt - rb, rmse_se),
                (
                    "mse",
                    (pair.sq_tree - pair.sq_baseline).mean(),
                    mcse(pair.sq_tree - pair.sq_baseline),
                ),
                (
                    "bias",
                    (pair.error_tree - pair.error_baseline).mean(),
                    mcse(pair.error_tree - pair.error_baseline),
                ),
                (
                    "coverage",
                    (pair.cover_tree - pair.cover_baseline).mean(),
                    mcse(pair.cover_tree - pair.cover_baseline),
                ),
            ]:
                row.update(
                    {
                        name + "_difference": delta,
                        name + "_mcse": delta_se,
                        name + "_lo": delta - Z95 * delta_se,
                        name + "_hi": delta + Z95 * delta_se,
                    }
                )
            pairs.append(row)
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--interaction-oracle",
        action="store_true",
        help="Compare the post hoc known-interaction spline with the saved primary tree fits",
    )
    args = parser.parse_args()
    if args.repetitions < 2 or args.workers < 1:
        parser.error("Give at least two draws and one worker.")
    scenarios = ("interaction",) if args.interaction_oracle else SCENARIOS
    jobs = [(i, s, r) for i in range(args.repetitions) for s in scenarios for r in REGIMES]
    methods = ("oracle_spline",) if args.interaction_oracle else METHODS
    # Each task fits one independent dataset. Parallelism lives entirely across datasets.
    for thread_setting in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "DML_N_JOBS",
    ):
        os.environ[thread_setting] = "1"
    started = perf_counter()
    results = []
    workers = min(args.workers, len(os.sched_getaffinity(0)), len(jobs))
    with threadpool_limits(limits=1), ProcessPoolExecutor(max_workers=workers) as pool:
        for completed, rows in enumerate(
            pool.map(partial(run_cell, methods=methods), jobs, chunksize=2), 1
        ):
            results.extend(rows)
            if completed % 250 == 0 or completed == len(jobs):
                print(
                    f"{completed}/{len(jobs)} datasets, {perf_counter() - started:.1f}s", flush=True
                )
    frame = pd.DataFrame(results)
    p = args.output
    prefix = "simulation"
    if args.interaction_oracle:
        reference = json.loads((p / "simulation-run.json").read_text())
        if (reference["seed"], reference["n"], reference["draws_per_cell"]) != (
            SEED,
            N,
            args.repetitions,
        ):
            raise ValueError("The primary dataset specification differs")
        primary = pd.read_csv(p / "simulation-draws.csv.gz")
        trees = primary.loc[(primary.scenario == "interaction") & (primary.method == "dml_trees")]
        key = ["rep", "scenario", "regime", "n_scored"]
        if set(trees[key].itertuples(index=False, name=None)) != set(
            frame[key].itertuples(index=False, name=None)
        ) or len(trees) != len(frame):
            raise ValueError("The primary run is missing matching tree fits")
        summary, paired = summaries(
            pd.concat([frame, trees], ignore_index=True), ("oracle_spline", "dml_trees")
        )
        prefix = "simulation-oracle"
    else:
        summary, paired = summaries(frame)
    p.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        p / f"{prefix}-draws.csv.gz",
        index=False,
        float_format="%.12g",
        compression={"method": "gzip", "mtime": 0},
    )
    summary.to_csv(p / f"{prefix}-summary.csv", index=False, float_format="%.12g")
    paired.to_csv(p / f"{prefix}-paired.csv", index=False, float_format="%.12g")
    (p / f"{prefix}-run.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "n": N,
                "draws_per_cell": args.repetitions,
                "workers": workers,
                "worker_threads": 1,
                "datasets": len(jobs),
                "estimates_attempted": len(frame),
                "failures": int(frame.theta.isna().sum()),
                "wall_seconds": perf_counter() - started,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
