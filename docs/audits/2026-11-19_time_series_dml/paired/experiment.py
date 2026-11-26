"""Matched retrospective splits and as-of rolling effects on known serial processes."""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import SplineTransformer
from temporalcv import TimeSeriesCrossValidator, newey_west_se
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from dml_ts.dml._utils import cross_fit_predictions, theta_via_fwl  # noqa: E402
from dml_ts.dml.double_ml import _influence_function_se  # noqa: E402

SEED = 20261121
SIZES = (300, 1200)
REGIMES = ("iid", "serial", "break", "hidden")
SPLITS = ("shuffled", "two_sided_buffer", "past", "past_buffer")
LEARNERS = ("linear", "spline", "trees")
GAP = 5
Z = 1.95996398454


@lru_cache
def folds(n):
    """Identical scoring dates and training counts in every design, fixed across seeds."""
    dates = np.arange(n)
    width = n // 8
    stop = n // 4 + 4 * width
    budget = n // 4 - GAP
    design = {}
    for name, gap in [("past", 0), ("past_buffer", GAP)]:
        cv = TimeSeriesCrossValidator(n_splits=4, gap=gap, test_size=width)
        design[name] = [(tr[-budget:], te) for tr, te in cv.split(dates[:stop])]
    design["two_sided_buffer"] = []
    for _, te in design["past"]:
        before = budget // 2
        train = np.r_[
            dates[te[0] - GAP - before : te[0] - GAP],
            dates[te[-1] + GAP + 1 : te[-1] + GAP + 1 + budget - before],
        ]
        design["two_sided_buffer"].append((train, te))
    scored = np.concatenate([te for _, te in design["past"]])
    design["shuffled"] = []
    rng = np.random.default_rng(SEED + n)
    for _, positions in KFold(4, shuffle=True, random_state=SEED).split(scored):
        te = scored[positions]
        tr = np.sort(rng.choice(np.setdiff1d(dates, te), budget, replace=False))
        design["shuffled"].append((tr, te))
    return design


def ar1(innovations, rho):
    """Unconditional unit variance from the first date."""
    inputs = innovations * np.sqrt(1 - rho**2)
    inputs[0] = innovations[0]
    return lfilter([1.0], [1.0, -rho], inputs, axis=0)


def generate(n, rep, regime):
    rng = np.random.default_rng([SEED, rep, n])
    raw = rng.normal(size=(n, 7))
    states = ar1(raw, 0.0 if regime == "iid" else 0.6)
    x = 4 * ndtr(states[:, :4]) - 2
    q = np.sin(2 * x[:, 0]) + 0.6 * x[:, 1] ** 2 + 2 * ((x[:, 0] > 0) & (x[:, 1] > 0))
    theta = np.ones(n)
    if regime == "break":
        theta[n // 2 :] = -0.5
    latent = states[:, 6] if regime == "hidden" else np.zeros(n)
    d = q + states[:, 4] + latent
    y = theta * d + q + states[:, 5] + latent
    return x, y, d, theta


def learner_design(name, x):
    if name == "trees":
        model = HistGradientBoostingRegressor(
            max_iter=32,
            max_leaf_nodes=12,
            max_bins=32,
            learning_rate=0.12,
            min_samples_leaf=10,
            l2_regularization=1,
            early_stopping=False,
            random_state=42,
        )
        return x, model
    if name == "spline":
        # Fixed support knots keep even retrospective preprocessing independent of held-out values.
        knots = np.tile(np.linspace(-2, 2, 5)[:, None], (1, x.shape[1]))
        x = SplineTransformer(knots=knots, include_bias=False).fit_transform(x)
    return x, LinearRegression()


def effect(yr, dr):
    est, _ = theta_via_fwl(yr, dr, no_variation_msg="No residual intervention variation")
    iid, scores = _influence_function_se(yr, dr, est)
    hac = float(newey_west_se(scores, bandwidth="auto").se)
    return est, iid, hac


def run_path(job):
    n, rep, regime = job
    x, y, d, theta = generate(n, rep, regime)
    schedule = folds(n)
    scored = np.concatenate([v for _, v in schedule["past"]])
    summary, rolling = [], []
    for name in LEARNERS:
        features, model = learner_design(name, x)
        for split in SPLITS:
            yh, dh = cross_fit_predictions(features, y, d, schedule[split], model, model)
            yr, dr = y - yh, d - dh
            est, iid, hac = effect(yr[scored], dr[scored])
            summary.append(
                (
                    n,
                    rep,
                    regime,
                    name,
                    split,
                    theta[scored].mean(),
                    est,
                    iid,
                    hac,
                    np.mean(yr[scored] ** 2),
                    np.mean(dr[scored] ** 2),
                )
            )
            if split == "past_buffer":
                for i, (train, later) in enumerate(schedule[split][1:], 1):
                    # All earlier scored outcomes at h=0 were observable at this forecast origin.
                    prior = scored[scored < later[0]]
                    frozen, _, old_se = effect(yr[prior], dr[prior])
                    later_est, _, new_se = effect(yr[later], dr[later])
                    rolling.append(
                        (
                            n,
                            rep,
                            regime,
                            name,
                            i,
                            later[0],
                            later[-1],
                            len(prior),
                            len(later),
                            train[-1],
                            theta[later].mean(),
                            frozen,
                            old_se,
                            later_est,
                            new_se,
                            np.mean(yr[later] ** 2),
                            np.mean(dr[later] ** 2),
                        )
                    )
    return summary, rolling


def oracle(n, rep):
    rng = np.random.default_rng([SEED, rep, n, 1])
    raw = rng.normal(size=(n, 2))
    u, v = ar1(raw, 0.6).T
    rows = []
    for design, residual, r in [("both_serial", v, 0.36), ("outcome_only", raw[:, 1], 0.0)]:
        psi = u * residual
        lag = np.arange(1, n)
        finite = 1 + 2 * np.sum((1 - lag / n) * r**lag)
        rows.append(
            (
                n,
                rep,
                design,
                (1 + r) / (1 - r),
                finite,
                psi.sum() ** 2 / n,
                n * float(newey_west_se(psi, bandwidth="auto").se) ** 2,
            )
        )
    return rows


def mean_se(series):
    return series.std(ddof=1) / np.sqrt(len(series))


def summarize(df, rolling, oracle_df):
    df = df.copy()
    df["error"] = df.estimate - df.truth
    df["squared"] = df.error**2
    for se in ("iid", "hac"):
        df["cover_" + se] = abs(df.error) <= Z * df[se]
        df["width_" + se] = 2 * Z * df[se]
    group = ["n", "regime", "learner", "split"]
    stats = (
        df.groupby(group)
        .agg(
            draws=("rep", "size"),
            bias=("error", "mean"),
            bias_mcse=("error", mean_se),
            mse=("squared", "mean"),
            mse_mcse=("squared", mean_se),
            cover_iid=("cover_iid", "mean"),
            cover_hac=("cover_hac", "mean"),
            coverage_mcse=("cover_hac", mean_se),
            width_iid=("width_iid", "mean"),
            width_hac=("width_hac", "mean"),
            outcome_mse=("outcome_mse", "mean"),
            treatment_mse=("treatment_mse", "mean"),
        )
        .reset_index()
    )
    stats["rmse"] = np.sqrt(stats.mse)
    stats["rmse_mcse"] = stats.mse_mcse / (2 * stats.rmse)
    keys = ["n", "regime", "learner", "rep"]
    ref = df[df.split == "shuffled"].set_index(keys)
    pairs = (
        df[df.split != "shuffled"]
        .set_index(keys)
        .join(ref[["squared", "error", "cover_hac"]], rsuffix="_shuffled")
    )
    for col in ["squared", "error", "cover_hac"]:
        pairs["delta_" + col] = pairs[col].astype(float) - pairs[col + "_shuffled"].astype(float)
    paired = (
        pairs.reset_index()
        .groupby(group)
        .agg(
            draws=("rep", "size"),
            delta_mse=("delta_squared", "mean"),
            mcse_mse=("delta_squared", mean_se),
            delta_bias=("delta_error", "mean"),
            mcse_bias=("delta_error", mean_se),
            delta_coverage=("delta_cover_hac", "mean"),
            mcse_coverage=("delta_cover_hac", mean_se),
        )
        .reset_index()
    )
    roll = rolling.copy()
    roll["error"] = roll.frozen - roll.truth
    roll["squared"] = roll.error**2
    roll["cover"] = abs(roll.error) <= Z * roll.prior_se
    roll["late_error"] = roll.later_est - roll.truth
    roll["late_cover"] = abs(roll.late_error) <= Z * roll.later_se
    roll["width"] = 2 * Z * roll.later_se
    rr = (
        roll.groupby(
            ["n", "regime", "learner", "block", "start", "end", "prior_n", "later_n", "truth"]
        )
        .agg(
            draws=("rep", "size"),
            effect_forecast=("frozen", "mean"),
            forecast_error=("error", "mean"),
            forecast_error_mcse=("error", mean_se),
            forecast_mse=("squared", "mean"),
            historical_ci_contains_later_truth=("cover", "mean"),
            later_estimate=("later_est", "mean"),
            later_bias_mcse=("late_error", mean_se),
            later_ci_coverage=("late_cover", "mean"),
            later_ci_width=("width", "mean"),
            outcome_mse=("outcome_mse", "mean"),
            treatment_mse=("treatment_mse", "mean"),
        )
        .reset_index()
    )
    oo = (
        oracle_df.groupby(["n", "design", "asymptotic", "finite"])
        .agg(
            draws=("rep", "size"),
            variance=("square_of_sum", "mean"),
            mcse=("square_of_sum", mean_se),
            mean_hac_variance=("hac_variance", "mean"),
        )
        .reset_index()
    )
    return {"summary": stats, "paired": paired, "rolling": rr, "oracle": oo}


DRAW = "n rep regime learner split truth estimate iid hac outcome_mse treatment_mse".split()
ROLL = (
    "n rep regime learner block start end prior_n later_n nuisance_train_end truth frozen "
    "prior_se later_est later_se outcome_mse treatment_mse"
).split()
ORACLE = "n rep design asymptotic finite square_of_sum hac_variance".split()


def save(frame, target):
    options = {"compression": {"method": "gzip", "mtime": 0}} if target.suffix == ".gz" else {}
    frame.to_csv(target, index=False, float_format="%.12g", **options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cores = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    )
    parser.add_argument("--workers", type=int, default=min(24, cores))
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--cache-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers < 1 or args.repetitions < 2:
        parser.error("positive workers and at least two repetitions are required")
    start = perf_counter()
    jobs = [(n, i, r) for i in range(args.repetitions) for n in SIZES for r in REGIMES]
    draws, roll = [], []
    workers = min(args.workers, cores, len(jobs))
    with (
        threadpool_limits(limits=1),
        ProcessPoolExecutor(
            max_workers=workers, initializer=threadpool_limits, initargs=(1,)
        ) as pool,
    ):
        for i, (ds, rs) in enumerate(pool.map(run_path, jobs, chunksize=2), 1):
            draws.extend(ds)
            roll.extend(rs)
            if i % 400 == 0:
                print(f"{i}/{len(jobs)} paths", flush=True)
    with threadpool_limits(limits=1):
        # Explicit one-worker replay catches split/learner randomness or nested-thread variation.
        check = run_path(jobs[0])
        np.testing.assert_allclose(
            [row[5:] for row in draws[: len(check[0])]],
            [row[5:] for row in check[0]],
            rtol=1e-11,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            [row[4:] for row in roll[: len(check[1])]],
            [row[4:] for row in check[1]],
            rtol=1e-11,
            atol=1e-12,
        )
        od = [row for n in SIZES for i in range(args.repetitions) for row in oracle(n, i)]
    frame, rolling, oo = (
        pd.DataFrame(v, columns=c) for v, c in [(draws, DRAW), (roll, ROLL), (od, ORACLE)]
    )
    for result in (frame, rolling, oo):
        if not np.isfinite(result.select_dtypes("number")).all().all():
            raise ValueError("A fitted result is nonfinite")
    output = args.output
    output.mkdir(exist_ok=True, parents=True)
    for name, data in summarize(frame, rolling, oo).items():
        save(data, output / f"{name}.csv")
    for name, data in [("draws", frame), ("rolling-draws", rolling), ("oracle-draws", oo)]:
        save(data, output / f"{name}.csv.gz")
    manifest = {
        n: {
            name: [{"train": tr.tolist(), "score": te.tolist()} for tr, te in splits]
            for name, splits in folds(n).items()
        }
        for n in SIZES
    }
    (output / "folds.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    meta = {
        "seed": SEED,
        "draws_per_cell": args.repetitions,
        "data_paths": len(jobs),
        "estimates": len(frame),
        "rolling_forecasts": len(rolling),
        "workers": workers,
        "threads_per_worker": 1,
        "serial_replay_path": jobs[0],
        "seconds": round(perf_counter() - start, 1),
    }
    (output / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
