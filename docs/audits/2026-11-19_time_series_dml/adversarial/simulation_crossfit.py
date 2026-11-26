"""Hold the fitted trees fixed and change whether a scored row was in their training sample."""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulation as sim  # noqa: E402

METHODS = ("dml_trees", "cycled_in_sample", "full_in_sample")


def fold_assignments(splits):
    """Each fitted pair predicts its held-out block and the next block in the cycle."""
    for i, (train, held_out) in enumerate(splits):
        included = splits[(i + 1) % len(splits)][1]
        if (
            len(held_out) != len(included)
            or np.intersect1d(train, held_out).size
            or not np.isin(included, train).all()
        ):
            raise ValueError(
                "Expected equal, disjoint K-fold blocks and their training complements"
            )
        yield train, held_out, included


def fit_trees(x, d, y):
    return sim.tree_model().fit(x, y), sim.tree_model().fit(x, d)


def predictions(models, x):
    return np.column_stack([model.predict(x) for model in models])


def score(y_d, predicted):
    yr, dr = (y_d - predicted).T
    theta, _ = sim.theta_via_fwl(
        yr, dr, treatment=y_d[:, 1], no_variation_msg="No residual treatment variation"
    )
    se, _ = sim._influence_function_se(yr, dr, theta)
    if not np.isfinite([theta, se]).all() or se <= 0:
        raise ValueError("Nonfinite fit or nonpositive standard error")
    return theta, se


def evaluate(job):
    rep, scenario = job
    x, d, y = sim.generate(rep, scenario, "iid")
    splits, used = sim.design("iid", rep)
    assert np.array_equal(used, np.arange(len(d)))
    y_d = np.column_stack([y, d])
    estimates = {method: np.full(y_d.shape, np.nan) for method in METHODS}
    failures = dict.fromkeys(METHODS, "")
    times = dict.fromkeys(METHODS, 0.0)
    try:
        for train, held_out, included in fold_assignments(splits):
            start = perf_counter()
            models = fit_trees(x[train], d[train], y[train])
            common_fit_time = perf_counter() - start
            for method, test in (("dml_trees", held_out), ("cycled_in_sample", included)):
                start = perf_counter()
                estimates[method][test] = predictions(models, x[test])
                times[method] += common_fit_time + perf_counter() - start
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        failures["dml_trees"] = failures["cycled_in_sample"] = str(exc)
    start = perf_counter()
    try:
        estimates["full_in_sample"] = predictions(fit_trees(x, d, y), x)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        failures["full_in_sample"] = str(exc)
    times["full_in_sample"] = perf_counter() - start
    rows = []
    for method in METHODS:
        theta = se = None
        if not failures[method]:
            try:
                theta, se = score(y_d, estimates[method])
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                failures[method] = str(exc)
        rows.append(
            {
                "rep": rep,
                "scenario": scenario,
                "regime": "iid",
                "method": method,
                "theta": theta,
                "se": se,
                "n_scored": len(used),
                "seconds": times[method],
                "failure": failures[method],
            }
        )
    return rows


def validate_reference(frame, reference):
    """Verify the regenerated held-out estimates against the primary study before reporting."""
    original = pd.read_csv(reference)
    primary = original.loc[(original.regime == "iid") & (original.method == "dml_trees")]
    result = frame.loc[frame.method == "dml_trees"]
    primary = primary.loc[primary.rep.isin(result.rep.unique())]
    keys = ["scenario", "rep"]
    original = primary.set_index(keys).sort_index()
    result = result.set_index(keys).sort_index()
    if not original.index.equals(result.index):
        raise ValueError("The archived primary schedule and the ablation differ")
    np.testing.assert_allclose(
        result[["theta", "se", "n_scored"]],
        original[["theta", "se", "n_scored"]],
        rtol=6e-12,
        atol=1e-13,
        equal_nan=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--primary", type=Path, default=Path(__file__).with_name("simulation-draws.csv.gz")
    )
    args = parser.parse_args()
    if args.repetitions < 2 or args.workers < 1:
        parser.error("Provide at least two replications and one worker")
    jobs = [(i, s) for i in range(args.repetitions) for s in sim.SCENARIOS]
    workers = min(args.workers, len(os.sched_getaffinity(0)), len(jobs))
    started = perf_counter()
    with threadpool_limits(limits=1), ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [row for dataset in pool.map(evaluate, jobs, chunksize=2) for row in dataset]
    frame = pd.DataFrame(rows)
    validate_reference(frame, args.primary)
    summary, paired = sim.summaries(frame, METHODS)
    p = args.output
    p.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        p / "simulation-crossfit-draws.csv.gz",
        index=False,
        float_format="%.12g",
        compression={"method": "gzip", "mtime": 0},
    )
    summary.to_csv(p / "simulation-crossfit-summary.csv", index=False, float_format="%.12g")
    paired.to_csv(p / "simulation-crossfit-paired.csv", index=False, float_format="%.12g")
    metadata = {
        "seed": sim.SEED,
        "repetitions": args.repetitions,
        "workers": workers,
        "worker_threads": 1,
        "datasets": len(jobs),
        "estimates_attempted": len(frame),
        "failed": int(frame.theta.isna().sum()),
        "validated_held_out_estimates": len(jobs),
        "wall_seconds": perf_counter() - started,
    }
    (p / "simulation-crossfit-run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
