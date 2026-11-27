"""Compare horizon-specific OLS and temporal PLR on an exogenous AR(1) shock."""

import argparse
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from temporalcv import TimeSeriesCrossValidator, newey_west_se
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from dml_ts.dml import TemporalPLRDML  # noqa: E402

HORIZONS = (0, 1, 4)
RHO = 0.6
BETA = 1.0
SEED = 20261121


def path(n, rep):
    rng = np.random.default_rng([SEED, n, rep, 303])
    shock, noise = rng.normal(size=(2, n + max(HORIZONS)))
    y0 = rng.normal(scale=np.sqrt((1 + BETA**2) / (1 - RHO**2)))
    y, _ = lfilter([1.0], [1.0, -RHO], BETA * shock + noise, zi=[RHO * y0])
    lag = np.r_[y0, y[: n - 1]]
    return shock[:n], lag, y


def estimate(job):
    n, rep = job
    shock, lag, y = path(n, rep)
    out = []
    for h in HORIZONS:
        outcome = y[h : h + n]
        trte = list(TimeSeriesCrossValidator(n_splits=3, gap=h).split(shock))
        if any(tr[-1] + h >= te[0] for tr, te in trte):
            raise AssertionError("Last available training outcome crosses the origin")
        scored = np.concatenate([te for _, te in trte])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            plr = TemporalPLRDML(
                n_lags=0, n_splits=3, gap=h, model_y="ridge", model_t="ridge", random_state=SEED
            ).fit(outcome, shock, lag[:, None], time_index=np.arange(n))
        if len(scored) != plr.n_periods or scored[0] != plr.dropped_initial_rows:
            raise AssertionError("The public PLR and local projection must score the same dates")
        w = np.c_[np.ones(len(scored)), lag[scored]]
        d = shock[scored]
        residual = d - w @ np.linalg.lstsq(w, d, rcond=None)[0]
        design = np.column_stack([w, d])
        coef = np.linalg.lstsq(design, outcome[scored], rcond=None)[0]
        psi = residual * (outcome[scored] - design @ coef) / np.mean(residual**2)
        ols_se = float(newey_west_se(psi, bandwidth="auto").se)
        for method, theta, se, nwarn in [
            ("ols", coef[-1], ols_se, 0),
            ("temporal_plr", plr.theta, plr.se, len(caught)),
        ]:
            out.append((n, rep, h, method, len(scored), BETA * RHO**h, theta, se, nwarn))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cores = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    )
    parser.add_argument("--workers", type=int, default=min(8, cores))
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--cache-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers < 1 or args.repetitions < 2:
        parser.error("positive workers and at least two repetitions are required")
    jobs = [(n, r) for r in range(args.repetitions) for n in (300, 1200)]
    workers = min(args.workers, cores, len(jobs))
    with (
        threadpool_limits(limits=1),
        ProcessPoolExecutor(
            max_workers=workers, initializer=threadpool_limits, initargs=(1,)
        ) as pool,
    ):
        rows = [row for draw in pool.map(estimate, jobs, chunksize=4) for row in draw]
        serial = estimate(jobs[0])
        np.testing.assert_allclose(
            [r[4:] for r in rows[: len(serial)]], [r[4:] for r in serial], atol=1e-12, rtol=1e-11
        )
    frame = pd.DataFrame(
        rows, columns="n rep horizon method scored truth estimate se warnings".split()
    )
    frame["error"] = frame.estimate - frame.truth
    frame["squared"] = frame.error**2
    frame["covered"] = abs(frame.error) <= 1.95996398454 * frame.se
    if not np.isfinite(frame[["estimate", "se"]]).all().all():
        raise ValueError("A fitted result is nonfinite")
    summary = (
        frame.groupby(["n", "horizon", "method", "truth", "scored"])
        .agg(
            draws=("rep", "size"),
            mean=("estimate", "mean"),
            bias=("error", "mean"),
            bias_mcse=("error", "sem"),
            mse=("squared", "mean"),
            coverage=("covered", "mean"),
            coverage_mcse=("covered", "sem"),
            mean_se=("se", "mean"),
            warnings=("warnings", "sum"),
        )
        .reset_index()
    )
    summary["rmse"] = np.sqrt(summary.mse)
    args.output.mkdir(exist_ok=True, parents=True)
    summary.to_csv(args.output / "impulse-summary.csv", index=False, float_format="%.12g")
    frame.to_csv(
        args.output / "impulse-draws.csv.gz",
        index=False,
        float_format="%.12g",
        compression={"method": "gzip", "mtime": 0},
    )
    meta = {
        "seed": SEED,
        "rho": RHO,
        "beta": BETA,
        "paths": len(jobs),
        "estimates": len(frame),
        "workers": workers,
        "threads_per_worker": 1,
        "serial_replay": jobs[0],
        "horizons": HORIZONS,
    }
    (args.output / "impulse-run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
