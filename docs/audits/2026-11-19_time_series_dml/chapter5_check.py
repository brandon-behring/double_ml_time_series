"""Independent Chapter 5 reconstruction. Run from the repository root."""

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from pathlib import Path

import numpy as np
from scipy.stats import norm
from temporalcv import newey_west_se

from dml_ts import TemporalPLRDML, double_ml


def chapter_data(rng, n):
    """Match the RNG draw order and parameter defaults in chapter_05.tex."""
    x = rng.standard_normal((n, 3))
    t = np.zeros(n)
    t[0] = rng.standard_normal()
    for i in range(1, n):
        t[i] = 0.3 * t[i - 1] + 0.5 * x[i, 0] + rng.standard_normal()
    e = np.zeros(n)
    e[0] = rng.standard_normal()
    for i in range(1, n):
        e[i] = 0.2 * e[i - 1] + rng.standard_normal()
    return 2 * t + x[:, 0] ** 2 + 0.5 * x[:, 1] + e, t, x


def persistent_data(rng, n):
    x = rng.normal(size=(n, 2))
    uv = rng.normal(size=(n, 2))
    for i in range(1, n):
        uv[i] = 0.8 * uv[i - 1] + 0.6 * uv[i]
    t = x[:, 0] + uv[:, 0]
    return 2 * t + x[:, 1] + uv[:, 1], t, x


def score_rows(name, fit):
    psi = fit.influence_scores
    ac = np.corrcoef(psi[1:], psi[:-1])[0, 1]
    zero = newey_west_se(psi, bandwidth=0).se
    return [
        (name + "/lag0", fit.theta, zero, len(psi), ac, 0),
        (name + "/auto", fit.theta, fit.se, len(psi), ac, fit.hac_bandwidth),
    ]


def run_seed(seed):
    y, t, x = chapter_data(np.random.default_rng(seed), 500)
    iid = double_ml(y, t, x, n_folds=5, model="random_forest", random_state=42)
    temporal = TemporalPLRDML(random_state=42).fit(y, t, x)
    rows = [("chapter/shuffled-iid", iid.theta, iid.se, len(y), np.nan, 0)]
    rows.extend(score_rows("chapter/forward", temporal))
    y, t, x = persistent_data(np.random.default_rng(10000 + seed), 500)
    fit = TemporalPLRDML(n_lags=0, model_y="ridge", model_t="ridge", random_state=42).fit(y, t, x)
    rows.extend(score_rows("persistent/forward", fit))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--repetitions", default=200, type=int)
    parser.add_argument("--draws", type=Path, help="Write each fitted interval to this CSV.")
    args = parser.parse_args()
    critical = norm.ppf(0.975)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        by_seed = list(pool.map(run_seed, range(args.repetitions)))
    if args.draws:
        with args.draws.open("w", newline="") as stream:
            draws = csv.writer(stream, lineterminator="\n")
            draws.writerow(
                ("replication", "fit", "theta", "se", "n_used", "score_ac1", "bandwidth", "covered")
            )
            for replication, fits in enumerate(by_seed):
                for name, theta, se, n, ac, bandwidth in fits:
                    covered = abs(theta - 2) <= critical * se
                    draws.writerow(
                        (
                            replication,
                            name,
                            theta,
                            se,
                            n,
                            ac if np.isfinite(ac) else "",
                            bandwidth,
                            int(covered),
                        )
                    )
    rows = list(chain.from_iterable(by_seed))
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        [
            "fit",
            "R",
            "bias",
            "empirical_sd",
            "mean_se",
            "coverage",
            "mcse",
            "mean_n",
            "mean_score_ac1",
            "mean_bandwidth",
        ]
    )
    for name in dict.fromkeys(row[0] for row in rows):
        theta, se, n, ac, bandwidth = np.array([r[1:] for r in rows if r[0] == name]).T
        coverage = np.mean(abs(theta - 2) <= critical * se)
        writer.writerow(
            [
                name,
                len(n),
                theta.mean() - 2,
                theta.std(ddof=1),
                se.mean(),
                coverage,
                np.sqrt(coverage * (1 - coverage) / len(n)),
                n.mean(),
                ac.mean(),
                bandwidth.mean(),
            ]
        )


if __name__ == "__main__":
    main()
