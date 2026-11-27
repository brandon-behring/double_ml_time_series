"""Paired Monte Carlo calibration on the myBP enrollment calendar (synthetic outcomes)."""

import argparse
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forward.reminder_rolling import (
    AGE,
    DAY,
    ENROLL,
    ID,
    ORIGINS,
    PRIOR_STEPS,
    fold_membership,
    load_metadata,
)

SCENARIOS = ("stable", "decline", "changing_assignment", "weak_support")
METHODS = ("zero", "historical", "aipw", "native_plr", "fixed_mix_aipw")
BREAK = np.datetime64("2023-01-01")
SEED = 20261120
# Predeclared sensitivity to limited information beyond one full evaluation cohort.
INFERENCE_ALPHA = 0.05


@dataclass
class Calendar:
    person: np.ndarray  # per participant-week cell
    n: np.ndarray
    x: np.ndarray
    enroll: np.ndarray  # per participant
    people: np.ndarray
    age_group: np.ndarray
    week: np.ndarray  # across participants
    decision_cells: np.ndarray


def calendar(frame):
    """Keep the real decision counts, issuance clock, participants and study/calendar weeks."""
    frame = frame.copy()
    frame["study_week"] = frame[DAY] // 7
    dates = frame[ENROLL] + pd.to_timedelta(frame[DAY], unit="D")
    frame["calendar_week"] = dates.dt.to_period("W").astype(int)
    frame["log_prior"] = np.log1p(frame[PRIOR_STEPS])
    cells = (
        frame.groupby([ID, "study_week", "calendar_week"], sort=True)
        .agg(
            n=(DAY, "size"),
            day=(DAY, "mean"),
            age=(AGE, "first"),
            enroll=(ENROLL, "first"),
            baseline=("base_step", "first"),
            log_prior=("log_prior", "mean"),
        )
        .reset_index()
    )
    persons = cells.drop_duplicates(ID)
    codes = pd.factorize(cells[ID], sort=True)[0]
    x = np.column_stack(
        (
            np.ones(len(cells)),
            (cells.age - 60) / 10,
            (cells.day - 55) / 30,
            np.log1p(cells.baseline),
            cells.log_prior,
        )
    )
    return Calendar(
        codes,
        cells.n.to_numpy(),
        x,
        persons.enroll.to_numpy(),
        persons[ID].to_numpy(),
        (persons.age.to_numpy() > 65).astype(int),
        pd.factorize(cells.calendar_week, sort=True)[0],
        np.repeat(np.arange(len(cells)), cells.n),
    )


def randomization(cal, scenario):
    late = cal.enroll >= BREAK
    group = cal.age_group.astype(bool)
    activity = np.full(len(group), 0.125)
    none = np.full(len(group), 0.75)
    tau = np.full(len(group), 0.12)
    if scenario == "decline":
        tau[late] -= 0.10
    elif scenario == "changing_assignment":
        activity = 0.875 * np.where(group ^ late, 0.5, 0.1)
        none = 0.875 - activity
        tau = np.where(group, 0.20, -0.20)
    elif scenario == "weak_support":
        activity = np.where(group, 0.0005, 0.125)
        none = 0.875 - activity
    elif scenario != "stable":
        raise ValueError(f"Unknown scenario {scenario}")
    return np.column_stack((activity, none)), tau


@dataclass
class Moments:
    n: np.ndarray
    xsum: np.ndarray
    xx: np.ndarray
    xy: np.ndarray
    y: np.ndarray


def moments(cal, probabilities, tau, uniforms, noise, cluster_noise, week_noise):
    person, x = cal.person, cal.x
    cells = cal.decision_cells
    p = probabilities[person[cells]]
    # One shared uniform per decision and shared person/week shocks across all scenarios.
    arm = (uniforms >= p[:, 0]).astype(int) + (uniforms >= p.sum(axis=1)).astype(int)
    counts = np.bincount(3 * cells + arm, minlength=3 * len(x)).reshape(-1, 3)[:, :2]
    base = (
        x @ np.array([0.5, 0.12, -0.04, 0.10, 0.25]) + cluster_noise[person] + week_noise[cal.week]
    )
    outcome_sum = counts * np.column_stack((base + tau[person], base)) + np.sqrt(counts) * noise

    def by_person(arr):
        out = np.zeros((len(cal.people), *arr.shape[1:]))
        np.add.at(out, person, arr)
        return out

    return Moments(
        np.bincount(person, weights=cal.n),
        by_person(cal.n[:, None] * x),
        by_person(counts[:, :, None, None] * x[:, None, :, None] * x[:, None, None, :]),
        by_person(outcome_sum[:, :, None] * x[:, None, :]),
        by_person(outcome_sum),
    )


@dataclass
class Estimate:
    value: float
    truth: float
    se: float


def estimate(ratios, contributions, truth):
    n = len(ratios)
    total = ratios.sum()
    value = contributions.sum() / total
    influence = (contributions - ratios * value) / total
    se = np.linalg.norm(influence) * np.sqrt(n / (n - 1)) if n > 1 else np.nan
    return Estimate(value, truth, se)


def fits(cal, m, ids, p, tau):
    """Cluster influences condition on the sampled calendar. PLR uses saturated strata."""
    if not len(ids):
        return {}
    n, x, xx, xy, y = (a[ids] for a in (m.n, m.xsum, m.xx, m.xy, m.y))
    p, tau, group = p[ids], tau[ids], cal.age_group[ids]
    truth = float(np.average(tau, weights=n))
    counts = xx[:, :, 0, 0]
    out = {}
    weighted_counts = counts / p
    if np.any(weighted_counts.sum(axis=0) == 0):
        return out
    arm_mean = (y / p).sum(axis=0) / weighted_counts.sum(axis=0)
    v = arm_mean[0] - arm_mean[1]
    components = (y / p - weighted_counts * arm_mean) / weighted_counts.sum(axis=0)
    scale = np.sqrt(len(n) / (len(n) - 1)) if len(n) > 1 else np.nan
    out["historical"] = Estimate(
        v, truth, scale * np.linalg.norm(components[:, 0] - components[:, 1])
    )

    # Within-stratum PLR. Its oracle estimand weights strata by residual treatment variation.
    strata = 2 * (cal.enroll[ids] >= BREAK).astype(int) + group
    total_by_stratum = np.bincount(strata, weights=counts.sum(axis=1))
    e = np.divide(
        np.bincount(strata, weights=counts[:, 0]),
        total_by_stratum,
        out=np.zeros_like(total_by_stratum),
        where=total_by_stratum > 0,
    )[strata]
    denom = np.sum((1 - e) ** 2 * counts[:, 0] + e**2 * counts[:, 1])
    if denom > 0:
        native = ((1 - e) * y[:, 0] - e * y[:, 1]).sum() / denom
        strata_y = np.bincount(strata, weights=y.sum(axis=1) - native * counts[:, 0])
        intercept = np.divide(
            strata_y, total_by_stratum, out=np.zeros_like(strata_y), where=total_by_stratum > 0
        )[strata]
        influence = (
            (1 - e) * (y[:, 0] - counts[:, 0] * (native + intercept))
            - e * (y[:, 1] - counts[:, 1] * intercept)
        ) / denom
        native_truth = np.average(tau, weights=n * p.prod(axis=1) / p.sum(axis=1))
        out["native_plr"] = Estimate(native, native_truth, scale * np.linalg.norm(influence))

    if len(ids) < 2:
        return out
    fold = fold_membership(cal.people[ids])
    contributions = np.zeros(len(ids))
    penalty = np.eye(x.shape[1])
    penalty[0, 0] = 0
    for label in np.unique(fold):
        train = fold != label
        if np.any(counts[train].sum(axis=0) == 0):
            return out
        beta = np.linalg.solve(xx[train].sum(axis=0) + penalty, xy[train].sum(axis=0)[:, :, None])[
            :, :, 0
        ]
        for arm, sign in ((0, 1), (1, -1)):
            adj_x = x[~train] - xx[~train, arm, 0] / p[~train, arm, None]
            contributions[~train] += sign * (adj_x @ beta[arm] + y[~train, arm] / p[~train, arm])
    out["aipw"] = estimate(n, contributions, truth)
    if all((counts[group == g].sum(axis=0) > 0).all() for g in (0, 1)):
        fixed = 0.0
        influence = np.zeros(len(ids))
        fixed_truth = 0.0
        for g in (0, 1):
            sel = group == g
            total = n[sel].sum()
            mean = contributions[sel].sum() / total
            fixed += 0.5 * mean
            fixed_truth += 0.5 * np.average(tau[sel], weights=n[sel])
            influence[sel] = 0.5 * (contributions[sel] - n[sel] * mean) / total
        out["fixed_mix_aipw"] = Estimate(fixed, fixed_truth, scale * np.linalg.norm(influence))
    return out


def run_replication(cal, rep, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, rep]))
    uniforms = rng.random(len(cal.decision_cells))
    residual = rng.normal(0, 1, (len(cal.n), 2))
    cluster_noise = rng.normal(0, 0.7, len(cal.people))
    week_noise = rng.normal(0, 0.25, cal.week.max() + 1)
    rows = []
    for scenario in SCENARIOS:
        p, tau = randomization(cal, scenario)
        m = moments(cal, p, tau, uniforms, residual, cluster_noise, week_noise)
        for start in ORIGINS:
            origin = start.date().isoformat()
            early = np.flatnonzero(cal.enroll + np.timedelta64(98, "D") < start.to_datetime64())
            late = np.flatnonzero(
                (cal.enroll >= start.to_datetime64())
                & (cal.enroll < (start + pd.DateOffset(months=3)).to_datetime64())
            )
            before = fits(cal, m, early, p, tau)
            after = fits(cal, m, late, p, tau)
            before["zero"] = Estimate(0.0, 0.0, 0.0)
            for method in METHODS:
                prior = before.get(method)
                evaluation = after.get(
                    method if method in ("native_plr", "fixed_mix_aipw") else "historical"
                )
                row = {
                    "rep": rep,
                    "scenario": scenario,
                    "origin": origin,
                    "method": method,
                    "forecast": None,
                    "reference": None,
                    "oracle": None,
                    "true_error": None,
                    "error": None,
                    "coverage": None,
                    "alert": None,
                    "forecast_loss": None,
                    "reason": "",
                }
                if prior is None or evaluation is None:
                    row["reason"] = "history or treatment support unavailable"
                else:
                    actual = prior.truth - evaluation.truth
                    error = prior.value - evaluation.value
                    row.update(
                        forecast=prior.value,
                        reference=evaluation.value,
                        oracle=evaluation.truth,
                        true_error=actual,
                        error=error,
                        forecast_loss=(prior.value - evaluation.truth) ** 2,
                    )
                    if np.isfinite([prior.se, evaluation.se]).all():
                        # Disjoint people. Conservative df is the smaller available cluster count minus one.
                        df = len(late) - 1 if method == "zero" else min(len(early), len(late)) - 1
                        width = t.ppf(1 - INFERENCE_ALPHA / 2, df) * np.hypot(
                            prior.se, evaluation.se
                        )
                        row.update(
                            coverage=int(abs(error - actual) <= width),
                            alert=int(abs(error) > width),
                        )
                    else:
                        row["reason"] = "point only: insufficient clusters for uncertainty"
                rows.append(row)
    return rows


_WORKER_CALENDAR = None


def _init(cal):
    global _WORKER_CALENDAR
    _WORKER_CALENDAR = cal
    threadpool_limits(limits=1)


def _worker(key):
    return run_replication(_WORKER_CALENDAR, *key)


def study(cal, reps, workers, seed=SEED):
    if reps < 1 or workers < 1:
        raise ValueError("Repetitions and workers must be positive")
    keys = [(i, seed) for i in range(reps)]
    with threadpool_limits(limits=1):
        if workers == 1:
            batches = [run_replication(cal, *key) for key in keys]
        else:
            core_count = (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else os.cpu_count() or 1
            )
            with ProcessPoolExecutor(
                min(workers, reps, core_count), initializer=_init, initargs=(cal,)
            ) as pool:
                batches = list(pool.map(_worker, keys, chunksize=3))
    return pd.DataFrame(r for batch in batches for r in batch)


def summarize(draws):
    index = ["rep", "scenario", "origin", "method"]
    expected = pd.MultiIndex.from_product(
        [range(draws.rep.nunique()), SCENARIOS, [d.date().isoformat() for d in ORIGINS], METHODS],
        names=index,
    )
    observed = pd.MultiIndex.from_frame(draws[index])
    if (
        draws.empty
        or not observed.is_unique
        or not observed.sort_values().equals(expected.sort_values())
    ):
        raise ValueError("Incomplete or duplicate scheduled simulations")
    output = []
    for (scenario, origin, method), g in draws.groupby(
        ["scenario", "origin", "method"], sort=False
    ):
        row = {
            "scenario": scenario,
            "origin": origin,
            "method": method,
            "reps": len(g),
        }
        for name, values in (
            ("scoreable", g.error.notna().astype(float)),
            ("with_interval", g.alert.notna().astype(float)),
            ("coverage", g.coverage),
            ("alert", g.alert),
            ("forecast_loss", g.forecast_loss),
            ("mean_signed_oracle_error", g.forecast - g.oracle),
        ):
            row[name] = values.mean()
            row[name + "_mcse"] = (
                values.std() / np.sqrt(values.count()) if values.count() > 1 else np.nan
            )
        output.append(row)
    return pd.DataFrame(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cal = calendar(load_metadata(args.cache_dir)[0])
    draws = study(cal, args.repetitions, args.workers)
    summary = summarize(draws)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        # mtime and header filename fixed for deterministic gzip bytes.
        import gzip

        with (args.output / "calibration-draws.csv.gz").open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as f:
                    draws.to_csv(f, index=False, float_format="%.12g")
        summary.to_csv(args.output / "calibration-summary.csv", index=False, float_format="%.12g")
    stable = summary.query("scenario=='stable' and method=='aipw'").coverage.dropna()
    print(
        json.dumps(
            {
                "repetitions": args.repetitions,
                "seed": SEED,
                "scheduled": len(draws),
                "participants": len(cal.people),
                "decisions": len(cal.decision_cells),
                "stable_aipw_coverage": stable.tolist(),
            }
        )
    )


if __name__ == "__main__":
    main()
