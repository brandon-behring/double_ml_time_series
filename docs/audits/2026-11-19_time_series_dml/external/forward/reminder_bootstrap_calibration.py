"""Calibrate the refitted person-bootstrap AIPW forecast-error intervals on synthetic myBP outcomes."""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forward import reminder_rolling as reminders  # noqa: E402
from forward import rolling_calibration as calibration  # noqa: E402

ORIGINS = reminders.ORIGINS[2:6]


def take(m, people, index):
    """Translate simulation moments into the empirical estimator's sufficient statistics."""
    return reminders.Sample(
        pd.Index(people[index]),
        m.n[index],
        m.xsum[index],
        m.xx[index],
        m.xy[index],
        m.y[index],
        {},
    )


def evaluate(cal, rep, seed, draws):
    # Match exactly the main Monte Carlo's synthetic data stream for this repetition.
    rng = np.random.default_rng(np.random.SeedSequence([seed, rep]))
    uniforms = rng.random(len(cal.decision_cells))
    residual = rng.normal(size=(len(cal.n), 2))
    person_shocks = rng.normal(0, 0.7, len(cal.people))
    week_shocks = rng.normal(0, 0.25, cal.week.max() + 1)
    p, stable_tau = calibration.randomization(cal, "stable")
    decline_p, decline_tau = calibration.randomization(cal, "decline")
    if not np.array_equal(p, decline_p):
        raise ValueError("Paired shift requires identical assignment probabilities")
    m = calibration.moments(cal, p, stable_tau, uniforms, residual, person_shocks, week_shocks)
    weights = np.vstack(
        (
            np.ones(len(cal.people)),
            reminders.joint_person_weights(
                cal.enroll, draws, np.random.SeedSequence([seed, rep, 6863])
            ),
        )
    )
    rows = []
    for origin in ORIGINS:
        start = origin.to_datetime64()
        early = np.flatnonzero(cal.enroll + np.timedelta64(98, "D") < start)
        late = np.flatnonzero(
            (cal.enroll >= start)
            & (cal.enroll < (origin + pd.DateOffset(months=3)).to_datetime64())
        )
        if not len(early) or not len(late):
            raise ValueError("The calibration needs a nonempty mature and evaluation cohort")
        # Both scenarios have identical early Y. After the break, each activity-arm Y
        # in a given evaluation cohort is shifted by exactly -0.10, for every weight vector.
        if np.any(decline_tau[early] != stable_tau[early]) or np.ptp(decline_tau[late]) > 0:
            raise ValueError("Cohort crosses the predeclared effect change")
        train = take(m, cal.people, early)
        later = take(m, cal.people, late)
        f = reminders.crossfit(train, weights[:, early])
        z = reminders.arm_difference(later, weights[:, late])
        support = all(np.all((s.xx[:, :, 0, 0] > 0).sum(axis=0) >= 2) for s in (train, later))
        for scenario, tau in [("stable", stable_tau), ("decline", decline_tau)]:
            if np.ptp(tau[early]) or np.ptp(tau[late]):
                raise ValueError("Shift calibration expects a common effect inside each cohort")
            truth = float(tau[early[0]] - tau[late[0]])
            row = {
                "rep": rep,
                "origin": origin.date().isoformat(),
                "scenario": scenario,
                "true_error": truth,
                "signed_error": np.nan,
                "covered": np.nan,
                "alert": np.nan,
                "width": np.nan,
            }
            if f is not None and z is not None:
                # As in the saved empirical freeze, round only the point forecast for scoring.
                diff = float(tau[late[0]] - stable_tau[late[0]])
                row["signed_error"] = float(np.round(f[0], 12) - (z[0] + diff))
                if support:
                    lower, upper = np.quantile(f[1:] - (z[1:] + diff), [0.025, 0.975])
                    row.update(
                        covered=float(lower <= truth <= upper),
                        alert=float(lower > 0 or upper < 0),
                        width=float(upper - lower),
                    )
            rows.append(row)
    return rows


def summarize(rows):
    output = []
    for (scenario, origin), g in rows.groupby(["scenario", "origin"], sort=False):
        available = g.signed_error.notna()
        intervals = g.covered.notna()
        row = {
            "scenario": scenario,
            "origin": origin,
            "replications": len(g),
            "fits": int(available.sum()),
            "intervals": int(intervals.sum()),
            "true_error": float(g.true_error.iloc[0]),
        }
        for name, values in [
            ("coverage", g.covered),
            ("alert", g.alert),
            ("mean_width", g.width),
            ("bias", g.signed_error - g.true_error),
        ]:
            row[name] = values.mean()
            row[name + "_mcse"] = (
                values.std() / np.sqrt(values.count()) if values.count() > 1 else np.nan
            )
        output.append(row)
    result = pd.DataFrame(output)
    dates = [o.date().isoformat() for o in ORIGINS]
    postbreak = [date for date in dates if np.datetime64(date) >= calibration.BREAK]
    for scenario, group in rows.groupby("scenario"):
        # Use one entry per full simulated panel so repeated shocks and training
        # samples stay paired. An unevaluable date is not evidence of no alert.
        panel = group.pivot(index="rep", columns="origin", values="alert")[dates].dropna()
        end = result.scenario.eq(scenario) & result.origin.eq(dates[-1])
        result.loc[end, "complete_panels"] = len(panel)
        for key, columns in [
            ("any_alert_four_dates", dates),
            ("any_alert_two_postbreak_dates", postbreak),
        ]:
            outcomes = panel[columns].astype(bool).any(axis=1)
            result.loc[end, key] = outcomes.mean()
            result.loc[end, key + "_mcse"] = (
                outcomes.std() / np.sqrt(len(outcomes)) if len(outcomes) > 1 else np.nan
            )
    return result


_WORKER_CALENDAR = None


def init(cal):
    global _WORKER_CALENDAR
    _WORKER_CALENDAR = cal
    threadpool_limits(1)


def worker(key):
    return evaluate(_WORKER_CALENDAR, *key)


def study(cal, repetitions=500, draws=999, workers=1, seed=calibration.SEED):
    if min(repetitions, workers) < 1 or draws < 2:
        raise ValueError(
            "Positive repetitions and workers and at least two bootstrap draws required"
        )
    keys = [(rep, seed, draws) for rep in range(repetitions)]
    with threadpool_limits(1):
        if workers == 1:
            groups = [evaluate(cal, *key) for key in keys]
        else:
            cores = (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else os.cpu_count() or 1
            )
            with ProcessPoolExecutor(
                min(workers, repetitions, cores), initializer=init, initargs=(cal,)
            ) as pool:
                groups = list(pool.map(worker, keys, chunksize=2))
    summary = summarize(pd.DataFrame([row for group in groups for row in group]))
    summary["bootstrap_draws"] = draws
    summary["seed"] = seed
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional summary CSV path")
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--draws", type=int, default=999)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cal = calibration.calendar(reminders.load_metadata(args.cache_dir)[0])
    result = study(cal, args.repetitions, args.draws, args.workers)
    if args.output:
        args.output.parent.mkdir(exist_ok=True, parents=True)
        result.to_csv(args.output, index=False, float_format="%.12g")
    print(result.to_json(orient="records", indent=2))
