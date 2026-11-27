"""Daily bikeshare partial slopes and fixed-trajectory treatment simulations."""

import argparse
import hashlib
import warnings
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import binomtest
from sklearn.linear_model import Ridge
from temporalcv import TimeSeriesCrossValidator

from dml_ts import TemporalPLRDML


def load_days(path: Path) -> pd.DataFrame:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401":
        raise ValueError("Source archive checksum differs from the audited release")
    with ZipFile(path) as data, data.open("day.csv") as source:
        df = pd.read_csv(source, parse_dates=["dteday"]).sort_values("dteday")
    assert df.dteday.tolist() == list(pd.date_range("2011-01-01", "2012-12-31"))
    assert df.notna().all().all() and (df.cnt == df.casual + df.registered).all()
    return df


def calendar(df: pd.DataFrame) -> np.ndarray:
    t = (df.dteday - df.dteday.iloc[0]).dt.days.to_numpy() / 365.25
    angle = 2 * np.pi * t
    weekday = pd.get_dummies(df.weekday, drop_first=True, dtype=float)
    return np.column_stack(
        [t, np.sin(angle), np.cos(angle), np.sin(2 * angle), np.cos(2 * angle), weekday, df.holiday]
    )


def fit(y, d, x, days, lags=0, gap=7, bandwidth=7):
    estimator = TemporalPLRDML(
        n_lags=lags,
        model_y="ridge",
        model_t="ridge",
        cv_strategy="time_series_split",
        n_splits=4,
        gap=gap,
        hac_bandwidth=bandwidth,
        random_state=42,
    )
    return estimator.fit(y, d, x, time_index=days)


def main(path: Path, repetitions: int) -> None:
    df = load_days(path)
    x = calendar(df)
    y = df.cnt.to_numpy()
    d = 10 * df.temp.to_numpy()
    dates = df.dteday.to_numpy()
    weather = np.column_stack([x, df.hum, df.windspeed, df.weathersit == 2, df.weathersit == 3])
    for train, test in TimeSeriesCrossValidator(n_splits=4, gap=7).split(x):
        print(
            "Test dates / observations / train and test temperature ranges:",
            df.dteday.iloc[test[0]].date(),
            df.dteday.iloc[test[-1]].date(),
            len(train),
            len(test),
            np.round([min(d[train]), max(d[train]), min(d[test]), max(d[test])], 2),
        )
    for name, controls, lags, gap, bw in [
        ("calendar", x, 0, 7, 7),
        ("white", x, 0, 7, 0),
        ("long HAC", x, 0, 7, 28),
        ("no gap", x, 0, 0, 7),
        ("past temperature", x, 1, 7, 7),
        ("co-weather", weather, 0, 7, 7),
    ]:
        r = fit(y, d, controls, dates, lags, gap, bw)
        print(
            f"{name}: {r.theta:.2f}, SE {r.se:.2f}, CI [{r.ci_lower:.2f}, {r.ci_upper:.2f}], "
            f"n {r.n_periods}, lag/CV loss {r.lagged_rows_dropped}/{r.dropped_initial_rows}, "
            f"R2(y,d) {r.outcome_r2_cv:.3f}/{r.treatment_r2_cv:.3f}, "
            f"mean residual d squared {np.mean(r.T_residual**2):.3f}"
        )
    print(f"Full-sample unadjusted OLS slope: {np.polyfit(d, y, 1)[0]:.2f}")
    print(
        f"Full-sample calendar-adjusted OLS slope: "
        f"{np.linalg.lstsq(np.column_stack([np.ones(len(d)), d, x]), y, rcond=None)[0][1]:.2f}"
    )

    p = expit(-0.4 + 0.8 * df.workingday.to_numpy() + 0.45 * x[:, 1])
    for persistence in (0.0, 0.7):
        for gap in (7, 0):
            calibration(y, x, dates, p, repetitions, persistence, gap)


def calibration(y, x, dates, p, repetitions, persistence, gap):
    print(
        f"Persistence {persistence:g}, gap {gap}, conditional propensity bounds "
        f"[{(1 - persistence) * min(p):.3f}, {persistence + (1 - persistence) * max(p):.3f}]"
    )
    targets = (0.0, 200.0)
    errors = np.empty((repetitions, len(targets)))
    design_errors = np.empty_like(errors)
    covered = np.zeros_like(errors, dtype=bool)
    ses = np.empty_like(errors)
    warnings_by_message = Counter()
    if persistence == 0:
        scored_days, residual_y = fixed_design_diagnostic(y, x, p, gap)
        observed_scores = np.empty(repetitions)
    for i, seed in enumerate(np.random.SeedSequence(20261119).spawn(repetitions)):
        rng = np.random.default_rng(seed)
        draw = rng.binomial(1, p).astype(float)
        # Use common fresh assignments and retention uniforms across all scenarios.
        retain = rng.random(len(p)) < persistence
        known_prob = p.copy()
        for t in range(1, len(p)):
            known_prob[t] = persistence * draw[t - 1] + (1 - persistence) * p[t]
            if retain[t]:
                draw[t] = draw[t - 1]
        if persistence == 0:
            observed_scores[i] = (draw[scored_days] - p[scored_days]) @ residual_y
        for j, target in enumerate(targets):
            with warnings.catch_warnings(record=True) as messages:
                warnings.simplefilter("always")
                res = fit(y + target * draw, draw, x, dates, gap=gap)
            warnings_by_message.update({str(w.message) for w in messages})
            errors[i, j] = res.theta - target
            # Known-propensity randomization benchmark on exactly the scored dates.
            # Subtract a fixed count to stabilize the Horvitz--Thompson contrast.
            scored = slice(-res.n_periods, None)
            centered = (y + target * draw - 4000.0)[scored]
            observed = draw[scored]
            prob = known_prob[scored]
            ht = np.mean(centered * (observed / prob - (1 - observed) / (1 - prob)))
            design_errors[i, j] = ht - target
            ses[i, j] = res.se
            covered[i, j] = res.ci_lower <= target <= res.ci_upper
    for j, target in enumerate(targets):
        err = errors[:, j]
        rate = covered[:, j].mean()
        band = binomtest(int(covered[:, j].sum()), repetitions).proportion_ci(method="wilson")
        print(
            f"target {target:g}, reps {repetitions}, mean {target + err.mean():.2f}, "
            f"bias {err.mean():.2f} +- MCSE {err.std(ddof=1) / np.sqrt(repetitions):.2f}, "
            f"RMSE {np.sqrt(np.mean(err**2)):.2f}, mean SE {ses[:, j].mean():.2f}, "
            f"coverage {rate:.3f} MCSE {np.sqrt(rate * (1 - rate) / repetitions):.3f} "
            f"Wilson [{band.low:.3f}, {band.high:.3f}]"
        )
        reference = design_errors[:, j]
        print(
            f"Known-propensity HT: bias {reference.mean():.2f}, MCSE "
            f"{reference.std(ddof=1) / np.sqrt(repetitions):.2f}, "
            f"RMSE {np.sqrt(np.mean(reference**2)):.2f}"
        )
    print(f"Maximum difference in paired errors {np.max(np.abs(errors[:, 0] - errors[:, 1])):.8g}")
    print("Simulation warnings (number of fits):", dict(warnings_by_message))
    if persistence == 0:
        # At the true constant effect, subtracting that effect from treated
        # outcomes restores y at every date under every possible assignment.
        # Simulate the two-sided sharp-null score test with an independent seed.
        rng = np.random.default_rng(20261120)
        prob = p[scored_days]
        batches = []
        for _ in range(100):
            fresh = rng.binomial(1, prob, size=(100, len(prob)))
            batches.append((fresh - prob) @ residual_y)
        reference = np.concatenate(batches)
        ordered = np.sort(abs(reference))
        at_least_as_extreme = len(reference) - np.searchsorted(ordered, abs(observed_scores))
        pvalues = (1 + at_least_as_extreme) / (1 + len(reference))
        print(
            f"Known-probability randomization test retained truth: {(pvalues > 0.05).sum()}/{repetitions}"
        )


def fixed_design_diagnostic(y, x, p, gap):
    """Compute exact null FWL moment expectations under independent Bernoulli assignment."""
    numerators, denominators, scored, residuals = [], [], [], []
    variance = p * (1 - p)
    for train, test in TimeSeriesCrossValidator(n_splits=4, gap=gap).split(x):
        # Ridge is linear in its response. Fitting each unit vector gives the
        # weights that take a training response vector to test predictions.
        weights = Ridge(alpha=1.0).fit(x[train], np.eye(len(train))).predict(x[test])
        residual_y = y[test] - weights @ y[train]
        mean_residual_d = p[test] - weights @ p[train]
        numerators.append(mean_residual_d @ residual_y)
        # Test draws and training predictions use disjoint, independent dates.
        prediction_variance = (weights**2) @ variance[train]
        denominators.append(np.sum(variance[test] + prediction_variance + mean_residual_d**2))
        scored.extend(test)
        residuals.extend(residual_y)
    print("Expected null FWL numerator contributions by test fold:", np.round(numerators, 2))
    print(
        f"Expected numerator {sum(numerators):.2f}, denominator {sum(denominators):.3f}, "
        f"ratio of expectations {sum(numerators) / sum(denominators):.3f}"
    )
    return np.asarray(scored), np.asarray(residuals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    parser.add_argument("--repetitions", type=int, default=200)
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("at least two repetitions are required")
    main(args.zip, args.repetitions)
