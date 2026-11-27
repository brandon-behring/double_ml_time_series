"""Compare the public temporal PLR and the UCI bike audit on identical offer hours."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import bike_dml as bike
import numpy as np
from temporalcv import PurgedWalkForward
from threadpoolctl import threadpool_limits

from dml_ts import TemporalPLRDML

SEED = 20261120
GAP = 168
LAG = 48


def digest(hours):
    return hashlib.sha256(np.asarray(hours, dtype="<i8").tobytes()).hexdigest()


def score(u, v, hours):
    """FWL and Bartlett sandwich, placing missing hours on the physical time axis."""
    denom = v @ v
    theta = (u @ v) / denom
    moment = np.zeros(hours[-1] - hours[0] + 1, dtype=float)
    moment[hours - hours[0]] = v * (u - theta * v)
    total = moment @ moment
    for lag in range(1, LAG + 1):
        total += 2 * (1 - lag / (LAG + 1)) * (moment[lag:] @ moment[:-lag])
    return float(theta), float(np.sqrt(total) / denom)


def ridge_residuals(x, y, d, folds):
    """Independent centered normal equations with an unpenalized intercept and alpha one."""
    out = []
    for train, test in folds:
        mean_x = x[train].mean(axis=0)
        centered = x[train] - mean_x
        targets = np.column_stack((y[train], d[train]))
        mean_y = targets.mean(axis=0)
        coef = np.linalg.solve(
            centered.T @ centered + np.eye(x.shape[1]), centered.T @ (targets - mean_y)
        )
        actual = np.column_stack((y[test], d[test]))
        out.append(actual - mean_y - (x[test] - mean_x) @ coef)
    return np.concatenate(out).T


def summarize(native, common, calendar_ridge, benchmarks, hours, u, v):
    positional = score(u, v, np.arange(len(v)))
    np.testing.assert_allclose(positional, [native.theta, native.se], rtol=1e-9, atol=1e-9)
    matched = score(u[common], v[common], hours[common])
    return {
        "native_all_oof_theta": native.theta,
        "native_all_oof_se_rows": native.se,
        "native_matched_theta": matched[0],
        "native_matched_se_hours": matched[1],
        "native_matched_se_rows": score(u[common], v[common], np.arange(common.sum()))[1],
        "calendar_matched_ridge_theta": calendar_ridge[0],
        "calendar_matched_ridge_se_hours": calendar_ridge[1],
        "historical": benchmarks,
    }


def evaluate(h, stamps, times, x, offers, outcomes, propensity):
    future = bike.shifted_indices(stamps, h)
    origins = np.flatnonzero(future >= 0)
    endpoints = future[origins]
    np.testing.assert_array_equal(stamps[endpoints], stamps[origins] + h)
    calendar = bike.fit_blocks(x, times, stamps, {h: future}, GAP)[h]
    history_test = np.concatenate([test for _, test, _ in calendar])
    cv = PurgedWalkForward(n_splits=5, purge_gap=GAP + h, embargo_pct=0.0)
    folds = list(cv.split(origins))
    native_test = origins[np.concatenate([test for _, test in folds])]
    assert np.all(np.diff(native_test) > 0)
    common = np.isin(native_test, history_test)
    np.testing.assert_array_equal(native_test[common], history_test)
    for train, test, _ in calendar:
        assert stamps[future[train]].max() < stamps[test].min() - GAP
    # A 168-row gap alone may give less than the specified separation for shifted outcomes.
    unadjusted = PurgedWalkForward(n_splits=5, purge_gap=GAP, embargo_pct=0.0)
    raw_separation = [
        int(stamps[origins[test]].min() - stamps[endpoints[train]].max())
        for train, test in unadjusted.split(origins)
    ]
    separation = [
        int(stamps[origins[test]].min() - stamps[endpoints[train]].max()) for train, test in folds
    ]
    assert min(separation) > GAP
    out = {
        "h": h,
        "target": bike.EFFECT[h] if h < len(bike.EFFECT) else 0,
        "eligible_pairs": len(origins),
        "native_scored": len(native_test),
        "matched_scored": len(history_test),
        "matched_hour_sha256": digest(stamps[history_test]),
        "native_train_outcome_to_test_hours": separation,
        "unadjusted_168_row_outcome_to_test_hours": raw_separation,
        "matched_row_lag_48_pairs_spanning_more_than_48_hours": int(
            np.count_nonzero(stamps[history_test[LAG:]] - stamps[history_test[:-LAG]] > LAG)
        ),
        "native_train_sizes": [len(train) for train, _ in folds],
        "calendar_train_sizes": [len(train) for train, _, _ in calendar],
        "draws": [],
    }
    calendar_folds = [
        (np.searchsorted(origins, tr), np.searchsorted(origins, te)) for tr, te, _ in calendar
    ]
    for (ti, vi), (train, test, _) in zip(calendar_folds, calendar, strict=True):
        np.testing.assert_array_equal(origins[ti], train)
        np.testing.assert_array_equal(origins[vi], test)
    for d, y in zip(offers, outcomes, strict=True):
        # Fit the public class directly. Its lagged controls are omitted in the original benchmark.
        model = TemporalPLRDML(
            n_lags=0,
            model_y="ridge",
            model_t="ridge",
            cv_strategy="purged_cv",
            n_splits=5,
            gap=GAP + h,
            hac_bandwidth=LAG,
            random_state=SEED,
        )
        result = model.fit(y[endpoints], d[origins], x[origins], time_index=stamps[origins])
        assert result.n_periods == len(native_test) and result.lagged_rows_dropped == 0
        u, v = result.Y_residual, result.T_residual
        ref = ridge_residuals(x[origins], y[endpoints], d[origins], folds)
        np.testing.assert_allclose([u, v], ref, rtol=1e-9, atol=1e-8)
        # Holding learners and score hours fixed isolates the effect of changing fold boundaries.
        ref_calendar = ridge_residuals(x[origins], y[endpoints], d[origins], calendar_folds)
        calendar_ridge = score(*ref_calendar, stamps[history_test])
        benchmarks = {}
        for method in ("known propensity", "hour and workday", "expanded calendar fit"):
            theta, se, n = bike.estimate(y, d, propensity, x, stamps, future, calendar, method, LAG)
            assert n == len(history_test)
            benchmarks[method] = {"theta": float(theta), "se_hours": float(se)}
        row = summarize(result, common, calendar_ridge, benchmarks, stamps[native_test], u, v)
        row["max_abs_ridge_residual_difference"] = float(np.max(np.abs(ref - [u, v])))
        row["native_matched_propensity_outside_unit"] = int(
            np.count_nonzero((d[history_test] - v[common] < 0) | (d[history_test] - v[common] > 1))
        )
        out["draws"].append(row)
    return out


def run(path, draws, horizons):
    stamps, times, base, x, propensity = bike.data(path)
    rng = np.random.default_rng(SEED)
    offers, outcomes = [], []
    for _ in range(draws):
        d = (rng.random(len(stamps)) < propensity).astype(float)
        y = base.copy()
        for h, effect in enumerate(bike.EFFECT):
            future = bike.shifted_indices(stamps, h)
            keep = future >= 0
            y[future[keep]] += effect * d[keep]
        offers.append(d)
        outcomes.append(y)
    return {
        "archive_sha256": bike.ZIP_HASH,
        "seed": SEED,
        "draws": draws,
        "offer_sha256": [hashlib.sha256(d.astype("u1").tobytes()).hexdigest() for d in offers],
        "gap_hours": GAP,
        "hac_bandwidth": LAG,
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("dml_ts", "temporalcv", "scikit-learn", "numpy", "scipy")
        },
        "horizons": [evaluate(h, stamps, times, x, offers, outcomes, propensity) for h in horizons],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    parser.add_argument("--draws", choices=range(1, 5), type=int, default=4)
    parser.add_argument(
        "--horizons", choices=bike.HORIZONS, type=int, nargs="+", default=bike.HORIZONS
    )
    args = parser.parse_args()
    if len(set(args.horizons)) != len(args.horizons):
        parser.error("each horizon must appear once")
    with threadpool_limits(limits=1):
        print(json.dumps(run(args.zip, args.draws, args.horizons), indent=2, allow_nan=False))
