"""Rerun every simulation quoted in the 2026-11-22 mathematical review.

Run from the repository root::

    uv run --locked python docs/audits/2026-11-22_mathematical_review/checks.py

The driver writes ``draws.csv.gz`` (one row per estimate), ``summary.csv`` (computed from that
file) and ``run.json`` (versions, source hashes, deterministic checks, warning counts). Seeds and
draw order are those of the original exploratory runs, so ``review_*`` studies replay the
reviewer's simulations and ``refute_*`` studies replay the attempt to refute findings F2 to F4.

Estimation goes through the package (``TemporalPLRDML``, ``cross_fit_predictions``, the learner
factory) and inference through ``temporalcv.newey_west_se``. Only what neither provides is
implemented here: the equal-weighted cosine variance, buffered two-sided folds, and recursive
g-estimation on each treatment's own-date state.
"""

from __future__ import annotations

import os

if __name__ == "__main__":
    # Read when numpy and dml_ts are imported. One thread per worker keeps the replay exact.
    os.environ.update(DML_N_JOBS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")

import argparse
import gzip
import hashlib
import io
import json
import platform
import subprocess
import time
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from temporalcv import (
    PurgedKFold,
    TimeSeriesCrossValidator,
    check_temporal_splitter,
    newey_west_se,
)
from threadpoolctl import threadpool_limits

import dml_ts
from dml_ts.dml import TemporalPLRDML
from dml_ts.dml._utils import ModelType, cross_fit_predictions, get_nuisance_model
from dml_ts.dml.dynamic_g_estimation import DynamicGEstimationDML
from dml_ts.sensitivity import compute_sensitivity_for_dml
from dml_ts.validation.dynamic_dgp import DynamicTreatmentDGP, DynamicTreatmentDGPResult

HERE = Path(__file__).resolve().parent
Z = float(stats.norm.ppf(0.975))

Floats = NDArray[np.float64]
Split = tuple[NDArray[np.int64], NDArray[np.int64]]
Row = dict[str, Any]
Sample = tuple[Floats, Floats, Floats]

COLUMNS = [
    "study",
    "design",
    "n",
    "variant",
    "component",
    "replicate",
    "truth",
    "error",
    "n_scored",
    "se_lag0",
    "se_nw",
    "se_prewhite",
    "se_long",
    "long_lags",
    "se_cosine",
    "se_sandwich",
    "score_acf1",
    "r2_warning",
]
FIXED_B_GRID = (0.02, 0.065, 0.0675, 0.103, 0.109, 0.15, 0.25, 0.5)


def require(condition: bool, message: str) -> None:
    """Raise ``ValueError(message)`` unless the condition holds.

    Examples
    --------
    >>> require(1 < 2, "never raised")
    >>> require(2 < 1, "expected 2 < 1")
    Traceback (most recent call last):
    ...
    ValueError: expected 2 < 1
    """
    if not condition:
        raise ValueError(message)


# --------------------------------------------------------------------- inference helpers


def cosine_dof(n: int) -> int:
    """Return nu = floor(0.4 n^(2/3)), at least 2 (Lazarus, Lewis, Stock and Watson 2018).

    Examples
    --------
    >>> cosine_dof(415)
    22
    """
    require(n >= 3, f"expected at least 3 scores, got {n}")
    return max(2, int(np.floor(0.4 * n ** (2 / 3))))


def cosine_long_run_variance(scores: Floats, nu: int | None = None) -> tuple[float, int]:
    """Return the equal-weighted cosine long-run variance and its degrees of freedom.

    The estimate is the mean of nu squared projections of the scores on type-II cosines. Each
    cosine is orthogonal to a constant, so no demeaning is needed. Cost is O(n nu).

    Parameters
    ----------
    scores
        Finite one-dimensional score series.
    nu
        Number of projections. Defaults to :func:`cosine_dof`.

    Returns
    -------
    tuple of float and int
        The variance and nu. Intervals take critical values from t with nu degrees of freedom.

    Examples
    --------
    >>> omega, nu = cosine_long_run_variance(np.sqrt(2) * np.cos(np.pi * (np.arange(8) + 0.5) / 8))
    >>> round(omega, 10), nu
    (4.0, 2)
    """
    require(scores.ndim == 1, f"expected a 1-D score series, got shape {scores.shape}")
    require(bool(np.isfinite(scores).all()), "expected finite scores, got NaN or inf")
    n = len(scores)
    nu = cosine_dof(n) if nu is None else nu
    require(1 <= nu < n, f"expected 1 <= nu < {n}, got {nu}")
    phases = np.outer(np.arange(1, nu + 1), np.arange(1, n + 1) - 0.5)
    basis = np.sqrt(2 / n) * np.cos(np.pi * phases / n)
    return float(np.mean((basis @ scores) ** 2)), nu


def fixed_b_critical_value(b: float | Floats) -> float | Floats:
    """Return the two-sided 5% Bartlett fixed-b critical value.

    The cubic was recalled from Kiefer and Vogelsang (2005), not read from the paper.
    :func:`fixed_b_simulation` checks it against simulated critical values.

    Examples
    --------
    >>> float(fixed_b_critical_value(0.0))
    1.96
    """
    require(bool(np.all((np.asarray(b) >= 0) & (np.asarray(b) <= 1))), f"b outside [0, 1]: {b}")
    return 1.9600 + 2.9694 * b + 0.4160 * b**2 - 0.5324 * b**3


def fixed_b_simulation(chunk: int, n: int = 2000, reps: int = 5000) -> Floats:
    """Simulate absolute Bartlett fixed-b t statistics on i.i.d. location samples.

    The Bartlett variance is computed as scaled squared partial sums over windows of b n rows,
    in O(reps n) per value of b.

    Examples
    --------
    >>> fixed_b_simulation(0, n=100, reps=4).shape
    (8, 4)
    """
    require(n >= 100 and reps >= 1, f"expected n >= 100 and reps >= 1, got {n}, {reps}")
    x = np.random.default_rng([20261122, chunk]).standard_normal((reps, n))
    mean = x.mean(axis=1)
    total = np.zeros((reps, n + 1))
    np.cumsum(x - mean[:, None], axis=1, out=total[:, 1:])
    out = np.empty((len(FIXED_B_GRID), reps))
    for i, b in enumerate(FIXED_B_GRID):
        size = round(b * n)
        starts = np.arange(-size + 1, n)
        windows = total[:, np.minimum(starts + size, n)] - total[:, np.maximum(starts, 0)]
        omega = (windows**2).sum(axis=1) / (n * size)
        out[i] = np.sqrt(n) * np.abs(mean) / np.sqrt(omega)
    return out


def fixed_b_table(simulated: Sequence[Floats]) -> list[Row]:
    """Set simulated fixed-b critical values beside the recalled cubic.

    Examples
    --------
    >>> fixed_b_table([fixed_b_simulation(0, n=100, reps=20)])[0]["b"]
    0.02
    """
    values = np.concatenate(simulated, axis=1)
    return [
        {
            "b": b,
            "simulated": float(np.quantile(values[i], 0.95)),
            "recalled_polynomial": float(fixed_b_critical_value(b)),
        }
        for i, b in enumerate(FIXED_B_GRID)
    ]


# ----------------------------------------------------------------------- cross-fit helpers


def buffered_two_sided_splits(n: int, n_splits: int, gap: int) -> list[Split]:
    """Return contiguous blocks, each trained on both sides outside a buffer.

    These are the folds of ``temporalcv.PurgedKFold(purge_gap=gap, embargo_pct=0.0)`` when
    ``n_splits`` divides ``n``. The two differ only in where the remainder rows go.

    Parameters
    ----------
    n
        Number of rows.
    n_splits
        Number of test blocks.
    gap
        Rows removed from training on each side of the test block.

    Returns
    -------
    list of (train, test) index arrays

    Examples
    --------
    >>> train, test = buffered_two_sided_splits(10, 5, 1)[1]
    >>> train.tolist(), test.tolist()
    ([0, 5, 6, 7, 8, 9], [2, 3])
    """
    require(2 <= n_splits <= n, f"expected 2 <= n_splits <= {n}, got {n_splits}")
    require(gap >= 0, f"expected gap >= 0, got {gap}")
    index = np.arange(n)
    folds = [
        (index[(index < block[0] - gap) | (index > block[-1] + gap)], block)
        for block in np.array_split(index, n_splits)
    ]
    require(all(len(train) for train, _ in folds), f"gap {gap} empties a training set")
    return folds


def make_splits(scheme: str, n: int, n_splits: int = 5, gap: int = 0) -> list[Split]:
    """Return the folds of one cross-fitting scheme.

    ``forward`` expands from the start, ``half`` expands but scores only the second half,
    ``two_sided`` is the upstream purged K-fold, and ``neighbours_out`` omits adjacent blocks.

    Examples
    --------
    >>> train, test = make_splits("half", 20)[0]
    >>> len(train), test.tolist()
    (10, [10, 11])
    """
    require(2 <= n_splits <= n and gap >= 0, f"invalid n={n}, n_splits={n_splits}, gap={gap}")
    rows = np.zeros((n, 1))
    if scheme == "forward":
        cv = TimeSeriesCrossValidator(n_splits=n_splits, gap=gap, expanding=True)
        return list(cv.split(rows))
    if scheme == "two_sided":
        return list(PurgedKFold(n_splits=n_splits, purge_gap=gap, embargo_pct=0.0).split(rows))
    if scheme == "half":
        require(n_splits <= n - n // 2 and gap < n // 2, f"too few rows for half: n={n}")
        blocks = np.array_split(np.arange(n // 2, n), n_splits)
        return [(np.arange(block[0] - gap), block) for block in blocks]
    if scheme == "neighbours_out":
        require(n_splits >= 4 and gap == 0, "neighbours_out needs 4+ blocks and takes no gap")
        blocks = np.array_split(np.arange(n), n_splits)
        return [
            (np.concatenate([b for j, b in enumerate(blocks) if abs(j - k) > 1]), blocks[k])
            for k in range(n_splits)
        ]
    raise ValueError(f"unknown cross-fitting scheme {scheme!r}")


class Fit(NamedTuple):
    """The parts of a partially linear fit that the checks use.

    Examples
    --------
    >>> Fit(2.0, np.zeros(3), np.ones(3, dtype=bool)).theta
    2.0
    """

    theta: float
    scores: Floats
    used: NDArray[np.bool_]


def residual_slope(ry: Floats, rt: Floats) -> tuple[float, Floats]:
    """Return theta = ry'rt / rt'rt and the scores (ry - theta rt) rt / mean(rt^2).

    Examples
    --------
    >>> residual_slope(np.array([2.0, -2.0]), np.array([1.0, -1.0]))[0]
    2.0
    """
    denominator = rt @ rt
    require(bool(denominator > 0), "treatment residuals have no variation")
    theta = float(ry @ rt / denominator)
    return theta, (ry - theta * rt) * rt / np.mean(rt**2)


def cross_fit_plr(
    Y: Floats,
    T: Floats,
    W: Floats,
    splits: Iterable[Split],
    model: BaseEstimator,
    first_row: int = 0,
) -> Fit:
    """Return the slope and scores, using the package's cross-fitting.

    Parameters
    ----------
    Y, T, W
        Outcome, treatment and controls.
    splits
        Train and test indices.
    model
        Learner cloned for both nuisances.
    first_row
        Earlier rows are fitted but not scored.

    Returns
    -------
    Fit

    Examples
    --------
    >>> Y, T, X = review_data(0, 200, 0.5)
    >>> fit = cross_fit_plr(Y, T, X, make_splits("two_sided", 200), Ridge())
    >>> int(fit.used.sum())
    200
    """
    require(len(Y) == len(T) == len(W), f"lengths differ: {len(Y)}, {len(T)}, {len(W)}")
    y_hat, t_hat = cross_fit_predictions(W, Y, T, splits, model, model)
    used = np.isfinite(y_hat) & np.isfinite(t_hat)
    used[:first_row] = False
    require(bool(used.any()), f"no row at or after {first_row} was scored")
    theta, scores = residual_slope((Y - y_hat)[used], (T - t_hat)[used])
    return Fit(theta, scores, used)


# -------------------------------------------------------------- g-estimation helpers


def series_total_blips(dgp: DynamicTreatmentDGP) -> Floats:
    """Return the true series blips: direct effect plus the path through the state.

    At lag k >= 1 this is theta_k + gamma' A^(k-1) b, holding later treatments fixed.

    Examples
    --------
    >>> series_total_blips(DynamicTreatmentDGP(3, [1, 0.5, 0.25], state_feedback=True)).tolist()
    [1.0, 1.0, 0.5]
    """
    total = dgp.theta_direct.astype(float).copy()
    if dgp.state_feedback:
        for k in range(1, dgp.n_periods):
            total[k] += float(dgp.gamma @ np.linalg.matrix_power(dgp.A, k - 1) @ dgp.b)
    return total


def own_date_residuals(
    Y: Floats,
    T: Floats,
    X: Floats,
    m: int,
    ends: NDArray[np.int64],
    splits: Sequence[Split],
    history: bool,
) -> list[Floats]:
    """For each lag k, residualize Y and T at lags 0..k on the state dated k periods back.

    Examples
    --------
    >>> Y, T, X = review_data(0, 60, 0.5)
    >>> ends = np.arange(1, 60)
    >>> folds = make_splits("two_sided", 59)
    >>> [r.shape for r in own_date_residuals(Y, T, X, 2, ends, folds, False)]
    [(59, 2), (59, 3)]
    """
    out = []
    for k in range(m):
        state = X[ends - k]
        if history:
            state = np.column_stack([state, *[T[ends - k - j] for j in range(1, m)]])
        design = np.column_stack([np.ones(len(ends)), state])
        target = np.column_stack([Y[ends], *[T[ends - j] for j in range(k + 1)]])
        residual = np.full(target.shape, np.nan)
        for train, test in splits:
            coef = np.linalg.lstsq(design[train], target[train], rcond=None)[0]
            residual[test] = target[test] - design[test] @ coef
        out.append(residual)
    return out


def peel_blips(residuals: Sequence[Floats]) -> tuple[Floats, Floats, Floats]:
    """Solve the triangular moments from lag 0 up, peeling off each blip once estimated.

    Returns the blips, the per-row moments and their Jacobian.

    Examples
    --------
    >>> peel_blips([np.array([[2.0, 1.0], [-2.0, -1.0]])])[0].tolist()
    [2.0]
    """
    keep = np.isfinite(residuals[0][:, 0])
    require(bool(keep.any()), "no block was scored")
    m, count = len(residuals), int(keep.sum())
    theta = np.zeros(m)
    moments, jacobian = np.zeros((count, m)), np.zeros((m, m))
    for k, residual in enumerate(residuals):
        ry, rt = residual[keep, 0], residual[keep, 1:]
        peeled = ry - rt[:, :k] @ theta[:k]
        theta[k] = rt[:, k] @ peeled / (rt[:, k] @ rt[:, k])
        moments[:, k] = rt[:, k] * (peeled - theta[k] * rt[:, k])
        jacobian[k, : k + 1] = -np.mean(rt[:, [k]] * rt[:, : k + 1], axis=0)
    return theta, moments, jacobian


def sandwich_errors(moments: Floats, jacobian: Floats, hac: bool) -> Floats:
    """Return joint-sandwich standard errors, G^-1 Omega G^-1' / n.

    Examples
    --------
    >>> sandwich_errors(np.array([[1.0], [-1.0]]), np.array([[-1.0]]), False).round(4).tolist()
    [0.7071]
    """
    count, m = moments.shape
    centred = moments - moments.mean(axis=0)
    meat = centred.T @ centred / count
    if hac:
        lags = int(np.floor(count ** (1 / 3))) + m
        require(lags < count, f"{count} rows cannot support {lags} HAC lags")
        for j in range(1, lags + 1):
            lagged = centred[j:].T @ centred[:-j] / count
            meat += (1 - j / (lags + 1)) * (lagged + lagged.T)
    bread = np.linalg.inv(jacobian)
    return np.sqrt(np.diag(bread @ meat @ bread.T / count))


def own_date_recursion(
    Y: Floats,
    T: Floats,
    X: Floats,
    m: int,
    ends: NDArray[np.int64],
    splits: Sequence[Split] | None = None,
    history: bool = False,
    hac: bool = False,
) -> tuple[Floats, Floats]:
    """Linear g-estimation that residualizes each treatment on the state at its own date.

    The recursion is that of Lewis and Syrgkanis (2021), Algorithm 2.

    Parameters
    ----------
    Y, T, X
        One series.
    m
        Number of blips.
    ends
        Dates of the final outcome in each block.
    splits
        Folds over ``ends``. ``None`` fits the nuisances in sample.
    history
        Also condition on the m - 1 treatments preceding the treated date.
    hac
        Add Bartlett autocovariances, for overlapping blocks.

    Returns
    -------
    tuple of arrays
        Blips for lags 0 to m - 1 and their standard errors.

    Examples
    --------
    >>> _, data = feedback_series(400, True, 0)
    >>> own_date_recursion(data.Y, data.T, data.X, 3, np.arange(2, 400))[0].shape
    (3,)
    """
    n = len(ends)
    require(m >= 1 and n > 0, f"expected m >= 1 and some blocks, got m={m}, blocks={n}")
    require(len(Y) == len(T) == len(X), f"lengths differ: {len(Y)}, {len(T)}, {len(X)}")
    earliest = (m - 1) * (2 if history else 1)
    # An earlier end would index from the end of the series without raising.
    require(
        bool(ends.min() >= earliest and ends.max() < len(Y)),
        f"block ends must lie in [{earliest}, {len(Y) - 1}]",
    )
    everything = np.arange(n)
    folds = [(everything, everything)] if splits is None else splits
    theta, moments, jacobian = peel_blips(own_date_residuals(Y, T, X, m, ends, folds, history))
    return theta, sandwich_errors(moments, jacobian, hac)


# -------------------------------------------------------------------- learner candidates


def scaled_ridge() -> BaseEstimator:
    """Return ridge on standardized inputs.

    Examples
    --------
    >>> [name for name, _ in scaled_ridge().steps]
    ['standardscaler', 'ridge']
    """
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def fixed_forest(n_estimators: int) -> RandomForestRegressor:
    """Return the repository's forest settings with a chosen number of trees.

    Examples
    --------
    >>> fixed_forest(50).max_depth
    5
    """
    return RandomForestRegressor(
        n_estimators=n_estimators, max_depth=5, min_samples_leaf=10, random_state=0, n_jobs=1
    )


class ForwardStack(RegressorMixin, BaseEstimator):
    """Ridge and forest, weighted by forward-chained predictions inside the training set.

    Examples
    --------
    >>> Y, _, X = review_data(0, 120, 0.5)
    >>> ForwardStack().fit(X, Y).predict(X[:3]).shape
    (3,)
    """

    def fit(self, X: Floats, y: Floats) -> ForwardStack:
        """Fit the weights, then refit both members on all training rows.

        Examples
        --------
        >>> Y, _, X = review_data(0, 120, 0.5)
        >>> ForwardStack().fit(X, Y).meta_.coef_.shape
        (2,)
        """
        held_out = np.full((len(y), 2), np.nan)
        for train, test in TimeSeriesSplit(n_splits=3).split(X):
            held_out[test, 0] = scaled_ridge().fit(X[train], y[train]).predict(X[test])
            held_out[test, 1] = fixed_forest(50).fit(X[train], y[train]).predict(X[test])
        seen = np.isfinite(held_out[:, 0])
        self.meta_ = LinearRegression().fit(held_out[seen], y[seen])
        self.linear_ = scaled_ridge().fit(X, y)
        self.forest_ = fixed_forest(50).fit(X, y)
        return self

    def predict(self, X: Floats) -> Floats:
        """Combine the members' predictions.

        Examples
        --------
        >>> Y, _, X = review_data(0, 120, 0.5)
        >>> ForwardStack().fit(X, Y).predict(X).shape
        (120,)
        """
        members = np.column_stack([self.linear_.predict(X), self.forest_.predict(X)])
        return np.asarray(self.meta_.predict(members), dtype=float)


class LinearThenForest(RegressorMixin, BaseEstimator):
    """Ridge, then a forest on the ridge residuals.

    Examples
    --------
    >>> Y, _, X = review_data(0, 80, 0.5)
    >>> LinearThenForest(5).fit(X, Y).predict(X[:2]).shape
    (2,)
    """

    def __init__(self, n_estimators: int = 50) -> None:
        """Set the number of trees.

        Examples
        --------
        >>> LinearThenForest(7).n_estimators
        7
        """
        self.n_estimators = n_estimators

    def fit(self, X: Floats, y: Floats) -> LinearThenForest:
        """Fit the linear part, then the forest on what it leaves.

        Examples
        --------
        >>> Y, _, X = review_data(0, 80, 0.5)
        >>> len(LinearThenForest(5).fit(X, Y).forest_.estimators_)
        5
        """
        self.linear_ = scaled_ridge().fit(X, y)
        self.forest_ = fixed_forest(self.n_estimators).fit(X, y - self.linear_.predict(X))
        return self

    def predict(self, X: Floats) -> Floats:
        """Add the two parts.

        Examples
        --------
        >>> Y, _, X = review_data(0, 80, 0.5)
        >>> LinearThenForest(5).fit(X, Y).predict(X).shape
        (80,)
        """
        return np.asarray(self.linear_.predict(X) + self.forest_.predict(X), dtype=float)


# ------------------------------------------------------------------- data generation


def unit_ar1(n: int, rho: float, rng: np.random.Generator) -> Floats:
    """Return an AR(1) series with unit marginal variance (the review's generator).

    Examples
    --------
    >>> unit_ar1(4, 0.5, np.random.default_rng(0)).shape
    (4,)
    """
    require(n >= 2 and abs(rho) < 1, f"expected n >= 2 and |rho| < 1, got {n}, {rho}")
    x = np.zeros(n)
    x[0] = rng.standard_normal()
    scale = np.sqrt(1 - rho**2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + scale * rng.standard_normal()
    return x


def review_data(seed: int, n: int, rho: float) -> Sample:
    """Draw T = X1 + u and Y = 2T + X2 + e, with independent AR(1) u and e.

    Examples
    --------
    >>> [a.shape for a in review_data(0, 5, 0.8)]
    [(5,), (5,), (5, 2)]
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    u, e = unit_ar1(n, rho, rng), unit_ar1(n, rho, rng)
    T = X[:, 0] + u
    return 2.0 * T + X[:, 1] + e, T, X


def review_complete(Y: Floats, T: Floats, X: Floats) -> Floats:
    """Return current X plus one lag of X, T and Y.

    Examples
    --------
    >>> review_complete(*review_data(0, 5, 0.8)).shape
    (4, 6)
    """
    return np.column_stack([X[1:], X[:-1], T[:-1], Y[:-1]])


def zero_start_ar1(n: int, rho: float, rng: np.random.Generator) -> Floats:
    """Return an AR(1) covariate with unit innovations that starts at zero.

    Examples
    --------
    >>> float(zero_start_ar1(3, 0.98, np.random.default_rng(0))[0])
    0.0
    """
    require(n >= 2 and abs(rho) < 1, f"expected n >= 2 and |rho| < 1, got {n}, {rho}")
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.standard_normal()
    return x


def covariate_data(seed: int, n: int, rho: float) -> Sample:
    """Draw T = 0.5x + noise and Y = 2T + x + noise, with AR(1) x.

    Examples
    --------
    >>> [a.shape for a in covariate_data(0, 5, 0.5)]
    [(5,), (5,), (5, 2)]
    """
    rng = np.random.default_rng(seed)
    x = zero_start_ar1(n, rho, rng)
    X = np.column_stack([x, rng.standard_normal(n)])
    T = 0.5 * x + rng.standard_normal(n)
    return 2.0 * T + x + rng.standard_normal(n), T, X


def repair_data(seed: int, n: int, rho: float, interaction: float) -> Sample:
    """Draw the linear design plus a threshold interaction in both equations.

    Examples
    --------
    >>> repair_data(0, 5, 0.5, 2.0)[2].shape
    (5, 3)
    """
    rng = np.random.default_rng(seed)
    x = zero_start_ar1(n, rho, rng)
    z = rng.standard_normal((n, 2))
    step = interaction * ((z[:, 0] > 0) & (z[:, 1] > 0))
    T = 0.5 * x + step + rng.standard_normal(n)
    Y = 2.0 * T + x + step + rng.standard_normal(n)
    return Y, T, np.column_stack([x, z])


def innovation_ar1(rng: np.random.Generator, shape: tuple[int, int], rho: float) -> Floats:
    """Return stationary AR(1) rows with unit innovations (the refutation generator).

    Examples
    --------
    >>> innovation_ar1(np.random.default_rng(0), (2, 5), 0.8).shape
    (2, 5)
    """
    require(abs(rho) < 1, f"expected |rho| < 1, got {rho}")
    z = rng.standard_normal(shape)
    if rho == 0:
        return z
    x = np.empty_like(z)
    x[:, 0] = z[:, 0] / np.sqrt(1 - rho**2)
    for t in range(1, shape[1]):
        x[:, t] = rho * x[:, t - 1] + z[:, t]
    return x


def feedback_outcomes(
    base: Floats, x2: Floats, latent: Floats, e: Floats, theta: float, phi: float
) -> tuple[Floats, Floats]:
    """Iterate T = phi Y(t-1) + base and Y = theta T + x2 + latent + e.

    Examples
    --------
    >>> one, zero = np.ones((1, 3)), np.zeros((1, 3))
    >>> feedback_outcomes(one, zero, zero, zero, 1.0, 0.5)[1].tolist()
    [[1.0, 1.5, 1.75]]
    """
    require(abs(phi * theta) < 1, f"explosive feedback: phi * theta = {phi * theta}")
    treated, outcome = np.empty(base.shape), np.empty(base.shape)
    previous = np.zeros(len(base))
    for t in range(base.shape[1]):
        treated[:, t] = phi * previous + base[:, t]
        outcome[:, t] = theta * treated[:, t] + x2[:, t] + latent[:, t] + e[:, t]
        previous = outcome[:, t]
    return treated, outcome


def stress_data(
    rng: np.random.Generator,
    reps: int,
    n: int,
    theta: float = 2.0,
    phi: float = 0.0,
    ru: float = 0.8,
    re: float = 0.8,
    rx: float = 0.0,
    g_t: float = 0.0,
    g_y: float = 0.0,
    rc: float = 0.8,
    noise_u: float = 0.0,
    noise_e: float = 0.0,
    burn: int = 100,
) -> dict[str, Any]:
    """Draw from the refutation generator, after a burn-in.

    T = phi Y(t-1) + X1 + g_t C + u and Y = theta T + X2 + g_y C + e. The AR(1) processes have
    unit innovations, ``noise_*`` adds white noise to an error, and C is latent.

    Examples
    --------
    >>> stress_data(np.random.default_rng(0), 2, 7, phi=0.3)["Y"].shape
    (2, 7)
    """
    require(reps >= 1 and n >= 2 and burn >= 0, f"invalid size: {reps}, {n}, {burn}")
    shape = (reps, n + burn)
    x1, x2, u, e = (innovation_ar1(rng, shape, rho) for rho in (rx, rx, ru, re))
    if noise_u:
        u = u + noise_u * rng.standard_normal(shape)
    if noise_e:
        e = e + noise_e * rng.standard_normal(shape)
    latent = innovation_ar1(rng, shape, rc) if (g_t or g_y) else np.zeros_like(u)
    base = x1 + g_t * latent + u
    if phi == 0:
        treated, outcome = base, theta * base + x2 + g_y * latent + e
    else:
        treated, outcome = feedback_outcomes(base, x2, g_y * latent, e, theta, phi)
    series = {"X1": x1, "X2": x2, "T": treated, "Y": outcome, "u": u, "e": e}
    return {name: value[:, burn:] for name, value in series.items()} | {"theta": theta}


def stress_controls(data: dict[str, Any], rep: int, kind: str, lags: int = 1) -> Sample:
    """Build one replicate's ``(Y, T, W)`` for a conditioning set.

    Examples
    --------
    >>> data = stress_data(np.random.default_rng(0), 1, 10)
    >>> stress_controls(data, 0, "full", 2)[2].shape
    (8, 10)
    """
    require(kind in {"current", "t_lag", "y_lag", "full"}, f"unknown conditioning set {kind!r}")
    x1, x2, T, Y = (data[name][rep] for name in ("X1", "X2", "T", "Y"))
    if kind == "current":
        return Y, T, np.stack([x1, x2], axis=1)
    n = len(Y)
    require(1 <= lags < n, f"expected 1 <= lags < {n}, got {lags}")
    lagged = {"t_lag": [T], "y_lag": [Y], "full": [Y, T, x1, x2]}[kind]
    columns = [x1[lags:], x2[lags:]]
    for k in range(1, lags + 1):
        columns += [value[lags - k : n - k] for value in lagged]
    return Y[lags:], T[lags:], np.stack(columns, axis=1)


def feedback_series(
    length: int, feedback: bool, seed: int
) -> tuple[DynamicTreatmentDGP, DynamicTreatmentDGPResult]:
    """Draw the package's series DGP with direct effects 1, 0.5 and 0.25.

    Examples
    --------
    >>> feedback_series(50, True, 0)[1].Y.shape
    (50,)
    """
    dgp = DynamicTreatmentDGP(
        n_periods=3,
        theta_t=np.array([1.0, 0.5, 0.25]),
        p=3,
        state_feedback=feedback,
        mode="series",
        series_length=length,
        random_state=seed,
    )
    return dgp, dgp.generate()


# ----------------------------------------------------------------------------- rows


def score_row(
    base: Row,
    theta: float,
    scores: Floats,
    truth: float = 2.0,
    long_lags: int | None = None,
    prewhiten: bool = False,
    acf: bool = False,
) -> Row:
    """Return one estimate with the standard errors that the checks compare.

    Examples
    --------
    >>> score_row({}, 2.5, np.sin(np.arange(30.0)))["error"]
    0.5
    """
    require(bool(np.isfinite(theta)), f"non-finite estimate in {base}")
    count = len(scores)
    row = base | {
        "truth": truth,
        "error": theta - truth,
        "n_scored": count,
        "se_lag0": newey_west_se(scores, bandwidth=0).se,
        "se_nw": newey_west_se(scores, bandwidth="auto").se,
        "se_cosine": np.sqrt(cosine_long_run_variance(scores)[0] / count),
    }
    if long_lags is not None:
        row |= {"long_lags": long_lags, "se_long": newey_west_se(scores, bandwidth=long_lags).se}
    if prewhiten:
        row["se_prewhite"] = newey_west_se(scores, prewhiten=True).se
    if acf:
        centred = scores - scores.mean()
        row["score_acf1"] = float(centred[1:] @ centred[:-1] / (centred @ centred))
    return row


def blip_rows(
    base: Row, truth: Floats, results: Mapping[str, tuple[Floats, Floats | None]]
) -> list[Row]:
    """Return one row per variant and lag.

    Examples
    --------
    >>> float(blip_rows({}, np.array([1.0]), {"v": (np.array([1.5]), None)})[0]["error"])
    0.5
    """
    rows = []
    for variant, (theta, se) in results.items():
        for k, target in enumerate(truth):
            row = base | {"variant": variant, "component": k}
            row |= {"truth": target, "error": theta[k] - target}
            rows.append(row if se is None else row | {"se_sandwich": se[k]})
    return rows


# --------------------------------------------------------------------------- designs

F3_DESIGNS: dict[str, tuple[dict[str, Any], str]] = {
    "A_complete_rho80": ({"theta": 2, "ru": 0.8, "re": 0.8}, "full"),
    "B_feedback_white": ({"theta": 1, "phi": 0.5, "ru": 0, "re": 0}, "current"),
    "B2_feedback_persistent_treatment": ({"theta": 1, "phi": 0.5, "ru": 0.8, "re": 0}, "current"),
    "C_feedback_confounded": ({"theta": 1, "phi": 0.5, "ru": 0, "re": 0.5}, "current"),
    "D_feedback_complete": ({"theta": 1, "phi": 0.5, "ru": 0.5, "re": 0.5}, "full"),
    "E_complete_rho95": ({"theta": 2, "ru": 0.95, "re": 0.95}, "full"),
    "E2_complete_rho95_x95": ({"theta": 2, "ru": 0.95, "re": 0.95, "rx": 0.95}, "full"),
    "E3_incomplete_rho95": ({"theta": 2, "ru": 0.95, "re": 0.95}, "current"),
}
F3_SCHEMES = [
    ("forward_gap0", "forward", 5, 0, False),
    ("forward_gap5", "forward", 5, 5, False),
    ("forward_second_half", "half", 5, 0, False),
    ("two_sided_gap0", "two_sided", 5, 0, False),
    ("two_sided_gap2", "two_sided", 5, 2, False),
    ("two_sided_gap2_forward_rows", "two_sided", 5, 2, True),
    ("neighbours_out_k5", "neighbours_out", 5, 0, False),
    ("neighbours_out_k10", "neighbours_out", 10, 0, False),
]
F2_DESIGNS: dict[str, tuple[dict[str, Any], list[tuple[str, int]]]] = {
    "both_persistent": (
        {"theta": 2, "ru": 0.8, "re": 0.8},
        [("current", 1), ("t_lag", 1), ("full", 1)],
    ),
    "treatment_residual_persistent": (
        {"theta": 2, "ru": 0.8, "re": 0},
        [("current", 1), ("t_lag", 1), ("full", 1)],
    ),
    "outcome_error_persistent": (
        {"theta": 2, "ru": 0, "re": 0.8},
        [("current", 1), ("full", 1)],
    ),
    "latent_confounder": (
        {"theta": 2, "ru": 0, "re": 0, "g_t": 0.5, "g_y": 0.5},
        [("current", 1), ("full", 1), ("full", 4)],
    ),
    "arma_errors": (
        {"theta": 2, "ru": 0.8, "re": 0.8, "noise_u": 1.5, "noise_e": 1.5},
        [("current", 1), ("full", 1), ("full", 4)],
    ),
    "feedback_white": (
        {"theta": 1, "phi": 0.5, "ru": 0, "re": 0},
        [("current", 1), ("y_lag", 1), ("full", 1)],
    ),
    "feedback_persistent_error": (
        {"theta": 1, "phi": 0.5, "ru": 0, "re": 0.8},
        [("current", 1), ("y_lag", 1), ("full", 1)],
    ),
}
F4_PAIRS = [(0.0, 0.0), (0.95, 0.0), (0.5, 0.5), (0.8, 0.8), (0.95, 0.95)]
TREE_DESIGNS: dict[str, tuple[dict[str, Any], str]] = {
    "T1_feedback_default_controls": ({"theta": 1, "phi": 0.8, "ru": 0, "re": 0}, "t_lag"),
    "T2_feedback_complete": ({"theta": 1, "phi": 0.5, "ru": 0.8, "re": 0.8}, "full"),
    "T3_complete_rho95": ({"theta": 2, "ru": 0.95, "re": 0.95}, "full"),
    "T4_feedback_incomplete": (
        {"theta": 1, "phi": 0.5, "ru": 0.8, "re": 0.8, "noise_u": 1.5, "noise_e": 1.5},
        "y_lag",
    ),
}
TREE_SCHEMES = [
    ("forward_gap0", "forward", 0),
    ("forward_gap5", "forward", 5),
    ("forward_second_half", "half", 0),
    ("two_sided_gap0", "two_sided", 0),
    ("two_sided_gap2", "two_sided", 2),
    ("two_sided_gap10", "two_sided", 10),
    ("neighbours_out_k5", "neighbours_out", 0),
]
REPAIR_DESIGNS = {
    "A_persistent_linear": (0.98, 0.0),
    "B_stationary_interaction": (0.5, 2.0),
    "C_persistent_and_interaction": (0.98, 2.0),
}


# --------------------------------------------------------------------------- studies


def f1_single(cell: tuple[bool], reps: range) -> list[Row]:
    """F1: one long series, seed 0, with and without treatment-to-state feedback.

    Examples
    --------
    >>> len(f1_single((False,), range(1)))
    9
    """
    (feedback,) = cell
    dgp, data = feedback_series(20000, feedback, 0)
    fit = DynamicGEstimationDML(
        n_periods=3, model_y="linear", model_t="linear", n_folds=5, random_state=0
    ).fit(data.Y, data.T, data.X, mode="series")
    recursion = own_date_recursion(data.Y, data.T, data.X, 3, np.arange(2, 20000))
    results = {
        "dgp_declared_truth": (data.theta_t, None),
        "repo_series": (fit.theta_t, fit.se_t),
        "own_date_in_sample": (recursion[0], None),
    }
    base = {"design": f"feedback={feedback}", "n": 20000, "replicate": reps[0]}
    return blip_rows(base, series_total_blips(dgp), results)


def recursion_prototypes(
    data: DynamicTreatmentDGPResult,
) -> dict[str, tuple[Floats, Floats | None]]:
    """Run the four prototypes: disjoint or overlapping blocks, with or without history.

    Examples
    --------
    >>> len(recursion_prototypes(feedback_series(600, True, 0)[1]))
    4
    """
    Y, T, X, m = data.Y, data.T, data.X, 3
    disjoint = np.arange(2 * (m - 1), len(Y), m)
    overlapping = np.arange(2 * (m - 1), len(Y))
    progressive = make_splits("half", len(disjoint))
    forward = make_splits("forward", len(overlapping), gap=m)
    results: dict[str, tuple[Floats, Floats | None]] = {}
    for history in (False, True):
        suffix = "_history" if history else ""
        results[f"disjoint_progressive{suffix}"] = own_date_recursion(
            Y, T, X, m, disjoint, progressive, history=history
        )
        results[f"overlapping_hac{suffix}"] = own_date_recursion(
            Y, T, X, m, overlapping, forward, history=history, hac=True
        )
    return results


def f1_coverage(cell: tuple[int], reps: range) -> list[Row]:
    """F1: coverage of the total blips by the repo mode and four recursion prototypes.

    Examples
    --------
    >>> len(f1_coverage((3000,), range(1)))
    15
    """
    (length,) = cell
    rows = []
    for r in reps:
        dgp, data = feedback_series(length, True, 1000 + r)
        fit = DynamicGEstimationDML(n_periods=3, model_y="linear", model_t="linear").fit(
            data.Y, data.T, data.X, mode="series"
        )
        results = {"repo_series": (fit.theta_t, fit.se_t)} | recursion_prototypes(data)
        base = {"design": "feedback=True", "n": length, "replicate": r}
        rows += blip_rows(base, series_total_blips(dgp), results)
    return rows


def review_f2_f4(cell: tuple[int, float], reps: range) -> list[Row]:
    """F2 and F4: current controls against one lag of Y, T and X.

    Examples
    --------
    >>> [row["variant"] for row in review_f2_f4((200, 0.8), range(1))]
    ['current', 'complete']
    """
    n, rho = cell
    rows = []
    for r in reps:
        Y, T, X = review_data(10000 + r, n, rho)
        for variant, W in (("current", X[1:]), ("complete", review_complete(Y, T, X))):
            fit = TemporalPLRDML(n_lags=0, model_y="ridge", model_t="ridge", n_splits=5).fit(
                Y[1:], T[1:], W
            )
            scores = fit.influence_scores
            base = {"design": f"rho={rho}", "n": n, "variant": variant, "replicate": r}
            # The exploratory run used ceil(1.3 sqrt(n)) lags, one more than the truncation rule.
            lags = int(np.ceil(1.3 * np.sqrt(len(scores))))
            rows.append(
                score_row(base, fit.theta, scores, long_lags=lags, prewhiten=True, acf=True)
            )
    return rows


def review_f3(cell: tuple[int], reps: range) -> list[Row]:
    """F3: forward against two-sided folds in the lag-completed design.

    Examples
    --------
    >>> len(review_f3((120,), range(1)))
    3
    """
    (n,) = cell
    ridge = get_nuisance_model("ridge")
    rows = []
    for r in reps:
        Y, T, X = review_data(50000 + r, n, 0.8)
        W = review_complete(Y, T, X)
        Y, T = Y[1:], T[1:]
        base = {"design": "complete_rho80", "n": n, "replicate": r}
        fit = TemporalPLRDML(n_lags=0, model_y="ridge", model_t="ridge", n_splits=5).fit(Y, T, W)
        rows.append(score_row(base | {"variant": "forward_gap0"}, fit.theta, fit.influence_scores))
        for variant, k, gap in (
            ("two_sided_gap2", 5, 2),
            ("two_sided_k10_gap_block", 10, len(Y) // 10),
        ):
            result = cross_fit_plr(Y, T, W, buffered_two_sided_splits(len(Y), k, gap), ridge)
            rows.append(score_row(base | {"variant": variant}, result.theta, result.scores))
    return rows


def refute_f3(cell: tuple[str, int], reps: range) -> list[Row]:
    """The attempt to refute F3 on eight designs with ridge nuisances.

    Examples
    --------
    >>> len(refute_f3(("B_feedback_white", 120), range(1)))
    8
    """
    design, n = cell
    params, kind = F3_DESIGNS[design]
    seed = 1000 * list(F3_DESIGNS).index(design) + n
    data = stress_data(np.random.default_rng(seed), 1000, n, **params)
    ridge = get_nuisance_model("ridge")
    size = n - (kind != "current")
    folds = {name: make_splits(s, size, k, gap) for name, s, k, gap, _ in F3_SCHEMES}
    rows = []
    for r in reps:
        Y, T, W = stress_controls(data, r, kind)
        for variant, _, _, _, restrict in F3_SCHEMES:
            first = int(folds["forward_gap0"][0][1][0]) if restrict else 0
            fit = cross_fit_plr(Y, T, W, folds[variant], ridge, first)
            base = {"design": design, "n": n, "variant": variant, "replicate": r}
            rows.append(score_row(base, fit.theta, fit.scores, data["theta"]))
    return rows


def refute_f2(cell: tuple[str, int], reps: range) -> list[Row]:
    """The attempt to refute F2 on seven designs and several conditioning sets.

    Examples
    --------
    >>> len(refute_f2(("outcome_error_persistent", 200), range(1)))
    4
    """
    design, n = cell
    params, conditioning = F2_DESIGNS[design]
    seed = 77000 + 10 * list(F2_DESIGNS).index(design) + n
    data = stress_data(np.random.default_rng(seed), 1000, n, **params)
    ridge = get_nuisance_model("ridge")
    rows = []
    for kind, lags in conditioning:
        size = n if kind == "current" else n - lags
        label = kind if kind == "current" else f"{kind}{lags}"
        folds = {
            "forward_gap0": make_splits("forward", size),
            "two_sided_gap2": make_splits("two_sided", size, gap=2),
        }
        for r in reps:
            Y, T, W = stress_controls(data, r, kind, lags)
            for scheme, splits in folds.items():
                fit = cross_fit_plr(Y, T, W, splits, ridge)
                base = {"design": design, "n": n, "variant": f"{label}/{scheme}", "replicate": r}
                rows.append(score_row(base, fit.theta, fit.scores, data["theta"], acf=True))
    return rows


def oracle_fit(u: Floats, e: Floats, used: NDArray[np.bool_]) -> Fit:
    """Return the fit that knows both nuisance functions, on the same rows.

    Examples
    --------
    >>> u = np.array([1.0, -1.0])
    >>> oracle_fit(u, np.zeros(2), np.ones(2, dtype=bool)).theta
    2.0
    """
    u, e = u[used], e[used]
    shift = float(u @ e / (u @ u))
    return Fit(2.0 + shift, (e - shift * u) * u / np.mean(u**2), used)


def refute_f4(cell: tuple[int, int], reps: range) -> list[Row]:
    """The attempt to refute F4, with fitted and oracle nuisances.

    Examples
    --------
    >>> [row["variant"] for row in refute_f4((0, 100), range(1))]
    ['fitted', 'oracle']
    """
    index, n = cell
    ru, re = F4_PAIRS[index]
    data = stress_data(np.random.default_rng(99000 + 10 * index + n), 2000, n, ru=ru, re=re)
    ridge = get_nuisance_model("ridge")
    folds = make_splits("forward", n)
    rows = []
    for r in reps:
        Y, T, W = stress_controls(data, r, "current")
        fitted = cross_fit_plr(Y, T, W, folds, ridge)
        oracle = oracle_fit(data["u"][r], data["e"][r], fitted.used)
        base = {"design": f"rho_u={ru},rho_e={re}", "n": n, "replicate": r}
        lags = int(np.ceil(1.3 * np.sqrt(len(fitted.scores)))) - 1
        for variant, fit in (("fitted", fitted), ("oracle", oracle)):
            row = base | {"variant": variant}
            rows.append(score_row(row, fit.theta, fit.scores, long_lags=lags, prewhiten=True))
    return rows


def refute_f3_trees(cell: tuple[str, int, ModelType], reps: range) -> list[Row]:
    """The attempt to refute F3 with the package's untuned tree learners.

    Examples
    --------
    >>> len(refute_f3_trees(("T1_feedback_default_controls", 200, "gradient_boosting"), range(1)))
    7
    """
    design, n, learner = cell
    params, kind = TREE_DESIGNS[design]
    index = list(TREE_DESIGNS).index(design)
    model = get_nuisance_model(learner, 0)
    rows = []
    for r in reps:
        data = stress_data(np.random.default_rng([index, n, r]), 1, n, **params)
        Y, T, W = stress_controls(data, 0, kind)
        for variant, scheme, gap in TREE_SCHEMES:
            fit = cross_fit_plr(Y, T, W, make_splits(scheme, len(Y), 5, gap), model)
            base = {"design": f"{design}/{learner}", "n": n, "variant": variant, "replicate": r}
            rows.append(score_row(base, fit.theta, fit.scores, data["theta"]))
    return rows


def default_forest_rows(base: Row, sample: Sample) -> list[Row]:
    """Fit the default forest; record its R-squared warning and its centred final stage.

    Examples
    --------
    >>> [r["variant"] for r in default_forest_rows({}, covariate_data(0, 150, 0.5))]
    ['default', 'default_centred_stage']
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = TemporalPLRDML(
            n_lags=0, model_y="random_forest", model_t="random_forest", n_splits=5, random_state=0
        ).fit(*sample)
    flagged = any("cross-validated R" in str(w.message) for w in caught)
    for w in caught:
        warnings.warn(w.message, stacklevel=2)
    first = score_row(base | {"variant": "default"}, fit.theta, fit.influence_scores)
    theta, scores = residual_slope(
        fit.Y_residual - fit.Y_residual.mean(), fit.T_residual - fit.T_residual.mean()
    )
    second = score_row(base | {"variant": "default_centred_stage"}, theta, scores)
    return [first | {"r2_warning": int(flagged)}, second]


def f5_default_forest(cell: tuple[float], reps: range) -> list[Row]:
    """F5: the default configuration on a linear design.

    Examples
    --------
    >>> len(f5_default_forest((0.5,), range(1)))
    4
    """
    (rho,) = cell
    n = 500
    folds = buffered_two_sided_splits(n, 5, 5)
    rows = []
    for r in reps:
        Y, T, X = covariate_data(900 + r, n, rho)
        base = {"design": f"covariate_ar={rho}", "n": n, "replicate": r}
        rows += default_forest_rows(base, (Y, T, X))
        two = cross_fit_plr(Y, T, X, folds, get_nuisance_model("random_forest", 0))
        rows.append(score_row(base | {"variant": "forest_two_sided"}, two.theta, two.scores))
        ridge = TemporalPLRDML(n_lags=0, model_y="ridge", model_t="ridge", n_splits=5).fit(Y, T, X)
        rows.append(score_row(base | {"variant": "ridge"}, ridge.theta, ridge.influence_scores))
    return rows


def f5_repair(cell: tuple[str], reps: range) -> list[Row]:
    """F5: which learner repairs the default's bias.

    Examples
    --------
    >>> len(f5_repair(("A_persistent_linear",), range(1)))
    6
    """
    (design,) = cell
    rho, interaction = REPAIR_DESIGNS[design]
    learners = {
        "ridge_standardized": scaled_ridge(),
        "forest_50_trees": fixed_forest(50),
        "forest_100_trees_default": get_nuisance_model("random_forest", 0),
        "stack_ridge_forest": ForwardStack(),
        "ridge_then_forest_50": LinearThenForest(50),
        "ridge_then_forest_100": LinearThenForest(100),
    }
    folds = make_splits("forward", 500)
    rows = []
    for r in reps:
        Y, T, X = repair_data(7000 + r, 500, rho, interaction)
        for variant, model in learners.items():
            fit = cross_fit_plr(Y, T, X, folds, model)
            base = {"design": design, "n": 500, "variant": variant, "replicate": r}
            rows.append(score_row(base, fit.theta, fit.scores))
    return rows


def f5_units(cell: tuple[ModelType, float], reps: range) -> list[Row]:
    """F5: the same data with the covariates expressed in other units.

    Examples
    --------
    >>> f5_units(("ridge", 0.01), range(1))[0]["design"]
    'units_times=0.01'
    """
    learner, scale = cell
    rows = []
    for r in reps:
        Y, T, X = covariate_data(900 + r, 500, 0.5)
        fit = TemporalPLRDML(n_lags=0, model_y=learner, model_t=learner).fit(Y, T, X * scale)
        base = {"design": f"units_times={scale}", "n": 500, "variant": learner, "replicate": r}
        rows.append(score_row(base, fit.theta, fit.influence_scores))
    return rows


class Study(NamedTuple):
    """A runner, its cells, the replications per cell and the replications per task.

    Examples
    --------
    >>> STUDIES["f5_repair"].replications
    60
    """

    run: Callable[[Any, range], list[Row]]
    cells: list[tuple[Any, ...]]
    replications: int
    chunk: int


STUDIES = {
    "f1_single": Study(f1_single, [(False,), (True,)], 1, 1),
    "f1_coverage": Study(f1_coverage, [(3000,)], 200, 10),
    "review_f2_f4": Study(review_f2_f4, [(200, 0.8), (500, 0.8), (500, 0.6)], 1000, 50),
    "review_f3": Study(review_f3, [(120,), (200,), (500,)], 1000, 50),
    "refute_f3": Study(refute_f3, [(d, n) for d in F3_DESIGNS for n in (120, 200, 500)], 1000, 50),
    "refute_f2": Study(refute_f2, [(d, n) for d in F2_DESIGNS for n in (200, 500)], 1000, 100),
    "refute_f4": Study(
        refute_f4, [(i, n) for i in range(len(F4_PAIRS)) for n in (100, 200, 500)], 2000, 200
    ),
    "refute_f3_trees": Study(
        refute_f3_trees,
        [(design, 200, "gradient_boosting") for design in TREE_DESIGNS]
        + [
            ("T1_feedback_default_controls", 400, "gradient_boosting"),
            ("T2_feedback_complete", 400, "gradient_boosting"),
            ("T2_feedback_complete", 200, "random_forest"),
        ],
        384,
        4,
    ),
    "f5_default_forest": Study(f5_default_forest, [(0.98,), (0.5,)], 120, 4),
    "f5_repair": Study(f5_repair, [(design,) for design in REPAIR_DESIGNS], 60, 3),
    "f5_units": Study(
        f5_units, [(m, scale) for m in ("ridge", "lasso") for scale in (1.0, 0.01)], 120, 20
    ),
}


def run_cell(
    study: str, cell: tuple[Any, ...], start: int, stop: int
) -> tuple[list[Row], Counter[str]]:
    """Run replications ``start`` to ``stop - 1`` of one cell, counting every warning.

    Examples
    --------
    >>> rows, warned = run_cell("review_f3", (120,), 0, 1)
    >>> len(rows), rows[0]["study"]
    (3, 'review_f3')
    """
    require(study in STUDIES, f"unknown study {study!r}")
    spec = STUDIES[study]
    require(cell in spec.cells, f"{cell} is not a cell of {study}")
    require(0 <= start < stop <= spec.replications, f"bad replication range {start}:{stop}")
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        rows = spec.run(cell, range(start, stop))
    counts = Counter(f"{w.category.__name__}: {str(w.message)[:110]}" for w in caught)
    return [{"component": 0} | row | {"study": study} for row in rows], counts


def run_studies(
    studies: Mapping[str, Study], pool: Executor
) -> tuple[list[Row], dict[str, dict[str, int]]]:
    """Run the studies in a pool and report every warning.

    ``map`` preserves task order, so the output does not depend on the number of workers.

    Examples
    --------
    >>> from concurrent.futures import ThreadPoolExecutor
    >>> with ThreadPoolExecutor(1) as pool:
    ...     rows, warned = run_studies({"f1_single": STUDIES["f1_single"]}, pool)
    f1_single: done
    >>> len(rows)
    18
    """
    rows: list[Row] = []
    warned: dict[str, dict[str, int]] = {}
    for name, study in studies.items():
        tasks = [
            (name, cell, start, min(start + study.chunk, study.replications))
            for cell in study.cells
            for start in range(0, study.replications, study.chunk)
        ]
        counts: Counter[str] = Counter()
        for part, seen in pool.map(run_cell, *zip(*tasks, strict=True)):
            rows.extend(part)
            counts.update(seen)
        warned[name] = dict(sorted(counts.items()))
        print(f"{name}: done", flush=True)
        for message, count in warned[name].items():
            print(f"  warning x{count}: {message}", flush=True)
    return rows, warned


# ------------------------------------------------------------------------- reporting


def critical_values(group: pd.DataFrame) -> dict[str, float]:
    """Return each interval's critical value in one cell.

    Examples
    --------
    >>> group = pd.DataFrame({"n_scored": [np.nan], "long_lags": [np.nan]})
    >>> sorted(critical_values(group))
    ['lag0', 'nw', 'prewhite', 'sandwich']
    """
    critical = {"lag0": Z, "nw": Z, "prewhite": Z, "sandwich": Z}
    if group.n_scored.notna().all():
        scored = int(group.n_scored.iloc[0])
        critical["cosine"] = float(stats.t.ppf(0.975, cosine_dof(scored)))
        if group.long_lags.notna().all():
            b = (group.long_lags.iloc[0] + 1) / scored
            critical["long"] = float(fixed_b_critical_value(b))
    return critical


def interval_summary(group: pd.DataFrame, sd: float) -> Row:
    """Return coverage and width for every interval recorded in one cell.

    Examples
    --------
    >>> group = pd.DataFrame({c: [np.nan, np.nan] for c in COLUMNS})
    >>> group["error"], group["se_sandwich"] = [0.0, 1.0], [0.1, 0.1]
    >>> interval_summary(group, 0.5)["coverage_sandwich"]
    0.5
    """
    error = group.error.to_numpy()
    default = Z * group.se_nw.mean() if group.se_nw.notna().all() else np.nan
    out: Row = {}
    for name, critical in critical_values(group).items():
        se = group[f"se_{name}"]
        if se.isna().any():
            continue
        half = critical * se.to_numpy()
        covered = float(np.mean(np.abs(error) <= half))
        out |= {
            f"mean_se_{name}": se.mean(),
            f"se_sd_{name}": se.mean() / sd,
            f"coverage_{name}": covered,
            f"coverage_mcse_{name}": np.sqrt(covered * (1 - covered) / len(group)),
            f"width_{name}": half.mean() / (Z * sd),
            f"relative_width_{name}": half.mean() / default,
        }
    return out


def cell_summary(group: pd.DataFrame) -> Row:
    """Return bias, spread, intervals and diagnostics for one cell.

    Examples
    --------
    >>> group = pd.DataFrame({c: [np.nan, np.nan] for c in COLUMNS})
    >>> group["truth"], group["error"] = 2.0, [0.0, 1.0]
    >>> float(cell_summary(group)["bias"])
    0.5
    """
    error = group.error.to_numpy()
    reps = len(group)
    truth, bias = group.truth.iloc[0], error.mean()
    out: Row = {"replications": reps, "truth": truth, "bias": bias, "estimate": truth + bias}
    out["rmse"] = np.sqrt(np.mean(error**2))
    sd = error.std(ddof=1) if reps > 1 else np.nan
    out |= {"sd": sd, "bias_mcse": sd / np.sqrt(reps)}
    if group.n_scored.notna().all():
        out["n_scored"] = int(group.n_scored.iloc[0])
        if group.long_lags.notna().all():
            out["b"] = (group.long_lags.iloc[0] + 1) / out["n_scored"]
    out |= interval_summary(group, sd)
    for column in ("score_acf1", "r2_warning"):
        if group[column].notna().all():
            out[column] = group[column].mean()
    return out


def summarize(draws: pd.DataFrame) -> pd.DataFrame:
    """Return one row per cell.

    ``width_x`` is the mean half-width over 1.96 empirical SD. ``relative_width_x`` is the mean
    half-width over that of the default interval.

    Examples
    --------
    >>> rows, _ = run_cell("review_f3", (120,), 0, 2)
    >>> len(summarize(pd.DataFrame(rows).reindex(columns=COLUMNS)))
    3
    """
    missing = set(COLUMNS) - set(draws.columns)
    require(not draws.empty and not missing, f"empty draws or missing columns {sorted(missing)}")
    keys = ["study", "design", "n", "variant", "component"]
    rows = [
        dict(zip(keys, key, strict=True)) | cell_summary(group)
        for key, group in draws.groupby(keys, sort=False)
    ]
    return pd.DataFrame(rows)


def rejection(call: Callable[[], object], expected: type[Exception]) -> str:
    """Say whether a call is accepted or raises the expected error.

    Examples
    --------
    >>> rejection(lambda: int("x"), ValueError).split(":")[0]
    'ValueError'
    """
    try:
        call()
    except expected as error:
        return f"{type(error).__name__}: {error}"
    return "accepted"


def chapter5_attainable_r2() -> tuple[float, float]:
    """Return population R-squared for T and Y given the controls in chapter 5's listing.

    Var(T) = (0.25 + 1) / (1 - 0.09), Var(x^2) = 2 and Cov(x, x^2) = 0.

    Examples
    --------
    >>> [round(value, 3) for value in chapter5_attainable_r2()]
    [0.272, 0.411]
    """
    var_t = (0.5**2 + 1) / (1 - 0.3**2)
    var_y = 2**2 * var_t + 2 + 1
    return 1 - 1 / var_t, 1 - (2**2 + 1) / var_y


def chapter5_listing() -> Row:
    """Fit chapter 5's listing, drawn in the listing's order.

    Examples
    --------
    >>> bool(chapter5_listing()["treatment_r2"] < 0.2)
    True
    """
    rng, n = np.random.RandomState(42), 500
    X = np.column_stack([rng.randn(n), np.sin(2 * np.pi * np.arange(n) / 100)])
    T = np.zeros(n)
    T[0] = rng.randn()
    for t in range(1, n):
        T[t] = 0.3 * T[t - 1] + 0.5 * X[t, 0] + rng.randn()
    Y = 2.0 * T + X[:, 0] ** 2 + rng.randn(n)
    fit = TemporalPLRDML(
        n_lags=3, model_y="random_forest", model_t="random_forest", random_state=42
    ).fit(Y, T, X)
    treatment, outcome = chapter5_attainable_r2()
    return {
        "theta": fit.theta,
        "treatment_r2": fit.treatment_r2_cv,
        "outcome_r2": fit.outcome_r2_cv,
        "attainable_treatment_r2": treatment,
        "attainable_outcome_r2": outcome,
    }


def sensitivity_relabelling() -> list[Row]:
    """Return the package's critical Gamma beside exp(|z| - 1.96), at two sample sizes.

    The package searches a grid of step 0.1, so agreement is to within one step.

    Examples
    --------
    >>> all(abs(r["gamma"] - r["exp_z_minus_1.96"]) <= 0.1 for r in sensitivity_relabelling())
    True
    """
    rows = []
    for z in (2.2, 2.5, 2.8):
        for n in (200, 20000):
            result = compute_sensitivity_for_dml(theta=z, se=1.0, n_samples=n, treatment_r2=0.3)
            rows.append(
                {
                    "z": z,
                    "n": n,
                    "gamma": round(float(result.gamma_critical), 4),
                    "exp_z_minus_1.96": float(np.exp(z - Z)),
                }
            )
    return rows


def blocked_cv_separation() -> list[Row]:
    """Return the rows that ``blocked_cv`` actually leaves between training and test.

    The package turns the requested gap in rows into a number of blocks, so the two differ.

    Examples
    --------
    >>> blocked_cv_separation()[0]
    {'n': 300, 'requested_gap': 0, 'rows_left_out': 25}
    """
    rows = []
    for n in (300, 600):
        for gap in (0, 1, 5, 10, 19, 20, 50):
            splitter = TemporalPLRDML(cv_strategy="blocked_cv", gap=gap)._create_cv(n)
            train, test = next(iter(splitter.split(np.zeros((n, 1)))))
            separation = int(test.min() - train.max() - 1)
            rows.append({"n": n, "requested_gap": gap, "rows_left_out": separation})
    return rows


def deterministic_checks() -> Row:
    """Return the quoted facts that need a run but no Monte Carlo study.

    Examples
    --------
    >>> deterministic_checks()["hac_bandwidth_auto"].split(":")[0]
    'ValueError'
    """
    two_sided = PurgedKFold(n_splits=5, purge_gap=2, embargo_pct=0.0)
    return {
        "hac_bandwidth_auto": rejection(
            lambda: TemporalPLRDML(hac_bandwidth="auto"),  # type: ignore[arg-type]
            ValueError,
        ),
        "purged_kfold_forward_only_contract": rejection(
            lambda: check_temporal_splitter(two_sided), AssertionError
        ),
        "chapter5_listing": chapter5_listing(),
        "sensitivity_relabelling": sensitivity_relabelling(),
        "blocked_cv_separation": blocked_cv_separation(),
        "probability_70pct_method_passes_20_rep_floor_80pct": float(stats.binom.sf(15, 20, 0.70)),
    }


def git_output(*args: str) -> str:
    """Return the output of a read-only git command in this repository.

    Examples
    --------
    >>> len(git_output("rev-parse", "HEAD"))
    40
    """
    command = ["git", "--no-optional-locks", "-C", str(HERE), *args]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def sha256(path: Path) -> str:
    """Return a file's SHA-256.

    Examples
    --------
    >>> len(sha256(Path(__file__)))
    64
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance() -> Row:
    """Return versions and hashes, without machine-specific paths.

    Examples
    --------
    >>> "checks.py" in provenance()["source_sha256"]
    True
    """
    package = Path(dml_ts.__file__).resolve().parent
    sources = [
        "dml/_utils.py",
        "dml/temporal_plr_dml.py",
        "dml/dynamic_g_estimation.py",
        "sensitivity/rosenbaum.py",
        "validation/dynamic_dgp.py",
    ]
    hashes = {f"dml_ts/{name}": sha256(package / name) for name in sources}
    hashes["checks.py"] = sha256(Path(__file__))
    packages = ["numpy", "scipy", "scikit-learn", "pandas", "temporalcv"]
    return {
        "python": platform.python_version(),
        "packages": {name: version(name) for name in packages},
        "repository_head": git_output("rev-parse", "HEAD"),
        "dml_ts_has_uncommitted_changes": bool(
            git_output("status", "--porcelain", "--", ":/dml_ts")
        ),
        "source_sha256": hashes,
    }


def write_draws(rows: Sequence[Row], path: Path) -> None:
    """Write byte-reproducible gzip: no timestamp or file name, six significant digits.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     write_draws([{"study": "s", "error": 0.5}], Path(folder) / "d.gz")
    ...     float(pd.read_csv(Path(folder) / "d.gz", compression="gzip").error[0])
    0.5
    """
    require(len(rows) > 0, "no draws to write")
    raw = pd.DataFrame(rows)
    # reindex would otherwise drop an unexpected column without a message.
    unknown = set(raw.columns) - set(COLUMNS)
    require(not unknown, f"unknown draw columns {sorted(unknown)}")
    frame = raw.reindex(columns=COLUMNS)
    for column in ("n_scored", "long_lags", "r2_warning"):
        frame[column] = frame[column].astype("Int64")
    with path.open("wb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                frame.to_csv(stream, index=False, float_format="%.6g", lineterminator="\n")


def write_outputs(output: Path, rows: Sequence[Row], record: Row) -> None:
    """Write the draws, the summary computed from them, and the run record.

    Examples
    --------
    >>> import tempfile
    >>> rows, _ = run_cell("f1_single", (True,), 0, 1)
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     write_outputs(Path(folder), rows, {})
    ...     sorted(path.name for path in Path(folder).iterdir())
    ['draws.csv.gz', 'run.json', 'summary.csv']
    """
    write_draws(rows, output / "draws.csv.gz")
    summary = summarize(pd.read_csv(output / "draws.csv.gz"))
    summary.to_csv(output / "summary.csv", index=False, float_format="%.6g")
    outputs = {name: sha256(output / name) for name in ("draws.csv.gz", "summary.csv")}
    full = record | {"rows": len(rows), "outputs_sha256": outputs}
    (output / "run.json").write_text(json.dumps(full, indent=2) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate the command line.

    Examples
    --------
    >>> parse_args(["--workers", "3"]).workers
    3
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Run every study and write the three output files.

    Examples
    --------
    >>> main(["--help"])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    SystemExit: 0
    """
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    record = provenance()  # before the run, so the hashes are of the code that ran
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows, warned = run_studies(STUDIES, pool)
        simulated = list(pool.map(fixed_b_simulation, range(40)))
    checks = deterministic_checks()
    checks["fixed_b_critical_values"] = fixed_b_table(simulated)
    record |= {
        "workers": args.workers,
        "seconds": round(time.monotonic() - started, 1),
        "replications": {name: study.replications for name, study in STUDIES.items()},
        "checks": checks,
        "warnings": warned,
    }
    write_outputs(args.output, rows, record)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
