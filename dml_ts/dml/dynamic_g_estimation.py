"""Recursive dynamic g-estimation (Lewis & Syrgkanis, 2021).

Implements ``DynamicGEstimationDML`` -- a Neyman-orthogonal, cross-fitted
g-estimator for *linear structural nested mean models* that recovers
**period-specific blips** ``theta = (theta_1, ..., theta_m)`` over an m-period
horizon. This is the recursive dynamic treatment-effect method that
``TemporalPLRDML`` is explicitly **not** (the latter returns a single scalar
effect after controlling for lagged treatment).

Reference:

- Lewis, G. and Syrgkanis, V. (2021). *Double/Debiased Machine Learning for
  Dynamic Treatment Effects via g-Estimation.* NeurIPS. arXiv:2002.07285.

Algorithm (panel of n i.i.d. trajectories x m periods), recursive backward
peeling -- a dynamic version of Robinson's partial linear model::

    for tau = m-1 ... 0:
        H_tau = history available before T_tau  (states X_0..tau, treatments T_0..tau-1)
        cross-fit (over units) the residual-maker M_tau, residualize on H_tau:
            R_tau = M_tau Y_final          (residualized final outcome)
            T~_s  = M_tau T_s   for all s  (residualized treatments)
        peeled residual outcome:  Y~^(tau) = R_tau - sum_{s>tau} theta_s T~_s
        orthogonal moment:        theta_tau = <T~_tau, Y~^(tau)> / <T~_tau, T~_tau>

Inference is the joint M-estimation sandwich over the stacked moments
``g_{i,tau} = T~_tau,i (R_tau,i - sum_{s>=tau} theta_s T~_s,i)``, whose Jacobian
is upper-triangular -- this correctly propagates the cross-stage coupling that
Lewis & Syrgkanis note breaks per-stage Neyman orthogonality. Clustered by unit
(panel) or HAC/Newey-West (single series).

``mode="series"`` is the single-unit, stationarity-based adaptation (also in
Lewis & Syrgkanis): a stationary distributed-lag g-estimation
``Y_t = sum_k theta_{k+1} T_{t-k} + g(X_t) + u_t`` with HAC inference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold

from .cross_fitting import TimeSeriesCrossValidator
from .hac import optimal_bandwidth

ArrayLike = Union[np.ndarray, List[float]]
ModelSpec = Union[str, BaseEstimator]
Mode = Literal["panel", "series"]

_DEFAULT_N_JOBS = int(os.environ.get("DML_N_JOBS", "-1"))


@dataclass
class DynamicGEstimationResult:
    """Result of recursive dynamic g-estimation.

    Attributes:
        theta_t: Per-period blips ``(m,)`` -- ``theta_t[k]`` is the effect of a
            unit blip in the period-``k`` treatment on the final-period outcome
            (panel) or the lag-``k`` distributed-lag effect (series).
        se_t: Standard errors ``(m,)`` from the joint sandwich (panel) or HAC
            (series).
        ci_lower_t: Lower confidence bounds ``(m,)``.
        ci_upper_t: Upper confidence bounds ``(m,)``.
        p_value_t: Two-sided p-values ``(m,)`` for ``H0: theta_k = 0``.
        cumulative_effect: ``sum(theta_t)``.
        cumulative_se: Standard error of the cumulative effect.
        n_units: Number of trajectories (1 for the series).
        n_periods: Horizon m.
        alpha: Significance level used for the intervals.
        cov: Joint covariance matrix ``(m, m)`` of ``theta_t``.
        nuisance_r2_t: Out-of-fold R^2 of the outcome nuisance at each stage.
        backend: ``"custom"`` (this estimator) or ``"econml"`` (reference).
        mode: ``"panel"`` or ``"series"``.
    """

    theta_t: NDArray[np.float64]
    se_t: NDArray[np.float64]
    ci_lower_t: NDArray[np.float64]
    ci_upper_t: NDArray[np.float64]
    p_value_t: NDArray[np.float64]
    cumulative_effect: float
    cumulative_se: float
    n_units: int
    n_periods: int
    alpha: float
    cov: NDArray[np.float64]
    nuisance_r2_t: NDArray[np.float64]
    backend: str = "custom"
    mode: str = "panel"

    def __repr__(self) -> str:
        blips = ", ".join(f"{v:.3f}" for v in self.theta_t)
        return (
            f"DynamicGEstimationResult(theta_t=[{blips}], "
            f"cumulative={self.cumulative_effect:.3f}, backend={self.backend!r})"
        )

    def summary(self) -> str:
        """Return a formatted per-period results table."""
        lines = [
            "Recursive Dynamic G-Estimation (Lewis-Syrgkanis)",
            "=" * 56,
            f"Backend: {self.backend}   Mode: {self.mode}   Units: {self.n_units}   "
            f"Periods (m): {self.n_periods}",
            "",
            f"{'period':>6} {'theta_t':>10} {'se':>9} "
            f"{'[' + str(int((1 - self.alpha) * 100)) + '% CI]':>20} {'p':>8}",
        ]
        for k in range(self.n_periods):
            ci = f"[{self.ci_lower_t[k]:.3f}, {self.ci_upper_t[k]:.3f}]"
            lines.append(
                f"{k:>6} {self.theta_t[k]:>10.4f} {self.se_t[k]:>9.4f} {ci:>20} "
                f"{self.p_value_t[k]:>8.4f}"
            )
        lines.append("-" * 56)
        lines.append(
            f"Cumulative effect: {self.cumulative_effect:.4f} " f"(se {self.cumulative_se:.4f})"
        )
        return "\n".join(lines)


def _resolve_model(spec: ModelSpec, random_state: Optional[int]) -> BaseEstimator:
    """Resolve a model spec (string alias or estimator) into a regressor.

    Conditional means ``E[Y|H]`` and ``E[T|H]`` are both estimated by
    regression (valid for continuous and binary treatment alike, where the
    treatment mean is the propensity).
    """
    if isinstance(spec, BaseEstimator):
        return clone(spec)
    if spec == "random_forest":
        return RandomForestRegressor(
            n_estimators=100, random_state=random_state, n_jobs=_DEFAULT_N_JOBS
        )
    if spec == "gradient_boosting":
        return GradientBoostingRegressor(random_state=random_state)
    if spec in ("linear", "ols"):
        return LinearRegression()
    if spec == "ridge":
        return Ridge(alpha=1.0)
    raise ValueError(
        f"Unknown model spec {spec!r}. Use an sklearn estimator or one of "
        "'random_forest', 'gradient_boosting', 'linear', 'ridge'."
    )


def _crossfit_residualize_kfold(
    H: NDArray[np.float64],
    targets: NDArray[np.float64],
    model: BaseEstimator,
    n_folds: int,
    random_state: Optional[int],
) -> NDArray[np.float64]:
    """Out-of-fold residuals of each column of ``targets`` on ``H`` (KFold over rows).

    Units are i.i.d. in the panel, so plain (shuffled) K-fold cross-fitting is
    the correct dimension. Returns ``targets - hat`` with the same shape.
    """
    n = H.shape[0]
    preds = np.zeros_like(targets, dtype=float)
    kf = KFold(n_splits=min(n_folds, n), shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(H):
        mod = clone(model)
        _yt = targets[train_idx]
        mod.fit(H[train_idx], _yt.ravel() if _yt.shape[1] == 1 else _yt)
        preds[test_idx] = np.asarray(mod.predict(H[test_idx])).reshape(len(test_idx), -1)
    return np.asarray(targets - preds, dtype=np.float64)


def _crossfit_residualize_timeseries(
    H: NDArray[np.float64],
    targets: NDArray[np.float64],
    model: BaseEstimator,
    n_splits: int,
    gap: int,
) -> NDArray[np.float64]:
    """Out-of-fold residuals using temporal CV (single-series mode).

    Rows uncovered by the expanding-window splits are returned as NaN so the
    caller can drop them (mirrors ``TemporalPLRDML``).
    """
    n = H.shape[0]
    preds = np.full_like(targets, np.nan, dtype=float)
    cv = TimeSeriesCrossValidator(n_splits=n_splits, gap=gap)
    for train_idx, test_idx in cv.split(H):
        mod = clone(model)
        _yt = targets[train_idx]
        mod.fit(H[train_idx], _yt.ravel() if _yt.shape[1] == 1 else _yt)
        preds[test_idx] = np.asarray(mod.predict(H[test_idx])).reshape(len(test_idx), -1)
    return np.asarray(targets - preds, dtype=np.float64)


def _r2(y: NDArray[np.float64], resid: NDArray[np.float64]) -> float:
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 1e-12 else 0.0


class DynamicGEstimationDML:
    """Recursive dynamic g-estimation for period-specific treatment blips.

    Args:
        n_periods: Horizon ``m`` for the series mode (number of distributed
            lags). For the panel mode ``m`` is inferred from ``groups`` and this
            may be left ``None``.
        model_y: Outcome nuisance ``E[Y|H]`` -- sklearn estimator or alias
            (``"random_forest"`` default, ``"linear"``, ``"ridge"``,
            ``"gradient_boosting"``).
        model_t: Treatment nuisance ``E[T|H]`` (same options).
        n_folds: Cross-fitting folds (panel KFold over units / series temporal CV).
        gap: Temporal-CV gap (series mode only).
        random_state: Seed for cross-fitting and nuisance models.

    The estimator is honest about its scope: it recovers a *constant* per-period
    blip (linear SNMM). Heterogeneous blips ``theta_tau(X)`` (the paper's Dynamic
    RLearner) are out of scope.
    """

    def __init__(
        self,
        n_periods: Optional[int] = None,
        model_y: ModelSpec = "random_forest",
        model_t: ModelSpec = "random_forest",
        n_folds: int = 5,
        gap: int = 0,
        random_state: Optional[int] = None,
    ):
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}.")
        self.n_periods = n_periods
        self.model_y = model_y
        self.model_t = model_t
        self.n_folds = int(n_folds)
        self.gap = int(gap)
        self.random_state = random_state

    def fit(
        self,
        Y: ArrayLike,
        T: ArrayLike,
        X: ArrayLike,
        W: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        mode: Optional[Mode] = None,
        alpha: float = 0.05,
    ) -> DynamicGEstimationResult:
        """Estimate the per-period blips.

        Args:
            Y: Outcome, stacked ``(n_rows,)``. Panel: per-period rows
                (unit-major). Series: the single series.
            T: Treatment, same stacking.
            X: Confounders/state, ``(n_rows, p)``.
            W: Optional extra controls appended to ``X`` (same stacking).
            groups: Unit id per row (panel). Required for ``mode="panel"``; each
                group must contribute the same number of consecutive periods.
            mode: ``"panel"`` (default when ``groups`` is given) or ``"series"``.
            alpha: Significance level for confidence intervals.

        Returns:
            A :class:`DynamicGEstimationResult`.

        Raises:
            ValueError: On inconsistent shapes, ragged panels, or missing
                ``groups`` for the panel mode.
        """
        Y = np.asarray(Y, dtype=float).ravel()
        T = np.asarray(T, dtype=float).ravel()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if W is not None:
            W = np.asarray(W, dtype=float)
            if W.ndim == 1:
                W = W.reshape(-1, 1)
            X = np.hstack([X, W])
        if not (len(Y) == len(T) == X.shape[0]):
            raise ValueError("Y, T, X must share the same number of rows.")

        resolved_mode: Mode = (
            mode if mode is not None else ("panel" if groups is not None else "series")
        )
        if resolved_mode == "panel":
            if groups is None:
                raise ValueError("mode='panel' requires groups (unit ids per row).")
            return self._fit_panel(Y, T, X, np.asarray(groups), alpha)
        return self._fit_series(Y, T, X, alpha)

    # ------------------------------------------------------------------ panel
    def _reshape_panel(
        self, Y: NDArray, T: NDArray, X: NDArray, groups: NDArray
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Reshape stacked rows into (n_units, m) / (n_units, m, p), validating regularity."""
        uniq, counts = np.unique(groups, return_counts=True)
        m = int(counts[0])
        if not np.all(counts == m):
            raise ValueError("Panel is ragged: every group must have the same number of periods.")
        if m < 2:
            raise ValueError(f"Need m >= 2 periods per unit, got {m}.")
        n = len(uniq)
        p = X.shape[1]
        order = np.argsort(np.argsort(groups, kind="stable"), kind="stable")  # noqa: F841
        # Rows are assumed unit-major and time-ordered within unit (the standard layout).
        Yp = Y.reshape(n, m)
        Tp = T.reshape(n, m)
        Xp = X.reshape(n, m, p)
        return Yp, Tp, Xp

    def _fit_panel(
        self, Y: NDArray, T: NDArray, X: NDArray, groups: NDArray, alpha: float
    ) -> DynamicGEstimationResult:
        Yp, Tp, Xp = self._reshape_panel(Y, T, X, groups)
        n, m, p = Xp.shape
        Y_final = Yp[:, -1]

        model_y = _resolve_model(self.model_y, self.random_state)
        model_t = _resolve_model(self.model_t, self.random_state)

        # Per stage tau: residualize [Y_final, T_0..T_{m-1}] on history H_tau.
        R: list[NDArray] = [np.empty(0)] * m
        Ttil: list[NDArray] = [np.empty(0)] * m  # Ttil[tau] is (n, m): col s = M_tau T_s
        nuis_r2 = np.zeros(m)
        for tau in range(m):
            cols = [Xp[:, k, :] for k in range(tau + 1)]  # states through tau
            if tau > 0:
                cols.append(Tp[:, :tau])  # past treatments
            H = np.hstack(cols)
            y_resid = _crossfit_residualize_kfold(
                H, Y_final.reshape(-1, 1), model_y, self.n_folds, self.random_state
            ).ravel()
            t_resid = _crossfit_residualize_kfold(
                H, Tp, model_t, self.n_folds, self.random_state
            )  # (n, m)
            R[tau] = y_resid
            Ttil[tau] = t_resid
            nuis_r2[tau] = _r2(Y_final, y_resid)

        # Backward recursion for point estimates.
        theta = np.zeros(m)
        for tau in range(m - 1, -1, -1):
            y_peeled = R[tau].copy()
            for s in range(tau + 1, m):
                y_peeled -= theta[s] * Ttil[tau][:, s]
            t_tau = Ttil[tau][:, tau]
            denom = float(t_tau @ t_tau)
            if denom < 1e-12:
                raise ValueError(
                    f"No residual treatment variation at period {tau} (weak/colinear treatment)."
                )
            theta[tau] = float(t_tau @ y_peeled) / denom

        # Joint sandwich variance over the stacked moments (upper-triangular Jacobian).
        g = np.zeros((n, m))
        G = np.zeros((m, m))
        for tau in range(m):
            t_tau = Ttil[tau][:, tau]
            resid_full = R[tau] - Ttil[tau][:, tau:] @ theta[tau:]
            g[:, tau] = t_tau * resid_full
            for s in range(tau, m):
                G[tau, s] = -float(np.mean(t_tau * Ttil[tau][:, s]))
        meat = (g.T @ g) / n
        G_inv = np.linalg.inv(G)
        cov = (G_inv @ meat @ G_inv.T) / n

        return self._package(theta, cov, alpha, n_units=n, m=m, nuis_r2=nuis_r2, mode="panel")

    # ----------------------------------------------------------------- series
    def _fit_series(
        self, Y: NDArray, T: NDArray, X: NDArray, alpha: float
    ) -> DynamicGEstimationResult:
        if self.n_periods is None:
            raise ValueError("mode='series' requires n_periods (number of distributed lags).")
        m = int(self.n_periods)
        L = len(Y)
        if L <= m + self.n_folds + 10:
            raise ValueError(f"series too short ({L}) for m={m} and n_folds={self.n_folds}.")

        # Distributed-lag design: Y_t on [T_t, T_{t-1}, ..., T_{t-m+1}], controls X_t.
        Tlag = np.column_stack(
            [T[(m - 1 - k) : (L - k)] for k in range(m)]
        )  # (Lv, m), col k = T_{t-k}
        Yv = Y[m - 1 :]
        Xv = X[m - 1 :]
        targets = np.column_stack([Yv, Tlag])
        resid = _crossfit_residualize_timeseries(
            Xv, targets, _resolve_model(self.model_y, self.random_state), self.n_folds, self.gap
        )
        keep = ~np.isnan(resid).any(axis=1)
        resid = resid[keep]
        Ytil = resid[:, 0]
        Ttil = resid[:, 1:]  # (Lk, m)
        lk = Ttil.shape[0]

        gram = Ttil.T @ Ttil
        theta = np.linalg.solve(gram, Ttil.T @ Ytil)

        # HAC sandwich: scores s_t = Ttil_t (Ytil_t - Ttil_t . theta); bread = (gram/Lk)^{-1}.
        resid_full = Ytil - Ttil @ theta
        scores = Ttil * resid_full[:, None]  # (Lk, m)
        bw = optimal_bandwidth(resid_full, method="newey_west")
        s_hac = _bartlett_long_run_cov(scores, bw)
        bread = np.linalg.inv(gram / lk)
        cov = (bread @ s_hac @ bread.T) / lk

        nuis_r2 = np.full(m, _r2(Yv[keep] if keep.shape[0] == len(Yv) else Ytil, Ytil))
        return self._package(theta, cov, alpha, n_units=1, m=m, nuis_r2=nuis_r2, mode="series")

    # ---------------------------------------------------------------- shared
    def _package(
        self,
        theta: NDArray,
        cov: NDArray,
        alpha: float,
        n_units: int,
        m: int,
        nuis_r2: NDArray,
        mode: str,
    ) -> DynamicGEstimationResult:
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        z = stats.norm.ppf(1 - alpha / 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            tstat = np.where(se > 0, theta / se, 0.0)
            pvals = 2 * (1 - stats.norm.cdf(np.abs(tstat)))
        cum = float(theta.sum())
        cum_se = float(np.sqrt(max(0.0, np.ones(m) @ cov @ np.ones(m))))
        return DynamicGEstimationResult(
            theta_t=theta,
            se_t=se,
            ci_lower_t=theta - z * se,
            ci_upper_t=theta + z * se,
            p_value_t=pvals,
            cumulative_effect=cum,
            cumulative_se=cum_se,
            n_units=n_units,
            n_periods=m,
            alpha=alpha,
            cov=cov,
            nuisance_r2_t=nuis_r2,
            backend="custom",
            mode=mode,
        )


def _bartlett_long_run_cov(scores: NDArray[np.float64], bandwidth: int) -> NDArray[np.float64]:
    """Newey-West (Bartlett) long-run covariance of a multivariate score series.

    ``S = Gamma_0 + sum_{j=1}^{bw} w_j (Gamma_j + Gamma_j^T)``, with Bartlett
    weights ``w_j = 1 - j/(bw+1)`` and ``Gamma_j = (1/T) sum_t s_t s_{t-j}^T``.
    """
    t, k = scores.shape
    s = scores - scores.mean(axis=0, keepdims=True)
    cov = (s.T @ s) / t
    bw = max(0, min(int(bandwidth), t - 1))
    for j in range(1, bw + 1):
        gamma_j = (s[j:].T @ s[:-j]) / t
        weight = 1.0 - j / (bw + 1)
        cov += weight * (gamma_j + gamma_j.T)
    return np.asarray(cov, dtype=np.float64)
