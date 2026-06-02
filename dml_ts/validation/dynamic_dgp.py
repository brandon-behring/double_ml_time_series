"""Dynamic-treatment DGP with known period-specific blips for g-estimation.

Generates data with a *known* sequence of per-period treatment effects
``theta_t = (theta_1, ..., theta_m)`` so that recursive dynamic g-estimation
(Lewis & Syrgkanis, 2021, arXiv:2002.07285) can be validated against ground
truth. This is the missing piece: the repo's existing DGPs expose only a scalar
effect, fixed lag coefficients, or piecewise regimes -- none a true sequence of
period-specific blips.

Two settings (both supported):

* ``mode="panel"`` -- n i.i.d. trajectories, each of length m periods. This is
  the canonical Lewis-Syrgkanis setting (the structure needed to identify m
  separate blips) and is directly comparable to EconML's ``DynamicDML`` (stacked
  rows + ``groups``).
* ``mode="series"`` -- a single long, stationary series with distributed-lag
  treatment effects ``Y_t = sum_k theta_{k+1} T_{t-k} + ...``. This is the
  single-unit, stationarity-based adaptation that Lewis & Syrgkanis also provide
  (their "progressive nuisance estimation analogue").

Structural model (panel), with optional treatment-dependent state:

    X_{i,tau} = A X_{i,tau-1} + b * T_{i,tau-1} + eps           (state, tau > 1)
    T_{i,tau} = pi . X_{i,tau} + eta                            (selection on state)
    Y_{i,tau} = sum_{s<=tau} theta_s * T_{i,s} + gamma . X_{i,tau} + u

When ``state_feedback=True`` the state depends on past treatment, so the
*total* effect of ``T_tau`` on the final outcome includes the indirect path
through future states. For the linear system above that total per-period blip is
available in closed form,

    theta*_tau = theta_tau + gamma^T A^{m-2-tau} b   (tau <= m-2),  theta*_{m-1} = theta_{m-1},

which is exactly the structural-nested-mean-model estimand that g-estimation
(and EconML's DynamicDML) recovers. ``DynamicTreatmentDGPResult.theta_t`` always
holds this total blip, so tests compare estimates against the correct target
regardless of the ``state_feedback`` setting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class DynamicTreatmentDGPResult:
    """Synthetic dynamic-treatment data with known per-period blips.

    Attributes:
        Y: Outcome, stacked as ``(n_rows,)``. For ``mode="panel"`` rows are
            unit-major: unit 0's m periods, then unit 1's, etc. (``n_rows =
            n_units * n_periods``). For ``mode="series"`` it is the single
            series ``(series_length,)``.
        T: Treatment, same stacking as ``Y``.
        X: Confounders, ``(n_rows, p)``, same stacking.
        groups: Unit id per row for the panel (EconML-compatible), or ``None``
            for ``mode="series"``.
        time_index: Within-unit period index ``0..m-1`` per row (panel), or
            ``0..L-1`` for the series.
        theta_t: True per-period total blip ``(m,)`` -- the g-estimation
            estimand (see module docstring).
        cumulative_effect: ``sum(theta_t)``.
        n_units: Number of trajectories (1 for the series).
        n_periods: Horizon m (number of blips).
        p: Number of confounders.
        mode: ``"panel"`` or ``"series"``.
        state_feedback: Whether the state depended on past treatment.
    """

    Y: NDArray[np.float64]
    T: NDArray[np.float64]
    X: NDArray[np.float64]
    groups: Optional[NDArray[np.int64]]
    time_index: NDArray[np.int64]
    theta_t: NDArray[np.float64]
    cumulative_effect: float
    n_units: int
    n_periods: int
    p: int
    mode: str
    state_feedback: bool = False

    def panels(self) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Reshape the panel stacking into ``(n_units, m)`` / ``(n_units, m, p)``.

        Returns:
            ``(Y_panel, T_panel, X_panel)`` with shapes ``(n_units, m)``,
            ``(n_units, m)`` and ``(n_units, m, p)``.

        Raises:
            ValueError: If called on a ``mode="series"`` result.
        """
        if self.mode != "panel":
            raise ValueError("panels() is only defined for mode='panel'.")
        m, p = self.n_periods, self.p
        return (
            self.Y.reshape(self.n_units, m),
            self.T.reshape(self.n_units, m),
            self.X.reshape(self.n_units, m, p),
        )

    def to_econml(
        self,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
        """Return ``(Y, T, X, groups)`` in EconML ``DynamicDML.fit`` layout.

        Raises:
            ValueError: If called on a ``mode="series"`` result (EconML
                ``DynamicDML`` requires panel ``groups``).
        """
        if self.mode != "panel" or self.groups is None:
            raise ValueError("to_econml() requires a panel result with groups.")
        return self.Y, self.T, self.X, self.groups


def _as_matrix(value: float | NDArray[np.float64], p: int) -> NDArray[np.float64]:
    """Coerce a scalar or array into a ``(p, p)`` state-transition matrix."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr) * np.eye(p)
    if arr.shape == (p, p):
        return arr
    raise ValueError(f"state_transition must be a scalar or ({p}, {p}) matrix, got {arr.shape}.")


def _as_vector(
    value: Optional[float | NDArray[np.float64]], p: int, default_first: float
) -> NDArray[np.float64]:
    """Coerce a scalar/array/None into a length-``p`` coefficient vector."""
    if value is None:
        vec = np.zeros(p)
        vec[0] = default_first
        return vec
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(p, float(arr))
    if arr.shape == (p,):
        return arr
    raise ValueError(f"coefficient vector must be scalar or length {p}, got {arr.shape}.")


class DynamicTreatmentDGP:
    """Generate dynamic-treatment data with known per-period blips ``theta_t``.

    See the module docstring for the structural model. The generator is fully
    deterministic given ``random_state`` and, with ``noise_level=0`` and linear
    nuisances, returns data whose blips are recovered exactly -- the basis for
    the estimator's deterministic-recovery test.

    Args:
        n_periods: Horizon ``m`` (number of per-period blips). Must be >= 2.
        theta_t: True per-period blips. Scalar (broadcast to all m periods) or a
            length-``m`` array. Interpreted as the *direct* structural
            coefficients; the result exposes the *total* blip (equal to these
            when ``state_feedback=False``).
        n_units: Number of i.i.d. trajectories (panel). Ignored for the series.
        p: Number of confounders.
        state_feedback: If True, the state depends on past treatment
            (``X_tau += b * T_{tau-1}``), creating genuine time-varying
            confounding; the result's ``theta_t`` is adjusted to the closed-form
            total effect. If False, the state is exogenous (matches EconML's
            example convention and yields an unambiguous estimand).
        state_transition: AR coefficient for the state -- scalar (``c * I``) or a
            ``(p, p)`` matrix ``A``.
        treatment_state_coef: ``b`` -- how past treatment shifts the state
            (scalar -> first coordinate, or length-``p`` vector). Only used when
            ``state_feedback=True``.
        treatment_policy_coef: ``pi`` -- selection of treatment on the current
            state (scalar -> first coordinate, or length-``p`` vector).
        confounding_coef: ``gamma`` -- effect of the state on the outcome
            (scalar -> all coordinates, or length-``p`` vector).
        treatment_type: ``"continuous"`` or ``"binary"``.
        noise_level: Default std for the state, treatment, and outcome noise
            terms (used wherever a specific channel below is left ``None``).
        state_noise: Std of the state shock ``eps`` (defaults to ``noise_level``).
        treatment_noise: Std of the treatment shock ``eta`` (defaults to
            ``noise_level``). This is the *independent* treatment variation that
            identifies the blips; it must be > 0 for recovery.
        outcome_noise: Std of the outcome shock ``u`` (defaults to
            ``noise_level``). Set to 0 for an exact (deterministic) recovery test.
        series_length: Length of the single series for ``mode="series"``.
        mode: ``"panel"`` (default) or ``"series"``.
        random_state: Seed for reproducibility.

    Raises:
        ValueError: On invalid shapes or parameters.
    """

    def __init__(
        self,
        n_periods: int,
        theta_t: float | NDArray[np.float64] = 1.0,
        n_units: int = 500,
        p: int = 3,
        state_feedback: bool = False,
        state_transition: float | NDArray[np.float64] = 0.5,
        treatment_state_coef: Optional[float | NDArray[np.float64]] = None,
        treatment_policy_coef: Optional[float | NDArray[np.float64]] = None,
        confounding_coef: Optional[float | NDArray[np.float64]] = 1.0,
        treatment_type: Literal["continuous", "binary"] = "continuous",
        noise_level: float = 1.0,
        state_noise: Optional[float] = None,
        treatment_noise: Optional[float] = None,
        outcome_noise: Optional[float] = None,
        series_length: int = 2000,
        mode: Literal["panel", "series"] = "panel",
        random_state: Optional[int] = None,
    ):
        if n_periods < 2:
            raise ValueError(f"n_periods (m) must be >= 2, got {n_periods}.")
        if p <= 0:
            raise ValueError(f"p must be positive, got {p}.")
        if mode not in ("panel", "series"):
            raise ValueError(f"mode must be 'panel' or 'series', got {mode!r}.")
        if treatment_type not in ("continuous", "binary"):
            raise ValueError(
                f"treatment_type must be 'continuous' or 'binary', got {treatment_type!r}."
            )

        self.n_periods = int(n_periods)
        self.n_units = int(n_units)
        self.p = int(p)
        self.state_feedback = bool(state_feedback)
        self.treatment_type = treatment_type
        self.noise_level = float(noise_level)
        self.state_noise = float(noise_level if state_noise is None else state_noise)
        self.treatment_noise = float(noise_level if treatment_noise is None else treatment_noise)
        self.outcome_noise = float(noise_level if outcome_noise is None else outcome_noise)
        self.series_length = int(series_length)
        self.mode = mode
        self.random_state = random_state

        m = self.n_periods
        theta_arr = np.asarray(theta_t, dtype=float)
        self.theta_direct = np.full(m, float(theta_arr)) if theta_arr.ndim == 0 else theta_arr
        if self.theta_direct.shape != (m,):
            raise ValueError(
                f"theta_t must be scalar or length {m}, got {self.theta_direct.shape}."
            )

        self.A = _as_matrix(state_transition, p)
        self.b = _as_vector(treatment_state_coef, p, default_first=0.5)
        self.pi = _as_vector(treatment_policy_coef, p, default_first=1.0)
        self.gamma = (
            _as_vector(confounding_coef, p, default_first=1.0)
            if confounding_coef is not None
            else np.zeros(p)
        )

    def _total_blip(self) -> NDArray[np.float64]:
        """Closed-form total per-period blip (the g-estimation estimand)."""
        m = self.n_periods
        theta_star = self.theta_direct.copy()
        if self.state_feedback:
            # SNMM blip = direct effect + indirect path through future states, holding
            # future *treatments* at baseline. T_tau enters X_{tau+1} (coef b) and then
            # propagates by A: d X_{m-1} / d T_tau = A^{m-2-tau} b. So
            #   theta*_tau = theta_tau + gamma^T A^{m-2-tau} b   (tau <= m-2; last unchanged).
            for tau in range(m - 1):
                power = np.linalg.matrix_power(self.A, m - 2 - tau)
                theta_star[tau] = self.theta_direct[tau] + float(self.gamma @ power @ self.b)
        return theta_star

    def _sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def generate(self) -> DynamicTreatmentDGPResult:
        """Generate the dataset.

        Returns:
            A :class:`DynamicTreatmentDGPResult` whose ``theta_t`` is the true
            total per-period blip.
        """
        if self.mode == "panel":
            return self._generate_panel()
        return self._generate_series()

    def _generate_panel(self) -> DynamicTreatmentDGPResult:
        rng = np.random.default_rng(self.random_state)
        n, m, p = self.n_units, self.n_periods, self.p

        X = np.zeros((n, m, p))
        T = np.zeros((n, m))
        X[:, 0, :] = rng.normal(scale=1.0, size=(n, p))
        for tau in range(m):
            if tau > 0:
                state = X[:, tau - 1, :] @ self.A.T + self.state_noise * rng.normal(size=(n, p))
                if self.state_feedback:
                    state = state + np.outer(T[:, tau - 1], self.b)
                X[:, tau, :] = state
            lin = X[:, tau, :] @ self.pi + self.treatment_noise * rng.normal(size=n)
            if self.treatment_type == "continuous":
                T[:, tau] = lin
            else:
                T[:, tau] = rng.binomial(1, self._sigmoid(lin)).astype(float)

        # Accumulated outcome: period-tau outcome carries blips of all s <= tau.
        cum_treatment = np.zeros((n, m))
        running = np.zeros(n)
        for tau in range(m):
            running = running + self.theta_direct[tau] * T[:, tau]
            cum_treatment[:, tau] = running
        Y = (
            cum_treatment
            + np.einsum("ntp,p->nt", X, self.gamma)
            + self.outcome_noise * rng.normal(size=(n, m))
        )

        groups = np.repeat(np.arange(n, dtype=np.int64), m)
        time_index = np.tile(np.arange(m, dtype=np.int64), n)
        theta_star = self._total_blip()
        return DynamicTreatmentDGPResult(
            Y=Y.reshape(n * m),
            T=T.reshape(n * m),
            X=X.reshape(n * m, p),
            groups=groups,
            time_index=time_index,
            theta_t=theta_star,
            cumulative_effect=float(theta_star.sum()),
            n_units=n,
            n_periods=m,
            p=p,
            mode="panel",
            state_feedback=self.state_feedback,
        )

    def _generate_series(self) -> DynamicTreatmentDGPResult:
        rng = np.random.default_rng(self.random_state)
        m, p, L = self.n_periods, self.p, self.series_length
        if L <= m + 10:
            raise ValueError(f"series_length ({L}) must exceed n_periods + 10.")

        X = np.zeros((L, p))
        T = np.zeros(L)
        X[0] = rng.normal(size=p)
        for t in range(L):
            if t > 0:
                state = self.A @ X[t - 1] + self.state_noise * rng.normal(size=p)
                if self.state_feedback:
                    state = state + self.b * T[t - 1]
                X[t] = state
            lin = float(self.pi @ X[t]) + self.treatment_noise * float(rng.normal())
            T[t] = (
                lin
                if self.treatment_type == "continuous"
                else float(rng.binomial(1, self._sigmoid(np.array(lin))))
            )

        # Distributed-lag outcome: Y_t = sum_{k=0..m-1} theta_{k+1} T_{t-k} + gamma.X_t + u.
        Y = X @ self.gamma + self.outcome_noise * rng.normal(size=L)
        for k in range(m):
            Y[k:] += self.theta_direct[k] * T[: L - k]

        theta_star = self.theta_direct.copy()  # series mode: distributed-lag coefs are the target
        return DynamicTreatmentDGPResult(
            Y=Y,
            T=T,
            X=X,
            groups=None,
            time_index=np.arange(L, dtype=np.int64),
            theta_t=theta_star,
            cumulative_effect=float(theta_star.sum()),
            n_units=1,
            n_periods=m,
            p=p,
            mode="series",
            state_feedback=self.state_feedback,
        )
