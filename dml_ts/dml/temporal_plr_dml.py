"""
Temporal partially linear DML for time series.

Implements scalar partially linear DML with lagged treatment controls,
time-series cross-fitting, and HAC inference. This module is not an
implementation of Lewis-Syrgkanis dynamic g-estimation.

References:

- Lewis, R. and Syrgkanis, V. (2021). Double/Debiased Machine Learning for
  Dynamic Treatment Effects. NeurIPS.
- Chernozhukov et al. (2018). Double Machine Learning for Treatment and
  Structural Parameters.

Core temporal partially linear DML classes.

Key components:
1. TemporalPLRDML - Scalar PLR DML with time-series cross-validation
2. RollingWindowDML - Local estimation for non-stationary effects
3. PanelDML - Fixed effects + DML for panel data structures

Usage:
    >>> import numpy as np
    >>> from dml_ts.dml import TemporalPLRDML
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> time = np.arange(n)
    >>> X = rng.standard_normal((n, 2))
    >>> T = 0.5 * X[:, 0] + rng.standard_normal(n)
    >>> Y = 2.0 * T + X[:, 1] + rng.standard_normal(n)
    >>> model = TemporalPLRDML(
    ...     n_lags=1,
    ...     model_y="ridge",
    ...     model_t="ridge",
    ...     cv_strategy="time_series_split",
    ...     n_splits=3,
    ... )
    >>> result = model.fit(Y=Y, T=T, X=X, time_index=time)
    >>> isinstance(result.theta, float)
    True
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.base import BaseEstimator
from temporalcv import (
    BlockedTimeSeriesCV,
    PurgedWalkForward,
    TimeSeriesCrossValidator,
    ci_ordered,
    coverage_in_unit,
    finite_se,
    newey_west_se,
)

from ._results import ResultBase
from ._utils import ModelType, cross_fit_predictions, theta_via_fwl, validate_lengths
from ._utils import compute_r2 as _compute_r2
from ._utils import get_nuisance_model as _get_nuisance_model

# Type aliases
ArrayLike = np.ndarray | list[float]
CVStrategy = Literal["time_series_split", "blocked_cv", "purged_cv"]
KernelType = Literal["bartlett", "parzen", "quadratic_spectral"]


@dataclass(frozen=True, slots=True, eq=False)
class TemporalPLRDMLResult(ResultBase):
    """Result container for TemporalPLRDML estimation.

    Attributes:
        theta: Scalar treatment effect estimate
        se: Standard error (HAC-adjusted for time series)
        t_stat: t-statistic = theta / se
        ci_lower: Lower 95% confidence interval bound
        ci_upper: Upper 95% confidence interval bound
        p_value: Two-sided p-value for H0: theta = 0
        n_samples: Number of observations
        n_periods: Number of observations used after lag and temporal-CV filtering
        outcome_r2_cv: Cross-validated R² for outcome model
        treatment_r2_cv: Cross-validated R² for treatment model
        hac_bandwidth: Bandwidth used for HAC covariance
        cv_strategy: Cross-validation strategy used
        Y_residual: Outcome residuals (Ỹ = Y - m̂(X))
        T_residual: Treatment residuals (T̃ = T - ℓ̂(X))
        influence_scores: Influence function values for each observation
        dropped_initial_rows: Rows excluded because temporal CV cannot produce
            out-of-fold predictions without training on future observations
        lagged_rows_dropped: Rows excluded by lag-feature construction

    Examples:
        >>> import numpy as np
        >>> from dml_ts.dml import TemporalPLRDML
        >>> np.random.seed(0)
        >>> n = 200
        >>> X = np.random.randn(n, 2)
        >>> T = 0.5 * X[:, 0] + np.random.randn(n)
        >>> Y = 2.0 * T + X[:, 1] + np.random.randn(n)
        >>> result = TemporalPLRDML(n_lags=1, model_y="ridge", model_t="ridge",
        ...                         n_splits=3).fit(Y, T, X)
        >>> isinstance(result.theta, float)
        True
        >>> bool(result.ci_lower < result.ci_upper)
        True
    """

    theta: float
    se: float
    t_stat: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_samples: int
    n_periods: int
    outcome_r2_cv: float
    treatment_r2_cv: float
    hac_bandwidth: int
    cv_strategy: str
    Y_residual: NDArray[np.float64]
    T_residual: NDArray[np.float64]
    influence_scores: NDArray[np.float64]
    dropped_initial_rows: int = 0
    lagged_rows_dropped: int = 0

    def _validate(self) -> None:
        """B3 numeric hard-fails at the result boundary."""
        if not np.isfinite(self.theta):
            raise ValueError(f"TemporalPLRDMLResult.theta is not finite: {self.theta!r}")
        finite_se(self.se, name="TemporalPLRDMLResult.se")
        ci_ordered(self.ci_lower, self.ci_upper, name="TemporalPLRDMLResult.ci")
        coverage_in_unit(self.p_value, name="TemporalPLRDMLResult.p_value")

    def __repr__(self) -> str:
        return (
            f"TemporalPLRDMLResult(θ={self.theta:.4f}, SE={self.se:.4f}, "
            f"95% CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
            f"p={self.p_value:.4f})"
        )

    def summary(self) -> str:
        """Return formatted summary of TemporalPLRDML results."""
        return f"""
Temporal PLR DML Results
========================
Treatment Effect (θ):    {self.theta:.4f}
HAC Standard Error:      {self.se:.4f}
t-statistic:             {self.t_stat:.2f}
p-value:                 {self.p_value:.4f}
95% Confidence Interval: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]

Sample Information:
  Observations:          {self.n_samples}
  Used observations:     {self.n_periods}
  Lag rows dropped:      {self.lagged_rows_dropped}
  CV rows dropped:       {self.dropped_initial_rows}

Nuisance Model Diagnostics:
  Outcome R² (CV):       {self.outcome_r2_cv:.3f}
  Treatment R² (CV):     {self.treatment_r2_cv:.3f}

HAC Inference:
  Bandwidth:             {self.hac_bandwidth}
  CV Strategy:           {self.cv_strategy}

Interpretation:
  A one-unit increase in treatment is associated with a
  {self.theta:.4f} unit change in outcome (p={self.p_value:.4f}).
  Standard errors are HAC-corrected for serial correlation.
"""


def _create_lagged_features(
    X: NDArray[np.float64],
    T: NDArray[np.float64],
    n_lags: int,
) -> tuple[NDArray[np.float64], int]:
    """Create lagged controls for temporal partially linear DML.

    Constructs a feature matrix with current controls and lagged treatment
    values. The resulting estimator remains scalar PLR DML, not a recursive
    dynamic treatment-effect estimator.

    Args:
        X: Covariates (n_samples, n_features)
        T: Treatment (n_samples,)
        n_lags: Number of lags to include

    Returns:
        Tuple of (augmented feature matrix, number of rows lost to lags)
    """
    n_samples = X.shape[0]

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Build lagged features: [X_t, T_{t-1}, T_{t-2}, ..., T_{t-n_lags}]
    # Also include lagged X if desired: [X_t, X_{t-1}, ..., T_{t-1}, ...]
    lagged_cols: list[NDArray[np.float64]] = []

    # Current X
    lagged_cols.append(X[n_lags:])

    # Lagged treatments
    for lag in range(1, n_lags + 1):
        T_lagged = T[n_lags - lag : n_samples - lag].reshape(-1, 1)
        lagged_cols.append(T_lagged)

    X_augmented = np.hstack(lagged_cols)
    return X_augmented, n_lags


def _cross_fit_nuisance_time_series(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    T: NDArray[np.float64],
    cv: TimeSeriesCrossValidator | BlockedTimeSeriesCV | PurgedWalkForward,
    outcome_model: BaseEstimator,
    treatment_model: BaseEstimator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate cross-fitted predictions for nuisance models using time series CV.

    For each fold, trains models on out-of-fold data (respecting temporal order)
    and predicts on in-fold data. This eliminates in-sample bias while
    maintaining temporal structure.

    Args:
        X: Covariates (n, p)
        Y: Outcome (n,)
        T: Treatment (n,)
        cv: Time series cross-validator
        outcome_model: Model for E[Y|X]
        treatment_model: Model for E[T|X]

    Returns:
        Tuple of (Y_hat, T_hat) cross-fitted predictions
    """
    # temporalcv splitters share split(X, y=None, groups=None) — splitting
    # is positional. Materialize the folds to detect silent shortfall.
    # temporalcv>=2.1.0 raises natively on under-provisioned configs across
    # the whole purged family (PurgedWalkForward since #32, its fixed-window
    # and embargo guards since #35/#36, the forward-only siblings since
    # v2.0), so this is splitter-agnostic defense-in-depth for any conforming
    # splitter that under-yields rather than raising.
    splits = list(cv.split(X, Y))
    if len(splits) < cv.get_n_splits():
        warnings.warn(
            f"CV yielded {len(splits)} of {cv.get_n_splits()} requested folds; "
            "the configuration is under-provisioned for this sample size. "
            "Uncovered rows are excluded via dropped_initial_rows.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Early observations may be uncovered by expanding-window temporal CV;
    # cross_fit_predictions leaves them NaN so the estimator excludes them
    # rather than training on future observations to fill the gaps.
    return cross_fit_predictions(X, Y, T, splits, outcome_model, treatment_model)


class TemporalPLRDML:
    """Temporal partially linear DML for scalar treatment effects.

    Implements scalar PLR DML for ordered data, handling:
    - Lagged treatment controls
    - Time-indexed confounders
    - Autocorrelation in residuals (HAC standard errors)
    - Temporal cross-validation (respects time ordering)

    The algorithm:
    1. Create lagged controls
    2. Use time series cross-validation to generate out-of-sample predictions
    3. Compute residuals: Ỹ = Y - m̂(X), T̃ = T - ℓ̂(X)
    4. Estimate θ via FWL: θ̂ = Σ(Ỹ·T̃) / Σ(T̃²)
    5. Compute HAC standard errors (Newey-West) for autocorrelation

    Args:
        n_lags: Number of lags to include for treatment and confounders
        model_y: ML model for outcome nuisance function
        model_t: ML model for treatment nuisance function
        cv_strategy: Cross-validation strategy for time series
        n_splits: Number of CV splits
        gap: Gap between train and test sets (to prevent leakage)
        hac_bandwidth: Bandwidth for HAC covariance (None = automatic; 0 = no lags)
        hac_kernel: Kernel for HAC estimation
        random_state: Random seed for reproducibility

    Examples:
        >>> import numpy as np
        >>> from dml_ts.dml import TemporalPLRDML
        >>> np.random.seed(42)
        >>> n = 300
        >>> time = np.arange(n)
        >>> X = np.random.randn(n, 3)
        >>> T = 0.5 * X[:, 0] + 0.3 * np.sin(time / 50) + np.random.randn(n)
        >>> Y = 2.0 * T + np.exp(X[:, 1] / 2) + np.random.randn(n)
        >>> model = TemporalPLRDML(n_lags=3, model_y="ridge", model_t="ridge", n_splits=3)
        >>> result = model.fit(Y, T, X, time_index=time)
        >>> bool(abs(result.theta - 2.0) < 0.3)  # Should be close to 2.0
        True
    """

    def __init__(
        self,
        n_lags: int = 1,
        model_y: ModelType = "random_forest",
        model_t: ModelType = "random_forest",
        cv_strategy: CVStrategy = "time_series_split",
        n_splits: int = 5,
        gap: int = 0,
        hac_bandwidth: int | None = None,
        hac_kernel: KernelType = "bartlett",
        random_state: int | None = None,
    ):
        """Initialize TemporalPLRDML.

        Args:
            n_lags: Number of lags for treatment and confounders
            model_y: Model type for outcome regression
            model_t: Model type for treatment propensity
            cv_strategy: Time series CV strategy
            n_splits: Number of cross-validation splits
            gap: Gap between train and test (prevents leakage)
            hac_bandwidth: HAC bandwidth (None = automatic floor(n^(1/3))
                selection; 0 = no lags, i.e. heteroskedasticity-only SEs)
            hac_kernel: Kernel function for HAC estimation
            random_state: Random seed

        Raises:
            ValueError: If hac_bandwidth is not None or a non-negative int.
        """
        if hac_bandwidth is not None and (
            isinstance(hac_bandwidth, bool)
            or not isinstance(hac_bandwidth, (int, np.integer))
            or hac_bandwidth < 0
        ):
            raise ValueError(
                f"hac_bandwidth must be None or a non-negative int, got {hac_bandwidth!r}"
            )
        self.n_lags = n_lags
        self.model_y = model_y
        self.model_t = model_t
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.gap = gap
        self.hac_bandwidth = hac_bandwidth
        self.hac_kernel = hac_kernel
        self.random_state = random_state

        # Fitted state
        self._outcome_model: BaseEstimator | None = None
        self._treatment_model: BaseEstimator | None = None
        self._is_fitted = False
        self._result: TemporalPLRDMLResult | None = None

        # Data state
        self._Y_residual: NDArray[np.float64] | None = None
        self._T_residual: NDArray[np.float64] | None = None
        self._X_augmented: NDArray[np.float64] | None = None

    def _create_cv(
        self, n_samples: int
    ) -> TimeSeriesCrossValidator | BlockedTimeSeriesCV | PurgedWalkForward:
        """Create the appropriate cross-validator based on strategy."""
        if self.cv_strategy == "time_series_split":
            return TimeSeriesCrossValidator(
                n_splits=self.n_splits,
                gap=self.gap,
                expanding=True,
            )
        elif self.cv_strategy == "blocked_cv":
            # BlockedTimeSeriesCV uses gap_blocks instead of gap
            # Convert gap (observations) to gap_blocks (number of blocks)
            return BlockedTimeSeriesCV(
                n_splits=self.n_splits,
                gap_blocks=max(1, self.gap // 10) if self.gap > 0 else 1,
            )
        elif self.cv_strategy == "purged_cv":
            # Forward-only purged walk (temporalcv). The retired
            # PurgedGroupTimeSeriesCV trained on observations AFTER each test
            # fold — temporal leakage — so this strategy now purges `gap`
            # observations before each test window and never trains on the
            # future. gap maps directly to the absolute purge_gap.
            return PurgedWalkForward(
                n_splits=self.n_splits,
                purge_gap=self.gap,
                # Explicit: gap maps EXACTLY to the purge distance; without
                # this, PurgedWalkForward's default 1%-of-test embargo would
                # add undisclosed extra separation.
                embargo_pct=0.0,
            )
        else:
            raise ValueError(f"Unknown CV strategy: {self.cv_strategy}")

    def fit(
        self,
        Y: ArrayLike,
        T: ArrayLike,
        X: ArrayLike,
        time_index: ArrayLike | None = None,
        alpha: float = 0.05,
    ) -> TemporalPLRDMLResult:
        """Fit the TemporalPLRDML model.

        Estimates the Average Treatment Effect (ATE) in the partially linear model:
            Y_t = θ·T_t + g(X_t, T_{t-1}, ..., T_{t-L}) + ε_t

        using Neyman-orthogonal moment conditions, time series cross-fitting,
        and HAC standard errors.

        Args:
            Y: Outcome variable (n_samples,)
            T: Treatment variable (n_samples,)
            X: Confounders (n_samples, n_features)
            time_index: Accepted for API compatibility and length-validated;
                splitting is positional (row order), so the values are not used
            alpha: Significance level for confidence intervals (default 0.05)

        Returns:
            TemporalPLRDMLResult with treatment effect, HAC SE, and diagnostics

        Raises:
            ValueError: If inputs have incompatible shapes
        """
        # Convert to numpy arrays
        Y = np.asarray(Y, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = len(Y)

        # Validate inputs
        validate_lengths(Y, T, X)

        # Create time index if not provided
        if time_index is not None and len(np.asarray(time_index)) != n_samples:
            raise ValueError(
                f"time_index length ({len(np.asarray(time_index))}) must match "
                f"Y length ({n_samples})"
            )

        # Create lagged features
        if self.n_lags > 0:
            X_augmented, n_dropped = _create_lagged_features(X, T, self.n_lags)
            Y_eff = Y[self.n_lags :]
            T_eff = T[self.n_lags :]
        else:
            X_augmented = X
            Y_eff = Y
            T_eff = T
            n_dropped = 0

        n_effective = len(Y_eff)

        # Set up nuisance models
        outcome_model = _get_nuisance_model(self.model_y, self.random_state)
        treatment_model = _get_nuisance_model(self.model_t, self.random_state)

        # Create time series cross-validator
        cv = self._create_cv(n_effective)

        # Cross-fit nuisance models
        Y_hat, T_hat = _cross_fit_nuisance_time_series(
            X=X_augmented,
            Y=Y_eff,
            T=T_eff,
            cv=cv,
            outcome_model=outcome_model,
            treatment_model=treatment_model,
        )

        # Expanding-window temporal CV cannot predict the earliest rows without
        # using future observations. Exclude rows without true out-of-fold
        # predictions from all downstream residual and inference calculations.
        valid_oof_mask = np.isfinite(Y_hat) & np.isfinite(T_hat)
        dropped_initial_rows = int(np.sum(~valid_oof_mask))
        if not valid_oof_mask.any():
            raise ValueError(
                "Temporal cross-fitting produced no out-of-fold predictions. "
                "Use more observations, fewer splits, or a smaller gap/purge setting."
            )

        Y_used = Y_eff[valid_oof_mask]
        T_used = T_eff[valid_oof_mask]
        X_used = X_augmented[valid_oof_mask]
        Y_hat_used = Y_hat[valid_oof_mask]
        T_hat_used = T_hat[valid_oof_mask]
        n_used = len(Y_used)

        # Compute cross-validated R² on rows with valid OOF predictions.
        outcome_r2_cv = _compute_r2(Y_used, Y_hat_used)
        treatment_r2_cv = _compute_r2(T_used, T_hat_used)
        if outcome_r2_cv < -0.25 or treatment_r2_cv < -0.25:
            warnings.warn(
                "Nuisance model diagnostics are poor: at least one temporal "
                "cross-validated R² is below -0.25. Check controls, stationarity, "
                "overlap, and nuisance model specification.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Compute residuals
        Y_tilde = Y_used - Y_hat_used
        T_tilde = T_used - T_hat_used

        # Store for later use
        self._Y_residual = Y_tilde
        self._T_residual = T_tilde
        self._X_augmented = X_used

        # Estimate theta via FWL
        theta, T_tilde_sq_sum = theta_via_fwl(
            Y_tilde,
            T_tilde,
            no_variation_msg=(
                "Treatment has no variation after controlling for X and lags. "
                "This may indicate perfect prediction of T by X."
            ),
        )
        T_tilde_sq_mean = np.mean(T_tilde**2)

        if T_tilde_sq_mean < 1e-6:
            warnings.warn(
                "Treatment residual variation is very low after controlling for "
                "X and lags. Overlap may be weak and inference may be unstable.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Compute influence scores
        influence_scores = (Y_tilde - theta * T_tilde) * T_tilde / T_tilde_sq_mean

        # Compute HAC standard errors via temporalcv. The "residuals" for
        # HAC are the influence scores; HACResult.se is sqrt(long-run
        # variance / n) — the variance of the MEAN — so no further scaling
        # (issue #7 was a second division by n at exactly this point).
        if not np.all(np.isfinite(influence_scores)):
            raise ValueError(
                "Influence scores contain non-finite values; check the "
                "nuisance model predictions and input data for NaN/inf."
            )
        hac_res = newey_west_se(
            influence_scores,
            bandwidth=self.hac_bandwidth if self.hac_bandwidth is not None else "auto",
            kernel=self.hac_kernel,
        )
        hac_se = float(hac_res.se)
        bandwidth_used = int(hac_res.bandwidth)

        # Compute inference statistics
        # Degenerate SEs must raise, never fabricate t=0/p=1 (issue #11).
        finite_se(hac_se, name="HAC standard error")
        t_stat = theta / hac_se
        p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat))))

        # Confidence interval
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_lower = theta - z_crit * hac_se
        ci_upper = theta + z_crit * hac_se

        # Store fitted models
        self._outcome_model = outcome_model
        self._treatment_model = treatment_model
        self._is_fitted = True

        # Create result
        result = TemporalPLRDMLResult(
            theta=theta,
            se=hac_se,
            t_stat=t_stat,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            n_samples=n_samples,
            n_periods=n_used,
            outcome_r2_cv=outcome_r2_cv,
            treatment_r2_cv=treatment_r2_cv,
            hac_bandwidth=bandwidth_used,
            cv_strategy=self.cv_strategy,
            Y_residual=Y_tilde,
            T_residual=T_tilde,
            influence_scores=influence_scores,
            dropped_initial_rows=dropped_initial_rows,
            lagged_rows_dropped=n_dropped,
        )

        self._result = result
        return result

    def effect(
        self,
        X: ArrayLike | None = None,
        T0: float = 0.0,
        T1: float = 1.0,
    ) -> np.ndarray:
        """Estimate treatment effect.

        For the partially linear model, the treatment effect is constant
        (homogeneous), so this returns the estimated θ for all observations.

        Args:
            X: Covariate values used only to set output length; unused by PLR
            T0: Baseline treatment value (default 0)
            T1: Counterfactual treatment value (default 1)

        Returns:
            Array of treatment effects (θ × (T1 - T0) for each observation)

        Raises:
            ValueError: If model not fitted
        """
        if not self._is_fitted or self._result is None:
            raise ValueError("Model must be fitted before calling effect(). Call fit() first.")

        # For partially linear model, effect is homogeneous
        theta = self._result.theta
        effect_size = theta * (T1 - T0)

        if X is not None:
            n = len(np.asarray(X))
            return np.full(n, effect_size)
        elif self._T_residual is not None:
            return np.full(len(self._T_residual), effect_size)
        else:
            return np.array([effect_size])

    def effect_interval(
        self,
        X: ArrayLike | None = None,
        alpha: float = 0.05,
        T0: float = 0.0,
        T1: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute confidence interval for treatment effect.

        Returns HAC-adjusted confidence intervals for the treatment effect.

        Args:
            X: Covariate values used only to set output length; unused by PLR
            alpha: Significance level (default 0.05 for 95% CI)
            T0: Baseline treatment value
            T1: Counterfactual treatment value

        Returns:
            Tuple of (lower_bound, upper_bound) arrays

        Raises:
            ValueError: If model not fitted
        """
        if not self._is_fitted or self._result is None:
            raise ValueError(
                "Model must be fitted before calling effect_interval(). Call fit() first."
            )

        z_crit = stats.norm.ppf(1 - alpha / 2)
        scale = T1 - T0

        lower = (self._result.theta - z_crit * self._result.se) * scale
        upper = (self._result.theta + z_crit * self._result.se) * scale

        if X is not None:
            n = len(np.asarray(X))
            return np.full(n, lower), np.full(n, upper)
        elif self._T_residual is not None:
            n = len(self._T_residual)
            return np.full(n, lower), np.full(n, upper)
        else:
            return np.array([lower]), np.array([upper])


class RollingWindowDML:
    """Rolling-window DML for non-stationary treatment effects.

    Estimates local treatment effects using a moving window approach,
    useful when treatment effects vary over time.

    The algorithm works as follows: for each window position t,
    select observations in [t - window_size/2, t + window_size/2],
    run DML on window data, and store the local treatment effect estimate.
    Returns a time series of local effects with HAC standard errors.

    Args:
        window_size: Size of rolling window
        step_size: Step between window centers
        model_y: ML model for outcome nuisance function
        model_t: ML model for treatment nuisance function
        n_folds: Number of CV folds within each window
        hac_bandwidth: HAC bandwidth (None = auto)
        random_state: Random seed

    Examples:
        >>> import numpy as np
        >>> from dml_ts.dml import RollingWindowDML
        >>> np.random.seed(0)
        >>> n = 200
        >>> X = np.random.randn(n, 2)
        >>> T = 0.5 * X[:, 0] + np.random.randn(n)
        >>> Y = 1.5 * T + X[:, 1] + np.random.randn(n)
        >>> model = RollingWindowDML(window_size=80, step_size=40, model_y="ridge",
        ...                          model_t="ridge", n_folds=3)
        >>> _ = model.fit(Y, T, X)
        >>> centers, thetas, ses = model.get_effects()
        >>> bool(len(centers) == len(thetas) == len(ses))
        True
    """

    def __init__(
        self,
        window_size: int = 100,
        step_size: int = 10,
        model_y: ModelType = "random_forest",
        model_t: ModelType = "random_forest",
        n_folds: int = 3,
        hac_bandwidth: int | None = None,
        random_state: int | None = None,
    ):
        """Initialize RollingWindowDML."""
        self.window_size = window_size
        self.step_size = step_size
        self.model_y = model_y
        self.model_t = model_t
        self.n_folds = n_folds
        self.hac_bandwidth = hac_bandwidth
        self.random_state = random_state

        self._theta_series: NDArray[np.float64] | None = None
        self._se_series: NDArray[np.float64] | None = None
        self._time_centers: NDArray[np.float64] | None = None
        self._is_fitted = False

    def fit(
        self,
        Y: ArrayLike,
        T: ArrayLike,
        X: ArrayLike,
        time_index: ArrayLike | None = None,
    ) -> RollingWindowDML:
        """Fit rolling-window DML model.

        Args:
            Y: Outcome variable (n_samples,)
            T: Treatment variable (n_samples,)
            X: Confounders (n_samples, n_features)
            time_index: Time index for each observation

        Returns:
            self: Fitted model
        """
        Y = np.asarray(Y, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = len(Y)

        # Compute window centers
        half_window = self.window_size // 2
        centers = list(range(half_window, n_samples - half_window, self.step_size))

        theta_list: list[float] = []
        se_list: list[float] = []
        kept_centers: list[int] = []
        skipped: list[tuple[int, str]] = []

        for center in centers:
            start_idx = max(0, center - half_window)
            end_idx = min(n_samples, center + half_window)

            Y_window = Y[start_idx:end_idx]
            T_window = T[start_idx:end_idx]
            X_window = X[start_idx:end_idx]

            if len(Y_window) < 30:
                skipped.append((center, f"window too small (n={len(Y_window)} < 30)"))
                continue

            # Fit TemporalPLRDML on window
            try:
                dml = TemporalPLRDML(
                    n_lags=0,  # No lags for rolling window
                    model_y=self.model_y,
                    model_t=self.model_t,
                    cv_strategy="time_series_split",
                    n_splits=min(self.n_folds, len(Y_window) // 10),
                    hac_bandwidth=self.hac_bandwidth,
                    random_state=self.random_state,
                )
                result = dml.fit(Y_window, T_window, X_window)
            except ValueError as e:
                skipped.append((center, str(e)))
                continue
            # Centers are appended ONLY on success: truncating the center
            # list to len(theta_list) silently relabeled every estimate after
            # a skipped middle window with the wrong timestamp (issue #10).
            theta_list.append(result.theta)
            se_list.append(result.se)
            kept_centers.append(center)

        if skipped:
            details = "; ".join(f"center {c}: {msg}" for c, msg in skipped[:5])
            more = f" (+{len(skipped) - 5} more)" if len(skipped) > 5 else ""
            warnings.warn(
                f"RollingWindowDML skipped {len(skipped)} of {len(centers)} "
                f"windows — {details}{more}. Reported estimates cover only "
                "the successful windows at their correct centers.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._theta_series = np.array(theta_list)
        self._se_series = np.array(se_list)
        self._time_centers = np.array(kept_centers)
        self._is_fitted = True

        return self

    def get_effects(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Get time series of local treatment effects.

        Returns:
            Tuple of (time_centers, theta_series, se_series)

        Raises:
            ValueError: If model not fitted
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first. Call fit().")

        assert self._time_centers is not None
        assert self._theta_series is not None
        assert self._se_series is not None

        return self._time_centers, self._theta_series, self._se_series


class PanelDML:
    """Panel DML with fixed effects.

    Implements DML for panel data with:
    - Individual fixed effects (within transformation)
    - Time fixed effects
    - Two-way fixed effects
    - Cluster-robust standard errors

    The algorithm:
    1. Demean data by individual/time (fixed effects transformation)
    2. Run DML on transformed data
    3. Compute cluster-robust standard errors

    Args:
        fixed_effects: Type of fixed effects
        cluster_se: Whether to use cluster-robust SE
        model_y: ML model for outcome nuisance function
        model_t: ML model for treatment nuisance function
        n_folds: Number of CV folds
        random_state: Random seed

    Examples:
        >>> import numpy as np
        >>> from dml_ts.dml import PanelDML
        >>> np.random.seed(0)
        >>> n_units, m = 40, 5
        >>> individual_id = np.repeat(np.arange(n_units), m)
        >>> time_id = np.tile(np.arange(m), n_units)
        >>> N = n_units * m
        >>> X = np.random.randn(N, 2)
        >>> T = 0.5 * X[:, 0] + np.random.randn(N)
        >>> Y = 2.0 * T + X[:, 1] + np.random.randn(N)
        >>> import warnings
        >>> model = PanelDML(fixed_effects="individual", model_y="ridge",
        ...                  model_t="ridge", n_folds=3)
        >>> with warnings.catch_warnings():  # early temporal-CV rows are uncovered
        ...     warnings.simplefilter("ignore", RuntimeWarning)
        ...     result = model.fit(Y, T, X, individual_id, time_id)
        >>> isinstance(result.theta, float)
        True
    """

    def __init__(
        self,
        fixed_effects: Literal["individual", "time", "twoway"] = "individual",
        cluster_se: bool = True,
        model_y: ModelType = "random_forest",
        model_t: ModelType = "random_forest",
        n_folds: int = 5,
        random_state: int | None = None,
    ):
        """Initialize PanelDML."""
        self.fixed_effects = fixed_effects
        self.cluster_se = cluster_se
        self.model_y = model_y
        self.model_t = model_t
        self.n_folds = n_folds
        self.random_state = random_state

        self._result: TemporalPLRDMLResult | None = None
        self._is_fitted = False

    def _demean_by_group(
        self,
        data: NDArray[np.float64],
        groups: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Demean data by group (within transformation)."""
        result = data.copy()
        unique_groups = np.unique(groups)

        for g in unique_groups:
            mask = groups == g
            if data.ndim == 1:
                result[mask] = data[mask] - np.mean(data[mask])
            else:
                result[mask] = data[mask] - np.mean(data[mask], axis=0)

        return result

    def fit(
        self,
        Y: ArrayLike,
        T: ArrayLike,
        X: ArrayLike,
        individual_id: ArrayLike,
        time_id: ArrayLike,
    ) -> TemporalPLRDMLResult:
        """Fit panel DML model.

        Args:
            Y: Outcome variable (n_samples,)
            T: Treatment variable (n_samples,)
            X: Confounders (n_samples, n_features)
            individual_id: Individual identifiers
            time_id: Time period identifiers

        Returns:
            TemporalPLRDMLResult with treatment effect and cluster-robust SE
        """
        Y = np.asarray(Y, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        individual_id = np.asarray(individual_id, dtype=np.int64)
        time_id = np.asarray(time_id, dtype=np.int64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Apply fixed effects transformation
        if self.fixed_effects == "individual":
            Y_fe = self._demean_by_group(Y, individual_id)
            T_fe = self._demean_by_group(T, individual_id)
            X_fe = self._demean_by_group(X, individual_id)
        elif self.fixed_effects == "time":
            Y_fe = self._demean_by_group(Y, time_id)
            T_fe = self._demean_by_group(T, time_id)
            X_fe = self._demean_by_group(X, time_id)
        elif self.fixed_effects == "twoway":
            # Two-way: demean by both individual and time
            Y_fe = self._demean_by_group(Y, individual_id)
            Y_fe = self._demean_by_group(Y_fe, time_id)
            T_fe = self._demean_by_group(T, individual_id)
            T_fe = self._demean_by_group(T_fe, time_id)
            X_fe = self._demean_by_group(X, individual_id)
            X_fe = self._demean_by_group(X_fe, time_id)
        else:
            raise ValueError(f"Unknown fixed_effects: {self.fixed_effects}")

        # Run TemporalPLRDML on transformed data
        dml = TemporalPLRDML(
            n_lags=0,
            model_y=self.model_y,
            model_t=self.model_t,
            cv_strategy="time_series_split",
            n_splits=self.n_folds,
            random_state=self.random_state,
        )

        # time_index is accepted for API compatibility only — splitting is
        # positional, so rows must already be in temporal order per unit
        result = dml.fit(Y_fe, T_fe, X_fe, time_index=time_id.astype(np.float64))

        # If cluster SE requested, recompute with clustering
        if self.cluster_se and result.influence_scores is not None:
            # Cluster by individual for standard errors
            unique_individuals = np.unique(individual_id)
            n_clusters = len(unique_individuals)

            if n_clusters < 2:
                raise ValueError(
                    f"Cluster-robust SEs require at least 2 clusters, got {n_clusters}. "
                    "Use cluster_se=False for a single cross-sectional unit."
                )

            # Compute cluster-level influence sums. The rows dropped by lag
            # construction and temporal-CV filtering form a contiguous prefix
            # under the hardcoded time_series_split strategy, so clusters
            # align as mask[cv_start:]. (Explicit retained-row indices on the
            # result will replace this bookkeeping in the B1 result refactor.)
            n_eff = len(result.influence_scores)
            cv_start = result.lagged_rows_dropped + result.dropped_initial_rows
            cluster_sums: list[float] = []
            n_empty_clusters = 0
            for ind in unique_individuals:
                mask_eff = (individual_id == ind)[cv_start:]
                if not mask_eff.any():
                    # Cluster fell entirely inside the dropped prefix: it
                    # contributes nothing to theta-hat and must not count
                    # toward the cluster degrees of freedom.
                    n_empty_clusters += 1
                    continue
                cluster_sums.append(float(np.sum(result.influence_scores[mask_eff])))

            n_clusters_eff = len(cluster_sums)
            if n_empty_clusters > 0:
                warnings.warn(
                    f"{n_empty_clusters} of {n_clusters} clusters have no observations "
                    "retained after lag/temporal-CV trimming; cluster-robust inference "
                    f"uses the remaining {n_clusters_eff} clusters.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if n_clusters_eff < 2:
                raise ValueError(
                    "Cluster-robust SEs require at least 2 clusters with observations "
                    f"retained after lag/temporal-CV trimming, got {n_clusters_eff} "
                    f"(of {n_clusters} nominal). Use cluster_se=False or reduce n_folds."
                )

            # Cluster-robust (CR1) variance of theta-hat: theta_hat - theta
            # ~= mean(psi) over the n_eff fitted observations, so
            # Var = G/(G-1) * sum_g S_g^2 / n_eff^2 where S_g are the
            # within-cluster SUMS of psi (already centered: sum_i psi_i = 0
            # by FWL construction) and G counts retained clusters. Treating
            # the sums as cluster means overstated the SE by ~the effective
            # cluster size (issue #9).
            cluster_psi = np.asarray(cluster_sums)
            cluster_var = (
                (n_clusters_eff / (n_clusters_eff - 1)) * float(np.sum(cluster_psi**2)) / (n_eff**2)
            )
            finite_se(cluster_var, name="cluster-robust variance")
            cluster_se = float(np.sqrt(cluster_var))

            # Degenerate cluster SEs must raise, never fabricate t=0/p=1.0
            # (issue #11; the <2-retained-clusters guard above catches the
            # all-trimmed case, this catches degenerate-by-value).
            finite_se(cluster_se, name="cluster-robust standard error")

            # Update result with cluster SE
            z_crit = stats.norm.ppf(0.975)
            result = TemporalPLRDMLResult(
                theta=result.theta,
                se=cluster_se,
                t_stat=result.theta / cluster_se,
                ci_lower=result.theta - z_crit * cluster_se,
                ci_upper=result.theta + z_crit * cluster_se,
                p_value=float(2 * (1 - stats.norm.cdf(abs(result.theta / cluster_se)))),
                n_samples=result.n_samples,
                n_periods=result.n_periods,
                outcome_r2_cv=result.outcome_r2_cv,
                treatment_r2_cv=result.treatment_r2_cv,
                hac_bandwidth=result.hac_bandwidth,
                cv_strategy=f"{result.cv_strategy}_clustered",
                Y_residual=result.Y_residual,
                T_residual=result.T_residual,
                influence_scores=result.influence_scores,
                dropped_initial_rows=result.dropped_initial_rows,
                lagged_rows_dropped=result.lagged_rows_dropped,
            )

        self._result = result
        self._is_fitted = True
        return result


def demonstrate_temporal_plr_dml(seed: int = 42) -> None:
    """Demonstrate TemporalPLRDML on synthetic time series data.

    Generates a DGP with:
    - Autocorrelated treatment
    - Time-varying confounders
    - True treatment effect θ = 2.0
    """
    np.random.seed(seed)

    print("=" * 70)
    print("TemporalPLRDML: Time Series Double Machine Learning")
    print("=" * 70)
    print()

    # Generate synthetic time series
    n = 500
    time = np.arange(n)

    # Confounders with time trends
    X = np.column_stack(
        [
            np.random.randn(n),  # Random confounder
            np.sin(2 * np.pi * time / 100),  # Seasonal confounder
            np.cumsum(np.random.randn(n) * 0.1),  # Random walk
        ]
    )

    # Treatment with confounding and autocorrelation
    T = np.zeros(n)
    T[0] = np.random.randn()
    for t in range(1, n):
        T[t] = 0.3 * T[t - 1] + 0.5 * X[t, 0] + 0.3 * X[t, 1] + np.random.randn()

    # Outcome with true effect θ = 2.0
    true_theta = 2.0
    Y = true_theta * T + 1.5 * X[:, 0] ** 2 + 2.0 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n)

    print(f"DGP: Y = {true_theta}·T + f(X) + ε")
    print(f"     T_t = 0.3·T_{'{t-1}'} + g(X) + η  (autocorrelated)")
    print(f"     n = {n} observations")
    print()

    # Fit TemporalPLRDML
    print("Fitting TemporalPLRDML with HAC standard errors...")
    model = TemporalPLRDML(
        n_lags=3,
        model_y="random_forest",
        model_t="random_forest",
        cv_strategy="time_series_split",
        n_splits=5,
        gap=0,
        hac_kernel="bartlett",
        random_state=seed,
    )

    result = model.fit(Y, T, X, time_index=time)

    print(result.summary())

    # Compare with true effect
    print("Comparison with True Effect:")
    print(f"  True θ:       {true_theta:.4f}")
    print(f"  Estimated θ:  {result.theta:.4f}")
    print(f"  Bias:         {result.theta - true_theta:.4f}")
    print(f"  Coverage:     {'✓' if result.ci_lower <= true_theta <= result.ci_upper else '✗'}")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_temporal_plr_dml()
