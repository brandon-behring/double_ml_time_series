"""
Stationarity Tests for Time Series DML Validation.

Implements diagnostic tests to verify stationarity assumptions
required for valid time series DML inference.

Implementations are from mathematical primitives (NumPy/SciPy) with
verification against statsmodels for accuracy.

Tests included:
1. ADF (Augmented Dickey-Fuller) - Unit root test
2. KPSS - Stationarity test (null is stationarity)
3. Phillips-Perron - Robust unit root test

Mathematical Foundations:

ADF Test:
    ΔY_t = α + β*t + γ*Y_{t-1} + Σ_{i=1}^{p} δ_i*ΔY_{t-i} + ε_t
    H0: γ = 0 (unit root, non-stationary)
    H1: γ < 0 (stationary)

KPSS Test:
    Y_t = ξ*t + r_t + ε_t, where r_t is a random walk
    LM = Σ S_t² / (T² * σ²)
    H0: σ²_η = 0 (stationary)
    H1: σ²_η > 0 (unit root)

Phillips-Perron Test:
    Same as ADF regression but uses Newey-West HAC correction
    for serial correlation in errors.

References:
- Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for
  autoregressive time series with a unit root.
- Kwiatkowski, D., et al. (1992). Testing the null hypothesis of stationarity
  against the alternative of a unit root.
- Phillips, P.C.B. & Perron, P. (1988). Testing for a unit root in time series
  regression. Biometrika 75, 335-346.

Usage:
    >>> from src.validation.stationarity import StationarityDiagnostic
    >>> diag = StationarityDiagnostic()
    >>> result = diag.test(Y, method="adf")
    >>> print(f"Stationary: {result.is_stationary}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.interpolate import interp1d


TestMethod = Literal["adf", "kpss", "pp"]


# =============================================================================
# Critical Value Tables
# =============================================================================

# ADF critical values from MacKinnon (1996)
# Interpolated for different sample sizes
# Keys: (regression_type, significance_level)
# Format: (tau_nc, tau_c, tau_ct) for n=[25, 50, 100, 250, 500, inf]
ADF_CRITICAL_VALUES = {
    # No constant, no trend ("n")
    ("n", 0.01): [-2.66, -2.62, -2.60, -2.58, -2.58, -2.58],
    ("n", 0.05): [-1.95, -1.95, -1.95, -1.95, -1.95, -1.95],
    ("n", 0.10): [-1.60, -1.61, -1.61, -1.62, -1.62, -1.62],
    # Constant only ("c")
    ("c", 0.01): [-3.75, -3.58, -3.51, -3.46, -3.44, -3.43],
    ("c", 0.05): [-2.99, -2.93, -2.89, -2.88, -2.87, -2.86],
    ("c", 0.10): [-2.64, -2.60, -2.58, -2.57, -2.57, -2.57],
    # Constant and trend ("ct")
    ("ct", 0.01): [-4.38, -4.15, -4.04, -3.99, -3.98, -3.96],
    ("ct", 0.05): [-3.60, -3.50, -3.45, -3.43, -3.42, -3.41],
    ("ct", 0.10): [-3.24, -3.18, -3.15, -3.13, -3.13, -3.12],
}

# KPSS critical values from Kwiatkowski et al. (1992)
# For LM statistic (asymptotic)
KPSS_CRITICAL_VALUES = {
    # Level stationarity ("c")
    ("c", 0.10): 0.347,
    ("c", 0.05): 0.463,
    ("c", 0.025): 0.574,
    ("c", 0.01): 0.739,
    # Trend stationarity ("ct")
    ("ct", 0.10): 0.119,
    ("ct", 0.05): 0.146,
    ("ct", 0.025): 0.176,
    ("ct", 0.01): 0.216,
}

# Sample sizes for critical value interpolation
SAMPLE_SIZES = np.array([25, 50, 100, 250, 500, 10000])


def _interpolate_critical_value(regression: str, alpha: float, n: int) -> float:
    """Interpolate ADF critical value for given sample size.

    Args:
        regression: "n", "c", or "ct"
        alpha: Significance level (0.01, 0.05, or 0.10)
        n: Sample size

    Returns:
        Interpolated critical value
    """
    key = (regression, alpha)
    if key not in ADF_CRITICAL_VALUES:
        raise ValueError(f"Invalid regression/alpha combination: {key}")

    cvs = np.array(ADF_CRITICAL_VALUES[key])
    # Log-linear interpolation
    interp_func = interp1d(np.log(SAMPLE_SIZES), cvs, kind="linear", fill_value="extrapolate")
    return float(interp_func(np.log(n)))


# =============================================================================
# Result Data Classes
# =============================================================================


@dataclass
class StationarityResult:
    """Result container for stationarity tests.

    Attributes:
        test_name: Name of the test performed
        statistic: Test statistic value
        p_value: P-value for the test
        critical_values: Dict of critical values at different significance levels
        is_stationary: Boolean indicating stationarity conclusion
        n_lags: Number of lags used in the test
        regression_type: Regression type (constant, trend, etc.)
        interpretation: Human-readable interpretation
    """

    test_name: str
    statistic: float
    p_value: float
    critical_values: Dict[str, float]
    is_stationary: bool
    n_lags: int
    regression_type: str
    interpretation: str

    def __repr__(self) -> str:
        return (
            f"StationarityResult({self.test_name}: stat={self.statistic:.4f}, "
            f"p={self.p_value:.4f}, stationary={self.is_stationary})"
        )


@dataclass
class ComprehensiveStationarityResult:
    """Result container for comprehensive stationarity analysis.

    Attributes:
        adf_result: ADF test result
        kpss_result: KPSS test result (optional)
        pp_result: Phillips-Perron test result (optional)
        overall_conclusion: Summary conclusion
        recommendation: Recommended action
    """

    adf_result: StationarityResult
    kpss_result: Optional[StationarityResult]
    pp_result: Optional[StationarityResult]
    overall_conclusion: str
    recommendation: str

    def __repr__(self) -> str:
        return f"ComprehensiveStationarityResult(conclusion='{self.overall_conclusion}')"


# =============================================================================
# Core Test Implementations
# =============================================================================


def _compute_adf_statistic(
    y: NDArray[np.float64],
    max_lags: Optional[int] = None,
    regression: str = "c",
) -> Tuple[float, int, float, NDArray[np.float64]]:
    """Compute ADF test statistic from regression.

    Implements:
        ΔY_t = α + β*t + γ*Y_{t-1} + Σ_{i=1}^{p} δ_i*ΔY_{t-i} + ε_t

    The test statistic is t = γ̂ / SE(γ̂).

    Args:
        y: Time series data
        max_lags: Maximum lags (None for automatic AIC selection)
        regression: "n" (none), "c" (constant), "ct" (constant + trend)

    Returns:
        Tuple of (t_statistic, n_lags, residual_variance, residuals)
    """
    n = len(y)

    # Determine max lags if not specified (Schwert 1989 rule)
    if max_lags is None:
        max_lags = int(12 * (n / 100) ** 0.25)
    max_lags = min(max_lags, n // 3 - 1)  # Ensure enough observations

    # First differences
    dy = np.diff(y)

    # Select optimal lag using AIC
    best_aic = np.inf
    best_lags = 0

    for p in range(max_lags + 1):
        # Build regression matrix
        # Dependent: ΔY_t for t = p+1, ..., T-1
        # Regressors: constant, trend (optional), Y_{t-1}, ΔY_{t-1}, ..., ΔY_{t-p}

        T_eff = n - p - 1  # Effective sample size
        if T_eff < 10:
            continue

        # Dependent variable
        Y_dep = dy[p:]

        # Build design matrix
        X_list = []

        # Constant
        if regression in ("c", "ct"):
            X_list.append(np.ones(T_eff))

        # Trend
        if regression == "ct":
            X_list.append(np.arange(p + 1, n)[:T_eff])

        # Lagged level Y_{t-1}
        X_list.append(y[p : n - 1])

        # Lagged differences ΔY_{t-i} for i = 1, ..., p
        for i in range(1, p + 1):
            X_list.append(dy[p - i : n - 1 - i])

        X = np.column_stack(X_list) if len(X_list) > 1 else X_list[0].reshape(-1, 1)

        # OLS estimation
        try:
            beta, residuals, rank, s = np.linalg.lstsq(X, Y_dep, rcond=None)
        except np.linalg.LinAlgError:
            continue

        # Compute AIC
        resid = Y_dep - X @ beta
        sse = np.sum(resid**2)
        k = X.shape[1]
        aic = T_eff * np.log(sse / T_eff) + 2 * k

        if aic < best_aic:
            best_aic = aic
            best_lags = p

    # Final estimation with optimal lags
    p = best_lags
    T_eff = n - p - 1
    Y_dep = dy[p:]

    X_list = []
    if regression in ("c", "ct"):
        X_list.append(np.ones(T_eff))
    if regression == "ct":
        X_list.append(np.arange(p + 1, n)[:T_eff])
    X_list.append(y[p : n - 1])
    for i in range(1, p + 1):
        X_list.append(dy[p - i : n - 1 - i])

    X = np.column_stack(X_list) if len(X_list) > 1 else X_list[0].reshape(-1, 1)

    # OLS
    beta = np.linalg.lstsq(X, Y_dep, rcond=None)[0]
    resid = Y_dep - X @ beta
    sse = np.sum(resid**2)
    sigma2 = sse / (T_eff - X.shape[1])

    # Coefficient index for γ (lagged level)
    gamma_idx = 0 if regression == "n" else (1 if regression == "c" else 2)

    # Standard error of γ
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        se_gamma = np.sqrt(sigma2 * XtX_inv[gamma_idx, gamma_idx])
    except np.linalg.LinAlgError:
        se_gamma = np.sqrt(sigma2 / T_eff)  # Fallback

    gamma_hat = beta[gamma_idx]
    t_stat = gamma_hat / se_gamma

    return t_stat, best_lags, sigma2, resid


def _compute_adf_pvalue(t_stat: float, n: int, regression: str) -> float:
    """Compute approximate p-value for ADF test using MacKinnon (1996).

    Uses interpolation between tabulated critical values.

    Args:
        t_stat: ADF t-statistic
        n: Sample size
        regression: Regression type

    Returns:
        Approximate p-value
    """
    # Get critical values for this regression and sample size
    cv_01 = _interpolate_critical_value(regression, 0.01, n)
    cv_05 = _interpolate_critical_value(regression, 0.05, n)
    cv_10 = _interpolate_critical_value(regression, 0.10, n)

    # Linear interpolation for p-value
    cvs = np.array([cv_01, cv_05, cv_10])
    pvals = np.array([0.01, 0.05, 0.10])

    if t_stat <= cv_01:
        return 0.001  # Very small
    elif t_stat >= cv_10:
        return 0.20 + 0.10 * (t_stat - cv_10)  # Extrapolate
    else:
        # Interpolate
        interp_func = interp1d(cvs, pvals, kind="linear", fill_value="extrapolate")
        return float(np.clip(interp_func(t_stat), 0.001, 0.999))


def _compute_kpss_statistic(
    y: NDArray[np.float64],
    regression: str = "c",
    n_lags: Optional[int] = None,
) -> Tuple[float, int, float]:
    """Compute KPSS stationarity test statistic.

    Implements:
        LM = (1/T²) * Σ_{t=1}^{T} S_t² / σ̂²

    where S_t = Σ_{s=1}^{t} e_s is the partial sum of residuals from the
    level or trend regression.

    Args:
        y: Time series data
        regression: "c" for level stationarity, "ct" for trend stationarity
        n_lags: Number of lags for HAC estimation (None for automatic)

    Returns:
        Tuple of (LM_statistic, n_lags_used, long_run_variance)
    """
    n = len(y)

    # Automatic lag selection (Schwert rule)
    if n_lags is None:
        n_lags = int(4 * (n / 100) ** 0.25)

    # Detrend the series
    if regression == "c":
        # Demean only
        resid = y - np.mean(y)
    else:  # "ct"
        # Remove linear trend
        t = np.arange(n)
        X = np.column_stack([np.ones(n), t])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta

    # Compute partial sums
    S = np.cumsum(resid)

    # Long-run variance estimation using Bartlett kernel
    # σ̂² = γ_0 + 2 * Σ_{k=1}^{L} (1 - k/(L+1)) * γ_k
    gamma_0 = np.mean(resid**2)
    lrv = gamma_0

    for k in range(1, n_lags + 1):
        gamma_k = np.mean(resid[k:] * resid[:-k])
        weight = 1 - k / (n_lags + 1)  # Bartlett weight
        lrv += 2 * weight * gamma_k

    # KPSS statistic
    eta = np.sum(S**2) / (n**2 * lrv)

    return eta, n_lags, lrv


def _compute_kpss_pvalue(eta: float, regression: str) -> float:
    """Compute approximate p-value for KPSS test.

    Uses interpolation between tabulated critical values.

    Args:
        eta: KPSS LM statistic
        regression: "c" or "ct"

    Returns:
        Approximate p-value (lower bound)
    """
    # Get critical values
    cv_10 = KPSS_CRITICAL_VALUES[(regression, 0.10)]
    cv_05 = KPSS_CRITICAL_VALUES[(regression, 0.05)]
    cv_025 = KPSS_CRITICAL_VALUES[(regression, 0.025)]
    cv_01 = KPSS_CRITICAL_VALUES[(regression, 0.01)]

    cvs = np.array([cv_10, cv_05, cv_025, cv_01])
    pvals = np.array([0.10, 0.05, 0.025, 0.01])

    if eta <= cv_10:
        return 0.15  # Greater than 10%
    elif eta >= cv_01:
        return 0.005  # Less than 1%
    else:
        interp_func = interp1d(cvs, pvals, kind="linear", fill_value="extrapolate")
        return float(np.clip(interp_func(eta), 0.005, 0.15))


def _compute_pp_statistic(
    y: NDArray[np.float64],
    regression: str = "c",
    n_lags: Optional[int] = None,
) -> Tuple[float, int, float, NDArray[np.float64]]:
    """Compute Phillips-Perron test statistic.

    Uses Newey-West HAC correction to the standard ADF regression.

    The PP statistic adjusts the ADF t-statistic:
        Z_α = T(α̂ - 1) - (1/2)(λ̂² - γ̂₀)/σ̂²

    where λ̂² is the long-run variance and γ̂₀ is the contemporaneous variance.

    Args:
        y: Time series data
        regression: "n", "c", or "ct"
        n_lags: Lags for HAC estimation (None for automatic)

    Returns:
        Tuple of (Z_t statistic, n_lags, long_run_variance, residuals)
    """
    n = len(y)

    # Automatic lag selection
    if n_lags is None:
        n_lags = int(4 * (n / 100) ** 0.25)

    # Build regression: Y_t = α + β*t + ρ*Y_{t-1} + u_t
    Y_dep = y[1:]
    T_eff = n - 1

    X_list = []
    if regression in ("c", "ct"):
        X_list.append(np.ones(T_eff))
    if regression == "ct":
        X_list.append(np.arange(1, n))
    X_list.append(y[:-1])

    X = np.column_stack(X_list) if len(X_list) > 1 else X_list[0].reshape(-1, 1)

    # OLS
    beta = np.linalg.lstsq(X, Y_dep, rcond=None)[0]
    resid = Y_dep - X @ beta
    sse = np.sum(resid**2)
    s2 = sse / T_eff  # Residual variance (biased)

    # Coefficient index for ρ
    rho_idx = 0 if regression == "n" else (1 if regression == "c" else 2)
    rho_hat = beta[rho_idx]

    # Compute contemporaneous variance γ₀
    gamma_0 = np.mean(resid**2)

    # Compute long-run variance using Bartlett kernel (Newey-West)
    lrv = gamma_0
    for k in range(1, n_lags + 1):
        gamma_k = np.mean(resid[k:] * resid[:-k])
        weight = 1 - k / (n_lags + 1)
        lrv += 2 * weight * gamma_k

    # Standard error from OLS
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        se_rho = np.sqrt(s2 * XtX_inv[rho_idx, rho_idx])
    except np.linalg.LinAlgError:
        se_rho = np.sqrt(s2 / T_eff)

    # Phillips-Perron correction
    # Z_t = sqrt(γ₀/λ²) * t - (λ² - γ₀) / (2*λ*σ̂*sqrt(T))
    # where t = (ρ̂ - 1) / SE(ρ̂)

    t_stat_uncorrected = (rho_hat - 1) / se_rho

    # Correction factor
    correction = (lrv - gamma_0) * T_eff / (2 * np.sqrt(lrv) * np.sqrt(sse))

    Z_t = np.sqrt(gamma_0 / lrv) * t_stat_uncorrected - correction

    return Z_t, n_lags, lrv, resid


# =============================================================================
# Main Class
# =============================================================================


class StationarityDiagnostic:
    """Stationarity diagnostic tool for time series DML.

    Performs stationarity tests to validate assumptions for time series DML.
    Non-stationary data requires differencing or detrending before DML.

    Args:
        significance_level: Significance level for hypothesis tests
        max_lags: Maximum lags for ADF test (None for automatic)
        regression: Regression type for tests ("n", "c", "ct")

    Example:
        >>> diag = StationarityDiagnostic(significance_level=0.05)
        >>> result = diag.test(Y, method="adf")
        >>> if not result.is_stationary:
        ...     Y_diff = np.diff(Y)  # Take first difference
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        max_lags: Optional[int] = None,
        regression: str = "c",
    ):
        """Initialize StationarityDiagnostic.

        Args:
            significance_level: Significance level (default 0.05)
            max_lags: Maximum lags for ADF (None for AIC-based selection)
            regression: Regression type ("c", "ct", "n")
        """
        if significance_level <= 0 or significance_level >= 1:
            raise ValueError("significance_level must be in (0, 1)")
        if regression not in ("n", "c", "ct"):
            raise ValueError("regression must be 'n', 'c', or 'ct'")

        self.significance_level = significance_level
        self.max_lags = max_lags
        self.regression = regression

    def test_adf(
        self,
        series: NDArray[np.float64],
        max_lags: Optional[int] = None,
        regression: Optional[str] = None,
    ) -> StationarityResult:
        """Perform Augmented Dickey-Fuller test.

        H0: Series has a unit root (non-stationary)
        H1: Series is stationary

        Reject H0 if t-statistic < critical value (test is one-sided).

        Args:
            series: Time series data
            max_lags: Maximum lags (None for automatic)
            regression: Regression type (None uses instance default)

        Returns:
            StationarityResult with test outcome
        """
        series = np.asarray(series, dtype=np.float64)
        if len(series) < 20:
            raise ValueError("Series too short for ADF test (need >= 20 observations)")

        max_lags = max_lags or self.max_lags
        regression = regression or self.regression

        t_stat, n_lags, sigma2, resid = _compute_adf_statistic(
            series, max_lags=max_lags, regression=regression
        )
        p_value = _compute_adf_pvalue(t_stat, len(series), regression)

        # Critical values
        n = len(series)
        critical_values = {
            "1%": _interpolate_critical_value(regression, 0.01, n),
            "5%": _interpolate_critical_value(regression, 0.05, n),
            "10%": _interpolate_critical_value(regression, 0.10, n),
        }

        # Conclusion: reject H0 (unit root) if t_stat < critical value
        cv = critical_values[f"{int(self.significance_level * 100)}%"]
        is_stationary = t_stat < cv

        # Interpretation
        if is_stationary:
            interpretation = (
                f"Reject H0 (unit root) at {self.significance_level:.0%} level. "
                f"Evidence suggests series is stationary."
            )
        else:
            interpretation = (
                f"Cannot reject H0 (unit root) at {self.significance_level:.0%} level. "
                f"Series may be non-stationary; consider differencing."
            )

        return StationarityResult(
            test_name="ADF",
            statistic=t_stat,
            p_value=p_value,
            critical_values=critical_values,
            is_stationary=is_stationary,
            n_lags=n_lags,
            regression_type=regression,
            interpretation=interpretation,
        )

    def test_kpss(
        self,
        series: NDArray[np.float64],
        regression: Optional[str] = None,
        n_lags: Optional[int] = None,
    ) -> StationarityResult:
        """Perform KPSS stationarity test.

        H0: Series is stationary
        H1: Series has a unit root

        Note: KPSS has opposite null hypothesis to ADF. Use both for
        robust stationarity conclusions.

        Reject H0 if LM statistic > critical value.

        Args:
            series: Time series data
            regression: "c" for level, "ct" for trend stationarity
            n_lags: Number of lags (None for automatic)

        Returns:
            StationarityResult with test outcome
        """
        series = np.asarray(series, dtype=np.float64)
        if len(series) < 20:
            raise ValueError("Series too short for KPSS test (need >= 20 observations)")

        regression = regression or self.regression
        if regression == "n":
            regression = "c"  # KPSS requires at least constant
        if regression not in ("c", "ct"):
            raise ValueError("KPSS regression must be 'c' or 'ct'")

        eta, n_lags_used, lrv = _compute_kpss_statistic(series, regression, n_lags)
        p_value = _compute_kpss_pvalue(eta, regression)

        # Critical values
        critical_values = {
            "10%": KPSS_CRITICAL_VALUES[(regression, 0.10)],
            "5%": KPSS_CRITICAL_VALUES[(regression, 0.05)],
            "2.5%": KPSS_CRITICAL_VALUES[(regression, 0.025)],
            "1%": KPSS_CRITICAL_VALUES[(regression, 0.01)],
        }

        # For KPSS, H0 is stationarity; reject if stat > cv
        # is_stationary = True if we DON'T reject H0
        cv_key = f"{int(self.significance_level * 100)}%"
        if cv_key not in critical_values:
            cv_key = "5%"
        cv = critical_values[cv_key]
        is_stationary = eta < cv

        if is_stationary:
            interpretation = (
                f"Cannot reject H0 (stationarity) at {self.significance_level:.0%} level. "
                f"Evidence suggests series is stationary."
            )
        else:
            interpretation = (
                f"Reject H0 (stationarity) at {self.significance_level:.0%} level. "
                f"Series may be non-stationary; consider differencing."
            )

        return StationarityResult(
            test_name="KPSS",
            statistic=eta,
            p_value=p_value,
            critical_values=critical_values,
            is_stationary=is_stationary,
            n_lags=n_lags_used,
            regression_type=regression,
            interpretation=interpretation,
        )

    def test_phillips_perron(
        self,
        series: NDArray[np.float64],
        regression: Optional[str] = None,
        n_lags: Optional[int] = None,
    ) -> StationarityResult:
        """Perform Phillips-Perron unit root test.

        H0: Series has a unit root (non-stationary)
        H1: Series is stationary

        More robust to heteroskedasticity than ADF due to HAC correction.

        Args:
            series: Time series data
            regression: Regression type
            n_lags: Number of lags for HAC estimation

        Returns:
            StationarityResult with test outcome
        """
        series = np.asarray(series, dtype=np.float64)
        if len(series) < 20:
            raise ValueError("Series too short for PP test (need >= 20 observations)")

        regression = regression or self.regression

        Z_t, n_lags_used, lrv, resid = _compute_pp_statistic(series, regression, n_lags)

        # P-value uses same distribution as ADF
        p_value = _compute_adf_pvalue(Z_t, len(series), regression)

        # Critical values (same as ADF)
        n = len(series)
        critical_values = {
            "1%": _interpolate_critical_value(regression, 0.01, n),
            "5%": _interpolate_critical_value(regression, 0.05, n),
            "10%": _interpolate_critical_value(regression, 0.10, n),
        }

        cv = critical_values[f"{int(self.significance_level * 100)}%"]
        is_stationary = Z_t < cv

        if is_stationary:
            interpretation = (
                f"Reject H0 (unit root) at {self.significance_level:.0%} level. "
                f"Evidence suggests series is stationary."
            )
        else:
            interpretation = (
                f"Cannot reject H0 (unit root) at {self.significance_level:.0%} level. "
                f"Series may be non-stationary; consider differencing."
            )

        return StationarityResult(
            test_name="Phillips-Perron",
            statistic=Z_t,
            p_value=p_value,
            critical_values=critical_values,
            is_stationary=is_stationary,
            n_lags=n_lags_used,
            regression_type=regression,
            interpretation=interpretation,
        )

    def test(
        self,
        series: NDArray[np.float64],
        method: TestMethod = "adf",
        **kwargs: Any,
    ) -> StationarityResult:
        """Perform stationarity test with specified method.

        Args:
            series: Time series data
            method: Test method ("adf", "kpss", "pp")
            **kwargs: Additional arguments for the test

        Returns:
            StationarityResult with test outcome
        """
        if method == "adf":
            return self.test_adf(series, **kwargs)
        elif method == "kpss":
            return self.test_kpss(series, **kwargs)
        elif method == "pp":
            return self.test_phillips_perron(series, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def comprehensive_test(
        self,
        series: NDArray[np.float64],
    ) -> ComprehensiveStationarityResult:
        """Perform comprehensive stationarity analysis.

        Runs ADF, KPSS, and PP tests and synthesizes conclusions:
        - If ADF rejects AND KPSS doesn't reject: Strong evidence of stationarity
        - If ADF doesn't reject AND KPSS rejects: Strong evidence of unit root
        - If both suggest stationary: Likely stationary
        - If neither suggests stationary: Likely non-stationary
        - Mixed results: Inconclusive

        Args:
            series: Time series data

        Returns:
            ComprehensiveStationarityResult with combined analysis
        """
        adf_result = self.test_adf(series)
        kpss_result = self.test_kpss(series)
        pp_result = self.test_phillips_perron(series)

        # Synthesize conclusions
        # ADF: is_stationary=True means rejected H0 (unit root) -> stationary
        # KPSS: is_stationary=True means didn't reject H0 (stationarity) -> stationary

        adf_stationary = adf_result.is_stationary
        kpss_stationary = kpss_result.is_stationary
        pp_stationary = pp_result.is_stationary

        # Count votes for stationarity
        n_stationary = sum([adf_stationary, kpss_stationary, pp_stationary])

        if n_stationary >= 2:
            if n_stationary == 3:
                overall_conclusion = "STATIONARY (strong evidence from all tests)"
                recommendation = "No differencing needed. Proceed with analysis."
            else:
                overall_conclusion = "LIKELY STATIONARY (majority of tests agree)"
                recommendation = "Proceed with caution; monitor residual autocorrelation."
        elif n_stationary == 1:
            overall_conclusion = "INCONCLUSIVE (tests disagree)"
            recommendation = "Consider visual inspection, longer sample, or try first difference."
        else:
            overall_conclusion = "NON-STATIONARY (strong evidence from all tests)"
            recommendation = "Apply first differencing before analysis."

        return ComprehensiveStationarityResult(
            adf_result=adf_result,
            kpss_result=kpss_result,
            pp_result=pp_result,
            overall_conclusion=overall_conclusion,
            recommendation=recommendation,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def run_stationarity_test(
    series: NDArray[np.float64],
    method: TestMethod = "adf",
    significance_level: float = 0.05,
    regression: str = "c",
) -> StationarityResult:
    """Convenience function for stationarity testing.

    Args:
        series: Time series data
        method: Test method ("adf", "kpss", "pp")
        significance_level: Significance level
        regression: Regression type

    Returns:
        StationarityResult

    Example:
        >>> result = run_stationarity_test(Y, method="adf")
        >>> print(f"Stationary: {result.is_stationary}")
    """
    diag = StationarityDiagnostic(significance_level=significance_level, regression=regression)
    return diag.test(series, method=method)


def suggest_differencing_order(
    series: NDArray[np.float64],
    max_d: int = 2,
    significance_level: float = 0.05,
) -> int:
    """Suggest differencing order to achieve stationarity.

    Tests differenced series until stationary or max_d reached.
    Uses ADF test as primary criterion.

    Args:
        series: Time series data
        max_d: Maximum differencing order to try
        significance_level: Significance level for tests

    Returns:
        Suggested differencing order (0, 1, or 2)

    Example:
        >>> d = suggest_differencing_order(Y)
        >>> Y_stationary = np.diff(Y, n=d) if d > 0 else Y
    """
    series = np.asarray(series, dtype=np.float64)
    diag = StationarityDiagnostic(significance_level=significance_level)

    current = series.copy()
    for d in range(max_d + 1):
        if len(current) < 20:
            # Not enough observations after differencing
            return max(0, d - 1)

        result = diag.test_adf(current)
        if result.is_stationary:
            return d

        # Difference for next iteration
        current = np.diff(current)

    # If still not stationary, return max_d
    return max_d
