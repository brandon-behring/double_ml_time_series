"""
HAC (Heteroskedasticity and Autocorrelation Consistent) Covariance Estimation.

Implements Newey-West and related HAC estimators for valid inference
in time series DML with autocorrelated residuals.

The standard OLS variance estimator assumes homoskedasticity and no
autocorrelation. When these assumptions fail (common in time series),
HAC estimators provide consistent variance estimates by accounting for
the serial correlation structure.

References:

- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance matrix.
  Econometrica, 55(3), 703-708.
- Andrews, D. W. (1991). Heteroskedasticity and autocorrelation consistent
  covariance matrix estimation. Econometrica, 59(3), 817-858.
- Newey, W. K., & West, K. D. (1994). Automatic lag selection in
  covariance matrix estimation. Review of Economic Studies, 61(4).

Usage:
    >>> from src.dml.hac import newey_west_se, HACEstimator
    >>> # Simple usage for single coefficient
    >>> se = newey_west_se(residuals, bandwidth=10)
    >>> # Full covariance matrix
    >>> cov = newey_west_covariance(residuals, X, bandwidth="auto")
    >>> # Object-oriented interface
    >>> hac = HACEstimator(kernel="bartlett", bandwidth="auto")
    >>> hac.fit(residuals, X)
    >>> se = hac.get_se()
"""

from typing import Callable, Literal, Optional, Union, cast
from dataclasses import dataclass
import numpy as np
from scipy import stats

KernelType = Literal["bartlett", "parzen", "quadratic_spectral"]
BandwidthMethod = Literal["auto", "newey_west", "andrews"]


@dataclass
class HACResult:
    """Result container for HAC estimation.

    Attributes:
        variance: HAC variance estimate (scalar for single coef)
        se: HAC standard error
        covariance: Full HAC covariance matrix (if X provided)
        bandwidth: Bandwidth used
        kernel: Kernel function used
        n_samples: Number of observations
        effective_dof: Effective degrees of freedom
    """

    variance: float
    se: float
    covariance: Optional[np.ndarray]
    bandwidth: int
    kernel: str
    n_samples: int
    effective_dof: float


def bartlett_kernel(lag: int, bandwidth: int) -> float:
    """Bartlett (triangular) kernel weight.

    The Bartlett kernel is the most common choice for HAC estimation.
    It's simple, always produces positive semi-definite matrices,
    and has good finite-sample properties.

    K(x) = 1 - abs(x) for abs(x) <= 1, else 0

    Args:
        lag: Lag index (can be negative)
        bandwidth: Bandwidth parameter (m)

    Returns:
        Kernel weight in [0, 1]

    Example:
        >>> bartlett_kernel(0, 10)  # Lag 0 always has weight 1
        1.0
        >>> bartlett_kernel(5, 10)  # Weight decreases with lag
        0.545...
        >>> bartlett_kernel(15, 10)  # Beyond bandwidth = 0
        0.0
    """
    if bandwidth <= 0:
        return 1.0 if lag == 0 else 0.0
    if abs(lag) <= bandwidth:
        return 1.0 - abs(lag) / (bandwidth + 1)
    return 0.0


def parzen_kernel(lag: int, bandwidth: int) -> float:
    """Parzen kernel weight.

    The Parzen kernel has a flatter top than Bartlett, giving more
    weight to intermediate lags. It's also positive semi-definite.

    Args:
        lag: Lag index
        bandwidth: Bandwidth parameter

    Returns:
        Kernel weight in [0, 1]
    """
    if bandwidth <= 0:
        return 1.0 if lag == 0 else 0.0

    x = abs(lag) / (bandwidth + 1)
    if x <= 0.5:
        return 1.0 - 6 * x**2 + 6 * x**3
    elif x <= 1.0:
        return 2.0 * (1 - x) ** 3
    return 0.0


def quadratic_spectral_kernel(lag: int, bandwidth: int) -> float:
    """Quadratic spectral (QS) kernel weight.

    The QS kernel is optimal in a mean squared error sense but
    can produce negative weights, requiring careful handling.

    Args:
        lag: Lag index
        bandwidth: Bandwidth parameter

    Returns:
        Kernel weight (can be slightly negative for large lags)
    """
    if lag == 0:
        return 1.0
    if bandwidth <= 0:
        return 0.0

    x = 6 * np.pi * lag / (5 * (bandwidth + 1))
    return float(3 * (np.sin(x) / x - np.cos(x)) / x**2)


def get_kernel_function(kernel: KernelType) -> Callable[[int, int], float]:
    """Get kernel function by name.

    Args:
        kernel: Kernel name

    Returns:
        Kernel function

    Raises:
        ValueError: If unknown kernel name
    """
    kernels = {
        "bartlett": bartlett_kernel,
        "parzen": parzen_kernel,
        "quadratic_spectral": quadratic_spectral_kernel,
    }
    if kernel not in kernels:
        raise ValueError(f"Unknown kernel '{kernel}'. Must be one of {list(kernels.keys())}")
    return kernels[kernel]


def optimal_bandwidth(
    residuals: np.ndarray,
    method: BandwidthMethod = "newey_west",
    kernel: KernelType = "bartlett",
) -> int:
    """Compute optimal bandwidth for HAC estimation.

    Implements automatic bandwidth selection:
    - "newey_west": Rule of thumb m = floor(T^(1/3))
    - "andrews": Data-driven plug-in based on AR(1) approximation
    - "auto": Uses newey_west method

    Args:
        residuals: Regression residuals (n_samples,)
        method: Bandwidth selection method
        kernel: Kernel function (affects Andrews method)

    Returns:
        Optimal bandwidth (integer >= 0)

    Example:
        >>> residuals = np.random.randn(100)
        >>> optimal_bandwidth(residuals, method="newey_west")
        4
        >>> optimal_bandwidth(residuals, method="andrews")
        3

    References:
        Newey & West (1994), Andrews (1991)
    """
    residuals = np.asarray(residuals).ravel()
    n = len(residuals)

    if n < 2:
        return 0

    if method in ("newey_west", "auto"):
        # Newey-West rule of thumb: m = floor(T^(1/3))
        # Conservative choice that works well in practice
        bandwidth = int(np.floor(n ** (1 / 3)))
        return max(0, bandwidth)

    elif method == "andrews":
        # Andrews (1991) data-driven bandwidth
        # Based on AR(1) approximation to residuals

        # Fit AR(1) to get autocorrelation
        if n < 3:
            return 0

        # Estimate rho from AR(1): e_t = rho * e_{t-1} + u_t
        e_lag = residuals[:-1]
        e_curr = residuals[1:]
        rho = np.corrcoef(e_lag, e_curr)[0, 1]

        # Bound rho to avoid numerical issues
        rho = np.clip(rho, -0.99, 0.99)

        if kernel == "bartlett":
            # Optimal bandwidth for Bartlett kernel
            # m* = 1.1447 * (alpha * T)^(1/3)
            # where alpha = 4*rho^2 / (1-rho)^4 for AR(1)
            alpha_num = 4 * rho**2
            alpha_den = (1 - rho) ** 4
            if alpha_den < 1e-10:
                alpha = 0.0
            else:
                alpha = alpha_num / alpha_den
            bandwidth = int(1.1447 * (alpha * n) ** (1 / 3))

        elif kernel == "parzen":
            # Optimal for Parzen: m* = 2.6614 * (alpha * T)^(1/5)
            alpha_num = 4 * rho**2
            alpha_den = (1 - rho) ** 4
            if alpha_den < 1e-10:
                alpha = 0.0
            else:
                alpha = alpha_num / alpha_den
            bandwidth = int(2.6614 * (alpha * n) ** (1 / 5))

        else:  # quadratic_spectral
            # Optimal for QS: m* = 1.3221 * (alpha * T)^(1/5)
            alpha_num = 4 * rho**2
            alpha_den = (1 - rho) ** 4
            if alpha_den < 1e-10:
                alpha = 0.0
            else:
                alpha = alpha_num / alpha_den
            bandwidth = int(1.3221 * (alpha * n) ** (1 / 5))

        return max(0, min(bandwidth, n - 1))

    else:
        raise ValueError(
            f"Unknown bandwidth method '{method}'. Use 'newey_west', 'andrews', or 'auto'"
        )


def _compute_long_run_variance(
    residuals: np.ndarray,
    bandwidth: int,
    kernel_func: Callable[[int, int], float],
) -> float:
    """Compute long-run variance of residuals.

    The long-run variance accounts for autocorrelation:
    Ω = γ₀ + 2 * Σⱼ₌₁ᵐ w(j) * γⱼ

    where γⱼ is the j-th autocovariance and w(j) is the kernel weight.

    Args:
        residuals: Regression residuals
        bandwidth: Lag truncation parameter
        kernel_func: Kernel weighting function

    Returns:
        Long-run variance estimate
    """
    n = len(residuals)
    residuals = residuals - np.mean(residuals)

    # Lag 0: variance
    gamma_0 = np.sum(residuals**2) / n

    # Add weighted autocovariances
    long_run_var = gamma_0

    for j in range(1, bandwidth + 1):
        if j >= n:
            break
        # Autocovariance at lag j
        gamma_j = np.sum(residuals[j:] * residuals[:-j]) / n
        weight = kernel_func(j, bandwidth)
        # Add both j and -j (symmetric)
        long_run_var += 2 * weight * gamma_j

    return float(max(0.0, long_run_var))  # Ensure non-negative


def newey_west_se(
    residuals: np.ndarray,
    X: Optional[np.ndarray] = None,
    bandwidth: Union[int, str] = "auto",
    kernel: KernelType = "bartlett",
) -> float:
    """Compute Newey-West HAC standard error.

    For a simple mean or single regression coefficient, computes the
    HAC-adjusted standard error that accounts for heteroskedasticity
    and autocorrelation.

    Args:
        residuals: Regression residuals (n_samples,)
        X: Regressors (optional). If None, computes SE for mean.
        bandwidth: Lag truncation parameter (int or "auto")
        kernel: Kernel function ("bartlett", "parzen", "quadratic_spectral")

    Returns:
        HAC-adjusted standard error

    Example:
        >>> np.random.seed(42)
        >>> # AR(1) residuals with rho=0.5
        >>> e = np.zeros(100)
        >>> for t in range(1, 100):
        ...     e[t] = 0.5 * e[t-1] + np.random.randn()
        >>> se_ols = np.std(e) / np.sqrt(100)  # Naive SE
        >>> se_hac = newey_west_se(e, bandwidth=5)  # HAC SE
        >>> se_hac > se_ols  # HAC should be larger with positive autocorr
        True
    """
    residuals = np.asarray(residuals).ravel()
    n = len(residuals)

    if n < 2:
        raise ValueError(f"Need at least 2 observations, got {n}")

    # Determine bandwidth
    if bandwidth is None or isinstance(bandwidth, str):
        method = cast(BandwidthMethod, bandwidth) if isinstance(bandwidth, str) else "andrews"
        bw = optimal_bandwidth(residuals, method=method, kernel=kernel)
    else:
        bw = int(bandwidth)

    bw = max(0, min(bw, n - 1))

    # Get kernel function
    kernel_func = get_kernel_function(kernel)

    if X is None:
        # Simple case: SE for the mean
        # SE = sqrt(Ω / n) where Ω is long-run variance
        long_run_var = _compute_long_run_variance(residuals, bw, kernel_func)
        return float(np.sqrt(long_run_var / n))
    else:
        # Regression case: use sandwich estimator
        # V = (X'X)^{-1} * Ω * (X'X)^{-1}
        # where Ω = Σᵢ Σⱼ w(abs(i-j)/m) * eᵢ * eⱼ * xᵢ * xⱼ'
        cov = newey_west_covariance(residuals, X, bandwidth=bw, kernel=kernel)
        # Return SE for first coefficient (or only coefficient)
        return float(np.sqrt(cov[0, 0]))


def newey_west_covariance(
    residuals: np.ndarray,
    X: np.ndarray,
    bandwidth: Union[int, str] = "auto",
    kernel: KernelType = "bartlett",
) -> np.ndarray:
    """Compute Newey-West HAC covariance matrix.

    Computes the sandwich covariance estimator:
    V = (X'X)⁻¹ * Ω * (X'X)⁻¹

    where Ω is the HAC "meat":
    Ω = Σᵢ Σⱼ w(abs(i-j)/m) * eᵢ * eⱼ * xᵢ * xⱼ'

    Args:
        residuals: Regression residuals (n_samples,)
        X: Design matrix (n_samples, n_features)
        bandwidth: Lag truncation parameter (int or "auto")
        kernel: Kernel function name

    Returns:
        HAC covariance matrix (n_features, n_features)

    Example:
        >>> np.random.seed(42)
        >>> n, k = 100, 3
        >>> X = np.random.randn(n, k)
        >>> X[:, 0] = 1  # Intercept
        >>> e = np.random.randn(n)
        >>> cov = newey_west_covariance(e, X, bandwidth=5)
        >>> cov.shape
        (3, 3)
    """
    residuals = np.asarray(residuals).ravel()
    X = np.asarray(X)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, k = X.shape

    if len(residuals) != n:
        raise ValueError(f"residuals length ({len(residuals)}) != X rows ({n})")

    if n < k:
        raise ValueError(f"Need n >= k, got n={n}, k={k}")

    # Determine bandwidth
    if bandwidth is None or isinstance(bandwidth, str):
        method = cast(BandwidthMethod, bandwidth) if isinstance(bandwidth, str) else "andrews"
        bw = optimal_bandwidth(residuals, method=method, kernel=kernel)
    else:
        bw = int(bandwidth)

    bw = max(0, min(bw, n - 1))

    # Get kernel function
    kernel_func = get_kernel_function(kernel)

    # Compute (X'X)^{-1}
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if singular
        XtX_inv = np.linalg.pinv(XtX)

    # Compute HAC "meat": Ω = Σⱼ w(j) * Σᵢ (eᵢ * xᵢ) (eᵢ₊ⱼ * xᵢ₊ⱼ)'
    # Start with lag 0 (heteroskedasticity-consistent part)
    u = residuals.reshape(-1, 1) * X  # n x k matrix of e_i * x_i
    omega = u.T @ u  # k x k

    # Add autocovariance terms
    for j in range(1, bw + 1):
        if j >= n:
            break
        weight = kernel_func(j, bw)
        # Cross-product at lag j
        gamma_j = u[j:].T @ u[:-j]  # k x k
        # Add both directions (symmetric)
        omega += weight * (gamma_j + gamma_j.T)

    # Sandwich: V = (X'X)^{-1} * Ω * (X'X)^{-1}
    cov = XtX_inv @ omega @ XtX_inv

    # Ensure symmetric (numerical precision)
    cov = np.asarray((cov + cov.T) / 2, dtype=np.float64)

    return cov


class HACEstimator:
    """HAC covariance estimator for DML inference.

    Provides a unified object-oriented interface for HAC estimation with:
    - Multiple kernel options (Bartlett, Parzen, Quadratic Spectral)
    - Automatic or manual bandwidth selection
    - Optional pre-whitening for improved finite-sample performance

    The estimator follows a fit/transform pattern:
    1. fit() computes the HAC covariance from residuals
    2. get_se() / get_covariance() retrieve results

    Args:
        kernel: Kernel function ("bartlett", "parzen", "quadratic_spectral")
        bandwidth: Lag truncation or "auto" for automatic selection
        prewhiten: Whether to apply AR(1) prewhitening (reduces bias)

    Example:
        >>> hac = HACEstimator(kernel="bartlett", bandwidth="auto")
        >>> hac.fit(residuals, X)
        >>> se = hac.get_se()
        >>> cov = hac.get_covariance()
    """

    def __init__(
        self,
        kernel: KernelType = "bartlett",
        bandwidth: Union[int, str] = "auto",
        prewhiten: bool = False,
    ):
        """Initialize HACEstimator.

        Args:
            kernel: Kernel function name
            bandwidth: Lag truncation or "auto"
            prewhiten: Whether to apply AR(1) prewhitening

        Raises:
            ValueError: If invalid kernel name
        """
        valid_kernels = {"bartlett", "parzen", "quadratic_spectral"}
        if kernel not in valid_kernels:
            raise ValueError(f"kernel must be one of {valid_kernels}, got {kernel}")

        self.kernel = kernel
        self.bandwidth = bandwidth
        self.prewhiten = prewhiten

        # Fitted attributes
        self._variance: Optional[float] = None
        self._covariance: Optional[np.ndarray] = None
        self._bandwidth_used: Optional[int] = None
        self._n_samples: Optional[int] = None
        self._is_fitted: bool = False

    def fit(
        self,
        residuals: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> "HACEstimator":
        """Fit the HAC estimator.

        Computes the HAC variance/covariance from regression residuals.

        Args:
            residuals: Regression residuals (n_samples,)
            X: Design matrix (n_samples, n_features). If None,
               computes variance for the mean.

        Returns:
            self: Fitted estimator

        Raises:
            ValueError: If residuals invalid
        """
        residuals = np.asarray(residuals).ravel()
        self._n_samples = len(residuals)

        if self._n_samples < 2:
            raise ValueError(f"Need at least 2 observations, got {self._n_samples}")

        # Determine bandwidth
        if self.bandwidth is None or isinstance(self.bandwidth, str):
            method = (
                cast(BandwidthMethod, self.bandwidth)
                if isinstance(self.bandwidth, str)
                else "andrews"
            )
            self._bandwidth_used = optimal_bandwidth(
                residuals,
                method=method,
                kernel=self.kernel,
            )
        else:
            self._bandwidth_used = int(self.bandwidth)

        self._bandwidth_used = max(0, min(self._bandwidth_used, self._n_samples - 1))

        # Optional prewhitening
        if self.prewhiten and self._n_samples >= 3:
            residuals, ar_coef = self._prewhiten(residuals)

        # Compute HAC estimates
        kernel_func = get_kernel_function(self.kernel)

        if X is None:
            # Simple case: variance of mean
            long_run_var = _compute_long_run_variance(residuals, self._bandwidth_used, kernel_func)
            self._variance = long_run_var / self._n_samples
            self._covariance = None
        else:
            # Regression case: full covariance
            self._covariance = newey_west_covariance(
                residuals, X, bandwidth=self._bandwidth_used, kernel=self.kernel
            )
            self._variance = float(self._covariance[0, 0])

            # Recolor if prewhitened
            if self.prewhiten:
                self._covariance = self._recolor_covariance(self._covariance, ar_coef)
                self._variance = float(self._covariance[0, 0])

        self._is_fitted = True
        return self

    def _prewhiten(self, residuals: np.ndarray) -> tuple[np.ndarray, float]:
        """Apply AR(1) prewhitening to residuals.

        Fits AR(1): e_t = phi * e_{t-1} + u_t
        Returns whitened residuals u_t and AR coefficient phi.

        Args:
            residuals: Original residuals

        Returns:
            Tuple of (whitened_residuals, ar_coefficient)
        """
        e_lag = residuals[:-1]
        e_curr = residuals[1:]

        # OLS estimate of AR(1) coefficient
        phi = np.sum(e_lag * e_curr) / np.sum(e_lag**2)
        phi = np.clip(phi, -0.99, 0.99)

        # Whitened residuals
        u = e_curr - phi * e_lag

        return u, phi

    def _recolor_covariance(self, cov: np.ndarray, ar_coef: float) -> np.ndarray:
        """Recolor covariance after prewhitening.

        Adjusts the HAC covariance to account for the AR(1) filtering.

        Args:
            cov: HAC covariance of whitened residuals
            ar_coef: AR(1) coefficient

        Returns:
            Recolored covariance matrix
        """
        # Recoloring factor: 1 / (1 - phi)^2
        recolor_factor = 1 / (1 - ar_coef) ** 2
        return cov * recolor_factor

    def get_se(self, index: int = 0) -> float:
        """Get HAC standard error.

        Args:
            index: Coefficient index (for multi-parameter case)

        Returns:
            HAC standard error

        Raises:
            RuntimeError: If estimator not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Estimator not fitted. Call fit() first.")

        if self._covariance is not None:
            return float(np.sqrt(self._covariance[index, index]))
        else:
            return float(np.sqrt(self._variance))

    def get_variance(self) -> float:
        """Get HAC variance estimate.

        Returns:
            HAC variance

        Raises:
            RuntimeError: If estimator not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Estimator not fitted. Call fit() first.")
        return self._variance

    def get_covariance(self) -> np.ndarray:
        """Get HAC covariance matrix.

        Returns:
            HAC covariance matrix

        Raises:
            RuntimeError: If estimator not fitted or no X provided
        """
        if not self._is_fitted:
            raise RuntimeError("Estimator not fitted. Call fit() first.")
        if self._covariance is None:
            raise RuntimeError("No covariance matrix. Fit with X to get covariance.")
        return self._covariance

    def get_result(self) -> HACResult:
        """Get complete HAC result.

        Returns:
            HACResult dataclass with all estimates

        Raises:
            RuntimeError: If estimator not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Estimator not fitted. Call fit() first.")

        return HACResult(
            variance=self._variance,
            se=self.get_se(),
            covariance=self._covariance,
            bandwidth=self._bandwidth_used,
            kernel=self.kernel,
            n_samples=self._n_samples,
            effective_dof=float(self._n_samples - self._bandwidth_used),
        )

    @property
    def bandwidth_used(self) -> int:
        """Return the bandwidth actually used (after auto-selection)."""
        if not self._is_fitted:
            raise RuntimeError("Estimator not fitted. Call fit() first.")
        return self._bandwidth_used


def hac_inference(
    theta: float,
    se_hac: float,
    alpha: float = 0.05,
) -> dict:
    """Perform inference with HAC standard errors.

    Computes confidence intervals and p-values using HAC-adjusted
    standard errors.

    Args:
        theta: Point estimate
        se_hac: HAC standard error
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        Dict with: theta, se, ci_lower, ci_upper, t_stat, p_value

    Example:
        >>> result = hac_inference(theta=2.5, se_hac=0.5, alpha=0.05)
        >>> result['ci_lower'], result['ci_upper']
        (1.52..., 3.47...)
    """
    z_crit = stats.norm.ppf(1 - alpha / 2)
    t_stat = theta / se_hac if se_hac > 0 else np.inf
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    return {
        "theta": theta,
        "se": se_hac,
        "ci_lower": theta - z_crit * se_hac,
        "ci_upper": theta + z_crit * se_hac,
        "t_stat": t_stat,
        "p_value": p_value,
    }
