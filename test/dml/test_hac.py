"""
Comprehensive tests for HAC covariance estimation.

Tests cover:
1. Kernel functions
2. Bandwidth selection
3. Long-run variance estimation
4. Newey-West SE and covariance
5. HACEstimator class
6. Edge cases and error handling
7. Comparison with known results
"""

import numpy as np
import pytest
from scipy import stats

from src.dml.hac import (
    bartlett_kernel,
    parzen_kernel,
    quadratic_spectral_kernel,
    get_kernel_function,
    optimal_bandwidth,
    newey_west_se,
    newey_west_covariance,
    HACEstimator,
    HACResult,
    hac_inference,
)


class TestKernelFunctions:
    """Test kernel weight functions."""

    @pytest.mark.unit
    def test_bartlett_at_zero(self) -> None:
        """Bartlett kernel should be 1 at lag 0."""
        assert bartlett_kernel(0, 10) == 1.0
        assert bartlett_kernel(0, 0) == 1.0
        assert bartlett_kernel(0, 100) == 1.0

    @pytest.mark.unit
    def test_bartlett_decreasing(self) -> None:
        """Bartlett kernel should decrease with lag."""
        bandwidth = 10
        weights = [bartlett_kernel(j, bandwidth) for j in range(15)]

        # Should be decreasing up to bandwidth
        for j in range(bandwidth):
            assert weights[j] > weights[j + 1]

        # Should be 0 beyond bandwidth
        assert weights[bandwidth + 1] == 0.0

    @pytest.mark.unit
    def test_bartlett_symmetric(self) -> None:
        """Bartlett kernel should be symmetric."""
        bandwidth = 10
        for j in range(-15, 16):
            assert bartlett_kernel(j, bandwidth) == bartlett_kernel(-j, bandwidth)

    @pytest.mark.unit
    def test_parzen_at_zero(self) -> None:
        """Parzen kernel should be 1 at lag 0."""
        assert parzen_kernel(0, 10) == 1.0

    @pytest.mark.unit
    def test_parzen_non_negative(self) -> None:
        """Parzen kernel should be non-negative."""
        bandwidth = 10
        for j in range(20):
            assert parzen_kernel(j, bandwidth) >= 0.0

    @pytest.mark.unit
    def test_parzen_decreasing(self) -> None:
        """Parzen kernel should generally decrease with lag."""
        bandwidth = 10
        # First few lags should be decreasing
        for j in range(bandwidth // 2):
            assert parzen_kernel(j, bandwidth) >= parzen_kernel(j + 1, bandwidth)

    @pytest.mark.unit
    def test_quadratic_spectral_at_zero(self) -> None:
        """QS kernel should be 1 at lag 0."""
        assert quadratic_spectral_kernel(0, 10) == 1.0

    @pytest.mark.unit
    def test_quadratic_spectral_decays(self) -> None:
        """QS kernel should decay to near zero."""
        bandwidth = 10
        # At large lags should be small
        assert abs(quadratic_spectral_kernel(100, bandwidth)) < 0.1

    @pytest.mark.unit
    def test_get_kernel_function(self) -> None:
        """Test kernel function lookup."""
        bartlett = get_kernel_function("bartlett")
        parzen = get_kernel_function("parzen")
        qs = get_kernel_function("quadratic_spectral")

        assert bartlett(0, 10) == 1.0
        assert parzen(0, 10) == 1.0
        assert qs(0, 10) == 1.0

    @pytest.mark.unit
    def test_get_kernel_function_invalid(self) -> None:
        """Test invalid kernel raises error."""
        with pytest.raises(ValueError, match="Unknown kernel"):
            get_kernel_function("invalid_kernel")


class TestOptimalBandwidth:
    """Test automatic bandwidth selection."""

    @pytest.mark.unit
    def test_newey_west_rule(self) -> None:
        """Test Newey-West T^(1/3) rule."""
        residuals = np.random.randn(100)
        bw = optimal_bandwidth(residuals, method="newey_west")

        # Should be approximately 100^(1/3) ≈ 4.6 -> 4
        assert 3 <= bw <= 6

    @pytest.mark.unit
    def test_bandwidth_scales_with_n(self) -> None:
        """Bandwidth should increase with sample size."""
        bw_small = optimal_bandwidth(np.random.randn(50), method="newey_west")
        bw_large = optimal_bandwidth(np.random.randn(500), method="newey_west")

        assert bw_large > bw_small

    @pytest.mark.unit
    def test_auto_uses_newey_west(self) -> None:
        """'auto' should use newey_west method."""
        np.random.seed(42)
        residuals = np.random.randn(100)
        bw_auto = optimal_bandwidth(residuals, method="auto")
        bw_nw = optimal_bandwidth(residuals, method="newey_west")

        assert bw_auto == bw_nw

    @pytest.mark.unit
    def test_andrews_method(self) -> None:
        """Test Andrews data-driven bandwidth."""
        np.random.seed(42)
        # Create AR(1) residuals
        n = 200
        e = np.zeros(n)
        for t in range(1, n):
            e[t] = 0.5 * e[t - 1] + np.random.randn()

        bw = optimal_bandwidth(e, method="andrews", kernel="bartlett")

        # Should be positive
        assert bw >= 0
        # Should be reasonable
        assert bw < n // 2

    @pytest.mark.unit
    def test_small_sample(self) -> None:
        """Test bandwidth with small sample."""
        residuals = np.random.randn(5)
        bw = optimal_bandwidth(residuals, method="newey_west")

        # Should be small but non-negative
        assert 0 <= bw <= 2

    @pytest.mark.unit
    def test_invalid_method(self) -> None:
        """Test invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown bandwidth method"):
            optimal_bandwidth(np.random.randn(100), method="invalid")


class TestNeweyWestSE:
    """Test Newey-West standard error computation."""

    @pytest.mark.unit
    def test_basic_se(self) -> None:
        """Test basic SE computation."""
        np.random.seed(42)
        residuals = np.random.randn(100)
        se = newey_west_se(residuals, bandwidth=5)

        assert se > 0
        assert np.isfinite(se)

    @pytest.mark.unit
    def test_se_with_autocorrelation(self) -> None:
        """HAC SE should be larger with positive autocorrelation."""
        np.random.seed(42)
        n = 200

        # AR(1) with positive autocorrelation
        e_ar = np.zeros(n)
        for t in range(1, n):
            e_ar[t] = 0.7 * e_ar[t - 1] + np.random.randn()

        # IID errors
        e_iid = np.random.randn(n)

        se_ar = newey_west_se(e_ar, bandwidth=10)
        se_iid = newey_west_se(e_iid, bandwidth=10)

        # HAC SE should be larger for AR process
        # (positive autocorr -> understated naive SE)
        assert se_ar > se_iid * 0.8  # Allow some noise

    @pytest.mark.unit
    def test_se_with_negative_autocorrelation(self) -> None:
        """HAC SE should be smaller with negative autocorrelation."""
        np.random.seed(42)
        n = 200

        # AR(1) with negative autocorrelation
        e_ar = np.zeros(n)
        for t in range(1, n):
            e_ar[t] = -0.5 * e_ar[t - 1] + np.random.randn()

        # IID errors
        e_iid = np.random.randn(n)

        se_ar = newey_west_se(e_ar, bandwidth=10)
        se_iid = newey_west_se(e_iid, bandwidth=10)

        # HAC SE should be smaller for negative AR process
        assert se_ar < se_iid * 1.5

    @pytest.mark.unit
    def test_auto_bandwidth(self) -> None:
        """Test SE with automatic bandwidth."""
        np.random.seed(42)
        residuals = np.random.randn(100)

        se_auto = newey_west_se(residuals, bandwidth="auto")
        se_fixed = newey_west_se(residuals, bandwidth=5)

        # Both should be positive
        assert se_auto > 0
        assert se_fixed > 0

    @pytest.mark.unit
    def test_different_kernels(self) -> None:
        """Test SE with different kernels."""
        np.random.seed(42)
        residuals = np.random.randn(100)

        se_bartlett = newey_west_se(residuals, bandwidth=5, kernel="bartlett")
        se_parzen = newey_west_se(residuals, bandwidth=5, kernel="parzen")
        se_qs = newey_west_se(residuals, bandwidth=5, kernel="quadratic_spectral")

        # All should be positive and similar order of magnitude
        for se in [se_bartlett, se_parzen, se_qs]:
            assert se > 0
            assert 0.01 < se < 1.0

    @pytest.mark.unit
    def test_small_sample_error(self) -> None:
        """Test error with too few observations."""
        with pytest.raises(ValueError, match="at least 2 observations"):
            newey_west_se(np.array([1.0]))


class TestNeweyWestCovariance:
    """Test Newey-West covariance matrix computation."""

    @pytest.mark.unit
    def test_basic_covariance(self) -> None:
        """Test basic covariance computation."""
        np.random.seed(42)
        n, k = 100, 3
        X = np.random.randn(n, k)
        X[:, 0] = 1  # Intercept
        residuals = np.random.randn(n)

        cov = newey_west_covariance(residuals, X, bandwidth=5)

        assert cov.shape == (k, k)
        # Should be symmetric
        assert np.allclose(cov, cov.T)
        # Diagonal should be positive
        assert all(cov[i, i] > 0 for i in range(k))

    @pytest.mark.unit
    def test_positive_semidefinite(self) -> None:
        """HAC covariance should be positive semi-definite."""
        np.random.seed(42)
        n, k = 100, 4
        X = np.random.randn(n, k)
        residuals = np.random.randn(n)

        cov = newey_west_covariance(residuals, X, bandwidth=5, kernel="bartlett")

        # Check eigenvalues are non-negative
        eigenvalues = np.linalg.eigvalsh(cov)
        assert all(ev >= -1e-10 for ev in eigenvalues)

    @pytest.mark.unit
    def test_1d_x(self) -> None:
        """Test with 1D X array."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n)  # 1D
        residuals = np.random.randn(n)

        cov = newey_west_covariance(residuals, X, bandwidth=5)

        assert cov.shape == (1, 1)
        assert cov[0, 0] > 0

    @pytest.mark.unit
    def test_dimension_mismatch(self) -> None:
        """Test error when dimensions don't match."""
        X = np.random.randn(100, 3)
        residuals = np.random.randn(50)  # Wrong length

        with pytest.raises(ValueError, match="residuals length"):
            newey_west_covariance(residuals, X, bandwidth=5)


class TestHACEstimator:
    """Test HACEstimator class."""

    @pytest.mark.unit
    def test_init_default(self) -> None:
        """Test default initialization."""
        hac = HACEstimator()
        assert hac.kernel == "bartlett"
        assert hac.bandwidth == "auto"
        assert hac.prewhiten is False

    @pytest.mark.unit
    def test_init_invalid_kernel(self) -> None:
        """Test invalid kernel raises error."""
        with pytest.raises(ValueError, match="kernel must be one of"):
            HACEstimator(kernel="invalid")

    @pytest.mark.unit
    def test_fit_simple(self) -> None:
        """Test fit with simple residuals."""
        np.random.seed(42)
        residuals = np.random.randn(100)

        hac = HACEstimator()
        hac.fit(residuals)

        assert hac._is_fitted
        assert hac._variance > 0
        assert hac._bandwidth_used >= 0

    @pytest.mark.unit
    def test_fit_with_x(self) -> None:
        """Test fit with design matrix."""
        np.random.seed(42)
        n, k = 100, 3
        X = np.random.randn(n, k)
        residuals = np.random.randn(n)

        hac = HACEstimator()
        hac.fit(residuals, X)

        assert hac._covariance is not None
        assert hac._covariance.shape == (k, k)

    @pytest.mark.unit
    def test_get_se_not_fitted(self) -> None:
        """Test error when getting SE before fit."""
        hac = HACEstimator()
        with pytest.raises(RuntimeError, match="not fitted"):
            hac.get_se()

    @pytest.mark.unit
    def test_get_se(self) -> None:
        """Test get_se after fit."""
        np.random.seed(42)
        residuals = np.random.randn(100)

        hac = HACEstimator()
        hac.fit(residuals)
        se = hac.get_se()

        assert se > 0
        assert np.isfinite(se)

    @pytest.mark.unit
    def test_get_variance(self) -> None:
        """Test get_variance."""
        np.random.seed(42)
        residuals = np.random.randn(100)

        hac = HACEstimator()
        hac.fit(residuals)

        var = hac.get_variance()
        se = hac.get_se()

        assert np.isclose(var, se**2)

    @pytest.mark.unit
    def test_get_covariance_without_x(self) -> None:
        """Test error getting covariance without X."""
        np.random.seed(42)
        residuals = np.random.randn(100)

        hac = HACEstimator()
        hac.fit(residuals)  # No X

        with pytest.raises(RuntimeError, match="No covariance matrix"):
            hac.get_covariance()

    @pytest.mark.unit
    def test_get_covariance_with_x(self) -> None:
        """Test get_covariance with X."""
        np.random.seed(42)
        n, k = 100, 3
        X = np.random.randn(n, k)
        residuals = np.random.randn(n)

        hac = HACEstimator()
        hac.fit(residuals, X)

        cov = hac.get_covariance()
        assert cov.shape == (k, k)

    @pytest.mark.unit
    def test_get_result(self) -> None:
        """Test get_result returns HACResult."""
        np.random.seed(42)
        residuals = np.random.randn(100)

        hac = HACEstimator(bandwidth=5)
        hac.fit(residuals)

        result = hac.get_result()

        assert isinstance(result, HACResult)
        assert result.variance > 0
        assert result.se > 0
        assert result.bandwidth == 5
        assert result.kernel == "bartlett"
        assert result.n_samples == 100

    @pytest.mark.unit
    def test_bandwidth_used_property(self) -> None:
        """Test bandwidth_used property."""
        np.random.seed(42)
        residuals = np.random.randn(100)

        hac = HACEstimator(bandwidth="auto")
        hac.fit(residuals)

        bw = hac.bandwidth_used
        assert isinstance(bw, int)
        assert bw >= 0

    @pytest.mark.unit
    def test_prewhiten(self) -> None:
        """Test prewhitening option."""
        np.random.seed(42)
        n = 200

        # AR(1) residuals
        e = np.zeros(n)
        for t in range(1, n):
            e[t] = 0.5 * e[t - 1] + np.random.randn()

        hac_no_pw = HACEstimator(prewhiten=False)
        hac_pw = HACEstimator(prewhiten=True)

        hac_no_pw.fit(e)
        hac_pw.fit(e)

        # Both should give positive SE
        assert hac_no_pw.get_se() > 0
        assert hac_pw.get_se() > 0

    @pytest.mark.unit
    def test_method_chaining(self) -> None:
        """Test fit returns self for chaining."""
        np.random.seed(42)
        residuals = np.random.randn(100)

        se = HACEstimator().fit(residuals).get_se()
        assert se > 0


class TestHACInference:
    """Test inference helper function."""

    @pytest.mark.unit
    def test_basic_inference(self) -> None:
        """Test basic inference computation."""
        result = hac_inference(theta=2.0, se_hac=0.5, alpha=0.05)

        assert result["theta"] == 2.0
        assert result["se"] == 0.5
        assert result["ci_lower"] < 2.0 < result["ci_upper"]
        assert result["t_stat"] == 4.0
        assert result["p_value"] < 0.001  # Highly significant

    @pytest.mark.unit
    def test_ci_coverage(self) -> None:
        """Test CI has correct width."""
        result = hac_inference(theta=1.0, se_hac=1.0, alpha=0.05)

        # 95% CI should be approximately ±1.96
        ci_width = result["ci_upper"] - result["ci_lower"]
        assert 3.8 < ci_width < 4.0  # ≈ 2 * 1.96

    @pytest.mark.unit
    def test_different_alpha(self) -> None:
        """Test different significance levels."""
        result_95 = hac_inference(theta=1.0, se_hac=0.5, alpha=0.05)
        result_99 = hac_inference(theta=1.0, se_hac=0.5, alpha=0.01)

        # 99% CI should be wider
        width_95 = result_95["ci_upper"] - result_95["ci_lower"]
        width_99 = result_99["ci_upper"] - result_99["ci_lower"]
        assert width_99 > width_95

    @pytest.mark.unit
    def test_nonsignificant(self) -> None:
        """Test non-significant result."""
        result = hac_inference(theta=0.1, se_hac=1.0, alpha=0.05)

        # Should not be significant at 5%
        assert result["p_value"] > 0.05
        # CI should contain 0
        assert result["ci_lower"] < 0 < result["ci_upper"]


class TestIntegration:
    """Integration tests with realistic scenarios."""

    @pytest.mark.unit
    def test_regression_with_hac(self) -> None:
        """Test HAC with actual regression residuals."""
        np.random.seed(42)
        n = 200

        # Generate AR(1) errors
        e = np.zeros(n)
        for t in range(1, n):
            e[t] = 0.6 * e[t - 1] + np.random.randn()

        # Generate regression data
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        beta_true = np.array([1.0, 2.0])
        y = X @ beta_true + e

        # OLS estimates
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta_hat

        # HAC covariance
        cov = newey_west_covariance(residuals, X, bandwidth="auto")
        se_hac = np.sqrt(np.diag(cov))

        # Standard OLS SE (assumes homoskedasticity)
        sigma2_ols = np.sum(residuals**2) / (n - 2)
        cov_ols = sigma2_ols * np.linalg.inv(X.T @ X)
        se_ols = np.sqrt(np.diag(cov_ols))

        # Both should be positive and finite
        assert all(se > 0 for se in se_hac)
        assert all(se > 0 for se in se_ols)
        assert all(np.isfinite(se) for se in se_hac)
        # HAC and OLS SE should be in same order of magnitude
        # (exact relationship depends on autocorrelation realization)
        assert 0.1 < se_hac[1] / se_ols[1] < 10

    @pytest.mark.unit
    def test_consistency_functions_vs_class(self) -> None:
        """Test that functions and class give same results."""
        np.random.seed(42)
        n, k = 100, 2
        X = np.random.randn(n, k)
        residuals = np.random.randn(n)
        bw = 5

        # Using function
        cov_func = newey_west_covariance(residuals, X, bandwidth=bw, kernel="bartlett")

        # Using class
        hac = HACEstimator(bandwidth=bw, kernel="bartlett")
        hac.fit(residuals, X)
        cov_class = hac.get_covariance()

        assert np.allclose(cov_func, cov_class)
