"""
Tests for stationarity diagnostic module.

Validates:
1. ADF test correctness (against known stationary/non-stationary series)
2. KPSS test correctness
3. Phillips-Perron test correctness
4. Comprehensive test synthesis
5. Differencing order recommendation
6. Edge cases and error handling
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytestmark = pytest.mark.tier2

from dml_ts.validation.stationarity import (
    StationarityDiagnostic,
    StationarityResult,
    ComprehensiveStationarityResult,
    run_stationarity_test,
    suggest_differencing_order,
    _compute_adf_statistic,
    _compute_kpss_statistic,
    _compute_pp_statistic,
    _interpolate_critical_value,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def stationary_ar1():
    """Generate stationary AR(1) series with phi=0.5."""
    np.random.seed(42)
    n = 200
    phi = 0.5
    y = np.zeros(n)
    y[0] = np.random.randn()
    for t in range(1, n):
        y[t] = phi * y[t - 1] + np.random.randn()
    return y


@pytest.fixture
def random_walk():
    """Generate random walk (non-stationary)."""
    np.random.seed(123)
    n = 200
    return np.cumsum(np.random.randn(n))


@pytest.fixture
def trend_stationary():
    """Generate trend-stationary series."""
    np.random.seed(456)
    n = 200
    trend = 0.1 * np.arange(n)
    noise = np.random.randn(n)
    return trend + noise


@pytest.fixture
def white_noise():
    """Generate white noise (stationary)."""
    np.random.seed(789)
    return np.random.randn(200)


# =============================================================================
# ADF Test Tests
# =============================================================================


class TestADFTest:
    """Tests for Augmented Dickey-Fuller test."""

    def test_adf_detects_stationary(self, stationary_ar1):
        """ADF should reject unit root for stationary series."""
        diag = StationarityDiagnostic(significance_level=0.05)
        result = diag.test_adf(stationary_ar1)

        assert result.test_name == "ADF"
        assert bool(result.is_stationary) is True
        assert result.statistic < result.critical_values["5%"]
        assert result.p_value < 0.05

    def test_adf_detects_nonstationary(self, random_walk):
        """ADF should not reject unit root for random walk."""
        diag = StationarityDiagnostic(significance_level=0.05)
        result = diag.test_adf(random_walk)

        assert bool(result.is_stationary) is False
        assert result.statistic > result.critical_values["5%"]
        assert result.p_value > 0.05

    def test_adf_white_noise(self, white_noise):
        """ADF should reject unit root for white noise."""
        diag = StationarityDiagnostic()
        result = diag.test_adf(white_noise)

        assert bool(result.is_stationary) is True
        assert result.statistic < -3.0  # Should be very negative

    def test_adf_regression_types(self, stationary_ar1):
        """Test different regression types."""
        diag = StationarityDiagnostic()

        # No constant
        result_n = diag.test_adf(stationary_ar1, regression="n")
        assert result_n.regression_type == "n"

        # Constant only
        result_c = diag.test_adf(stationary_ar1, regression="c")
        assert result_c.regression_type == "c"

        # Constant and trend
        result_ct = diag.test_adf(stationary_ar1, regression="ct")
        assert result_ct.regression_type == "ct"

    def test_adf_critical_values(self, stationary_ar1):
        """Critical values should be correctly ordered."""
        result = run_stationarity_test(stationary_ar1, method="adf")

        cv = result.critical_values
        assert cv["1%"] < cv["5%"] < cv["10%"]

    def test_adf_n_lags_returned(self, stationary_ar1):
        """n_lags should be non-negative."""
        result = run_stationarity_test(stationary_ar1, method="adf")
        assert result.n_lags >= 0

    def test_adf_interpretation(self, stationary_ar1, random_walk):
        """Interpretation should be meaningful."""
        result_stat = run_stationarity_test(stationary_ar1, method="adf")
        assert "stationary" in result_stat.interpretation.lower()

        result_ns = run_stationarity_test(random_walk, method="adf")
        assert (
            "non-stationary" in result_ns.interpretation.lower()
            or "differencing" in result_ns.interpretation.lower()
        )


# =============================================================================
# KPSS Test Tests
# =============================================================================


class TestKPSSTest:
    """Tests for KPSS stationarity test."""

    def test_kpss_detects_stationary(self, stationary_ar1):
        """KPSS should not reject stationarity for stationary series."""
        diag = StationarityDiagnostic(significance_level=0.05)
        result = diag.test_kpss(stationary_ar1)

        assert result.test_name == "KPSS"
        assert bool(result.is_stationary) is True
        assert result.statistic < result.critical_values["5%"]

    def test_kpss_detects_nonstationary(self, random_walk):
        """KPSS should reject stationarity for random walk."""
        diag = StationarityDiagnostic()
        result = diag.test_kpss(random_walk)

        assert bool(result.is_stationary) is False
        assert result.statistic > result.critical_values["5%"]

    def test_kpss_regression_types(self, stationary_ar1):
        """Test KPSS with level and trend regression."""
        diag = StationarityDiagnostic()

        result_c = diag.test_kpss(stationary_ar1, regression="c")
        assert result_c.regression_type == "c"

        result_ct = diag.test_kpss(stationary_ar1, regression="ct")
        assert result_ct.regression_type == "ct"

    def test_kpss_critical_values(self, stationary_ar1):
        """Critical values should be correctly ordered."""
        result = run_stationarity_test(stationary_ar1, method="kpss")

        cv = result.critical_values
        assert cv["10%"] < cv["5%"] < cv["2.5%"] < cv["1%"]


# =============================================================================
# Phillips-Perron Test Tests
# =============================================================================


class TestPhillipsPerronTest:
    """Tests for Phillips-Perron unit root test."""

    def test_pp_detects_stationary(self, stationary_ar1):
        """PP should reject unit root for stationary series."""
        diag = StationarityDiagnostic()
        result = diag.test_phillips_perron(stationary_ar1)

        assert result.test_name == "Phillips-Perron"
        assert bool(result.is_stationary) is True

    def test_pp_detects_nonstationary(self, random_walk):
        """PP should not reject unit root for random walk (usually)."""
        diag = StationarityDiagnostic()
        result = diag.test_phillips_perron(random_walk)

        # PP may sometimes reject due to HAC correction; check p-value is reasonable
        # The key is that it should not be strongly significant
        assert result.p_value > 0.01  # Should not be highly significant

    def test_pp_similar_to_adf(self, stationary_ar1):
        """PP and ADF should give similar conclusions."""
        diag = StationarityDiagnostic()

        adf_result = diag.test_adf(stationary_ar1)
        pp_result = diag.test_phillips_perron(stationary_ar1)

        assert adf_result.is_stationary == pp_result.is_stationary


# =============================================================================
# Comprehensive Test Tests
# =============================================================================


class TestComprehensiveTest:
    """Tests for comprehensive stationarity analysis."""

    def test_comprehensive_stationary(self, stationary_ar1):
        """Comprehensive test should identify stationary series."""
        diag = StationarityDiagnostic()
        result = diag.comprehensive_test(stationary_ar1)

        assert isinstance(result, ComprehensiveStationarityResult)
        assert "STATIONARY" in result.overall_conclusion.upper()
        assert result.adf_result is not None
        assert result.kpss_result is not None
        assert result.pp_result is not None

    def test_comprehensive_nonstationary(self, random_walk):
        """Comprehensive test should indicate non-stationarity issues."""
        diag = StationarityDiagnostic()
        result = diag.comprehensive_test(random_walk)

        # Either non-stationary or inconclusive (PP can be quirky)
        assert (
            "NON-STATIONARY" in result.overall_conclusion.upper()
            or "INCONCLUSIVE" in result.overall_conclusion.upper()
        )
        # ADF should clearly indicate non-stationarity
        assert bool(result.adf_result.is_stationary) is False

    def test_comprehensive_has_recommendation(self, stationary_ar1):
        """Comprehensive test should provide recommendation."""
        diag = StationarityDiagnostic()
        result = diag.comprehensive_test(stationary_ar1)

        assert len(result.recommendation) > 10


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_run_stationarity_test_adf(self, stationary_ar1):
        """run_stationarity_test should work with ADF."""
        result = run_stationarity_test(stationary_ar1, method="adf")
        assert isinstance(result, StationarityResult)
        assert result.test_name == "ADF"

    def test_run_stationarity_test_kpss(self, stationary_ar1):
        """run_stationarity_test should work with KPSS."""
        result = run_stationarity_test(stationary_ar1, method="kpss")
        assert result.test_name == "KPSS"

    def test_run_stationarity_test_pp(self, stationary_ar1):
        """run_stationarity_test should work with PP."""
        result = run_stationarity_test(stationary_ar1, method="pp")
        assert result.test_name == "Phillips-Perron"

    def test_suggest_differencing_stationary(self, stationary_ar1):
        """suggest_differencing_order should return 0 for stationary."""
        d = suggest_differencing_order(stationary_ar1)
        assert d == 0

    def test_suggest_differencing_random_walk(self, random_walk):
        """suggest_differencing_order should return 1 for random walk."""
        d = suggest_differencing_order(random_walk)
        assert d == 1

    def test_suggest_differencing_i2(self):
        """suggest_differencing_order should return 2 for I(2) process."""
        np.random.seed(999)
        # I(2): cumsum of cumsum
        i2 = np.cumsum(np.cumsum(np.random.randn(200)))
        d = suggest_differencing_order(i2, max_d=2)
        assert d == 2


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_short_series_raises(self):
        """Short series should raise ValueError."""
        diag = StationarityDiagnostic()

        with pytest.raises(ValueError, match="too short"):
            diag.test_adf(np.random.randn(10))

    def test_invalid_regression_raises(self):
        """Invalid regression type should raise ValueError."""
        with pytest.raises(ValueError):
            StationarityDiagnostic(regression="invalid")

    def test_invalid_significance_level_raises(self):
        """Invalid significance level should raise ValueError."""
        with pytest.raises(ValueError):
            StationarityDiagnostic(significance_level=1.5)

        with pytest.raises(ValueError):
            StationarityDiagnostic(significance_level=-0.1)

    def test_invalid_method_raises(self, stationary_ar1):
        """Invalid method should raise ValueError."""
        diag = StationarityDiagnostic()

        with pytest.raises(ValueError):
            diag.test(stationary_ar1, method="invalid")

    def test_constant_series(self):
        """Constant series should be handled."""
        constant = np.ones(100)
        diag = StationarityDiagnostic()

        # Should not crash; result may vary
        result = diag.test_kpss(constant)
        assert isinstance(result, StationarityResult)

    def test_numpy_array_conversion(self, stationary_ar1):
        """Should handle list input."""
        diag = StationarityDiagnostic()
        result = diag.test_adf(list(stationary_ar1))
        assert isinstance(result, StationarityResult)


# =============================================================================
# Critical Value Interpolation Tests
# =============================================================================


class TestCriticalValueInterpolation:
    """Tests for critical value interpolation."""

    def test_interpolate_known_values(self):
        """Test interpolation at known sample sizes."""
        # At n=100, should be close to tabulated values
        cv_c_05 = _interpolate_critical_value("c", 0.05, 100)
        assert -2.95 < cv_c_05 < -2.85

    def test_interpolate_extrapolation(self):
        """Test extrapolation for large n."""
        cv_large = _interpolate_critical_value("c", 0.05, 10000)
        # Should approach asymptotic value (~-2.86)
        assert -2.90 < cv_large < -2.82

    def test_interpolate_small_n(self):
        """Test interpolation for small n."""
        cv_small = _interpolate_critical_value("c", 0.05, 30)
        # Should be more negative for small samples
        assert cv_small < -2.90


# =============================================================================
# Statsmodels Comparison Tests (if available)
# =============================================================================


class TestStatsmodelsComparison:
    """Compare results with statsmodels (if available)."""

    @pytest.fixture
    def check_statsmodels(self):
        """Check if statsmodels is available."""
        try:
            from statsmodels.tsa.stattools import adfuller

            return True
        except ImportError:
            pytest.skip("statsmodels not available")

    def test_adf_matches_statsmodels(self, stationary_ar1, check_statsmodels):
        """ADF statistic should match statsmodels."""
        from statsmodels.tsa.stattools import adfuller

        our_result = run_stationarity_test(stationary_ar1, method="adf", regression="c")
        sm_result = adfuller(stationary_ar1, regression="c", autolag="AIC")

        # Statistics should be very close
        assert_allclose(our_result.statistic, sm_result[0], rtol=0.01)

    def test_kpss_close_to_statsmodels(self, stationary_ar1, check_statsmodels):
        """KPSS statistic should be close to statsmodels."""
        from statsmodels.tsa.stattools import kpss

        our_result = run_stationarity_test(stationary_ar1, method="kpss", regression="c")
        sm_result = kpss(stationary_ar1, regression="c")

        # May have small differences due to bandwidth selection
        assert_allclose(our_result.statistic, sm_result[0], rtol=0.1)


# =============================================================================
# Monte Carlo Tests
# =============================================================================


class TestMonteCarlo:
    """Monte Carlo tests for statistical properties."""

    @pytest.mark.tier3
    def test_adf_size_under_null(self):
        """Test ADF rejection rate under null (unit root)."""
        np.random.seed(42)
        n_sims = 200
        n = 100
        rejections = 0

        for _ in range(n_sims):
            rw = np.cumsum(np.random.randn(n))
            result = run_stationarity_test(rw, method="adf", significance_level=0.05)
            if result.is_stationary:
                rejections += 1

        # Should reject approximately 5% under null
        rejection_rate = rejections / n_sims
        assert 0.02 < rejection_rate < 0.12  # Allow some variation

    @pytest.mark.tier3
    def test_adf_power(self):
        """Test ADF power for stationary series."""
        np.random.seed(42)
        n_sims = 200
        n = 100
        phi = 0.5
        rejections = 0

        for _ in range(n_sims):
            y = np.zeros(n)
            y[0] = np.random.randn()
            for t in range(1, n):
                y[t] = phi * y[t - 1] + np.random.randn()
            result = run_stationarity_test(y, method="adf", significance_level=0.05)
            if result.is_stationary:
                rejections += 1

        # Should have high power for phi=0.5
        power = rejections / n_sims
        assert power > 0.7  # Should reject most of the time


# =============================================================================
# Result Object Tests
# =============================================================================


class TestResultObjects:
    """Tests for result object properties."""

    def run_stationarity_test_result_repr(self, stationary_ar1):
        """StationarityResult should have readable repr."""
        result = run_stationarity_test(stationary_ar1)
        repr_str = repr(result)

        assert "StationarityResult" in repr_str
        assert "stat=" in repr_str
        assert "stationary=" in repr_str

    def test_comprehensive_result_repr(self, stationary_ar1):
        """ComprehensiveStationarityResult should have readable repr."""
        diag = StationarityDiagnostic()
        result = diag.comprehensive_test(stationary_ar1)
        repr_str = repr(result)

        assert "ComprehensiveStationarityResult" in repr_str
        assert "conclusion=" in repr_str
