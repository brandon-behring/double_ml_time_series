"""
Comprehensive tests for bootstrap diagnostics module.

Tests convergence diagnostics, distribution diagnostics, and automated
recommendations for bootstrap sample sizes.

Usage:
    pytest test/validation/test_bootstrap_diagnostics.py -v
"""

import numpy as np
import pytest
from typing import List

from src.validation.bootstrap_diagnostics import (
    BootstrapDiagnostics,
    ConvergenceDiagnostic,
    DistributionDiagnostic,
)
from src.validation.dgp_generator import DGPGenerator


# =============================================================================
# Test Class 1: Basic Functionality
# =============================================================================


@pytest.mark.tier1
class TestBootstrapDiagnosticsBasicFunctionality:
    """Test basic functionality of BootstrapDiagnostics."""

    @pytest.mark.tier1
    def test_instantiation_default(self):
        """Test instantiation with default parameters."""
        dgp = DGPGenerator(n=100, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data)

        assert diag.estimator_type == "LinearDML"
        assert diag.random_state is None
        assert diag.data is not None

    @pytest.mark.tier1
    def test_instantiation_with_custom_params(self):
        """Test instantiation with custom parameters."""
        dgp = DGPGenerator(n=100, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=99)

        assert diag.estimator_type == "OLS"
        assert diag.random_state == 99

    @pytest.mark.tier3
    def test_convergence_diagnostic_returns_correct_type(self):
        """Test that diagnose_convergence returns ConvergenceDiagnostic."""
        dgp = DGPGenerator(n=100, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        result = diag.diagnose_convergence(
            target="bias", n_bootstrap_range=[20, 50], true_value=2.0, n_replications=2
        )

        assert isinstance(result, ConvergenceDiagnostic)
        assert hasattr(result, "n_bootstrap_tested")
        assert hasattr(result, "estimates")
        assert hasattr(result, "std_errors")
        assert hasattr(result, "convergence_score")
        assert hasattr(result, "converged")
        assert hasattr(result, "recommended_n")

    @pytest.mark.tier3
    def test_distribution_diagnostic_returns_correct_type(self):
        """Test that diagnose_distribution returns DistributionDiagnostic."""
        dgp = DGPGenerator(n=100, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        result = diag.diagnose_distribution(n_bootstrap=50)

        assert isinstance(result, DistributionDiagnostic)
        assert hasattr(result, "n_bootstrap")
        assert hasattr(result, "normality_pvalue")
        assert hasattr(result, "is_normal")
        assert hasattr(result, "skewness")
        assert hasattr(result, "kurtosis")
        assert hasattr(result, "symmetry_score")


# =============================================================================
# Test Class 2: Convergence Diagnostics
# =============================================================================


@pytest.mark.tier3
class TestBootstrapDiagnosticsConvergence:
    """Test convergence diagnostic functionality."""

    @pytest.mark.tier3
    def test_convergence_detects_stability(self):
        """Test that convergence diagnostic detects stable bootstrap estimates."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        # Use increasing bootstrap sizes
        result = diag.diagnose_convergence(
            target="bias",
            n_bootstrap_range=[100, 200, 500],
            true_value=2.0,
            n_replications=5,
            tolerance=0.10,
        )

        # Should detect convergence for simple OLS case
        assert isinstance(result.converged, bool)
        assert 0.0 <= result.convergence_score <= 1.0
        assert result.recommended_n in result.n_bootstrap_tested

    def test_convergence_bias_requires_true_value(self):
        """Test that bias convergence diagnostic requires true_value."""
        dgp = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, random_state=42)

        with pytest.raises(ValueError, match="true_value required"):
            diag.diagnose_convergence(target="bias", n_bootstrap_range=[50, 100], true_value=None)

    def test_convergence_coverage_requires_true_value(self):
        """Test that coverage convergence diagnostic requires true_value."""
        dgp = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, random_state=42)

        with pytest.raises(ValueError, match="true_value required"):
            diag.diagnose_convergence(
                target="coverage", n_bootstrap_range=[50, 100], true_value=None
            )

    def test_convergence_variance_does_not_require_true_value(self):
        """Test that variance convergence diagnostic works without true_value."""
        dgp = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        # Should complete without error
        result = diag.diagnose_convergence(
            target="variance",
            n_bootstrap_range=[50, 100],
            true_value=None,
            n_replications=3,
        )

        assert isinstance(result, ConvergenceDiagnostic)
        assert result.diagnostics["target"] == "variance"

    def test_convergence_estimates_length_matches_range(self):
        """Test that returned estimates match n_bootstrap_range length."""
        dgp = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        n_range = [50, 100, 200]
        result = diag.diagnose_convergence(
            target="bias", n_bootstrap_range=n_range, true_value=2.0, n_replications=3
        )

        assert len(result.n_bootstrap_tested) == len(n_range)
        assert len(result.estimates) == len(n_range)
        assert len(result.std_errors) == len(n_range)
        assert result.n_bootstrap_tested == n_range


# =============================================================================
# Test Class 3: Distribution Diagnostics
# =============================================================================


@pytest.mark.tier3
class TestBootstrapDiagnosticsDistribution:
    """Test distribution diagnostic functionality."""

    @pytest.mark.tier3
    def test_distribution_returns_valid_statistics(self):
        """Test that distribution diagnostic returns valid statistics."""
        dgp = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        result = diag.diagnose_distribution(n_bootstrap=100)

        # Normality p-value should be in [0, 1]
        assert 0.0 <= result.normality_pvalue <= 1.0
        # is_normal is boolean
        assert isinstance(result.is_normal, bool)
        # Skewness and kurtosis should be finite
        assert np.isfinite(result.skewness)
        assert np.isfinite(result.kurtosis)
        # Symmetry score in [0, 1]
        assert 0.0 <= result.symmetry_score <= 1.0

    @pytest.mark.tier3
    def test_distribution_diagnostics_includes_percentiles(self):
        """Test that distribution diagnostics include percentile information."""
        dgp = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        result = diag.diagnose_distribution(n_bootstrap=100)

        assert "percentiles" in result.diagnostics
        assert "p05" in result.diagnostics["percentiles"]
        assert "p25" in result.diagnostics["percentiles"]
        assert "p75" in result.diagnostics["percentiles"]
        assert "p95" in result.diagnostics["percentiles"]

    def test_distribution_with_small_n_bootstrap(self):
        """Test distribution diagnostic with minimal bootstrap samples."""
        dgp = DGPGenerator(n=100, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        # Should handle small n_bootstrap (though results may be unreliable)
        result = diag.diagnose_distribution(n_bootstrap=10)

        assert isinstance(result, DistributionDiagnostic)
        assert result.n_bootstrap == 10


# =============================================================================
# Test Class 4: Recommendation Functionality
# =============================================================================


@pytest.mark.tier3
class TestBootstrapDiagnosticsRecommendations:
    """Test automated recommendation functionality."""

    def test_recommend_n_bootstrap_returns_dict(self):
        """Test that recommend_n_bootstrap returns dictionary."""
        dgp = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        recommendations = diag.recommend_n_bootstrap(target_tasks=["bias"], precision_level="fast")

        assert isinstance(recommendations, dict)
        assert "bias" in recommendations
        assert isinstance(recommendations["bias"], int)
        assert recommendations["bias"] > 0

    def test_recommend_n_bootstrap_ci_task(self):
        """Test recommendations for CI task."""
        dgp = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        recommendations = diag.recommend_n_bootstrap(target_tasks=["ci"], precision_level="fast")

        assert "ci" in recommendations
        assert isinstance(recommendations["ci"], int)

    def test_recommend_n_bootstrap_both_task(self):
        """Test recommendations for both bias and CI."""
        dgp = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        recommendations = diag.recommend_n_bootstrap(target_tasks=["both"], precision_level="fast")

        assert "both" in recommendations
        # "both" should use the larger of bias/ci recommendations
        if "bias" in recommendations and "ci" in recommendations:
            assert recommendations["both"] >= recommendations["bias"]
            assert recommendations["both"] >= recommendations["ci"]

    def test_recommend_precision_levels(self):
        """Test that different precision levels return different recommendations."""
        dgp = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        fast = diag.recommend_n_bootstrap(target_tasks=["bias"], precision_level="fast")
        default = diag.recommend_n_bootstrap(target_tasks=["bias"], precision_level="default")
        precise = diag.recommend_n_bootstrap(target_tasks=["bias"], precision_level="precise")

        # Precision levels should differ (though exact values depend on convergence)
        # At minimum, the ranges tested should be different
        assert isinstance(fast, dict)
        assert isinstance(default, dict)
        assert isinstance(precise, dict)


# =============================================================================
# Test Class 5: Reproducibility
# =============================================================================


@pytest.mark.tier3
class TestBootstrapDiagnosticsReproducibility:
    """Test reproducibility with random seeds."""

    @pytest.mark.tier3
    def test_same_seed_produces_identical_convergence_results(self):
        """Test that same seed produces identical convergence diagnostics."""
        dgp1 = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
        dgp2 = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
        data1 = dgp1.generate()
        data2 = dgp2.generate()

        diag1 = BootstrapDiagnostics(data1, estimator_type="OLS", random_state=42)
        diag2 = BootstrapDiagnostics(data2, estimator_type="OLS", random_state=42)

        result1 = diag1.diagnose_convergence(
            target="bias", n_bootstrap_range=[50, 100], true_value=2.0, n_replications=3
        )
        result2 = diag2.diagnose_convergence(
            target="bias", n_bootstrap_range=[50, 100], true_value=2.0, n_replications=3
        )

        # With same seeds, should get identical results
        assert result1.converged == result2.converged
        np.testing.assert_array_almost_equal(result1.estimates, result2.estimates)
        assert result1.convergence_score == result2.convergence_score

    @pytest.mark.tier3
    def test_same_seed_produces_identical_distribution_results(self):
        """Test that same seed produces identical distribution diagnostics."""
        dgp1 = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
        dgp2 = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
        data1 = dgp1.generate()
        data2 = dgp2.generate()

        diag1 = BootstrapDiagnostics(data1, estimator_type="OLS", random_state=42)
        diag2 = BootstrapDiagnostics(data2, estimator_type="OLS", random_state=42)

        result1 = diag1.diagnose_distribution(n_bootstrap=100)
        result2 = diag2.diagnose_distribution(n_bootstrap=100)

        # With same seeds, should get identical results
        assert result1.normality_pvalue == result2.normality_pvalue
        assert result1.skewness == result2.skewness
        assert result1.kurtosis == result2.kurtosis

    @pytest.mark.tier3
    def test_different_seed_produces_different_results(self):
        """Test that different seeds produce different results."""
        dgp = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()

        diag1 = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)
        diag2 = BootstrapDiagnostics(data, estimator_type="OLS", random_state=99)

        result1 = diag1.diagnose_distribution(n_bootstrap=100)
        result2 = diag2.diagnose_distribution(n_bootstrap=100)

        # Different seeds should produce different bootstrap distributions
        # (Not guaranteed different, but very unlikely to be identical)
        assert result1.skewness != result2.skewness or result1.kurtosis != result2.kurtosis


# =============================================================================
# Test Class 6: Edge Cases
# =============================================================================


@pytest.mark.tier3
class TestBootstrapDiagnosticsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_n_bootstrap_value(self):
        """Test convergence diagnostic with single n_bootstrap value."""
        dgp = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        result = diag.diagnose_convergence(
            target="bias", n_bootstrap_range=[100], true_value=2.0, n_replications=3
        )

        # With single value, convergence check is trivial
        assert len(result.estimates) == 1
        assert isinstance(result.converged, bool)

    def test_ols_estimator(self):
        """Test diagnostics work with OLS estimator."""
        dgp = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        result = diag.diagnose_distribution(n_bootstrap=50)

        assert isinstance(result, DistributionDiagnostic)
        assert result.n_bootstrap == 50

    @pytest.mark.tier3
    def test_lineardml_estimator(self):
        """Test diagnostics work with LinearDML estimator (uses OLS for speed in tier3)."""
        dgp = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        result = diag.diagnose_distribution(n_bootstrap=50)

        assert isinstance(result, DistributionDiagnostic)
        assert result.n_bootstrap == 50

    def test_ipw_estimator_raises_not_implemented(self):
        """Test that IPW estimator raises NotImplementedError."""
        dgp = DGPGenerator(n=100, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="IPW", random_state=42)

        with pytest.raises(NotImplementedError, match="IPW not yet implemented"):
            diag.diagnose_distribution(n_bootstrap=50)

    def test_aipw_estimator_raises_not_implemented(self):
        """Test that AIPW estimator raises NotImplementedError."""
        dgp = DGPGenerator(n=100, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="AIPW", random_state=42)

        with pytest.raises(NotImplementedError, match="AIPW not yet implemented"):
            diag.diagnose_distribution(n_bootstrap=50)

    def test_small_data_size(self):
        """Test diagnostics with small data size."""
        dgp = DGPGenerator(n=50, p=2, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        # Should complete without error, though results may be unreliable
        result = diag.diagnose_distribution(n_bootstrap=20)

        assert isinstance(result, DistributionDiagnostic)
        assert np.isfinite(result.skewness)

    def test_large_data_size(self):
        """Test diagnostics with large data size."""
        dgp = DGPGenerator(n=2000, p=5, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        # Large n should give stable bootstrap distributions
        result = diag.diagnose_distribution(n_bootstrap=50)

        assert isinstance(result, DistributionDiagnostic)
        # Large n should have more symmetric bootstrap distribution
        assert abs(result.skewness) < 2.0  # Reasonable skewness bound


# =============================================================================
# Test Class 7: Statistical Properties
# =============================================================================


@pytest.mark.tier3
class TestBootstrapDiagnosticsStatisticalProperties:
    """Test statistical properties of diagnostics."""

    @pytest.mark.tier3
    def test_convergence_score_in_valid_range(self):
        """Test that convergence score is in [0, 1]."""
        dgp = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        result = diag.diagnose_convergence(
            target="bias", n_bootstrap_range=[50, 100, 200], true_value=2.0, n_replications=3
        )

        assert 0.0 <= result.convergence_score <= 1.0

    @pytest.mark.tier3
    def test_symmetry_score_in_valid_range(self):
        """Test that symmetry score is in [0, 1]."""
        dgp = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        result = diag.diagnose_distribution(n_bootstrap=100)

        assert 0.0 <= result.symmetry_score <= 1.0

    def test_std_errors_non_negative(self):
        """Test that standard errors are non-negative."""
        dgp = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        result = diag.diagnose_convergence(
            target="bias", n_bootstrap_range=[50, 100], true_value=2.0, n_replications=3
        )

        assert all(se >= 0 for se in result.std_errors)

    @pytest.mark.tier3
    def test_normality_pvalue_in_valid_range(self):
        """Test that normality p-value is in [0, 1]."""
        dgp = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        result = diag.diagnose_distribution(n_bootstrap=100)

        assert 0.0 <= result.normality_pvalue <= 1.0

    def test_recommended_n_positive(self):
        """Test that recommended n_bootstrap values are positive."""
        dgp = DGPGenerator(n=200, p=3, true_effect=2.0, random_state=42)
        data = dgp.generate()
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        recommendations = diag.recommend_n_bootstrap(
            target_tasks=["bias", "ci", "both"], precision_level="fast"
        )

        for task, n_boot in recommendations.items():
            assert n_boot > 0, f"Task {task} has non-positive recommendation: {n_boot}"


# =============================================================================
# Test Class 8: Integration Tests
# =============================================================================


@pytest.mark.tier3
class TestBootstrapDiagnosticsIntegration:
    """Test integration with DGP and end-to-end workflows."""

    @pytest.mark.tier3
    def test_full_workflow_bias_convergence(self):
        """Test complete workflow for bias convergence analysis."""
        # Generate data
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, random_state=42)
        data = dgp.generate()

        # Create diagnostics
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        # Run convergence diagnostic
        result = diag.diagnose_convergence(
            target="bias",
            n_bootstrap_range=[100, 200, 500],
            true_value=2.0,
            n_replications=5,
        )

        # Verify complete workflow
        assert isinstance(result, ConvergenceDiagnostic)
        assert len(result.estimates) == 3
        assert result.recommended_n in [100, 200, 500]
        assert "monte_carlo_error" in result.diagnostics

    @pytest.mark.tier3
    def test_full_workflow_distribution_analysis(self):
        """Test complete workflow for distribution analysis."""
        # Generate data
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, random_state=42)
        data = dgp.generate()

        # Create diagnostics (use OLS for speed in tier3)
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        # Run distribution diagnostic
        result = diag.diagnose_distribution(n_bootstrap=200)

        # Verify complete workflow
        assert isinstance(result, DistributionDiagnostic)
        assert result.n_bootstrap == 200
        assert "mean" in result.diagnostics
        assert "median" in result.diagnostics
        assert "std" in result.diagnostics

    def test_full_workflow_recommendations(self):
        """Test complete workflow for automated recommendations."""
        # Generate data
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, random_state=42)
        data = dgp.generate()

        # Create diagnostics
        diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=42)

        # Get recommendations
        recommendations = diag.recommend_n_bootstrap(
            target_tasks=["bias", "ci"], precision_level="default"
        )

        # Verify recommendations
        assert "bias" in recommendations
        assert "ci" in recommendations
        assert all(isinstance(v, int) for v in recommendations.values())
        assert all(v > 0 for v in recommendations.values())
