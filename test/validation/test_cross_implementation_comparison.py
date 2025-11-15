"""
Tests for cross-implementation comparison module.

Test coverage for validating DML implementation consistency across packages.
"""

import pytest
import numpy as np
from src.validation.cross_implementation_comparison import (
    CrossImplementationComparison,
    ImplementationComparisonResult,
)
from src.validation.dgp_generator import DGPGenerator


class TestCrossImplementationComparisonBasicFunctionality:
    """Test basic functionality and interface."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        comp = CrossImplementationComparison()

        assert comp.n_simulations == 100
        assert comp.alpha == 0.05
        assert comp.bias_tolerance == 0.05
        assert comp.variance_tolerance == 0.10
        assert comp.coverage_tolerance == 0.02

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        comp = CrossImplementationComparison(
            n_simulations=50,
            alpha=0.10,
            bias_tolerance=0.03,
            variance_tolerance=0.15,
            coverage_tolerance=0.01,
            random_state=42,
        )

        assert comp.n_simulations == 50
        assert comp.alpha == 0.10
        assert comp.bias_tolerance == 0.03
        assert comp.variance_tolerance == 0.15
        assert comp.coverage_tolerance == 0.01
        assert comp.random_state == 42

    def test_compare_implementations_returns_result(self):
        """Test that compare_implementations returns proper result object."""
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, random_state=42)
        comp = CrossImplementationComparison(n_simulations=10, random_state=42)

        result = comp.compare_implementations(dgp)

        assert isinstance(result, ImplementationComparisonResult)
        assert result.implementation1 == "LinearDML_current"
        assert result.implementation2 == "LinearDML_econml"
        assert result.n_simulations == 10

    def test_reproducibility_with_random_state(self):
        """Test that results are reproducible with same random_state."""
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, random_state=42)

        comp1 = CrossImplementationComparison(n_simulations=10, random_state=42)
        result1 = comp1.compare_implementations(dgp)

        comp2 = CrossImplementationComparison(n_simulations=10, random_state=42)
        result2 = comp2.compare_implementations(dgp)

        assert np.isclose(result1.bias1, result2.bias1, atol=1e-10)
        assert np.isclose(result1.variance1, result2.variance1, atol=1e-10)
        assert np.isclose(result1.coverage1, result2.coverage1, atol=1e-10)


class TestCrossImplementationComparisonConsistency:
    """Test consistency between identical implementations."""

    def test_identical_implementations_have_zero_difference(self):
        """Test that comparing LinearDML with itself gives near-zero differences."""
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, random_state=42)
        comp = CrossImplementationComparison(n_simulations=20, random_state=42)

        result = comp.compare_implementations(
            dgp, implementation1="LinearDML_current", implementation2="LinearDML_econml"
        )

        # Since both implementations are identical (EconML), differences should be exactly zero
        assert np.isclose(result.bias_difference, 0.0, atol=1e-10)
        assert np.isclose(result.variance_difference, 0.0, atol=1e-10)
        assert np.isclose(result.coverage_difference, 0.0, atol=1e-10)

    def test_status_pass_for_identical_implementations(self):
        """Test that comparing identical implementations returns PASS status."""
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, random_state=42)
        comp = CrossImplementationComparison(n_simulations=20, random_state=42)

        result = comp.compare_implementations(dgp)

        assert result.status == "PASS"


class TestCrossImplementationComparisonMetrics:
    """Test calculation of comparison metrics."""

    def test_bias_calculation(self):
        """Test that bias is calculated correctly for both implementations."""
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, random_state=42)
        comp = CrossImplementationComparison(n_simulations=20, random_state=42)

        result = comp.compare_implementations(dgp)

        # Bias should be difference between mean estimate and true effect
        # For well-performing DML, bias should be small
        assert abs(result.bias1) < 0.5  # Reasonable bias
        assert abs(result.bias2) < 0.5
        assert isinstance(result.bias_difference, float)
        assert result.bias_difference >= 0.0

    def test_variance_calculation(self):
        """Test that variance is calculated correctly for both implementations."""
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, random_state=42)
        comp = CrossImplementationComparison(n_simulations=20, random_state=42)

        result = comp.compare_implementations(dgp)

        # Variance should be positive
        assert result.variance1 > 0
        assert result.variance2 > 0
        # Variance difference is relative
        assert 0.0 <= result.variance_difference <= 1.0
        assert isinstance(result.variance_difference, float)

    def test_coverage_calculation(self):
        """Test that coverage is calculated correctly for both implementations."""
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, random_state=42)
        comp = CrossImplementationComparison(n_simulations=30, random_state=42)

        result = comp.compare_implementations(dgp)

        # Coverage should be between 0 and 1
        assert 0.0 <= result.coverage1 <= 1.0
        assert 0.0 <= result.coverage2 <= 1.0
        # Coverage difference should be absolute
        assert result.coverage_difference >= 0.0
        assert isinstance(result.coverage_difference, float)


class TestCrossImplementationComparisonStatusDetermination:
    """Test status determination logic."""

    def test_status_pass_when_all_within_tolerance(self):
        """Test PASS status when all metrics within tolerance."""
        comp = CrossImplementationComparison(
            bias_tolerance=0.10,
            variance_tolerance=0.15,
            coverage_tolerance=0.05,
        )

        # All within tolerance
        status = comp._determine_status(bias_diff=0.05, var_diff=0.10, cov_diff=0.02)
        assert status == "PASS"

    def test_status_warning_when_one_exceeds_tolerance(self):
        """Test WARNING status when exactly one metric exceeds tolerance."""
        comp = CrossImplementationComparison(
            bias_tolerance=0.05,
            variance_tolerance=0.10,
            coverage_tolerance=0.02,
        )

        # Only bias exceeds
        status = comp._determine_status(bias_diff=0.10, var_diff=0.05, cov_diff=0.01)
        assert status == "WARNING"

    def test_status_fail_when_two_exceed_tolerance(self):
        """Test FAIL status when two or more metrics exceed tolerance."""
        comp = CrossImplementationComparison(
            bias_tolerance=0.05,
            variance_tolerance=0.10,
            coverage_tolerance=0.02,
        )

        # Bias and variance exceed
        status = comp._determine_status(bias_diff=0.10, var_diff=0.15, cov_diff=0.01)
        assert status == "FAIL"

    def test_status_fail_when_all_exceed_tolerance(self):
        """Test FAIL status when all metrics exceed tolerance."""
        comp = CrossImplementationComparison(
            bias_tolerance=0.05,
            variance_tolerance=0.10,
            coverage_tolerance=0.02,
        )

        # All exceed
        status = comp._determine_status(bias_diff=0.10, var_diff=0.15, cov_diff=0.05)
        assert status == "FAIL"


class TestCrossImplementationComparisonEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_sample_size(self):
        """Test comparison with small sample sizes."""
        dgp = DGPGenerator(n=100, p=3, true_effect=2.0, random_state=42)
        comp = CrossImplementationComparison(n_simulations=10, random_state=42)

        result = comp.compare_implementations(dgp)

        # Should still return valid result
        assert isinstance(result, ImplementationComparisonResult)
        assert result.n_simulations == 10

    def test_large_sample_size(self):
        """Test comparison with large sample sizes."""
        dgp = DGPGenerator(n=2000, p=10, true_effect=2.0, random_state=42)
        comp = CrossImplementationComparison(n_simulations=10, random_state=42)

        result = comp.compare_implementations(dgp)

        # Should still return valid result
        assert isinstance(result, ImplementationComparisonResult)
        # With large n, bias should be smaller
        assert abs(result.bias1) < 0.3

    def test_zero_true_effect(self):
        """Test comparison when true treatment effect is zero."""
        dgp = DGPGenerator(n=500, p=5, true_effect=0.0, random_state=42)
        comp = CrossImplementationComparison(n_simulations=20, random_state=42)

        result = comp.compare_implementations(dgp)

        # Should handle zero effect correctly
        assert isinstance(result, ImplementationComparisonResult)
        assert result.status in ["PASS", "WARNING", "FAIL"]

    def test_negative_true_effect(self):
        """Test comparison with negative true treatment effect."""
        dgp = DGPGenerator(n=500, p=5, true_effect=-1.5, random_state=42)
        comp = CrossImplementationComparison(n_simulations=20, random_state=42)

        result = comp.compare_implementations(dgp)

        # Should handle negative effects correctly
        assert isinstance(result, ImplementationComparisonResult)
        # Bias should be around the negative true effect
        assert result.bias1 < 0 or abs(result.bias1 + 1.5) < 1.0

    def test_very_few_simulations(self):
        """Test with minimal number of simulations."""
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, random_state=42)
        comp = CrossImplementationComparison(n_simulations=5, random_state=42)

        result = comp.compare_implementations(dgp)

        # Should still work but with higher variance
        assert isinstance(result, ImplementationComparisonResult)
        assert result.n_simulations == 5


class TestCrossImplementationComparisonMetadata:
    """Test metadata storage and reporting."""

    def test_metadata_contains_dgp_info(self):
        """Test that result metadata contains DGP parameters."""
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, confounding_strength=1.5, random_state=42)
        comp = CrossImplementationComparison(n_simulations=10, random_state=42)

        result = comp.compare_implementations(dgp)

        assert result.metadata["dgp_n"] == 500
        assert result.metadata["dgp_p"] == 5
        assert result.metadata["dgp_true_effect"] == 2.0
        assert result.metadata["dgp_confounding"] == 1.5

    def test_metadata_contains_tolerance_info(self):
        """Test that result metadata contains tolerance thresholds."""
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, random_state=42)
        comp = CrossImplementationComparison(
            n_simulations=10,
            bias_tolerance=0.03,
            variance_tolerance=0.12,
            coverage_tolerance=0.015,
            random_state=42,
        )

        result = comp.compare_implementations(dgp)

        assert result.metadata["bias_tolerance"] == 0.03
        assert result.metadata["variance_tolerance"] == 0.12
        assert result.metadata["coverage_tolerance"] == 0.015

    def test_timestamp_populated(self):
        """Test that result contains timestamp."""
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, random_state=42)
        comp = CrossImplementationComparison(n_simulations=10, random_state=42)

        result = comp.compare_implementations(dgp)

        assert result.timestamp is not None
        from datetime import datetime

        assert isinstance(result.timestamp, datetime)


class TestCrossImplementationComparisonUnknownImplementation:
    """Test error handling for unknown implementations."""

    def test_unknown_implementation_raises_error(self):
        """Test that specifying unknown implementation raises ValueError."""
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, random_state=42)
        comp = CrossImplementationComparison(n_simulations=10, random_state=42)

        with pytest.raises(ValueError, match="Unknown implementation"):
            comp.compare_implementations(
                dgp, implementation1="NonexistentImplementation", implementation2="LinearDML_econml"
            )


class TestCrossImplementationComparisonIntegration:
    """Integration tests with realistic scenarios."""

    def test_realistic_comparison_scenario(self):
        """Test comparison on realistic DGP scenario."""
        # Moderate sample size, moderate confounding
        dgp = DGPGenerator(
            n=1000,
            p=10,
            true_effect=1.5,
            confounding_strength=1.0,
            random_state=42,
        )
        comp = CrossImplementationComparison(n_simulations=50, random_state=42)

        result = comp.compare_implementations(dgp)

        # Should pass with identical implementations
        assert result.status == "PASS"
        # Bias should be reasonably small
        assert abs(result.bias1) < 0.3
        # Coverage should be near nominal
        assert 0.85 <= result.coverage1 <= 1.0

    def test_challenging_comparison_scenario(self):
        """Test comparison on challenging DGP scenario (small n, high confounding)."""
        dgp = DGPGenerator(
            n=200,
            p=15,
            true_effect=1.0,
            confounding_strength=2.0,
            random_state=42,
        )
        comp = CrossImplementationComparison(n_simulations=30, random_state=42)

        result = comp.compare_implementations(dgp)

        # Should still return valid result even if challenging
        assert isinstance(result, ImplementationComparisonResult)
        assert result.status in ["PASS", "WARNING", "FAIL"]
