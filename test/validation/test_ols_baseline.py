"""
Test suite for OLS baseline estimators.

Tests NaiveOLS and OLSWithControls to ensure they:
1. Properly detect confounding bias
2. Control for confounders when X is included
3. Maintain correct statistical properties
"""

import numpy as np
import pytest

from src.validation.dgp_generator import DGPGenerator
from src.validation.ols_baseline import NaiveOLS, OLSWithControls
from src.validation.validation_result import ValidationResult


class TestNaiveOLS:
    """Test suite for NaiveOLS estimator."""

    def test_instantiation(self):
        """Test basic instantiation."""
        estimator = NaiveOLS(n_simulations=10, alpha=0.05, random_state=42)
        assert estimator.n_simulations == 10
        assert estimator.alpha == 0.05
        assert estimator.random_state == 42

    def test_instantiation_with_custom_params(self):
        """Test instantiation with custom parameters."""
        estimator = NaiveOLS(n_simulations=50, alpha=0.01, random_state=123)
        assert estimator.n_simulations == 50
        assert estimator.alpha == 0.01
        assert estimator.random_state == 123

    def test_validate_returns_validation_result(self):
        """Test that validate() returns ValidationResult."""
        estimator = NaiveOLS(n_simulations=10, random_state=42)
        dgp = DGPGenerator(n=200, p=2, true_effect=2.0, random_state=42)

        result = estimator.validate(dgp)

        assert isinstance(result, ValidationResult)
        assert result.method == "NaiveOLS"
        assert result.n_simulations == 10

    def test_detects_confounding_bias(self):
        """Test that NaiveOLS detects significant bias with confounding."""
        # Create DGP with strong confounding
        dgp = DGPGenerator(
            n=1000,
            p=5,
            true_effect=2.0,
            confounding_strength=1.0,  # Strong confounding
            random_state=42,
        )

        estimator = NaiveOLS(n_simulations=20, random_state=42)
        result = estimator.validate(dgp)

        # NaiveOLS should show significant bias (FAIL or WARNING)
        assert result.status in ["FAIL", "WARNING"]
        # Bias should be substantial due to confounding
        assert abs(result.bias) > 0.1

    def test_less_biased_without_confounding(self):
        """Test NaiveOLS with weak confounding (no confounding)."""
        # Create DGP with no confounding
        dgp = DGPGenerator(
            n=1000,
            p=5,
            true_effect=2.0,
            confounding_strength=0.0,  # No confounding
            random_state=42,
        )

        estimator = NaiveOLS(n_simulations=20, random_state=42)
        result = estimator.validate(dgp)

        # Without confounding, NaiveOLS should have small bias
        # Note: Bias may still be non-zero due to finite samples and randomness
        assert abs(result.bias) < 0.5  # Relaxed threshold for statistical variation

    def test_reproducibility_same_seed(self):
        """Test reproducibility with same random seed."""
        dgp1 = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        dgp2 = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)

        estimator1 = NaiveOLS(n_simulations=15, random_state=123)
        estimator2 = NaiveOLS(n_simulations=15, random_state=123)

        result1 = estimator1.validate(dgp1)
        result2 = estimator2.validate(dgp2)

        assert np.isclose(result1.bias, result2.bias, atol=1e-10)
        assert np.isclose(result1.mse, result2.mse, atol=1e-10)

    def test_different_seed_produces_different_results(self):
        """Test that different seeds produce different results."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)

        estimator1 = NaiveOLS(n_simulations=15, random_state=123)
        estimator2 = NaiveOLS(n_simulations=15, random_state=456)

        result1 = estimator1.validate(dgp)
        result2 = estimator2.validate(dgp)

        # Results should differ (with high probability)
        assert not np.isclose(result1.bias, result2.bias, atol=0.05)

    def test_mse_non_negative(self):
        """Test that MSE is always non-negative."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        estimator = NaiveOLS(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        assert result.mse >= 0

    def test_coverage_in_valid_range(self):
        """Test that coverage is in [0, 1]."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        estimator = NaiveOLS(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        assert 0 <= result.coverage <= 1

    def test_confidence_interval_validity(self):
        """Test that CI bounds are valid."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        estimator = NaiveOLS(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        # Lower bound should be less than upper bound
        assert result.ci_lower < result.ci_upper

    def test_metadata_contains_dgp_info(self):
        """Test that metadata contains DGP information."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.5, random_state=42)
        estimator = NaiveOLS(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        assert result.metadata["dgp_n"] == 500
        assert result.metadata["dgp_p"] == 3
        assert result.metadata["dgp_true_effect"] == 2.5
        assert result.metadata["controls_used"] == False

    def test_result_contains_all_required_fields(self):
        """Test that result has all required fields."""
        dgp = DGPGenerator(n=300, p=2, true_effect=2.0, random_state=42)
        estimator = NaiveOLS(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        # Check required fields
        assert hasattr(result, "method")
        assert hasattr(result, "status")
        assert hasattr(result, "bias")
        assert hasattr(result, "mse")
        assert hasattr(result, "coverage")
        assert hasattr(result, "n_simulations")
        assert hasattr(result, "metadata")
        assert hasattr(result, "bias_p_value")


class TestOLSWithControls:
    """Test suite for OLSWithControls estimator."""

    def test_instantiation(self):
        """Test basic instantiation."""
        estimator = OLSWithControls(n_simulations=10, alpha=0.05, random_state=42)
        assert estimator.n_simulations == 10
        assert estimator.alpha == 0.05
        assert estimator.random_state == 42

    def test_instantiation_with_custom_params(self):
        """Test instantiation with custom parameters."""
        estimator = OLSWithControls(n_simulations=50, alpha=0.01, random_state=123)
        assert estimator.n_simulations == 50
        assert estimator.alpha == 0.01
        assert estimator.random_state == 123

    def test_validate_returns_validation_result(self):
        """Test that validate() returns ValidationResult."""
        estimator = OLSWithControls(n_simulations=10, random_state=42)
        dgp = DGPGenerator(n=200, p=2, true_effect=2.0, random_state=42)

        result = estimator.validate(dgp)

        assert isinstance(result, ValidationResult)
        assert result.method == "OLSWithControls"
        assert result.n_simulations == 10

    def test_less_biased_than_naive(self):
        """Test OLSWithControls is less biased than NaiveOLS."""
        dgp = DGPGenerator(
            n=1000,
            p=5,
            true_effect=2.0,
            confounding_strength=1.0,
            random_state=42,
        )

        naive = NaiveOLS(n_simulations=20, random_state=42)
        controls = OLSWithControls(n_simulations=20, random_state=42)

        result_naive = naive.validate(dgp)
        result_controls = controls.validate(dgp)

        # OLSWithControls should have smaller bias
        assert abs(result_controls.bias) < abs(result_naive.bias)

    def test_reproducibility_same_seed(self):
        """Test reproducibility with same random seed."""
        dgp1 = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        dgp2 = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)

        estimator1 = OLSWithControls(n_simulations=15, random_state=123)
        estimator2 = OLSWithControls(n_simulations=15, random_state=123)

        result1 = estimator1.validate(dgp1)
        result2 = estimator2.validate(dgp2)

        assert np.isclose(result1.bias, result2.bias, atol=1e-10)
        assert np.isclose(result1.mse, result2.mse, atol=1e-10)

    def test_mse_non_negative(self):
        """Test that MSE is always non-negative."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        estimator = OLSWithControls(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        assert result.mse >= 0

    def test_coverage_in_valid_range(self):
        """Test that coverage is in [0, 1]."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        estimator = OLSWithControls(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        assert 0 <= result.coverage <= 1

    def test_confidence_interval_validity(self):
        """Test that CI bounds are valid."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        estimator = OLSWithControls(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        # Lower bound should be less than upper bound
        assert result.ci_lower < result.ci_upper

    def test_metadata_contains_dgp_info(self):
        """Test that metadata contains DGP information."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.5, random_state=42)
        estimator = OLSWithControls(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        assert result.metadata["dgp_n"] == 500
        assert result.metadata["dgp_p"] == 3
        assert result.metadata["dgp_true_effect"] == 2.5
        assert result.metadata["controls_used"] == True

    def test_result_contains_all_required_fields(self):
        """Test that result has all required fields."""
        dgp = DGPGenerator(n=300, p=2, true_effect=2.0, random_state=42)
        estimator = OLSWithControls(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        # Check required fields
        assert hasattr(result, "method")
        assert hasattr(result, "status")
        assert hasattr(result, "bias")
        assert hasattr(result, "mse")
        assert hasattr(result, "coverage")
        assert hasattr(result, "n_simulations")
        assert hasattr(result, "metadata")
        assert hasattr(result, "bias_p_value")
