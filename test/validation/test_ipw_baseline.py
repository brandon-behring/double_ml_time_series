"""
Test suite for IPW baseline estimators.

Tests IPWEstimator and AugmentedIPW to ensure they:
1. Provide unbiased estimates with good overlap
2. Remain stable even with poor overlap
3. Show double robustness property (AugmentedIPW)
4. Maintain correct statistical properties
"""

import numpy as np
import pytest

from src.validation.dgp_generator import DGPGenerator
from src.validation.ipw_baseline import IPWEstimator, AugmentedIPW
from src.validation.validation_result import ValidationResult


class TestIPWEstimator:
    """Test suite for IPWEstimator."""

    def test_instantiation(self):
        """Test basic instantiation."""
        estimator = IPWEstimator(n_simulations=10, alpha=0.05, random_state=42)
        assert estimator.n_simulations == 10
        assert estimator.alpha == 0.05
        assert estimator.random_state == 42

    @pytest.mark.slow
    def test_instantiation_with_custom_params(self):
        """Test instantiation with custom parameters."""
        estimator = IPWEstimator(n_simulations=50, alpha=0.01, random_state=123)
        assert estimator.n_simulations == 50
        assert estimator.alpha == 0.01
        assert estimator.random_state == 123

    def test_validate_returns_validation_result(self):
        """Test that validate() returns ValidationResult."""
        estimator = IPWEstimator(n_simulations=10, random_state=42)
        dgp = DGPGenerator(n=200, p=2, true_effect=2.0, random_state=42)

        result = estimator.validate(dgp)

        assert isinstance(result, ValidationResult)
        assert result.method == "IPWEstimator"
        assert result.n_simulations == 10

    def test_unbiased_with_good_overlap(self):
        """Test that IPW is approximately unbiased with good overlap."""
        # Create DGP with weak confounding (good overlap)
        dgp = DGPGenerator(
            n=1000,
            p=5,
            true_effect=2.0,
            confounding_strength=0.5,  # Weak confounding
            random_state=42,
        )

        estimator = IPWEstimator(n_simulations=20, random_state=42)
        result = estimator.validate(dgp)

        # IPW should be approximately unbiased with good overlap
        assert abs(result.bias) < 0.5

    def test_reproducibility_same_seed(self):
        """Test reproducibility with same random seed."""
        dgp1 = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        dgp2 = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)

        estimator1 = IPWEstimator(n_simulations=15, random_state=123)
        estimator2 = IPWEstimator(n_simulations=15, random_state=123)

        result1 = estimator1.validate(dgp1)
        result2 = estimator2.validate(dgp2)

        assert np.isclose(result1.bias, result2.bias, atol=1e-10)
        assert np.isclose(result1.mse, result2.mse, atol=1e-10)

    def test_mse_non_negative(self):
        """Test that MSE is always non-negative."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        estimator = IPWEstimator(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        assert result.mse >= 0

    def test_coverage_in_valid_range(self):
        """Test that coverage is in [0, 1]."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        estimator = IPWEstimator(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        assert 0 <= result.coverage <= 1

    def test_confidence_interval_validity(self):
        """Test that CI bounds are valid."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        estimator = IPWEstimator(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        # Lower bound should be less than upper bound
        assert result.ci_lower < result.ci_upper

    def test_metadata_contains_dgp_info(self):
        """Test that metadata contains DGP information."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.5, random_state=42)
        estimator = IPWEstimator(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        assert result.metadata["dgp_n"] == 500
        assert result.metadata["dgp_p"] == 3
        assert result.metadata["dgp_true_effect"] == 2.5
        assert result.metadata["propensity_model"] == "LogisticRegression"

    def test_result_contains_all_required_fields(self):
        """Test that result has all required fields."""
        dgp = DGPGenerator(n=300, p=2, true_effect=2.0, random_state=42)
        estimator = IPWEstimator(n_simulations=10, random_state=42)
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


class TestAugmentedIPW:
    """Test suite for AugmentedIPW estimator."""

    def test_instantiation(self):
        """Test basic instantiation."""
        estimator = AugmentedIPW(n_simulations=10, alpha=0.05, random_state=42)
        assert estimator.n_simulations == 10
        assert estimator.alpha == 0.05
        assert estimator.random_state == 42

    @pytest.mark.slow
    def test_instantiation_with_custom_params(self):
        """Test instantiation with custom parameters."""
        estimator = AugmentedIPW(n_simulations=50, alpha=0.01, random_state=123)
        assert estimator.n_simulations == 50
        assert estimator.alpha == 0.01
        assert estimator.random_state == 123

    def test_validate_returns_validation_result(self):
        """Test that validate() returns ValidationResult."""
        estimator = AugmentedIPW(n_simulations=10, random_state=42)
        dgp = DGPGenerator(n=200, p=2, true_effect=2.0, random_state=42)

        result = estimator.validate(dgp)

        assert isinstance(result, ValidationResult)
        assert result.method == "AugmentedIPW"
        assert result.n_simulations == 10

    def test_double_robustness_property(self):
        """Test that AugmentedIPW is approximately unbiased even with confounding."""
        # Create DGP with moderate confounding
        dgp = DGPGenerator(
            n=1000,
            p=5,
            true_effect=2.0,
            confounding_strength=1.0,  # Moderate confounding
            random_state=42,
        )

        estimator = AugmentedIPW(n_simulations=20, random_state=42)
        result = estimator.validate(dgp)

        # AugmentedIPW should be approximately unbiased due to double robustness
        # (outcome model correct if linear, or propensity model correct if logistic)
        assert abs(result.bias) < 0.5

    def test_more_robust_than_pure_ipw(self):
        """Test that AugmentedIPW is more robust than pure IPW."""
        # Create DGP with confounding
        dgp = DGPGenerator(
            n=800,
            p=5,
            true_effect=2.0,
            confounding_strength=1.0,
            random_state=42,
        )

        ipw = IPWEstimator(n_simulations=20, random_state=42)
        aipw = AugmentedIPW(n_simulations=20, random_state=42)

        result_ipw = ipw.validate(dgp)
        result_aipw = aipw.validate(dgp)

        # AugmentedIPW should have smaller or equal bias
        assert abs(result_aipw.bias) <= abs(result_ipw.bias) + 0.1  # Allow some tolerance

    def test_reproducibility_same_seed(self):
        """Test reproducibility with same random seed."""
        dgp1 = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        dgp2 = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)

        estimator1 = AugmentedIPW(n_simulations=15, random_state=123)
        estimator2 = AugmentedIPW(n_simulations=15, random_state=123)

        result1 = estimator1.validate(dgp1)
        result2 = estimator2.validate(dgp2)

        assert np.isclose(result1.bias, result2.bias, atol=1e-10)
        assert np.isclose(result1.mse, result2.mse, atol=1e-10)

    def test_mse_non_negative(self):
        """Test that MSE is always non-negative."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        estimator = AugmentedIPW(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        assert result.mse >= 0

    def test_coverage_in_valid_range(self):
        """Test that coverage is in [0, 1]."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        estimator = AugmentedIPW(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        assert 0 <= result.coverage <= 1

    def test_confidence_interval_validity(self):
        """Test that CI bounds are valid."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        estimator = AugmentedIPW(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        # Lower bound should be less than upper bound
        assert result.ci_lower < result.ci_upper

    def test_metadata_contains_dgp_info(self):
        """Test that metadata contains DGP information."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.5, random_state=42)
        estimator = AugmentedIPW(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        assert result.metadata["dgp_n"] == 500
        assert result.metadata["dgp_p"] == 3
        assert result.metadata["dgp_true_effect"] == 2.5
        assert result.metadata["doubly_robust"] is True

    def test_result_contains_all_required_fields(self):
        """Test that result has all required fields."""
        dgp = DGPGenerator(n=300, p=2, true_effect=2.0, random_state=42)
        estimator = AugmentedIPW(n_simulations=10, random_state=42)
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


class TestIPWComparison:
    """Test suite for comparing IPW and AugmentedIPW."""

    def test_both_methods_can_run_on_same_dgp(self):
        """Test that both methods can be run on the same DGP."""
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, confounding_strength=0.8, random_state=42)

        ipw = IPWEstimator(n_simulations=10, random_state=42)
        aipw = AugmentedIPW(n_simulations=10, random_state=42)

        result_ipw = ipw.validate(dgp)
        result_aipw = aipw.validate(dgp)

        # Both should return valid results
        assert isinstance(result_ipw, ValidationResult)
        assert isinstance(result_aipw, ValidationResult)
        assert result_ipw.method == "IPWEstimator"
        assert result_aipw.method == "AugmentedIPW"

    def test_results_are_different_but_reasonable(self):
        """Test that results from both methods are different but both reasonable."""
        dgp = DGPGenerator(n=600, p=5, true_effect=2.0, confounding_strength=1.0, random_state=42)

        ipw = IPWEstimator(n_simulations=20, random_state=42)
        aipw = AugmentedIPW(n_simulations=20, random_state=42)

        result_ipw = ipw.validate(dgp)
        result_aipw = aipw.validate(dgp)

        # Biases should be different but in reasonable ranges
        assert abs(result_ipw.bias - result_aipw.bias) > 0.01  # Different estimates
        assert abs(result_ipw.bias) < 1.5  # Reasonable for IPW
        assert abs(result_aipw.bias) < 1.5  # Reasonable for AIPW
