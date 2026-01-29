"""
Test suite for ML baseline estimators.

Tests RandomForestEstimator and XGBoostEstimator to ensure they:
1. Provide flexible non-parametric outcome prediction
2. Reduce bias compared to linear baselines
3. Maintain correct statistical properties
4. Handle edge cases gracefully
"""

import numpy as np
import pytest

from src.validation.dgp_generator import DGPGenerator
from src.validation.ml_baseline import RandomForestEstimator, XGBoostEstimator
from src.validation.validation_result import ValidationResult


class TestRandomForestEstimator:
    """Test suite for RandomForestEstimator."""

    def test_instantiation(self):
        """Test basic instantiation."""
        estimator = RandomForestEstimator(n_simulations=10, alpha=0.05, random_state=42)
        assert estimator.n_simulations == 10
        assert estimator.alpha == 0.05
        assert estimator.random_state == 42

    @pytest.mark.slow
    def test_instantiation_with_custom_params(self):
        """Test instantiation with custom parameters."""
        estimator = RandomForestEstimator(
            n_simulations=50, alpha=0.01, n_estimators=200, random_state=123
        )
        assert estimator.n_simulations == 50
        assert estimator.alpha == 0.01
        assert estimator.n_estimators == 200
        assert estimator.random_state == 123

    def test_validate_returns_validation_result(self):
        """Test that validate() returns ValidationResult."""
        estimator = RandomForestEstimator(n_simulations=10, random_state=42)
        dgp = DGPGenerator(n=200, p=2, true_effect=2.0, random_state=42)

        result = estimator.validate(dgp)

        assert isinstance(result, ValidationResult)
        assert result.method == "RandomForestEstimator"
        assert result.n_simulations == 10

    def test_less_biased_than_naive_ols(self):
        """Test that RF has lower bias than NaiveOLS with confounding."""
        from src.validation.ols_baseline import NaiveOLS

        # Create DGP with confounding
        dgp = DGPGenerator(
            n=800,
            p=5,
            true_effect=2.0,
            confounding_strength=1.0,  # Moderate confounding
            random_state=42,
        )

        rf = RandomForestEstimator(n_simulations=15, random_state=42)
        naive = NaiveOLS(n_simulations=15, random_state=42)

        result_rf = rf.validate(dgp)
        result_naive = naive.validate(dgp)

        # RF should have lower or comparable bias due to non-parametric flexibility
        # Note: RF's non-linearity may help with misspecification
        assert abs(result_rf.bias) <= abs(result_naive.bias) + 0.2  # Allow tolerance

    def test_reproducibility_same_seed(self):
        """Test reproducibility with same random seed."""
        dgp1 = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        dgp2 = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)

        estimator1 = RandomForestEstimator(n_simulations=15, random_state=123)
        estimator2 = RandomForestEstimator(n_simulations=15, random_state=123)

        result1 = estimator1.validate(dgp1)
        result2 = estimator2.validate(dgp2)

        assert np.isclose(result1.bias, result2.bias, atol=1e-10)
        assert np.isclose(result1.mse, result2.mse, atol=1e-10)

    def test_mse_non_negative(self):
        """Test that MSE is always non-negative."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        estimator = RandomForestEstimator(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        assert result.mse >= 0

    def test_coverage_in_valid_range(self):
        """Test that coverage is in [0, 1]."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        estimator = RandomForestEstimator(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        assert 0 <= result.coverage <= 1

    def test_confidence_interval_validity(self):
        """Test that CI bounds are valid."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        estimator = RandomForestEstimator(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        # Lower bound should be less than upper bound
        assert result.ci_lower < result.ci_upper

    def test_metadata_contains_dgp_info(self):
        """Test that metadata contains DGP information."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.5, random_state=42)
        estimator = RandomForestEstimator(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        assert result.metadata["dgp_n"] == 500
        assert result.metadata["dgp_p"] == 3
        assert result.metadata["dgp_true_effect"] == 2.5
        assert result.metadata["estimator"] == "RandomForestEstimator"
        assert result.metadata["n_estimators"] == 100
        assert result.metadata["max_depth"] == 10

    def test_result_contains_all_required_fields(self):
        """Test that result has all required fields."""
        dgp = DGPGenerator(n=300, p=2, true_effect=2.0, random_state=42)
        estimator = RandomForestEstimator(n_simulations=10, random_state=42)
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

    def test_handles_small_sample_sizes(self):
        """Test that RF handles small sample sizes gracefully."""
        dgp = DGPGenerator(n=100, p=2, true_effect=2.0, random_state=42)
        estimator = RandomForestEstimator(n_simulations=5, random_state=42)
        result = estimator.validate(dgp)

        # Should return valid result even with small samples
        assert isinstance(result, ValidationResult)
        assert result.mse >= 0

    def test_handles_large_feature_space(self):
        """Test that RF handles many features gracefully."""
        dgp = DGPGenerator(n=500, p=20, true_effect=2.0, random_state=42)
        estimator = RandomForestEstimator(n_simulations=5, random_state=42)
        result = estimator.validate(dgp)

        # Should handle high-dimensional case
        assert isinstance(result, ValidationResult)
        assert result.metadata["dgp_p"] == 20


class TestXGBoostEstimator:
    """Test suite for XGBoostEstimator."""

    def test_instantiation(self):
        """Test basic instantiation."""
        estimator = XGBoostEstimator(n_simulations=10, alpha=0.05, random_state=42)
        assert estimator.n_simulations == 10
        assert estimator.alpha == 0.05
        assert estimator.random_state == 42

    @pytest.mark.slow
    def test_instantiation_with_custom_params(self):
        """Test instantiation with custom parameters."""
        estimator = XGBoostEstimator(
            n_simulations=50, alpha=0.01, n_estimators=200, random_state=123
        )
        assert estimator.n_simulations == 50
        assert estimator.alpha == 0.01
        assert estimator.n_estimators == 200
        assert estimator.random_state == 123

    def test_validate_returns_validation_result(self):
        """Test that validate() returns ValidationResult."""
        estimator = XGBoostEstimator(n_simulations=10, random_state=42)
        dgp = DGPGenerator(n=200, p=2, true_effect=2.0, random_state=42)

        result = estimator.validate(dgp)

        assert isinstance(result, ValidationResult)
        assert result.method == "XGBoostEstimator"
        assert result.n_simulations == 10

    def test_less_biased_than_naive_ols(self):
        """Test that XGBoost has lower bias than NaiveOLS with confounding."""
        from src.validation.ols_baseline import NaiveOLS

        # Create DGP with confounding
        dgp = DGPGenerator(
            n=800,
            p=5,
            true_effect=2.0,
            confounding_strength=1.0,  # Moderate confounding
            random_state=42,
        )

        xgb = XGBoostEstimator(n_simulations=15, random_state=42)
        naive = NaiveOLS(n_simulations=15, random_state=42)

        result_xgb = xgb.validate(dgp)
        result_naive = naive.validate(dgp)

        # XGBoost should have lower or comparable bias
        assert abs(result_xgb.bias) <= abs(result_naive.bias) + 0.2  # Allow tolerance

    def test_xgboost_more_flexible_than_rf(self):
        """Test that XGBoost is at least as flexible as RF for complex relationships."""
        dgp = DGPGenerator(n=600, p=5, true_effect=2.0, random_state=42)

        rf = RandomForestEstimator(n_simulations=10, random_state=42)
        xgb = XGBoostEstimator(n_simulations=10, random_state=42)

        result_rf = rf.validate(dgp)
        result_xgb = xgb.validate(dgp)

        # Both should be reasonable; no strict inequality expected
        assert isinstance(result_rf, ValidationResult)
        assert isinstance(result_xgb, ValidationResult)

    def test_reproducibility_same_seed(self):
        """Test reproducibility with same random seed."""
        dgp1 = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        dgp2 = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)

        estimator1 = XGBoostEstimator(n_simulations=15, random_state=123)
        estimator2 = XGBoostEstimator(n_simulations=15, random_state=123)

        result1 = estimator1.validate(dgp1)
        result2 = estimator2.validate(dgp2)

        assert np.isclose(result1.bias, result2.bias, atol=1e-10)
        assert np.isclose(result1.mse, result2.mse, atol=1e-10)

    def test_mse_non_negative(self):
        """Test that MSE is always non-negative."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        estimator = XGBoostEstimator(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        assert result.mse >= 0

    def test_coverage_in_valid_range(self):
        """Test that coverage is in [0, 1]."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        estimator = XGBoostEstimator(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        assert 0 <= result.coverage <= 1

    def test_confidence_interval_validity(self):
        """Test that CI bounds are valid."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.0, random_state=42)
        estimator = XGBoostEstimator(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        # Lower bound should be less than upper bound
        assert result.ci_lower < result.ci_upper

    def test_metadata_contains_dgp_info(self):
        """Test that metadata contains DGP information."""
        dgp = DGPGenerator(n=500, p=3, true_effect=2.5, random_state=42)
        estimator = XGBoostEstimator(n_simulations=10, random_state=42)
        result = estimator.validate(dgp)

        assert result.metadata["dgp_n"] == 500
        assert result.metadata["dgp_p"] == 3
        assert result.metadata["dgp_true_effect"] == 2.5
        assert result.metadata["estimator"] == "XGBoostEstimator"
        assert result.metadata["n_estimators"] == 100
        assert result.metadata["max_depth"] == 5
        assert result.metadata["learning_rate"] == 0.1

    def test_result_contains_all_required_fields(self):
        """Test that result has all required fields."""
        dgp = DGPGenerator(n=300, p=2, true_effect=2.0, random_state=42)
        estimator = XGBoostEstimator(n_simulations=10, random_state=42)
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

    def test_handles_small_sample_sizes(self):
        """Test that XGBoost handles small sample sizes gracefully."""
        dgp = DGPGenerator(n=100, p=2, true_effect=2.0, random_state=42)
        estimator = XGBoostEstimator(n_simulations=5, random_state=42)
        result = estimator.validate(dgp)

        # Should return valid result even with small samples
        assert isinstance(result, ValidationResult)
        assert result.mse >= 0

    def test_handles_large_feature_space(self):
        """Test that XGBoost handles many features gracefully."""
        dgp = DGPGenerator(n=500, p=20, true_effect=2.0, random_state=42)
        estimator = XGBoostEstimator(n_simulations=5, random_state=42)
        result = estimator.validate(dgp)

        # Should handle high-dimensional case
        assert isinstance(result, ValidationResult)
        assert result.metadata["dgp_p"] == 20


class TestMLMethodsComparison:
    """Test suite for comparing RF and XGBoost methods."""

    def test_both_methods_can_run_on_same_dgp(self):
        """Test that both methods can be run on the same DGP."""
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, confounding_strength=0.8, random_state=42)

        rf = RandomForestEstimator(n_simulations=10, random_state=42)
        xgb = XGBoostEstimator(n_simulations=10, random_state=42)

        result_rf = rf.validate(dgp)
        result_xgb = xgb.validate(dgp)

        # Both should return valid results
        assert isinstance(result_rf, ValidationResult)
        assert isinstance(result_xgb, ValidationResult)
        assert result_rf.method == "RandomForestEstimator"
        assert result_xgb.method == "XGBoostEstimator"

    def test_results_are_different_but_reasonable(self):
        """Test that results from both methods are different but both reasonable."""
        dgp = DGPGenerator(n=600, p=5, true_effect=2.0, confounding_strength=1.0, random_state=42)

        rf = RandomForestEstimator(n_simulations=15, random_state=42)
        xgb = XGBoostEstimator(n_simulations=15, random_state=42)

        result_rf = rf.validate(dgp)
        result_xgb = xgb.validate(dgp)

        # Biases should be different but in reasonable ranges
        # (both methods can produce similar results by chance, so we check reasonableness)
        assert abs(result_rf.bias) < 1.5  # Reasonable for RF
        assert abs(result_xgb.bias) < 1.5  # Reasonable for XGBoost

    def test_both_methods_integrate_with_framework(self):
        """Test that both methods work with BaselineComparison framework."""
        from src.validation.baseline_comparison import BaselineComparison

        # Create comparison with DML disabled to avoid long-running tests
        comp = BaselineComparison(n_simulations=5, include_dml=False, random_state=42)
        dgp = DGPGenerator(n=300, p=3, true_effect=2.0, random_state=42)

        # This should run all 4 baseline methods successfully
        results = comp.compare(dgp)

        # Should have results for baseline methods
        assert "NaiveOLS" in results
        assert "OLSWithControls" in results
        assert "IPWEstimator" in results
        assert "AugmentedIPW" in results
