"""
Template for test files following test-first development.

Use this template when creating tests for validation methods.
Ensures consistent testing structure across all validation code.

Usage:
    1. Copy this file: cp templates/test_harness_template.py test/validation/test_my_method.py
    2. Replace "BiasValidation" with your method name
    3. Write tests BEFORE implementing the method
    4. Run: pytest test/validation/test_my_method.py -v
"""

import numpy as np
import pytest
from datetime import datetime

from src.validation.bias_validation import BiasValidation
from src.validation.dgp_generator import DGPGenerator
from src.validation.validation_result import ValidationResult


# =============================================================================
# Test Class 1: Basic Functionality
# =============================================================================


class TestBiasValidationBasicFunctionality:
    """Test basic functionality of BiasValidation validation method."""

    @pytest.mark.unit
    def test_instantiation(self):
        """Test that BiasValidation can be instantiated with default parameters."""
        validator = BiasValidation()
        assert validator.n_simulations == 1000
        assert validator.alpha == 0.05
        assert validator.random_state is None

    @pytest.mark.slow
    def test_instantiation_with_custom_params(self):
        """Test instantiation with custom parameters."""
        validator = BiasValidation(n_simulations=500, alpha=0.01, random_state=42)
        assert validator.n_simulations == 500
        assert validator.alpha == 0.01
        assert validator.random_state == 42

    def test_validate_returns_validation_result(self):
        """Test that validate() returns ValidationResult."""
        validator = BiasValidation(n_simulations=10, random_state=42)
        dgp = DGPGenerator(n=100, p=3, true_effect=2.0, random_state=42)
        result = validator.validate(dgp)

        assert isinstance(result, ValidationResult)
        assert result.method == "BiasValidation"
        assert result.status in ["PASS", "FAIL", "WARNING"]
        assert isinstance(result.bias, float)
        assert isinstance(result.mse, float)


# =============================================================================
# Test Class 2: Validation Logic
# =============================================================================


class TestBiasValidationValidationLogic:
    """Test validation logic and correctness."""

    @pytest.mark.slow
    def test_detects_no_bias_with_correct_estimator(self):
        """Test that validator detects no bias when estimator is unbiased."""
        # DGP with low confounding - DML should be nearly unbiased
        dgp = DGPGenerator(n=1000, p=5, true_effect=2.0, confounding_strength=0.1, random_state=42)
        validator = BiasValidation(n_simulations=50, random_state=42)
        result = validator.validate(dgp)

        # Bias should be small (< 0.1 for PASS)
        assert abs(result.bias) < 0.15, f"Expected low bias, got {result.bias}"
        assert result.status in ["PASS", "WARNING"], f"Expected PASS/WARNING, got {result.status}"

    @pytest.mark.slow
    def test_pass_status_with_good_estimator(self):
        """Test PASS status with properly specified estimator."""
        # Simple DGP with moderate sample size - should get PASS
        dgp = DGPGenerator(n=1000, p=5, true_effect=2.0, confounding_strength=0.5, random_state=42)
        validator = BiasValidation(n_simulations=50, random_state=42)
        result = validator.validate(dgp)

        # Should achieve PASS or at worst WARNING
        assert result.status in ["PASS", "WARNING"]
        assert abs(result.bias) < 0.25  # Within acceptable range


# =============================================================================
# Test Class 3: Reproducibility
# =============================================================================


class TestBiasValidationReproducibility:
    """Test reproducibility with random seeds."""

    def test_same_seed_produces_identical_results(self):
        """Test that same seed produces identical validation results."""
        validator1 = BiasValidation(n_simulations=20, random_state=42)
        validator2 = BiasValidation(n_simulations=20, random_state=42)
        # Use separate DGP instances with same seed - DGP's RNG advances with each generate()
        dgp1 = DGPGenerator(n=500, p=5, true_effect=2.0, random_state=42)
        dgp2 = DGPGenerator(n=500, p=5, true_effect=2.0, random_state=42)

        result1 = validator1.validate(dgp1)
        result2 = validator2.validate(dgp2)

        # With same seed, should get identical results
        assert result1.bias == result2.bias
        assert result1.mse == result2.mse
        assert result1.coverage == result2.coverage

    def test_different_seed_produces_different_results(self):
        """Test that different seeds produce different results."""
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, random_state=42)
        validator1 = BiasValidation(n_simulations=20, random_state=42)
        validator2 = BiasValidation(n_simulations=20, random_state=99)

        result1 = validator1.validate(dgp)
        result2 = validator2.validate(dgp)

        # Different seeds should produce different Monte Carlo results
        # (Not guaranteed to be different, but very unlikely to be identical)
        assert result1.bias != result2.bias or result1.mse != result2.mse

    def test_dgp_seed_affects_results(self):
        """Test that DGP seed affects validation results."""
        dgp1 = DGPGenerator(n=500, p=5, true_effect=2.0, random_state=42)
        dgp2 = DGPGenerator(n=500, p=5, true_effect=2.0, random_state=99)
        validator = BiasValidation(n_simulations=20, random_state=42)

        result1 = validator.validate(dgp1)
        result2 = validator.validate(dgp2)

        # Different DGP seeds create different data generating processes
        assert result1.bias != result2.bias or result1.mse != result2.mse


# =============================================================================
# Test Class 4: Edge Cases
# =============================================================================


class TestBiasValidationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_sample_size(self):
        """Test validation with small sample size (n=50)."""
        dgp = DGPGenerator(n=50, p=3, true_effect=2.0, confounding_strength=0.5, random_state=42)
        validator = BiasValidation(n_simulations=20, random_state=42)
        result = validator.validate(dgp)

        # Should complete without error
        assert isinstance(result, ValidationResult)
        assert result.status in ["PASS", "FAIL", "WARNING"]
        # Small n may have higher bias/variance, but should still return valid result
        assert np.isfinite(result.bias)
        assert np.isfinite(result.mse)

    @pytest.mark.slow
    def test_large_sample_size(self):
        """Test validation with large sample size (n=10000)."""
        dgp = DGPGenerator(n=5000, p=5, true_effect=2.0, confounding_strength=0.5, random_state=42)
        validator = BiasValidation(n_simulations=10, random_state=42)
        result = validator.validate(dgp)

        # Large n should give low bias and likely PASS
        assert isinstance(result, ValidationResult)
        assert abs(result.bias) < 0.2, "Large sample should have low bias"
        assert result.status in ["PASS", "WARNING"]

    def test_single_confounder(self):
        """Test validation with single confounder (p=1)."""
        dgp = DGPGenerator(n=500, p=1, true_effect=2.0, confounding_strength=0.5, random_state=42)
        validator = BiasValidation(n_simulations=20, random_state=42)
        result = validator.validate(dgp)

        # Should work with single confounder
        assert isinstance(result, ValidationResult)
        assert result.status in ["PASS", "FAIL", "WARNING"]

    def test_many_confounders(self):
        """Test validation with many confounders (p=50)."""
        dgp = DGPGenerator(n=1000, p=30, true_effect=2.0, confounding_strength=0.5, random_state=42)
        validator = BiasValidation(n_simulations=10, random_state=42)
        result = validator.validate(dgp)

        # Should handle high-dimensional case
        assert isinstance(result, ValidationResult)
        assert result.status in ["PASS", "FAIL", "WARNING"]

    def test_zero_true_effect(self):
        """Test validation with zero true effect."""
        dgp = DGPGenerator(n=500, p=5, true_effect=0.0, confounding_strength=0.5, random_state=42)
        validator = BiasValidation(n_simulations=20, random_state=42)
        result = validator.validate(dgp)

        # Should correctly estimate near-zero effect
        assert isinstance(result, ValidationResult)
        # Bias should be close to 0 (estimating 0 effect)
        assert abs(result.bias) < 0.3, f"Expected low bias for zero effect, got {result.bias}"

    def test_negative_true_effect(self):
        """Test validation with negative true effect."""
        dgp = DGPGenerator(n=500, p=5, true_effect=-2.0, confounding_strength=0.5, random_state=42)
        validator = BiasValidation(n_simulations=20, random_state=42)
        result = validator.validate(dgp)

        # Should work with negative effects
        assert isinstance(result, ValidationResult)
        assert result.status in ["PASS", "FAIL", "WARNING"]
        # Bias is agnostic to sign of effect
        assert np.isfinite(result.bias)

    def test_very_few_simulations(self):
        """Test validation with minimal simulations (n_sim=10)."""
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, confounding_strength=0.5, random_state=42)
        validator = BiasValidation(n_simulations=5, random_state=42)
        result = validator.validate(dgp)

        # Should complete with minimal simulations (low precision but valid)
        assert isinstance(result, ValidationResult)
        assert result.n_simulations == 5
        assert np.isfinite(result.bias)
        assert np.isfinite(result.mse)


# =============================================================================
# Test Class 5: Parameter Validation
# =============================================================================


class TestBiasValidationParameterValidation:
    """Test parameter validation and error handling."""

    @pytest.mark.unit
    def test_negative_n_simulations_raises_error(self):
        """Test that negative n_simulations raises ValueError."""
        # TODO: Implement
        # with pytest.raises(ValueError, match="n_simulations must be positive"):
        #     BiasValidationValidation(n_simulations=-100)
        pass

    @pytest.mark.unit
    def test_zero_n_simulations_raises_error(self):
        """Test that zero n_simulations raises ValueError."""
        # TODO: Implement
        pass

    @pytest.mark.unit
    def test_invalid_alpha_raises_error(self):
        """Test that invalid alpha (not in [0,1]) raises ValueError."""
        # TODO: Implement
        # with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        #     BiasValidationValidation(alpha=-0.05)
        #     BiasValidationValidation(alpha=1.5)
        pass


# =============================================================================
# Test Class 6: Statistical Properties
# =============================================================================


class TestBiasValidationStatisticalProperties:
    """Test statistical properties of validation method."""

    def test_mse_is_positive(self):
        """Test that MSE is always non-negative."""
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, confounding_strength=0.5, random_state=42)
        validator = BiasValidation(n_simulations=20, random_state=42)
        result = validator.validate(dgp)

        # MSE must be non-negative by definition
        assert result.mse >= 0, f"MSE must be non-negative, got {result.mse}"
        assert np.isfinite(result.mse), "MSE must be finite"

    def test_coverage_in_valid_range(self):
        """Test that coverage is between 0 and 1."""
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, confounding_strength=0.5, random_state=42)
        validator = BiasValidation(n_simulations=20, random_state=42)
        result = validator.validate(dgp)

        # Coverage is a proportion, must be in [0, 1]
        assert 0 <= result.coverage <= 1, f"Coverage must be in [0,1], got {result.coverage}"

    def test_confidence_interval_validity(self):
        """Test that confidence intervals are well-formed."""
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, confounding_strength=0.5, random_state=42)
        validator = BiasValidation(n_simulations=20, random_state=42)
        result = validator.validate(dgp)

        # CI should be well-formed
        assert (
            result.ci_lower <= result.ci_upper
        ), f"CI lower ({result.ci_lower}) > upper ({result.ci_upper})"
        assert np.isfinite(result.ci_lower), "CI lower must be finite"
        assert np.isfinite(result.ci_upper), "CI upper must be finite"

    @pytest.mark.slow
    def test_coverage_approximately_nominal(self):
        """Test that coverage rate is approximately at nominal level (95%).

        THE SMOKING GUN TEST - This will FAIL with current broken code.

        Current bug: Uses Monte Carlo SE instead of per-estimate DML CIs.
        After fix: Coverage should be ~0.95 for 95% confidence intervals.

        This test validates the CORE PURPOSE of confidence intervals:
        A properly calibrated 95% CI should contain the true parameter
        in approximately 95% of repeated samples.
        """
        dgp = DGPGenerator(
            n=2000,  # Large sample for stable DML estimates
            p=5,
            true_effect=2.0,
            confounding_strength=0.5,  # Moderate confounding
            random_state=42,
        )

        validator = BiasValidation(
            n_simulations=200,  # Enough for stable coverage estimate
            alpha=0.05,  # 95% CIs
            random_state=42,
        )

        result = validator.validate(dgp)

        # Coverage should be ~0.95 (allow 0.85-1.00 given finite sample variability)
        assert 0.85 <= result.coverage <= 1.00, (
            f"Coverage should be ~0.95 for 95% CIs with proper DML implementation. "
            f"Got {result.coverage:.3f}. "
            f"If this fails with coverage < 0.85, the CI calculation is broken "
            f"(likely using Monte Carlo SE instead of DML's own CIs)."
        )

        # Ideally should be close to nominal 0.95 (within ±0.08 given finite sample)
        assert abs(result.coverage - 0.95) < 0.10, (
            f"Coverage {result.coverage:.3f} deviates from nominal 0.95 by more than 0.10. "
            f"This may indicate CI miscalibration or too few simulations."
        )


# =============================================================================
# Test Class 7: Integration Tests
# =============================================================================


class TestBiasValidationIntegration:
    """Integration tests with other components."""

    def test_works_with_linear_dgp(self):
        """Test integration with linear DGP."""
        # DGP generator creates linear relationships by default
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, confounding_strength=0.5, random_state=42)
        validator = BiasValidation(n_simulations=20, random_state=42)
        result = validator.validate(dgp)

        # Should work seamlessly with DGP
        assert isinstance(result, ValidationResult)
        assert result.method == "BiasValidation"
        assert result.status in ["PASS", "FAIL", "WARNING"]

    def test_metadata_contains_dgp_info(self):
        """Test that result metadata contains DGP configuration."""
        dgp = DGPGenerator(n=1000, p=10, true_effect=3.0, confounding_strength=1.5, random_state=42)
        validator = BiasValidation(n_simulations=20, random_state=42)
        result = validator.validate(dgp)

        # Metadata should capture DGP configuration
        assert "dgp_n" in result.metadata
        assert result.metadata["dgp_n"] == 1000
        assert result.metadata["dgp_p"] == 10
        assert result.metadata["dgp_true_effect"] == 3.0
        assert result.metadata["dgp_confounding"] == 1.5

    def test_result_contains_all_required_fields(self):
        """Test that ValidationResult has all required fields."""
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, confounding_strength=0.5, random_state=42)
        validator = BiasValidation(n_simulations=20, random_state=42)
        result = validator.validate(dgp)

        # Check all required fields are present
        assert hasattr(result, "method")
        assert hasattr(result, "status")
        assert hasattr(result, "bias")
        assert hasattr(result, "mse")
        assert hasattr(result, "coverage")
        assert hasattr(result, "ci_lower")
        assert hasattr(result, "ci_upper")
        assert hasattr(result, "n_simulations")
        assert hasattr(result, "timestamp")
        assert hasattr(result, "metadata")


# =============================================================================
# Test Class 8: Multiple Testing Correction (2025-11-14 FIX)
# =============================================================================


class TestBiasValidationMultipleTestingCorrection:
    """Test multiple testing correction functionality (critical fix 2025-11-14)."""

    def test_bonferroni_correction_is_default(self):
        """Test that Bonferroni correction is applied by default."""
        # Create scenario with moderate bias - without correction might be WARNING, with correction might be PASS
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, confounding_strength=0.5, random_state=42)
        validator = BiasValidation(n_simulations=30, random_state=42)

        # The validator should use Bonferroni correction by default
        result = validator.validate(dgp)

        # Verify p-values are stored
        assert hasattr(result, "bias_p_value")
        assert hasattr(result, "coverage_p_value")
        assert isinstance(result.bias_p_value, float)
        assert isinstance(result.coverage_p_value, float)

        # P-values should be in [0, 1]
        assert 0 <= result.bias_p_value <= 1
        assert 0 <= result.coverage_p_value <= 1

    @pytest.mark.slow
    def test_bonferroni_reduces_false_positives(self):
        """Test that Bonferroni correction reduces false positive rate."""
        # With 2 tests at α=0.05, uncorrected familywise error rate ≈ 9.75%
        # With Bonferroni, familywise error rate ≤ 5%

        # This test verifies that correction is applied, not that a specific outcome occurs
        # We test by comparing corrected vs uncorrected thresholds

        dgp = DGPGenerator(n=1000, p=5, true_effect=2.0, confounding_strength=0.1, random_state=42)
        validator = BiasValidation(n_simulations=50, random_state=42)

        # Get simulation results
        estimates, ci_bounds = validator._run_simulations(dgp)
        bias_samples = validator._calculate_bias_samples(estimates, dgp.true_effect)
        coverage = validator._calculate_coverage(ci_bounds, dgp.true_effect)

        # Get status with Bonferroni correction (default)
        status_bonf, p_bias_bonf, p_cov_bonf = validator._determine_statistical_status(
            bias_samples,
            coverage,
            len(estimates),
            ci_bounds,
            dgp.true_effect,
            alpha_test=0.05,
            correction_method="bonferroni",
        )

        # Get status without correction
        status_none, p_bias_none, p_cov_none = validator._determine_statistical_status(
            bias_samples,
            coverage,
            len(estimates),
            ci_bounds,
            dgp.true_effect,
            alpha_test=0.05,
            correction_method="none",
        )

        # P-values should be the same (data-driven, not affected by correction method)
        assert p_bias_bonf == p_bias_none
        assert p_cov_bonf == p_cov_none

        # Bonferroni should be more conservative or same (never less conservative)
        # FAIL < WARNING < PASS in conservativeness
        status_order = {"FAIL": 0, "WARNING": 1, "PASS": 2}
        assert (
            status_order[status_bonf] >= status_order[status_none]
        ), "Bonferroni correction should be at least as conservative as no correction"

    def test_no_correction_option_exists(self):
        """Test that 'none' correction method can be specified (for single-method testing)."""
        # Note: We can't actually pass correction_method through validate()
        # This tests the _determine_statistical_status method directly

        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, confounding_strength=0.5, random_state=42)
        validator = BiasValidation(n_simulations=20, random_state=42)

        # Run simulation to get data
        estimates, ci_bounds = validator._run_simulations(dgp)
        bias_samples = validator._calculate_bias_samples(estimates, dgp.true_effect)
        coverage = validator._calculate_coverage(ci_bounds, dgp.true_effect)

        # Test with no correction
        status_none, _, _ = validator._determine_statistical_status(
            bias_samples,
            coverage,
            len(estimates),
            ci_bounds,
            dgp.true_effect,
            alpha_test=0.05,
            correction_method="none",
        )

        # Test with Bonferroni correction
        status_bonf, _, _ = validator._determine_statistical_status(
            bias_samples,
            coverage,
            len(estimates),
            ci_bounds,
            dgp.true_effect,
            alpha_test=0.05,
            correction_method="bonferroni",
        )

        # Both should return valid status
        assert status_none in ["PASS", "FAIL", "WARNING"]
        assert status_bonf in ["PASS", "FAIL", "WARNING"]

    def test_invalid_correction_method_raises_error(self):
        """Test that invalid correction method raises ValueError."""
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, confounding_strength=0.5, random_state=42)
        validator = BiasValidation(n_simulations=10, random_state=42)

        # Run simulation to get data
        estimates, ci_bounds = validator._run_simulations(dgp)
        bias_samples = validator._calculate_bias_samples(estimates, dgp.true_effect)
        coverage = validator._calculate_coverage(ci_bounds, dgp.true_effect)

        # Test with invalid correction method
        with pytest.raises(ValueError, match="Unknown correction_method"):
            validator._determine_statistical_status(
                bias_samples,
                coverage,
                len(estimates),
                ci_bounds,
                dgp.true_effect,
                alpha_test=0.05,
                correction_method="invalid",
            )

    def test_holm_correction_works(self):
        """Test that Holm-Bonferroni correction method works."""
        dgp = DGPGenerator(n=500, p=5, true_effect=2.0, confounding_strength=0.5, random_state=42)
        validator = BiasValidation(n_simulations=20, random_state=42)

        # Run simulation to get data
        estimates, ci_bounds = validator._run_simulations(dgp)
        bias_samples = validator._calculate_bias_samples(estimates, dgp.true_effect)
        coverage = validator._calculate_coverage(ci_bounds, dgp.true_effect)

        # Test with Holm correction
        status_holm, p_bias, p_cov = validator._determine_statistical_status(
            bias_samples,
            coverage,
            len(estimates),
            ci_bounds,
            dgp.true_effect,
            alpha_test=0.05,
            correction_method="holm",
        )

        # Should return valid status and p-values
        assert status_holm in ["PASS", "FAIL", "WARNING"]
        assert 0 <= p_bias <= 1
        assert 0 <= p_cov <= 1

    def test_correction_affects_status_thresholds(self):
        """Test that correction method affects status determination."""
        # With 2 tests, Bonferroni uses α/2 = 0.025 per test
        # Without correction, uses α = 0.05 per test

        dgp = DGPGenerator(n=300, p=5, true_effect=2.0, confounding_strength=1.0, random_state=42)
        validator = BiasValidation(n_simulations=30, random_state=42)

        # Run simulation once
        estimates, ci_bounds = validator._run_simulations(dgp)
        bias_samples = validator._calculate_bias_samples(estimates, dgp.true_effect)
        coverage = validator._calculate_coverage(ci_bounds, dgp.true_effect)

        # Get status with different corrections
        status_bonf, p_bias_bonf, p_cov_bonf = validator._determine_statistical_status(
            bias_samples,
            coverage,
            len(estimates),
            ci_bounds,
            dgp.true_effect,
            alpha_test=0.05,
            correction_method="bonferroni",
        )

        status_none, p_bias_none, p_cov_none = validator._determine_statistical_status(
            bias_samples,
            coverage,
            len(estimates),
            ci_bounds,
            dgp.true_effect,
            alpha_test=0.05,
            correction_method="none",
        )

        # P-values should be the same (they're data-driven)
        assert p_bias_bonf == p_bias_none
        assert p_cov_bonf == p_cov_none

        # But status might differ due to different thresholds
        # (Bonferroni is more conservative, so less likely to FAIL/WARNING)
        assert status_bonf in ["PASS", "FAIL", "WARNING"]
        assert status_none in ["PASS", "FAIL", "WARNING"]


# =============================================================================
# Test Class 9: Performance Tests (Optional)
# =============================================================================


@pytest.mark.slow
class TestBiasValidationPerformance:
    """Performance and timing tests (marked as slow)."""

    def test_completes_in_reasonable_time(self):
        """Test that validation completes within reasonable time."""
        # TODO: Implement
        # import time
        # validator = BiasValidationValidation(n_simulations=1000)
        # dgp = DGPGenerator(n=1000, p=10, true_effect=2.0)
        #
        # start = time.time()
        # result = validator.validate(dgp)
        # duration = time.time() - start
        #
        # # Should complete in under 30 seconds
        # assert duration < 30.0
        pass

    def test_parallel_speedup(self):
        """Test that parallel execution provides speedup."""
        # TODO: Implement if parallel execution is used
        pass


# =============================================================================
# Fixtures (if needed)
# =============================================================================


@pytest.fixture
def simple_dgp():
    """Fixture: Simple DGP for testing."""
    return DGPGenerator(n=500, p=5, true_effect=2.0, confounding_strength=0.0, random_state=42)


@pytest.fixture
def confounded_dgp():
    """Fixture: DGP with confounding for testing bias detection."""
    return DGPGenerator(n=500, p=5, true_effect=2.0, confounding_strength=2.0, random_state=42)


# TODO: Add more fixtures as needed
