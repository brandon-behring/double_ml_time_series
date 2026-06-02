"""
Unit tests for EnhancedDGPGenerator.

Tests heterogeneous treatment effects, misspecification scenarios,
and assumption violations.

All tests in this module are unit tests (data generation, no estimation).
"""

import pytest
import numpy as np
from dml_ts.validation.enhanced_dgp import EnhancedDGPGenerator, EnhancedDGPResult

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.tier1


class TestEnhancedDGPBasicFunctionality:
    """Test basic DGP generation functionality."""

    def test_generate_returns_enhanced_result(self):
        """Test that generate() returns EnhancedDGPResult."""
        dgp = EnhancedDGPGenerator(n=100, p=5, base_effect=2.0, random_state=42)
        result = dgp.generate()

        assert isinstance(result, EnhancedDGPResult)
        assert result.Y.shape == (100,)
        assert result.T.shape == (100,)
        assert result.X.shape == (100, 5)
        assert result.true_effect.shape == (100,)
        assert isinstance(result.true_ate, float)

    def test_reproducibility_with_random_state(self):
        """Test that same random_state produces identical results."""
        dgp1 = EnhancedDGPGenerator(n=100, p=5, base_effect=2.0, random_state=42)
        dgp2 = EnhancedDGPGenerator(n=100, p=5, base_effect=2.0, random_state=42)

        result1 = dgp1.generate()
        result2 = dgp2.generate()

        np.testing.assert_array_equal(result1.Y, result2.Y)
        np.testing.assert_array_equal(result1.T, result2.T)
        np.testing.assert_array_equal(result1.X, result2.X)
        np.testing.assert_array_equal(result1.true_effect, result2.true_effect)

    def test_different_random_states_produce_different_results(self):
        """Test that different random_state values produce different results."""
        dgp1 = EnhancedDGPGenerator(n=100, p=5, base_effect=2.0, random_state=42)
        dgp2 = EnhancedDGPGenerator(n=100, p=5, base_effect=2.0, random_state=43)

        result1 = dgp1.generate()
        result2 = dgp2.generate()

        assert not np.array_equal(result1.Y, result2.Y)
        assert not np.array_equal(result1.T, result2.T)

    def test_binary_treatment(self):
        """Test that treatment is binary {0, 1}."""
        dgp = EnhancedDGPGenerator(n=1000, p=5, base_effect=2.0, random_state=42)
        result = dgp.generate()

        unique_values = np.unique(result.T)
        assert set(unique_values).issubset({0.0, 1.0})
        assert 0.0 in unique_values
        assert 1.0 in unique_values

    def test_propensity_scores_in_valid_range(self):
        """Test that propensity scores are in [0, 1]."""
        dgp = EnhancedDGPGenerator(n=100, p=5, base_effect=2.0, random_state=42)
        result = dgp.generate()

        assert np.all(result.propensity_score >= 0)
        assert np.all(result.propensity_score <= 1)
        assert result.propensity_score.shape == (100,)


class TestHeterogeneousTreatmentEffects:
    """Test HTE functionality."""

    def test_constant_effect_when_hte_strength_zero(self):
        """Test that effects are constant when hte_strength=0."""
        dgp = EnhancedDGPGenerator(n=100, p=5, base_effect=2.0, hte_strength=0.0, random_state=42)
        result = dgp.generate()

        # All effects should be exactly base_effect
        np.testing.assert_allclose(result.true_effect, 2.0, atol=1e-10)
        assert result.true_ate == pytest.approx(2.0, abs=1e-10)

    def test_heterogeneous_effects_when_hte_strength_positive(self):
        """Test that effects vary when hte_strength > 0."""
        dgp = EnhancedDGPGenerator(n=1000, p=5, base_effect=2.0, hte_strength=1.0, random_state=42)
        result = dgp.generate()

        # Effects should vary across individuals
        effect_std = np.std(result.true_effect)
        assert effect_std > 0.5  # Should have meaningful variation

        # Mean should be close to base_effect
        assert result.true_ate == pytest.approx(2.0, abs=0.5)

    def test_linear_hte(self):
        """Test linear HTE specification."""
        dgp = EnhancedDGPGenerator(
            n=1000,
            p=5,
            base_effect=2.0,
            hte_strength=1.0,
            hte_nonlinear=False,
            random_state=42,
        )
        result = dgp.generate()

        # Linear HTE should create variation
        assert np.std(result.true_effect) > 0
        assert result.true_effect.min() != result.true_effect.max()

    def test_nonlinear_hte(self):
        """Test non-linear HTE specification."""
        dgp = EnhancedDGPGenerator(
            n=1000,
            p=5,
            base_effect=2.0,
            hte_strength=1.0,
            hte_nonlinear=True,
            random_state=42,
        )
        result = dgp.generate()

        # Non-linear HTE should also create variation
        assert np.std(result.true_effect) > 0
        # Effects should be different from linear case
        dgp_linear = EnhancedDGPGenerator(
            n=1000,
            p=5,
            base_effect=2.0,
            hte_strength=1.0,
            hte_nonlinear=False,
            random_state=42,
        )
        result_linear = dgp_linear.generate()
        assert not np.allclose(result.true_effect, result_linear.true_effect)

    def test_ate_is_mean_of_individual_effects(self):
        """Test that true_ate equals mean of true_effect."""
        dgp = EnhancedDGPGenerator(n=1000, p=5, base_effect=2.0, hte_strength=1.0, random_state=42)
        result = dgp.generate()

        expected_ate = np.mean(result.true_effect)
        assert result.true_ate == pytest.approx(expected_ate, abs=1e-10)


class TestOmittedVariableBias:
    """Test omitted variable functionality."""

    def test_no_omitted_vars_by_default(self):
        """Test that p_omitted=0 by default."""
        dgp = EnhancedDGPGenerator(n=100, p=5, base_effect=2.0, random_state=42)
        result = dgp.generate()

        assert result.p_omitted == 0
        assert result.X_true.shape == (100, 5)  # Same as observed
        np.testing.assert_array_equal(result.X, result.X_true)

    def test_omitted_variables_included_in_x_true(self):
        """Test that X_true includes omitted variables."""
        dgp = EnhancedDGPGenerator(n=100, p=5, p_omitted=3, base_effect=2.0, random_state=42)
        result = dgp.generate()

        assert result.p == 5  # Observed confounders
        assert result.p_omitted == 3
        assert result.X.shape == (100, 5)  # Only observed
        assert result.X_true.shape == (100, 8)  # Observed + omitted

    def test_omitted_variables_affect_treatment(self):
        """Test that omitted variables create confounding."""
        dgp_no_omit = EnhancedDGPGenerator(
            n=1000,
            p=5,
            p_omitted=0,
            base_effect=2.0,
            confounding_strength=1.0,
            random_state=42,
        )
        dgp_with_omit = EnhancedDGPGenerator(
            n=1000,
            p=5,
            p_omitted=3,
            omitted_var_strength=2.0,
            base_effect=2.0,
            confounding_strength=1.0,
            random_state=42,
        )

        result_no_omit = dgp_no_omit.generate()
        result_with_omit = dgp_with_omit.generate()

        # Outcomes should differ due to omitted variable confounding
        assert not np.array_equal(result_no_omit.Y, result_with_omit.Y)


class TestMeasurementError:
    """Test measurement error functionality."""

    def test_no_measurement_error_by_default(self):
        """Test that measurement_error_std=0 by default."""
        dgp = EnhancedDGPGenerator(n=100, p=5, base_effect=2.0, random_state=42)
        result = dgp.generate()

        assert result.measurement_error_std == 0.0
        # X should equal X_true when no omitted vars and no measurement error
        np.testing.assert_array_equal(result.X, result.X_true)

    def test_measurement_error_creates_difference(self):
        """Test that measurement error creates X != X_true."""
        dgp = EnhancedDGPGenerator(
            n=100, p=5, base_effect=2.0, measurement_error_std=0.5, random_state=42
        )
        result = dgp.generate()

        assert result.measurement_error_std == 0.5
        # X should differ from X_true due to measurement error
        assert not np.allclose(result.X, result.X_true)

        # Difference should have appropriate magnitude
        diff = result.X - result.X_true
        assert np.std(diff) > 0.3  # Should be noticeable
        assert np.std(diff) < 0.7  # But not huge

    def test_measurement_error_increases_with_std(self):
        """Test that larger std creates larger errors."""
        dgp_small = EnhancedDGPGenerator(
            n=1000, p=5, base_effect=2.0, measurement_error_std=0.1, random_state=42
        )
        dgp_large = EnhancedDGPGenerator(
            n=1000, p=5, base_effect=2.0, measurement_error_std=1.0, random_state=42
        )

        result_small = dgp_small.generate()
        result_large = dgp_large.generate()

        diff_small = result_small.X - result_small.X_true
        diff_large = result_large.X - result_large.X_true

        assert np.std(diff_large) > np.std(diff_small)


class TestPropensityScoreExtremeness:
    """Test propensity score extremeness functionality."""

    def test_default_propensity_extremeness(self):
        """Test default extremeness=1.0."""
        dgp = EnhancedDGPGenerator(n=1000, p=5, base_effect=2.0, random_state=42)
        result = dgp.generate()

        # With default extremeness, scores can be anywhere in (0,1)
        # Check scores are strictly in (0,1) not on boundaries
        assert result.propensity_score.min() > 0.0
        assert result.propensity_score.max() < 1.0
        # Check for some variation (not all identical)
        assert np.std(result.propensity_score) > 0.1

    def test_increased_extremeness_creates_more_extreme_scores(self):
        """Test that extremeness > 1 pushes scores toward 0/1."""
        dgp_normal = EnhancedDGPGenerator(
            n=1000,
            p=5,
            base_effect=2.0,
            propensity_extremeness=1.0,
            confounding_strength=1.0,
            random_state=42,
        )
        dgp_extreme = EnhancedDGPGenerator(
            n=1000,
            p=5,
            base_effect=2.0,
            propensity_extremeness=3.0,
            confounding_strength=1.0,
            random_state=42,
        )

        result_normal = dgp_normal.generate()
        result_extreme = dgp_extreme.generate()

        # Extreme should have scores closer to 0 or 1
        # Measure by distance from 0.5
        dist_normal = np.abs(result_normal.propensity_score - 0.5).mean()
        dist_extreme = np.abs(result_extreme.propensity_score - 0.5).mean()

        assert dist_extreme > dist_normal


class TestPolynomialTerms:
    """Test polynomial terms functionality."""

    def test_linear_by_default(self):
        """Test that polynomial_degree=1 by default."""
        dgp = EnhancedDGPGenerator(n=100, p=5, base_effect=2.0, random_state=42)
        result = dgp.generate()

        # Just check it runs without error
        assert result.Y.shape == (100,)

    def test_quadratic_terms(self):
        """Test polynomial_degree=2 adds quadratic terms."""
        dgp_linear = EnhancedDGPGenerator(
            n=1000, p=5, base_effect=2.0, polynomial_degree=1, random_state=42
        )
        dgp_quad = EnhancedDGPGenerator(
            n=1000, p=5, base_effect=2.0, polynomial_degree=2, random_state=42
        )

        result_linear = dgp_linear.generate()
        result_quad = dgp_quad.generate()

        # Outcomes should differ due to quadratic terms
        assert not np.allclose(result_linear.Y, result_quad.Y)


class TestScenarioDescription:
    """Test scenario description functionality."""

    def test_basic_description(self):
        """Test get_scenario_description() basic case."""
        dgp = EnhancedDGPGenerator(n=100, p=5, base_effect=2.0, random_state=42)
        desc = dgp.get_scenario_description()

        assert "n=100" in desc
        assert "p=5" in desc
        assert "2.00" in desc  # base_effect

    def test_description_includes_hte(self):
        """Test description includes HTE when present."""
        dgp = EnhancedDGPGenerator(n=100, p=5, base_effect=2.0, hte_strength=1.0, random_state=42)
        desc = dgp.get_scenario_description()

        assert "HTE" in desc
        assert "linear" in desc  # hte_nonlinear=False

    def test_description_includes_omitted_vars(self):
        """Test description includes omitted variables when present."""
        dgp = EnhancedDGPGenerator(n=100, p=5, base_effect=2.0, p_omitted=3, random_state=42)
        desc = dgp.get_scenario_description()

        assert "Omitted" in desc
        assert "3" in desc  # p_omitted

    def test_description_includes_measurement_error(self):
        """Test description includes measurement error when present."""
        dgp = EnhancedDGPGenerator(
            n=100, p=5, base_effect=2.0, measurement_error_std=0.5, random_state=42
        )
        desc = dgp.get_scenario_description()

        assert "Measurement error" in desc
        assert "0.5" in desc or "0.50" in desc


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_sample_size(self):
        """Test with very small n."""
        dgp = EnhancedDGPGenerator(n=10, p=5, base_effect=2.0, random_state=42)
        result = dgp.generate()

        assert result.Y.shape == (10,)
        assert result.T.shape == (10,)

    def test_large_sample_size(self):
        """Test with large n."""
        dgp = EnhancedDGPGenerator(n=10000, p=5, base_effect=2.0, random_state=42)
        result = dgp.generate()

        assert result.Y.shape == (10000,)
        assert result.T.shape == (10000,)

    def test_single_confounder(self):
        """Test with p=1."""
        dgp = EnhancedDGPGenerator(n=100, p=1, base_effect=2.0, random_state=42)
        result = dgp.generate()

        assert result.X.shape == (100, 1)

    def test_many_confounders(self):
        """Test with large p."""
        dgp = EnhancedDGPGenerator(n=100, p=50, base_effect=2.0, random_state=42)
        result = dgp.generate()

        assert result.X.shape == (100, 50)

    def test_zero_base_effect(self):
        """Test with base_effect=0."""
        dgp = EnhancedDGPGenerator(n=100, p=5, base_effect=0.0, random_state=42)
        result = dgp.generate()

        assert result.true_ate == pytest.approx(0.0, abs=1e-10)

    def test_negative_base_effect(self):
        """Test with negative base_effect."""
        dgp = EnhancedDGPGenerator(n=100, p=5, base_effect=-2.0, random_state=42)
        result = dgp.generate()

        assert result.true_ate == pytest.approx(-2.0, abs=1e-10)


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_all_features_together(self):
        """Test DGP with all features enabled."""
        dgp = EnhancedDGPGenerator(
            n=1000,
            p=10,
            base_effect=2.0,
            hte_strength=1.0,
            hte_nonlinear=True,
            p_omitted=3,
            omitted_var_strength=1.5,
            polynomial_degree=2,
            propensity_extremeness=2.0,
            measurement_error_std=0.3,
            confounding_strength=1.0,
            noise_level=1.0,
            random_state=42,
        )
        result = dgp.generate()

        # All dimensions correct
        assert result.Y.shape == (1000,)
        assert result.T.shape == (1000,)
        assert result.X.shape == (1000, 10)  # Observed only
        assert result.X_true.shape == (1000, 13)  # Observed + omitted
        assert result.true_effect.shape == (1000,)

        # HTE creates variation
        assert np.std(result.true_effect) > 0.5

        # Measurement error creates difference
        assert not np.allclose(result.X, result.X_true[:, :10])

        # Propensity scores valid
        assert np.all(result.propensity_score >= 0)
        assert np.all(result.propensity_score <= 1)

    def test_realistic_misspecification_scenario(self):
        """Test realistic DML assumption violation scenario."""
        # Simulates: omitted confounders + measurement error + moderate HTE
        dgp = EnhancedDGPGenerator(
            n=500,
            p=8,
            base_effect=1.5,
            hte_strength=0.5,  # Moderate heterogeneity
            p_omitted=2,  # Unobserved confounders
            omitted_var_strength=1.0,
            measurement_error_std=0.2,  # Noisy covariates
            confounding_strength=1.5,
            random_state=42,
        )
        result = dgp.generate()

        # Should still produce valid data
        assert result.Y.shape == (500,)
        assert result.p == 8
        assert result.p_omitted == 2

        # Treatment should have reasonable prevalence
        prevalence = result.T.mean()
        assert 0.2 < prevalence < 0.8
