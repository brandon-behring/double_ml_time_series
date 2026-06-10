"""
Tests for DGP (Data Generating Process) Generator.

Test-first development: These tests are written BEFORE implementation.
The DGP generator must create synthetic data with known properties for validation.

All tests in this module are unit tests (data generation, no estimation).
"""

from dataclasses import asdict

import numpy as np
import pytest

from dml_ts.validation.dgp_generator import DGPGenerator, DGPResult

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.tier1


class TestDGPResult:
    """Test the DGPResult dataclass."""

    def test_dgp_result_creation(self):
        """Test DGPResult can be created with required fields."""
        result = DGPResult(
            Y=np.array([1.0, 2.0, 3.0]),
            T=np.array([0, 1, 0]),
            X=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            true_effect=2.0,
            n=3,
            p=2,
        )
        assert result.n == 3
        assert result.p == 2
        assert result.true_effect == 2.0
        assert len(result.Y) == 3
        assert len(result.T) == 3
        assert result.X.shape == (3, 2)

    def test_dgp_result_to_dict(self):
        """Test DGPResult can be converted to dictionary."""
        result = DGPResult(
            Y=np.array([1.0]),
            T=np.array([0]),
            X=np.array([[1.0]]),
            true_effect=2.0,
            n=1,
            p=1,
        )
        d = asdict(result)
        assert "Y" in d
        assert "T" in d
        assert "X" in d
        assert d["true_effect"] == 2.0


class TestDGPGeneratorLinear:
    """Test DGP generator with linear specification."""

    def test_linear_no_confounding(self):
        """Test linear DGP with no confounding returns correct shapes."""
        dgp = DGPGenerator(
            n=1000,
            p=5,
            true_effect=3.0,
            confounding_strength=0.0,  # No confounding
            treatment_model="linear",
            outcome_model="linear",
            random_state=42,
        )

        result = dgp.generate()

        assert result.n == 1000
        assert result.p == 5
        assert result.true_effect == 3.0
        assert result.Y.shape == (1000,)
        assert result.T.shape == (1000,)
        assert result.X.shape == (1000, 5)

    def test_linear_dgp_recovers_true_effect(self):
        """Test that linear DGP with no confounding can recover true effect."""
        dgp = DGPGenerator(
            n=5000,
            p=3,
            true_effect=2.5,
            confounding_strength=0.0,  # No confounding
            treatment_model="linear",
            outcome_model="linear",
            random_state=42,
        )

        result = dgp.generate()

        # Simple OLS should recover true effect with no confounding
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        X_with_T = np.column_stack([result.T, result.X])
        model.fit(X_with_T, result.Y)

        # Coefficient on treatment should be close to true effect
        estimated_effect = model.coef_[0]
        assert abs(estimated_effect - 2.5) < 0.1  # Within 0.1

    def test_reproducibility_with_seed(self):
        """Test that same seed produces identical data."""
        dgp1 = DGPGenerator(n=100, p=5, true_effect=2.0, random_state=42)
        dgp2 = DGPGenerator(n=100, p=5, true_effect=2.0, random_state=42)

        result1 = dgp1.generate()
        result2 = dgp2.generate()

        np.testing.assert_array_equal(result1.Y, result2.Y)
        np.testing.assert_array_equal(result1.T, result2.T)
        np.testing.assert_array_equal(result1.X, result2.X)

    def test_different_seed_produces_different_data(self):
        """Test that different seeds produce different data."""
        dgp1 = DGPGenerator(n=100, p=5, true_effect=2.0, random_state=42)
        dgp2 = DGPGenerator(n=100, p=5, true_effect=2.0, random_state=43)

        result1 = dgp1.generate()
        result2 = dgp2.generate()

        assert not np.array_equal(result1.Y, result2.Y)
        assert not np.array_equal(result1.T, result2.T)


class TestDGPGeneratorNonlinear:
    """Test DGP generator with nonlinear specification."""

    def test_nonlinear_treatment_model(self):
        """Test nonlinear treatment model creates valid data."""
        dgp = DGPGenerator(
            n=1000,
            p=5,
            true_effect=2.0,
            treatment_model="nonlinear",
            outcome_model="linear",
            random_state=42,
        )

        result = dgp.generate()

        # Treatment should still be binary
        assert set(result.T).issubset({0, 1})
        # Should have both treatment and control
        assert 0 in result.T
        assert 1 in result.T

    def test_nonlinear_outcome_model(self):
        """Test nonlinear outcome model creates valid data."""
        dgp = DGPGenerator(
            n=1000,
            p=5,
            true_effect=2.0,
            treatment_model="linear",
            outcome_model="nonlinear",
            random_state=42,
        )

        result = dgp.generate()

        # Outcome should be continuous
        assert result.Y.dtype == np.float64
        # Should have variation
        assert np.std(result.Y) > 0

    def test_nonlinear_both_models(self):
        """Test fully nonlinear DGP."""
        dgp = DGPGenerator(
            n=1000,
            p=5,
            true_effect=2.0,
            treatment_model="nonlinear",
            outcome_model="nonlinear",
            random_state=42,
        )

        result = dgp.generate()

        assert result.Y.shape == (1000,)
        assert result.T.shape == (1000,)
        assert result.X.shape == (1000, 5)


class TestDGPGeneratorConfounding:
    """Test DGP generator with confounding."""

    def test_confounding_creates_bias(self):
        """Test that confounding creates bias in naive estimator."""
        dgp = DGPGenerator(
            n=5000,
            p=5,
            true_effect=2.0,
            confounding_strength=2.0,  # Strong confounding
            treatment_model="linear",
            outcome_model="linear",
            random_state=42,
        )

        result = dgp.generate()

        # Naive OLS (without controls) should be biased
        from sklearn.linear_model import LinearRegression

        naive_model = LinearRegression()
        naive_model.fit(result.T.reshape(-1, 1), result.Y)
        naive_effect = naive_model.coef_[0]

        # Should be significantly different from true effect
        assert abs(naive_effect - 2.0) > 0.2  # At least 0.2 bias

    def test_confounding_controlled_recovers_effect(self):
        """Test that controlling for X recovers true effect with confounding."""
        dgp = DGPGenerator(
            n=5000,
            p=5,
            true_effect=2.5,
            confounding_strength=1.5,
            treatment_model="linear",
            outcome_model="linear",
            random_state=42,
        )

        result = dgp.generate()

        # OLS with controls should recover true effect
        from sklearn.linear_model import LinearRegression

        controlled_model = LinearRegression()
        X_with_T = np.column_stack([result.T, result.X])
        controlled_model.fit(X_with_T, result.Y)
        controlled_effect = controlled_model.coef_[0]

        # Should be close to true effect
        assert abs(controlled_effect - 2.5) < 0.15


class TestDGPGeneratorParameterValidation:
    """Test parameter validation in DGP generator."""

    def test_negative_n_raises_error(self):
        """Test that negative n raises ValueError."""
        with pytest.raises(ValueError, match="n must be positive"):
            DGPGenerator(n=-100, p=5, true_effect=2.0)

    def test_zero_n_raises_error(self):
        """Test that zero n raises ValueError."""
        with pytest.raises(ValueError, match="n must be positive"):
            DGPGenerator(n=0, p=5, true_effect=2.0)

    def test_negative_p_raises_error(self):
        """Test that negative p raises ValueError."""
        with pytest.raises(ValueError, match="p must be positive"):
            DGPGenerator(n=100, p=-5, true_effect=2.0)

    def test_zero_p_raises_error(self):
        """Test that zero p raises ValueError."""
        with pytest.raises(ValueError, match="p must be positive"):
            DGPGenerator(n=100, p=0, true_effect=2.0)

    def test_invalid_treatment_model_raises_error(self):
        """Test that invalid treatment model raises ValueError."""
        with pytest.raises(ValueError, match="treatment_model must be"):
            DGPGenerator(n=100, p=5, true_effect=2.0, treatment_model="invalid")

    def test_invalid_outcome_model_raises_error(self):
        """Test that invalid outcome model raises ValueError."""
        with pytest.raises(ValueError, match="outcome_model must be"):
            DGPGenerator(n=100, p=5, true_effect=2.0, outcome_model="invalid")


class TestDGPGeneratorEdgeCases:
    """Test edge cases for DGP generator."""

    def test_small_sample_size(self):
        """Test DGP works with small sample size."""
        dgp = DGPGenerator(n=10, p=2, true_effect=2.0, random_state=42)
        result = dgp.generate()

        assert result.n == 10
        assert len(result.Y) == 10

    def test_single_confounder(self):
        """Test DGP works with single confounder."""
        dgp = DGPGenerator(n=100, p=1, true_effect=2.0, random_state=42)
        result = dgp.generate()

        assert result.p == 1
        assert result.X.shape == (100, 1)

    def test_many_confounders(self):
        """Test DGP works with many confounders."""
        dgp = DGPGenerator(n=100, p=50, true_effect=2.0, random_state=42)
        result = dgp.generate()

        assert result.p == 50
        assert result.X.shape == (100, 50)

    def test_zero_true_effect(self):
        """Test DGP works with zero true effect."""
        dgp = DGPGenerator(n=1000, p=5, true_effect=0.0, random_state=42)
        result = dgp.generate()

        assert result.true_effect == 0.0

    def test_negative_true_effect(self):
        """Test DGP works with negative true effect."""
        dgp = DGPGenerator(n=1000, p=5, true_effect=-2.5, random_state=42)
        result = dgp.generate()

        assert result.true_effect == -2.5


class TestDGPGeneratorMultipleGeneration:
    """Test multiple data generations from same DGP."""

    def test_multiple_generations_different_data(self):
        """Test that multiple generations produce different data."""
        dgp = DGPGenerator(n=100, p=5, true_effect=2.0, random_state=42)

        result1 = dgp.generate()
        result2 = dgp.generate()

        # Should be different realizations
        assert not np.array_equal(result1.Y, result2.Y)
        assert not np.array_equal(result1.T, result2.T)

    def test_resetting_seed_reproduces_data(self):
        """Test that resetting random state reproduces data."""
        dgp = DGPGenerator(n=100, p=5, true_effect=2.0, random_state=42)
        result1 = dgp.generate()

        # Create new generator with same seed
        dgp2 = DGPGenerator(n=100, p=5, true_effect=2.0, random_state=42)
        result2 = dgp2.generate()

        np.testing.assert_array_equal(result1.Y, result2.Y)
        np.testing.assert_array_equal(result1.T, result2.T)
        np.testing.assert_array_equal(result1.X, result2.X)


class TestDGPGeneratorStatisticalProperties:
    """Test statistical properties of generated data."""

    def test_treatment_balance(self):
        """Test treatment assignment is reasonably balanced."""
        dgp = DGPGenerator(
            n=10000,
            p=5,
            true_effect=2.0,
            treatment_model="linear",
            random_state=42,
        )

        result = dgp.generate()

        treatment_rate = result.T.mean()
        # Should be between 20% and 80%
        assert 0.2 < treatment_rate < 0.8

    def test_confounders_standardized(self):
        """Test that confounders are approximately standardized."""
        dgp = DGPGenerator(n=10000, p=5, true_effect=2.0, random_state=42)

        result = dgp.generate()

        # Mean should be close to 0
        assert np.abs(result.X.mean()) < 0.1
        # Std should be close to 1
        assert 0.9 < np.std(result.X) < 1.1

    def test_outcome_has_variation(self):
        """Test that outcome variable has meaningful variation."""
        dgp = DGPGenerator(n=1000, p=5, true_effect=2.0, random_state=42)

        result = dgp.generate()

        # Outcome should have standard deviation > 0
        assert np.std(result.Y) > 0.5
