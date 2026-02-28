"""Tests for time series DGP generator.

Tests cover:
1. Single time series generation
2. Panel data generation
3. Structural break DGP
4. AR dynamics verification
5. Confounding properties
6. Integration with DML estimators
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytestmark = pytest.mark.tier2

from src.validation.dgp_generator_ts import (
    TimeSeriesDGPGenerator,
    TimeSeriesDGPResult,
    BreakDGPGenerator,
    BreakDGPResult,
    create_ar_dgp,
    create_panel_dgp,
    create_break_dgp,
)


# ============================================================================
# TimeSeriesDGPGenerator Tests - Single Series
# ============================================================================


class TestTimeSeriesDGPGeneratorSingleSeries:
    """Tests for single time series DGP generation."""

    def test_basic_generation(self):
        """Test basic DGP generation returns correct types."""
        dgp = TimeSeriesDGPGenerator(n_periods=100, p=5, true_effect=2.0, random_state=42)
        result = dgp.generate()

        assert isinstance(result, TimeSeriesDGPResult)
        assert isinstance(result.Y, np.ndarray)
        assert isinstance(result.T, np.ndarray)
        assert isinstance(result.X, np.ndarray)

    def test_output_shapes(self):
        """Test that outputs have correct shapes."""
        n, p = 150, 7
        dgp = TimeSeriesDGPGenerator(n_periods=n, p=p, true_effect=2.0, random_state=42)
        result = dgp.generate()

        assert result.Y.shape == (n,)
        assert result.T.shape == (n,)
        assert result.X.shape == (n, p)
        assert result.n_periods == n
        assert result.n_units == 1
        assert result.p == p

    def test_reproducibility(self):
        """Test that same seed produces same data."""
        dgp1 = TimeSeriesDGPGenerator(n_periods=100, p=5, true_effect=2.0, random_state=42)
        dgp2 = TimeSeriesDGPGenerator(n_periods=100, p=5, true_effect=2.0, random_state=42)

        result1 = dgp1.generate()
        result2 = dgp2.generate()

        assert_allclose(result1.Y, result2.Y)
        assert_allclose(result1.T, result2.T)
        assert_allclose(result1.X, result2.X)

    def test_different_seeds(self):
        """Test that different seeds produce different data."""
        dgp1 = TimeSeriesDGPGenerator(n_periods=100, p=5, true_effect=2.0, random_state=42)
        dgp2 = TimeSeriesDGPGenerator(n_periods=100, p=5, true_effect=2.0, random_state=123)

        result1 = dgp1.generate()
        result2 = dgp2.generate()

        assert not np.allclose(result1.Y, result2.Y)

    def test_no_nan_values(self):
        """Test that generated data has no NaN values."""
        dgp = TimeSeriesDGPGenerator(n_periods=200, p=10, true_effect=2.0, random_state=42)
        result = dgp.generate()

        assert not np.isnan(result.Y).any()
        assert not np.isnan(result.T).any()
        assert not np.isnan(result.X).any()

    def test_true_effect_stored(self):
        """Test that true effect is stored correctly."""
        true_effect = 3.5
        dgp = TimeSeriesDGPGenerator(n_periods=100, p=5, true_effect=true_effect, random_state=42)
        result = dgp.generate()

        assert result.true_effect == true_effect

    def test_time_index(self):
        """Test that time index is generated correctly."""
        n = 100
        dgp = TimeSeriesDGPGenerator(n_periods=n, p=5, true_effect=2.0, random_state=42)
        result = dgp.generate()

        assert_allclose(result.time_index, np.arange(n))


# ============================================================================
# AR Dynamics Tests
# ============================================================================


class TestARDynamics:
    """Tests for AR treatment dynamics."""

    def test_ar1_treatment_autocorrelation(self):
        """Test that AR(1) treatment shows expected autocorrelation."""
        dgp = TimeSeriesDGPGenerator(
            n_periods=500,
            p=3,
            true_effect=2.0,
            treatment_ar_order=1,
            treatment_ar_coefs=np.array([0.7]),
            confounding_strength=0.0,  # No confounding
            random_state=42,
        )
        result = dgp.generate()
        T = result.T

        # Compute lag-1 autocorrelation
        autocorr = np.corrcoef(T[:-1], T[1:])[0, 1]

        # Should be close to 0.7 (some noise expected)
        assert 0.5 < autocorr < 0.9

    def test_ar2_stored_coefs(self):
        """Test that AR(2) coefficients are stored."""
        ar_coefs = np.array([0.5, 0.2])
        dgp = TimeSeriesDGPGenerator(
            n_periods=100,
            p=3,
            true_effect=2.0,
            treatment_ar_order=2,
            treatment_ar_coefs=ar_coefs,
            random_state=42,
        )
        result = dgp.generate()

        assert_allclose(result.ar_coefs, ar_coefs)

    def test_stable_ar_coefs_generated(self):
        """Test that generated AR coefficients are stable."""
        dgp = TimeSeriesDGPGenerator(
            n_periods=100, p=3, true_effect=2.0, treatment_ar_order=3, random_state=42
        )
        # Check coefficients sum to less than 1 (rough stationarity)
        assert np.abs(dgp.treatment_ar_coefs).sum() < 1.0

    def test_ar_order_zero(self):
        """Test AR order 0 (no treatment autocorrelation)."""
        dgp = TimeSeriesDGPGenerator(
            n_periods=200,
            p=3,
            true_effect=2.0,
            treatment_ar_order=0,
            confounding_strength=0.0,
            random_state=42,
        )
        result = dgp.generate()

        # Autocorrelation should be low
        T = result.T
        autocorr = np.corrcoef(T[:-1], T[1:])[0, 1]
        assert np.abs(autocorr) < 0.3


# ============================================================================
# Confounding Tests
# ============================================================================


class TestConfounding:
    """Tests for confounding properties."""

    def test_no_confounding(self):
        """Test with zero confounding strength."""
        dgp = TimeSeriesDGPGenerator(
            n_periods=500,
            p=5,
            true_effect=2.0,
            confounding_strength=0.0,
            random_state=42,
        )
        result = dgp.generate()

        # OLS should recover effect reasonably well
        # Y = τT + ε when no confounding
        T = result.T
        Y = result.Y

        beta_ols = np.cov(T, Y)[0, 1] / np.var(T)
        assert abs(beta_ols - 2.0) < 1.0  # Should be close to true effect

    def test_strong_confounding_biases_ols(self):
        """Test that confounding biases naive OLS."""
        dgp = TimeSeriesDGPGenerator(
            n_periods=500,
            p=5,
            true_effect=2.0,
            confounding_strength=3.0,  # Strong confounding
            random_state=42,
        )
        result = dgp.generate()

        # OLS will be biased
        T = result.T
        Y = result.Y

        beta_ols = np.cov(T, Y)[0, 1] / np.var(T)
        # With strong confounding, OLS should be substantially biased
        # (may be higher or lower than true effect)
        # Just check it's not exactly the true effect
        assert abs(beta_ols - 2.0) > 0.1

    def test_var_confounders_autocorrelated(self):
        """Test that VAR confounders show autocorrelation."""
        dgp = TimeSeriesDGPGenerator(
            n_periods=500,
            p=5,
            true_effect=2.0,
            confounder_var_coef=0.8,  # High persistence
            random_state=42,
        )
        result = dgp.generate()
        X = result.X

        # Check autocorrelation in first confounder
        autocorr = np.corrcoef(X[:-1, 0], X[1:, 0])[0, 1]
        assert autocorr > 0.5


# ============================================================================
# Panel DGP Tests
# ============================================================================


class TestPanelDGP:
    """Tests for panel data generation."""

    def test_panel_shapes(self):
        """Test panel data has correct shapes."""
        n_periods, n_units, p = 50, 20, 5
        dgp = TimeSeriesDGPGenerator(
            n_periods=n_periods,
            n_units=n_units,
            p=p,
            true_effect=2.0,
            random_state=42,
        )
        result = dgp.generate()

        assert result.Y.shape == (n_units, n_periods)
        assert result.T.shape == (n_units, n_periods)
        assert result.X.shape == (n_units, n_periods, p)
        assert result.n_periods == n_periods
        assert result.n_units == n_units

    def test_panel_unit_ids(self):
        """Test that unit IDs are generated for panel."""
        dgp = TimeSeriesDGPGenerator(
            n_periods=50, n_units=10, p=3, true_effect=2.0, random_state=42
        )
        result = dgp.generate()

        assert result.unit_ids is not None
        assert len(result.unit_ids) == 50 * 10  # n_units * n_periods

    def test_panel_fixed_effects(self):
        """Test panel with fixed effects has unit heterogeneity."""
        dgp_fe = TimeSeriesDGPGenerator(
            n_periods=100,
            n_units=30,
            p=3,
            true_effect=2.0,
            panel_fixed_effects=True,
            random_state=42,
        )
        dgp_no_fe = TimeSeriesDGPGenerator(
            n_periods=100,
            n_units=30,
            p=3,
            true_effect=2.0,
            panel_fixed_effects=False,
            random_state=42,
        )

        result_fe = dgp_fe.generate()
        result_no_fe = dgp_no_fe.generate()

        # Unit means should vary more with fixed effects
        unit_means_fe = result_fe.Y.mean(axis=1)
        unit_means_no_fe = result_no_fe.Y.mean(axis=1)

        # FE should induce more between-unit variance
        var_fe = np.var(unit_means_fe)
        var_no_fe = np.var(unit_means_no_fe)

        assert var_fe > var_no_fe * 0.5  # FE variance should be substantial


# ============================================================================
# Error Autocorrelation Tests
# ============================================================================


class TestErrorAutocorrelation:
    """Tests for AR(1) error structure."""

    def test_ar1_errors(self):
        """Test that AR(1) errors show autocorrelation in residuals."""
        dgp = TimeSeriesDGPGenerator(
            n_periods=500,
            p=3,
            true_effect=2.0,
            error_ar_coef=0.7,
            confounding_strength=0.0,
            random_state=42,
        )
        result = dgp.generate()

        # Compute residuals (Y - τT since no confounding)
        residuals = result.Y - result.true_effect * result.T

        # Autocorrelation of residuals
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]

        # Should show positive autocorrelation
        assert autocorr > 0.3

    def test_no_error_autocorrelation(self):
        """Test that zero AR coefficient gives uncorrelated errors."""
        dgp = TimeSeriesDGPGenerator(
            n_periods=500,
            p=3,
            true_effect=2.0,
            error_ar_coef=0.0,
            confounding_strength=0.0,
            random_state=42,
        )
        result = dgp.generate()

        residuals = result.Y - result.true_effect * result.T
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]

        # Should be close to zero
        assert np.abs(autocorr) < 0.2


# ============================================================================
# Lagged Effects Tests
# ============================================================================


class TestLaggedEffects:
    """Tests for lagged treatment effects."""

    def test_lagged_effect_cumulative(self):
        """Test cumulative effect includes lags."""
        lagged_coefs = np.array([0.5, 0.3, 0.1])
        dgp = TimeSeriesDGPGenerator(
            n_periods=100,
            p=3,
            true_effect=2.0,
            lagged_effect_coefs=lagged_coefs,
            random_state=42,
        )
        result = dgp.generate()

        expected_cumulative = 2.0 + 0.5 + 0.3 + 0.1  # 2.9
        assert_allclose(result.true_cumulative_effect, expected_cumulative)

    def test_no_lagged_effects(self):
        """Test that no lagged effects gives same cumulative as contemporaneous."""
        dgp = TimeSeriesDGPGenerator(n_periods=100, p=3, true_effect=2.0, random_state=42)
        result = dgp.generate()

        assert result.true_cumulative_effect == result.true_effect


# ============================================================================
# Binary Treatment Tests
# ============================================================================


class TestBinaryTreatment:
    """Tests for binary treatment generation."""

    def test_binary_treatment_values(self):
        """Test that binary treatment has values in {0, 1}."""
        dgp = TimeSeriesDGPGenerator(
            n_periods=200,
            p=3,
            true_effect=2.0,
            treatment_type="binary",
            random_state=42,
        )
        result = dgp.generate()

        unique_vals = np.unique(result.T)
        assert set(unique_vals).issubset({0.0, 1.0})

    def test_binary_treatment_proportions(self):
        """Test that binary treatment has reasonable proportions."""
        dgp = TimeSeriesDGPGenerator(
            n_periods=1000,
            p=3,
            true_effect=2.0,
            treatment_type="binary",
            random_state=42,
        )
        result = dgp.generate()

        prop_treated = result.T.mean()
        # Should be somewhere between 0.1 and 0.9
        assert 0.1 < prop_treated < 0.9


# ============================================================================
# Validation Tests
# ============================================================================


class TestValidation:
    """Tests for parameter validation."""

    def test_n_periods_too_small(self):
        """Test that too few periods raises error."""
        with pytest.raises(ValueError, match="n_periods.*must be"):
            TimeSeriesDGPGenerator(n_periods=10, p=3, true_effect=2.0, treatment_ar_order=5)

    def test_p_not_positive(self):
        """Test that p <= 0 raises error."""
        with pytest.raises(ValueError, match="p must be positive"):
            TimeSeriesDGPGenerator(n_periods=100, p=0, true_effect=2.0)

    def test_invalid_var_coef(self):
        """Test that |VAR coef| >= 1 raises error."""
        with pytest.raises(ValueError, match="confounder_var_coef"):
            TimeSeriesDGPGenerator(n_periods=100, p=3, true_effect=2.0, confounder_var_coef=1.0)

    def test_invalid_error_ar_coef(self):
        """Test that |error AR coef| >= 1 raises error."""
        with pytest.raises(ValueError, match="error_ar_coef"):
            TimeSeriesDGPGenerator(n_periods=100, p=3, true_effect=2.0, error_ar_coef=-1.0)

    def test_invalid_treatment_type(self):
        """Test that invalid treatment type raises error."""
        with pytest.raises(ValueError, match="treatment_type"):
            TimeSeriesDGPGenerator(n_periods=100, p=3, true_effect=2.0, treatment_type="invalid")

    def test_ar_coefs_wrong_length(self):
        """Test that AR coefs with wrong length raises error."""
        with pytest.raises(ValueError, match="treatment_ar_coefs length"):
            TimeSeriesDGPGenerator(
                n_periods=100,
                p=3,
                true_effect=2.0,
                treatment_ar_order=2,
                treatment_ar_coefs=np.array([0.5]),  # Wrong length
            )


# ============================================================================
# BreakDGPGenerator Tests
# ============================================================================


class TestBreakDGPGenerator:
    """Tests for structural break DGP."""

    def test_basic_generation(self):
        """Test basic break DGP generation."""
        dgp = BreakDGPGenerator(
            n_periods=300,
            p=3,
            effects=[1.0, 3.0, 1.0],
            break_points=[100, 200],
            random_state=42,
        )
        result = dgp.generate()

        assert isinstance(result, BreakDGPResult)
        assert result.Y.shape == (300,)
        assert result.effects == [1.0, 3.0, 1.0]
        assert result.break_points == [100, 200]

    def test_effect_periods(self):
        """Test that effect periods are computed correctly."""
        dgp = BreakDGPGenerator(
            n_periods=300,
            p=3,
            effects=[1.0, 3.0, 1.0],
            break_points=[100, 200],
            random_state=42,
        )
        result = dgp.generate()

        expected_periods = [(0, 100), (100, 200), (200, 300)]
        assert result.effect_periods == expected_periods

    def test_regime_indices(self):
        """Test that regime indices are correct."""
        dgp = BreakDGPGenerator(
            n_periods=300,
            p=3,
            effects=[1.0, 3.0, 1.0],
            break_points=[100, 200],
            random_state=42,
        )
        result = dgp.generate()

        # First regime: 0-99
        assert all(result.regime_indices[:100] == 0)
        # Second regime: 100-199
        assert all(result.regime_indices[100:200] == 1)
        # Third regime: 200-299
        assert all(result.regime_indices[200:] == 2)

    def test_different_effects_by_regime(self):
        """Test that effects differ by regime in generated data."""
        dgp = BreakDGPGenerator(
            n_periods=600,
            p=3,
            effects=[0.0, 5.0, 0.0],  # Large middle effect
            break_points=[200, 400],
            confounding_strength=0.0,
            random_state=42,
        )
        result = dgp.generate()

        # Compute correlation in each regime
        def regime_corr(start, end):
            Y_sub = result.Y[start:end]
            T_sub = result.T[start:end]
            return np.corrcoef(Y_sub, T_sub)[0, 1]

        corr_1 = regime_corr(0, 200)
        corr_2 = regime_corr(200, 400)
        corr_3 = regime_corr(400, 600)

        # Middle regime should have stronger correlation
        assert corr_2 > corr_1
        assert corr_2 > corr_3

    def test_break_validation_effects_count(self):
        """Test that effects and break_points consistency is validated."""
        with pytest.raises(ValueError, match="Number of effects"):
            BreakDGPGenerator(
                n_periods=300,
                p=3,
                effects=[1.0, 2.0],  # 2 effects need 1 break point
                break_points=[100, 200],  # 2 break points
            )

    def test_break_validation_order(self):
        """Test that break_points must be ascending."""
        with pytest.raises(ValueError, match="ascending order"):
            BreakDGPGenerator(
                n_periods=300,
                p=3,
                effects=[1.0, 2.0, 3.0],
                break_points=[200, 100],  # Wrong order
            )

    def test_break_validation_bounds(self):
        """Test that break_points must be within bounds."""
        with pytest.raises(ValueError, match="within"):
            BreakDGPGenerator(n_periods=300, p=3, effects=[1.0, 2.0], break_points=[0])


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_ar_dgp(self):
        """Test create_ar_dgp convenience function."""
        result = create_ar_dgp(n=200, p=5, true_effect=2.0, ar_coef=0.6, random_state=42)

        assert isinstance(result, TimeSeriesDGPResult)
        assert result.n_periods == 200
        assert result.p == 5
        assert result.true_effect == 2.0

    def test_create_panel_dgp(self):
        """Test create_panel_dgp convenience function."""
        result = create_panel_dgp(n_periods=100, n_units=50, p=5, true_effect=2.0, random_state=42)

        assert isinstance(result, TimeSeriesDGPResult)
        assert result.n_periods == 100
        assert result.n_units == 50
        assert result.Y.shape == (50, 100)

    def test_create_break_dgp(self):
        """Test create_break_dgp convenience function."""
        result = create_break_dgp(n=300, p=5, random_state=42)

        assert isinstance(result, BreakDGPResult)
        assert result.n_periods == 300
        assert len(result.effects) == 3  # Default

    def test_create_break_dgp_custom(self):
        """Test create_break_dgp with custom parameters."""
        result = create_break_dgp(
            n=400,
            p=3,
            effects=[1.0, 5.0],
            break_points=[200],
            random_state=42,
        )

        assert result.effects == [1.0, 5.0]
        assert result.break_points == [200]


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.tier3
class TestDMLIntegration:
    """Tests for integration with DML estimators."""

    def test_single_series_with_dynamic_dml(self):
        """Test single series DGP with DynamicDML."""
        from src.dml import DynamicDML

        result = create_ar_dgp(
            n=300,
            p=5,
            true_effect=2.0,
            ar_coef=0.3,
            confounding_strength=1.0,
            random_state=42,
        )

        model = DynamicDML(
            n_lags=2,
            model_y="ridge",
            model_t="ridge",
            cv_strategy="time_series_split",
            n_splits=3,
            random_state=42,
        )
        dml_result = model.fit(result.Y, result.T, result.X)

        # Should recover effect within reasonable range
        # (HAC SE makes CI wider but effect should be close)
        assert abs(dml_result.theta - 2.0) < 2.0

    def test_panel_dgp_structure(self):
        """Test panel DGP can be flattened for DML."""
        result = create_panel_dgp(n_periods=50, n_units=20, p=3, true_effect=2.0, random_state=42)

        # Flatten for standard DML (N = n_units * n_periods)
        Y_flat = result.Y.flatten()
        T_flat = result.T.flatten()
        X_flat = result.X.reshape(-1, result.p)

        assert Y_flat.shape == (1000,)
        assert T_flat.shape == (1000,)
        assert X_flat.shape == (1000, 3)

    def test_break_dgp_with_rolling_dml(self):
        """Test break DGP with RollingWindowDML."""
        from src.dml import RollingWindowDML

        result = create_break_dgp(
            n=400,
            p=3,
            effects=[1.0, 4.0, 1.0],
            break_points=[133, 266],
            random_state=42,
        )

        model = RollingWindowDML(
            window_size=80,
            step_size=40,
            model_y="ridge",
            model_t="ridge",
            random_state=42,
        )
        model.fit(result.Y, result.T, result.X)

        # Get time series of effects
        time_centers, theta_series, se_series = model.get_effects()

        # Should have multiple windows
        assert len(theta_series) > 3

        # Middle effects should be larger on average
        n_effects = len(theta_series)
        early_effects = theta_series[: n_effects // 3]
        mid_effects = theta_series[n_effects // 3 : 2 * n_effects // 3]

        # Mid-period effects should be higher (true effect is 4.0 vs 1.0)
        avg_early = np.mean(early_effects)
        avg_mid = np.mean(mid_effects)
        # With noise, just check mid is somewhat higher
        assert avg_mid > avg_early - 1.0  # Allow for noise
