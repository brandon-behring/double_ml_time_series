"""Tests for TemporalPLRDML time series implementation.

Tests cover:
1. TemporalPLRDML core functionality
2. RollingWindowDML for time-varying effects
3. PanelDML with fixed effects
4. Integration with HAC and cross-fitting components
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats as scipy_stats
from sklearn.dummy import DummyRegressor

from dml_ts.dml.cross_fitting import TimeSeriesCrossValidator
from dml_ts.dml.hac import HACEstimator, newey_west_se
from dml_ts.dml.temporal_plr_dml import (
    TemporalPLRDML,
    TemporalPLRDMLResult,
    PanelDML,
    RollingWindowDML,
    _compute_r2,
    _create_lagged_features,
    _cross_fit_nuisance_time_series,
    _get_nuisance_model,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_time_series():
    """Generate simple time series data with known treatment effect."""
    np.random.seed(42)
    n = 200
    time = np.arange(n)

    # Simple confounders
    X = np.column_stack([np.random.randn(n), np.sin(time / 20)])

    # Treatment: correlated with X
    T = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n)

    # Outcome: true effect = 2.0
    true_theta = 2.0
    Y = true_theta * T + X[:, 0] + X[:, 1] ** 2 + np.random.randn(n)

    return Y, T, X, time, true_theta


@pytest.fixture
def autocorrelated_time_series():
    """Generate time series with autocorrelated treatment."""
    np.random.seed(123)
    n = 300
    time = np.arange(n)

    X = np.random.randn(n, 2)

    # Autocorrelated treatment
    T = np.zeros(n)
    T[0] = np.random.randn()
    for t in range(1, n):
        T[t] = 0.5 * T[t - 1] + 0.3 * X[t, 0] + np.random.randn()

    # Outcome
    true_theta = 1.5
    Y = true_theta * T + X[:, 0] ** 2 + np.random.randn(n)

    return Y, T, X, time, true_theta


@pytest.fixture
def panel_data():
    """Generate panel data with individual fixed effects."""
    np.random.seed(456)
    n_individuals = 50
    n_periods = 20
    n_total = n_individuals * n_periods

    # Create IDs
    individual_id = np.repeat(np.arange(n_individuals), n_periods)
    time_id = np.tile(np.arange(n_periods), n_individuals)

    # Individual fixed effects
    alpha_i = np.random.randn(n_individuals)
    alpha_expanded = alpha_i[individual_id]

    # Covariates
    X = np.random.randn(n_total, 2)

    # Treatment
    T = 0.5 * X[:, 0] + alpha_expanded + np.random.randn(n_total)

    # Outcome with true effect = 3.0
    true_theta = 3.0
    Y = true_theta * T + X[:, 0] + alpha_expanded + np.random.randn(n_total)

    return Y, T, X, individual_id, time_id, true_theta


# ============================================================================
# Helper Function Tests
# ============================================================================


@pytest.mark.tier1
class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_nuisance_model_ridge(self):
        """Test ridge model creation."""
        model = _get_nuisance_model("ridge")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_get_nuisance_model_lasso(self):
        """Test lasso model creation."""
        model = _get_nuisance_model("lasso")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_get_nuisance_model_random_forest(self):
        """Test random forest model creation."""
        model = _get_nuisance_model("random_forest", random_state=42)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_get_nuisance_model_gradient_boosting(self):
        """Test gradient boosting model creation."""
        model = _get_nuisance_model("gradient_boosting", random_state=42)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_get_nuisance_model_invalid(self):
        """Test invalid model type raises error."""
        with pytest.raises(ValueError, match="Unknown model type"):
            _get_nuisance_model("invalid_model")  # type: ignore

    def test_compute_r2_perfect(self):
        """Test R² for perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        r2 = _compute_r2(y_true, y_pred)
        assert_allclose(r2, 1.0)

    def test_compute_r2_zero(self):
        """Test R² when predictions are mean."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full_like(y_true, np.mean(y_true))
        r2 = _compute_r2(y_true, y_pred)
        assert_allclose(r2, 0.0, atol=1e-10)

    def test_create_lagged_features_basic(self):
        """Test lagged feature creation."""
        X = np.arange(10).reshape(-1, 1).astype(float)
        T = np.arange(10).astype(float) * 0.1
        n_lags = 2

        X_aug, n_dropped = _create_lagged_features(X, T, n_lags)

        assert n_dropped == 2
        assert X_aug.shape[0] == 8  # 10 - 2 lags
        assert X_aug.shape[1] == 3  # 1 original + 2 lags

    def test_create_lagged_features_values(self):
        """Test lagged feature values are correct."""
        X = np.ones((5, 1))
        T = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        n_lags = 2

        X_aug, _ = _create_lagged_features(X, T, n_lags)

        # First row should be: X[2], T[1], T[0]
        expected_first = [1.0, 2.0, 1.0]
        assert_allclose(X_aug[0], expected_first)

        # Last row should be: X[4], T[3], T[2]
        expected_last = [1.0, 4.0, 3.0]
        assert_allclose(X_aug[-1], expected_last)

    def test_temporal_cross_fit_does_not_backfill_early_rows(self):
        """Early rows without temporal OOF predictions stay missing."""
        n = 30
        X = np.arange(n, dtype=float).reshape(-1, 1)
        Y = np.arange(n, dtype=float)
        T = np.arange(n, dtype=float) * 0.5
        cv = TimeSeriesCrossValidator(n_splits=3)

        Y_hat, T_hat = _cross_fit_nuisance_time_series(
            X=X,
            Y=Y,
            T=T,
            cv=cv,
            outcome_model=DummyRegressor(strategy="mean"),
            treatment_model=DummyRegressor(strategy="mean"),
        )

        first_predicted = min(int(test_idx[0]) for _, test_idx in cv.split(X, Y))
        assert np.isnan(Y_hat[:first_predicted]).all()
        assert np.isnan(T_hat[:first_predicted]).all()
        assert np.isfinite(Y_hat[first_predicted:]).all()
        assert np.isfinite(T_hat[first_predicted:]).all()

    @pytest.mark.tier1
    def test_temporal_cross_fit_trains_only_on_past(self):
        """No fold trains on future rows: max(train) < min(test) per split.

        Regression guard for audit finding F5 (temporal leakage). Complements
        test_temporal_cross_fit_does_not_backfill_early_rows: that test checks the
        *output* (early rows stay NaN); this checks the *input* — every cross-fitted
        prediction came from a model trained only on strictly-earlier indices.
        """
        n = 120
        X = np.random.RandomState(0).randn(n, 2)
        Y = np.random.RandomState(1).randn(n)
        T = np.random.RandomState(2).randn(n)
        cv = TimeSeriesCrossValidator(n_splits=5, gap=2)

        predicted_by_fold = np.full(n, -1, dtype=int)
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, Y)):
            assert (
                train_idx.max() < test_idx.min()
            ), f"Fold {fold}: train max {train_idx.max()} >= test min {test_idx.min()}"
            predicted_by_fold[test_idx] = fold

        Y_hat, T_hat = _cross_fit_nuisance_time_series(
            X=X,
            Y=Y,
            T=T,
            cv=cv,
            outcome_model=DummyRegressor(strategy="mean"),
            treatment_model=DummyRegressor(strategy="mean"),
        )
        finite = np.isfinite(Y_hat)
        assert (predicted_by_fold[finite] >= 0).all()
        assert np.isfinite(T_hat[finite]).all()


# ============================================================================
# TemporalPLRDMLResult Tests
# ============================================================================


@pytest.mark.tier1
class TestTemporalPLRDMLResult:
    """Tests for TemporalPLRDMLResult dataclass."""

    def test_result_repr(self):
        """Test result string representation."""
        result = TemporalPLRDMLResult(
            theta=2.0,
            se=0.1,
            t_stat=20.0,
            ci_lower=1.8,
            ci_upper=2.2,
            p_value=0.001,
            n_samples=100,
            n_periods=97,
            outcome_r2_cv=0.5,
            treatment_r2_cv=0.3,
            hac_bandwidth=5,
            cv_strategy="time_series_split",
            Y_residual=np.zeros(97),
            T_residual=np.zeros(97),
            influence_scores=np.zeros(97),
        )

        repr_str = repr(result)
        assert "θ=2.0000" in repr_str
        assert "SE=0.1000" in repr_str

    def test_result_summary(self):
        """Test result summary output."""
        result = TemporalPLRDMLResult(
            theta=2.0,
            se=0.1,
            t_stat=20.0,
            ci_lower=1.8,
            ci_upper=2.2,
            p_value=0.001,
            n_samples=100,
            n_periods=97,
            outcome_r2_cv=0.5,
            treatment_r2_cv=0.3,
            hac_bandwidth=5,
            cv_strategy="time_series_split",
            Y_residual=np.zeros(97),
            T_residual=np.zeros(97),
            influence_scores=np.zeros(97),
        )

        summary = result.summary()
        assert "Temporal PLR DML Results" in summary
        assert "HAC Standard Error:" in summary
        assert "Bandwidth:" in summary


# ============================================================================
# TemporalPLRDML Core Tests
# ============================================================================


@pytest.mark.tier1
class TestTemporalPLRDMLInit:
    """Tests for TemporalPLRDML initialization."""

    def test_default_init(self):
        """Test default initialization."""
        model = TemporalPLRDML()
        assert model.n_lags == 1
        assert model.model_y == "random_forest"
        assert model.model_t == "random_forest"
        assert model.cv_strategy == "time_series_split"
        assert model.n_splits == 5
        assert not model._is_fitted

    def test_custom_init(self):
        """Test custom initialization."""
        model = TemporalPLRDML(
            n_lags=3,
            model_y="ridge",
            model_t="lasso",
            cv_strategy="blocked_cv",
            n_splits=10,
            gap=5,
            hac_bandwidth=10,
            hac_kernel="parzen",
            random_state=123,
        )
        assert model.n_lags == 3
        assert model.model_y == "ridge"
        assert model.model_t == "lasso"
        assert model.cv_strategy == "blocked_cv"
        assert model.n_splits == 10
        assert model.gap == 5
        assert model.hac_bandwidth == 10
        assert model.hac_kernel == "parzen"
        assert model.random_state == 123


@pytest.mark.tier2
class TestTemporalPLRDMLFit:
    """Tests for TemporalPLRDML fit method."""

    def test_fit_returns_result(self, simple_time_series):
        """Test that fit returns TemporalPLRDMLResult."""
        Y, T, X, time, _ = simple_time_series
        model = TemporalPLRDML(n_lags=1, random_state=42)
        result = model.fit(Y, T, X, time_index=time)

        assert isinstance(result, TemporalPLRDMLResult)
        assert model._is_fitted

    def test_fit_without_time_index(self, simple_time_series):
        """Test fit without explicit time index."""
        Y, T, X, _, _ = simple_time_series
        model = TemporalPLRDML(n_lags=1, random_state=42)
        result = model.fit(Y, T, X)  # No time_index

        assert isinstance(result, TemporalPLRDMLResult)

    def test_fit_estimates_reasonable(self, simple_time_series):
        """Test that estimates are in reasonable range of true value."""
        Y, T, X, time, true_theta = simple_time_series
        model = TemporalPLRDML(n_lags=2, random_state=42)
        result = model.fit(Y, T, X, time_index=time)

        # Theta should be within 50% of true value
        assert abs(result.theta - true_theta) < 0.5 * abs(true_theta)

    def test_fit_hac_se_positive(self, simple_time_series):
        """Test that HAC SE is positive."""
        Y, T, X, time, _ = simple_time_series
        model = TemporalPLRDML(n_lags=1, random_state=42)
        result = model.fit(Y, T, X, time_index=time)

        assert result.se > 0

    def test_fit_with_autocorrelation(self, autocorrelated_time_series):
        """Test fit with autocorrelated treatment."""
        Y, T, X, time, true_theta = autocorrelated_time_series
        model = TemporalPLRDML(
            n_lags=3,
            hac_kernel="bartlett",
            random_state=42,
        )
        result = model.fit(Y, T, X, time_index=time)

        # Should still get reasonable estimate
        assert abs(result.theta - true_theta) < 1.0

    def test_fit_stores_residuals(self, simple_time_series):
        """Test that residuals are stored after fit."""
        Y, T, X, time, _ = simple_time_series
        model = TemporalPLRDML(n_lags=1, random_state=42)
        model.fit(Y, T, X, time_index=time)

        assert model._Y_residual is not None
        assert model._T_residual is not None
        assert len(model._Y_residual) == len(Y) - model.n_lags - model._result.dropped_initial_rows

    def test_fit_with_different_cv_strategies(self, simple_time_series):
        """Test fit with different CV strategies."""
        Y, T, X, time, _ = simple_time_series

        for cv_strategy in ["time_series_split", "blocked_cv", "purged_cv"]:
            model = TemporalPLRDML(
                n_lags=1,
                cv_strategy=cv_strategy,  # type: ignore
                random_state=42,
            )
            result = model.fit(Y, T, X, time_index=time)
            assert result.cv_strategy == cv_strategy

    def test_fit_with_different_models(self, simple_time_series):
        """Test fit with different ML models."""
        Y, T, X, time, _ = simple_time_series

        for model_type in ["ridge", "lasso", "random_forest", "gradient_boosting"]:
            model = TemporalPLRDML(
                n_lags=1,
                model_y=model_type,  # type: ignore
                model_t=model_type,  # type: ignore
                random_state=42,
            )
            result = model.fit(Y, T, X, time_index=time)
            assert isinstance(result.theta, float)

    def test_fit_input_validation(self):
        """Test input validation in fit."""
        model = TemporalPLRDML()

        Y = np.array([1.0, 2.0, 3.0])
        T = np.array([1.0, 2.0])  # Wrong length
        X = np.array([[1.0], [2.0], [3.0]])

        with pytest.raises(ValueError, match="must match Y length"):
            model.fit(Y, T, X)

    def test_fit_x_validation(self):
        """Test X shape validation in fit."""
        model = TemporalPLRDML()

        Y = np.array([1.0, 2.0, 3.0])
        T = np.array([1.0, 2.0, 3.0])
        X = np.array([[1.0], [2.0]])  # Wrong rows

        with pytest.raises(ValueError, match="X rows"):
            model.fit(Y, T, X)

    def test_fit_no_treatment_variation(self):
        """Test error when treatment has no variation after partialing out."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 1)
        # Create T that is EXACTLY a linear function of X
        # Ridge with high regularization won't perfectly predict, so use
        # a constant T which will have zero residual variance
        T = np.zeros(n)  # Constant treatment = no variation
        Y = 2.0 * T + np.random.randn(n)

        model = TemporalPLRDML(n_lags=0, model_y="ridge", model_t="ridge", random_state=42)

        with pytest.raises(ValueError, match="no variation"):
            model.fit(Y, T, X)


@pytest.mark.tier2
class TestTemporalPLRDMLEffect:
    """Tests for TemporalPLRDML effect methods."""

    def test_effect_before_fit(self):
        """Test effect raises error before fit."""
        model = TemporalPLRDML()
        with pytest.raises(ValueError, match="must be fitted"):
            model.effect()

    def test_effect_after_fit(self, simple_time_series):
        """Test effect returns correct shape."""
        Y, T, X, time, _ = simple_time_series
        model = TemporalPLRDML(n_lags=1, random_state=42)
        model.fit(Y, T, X, time_index=time)

        effects = model.effect()
        assert len(effects) == model._result.n_periods

    def test_effect_with_X(self, simple_time_series):
        """Test effect with new X values."""
        Y, T, X, time, _ = simple_time_series
        model = TemporalPLRDML(n_lags=1, random_state=42)
        model.fit(Y, T, X, time_index=time)

        X_new = np.random.randn(10, X.shape[1])
        effects = model.effect(X=X_new)
        assert len(effects) == 10

    def test_effect_scaling(self, simple_time_series):
        """Test effect scaling with T0 and T1."""
        Y, T, X, time, _ = simple_time_series
        model = TemporalPLRDML(n_lags=1, random_state=42)
        result = model.fit(Y, T, X, time_index=time)

        effects_default = model.effect(T0=0, T1=1)
        effects_scaled = model.effect(T0=0, T1=2)

        assert_allclose(effects_scaled, 2 * effects_default)

    def test_effect_interval_before_fit(self):
        """Test effect_interval raises error before fit."""
        model = TemporalPLRDML()
        with pytest.raises(ValueError, match="must be fitted"):
            model.effect_interval()

    def test_effect_interval_after_fit(self, simple_time_series):
        """Test effect_interval returns correct shape."""
        Y, T, X, time, _ = simple_time_series
        model = TemporalPLRDML(n_lags=1, random_state=42)
        model.fit(Y, T, X, time_index=time)

        lower, upper = model.effect_interval()
        assert len(lower) == model._result.n_periods
        assert len(upper) == model._result.n_periods

    def test_effect_interval_ordering(self, simple_time_series):
        """Test that lower < upper for confidence intervals."""
        Y, T, X, time, _ = simple_time_series
        model = TemporalPLRDML(n_lags=1, random_state=42)
        model.fit(Y, T, X, time_index=time)

        lower, upper = model.effect_interval(alpha=0.05)
        assert np.all(lower < upper)


# ============================================================================
# RollingWindowDML Tests
# ============================================================================


@pytest.mark.tier2
class TestRollingWindowDML:
    """Tests for RollingWindowDML."""

    def test_init(self):
        """Test initialization."""
        model = RollingWindowDML(
            window_size=50,
            step_size=5,
            model_y="ridge",
            random_state=42,
        )
        assert model.window_size == 50
        assert model.step_size == 5
        assert not model._is_fitted

    @pytest.mark.tier2
    def test_fit_returns_self(self, autocorrelated_time_series):
        """Test fit returns self."""
        Y, T, X, time, _ = autocorrelated_time_series
        model = RollingWindowDML(
            window_size=100,
            step_size=20,
            model_y="ridge",
            model_t="ridge",
            random_state=42,
        )
        result = model.fit(Y, T, X, time_index=time)

        assert result is model
        assert model._is_fitted

    @pytest.mark.tier2
    def test_get_effects_before_fit(self):
        """Test get_effects raises error before fit."""
        model = RollingWindowDML()
        with pytest.raises(ValueError, match="fitted first"):
            model.get_effects()

    @pytest.mark.tier2
    def test_get_effects_after_fit(self, autocorrelated_time_series):
        """Test get_effects returns arrays after fit."""
        Y, T, X, time, _ = autocorrelated_time_series
        model = RollingWindowDML(
            window_size=100,
            step_size=20,
            model_y="ridge",
            model_t="ridge",
            random_state=42,
        )
        model.fit(Y, T, X, time_index=time)

        time_centers, theta_series, se_series = model.get_effects()

        assert len(time_centers) > 0
        assert len(theta_series) == len(time_centers)
        assert len(se_series) == len(time_centers)

    @pytest.mark.tier2
    def test_window_se_matches_standalone_fit(self, autocorrelated_time_series):
        """Window SEs delegate to TemporalPLRDML without divergent rescaling.

        Consistency test between call paths: a rolling window's SE must equal
        the SE of a standalone TemporalPLRDML fit on the same slice. Guards
        against a future "fix" applied only inside RollingWindowDML's loop.
        """
        Y, T, X, time, _ = autocorrelated_time_series
        model = RollingWindowDML(
            window_size=100,
            step_size=20,
            model_y="ridge",
            model_t="ridge",
            random_state=42,
        )
        model.fit(Y, T, X, time_index=time)
        _, _, se_series = model.get_effects()

        # n=300, window 100, step 20 -> centers range(50, 250, 20) = 10
        # windows. A silent window skip would misalign every later center.
        assert len(se_series) == 10
        assert np.all(se_series > 0)

        # First window covers [0, 100); replicate its exact internal fit.
        standalone = TemporalPLRDML(
            n_lags=0,
            model_y="ridge",
            model_t="ridge",
            cv_strategy="time_series_split",
            n_splits=3,
            random_state=42,
        )
        first_window = standalone.fit(Y[:100], T[:100], X[:100])

        assert_allclose(se_series[0], first_window.se, rtol=1e-10)


# ============================================================================
# PanelDML Tests
# ============================================================================


@pytest.mark.tier2
class TestPanelDML:
    """Tests for PanelDML."""

    def test_init(self):
        """Test initialization."""
        model = PanelDML(
            fixed_effects="twoway",
            cluster_se=True,
            random_state=42,
        )
        assert model.fixed_effects == "twoway"
        assert model.cluster_se is True

    @pytest.mark.tier2
    def test_fit_individual_fe(self, panel_data):
        """Test fit with individual fixed effects."""
        Y, T, X, individual_id, time_id, true_theta = panel_data
        model = PanelDML(
            fixed_effects="individual",
            cluster_se=False,
            model_y="ridge",
            model_t="ridge",
            random_state=42,
        )
        result = model.fit(Y, T, X, individual_id, time_id)

        assert isinstance(result, TemporalPLRDMLResult)
        # With proper FE, should get close to true effect
        assert abs(result.theta - true_theta) < 1.5

    @pytest.mark.tier2
    def test_fit_time_fe(self, panel_data):
        """Test fit with time fixed effects."""
        Y, T, X, individual_id, time_id, _ = panel_data
        model = PanelDML(
            fixed_effects="time",
            model_y="ridge",
            model_t="ridge",
            random_state=42,
        )
        result = model.fit(Y, T, X, individual_id, time_id)

        assert isinstance(result, TemporalPLRDMLResult)

    @pytest.mark.tier2
    def test_fit_twoway_fe(self, panel_data):
        """Test fit with two-way fixed effects."""
        Y, T, X, individual_id, time_id, _ = panel_data
        model = PanelDML(
            fixed_effects="twoway",
            model_y="ridge",
            model_t="ridge",
            random_state=42,
        )
        result = model.fit(Y, T, X, individual_id, time_id)

        assert isinstance(result, TemporalPLRDMLResult)

    @pytest.mark.tier2
    def test_cluster_se(self, panel_data):
        """Test cluster-robust standard errors."""
        Y, T, X, individual_id, time_id, _ = panel_data

        model_no_cluster = PanelDML(
            fixed_effects="individual",
            cluster_se=False,
            model_y="ridge",
            model_t="ridge",
            random_state=42,
        )
        result_no_cluster = model_no_cluster.fit(Y, T, X, individual_id, time_id)

        model_cluster = PanelDML(
            fixed_effects="individual",
            cluster_se=True,
            model_y="ridge",
            model_t="ridge",
            random_state=42,
        )
        result_cluster = model_cluster.fit(Y, T, X, individual_id, time_id)

        # Cluster SE should typically be larger (accounting for within-cluster correlation)
        # But not always - just check both are positive
        assert result_no_cluster.se > 0
        assert result_cluster.se > 0


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.tier2
class TestTemporalPLRDMLIntegration:
    """Integration tests for TemporalPLRDML with other components."""

    def test_integration_with_hac(self, autocorrelated_time_series):
        """Test integration with HAC covariance estimation."""
        Y, T, X, time, _ = autocorrelated_time_series

        # Different HAC kernels should all work
        for kernel in ["bartlett", "parzen", "quadratic_spectral"]:
            model = TemporalPLRDML(
                n_lags=2,
                hac_kernel=kernel,  # type: ignore
                random_state=42,
            )
            result = model.fit(Y, T, X, time_index=time)
            assert result.se > 0
            assert result.hac_bandwidth > 0

    def test_hac_se_consistency_with_hac_module(self, simple_time_series):
        """SE equals the HAC variance-of-the-mean of the influence scores.

        Regression test for issue #7: the estimator previously divided the
        HAC variance (already long-run variance / n) by n again, understating
        SEs by a factor of sqrt(n).
        """
        Y, T, X, time, _ = simple_time_series
        model = TemporalPLRDML(n_lags=1, random_state=42)
        result = model.fit(Y, T, X, time_index=time)

        hac = HACEstimator(kernel=model.hac_kernel, bandwidth=result.hac_bandwidth)
        hac.fit(result.influence_scores)

        assert_allclose(result.se, np.sqrt(hac.get_variance()), rtol=1e-12)
        assert_allclose(result.se, hac.get_se(), rtol=1e-12)
        assert_allclose(
            result.se,
            newey_west_se(
                result.influence_scores,
                bandwidth=result.hac_bandwidth,
                kernel=model.hac_kernel,
            ),
            rtol=1e-12,
        )

        # Inference statistics must all derive from the same se (guards the
        # block at temporal_plr_dml.py:569-575 through future refactors).
        z_crit = scipy_stats.norm.ppf(0.975)
        assert_allclose(result.t_stat, result.theta / result.se, rtol=1e-12)
        assert_allclose(result.ci_upper - result.ci_lower, 2 * z_crit * result.se, rtol=1e-12)
        assert_allclose(
            result.p_value,
            2 * (1 - scipy_stats.norm.cdf(abs(result.t_stat))),
            rtol=1e-9,
            atol=1e-12,
        )

        # Two-sided magnitude guard: the HAC SE must be the same order as the
        # iid SE of the influence scores. Under the pre-fix double division it
        # was sqrt(n_used) (here ~13x) smaller and the lower bound fails hard;
        # the upper bound catches an hac-module-internal inflation (both sides
        # of the equality asserts above would move together).
        n_used = len(result.influence_scores)
        naive_se = np.std(result.influence_scores, ddof=1) / np.sqrt(n_used)
        assert result.se > 0.2 * naive_se
        assert result.se < 5 * naive_se

    def test_integration_with_cross_fitting(self, simple_time_series):
        """Test integration with time series cross-validation."""
        Y, T, X, time, _ = simple_time_series

        model = TemporalPLRDML(
            n_lags=1,
            cv_strategy="time_series_split",
            n_splits=5,
            gap=2,  # Add gap to prevent leakage
            random_state=42,
        )
        result = model.fit(Y, T, X, time_index=time)

        # Cross-validation should produce reasonable R² values
        assert 0 <= result.outcome_r2_cv <= 1
        assert 0 <= result.treatment_r2_cv <= 1

    @pytest.mark.tier3
    def test_confidence_interval_coverage(self):
        """Nominal-0.95 CIs must achieve near-nominal empirical coverage.

        Regression test for issue #7: with SEs understated by sqrt(n) the
        empirical coverage collapses to ~0.10 (theory: 2*Phi(1.96/sqrt(n))-1).
        The previous assertion (coverage_rate >= 0.0) was vacuous and masked
        exactly that.
        """
        np.random.seed(42)
        n_sims = 20  # Limited for speed
        true_theta = 2.0
        n = 300  # Larger sample for better asymptotics
        coverage_count = 0
        bias_list = []

        for _ in range(n_sims):
            # Generate simple i.i.d. DGP (no autocorrelation)
            X = np.random.randn(n, 2)
            T = 0.5 * X[:, 0] + np.random.randn(n)
            Y = true_theta * T + X[:, 0] + np.random.randn(n)

            model = TemporalPLRDML(
                n_lags=0,
                model_y="ridge",
                model_t="ridge",
                random_state=None,
            )
            result = model.fit(Y, T, X)
            bias_list.append(result.theta - true_theta)

            if result.ci_lower <= true_theta <= result.ci_upper:
                coverage_count += 1

        coverage_rate = coverage_count / n_sims
        mean_bias = np.mean(bias_list)
        rmse = np.sqrt(np.mean(np.array(bias_list) ** 2))

        # Check that estimates are reasonable (low bias, reasonable RMSE)
        assert abs(mean_bias) < 0.3, f"Mean bias {mean_bias:.3f} too large"
        assert rmse < 0.5, f"RMSE {rmse:.3f} too large"

        # 16/20 at nominal 0.95: loose enough for Monte Carlo noise with
        # 20 simulations, tight enough that sqrt(n)-understated SEs
        # (empirical coverage ~0.10) fail hard.
        assert (
            coverage_rate >= 0.80
        ), f"Coverage {coverage_rate:.2f} far below nominal 0.95 — HAC SEs understated"


# ============================================================================
# Edge Case Tests
# ============================================================================


@pytest.mark.tier2
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_small_sample(self):
        """Test with small sample size."""
        np.random.seed(42)
        n = 50
        X = np.random.randn(n, 2)
        T = 0.5 * X[:, 0] + np.random.randn(n)
        Y = 2.0 * T + X[:, 0] + np.random.randn(n)

        model = TemporalPLRDML(
            n_lags=1,
            n_splits=3,  # Fewer splits for small sample
            model_y="ridge",
            model_t="ridge",
            random_state=42,
        )
        result = model.fit(Y, T, X)

        assert isinstance(result.theta, float)
        assert result.se > 0

    def test_1d_X(self):
        """Test with 1D covariate array."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n)  # 1D
        T = 0.5 * X + np.random.randn(n)
        Y = 2.0 * T + X + np.random.randn(n)

        model = TemporalPLRDML(n_lags=1, random_state=42)
        result = model.fit(Y, T, X)

        assert isinstance(result.theta, float)

    def test_zero_lags(self):
        """Test with no lags."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        T = 0.5 * X[:, 0] + np.random.randn(n)
        Y = 2.0 * T + X[:, 0] + np.random.randn(n)

        model = TemporalPLRDML(n_lags=0, random_state=42)
        result = model.fit(Y, T, X)

        # No lag rows should be dropped, but early temporal-CV rows are excluded.
        assert result.lagged_rows_dropped == 0
        assert result.n_periods + result.dropped_initial_rows == n

    def test_many_lags(self):
        """Test with many lags."""
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 2)
        T = 0.5 * X[:, 0] + np.random.randn(n)
        Y = 2.0 * T + X[:, 0] + np.random.randn(n)

        model = TemporalPLRDML(n_lags=10, random_state=42)
        result = model.fit(Y, T, X)

        # 10 lag rows are dropped before temporal-CV filtering.
        assert result.lagged_rows_dropped == 10
        assert result.n_periods + result.dropped_initial_rows == n - 10

    def test_list_inputs(self):
        """Test with list inputs instead of arrays."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2).tolist()
        T = (0.5 * np.array(X)[:, 0] + np.random.randn(n)).tolist()
        Y = (2.0 * np.array(T) + np.array(X)[:, 0] + np.random.randn(n)).tolist()

        model = TemporalPLRDML(n_lags=1, random_state=42)
        result = model.fit(Y, T, X)

        assert isinstance(result.theta, float)


# ============================================================================
# Benchmark Tests (slow)
# ============================================================================


@pytest.mark.tier3
class TestBenchmarks:
    """Benchmark tests for larger datasets."""

    def test_large_sample(self):
        """Test with large sample size."""
        np.random.seed(42)
        n = 2000
        X = np.random.randn(n, 5)
        T = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n)
        Y = 2.0 * T + X[:, 0] ** 2 + np.random.randn(n)
        time = np.arange(n)

        model = TemporalPLRDML(
            n_lags=5,
            model_y="random_forest",
            model_t="random_forest",
            random_state=42,
        )
        result = model.fit(Y, T, X, time_index=time)

        # With large sample, should get close to true effect
        assert abs(result.theta - 2.0) < 0.3

    def test_many_features(self):
        """Test with many features."""
        np.random.seed(42)
        n = 500
        p = 20
        X = np.random.randn(n, p)
        T = 0.5 * X[:, 0] + np.random.randn(n)
        Y = 2.0 * T + X[:, 0] + X[:, 1] ** 2 + np.random.randn(n)

        model = TemporalPLRDML(
            n_lags=2,
            model_y="random_forest",
            model_t="random_forest",
            random_state=42,
        )
        result = model.fit(Y, T, X)

        assert isinstance(result.theta, float)
        assert result.se > 0
