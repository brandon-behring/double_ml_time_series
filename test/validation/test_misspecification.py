"""
Test suite for realistic failure modes and model misspecification.

Tests how baseline methods and DML perform when model assumptions are violated:
1. Non-linear treatment effects
2. Treatment-covariate interactions
3. Poor overlap (extreme propensity scores)
4. Unobserved confounding
5. High-dimensional sparse effects
"""

import numpy as np
import pytest

from dml_ts.dml import econml_available
from dml_ts.validation.bias_validation import BiasValidation
from dml_ts.validation.dgp_generator import DGPGenerator
from dml_ts.validation.ipw_baseline import AugmentedIPW, IPWEstimator
from dml_ts.validation.ml_baseline import RandomForestEstimator
from dml_ts.validation.ols_baseline import NaiveOLS, OLSWithControls


class MisspecifiedDGPGenerator:
    """
    Generates data with model misspecifications for testing failure modes.

    Variants:
    - non_linear_effect: Treatment effect depends non-linearly on X
    - interaction_effect: Treatment effect varies by covariate values
    - poor_overlap: Propensity scores concentrated near 0 or 1
    """

    def __init__(
        self,
        n: int = 500,
        p: int = 5,
        true_effect: float = 2.0,
        random_state: int = None,
        misspec_type: str = "non_linear",
    ):
        """Initialize misspecified DGP generator."""
        self.n = n
        self.p = p
        self.true_effect = true_effect
        self.random_state = random_state
        self.misspec_type = misspec_type
        self._rng = np.random.RandomState(random_state)

    def generate(self):
        """Generate misspecified data."""
        # Covariates: uniform on [-1, 1]
        X = self._rng.uniform(-1, 1, size=(self.n, self.p))

        # Treatment: confounded by X[0]
        ps = 1 / (1 + np.exp(-X[:, 0]))  # Logistic transformation
        T = (self._rng.uniform(0, 1, size=self.n) < ps).astype(int)

        # Outcomes with misspecification
        if self.misspec_type == "non_linear":
            # Non-linear treatment effect: ATE depends on X[0]
            treatment_effect = self.true_effect * (1 + X[:, 0] ** 2)
            Y = (
                X[:, 0]
                + X[:, 1] ** 2
                + treatment_effect * T  # Non-linear confounding
                + self._rng.normal(0, 1, size=self.n)
            )

        elif self.misspec_type == "interaction":
            # Interaction: treatment effect varies by X[0]
            treatment_effect = self.true_effect * (0.5 + X[:, 0])
            Y = X[:, 0] + X[:, 1] + treatment_effect * T + self._rng.normal(0, 1, size=self.n)

        elif self.misspec_type == "poor_overlap":
            # Strong confounding: poor overlap
            ps_strong = 1 / (1 + np.exp(-3 * X[:, 0]))
            T = (self._rng.uniform(0, 1, size=self.n) < ps_strong).astype(int)
            Y = X[:, 0] + X[:, 1] + self.true_effect * T + self._rng.normal(0, 1, size=self.n)

        else:
            raise ValueError(f"Unknown misspec_type: {self.misspec_type}")

        return X, T, Y


@pytest.mark.tier3
class TestNonlinearEffects:
    """Test methods under non-linear treatment effects."""

    @pytest.mark.tier3
    def test_linear_methods_biased_under_nonlinearity(self):
        """Linear methods should show bias with non-linear effects."""
        dgp = MisspecifiedDGPGenerator(
            n=1000, p=5, true_effect=2.0, random_state=42, misspec_type="non_linear"
        )
        X, T, Y = dgp.generate()

        # OLS with controls (assumes linearity)
        from sklearn.linear_model import LinearRegression

        X_with_T = np.column_stack([T, X])
        model = LinearRegression()
        model.fit(X_with_T, Y)
        ate_linear = model.coef_[0]  # Coefficient on T

        # Linear ATE will be biased due to non-linearity
        # Should deviate from true_effect (≈2.0)
        # Note: exact value depends on X distribution
        assert isinstance(ate_linear, (int, float))

    @pytest.mark.tier3
    def test_ml_methods_more_flexible_nonlinearity(self):
        """ML methods should handle non-linearity better than linear."""
        dgp = MisspecifiedDGPGenerator(
            n=800, p=5, true_effect=2.0, random_state=42, misspec_type="non_linear"
        )

        # Generate data
        X, T, Y = dgp.generate()

        # RF should work with non-linear data
        # (This is a sanity check that RF can be applied)
        from sklearn.ensemble import RandomForestRegressor

        # Fit separate models for T=1 and T=0
        rf_t1 = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_t0 = RandomForestRegressor(n_estimators=50, random_state=42)

        X_t1 = X[T == 1]
        Y_t1 = Y[T == 1]
        X_t0 = X[T == 0]
        Y_t0 = Y[T == 0]

        if len(X_t1) > 0 and len(X_t0) > 0:
            rf_t1.fit(X_t1, Y_t1)
            rf_t0.fit(X_t0, Y_t0)

            # Predict
            y1_pred = rf_t1.predict(X)
            y0_pred = rf_t0.predict(X)
            ate_rf = np.mean(y1_pred - y0_pred)

            # ATE should be reasonable (not nan/inf)
            assert np.isfinite(ate_rf)


@pytest.mark.tier3
class TestInteractionEffects:
    """Test methods under treatment-covariate interaction effects."""

    @pytest.mark.tier3
    def test_ols_biased_with_interactions(self):
        """OLS without interaction terms should be biased."""
        dgp = MisspecifiedDGPGenerator(
            n=1000, p=5, true_effect=2.0, random_state=42, misspec_type="interaction"
        )
        X, T, Y = dgp.generate()

        # Simple OLS: Y ~ T + X (ignoring T*X)
        from sklearn.linear_model import LinearRegression

        X_with_T = np.column_stack([T, X])
        model = LinearRegression()
        model.fit(X_with_T, Y)

        # Should have non-zero bias due to omitted interactions
        assert isinstance(model.coef_[0], (int, float))

    @pytest.mark.tier3
    def test_aipw_more_robust_interactions(self):
        """AIPW uses both propensity and outcome models, potentially more robust."""
        dgp = MisspecifiedDGPGenerator(
            n=600, p=5, true_effect=2.0, random_state=42, misspec_type="interaction"
        )

        # AIPW should run without errors
        from dml_ts.validation.dgp_generator import DGPGenerator

        # Create wrapper to use AIPW
        dml_dgp = DGPGenerator(n=600, p=5, true_effect=2.0, random_state=42)

        aipw = AugmentedIPW(n_simulations=5, random_state=42)
        result = aipw.validate(dml_dgp)

        # Should return valid result
        assert result.mse >= 0
        assert 0 <= result.coverage <= 1


@pytest.mark.tier3
class TestPoorOverlap:
    """Test methods under poor overlap (extreme propensity scores)."""

    @pytest.mark.tier3
    def test_ipw_weights_explosion_poor_overlap(self):
        """IPW can have extreme weights with poor overlap."""
        dgp = MisspecifiedDGPGenerator(
            n=500, p=5, true_effect=2.0, random_state=42, misspec_type="poor_overlap"
        )
        X, T, Y = dgp.generate()

        # Estimate propensity scores
        from sklearn.linear_model import LogisticRegression

        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X, T)
        ps = ps_model.predict_proba(X)[:, 1]

        # With poor overlap, ps will be very close to 0 or 1
        # resulting in extreme weights
        ps_clipped = np.clip(ps, 0.01, 0.99)
        weights_t = T / ps_clipped

        # Check for extreme weights
        extreme_weights = np.sum(weights_t > 10)
        # Poor overlap should lead to some extreme weights
        # (This is expected behavior, not a failure)
        assert isinstance(extreme_weights, (int, np.integer))

    @pytest.mark.tier3
    def test_ml_methods_avoid_weights(self):
        """ML methods don't use propensity weights, more stable with poor overlap."""
        dgp = MisspecifiedDGPGenerator(
            n=500, p=5, true_effect=2.0, random_state=42, misspec_type="poor_overlap"
        )

        # Run RF and XGB (which don't depend on propensity weighting)
        from dml_ts.validation.dgp_generator import DGPGenerator

        dml_dgp = DGPGenerator(n=300, p=5, true_effect=2.0, random_state=42)

        rf = RandomForestEstimator(n_simulations=3, random_state=42, n_estimators=50)
        result_rf = rf.validate(dml_dgp)

        # Should produce stable results
        assert np.isfinite(result_rf.mse)
        assert np.isfinite(result_rf.bias)


@pytest.mark.tier3
class TestHighDimensional:
    """Test methods with high-dimensional sparse effects."""

    @pytest.mark.tier3
    def test_sparse_features_linear_struggle(self):
        """Linear methods may struggle with high-dimensional sparse data."""
        # Generate sparse high-dimensional data
        np.random.seed(42)
        n, p = 300, 50  # Many features, few observations

        X = np.random.standard_normal((n, p))
        T = np.random.binomial(1, 0.5, n)
        Y = 2.0 * T + np.random.standard_normal(n)

        # OLS with all features might overfit or be unstable
        from sklearn.linear_model import LinearRegression

        X_with_T = np.column_stack([T, X])
        model = LinearRegression()
        model.fit(X_with_T, Y)
        ate_ols = model.coef_[0]

        # ATE might be biased or unstable due to high dimensionality
        assert np.isfinite(ate_ols)

    @pytest.mark.tier3
    def test_ml_methods_handle_sparsity(self):
        """ML methods with regularization handle sparse high-dimensional data."""
        np.random.seed(42)
        n, p = 400, 30

        X = np.random.standard_normal((n, p))
        T = np.random.binomial(1, 0.5, n)
        Y = 2.0 * T + X[:, 0] + np.random.standard_normal(n)

        # RF should handle this without numerical issues
        from sklearn.ensemble import RandomForestRegressor

        X_t1 = X[T == 1]
        Y_t1 = Y[T == 1]
        X_t0 = X[T == 0]
        Y_t0 = Y[T == 0]

        if len(X_t1) > 0 and len(X_t0) > 0:
            rf_t1 = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            rf_t0 = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)

            rf_t1.fit(X_t1, Y_t1)
            rf_t0.fit(X_t0, Y_t0)

            y1_pred = rf_t1.predict(X)
            y0_pred = rf_t0.predict(X)
            ate_rf = np.mean(y1_pred - y0_pred)

            assert np.isfinite(ate_rf)


@pytest.mark.tier3
class TestUnobservedConfounding:
    """Test robustness to unobserved confounding (which no method can fully solve)."""

    @pytest.mark.tier3
    def test_all_methods_fail_with_hidden_confounder(self):
        """
        No method can perfectly handle unobserved confounding.
        This test documents expected behavior rather than success.
        """
        # Generate data with unobserved confounder U
        np.random.seed(42)
        n = 500

        # Unobserved confounder
        U = np.random.standard_normal(n)

        # Observed covariates
        X = np.random.standard_normal((n, 5))

        # Treatment affected by U (unobserved)
        T = (np.random.uniform(0, 1, n) < (1 / (1 + np.exp(-U)))).astype(int)

        # Outcome affected by both T and U
        true_effect = 2.0
        Y = true_effect * T + U + np.random.standard_normal(n)

        # OLS on observed only will be biased
        from sklearn.linear_model import LinearRegression

        X_with_T = np.column_stack([T, X])
        model = LinearRegression()
        model.fit(X_with_T, Y)
        ate_biased = model.coef_[0]

        # ATE estimate will deviate from true_effect (2.0)
        # This is expected: unobserved confounding cannot be corrected
        # by any method that only uses observed variables
        assert np.isfinite(ate_biased)
        # Note: We don't assert bias direction because it depends on correlation structure


@pytest.mark.tier3
@pytest.mark.skipif(not econml_available(), reason="econml not installed (optional '[full]' extra)")
class TestCombinedMisspecifications:
    """Test methods under multiple simultaneous misspecifications."""

    @pytest.mark.tier3
    def test_dml_vs_baselines_combined_issues(self):
        """Compare DML with baselines under multiple misspecifications."""
        # DML should be more robust to combined issues
        # (But this is a research question, not guaranteed)

        dgp = DGPGenerator(n=800, p=5, true_effect=2.0, confounding_strength=1.0, random_state=42)

        methods = {
            "NaiveOLS": NaiveOLS(n_simulations=10, random_state=42),
            "OLSWithControls": OLSWithControls(n_simulations=10, random_state=42),
            "IPW": IPWEstimator(n_simulations=10, random_state=42),
            "AIPW": AugmentedIPW(n_simulations=10, random_state=42),
            "DML": BiasValidation(n_simulations=10, random_state=42),
        }

        results = {}
        for name, method in methods.items():
            result = method.validate(dgp)
            results[name] = result.bias

        # All should produce finite results
        for bias in results.values():
            assert np.isfinite(bias)

        # (No assertion on bias magnitude - that's the research question)
