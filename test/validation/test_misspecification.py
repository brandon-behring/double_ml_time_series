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
from dml_ts.validation.baseline_comparison import BaselineComparison
from dml_ts.validation.bootstrap_config import BootstrapConfig
from dml_ts.validation.dgp_generator import DGPGenerator, DGPResult
from dml_ts.validation.ipw_baseline import AugmentedIPW
from dml_ts.validation.ml_baseline import RandomForestEstimator


class MisspecifiedDGPGenerator(DGPGenerator):
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
        assignment_scale: float = 3.0,
    ):
        """Initialize misspecified DGP generator."""
        # 'true_effect' in the fixture constructor is the coefficient multiplying
        # the heterogeneous response. Validators compare with the population ATE.
        multiplier = {"non_linear": 4 / 3, "interaction": 0.5, "poor_overlap": 1}[misspec_type]
        super().__init__(n, p, multiplier * true_effect, random_state=random_state)
        self.response_scale = true_effect
        self.misspec_type = misspec_type
        self.assignment_scale = assignment_scale

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
            treatment_effect = self.response_scale * (1 + X[:, 0] ** 2)
            Y = (
                X[:, 0]
                + X[:, 1] ** 2
                + treatment_effect * T  # Non-linear confounding
                + self._rng.normal(0, 1, size=self.n)
            )

        elif self.misspec_type == "interaction":
            # Interaction: treatment effect varies by X[0]
            treatment_effect = self.response_scale * (0.5 + X[:, 0])
            Y = X[:, 0] + X[:, 1] + treatment_effect * T + self._rng.normal(0, 1, size=self.n)

        elif self.misspec_type == "poor_overlap":
            # Strong confounding: poor overlap
            ps_strong = 1 / (1 + np.exp(-self.assignment_scale * X[:, 0]))
            T = (self._rng.uniform(0, 1, size=self.n) < ps_strong).astype(int)
            Y = X[:, 0] + X[:, 1] + self.true_effect * T + self._rng.normal(0, 1, size=self.n)

        else:
            raise ValueError(f"Unknown misspec_type: {self.misspec_type}")

        return DGPResult(Y, T, X, self.true_effect, self.n, self.p)


@pytest.mark.tier3
class TestNonlinearEffects:
    """Test methods under non-linear treatment effects."""

    @pytest.mark.tier3
    def test_nonlinear_effect_projection_matches_population_moments(self):
        from scipy.special import expit
        from sklearn.linear_model import LinearRegression

        # Partial out the population *linear* projection of T on (1,X).
        # For E[Y|T,X] = g(X)+tau(X)*T, its treatment slope weights tau by
        # E[(T-q(X))*T|X]=(1-q(X))*p(X). The unrelated X1² term drops out.
        x, w = np.polynomial.legendre.leggauss(32)
        p = expit(x)
        q = 0.5 + x * ((w @ (x * p)) / (w @ (x * x)))
        measure = w * (1 - q) * p
        target = measure @ (2 * (1 + x * x)) / measure.sum()
        dgp = MisspecifiedDGPGenerator(n=6000, p=5, true_effect=2.0, random_state=42)
        fits = []
        for _ in range(5):
            d = dgp.generate()
            fits.append(LinearRegression().fit(np.column_stack([d.T, d.X]), d.Y).coef_[0])
        assert np.mean(fits) == pytest.approx(target, abs=0.06)
        assert dgp.true_effect == pytest.approx(8 / 3)

    @pytest.mark.tier3
    def test_ml_methods_more_flexible_nonlinearity(self):
        """ML methods should handle non-linearity better than linear."""
        dgp = MisspecifiedDGPGenerator(
            n=800, p=5, true_effect=2.0, random_state=42, misspec_type="non_linear"
        )

        # Generate data
        data = dgp.generate()
        X, T, Y = data.X, data.T, data.Y

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
            assert ate_rf == pytest.approx(dgp.true_effect, abs=0.4)


@pytest.mark.tier3
class TestInteractionEffects:
    """Test methods under treatment-covariate interaction effects."""

    @pytest.mark.tier3
    def test_symmetric_assignment_interaction_recovers_mean_effect(self):
        from sklearn.linear_model import LinearRegression

        # Under the fixture's symmetric logistic assignment, the omitted T*X0
        # interaction has zero slope on T after linear adjustment for X0.
        dgp = MisspecifiedDGPGenerator(
            n=5000, p=5, true_effect=2.0, random_state=42, misspec_type="interaction"
        )
        coefficients = []
        for _ in range(5):
            d = dgp.generate()
            coefficients.append(LinearRegression().fit(np.column_stack([d.T, d.X]), d.Y).coef_[0])
        assert np.mean(coefficients) == pytest.approx(dgp.true_effect, abs=0.06)

    @pytest.mark.tier3
    def test_aipw_more_robust_interactions(self):
        """AIPW uses both propensity and outcome models, potentially more robust."""
        dgp = MisspecifiedDGPGenerator(
            n=600, p=5, true_effect=2.0, random_state=42, misspec_type="interaction"
        )

        aipw = AugmentedIPW(
            n_simulations=8, bootstrap_config=BootstrapConfig.tier2(), random_state=42
        )
        result = aipw.validate(dgp)
        assert result.metadata["dgp_true_effect"] == 1.0
        assert abs(result.bias) < 0.12


@pytest.mark.tier3
class TestPoorOverlap:
    """Test methods under poor overlap (extreme propensity scores)."""

    @pytest.mark.tier3
    def test_weak_overlap_reduces_effective_sample_size(self):
        args = {
            "n": 5000,
            "p": 5,
            "true_effect": 2.0,
            "random_state": 42,
            "misspec_type": "poor_overlap",
        }
        ess, extreme = [], []
        for scale in (1, 3):
            data = MisspecifiedDGPGenerator(**args, assignment_scale=scale).generate()
            ps = 1 / (1 + np.exp(-scale * data.X[:, 0]))
            weights = np.where(data.T == 1, 1 / ps, 1 / (1 - ps))
            ess.append(weights.sum() ** 2 / (weights @ weights))
            extreme.append(np.mean((ps < 0.1) | (ps > 0.9)))
        assert extreme[0] == 0 and extreme[1] > 0.2
        assert ess[1] < 0.75 * ess[0]

    @pytest.mark.tier3
    def test_random_forest_on_poor_overlap_draws(self):
        dgp = MisspecifiedDGPGenerator(
            n=600, p=5, true_effect=2.0, random_state=42, misspec_type="poor_overlap"
        )
        rf = RandomForestEstimator(
            n_simulations=5,
            random_state=42,
            n_estimators=30,
            bootstrap_config=BootstrapConfig.tier2(),
        )
        result = rf.validate(dgp)
        assert result.metadata["dgp_n"] == 600
        assert abs(result.bias) < 0.35


@pytest.mark.tier3
class TestHighDimensional:
    """Test methods with high-dimensional sparse effects."""

    @pytest.mark.tier3
    def test_randomized_effect_with_irrelevant_controls(self):
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
        assert ate_ols == pytest.approx(2.0, abs=0.45)

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

            assert ate_rf == pytest.approx(2.0, abs=0.45)


@pytest.mark.tier3
class TestUnobservedConfounding:
    """Test robustness to unobserved confounding (which no method can fully solve)."""

    @pytest.mark.tier3
    def test_hidden_and_observed_confounder_have_distinct_projection_targets(self):
        from scipy.special import expit

        from dml_ts.dml import double_ml, fwl_estimate

        # Under symmetric U ~ N(0,1), P(T=1)=1/2 and Var(T)=1/4.
        # Compute 2 + 4 E[U expit(U)] with deterministic Gaussian quadrature.
        nodes, weights = np.polynomial.hermite.hermgauss(32)
        u_grid = np.sqrt(2) * nodes
        hidden_target = 2 + 4 * (weights @ (u_grid * expit(u_grid))) / np.sqrt(np.pi)
        rng = np.random.default_rng(42)
        fits = []
        for _ in range(8):
            u, x, noise = rng.normal(size=(3, 4000))
            t = rng.binomial(1, expit(u))
            y = 2 * t + u + noise
            row = []
            for controls in (x[:, None], np.column_stack([x, u])):
                row.extend(
                    (
                        fwl_estimate(y, t, controls, method="ols").theta,
                        double_ml(y, t, controls, model="ridge").theta,
                    )
                )
            fits.append(row)
        np.testing.assert_allclose(np.mean(fits, axis=0), [hidden_target] * 2 + [2] * 2, atol=0.065)


@pytest.mark.tier3
@pytest.mark.skipif(not econml_available(), reason="econml not installed (optional '[full]' extra)")
class TestCombinedMisspecifications:
    """Test methods under multiple simultaneous misspecifications."""

    @pytest.mark.tier3
    def test_dml_vs_baselines_combined_issues(self):
        """Compare DML with baselines under multiple misspecifications."""
        dgp = DGPGenerator(
            n=800,
            p=5,
            true_effect=2.0,
            confounding_strength=1.0,
            treatment_model="nonlinear",
            outcome_model="nonlinear",
            random_state=42,
        )
        comparison = BaselineComparison(n_simulations=10, random_state=42, include_dml=True)
        results = comparison.compare(dgp)
        assert all(result.metadata["comparison_data_seed"] == 42 for result in results.values())
        assert all(np.isfinite(result.bias) for result in results.values())
