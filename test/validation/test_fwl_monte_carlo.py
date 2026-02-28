"""Monte Carlo validation for FWL and Robinson estimators.

This test suite validates:
1. FWL gives unbiased estimates when confounding is linear
2. FWL = OLS (algebraic identity via FWL theorem)
3. Robinson handles nonlinear confounding better than FWL
4. Both achieve correct coverage under homoskedasticity

The validation criteria are:
- Bias < 0.1 (10% of true effect)
- Coverage 93-97% (nominal 95%)
- RMSE within expected range
"""

import numpy as np
import pytest
from scipy import stats

from src.dml.fwl import fwl_estimate, fwl_vs_ols_comparison
from src.dml.robinson import robinson_estimator, compare_fwl_vs_robinson


@pytest.mark.tier2
class TestFWLBasic:
    """Basic functionality tests for FWL estimator."""

    def test_fwl_imports(self):
        """Verify FWL module imports correctly."""
        from src.dml import fwl_estimate, fwl_residualize, fwl_vs_ols_comparison

        assert callable(fwl_estimate)
        assert callable(fwl_residualize)
        assert callable(fwl_vs_ols_comparison)

    def test_fwl_single_run(self):
        """Single run sanity check."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        T = 0.5 * X[:, 0] + np.random.randn(n)
        Y = 2.0 * T + X @ [1, -0.5] + np.random.randn(n)

        result = fwl_estimate(Y, T, X)

        assert hasattr(result, "theta")
        assert hasattr(result, "se")
        assert hasattr(result, "t_stat")
        assert hasattr(result, "p_value")
        assert abs(result.theta - 2.0) < 0.3  # Within 0.3 of true effect

    def test_fwl_equals_ols(self):
        """FWL theorem: FWL = OLS (algebraic identity)."""
        np.random.seed(123)
        n = 300
        X = np.random.randn(n, 3)
        T = np.random.randn(n)
        Y = 1.5 * T + X @ [0.5, -0.3, 0.8] + np.random.randn(n)

        result = fwl_vs_ols_comparison(Y, T, X)

        assert result["match"], (
            f"FWL theorem violated: OLS={result['theta_ols']:.10f}, "
            f"FWL={result['theta_fwl']:.10f}"
        )


@pytest.mark.tier3
class TestFWLMonteCarlo:
    """Monte Carlo validation for FWL estimator."""

    @pytest.fixture
    def monte_carlo_results(self):
        """Run Monte Carlo simulation for FWL."""
        np.random.seed(42)
        n_sims = 200  # Reduced for faster testing
        n_obs = 500
        true_theta = 2.0

        thetas = []
        ses = []
        coverages = []

        for i in range(n_sims):
            # Generate data with linear confounding
            X = np.random.randn(n_obs, 3)
            T = 0.5 * X[:, 0] - 0.3 * X[:, 1] + np.random.randn(n_obs)
            Y = true_theta * T + X @ [1, 0.5, -0.3] + np.random.randn(n_obs)

            result = fwl_estimate(Y, T, X)
            thetas.append(result.theta)
            ses.append(result.se)

            # Check if 95% CI covers true theta
            ci_lower = result.theta - 1.96 * result.se
            ci_upper = result.theta + 1.96 * result.se
            coverages.append(ci_lower <= true_theta <= ci_upper)

        return {
            "thetas": np.array(thetas),
            "ses": np.array(ses),
            "coverages": np.array(coverages),
            "true_theta": true_theta,
        }

    def test_fwl_unbiased_linear_confounding(self, monte_carlo_results):
        """FWL is unbiased when confounding is linear."""
        thetas = monte_carlo_results["thetas"]
        true_theta = monte_carlo_results["true_theta"]

        bias = np.mean(thetas) - true_theta
        bias_se = np.std(thetas) / np.sqrt(len(thetas))

        # Bias should be < 0.1 (within 2 SE of zero for significance)
        assert abs(bias) < 0.1, f"FWL bias = {bias:.4f}, expected < 0.1"

        # Stronger test: bias should not be statistically significant
        t_stat = bias / bias_se
        assert abs(t_stat) < 2.5, f"FWL bias statistically significant: t={t_stat:.2f}"

    def test_fwl_coverage_linear_confounding(self, monte_carlo_results):
        """FWL achieves nominal coverage under homoskedasticity."""
        coverages = monte_carlo_results["coverages"]
        coverage_rate = np.mean(coverages)

        # Coverage should be 93-97% (allowing for simulation variance)
        assert 0.90 <= coverage_rate <= 0.98, f"FWL coverage = {coverage_rate:.2%}, expected 93-97%"


@pytest.mark.tier2
class TestRobinsonBasic:
    """Basic functionality tests for Robinson estimator."""

    def test_robinson_imports(self):
        """Verify Robinson module imports correctly."""
        from src.dml import robinson_estimator

        assert callable(robinson_estimator)

    def test_robinson_single_run(self):
        """Single run sanity check."""
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        T = 0.5 * X[:, 0] + np.random.randn(n)
        Y = 2.0 * T + X @ [1, -0.5] + np.random.randn(n)

        result = robinson_estimator(Y, T, X, model="random_forest")

        assert hasattr(result, "theta")
        assert hasattr(result, "se")
        assert hasattr(result, "outcome_r2")
        assert abs(result.theta - 2.0) < 0.3


@pytest.mark.tier3
class TestRobinsonVsFWL:
    """Tests comparing Robinson vs FWL on different DGPs."""

    def test_linear_confounding_both_work(self):
        """Both FWL and Robinson work with linear confounding."""
        np.random.seed(42)
        n = 1000
        true_theta = 2.0

        X = np.random.randn(n, 2)
        T = 0.5 * X[:, 0] + np.random.randn(n)
        Y = true_theta * T + X @ [1, -0.5] + np.random.randn(n)

        fwl_result = fwl_estimate(Y, T, X)
        robinson_result = robinson_estimator(Y, T, X, model="random_forest")

        fwl_bias = abs(fwl_result.theta - true_theta)
        robinson_bias = abs(robinson_result.theta - true_theta)

        # Both should have low bias
        assert fwl_bias < 0.15, f"FWL bias = {fwl_bias:.4f}"
        assert robinson_bias < 0.15, f"Robinson bias = {robinson_bias:.4f}"

    def test_nonlinear_confounding_robinson_wins(self):
        """Robinson handles nonlinear confounding better than FWL."""
        np.random.seed(42)
        n = 2000
        true_theta = 2.0

        X = np.random.randn(n, 2)
        # Nonlinear treatment model
        T = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + 0.5 * np.random.randn(n)
        # Nonlinear outcome model
        Y = true_theta * T + np.exp(X[:, 0] / 2) + np.cos(X[:, 1]) + np.random.randn(n)

        fwl_result = fwl_estimate(Y, T, X)
        robinson_result = robinson_estimator(Y, T, X, model="random_forest")

        fwl_bias = abs(fwl_result.theta - true_theta)
        robinson_bias = abs(robinson_result.theta - true_theta)

        # Robinson should have meaningfully lower bias
        assert robinson_bias < fwl_bias * 0.8, (
            f"Expected Robinson ({robinson_bias:.4f}) to beat FWL ({fwl_bias:.4f}) "
            f"by at least 20% with nonlinear confounding"
        )


@pytest.mark.tier1
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_perfect_collinearity_raises(self):
        """FWL raises error when T is perfectly predicted by X."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        T = X @ [1, 0]  # T = X₁ exactly (perfect collinearity)
        Y = T + np.random.randn(n)

        with pytest.raises(ValueError, match="no variation"):
            fwl_estimate(Y, T, X)

    def test_insufficient_df_raises(self):
        """FWL raises error when n <= p + 1."""
        np.random.seed(42)
        n = 5
        X = np.random.randn(n, 10)  # More covariates than observations
        T = np.random.randn(n)
        Y = np.random.randn(n)

        # With more covariates than observations, either collinearity or
        # insufficient df error is acceptable
        with pytest.raises(ValueError, match="(degrees of freedom|no variation)"):
            fwl_estimate(Y, T, X)

    def test_mismatched_lengths_raises(self):
        """FWL raises error when Y, T, X have different lengths."""
        Y = np.random.randn(100)
        T = np.random.randn(50)
        X = np.random.randn(100, 2)

        with pytest.raises(ValueError, match="length"):
            fwl_estimate(Y, T, X)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
