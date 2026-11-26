"""Test suite for Rosenbaum bounds sensitivity analysis.

Tests the RosenbaumBounds class to ensure correct computation of
critical gamma, p-values, and sensitivity interpretations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dml_ts.sensitivity.rosenbaum import (
    RosenbaumBounds,
    SensitivityResult,
    compute_sensitivity_for_dml,
)


class TestRosenbaumBounds:
    """Test suite for RosenbaumBounds class."""

    @pytest.fixture
    def bounds(self) -> RosenbaumBounds:
        """Create default RosenbaumBounds instance."""
        return RosenbaumBounds()

    @pytest.mark.tier1
    def test_instantiation_default(self) -> None:
        """Test default instantiation."""
        bounds = RosenbaumBounds()
        assert bounds.gamma_max == 3.0
        assert bounds.gamma_step == 0.1
        assert bounds.alpha == 0.05

    @pytest.mark.tier1
    def test_instantiation_custom(self) -> None:
        """Test instantiation with custom parameters."""
        bounds = RosenbaumBounds(gamma_max=5.0, gamma_step=0.05, alpha=0.01)
        assert bounds.gamma_max == 5.0
        assert bounds.gamma_step == 0.05
        assert bounds.alpha == 0.01

    @pytest.mark.tier1
    def test_instantiation_invalid_gamma_max(self) -> None:
        """Test that invalid gamma_max raises ValueError."""
        with pytest.raises(ValueError, match="gamma_max must be > 1.0"):
            RosenbaumBounds(gamma_max=0.5)

    @pytest.mark.tier1
    def test_instantiation_invalid_alpha(self) -> None:
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            RosenbaumBounds(alpha=1.5)

    @pytest.mark.tier1
    def test_analyze_returns_sensitivity_result(self, bounds: RosenbaumBounds) -> None:
        """Test that analyze returns SensitivityResult."""
        result = bounds.analyze(theta=2.0, se=0.5, n_treated=500, n_control=500)
        assert isinstance(result, SensitivityResult)

    @pytest.mark.tier1
    def test_gamma_critical_increases_with_effect_size(self, bounds: RosenbaumBounds) -> None:
        """Test that larger effect sizes have higher critical gamma."""
        # Small effect
        small = bounds.analyze(theta=1.0, se=0.5, n_treated=500, n_control=500)

        # Large effect (same SE)
        large = bounds.analyze(theta=3.0, se=0.5, n_treated=500, n_control=500)

        # Larger effect should be more robust (higher gamma_critical)
        assert large.gamma_critical >= small.gamma_critical

    @pytest.mark.tier1
    def test_gamma_critical_decreases_with_se(self, bounds: RosenbaumBounds) -> None:
        """Test that larger SE leads to lower critical gamma."""
        # Low SE (precise estimate)
        precise = bounds.analyze(theta=2.0, se=0.3, n_treated=500, n_control=500)

        # High SE (imprecise estimate)
        imprecise = bounds.analyze(theta=2.0, se=0.8, n_treated=500, n_control=500)

        # Precise estimate should be more robust
        assert precise.gamma_critical >= imprecise.gamma_critical

    @pytest.mark.tier1
    def test_p_value_at_gamma_1_matches_standard(self, bounds: RosenbaumBounds) -> None:
        """Test that p-value at gamma=1 matches standard z-test."""
        theta = 2.0
        se = 0.5
        result = bounds.analyze(theta=theta, se=se, n_treated=500, n_control=500)

        # Standard two-sided p-value
        from scipy import stats

        expected_p = 2 * (1 - stats.norm.cdf(abs(theta) / se))

        actual_p = result.p_values_by_gamma.get(1.0)
        assert actual_p is not None
        assert np.isclose(actual_p, expected_p, rtol=0.01)

    @pytest.mark.tier1
    def test_p_values_increase_with_gamma(self, bounds: RosenbaumBounds) -> None:
        """Test that p-values increase as gamma increases."""
        result = bounds.analyze(theta=2.0, se=0.5, n_treated=500, n_control=500)

        gammas = sorted(result.p_values_by_gamma.keys())
        p_values = [result.p_values_by_gamma[g] for g in gammas]

        # p-values should be non-decreasing
        for i in range(1, len(p_values)):
            assert p_values[i] >= p_values[i - 1] - 1e-10  # Allow small numerical error

    @pytest.mark.tier1
    def test_interpretation_robust(self, bounds: RosenbaumBounds) -> None:
        """Test 'Robust' interpretation for high gamma_critical."""
        # Very large effect relative to SE should be robust
        result = bounds.analyze(theta=5.0, se=0.5, n_treated=1000, n_control=1000)
        assert result.interpretation == "Robust"

    @pytest.mark.tier1
    def test_interpretation_fragile(self) -> None:
        """Test 'Fragile' interpretation for low gamma_critical."""
        bounds = RosenbaumBounds(gamma_max=1.5)
        # Small effect with large SE
        result = bounds.analyze(theta=0.5, se=0.3, n_treated=100, n_control=100)

        # Should be fragile or at least sensitive
        assert result.interpretation in ["Fragile", "Sensitive"]

    @pytest.mark.tier1
    def test_invalid_se_raises(self, bounds: RosenbaumBounds) -> None:
        """Test that non-positive SE raises ValueError."""
        with pytest.raises(ValueError, match="Standard error must be positive"):
            bounds.analyze(theta=2.0, se=0.0, n_treated=500, n_control=500)

        with pytest.raises(ValueError, match="Standard error must be positive"):
            bounds.analyze(theta=2.0, se=-0.5, n_treated=500, n_control=500)

    @pytest.mark.tier1
    def test_invalid_sample_size_raises(self, bounds: RosenbaumBounds) -> None:
        """Test that non-positive sample sizes raise ValueError."""
        with pytest.raises(ValueError, match="Sample sizes must be positive"):
            bounds.analyze(theta=2.0, se=0.5, n_treated=0, n_control=500)

    @pytest.mark.tier1
    def test_result_summary(self, bounds: RosenbaumBounds) -> None:
        """Test that summary returns formatted string."""
        result = bounds.analyze(theta=2.0, se=0.5, n_treated=500, n_control=500)
        summary = result.summary()

        assert "Rosenbaum Bounds Sensitivity Analysis" in summary
        assert "Critical Gamma" in summary
        assert result.gamma_label in summary

    @pytest.mark.tier1
    def test_result_repr(self, bounds: RosenbaumBounds) -> None:
        """Test __repr__ method."""
        result = bounds.analyze(theta=2.0, se=0.5, n_treated=500, n_control=500)
        repr_str = repr(result)

        assert "SensitivityResult" in repr_str
        assert "Γ_crit" in repr_str


class TestPlotSensitivity:
    """Test suite for plot_sensitivity method."""

    @pytest.fixture
    def bounds(self) -> RosenbaumBounds:
        """Create default RosenbaumBounds instance."""
        return RosenbaumBounds()

    @pytest.fixture
    def result(self, bounds: RosenbaumBounds) -> SensitivityResult:
        """Create a sensitivity result for plotting tests."""
        return bounds.analyze(theta=2.0, se=0.5, n_treated=500, n_control=500)

    @pytest.mark.tier1
    def test_plot_returns_figure(self, bounds: RosenbaumBounds, result: SensitivityResult) -> None:
        """Test that plot_sensitivity returns a Figure."""
        fig = bounds.plot_sensitivity(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    @pytest.mark.tier1
    def test_plot_custom_figsize(self, bounds: RosenbaumBounds, result: SensitivityResult) -> None:
        """Test custom figure size."""
        fig = bounds.plot_sensitivity(result, figsize=(12, 8))
        assert fig.get_figwidth() == 12
        assert fig.get_figheight() == 8
        plt.close(fig)

    @pytest.mark.tier1
    def test_plot_custom_title(self, bounds: RosenbaumBounds, result: SensitivityResult) -> None:
        """Test custom title."""
        custom_title = "My Custom Title"
        fig = bounds.plot_sensitivity(result, title=custom_title)
        ax = fig.gca()
        assert ax.get_title() == custom_title
        plt.close(fig)


class TestComputeSensitivityForDML:
    """Test suite for the convenience function."""

    @pytest.mark.tier1
    def test_returns_sensitivity_result(self) -> None:
        """Test that function returns SensitivityResult."""
        result = compute_sensitivity_for_dml(theta=2.5, se=0.5, n_samples=1000, treatment_r2=0.3)
        assert isinstance(result, SensitivityResult)

    @pytest.mark.tier1
    def test_custom_parameters(self) -> None:
        """Test with custom gamma_max and alpha."""
        result = compute_sensitivity_for_dml(
            theta=2.5,
            se=0.5,
            n_samples=1000,
            treatment_r2=0.3,
            gamma_max=5.0,
            alpha=0.01,
        )
        assert result.alpha == 0.01

    @pytest.mark.tier1
    def test_balanced_proxy_cancels_sample_size_and_treatment_r2(self) -> None:
        from scipy.stats import norm

        expected = compute_sensitivity_for_dml(1.4, 0.5, 1000, treatment_r2=0.1)
        for n, r2 in [(1000, 0.8), (57, 0.9)]:
            got = compute_sensitivity_for_dml(1.4, 0.5, n, treatment_r2=r2)
            assert got.p_values_by_gamma == expected.p_values_by_gamma
            for gamma, p in got.p_values_by_gamma.items():
                assert p == pytest.approx(2 * norm.sf(max(0, 2.8 - np.log(gamma))))


@pytest.mark.tier1
@pytest.mark.parametrize("step,limit", [(0.2, 1.25), (0.004, 1.015), (0.1, 1.3)])
def test_grid_endpoint_and_censoring(step, limit):
    from scipy.stats import norm

    bounds = RosenbaumBounds(gamma_max=limit, gamma_step=step)
    result = bounds.analyze(10, 1, 30, 30)
    assert list(result.p_values_by_gamma) == list(result.gamma_values)
    assert np.all(np.diff(result.gamma_values) > 0)
    assert (result.gamma_values[0], result.gamma_values[-1]) == (1, limit)
    assert result.gamma_critical == limit and not result.crossing_observed
    assert "No crossing" in result.summary()
    theta = norm.isf(0.025) + np.log(np.mean(result.gamma_values[-2:]))
    endpoint = bounds.analyze(theta, 1, 30, 30)
    assert endpoint.gamma_critical == limit and endpoint.crossing_observed
    del result.p_values_by_gamma[limit]  # Legacy grids could omit the upper endpoint.
    assert not result.crossing_observed
    assert "No crossing" in result.summary()
