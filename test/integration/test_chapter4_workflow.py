"""Integration tests for Chapter 4 workflow.

Tests the complete pipeline from OJ dataset loading through DML estimation
to sensitivity analysis. These tests verify that all components work
together correctly.
"""

import numpy as np
import pytest

from dml_ts.data import OJDataLoader, OJDataset
from dml_ts.dml import DMLResult, double_ml
from dml_ts.sensitivity import RosenbaumBounds, SensitivityResult, compute_sensitivity_for_dml


@pytest.mark.network
class TestChapter4Workflow:
    """Integration tests for Chapter 4: Cross-Sectional Application.

    OJDataLoader is constructed with cache_dir=None (no caching), so every
    load() downloads the Dominick's CSV — the whole class carries the
    network marker.
    """

    @pytest.fixture(scope="class")
    def oj_data(self) -> OJDataset:
        """Load OJ dataset once for all integration tests."""
        loader = OJDataLoader()
        return loader.load()

    @pytest.mark.tier2
    def test_oj_dml_pipeline(self, oj_data: OJDataset) -> None:
        """Reproduce the pooled slope and its four uncertainty calculations."""
        import statsmodels.api as sm
        from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

        # Run DML on full OJ dataset
        result = double_ml(
            Y=oj_data.Y,
            T=oj_data.T,
            X=oj_data.X,
            n_folds=5,
            model="random_forest",
            random_state=42,
        )

        # Basic sanity checks
        assert isinstance(result, DMLResult)
        assert not np.isnan(result.theta)
        assert not np.isnan(result.se)
        assert result.se > 0

        # Compare to measured outputs on the pinned extract and verify the
        # two-way correction independently on the normalized influence scores.
        frame = oj_data.raw_df
        assert frame is not None
        assert frame.groupby(["store", "week"]).size().eq(3).all()
        assert not frame.duplicated(["store", "brand", "week"]).any()
        assert result.theta == pytest.approx(-1.1673, abs=0.001)
        assert result.se == pytest.approx(0.01584, abs=0.0001)
        final = sm.OLS(result.Y_residual, result.T_residual, hasconst=False).fit()
        both, store, week = cov_cluster_2groups(
            final, frame.store.to_numpy(), frame.week.to_numpy()
        )
        scores = frame.assign(score=result.influence_scores)
        components = []
        for keys in [["store"], ["week"], ["store", "week"]]:
            sums = scores.groupby(keys)["score"].sum().to_numpy()
            components.append(len(sums) / (len(sums) - 1) * sums @ sums / len(scores) ** 2)
        np.testing.assert_allclose(
            [store.item(), week.item(), both.item()],
            [components[0], components[1], components[0] + components[1] - components[2]],
        )
        np.testing.assert_allclose(
            np.sqrt([store.item(), week.item(), both.item()]), [0.0710, 0.0945, 0.1171], atol=0.0003
        )

    @pytest.mark.tier2
    def test_baseline_comparison_oj(self, oj_data: OJDataset) -> None:
        """Compare the three pooled conditional slopes on the same extract."""
        from sklearn.linear_model import LinearRegression

        # Naive OLS: regress Y on T only (ignores confounders)
        naive_model = LinearRegression()
        naive_model.fit(oj_data.T.reshape(-1, 1), oj_data.Y)
        naive_theta = naive_model.coef_[0]

        # OLS with controls: regress Y on [T, X]
        TX = np.column_stack([oj_data.T, oj_data.X])
        controls_model = LinearRegression()
        controls_model.fit(TX, oj_data.Y)
        controls_theta = controls_model.coef_[0]

        # DML estimate
        dml_result = double_ml(
            Y=oj_data.Y,
            T=oj_data.T,
            X=oj_data.X,
            n_folds=5,
            model="random_forest",
            random_state=42,
        )

        assert naive_theta == pytest.approx(-1.6013, abs=0.0001)
        assert controls_theta == pytest.approx(-1.1697, abs=0.0001)
        assert dml_result.theta == pytest.approx(-1.1673, abs=0.001)

    @pytest.mark.tier2
    def test_sensitivity_with_dml_results(self, oj_data: OJDataset) -> None:
        """Test sensitivity analysis on DML price elasticity estimate."""
        # Run DML
        dml_result = double_ml(
            Y=oj_data.Y,
            T=oj_data.T,
            X=oj_data.X,
            n_folds=5,
            model="random_forest",
            random_state=42,
        )

        # Run sensitivity analysis
        sensitivity = compute_sensitivity_for_dml(
            theta=dml_result.theta,
            se=dml_result.se,
            n_samples=oj_data.n_samples,
            treatment_r2=dml_result.treatment_r2_cv,
            gamma_max=3.0,
            alpha=0.05,
        )

        # Basic validation
        assert isinstance(sensitivity, SensitivityResult)
        assert sensitivity.gamma_critical >= 1.0

        # Result should have expected attributes
        assert len(sensitivity.p_values_by_gamma) > 0
        assert sensitivity.interpretation in [
            "Robust",
            "Moderately Robust",
            "Sensitive",
            "Fragile",
        ]

        # p-value at gamma=1 should match (approximately) the DML p-value
        # Both should indicate significance
        p_at_1 = sensitivity.p_values_by_gamma.get(1.0, 1.0)
        assert p_at_1 < 0.05, f"p-value at Γ=1 should be significant, got {p_at_1}"

        # Check the diagnostic interface with observed values as inputs.
        assert sensitivity.gamma_critical > 1.0, (
            f"Expected gamma_critical > 1 for significant effect, got {sensitivity.gamma_critical}"
        )

    @pytest.mark.tier2
    def test_full_chapter4_pipeline_reproducible(self) -> None:
        """Test that full pipeline is reproducible with same random state."""
        loader = OJDataLoader()
        data = loader.load()

        # Run 1
        result1 = double_ml(
            Y=data.Y,
            T=data.T,
            X=data.X,
            n_folds=5,
            model="random_forest",
            random_state=42,
        )

        sens1 = compute_sensitivity_for_dml(
            theta=result1.theta,
            se=result1.se,
            n_samples=data.n_samples,
            treatment_r2=result1.treatment_r2_cv,
        )

        # Run 2 (same random state)
        result2 = double_ml(
            Y=data.Y,
            T=data.T,
            X=data.X,
            n_folds=5,
            model="random_forest",
            random_state=42,
        )

        sens2 = compute_sensitivity_for_dml(
            theta=result2.theta,
            se=result2.se,
            n_samples=data.n_samples,
            treatment_r2=result2.treatment_r2_cv,
        )

        # Results should be identical
        assert np.isclose(result1.theta, result2.theta)
        assert np.isclose(result1.se, result2.se)
        assert np.isclose(sens1.gamma_critical, sens2.gamma_critical)

    @pytest.mark.tier2
    def test_brand_specific_analysis(self) -> None:
        """Test DML analysis on single brand (Tropicana)."""
        loader = OJDataLoader(brand="tropicana")
        data = loader.load()

        # Should have fewer samples than full dataset
        assert data.n_samples < 15_000

        # Run DML
        result = double_ml(
            Y=data.Y,
            T=data.T,
            X=data.X,
            n_folds=5,
            model="random_forest",
            random_state=42,
        )

        # Should still get negative elasticity
        assert result.theta < 0

        # Should be significant
        assert result.p_value < 0.10  # More lenient with smaller sample


@pytest.mark.network
class TestEdgeCases:
    """Edge case tests for Chapter 4 workflow.

    These construct their own OJDataLoader instances (cache_dir=None), so
    every run downloads the CSV — network-marked, and tier2 because both
    tests run DML estimation.
    """

    @pytest.mark.tier2
    def test_dml_with_extended_features(self) -> None:
        """Test DML with more confounders."""
        loader = OJDataLoader(features=["feat", "INCOME", "AGE60", "EDUC", "ETHNIC", "HHLARGE"])
        data = loader.load()

        result = double_ml(
            Y=data.Y,
            T=data.T,
            X=data.X,
            n_folds=3,  # Fewer folds for speed
            model="ridge",  # Faster model
            random_state=42,
        )

        assert not np.isnan(result.theta)
        assert result.theta < 0  # Still expect negative elasticity

    @pytest.mark.tier2
    def test_sensitivity_plot_integration(self) -> None:
        """Test that sensitivity plot works with real DML results."""
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend for testing

        # Simple DML run
        loader = OJDataLoader()
        data = loader.load()

        # Use ridge for speed, subset for faster test
        n_subset = 5000
        indices = np.random.RandomState(42).choice(data.n_samples, n_subset, replace=False)

        result = double_ml(
            Y=data.Y[indices],
            T=data.T[indices],
            X=data.X[indices],
            n_folds=3,
            model="ridge",
            random_state=42,
        )

        # Sensitivity analysis
        bounds = RosenbaumBounds(gamma_max=2.5)
        sensitivity = bounds.analyze(
            theta=result.theta,
            se=result.se,
            n_treated=n_subset // 2,
            n_control=n_subset // 2,
        )

        # Plot should not raise
        fig = bounds.plot_sensitivity(sensitivity)
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close(fig)
