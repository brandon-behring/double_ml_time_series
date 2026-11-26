"""
Test suite for BaselineComparison framework.

Tests that BaselineComparison properly:
1. Runs all methods on same DGP
2. Generates summary statistics and tables
3. Handles comparison across multiple DGP configurations
4. Correctly identifies best/worst performing methods
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from dml_ts.dml import econml_available
from dml_ts.validation.baseline_comparison import BaselineComparison
from dml_ts.validation.bootstrap_config import BootstrapConfig
from dml_ts.validation.dgp_generator import DGPGenerator
from dml_ts.validation.ipw_baseline import AugmentedIPW, IPWEstimator
from dml_ts.validation.ml_baseline import RandomForestEstimator, XGBoostEstimator
from dml_ts.validation.ols_baseline import NaiveOLS, OLSWithControls


class TestBaselineComparisonBasic:
    """Test basic functionality of BaselineComparison."""

    @pytest.mark.tier1
    def test_instantiation_without_dml(self):
        """Test instantiation without DML method."""
        comp = BaselineComparison(n_simulations=10, include_dml=False)
        assert comp.n_simulations == 10
        assert not comp.include_dml
        assert len(comp.methods) == 4  # OLS + IPW methods only

    @pytest.mark.tier1
    def test_instantiation_with_dml(self):
        """Test instantiation with DML method included."""
        comp = BaselineComparison(n_simulations=10, include_dml=True)
        assert comp.n_simulations == 10
        assert comp.include_dml
        assert len(comp.methods) == 5  # OLS + IPW + DML

    @pytest.mark.tier2
    def test_compare_returns_all_methods(self):
        """Test that compare() returns results for all configured methods."""
        comp = BaselineComparison(n_simulations=5, include_dml=False, random_state=42)
        dgp = DGPGenerator(n=200, p=2, true_effect=2.0, random_state=42)

        results = comp.compare(dgp)

        # Should have results for all 4 baseline methods
        assert len(results) == 4
        assert "NaiveOLS" in results
        assert "OLSWithControls" in results
        assert "IPWEstimator" in results
        assert "AugmentedIPW" in results

    @pytest.mark.tier2
    @pytest.mark.parametrize(
        "estimator",
        [
            NaiveOLS,
            OLSWithControls,
            IPWEstimator,
            AugmentedIPW,
            RandomForestEstimator,
            XGBoostEstimator,
        ],
    )
    def test_bias_p_value_matches_simulation_moments(self, estimator):
        n = 5
        method = estimator(
            n_simulations=n,
            bootstrap_config=BootstrapConfig(n_bootstrap_bias=23, n_bootstrap_ci=2),
            random_state=123,
        )
        dgp = DGPGenerator(n=100, p=2, true_effect=2.0, random_state=123)
        result = method.validate(dgp)

        # The reported MSE and bias give the second and first raw moments of the n errors.
        standard_error = np.sqrt((result.mse - result.bias**2) / (n - 1))
        expected = 2 * stats.t.sf(abs(result.bias) / standard_error, df=n - 1)
        assert result.bias_p_value == pytest.approx(expected)

    @pytest.mark.tier2
    @pytest.mark.parametrize("estimator", [RandomForestEstimator, XGBoostEstimator])
    def test_bias_test_counts_draws_with_both_arms(self, estimator, monkeypatch):
        dgp = DGPGenerator(n=100, p=2, true_effect=2.0, random_state=123)
        draws = [dgp.generate() for _ in range(4)]
        draws[0].T[:] = 0
        monkeypatch.setattr(dgp, "generate", iter(draws).__next__)
        method = estimator(
            n_simulations=4,
            n_estimators=3,
            bootstrap_config=BootstrapConfig(n_bootstrap_bias=10, n_bootstrap_ci=2),
            random_state=123,
        )

        result = method.validate(dgp)

        assert result.n_simulations == 4
        assert result.metadata["n_bias_simulations"] == 3
        standard_error = np.sqrt((result.mse - result.bias**2) / 2)
        assert result.bias_p_value == pytest.approx(
            2 * stats.t.sf(abs(result.bias) / standard_error, df=2)
        )


@pytest.mark.tier2
class TestBaselineComparisonTables:
    """Test table generation functionality."""

    def test_create_comparison_table_format(self):
        """Test that comparison table has correct format."""
        comp = BaselineComparison(n_simulations=5, include_dml=False, random_state=42)
        dgp = DGPGenerator(n=200, p=2, true_effect=2.0, random_state=42)

        table = comp.create_comparison_table(dgp)

        # Check table structure
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 4  # One row per method
        assert "Method" in table.columns
        assert "Bias" in table.columns
        assert "MSE" in table.columns
        assert "RMSE" in table.columns
        assert "Coverage" in table.columns
        assert "Status" in table.columns

    def test_comparison_table_contains_all_methods(self):
        """Test that comparison table includes all methods."""
        comp = BaselineComparison(n_simulations=5, include_dml=False, random_state=42)
        dgp = DGPGenerator(n=200, p=2, true_effect=2.0, random_state=42)

        table = comp.create_comparison_table(dgp)

        methods_in_table = set(table["Method"].values)
        assert "NaiveOLS" in methods_in_table
        assert "OLSWithControls" in methods_in_table
        assert "IPWEstimator" in methods_in_table
        assert "AugmentedIPW" in methods_in_table

    def test_create_detailed_comparison_table(self):
        """Test detailed comparison table generation."""
        comp = BaselineComparison(n_simulations=5, include_dml=False, random_state=42)
        dgp = DGPGenerator(n=200, p=2, true_effect=2.0, random_state=42)

        table = comp.create_detailed_comparison_table(dgp)

        # Check for additional columns in detailed table
        assert "Abs_Bias" in table.columns
        assert "RMSE" in table.columns
        assert "CI_Width" in table.columns
        assert "P_Value" in table.columns

        # Values should be numeric
        assert pd.api.types.is_numeric_dtype(table["Bias"])
        assert pd.api.types.is_numeric_dtype(table["MSE"])


@pytest.mark.tier2
class TestBaselineComparisonAcrossDGPs:
    """Test comparison across multiple DGP configurations."""

    def test_compare_across_dgps_single_config(self):
        """Test comparison across single DGP configuration."""
        comp = BaselineComparison(n_simulations=5, include_dml=False, random_state=42)

        configs = [{"n": 300, "p": 3, "true_effect": 2.0, "random_state": 42}]

        results = comp.compare_across_dgps(configs)

        # Should have one table
        assert len(results) == 1
        assert isinstance(list(results.values())[0], pd.DataFrame)

    def test_compare_across_dgps_multiple_configs(self):
        """Test comparison across multiple DGP configurations."""
        comp = BaselineComparison(n_simulations=5, include_dml=False, random_state=42)

        configs = [
            {"n": 300, "p": 3, "true_effect": 2.0, "random_state": 42},
            {
                "n": 300,
                "p": 3,
                "true_effect": 2.0,
                "random_state": 42,
                "outcome_model": "nonlinear",
            },
        ]

        results = comp.compare_across_dgps(configs)

        # Should have one table per config
        assert len(results) == len(configs)
        assert any("outcome_model=nonlinear" in key for key in results)
        assert all(isinstance(v, pd.DataFrame) for v in results.values())

    def test_config_to_string_formatting(self):
        """Test that config strings are properly formatted."""
        comp = BaselineComparison(n_simulations=5, include_dml=False)

        config = {"n": 500, "p": 5, "true_effect": 2.0, "confounding_strength": 1.0}
        config_str = comp._config_to_string(config)

        # Should contain all config parameters
        assert "n=500" in config_str
        assert "p=5" in config_str
        assert "true_effect=2.0" in config_str
        assert "confounding_strength=1.0" in config_str


@pytest.mark.tier2
class TestBaselineComparisonSummaryStats:
    """Test summary statistics generation."""

    def test_generate_summary_statistics_structure(self):
        """Test that summary statistics have correct structure."""
        comp = BaselineComparison(n_simulations=5, include_dml=False, random_state=42)
        dgp = DGPGenerator(n=300, p=3, true_effect=2.0, random_state=42)

        summary = comp.generate_summary_statistics(dgp)

        # Check required keys
        assert "dgp_config" in summary
        assert "best_bias_method" in summary
        assert "worst_bias_method" in summary
        assert "best_mse_method" in summary
        assert "all_biases" in summary
        assert "all_mses" in summary
        assert "mean_absolute_bias" in summary

    def test_best_worst_methods_are_valid(self):
        """Test that best/worst method identification is valid."""
        comp = BaselineComparison(n_simulations=5, include_dml=False, random_state=42)
        dgp = DGPGenerator(n=300, p=3, true_effect=2.0, random_state=42)

        summary = comp.generate_summary_statistics(dgp)

        # Best and worst should be different methods
        best_bias = summary["best_bias_method"]
        worst_bias = summary["worst_bias_method"]

        # Should be valid method names
        valid_methods = {"NaiveOLS", "OLSWithControls", "IPWEstimator", "AugmentedIPW"}
        assert best_bias in valid_methods
        assert worst_bias in valid_methods

    def test_summary_statistics_biases_match_all_biases(self):
        """Test that summary stats biases match individual biases."""
        comp = BaselineComparison(n_simulations=5, include_dml=False, random_state=42)
        dgp = DGPGenerator(n=300, p=3, true_effect=2.0, random_state=42)

        summary = comp.generate_summary_statistics(dgp)

        # Best bias should be lowest in all_biases
        best_bias_method = summary["best_bias_method"]
        all_biases = summary["all_biases"]
        best_bias_value = all_biases[best_bias_method]

        # Should be the minimum
        assert best_bias_value == min(all_biases.values())

    def test_mean_metrics_are_reasonable(self):
        """Test that mean metrics are within reasonable ranges."""
        comp = BaselineComparison(n_simulations=5, include_dml=False, random_state=42)
        dgp = DGPGenerator(n=300, p=3, true_effect=2.0, random_state=42)

        summary = comp.generate_summary_statistics(dgp)

        # Mean metrics should be non-negative
        assert summary["mean_absolute_bias"] >= 0
        assert summary["mean_mse"] >= 0

        # Coverage should be in [0, 1]
        assert 0 <= summary["mean_coverage"] <= 1


@pytest.mark.tier4
@pytest.mark.skipif(not econml_available(), reason="econml not installed (optional '[full]' extra)")
class TestBaselineComparisonWithDML:
    """Test comparison framework with DML included (requires econml)."""

    def test_compare_with_dml_included(self):
        """Test that comparison works with DML included."""
        comp = BaselineComparison(n_simulations=5, include_dml=True, random_state=42)
        dgp = DGPGenerator(n=300, p=3, true_effect=2.0, random_state=42)

        results = comp.compare(dgp)

        # Should have 5 methods (4 baselines + DML)
        assert len(results) == 5
        assert "DML" in results

    def test_dml_appears_in_comparison_table(self):
        """Test that DML appears in comparison table when included."""
        comp = BaselineComparison(n_simulations=5, include_dml=True, random_state=42)
        dgp = DGPGenerator(n=300, p=3, true_effect=2.0, random_state=42)

        table = comp.create_comparison_table(dgp)

        # DML should be in the table
        assert "DML" in table["Method"].values


@pytest.mark.tier2
class TestBaselineComparisonReproducibility:
    """Test reproducibility of comparison results."""

    def test_same_seed_produces_same_results(self):
        """Test that same random seed produces identical results."""
        comp1 = BaselineComparison(n_simulations=5, include_dml=False, random_state=42)
        comp2 = BaselineComparison(n_simulations=5, include_dml=False, random_state=42)

        dgp1 = DGPGenerator(n=300, p=3, true_effect=2.0, random_state=42)
        dgp2 = DGPGenerator(n=300, p=3, true_effect=2.0, random_state=42)

        results1 = comp1.compare(dgp1)
        results2 = comp2.compare(dgp2)

        # Results should match
        for method_name in results1:
            assert np.isclose(results1[method_name].bias, results2[method_name].bias, atol=1e-10)

    def test_different_seed_produces_different_results(self):
        """Test that different DGP seeds produce different results (C3 fix ensures deterministic comparisons)."""
        # After C3 fix: same DGP → same results (deterministic comparison)
        # Different DGP seeds → different results
        comp1 = BaselineComparison(n_simulations=10, include_dml=False, random_state=42)
        comp2 = BaselineComparison(n_simulations=10, include_dml=False, random_state=42)

        dgp1 = DGPGenerator(n=300, p=3, true_effect=2.0, random_state=42)
        dgp2 = DGPGenerator(n=300, p=3, true_effect=2.0, random_state=123)

        results1 = comp1.compare(dgp1)
        results2 = comp2.compare(dgp2)

        # At least some results should differ (different DGPs)
        biases_differ = False
        for method_name in results1:
            if not np.isclose(results1[method_name].bias, results2[method_name].bias, atol=0.01):
                biases_differ = True
                break

        assert biases_differ


@pytest.mark.tier1
@pytest.mark.parametrize(("dgp_seed", "comparison_seed"), [(317, None), (None, 317), (None, None)])
def test_paired_inputs_are_replayable_and_method_order_independent(dgp_seed, comparison_seed):
    from datetime import datetime
    from hashlib import sha256

    from dml_ts.validation.validation_result import ValidationResult

    class Recorder:
        def validate(self, dgp):
            hashes = []
            for _ in range(3):
                draw = dgp.generate()
                hashes.append(
                    sha256(draw.X.tobytes() + draw.T.tobytes() + draw.Y.tobytes()).hexdigest()
                )
            return ValidationResult(
                "record", "PASS", 0.0, 0.0, 1.0, 0.0, 0.0, 3, datetime.now(), {"hashes": hashes}
            )

    dgp = DGPGenerator(n=30, p=2, true_effect=2, random_state=dgp_seed)
    comparison = BaselineComparison(random_state=comparison_seed)
    comparison.methods = {"a": Recorder(), "b": Recorder()}
    first = comparison.compare(dgp)
    assert first["a"].metadata == first["b"].metadata
    assert len(set(first["a"].metadata["hashes"])) == 3
    comparison.random_state = first["a"].metadata["comparison_data_seed"]
    comparison.methods = dict(reversed(comparison.methods.items()))
    replay = comparison.compare(dgp)
    assert replay["b"].metadata == first["a"].metadata == replay["a"].metadata


@pytest.mark.tier1
def test_best_coverage_uses_distance_from_configured_nominal(monkeypatch):
    from datetime import datetime

    from dml_ts.validation.validation_result import ValidationResult

    def result(coverage):
        return ValidationResult(
            "test", "PASS", 0.0, 0.0, coverage, 0.0, 0.0, 10, datetime.now(), {"alpha": 0.2}
        )

    comparison = BaselineComparison()
    monkeypatch.setattr(
        comparison, "compare", lambda dgp: {"always": result(1.0), "calibrated": result(0.8)}
    )
    summary = comparison.generate_summary_statistics(DGPGenerator(n=30, p=1, true_effect=2.0))
    assert summary["best_coverage_method"] == "calibrated"
