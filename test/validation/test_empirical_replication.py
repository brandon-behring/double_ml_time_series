"""
Tests for empirical replication module (401(k) analysis).

Test coverage for Chernozhukov et al. (2018) 401(k) DML replication.

Tier assignments:
    tier1: Init, constants, dataclass tests (no estimation)
    tier2: Data loading, preprocessing (network → cached)
    tier4: Full PLR replication (RF/Lasso), integration workflows
"""

import numpy as np
import pytest

from dml_ts.dml import econml_available
from dml_ts.validation.empirical_replication import (
    FourZeroOneKReplication,
    ReplicationResult,
)

# replicate_plr_rf / replicate_plr_lasso estimate via EconML LinearDML, so these
# classes need the optional [full] extra; skip rather than hard-fail when absent.
requires_econml = pytest.mark.skipif(
    not econml_available(), reason="econml not installed (optional '[full]' extra)"
)


@pytest.mark.tier1
class TestFourZeroOneKReplicationBasicFunctionality:
    """Test basic functionality and interface."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        replicator = FourZeroOneKReplication()

        assert replicator.data_path is None
        assert replicator.random_state is None
        assert replicator.tolerance == 0.15

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        replicator = FourZeroOneKReplication(
            data_path="/path/to/data.csv",
            random_state=42,
            tolerance=0.10,
        )

        assert replicator.data_path == "/path/to/data.csv"
        assert replicator.random_state == 42
        assert replicator.tolerance == 0.10

    def test_published_ates_constants(self):
        """Test that published ATE constants are defined correctly."""
        assert FourZeroOneKReplication.PUBLISHED_ATES["PLR_RF"] == 9247.0
        assert FourZeroOneKReplication.PUBLISHED_ATES["PLR_Lasso"] == 8187.0
        # IRM_RF removed — IRM replication deferred (see Appendix A)
        assert "IRM_RF" not in FourZeroOneKReplication.PUBLISHED_ATES


@pytest.mark.tier1
@pytest.mark.parametrize("treatment", ["e401", "p401"])
@pytest.mark.parametrize(
    "method,target", [("replicate_plr_rf", 9247), ("replicate_plr_lasso", 8187)]
)
def test_mean_inference_and_benchmark_scope(monkeypatch, method, target, treatment):
    import sys
    from types import SimpleNamespace

    import pandas as pd

    sample = pd.DataFrame(
        {
            "net_tfa": [10.0, 20.0, 30.0],
            "e401": [0, 1, 1],
            "p401": [0, 0, 1],
            "age": [45, 51, 47],
        }
    )
    replicator = FourZeroOneKReplication(random_state=42, tolerance=0)
    monkeypatch.setattr(replicator, "load_data", lambda: sample)

    class FittedModel:
        def __init__(self, **kwargs):
            assert kwargs["cv"] == 5
            assert kwargs["discrete_treatment"] is True

        def fit(self, Y, T, X):
            np.testing.assert_array_equal(T, sample[treatment].to_numpy())
            np.testing.assert_array_equal(X, sample[["age"]].to_numpy())

        def ate_inference(self, X):
            return SimpleNamespace(
                mean_point=target,
                stderr_mean=125.0,
                conf_int_mean=lambda *, alpha: (target - 245.0, target + 245.0),
            )

    monkeypatch.setitem(sys.modules, "econml.dml", SimpleNamespace(LinearDML=FittedModel))
    result = getattr(replicator, method)(treatment=treatment)
    assert result.ate_estimate == target
    assert result.std_error == 125.0
    assert (result.ci_lower, result.ci_upper) == (target - 245.0, target + 245.0)
    assert result.p_value is None
    assert result.metadata["control_columns"] == ["age"]
    assert result.metadata["n_partitions"] == 1
    if treatment == "e401":
        assert result.published_ate == target
        assert result.difference == result.rel_difference == 0
        assert result.status == "WITHIN_TOLERANCE"
        assert "Panel B" in result.metadata["benchmark"]
    else:
        assert result.published_ate is result.difference is result.rel_difference is None
        assert result.status == "NO_BENCHMARK"
        assert result.metadata["benchmark"] is None


@pytest.mark.tier2
class TestFourZeroOneKReplicationDataLoading:
    """Test data loading functionality."""

    def test_load_local_sipp_fixture(self):
        """Load the checked-in SIPP sample."""
        replicator = FourZeroOneKReplication(data_path="test/fixtures/401k_data.csv")
        df = replicator.load_data()

        # Check shape (n=9,915, 14 variables)
        assert df.shape == (9915, 14)

        # Check key columns exist
        assert "net_tfa" in df.columns
        assert "e401" in df.columns
        assert "p401" in df.columns
        assert "age" in df.columns
        assert "inc" in df.columns

    def test_load_data_caching(self):
        """Test that data is cached after first load."""
        replicator = FourZeroOneKReplication(data_path="test/fixtures/401k_data.csv")

        # First load
        df1 = replicator.load_data()

        # Second load should return cached data
        df2 = replicator.load_data()

        assert df1 is df2  # Same object reference


@pytest.mark.tier2
class TestFourZeroOneKReplicationPreprocessing:
    """Test data preprocessing functionality."""

    def test_preprocess_data_with_e401(self):
        """Test preprocessing with e401 (eligibility) treatment."""
        replicator = FourZeroOneKReplication(
            data_path="test/fixtures/401k_data.csv", random_state=42
        )
        Y, T, X = replicator.preprocess_data(treatment="e401")

        # Check shapes
        assert Y.shape == (9915,)
        assert T.shape == (9915,)
        assert X.shape[0] == 9915
        assert X.shape[1] == 11  # Nine published controls plus nifa and tw.

    def test_preprocess_data_with_p401(self):
        """Test preprocessing with p401 (participation) treatment."""
        replicator = FourZeroOneKReplication(
            data_path="test/fixtures/401k_data.csv", random_state=42
        )
        Y, T, X = replicator.preprocess_data(treatment="p401")

        # Check shapes
        assert Y.shape == (9915,)
        assert T.shape == (9915,)
        assert X.shape[0] == 9915

    def test_preprocess_data_invalid_treatment(self):
        """Test that invalid treatment raises error."""
        replicator = FourZeroOneKReplication(data_path="test/fixtures/401k_data.csv")

        with pytest.raises(ValueError, match="treatment must be"):
            replicator.preprocess_data(treatment="invalid")

    def test_preprocess_data_caching(self):
        """Test that preprocessed data is cached."""
        replicator = FourZeroOneKReplication(
            data_path="test/fixtures/401k_data.csv", random_state=42
        )

        # First preprocessing
        Y1, T1, X1 = replicator.preprocess_data()

        # Second preprocessing should return same arrays
        Y2, T2, X2 = replicator.preprocess_data()

        assert np.array_equal(Y1, Y2)
        assert np.array_equal(T1, T2)
        assert np.array_equal(X1, X2)


@pytest.mark.tier4
@pytest.mark.network
@requires_econml
class TestFourZeroOneKReplicationPLRRandomForest:
    """Test PLR Random Forest replication."""

    def test_replicate_plr_rf_returns_result(self):
        """Test that replicate_plr_rf returns proper result object."""
        replicator = FourZeroOneKReplication(random_state=42)
        result = replicator.replicate_plr_rf()

        assert isinstance(result, ReplicationResult)
        assert result.method == "PLR_RF"
        assert result.published_ate == 9247.0

    def test_replicate_plr_rf_ate_in_reasonable_range(self):
        """Test that ATE estimate is in reasonable range."""
        replicator = FourZeroOneKReplication(random_state=42)
        result = replicator.replicate_plr_rf()

        # Companion single-split smoke-test range
        # Allow ±50% range (very generous for first test)
        assert 4500 <= result.ate_estimate <= 14000

    def test_replicate_plr_rf_has_valid_ci(self):
        """Test that confidence interval is valid."""
        replicator = FourZeroOneKReplication(random_state=42)
        result = replicator.replicate_plr_rf()

        # CI should contain ATE
        assert result.ci_lower <= result.ate_estimate <= result.ci_upper

        # CI should be non-degenerate
        assert result.ci_upper > result.ci_lower

    def test_replicate_plr_rf_std_error_positive(self):
        """Test that standard error is positive."""
        replicator = FourZeroOneKReplication(random_state=42)
        result = replicator.replicate_plr_rf()

        assert result.std_error > 0

    def test_replicate_plr_rf_reproducibility(self):
        """Test that results are reproducible with same random_state."""
        rep1 = FourZeroOneKReplication(random_state=42)
        result1 = rep1.replicate_plr_rf()

        rep2 = FourZeroOneKReplication(random_state=42)
        result2 = rep2.replicate_plr_rf()

        assert np.isclose(result1.ate_estimate, result2.ate_estimate, rtol=0.01)
        assert np.isclose(result1.std_error, result2.std_error, rtol=0.01)

    def test_replicate_plr_rf_metadata(self):
        """Test that result metadata contains expected fields."""
        replicator = FourZeroOneKReplication(random_state=42)
        result = replicator.replicate_plr_rf()

        assert "treatment" in result.metadata
        assert "n_estimators" in result.metadata
        assert "n_obs" in result.metadata
        assert "n_controls" in result.metadata

        assert result.metadata["treatment"] == "e401"
        assert result.metadata["n_obs"] == 9915
        assert result.metadata["n_controls"] == 11


@pytest.mark.tier4
@pytest.mark.network
@requires_econml
class TestFourZeroOneKReplicationPLRLasso:
    """Test PLR Lasso replication."""

    def test_replicate_plr_lasso_returns_result(self):
        """Test that replicate_plr_lasso returns proper result object."""
        replicator = FourZeroOneKReplication(random_state=42)
        result = replicator.replicate_plr_lasso()

        assert isinstance(result, ReplicationResult)
        assert result.method == "PLR_Lasso"
        assert result.published_ate == 8187.0

    def test_replicate_plr_lasso_ate_in_reasonable_range(self):
        """Test that ATE estimate is in reasonable range."""
        replicator = FourZeroOneKReplication(random_state=42)
        result = replicator.replicate_plr_lasso()

        # Companion single-split smoke-test range
        # Allow ±50% range
        assert 4500 <= result.ate_estimate <= 14500

    def test_replicate_plr_lasso_has_valid_ci(self):
        """Test that confidence interval is valid."""
        replicator = FourZeroOneKReplication(random_state=42)
        result = replicator.replicate_plr_lasso()

        # CI should contain ATE
        assert result.ci_lower <= result.ate_estimate <= result.ci_upper


@pytest.mark.tier4
@pytest.mark.network
@requires_econml
class TestFourZeroOneKReplicationComparison:
    """Test comparison with published results."""

    def test_status_match_when_within_tolerance(self):
        """The relative gap is within the wider descriptive cutoff."""
        replicator = FourZeroOneKReplication(random_state=42, tolerance=0.50)  # Very generous
        result = replicator.replicate_plr_rf()

        assert abs(result.rel_difference) <= 0.50
        assert result.status == "WITHIN_TOLERANCE"

    def test_status_mismatch_when_exceeds_tolerance(self):
        """Report a relative gap outside the narrower descriptive cutoff."""
        replicator = FourZeroOneKReplication(random_state=42, tolerance=0.01)  # Very strict
        result = replicator.replicate_plr_rf()

        assert abs(result.rel_difference) > 0.01
        assert result.status == "OUTSIDE_TOLERANCE"

    def test_difference_calculation(self):
        """Test that difference is calculated correctly."""
        replicator = FourZeroOneKReplication(random_state=42)
        result = replicator.replicate_plr_rf()

        # Difference = estimate - published
        expected_diff = result.ate_estimate - result.published_ate
        assert np.isclose(result.difference, expected_diff)

    def test_relative_difference_calculation(self):
        """Test that relative difference is calculated correctly."""
        replicator = FourZeroOneKReplication(random_state=42)
        result = replicator.replicate_plr_rf()

        # Relative difference = (estimate - published) / published
        expected_rel_diff = result.difference / result.published_ate
        assert np.isclose(result.rel_difference, expected_rel_diff)

    def test_same_sample_gap_has_no_p_value(self):
        """Inference for one estimate cannot test a difference from a same-sample estimate."""
        replicator = FourZeroOneKReplication(random_state=42)
        result = replicator.replicate_plr_rf()

        assert result.p_value is None


@pytest.mark.tier4
@pytest.mark.network
@requires_econml
class TestFourZeroOneKReplicationResultDataclass:
    """Test ReplicationResult dataclass."""

    def test_result_has_all_required_fields(self):
        """Test that ReplicationResult has all required fields."""
        replicator = FourZeroOneKReplication(random_state=42)
        result = replicator.replicate_plr_rf()

        # Check all fields exist
        assert hasattr(result, "method")
        assert hasattr(result, "ate_estimate")
        assert hasattr(result, "std_error")
        assert hasattr(result, "ci_lower")
        assert hasattr(result, "ci_upper")
        assert hasattr(result, "published_ate")
        assert hasattr(result, "difference")
        assert hasattr(result, "rel_difference")
        assert hasattr(result, "p_value")
        assert hasattr(result, "status")
        assert hasattr(result, "tolerance")
        assert hasattr(result, "timestamp")
        assert hasattr(result, "metadata")

    def test_result_timestamp_populated(self):
        """Test that timestamp is populated."""
        replicator = FourZeroOneKReplication(random_state=42)
        result = replicator.replicate_plr_rf()

        assert result.timestamp is not None
        from datetime import datetime

        assert isinstance(result.timestamp, datetime)


@pytest.mark.tier4
@pytest.mark.network
@requires_econml
class TestFourZeroOneKReplicationEdgeCases:
    """Test edge cases and robustness."""

    def test_different_treatment_variables(self):
        """Test replication with different treatment variables."""
        replicator = FourZeroOneKReplication(random_state=42)

        # Test with e401 (eligibility)
        result_e401 = replicator.replicate_plr_rf(treatment="e401")
        assert isinstance(result_e401, ReplicationResult)

        # Reset replicator for p401 (participation)
        replicator2 = FourZeroOneKReplication(random_state=42)
        result_p401 = replicator2.replicate_plr_rf(treatment="p401")
        assert isinstance(result_p401, ReplicationResult)

        assert result_p401.status == "NO_BENCHMARK"
        assert result_p401.published_ate is None
        assert result_p401.difference is None
        assert result_p401.rel_difference is None
        assert result_p401.p_value is None

        # Estimates should differ (different treatments)
        assert result_e401.ate_estimate != result_p401.ate_estimate

    def test_different_random_forests_settings(self):
        """Test replication with different RF hyperparameters."""
        # Small forest
        rep1 = FourZeroOneKReplication(random_state=42)
        result1 = rep1.replicate_plr_rf(n_estimators=100, max_depth=5)
        assert isinstance(result1, ReplicationResult)

        # Large forest
        rep2 = FourZeroOneKReplication(random_state=42)
        result2 = rep2.replicate_plr_rf(n_estimators=500, max_depth=None)
        assert isinstance(result2, ReplicationResult)

        # Both should produce valid estimates
        assert 4000 <= result1.ate_estimate <= 15000
        assert 4000 <= result2.ate_estimate <= 15000


@pytest.mark.tier4
@pytest.mark.network
@requires_econml
class TestFourZeroOneKReplicationIntegration:
    """Integration tests with realistic scenarios."""

    def test_full_replication_workflow(self):
        """Test complete replication workflow."""
        # Initialize
        replicator = FourZeroOneKReplication(random_state=42, tolerance=0.15)

        # Load data
        df = replicator.load_data()
        assert df.shape == (9915, 14)

        # Preprocess
        Y, T, X = replicator.preprocess_data()
        assert Y.shape == (9915,)

        # Replicate PLR RF
        result_rf = replicator.replicate_plr_rf()
        assert isinstance(result_rf, ReplicationResult)
        assert result_rf.method == "PLR_RF"

        # Reset for Lasso
        replicator2 = FourZeroOneKReplication(random_state=42, tolerance=0.15)

        # Replicate PLR Lasso
        result_lasso = replicator2.replicate_plr_lasso()
        assert isinstance(result_lasso, ReplicationResult)
        assert result_lasso.method == "PLR_Lasso"

        # Both should produce valid results
        assert result_rf.status in ["WITHIN_TOLERANCE", "OUTSIDE_TOLERANCE"]
        assert result_lasso.status in ["WITHIN_TOLERANCE", "OUTSIDE_TOLERANCE"]

    def test_comparison_across_methods(self):
        """Test that different methods produce different estimates."""
        rep_rf = FourZeroOneKReplication(random_state=42)
        result_rf = rep_rf.replicate_plr_rf()

        rep_lasso = FourZeroOneKReplication(random_state=42)
        result_lasso = rep_lasso.replicate_plr_lasso()

        # Methods should differ
        assert result_rf.method != result_lasso.method
        assert result_rf.published_ate != result_lasso.published_ate

        # Both should target published estimates
        assert result_rf.published_ate == 9247.0
        assert result_lasso.published_ate == 8187.0
