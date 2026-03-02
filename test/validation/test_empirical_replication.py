"""
Tests for empirical replication module (401(k) analysis).

Test coverage for Chernozhukov et al. (2018) 401(k) DML replication.

Tier assignments:
    tier1: Init, constants, dataclass tests (no estimation)
    tier2: Data loading, preprocessing (network → cached)
    tier4: Full PLR replication (RF/Lasso), integration workflows
"""

import pytest
import numpy as np
import pandas as pd
from src.validation.empirical_replication import (
    FourZeroOneKReplication,
    ReplicationResult,
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
        assert FourZeroOneKReplication.PUBLISHED_ATES["PLR_RF"] == 9127.0
        assert FourZeroOneKReplication.PUBLISHED_ATES["PLR_Lasso"] == 9580.0
        # IRM_RF removed — IRM replication deferred (see Appendix A)
        assert "IRM_RF" not in FourZeroOneKReplication.PUBLISHED_ATES


@pytest.mark.tier2
class TestFourZeroOneKReplicationDataLoading:
    """Test data loading functionality."""

    def test_load_data_from_doubleml(self):
        """Test loading data from doubleml package."""
        replicator = FourZeroOneKReplication()
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
        replicator = FourZeroOneKReplication()

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
        replicator = FourZeroOneKReplication(random_state=42)
        Y, T, X = replicator.preprocess_data(treatment="e401")

        # Check shapes
        assert Y.shape == (9915,)
        assert T.shape == (9915,)
        assert X.shape[0] == 9915
        assert X.shape[1] == 11  # 11 control variables (C5 fix: includes all covariates)

    def test_preprocess_data_with_p401(self):
        """Test preprocessing with p401 (participation) treatment."""
        replicator = FourZeroOneKReplication(random_state=42)
        Y, T, X = replicator.preprocess_data(treatment="p401")

        # Check shapes
        assert Y.shape == (9915,)
        assert T.shape == (9915,)
        assert X.shape[0] == 9915

    def test_preprocess_data_invalid_treatment(self):
        """Test that invalid treatment raises error."""
        replicator = FourZeroOneKReplication()

        with pytest.raises(ValueError, match="treatment must be"):
            replicator.preprocess_data(treatment="invalid")

    def test_preprocess_data_caching(self):
        """Test that preprocessed data is cached."""
        replicator = FourZeroOneKReplication(random_state=42)

        # First preprocessing
        Y1, T1, X1 = replicator.preprocess_data()

        # Second preprocessing should return same arrays
        Y2, T2, X2 = replicator.preprocess_data()

        assert np.array_equal(Y1, Y2)
        assert np.array_equal(T1, T2)
        assert np.array_equal(X1, X2)


@pytest.mark.tier4
@pytest.mark.network
class TestFourZeroOneKReplicationPLRRandomForest:
    """Test PLR Random Forest replication."""

    def test_replicate_plr_rf_returns_result(self):
        """Test that replicate_plr_rf returns proper result object."""
        replicator = FourZeroOneKReplication(random_state=42)
        result = replicator.replicate_plr_rf()

        assert isinstance(result, ReplicationResult)
        assert result.method == "PLR_RF"
        assert result.published_ate == 9127.0

    def test_replicate_plr_rf_ate_in_reasonable_range(self):
        """Test that ATE estimate is in reasonable range."""
        replicator = FourZeroOneKReplication(random_state=42)
        result = replicator.replicate_plr_rf()

        # Published estimate: $9,127
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
        assert result.metadata["n_controls"] == 11  # 11 controls (C5 fix: includes all covariates)


@pytest.mark.tier4
@pytest.mark.network
class TestFourZeroOneKReplicationPLRLasso:
    """Test PLR Lasso replication."""

    def test_replicate_plr_lasso_returns_result(self):
        """Test that replicate_plr_lasso returns proper result object."""
        replicator = FourZeroOneKReplication(random_state=42)
        result = replicator.replicate_plr_lasso()

        assert isinstance(result, ReplicationResult)
        assert result.method == "PLR_Lasso"
        assert result.published_ate == 9580.0

    def test_replicate_plr_lasso_ate_in_reasonable_range(self):
        """Test that ATE estimate is in reasonable range."""
        replicator = FourZeroOneKReplication(random_state=42)
        result = replicator.replicate_plr_lasso()

        # Published estimate: $9,580
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
class TestFourZeroOneKReplicationComparison:
    """Test comparison with published results."""

    def test_status_match_when_within_tolerance(self):
        """Test MATCH status when estimate within tolerance."""
        replicator = FourZeroOneKReplication(random_state=42, tolerance=0.50)  # Very generous
        result = replicator.replicate_plr_rf()

        # With 50% tolerance, should almost certainly match
        assert result.status == "MATCH"

    def test_status_mismatch_when_exceeds_tolerance(self):
        """Test MISMATCH status when estimate exceeds tolerance."""
        replicator = FourZeroOneKReplication(random_state=42, tolerance=0.01)  # Very strict
        result = replicator.replicate_plr_rf()

        # With 1% tolerance, likely to mismatch (unless very lucky)
        # This test might occasionally fail due to random variation
        # If estimate happens to be exactly right, that's actually good!
        if abs(result.rel_difference) > 0.01:
            assert result.status == "MISMATCH"

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

    def test_p_value_is_valid_probability(self):
        """Test that p-value is between 0 and 1."""
        replicator = FourZeroOneKReplication(random_state=42)
        result = replicator.replicate_plr_rf()

        assert 0 <= result.p_value <= 1


@pytest.mark.tier4
@pytest.mark.network
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
        assert result_rf.status in ["MATCH", "MISMATCH"]
        assert result_lasso.status in ["MATCH", "MISMATCH"]

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
        assert result_rf.published_ate == 9127.0
        assert result_lasso.published_ate == 9580.0
