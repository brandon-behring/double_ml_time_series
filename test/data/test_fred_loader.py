"""Tests for FRED data loader.

Tests cover:
1. Synthetic data generation (no API required)
2. Data structure and metadata
3. Frequency conversion
4. Integration with DML
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from dml_ts.data.fred_loader import (
    FREDLoader,
    FREDSeries,
    MacroControlsResult,
    MACRO_CONTROL_SETS,
    STANDARD_MACRO_SERIES,
    create_synthetic_fred_data,
)

pytestmark = pytest.mark.tier2


# ============================================================================
# Synthetic Data Tests (No API Required)
# ============================================================================


class TestSyntheticFREDData:
    """Tests for synthetic FRED data generation."""

    def test_create_synthetic_data_basic(self):
        """Test basic synthetic data creation."""
        result = create_synthetic_fred_data(
            start_date="2015-01-01",
            end_date="2020-12-31",
            seed=42,
        )

        assert isinstance(result, MacroControlsResult)
        assert isinstance(result.data, pd.DataFrame)

    def test_synthetic_data_columns(self):
        """Test that synthetic data has expected columns."""
        result = create_synthetic_fred_data(seed=42)

        expected_cols = ["GDPC1", "CPIAUCSL", "UNRATE", "FEDFUNDS"]
        assert list(result.data.columns) == expected_cols

    def test_synthetic_data_shape(self):
        """Test synthetic data has correct shape."""
        result = create_synthetic_fred_data(
            start_date="2020-01-01",
            end_date="2020-12-31",
            frequency="M",
            seed=42,
        )

        # 12 months in 2020
        assert result.data.shape[0] == 12
        assert result.data.shape[1] == 4

    def test_synthetic_data_reproducibility(self):
        """Test that same seed produces same data."""
        result1 = create_synthetic_fred_data(seed=42)
        result2 = create_synthetic_fred_data(seed=42)

        pd.testing.assert_frame_equal(result1.data, result2.data)

    def test_synthetic_data_different_seeds(self):
        """Test that different seeds produce different data."""
        result1 = create_synthetic_fred_data(seed=42)
        result2 = create_synthetic_fred_data(seed=123)

        assert not result1.data.equals(result2.data)

    def test_synthetic_data_quarterly(self):
        """Test synthetic data at quarterly frequency."""
        result = create_synthetic_fred_data(
            start_date="2015-01-01",
            end_date="2020-12-31",
            frequency="Q",
            seed=42,
        )

        # 6 years * 4 quarters = 24
        assert result.data.shape[0] == 24

    def test_synthetic_data_no_nans(self):
        """Test that synthetic data has no missing values."""
        result = create_synthetic_fred_data(seed=42)

        assert not result.data.isna().any().any()

    def test_synthetic_data_metadata(self):
        """Test that metadata is populated."""
        result = create_synthetic_fred_data(seed=42)

        assert len(result.metadata) == 4
        assert "GDPC1" in result.metadata
        assert "name" in result.metadata["GDPC1"]

    def test_synthetic_data_missing_pct(self):
        """Test that missing_pct is all zeros for synthetic data."""
        result = create_synthetic_fred_data(seed=42)

        for series_id, pct in result.missing_pct.items():
            assert pct == 0.0

    def test_synthetic_gdp_reasonable_range(self):
        """Test that synthetic GDP growth is in reasonable range."""
        result = create_synthetic_fred_data(seed=42)

        gdp = result.data["GDPC1"]
        assert gdp.mean() > 0  # Positive average growth
        assert gdp.mean() < 10  # Not crazy high
        assert gdp.std() > 0  # Has variation

    def test_synthetic_unemployment_bounded(self):
        """Test that unemployment is bounded to reasonable range."""
        result = create_synthetic_fred_data(seed=42)

        unrate = result.data["UNRATE"]
        assert unrate.min() >= 2.0
        assert unrate.max() <= 15.0

    def test_synthetic_fed_rate_bounded(self):
        """Test that fed funds rate is non-negative and bounded."""
        result = create_synthetic_fred_data(seed=42)

        fedrate = result.data["FEDFUNDS"]
        assert fedrate.min() >= 0.0
        assert fedrate.max() <= 10.0


# ============================================================================
# FRED Loader Configuration Tests
# ============================================================================


class TestFREDLoaderConfig:
    """Tests for FREDLoader configuration."""

    def test_standard_macro_series_defined(self):
        """Test that standard series dictionary is populated."""
        assert len(STANDARD_MACRO_SERIES) > 0
        assert "GDP" in STANDARD_MACRO_SERIES
        assert "CPIAUCSL" in STANDARD_MACRO_SERIES

    def test_macro_control_sets_defined(self):
        """Test that control sets are defined."""
        assert "basic" in MACRO_CONTROL_SETS
        assert "comprehensive" in MACRO_CONTROL_SETS
        assert "financial" in MACRO_CONTROL_SETS

    def test_basic_set_has_required_series(self):
        """Test basic control set has essential series."""
        basic = MACRO_CONTROL_SETS["basic"]
        assert "GDPC1" in basic or "GDP" in basic
        assert "CPIAUCSL" in basic
        assert "UNRATE" in basic

    def test_loader_init_no_explicit_api_key(self):
        """Test loader initializes and may use env var."""
        import os

        # Save original env var if exists
        original_key = os.environ.get("FRED_API_KEY")

        # Test with env var cleared
        if "FRED_API_KEY" in os.environ:
            del os.environ["FRED_API_KEY"]

        try:
            loader = FREDLoader(api_key=None)
            assert loader.api_key is None
        finally:
            # Restore original env var
            if original_key is not None:
                os.environ["FRED_API_KEY"] = original_key

    def test_loader_cache_dir_created(self, tmp_path):
        """Test loader creates cache directory."""
        cache_dir = tmp_path / "fred_cache"
        loader = FREDLoader(cache_dir=cache_dir)

        assert loader.cache_dir.exists()

    def test_list_available_series(self):
        """Test listing available series."""
        series = FREDLoader.list_available_series()

        assert isinstance(series, dict)
        assert len(series) > 0
        assert "GDP" in series

    def test_list_control_sets(self):
        """Test listing control sets."""
        sets = FREDLoader.list_control_sets()

        assert isinstance(sets, dict)
        assert "basic" in sets


# ============================================================================
# MacroControlsResult Tests
# ============================================================================


class TestMacroControlsResult:
    """Tests for MacroControlsResult dataclass."""

    def test_result_attributes(self):
        """Test that result has all expected attributes."""
        result = create_synthetic_fred_data(seed=42)

        assert hasattr(result, "data")
        assert hasattr(result, "metadata")
        assert hasattr(result, "start_date")
        assert hasattr(result, "end_date")
        assert hasattr(result, "frequency")
        assert hasattr(result, "missing_pct")

    def test_result_data_is_dataframe(self):
        """Test that data is a DataFrame."""
        result = create_synthetic_fred_data(seed=42)

        assert isinstance(result.data, pd.DataFrame)

    def test_result_metadata_is_dict(self):
        """Test that metadata is a dictionary."""
        result = create_synthetic_fred_data(seed=42)

        assert isinstance(result.metadata, dict)


# ============================================================================
# Integration Tests
# ============================================================================


class TestFREDDMLIntegration:
    """Tests for integration between FRED data and DML."""

    def test_synthetic_data_usable_as_controls(self):
        """Test that synthetic data can be used as DML controls."""
        result = create_synthetic_fred_data(
            start_date="2015-01-01",
            end_date="2020-12-31",
            seed=42,
        )

        # Convert to numpy for DML
        X_macro = result.data.values

        assert X_macro.shape[0] > 0
        assert X_macro.shape[1] == 4
        assert not np.isnan(X_macro).any()

    def test_synthetic_data_with_temporal_plr_dml(self):
        """Test synthetic FRED data with TemporalPLRDML."""
        from dml_ts.dml import TemporalPLRDML

        # Get synthetic macro data
        fred_result = create_synthetic_fred_data(
            start_date="2015-01-01",
            end_date="2020-12-31",
            seed=42,
        )

        n = len(fred_result.data)
        X_macro = fred_result.data.values

        # Generate synthetic outcome and treatment
        rng = np.random.default_rng(42)
        T = 0.5 * X_macro[:, 0] + rng.standard_normal(n)  # Correlated with GDP
        Y = 2.0 * T + X_macro[:, 2] + rng.standard_normal(n)  # Effect + unemployment

        # Fit TemporalPLRDML
        model = TemporalPLRDML(
            n_lags=1,
            model_y="ridge",
            model_t="ridge",
            random_state=42,
        )
        result = model.fit(Y, T, X_macro)

        # Should estimate effect close to 2.0
        assert abs(result.theta - 2.0) < 1.0  # Within 1.0 of true effect

    def test_date_alignment(self):
        """Test that date index is properly aligned."""
        result = create_synthetic_fred_data(
            start_date="2015-01-01",
            end_date="2015-12-31",
            frequency="M",
            seed=42,
        )

        # Check date index
        assert result.data.index[0].year == 2015
        assert result.data.index[0].month == 1
        assert result.data.index[-1].year == 2015
        assert result.data.index[-1].month == 12


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_short_date_range(self):
        """Test with very short date range."""
        result = create_synthetic_fred_data(
            start_date="2020-01-01",
            end_date="2020-03-31",
            frequency="M",
            seed=42,
        )

        assert result.data.shape[0] == 3

    def test_annual_frequency(self):
        """Test annual frequency."""
        result = create_synthetic_fred_data(
            start_date="2015-01-01",
            end_date="2020-12-31",
            frequency="A",
            seed=42,
        )

        # 6 years
        assert result.data.shape[0] == 6

    def test_weekly_frequency(self):
        """Test weekly frequency."""
        result = create_synthetic_fred_data(
            start_date="2020-01-01",
            end_date="2020-03-31",
            frequency="W",
            seed=42,
        )

        # Approximately 13 weeks in Q1
        assert 12 <= result.data.shape[0] <= 14
