"""Test suite for OJ dataset loader.

Tests the OJDataLoader class to ensure correct data loading,
preprocessing, and validation for DML price elasticity analysis.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from src.data.oj_loader import OJDataset, OJDataLoader


@pytest.mark.tier2
class TestOJDataLoader:
    """Test suite for OJDataLoader."""

    @pytest.fixture(scope="class")
    def loader(self) -> OJDataLoader:
        """Create a loader instance."""
        return OJDataLoader()

    @pytest.fixture(scope="class")
    def dataset(self, loader: OJDataLoader) -> OJDataset:
        """Load dataset once for all tests in class."""
        return loader.load()

    @pytest.mark.tier1
    def test_load_returns_oj_dataset(self, dataset: OJDataset) -> None:
        """Test that load() returns an OJDataset instance."""
        assert isinstance(dataset, OJDataset)

    @pytest.mark.tier1
    def test_dataset_shapes_consistent(self, dataset: OJDataset) -> None:
        """Test that Y, T, X arrays have consistent shapes."""
        n = dataset.n_samples

        # All arrays should have n samples
        assert dataset.Y.shape == (n,), f"Y shape {dataset.Y.shape} != ({n},)"
        assert dataset.T.shape == (n,), f"T shape {dataset.T.shape} != ({n},)"
        assert dataset.X.shape[0] == n, f"X rows {dataset.X.shape[0]} != {n}"

        # Feature dimension should match feature_names
        assert dataset.X.shape[1] == len(dataset.feature_names)
        assert dataset.n_features == len(dataset.feature_names)

    @pytest.mark.tier1
    def test_default_feature_names(self, dataset: OJDataset) -> None:
        """Test that default features are feat, INCOME, AGE60."""
        expected = ["feat", "INCOME", "AGE60"]
        assert dataset.feature_names == expected

    @pytest.mark.tier1
    def test_no_missing_values(self, dataset: OJDataset) -> None:
        """Test that dataset has no NaN values after preprocessing."""
        assert not np.any(np.isnan(dataset.Y)), "Y contains NaN"
        assert not np.any(np.isnan(dataset.T)), "T contains NaN"
        assert not np.any(np.isnan(dataset.X)), "X contains NaN"

    @pytest.mark.tier1
    def test_reasonable_sample_size(self, dataset: OJDataset) -> None:
        """Test that dataset has expected ~28k observations."""
        # The full OJ dataset has ~28,947 observations
        assert dataset.n_samples > 20_000, f"Too few samples: {dataset.n_samples}"
        assert dataset.n_samples < 50_000, f"Too many samples: {dataset.n_samples}"

    @pytest.mark.tier1
    def test_treatment_is_log_price(self, dataset: OJDataset) -> None:
        """Test that treatment T is log(price), not raw price."""
        # Log prices should typically be in range (0, 3) for dollar amounts
        assert dataset.T.min() > -1.0, f"T min {dataset.T.min()} too low for log price"
        assert dataset.T.max() < 3.0, f"T max {dataset.T.max()} too high for log price"

    @pytest.mark.tier1
    def test_custom_features(self) -> None:
        """Test loading with custom feature set."""
        features = ["feat", "INCOME", "AGE60", "EDUC"]
        loader = OJDataLoader(features=features)
        data = loader.load()

        assert data.n_features == 4
        assert data.feature_names == features

    @pytest.mark.tier1
    def test_brand_filter(self) -> None:
        """Test filtering to single brand."""
        loader = OJDataLoader(brand="tropicana")
        data = loader.load()

        # Should have fewer samples than full dataset
        assert data.n_samples < 15_000

        # Raw df should only contain tropicana
        assert data.raw_df is not None
        assert all(data.raw_df["brand"] == "tropicana")

    @pytest.mark.tier1
    def test_invalid_feature_raises(self) -> None:
        """Test that invalid feature names raise ValueError."""
        loader = OJDataLoader(features=["nonexistent_feature"])

        with pytest.raises(ValueError, match="Features not in dataset"):
            loader.load()

    @pytest.mark.tier1
    def test_dataset_summary(self, dataset: OJDataset) -> None:
        """Test that summary() returns formatted string."""
        summary = dataset.summary()

        assert "Orange Juice Dataset Summary" in summary
        # n_samples is formatted with commas (28,947)
        assert f"{dataset.n_samples:,}" in summary
        assert "INCOME" in summary

    @pytest.mark.tier1
    def test_dataset_repr(self, dataset: OJDataset) -> None:
        """Test __repr__ method."""
        repr_str = repr(dataset)

        assert "OJDataset" in repr_str
        assert "n_samples" in repr_str
        assert "n_features" in repr_str

    @pytest.mark.tier2
    def test_caching(self) -> None:
        """Test that caching works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # First load - downloads
            loader1 = OJDataLoader(cache_dir=cache_dir)
            data1 = loader1.load()

            # Cache file should exist
            cache_file = cache_dir / "oj_large.csv"
            assert cache_file.exists()

            # Second load - from cache
            loader2 = OJDataLoader(cache_dir=cache_dir)
            data2 = loader2.load()

            # Should be identical
            np.testing.assert_array_equal(data1.Y, data2.Y)
            np.testing.assert_array_equal(data1.T, data2.T)
            np.testing.assert_array_equal(data1.X, data2.X)

    @pytest.mark.tier1
    def test_get_available_features(self, loader: OJDataLoader) -> None:
        """Test getting list of available features."""
        features = loader.get_available_features()

        # Should include key columns
        assert "logmove" in features
        assert "price" in features
        assert "INCOME" in features
        assert "brand" in features

    @pytest.mark.tier1
    def test_get_brand_options(self, loader: OJDataLoader) -> None:
        """Test getting list of brand options."""
        brands = loader.get_brand_options()

        assert len(brands) == 3
        assert "tropicana" in brands
        assert "dominicks" in brands
        assert "minute.maid" in brands
