"""
Comprehensive tests for time series cross-validation.

Tests cover:
1. Basic splitting behavior
2. Gap and purge functionality
3. Expanding vs sliding windows
4. Blocked cross-validation
5. Purged/embargoed cross-validation
6. Edge cases and error handling
7. Comparison with sklearn TimeSeriesSplit
"""

import numpy as np
import pytest
from sklearn.model_selection import TimeSeriesSplit

from dml_ts.dml.cross_fitting import (
    TimeSeriesCrossValidator,
    BlockedTimeSeriesCV,
    PurgedGroupTimeSeriesCV,
    CVFold,
    create_time_series_cv,
)


class TestTimeSeriesCrossValidatorBasic:
    """Test basic TimeSeriesCrossValidator functionality."""

    @pytest.mark.tier1
    def test_init_default(self) -> None:
        """Test default initialization."""
        cv = TimeSeriesCrossValidator()
        assert cv.n_splits == 5
        assert cv.gap == 0
        assert cv.purge_length == 0
        assert cv.test_size is None
        assert cv.expanding is True

    @pytest.mark.tier1
    def test_init_custom(self) -> None:
        """Test initialization with custom parameters."""
        cv = TimeSeriesCrossValidator(
            n_splits=3,
            gap=10,
            purge_length=5,
            test_size=50,
            expanding=False,
        )
        assert cv.n_splits == 3
        assert cv.gap == 10
        assert cv.purge_length == 5
        assert cv.test_size == 50
        assert cv.expanding is False

    @pytest.mark.tier1
    def test_init_invalid_n_splits(self) -> None:
        """Test that invalid n_splits raises ValueError."""
        with pytest.raises(ValueError, match="n_splits must be >= 1"):
            TimeSeriesCrossValidator(n_splits=0)

    @pytest.mark.tier1
    def test_init_invalid_gap(self) -> None:
        """Test that negative gap raises ValueError."""
        with pytest.raises(ValueError, match="gap must be >= 0"):
            TimeSeriesCrossValidator(gap=-1)

    @pytest.mark.tier1
    def test_init_invalid_purge(self) -> None:
        """Test that negative purge_length raises ValueError."""
        with pytest.raises(ValueError, match="purge_length must be >= 0"):
            TimeSeriesCrossValidator(purge_length=-1)

    @pytest.mark.tier1
    def test_get_n_splits(self) -> None:
        """Test get_n_splits returns correct value."""
        cv = TimeSeriesCrossValidator(n_splits=7)
        assert cv.get_n_splits() == 7
        assert cv.get_n_splits(np.zeros(100)) == 7  # Args ignored

    @pytest.mark.tier1
    def test_basic_split(self) -> None:
        """Test basic split produces correct number of folds."""
        cv = TimeSeriesCrossValidator(n_splits=3)
        X = np.arange(100).reshape(-1, 1)

        folds = list(cv.split(X))
        assert len(folds) == 3

        for train_idx, test_idx in folds:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # Train should come before test
            assert train_idx[-1] < test_idx[0]


class TestTimeSeriesCrossValidatorTemporalOrder:
    """Test temporal ordering is respected."""

    @pytest.mark.tier1
    def test_train_before_test(self) -> None:
        """Test that all train indices are before all test indices."""
        cv = TimeSeriesCrossValidator(n_splits=5)
        X = np.arange(200).reshape(-1, 1)

        for train_idx, test_idx in cv.split(X):
            assert (
                train_idx.max() < test_idx.min()
            ), f"Train max ({train_idx.max()}) >= Test min ({test_idx.min()})"

    @pytest.mark.tier1
    def test_no_overlap(self) -> None:
        """Test train and test indices don't overlap."""
        cv = TimeSeriesCrossValidator(n_splits=5, gap=0)
        X = np.arange(200).reshape(-1, 1)

        for train_idx, test_idx in cv.split(X):
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0, f"Found overlapping indices: {overlap}"

    @pytest.mark.tier1
    def test_expanding_window(self) -> None:
        """Test expanding window grows with each fold."""
        cv = TimeSeriesCrossValidator(n_splits=3, expanding=True)
        X = np.arange(120).reshape(-1, 1)

        train_sizes = []
        for train_idx, _ in cv.split(X):
            train_sizes.append(len(train_idx))

        # Each fold should have more training data
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1], (
                f"Fold {i} train size ({train_sizes[i]}) not larger than "
                f"fold {i-1} ({train_sizes[i-1]})"
            )

    @pytest.mark.tier1
    def test_test_sets_sequential(self) -> None:
        """Test that test sets are sequential (later folds test later data)."""
        cv = TimeSeriesCrossValidator(n_splits=4)
        X = np.arange(200).reshape(-1, 1)

        test_starts = []
        for _, test_idx in cv.split(X):
            test_starts.append(test_idx[0])

        for i in range(1, len(test_starts)):
            assert test_starts[i] > test_starts[i - 1], (
                f"Fold {i} test start ({test_starts[i]}) not after "
                f"fold {i-1} ({test_starts[i-1]})"
            )


class TestTimeSeriesCrossValidatorGapAndPurge:
    """Test gap and purge functionality."""

    @pytest.mark.tier1
    def test_gap_creates_separation(self) -> None:
        """Test that gap creates separation between train and test."""
        gap = 10
        cv = TimeSeriesCrossValidator(n_splits=3, gap=gap)
        X = np.arange(150).reshape(-1, 1)

        for train_idx, test_idx in cv.split(X):
            actual_gap = test_idx[0] - train_idx[-1] - 1
            assert actual_gap >= gap, f"Actual gap ({actual_gap}) less than requested ({gap})"

    @pytest.mark.tier1
    def test_purge_removes_from_train(self) -> None:
        """Test that purge removes observations from train end."""
        purge_length = 5
        cv_no_purge = TimeSeriesCrossValidator(n_splits=3, purge_length=0)
        cv_with_purge = TimeSeriesCrossValidator(n_splits=3, purge_length=purge_length)
        X = np.arange(150).reshape(-1, 1)

        for (train_no, _), (train_with, _) in zip(cv_no_purge.split(X), cv_with_purge.split(X)):
            size_diff = len(train_no) - len(train_with)
            assert (
                size_diff == purge_length
            ), f"Purge removed {size_diff} samples, expected {purge_length}"

    @pytest.mark.tier1
    def test_combined_gap_and_purge(self) -> None:
        """Test gap and purge work together."""
        gap = 10
        purge = 5
        cv = TimeSeriesCrossValidator(n_splits=3, gap=gap, purge_length=purge)
        X = np.arange(200).reshape(-1, 1)

        for train_idx, test_idx in cv.split(X):
            # Total separation should be gap + purge
            total_sep = test_idx[0] - train_idx[-1] - 1
            assert (
                total_sep >= gap + purge
            ), f"Total separation ({total_sep}) less than gap+purge ({gap + purge})"


class TestTimeSeriesCrossValidatorSlidingWindow:
    """Test sliding window functionality."""

    @pytest.mark.tier1
    def test_sliding_window_constant_train_growth(self) -> None:
        """Test sliding window doesn't grow as fast as expanding."""
        cv_expanding = TimeSeriesCrossValidator(n_splits=3, expanding=True)
        cv_sliding = TimeSeriesCrossValidator(n_splits=3, expanding=False)
        X = np.arange(150).reshape(-1, 1)

        exp_sizes = [len(train) for train, _ in cv_expanding.split(X)]
        slid_sizes = [len(train) for train, _ in cv_sliding.split(X)]

        # Expanding should grow faster
        exp_growth = exp_sizes[-1] - exp_sizes[0]
        slid_growth = slid_sizes[-1] - slid_sizes[0]

        # Sliding should have less growth (or same)
        assert slid_growth <= exp_growth


class TestTimeSeriesCrossValidatorComparison:
    """Compare with sklearn's TimeSeriesSplit."""

    @pytest.mark.tier1
    def test_matches_sklearn_basic(self) -> None:
        """Test basic behavior matches sklearn TimeSeriesSplit."""
        n_splits = 5
        cv_ours = TimeSeriesCrossValidator(n_splits=n_splits, gap=0)
        cv_sklearn = TimeSeriesSplit(n_splits=n_splits)
        X = np.arange(100).reshape(-1, 1)

        ours_folds = list(cv_ours.split(X))
        sklearn_folds = list(cv_sklearn.split(X))

        assert len(ours_folds) == len(sklearn_folds)

        # Test sets should be similar in size
        for (_, test_ours), (_, test_sklearn) in zip(ours_folds, sklearn_folds):
            # Allow some difference due to different algorithms
            size_diff = abs(len(test_ours) - len(test_sklearn))
            assert size_diff < 10, f"Test size difference {size_diff} too large"


class TestTimeSeriesCrossValidatorEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.tier1
    def test_small_dataset(self) -> None:
        """Test with small dataset."""
        cv = TimeSeriesCrossValidator(n_splits=2)
        X = np.arange(20).reshape(-1, 1)

        folds = list(cv.split(X))
        assert len(folds) == 2

    @pytest.mark.tier1
    def test_dataset_too_small_raises(self) -> None:
        """Test that too-small dataset raises ValueError."""
        cv = TimeSeriesCrossValidator(n_splits=5, gap=50)
        X = np.arange(20).reshape(-1, 1)

        with pytest.raises(ValueError, match="Not enough samples"):
            list(cv.split(X))

    @pytest.mark.tier1
    def test_1d_array(self) -> None:
        """Test with 1D array input."""
        cv = TimeSeriesCrossValidator(n_splits=3)
        X = np.arange(100)  # 1D, not 2D

        folds = list(cv.split(X))
        assert len(folds) == 3

    @pytest.mark.tier1
    def test_custom_test_size(self) -> None:
        """Test with custom test_size."""
        test_size = 20
        cv = TimeSeriesCrossValidator(n_splits=3, test_size=test_size)
        X = np.arange(200).reshape(-1, 1)

        for _, test_idx in cv.split(X):
            assert len(test_idx) == test_size

    @pytest.mark.tier1
    def test_get_fold_info(self) -> None:
        """Test get_fold_info returns correct CVFold objects."""
        cv = TimeSeriesCrossValidator(n_splits=3)
        X = np.arange(100).reshape(-1, 1)

        folds = cv.get_fold_info(X)
        assert len(folds) == 3

        for i, fold in enumerate(folds):
            assert isinstance(fold, CVFold)
            assert fold.fold_number == i
            assert fold.train_start >= 0
            assert fold.train_end <= 100
            assert fold.test_start >= fold.train_end


class TestBlockedTimeSeriesCV:
    """Test BlockedTimeSeriesCV functionality."""

    @pytest.mark.tier1
    def test_init_default(self) -> None:
        """Test default initialization."""
        cv = BlockedTimeSeriesCV()
        assert cv.n_splits == 5
        assert cv.block_size is None
        assert cv.gap_blocks == 1

    @pytest.mark.tier1
    def test_init_invalid(self) -> None:
        """Test invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            BlockedTimeSeriesCV(n_splits=0)
        with pytest.raises(ValueError):
            BlockedTimeSeriesCV(gap_blocks=-1)

    @pytest.mark.tier1
    def test_basic_split(self) -> None:
        """Test basic blocked split."""
        cv = BlockedTimeSeriesCV(n_splits=3, block_size=20, gap_blocks=1)
        X = np.arange(200).reshape(-1, 1)

        folds = list(cv.split(X))
        assert len(folds) <= 3  # May have fewer if not enough blocks

        for train_idx, test_idx in folds:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # Train indices should be multiples of block_size
            assert len(train_idx) % 20 == 0 or train_idx[-1] == len(train_idx) - 1

    @pytest.mark.tier1
    def test_blocks_not_split(self) -> None:
        """Test that individual blocks are not split between train/test."""
        block_size = 10
        cv = BlockedTimeSeriesCV(n_splits=2, block_size=block_size, gap_blocks=0)
        X = np.arange(100).reshape(-1, 1)

        for train_idx, test_idx in cv.split(X):
            # Train should end at a block boundary
            if len(train_idx) > 0:
                assert (
                    train_idx[-1] % block_size == block_size - 1
                    or train_idx[-1] == len(train_idx) - 1
                )
            # Test should start at a block boundary
            if len(test_idx) > 0:
                assert test_idx[0] % block_size == 0

    @pytest.mark.tier1
    def test_gap_blocks(self) -> None:
        """Test gap_blocks creates block-level separation."""
        block_size = 10
        gap_blocks = 2
        cv = BlockedTimeSeriesCV(n_splits=2, block_size=block_size, gap_blocks=gap_blocks)
        X = np.arange(200).reshape(-1, 1)

        for train_idx, test_idx in cv.split(X):
            if len(train_idx) > 0 and len(test_idx) > 0:
                # Gap should be at least gap_blocks * block_size
                actual_gap = test_idx[0] - train_idx[-1] - 1
                expected_min_gap = gap_blocks * block_size
                assert (
                    actual_gap >= expected_min_gap
                ), f"Actual gap ({actual_gap}) less than expected ({expected_min_gap})"


class TestPurgedGroupTimeSeriesCV:
    """Test PurgedGroupTimeSeriesCV functionality."""

    @pytest.mark.tier1
    def test_init_default(self) -> None:
        """Test default initialization."""
        cv = PurgedGroupTimeSeriesCV()
        assert cv.n_splits == 5
        assert cv.embargo_pct == 0.01
        assert cv.purge_pct == 0.0

    @pytest.mark.tier1
    def test_init_invalid(self) -> None:
        """Test invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            PurgedGroupTimeSeriesCV(n_splits=0)
        with pytest.raises(ValueError):
            PurgedGroupTimeSeriesCV(embargo_pct=-0.1)
        with pytest.raises(ValueError):
            PurgedGroupTimeSeriesCV(embargo_pct=1.5)
        with pytest.raises(ValueError):
            PurgedGroupTimeSeriesCV(purge_pct=-0.1)

    @pytest.mark.tier1
    def test_basic_split(self) -> None:
        """Test basic purged split."""
        cv = PurgedGroupTimeSeriesCV(n_splits=3, embargo_pct=0.05, purge_pct=0.02)
        X = np.arange(1000).reshape(-1, 1)

        folds = list(cv.split(X))
        assert len(folds) == 3

        for train_idx, test_idx in folds:
            assert len(train_idx) > 0
            assert len(test_idx) > 0

    @pytest.mark.tier1
    def test_embargo_excludes_post_test(self) -> None:
        """Test embargo excludes samples after test set."""
        embargo_pct = 0.1
        cv = PurgedGroupTimeSeriesCV(n_splits=2, embargo_pct=embargo_pct, purge_pct=0.0)
        X = np.arange(100).reshape(-1, 1)

        for train_idx, test_idx in cv.split(X):
            test_end = test_idx[-1]
            # Samples immediately after test should not be in train
            embargo_size = int(100 * embargo_pct)
            embargo_range = np.arange(test_end + 1, min(test_end + 1 + embargo_size, 100))

            for idx in embargo_range:
                assert idx not in train_idx, f"Index {idx} in embargo range found in train"

    @pytest.mark.tier1
    def test_purge_excludes_pre_test(self) -> None:
        """Test purge excludes samples before test set."""
        purge_pct = 0.1
        cv = PurgedGroupTimeSeriesCV(n_splits=2, embargo_pct=0.0, purge_pct=purge_pct)
        X = np.arange(100).reshape(-1, 1)

        for train_idx, test_idx in cv.split(X):
            test_start = test_idx[0]
            # Samples immediately before test should not be in train
            purge_size = int(100 * purge_pct)
            purge_range = np.arange(max(0, test_start - purge_size), test_start)

            for idx in purge_range:
                assert idx not in train_idx, f"Index {idx} in purge range found in train"


class TestCreateTimeSeriesCV:
    """Test factory function."""

    @pytest.mark.tier1
    def test_expanding(self) -> None:
        """Test creating expanding window CV."""
        cv = create_time_series_cv("expanding", n_splits=3)
        assert isinstance(cv, TimeSeriesCrossValidator)
        assert cv.expanding is True

    @pytest.mark.tier1
    def test_time_series_split(self) -> None:
        """Test 'time_series_split' alias."""
        cv = create_time_series_cv("time_series_split", n_splits=3)
        assert isinstance(cv, TimeSeriesCrossValidator)
        assert cv.expanding is True

    @pytest.mark.tier1
    def test_sliding(self) -> None:
        """Test creating sliding window CV."""
        cv = create_time_series_cv("sliding", n_splits=3)
        assert isinstance(cv, TimeSeriesCrossValidator)
        assert cv.expanding is False

    @pytest.mark.tier1
    def test_blocked(self) -> None:
        """Test creating blocked CV."""
        cv = create_time_series_cv("blocked", n_splits=3, block_size=10)
        assert isinstance(cv, BlockedTimeSeriesCV)
        assert cv.block_size == 10

    @pytest.mark.tier1
    def test_purged(self) -> None:
        """Test creating purged CV."""
        cv = create_time_series_cv("purged", n_splits=3, embargo_pct=0.05)
        assert isinstance(cv, PurgedGroupTimeSeriesCV)
        assert cv.embargo_pct == 0.05

    @pytest.mark.tier1
    def test_invalid_strategy(self) -> None:
        """Test invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            create_time_series_cv("invalid_strategy")

    @pytest.mark.tier1
    def test_kwargs_passed(self) -> None:
        """Test kwargs are passed to CV class."""
        cv = create_time_series_cv("expanding", n_splits=5, gap=10, purge_length=5)
        assert cv.gap == 10
        assert cv.purge_length == 5


class TestIntegration:
    """Integration tests with realistic scenarios."""

    @pytest.mark.tier1
    def test_with_actual_model_training(self) -> None:
        """Test CV can be used for actual model training."""
        from sklearn.linear_model import Ridge

        # Generate synthetic time series data
        np.random.seed(42)
        n_samples = 200
        X = np.random.randn(n_samples, 5)
        y = X @ np.array([1, 2, 3, 4, 5]) + np.random.randn(n_samples) * 0.1

        cv = TimeSeriesCrossValidator(n_splits=3, gap=5)
        scores = []

        for train_idx, test_idx in cv.split(X):
            model = Ridge()
            model.fit(X[train_idx], y[train_idx])
            score = model.score(X[test_idx], y[test_idx])
            scores.append(score)

        # All scores should be reasonable
        assert all(s > 0.5 for s in scores), f"Scores too low: {scores}"

    @pytest.mark.tier1
    def test_sklearn_cross_val_score_compatible(self) -> None:
        """Test CV is compatible with sklearn cross_val_score."""
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import Ridge

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X.sum(axis=1) + np.random.randn(100) * 0.1

        cv = TimeSeriesCrossValidator(n_splits=3)
        scores = cross_val_score(Ridge(), X, y, cv=cv)

        assert len(scores) == 3
        assert all(np.isfinite(scores))
