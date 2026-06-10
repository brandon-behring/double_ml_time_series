"""
Time Series Cross-Validation for Double Machine Learning.

Implements cross-fitting strategies that respect temporal ordering:
1. TimeSeriesCrossValidator - Expanding/sliding window with gap and purging
2. BlockedTimeSeriesCV - Blocked cross-validation with gaps
3. PurgedGroupTimeSeriesCV - Purged cross-validation to prevent leakage

These are essential for valid inference in time series DML where
standard K-Fold would violate the temporal structure.

References:

- Bergmeir, C., Hyndman, R. J., & Koo, B. (2018). A note on the validity
  of cross-validation for evaluating autoregressive time series prediction.
- de Prado, M. L. (2018). Advances in Financial Machine Learning. Chapter 7.

Usage:
    >>> from dml_ts.dml.cross_fitting import TimeSeriesCrossValidator
    >>> cv = TimeSeriesCrossValidator(n_splits=5, gap=10, purge_length=5)
    >>> for train_idx, test_idx in cv.split(X):
    ...     model.fit(X[train_idx], Y[train_idx])
    ...     pred = model.predict(X[test_idx])
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import BaseCrossValidator

ArrayLike = np.ndarray | list[float]


@dataclass
class CVFold:
    """Container for a single cross-validation fold.

    Attributes:
        train_idx: Indices for training set
        test_idx: Indices for test set
        fold_number: Fold index (0-indexed)
        train_start: Start index of training window
        train_end: End index of training window
        test_start: Start index of test window
        test_end: End index of test window
    """

    train_idx: np.ndarray
    test_idx: np.ndarray
    fold_number: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


class TimeSeriesCrossValidator(BaseCrossValidator):
    """Time series cross-validator with purging and gap.

    Implements time-series aware cross-validation that:
    1. Respects temporal ordering (train always before test)
    2. Includes gap between train and test to prevent leakage
    3. Optionally purges observations near the boundary
    4. Supports expanding or sliding window

    The split structure for expanding window with n_splits=3:

    ```
    Fold 0: [===TRAIN===]--gap--[TEST]................
    Fold 1: [======TRAIN======]--gap--[TEST].........
    Fold 2: [=========TRAIN=========]--gap--[TEST]...
    ```

    For sliding window:

    ```
    Fold 0: [===TRAIN===]--gap--[TEST]................
    Fold 1: ...[===TRAIN===]--gap--[TEST].............
    Fold 2: ......[===TRAIN===]--gap--[TEST]..........
    ```

    Args:
        n_splits: Number of CV folds (default 5)
        gap: Number of periods between train end and test start (default 0)
        purge_length: Periods to remove from train end (default 0)
        test_size: Size of test set (None for automatic sizing)
        expanding: Use expanding (True) or sliding (False) window (default True)
        min_train_size: Minimum training set size (None for n_samples // (n_splits + 1))

    Example:
        >>> cv = TimeSeriesCrossValidator(n_splits=5, gap=10, purge_length=5)
        >>> X = np.random.randn(1000, 10)
        >>> for train, test in cv.split(X):
        ...     print(f"Train: {len(train)}, Test: {len(test)}")
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        purge_length: int = 0,
        test_size: int | None = None,
        expanding: bool = True,
        min_train_size: int | None = None,
    ):
        """Initialize TimeSeriesCrossValidator.

        Args:
            n_splits: Number of cross-validation folds
            gap: Periods between train end and test start
            purge_length: Periods to remove from train end
            test_size: Size of test set (None for automatic)
            expanding: Use expanding (True) or sliding (False) window
            min_train_size: Minimum training set size

        Raises:
            ValueError: If n_splits < 1 or gap/purge_length < 0
        """
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")
        if gap < 0:
            raise ValueError(f"gap must be >= 0, got {gap}")
        if purge_length < 0:
            raise ValueError(f"purge_length must be >= 0, got {purge_length}")

        self.n_splits = n_splits
        self.gap = gap
        self.purge_length = purge_length
        self.test_size = test_size
        self.expanding = expanding
        self.min_train_size = min_train_size

    def split(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
        time_index: ArrayLike | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices for each fold.

        Args:
            X: Features array (n_samples, n_features) or (n_samples,)
            y: Target array (unused, for sklearn compatibility)
            groups: Group labels (unused, for sklearn compatibility)
            time_index: Time index for each observation (unused in basic impl)

        Yields:
            Tuple of (train_indices, test_indices) for each fold

        Raises:
            ValueError: If dataset too small for requested splits

        Example:
            >>> cv = TimeSeriesCrossValidator(n_splits=3, gap=5)
            >>> X = np.arange(100).reshape(-1, 1)
            >>> for train, test in cv.split(X):
            ...     print(f"Train: {train[0]}-{train[-1]}, Test: {test[0]}-{test[-1]}")
        """
        X_arr = np.asarray(X)
        n_samples = X_arr.shape[0]

        # Calculate test size if not specified
        if self.test_size is not None:
            test_size = self.test_size
        else:
            # Default: divide remaining samples equally among folds
            test_size = n_samples // (self.n_splits + 1)

        # Calculate minimum train size
        if self.min_train_size is not None:
            min_train_size = self.min_train_size
        else:
            min_train_size = n_samples // (self.n_splits + 1)

        # Validate we have enough samples
        min_required = min_train_size + self.gap + self.purge_length + test_size
        if n_samples < min_required:
            raise ValueError(
                f"Not enough samples ({n_samples}) for requested configuration. "
                f"Need at least {min_required} samples "
                f"(min_train={min_train_size}, gap={self.gap}, "
                f"purge={self.purge_length}, test_size={test_size})"
            )

        # Generate folds
        for fold_idx in range(self.n_splits):
            train_idx, test_idx = self._get_fold_indices(
                n_samples=n_samples,
                fold_idx=fold_idx,
                test_size=test_size,
                min_train_size=min_train_size,
            )
            yield train_idx, test_idx

    def _get_fold_indices(
        self,
        n_samples: int,
        fold_idx: int,
        test_size: int,
        min_train_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate train/test indices for a single fold.

        Args:
            n_samples: Total number of samples
            fold_idx: Index of current fold (0-indexed)
            test_size: Size of test set
            min_train_size: Minimum training set size

        Returns:
            Tuple of (train_indices, test_indices)
        """
        # Calculate test set boundaries
        # Test sets are at the end, working backwards for each fold
        # Fold n_splits-1 gets the last test set, fold 0 gets the earliest
        test_end = n_samples - (self.n_splits - 1 - fold_idx) * test_size
        test_start = test_end - test_size

        # Calculate train set boundaries
        # Train ends at gap + purge_length before test starts
        train_end_raw = test_start - self.gap
        train_end = train_end_raw - self.purge_length  # Purge removes from train end

        if self.expanding:
            # Expanding window: train always starts at 0
            train_start = 0
        else:
            # Sliding window: fixed training window size
            window_size = min_train_size + fold_idx * test_size
            train_start = max(0, train_end - window_size)

        # Ensure minimum train size
        if train_end - train_start < min_train_size:
            train_start = max(0, train_end - min_train_size)

        # Create index arrays
        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(test_start, test_end)

        return train_idx, test_idx

    def get_n_splits(
        self,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> int:
        """Return the number of splitting iterations.

        Args:
            X: Features (unused)
            y: Target (unused)
            groups: Groups (unused)

        Returns:
            Number of folds
        """
        return self.n_splits

    def get_fold_info(
        self,
        X: ArrayLike,
    ) -> list[CVFold]:
        """Get detailed information about all folds.

        Args:
            X: Features array

        Returns:
            List of CVFold objects with detailed fold information

        Example:
            >>> cv = TimeSeriesCrossValidator(n_splits=3)
            >>> X = np.arange(100).reshape(-1, 1)
            >>> folds = cv.get_fold_info(X)
            >>> for f in folds:
            ...     print(f"Fold {f.fold_number}: train {f.train_start}-{f.train_end}")
        """
        folds = []
        for fold_idx, (train_idx, test_idx) in enumerate(self.split(X)):
            fold = CVFold(
                train_idx=train_idx,
                test_idx=test_idx,
                fold_number=fold_idx,
                train_start=int(train_idx[0]) if len(train_idx) > 0 else 0,
                train_end=int(train_idx[-1]) + 1 if len(train_idx) > 0 else 0,
                test_start=int(test_idx[0]) if len(test_idx) > 0 else 0,
                test_end=int(test_idx[-1]) + 1 if len(test_idx) > 0 else 0,
            )
            folds.append(fold)
        return folds


class BlockedTimeSeriesCV(BaseCrossValidator):
    """Blocked time series cross-validation.

    Groups observations into blocks and ensures:
    1. No data leakage between blocks
    2. Entire blocks are in train or test (not split)
    3. Temporal ordering preserved

    Useful for data with grouped observations (e.g., weekly data grouped by month,
    or when autocorrelation extends across multiple periods).

    The split structure for n_splits=3 with block_size=10:

    ```
    Fold 0: [B0][B1][B2]--gap--[B3][B4]..............
    Fold 1: [B0][B1][B2][B3][B4]--gap--[B5][B6]......
    Fold 2: [B0][B1][B2][B3][B4][B5][B6]--gap--[B7][B8]
    ```

    Args:
        n_splits: Number of CV folds (default 5)
        block_size: Size of each block (None for automatic)
        gap_blocks: Number of blocks between train and test (default 1)
        test_blocks: Number of blocks in test set (None for automatic)

    Example:
        >>> cv = BlockedTimeSeriesCV(n_splits=3, block_size=20, gap_blocks=1)
        >>> X = np.random.randn(200, 10)
        >>> for train, test in cv.split(X):
        ...     print(f"Train blocks: {len(train)//20}, Test blocks: {len(test)//20}")
    """

    def __init__(
        self,
        n_splits: int = 5,
        block_size: int | None = None,
        gap_blocks: int = 1,
        test_blocks: int | None = None,
    ):
        """Initialize BlockedTimeSeriesCV.

        Args:
            n_splits: Number of cross-validation folds
            block_size: Size of each block (None for automatic)
            gap_blocks: Number of blocks between train and test
            test_blocks: Number of blocks in test set (None for automatic)

        Raises:
            ValueError: If n_splits < 1 or gap_blocks < 0
        """
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")
        if gap_blocks < 0:
            raise ValueError(f"gap_blocks must be >= 0, got {gap_blocks}")

        self.n_splits = n_splits
        self.block_size = block_size
        self.gap_blocks = gap_blocks
        self.test_blocks = test_blocks

    def split(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate blocked train/test indices.

        Args:
            X: Features array (n_samples, n_features)
            y: Target array (unused, for sklearn compatibility)
            groups: Group labels (unused, for sklearn compatibility)

        Yields:
            Tuple of (train_indices, test_indices) for each fold

        Raises:
            ValueError: If not enough blocks for requested splits
        """
        X_arr = np.asarray(X)
        n_samples = X_arr.shape[0]

        # Calculate block size if not specified
        if self.block_size is not None:
            block_size = self.block_size
        else:
            # Default: create enough blocks for splits + gaps + tests
            n_blocks_needed = self.n_splits * 2 + self.gap_blocks + 1
            block_size = max(1, n_samples // n_blocks_needed)

        # Calculate number of blocks
        n_blocks = n_samples // block_size

        # Calculate test blocks if not specified
        if self.test_blocks is not None:
            test_blocks = self.test_blocks
        else:
            test_blocks = max(1, n_blocks // (self.n_splits + 1))

        # Validate we have enough blocks
        min_blocks = 1 + self.gap_blocks + test_blocks
        if n_blocks < min_blocks:
            raise ValueError(
                f"Not enough blocks ({n_blocks}) for requested configuration. "
                f"Need at least {min_blocks} blocks "
                f"(min_train=1, gap={self.gap_blocks}, test={test_blocks})"
            )

        # Generate folds
        for fold_idx in range(self.n_splits):
            # Test blocks are at the end
            test_block_end = n_blocks - (self.n_splits - 1 - fold_idx) * test_blocks
            test_block_start = test_block_end - test_blocks

            # Train blocks end before gap
            train_block_end = test_block_start - self.gap_blocks

            if train_block_end <= 0:
                continue  # Skip fold if not enough train blocks

            # Convert block indices to sample indices
            train_end_sample = train_block_end * block_size
            test_start_sample = test_block_start * block_size
            test_end_sample = min(test_block_end * block_size, n_samples)

            train_idx = np.arange(0, train_end_sample)
            test_idx = np.arange(test_start_sample, test_end_sample)

            yield train_idx, test_idx

    def get_n_splits(
        self,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> int:
        """Return the number of splits."""
        return self.n_splits


class PurgedGroupTimeSeriesCV(BaseCrossValidator):
    """Purged and embargoed group time series cross-validation.

    Implements the purged K-Fold from de Prado (2018):
    1. Groups observations by time period
    2. Purges observations within embargo window after test set
    3. Prevents information leakage from overlapping labels

    This is critical for financial time series where labels
    may span multiple periods (e.g., forward returns).

    Args:
        n_splits: Number of CV folds (default 5)
        embargo_pct: Percentage of samples to embargo after test (default 0.01)
        purge_pct: Percentage of samples to purge from train end (default 0.0)

    References:
        de Prado, M. L. (2018). Advances in Financial Machine Learning.
        Chapter 7: Cross-Validation in Finance.

    Example:
        >>> cv = PurgedGroupTimeSeriesCV(n_splits=5, embargo_pct=0.02)
        >>> X = np.random.randn(1000, 10)
        >>> for train, test in cv.split(X):
        ...     # Train indices exclude embargo period after test
        ...     print(f"Train: {len(train)}, Test: {len(test)}")
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.0,
    ):
        """Initialize PurgedGroupTimeSeriesCV.

        Args:
            n_splits: Number of cross-validation folds
            embargo_pct: Percentage of samples to embargo after test
            purge_pct: Percentage of samples to purge from train end

        Raises:
            ValueError: If percentages not in [0, 1] or n_splits < 1
        """
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")
        if not 0 <= embargo_pct <= 1:
            raise ValueError(f"embargo_pct must be in [0, 1], got {embargo_pct}")
        if not 0 <= purge_pct <= 1:
            raise ValueError(f"purge_pct must be in [0, 1], got {purge_pct}")

        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

    def split(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
        event_spans: ArrayLike | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate purged train/test indices.

        For each fold:
        1. Select test set from time-ordered data
        2. Purge observations from train that are within purge_pct of test start
        3. Embargo observations after test that overlap with test labels

        Args:
            X: Features array (n_samples, n_features)
            y: Target array (unused, for sklearn compatibility)
            groups: Group labels for observations (unused in basic impl)
            event_spans: (start, end) indices for each event's span (unused in basic impl)

        Yields:
            Tuple of (train_indices, test_indices) for each fold
        """
        X_arr = np.asarray(X)
        n_samples = X_arr.shape[0]

        # Calculate purge and embargo sizes
        purge_size = int(n_samples * self.purge_pct)
        embargo_size = int(n_samples * self.embargo_pct)

        # Test size (equal splits)
        test_size = n_samples // (self.n_splits + 1)

        for fold_idx in range(self.n_splits):
            # Test set position (from end, working backwards)
            test_end = n_samples - (self.n_splits - 1 - fold_idx) * test_size
            test_start = test_end - test_size

            # Train indices: everything before test (minus purge) and after embargo
            train_before_end = test_start - purge_size
            train_after_start = test_end + embargo_size

            # Combine train indices
            train_before = np.arange(0, max(0, train_before_end))
            train_after = np.arange(min(train_after_start, n_samples), n_samples)
            train_idx = np.concatenate([train_before, train_after])

            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx

    def get_n_splits(
        self,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> int:
        """Return the number of splits."""
        return self.n_splits


def create_time_series_cv(
    strategy: str = "time_series_split",
    n_splits: int = 5,
    **kwargs: Any,
) -> BaseCrossValidator:
    """Factory function to create time series cross-validators.

    Args:
        strategy: CV strategy name. Options:
            - "time_series_split" or "expanding": TimeSeriesCrossValidator (expanding)
            - "sliding": TimeSeriesCrossValidator with expanding=False
            - "blocked": BlockedTimeSeriesCV
            - "purged": PurgedGroupTimeSeriesCV
        n_splits: Number of folds
        **kwargs: Additional arguments for the CV class

    Returns:
        Configured cross-validator instance

    Raises:
        ValueError: If unknown strategy

    Example:
        >>> cv = create_time_series_cv("expanding", n_splits=5, gap=10)
        >>> cv = create_time_series_cv("blocked", n_splits=3, block_size=20)
        >>> cv = create_time_series_cv("purged", n_splits=5, embargo_pct=0.02)
    """
    strategy_lower = strategy.lower()

    if strategy_lower in ("time_series_split", "expanding"):
        return TimeSeriesCrossValidator(n_splits=n_splits, expanding=True, **kwargs)
    elif strategy_lower == "sliding":
        return TimeSeriesCrossValidator(n_splits=n_splits, expanding=False, **kwargs)
    elif strategy_lower == "blocked":
        return BlockedTimeSeriesCV(n_splits=n_splits, **kwargs)
    elif strategy_lower == "purged":
        return PurgedGroupTimeSeriesCV(n_splits=n_splits, **kwargs)
    else:
        valid_strategies = ["time_series_split", "expanding", "sliding", "blocked", "purged"]
        raise ValueError(f"Unknown strategy '{strategy}'. Must be one of {valid_strategies}")
