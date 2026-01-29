"""
Time Series Cross-Validation for Double Machine Learning.

Implements cross-fitting strategies that respect temporal ordering:
1. TimeSeriesSplit - Standard expanding window
2. BlockedCV - Blocked cross-validation with gaps
3. PurgedCV - Purged cross-validation to prevent leakage

These are essential for valid inference in time series DML where
standard K-Fold would violate the temporal structure.

Phase 2A Implementation - NOT YET IMPLEMENTED
Current status: Skeleton with interface definitions

References:
- Bergmeir, C., Hyndman, R. J., & Koo, B. (2018). A note on the validity
  of cross-validation for evaluating autoregressive time series prediction.
- de Prado, M. L. (2018). Advances in Financial Machine Learning. Chapter 7.

Usage (planned):
    >>> from src.dml.cross_fitting import TimeSeriesCrossValidator
    >>> cv = TimeSeriesCrossValidator(
    ...     n_splits=5,
    ...     gap=10,
    ...     purge_length=5,
    ... )
    >>> for train_idx, test_idx in cv.split(X, time_index=time):
    ...     # Fit model on train, predict on test
    ...     pass
"""

from typing import Any, Generator, Iterator, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import BaseCrossValidator


ArrayLike = Union[np.ndarray, List[float]]


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
    """Time series cross-validator with purging and blocking.

    NOT YET IMPLEMENTED - Phase 2A skeleton.

    Implements time-series aware cross-validation that:
    1. Respects temporal ordering (train always before test)
    2. Includes gap between train and test to prevent leakage
    3. Optionally purges observations near the boundary
    4. Supports blocked/grouped cross-validation

    Args:
        n_splits: Number of CV folds
        gap: Number of periods between train and test
        purge_length: Number of periods to purge from train end
        test_size: Size of test set (None for equal splits)
        expanding: Whether to use expanding window (vs sliding)

    Example (planned):
        >>> cv = TimeSeriesCrossValidator(n_splits=5, gap=10, purge_length=5)
        >>> for train, test in cv.split(X, time_index=time):
        ...     model.fit(X[train], Y[train])
        ...     pred = model.predict(X[test])
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        purge_length: int = 0,
        test_size: Optional[int] = None,
        expanding: bool = True,
    ):
        """Initialize TimeSeriesCrossValidator.

        Args:
            n_splits: Number of cross-validation folds
            gap: Periods between train end and test start
            purge_length: Periods to remove from train end
            test_size: Size of test set (None for automatic)
            expanding: Use expanding (True) or sliding (False) window
        """
        self.n_splits = n_splits
        self.gap = gap
        self.purge_length = purge_length
        self.test_size = test_size
        self.expanding = expanding

    def split(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        time_index: Optional[ArrayLike] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices for each fold.

        NOT YET IMPLEMENTED.

        Args:
            X: Features array
            y: Target array (unused, for sklearn compatibility)
            groups: Group labels (unused, for sklearn compatibility)
            time_index: Time index for each observation

        Yields:
            Tuple of (train_indices, test_indices) for each fold

        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError(
            "TimeSeriesCrossValidator.split() is not yet implemented. "
            "This is a Phase 2A skeleton."
        )

    def get_n_splits(
        self,
        X: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
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


class BlockedTimeSeriesCV(BaseCrossValidator):
    """Blocked time series cross-validation.

    NOT YET IMPLEMENTED - Phase 2A skeleton.

    Groups observations into blocks and ensures:
    1. No data leakage between blocks
    2. Entire blocks are in train or test (not split)
    3. Temporal ordering preserved

    Useful for data with grouped observations (e.g., weekly data grouped by month).
    """

    def __init__(
        self,
        n_splits: int = 5,
        block_size: Optional[int] = None,
        gap_blocks: int = 1,
    ):
        """Initialize BlockedTimeSeriesCV.

        Args:
            n_splits: Number of cross-validation folds
            block_size: Size of each block (None for automatic)
            gap_blocks: Number of blocks between train and test
        """
        self.n_splits = n_splits
        self.block_size = block_size
        self.gap_blocks = gap_blocks

    def split(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate blocked train/test indices.

        NOT YET IMPLEMENTED.
        """
        raise NotImplementedError(
            "BlockedTimeSeriesCV.split() is not yet implemented. " "This is a Phase 2A skeleton."
        )

    def get_n_splits(
        self,
        X: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
    ) -> int:
        """Return the number of splits."""
        return self.n_splits


class PurgedGroupTimeSeriesCV(BaseCrossValidator):
    """Purged and embarassed group time series cross-validation.

    NOT YET IMPLEMENTED - Phase 2A skeleton.

    Implements the purged K-Fold from de Prado (2018):
    1. Groups observations by time period
    2. Purges observations within embargo window
    3. Prevents information leakage from overlapping labels

    This is critical for financial time series where labels
    may span multiple periods.

    References:
        de Prado, M. L. (2018). Advances in Financial Machine Learning.
        Chapter 7: Cross-Validation in Finance.
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
            purge_pct: Percentage of samples to purge from train
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

    def split(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        event_spans: Optional[ArrayLike] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate purged train/test indices.

        NOT YET IMPLEMENTED.

        Args:
            X: Features array
            y: Target array
            groups: Group labels for observations
            event_spans: (start, end) indices for each event's span

        Yields:
            Tuple of (train_indices, test_indices) for each fold
        """
        raise NotImplementedError(
            "PurgedGroupTimeSeriesCV.split() is not yet implemented. "
            "This is a Phase 2A skeleton."
        )

    def get_n_splits(
        self,
        X: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
    ) -> int:
        """Return the number of splits."""
        return self.n_splits


def create_time_series_cv(
    strategy: str = "time_series_split",
    n_splits: int = 5,
    **kwargs: Any,
) -> BaseCrossValidator:
    """Factory function to create time series cross-validators.

    NOT YET IMPLEMENTED.

    Args:
        strategy: CV strategy name
        n_splits: Number of folds
        **kwargs: Additional arguments for the CV class

    Returns:
        Configured cross-validator instance

    Raises:
        NotImplementedError: This function is not yet implemented
    """
    raise NotImplementedError(
        "create_time_series_cv() is not yet implemented. " "This is a Phase 2A skeleton."
    )
