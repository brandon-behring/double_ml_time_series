"""Tests for the cv factory (thin wrapper over temporalcv splitters)."""

import numpy as np
import pytest
from temporalcv import (
    BlockedTimeSeriesCV,
    PurgedWalkForward,
    TimeSeriesCrossValidator,
)

from dml_ts.dml import create_time_series_cv

pytestmark = pytest.mark.tier1


class TestCreateTimeSeriesCV:
    def test_expanding(self) -> None:
        cv = create_time_series_cv("expanding", n_splits=3)
        assert isinstance(cv, TimeSeriesCrossValidator)
        assert cv.expanding is True

    def test_time_series_split(self) -> None:
        cv = create_time_series_cv("time_series_split", n_splits=3)
        assert isinstance(cv, TimeSeriesCrossValidator)
        assert cv.expanding is True

    def test_sliding(self) -> None:
        cv = create_time_series_cv("sliding", n_splits=3)
        assert isinstance(cv, TimeSeriesCrossValidator)
        assert cv.expanding is False

    def test_blocked(self) -> None:
        cv = create_time_series_cv("blocked", n_splits=3, block_size=10)
        assert isinstance(cv, BlockedTimeSeriesCV)

    def test_purged_is_forward_only(self) -> None:
        """The purged strategy returns the forward-only walk (PR-8 leakage
        fix: the retired splitter trained on observations after each test
        fold)."""
        cv = create_time_series_cv("purged", n_splits=3, purge_gap=5)
        assert isinstance(cv, PurgedWalkForward)
        X = np.zeros((200, 1))
        for train_idx, test_idx in cv.split(X):
            assert np.max(train_idx) + 5 < np.min(test_idx)

    def test_invalid_strategy(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            create_time_series_cv("nope")
