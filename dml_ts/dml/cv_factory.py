"""Factory for time-series cross-validators (thin wrapper over temporalcv).

The splitters themselves live upstream in temporalcv (the dml_ts
implementations were ported there verbatim with bit-exact golden parity,
then retired here). This factory survives as the dml_ts-style string
dispatch used by the book and the estimators.

The ``"purged"`` strategy now returns the forward-only
:class:`temporalcv.PurgedWalkForward`. The retired
``PurgedGroupTimeSeriesCV`` was a bidirectional purged K-fold that trained
on observations AFTER each test fold — temporal leakage under this repo's
no-lookahead standard — and was deliberately not ported upstream.
"""

from typing import Any

from temporalcv import (
    BlockedTimeSeriesCV,
    PurgedWalkForward,
    TimeSeriesCrossValidator,
)

Splitter = TimeSeriesCrossValidator | BlockedTimeSeriesCV | PurgedWalkForward


def create_time_series_cv(
    strategy: str = "time_series_split",
    n_splits: int = 5,
    **kwargs: Any,
) -> Splitter:
    """Factory function to create time series cross-validators.

    Args:
        strategy: CV strategy name. Options:
            - "time_series_split" or "expanding": TimeSeriesCrossValidator (expanding)
            - "sliding": TimeSeriesCrossValidator with expanding=False
            - "blocked": BlockedTimeSeriesCV
            - "purged": PurgedWalkForward (forward-only purged walk)
        n_splits: Number of folds
        **kwargs: Additional arguments for the CV class. NOTE: for the
            "purged" strategy, ``embargo_pct`` is also a valid
            PurgedWalkForward parameter but with DIFFERENT geometry than the
            retired splitter's parameter of the same name (forward-only
            pre-test trim vs the old bidirectional post-test embargo); the
            new primary control is the absolute ``purge_gap``.

    Returns:
        Configured cross-validator instance

    Raises:
        ValueError: If unknown strategy

    Example:
        >>> cv = create_time_series_cv("expanding", n_splits=5, gap=10)
        >>> cv = create_time_series_cv("blocked", n_splits=3, block_size=20)
        >>> cv = create_time_series_cv("purged", n_splits=5, purge_gap=5)
    """
    strategy_lower = strategy.lower()

    if strategy_lower in ("time_series_split", "expanding"):
        return TimeSeriesCrossValidator(n_splits=n_splits, expanding=True, **kwargs)
    elif strategy_lower == "sliding":
        return TimeSeriesCrossValidator(n_splits=n_splits, expanding=False, **kwargs)
    elif strategy_lower == "blocked":
        return BlockedTimeSeriesCV(n_splits=n_splits, **kwargs)
    elif strategy_lower == "purged":
        return PurgedWalkForward(n_splits=n_splits, **kwargs)
    else:
        valid_strategies = ["time_series_split", "expanding", "sliding", "blocked", "purged"]
        raise ValueError(f"Unknown strategy '{strategy}'. Must be one of {valid_strategies}")
