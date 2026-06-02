"""Executable public snippets for README and Sphinx examples."""

import numpy as np
import pytest

from dml_ts.data import create_synthetic_fred_data
from dml_ts.dml import TemporalPLRDML, TimeSeriesCrossValidator, double_ml
from dml_ts.validation import create_insurance_dgp

pytestmark = pytest.mark.tier1


def test_readme_cross_sectional_double_ml_snippet() -> None:
    """README cross-sectional PLR DML snippet uses the live API."""
    rng = np.random.default_rng(42)
    n = 120
    X = rng.normal(size=(n, 5))
    T = X[:, 0] + rng.normal(size=n)
    Y = 2.0 * T + X[:, 1] ** 2 + rng.normal(size=n)

    result = double_ml(Y, T, X, n_folds=3, model="ridge", random_state=42)

    assert np.isfinite(result.theta)
    assert result.ci_lower < result.ci_upper


def test_readme_temporal_plr_dml_snippet() -> None:
    """README temporal PLR snippet uses the live API and result fields."""
    rng = np.random.default_rng(42)
    n = 120
    time_index = np.arange(n)
    X = np.column_stack([rng.normal(size=n), np.sin(time_index / 12)])
    T = 0.4 * X[:, 0] + rng.normal(size=n)
    Y = 1.5 * T + X[:, 1] + rng.normal(size=n)

    model = TemporalPLRDML(
        n_lags=2,
        model_y="ridge",
        model_t="ridge",
        n_splits=3,
        gap=1,
        hac_bandwidth=4,
        random_state=42,
    )
    result = model.fit(Y, T, X, time_index=time_index)

    assert np.isfinite(result.theta)
    assert result.se > 0
    assert result.lagged_rows_dropped == 2
    assert result.n_periods + result.dropped_initial_rows == n - 2


def test_sphinx_time_series_cross_validator_snippet() -> None:
    """Sphinx temporal CV snippet uses purge_length, not a stale purge argument."""
    X = np.arange(100).reshape(-1, 1)
    cv = TimeSeriesCrossValidator(n_splits=5, gap=3, purge_length=2)

    for train_idx, test_idx in cv.split(X):
        assert train_idx[-1] + 3 + 2 <= test_idx[0]


def test_readme_synthetic_fred_snippet() -> None:
    """Synthetic FRED snippet uses date-based arguments and MacroControlsResult.data."""
    macro = create_synthetic_fred_data(
        start_date="2018-01-01",
        end_date="2020-12-31",
        frequency="M",
        seed=42,
    )

    assert list(macro.data.columns) == ["GDPC1", "CPIAUCSL", "UNRATE", "FEDFUNDS"]
    assert macro.data.shape[0] > 0


def test_sphinx_insurance_dgp_snippet() -> None:
    """Insurance DGP snippet uses the current factory signature."""
    data = create_insurance_dgp(
        realism="moderate",
        n_periods=24,
        n_products=4,
        true_tau=-0.8,
        seed=42,
    )

    assert len(data.Y) == 96
    assert data.X.shape[0] == 96
    assert data.true_params.tau == -0.8
