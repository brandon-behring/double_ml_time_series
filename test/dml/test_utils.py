"""Tests for the shared DML-layer helpers (Track B1b)."""

import numpy as np
import pytest

from dml_ts.dml import double_ml
from dml_ts.dml._utils import (
    cross_fit_predictions,
    get_nuisance_model,
    theta_via_fwl,
    validate_lengths,
)
from dml_ts.dml.robinson import robinson_estimator


def _plr_data(seed: int = 0, n: int = 150) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    T = 0.5 * X[:, 0] + rng.normal(size=n)
    Y = 2.0 * T + X[:, 0] + rng.normal(size=n)
    return Y, T, X


@pytest.mark.tier1
class TestFactory:
    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model type"):
            get_nuisance_model("xgboost")  # type: ignore[arg-type]

    def test_random_state_is_wired(self) -> None:
        rf = get_nuisance_model("random_forest", random_state=7)
        assert rf.get_params()["random_state"] == 7


@pytest.mark.tier1
class TestThetaViaFWL:
    def test_no_variation_raises_with_caller_message(self) -> None:
        z = np.zeros(50)
        with pytest.raises(ValueError, match="custom no variation"):
            theta_via_fwl(z, z, no_variation_msg="custom no variation")

    def test_nan_residuals_raise_loudly(self) -> None:
        """NaN must not slip through the < 1e-10 guard (NaN < x is False)."""
        y = np.ones(50)
        t = np.ones(50)
        t[3] = np.nan
        with pytest.raises(ValueError, match="Non-finite treatment residuals"):
            theta_via_fwl(y, t, no_variation_msg="unused")


@pytest.mark.tier1
class TestValidateLengths:
    def test_t_mismatch(self) -> None:
        with pytest.raises(ValueError, match="must match Y length"):
            validate_lengths(np.ones(10), np.ones(9), np.ones((10, 2)))

    def test_x_mismatch(self) -> None:
        with pytest.raises(ValueError, match="must match Y length"):
            validate_lengths(np.ones(10), np.ones(10), np.ones((9, 2)))


@pytest.mark.tier2
class TestLassoWidening:
    """double_ml/robinson previously raised on model='lasso'; the shared
    factory deliberately widened both to its full option set."""

    def test_double_ml_accepts_lasso(self) -> None:
        Y, T, X = _plr_data()
        r = double_ml(Y, T, X, n_folds=3, model="lasso", random_state=0)
        assert np.isfinite(r.theta) and np.isfinite(r.se)

    def test_robinson_accepts_lasso(self) -> None:
        Y, T, X = _plr_data()
        r = robinson_estimator(Y, T, X, model="lasso")
        assert np.isfinite(r.theta) and np.isfinite(r.se)


@pytest.mark.tier2
class TestCrossFitPredictions:
    def test_uncovered_rows_stay_nan(self) -> None:
        Y, T, X = _plr_data(n=60)
        from sklearn.linear_model import Ridge

        # One fold covering only rows 30..59: the prefix must stay NaN.
        splits = [(np.arange(0, 30), np.arange(30, 60))]
        Y_hat, T_hat = cross_fit_predictions(X, Y, T, splits, Ridge(), Ridge())
        assert np.isnan(Y_hat[:30]).all() and np.isnan(T_hat[:30]).all()
        assert np.isfinite(Y_hat[30:]).all() and np.isfinite(T_hat[30:]).all()
