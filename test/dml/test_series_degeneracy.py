"""Series-mode ``DynamicGEstimationDML`` refuses lags it cannot identify.

The series fit solved its treatment Gram system with no check. A lag that the controls
reproduce exactly gave a slope of order 1/eps (``T = 0.5 * x0`` returned 1e13 to 1e15 under
the linear learner), and a constant treatment ended in ``LinAlgError: Singular matrix``.
Each lag is now held to the bound the other estimators use, and a singular Gram system is
reported as such.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dml_ts.dml import DynamicGEstimationDML


def series_draw(kind, length=400, seed=0, level=0.0):
    """A two-lag series: Y_t = 1.0 T_t + 0.5 T_{t-1} + X_t0 + noise, with a chosen treatment."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((length, 2)) + level
    noise = rng.standard_normal(length)
    treatment = {
        "healthy": 0.5 * (x[:, 0] - level) + noise,
        "collinear": 0.5 * x[:, 0],
        "alternating": np.where(np.arange(length) % 2 == 0, 1.0, -1.0),
    }[kind]
    y = np.zeros(length)
    y[1:] = 1.0 * treatment[1:] + 0.5 * treatment[:-1] + x[1:, 0] + rng.standard_normal(length - 1)
    return y, treatment, x


def series_fit(y, t, x, learner="linear"):
    estimator = DynamicGEstimationDML(
        n_periods=2, model_y=learner, model_t=learner, n_folds=3, random_state=0
    )
    return estimator.fit(y, t, x, mode="series")


@pytest.mark.tier2
@pytest.mark.parametrize("constant", [0.0, 3.0, 1e6])
@pytest.mark.parametrize("learner", ["linear", "ridge"])
def test_series_constant_treatment_raises_for_both_learners(learner, constant):
    """A constant treatment ended in ``LinAlgError: Singular matrix``."""
    y, _, x = series_draw("healthy")
    with pytest.raises(ValueError, match="No residual treatment variation at period 0"):
        series_fit(y, np.full(len(y), constant), x, learner)


@pytest.mark.tier2
@pytest.mark.parametrize("level", [0.0, 1e3])
@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
def test_series_treatment_reproduced_by_the_controls_raises(scale, level):
    """``T = 0.5 * x0`` returned a lag-0 slope of order 1e13 to 1e15 under the linear learner."""
    y, t, x = series_draw("collinear", level=level)
    with pytest.raises(ValueError, match=r"period 0 .*norm of the treatment residuals .* <= bound"):
        series_fit(y, scale * t, x)


@pytest.mark.tier2
def test_series_lagged_treatment_among_the_controls_names_the_lag():
    """With T_{t-1} as a control, lag 1 has no residual variation and lag 0 does."""
    y, t, x = series_draw("healthy")
    lagged = np.concatenate([[0.0], t[:-1]])
    with pytest.raises(ValueError, match="No residual treatment variation at period 1"):
        series_fit(y, t, np.column_stack([x, lagged]))


@pytest.mark.tier2
@pytest.mark.parametrize("learner", ["linear", "ridge"])
def test_series_linearly_dependent_lags_warn_then_raise_naming_the_gram(learner):
    """Each lag of an alternating treatment varies, but lag 1 is minus lag 0."""
    y, t, x = series_draw("alternating")
    with (
        pytest.warns(RuntimeWarning, match="series treatment Gram is ill-conditioned"),
        pytest.raises(ValueError, match="series treatment Gram matrix is singular"),
    ):
        series_fit(y, t, x, learner)


@pytest.mark.tier2
@pytest.mark.parametrize("learner", ["linear", "ridge"])
def test_series_healthy_fit_recovers_both_lags(learner):
    y, t, x = series_draw("healthy")
    result = series_fit(y, t, x, learner)
    assert_allclose(result.theta_t, [1.0, 0.5], atol=0.15)


@pytest.mark.tier2
@pytest.mark.parametrize("scale", [1e-8, 1e8])
def test_series_estimate_scales_with_the_units_of_the_treatment(scale):
    """The lag check is relative: a change of units is not mistaken for degeneracy."""
    y, t, x = series_draw("healthy")
    reference = series_fit(y, t, x).theta_t
    assert_allclose(series_fit(y, scale * t, x).theta_t, reference / scale, rtol=1e-6)


@pytest.mark.tier2
def test_series_estimate_ignores_a_shift_of_the_treatment():
    y, t, x = series_draw("healthy")
    assert_allclose(series_fit(y, t + 1e3, x).theta_t, series_fit(y, t, x).theta_t, rtol=1e-6)
