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
    message = r"No treatment variation at period 0: the treatment is constant \(.*\) on the"
    with pytest.raises(ValueError, match=message):
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
def test_series_linearly_dependent_lags_raise_naming_the_gram(learner):
    """Each lag of an alternating treatment varies, but lag 1 is minus lag 0."""
    y, t, x = series_draw("alternating")
    message = "treatment Gram matrix is singular.*lags are linearly dependent"
    with pytest.raises(ValueError, match=message):
        series_fit(y, t, x, learner)


@pytest.mark.tier2
@pytest.mark.parametrize("jitter", [1e-13, 1e-10])
@pytest.mark.parametrize("learner", ["linear", "ridge"])
def test_series_lags_dependent_up_to_rounding_raise(learner, jitter):
    """Dependence need not be exact in floating point to make the solution meaningless.

    With lags equal up to 1e-13 the solver meets no zero pivot. It returned lag effects of
    -5.4 and -5.8 with standard errors of 1.9e6 on one machine and failed in the later
    inversion on another, so the refusal cannot rest on an exception from the solver.
    """
    y, t, x = series_draw("alternating")
    nudged = t + jitter * np.random.default_rng(1).standard_normal(len(t))
    with pytest.raises(ValueError, match="lags are linearly dependent"):
        series_fit(y, nudged, x, learner)


@pytest.mark.tier2
def test_series_nearly_dependent_lags_warn_about_the_gram():
    """Above the refusal bound and below a well-conditioned design, the fit warns.

    The scaled singular-value ratio is 2e-7 here: more than sqrt(eps), so the design is not
    refused, and the Gram matrix has a condition number near 2e13, above the warning bound
    of 1e12. The result validator may still reject the covariance afterwards, which is not
    this test's subject.
    """
    y, t, x = series_draw("alternating")
    nudged = t + 3e-7 * np.random.default_rng(1).standard_normal(len(t))
    with pytest.warns(RuntimeWarning, match="series treatment Gram is ill-conditioned"):
        try:
            series_fit(y, nudged, x)
        except ValueError as error:
            assert "linearly dependent" not in str(error)


@pytest.mark.tier2
@pytest.mark.parametrize("learner", ["linear", "ridge"])
def test_series_persistent_treatment_is_not_refused(learner):
    """An AR(1) treatment with coefficient 0.99 has correlated lags, not dependent ones."""
    rng = np.random.default_rng(3)
    length = 400
    x = rng.standard_normal((length, 2))
    shocks = rng.standard_normal(length)
    t = np.zeros(length)
    for i in range(1, length):
        t[i] = 0.99 * t[i - 1] + shocks[i]
    y = np.zeros(length)
    y[1:] = 1.0 * t[1:] + 0.5 * t[:-1] + x[1:, 0] + rng.standard_normal(length - 1)
    result = series_fit(y, t, x, learner)
    assert_allclose(result.theta_t, [1.0, 0.5], atol=0.25)


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


# ------------------------------------------------- checks that hold for every learner

LEARNERS = ["linear", "ridge", "random_forest", "gradient_boosting"]


@pytest.mark.tier2
@pytest.mark.parametrize("learner", LEARNERS)
def test_series_treatment_constant_on_the_scored_rows_raises(learner):
    """A step at row 60 varies in the series and is constant on every scored row.

    The first block of a time-series split is only trained on. With the step inside it, the
    "residual" on the scored rows is the error of the nuisance prediction and nothing else,
    and the fit returned two lag effects with no error and no warning.
    """
    y, _, x = series_draw("healthy")
    step = (np.arange(len(y)) >= 60).astype(float)
    with pytest.raises(ValueError, match=r"period 0: the treatment is constant \(1\) on the"):
        series_fit(y, step, x, learner)


@pytest.mark.tier2
@pytest.mark.parametrize("learner", LEARNERS)
@pytest.mark.parametrize("levels", [(1.0, -1.0), (1.0, 0.0), (5.0, 3.0)])
def test_series_two_level_alternation_raises_under_every_learner(learner, levels):
    """Lag 1 of an on/off treatment is a constant minus lag 0, whatever the two levels.

    A learner that only approximates the treatment leaves residuals that are not exactly
    dependent: for a 0/1 treatment gradient boosting returned two lag effects near 5.5 with
    p = 0.03 where the truth is 1.0 and 0.5. The dependence is a property of the lags and
    is tested on them.
    """
    y, _, x = series_draw("healthy")
    high, low = levels
    treatment = np.where(np.arange(len(y)) % 2 == 0, high, low)
    with pytest.raises(ValueError, match="Gram matrix is singular.*linearly dependent"):
        series_fit(y, treatment, x, learner)


@pytest.mark.tier2
def test_series_lags_dependent_only_given_the_controls_raise():
    """With the two-period total T_t + T_{t-1} as a control, the lags stay distinct.

    Their residuals are not: what is left of T_t is minus what is left of T_{t-1}. No
    check on the lags sees this, and each lag still has residual variation, so the test is
    the one on the residuals.
    """
    y, t, x = series_draw("healthy")
    total = t + np.concatenate([[0.0], t[:-1]])
    with pytest.raises(ValueError, match="residualized treatment lags are linearly dependent"):
        series_fit(y, t, np.column_stack([x, total]))


@pytest.mark.tier2
def test_series_period_four_treatment_with_three_lags_raises():
    """sin(pi t / 2) is 0, 1, 0, -1: lag 2 is minus lag 0, while lag 1 is free."""
    y, _, x = series_draw("healthy")
    treatment = np.round(np.sin(np.pi * np.arange(len(y)) / 2))
    estimator = DynamicGEstimationDML(
        n_periods=3, model_y="linear", model_t="linear", n_folds=3, random_state=0
    )
    with pytest.raises(ValueError, match="treatment lags are linearly dependent"):
        estimator.fit(y, treatment, x, mode="series")


@pytest.mark.tier2
def test_series_with_no_more_scored_rows_than_lags_raises():
    """55 rows and 25 lags pass the length check and leave about 21 scored rows."""
    rng = np.random.default_rng(0)
    estimator = DynamicGEstimationDML(
        n_periods=25, model_y="linear", model_t="linear", n_folds=2, random_state=0
    )
    with pytest.raises(ValueError, match=r"scored \d+ rows for 25 lags"):
        estimator.fit(
            rng.standard_normal(55),
            rng.standard_normal(55),
            rng.standard_normal((55, 1)),
            mode="series",
        )


@pytest.mark.tier2
@pytest.mark.parametrize("learner", LEARNERS)
def test_series_binary_treatment_with_both_values_scored_is_fitted(learner):
    """The raw checks refuse nothing that varies: a binary treatment is fitted."""
    y, t, x = series_draw("healthy")
    result = series_fit(y, (t > 0).astype(float), x, learner)
    assert np.isfinite(result.theta_t).all()
    assert np.isfinite(result.se_t).all()


# ------------------------------------------------- the size of the per-lag bound


def nearly_reproduced(relative):
    """A treatment the controls reproduce up to noise of the given relative size."""
    y, _, x = series_draw("healthy")
    rng = np.random.default_rng(7)
    return y, 0.5 * x[:, 0] + relative * rng.standard_normal(len(y)), x


@pytest.mark.tier2
def test_series_bound_refuses_an_unexplained_share_below_one_part_in_a_million():
    """The first term of the bound is 1e-6 of the centered norm of the lag.

    Noise of relative size 2e-7 leaves residuals of about 4e-7 of that norm: refused. A bound
    of a few ulps of norm(T), or of sqrt(eps) * norm(T) alone, would accept it.
    """
    y, t, x = nearly_reproduced(2e-7)
    with pytest.raises(ValueError, match="No residual treatment variation at period 0"):
        series_fit(y, t, x)


@pytest.mark.tier2
def test_series_bound_accepts_an_unexplained_share_above_one_part_in_a_million():
    """Noise of relative size 1e-4 leaves residuals of 2e-4 of the centered norm: fitted."""
    y, t, x = nearly_reproduced(1e-4)
    assert np.isfinite(series_fit(y, t, x).theta_t).all()


@pytest.mark.tier2
def test_series_treatment_with_spread_below_sqrt_eps_of_its_level_is_refused():
    """The second term of the bound: variation below 1.5e-8 of the level is refused.

    This is the documented limit of the shared check, so a shift of 1e8 is refused where a
    shift of 1e3 changes nothing. Subtract the level before fitting.
    """
    y, t, x = series_draw("healthy")
    with pytest.raises(ValueError, match="No residual treatment variation at period 0.*<= bound"):
        series_fit(y, t + 1e8, x)
