"""Only scored rows count as treatment variation in ``TemporalPLRDML`` (#76).

``TemporalPLRDML`` scores the rows that have an out-of-fold prediction; the first block of a
time-series split is trained on and never scored. A treatment that varies in that block alone
is constant on every scored row. Its "residual" there is the error of the nuisance prediction,
and the fit returned an estimate with a small standard error (1.63 with 0.12 in one draw, 2.09
with 0.12 in another, true slope 2).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dml_ts.dml import PanelDML, RollingWindowDML, TemporalPLRDML

LEARNERS = ["ridge", "lasso", "random_forest", "gradient_boosting"]


def temporal_draw(treatment, seed=0):
    """Outcome with slope 2 on the given treatment and one control."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((len(treatment), 2))
    y = 2.0 * treatment + x[:, 1] + rng.standard_normal(len(treatment))
    return y, x


def temporal_fit(y, t, x, learner="ridge"):
    return TemporalPLRDML(n_lags=0, model_y=learner, model_t=learner, random_state=0).fit(y, t, x)


@pytest.mark.tier2
@pytest.mark.parametrize("learner", LEARNERS)
@pytest.mark.parametrize("levels", [(1.0, 0.0), (0.0, 1.0), (5.0, 3.0)])
def test_treatment_that_varies_only_before_the_scored_rows_is_refused(learner, levels):
    """A switch at row 60 of 400 lies inside the first block, which is trained on only."""
    before, after = levels
    treatment = np.where(np.arange(400) < 60, before, after)
    y, x = temporal_draw(treatment)
    message = rf"treatment is constant \({after:g}\) on the \d+ rows that have an out-of-fold"
    with pytest.raises(ValueError, match=message):
        temporal_fit(y, treatment, x, learner)


@pytest.mark.tier2
def test_rare_binary_treatment_with_no_treated_scored_row_is_refused():
    treatment = np.zeros(400)
    treatment[[5, 17, 40]] = 1.0
    y, x = temporal_draw(treatment)
    with pytest.raises(ValueError, match="Treatment has no variation on the scored rows"):
        temporal_fit(y, treatment, x)


@pytest.mark.tier2
@pytest.mark.parametrize("learner", LEARNERS)
def test_treatment_that_switches_among_the_scored_rows_is_fitted(learner):
    """The same step at row 200 is variation the scored rows see; nothing is refused."""
    treatment = (np.arange(400) >= 200).astype(float)
    y, x = temporal_draw(treatment)
    result = temporal_fit(y, treatment, x, learner)
    assert np.isfinite(result.theta) and np.isfinite(result.se)


@pytest.mark.tier2
def test_one_treated_scored_row_is_variation_and_is_fitted():
    """The check refuses a constant, not a rare, treatment."""
    treatment = np.zeros(400)
    treatment[300] = 1.0
    y, x = temporal_draw(treatment)
    assert np.isfinite(temporal_fit(y, treatment, x).theta)


@pytest.mark.tier2
def test_rolling_window_with_a_constant_treatment_is_a_named_skip():
    """Windows are fitted by the same estimator: the constant one is skipped and said to be."""
    rng = np.random.default_rng(2)
    treatment = rng.standard_normal(400)
    treatment[100:200] = 0.0
    y, x = temporal_draw(treatment, seed=2)
    rolling = RollingWindowDML(window_size=100, step_size=100, model_y="ridge", model_t="ridge")
    with pytest.warns(RuntimeWarning, match="center 150: Treatment has no variation on the scored"):
        rolling.fit(y, treatment, x)
    centers, slopes, _ = rolling.get_effects()
    assert list(centers) == [50, 250, 350]
    assert_allclose(slopes, 2.0, atol=0.6)


@pytest.mark.tier2
@pytest.mark.parametrize("treated_units", [1, 5])
def test_panel_whose_treated_units_are_never_scored_is_refused(treated_units):
    """PanelDML is not edited here, but its inner estimator is this one.

    It hands unit-major rows to the same time-series split, so the first block that is
    trained on and never scored is the first UNITS. With the treatment varying only in those
    units the fit returned 17.2 with a standard error of 5.7 (one unit) and 4.4 with 1.0 (five
    units) for a true slope of 2. How panels should be split belongs to the PanelDML redesign;
    until then such a panel is refused.
    """
    rng = np.random.default_rng(0)
    units, periods = 30, 10
    individual = np.repeat(np.arange(units), periods)
    period = np.tile(np.arange(periods), units)
    x = rng.standard_normal((units * periods, 2))
    treatment = np.where(individual < treated_units, rng.standard_normal(units * periods), 0.0)
    y = 2.0 * treatment + x[:, 1] + rng.standard_normal(units * periods)
    with pytest.raises(ValueError, match="Treatment has no variation on the scored rows"):
        PanelDML(model_y="ridge", model_t="ridge").fit(y, treatment, x, individual, period)


@pytest.mark.tier2
def test_panel_with_every_unit_treated_keeps_its_estimate():
    rng = np.random.default_rng(0)
    units, periods = 30, 10
    individual = np.repeat(np.arange(units), periods)
    period = np.tile(np.arange(periods), units)
    x = rng.standard_normal((units * periods, 2))
    treatment = rng.standard_normal(units * periods)
    y = 2.0 * treatment + x[:, 1] + rng.standard_normal(units * periods)
    result = PanelDML(model_y="ridge", model_t="ridge").fit(y, treatment, x, individual, period)
    assert_allclose(result.theta, 2.0, atol=0.2)
