"""``TemporalPLRDML`` refuses a treatment that is constant on the rows it scores (#76).

``TemporalPLRDML`` scores the rows that have an out-of-fold prediction; the first block of a
time-series split is trained on and never scored. A treatment that varies in that block alone
is constant on every scored row. Its "residual" there is the error of the nuisance prediction,
so the slope rests on how the two nuisance fits err together and not on held-out variation.
Over 400 draws of a switch at row 60 of 400 the fit returned estimates centered on the true
slope with a standard error about 30% too small (coverage 84.5%).

The check refuses exact constancy and nothing else. A treatment that varies on a few scored
rows is fitted, and its interval can be far too short; that is a separate defect (#81).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dml_ts.dml import PanelDML, RollingWindowDML, TemporalPLRDML

LEARNERS = ["ridge", "lasso", "random_forest", "gradient_boosting"]
STRATEGIES = ["time_series_split", "blocked_cv", "purged_cv"]


def temporal_draw(treatment, seed=0):
    """Outcome with slope 2 on the given treatment and one control."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((len(treatment), 2))
    y = 2.0 * treatment + x[:, 1] + rng.standard_normal(len(treatment))
    return y, x


def temporal_fit(y, t, x, learner="ridge", **settings):
    settings.setdefault("n_lags", 0)
    return TemporalPLRDML(model_y=learner, model_t=learner, random_state=0, **settings).fit(y, t, x)


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
@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("n_lags", [0, 1])
def test_the_refusal_holds_for_every_split_strategy_and_with_lagged_controls(n_lags, strategy):
    """``n_lags=1`` is the class default; the scored rows differ between the strategies."""
    treatment = (np.arange(400) < 60).astype(float)
    y, x = temporal_draw(treatment)
    with pytest.raises(ValueError, match="Treatment has no variation on the scored rows"):
        temporal_fit(y, treatment, x, n_lags=n_lags, cv_strategy=strategy)


@pytest.mark.tier2
def test_treatment_constant_up_to_rounding_on_the_scored_rows_is_refused():
    """0.1 + 0.2 and 0.3 differ in the last bit; a test for exact equality would fit them."""
    treatment = np.where(np.arange(400) % 2 == 0, 0.1 + 0.2, 0.3)
    treatment[:60] = np.random.default_rng(6).standard_normal(60)
    assert np.ptp(treatment[60:]) > 0
    y, x = temporal_draw(treatment)
    with pytest.raises(ValueError, match="Treatment has no variation on the scored rows"):
        temporal_fit(y, treatment, x)


@pytest.mark.tier2
def test_rare_binary_treatment_with_no_treated_scored_row_is_refused():
    treatment = np.zeros(400)
    treatment[[5, 17, 40]] = 1.0
    y, x = temporal_draw(treatment)
    with pytest.raises(ValueError, match="Treatment has no variation on the scored rows"):
        temporal_fit(y, treatment, x)


@pytest.mark.tier2
@pytest.mark.parametrize("learner", LEARNERS)
def test_treatment_that_switches_among_the_scored_rows_is_not_refused(learner):
    """The same step at row 200 is variation the scored rows see, and the slope is recovered."""
    treatment = (np.arange(400) >= 200).astype(float)
    y, x = temporal_draw(treatment)
    result = temporal_fit(y, treatment, x, learner)
    assert result.n_periods == 330
    assert_allclose(result.theta, 2.0, atol=0.5)


@pytest.mark.tier2
def test_one_treated_scored_row_is_outside_this_check():
    """The check refuses a constant treatment, and a single treated scored row is not constant.

    This marks the boundary of the check, not a design that works: over 400 draws of this
    design the estimates have a standard deviation of 0.97 against a mean reported standard
    error of 0.056, and the 95% interval covers the truth in 7.5% of them. The test passes
    when the fit returns; it says nothing about the estimate.
    """
    treatment = np.zeros(400)
    treatment[300] = 1.0
    y, x = temporal_draw(treatment)
    assert temporal_fit(y, treatment, x).n_periods == 330


@pytest.mark.tier2
def test_rolling_window_whose_treatment_varies_only_in_its_first_block_is_a_named_skip():
    """Each window is fitted by the same estimator and has its own unscored first block.

    The second window's treatment varies in its first 20 rows and is constant after them.
    It returned an estimate before; it is now skipped, and the warning names it.
    """
    rng = np.random.default_rng(2)
    treatment = rng.standard_normal(400)
    treatment[120:200] = 0.0
    y, x = temporal_draw(treatment, seed=2)
    rolling = RollingWindowDML(window_size=100, step_size=100, model_y="ridge", model_t="ridge")
    with pytest.warns(RuntimeWarning, match="center 150: Treatment has no variation on the scored"):
        rolling.fit(y, treatment, x)
    centers, slopes, _ = rolling.get_effects()
    assert list(centers) == [50, 250, 350]
    assert_allclose(slopes, 2.0, atol=0.6)


@pytest.mark.tier2
@pytest.mark.parametrize("treated_units", [1, 5])
def test_panel_whose_only_treated_units_are_never_scored_is_refused(treated_units):
    """PanelDML is not edited here, but its inner estimator is this one.

    It hands unit-major rows to the same time-series split, so the block that is trained on
    and never scored is the first UNITS. With the treatment varying only in those units and
    exactly zero elsewhere, the fit returned 17.2 with a standard error of 5.7 (one unit) and
    4.4 with 1.0 (five units) for a true slope of 2.

    The refusal reaches PanelDML only when the demeaned treatment is exactly constant on the
    scored rows. A constant non-zero dose leaves rounding residue after demeaning and is
    still fitted; how panels are split and checked belongs to the PanelDML redesign.
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
