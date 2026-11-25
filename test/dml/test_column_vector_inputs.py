"""Column-vector outcomes and treatments are refused, not broadcast (#72).

An outcome or treatment of shape ``(n, 1)`` broadcast against ``(n,)`` residuals to an
``(n, n)`` product. ``fwl_estimate`` then returned a slope of order 1e-16 for a true slope of
2, ``double_ml`` returned 0.0005, ``robinson_estimator`` returned -0.002 with a standard error
of 0.004, and ``RollingWindowDML`` returned no window at all.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from dml_ts.dml import PanelDML, RollingWindowDML, TemporalPLRDML, double_ml
from dml_ts.dml._utils import theta_via_fwl, validate_lengths
from dml_ts.dml.fwl import fwl_estimate
from dml_ts.dml.robinson import robinson_estimator

TRUE_SLOPE = 2.0


def plr_draw(n=300, seed=0):
    """A partially linear draw with slope ``TRUE_SLOPE`` and one confounder."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    t = x[:, 0] + rng.normal(size=n)
    y = TRUE_SLOPE * t + x[:, 1] + rng.normal(size=n)
    return y, t, x


def rolling_slopes(y, t, x):
    """Slopes of three disjoint windows of 100 rows."""
    model = RollingWindowDML(window_size=100, step_size=100, model_y="ridge", model_t="ridge")
    return model.fit(y, t, x).get_effects()[1]


# The five estimators that share ``validate_lengths``; each returns its slope(s).
SLOPES = {
    "fwl_estimate": lambda y, t, x: fwl_estimate(y, t, x).theta,
    "double_ml": lambda y, t, x: double_ml(y, t, x, model="ridge").theta,
    "robinson_estimator": lambda y, t, x: robinson_estimator(y, t, x, model="ridge").theta,
    "TemporalPLRDML": lambda y, t, x: (
        TemporalPLRDML(n_lags=0, model_y="ridge", model_t="ridge").fit(y, t, x).theta
    ),
    "RollingWindowDML": rolling_slopes,
}

COLUMN_CASES = {
    "treatment_column": (False, True, "T"),
    "outcome_column": (True, False, "Y"),
    "both_columns": (True, True, "Y"),
}


@pytest.mark.tier2
@pytest.mark.parametrize("case", sorted(COLUMN_CASES))
@pytest.mark.parametrize("name", sorted(SLOPES))
def test_column_vector_outcome_or_treatment_is_refused(name, case):
    """Every shape that used to broadcast raises, and the message names the argument."""
    y, t, x = plr_draw()
    y_is_column, t_is_column, first_refused = COLUMN_CASES[case]
    y_in = y[:, None] if y_is_column else y
    t_in = t[:, None] if t_is_column else t
    message = rf"{first_refused} must be one-dimensional, got shape \(300, 1\).*\.ravel\(\)\."
    with pytest.raises(ValueError, match=message):
        SLOPES[name](y_in, t_in, x)


@pytest.mark.tier2
@pytest.mark.parametrize("name", sorted(SLOPES))
def test_one_dimensional_inputs_still_give_the_slope(name):
    """The shape check refuses nothing it should accept: flat inputs recover the slope."""
    y, t, x = plr_draw()
    slopes = np.atleast_1d(SLOPES[name](y, t, x))
    assert slopes.size >= 1
    assert_allclose(slopes, TRUE_SLOPE, atol=0.35)


@pytest.mark.tier2
def test_a_dataframe_column_is_refused_and_its_flat_form_is_accepted():
    """``df[["t"]].to_numpy()`` is the shape pandas users produce; it gave -2.6e-16."""
    y, t, x = plr_draw()
    column = pd.DataFrame({"t": t})[["t"]].to_numpy()
    assert column.shape == (300, 1)
    with pytest.raises(ValueError, match="T must be one-dimensional"):
        fwl_estimate(y, column, x)
    assert fwl_estimate(y, column.ravel(), x).theta == fwl_estimate(y, t, x).theta


@pytest.mark.tier2
def test_a_two_column_treatment_is_refused_with_its_shape():
    y, t, x = plr_draw()
    with pytest.raises(ValueError, match=r"T must be one-dimensional, got shape \(300, 2\)"):
        double_ml(y, np.column_stack([t, t]), x, model="ridge")


@pytest.mark.tier2
@pytest.mark.parametrize("case", sorted(COLUMN_CASES))
def test_panel_dml_refuses_a_column_vector_and_keeps_its_flat_estimate(case):
    """PanelDML's own code is untouched here; it must not start returning a number."""
    y, t, x = plr_draw()
    individual, period = np.repeat(np.arange(30), 10), np.tile(np.arange(10), 30)
    y_is_column, t_is_column, _ = COLUMN_CASES[case]
    y_in = y[:, None] if y_is_column else y
    t_in = t[:, None] if t_is_column else t
    panel = PanelDML(model_y="ridge", model_t="ridge")
    with pytest.raises(ValueError, match="one-dimensional|1-D"):
        panel.fit(y_in, t_in, x, individual, period)
    assert_allclose(panel.fit(y, t, x, individual, period).theta, TRUE_SLOPE, atol=0.35)


@pytest.mark.tier1
@pytest.mark.parametrize("name,bad", [("Y", 0), ("T", 1)])
@pytest.mark.parametrize("shape", [(6, 1), (1, 6), (6, 2), ()])
def test_validate_lengths_requires_flat_outcome_and_treatment(name, bad, shape):
    arguments = [np.ones(6), np.ones(6), np.ones((6, 2))]
    arguments[bad] = np.ones(shape)
    with pytest.raises(ValueError, match=rf"{name} must be one-dimensional, got shape"):
        validate_lengths(*arguments)


@pytest.mark.tier1
def test_validate_lengths_accepts_flat_arrays_and_a_flat_x():
    assert validate_lengths(np.ones(6), np.ones(6), np.ones((6, 2))) is None
    assert validate_lengths(np.ones(6), np.ones(6), np.ones(6)) is None


@pytest.mark.tier1
@pytest.mark.parametrize(
    "y_shape,t_shape",
    [((5,), (5, 1)), ((5, 1), (5,)), ((5, 1), (5, 1)), ((5, 5), (5, 5)), ((5,), (4,))],
)
def test_theta_via_fwl_refuses_residuals_that_would_broadcast(y_shape, t_shape):
    """The closed form is the last line of defense for callers that skip the entry check."""
    rng = np.random.default_rng(1)
    message = r"must be one-dimensional and of equal shape, got \(5,"
    with pytest.raises(ValueError, match=message):
        theta_via_fwl(rng.normal(size=y_shape), rng.normal(size=t_shape), no_variation_msg="unused")


@pytest.mark.tier1
def test_theta_via_fwl_of_flat_residuals_is_the_ratio_of_sums():
    y_tilde, t_tilde = np.array([1.0, -2.0, 0.5]), np.array([2.0, -1.0, 1.0])
    theta, t_sq = theta_via_fwl(y_tilde, t_tilde, no_variation_msg="unused")
    assert theta == pytest.approx(4.5 / 6.0)
    assert t_sq == pytest.approx(6.0)


@pytest.mark.tier2
@pytest.mark.parametrize("name", sorted(SLOPES))
def test_a_zero_dimensional_outcome_is_refused_by_the_shape_check(name):
    """The shape check runs before anything calls len() on the outcome."""
    _, t, x = plr_draw()
    with pytest.raises(ValueError, match=r"Y must be one-dimensional, got shape \(\)"):
        SLOPES[name](np.float64(1.0), t, x)


@pytest.mark.tier2
def test_the_hint_in_the_message_works_on_a_dataframe_column():
    """A DataFrame has no ``ravel``; ``np.asarray(T).ravel()`` is what the message names."""
    y, t, x = plr_draw()
    column = pd.DataFrame({"t": t})[["t"]]
    with pytest.raises(ValueError, match=r"np\.asarray\(T\)\.ravel\(\)"):
        fwl_estimate(y, column, x)
    assert fwl_estimate(y, np.asarray(column).ravel(), x).theta == fwl_estimate(y, t, x).theta
