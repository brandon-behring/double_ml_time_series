"""Guards of the core estimators: degenerate treatments, constant targets, argument checks.

Each test pins a behavior that the adversarial review of #69 found wrong or unpinned.
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier
from sklearn.model_selection import KFold

from dml_ts.dml import (
    DynamicGEstimationDML,
    PanelDML,
    RollingWindowDML,
    TemporalPLRDML,
    double_ml,
    dynamic_g_estimation,
    temporal_plr_dml,
)
from dml_ts.dml._utils import (
    compute_r2,
    is_numerically_constant,
    no_variation_tolerance,
    theta_via_fwl,
)
from dml_ts.dml.fwl import fwl_estimate, fwl_residualize
from dml_ts.dml.robinson import robinson_estimator

LINEAR_FITS = {
    "fwl_qr": lambda y, t, x: fwl_estimate(y, t, x),
    "fwl_ols": lambda y, t, x: fwl_estimate(y, t, x, method="ols"),
    "double_ml": lambda y, t, x: double_ml(y, t, x, model=LinearRegression(), n_folds=3),
    "robinson": lambda y, t, x: robinson_estimator(y, t, x, model=LinearRegression()),
}


def plr_draw(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    t = x[:, 0] + rng.normal(size=n)
    y = 2 * t + x[:, 1] + rng.normal(size=n)
    return y, t, x


# --- No residual treatment variation -----------------------------------------


@pytest.mark.tier2
@pytest.mark.parametrize("level", [0.0, 100.0, 1000.0, 1e5])
@pytest.mark.parametrize("name", sorted(LINEAR_FITS))
def test_exact_collinearity_raises_when_the_controls_have_a_level(name, level):
    """A bound of a few ulps of norm(T) let most of these through at level >= 100."""
    for seed in range(6):
        rng = np.random.default_rng(seed)
        x = level + rng.normal(size=(100, 3))
        y = rng.normal(size=100)
        standardized = (x[:, 2] - x[:, 2].mean()) / x[:, 2].std()
        for t in (x[:, 1] - x[:, 0], standardized):
            with pytest.raises(ValueError, match="no variation"):
                LINEAR_FITS[name](y, t, x)


@pytest.mark.tier2
def test_treatment_in_the_span_of_a_few_rows_raises_under_ols():
    rng = np.random.default_rng(7)
    for n in (6, 10, 20):
        for _ in range(150):
            x = rng.normal(size=(n, 3))
            with pytest.raises(ValueError, match="no variation"):
                fwl_estimate(rng.normal(size=n), x @ rng.normal(size=3), x, method="ols")


@pytest.mark.tier2
@pytest.mark.parametrize("scale", [1e-10, 1.0, 1e10, 1e12])
def test_exact_collinearity_raises_in_every_unit_of_the_treatment(scale):
    """The absolute cutoff sum(T_tilde**2) < 1e-10 accepted the two large scales."""
    y, _, x = plr_draw(120, 4)
    for name, fit in LINEAR_FITS.items():
        with pytest.raises(ValueError, match="no variation"):
            fit(y, scale * (x[:, 0] - 2 * x[:, 1]), x)
        assert np.isfinite(fit(y, scale * plr_draw(120, 4)[1], x).theta), name


@pytest.mark.tier2
@pytest.mark.parametrize("constant", [0.0, 0.1, 5.0, 1e6])
@pytest.mark.parametrize("level", [0.0, 1e3])
def test_constant_treatment_raises_whatever_the_level_of_the_controls(constant, level):
    rng = np.random.default_rng(11)
    x = level + rng.normal(size=(150, 3))
    y = rng.normal(size=150)
    t = np.full(150, constant)
    fits = dict(LINEAR_FITS)
    fits["double_ml_ridge"] = lambda y, t, x: double_ml(y, t, x, model="ridge", n_folds=3)
    fits["double_ml_forest"] = lambda y, t, x: double_ml(y, t, x, model="random_forest", n_folds=3)
    for fit in fits.values():
        with pytest.raises(ValueError, match="no variation"):
            fit(y, t, x)


@pytest.mark.tier2
@pytest.mark.parametrize("offset", [1e3, 1e6])
def test_adding_a_constant_to_the_treatment_leaves_the_estimate(offset):
    y, t, x = plr_draw(200, 11)
    base = fwl_estimate(y, t, x)
    shifted = fwl_estimate(y, t + offset, x)
    assert_allclose([shifted.theta, shifted.se], [base.theta, base.se], rtol=1e-7)


@pytest.mark.tier2
def test_variation_below_sqrt_eps_of_the_level_is_refused_with_the_numbers():
    """The documented band: residual spread under 1.5e-8 of the treatment's level."""
    y, t, x = plr_draw(200, 11)
    with pytest.raises(ValueError, match=r"no variation.*<= bound"):
        fwl_estimate(y, t + 1e9, x)


@pytest.mark.tier2
@pytest.mark.parametrize("fixed_effects", ["individual", "time", "twoway"])
def test_panel_treatment_absorbed_by_the_fixed_effects_raises(fixed_effects):
    """Demeaning leaves rounding residue, which a scale taken after demeaning accepts."""
    rng = np.random.default_rng(3)
    n_units, n_times = 40, 18
    unit = np.repeat(np.arange(n_units), n_times)
    time = np.tile(np.arange(n_times), n_units)
    by_unit = rng.normal(size=n_units)[unit]
    by_time = rng.normal(size=n_times)[time]
    t = {"individual": by_unit, "time": by_time, "twoway": by_unit + by_time}[fixed_effects]
    x = rng.normal(size=(len(t), 2))
    y = rng.normal(size=len(t))
    model = PanelDML(fixed_effects=fixed_effects, model_y="ridge", model_t="ridge", n_folds=3)
    with pytest.raises(ValueError, match="no variation after removing"):
        model.fit(y, t, x, unit, time)
    varying = t + rng.normal(size=len(t))
    assert np.isfinite(model.fit(y + varying, varying, x, unit, time).theta)


@pytest.mark.tier1
def test_tolerance_scales_with_the_units_and_ignores_a_shift_of_the_treatment():
    t = np.random.default_rng(2).normal(size=300)
    base = no_variation_tolerance(t)
    for scale in (1e-9, -3.0, 1e9):
        assert_allclose(no_variation_tolerance(scale * t), abs(scale) * base, rtol=1e-12)
    assert_allclose(no_variation_tolerance(t + 10.0), base, rtol=1e-12)
    with pytest.raises(ValueError, match="Non-finite treatment"):
        no_variation_tolerance(np.array([1.0, np.nan, 2.0]))


@pytest.mark.tier1
def test_without_a_treatment_scale_only_exactly_zero_residuals_raise():
    y = np.array([1.0, 2.0, 4.0])
    with pytest.raises(ValueError, match="nothing left"):
        theta_via_fwl(y, np.zeros(3), no_variation_msg="nothing left")
    theta, _ = theta_via_fwl(y, 1e-9 * np.array([1.0, -1.0, 0.5]), no_variation_msg="nothing left")
    assert np.isfinite(theta)


# --- R² of a constant target -------------------------------------------------


@pytest.mark.tier1
@pytest.mark.parametrize("constant", [0.0, 0.001, 0.1, 0.5, 1e8])
@pytest.mark.parametrize("n", [100, 257, 1000])
def test_r2_of_a_constant_target_is_zero_in_every_helper(constant, n):
    """mean(np.full(n, 0.1)) != 0.1, so 'ss_tot > 0' is true and R² was rounding noise."""
    y = np.full(n, constant)
    design = np.column_stack([np.ones(n), np.random.default_rng(n).normal(size=n)])
    assert is_numerically_constant(y)
    assert compute_r2(y, 0.9 * y + 0.01) == 0.0
    assert dynamic_g_estimation._r2(y, np.full(n, 0.01)) == 0.0
    assert fwl_residualize(y, design)[1] == 0.0


@pytest.mark.tier1
@pytest.mark.parametrize("scale", [1e-9, 1e-3, 1e9])
def test_r2_does_not_depend_on_the_units_of_the_target(scale):
    """The absolute cutoffs 1e-10 and 1e-12 returned 0.0 for the smallest unit."""
    y = np.random.default_rng(5).normal(size=200)
    pred = 0.9 * y
    design = np.column_stack([np.ones(200), y + np.random.default_rng(6).normal(size=200)])
    assert not is_numerically_constant(scale * y)
    assert_allclose(compute_r2(y, pred), 0.99, atol=1e-3)
    assert_allclose(compute_r2(scale * y, scale * pred), compute_r2(y, pred), rtol=1e-10)
    assert_allclose(
        dynamic_g_estimation._r2(scale * y, scale * (y - pred)),
        dynamic_g_estimation._r2(y, y - pred),
        rtol=1e-10,
    )
    assert_allclose(
        fwl_residualize(scale * y, design)[1], fwl_residualize(y, design)[1], rtol=1e-10
    )


# --- Argument checks ---------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_temporal_alpha_outside_the_open_unit_interval_raises(alpha):
    y, t, x = plr_draw(120, 3)
    model = TemporalPLRDML(n_lags=0, model_y="ridge", model_t="ridge", n_splits=3)
    with pytest.raises(ValueError, match="alpha must be in"):
        model.fit(y, t, x, alpha=alpha)
    model.fit(y, t, x)
    with pytest.raises(ValueError, match="alpha must be in"):
        model.effect_interval(alpha=alpha)


@pytest.mark.tier1
@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_size": 0},
        {"window_size": -5},
        {"window_size": 50.0},
        {"window_size": True},
        {"window_size": 50, "step_size": 0},
        {"window_size": 50, "step_size": 2.5},
    ],
)
def test_rolling_window_arguments_must_be_positive_integers(kwargs):
    with pytest.raises(ValueError, match="must be a positive integer"):
        RollingWindowDML(**kwargs)
    assert RollingWindowDML(window_size=np.int64(50), step_size=np.int64(5)).window_size == 50


@pytest.mark.tier1
def test_rolling_rejects_inputs_of_unequal_length():
    y, t, x = plr_draw(120, 3)
    model = RollingWindowDML(window_size=40, step_size=20, model_y="ridge", model_t="ridge")
    for args in ((y[:-1], t, x), (y, t[:-1], x), (y, t, x[:-1])):
        with pytest.raises(ValueError, match="length|rows"):
            model.fit(*args)
    with pytest.raises(ValueError, match="time_index length"):
        model.fit(y, t, x, time_index=np.arange(119))


# --- Weak-overlap warning ----------------------------------------------------


@pytest.mark.tier2
@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
@pytest.mark.parametrize("unexplained,warns", [(1e-8, True), (1e-3, False)])
def test_weak_overlap_warning_follows_the_unexplained_share_in_every_unit(
    monkeypatch, scale, unexplained, warns
):
    """An absolute cutoff warned at 1e-8 units and stayed silent at 1e8 units."""
    y, t, x = plr_draw(300, 5)
    t = scale * t
    noise = np.random.default_rng(9).normal(size=len(t))
    noise = (noise - noise.mean()) / noise.std()

    def nuisance(X, Y, T, cv, outcome_model, treatment_model):
        return np.zeros_like(Y), T - np.sqrt(unexplained) * T.std() * noise

    monkeypatch.setattr(temporal_plr_dml, "_cross_fit_nuisance_time_series", nuisance)
    model = TemporalPLRDML(n_lags=0, model_y="ridge", model_t="ridge", n_splits=3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(y, t, x)
    weak = [w for w in caught if "residual variation is very low" in str(w.message)]
    assert bool(weak) is warns


# --- Classifier nuisances ----------------------------------------------------


def binary_draw(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-1.5 * x[:, 0]))).astype(float)
    y = 1.5 * t + x[:, 0] + rng.normal(size=n)
    return y, t, x


@pytest.mark.tier2
def test_classifier_treatment_model_enters_through_its_class_probabilities():
    y, t, x = binary_draw(400, 21)
    result = double_ml(
        y,
        t,
        x,
        outcome_model=LinearRegression(),
        treatment_model=LogisticRegression(),
        n_folds=3,
        random_state=0,
    )
    by_probability = np.empty(len(t))
    by_label = np.empty(len(t))
    y_hat = np.empty(len(t))
    for train, test in KFold(n_splits=3, shuffle=True, random_state=0).split(x):
        classifier = LogisticRegression().fit(x[train], t[train])
        by_probability[test] = classifier.predict_proba(x[test])[:, 1]
        by_label[test] = classifier.predict(x[test])
        y_hat[test] = LinearRegression().fit(x[train], y[train]).predict(x[test])

    def slope(t_hat):
        return ((y - y_hat) @ (t - t_hat)) / ((t - t_hat) @ (t - t_hat))

    assert_allclose(result.theta, slope(by_probability), rtol=1e-12)
    assert abs(slope(by_label) - slope(by_probability)) > 1e-3
    recoded = double_ml(
        y,
        t + 1.0,
        x,
        outcome_model=LinearRegression(),
        treatment_model=LogisticRegression(),
        n_folds=3,
        random_state=0,
    )
    assert_allclose(recoded.theta, result.theta, rtol=1e-10)


@pytest.mark.tier2
def test_classifier_without_probabilities_is_refused():
    y, t, x = binary_draw(200, 22)
    with pytest.raises(ValueError, match="predict_proba"):
        double_ml(y, t, x, outcome_model=LinearRegression(), treatment_model=RidgeClassifier())
    panel = {"groups": np.repeat(np.arange(100), 2)}
    with pytest.raises(ValueError, match="predict_proba"):
        DynamicGEstimationDML(n_periods=2, model_y="ridge", model_t=RidgeClassifier()).fit(
            y, t, x, **panel
        )


# --- Fold reuse --------------------------------------------------------------


@pytest.mark.tier2
def test_dynamic_panel_draws_one_set_of_folds_per_fit(monkeypatch):
    """With random_state=None a split per residualization gave each its own shuffle."""
    seen = []
    original = dynamic_g_estimation._crossfit_residualize

    def spy(H, targets, model, splits):
        seen.append(splits)
        return original(H, targets, model, splits)

    monkeypatch.setattr(dynamic_g_estimation, "_crossfit_residualize", spy)
    y, t, x = plr_draw(96, 24)
    DynamicGEstimationDML(
        n_periods=2, model_y="ridge", model_t="ridge", n_folds=3, random_state=None
    ).fit(y, t, x, groups=np.repeat(np.arange(48), 2))
    assert len(seen) == 4  # two stages, each with an outcome and a treatment residualization
    assert all(splits is seen[0] for splits in seen)
    held_out = np.sort(np.concatenate([test for _, test in seen[0]]))
    assert held_out.tolist() == list(range(48))
