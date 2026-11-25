"""Properties of the public estimators under changes of units, rows and folds."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from dml_ts.dml import DynamicGEstimationDML, RollingWindowDML, TemporalPLRDML, double_ml
from dml_ts.dml.fwl import fwl_estimate
from dml_ts.dml.robinson import robinson_estimator

pytestmark = pytest.mark.tier2


def plr_draw(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    t = x[:, 0] + rng.normal(size=n)
    y = 2 * t + x[:, 1] + rng.normal(size=n)
    return y, t, x


@pytest.mark.parametrize("seed", [19, 53, 71])
def test_changing_treatment_units_preserves_physical_effect_and_uncertainty(seed):
    rng = np.random.default_rng(seed)
    y, t, x = plr_draw(rng.integers(80, 140), seed)
    scales = [1e-8, -1e-8, -1e4, 10 ** rng.uniform(-6, 6)]

    def temporal(y, t, x):
        return TemporalPLRDML(n_lags=0, model_t="ridge", model_y="ridge", n_splits=3).fit(y, t, x)

    for fit in (
        fwl_estimate,
        lambda y, t, x: double_ml(y, t, x, model="ridge", n_folds=3),
        lambda y, t, x: robinson_estimator(y, t, x, model="ridge"),
        temporal,
    ):
        original = fit(y, t, x)
        for scale in scales:
            converted = fit(y, scale * t, x)
            assert_allclose(converted.theta * scale, original.theta, rtol=1e-10)
            assert_allclose(converted.se * abs(scale), original.se, rtol=1e-10)


@pytest.mark.parametrize("add_intercept", [True, False])
def test_fwl_ols_redundant_controls_against_rank_aware_full_design(add_intercept):
    for seed in range(4):
        y, t, x = plr_draw(100 + seed, seed)
        redundant_x = np.column_stack((x, x[:, 1], 2 * x[:, 0] - x[:, 1]))
        design = np.column_stack((t, redundant_x))
        if add_intercept:
            design = np.column_stack((design, np.ones(len(y))))
        pinv = np.linalg.pinv(design)
        beta = pinv @ y
        resid = y - design @ beta
        df = len(y) - np.linalg.matrix_rank(design)
        expected_se = np.sqrt((resid @ resid / df) * (pinv[0] @ pinv[0]))
        result = fwl_estimate(y, t, redundant_x, method="ols", add_intercept=add_intercept)
        assert_allclose([result.theta, result.se], [beta[0], expected_se], rtol=1e-11)
        for scale in [1e-8, 1e8]:
            with pytest.raises(ValueError, match="no variation"):
                fwl_estimate(y, scale * x[:, 1], redundant_x, method="ols")


def test_contrast_interval_uses_signed_effect_and_absolute_standard_error():
    y, t, x = plr_draw(100, 19)
    model = TemporalPLRDML(n_lags=0, model_y="ridge", model_t="ridge", n_splits=3)
    fit = model.fit(y, t, x)
    for alpha in (0.01, 0.05, 0.20):
        for start, stop in ((0, 1), (1, 0), (2, -5), (1.5, 1.5)):
            difference = stop - start
            margin = norm.isf(alpha / 2) * abs(difference) * fit.se
            lower, upper = model.effect_interval(X=x[:4], alpha=alpha, T0=start, T1=stop)
            assert_allclose(lower, fit.theta * difference - margin)
            assert_allclose(upper, fit.theta * difference + margin)


@pytest.mark.parametrize("n,width,step", [(40, 40, 1), (41, 41, 2), (42, 40, 1), (83, 33, 10)])
def test_rolling_exact_rows_against_every_standalone_fit(n, width, step):
    y, t, x = plr_draw(n, 19)
    model = RollingWindowDML(window_size=width, step_size=step, model_y="ridge", model_t="ridge")
    centers, slopes, ses = model.fit(y, t, x, time_index=2010 + np.arange(n)).get_effects()
    starts = np.arange(0, n - width + 1, step)
    assert_array_equal(centers, starts + width // 2)
    for i, start in enumerate(starts):
        rows = slice(start, start + width)
        ref = TemporalPLRDML(n_lags=0, model_y="ridge", model_t="ridge", n_splits=3)
        fit = ref.fit(y[rows], t[rows], x[rows])
        assert_allclose([slopes[i], ses[i]], [fit.theta, fit.se], rtol=1e-12)


@pytest.mark.parametrize("strategy", ["time_series_split", "blocked_cv", "purged_cv"])
def test_future_changes_cannot_alter_earlier_scored_residuals(strategy):
    n = 241
    y, t, x = plr_draw(n, 17)
    options = {
        "n_lags": 2,
        "model_y": "ridge",
        "model_t": "ridge",
        "n_splits": 3,
        "gap": 2,
        "cv_strategy": strategy,
    }
    fit = TemporalPLRDML(**options).fit(y, t, x)
    cutoff = n // 2
    rng = np.random.default_rng(504)
    for values in (y, t, x):
        values[cutoff:] += rng.normal(size=values[cutoff:].shape)
    changed = TemporalPLRDML(**options).fit(y, t, x)
    prefix = cutoff - fit.lagged_rows_dropped - fit.dropped_initial_rows
    assert prefix > 0
    assert_array_equal(fit.Y_residual[:prefix], changed.Y_residual[:prefix])
    assert_array_equal(fit.T_residual[:prefix], changed.T_residual[:prefix])


class RecordingRegressor(RegressorMixin, BaseEstimator):
    """Single-output learner exposing which rows and targets each clone received."""

    calls = []

    def __init__(self, role):
        self.role = role

    def fit(self, x, y):
        assert y.ndim == 1
        self.train_ = x[:, 0].astype(int)
        self.y_ = y.copy()
        return self

    def predict(self, x):
        self.calls.append((self.role, self.train_, x[:, 0].astype(int), self.y_))
        return np.full(len(x), self.y_.mean())


def test_dynamic_series_routes_each_target_and_preserves_the_training_gap(monkeypatch):
    y, t, x = plr_draw(120, 35)
    x[:, 0] = np.arange(len(y))
    monkeypatch.setattr(RecordingRegressor, "calls", [])
    est = DynamicGEstimationDML(
        n_periods=2,
        model_y=RecordingRegressor("outcome"),
        model_t=RecordingRegressor("treatment"),
        n_folds=3,
        gap=2,
        random_state=13,
    )
    est.fit(y, t, x, mode="series")
    calls = RecordingRegressor.calls
    folds = {tuple(test) for _, _, test, _ in calls}
    assert len(folds) == 3
    for rows in folds:
        per_fold = [
            (role, train, target) for role, train, test, target in calls if tuple(test) == rows
        ]
        assert [role for role, _, _ in per_fold] == ["outcome", "treatment", "treatment"]
        train = per_fold[0][1]
        assert train.max() + 2 < min(rows)
        for (_, actual_train, target), reference in zip(
            per_fold, (y[train], t[train], t[train - 1]), strict=True
        ):
            assert_array_equal(actual_train, train)
            assert_array_equal(target, reference)
    with pytest.raises(ValueError, match="Unknown model spec"):
        DynamicGEstimationDML(n_periods=2, model_y="ridge", model_t="invalid").fit(y, t, x)


@pytest.mark.parametrize("mode", ["panel", "series"])
def test_dynamic_gradient_boosting_alias_matches_explicit_single_output_models(mode):
    y, t, x = plr_draw(96, 24)
    groups = np.repeat(np.arange(48), 2) if mode == "panel" else None
    options = {"n_periods": 2, "model_y": "ridge", "n_folds": 2, "random_state": 17}
    actual = DynamicGEstimationDML(**options, model_t="gradient_boosting").fit(
        y, t, x, groups=groups
    )
    reg = MultiOutputRegressor(GradientBoostingRegressor(random_state=17))
    expected = DynamicGEstimationDML(**options, model_t=reg).fit(y, t, x, groups=groups)
    assert_allclose(actual.theta_t, expected.theta_t)
    assert_allclose(actual.cov, expected.cov)
