"""Exact causal designs with known projection, assignment, and intervention targets."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from dml_ts import double_ml
from dml_ts.production.dml_pipeline import InsuranceDMLPipeline, PipelineConfig

pytestmark = pytest.mark.tier2


class KnownLinearMean(RegressorMixin, BaseEstimator):
    """Supply a population nuisance function on the single raw covariate."""

    def __init__(self, intercept, slope):
        self.intercept = intercept
        self.slope = slope

    def fit(self, x, y):
        return self

    def predict(self, x):
        return self.intercept + self.slope * x[:, 0]


def test_heterogeneous_plr_targets_the_variance_weighted_effect():
    x = np.repeat([0.0, 1.0], 100).reshape(-1, 1)
    treatment = np.r_[np.ones(10), np.zeros(90), np.ones(50), np.zeros(50)]
    effects = 2 * x[:, 0]
    result = double_ml(
        effects * treatment,
        treatment,
        x,
        outcome_model=KnownLinearMean(0, 1),
        treatment_model=KnownLinearMean(0.1, 0.4),
    )
    expected = (0.09 * 0 + 0.25 * 2) / (0.09 + 0.25)
    assert effects.mean() == 1
    assert result.theta == pytest.approx(expected)


def test_orthogonal_score_is_quadratic_in_joint_nuisance_error():
    # Every combination of the covariate and two independent mean-zero shocks occurs.
    x, v, e = np.tile(np.array(np.meshgrid(*[[-1.0, 1.0]] * 3)).reshape(3, -1), 6)
    treatment = x + v
    outcome = 2 * treatment + 0.5 * x + e
    for delta in (0, 0.25, 0.125, -0.125):
        fit = double_ml(
            outcome,
            treatment,
            x[:, None],
            outcome_model=KnownLinearMean(0, 2.5 + delta),
            treatment_model=KnownLinearMean(0, 1 + delta),
        )
        score_at_truth = (fit.theta - 2) * np.mean(fit.T_residual**2)
        assert score_at_truth == pytest.approx(-(delta**2), abs=1e-14)


def test_observational_equivalence_cannot_identify_the_intervention():
    rng = np.random.default_rng(42)
    u, v = rng.standard_normal((2, 1000))
    noise = rng.normal(scale=0.5, size=1000)
    x = rng.standard_normal((1000, 2))
    treatment, independent_error = (u + v) / np.sqrt(2), (u - v) / np.sqrt(2)
    # In the first model T is exogenous. The second structural outcome ignores T.
    baseline = 0.5 * x[:, 0] + noise
    causal = treatment + independent_error + baseline
    confounded = np.sqrt(2) * u + baseline
    np.testing.assert_allclose(causal, confounded, atol=2e-15)
    fits = [double_ml(y, treatment, x, model="ridge") for y in (causal, confounded)]
    a, b = fits
    np.testing.assert_allclose(
        [a.theta, a.se, a.ci_lower, a.ci_upper], [b.theta, b.se, b.ci_lower, b.ci_upper], atol=1e-13
    )
    assert 0 < a.ci_lower < 1 < a.ci_upper


@pytest.mark.parametrize(
    "true_prop,true_outcome,expected",
    [(True, True, 2.5), (True, False, 2.5), (False, True, 2.5), (False, False, 4.25)],
)
def test_binary_aipw_with_one_or_two_wrong_nuisances(tmp_path, true_prop, true_outcome, expected):
    # Construct identical treatment counts in each training and scoring stratum.
    x = np.empty((32, 1))
    t = np.empty(32)
    for _, held in KFold(2, shuffle=True, random_state=42).split(x):
        x[held, 0] = np.repeat([0, 1], 8)
        t[held] = np.r_[np.ones(2), np.zeros(6), np.ones(6), np.zeros(2)]
    y = 2 * x[:, 0] + t * (1 + 3 * x[:, 0])
    config = PipelineConfig(
        n_folds=2,
        random_state=42,
        use_hac=False,
        monitoring_enabled=False,
        model_registry_path=str(tmp_path),
        propensity_model=DecisionTreeClassifier(max_depth=1, random_state=42)
        if true_prop
        else DummyClassifier(strategy="prior"),
        outcome_model=LinearRegression()
        if true_outcome
        else DummyRegressor(strategy="constant", constant=0),
    )
    assert InsuranceDMLPipeline(config).fit(x, t, y).ate == pytest.approx(expected)
