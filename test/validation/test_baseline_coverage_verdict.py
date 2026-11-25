"""The six baselines judge interval coverage as the DML validator does.

They classified the bias p-value alone, so ``coverage_status`` was ``None`` and the overall
status ignored the intervals: ``OLSWithControls.validate()`` returned PASS with 76% of its 95%
intervals covering the truth, and ``NaiveOLS`` returned PASS with 34%.
"""

import numpy as np
import pytest
from scipy import stats

from dml_ts.validation.bias_validation import assess_simulation_draws
from dml_ts.validation.bootstrap_config import BootstrapConfig
from dml_ts.validation.dgp_generator import DGPGenerator
from dml_ts.validation.ipw_baseline import AugmentedIPW, IPWEstimator
from dml_ts.validation.ml_baseline import RandomForestEstimator, XGBoostEstimator
from dml_ts.validation.ols_baseline import NaiveOLS, OLSWithControls

RANK = {"PASS": 0, "WARNING": 1, "FAIL": 2}
BASELINES = [
    NaiveOLS,
    OLSWithControls,
    IPWEstimator,
    AugmentedIPW,
    RandomForestEstimator,
    XGBoostEstimator,
]


def centered_errors(n):
    """Signed errors with mean zero and spread, so the bias test does not reject."""
    return np.linspace(-1.0, 1.0, n)


def intervals_covering(errors, covered):
    """Intervals around each error; the first ``covered`` contain zero, the rest do not."""
    half_width = np.where(np.arange(errors.size) < covered, np.abs(errors) + 0.5, 0.0)
    shift = np.where(np.arange(errors.size) < covered, 0.0, 5.0)
    return np.column_stack([errors - half_width + shift, errors + half_width + shift])


# ---------------------------------------------------------------- the shared assessment


@pytest.mark.tier1
@pytest.mark.parametrize(
    "covered,coverage_verdict,overall",
    [
        (190, "PASS", "PASS"),
        (152, "FAIL", "FAIL"),
        (170, "FAIL", "FAIL"),
        (200, "WARNING", "WARNING"),
    ],
)
def test_coverage_verdict_is_the_exact_binomial_test_against_the_nominal_level(
    covered, coverage_verdict, overall
):
    """A deficit that the binomial test rejects fails; an excess warns; the status is the worse."""
    errors = centered_errors(200)
    verdicts, p_bias, p_coverage = assess_simulation_draws(
        errors, intervals_covering(errors, covered), 0.0, nominal=0.95
    )
    assert p_coverage == stats.binomtest(covered, 200, 0.95).pvalue
    assert p_bias == pytest.approx(1.0)
    assert verdicts.bias == "PASS"
    assert verdicts.coverage == coverage_verdict
    assert verdicts.overall == overall


@pytest.mark.tier1
def test_coverage_is_counted_against_the_true_effect_and_the_given_nominal_level():
    """The same intervals pass at a nominal level of 80% and fail at 95%."""
    errors = centered_errors(200) + 3.0
    bounds = intervals_covering(errors - 3.0, 164) + 3.0
    at_80, _, p_80 = assess_simulation_draws(errors - 3.0, bounds, 3.0, nominal=0.80)
    at_95, _, p_95 = assess_simulation_draws(errors - 3.0, bounds, 3.0, nominal=0.95)
    assert (p_80, p_95) == (
        stats.binomtest(164, 200, 0.80).pvalue,
        stats.binomtest(164, 200, 0.95).pvalue,
    )
    assert (at_80.coverage, at_95.coverage) == ("PASS", "FAIL")


@pytest.mark.tier1
@pytest.mark.parametrize(
    "spoil,message",
    [
        (lambda e, b: (e[:-1], b), "one simulation error and one confidence interval"),
        (lambda e, b: (e, b[:, :1]), "one simulation error and one confidence interval"),
        (lambda e, b: (e, np.where(np.arange(b.size).reshape(b.shape) == 3, np.nan, b)), "no NaN"),
        (lambda e, b: (e, b[:, ::-1]), "ordered endpoints"),
    ],
)
def test_assessment_refuses_draws_it_cannot_count(spoil, message):
    errors = centered_errors(50)
    bad_errors, bad_bounds = spoil(errors, intervals_covering(errors, 48))
    with pytest.raises(ValueError, match=message):
        assess_simulation_draws(bad_errors, bad_bounds, 0.0, nominal=0.95)


# ---------------------------------------------------------------- the baselines


@pytest.mark.tier2
@pytest.mark.parametrize("baseline,coverage", [(OLSWithControls, 0.76), (NaiveOLS, 0.34)])
def test_an_unbiased_baseline_with_short_intervals_fails(baseline, coverage):
    """The stored check of the end-to-end review: status PASS at 76% and at 34% coverage.

    ``DGPGenerator`` redraws its coefficients on every draw, so the mean error is near zero
    and the bias verdict passes for both. Their intervals do not cover: naive OLS ignores the
    confounders, and ``OLSWithControls`` leaves the intercept out of its standard error (#61).
    """
    dgp = DGPGenerator(n=200, p=5, true_effect=1.0, random_state=42)
    result = baseline(n_simulations=300, random_state=42).validate(dgp)
    assert result.coverage == pytest.approx(coverage, abs=0.005)
    assert result.bias_status == "PASS"
    assert result.coverage_status == "FAIL"
    assert result.status == "FAIL"
    covered = round(result.coverage * 300)
    assert result.coverage_p_value == stats.binomtest(covered, 300, 0.95).pvalue


@pytest.mark.tier2
@pytest.mark.parametrize("baseline", BASELINES)
def test_every_baseline_reports_both_verdicts_and_the_worse_one_as_its_status(baseline):
    dgp = DGPGenerator(n=150, p=2, true_effect=2.0, random_state=7)
    config = BootstrapConfig(n_bootstrap_bias=20, n_bootstrap_ci=20)
    result = baseline(n_simulations=12, bootstrap_config=config, random_state=7).validate(dgp)
    assert result.bias_status in RANK
    assert result.coverage_status in RANK
    assert result.status == max(result.bias_status, result.coverage_status, key=lambda v: RANK[v])
    covered = round(result.coverage * 12)
    assert result.coverage_p_value == stats.binomtest(covered, 12, 0.95).pvalue


@pytest.mark.tier2
@pytest.mark.parametrize("baseline", [RandomForestEstimator, XGBoostEstimator])
def test_coverage_is_tested_on_the_draws_the_bias_test_uses(baseline, monkeypatch):
    """A draw with an empty treatment arm has no estimate; neither test counts it."""
    dgp = DGPGenerator(n=100, p=2, true_effect=2.0, random_state=123)
    draws = [dgp.generate() for _ in range(6)]
    draws[0].T[:] = 0
    monkeypatch.setattr(dgp, "generate", iter(draws).__next__)
    config = BootstrapConfig(n_bootstrap_bias=10, n_bootstrap_ci=10)
    result = baseline(
        n_simulations=6, n_estimators=3, bootstrap_config=config, random_state=123
    ).validate(dgp)
    assert result.metadata["n_bias_simulations"] == 5
    # result.coverage is a share of all six draws, and the empty draw covers nothing.
    covered = round(result.coverage * 6)
    assert result.coverage_p_value == stats.binomtest(covered, 5, 0.95).pvalue
