"""The six baselines judge interval coverage as the DML validator does.

They classified the bias p-value alone, so ``coverage_status`` was ``None`` and the overall
status ignored the intervals: ``OLSWithControls.validate()`` returned PASS with 76% of its 95%
intervals covering the truth, and ``NaiveOLS`` returned PASS with 34%.
"""

import sys
import warnings

import numpy as np
import pytest
from scipy import stats

from dml_ts.validation._bootstrap_folds import original_fold_ids
from dml_ts.validation.bias_validation import (
    SimulationVerdicts,
    assess_simulation_draws,
    classify_bias_and_coverage,
)
from dml_ts.validation.bootstrap_config import BootstrapConfig
from dml_ts.validation.bootstrap_diagnostics import BootstrapDiagnostics
from dml_ts.validation.dgp_generator import DGPGenerator
from dml_ts.validation.enhanced_dgp import EnhancedDGPGenerator
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
@pytest.mark.parametrize("baseline", BASELINES)
def test_every_baseline_passes_its_level_in_and_the_two_verdicts_out(baseline, monkeypatch):
    """Each baseline hands its own nominal level to the assessment and reports what comes back.

    The verdicts are forced apart (bias PASS, coverage FAIL), so a baseline that reported the
    bias verdict as its status, or dropped a field, is seen.
    """
    seen = {}

    def forced(errors, bounds, true_effect, nominal):
        seen.update(rows=len(errors), bounds=np.shape(bounds), truth=true_effect, nominal=nominal)
        return SimulationVerdicts("PASS", "FAIL"), 0.7, 0.001

    monkeypatch.setattr(sys.modules[baseline.__module__], "assess_simulation_draws", forced)
    dgp = DGPGenerator(n=150, p=2, true_effect=2.0, random_state=7)
    config = BootstrapConfig(n_bootstrap_bias=20, n_bootstrap_ci=20)
    result = baseline(n_simulations=6, alpha=0.1, bootstrap_config=config, random_state=7).validate(
        dgp
    )
    assert seen == {"rows": 6, "bounds": (6, 2), "truth": 2.0, "nominal": pytest.approx(0.9)}
    assert (result.status, result.bias_status, result.coverage_status) == ("FAIL", "PASS", "FAIL")
    assert (result.bias_p_value, result.coverage_p_value) == (0.7, 0.001)


@pytest.mark.tier2
@pytest.mark.parametrize("baseline", [RandomForestEstimator, XGBoostEstimator])
def test_a_draw_with_an_empty_arm_is_left_out_of_every_number_and_named(baseline, monkeypatch):
    """Bias, MSE, coverage and both tests use the fitted draws, and the result says so.

    The reported coverage was a share of all draws while the test counted the fitted ones:
    with 5 of 12 intervals covering and 2 empty draws it read 41.7% beside a test of 5 in 10.
    """
    dgp = DGPGenerator(n=100, p=2, true_effect=2.0, random_state=123)
    draws = [dgp.generate() for _ in range(6)]
    draws[0].T[:] = 0
    monkeypatch.setattr(dgp, "generate", iter(draws).__next__)
    config = BootstrapConfig(n_bootstrap_bias=10, n_bootstrap_ci=10)
    estimator = baseline(n_simulations=6, n_estimators=3, bootstrap_config=config, random_state=123)
    with pytest.warns(RuntimeWarning, match="1 of 6 simulations had an empty treatment arm"):
        result = estimator.validate(dgp)
    assert result.metadata["n_bias_simulations"] == 5
    covered = result.coverage * 5
    assert covered == pytest.approx(round(covered))
    assert result.coverage_p_value == stats.binomtest(round(covered), 5, 0.95).pvalue


# ---------------------------------------------------------------- the rule and the levels


@pytest.mark.tier1
@pytest.mark.parametrize(
    "bias,p_bias,p_coverage,epsilon,verdict",
    [
        (0.16, 0.80, 0.80, 0.10, "FAIL"),  # above 0.15: FAIL with no test
        (0.14, 0.80, 0.80, 0.10, "PASS"),  # below 0.15 and not rejected
        (0.12, 0.001, 0.80, 0.10, "FAIL"),  # at least epsilon, rejected at the 1% level
        (0.12, 0.001, 0.80, 0.13, "WARNING"),  # the same bias is below a larger epsilon
        (0.12, 0.006, 0.80, 0.10, "WARNING"),  # rejected at 5% (0.025) and not at 1% (0.005)
        (0.12, 0.020, 1e-9, 0.10, "WARNING"),  # a strict rejection of coverage is not one of bias
        (0.02, 0.001, 0.80, 0.10, "WARNING"),  # significant and small
    ],
)
def test_bias_verdict_follows_each_clause_of_the_rule(bias, p_bias, p_coverage, epsilon, verdict):
    verdicts = classify_bias_and_coverage(
        bias, p_bias, 0.95, 0.95, p_coverage, practical_epsilon=epsilon
    )
    assert verdicts.bias == verdict


@pytest.mark.tier1
@pytest.mark.parametrize(
    "truth,nominal,message",
    [
        (float("nan"), 0.95, "true_effect must be finite"),
        (float("inf"), 0.95, "true_effect must be finite"),
        (0.0, 1.0, "nominal coverage must be in"),
        (0.0, 0.0, "nominal coverage must be in"),
    ],
)
def test_assessment_refuses_a_truth_or_a_level_it_cannot_use(truth, nominal, message):
    errors = centered_errors(50)
    with pytest.raises(ValueError, match=message):
        assess_simulation_draws(errors, intervals_covering(errors, 48), truth, nominal=nominal)


@pytest.mark.tier2
def test_a_simulation_without_noise_is_refused_instead_of_failed():
    """An exact estimator on a noise-free generator leaves errors of order 1e-15.

    The t-test and the coverage count then judged rounding residue: status FAIL with 28% to
    36% coverage for an estimator with no error at all.
    """
    dgp = DGPGenerator(n=200, p=3, true_effect=2.0, noise_level=0.0, random_state=0)
    with pytest.raises(ValueError, match="rounding level of the true effect"):
        OLSWithControls(n_simulations=50, random_state=0).validate(dgp)


@pytest.mark.tier1
def test_rounding_guard_keeps_small_real_errors():
    """Errors of 1e-6 of the true effect are noise to be tested, not rounding."""
    errors = 2e-6 * centered_errors(50)
    verdicts, _, _ = assess_simulation_draws(errors, intervals_covering(errors, 48), 2.0, 0.95)
    assert verdicts.bias == "PASS"


@pytest.mark.tier2
@pytest.mark.parametrize("baseline", BASELINES)
def test_interval_and_bias_interval_follow_alpha(baseline, monkeypatch):
    """Both intervals narrow when alpha goes from 0.05 to 0.2 (normal ratio 0.654)."""
    widths = {}

    def recording(errors, bounds, true_effect, nominal):
        widths[round(nominal, 2)] = float(np.mean(np.diff(np.asarray(bounds), axis=1)))
        return SimulationVerdicts("PASS", "PASS"), 0.5, 0.5

    monkeypatch.setattr(sys.modules[baseline.__module__], "assess_simulation_draws", recording)
    trees = baseline in (RandomForestEstimator, XGBoostEstimator)
    config = BootstrapConfig(n_bootstrap_bias=400, n_bootstrap_ci=25 if trees else 200)
    extra = {"n_estimators": 3} if trees else {}
    bias_widths = {}
    for alpha in (0.05, 0.2):
        dgp = DGPGenerator(n=150, p=2, true_effect=2.0, random_state=7)
        result = baseline(
            n_simulations=5 if trees else 30,
            alpha=alpha,
            bootstrap_config=config,
            random_state=7,
            **extra,
        ).validate(dgp)
        bias_widths[alpha] = result.ci_upper - result.ci_lower
    # A level fixed at 95% gives a ratio of exactly 1; the bounds leave room for bootstrap noise.
    assert 0.4 < widths[0.8] / widths[0.95] < 0.9
    assert 0.4 < bias_widths[0.2] / bias_widths[0.05] < 0.9


# ---------------------------------------------------------------- folds and generators


@pytest.mark.tier1
def test_stratified_folds_spread_a_rare_treatment_over_every_fold():
    """Five treated units in 25: stratified folds hold one each; plain folds need not."""
    treatment = np.array([1.0] * 5 + [0.0] * 20)
    stratified = original_fold_ids(treatment, seed=0, stratified=True)
    plain = original_fold_ids(treatment, seed=0, stratified=False)
    assert [int(treatment[stratified == k].sum()) for k in range(5)] == [1, 1, 1, 1, 1]
    assert sorted(int(treatment[plain == k].sum()) for k in range(5)) != [1, 1, 1, 1, 1]


@pytest.mark.tier2
def test_bootstrap_dml_fit_receives_the_folds_of_the_original_observations(monkeypatch):
    """The mapped folds must reach the estimator: no observation is scored by a fit that saw it.

    The existing test replaces ``_estimate_dml``, the one place where the folds are handed
    to EconML, so it passes when that method ignores them.
    """
    econml_dml = pytest.importorskip("econml.dml")
    seen = []

    class Recorder:
        def __init__(self, **kwargs):
            self.folds = kwargs["cv"]

        def fit(self, Y, T, X):
            seen.append((self.folds, np.asarray(Y)))
            return self

        def effect(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr(econml_dml, "LinearDML", Recorder)
    data = DGPGenerator(n=60, p=2, true_effect=1.0, random_state=0).generate()
    diag = BootstrapDiagnostics(data, estimator_type="LinearDML", random_state=0)
    diag._bootstrap_estimates(4)
    assert len(seen) == 4
    for folds, outcome in seen:
        assert isinstance(folds, list) and 2 <= len(folds) <= 5
        assert len(set(outcome)) < 60  # a draw with replacement repeats observations
        assert sorted(np.concatenate([test for _, test in folds])) == list(range(60))
        for train, test in folds:
            # Copies of one observation share its outcome value, so no value may sit on both sides.
            assert not set(outcome[train]) & set(outcome[test])


@pytest.mark.tier1
def test_omitted_covariates_without_strength_warn_that_they_are_inert():
    with pytest.warns(RuntimeWarning, match="omitted covariates enter neither"):
        EnhancedDGPGenerator(n=50, p=2, base_effect=2.0, p_omitted=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        EnhancedDGPGenerator(n=50, p=2, base_effect=2.0, p_omitted=3, omitted_var_strength=1.0)
        EnhancedDGPGenerator(n=50, p=2, base_effect=2.0)
