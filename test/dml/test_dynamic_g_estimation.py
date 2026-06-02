"""Tests for recursive dynamic g-estimation (DynamicGEstimationDML).

Validation strategy:
- tier1: result shape/fields and input validation.
- tier2: the recursion recovers known blips exactly (deterministic), recovers
  the total blip under time-varying confounding, and the series adaptation works.
- tier3: Monte-Carlo bias + CI coverage of the joint sandwich variance, and
  numerical agreement with the EconML reference (skipped if econml absent).
"""

import numpy as np
import pytest

from dml_ts.dml import (
    DynamicGEstimationDML,
    DynamicGEstimationResult,
    econml_available,
    fit_econml_reference,
)
from dml_ts.validation import DynamicTreatmentDGP


@pytest.fixture
def deterministic_panel():
    """Identified (treatment noise) but noiseless outcome/state => exact recovery."""
    return DynamicTreatmentDGP(
        n_periods=3,
        theta_t=[1.0, 2.0, 3.0],
        n_units=400,
        p=2,
        state_noise=0.0,
        treatment_noise=1.0,
        outcome_noise=0.0,
        random_state=0,
    ).generate()


def _linear_estimator(**kw):
    return DynamicGEstimationDML(model_y="linear", model_t="linear", random_state=0, **kw)


@pytest.mark.tier1
class TestResultAndValidation:
    def test_fit_returns_result_with_shapes(self, deterministic_panel):
        d = deterministic_panel
        res = _linear_estimator().fit(d.Y, d.T, d.X, groups=d.groups)
        assert isinstance(res, DynamicGEstimationResult)
        assert res.theta_t.shape == (3,) and res.se_t.shape == (3,)
        assert res.ci_lower_t.shape == (3,) and res.ci_upper_t.shape == (3,)
        assert res.cov.shape == (3, 3)
        assert res.n_periods == 3 and res.n_units == 400 and res.backend == "custom"
        assert res.cumulative_effect == pytest.approx(res.theta_t.sum())

    def test_panel_requires_groups(self, deterministic_panel):
        d = deterministic_panel
        with pytest.raises(ValueError, match="groups"):
            _linear_estimator().fit(d.Y, d.T, d.X, mode="panel")

    def test_ragged_panel_rejected(self):
        y = np.zeros(5)
        t = np.zeros(5)
        x = np.zeros((5, 2))
        groups = np.array([0, 0, 1, 1, 1])
        with pytest.raises(ValueError, match="ragged"):
            _linear_estimator().fit(y, t, x, groups=groups)

    def test_series_requires_n_periods(self):
        with pytest.raises(ValueError, match="n_periods"):
            DynamicGEstimationDML(n_periods=None).fit(
                np.zeros(100), np.zeros(100), np.zeros((100, 2)), mode="series"
            )

    def test_mismatched_rows_rejected(self):
        with pytest.raises(ValueError, match="same number of rows"):
            _linear_estimator().fit(
                np.zeros(9), np.zeros(10), np.zeros((10, 2)), groups=np.arange(10)
            )

    def test_summary_renders(self, deterministic_panel):
        d = deterministic_panel
        res = _linear_estimator().fit(d.Y, d.T, d.X, groups=d.groups)
        text = res.summary()
        assert "Dynamic G-Estimation" in text and "Cumulative" in text


@pytest.mark.tier2
class TestRecovery:
    def test_exact_recovery_deterministic(self, deterministic_panel):
        d = deterministic_panel
        res = _linear_estimator(n_folds=5).fit(d.Y, d.T, d.X, groups=d.groups)
        np.testing.assert_allclose(res.theta_t, [1.0, 2.0, 3.0], atol=1e-6)

    def test_recovers_total_blip_under_time_varying_confounding(self):
        d = DynamicTreatmentDGP(
            n_periods=3,
            theta_t=1.0,
            n_units=4000,
            p=2,
            state_feedback=True,
            state_transition=0.5,
            treatment_state_coef=0.8,
            treatment_policy_coef=1.0,
            confounding_coef=1.0,
            treatment_noise=1.0,
            outcome_noise=0.5,
            random_state=3,
        ).generate()
        res = _linear_estimator(n_folds=5).fit(d.Y, d.T, d.X, groups=d.groups)
        np.testing.assert_allclose(res.theta_t, d.theta_t, atol=0.2)

    def test_series_mode_recovers_distributed_lag(self):
        s = DynamicTreatmentDGP(
            n_periods=2,
            theta_t=[1.0, 0.5],
            p=2,
            mode="series",
            series_length=3000,
            noise_level=0.5,
            treatment_noise=1.0,
            random_state=2,
        ).generate()
        est = DynamicGEstimationDML(n_periods=2, model_y="linear", model_t="linear", random_state=0)
        res = est.fit(s.Y, s.T, s.X, mode="series")
        assert res.mode == "series"
        np.testing.assert_allclose(res.theta_t, s.theta_t, atol=0.15)


@pytest.mark.tier3
class TestInferenceAndReference:
    def test_monte_carlo_bias_and_coverage(self):
        truth = np.array([1.0, -1.0, 2.0])
        m, reps = 3, 100
        ests = np.zeros((reps, m))
        covered = np.zeros((reps, m), dtype=bool)
        est = _linear_estimator(n_folds=4)
        for r in range(reps):
            d = DynamicTreatmentDGP(
                n_periods=m,
                theta_t=truth,
                n_units=600,
                p=2,
                noise_level=1.0,
                treatment_noise=1.0,
                outcome_noise=1.0,
                random_state=500 + r,
            ).generate()
            res = est.fit(d.Y, d.T, d.X, groups=d.groups)
            ests[r] = res.theta_t
            covered[r] = (res.ci_lower_t <= truth) & (truth <= res.ci_upper_t)
        bias = ests.mean(axis=0) - truth
        coverage = covered.mean(axis=0)
        assert np.abs(bias).max() < 0.05, f"bias too large: {bias}"
        assert (np.abs(coverage - 0.95) < 0.07).all(), f"coverage off nominal: {coverage}"

    @pytest.mark.skipif(
        not econml_available(), reason="econml not installed (optional '.[full]' extra)"
    )
    def test_numerical_agreement_with_econml(self):
        d = DynamicTreatmentDGP(
            n_periods=3,
            theta_t=[1.0, -0.5, 2.0],
            n_units=1500,
            p=3,
            noise_level=1.0,
            treatment_noise=1.0,
            outcome_noise=1.0,
            random_state=11,
        ).generate()
        custom = _linear_estimator(n_folds=5).fit(d.Y, d.T, d.X, groups=d.groups)
        reference = fit_econml_reference(d.Y, d.T, d.X, groups=d.groups, cv=5, random_state=0)
        # Both target the same estimand and recover the truth within sampling error.
        np.testing.assert_allclose(custom.theta_t, d.theta_t, atol=0.15)
        np.testing.assert_allclose(reference.theta_t, d.theta_t, atol=0.15)
        np.testing.assert_allclose(custom.theta_t, reference.theta_t, atol=0.12)
