"""Fail-loud regression tests for validation/ silent-failure paths (audit #41 PR-A).

Each test pins a path that previously fabricated a verdict-driving value (a p-value, a
normality flag) or silently dropped failures, instead of raising/warning — the repo's #1
rule ("never fail silently"; CONTRIBUTING #7/#9).
"""

from __future__ import annotations

import numpy as np
import pytest

from dml_ts.validation.bias_validation import BiasValidation
from dml_ts.validation.bootstrap_diagnostics import BootstrapDiagnostics
from dml_ts.validation.dgp_generator import DGPGenerator
from dml_ts.validation.dgp_generator_ts import TimeSeriesDGPGenerator


# ------------------------------------------------------------------- bias_validation t=0/p=1
@pytest.mark.tier1
def test_bias_validation_raises_on_degenerate_bootstrap() -> None:
    """Zero-variance bias bootstrap must raise, not fabricate p=0/p=1 driving the verdict."""
    validator = BiasValidation(n_simulations=100, random_state=42)
    ci_bounds = np.tile([1.9, 2.1], (100, 1))
    with pytest.raises(ValueError, match="[Dd]egenerate"):
        validator._determine_statistical_status(
            bias_samples=np.full(100, 0.5),  # identical samples -> se_bias == 0
            coverage=1.0,
            n_simulations=100,
            ci_bounds=ci_bounds,
            true_effect=2.0,
        )


@pytest.mark.tier1
def test_bias_validation_accepts_varied_bootstrap() -> None:
    """Sanity: with genuine sampling variance the t-test runs and returns a finite p."""
    validator = BiasValidation(n_simulations=100, random_state=42)
    ci_bounds = np.tile([1.9, 2.1], (100, 1))
    status, p_bias, _p_cov = validator._determine_statistical_status(
        bias_samples=-0.004 + np.linspace(-0.05, 0.05, 100),
        coverage=1.0,
        n_simulations=100,
        ci_bounds=ci_bounds,
        true_effect=2.0,
    )
    assert np.isfinite(p_bias)
    assert status in {"PASS", "WARNING", "FAIL"}


# --------------------------------------------------------------- bootstrap_diagnostics < 3 samples
@pytest.mark.tier1
def test_bootstrap_diagnose_distribution_raises_below_three() -> None:
    """< 3 bootstrap samples can't support Shapiro-Wilk → raise, not fabricate non-normal."""
    data = DGPGenerator(n=80, p=2, true_effect=1.0, random_state=0).generate()
    diag = BootstrapDiagnostics(data, estimator_type="OLS", random_state=0)
    with pytest.raises(ValueError, match="n_bootstrap >= 3"):
        diag.diagnose_distribution(n_bootstrap=2)


# ------------------------------------------------------------ insurance_dgp survivorship / narrowing
class _FakeResult:
    theta = 2.0
    se = 0.1
    ci_lower = 1.8
    ci_upper = 2.2


@pytest.mark.tier2
def test_validate_dgp_recovery_counts_failures_and_warns(monkeypatch) -> None:
    """A degenerate sim is excluded with a survivorship warning and counted in n_failed."""
    from dml_ts.validation import insurance_dgp as mod

    state = {"n": 0}

    class _FakePanel:
        def __init__(self, **kwargs) -> None:
            pass

        def fit(self, *args, **kwargs):
            state["n"] += 1
            if state["n"] == 1:
                raise ValueError("singular design")
            return _FakeResult()

    monkeypatch.setattr("dml_ts.dml.PanelDML", _FakePanel)

    with pytest.warns(RuntimeWarning, match="survivorship"):
        out = mod.validate_dgp_recovery(
            realism="simple", n_periods=60, n_products=5, n_sims=4, seed=0
        )
    assert out["n_failed"] == 1
    assert out["n_successful"] == 3


@pytest.mark.tier2
def test_validate_dgp_recovery_propagates_unexpected_errors(monkeypatch) -> None:
    """A non-(LinAlgError/ValueError) is a genuine bug and must surface, not be swallowed."""
    from dml_ts.validation import insurance_dgp as mod

    class _FakePanel:
        def __init__(self, **kwargs) -> None:
            pass

        def fit(self, *args, **kwargs):
            raise KeyError("genuine bug")

    monkeypatch.setattr("dml_ts.dml.PanelDML", _FakePanel)

    with pytest.raises(KeyError):
        mod.validate_dgp_recovery(realism="simple", n_periods=60, n_products=5, n_sims=2, seed=0)


# ----------------------------------------------------------------- dgp_generator_ts AR / floor
@pytest.mark.tier1
def test_timeseries_dgp_rejects_explosive_user_ar() -> None:
    """An explosive user AR ([0.6, 0.6]) must raise — a length check alone would accept it."""
    with pytest.raises(ValueError, match="non-stationary"):
        TimeSeriesDGPGenerator(
            n_periods=50,
            p=2,
            true_effect=1.0,
            treatment_ar_order=2,
            treatment_ar_coefs=np.array([0.6, 0.6]),
        )


@pytest.mark.tier1
def test_timeseries_dgp_accepts_stationary_user_ar() -> None:
    """A stationary user AR ([0.5]) constructs without raising or warning about the floor."""
    gen = TimeSeriesDGPGenerator(
        n_periods=50,
        p=2,
        true_effect=1.0,
        treatment_ar_order=1,
        treatment_ar_coefs=np.array([0.5]),
    )
    assert gen.treatment_ar_coefs.shape == (1,)


@pytest.mark.tier1
def test_timeseries_dgp_warns_on_near_unit_root_confounder() -> None:
    """A near-unit-root confounder triggers the variance floor → must warn, not silently fabricate."""
    with pytest.warns(RuntimeWarning, match="near unit-root"):
        TimeSeriesDGPGenerator(n_periods=50, p=2, true_effect=1.0, confounder_var_coef=0.999)
