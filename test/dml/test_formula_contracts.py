"""Exact-value formula-contract + fail-loud unit tests for the core estimators (audit #43).

The core SE/theta formulas were tested only INDIRECTLY (Monte-Carlo brackets), so a wrong
constant or ``ddof`` would ship undetected. Each test here pins the estimator's output against
an INDEPENDENT textbook re-derivation at ``rtol<=1e-10`` (mirroring the exemplary
``test_cluster_se_matches_cr1_formula`` / ``test_hac_se_consistency_with_hac_module`` in
``test_temporal_plr_dml.py``), and pins each degenerate path to a loud raise (the repo's #1 rule).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from dml_ts.dml.double_ml import _influence_function_se, double_ml
from dml_ts.dml.dynamic_g_estimation import DynamicGEstimationDML, _bartlett_long_run_cov
from dml_ts.dml.fwl import fwl_estimate
from dml_ts.dml.robinson import robinson_estimator

pytestmark = pytest.mark.tier1


# --------------------------------------------------------------------- DML influence-function SE
def test_dml_influence_function_se_matches_closed_form() -> None:
    """``_influence_function_se`` == sqrt(var(psi, ddof=1)/n), psi=(Ỹ-θT̃)·T̃/mean(T̃²)."""
    Y_tilde = np.array([1.0, -2.0, 0.5, 3.0, -1.5, 2.0, -0.25, 1.75])
    T_tilde = np.array([0.5, 1.0, -1.0, 2.0, -0.5, 1.5, -2.0, 0.75])
    theta = 1.3

    se, psi = _influence_function_se(Y_tilde, T_tilde, theta)

    mean_t_sq = np.mean(T_tilde**2)
    psi_expected = (Y_tilde - theta * T_tilde) * T_tilde / mean_t_sq
    se_expected = np.sqrt(np.var(psi_expected, ddof=1) / len(Y_tilde))

    assert_allclose(psi, psi_expected, rtol=1e-12)
    assert_allclose(se, se_expected, rtol=1e-12)


def test_double_ml_inference_derives_from_theta_and_se() -> None:
    """Headline inference (psi, se, t, CI, p) all derive from the documented closed forms."""
    rng = np.random.default_rng(7)
    n = 400
    X = rng.standard_normal((n, 3))
    T = 0.5 * X[:, 0] + rng.standard_normal(n)
    Y = 2.0 * T + X @ np.array([1.0, 0.5, -0.3]) + rng.standard_normal(n)

    result = double_ml(Y, T, X, model="ridge", n_folds=5, alpha=0.05)

    # psi and se reproduce the influence-function closed form from the exposed residuals.
    mean_t_sq = np.mean(result.T_residual**2)
    psi_expected = (
        (result.Y_residual - result.theta * result.T_residual) * result.T_residual / mean_t_sq
    )
    assert_allclose(result.influence_scores, psi_expected, rtol=1e-10)
    assert_allclose(result.se, np.sqrt(np.var(psi_expected, ddof=1) / n), rtol=1e-10)

    # t-stat, CI, p-value are the standard normal-theory transforms of (theta, se).
    z = stats.norm.ppf(0.975)
    assert_allclose(result.t_stat, result.theta / result.se, rtol=1e-12)
    assert_allclose(result.ci_lower, result.theta - z * result.se, rtol=1e-12)
    assert_allclose(result.ci_upper, result.theta + z * result.se, rtol=1e-12)
    assert_allclose(
        result.p_value, 2 * (1 - stats.norm.cdf(abs(result.t_stat))), rtol=1e-9, atol=1e-12
    )


# --------------------------------------------------------------------- FWL homoskedastic SE / theta
def test_fwl_se_and_theta_match_textbook_ols() -> None:
    """FWL θ and homoskedastic SE equal OLS on the full design ``D=[1, T, X]``.

    Independent re-derivation: ``se_T = sqrt(σ̂²·inv(DᵀD)[1,1])`` with ``σ̂² = RSS/(n-k)``.
    The partitioned-inverse identity ``inv(DᵀD)[1,1] == 1/ΣT̃²`` is exactly what ``fwl.py``
    exploits, so a wrong ``ddof``/denominator or a wrong df breaks this pin.
    """
    rng = np.random.default_rng(11)
    n = 250
    X = rng.standard_normal((n, 3))
    T = rng.standard_normal(n)
    Y = 1.5 * T + X @ np.array([0.7, -0.4, 0.2]) + rng.standard_normal(n)

    result = fwl_estimate(Y, T, X)

    design = np.column_stack([np.ones(n), T, X])
    beta, *_ = np.linalg.lstsq(design, Y, rcond=None)
    resid = Y - design @ beta
    k = design.shape[1]
    sigma_sq = float(resid @ resid) / (n - k)
    se_t = np.sqrt(sigma_sq * np.linalg.inv(design.T @ design)[1, 1])

    assert_allclose(result.theta, beta[1], rtol=1e-10)
    assert_allclose(result.se, se_t, rtol=1e-10)


# --------------------------------------------------------------------------- Robinson naive SE
def test_robinson_naive_se_matches_closed_form() -> None:
    """Robinson naive SE == sqrt(var(Ỹ-θT̃, ddof=1)/ΣT̃²), recomputed from exposed residuals."""
    rng = np.random.default_rng(13)
    n = 300
    X = rng.standard_normal((n, 2))
    T = 0.5 * X[:, 0] + rng.standard_normal(n)
    Y = 1.5 * T + X[:, 1] + rng.standard_normal(n)

    result = robinson_estimator(Y, T, X, model="ridge")  # ridge nuisance is deterministic

    Y_tilde, T_tilde = result.Y_residual, result.T_residual
    theta_expected = float(Y_tilde @ T_tilde) / float(T_tilde @ T_tilde)
    final_resid = Y_tilde - result.theta * T_tilde
    se_expected = np.sqrt(np.var(final_resid, ddof=1) / np.sum(T_tilde**2))

    assert_allclose(result.theta, theta_expected, rtol=1e-10)
    assert_allclose(result.se, se_expected, rtol=1e-10)


# ----------------------------------------------------------------------- dynamic-g sandwich SE
def test_dynamic_g_cumulative_se_is_full_quadratic_form() -> None:
    """cumulative_se == sqrt(1ᵀ·cov·1) — NOT sqrt(Σ diag), which drops cross-period covariance."""
    rng = np.random.default_rng(0)
    n_units, m = 150, 3
    groups = np.repeat(np.arange(n_units), m)
    X = rng.standard_normal((n_units * m, 1))
    T = 0.5 * X[:, 0] + rng.standard_normal(n_units * m)
    Y = 1.0 * T + X[:, 0] + rng.standard_normal(n_units * m)

    result = DynamicGEstimationDML(model_y="ridge", model_t="ridge", n_folds=3, random_state=0).fit(
        Y, T, X, groups=groups
    )

    ones = np.ones(result.n_periods)
    cum_se_full = np.sqrt(float(ones @ result.cov @ ones))
    assert_allclose(result.cumulative_se, cum_se_full, rtol=1e-10)

    # Cross-period covariance is non-zero here, so the full quadratic form genuinely differs
    # from the diagonal-only sum — proving the pin discriminates the #11-class bug.
    assert abs(float(result.cov[0, -1])) > 1e-12
    diag_only = np.sqrt(float(np.sum(np.diag(result.cov))))
    assert not np.isclose(result.cumulative_se, diag_only)


def test_bartlett_long_run_cov_matches_kernel_definition() -> None:
    """Direct re-derivation of S = Γ₀ + Σ wⱼ(Γⱼ+Γⱼᵀ), wⱼ=1-j/(bw+1), on demeaned scores."""
    rng = np.random.default_rng(3)
    scores = rng.standard_normal((60, 2))
    bw = 4

    s_long_run = _bartlett_long_run_cov(scores, bw)

    s = scores - scores.mean(axis=0, keepdims=True)
    t = scores.shape[0]
    expected = (s.T @ s) / t
    for j in range(1, bw + 1):
        gamma_j = (s[j:].T @ s[:-j]) / t
        weight = 1.0 - j / (bw + 1)
        expected += weight * (gamma_j + gamma_j.T)

    assert_allclose(s_long_run, expected, rtol=1e-12)


# ----------------------------------------------------------------- fail-loud on degenerate inputs
def test_double_ml_raises_on_zero_variation_treatment() -> None:
    """A constant treatment has no residual variation → DML must raise, not divide by ~0."""
    rng = np.random.default_rng(0)
    n = 200
    X = rng.standard_normal((n, 2))
    T = np.ones(n)
    Y = rng.standard_normal(n)
    with pytest.raises(ValueError, match="no variation"):
        double_ml(Y, T, X, model="ridge", n_folds=3)


def test_robinson_raises_on_zero_variation_treatment() -> None:
    rng = np.random.default_rng(0)
    n = 200
    X = rng.standard_normal((n, 2))
    T = np.ones(n)
    Y = rng.standard_normal(n)
    with pytest.raises(ValueError, match="no variation"):
        robinson_estimator(Y, T, X, model="ridge")


def test_dynamic_g_raises_on_weak_colinear_treatment() -> None:
    """A constant treatment leaves zero residual variation at a period → recursion must raise."""
    rng = np.random.default_rng(0)
    n_units, m = 80, 2
    groups = np.repeat(np.arange(n_units), m)
    X = rng.standard_normal((n_units * m, 1))
    T = np.ones(n_units * m)
    Y = rng.standard_normal(n_units * m)
    with pytest.raises(ValueError, match="weak/colinear"):
        DynamicGEstimationDML(model_y="ridge", model_t="ridge", n_folds=3, random_state=0).fit(
            Y, T, X, groups=groups
        )
