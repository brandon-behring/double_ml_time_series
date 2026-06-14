"""Fail-loud hardening regression tests (multi-reviewer audit, 2026-06).

Each test pins a degenerate path that previously returned a plausible-but-wrong
value or empty/garbage output instead of raising/warning — the repo's #1 rule
("never fail silently"; CONTRIBUTING #7/#9).
"""

import warnings

import numpy as np
import pytest

from dml_ts.dml import RollingWindowDML, TemporalPLRDML
from dml_ts.dml._utils import theta_via_fwl
from dml_ts.dml.dynamic_g_estimation import _warn_if_ill_conditioned
from dml_ts.dml.fwl import fwl_residualize
from dml_ts.dml.inference import hac_inference

pytestmark = pytest.mark.tier1


def test_fwl_residualize_qr_raises_on_rank_deficient_controls() -> None:
    """QR residualization must raise, not silently over-project, on collinear X."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    X = np.column_stack([np.ones(200), x, x])  # rank 2 in 3 columns
    Y = rng.standard_normal(200)
    with pytest.raises(ValueError, match="[Rr]ank-deficient"):
        fwl_residualize(Y, X, method="qr")
    # the rank-safe "ols" path still returns finite residuals
    resid, _ = fwl_residualize(Y, X, method="ols")
    assert np.isfinite(resid).all()
    # a full-rank but SCALE-DISPARATE design must NOT raise (scale != rank deficiency)
    X_scaled = np.column_stack(
        [np.ones(200), 1e-4 * rng.standard_normal(200), 1e8 * rng.standard_normal(200)]
    )
    resid_qr, _ = fwl_residualize(Y, X_scaled, method="qr")
    assert np.isfinite(resid_qr).all()


def test_hac_inference_rejects_nonfinite_theta_and_bad_alpha() -> None:
    with pytest.raises(ValueError, match="theta must be finite"):
        hac_inference(theta=float("nan"), se_hac=0.5)
    for bad_alpha in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            hac_inference(theta=2.0, se_hac=0.5, alpha=bad_alpha)
    out = hac_inference(theta=2.0, se_hac=0.5, alpha=0.05)
    assert out["p_value"] < 0.001


def test_theta_via_fwl_rejects_nonfinite_outcome_residuals() -> None:
    """Symmetric to the treatment-side guard: a NaN on the OUTCOME side must raise."""
    T = np.array([1.0, -1.0, 2.0, -2.0, 0.5])
    Y = np.array([1.0, np.nan, 2.0, -1.0, 0.0])
    with pytest.raises(ValueError, match="[Nn]on-finite outcome residuals"):
        theta_via_fwl(Y, T, no_variation_msg="no variation")


def test_temporal_plr_dml_rejects_invalid_n_lags() -> None:
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError, match="n_lags"):
            TemporalPLRDML(n_lags=bad)


def test_rolling_window_raises_when_window_exceeds_series() -> None:
    rng = np.random.default_rng(0)
    n = 50
    Y = rng.standard_normal(n)
    T = rng.standard_normal(n)
    X = rng.standard_normal((n, 2))
    with pytest.raises(ValueError, match="too large"):
        RollingWindowDML(window_size=100, step_size=10).fit(Y, T, X)


def test_warn_if_ill_conditioned() -> None:
    with pytest.warns(RuntimeWarning, match="ill-conditioned"):
        _warn_if_ill_conditioned(np.array([[1.0, 0.0], [0.0, 1e-16]]), "test matrix")
    # a well-conditioned matrix must not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_ill_conditioned(np.eye(2), "test matrix")
