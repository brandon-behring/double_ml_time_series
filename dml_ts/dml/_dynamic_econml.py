"""EconML reference backend for recursive dynamic g-estimation (optional).

Thin adapter over EconML's ``DynamicDML`` -- the reference implementation of
Lewis & Syrgkanis (2021) -- used to *cross-validate* the native
:class:`~dml_ts.dml.dynamic_g_estimation.DynamicGEstimationDML`. It returns the
same :class:`DynamicGEstimationResult` so the two are directly comparable. This
is a legitimate custom-vs-reference check, not the tautological EconML-vs-EconML
comparison the repo deleted earlier.

``econml`` is an optional dependency (the ``full`` extra). Importing this module
does not require it; only :func:`fit_econml_reference` does, and it raises a
clear, actionable error if it is missing.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator

from .dynamic_g_estimation import DynamicGEstimationResult

ArrayLike = np.ndarray | list[float]


def econml_available() -> bool:
    """Return True if EconML's ``DynamicDML`` can be imported."""
    try:
        from econml.panel.dml import DynamicDML  # noqa: F401

        return True
    except Exception:
        return False


def fit_econml_reference(
    Y: ArrayLike,
    T: ArrayLike,
    X: ArrayLike,
    groups: ArrayLike,
    alpha: float = 0.05,
    model_y: BaseEstimator | None = None,
    model_t: BaseEstimator | None = None,
    cv: int = 5,
    random_state: int | None = None,
) -> DynamicGEstimationResult:
    """Fit EconML ``DynamicDML`` and return a comparable result.

    The confounders ``X`` are passed to EconML as controls ``W`` (no
    heterogeneity features), so ``intercept_`` holds the constant per-period
    blips -- the same estimand as the native estimator.

    Args:
        Y, T: Outcome and treatment, stacked unit-major ``(n_units * m,)``.
        X: Confounders/controls ``(n_rows, p)`` (passed as EconML ``W``).
        groups: Unit id per row.
        alpha: Significance level.
        model_y, model_t: Nuisance models (default EconML's auto).
        cv: Cross-fitting folds.
        random_state: Seed.

    Returns:
        :class:`DynamicGEstimationResult` with ``backend="econml"``.

    Raises:
        ImportError: If ``econml`` is not installed (``pip install -e ".[full]"``).
    """
    try:
        from econml.panel.dml import DynamicDML
    except Exception as exc:  # pragma: no cover - exercised only without econml
        raise ImportError(
            "fit_econml_reference requires EconML. Install the optional extra:\n"
            '    uv pip install -e ".[full]"   (or: pip install econml)'
        ) from exc

    Y = np.asarray(Y, dtype=float).ravel()
    T = np.asarray(T, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    groups = np.asarray(groups)

    m = int(np.bincount(groups.astype(int)).max())
    n_units = int(len(np.unique(groups)))

    kwargs = {"cv": cv, "random_state": random_state}
    if model_y is not None:
        kwargs["model_y"] = model_y
    if model_t is not None:
        kwargs["model_t"] = model_t
    est = DynamicDML(**kwargs)
    # X=None -> constant (non-heterogeneous) blips; confounders go to W.
    est.fit(Y, T, X=None, W=X, groups=groups, inference="auto")

    theta = np.asarray(est.intercept_, dtype=float).reshape(-1)[:m]

    # Per-period inference: prefer EconML's interval accessor; derive se from it.
    z = stats.norm.ppf(1 - alpha / 2)
    try:
        lo, hi = est.intercept__interval(alpha=alpha)
        ci_lower = np.asarray(lo, dtype=float).reshape(-1)[:m]
        ci_upper = np.asarray(hi, dtype=float).reshape(-1)[:m]
        se = (ci_upper - ci_lower) / (2 * z)
    except Exception:  # pragma: no cover - depends on econml version
        inf = est.intercept__inference()
        se = np.asarray(inf.stderr, dtype=float).reshape(-1)[:m]
        ci_lower, ci_upper = theta - z * se, theta + z * se

    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, theta / se, 0.0)
        pvals = 2 * (1 - stats.norm.cdf(np.abs(tstat)))

    cov = np.diag(se**2)  # reference backend: diagonal approximation of the joint cov
    cum_se = float(np.sqrt(np.sum(se**2)))
    return DynamicGEstimationResult(
        theta_t=theta,
        se_t=se,
        ci_lower_t=ci_lower,
        ci_upper_t=ci_upper,
        p_value_t=pvals,
        cumulative_effect=float(theta.sum()),
        cumulative_se=cum_se,
        n_units=n_units,
        n_periods=m,
        alpha=alpha,
        cov=cov,
        nuisance_r2_t=np.full(m, np.nan),
        backend="econml",
        mode="panel",
    )
