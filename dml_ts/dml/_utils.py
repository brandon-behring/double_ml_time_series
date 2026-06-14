"""Shared DML-layer helpers (Track B1b).

Consolidates the near-copies that lived in ``double_ml``, ``robinson``, and
``temporal_plr_dml``: the nuisance-model factory, cross-validated R², the FWL
closed form with its no-variation guard, the cross-fit prediction loop, and
Y/T/X length validation. Intentional differences are parameterized, never
papered over: per-estimator no-variation messages, per-caller ``random_state``
(``double_ml``/``robinson`` historically pin the nuisance seed to 42
independently of their fold seed — preserved at their call sites). One
DELIBERATE widening: ``double_ml``/``robinson`` now accept ``"lasso"``
(they previously raised), aligning all consumers on this factory's full
option set; their public signatures document it.

``fwl.py`` (pure OLS/QR, no nuisance models) and ``dynamic_g_estimation.py``
(its own deterministic linear/ridge reference factory) deliberately keep
their own machinery.
"""

import os
from collections.abc import Iterable
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge

# Configurable parallelism: default uses all cores; tests set DML_N_JOBS=1.
_DEFAULT_N_JOBS = int(os.environ.get("DML_N_JOBS", "-1"))

ModelType = Literal["ridge", "lasso", "random_forest", "gradient_boosting"]


def get_nuisance_model(
    model_type: ModelType,
    random_state: int | None = None,
) -> BaseEstimator:
    """Get a nuisance model for E[Y|X] or E[T|X] estimation.

    Args:
        model_type: Type of model to use.
        random_state: Random seed for the stochastic learners. ``double_ml``
            and ``robinson`` pass 42 (their historical hardcoded pin);
            ``TemporalPLRDML`` passes its configured seed.

    Returns:
        Unfitted sklearn estimator with fit() and predict() methods.

    Raises:
        ValueError: If model_type is not recognized.
    """
    if model_type == "ridge":
        return Ridge(alpha=1.0)
    elif model_type == "lasso":
        return Lasso(alpha=0.1, max_iter=2000)
    elif model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=_DEFAULT_N_JOBS,
        )
    elif model_type == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=random_state,
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            "Choose from: 'ridge', 'lasso', 'random_forest', 'gradient_boosting'"
        )


def compute_r2(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Compute R² (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0


def theta_via_fwl(
    Y_tilde: NDArray[np.float64],
    T_tilde: NDArray[np.float64],
    *,
    no_variation_msg: str,
) -> tuple[float, float]:
    """FWL closed form ``theta = sum(Y_tilde*T_tilde) / sum(T_tilde**2)``.

    Args:
        Y_tilde: Outcome residuals.
        T_tilde: Treatment residuals.
        no_variation_msg: Estimator-specific ValueError message raised when
            the treatment residuals carry (numerically) no variation.

    Returns:
        Tuple of ``(theta, T_tilde_sq_sum)`` — the sum is returned because
        callers reuse it for their variance constructions.
    """
    T_tilde_sq_sum = float(np.sum(T_tilde**2))
    if not np.isfinite(T_tilde_sq_sum):
        raise ValueError(
            "Non-finite treatment residuals (sum of squares = "
            f"{T_tilde_sq_sum}); check for NaN/inf in inputs or uncovered "
            "cross-fit rows."
        )
    if T_tilde_sq_sum < 1e-10:
        raise ValueError(no_variation_msg)
    numerator = float(np.sum(Y_tilde * T_tilde))
    if not np.isfinite(numerator):
        # Symmetric to the treatment-side guard above: a NaN/inf on the OUTCOME
        # side must not silently produce a NaN theta that rides downstream.
        raise ValueError(
            "Non-finite outcome residuals (Y_tilde·T_tilde sum = "
            f"{numerator}); check for NaN/inf in inputs or uncovered cross-fit rows."
        )
    theta = numerator / T_tilde_sq_sum
    return theta, T_tilde_sq_sum


def cross_fit_predictions(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    T: NDArray[np.float64],
    splits: Iterable[tuple[NDArray[np.int64], NDArray[np.int64]]],
    outcome_model: BaseEstimator,
    treatment_model: BaseEstimator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cross-fitted out-of-fold predictions over an injected split source.

    For each fold, clones and trains the models on the train indices and
    predicts on the test indices. Rows never covered by a test fold remain
    NaN — temporal expanding-window CV leaves an uncovered prefix that the
    caller must exclude rather than backfill. For full-coverage K-fold, an
    uncovered row propagates NaN into the residuals, where theta_via_fwl's
    isfinite guard raises (the old zeros-init silently passed a fabricated
    0.0 prediction downstream instead).

    Returns:
        Tuple of ``(Y_hat, T_hat)`` predictions, NaN where uncovered.
    """
    n = len(Y)
    Y_hat = np.full(n, np.nan)
    T_hat = np.full(n, np.nan)

    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]

        outcome_mod = clone(outcome_model)
        outcome_mod.fit(X_train, Y[train_idx])
        Y_hat[test_idx] = outcome_mod.predict(X_test)

        treatment_mod = clone(treatment_model)
        treatment_mod.fit(X_train, T[train_idx])
        T_hat[test_idx] = treatment_mod.predict(X_test)

    return Y_hat, T_hat


def validate_lengths(
    Y: NDArray[np.float64], T: NDArray[np.float64], X: NDArray[np.float64]
) -> None:
    """Y/T/X first-dimension agreement; messages are test-pinned."""
    n = len(Y)
    if len(T) != n:
        raise ValueError(f"T length ({len(T)}) must match Y length ({n})")
    if X.shape[0] != n:
        raise ValueError(f"X rows ({X.shape[0]}) must match Y length ({n})")
