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
from sklearn.base import BaseEstimator, clone, is_classifier
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


_EPS = float(np.finfo(float).eps)

# X explains all but 1e-12 of the treatment's variance: (1e-6)**2.
_UNEXPLAINED_NORM_RATIO = 1e-6

# Residuals smaller than sqrt(eps) of the treatment's level cannot be told from
# the rounding of an ill-conditioned fit. R's lm() uses 1e-7 for the same test.
_LEVEL_NORM_RATIO = float(np.sqrt(_EPS))


def is_numerically_constant(values: NDArray[np.float64]) -> bool:
    """Whether the spread of ``values`` is rounding residue of subtracting their mean.

    The mean of a constant array need not equal the constant (``np.full(100, 0.1)``
    has a mean one ulp away), so the centered values are compared with the level
    of the array and never with exact zero.
    """
    values = np.asarray(values, dtype=float)
    spread = np.linalg.norm(values - values.mean())
    return bool(spread <= np.linalg.norm(values) * values.size * _EPS)


def no_variation_tolerance(treatment: NDArray[np.float64]) -> float:
    """Largest norm of treatment residuals that counts as no residual variation.

    The bound is the larger of two terms, both in the units of the treatment:

    - ``1e-6 * norm(T - mean(T))``: the controls explain all but ``1e-12`` of
      the treatment's variance. This term does not change when a constant is
      added to the treatment.
    - ``sqrt(eps) * norm(T)``: the residuals are below ``1.5e-8`` of the
      treatment's level. A constant treatment leaves residuals of this kind,
      and so does a treatment whose variation is that small beside its level.

    The rounding error of a fit grows with the level and the conditioning of the
    controls, not with ``norm(T)`` alone, so a bound of a few ulps of ``norm(T)``
    accepts exact collinearity whenever the controls have a non-zero mean.

    Raises:
        ValueError: If the treatment holds a NaN or an infinity.
    """
    treatment = np.asarray(treatment, dtype=float)
    explained = _UNEXPLAINED_NORM_RATIO * np.linalg.norm(treatment - treatment.mean())
    level = _LEVEL_NORM_RATIO * np.linalg.norm(treatment)
    tolerance = float(max(explained, level))
    if not np.isfinite(tolerance):
        raise ValueError("Non-finite treatment values; check for NaN/inf in the treatment.")
    return tolerance


def compute_r2(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Compute R² (coefficient of determination).

    Returns 0.0 when ``y_true`` is numerically constant, where R² is undefined.
    """
    if is_numerically_constant(y_true):
        return 0.0
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def theta_via_fwl(
    Y_tilde: NDArray[np.float64],
    T_tilde: NDArray[np.float64],
    *,
    treatment: NDArray[np.float64] | None = None,
    no_variation_msg: str,
) -> tuple[float, float]:
    """FWL closed form ``theta = sum(Y_tilde*T_tilde) / sum(T_tilde**2)``.

    Args:
        Y_tilde: Outcome residuals.
        T_tilde: Treatment residuals.
        treatment: Treatment values before residualization on the same rows.
            These supply the units for distinguishing a numerically exact fit
            from residuals measured on a small scale; see
            :func:`no_variation_tolerance`. Without them only residuals that
            are exactly zero raise.
        no_variation_msg: Estimator-specific ValueError message raised when
            the treatment residuals carry (numerically) no variation. The
            residual norm and the bound are appended to it.

    Returns:
        Tuple of ``(theta, T_tilde_sq_sum)`` — the sum is returned because
        callers reuse it for their variance constructions.

    Raises:
        ValueError: If the two residual arrays are not one-dimensional and of
            equal shape; if either holds a NaN or an infinity; or if the
            treatment residuals carry no variation.
    """
    y_shape, t_shape = np.shape(Y_tilde), np.shape(T_tilde)
    if len(y_shape) != 1 or y_shape != t_shape:
        # Shapes (n, 1) and (n,) broadcast to an (n, n) product, and the slope
        # computed from it is wrong with no error (#72).
        raise ValueError(
            "Outcome and treatment residuals must be one-dimensional and of equal shape, "
            f"got {y_shape} and {t_shape}."
        )
    T_tilde_sq_sum = float(np.sum(T_tilde**2))
    if not np.isfinite(T_tilde_sq_sum):
        raise ValueError(
            "Non-finite treatment residuals (sum of squares = "
            f"{T_tilde_sq_sum}); check for NaN/inf in inputs or uncovered "
            "cross-fit rows."
        )
    residual_norm = float(np.sqrt(T_tilde_sq_sum))
    tolerance = 0.0 if treatment is None else no_variation_tolerance(treatment)
    if residual_norm <= tolerance:
        raise ValueError(
            f"{no_variation_msg} (norm of the treatment residuals {residual_norm:.3g} "
            f"<= bound {tolerance:.3g})"
        )
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


def predict_conditional_mean(model: BaseEstimator, X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Predict a numeric mean, including probability-weighted numeric class labels."""
    if not is_classifier(model):
        return np.asarray(model.predict(X), dtype=float)
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "A classifier nuisance must implement predict_proba to estimate a conditional mean."
        )
    probabilities = model.predict_proba(X)
    # sklearn multioutput classifiers return one probability matrix and class
    # array per target; labels can differ across treatment periods.
    if isinstance(probabilities, list):
        return np.column_stack(
            [
                np.asarray(p) @ np.asarray(labels, dtype=float)
                for p, labels in zip(probabilities, model.classes_, strict=True)
            ]
        )
    return np.asarray(probabilities @ np.asarray(model.classes_, dtype=float), dtype=float)


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
        Y_hat[test_idx] = predict_conditional_mean(outcome_mod, X_test)

        treatment_mod = clone(treatment_model)
        treatment_mod.fit(X_train, T[train_idx])
        T_hat[test_idx] = predict_conditional_mean(treatment_mod, X_test)

    return Y_hat, T_hat


def as_validated_arrays(
    Y: object, T: object, X: object
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convert Y, T and X to float arrays, check their shapes, and make X two-dimensional.

    The conversion comes first because a pandas ``Series`` is indexed by label and an
    array by position. With a reordered index, ``Y[train_idx]`` picked other rows than
    ``X[train_idx]``, and two ``Series`` with different labels were aligned by label when
    multiplied; both gave a wrong slope with no error (#75). After the conversion every
    row is addressed by its position, as ``TemporalPLRDML.fit`` has always done.

    Args:
        Y: Outcome, anything ``np.asarray`` reads as a one-dimensional float array.
        T: Treatment, likewise.
        X: Controls, one- or two-dimensional; a one-dimensional X is one column.

    Returns:
        ``(Y, T, X)`` as float arrays, X of shape ``(n, p)``.

    Raises:
        ValueError: If Y or T is not one-dimensional or the lengths disagree
            (see ``validate_lengths``), or if a value cannot be read as a float.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> y = pd.Series([1.0, 2.0, 3.0], index=[2, 0, 1])
        >>> Y, T, X = as_validated_arrays(y, [0, 1, 0], [5.0, 6.0, 7.0])
        >>> Y.tolist(), T.tolist(), X.shape
        ([1.0, 2.0, 3.0], [0.0, 1.0, 0.0], (3, 1))
    """
    Y_arr = np.asarray(Y, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    X_arr = np.asarray(X, dtype=float)
    validate_lengths(Y_arr, T_arr, X_arr)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    return Y_arr, T_arr, X_arr


def validate_lengths(
    Y: NDArray[np.float64], T: NDArray[np.float64], X: NDArray[np.float64]
) -> None:
    """Check that Y and T are one-dimensional and that Y, T and X agree in length.

    A column vector of shape ``(n, 1)`` is refused, not flattened. Its residuals
    broadcast against ``(n,)`` predictions to an ``(n, n)`` matrix, and the slope
    computed from that matrix is wrong with no error (#72). The two length
    messages are test-pinned.

    Raises:
        ValueError: If Y or T is not one-dimensional, or the lengths disagree.

    Examples:
        >>> import numpy as np
        >>> validate_lengths(np.ones(4), np.ones(4), np.ones((4, 2)))
        >>> validate_lengths(np.ones(4), np.ones((4, 1)), np.ones((4, 2)))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: T must be one-dimensional, got shape (4, 1). ...
    """
    for name, values in (("Y", Y), ("T", T)):
        shape = np.shape(values)
        if len(shape) != 1:
            raise ValueError(
                f"{name} must be one-dimensional, got shape {shape}. "
                f"Pass a single column as a flat array, for example np.asarray({name}).ravel()."
            )
    n = len(Y)
    if len(T) != n:
        raise ValueError(f"T length ({len(T)}) must match Y length ({n})")
    if X.shape[0] != n:
        raise ValueError(f"X rows ({X.shape[0]}) must match Y length ({n})")
