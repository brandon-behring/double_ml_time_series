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

import decimal
import numbers
import os
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse
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
    of the array and never with exact zero. The values are first divided by their
    largest magnitude: a norm of values near 1e300 overflows, and one of values near
    1e-300 underflows, and either would make a varying array look constant.

    Examples:
        >>> import numpy as np
        >>> is_numerically_constant(np.full(100, 0.1))
        True
        >>> is_numerically_constant(1e300 * np.array([1.0, 2.0, 3.0]))
        False
    """
    values = np.asarray(values, dtype=float)
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    if not np.isfinite(scale):
        return False
    if scale == 0.0:
        return True
    scaled = values / scale
    spread = np.linalg.norm(scaled - scaled.mean())
    return bool(spread <= np.linalg.norm(scaled) * scaled.size * _EPS)


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


_NOTHING = object()


def _is_real_number_or_missing(value: object) -> bool:
    """Whether one element of an object array is a real number or a missing value.

    ``np.timedelta64`` is excluded by name: NumPy derives it from its signed integers, so
    it passes for a number.

    Examples:
        >>> import decimal
        >>> import numpy as np
        >>> [_is_real_number_or_missing(v) for v in (1, 2.5, True, None, decimal.Decimal("1"))]
        [True, True, True, True, True]
        >>> [_is_real_number_or_missing(v) for v in (1j, np.timedelta64(1, "s"), b"1")]
        [False, False, False]
    """
    if value is None or value is pd.NA:
        return True
    if isinstance(value, np.timedelta64):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim == 0 and value.dtype.kind in "fiub"
    return isinstance(value, (numbers.Real, decimal.Decimal, np.bool_))


def _example(value: object, limit: int = 60) -> str:
    """A value as it is shown in a refusal: its ``repr``, cut at ``limit`` characters."""
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _read_as_floats(name: str, values: object) -> NDArray[np.float64]:
    """Read an outcome or a treatment as a float array, or say why it is not one.

    Numbers (floats, integers, booleans, and objects such as ``Decimal``) are read by their
    values. ``None``, NaN and the missing values of a pandas column become NaN; ``pd.NA``
    inside a list or a NumPy object array is refused. Anything whose conversion would
    change its meaning is refused: a datetime or timedelta would become integer ticks in
    its storage unit, a complex value would lose its imaginary part, a masked array would
    be read under its mask, and text is a label even when it spells a number.

    An object array is judged by its elements and not by whether it can be cast:
    ``np.asarray(values, dtype=float)`` turns a timezone-aware pandas column into ticks,
    and an object array of ``np.timedelta64`` or ``np.complex128`` scalars into floats,
    without an error. An element must be a real number (``numbers.Real``, ``Decimal``, a
    NumPy boolean, a zero-dimensional array of numbers) or a missing value.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> _read_as_floats("T", [0, 10, None]).tolist()
        [0.0, 10.0, nan]
        >>> _read_as_floats("Y", np.array([1, 2], dtype="timedelta64[s]"))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: Y has dtype timedelta64[s], which is not a number. ...
        >>> stamps = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True))
        >>> _read_as_floats("Y", stamps)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: Y holds Timestamp values (for example ...), not real numbers. ...
    """
    if isinstance(values, np.ma.MaskedArray) and np.ma.is_masked(values):
        raise ValueError(
            f"{name} is a masked array with masked entries. Its values would be read under "
            "the mask; drop or fill those rows first."
        )
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} could not be read as an array: {exc}") from exc
    if array.dtype.kind == "O":
        text = next((v for v in array.ravel() if isinstance(v, (str, bytes))), None)
        if text is not None:
            raise ValueError(
                f"{name} holds text (for example {_example(text)}), not numbers. "
                "Convert it to floats first."
            )
        # A default of None would hide an element that is None.
        other = next((v for v in array.ravel() if not _is_real_number_or_missing(v)), _NOTHING)
        if other is not _NOTHING:
            raise ValueError(
                f"{name} holds {type(other).__name__} values (for example {_example(other)}), not real "
                "numbers. Convert it to floats in the unit you mean: a datetime or timedelta "
                "would be read as integer ticks, and a complex value would lose its "
                "imaginary part."
            )
        try:
            # From the input, not from the object array: pandas turns the missing values
            # of its columns into NaN when it is asked to.
            if isinstance(values, (pd.Series, pd.DataFrame)):
                array = values.to_numpy(dtype=float, na_value=np.nan)
            else:
                array = np.asarray(values, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} could not be read as a float array: {exc}") from exc
    if array.dtype.kind not in "fiub":
        raise ValueError(
            f"{name} has dtype {array.dtype}, which is not a number. Convert it to floats in "
            "the unit you mean: a datetime or timedelta would be read as integer ticks, a "
            "complex value would lose its imaginary part, and text is a label."
        )
    return array.astype(float)


def _refuse_different_indexes(labelled: list[tuple[str, Any]]) -> None:
    """Refuse indexes that differ in value or order, or that name their levels differently.

    An index is anything with an ``equals`` method: a pandas index, or that of a container
    that follows pandas.

    ``Index.equals`` compares values and order and ignores names. Two indexes whose levels
    are named differently label different things even when their values agree: an outcome
    indexed by (unit, period) and a treatment indexed by (period, unit) over a square panel
    have equal tuples in other rows.
    """
    first_name, first = labelled[0]
    for name, index in labelled[1:]:
        if not index.equals(first):
            raise ValueError(
                f"{first_name} and {name} carry different pandas indexes. Rows are paired by "
                "position, and labels are not used to reorder them. If the rows are already "
                "in the same order, pass arrays with .to_numpy(). If they are not, bring "
                "them into one order first (for example with .loc or .reindex on labels "
                "that mean the same thing in both)."
            )
    # Equal values are transitive; agreement of names is not, because an unnamed level
    # agrees with every name. So the names are compared for every pair.
    named = [
        (name, list(index.names))
        for name, index in labelled
        if getattr(index, "names", None) is not None
    ]
    for position, (first_name, names) in enumerate(named):
        for name, other in named[position + 1 :]:
            clash = any(
                a is not None and b is not None and a != b
                for a, b in zip(names, other, strict=False)
            )
            if clash:
                raise ValueError(
                    f"{first_name} and {name} carry different pandas indexes: their values "
                    f"are equal, but their levels are named {names} and {other}. Rows are "
                    "paired by position; pass arrays with .to_numpy() if the rows are in "
                    "the same order."
                )


def as_validated_arrays(
    Y: object, T: object, X: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Read Y and T as float arrays, check the shapes, and make a flat X a column.

    Rows are paired by position. ``double_ml`` indexed a pandas ``Series`` by label where
    it indexed the controls by position: with a reordered frame its slope was wrong with
    no error, and with other labels it raised ``KeyError``. ``fwl_estimate`` and
    ``robinson_estimator`` multiplied two residual ``Series`` that pandas aligned by label:
    they were right for a reordered frame and wrong, with no error, when the outcome and
    the treatment carried different labels (#75). Reading Y and T by value removes the
    labels.

    Pairing by position is only right when the labels agree. Two pandas inputs whose
    indexes differ in value or in order are therefore refused, and so are two whose levels
    are named differently: the same units in another order would be paired wrongly, and
    nothing in the numbers would show it. What the check cannot see: an input whose index
    was reset after it was reordered; rows permuted among repeated labels; a level that
    one input names and the other does not; and a container whose ``index`` has no
    ``equals`` method, which is read by position.

    This function does not convert a two-dimensional X; it checks its row count and
    returns it as given. A list is read as an array, and a one-dimensional X becomes one
    column of an array, also when it was a ``Series``. What then happens to X is the
    caller's affair: ``robinson_estimator`` hands it to the nuisance model, so a DataFrame
    keeps its column names and dtypes and a sparse matrix stays sparse there;
    ``double_ml`` selects fold rows with ``X[rows]``, which works for an array and a CSR
    or CSC matrix and raises ``KeyError`` for a DataFrame; ``fwl_estimate`` stacks X into a
    dense design, which a DataFrame of numbers survives and a sparse matrix does not.

    Args:
        Y: Outcome: a one-dimensional sequence of numbers.
        T: Treatment: likewise.
        X: Controls with one row per observation.

    Returns:
        ``(Y, T, X)``: Y and T as float arrays; X as given, or as an array if it was a
        list, or as one column if it was one-dimensional.

    Raises:
        ValueError: If Y or T is not numeric, is masked, or cannot be read; if Y or T is
            not one-dimensional or the lengths disagree (see ``validate_lengths``); if X
            has no rows to count; or if two of the inputs are pandas objects whose indexes
            differ in value, in order or in the names of their levels.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> y = pd.Series([1.0, 2.0, 3.0], index=[2, 0, 1])
        >>> Y, T, X = as_validated_arrays(y, [0, 1, 0], np.array([5.0, 6.0, 7.0]))
        >>> Y.tolist(), T.tolist(), X.shape
        ([1.0, 2.0, 3.0], [0.0, 1.0, 0.0], (3, 1))
        >>> as_validated_arrays(y, y.sort_index(), X)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: Y and T carry different pandas indexes. ...
    """
    Y_arr = _read_as_floats("Y", Y)
    T_arr = _read_as_floats("T", T)
    given = X
    if not hasattr(X, "shape"):
        X = np.asarray(X)
    if len(X.shape) == 0:
        raise ValueError(
            f"X must hold the controls with one row per observation, got an object of type "
            f"{type(given).__name__} without rows."
        )
    validate_lengths(Y_arr, T_arr, X)
    # A pandas Series or DataFrame is recognized by an index that can be compared;
    # a list also has an ``index`` attribute, which is a method.
    labelled = [
        (name, index)
        for name, index in (
            ("Y", getattr(Y, "index", None)),
            ("T", getattr(T, "index", None)),
            ("X", getattr(X, "index", None)),
        )
        if hasattr(index, "equals")
    ]
    if labelled:
        _refuse_different_indexes(labelled)
    if len(X.shape) == 1:
        # np.asarray of a sparse array is a zero-dimensional object, not its values.
        X = X.reshape((-1, 1)) if sparse.issparse(X) else np.asarray(X).reshape(-1, 1)
    return Y_arr, T_arr, X


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
