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
from collections.abc import Iterable, Mapping
from typing import Any, Literal, overload

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


def take_rows(X: Any, rows: NDArray[np.int64]) -> Any:
    """Rows of ``X`` at the positions ``rows``, whatever ``X`` is.

    ``X[rows]`` selects rows of an array and of a sparse matrix, but columns of a
    DataFrame, by label. A DataFrame is therefore indexed with ``.iloc``. A sparse
    matrix in a format other than CSR or CSC is converted to CSR first: a ``coo_matrix``
    cannot be indexed at all.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=[10, 20, 30])
        >>> take_rows(frame, np.array([0, 2]))["a"].tolist()
        [1.0, 3.0]
        >>> take_rows(np.arange(6).reshape(3, 2), np.array([2])).tolist()
        [[4, 5]]
    """
    if hasattr(X, "iloc"):
        return X.iloc[rows]
    if sparse.issparse(X) and X.format not in ("csr", "csc"):
        X = X.tocsr()
    return X[rows]


def cross_fit_predictions(
    X: Any,
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

    ``X`` may be an array, a sparse matrix or a DataFrame: its rows are selected
    by position (:func:`take_rows`) and handed to the models as they are. ``Y``
    and ``T`` must be arrays.

    Returns:
        Tuple of ``(Y_hat, T_hat)`` predictions, NaN where uncovered.
    """
    n = len(Y)
    Y_hat = np.full(n, np.nan)
    T_hat = np.full(n, np.nan)

    for train_idx, test_idx in splits:
        X_train, X_test = take_rows(X, train_idx), take_rows(X, test_idx)

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


def refuse_masked_entries(name: str, values: object) -> None:
    """Refuse a masked array that has masked entries; NumPy and scikit-learn read under the mask."""
    if isinstance(values, np.ma.MaskedArray) and np.ma.is_masked(values):
        raise ValueError(
            f"{name} is a masked array with masked entries. Its values would be read under "
            "the mask; drop or fill those rows first."
        )


def _example(value: object, limit: int = 60) -> str:
    """A value as it is shown in a refusal: its ``repr``, cut at ``limit`` characters."""
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


def read_as_floats(name: str, values: object) -> NDArray[np.float64]:
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
        >>> read_as_floats("T", [0, 10, None]).tolist()
        [0.0, 10.0, nan]
        >>> read_as_floats("Y", np.array([1, 2], dtype="timedelta64[s]"))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: Y has dtype timedelta64[s], which is not a number. ...
        >>> stamps = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True))
        >>> read_as_floats("Y", stamps)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: Y holds Timestamp values (for example ...), not real numbers. ...
    """
    refuse_masked_entries(name, values)
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


def read_controls_as_floats(name: str, values: object) -> NDArray[np.float64]:
    """Read controls as a two-dimensional float array, or say why they are not one.

    For the estimators that compute on the controls themselves. The refusals are those
    of an outcome or a treatment (a datetime or timedelta, a complex value, text, a
    masked entry); a sparse matrix is refused too, because these estimators need the
    dense values. A one-dimensional input becomes one column.

    Examples:
        >>> import pandas as pd
        >>> read_controls_as_floats("X", pd.DataFrame({"a": [1, 2], "b": [True, False]})).tolist()
        [[1.0, 1.0], [2.0, 0.0]]
        >>> read_controls_as_floats("X", [1.0, 2.0]).shape
        (2, 1)
        >>> read_controls_as_floats("W", pd.DataFrame({"g": ["a", "b"]}))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: W holds text (for example 'a'), not numbers. ...
    """
    if sparse.issparse(values):
        raise ValueError(
            f"{name} is a sparse matrix. This estimator computes on the controls as a dense "
            f"array; pass {name}.toarray() if it fits in memory."
        )
    array = read_as_floats(name, values)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(
            f"{name} must hold the controls with one row per observation and one column per "
            f"control, got shape {array.shape}."
        )
    return array


_OWN_LEARNER_ADVICE = (
    " The check applies because one of the package's own learners (a string name or the "
    "default) receives the controls. A learner you supply receives them as given: to "
    "encode such a column yourself, pass your own learner, in double_ml for both "
    "nuisance models."
)


def refuse_non_numeric_controls(name: str, values: object) -> None:
    """Refuse controls that one of the package's own learners would misread or cannot read.

    For ``robinson_estimator`` and ``double_ml``, which hand the controls to a nuisance
    model as given. When that model is one of the package's own (a string name, or the
    default forest), the caller has not said how a column that is not a number is to be
    encoded, and the learners disagree about it: the forest reads controls that are all
    datetime or timedelta as integer ticks where ridge raises, and each of them parses
    text that spells numbers in an object column. Such controls are refused here under
    the refusals of :func:`read_controls_as_floats`: a datetime or timedelta, a complex
    value, text, a masked entry, and more than two dimensions. A learner the caller
    supplies is not checked, because a pipeline may encode such columns.

    This is a check and nothing else. It returns nothing, and the caller hands on the
    object it was given: a converted array would change the numbers (``float32`` controls
    are fitted in ``float32``) and could not stand for a sparse matrix. A sparse matrix is
    judged by its dtype alone and is never made dense.

    The cost. A sparse matrix, and a DataFrame whose columns are all stored as numbers,
    are judged by their dtypes and not read. Other dense controls pay one conversion
    that is thrown away: a copy for an array of numbers, and for an object array or a
    frame with a column that is not stored as numbers a pass over its elements in Python.

    Raises:
        ValueError: Naming the argument, and saying that a learner the caller supplies
            receives the controls as given.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from scipy import sparse
        >>> refuse_non_numeric_controls("X", np.ones((3, 2), dtype=np.float32))
        >>> refuse_non_numeric_controls("X", sparse.csr_matrix(np.eye(3)))
        >>> days = pd.DataFrame({"day": pd.date_range("2020-01-01", periods=3)})
        >>> refuse_non_numeric_controls("X", days)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: X has dtype datetime64[...], which is not a number. ...
    """
    if sparse.issparse(values):
        dtype = values.dtype  # type: ignore[attr-defined]
        if dtype.kind not in "fiub":
            raise ValueError(
                f"{name} is a sparse matrix of dtype {dtype}, which is not a number."
                + _OWN_LEARNER_ADVICE
            )
        return
    if isinstance(values, pd.DataFrame) and all(
        getattr(dtype, "kind", "O") in "fiub" for dtype in values.dtypes
    ):
        # Every column is stored as numbers (a nullable column with its missing values
        # included), so the conversion below could refuse nothing. It is skipped because
        # float beside boolean columns convert through an object array, read element by
        # element: 1.4 s for 100,000 rows of 20 columns, thirty times a ridge fit.
        return
    try:
        read_controls_as_floats(name, values)
    except ValueError as exc:
        raise ValueError(f"{exc}{_OWN_LEARNER_ADVICE}") from exc


def refuse_unequal_indexes(inputs: Mapping[str, object]) -> None:
    """Refuse pandas inputs whose indexes differ in value or order, or name their levels differently.

    Rows are paired by position, and labels are never used to reorder them: a time-series
    estimator depends on the order of its rows. Pairing by position is only right when the
    labels agree, so two labelled inputs with different indexes are refused. Inputs
    without a pandas index (arrays, lists, ``None``) take no part in the comparison.

    What the comparison cannot see: an input whose index was reset after it was reordered;
    rows permuted among repeated labels; a level that one input names and the other does
    not; a single labelled input beside arrays; and a container whose ``index`` has no
    ``equals`` method, which is read by position.

    Raises:
        ValueError: Naming the two labelled inputs whose indexes differ.

    Examples:
        >>> import pandas as pd
        >>> y = pd.Series([1.0, 2.0, 3.0], index=[2, 0, 1])
        >>> refuse_unequal_indexes({"Y": y, "T": [0, 1, 0], "groups": None})
        >>> refuse_unequal_indexes({"Y": y, "T": y.sort_index()})  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: Y and T carry different pandas indexes. ...
    """
    # A pandas Series or DataFrame is recognized by an index that can be compared;
    # a list also has an ``index`` attribute, which is a method.
    labelled = [
        (name, index)
        for name, index in ((name, getattr(value, "index", None)) for name, value in inputs.items())
        if hasattr(index, "equals")
    ]
    if labelled:
        _refuse_different_indexes(labelled)


@overload
def as_validated_arrays(
    Y: object,
    T: object,
    X: Any,
    *,
    numeric_controls: Literal[True],
    also_labelled: Mapping[str, object] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...


@overload
def as_validated_arrays(
    Y: object,
    T: object,
    X: Any,
    *,
    numeric_controls: Literal[False] = False,
    also_labelled: Mapping[str, object] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], Any]: ...


def as_validated_arrays(
    Y: object,
    T: object,
    X: Any,
    *,
    numeric_controls: bool = False,
    also_labelled: Mapping[str, object] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], Any]:
    """The one input boundary of the estimators: read Y, T and X, and check that they pair.

    The estimators of ``dml_ts.dml`` read their inputs here, with three exceptions:
    ``fwl_residualize`` takes no treatment and uses the same readers directly;
    ``PanelDML`` keeps its own reading until its redesign; and the EconML cross-check
    ``fit_econml_reference`` converts its arguments itself.

    1. **Y and T are read by value as floats.** Lists, integers, booleans, ``Decimal``
       objects and nullable pandas columns are read by their values. What conversion
       would misread is refused by the argument's name: a datetime or timedelta, a
       complex value, text, a masked array with masked entries, and anything that is
       not one-dimensional.
    2. **Rows are paired by position, and unequal pandas indexes are refused.** Labels
       are never used to reorder rows. The same units in another order, or a ``Series``
       rebuilt with a fresh index beside one with a date index, are refused; the message
       names ``.to_numpy()`` as the way through when the rows already agree. What the
       comparison cannot see is listed in :func:`refuse_unequal_indexes`.
    3. **Controls.** With ``numeric_controls=False`` (``robinson_estimator``,
       ``double_ml``) a two-dimensional X is handed to the nuisance model as it was
       given: a DataFrame keeps its column names and dtypes and a sparse matrix stays
       sparse; only the row count is checked, and rows are selected by position. (A
       masked array with masked entries is refused there too: a learner would read under
       the mask.) Those two estimators then check the controls, without converting them,
       when the model is one of the package's own (:func:`refuse_non_numeric_controls`):
       it would parse numeric text in an object column, and the default forest would
       read controls that are all datetime or timedelta as ticks. What a model the
       caller supplies makes of a column is that model's affair. With
       ``numeric_controls=True`` (``fwl_estimate``,
       ``fwl_vs_ols_comparison``, ``TemporalPLRDML``, ``RollingWindowDML``,
       ``DynamicGEstimationDML``) the estimator computes on the controls, so X is
       converted to a float array under the refusals of point 1, and a sparse matrix is
       refused.

    Args:
        Y: Outcome: a one-dimensional sequence of numbers.
        T: Treatment: likewise.
        X: Controls with one row per observation.
        numeric_controls: Whether the caller computes on X (see point 3).
        also_labelled: Further inputs, by argument name, whose pandas indexes must equal
            those of Y, T and X (for example ``{"W": W, "groups": groups}``). Only their
            indexes are read here.

    Returns:
        ``(Y, T, X)``: Y and T as float arrays. X as a two-dimensional float array if
        ``numeric_controls``; otherwise as given, or as an array if it was a list, or as
        one column if it was one-dimensional.

    Raises:
        ValueError: If Y or T is not numeric, is masked, or cannot be read; if Y or T is
            not one-dimensional or the lengths disagree (see ``validate_lengths``); if X
            has no rows to count, or cannot be read as numbers when ``numeric_controls``;
            or if two of the inputs are pandas objects with different indexes.

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
        >>> frame = pd.DataFrame({"a": [5, 6, 7]}, index=y.index)
        >>> as_validated_arrays(y, [0, 1, 0], frame)[2] is frame
        True
        >>> as_validated_arrays(y, [0, 1, 0], frame, numeric_controls=True)[2].dtype
        dtype('float64')
    """
    Y_arr = read_as_floats("Y", Y)
    T_arr = read_as_floats("T", T)
    given = X
    if numeric_controls:
        X = read_controls_as_floats("X", X)
    else:
        # Not converted, with one exception to "as given": a learner would read masked
        # entries under their mask, exactly as a conversion here would.
        refuse_masked_entries("X", X)
        if not hasattr(X, "shape"):
            X = np.asarray(X)
        if len(X.shape) == 0:
            raise ValueError(
                f"X must hold the controls with one row per observation, got an object of "
                f"type {type(given).__name__} without rows."
            )
    validate_lengths(Y_arr, T_arr, X)
    refuse_unequal_indexes({"Y": Y, "T": T, "X": given, **(also_labelled or {})})
    if len(X.shape) == 1:
        # np.asarray of a sparse array is a zero-dimensional object, not its values.
        X = X.reshape((-1, 1)) if sparse.issparse(X) else np.asarray(X).reshape(-1, 1)
    return Y_arr, T_arr, X


def require_one_dimensional(name: str, values: Any) -> None:
    """Refuse an outcome or a treatment that is not one-dimensional, by its name.

    A single column is told how to flatten itself. A matrix with several rows and columns
    is not: flattening it is the misreading that the refusal is there to stop.

    Raises:
        ValueError: Naming the argument and its shape.

    Examples:
        >>> import numpy as np
        >>> require_one_dimensional("Y", np.ones(4))
        >>> require_one_dimensional("Y", np.ones((4, 1)))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: Y must be one-dimensional, got shape (4, 1). ...
    """
    shape = np.shape(values)
    if len(shape) != 1:
        if len(shape) == 2 and 1 in shape:
            advice = (
                f"Pass a single column as a flat array, for example np.asarray({name}).ravel()."
            )
        else:
            # Flattening a matrix reads its rows in whatever order it happens to have; a
            # (periods, units) matrix flattened as (units, periods) gives other numbers.
            advice = (
                "Pass one value per observation, stacked in the row order the estimator "
                "documents; a matrix is not flattened for you."
            )
        raise ValueError(f"{name} must be one-dimensional, got shape {shape}. {advice}")


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
    require_one_dimensional("Y", Y)
    require_one_dimensional("T", T)
    n = len(Y)
    if len(T) != n:
        raise ValueError(f"T length ({len(T)}) must match Y length ({n})")
    if X.shape[0] != n:
        raise ValueError(f"X rows ({X.shape[0]}) must match Y length ({n})")
