"""The input contract of the estimators, as one matrix (#77).

The public callables of ``dml_ts.dml`` that read data against the kinds of input, with two
left out by name and reason below (``PanelDML`` and the EconML cross-check). A cell expects
one of three things:

- the result that plain float arrays of the same values give;
- a ``ValueError`` that names the argument at fault;
- for ``robinson_estimator`` and ``double_ml`` with controls their model cannot read, the
  model's own error. These two hand the controls to the nuisance model as given, so what the
  model does with them is the model's affair, and that error does not name the argument.

No estimator is left out of a parametrization: where they differ by design, the difference is
written in the case. The other estimators compute on the controls and convert them to floats
under the refusals that hold for the outcome and the treatment.

What the matrix does not show. The learner is ridge throughout: the default forest reads a
datetime or timedelta column of controls as ticks where ridge raises. An agreement between
the two sides of a cell says that the container made no difference, not that the estimator
is right; the arrays' results are checked elsewhere (golden parity, the formula contracts).

The contract itself is stated in ``dml_ts.dml._utils.as_validated_arrays``.
"""

from __future__ import annotations

import decimal
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sparse
from numpy.testing import assert_allclose
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

import dml_ts.dml as dml_package
from dml_ts.dml import (
    DynamicGEstimationDML,
    RollingWindowDML,
    TemporalPLRDML,
    double_ml,
    fwl_estimate,
    fwl_residualize,
    fwl_vs_ols_comparison,
    robinson_estimator,
)
from dml_ts.dml._utils import read_controls_as_floats, take_rows

pytestmark = pytest.mark.tier2

N, UNITS, PERIODS = 300, 100, 3
GROUPS = np.repeat(np.arange(UNITS), PERIODS)

# The same values in another container can differ in the last bits through memory layout.
# Rows paired wrongly, or values misread, differ in the first digits.
RTOL = 1e-10


def _draw() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261126)
    x = rng.standard_normal((N, 2))
    t = 0.6 * x[:, 0] + rng.standard_normal(N)
    y = 2.0 * t + x[:, 1] + rng.standard_normal(N)
    return y, t, x


Y0, T0, X0 = _draw()
DATES = pd.date_range("2020-01-01", periods=N, freq="D")
BINARY = T0 > 0
PERM = np.random.default_rng(5).permutation(N)


# --------------------------------------------------------------------------- the estimators


def _flat(*values: Any) -> tuple[float, ...]:
    return tuple(float(v) for value in values for v in np.ravel(value))


def _fwl(Y, T, X):
    fit = fwl_estimate(Y, T, X)
    return _flat(fit.theta, fit.se)


def _fwl_vs_ols(Y, T, X):
    both = fwl_vs_ols_comparison(Y, T, X)
    return _flat(both["theta_ols"], both["theta_fwl"])


def _fwl_residualize(Y, T, X):
    residuals, r_squared = fwl_residualize(Y, X)
    assert type(residuals) is np.ndarray
    return _flat(residuals, r_squared)


def _robinson(Y, T, X):
    fit = robinson_estimator(Y, T, X, model="ridge")
    return _flat(fit.theta, fit.se)


def _double_ml(Y, T, X):
    fit = double_ml(Y, T, X, model="ridge", random_state=0)
    return _flat(fit.theta, fit.se)


def _fwl_ols(Y, T, X):
    fit = fwl_estimate(Y, T, X, add_intercept=False, method="ols")
    return _flat(fit.theta, fit.se)


def _fwl_residualize_ols(Y, T, X):
    residuals, r_squared = fwl_residualize(Y, X, method="ols")
    assert type(residuals) is np.ndarray
    return _flat(residuals, r_squared)


def _temporal(Y, T, X):
    fit = TemporalPLRDML(n_lags=1, model_y="ridge", model_t="ridge", random_state=0).fit(Y, T, X)
    return _flat(fit.theta, fit.se)


def _temporal_without_lags(Y, T, X):
    fit = TemporalPLRDML(n_lags=0, model_y="ridge", model_t="ridge", random_state=0).fit(Y, T, X)
    return _flat(fit.theta, fit.se)


def _rolling(Y, T, X):
    model = RollingWindowDML(window_size=100, step_size=100, model_y="ridge", model_t="ridge")
    centers, effects, errors = model.fit(Y, T, X).get_effects()
    assert len(effects) == 3
    return _flat(centers, effects, errors)


def _g_panel(Y, T, X):
    model = DynamicGEstimationDML(model_y="ridge", model_t="ridge", n_folds=3, random_state=0)
    fit = model.fit(Y, T, X, groups=GROUPS)
    return _flat(fit.theta_t, fit.se_t)


def _g_series(Y, T, X):
    model = DynamicGEstimationDML(
        n_periods=2, model_y="ridge", model_t="ridge", n_folds=3, random_state=0
    )
    fit = model.fit(Y, T, X, mode="series")
    return _flat(fit.theta_t, fit.se_t)


@dataclass(frozen=True)
class Entry:
    """One public entry point, and what it does with its arguments."""

    name: str
    fit: Callable[[Any, Any, Any], tuple[float, ...]]
    computes_on_controls: bool
    reads_treatment: bool = True

    def __str__(self) -> str:
        return self.name


ENTRIES = (
    Entry("fwl_estimate", _fwl, computes_on_controls=True),
    Entry("fwl_vs_ols_comparison", _fwl_vs_ols, computes_on_controls=True),
    Entry("fwl_estimate[ols, no intercept]", _fwl_ols, computes_on_controls=True),
    Entry("fwl_residualize", _fwl_residualize, computes_on_controls=True, reads_treatment=False),
    Entry(
        "fwl_residualize[ols]",
        _fwl_residualize_ols,
        computes_on_controls=True,
        reads_treatment=False,
    ),
    Entry("robinson_estimator", _robinson, computes_on_controls=False),
    Entry("double_ml", _double_ml, computes_on_controls=False),
    Entry("TemporalPLRDML", _temporal, computes_on_controls=True),
    Entry("TemporalPLRDML[n_lags=0]", _temporal_without_lags, computes_on_controls=True),
    Entry("RollingWindowDML", _rolling, computes_on_controls=True),
    Entry("DynamicGEstimationDML[panel]", _g_panel, computes_on_controls=True),
    Entry("DynamicGEstimationDML[series]", _g_series, computes_on_controls=True),
)

# Public names of the package that the matrix does not cover, each with its reason.
NOT_IN_THE_MATRIX = {
    "PanelDML": "held for its redesign; its identifiers are part of that work (#77)",
    "fit_econml_reference": (
        "the EconML cross-check converts its arguments itself, with np.asarray and ravel, "
        "and is outside this change (#42)"
    ),
    "create_time_series_cv": "builds a splitter and reads no data",
    "econml_available": "reads no data",
    "DMLResult": "a result type",
    "TemporalPLRDMLResult": "a result type",
    "DynamicGEstimationResult": "a result type",
}


def test_the_matrix_covers_every_public_name_or_says_why_not():
    covered = {entry.name.split("[")[0] for entry in ENTRIES}
    assert covered | set(NOT_IN_THE_MATRIX) == set(dml_package.__all__)
    assert not covered & set(NOT_IN_THE_MATRIX)


# --------------------------------------------------------------------------------- the cases


@dataclass(frozen=True)
class Refusal:
    """A ``ValueError`` that names ``argument``; ``pattern`` is matched against its message."""

    argument: str
    pattern: str


@dataclass(frozen=True)
class Case:
    """One kind of input.

    ``given`` builds ``(Y, T, X)``. ``reference`` builds the float arrays that hold the same
    values, for the cells that must agree with arrays. ``refused`` holds for every estimator
    that reads the argument at fault. ``refused_where_computed`` holds only for the estimators
    that compute on the controls; the estimators that hand the controls to the nuisance model
    then either give the arrays' result or, with ``the_model_raises``, fail inside the model
    (ridge here) with that exception type and message. ``rtol`` is looser where a learner
    takes another numerical path for the same values.
    """

    name: str
    given: Callable[[], tuple[Any, Any, Any]]
    reference: Callable[[], tuple[Any, Any, Any]] = lambda: (Y0, T0, X0)
    refused: Refusal | None = None
    refused_where_computed: Refusal | None = None
    the_model_raises: tuple[type[Exception], str] | None = None
    rtol: float = RTOL

    def __str__(self) -> str:
        return self.name


def _series(values, index=None, dtype=None) -> pd.Series:
    return pd.Series(np.asarray(values), index=index, dtype=dtype)


def _frame(values=X0, index=None) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(values), columns=["x0", "x1"], index=index)


def _masked(values: np.ndarray) -> np.ma.MaskedArray:
    mask = np.zeros(values.shape, dtype=bool)
    mask[:3] = True
    return np.ma.array(values, mask=mask)


def _shuffled():
    frame = pd.DataFrame({"y": Y0, "t": T0, "x0": X0[:, 0], "x1": X0[:, 1]}, index=DATES)
    return frame.sample(frac=1, random_state=1)


X5 = np.column_stack([X0, np.random.default_rng(11).standard_normal((N, 3))])
TEXT_FRAME = _frame().assign(region=["a", "b", "c"] * (N // 3))
DATE_FRAME = _frame().assign(day=DATES)
DIFFERENT_INDEXES = r"carry different pandas indexes.*\.to_numpy\(\)"

SAME_AS_ARRAYS = (
    Case("float arrays", lambda: (Y0, T0, X0)),
    Case("lists", lambda: (Y0.tolist(), T0.tolist(), X0.tolist())),
    Case("tuples", lambda: (tuple(Y0), tuple(T0), tuple(map(tuple, X0)))),
    Case(
        "float32 outcome and treatment",
        lambda: (Y0.astype(np.float32), T0.astype(np.float32), X0),
        lambda: (Y0.astype(np.float32).astype(float), T0.astype(np.float32).astype(float), X0),
    ),
    Case(
        "integer outcome, boolean treatment",
        lambda: (np.round(10 * Y0).astype(int), BINARY, X0),
        lambda: (np.round(10 * Y0), BINARY.astype(float), X0),
    ),
    Case(
        "list of booleans as the treatment",
        lambda: (Y0, [bool(v) for v in BINARY], X0),
        lambda: (Y0, BINARY.astype(float), X0),
    ),
    Case("Decimal objects", lambda: ([decimal.Decimal(repr(float(v))) for v in Y0], T0, X0)),
    Case("pandas, default index", lambda: (_series(Y0), _series(T0), _frame())),
    Case(
        "pandas, one date index",
        lambda: (_series(Y0, DATES), _series(T0, DATES), _frame(index=DATES)),
    ),
    Case(
        "pandas, one frame shuffled as a whole",
        lambda: (_shuffled().y, _shuffled().t, _shuffled()[["x0", "x1"]]),
        lambda: (
            _shuffled().y.to_numpy(),
            _shuffled().t.to_numpy(),
            _shuffled()[["x0", "x1"]].to_numpy(),
        ),
    ),
    Case("Series with a date index, array controls", lambda: (_series(Y0, DATES), T0, X0)),
    Case(
        "integer labels beside the same labels as floats",
        lambda: (_series(Y0, np.arange(N)), _series(T0, np.arange(N).astype(float)), X0),
    ),
    Case(
        "nullable Float64 outcome, Int64 treatment",
        lambda: (
            pd.Series(pd.array(Y0, dtype="Float64")),
            pd.Series(pd.array(BINARY.astype(int), dtype="Int64")),
            X0,
        ),
        lambda: (Y0, BINARY.astype(float), X0),
    ),
    Case(
        "categorical treatment whose codes are not its values",
        lambda: (
            Y0,
            pd.Series(pd.Categorical(np.where(BINARY, 10, 20), categories=[20, 10])),
            X0,
        ),
        lambda: (Y0, np.where(BINARY, 10.0, 20.0), X0),
    ),
    Case(
        "masked arrays without masked entries",
        lambda: (np.ma.array(Y0), np.ma.array(T0), np.ma.array(X0)),
    ),
    Case("one-dimensional controls", lambda: (Y0, T0, X0[:, 0]), lambda: (Y0, T0, X0[:, :1])),
    Case(
        "one-dimensional Series of controls with the outcome's index",
        lambda: (_series(Y0, DATES), _series(T0, DATES), _series(X0[:, 0], DATES)),
        lambda: (Y0, T0, X0[:, :1]),
    ),
    Case(
        "frame of controls with integer and boolean columns",
        lambda: (
            Y0,
            T0,
            pd.DataFrame({"count": np.round(3 * X0[:, 0]).astype(int), "flag": X0[:, 1] > 0}),
        ),
        lambda: (Y0, T0, np.column_stack([np.round(3 * X0[:, 0]), (X0[:, 1] > 0).astype(float)])),
    ),
)

REFUSED_EVERYWHERE = (
    Case(
        "column vector as the outcome",
        lambda: (Y0[:, None], T0, X0),
        refused=Refusal("Y", r"Y must be one-dimensional, got shape \(300, 1\)"),
    ),
    Case(
        "column vector as the treatment",
        lambda: (Y0, T0[:, None], X0),
        refused=Refusal("T", r"T must be one-dimensional, got shape \(300, 1\)"),
    ),
    Case(
        "timedelta treatment",
        lambda: (Y0, (1000 * T0).astype("int64").astype("timedelta64[s]"), X0),
        refused=Refusal("T", r"T has dtype timedelta64\[s\], which is not a number"),
    ),
    Case(
        "datetime treatment",
        lambda: (Y0, DATES, X0),
        refused=Refusal("T", r"T has dtype datetime64\[\w+\], which is not a number"),
    ),
    Case(
        "complex outcome",
        lambda: (Y0 + 1j, T0, X0),
        refused=Refusal("Y", r"Y has dtype complex128, which is not a number"),
    ),
    Case(
        "numeric text in an object column as the treatment",
        lambda: (Y0, _series(T0).astype(str).astype(object), X0),
        refused=Refusal("T", r"T holds text \(for example '.*'\), not numbers"),
    ),
    Case(
        "array of text as the outcome",
        lambda: (Y0.astype(str), T0, X0),
        refused=Refusal("Y", r"Y has dtype <U\d+, which is not a number"),
    ),
    Case(
        "masked outcome with masked entries",
        lambda: (_masked(Y0), T0, X0),
        refused=Refusal("Y", r"Y is a masked array with masked entries"),
    ),
    Case(
        "masked treatment with masked entries",
        lambda: (Y0, _masked(T0), X0),
        refused=Refusal("T", r"T is a masked array with masked entries"),
    ),
    Case(
        "masked controls with masked entries",
        lambda: (Y0, T0, _masked(X0)),
        refused=Refusal("X", r"X is a masked array with masked entries"),
    ),
    Case(
        "ragged outcome",
        lambda: ([[1.0, 2.0], [3.0]] + Y0.tolist()[2:], T0, X0),
        refused=Refusal("Y", r"Y could not be read as an array"),
    ),
    Case(
        "treatment with the outcome's labels in another order",
        lambda: (_series(Y0, DATES), _series(T0, DATES).sample(frac=1, random_state=2), X0),
        refused=Refusal("T", r"Y and T " + DIFFERENT_INDEXES),
    ),
    Case(
        "treatment rebuilt with a fresh index beside a date index",
        lambda: (_series(Y0, DATES), _series(T0), X0),
        refused=Refusal("T", r"Y and T " + DIFFERENT_INDEXES),
    ),
    Case(
        "frame of controls with another index than the outcome",
        lambda: (_series(Y0, DATES), T0, _frame()),
        refused=Refusal("X", r"Y and X " + DIFFERENT_INDEXES),
    ),
    Case(
        "treatment and controls with other labels beside an unlabelled outcome",
        lambda: (Y0, _series(T0, DATES), _frame()),
        refused=Refusal("T", r"T and X " + DIFFERENT_INDEXES),
    ),
    Case(
        "treatment shorter than the outcome",
        lambda: (Y0, T0[:-1], X0),
        refused=Refusal("T", r"T length \(299\) must match Y length \(300\)"),
    ),
    Case(
        "controls with fewer rows than the outcome",
        lambda: (Y0, T0, X0[:-1]),
        refused=Refusal("X", r"X rows \(299\) must match Y length \(300\)"),
    ),
    Case(
        "a number as the controls",
        lambda: (Y0, T0, 5.0),
        refused=Refusal("X", r"X must hold the controls with one row per observation"),
    ),
)

CONTROLS_THAT_SPLIT_THE_ESTIMATORS = (
    # Ridge solves a sparse system iteratively: it agrees with the dense solution to about
    # 1e-6 with several columns (to rounding with these two), so the cells are compared at 1e-4.
    Case(
        "sparse CSR controls",
        lambda: (Y0, T0, sparse.csr_matrix(X0)),
        refused_where_computed=Refusal("X", r"X is a sparse matrix.*X\.toarray\(\)"),
        rtol=1e-4,
    ),
    Case(
        "sparse COO controls",
        lambda: (Y0, T0, sparse.coo_matrix(X0)),
        refused_where_computed=Refusal("X", r"X is a sparse matrix.*X\.toarray\(\)"),
        rtol=1e-4,
    ),
    Case(
        "sparse CSR controls with five columns",
        lambda: (Y0, T0, sparse.csr_matrix(X5)),
        lambda: (Y0, T0, X5),
        refused_where_computed=Refusal("X", r"X is a sparse matrix.*X\.toarray\(\)"),
        rtol=1e-4,
    ),
    Case(
        "frame of controls with a text column",
        lambda: (Y0, T0, TEXT_FRAME),
        refused_where_computed=Refusal("X", r"X holds text \(for example 'a'\), not numbers"),
        the_model_raises=(ValueError, r"could not convert string to float: 'a'"),
    ),
    Case(
        "frame of controls with a datetime column",
        lambda: (Y0, T0, DATE_FRAME),
        refused_where_computed=Refusal(
            "X", r"X holds Timestamp values \(for example .*\), not real"
        ),
        # numpy's DTypePromotionError, a TypeError, raised inside scikit-learn's validation
        the_model_raises=(TypeError, r"DateTime64DType.* could not be promoted"),
    ),
    Case(
        "complex controls",
        lambda: (Y0, T0, X0 + 1j),
        refused_where_computed=Refusal("X", r"X has dtype complex128, which is not a number"),
        the_model_raises=(ValueError, r"Complex data not supported"),
    ),
    Case(
        "three-dimensional controls",
        lambda: (Y0, T0, np.stack([X0, X0], axis=2)),
        refused_where_computed=Refusal(
            "X", r"X must hold the controls with one row per observation and one column"
        ),
        the_model_raises=(ValueError, r"Found array with dim 3"),
    ),
)

CASES = SAME_AS_ARRAYS + REFUSED_EVERYWHERE + CONTROLS_THAT_SPLIT_THE_ESTIMATORS


@cache
def _from_arrays(entry: Entry, case: Case) -> tuple[float, ...]:
    Y, T, X = case.reference()
    return entry.fit(np.asarray(Y, dtype=float), np.asarray(T, dtype=float), np.asarray(X))


@pytest.mark.parametrize("case", CASES, ids=str)
@pytest.mark.parametrize("entry", ENTRIES, ids=str)
def test_every_cell_gives_the_arrays_result_or_refuses_by_name(entry: Entry, case: Case):
    refusal = case.refused
    if refusal is None and entry.computes_on_controls:
        refusal = case.refused_where_computed
    if refusal is not None and refusal.argument == "T" and not entry.reads_treatment:
        refusal = None  # fwl_residualize takes no treatment: the rest of the input is valid

    if refusal is not None:
        assert refusal.argument in refusal.pattern
        with pytest.raises(ValueError, match=refusal.pattern):
            entry.fit(*case.given())
        return
    if case.the_model_raises is not None and not entry.computes_on_controls:
        kind, pattern = case.the_model_raises
        with pytest.raises(kind, match=pattern):
            entry.fit(*case.given())
        return

    result = entry.fit(*case.given())
    assert_allclose(result, _from_arrays(entry, case), rtol=case.rtol, atol=0)


def test_the_matrix_separates_right_pairing_from_wrong():
    """The tolerance means something: the same rows paired wrongly are far outside it."""
    for entry in ENTRIES:
        right = np.array(entry.fit(Y0, T0, X0))
        wrong = np.array(entry.fit(Y0, T0, X0[PERM]))
        relative = np.abs(wrong - right) / np.maximum(np.abs(right), 1e-12)
        assert np.max(relative) > 1e-3, entry.name


# ------------------------------------------------- controls handed to the model as they are


def _by_name() -> Any:
    """A nuisance model that needs the frame: it selects its columns by name."""
    return make_pipeline(
        ColumnTransformer([("numbers", "passthrough", ["x0", "x1"])]), Ridge(alpha=1.0)
    )


@pytest.mark.parametrize(
    "fit",
    [
        lambda Y, T, X, model: robinson_estimator(Y, T, X, model=model),
        lambda Y, T, X, model: double_ml(Y, T, X, model=model, random_state=0),
    ],
    ids=["robinson_estimator", "double_ml"],
)
@pytest.mark.parametrize("index", [None, DATES], ids=["default index", "date index"])
def test_a_frame_of_controls_reaches_the_model_with_its_columns(fit, index):
    """The text column would stop ridge; the model drops it by name, so it must see a frame."""
    controls = TEXT_FRAME.set_axis(index) if index is not None else TEXT_FRAME
    with_frame = fit(Y0, T0, controls, _by_name())
    with_arrays = fit(Y0, T0, X0, Ridge(alpha=1.0))
    assert_allclose(with_frame.theta, with_arrays.theta, rtol=RTOL)
    assert_allclose(with_frame.se, with_arrays.se, rtol=RTOL)


def test_take_rows_selects_positions_in_every_container():
    rows = np.array([4, 0, 2])
    labelled = _frame(index=DATES)
    assert take_rows(labelled, rows).index.equals(DATES[rows])
    assert_allclose(take_rows(labelled, rows).to_numpy(), X0[rows])
    assert_allclose(take_rows(X0, rows), X0[rows])
    for build in (
        sparse.csr_matrix,
        sparse.csc_matrix,
        sparse.coo_matrix,
        sparse.lil_matrix,
        sparse.dia_matrix,
        sparse.csr_array,
        sparse.coo_array,
    ):
        assert_allclose(take_rows(build(X0), rows).toarray(), X0[rows])
    # Integer column labels: ``frame[rows]`` would select columns 4, 0 and 2 by label.
    wide = pd.DataFrame(np.arange(50.0).reshape(5, 10))
    assert_allclose(take_rows(wide, rows).to_numpy(), wide.to_numpy()[rows])
    # Repeated row labels: selecting the labels found at these positions returns more rows.
    repeated = _frame(index=np.arange(N) % 7)
    assert take_rows(repeated, rows).shape == (3, 2)
    assert_allclose(take_rows(repeated, rows).to_numpy(), X0[rows])


def test_double_ml_takes_fold_rows_by_position_under_repeated_row_labels():
    repeated = np.arange(N) % 7
    with_frame = double_ml(
        _series(Y0, repeated), _series(T0, repeated), _frame(index=repeated), model="ridge"
    )
    assert_allclose(with_frame.theta, double_ml(Y0, T0, X0, model="ridge").theta, rtol=RTOL)


def test_a_frame_of_controls_with_a_missing_value_is_read_as_nan():
    """pandas converts its own missing values; NumPy cannot cast ``pd.NA`` among objects."""
    frame = pd.DataFrame(
        {"a": pd.array([1.0, None, 3.0], dtype="Float64"), "b": pd.array([4, 5, 6], dtype="Int64")}
    )
    read = read_controls_as_floats("X", frame)
    assert np.isnan(read[1, 0])
    assert_allclose(np.delete(read.ravel(), 2), [1.0, 4.0, 5.0, 3.0, 6.0])


# ------------------------------------- further arguments of the dynamic g-estimator: W, groups


def _g_with(W=None, groups=GROUPS, Y=Y0, T=T0, X=X0):
    model = DynamicGEstimationDML(model_y="ridge", model_t="ridge", n_folds=3, random_state=0)
    return model.fit(Y, T, X, W=W, groups=groups).theta_t


W0 = np.random.default_rng(9).standard_normal((N, 1))


@pytest.mark.parametrize(
    "W",
    [
        W0.tolist(),
        W0[:, 0],
        pd.DataFrame(W0, columns=["w"]),
        pd.Series(W0[:, 0]),
        np.ma.array(W0),
    ],
    ids=["list", "one-dimensional", "frame", "Series", "masked without masked entries"],
)
def test_further_controls_are_read_by_value(W):
    assert_allclose(_g_with(W=W), _g_with(W=W0), rtol=RTOL)
    assert not np.allclose(_g_with(W=W0), _g_with(W=None), rtol=1e-6)


@pytest.mark.parametrize(
    ("W", "pattern"),
    [
        (pd.DataFrame({"w": ["a", "b", "c"] * (N // 3)}), r"W holds text \(for example 'a'\)"),
        (sparse.csr_matrix(W0), r"W is a sparse matrix"),
        (_masked(W0), r"W is a masked array with masked entries"),
        (W0.astype("timedelta64[s]"), r"W has dtype timedelta64\[s\]"),
        (W0[:-1], r"W rows \(299\) must match Y length \(300\)"),
        (pd.DataFrame(W0, index=DATES), r"Y and W " + DIFFERENT_INDEXES),
    ],
    ids=["text", "sparse", "masked entries", "timedelta", "fewer rows", "another index"],
)
def test_further_controls_are_refused_by_name(W, pattern):
    with pytest.raises(ValueError, match=pattern):
        _g_with(W=W, Y=_series(Y0))


def test_labelled_groups_join_the_index_comparison():
    labelled = pd.Series(GROUPS, index=DATES)
    assert_allclose(_g_with(groups=labelled, Y=_series(Y0, DATES)), _g_with(), rtol=RTOL)
    assert_allclose(_g_with(groups=labelled), _g_with(), rtol=RTOL)  # nothing to compare with
    with pytest.raises(ValueError, match=r"Y and groups " + DIFFERENT_INDEXES):
        _g_with(groups=labelled, Y=_series(Y0))
    with pytest.raises(ValueError, match=r"Y and groups " + DIFFERENT_INDEXES):
        _g_with(groups=labelled.sample(frac=1, random_state=3), Y=_series(Y0, DATES))


def _g_series_with(W=None, Y=Y0, T=T0, X=X0):
    model = DynamicGEstimationDML(
        n_periods=2, model_y="ridge", model_t="ridge", n_folds=3, random_state=0
    )
    return model.fit(Y, T, X, W=W, mode="series").theta_t


def test_further_controls_follow_the_same_rules_in_series_mode():
    """``W`` is read in ``fit`` before the modes part; a test in one mode alone would not show it."""
    with_w = _g_series_with(W=W0)
    assert not np.allclose(with_w, _g_series_with(W=None), rtol=1e-6)
    assert_allclose(_g_series_with(W=pd.DataFrame(W0, columns=["w"])), with_w, rtol=RTOL)
    assert_allclose(_g_series_with(W=W0[:, 0].tolist()), with_w, rtol=RTOL)
    with pytest.raises(ValueError, match=r"W holds text"):
        _g_series_with(W=pd.DataFrame({"w": ["a", "b", "c"] * (N // 3)}))
    with pytest.raises(ValueError, match=r"Y and W " + DIFFERENT_INDEXES):
        _g_series_with(W=pd.DataFrame(W0, index=DATES), Y=_series(Y0))
    with pytest.raises(ValueError, match=r"W rows \(299\) must match Y length \(300\)"):
        _g_series_with(W=W0[:-1])


def test_group_identifiers_are_one_per_row_and_never_read_under_a_mask():
    masked = np.ma.array(GROUPS, mask=np.arange(N) < 3)
    with pytest.raises(ValueError, match=r"^groups is a masked array with masked entries"):
        _g_with(groups=masked)
    assert_allclose(_g_with(groups=np.ma.array(GROUPS)), _g_with(), rtol=RTOL)
    with pytest.raises(
        ValueError, match=r"^groups must hold one unit id per row: got shape \(297,\)"
    ):
        _g_with(groups=GROUPS[:-3])
    with pytest.raises(
        ValueError, match=r"^groups must hold one unit id per row: got shape \(300, 1\)"
    ):
        _g_with(groups=GROUPS[:, None])
    # Identifiers are compared as given: text labels in unit-major order are the same panel.
    assert_allclose(
        _g_with(groups=np.array([f"unit {g:03d}" for g in GROUPS])), _g_with(), rtol=RTOL
    )


@pytest.mark.parametrize("mode", ["panel", "series"])
def test_the_dynamic_estimator_no_longer_flattens_a_matrix(mode):
    """A (periods, units) matrix flattened in the wrong order gave other numbers (#77)."""
    wide_y, wide_t = Y0.reshape(UNITS, PERIODS), T0.reshape(UNITS, PERIODS)
    model = DynamicGEstimationDML(
        n_periods=2, model_y="ridge", model_t="ridge", n_folds=3, random_state=0
    )
    kwargs = {"groups": GROUPS} if mode == "panel" else {"mode": "series"}
    cases = (
        (wide_y, T0, "Y"),
        (Y0, wide_t, "T"),
        (Y0, wide_t.T.copy(), "T"),
        (wide_y.T.copy(), wide_t.T.copy(), "Y"),
    )
    for Y, T, name in cases:
        with pytest.raises(ValueError, match=rf"{name} must be one-dimensional") as refusal:
            model.fit(Y, T, X0, **kwargs)
        # Flattening is how a matrix in the wrong order was misread; the refusal must not
        # recommend it.
        assert "ravel" not in str(refusal.value)
        assert "not flattened for you" in str(refusal.value)
    with pytest.raises(ValueError, match=r"np\.asarray\(Y\)\.ravel\(\)"):
        model.fit(Y0[:, None], T0, X0, **kwargs)


# --------------------------------------------------------------------------- the time index


@pytest.mark.parametrize(
    "fit",
    [
        lambda Y, T, X, index: (
            TemporalPLRDML(n_lags=1, model_y="ridge", model_t="ridge", random_state=0)
            .fit(Y, T, X, time_index=index)
            .theta
        ),
        lambda Y, T, X, index: (
            RollingWindowDML(window_size=100, step_size=100, model_y="ridge", model_t="ridge")
            .fit(Y, T, X, time_index=index)
            .get_effects()[1]
        ),
    ],
    ids=["TemporalPLRDML", "RollingWindowDML"],
)
def test_a_labelled_time_index_joins_the_index_comparison(fit):
    """Its values are not used, but a labelled one in another order says the rows are too."""
    outcome = _series(Y0, DATES)
    stamps = pd.Series(DATES, index=DATES)
    reference = fit(Y0, T0, X0, None)
    assert_allclose(fit(outcome, T0, X0, stamps), reference, rtol=RTOL)
    assert_allclose(fit(outcome, T0, X0, DATES), reference, rtol=RTOL)  # an index has no labels
    with pytest.raises(ValueError, match=r"Y and time_index " + DIFFERENT_INDEXES):
        fit(outcome, T0, X0, stamps.sample(frac=1, random_state=4))


# ------------------------------------------------------------------------- fwl_residualize


def test_fwl_residualize_returns_an_array_for_a_series():
    """It returned a labelled Series, so the recipe by hand multiplied residuals by label."""
    design = np.column_stack([np.ones(N), X0])
    shifted = _series(Y0, np.arange(50, 50 + N))
    residuals, _ = fwl_residualize(shifted, design)
    assert type(residuals) is np.ndarray
    by_hand_y = fwl_residualize(shifted, design)[0]
    by_hand_t = fwl_residualize(_series(T0), design)[0]
    slope = float(by_hand_y @ by_hand_t / (by_hand_t @ by_hand_t))
    assert_allclose(slope, fwl_estimate(Y0, T0, X0).theta, rtol=RTOL)
