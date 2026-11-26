"""The outcome and the treatment are read by value and paired by position (#75).

``double_ml`` indexed a pandas ``Series`` by label where it indexed the controls by position:
after ``df.sample(frac=1)`` it returned 1.82 where the arrays give 2.03. ``fwl_estimate`` and
``robinson_estimator`` multiplied two residual ``Series`` that pandas aligned by label: with
different labels on the outcome and the treatment they returned -0.15 for a slope of 2.

For ``robinson_estimator`` and ``double_ml`` the shared helper does not convert
two-dimensional controls, which a first version of this change did: a DataFrame or a sparse
matrix reaches the nuisance model as given, and fold rows are selected by position.
``fwl_estimate`` computes on its controls and converts them to floats. The matrix of every
estimator against every kind of input is in ``test_input_contract.py``.

An object column is judged by its elements. NumPy and pandas cast a timezone-aware datetime,
and an object array of ``np.timedelta64`` or ``np.complex128`` scalars, to floats without an
error; a second version of this change returned 7,348 for such an outcome where the floats
give 2.04.
"""

import decimal
import fractions

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sparse
from numpy.testing import assert_allclose
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from dml_ts.dml import double_ml
from dml_ts.dml._utils import as_validated_arrays, is_numerically_constant, read_as_floats
from dml_ts.dml.fwl import fwl_estimate
from dml_ts.dml.robinson import robinson_estimator

SLOPES = {
    "double_ml": lambda y, t, x: double_ml(y, t, x, model="ridge").theta,
    "robinson_estimator": lambda y, t, x: robinson_estimator(y, t, x, model="ridge").theta,
    "fwl_estimate": lambda y, t, x: fwl_estimate(y, t, x).theta,
}

# The same rows in another container or memory layout can differ in the last bit. Rows paired
# wrongly differ in the first digit, so this tolerance separates the two by eleven orders.
RTOL = 1e-12


def frame(n=300, seed=0):
    """A partially linear draw with slope 2, as a DataFrame with a sort key."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    t = x[:, 0] + rng.normal(size=n)
    y = 2 * t + x[:, 1] + rng.normal(size=n)
    return pd.DataFrame({"y": y, "t": t, "x0": x[:, 0], "x1": x[:, 1], "key": rng.normal(size=n)})


REORDERINGS = {
    "shuffled": lambda df: df.sample(frac=1, random_state=0),
    "sorted": lambda df: df.sort_values("key"),
    "filtered": lambda df: df[df["key"] > -0.5],
    "reversed": lambda df: df.iloc[::-1],
}


# ---------------------------------------------------------------- position, not label


@pytest.mark.tier2
@pytest.mark.parametrize("reordering", sorted(REORDERINGS))
@pytest.mark.parametrize("name", sorted(SLOPES))
def test_a_reordered_frame_gives_the_slope_of_its_arrays(name, reordering):
    """The index of a Series must not decide which rows are paired."""
    df = REORDERINGS[reordering](frame())
    assert not df.index.equals(pd.RangeIndex(len(df)))
    x = df[["x0", "x1"]].to_numpy()
    from_arrays = SLOPES[name](df["y"].to_numpy(), df["t"].to_numpy(), x)
    assert_allclose(SLOPES[name](df["y"], df["t"], x), from_arrays, rtol=RTOL)
    assert_allclose(from_arrays, 2.0, atol=0.35)


@pytest.mark.tier2
@pytest.mark.parametrize("name", sorted(SLOPES))
def test_pandas_inputs_with_different_labels_are_refused(name):
    """Labels 50..349 against 0..299: aligned by label, 250 rows met the wrong partner.

    Pairing by position is right only if the labels agree, so the call is refused and the
    arrays are accepted.
    """
    df = frame()
    x = df[["x0", "x1"]].to_numpy()
    outcome = pd.Series(df["y"].to_numpy(), index=range(50, 350))
    treatment = pd.Series(df["t"].to_numpy(), index=range(300))
    with pytest.raises(ValueError, match="Y and T carry different pandas indexes"):
        SLOPES[name](outcome, treatment, x)
    from_arrays = SLOPES[name](df["y"].to_numpy(), df["t"].to_numpy(), x)
    assert_allclose(
        SLOPES[name](outcome.to_numpy(), treatment.to_numpy(), x), from_arrays, rtol=RTOL
    )


@pytest.mark.tier2
@pytest.mark.parametrize("name", sorted(SLOPES))
def test_the_same_units_in_another_order_are_refused(name):
    """A treatment taken from a reordered copy of the frame has the same labels, shuffled.

    Read by position it is paired with the wrong outcomes: ``fwl_estimate`` returned 0.17
    where the aligned inputs give 2.03.
    """
    df = frame()
    shuffled = df.sample(frac=1, random_state=0)
    x = df[["x0", "x1"]].to_numpy()
    with pytest.raises(ValueError, match="Y and T carry different pandas indexes"):
        SLOPES[name](df["y"], shuffled["t"], x)
    with pytest.raises(ValueError, match="Y and X carry different pandas indexes"):
        SLOPES[name](df["y"], df["t"], shuffled[["x0", "x1"]])
    aligned = shuffled["t"].loc[df.index]
    from_arrays = SLOPES[name](df["y"].to_numpy(), df["t"].to_numpy(), x)
    assert_allclose(SLOPES[name](df["y"], aligned, x), from_arrays, rtol=RTOL)


@pytest.mark.tier2
@pytest.mark.parametrize("name", sorted(SLOPES))
def test_one_labelled_input_among_arrays_is_read_by_position(name):
    """With a single pandas input there are no two sets of labels to disagree."""
    df = frame().sample(frac=1, random_state=3)
    x = df[["x0", "x1"]].to_numpy()
    from_arrays = SLOPES[name](df["y"].to_numpy(), df["t"].to_numpy(), x)
    assert_allclose(SLOPES[name](df["y"], df["t"].to_numpy(), x), from_arrays, rtol=RTOL)
    assert_allclose(SLOPES[name](df["y"].to_numpy(), df["t"], x), from_arrays, rtol=RTOL)


@pytest.mark.tier2
@pytest.mark.parametrize("name", sorted(SLOPES))
def test_a_treatment_and_controls_that_disagree_are_refused_when_the_outcome_has_no_labels(name):
    """The comparison is among the labelled inputs, whichever they are."""
    df = frame()
    shuffled = df.sample(frac=1, random_state=0)
    with pytest.raises(ValueError, match="T and X carry different pandas indexes"):
        SLOPES[name](df["y"].to_numpy(), df["t"], shuffled[["x0", "x1"]])


LABELS = {
    "dates": lambda n: pd.date_range("2020-01-01", periods=n, freq="D"),
    "strings": lambda n: pd.Index([f"r{i:03d}" for i in range(n)]),
    "repeated": lambda n: pd.Index(np.arange(n) % 7),
    "two_levels": lambda n: pd.MultiIndex.from_arrays([np.arange(n) // 10, np.arange(n) % 10]),
}


@pytest.mark.tier2
@pytest.mark.parametrize("labels", sorted(LABELS))
@pytest.mark.parametrize("name", sorted(SLOPES))
def test_the_index_comparison_does_not_depend_on_the_kind_of_label(name, labels):
    """Dates, strings, repeated labels and tuples: equal indexes pass, reordered ones do not."""
    df = frame()
    index = LABELS[labels](len(df))
    y, t = df["y"].to_numpy(), df["t"].to_numpy()
    x = df[["x0", "x1"]].to_numpy()
    outcome, treatment = pd.Series(y, index=index), pd.Series(t, index=index)
    assert_allclose(SLOPES[name](outcome, treatment, x), SLOPES[name](y, t, x), rtol=RTOL)
    with pytest.raises(ValueError, match="Y and T carry different pandas indexes"):
        SLOPES[name](outcome, treatment.iloc[::-1], x)


@pytest.mark.tier2
@pytest.mark.parametrize("name", sorted(SLOPES))
def test_equal_values_under_other_level_names_are_refused(name):
    """(unit, period) against (period, unit) over a square panel: equal tuples, other rows.

    ``Index.equals`` ignores names, so the two indexes compare equal although row ``i`` of
    one is another observation than row ``i`` of the other.
    """
    rng = np.random.default_rng(8)
    units = periods = 12
    unit, period = np.repeat(np.arange(units), periods), np.tile(np.arange(periods), units)
    x = rng.normal(size=(units * periods, 2))
    t = x[:, 0] + rng.normal(size=units * periods)
    y = 2 * t + x[:, 1] + rng.normal(size=units * periods)
    by_unit = pd.MultiIndex.from_arrays([unit, period], names=["unit", "period"])
    outcome = pd.Series(y, index=by_unit)
    treatment = pd.Series(t, index=by_unit).swaplevel().sort_index()
    assert outcome.index.equals(treatment.index)
    with pytest.raises(ValueError, match=r"levels are named \['unit', 'period'\] and \['period'"):
        SLOPES[name](outcome, treatment, x)
    # One name against another on a single level is the same doubt.
    with pytest.raises(ValueError, match=r"levels are named \['household'\] and \['week'\]"):
        SLOPES[name](pd.Series(y).rename_axis("household"), pd.Series(t).rename_axis("week"), x)
    # A named index beside an unnamed one with the same values is no disagreement.
    reference = SLOPES[name](y, t, x)
    assert_allclose(SLOPES[name](pd.Series(y).rename_axis("row"), pd.Series(t), x), reference)


class _LabelsLikePandas:
    """The index of a container that follows pandas without being pandas."""

    def __init__(self, labels):
        self.labels = list(labels)

    def equals(self, other):
        return self.labels == list(getattr(other, "labels", other))


class _SeriesLikePandas:
    def __init__(self, values, labels):
        self.values, self.index = np.asarray(values, dtype=float), _LabelsLikePandas(labels)

    def __array__(self, dtype=None, copy=None):
        return self.values if dtype is None else self.values.astype(dtype)


@pytest.mark.tier2
@pytest.mark.parametrize("name", sorted(SLOPES))
def test_an_index_is_whatever_can_be_compared_not_only_a_pandas_index(name):
    """Containers that follow pandas carry labels too; the comparison must not skip them."""
    df = frame()
    shuffled = df.sample(frac=1, random_state=0)
    x = df[["x0", "x1"]].to_numpy()
    outcome = _SeriesLikePandas(df["y"], df.index)
    reference = SLOPES[name](df["y"].to_numpy(), df["t"].to_numpy(), x)
    assert_allclose(SLOPES[name](outcome, _SeriesLikePandas(df["t"], df.index), x), reference)
    with pytest.raises(ValueError, match="Y and T carry different pandas indexes"):
        SLOPES[name](outcome, _SeriesLikePandas(shuffled["t"], shuffled.index), x)


@pytest.mark.tier1
def test_the_refusal_names_the_safe_way_through_first():
    """Aligning by label mispairs rows that were already in the same order under other labels."""
    outcome = pd.Series([1.0, 2.0, 3.0], index=[10, 11, 12])
    with pytest.raises(ValueError) as refusal:
        as_validated_arrays(outcome, pd.Series([0.0, 1.0, 0.0]), np.ones((3, 1)))
    message = str(refusal.value)
    assert 0 < message.index(".to_numpy()") < message.index(".loc")


# ---------------------------------------------------------------- values, not codes or ticks


@pytest.mark.tier2
@pytest.mark.parametrize("name", sorted(SLOPES))
def test_integer_boolean_and_list_inputs_are_read_as_their_values(name):
    """Doses of 0, 10 and 50 stored as integers are those numbers, not category codes."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=(300, 2))
    doses = rng.choice([0, 10, 50], size=300)
    y = 2.0 * doses + x[:, 1] + rng.normal(size=300)
    reference = SLOPES[name](y, doses.astype(float), x)
    assert_allclose(reference, 2.0, atol=0.05)
    assert_allclose(SLOPES[name](y, doses, x), reference, rtol=RTOL)
    assert_allclose(SLOPES[name](y.tolist(), doses.tolist(), x), reference, rtol=RTOL)
    assert_allclose(SLOPES[name](y, pd.Series(doses, dtype="Int64"), x), reference, rtol=RTOL)
    binary = doses > 0
    assert_allclose(SLOPES[name](y, binary, x), SLOPES[name](y, binary.astype(float), x), rtol=RTOL)


def unreadable(kind, y, t):
    """An outcome or a treatment whose conversion to floats would change its meaning."""
    days = pd.to_timedelta(np.arange(len(t)), unit="D")
    return {
        "timedelta_outcome": (pd.to_timedelta(y, unit="h"), t, "Y has dtype timedelta64"),
        "datetime_treatment": (y, pd.Timestamp("2020-01-01") + days, "T has dtype datetime64"),
        "complex_outcome": (y + 3j * t, t, "Y has dtype complex128"),
        "masked_outcome": (
            np.ma.array(y, mask=np.arange(len(y)) % 10 == 0),
            t,
            "Y is a masked array with masked entries",
        ),
        "string_treatment": (y, np.array(["a", "b"] * (len(t) // 2)), "T has dtype <U1"),
        "numeric_text": (pd.Series([str(v) for v in y], dtype=object), t, "Y holds text"),
        "text_categories": (y, pd.Categorical(["0", "1"] * (len(t) // 2)), "T holds text"),
        "mixed_objects": (y, [None, object()] * (len(t) // 2), "T holds object values"),
    }[kind]


@pytest.mark.tier2
@pytest.mark.parametrize(
    "kind",
    [
        "timedelta_outcome",
        "datetime_treatment",
        "complex_outcome",
        "masked_outcome",
        "string_treatment",
        "numeric_text",
        "text_categories",
        "mixed_objects",
    ],
)
@pytest.mark.parametrize("name", sorted(SLOPES))
def test_values_that_are_not_numbers_are_refused_by_name(name, kind):
    """Read as floats, a timedelta becomes integer ticks and its slope depends on the unit.

    A first version of this change converted everything with ``dtype=float``: a timedelta
    outcome gave slopes of 7.3e3, 7.3e6 and 7.3e9 for seconds, milliseconds and
    microseconds, a complex outcome lost its imaginary part, and a masked one was read
    under its mask.
    """
    df = frame()
    y, t, x = df["y"].to_numpy(), df["t"].to_numpy(), df[["x0", "x1"]].to_numpy()
    outcome, treatment, message = unreadable(kind, y, t)
    with pytest.raises(ValueError, match=f"^{message}"):
        SLOPES[name](outcome, treatment, x)


@pytest.mark.tier1
def test_a_categorical_of_numbers_is_read_by_its_values_not_its_codes():
    """Categories listed as [20, 10] give 20 the code 0; the values are what was measured."""
    doses = pd.Categorical([10, 20, 20, 10], categories=[20, 10])
    assert doses.codes.tolist() == [1, 0, 0, 1]
    _, T, _ = as_validated_arrays([1.0, 2.0, 3.0, 4.0], doses, np.ones((4, 1)))
    assert T.tolist() == [10.0, 20.0, 20.0, 10.0]


def object_column(kind, y):
    """An outcome that NumPy or pandas would cast to floats although it is not a number."""
    hours = np.round(y * 3600)
    stamps = pd.Timestamp("2020-01-01", tz="UTC") + pd.to_timedelta(hours, unit="s")
    return {
        "tz_aware_series_s": (pd.Series(stamps).dt.as_unit("s"), "Timestamp"),
        "tz_aware_series_ns": (pd.Series(stamps).dt.as_unit("ns"), "Timestamp"),
        "tz_aware_index": (pd.DatetimeIndex(stamps), "Timestamp"),
        "tz_aware_categorical": (pd.Series(pd.Categorical(stamps)), "Timestamp"),
        "numpy_complex_objects": (
            np.array([np.complex128(complex(v, 5.0)) for v in y], dtype=object),
            "complex128",
        ),
        "python_complex_objects": (np.array([complex(v, 5.0) for v in y], dtype=object), "complex"),
        "numpy_timedelta_objects": (
            np.array([np.timedelta64(int(v), "ms") for v in hours], dtype=object),
            "timedelta64",
        ),
        "numpy_datetime_objects": (
            np.array([np.datetime64("2020-01-01") + np.timedelta64(i, "D") for i in range(len(y))])
            .astype(object)
            .astype("O"),
            "date",
        ),
        "bytearray_objects": (
            np.array([bytearray(repr(v).encode()) for v in y.tolist()] + [None], dtype=object)[:-1],
            "bytearray",
        ),
        "memoryview_objects": (
            np.array([memoryview(repr(v).encode()) for v in y.tolist()] + [None], dtype=object)[
                :-1
            ],
            "memoryview",
        ),
        "not_a_time": (np.array([*y.tolist()[:-1], pd.NaT], dtype=object), "NaTType"),
    }[kind]


@pytest.mark.tier2
@pytest.mark.parametrize(
    "kind",
    [
        "tz_aware_series_s",
        "tz_aware_series_ns",
        "tz_aware_index",
        "tz_aware_categorical",
        "numpy_complex_objects",
        "python_complex_objects",
        "numpy_timedelta_objects",
        "numpy_datetime_objects",
        "bytearray_objects",
        "memoryview_objects",
        "not_a_time",
    ],
)
@pytest.mark.parametrize("name", sorted(SLOPES))
def test_an_object_column_is_judged_by_its_elements(name, kind):
    """What can be cast to floats is not thereby a number.

    ``np.asarray(values, dtype=float)`` turns a timezone-aware datetime into ticks, drops the
    imaginary part of ``np.complex128`` objects with a warning, and parses a ``bytearray``.
    The unreworked functions raised on all of these.
    """
    df = frame()
    y, t, x = df["y"].to_numpy(), df["t"].to_numpy(), df[["x0", "x1"]].to_numpy()
    outcome, element = object_column(kind, y)
    with pytest.raises(
        ValueError, match=rf"^Y holds {element} values \(for example .*\), not real"
    ):
        SLOPES[name](outcome, t, x)
    with pytest.raises(ValueError, match=rf"^T holds {element} values"):
        SLOPES[name](y, outcome, x)


@pytest.mark.tier1
def test_real_numbers_in_an_object_column_are_read():
    """The element check admits every real number, and nothing is lost on the way."""
    mixed = [1, 2.5, True, np.float32(0.5), np.int8(-3), np.bool_(False), np.uint16(7)]
    mixed += [decimal.Decimal("0.25"), fractions.Fraction(1, 8), None, float("nan")]
    read = read_as_floats("Y", np.array(mixed, dtype=object))
    assert read[:9].tolist() == [1.0, 2.5, 1.0, 0.5, -3.0, 0.0, 7.0, 0.25, 0.125]
    assert np.isnan(read[9:]).all()
    assert read_as_floats("T", np.array([3, 4], dtype=np.uint64)).tolist() == [3.0, 4.0]


@pytest.mark.tier1
def test_text_is_refused_wherever_it_stands_and_whatever_its_type():
    with pytest.raises(ValueError, match=r"^Y holds text \(for example 'b'\)"):
        read_as_floats("Y", np.array([1.0, 2.0, "b", 4.0], dtype=object))
    with pytest.raises(ValueError, match=r"^Y holds text \(for example b'7'\)"):
        read_as_floats("Y", np.array([1.0, b"7"], dtype=object))


@pytest.mark.tier1
def test_inputs_that_cannot_be_read_are_refused_by_name():
    """A ragged list, a missing value NumPy cannot cast, and an integer too large for a float."""
    with pytest.raises(ValueError, match="^Y could not be read as an array"):
        read_as_floats("Y", [[1.0, 2.0], [3.0]])
    with pytest.raises(ValueError, match="^T could not be read as a float array"):
        read_as_floats("T", [1.0, pd.NA])
    with pytest.raises(ValueError, match="^T could not be read as a float array"):
        read_as_floats("T", pd.Series([1.0, pd.NA], dtype=object))
    with pytest.raises(ValueError, match="^Y could not be read as a float array"):
        read_as_floats("Y", [10**400, 1])
    assert np.isnan(read_as_floats("Y", [1.0, None])[1])


@pytest.mark.tier1
def test_a_mask_without_masked_entries_and_missing_values_are_read():
    """An unmasked masked array is its data; ``pd.NA`` is NaN for the finite checks to refuse."""
    values = np.ma.array([1.0, 2.0, 3.0], mask=False)
    Y, _, _ = as_validated_arrays(values, [0, 1, 0], np.ones((3, 1)))
    assert Y.tolist() == [1.0, 2.0, 3.0]
    nullable = pd.Series([1.0, 2.0, pd.NA], dtype="Float64")
    Y, _, _ = as_validated_arrays(nullable, pd.Series([0, 1, 0], dtype="Int64"), np.ones((3, 1)))
    assert np.isnan(Y[2]) and Y[:2].tolist() == [1.0, 2.0]
    # A nullable boolean column with a missing value is the one that arrives as objects.
    flags = pd.Series([True, False, pd.NA], dtype="boolean")
    assert np.asarray(flags).dtype == object
    _, T, _ = as_validated_arrays([1.0, 2.0, 3.0], flags, np.ones((3, 1)))
    assert T[:2].tolist() == [1.0, 0.0] and np.isnan(T[2])


@pytest.mark.tier2
@pytest.mark.parametrize("name", sorted(SLOPES))
def test_a_one_column_frame_is_still_refused(name):
    """Reading by value comes before the shape check of #72 and must not weaken it."""
    df = frame()
    x = df[["x0", "x1"]].to_numpy()
    with pytest.raises(ValueError, match=r"T must be one-dimensional, got shape \(300, 1\)"):
        SLOPES[name](df["y"], df[["t"]], x)
    with pytest.raises(ValueError, match=r"Y must be one-dimensional, got shape \(300, 1\)"):
        SLOPES[name](df[["y"]], df["t"], x)


# ---------------------------------------------------------------- the controls are left alone


@pytest.mark.tier2
def test_robinson_hands_a_frame_of_controls_to_the_model_as_given():
    """A pipeline that selects columns by name and encodes a string column needs the frame."""
    df = frame()
    rng = np.random.default_rng(5)
    controls = df[["x0", "x1"]].assign(region=rng.choice(["west", "east"], size=len(df)))
    by_name = make_pipeline(
        ColumnTransformer([("region", OneHotEncoder(), ["region"])], remainder="passthrough"),
        Ridge(),
    )
    result = robinson_estimator(df["y"], df["t"], controls, model=by_name)
    assert_allclose(result.theta, 2.0, atol=0.35)


@pytest.mark.tier2
def test_double_ml_takes_a_frame_of_controls():
    """Fold rows are selected by position, so a frame gives the result of its arrays.

    ``double_ml`` selected them with ``X[rows]``, which reads a DataFrame's columns by label:
    every frame of controls raised ``KeyError`` before a model saw it.
    """
    df = frame()
    y, t, x = df["y"].to_numpy(), df["t"].to_numpy(), df[["x0", "x1"]].to_numpy()
    reference = double_ml(y, t, x, model="ridge").theta
    assert_allclose(double_ml(df["y"], df["t"], df[["x0", "x1"]], model="ridge").theta, reference)
    assert_allclose(double_ml(y, t, df[["x0", "x1"]], model="ridge").theta, reference, rtol=RTOL)


@pytest.mark.tier2
def test_fwl_estimate_refuses_sparse_controls_by_name():
    """It computes on a dense design; the refusal was NumPy's, from stacking the intercept."""
    df = frame()
    with pytest.raises(ValueError, match=r"^X is a sparse matrix.*X\.toarray\(\)"):
        fwl_estimate(
            df["y"].to_numpy(), df["t"].to_numpy(), sparse.csr_matrix(df[["x0", "x1"]].to_numpy())
        )


@pytest.mark.tier2
@pytest.mark.parametrize("name", ["double_ml", "robinson_estimator"])
def test_sparse_controls_give_the_dense_result(name):
    df = frame()
    x = df[["x0", "x1"]].to_numpy()
    y, t = df["y"].to_numpy(), df["t"].to_numpy()
    assert_allclose(SLOPES[name](y, t, sparse.csr_matrix(x)), SLOPES[name](y, t, x), rtol=1e-9)


@pytest.mark.tier2
@pytest.mark.parametrize("name", ["fwl_estimate", "robinson_estimator"])
def test_a_numeric_frame_of_controls_gives_the_array_result(name):
    df = frame()
    y, t = df["y"].to_numpy(), df["t"].to_numpy()
    controls = df[["x0", "x1"]]
    assert_allclose(
        SLOPES[name](df["y"], df["t"], controls), SLOPES[name](y, t, controls.to_numpy()), rtol=RTOL
    )


@pytest.mark.tier2
@pytest.mark.parametrize("name", sorted(SLOPES))
def test_controls_given_as_a_list_or_as_one_series_give_the_array_result(name):
    """A list has no column names or dtypes to keep; a Series is one column of controls."""
    df = frame()
    y, t = df["y"].to_numpy(), df["t"].to_numpy()
    x = df[["x0", "x1"]].to_numpy()
    assert_allclose(SLOPES[name](y, t, x.tolist()), SLOPES[name](y, t, x), rtol=RTOL)
    one = df["x0"]
    assert_allclose(
        SLOPES[name](df["y"], df["t"], one),
        SLOPES[name](y, t, one.to_numpy().reshape(-1, 1)),
        rtol=RTOL,
    )


@pytest.mark.tier2
@pytest.mark.parametrize("controls", [None, 3.0])
@pytest.mark.parametrize("name", sorted(SLOPES))
def test_controls_without_rows_are_refused_by_name(name, controls):
    df = frame()
    with pytest.raises(ValueError, match="^X must hold the controls with one row per observation"):
        SLOPES[name](df["y"].to_numpy(), df["t"].to_numpy(), controls)


# ---------------------------------------------------------------- the helpers


@pytest.mark.tier1
def test_as_validated_arrays_returns_float_arrays_and_a_column():
    y = pd.Series([1.0, 2.0, 3.0], index=[2, 0, 1])
    Y, T, X = as_validated_arrays(y, [0, 1, 0], np.array([5.0, 6.0, 7.0]))
    assert (type(Y), type(T)) == (np.ndarray, np.ndarray)
    assert (Y.dtype, T.dtype) == (np.float64, np.float64)
    assert Y.tolist() == [1.0, 2.0, 3.0]
    assert T.tolist() == [0.0, 1.0, 0.0]
    assert X.tolist() == [[5.0], [6.0], [7.0]]


@pytest.mark.tier1
def test_as_validated_arrays_returns_the_controls_it_was_given():
    controls = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["u", "v", "w"]})
    _, _, X = as_validated_arrays([1.0, 2.0, 3.0], [0, 1, 0], controls)
    assert X is controls
    matrix = sparse.csr_matrix(np.eye(3))
    _, _, X = as_validated_arrays([1.0, 2.0, 3.0], [0, 1, 0], matrix)
    assert X is matrix


@pytest.mark.tier1
def test_as_validated_arrays_keeps_the_length_and_shape_errors():
    column = np.ones((3, 1))
    with pytest.raises(ValueError, match="T length \\(2\\) must match Y length \\(3\\)"):
        as_validated_arrays([1.0, 2.0, 3.0], [1.0, 2.0], column)
    with pytest.raises(ValueError, match="X rows \\(2\\) must match Y length \\(3\\)"):
        as_validated_arrays([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], np.ones((2, 1)))
    with pytest.raises(ValueError, match="Y must be one-dimensional"):
        as_validated_arrays([[1.0], [2.0]], [1.0, 2.0], np.ones((2, 1)))


@pytest.mark.tier1
@pytest.mark.parametrize("scale", [1e-300, 1e-150, 1.0, 1e150, 1e300])
def test_constancy_is_judged_the_same_at_every_scale(scale):
    """Norms overflow above about 1e154 and underflow below 1e-162; the verdict must not."""
    varying = scale * np.array([1.0, 2.0, 3.0, 5.0])
    assert not is_numerically_constant(varying)
    assert is_numerically_constant(np.full(50, 0.1 * scale))
    assert is_numerically_constant(scale * np.array([0.1 + 0.2, 0.3, 0.3, 0.1 + 0.2]))


@pytest.mark.tier1
def test_constancy_edge_cases():
    """An empty array has nothing that varies; an infinite entry is not a constant level."""
    assert is_numerically_constant(np.array([]))
    assert not is_numerically_constant(np.array([1.0, np.inf]))
    assert not is_numerically_constant(np.array([1.0, np.nan]))
    assert is_numerically_constant(np.zeros(5))


@pytest.mark.tier1
def test_the_constancy_bound_grows_with_the_number_of_values():
    """Rounding of a mean accumulates with the count: eight ulps apart is residue in 50 values."""
    eps = np.finfo(float).eps
    wobble = np.where(np.arange(50) % 2 == 0, 0.3, 0.3 * (1 + 8 * eps))
    assert np.ptp(wobble) > 0
    assert is_numerically_constant(wobble)
    assert not is_numerically_constant(np.where(np.arange(50) % 2 == 0, 0.3, 0.3 * (1 + 1e-9)))
