"""The outcome and the treatment are read by value and paired by position (#75).

``double_ml`` indexed a pandas ``Series`` by label where it indexed the controls by position:
after ``df.sample(frac=1)`` it returned 1.82 where the arrays give 2.03. ``fwl_estimate`` and
``robinson_estimator`` multiplied two residual ``Series`` that pandas aligned by label: with
different labels on the outcome and the treatment they returned -0.15 for a slope of 2.

The controls are not converted. A DataFrame or a sparse matrix reaches the nuisance model as
it was given, which a first version of this change broke.
"""

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
from dml_ts.dml._utils import as_validated_arrays, is_numerically_constant
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
        "mixed_objects": (y, [None, object()] * (len(t) // 2), "T could not be read as a float"),
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


@pytest.mark.tier1
def test_a_mask_without_masked_entries_and_missing_values_are_read():
    """An unmasked masked array is its data; ``pd.NA`` is NaN for the finite checks to refuse."""
    values = np.ma.array([1.0, 2.0, 3.0], mask=False)
    Y, _, _ = as_validated_arrays(values, [0, 1, 0], np.ones((3, 1)))
    assert Y.tolist() == [1.0, 2.0, 3.0]
    nullable = pd.Series([1.0, 2.0, pd.NA], dtype="Float64")
    Y, _, _ = as_validated_arrays(nullable, pd.Series([0, 1, 0], dtype="Int64"), np.ones((3, 1)))
    assert np.isnan(Y[2]) and Y[:2].tolist() == [1.0, 2.0]


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
