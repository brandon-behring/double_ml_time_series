"""Inputs are read by position, whatever container they come in (#75).

``double_ml``, ``fwl_estimate`` and ``robinson_estimator`` indexed a pandas ``Series`` by label
where they indexed the controls by position. After ``df.sample(frac=1)`` ``double_ml`` returned
1.82 where the arrays give 2.03, and with different labels on the outcome and the treatment
``fwl_estimate`` and ``robinson_estimator`` returned -0.15 for a slope of 2.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from dml_ts.dml import double_ml
from dml_ts.dml._utils import as_validated_arrays
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


@pytest.mark.tier2
@pytest.mark.parametrize("reordering", sorted(REORDERINGS))
@pytest.mark.parametrize("name", sorted(SLOPES))
def test_a_reordered_frame_gives_the_slope_of_its_arrays(name, reordering):
    """The index of a Series must not decide which rows are paired."""
    df = REORDERINGS[reordering](frame())
    assert not df.index.equals(pd.RangeIndex(len(df)))
    controls = df[["x0", "x1"]]
    from_arrays = SLOPES[name](df["y"].to_numpy(), df["t"].to_numpy(), controls.to_numpy())
    assert_allclose(SLOPES[name](df["y"], df["t"], controls.to_numpy()), from_arrays, rtol=RTOL)
    assert_allclose(SLOPES[name](df["y"], df["t"], controls), from_arrays, rtol=RTOL)
    assert_allclose(from_arrays, 2.0, atol=0.35)


@pytest.mark.tier2
@pytest.mark.parametrize("name", sorted(SLOPES))
def test_different_labels_on_outcome_and_treatment_are_paired_by_position(name):
    """Labels 50..349 against 0..299: aligned by label, 250 rows met the wrong partner."""
    df = frame()
    x = df[["x0", "x1"]].to_numpy()
    outcome = pd.Series(df["y"].to_numpy(), index=range(50, 350))
    treatment = pd.Series(df["t"].to_numpy(), index=range(300))
    from_arrays = SLOPES[name](df["y"].to_numpy(), df["t"].to_numpy(), x)
    assert_allclose(SLOPES[name](outcome, treatment, x), from_arrays, rtol=RTOL)


@pytest.mark.tier2
@pytest.mark.parametrize("name", sorted(SLOPES))
def test_lists_and_integer_treatments_are_read_as_their_values(name):
    """A list, and a 0/1 treatment stored as integers or booleans, give the float result."""
    df = frame()
    x = df[["x0", "x1"]].to_numpy()
    y = df["y"].to_numpy()
    binary = (df["t"].to_numpy() > 0).astype(float)
    reference = SLOPES[name](y, binary, x)
    assert_allclose(SLOPES[name](y.tolist(), binary.tolist(), x.tolist()), reference, rtol=RTOL)
    assert_allclose(SLOPES[name](y, binary.astype(int), x), reference, rtol=RTOL)
    assert_allclose(SLOPES[name](y, binary.astype(bool), x), reference, rtol=RTOL)


@pytest.mark.tier2
@pytest.mark.parametrize("name", sorted(SLOPES))
def test_a_one_column_frame_is_still_refused_after_the_conversion(name):
    """Conversion comes before the shape check of #72 and must not weaken it."""
    df = frame()
    x = df[["x0", "x1"]].to_numpy()
    with pytest.raises(ValueError, match=r"T must be one-dimensional, got shape \(300, 1\)"):
        SLOPES[name](df["y"], df[["t"]], x)
    with pytest.raises(ValueError, match=r"Y must be one-dimensional, got shape \(300, 1\)"):
        SLOPES[name](df[["y"]], df["t"], x)


@pytest.mark.tier1
def test_as_validated_arrays_returns_positional_float_arrays():
    y = pd.Series([1.0, 2.0, 3.0], index=[2, 0, 1])
    Y, T, X = as_validated_arrays(y, [0, 1, 0], pd.Series([5, 6, 7], index=[9, 8, 7]))
    assert (type(Y), type(T), type(X)) == (np.ndarray, np.ndarray, np.ndarray)
    assert (Y.dtype, T.dtype, X.dtype) == (np.float64, np.float64, np.float64)
    assert Y.tolist() == [1.0, 2.0, 3.0]
    assert T.tolist() == [0.0, 1.0, 0.0]
    assert X.tolist() == [[5.0], [6.0], [7.0]]


@pytest.mark.tier1
def test_as_validated_arrays_keeps_the_length_and_shape_errors():
    with pytest.raises(ValueError, match="T length \\(2\\) must match Y length \\(3\\)"):
        as_validated_arrays([1.0, 2.0, 3.0], [1.0, 2.0], [[1.0], [2.0], [3.0]])
    with pytest.raises(ValueError, match="X rows \\(2\\) must match Y length \\(3\\)"):
        as_validated_arrays([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [[1.0], [2.0]])
    with pytest.raises(ValueError, match="Y must be one-dimensional"):
        as_validated_arrays([[1.0], [2.0]], [1.0, 2.0], [[1.0], [2.0]])
