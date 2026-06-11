"""Contract tests for the frozen result objects (Track B1).

Every estimator result is @dataclass(frozen=True, slots=True, eq=False) on
ResultBase: immutable, slotted, identity-equality, with a shallow to_dict().
"""

import dataclasses
from typing import Any

import numpy as np
import pytest

from dml_ts.dml import TemporalPLRDML, double_ml
from dml_ts.dml.dynamic_g_estimation import DynamicGEstimationResult
from dml_ts.dml.fwl import FWLResult, fwl_estimate
from dml_ts.dml.robinson import robinson_estimator

pytestmark = pytest.mark.tier2


def _plr_data(seed: int = 0, n: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    T = 0.5 * X[:, 0] + rng.normal(size=n)
    Y = 2.0 * T + X[:, 0] + rng.normal(size=n)
    return Y, T, X


def _result_instances() -> list[Any]:
    Y, T, X = _plr_data()
    return [
        fwl_estimate(Y, T, X),
        robinson_estimator(Y, T, X, model="ridge"),
        double_ml(Y, T, X, n_folds=3, model="ridge", random_state=0),
        TemporalPLRDML(n_lags=0, model_y="ridge", model_t="ridge", random_state=0).fit(Y, T, X),
        # Direct construction: also covers the default-valued fields
        # (backend/mode) under frozen+slots.
        DynamicGEstimationResult(
            theta_t=np.array([1.0, 2.0]),
            se_t=np.array([0.1, 0.1]),
            ci_lower_t=np.array([0.8, 1.8]),
            ci_upper_t=np.array([1.2, 2.2]),
            p_value_t=np.array([0.001, 0.001]),
            cumulative_effect=3.0,
            cumulative_se=0.15,
            n_units=100,
            n_periods=2,
            alpha=0.05,
            cov=np.eye(2),
            nuisance_r2_t=np.array([0.9, 0.85]),
        ),
    ]


@pytest.fixture(scope="module")
def result_instances() -> list[Any]:
    return _result_instances()


def test_results_are_frozen(result_instances: list[Any]) -> None:
    """Post-construction field mutation is a hard error, never silent."""
    for r in result_instances:
        first_field = dataclasses.fields(r)[0].name
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(r, first_field, object())


def test_results_are_slotted(result_instances: list[Any]) -> None:
    """No silent attribute growth: unknown attributes raise.

    CPython quirk (present in 3.13.x as of 2026-06, tracked at
    python/cpython#105936): frozen+slots dataclasses raise TypeError (stale
    class cell in the generated __setattr__'s super() call) instead of
    AttributeError for unknown names. Loud either way — the contract is
    that the write NEVER succeeds silently.
    """
    for r in result_instances:
        with pytest.raises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):
            r.definitely_not_a_field = 1


def test_to_dict_covers_all_fields(result_instances: list[Any]) -> None:
    for r in result_instances:
        d = r.to_dict()
        assert set(d) == {f.name for f in dataclasses.fields(r)}
        # shallow: array fields are the same objects, not copies
        for f in dataclasses.fields(r):
            v = getattr(r, f.name)
            if isinstance(v, np.ndarray):
                assert d[f.name] is v


def test_eq_is_identity(result_instances: list[Any]) -> None:
    """eq=False: array-bearing dataclasses must not value-compare."""
    for r in result_instances:
        assert r == r
        assert hash(r) == hash(r)  # usable in sets/dicts (identity hash)
    # two equal-valued results are still distinct objects
    z = np.zeros(3)
    a = FWLResult(1.0, 0.1, 10.0, 0.0, z, z, 0.5, 0.5)
    b = FWLResult(1.0, 0.1, 10.0, 0.0, z, z, 0.5, 0.5)
    assert a != b


def test_validate_hook_fires_for_every_result_class(monkeypatch: pytest.MonkeyPatch) -> None:
    """__post_init__ dispatches to _validate on construction of ALL classes.

    This is the seam the B3 validators fill; if a future result class
    defines its own __post_init__ without chaining, _validate silently
    never runs — this test makes that loud for every shipped class.
    """
    from dml_ts.dml._results import ResultBase

    calls: list[str] = []
    monkeypatch.setattr(ResultBase, "_validate", lambda self: calls.append(type(self).__name__))

    instances = _result_instances()
    constructed = [type(r).__name__ for r in instances]
    for name in constructed:
        assert name in calls, f"_validate did not fire for {name}"
