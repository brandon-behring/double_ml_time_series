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


# ---------------------------------------------------------------------------
# B3: _validate is no longer a no-op — degenerate numerics fail at
# construction (issue #11). These replace the pre-B3 seam-fires test.
# ---------------------------------------------------------------------------


def _fwl_kwargs(**over: Any) -> dict[str, Any]:
    z = np.zeros(3)
    kw: dict[str, Any] = {
        "theta": 1.0,
        "se": 0.1,
        "t_stat": 10.0,
        "p_value": 0.0,
        "Y_residual": z,
        "T_residual": z,
        "r2_Y": 0.5,
        "r2_T": 0.5,
    }
    kw.update(over)
    return kw


def _dyn_kwargs(**over: Any) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "theta_t": np.array([1.0, 2.0]),
        "se_t": np.array([0.1, 0.1]),
        "ci_lower_t": np.array([0.8, 1.8]),
        "ci_upper_t": np.array([1.2, 2.2]),
        "p_value_t": np.array([0.001, 0.001]),
        "cumulative_effect": 3.0,
        "cumulative_se": 0.15,
        "n_units": 100,
        "n_periods": 2,
        "alpha": 0.05,
        "cov": np.eye(2),
        "nuisance_r2_t": np.array([0.9, 0.85]),
    }
    kw.update(over)
    return kw


def _scalar_inference_kwargs(**over: Any) -> dict[str, Any]:
    """Common scalar-inference fields shared by DMLResult-style classes."""
    z = np.zeros(3)
    kw: dict[str, Any] = {
        "theta": 1.0,
        "se": 0.1,
        "t_stat": 10.0,
        "p_value": 0.0,
        "ci_lower": 0.8,
        "ci_upper": 1.2,
        "Y_residual": z,
        "T_residual": z,
    }
    kw.update(over)
    return kw


def _make_dml(**over: Any) -> Any:
    from dml_ts.dml import DMLResult

    kw = _scalar_inference_kwargs(**over)
    kw.update(influence_scores=np.zeros(3), outcome_r2_cv=0.5, treatment_r2_cv=0.5, n_folds=3)
    return DMLResult(**kw)


def _make_tplr(**over: Any) -> Any:
    from dml_ts.dml import TemporalPLRDMLResult

    kw = _scalar_inference_kwargs(**over)
    kw.update(
        influence_scores=np.zeros(3),
        outcome_r2_cv=0.5,
        treatment_r2_cv=0.5,
        n_samples=3,
        n_periods=3,
        hac_bandwidth=1,
        cv_strategy="time_series_split",
    )
    return TemporalPLRDMLResult(**kw)


def _make_robinson(**over: Any) -> Any:
    from dml_ts.dml.robinson import RobinsonResult

    z = np.zeros(3)
    kw: dict[str, Any] = {
        "theta": 1.0,
        "se": 0.1,
        "Y_residual": z,
        "T_residual": z,
        "outcome_r2": 0.5,
        "treatment_r2": 0.5,
    }
    kw.update(over)
    return RobinsonResult(**kw)


class TestValidateRejectsDegenerateNumerics:
    @pytest.mark.parametrize("bad_se", [0.0, -0.5, float("nan"), float("inf")])
    def test_fwl_degenerate_se_raises(self, bad_se: float) -> None:
        with pytest.raises(ValueError, match="se"):
            FWLResult(**_fwl_kwargs(se=bad_se))

    def test_fwl_nonfinite_theta_raises(self) -> None:
        with pytest.raises(ValueError, match="theta"):
            FWLResult(**_fwl_kwargs(theta=float("nan")))

    def test_fwl_pvalue_outside_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="p_value"):
            FWLResult(**_fwl_kwargs(p_value=1.5))

    def test_dynamic_degenerate_se_raises(self) -> None:
        with pytest.raises(ValueError, match="se_t"):
            DynamicGEstimationResult(**_dyn_kwargs(se_t=np.array([0.1, 0.0])))

    def test_dynamic_inverted_ci_raises(self) -> None:
        with pytest.raises(ValueError, match="ci_t"):
            DynamicGEstimationResult(**_dyn_kwargs(ci_lower_t=np.array([2.0, 1.8])))

    def test_dynamic_non_psd_cov_raises(self) -> None:
        with pytest.raises(ValueError, match="cov"):
            DynamicGEstimationResult(**_dyn_kwargs(cov=np.array([[1.0, 2.0], [2.0, 1.0]])))

    def test_estimator_results_pass_validation(self, result_instances: list[Any]) -> None:
        """Every real fit constructs cleanly — validators reject only
        degenerate numerics, never healthy paths (golden-gated too)."""
        assert len(result_instances) >= 4

    @pytest.mark.parametrize("factory", [_make_dml, _make_tplr, _make_robinson])
    def test_scalar_results_degenerate_se_raises(self, factory: Any) -> None:
        """Every scalar result class hard-fails on degenerate SEs — a deleted
        hook would let these constructions pass (mutation-probe finding)."""
        with pytest.raises(ValueError, match=r"\.se"):
            factory(se=0.0)
        with pytest.raises(ValueError, match=r"\.se"):
            factory(se=float("nan"))
        with pytest.raises(ValueError, match="theta"):
            factory(theta=float("inf"))

    @pytest.mark.parametrize("factory", [_make_dml, _make_tplr])
    def test_scalar_results_inverted_ci_raises(self, factory: Any) -> None:
        with pytest.raises(ValueError, match=r"\.ci"):
            factory(ci_lower=2.0, ci_upper=1.0)
