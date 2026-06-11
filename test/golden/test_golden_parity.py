"""Golden-snapshot parity gate for the Track B migration (B2.0).

Pins the numerical behavior of every splitter, HAC function, and estimator
that later Track B PRs refactor or retire. Captured AFTER the #7 (sqrt(n)
HAC SE) and #9 (cluster SE) fixes — goldens are born correct.

Rules of the harness:

- Input data is generated INLINE with frozen ``np.random.default_rng`` seeds —
  never via the validation DGP classes, so DGP-internal changes cannot
  silently invalidate these snapshots.
- Snapshots live in ``test/golden/snapshots/*.json`` (pretty-printed, sorted
  keys). Regenerate with ``DML_REGEN_GOLDEN=1 venv/bin/python -m pytest
  test/golden -q``. Any PR that intentionally changes values must regenerate
  ONLY the affected keys and declare them in the file's ``_provenance`` map —
  review audits the JSON diff.
- Tolerance: ``rtol=1e-9, atol=0`` for floats (BLAS run-to-run noise is far
  below this; real regressions like #7 are ~10x-30x). Ints, strings, and
  hashes compare exactly. If the gate ever flakes, tighten the harness
  (seeds/threads), never the tolerance.
- A missing snapshot or key FAILS loudly with regen instructions — no silent
  skips. Regen is a LOCAL, single-process operation: it is blocked in CI and
  must never run under pytest-xdist (read-modify-write per test).
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dml_ts.dml import (
    DynamicGEstimationDML,
    PanelDML,
    RollingWindowDML,
    TemporalPLRDML,
    double_ml,
)
from dml_ts.dml.cross_fitting import (
    BlockedTimeSeriesCV,
    PurgedGroupTimeSeriesCV,
    TimeSeriesCrossValidator,
)
from dml_ts.dml.fwl import fwl_estimate
from dml_ts.dml.hac import newey_west_covariance, newey_west_se, optimal_bandwidth
from dml_ts.dml.robinson import robinson_estimator

SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"
REGEN = os.environ.get("DML_REGEN_GOLDEN") == "1"
RTOL = 1e-9

if REGEN and os.environ.get("CI"):
    raise RuntimeError(
        "DML_REGEN_GOLDEN=1 is set in CI: the golden gate would pass without "
        "comparing anything. Regeneration is a local-only operation."
    )


# ---------------------------------------------------------------------------
# Snapshot plumbing
# ---------------------------------------------------------------------------


def _to_jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays to plain JSON types."""
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def _assert_matches(expected: Any, computed: Any, path: str) -> None:
    """Recursively compare a computed entry against its golden value."""
    if isinstance(expected, dict):
        assert isinstance(computed, dict), f"{path}: type changed to {type(computed)}"
        assert set(expected) == set(computed), (
            f"{path}: key set changed: {sorted(set(expected) ^ set(computed))}"
        )
        for k in expected:
            _assert_matches(expected[k], computed[k], f"{path}.{k}")
    elif isinstance(expected, list):
        assert isinstance(computed, list), f"{path}: type changed to {type(computed)}"
        assert len(expected) == len(computed), f"{path}: length {len(expected)} -> {len(computed)}"
        for i, (e, c) in enumerate(zip(expected, computed)):
            _assert_matches(e, c, f"{path}[{i}]")
    elif isinstance(expected, float):
        # Scalar pin: without this, numpy broadcasting would silently accept
        # a scalar->array refactor (assert_allclose(1.5, [1.5, 1.5]) passes).
        assert isinstance(computed, (int, float)), f"{path}: scalar -> {type(computed)}"
        np.testing.assert_allclose(
            computed, expected, rtol=RTOL, atol=0.0, equal_nan=False, err_msg=path
        )
    else:
        # Strict type identity: int golden vs float computed (or bool vs int)
        # is API drift the migration must catch, not tolerate via ==.
        assert type(computed) is type(expected), f"{path}: {type(expected)} -> {type(computed)}"
        assert computed == expected, f"{path}: {expected!r} -> {computed!r}"


def _load_entries(snapshot_path: Path) -> dict[str, Any]:
    try:
        return json.loads(snapshot_path.read_text(encoding="utf-8"))["entries"]
    except (json.JSONDecodeError, KeyError) as e:
        pytest.fail(
            f"Corrupt golden snapshot {snapshot_path} ({e!r}). "
            "Regenerate with DML_REGEN_GOLDEN=1 pytest test/golden -q"
        )


def _check(snapshot_name: str, key: str, computed: dict[str, Any]) -> None:
    """Compare ``computed`` against the golden entry, or write it under regen."""
    computed = _to_jsonable(computed)
    snapshot_path = SNAPSHOT_DIR / f"{snapshot_name}.json"

    if REGEN:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {"_provenance": {}, "entries": {}}
        if snapshot_path.exists():
            data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        data["entries"][key] = computed
        data["_provenance"].setdefault(
            "initial",
            {
                "pr": "track-B PR-4",
                "reason": "initial capture after the #7 and #9 fixes (goldens born correct)",
            },
        )
        snapshot_path.write_text(
            json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return

    if not snapshot_path.exists():
        pytest.fail(
            f"Missing golden snapshot {snapshot_path}. "
            "Regenerate with DML_REGEN_GOLDEN=1 pytest test/golden -q"
        )
    entries = _load_entries(snapshot_path)
    if key not in entries:
        pytest.fail(
            f"Missing golden key {key!r} in {snapshot_path.name}. "
            "Regenerate with DML_REGEN_GOLDEN=1 pytest test/golden -q "
            "and declare the new key in _provenance."
        )
    _assert_matches(entries[key], computed, key)


# ---------------------------------------------------------------------------
# Inline frozen data generators (NEVER the validation DGP classes)
# ---------------------------------------------------------------------------


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float = 1.0) -> np.ndarray:
    e = rng.normal(0.0, sigma, n)
    x = np.zeros(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def _iid_plr(seed: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    T = 0.5 * X[:, 0] + rng.normal(size=n)
    Y = 2.0 * T + X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=n)
    return Y, T, X


def _ts_plr(seed: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    T = _ar1(rng, n, phi=0.5) + 0.3 * X[:, 0]
    Y = 1.5 * T + X[:, 0] + rng.normal(size=n)
    return Y, T, X


def _panel(
    seed: int, n_units: int, n_periods: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = n_units * n_periods
    individual_id = np.repeat(np.arange(n_units), n_periods)
    time_id = np.tile(np.arange(n_periods), n_units)
    alpha = rng.normal(size=n_units)[individual_id]
    X = rng.normal(size=(n, 2))
    T = 0.5 * X[:, 0] + alpha + rng.normal(size=n)
    Y = 3.0 * T + X[:, 0] + alpha + rng.normal(size=n)
    return Y, T, X, individual_id, time_id


def _dynamic_panel(
    seed: int, n_units: int, m: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stacked per-period rows, unit-major, with groups — shape-valid input.

    The golden pins behavior on fixed inputs; SNMM fidelity is irrelevant here.
    """
    rng = np.random.default_rng(seed)
    rows = n_units * m
    groups = np.repeat(np.arange(n_units), m)
    period = np.tile(np.arange(m), n_units)
    X = rng.normal(size=(rows, 2))
    T = 0.5 * X[:, 0] + rng.normal(size=rows)
    Y = (1.0 + 0.5 * period) * T + X[:, 0] + 0.2 * rng.normal(size=rows)
    return Y, T, X, groups


# ---------------------------------------------------------------------------
# Layer 1: splitter indices
# ---------------------------------------------------------------------------

SPLITTER_CONFIGS: dict[str, tuple[Any, int]] = {}
for _n in (50, 200, 1000):
    for _ns in (3, 5):
        for _gap in (0, 5):
            SPLITTER_CONFIGS[f"tscv_n{_n}_s{_ns}_g{_gap}"] = (
                TimeSeriesCrossValidator(n_splits=_ns, gap=_gap),
                _n,
            )
for _n in (200, 1000):
    for _ns in (3, 5):
        SPLITTER_CONFIGS[f"blocked_n{_n}_s{_ns}"] = (
            BlockedTimeSeriesCV(n_splits=_ns, gap_blocks=1),
            _n,
        )
# Embargo values spanning TemporalPLRDML._create_cv's clamp range [0.01, 0.1]
# (it emits clamp(gap/n, 0.01, 0.1) with its configured n_splits; these pins
# parametrize the splitter directly at the floor, an interior value, and the
# ceiling for two sample sizes).
for _n, _embargo in ((200, 0.01), (200, 0.025), (200, 0.1), (600, 0.01), (600, 0.025), (600, 0.1)):
    SPLITTER_CONFIGS[f"purged_n{_n}_e{_embargo}"] = (
        PurgedGroupTimeSeriesCV(n_splits=5, embargo_pct=_embargo),
        _n,
    )


@pytest.mark.tier1
class TestSplitterGoldens:
    @pytest.mark.parametrize("key", sorted(SPLITTER_CONFIGS))
    def test_splitter_indices(self, key: str) -> None:
        splitter, n = SPLITTER_CONFIGS[key]
        X = np.zeros((n, 1))
        digest = hashlib.sha256()
        folds = []
        for train_idx, test_idx in splitter.split(X):
            # Explicit little-endian dtype: tobytes() uses native order and
            # would otherwise produce different hashes on big-endian hosts.
            train_idx = np.asarray(train_idx, dtype="<i8")
            test_idx = np.asarray(test_idx, dtype="<i8")
            digest.update(train_idx.tobytes())
            digest.update(b"|")
            digest.update(test_idx.tobytes())
            digest.update(b";")
            folds.append(
                {
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "train_head": train_idx[:3],
                    "train_tail": train_idx[-3:],
                    "test_head": test_idx[:3],
                    "test_tail": test_idx[-3:],
                }
            )
        entry = {"n_folds": len(folds), "folds": folds, "sha256": digest.hexdigest()}
        _check("splitters", key, entry)


# ---------------------------------------------------------------------------
# Layer 2: HAC function values (Andrews deliberately excluded: documented
# deviation — temporalcv fixes the Bartlett constant to alpha(1))
# ---------------------------------------------------------------------------


def _hac_series(seed: int, n: int, phi: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if phi == 0.0:
        return rng.normal(size=n)
    return _ar1(rng, n, phi=phi)


HAC_CONFIGS = {
    **{f"ar06_n120_seed{s}": (s, 120, 0.6) for s in range(5)},
    **{f"wn_n200_seed{s}": (s, 200, 0.0) for s in (10, 11, 12)},
    **{f"ar09_n300_seed{s}": (s, 300, 0.9) for s in (20, 21)},
}


@pytest.mark.tier1
class TestHACGoldens:
    @pytest.mark.parametrize("key", sorted(HAC_CONFIGS))
    def test_hac_values(self, key: str) -> None:
        seed, n, phi = HAC_CONFIGS[key]
        e = _hac_series(seed, n, phi)
        rng = np.random.default_rng(seed + 1000)
        X = np.column_stack([np.ones(n), rng.normal(size=n)])

        entry = {
            "nw_se_auto": newey_west_se(e),
            "nw_se_bw5": newey_west_se(e, bandwidth=5),
            "nw_se_parzen_bw5": newey_west_se(e, bandwidth=5, kernel="parzen"),
            "bw_newey_west": optimal_bandwidth(e, method="newey_west"),
            "sandwich_00_auto": float(newey_west_covariance(e, X)[0, 0]),
        }
        _check("hac", key, entry)


# ---------------------------------------------------------------------------
# Layer 3: estimator end-to-end
# ---------------------------------------------------------------------------


def _entry_double_ml(model: str, seed: int, n: int) -> dict[str, Any]:
    Y, T, X = _iid_plr(seed, n)
    r = double_ml(Y, T, X, n_folds=5, model=model, random_state=0)
    return {
        "theta": r.theta,
        "se": r.se,
        "t_stat": r.t_stat,
        "p_value": r.p_value,
        "ci_lower": r.ci_lower,
        "ci_upper": r.ci_upper,
        "outcome_r2_cv": r.outcome_r2_cv,
        "treatment_r2_cv": r.treatment_r2_cv,
    }


def _entry_tplr(
    cv_strategy: str, model: str, n_lags: int, gap: int, seed: int, n: int
) -> dict[str, Any]:
    Y, T, X = _ts_plr(seed, n)
    est = TemporalPLRDML(
        n_lags=n_lags,
        model_y=model,
        model_t=model,
        cv_strategy=cv_strategy,  # type: ignore[arg-type]
        n_splits=5,
        gap=gap,
        random_state=0,
    )
    r = est.fit(Y, T, X)
    return {
        "theta": r.theta,
        "se": r.se,
        "t_stat": r.t_stat,
        "p_value": r.p_value,
        "ci_lower": r.ci_lower,
        "ci_upper": r.ci_upper,
        "n_periods": r.n_periods,
        "hac_bandwidth": r.hac_bandwidth,
        "dropped_initial_rows": r.dropped_initial_rows,
        "lagged_rows_dropped": r.lagged_rows_dropped,
        "outcome_r2_cv": r.outcome_r2_cv,
        "treatment_r2_cv": r.treatment_r2_cv,
    }


ESTIMATOR_BUILDERS = {
    "double_ml_ridge": lambda: _entry_double_ml("ridge", seed=1, n=400),
    "double_ml_rf": lambda: _entry_double_ml("random_forest", seed=2, n=300),
    "fwl_qr": lambda: (
        lambda r: {
            "theta": r.theta,
            "se": r.se,
            "t_stat": r.t_stat,
            "p_value": r.p_value,
            "r2_Y": r.r2_Y,
            "r2_T": r.r2_T,
        }
    )(fwl_estimate(*_iid_plr(3, 300))),
    # RobinsonResult has no t_stat/p_value/ci fields today (naive SE only).
    # If the B1 result refactor adds inference fields, extend this entry and
    # declare the regenerated key in _provenance.
    "robinson_ridge": lambda: (
        lambda r: {
            "theta": r.theta,
            "se": r.se,
            "outcome_r2": r.outcome_r2,
            "treatment_r2": r.treatment_r2,
        }
    )(robinson_estimator(*_iid_plr(4, 300), model="ridge")),
    "tplr_time_series_ridge": lambda: _entry_tplr(
        "time_series_split", "ridge", n_lags=1, gap=2, seed=5, n=300
    ),
    "tplr_blocked_ridge": lambda: _entry_tplr(
        "blocked_cv", "ridge", n_lags=1, gap=2, seed=5, n=300
    ),
    "tplr_purged_ridge": lambda: _entry_tplr("purged_cv", "ridge", n_lags=1, gap=5, seed=5, n=300),
    "tplr_time_series_rf": lambda: _entry_tplr(
        "time_series_split", "random_forest", n_lags=0, gap=0, seed=6, n=200
    ),
    # RF on the purged path: the CV strategy's fold structure interacting with
    # a non-linear nuisance is where refactor regressions surface; ridge alone
    # would not exercise it.
    "tplr_purged_rf": lambda: _entry_tplr(
        "purged_cv", "random_forest", n_lags=0, gap=5, seed=6, n=200
    ),
}

# Keys written by the non-parametrized test methods below. The exhaustiveness
# test keeps snapshots free of stale keys — a renamed config must regenerate.
LITERAL_ESTIMATOR_KEYS = {
    "rolling_window_ridge",
    "panel_ridge_clusterFalse",
    "panel_ridge_clusterTrue",
    "dynamic_g_panel_linear",
    "dynamic_g_series_linear",
}

EXPECTED_KEYS = {
    "splitters": set(SPLITTER_CONFIGS),
    "hac": set(HAC_CONFIGS),
    "estimators": set(ESTIMATOR_BUILDERS) | LITERAL_ESTIMATOR_KEYS,
}


@pytest.mark.tier1
@pytest.mark.parametrize("name", sorted(EXPECTED_KEYS))
def test_snapshot_keys_exhaustive(name: str) -> None:
    """Snapshot files contain exactly the keys the tests pin — no stale keys.

    A renamed/retired config must regenerate its snapshot; otherwise the old
    key would linger forever, looking pinned while gating nothing.
    """
    if REGEN:
        pytest.skip("regen mode: key sets are checked on the next compare run")
    path = SNAPSHOT_DIR / f"{name}.json"
    if not path.exists():
        pytest.fail(f"Missing {path}; regenerate with DML_REGEN_GOLDEN=1 pytest test/golden -q")
    on_disk = set(_load_entries(path))
    assert on_disk == EXPECTED_KEYS[name], (
        f"{name}.json stale keys: {sorted(on_disk - EXPECTED_KEYS[name])}; "
        f"unpinned configs: {sorted(EXPECTED_KEYS[name] - on_disk)}"
    )


@pytest.mark.tier2
class TestEstimatorGoldens:
    @pytest.mark.parametrize("key", sorted(ESTIMATOR_BUILDERS))
    def test_estimator_outputs(self, key: str) -> None:
        _check("estimators", key, ESTIMATOR_BUILDERS[key]())

    def test_rolling_window_ridge(self) -> None:
        Y, T, X = _ts_plr(7, 300)
        est = RollingWindowDML(
            window_size=100, step_size=30, model_y="ridge", model_t="ridge", random_state=0
        )
        est.fit(Y, T, X)
        centers, thetas, ses = est.get_effects()
        entry = {
            "centers": centers,
            "theta_series": thetas,
            "se_series": ses,
        }
        _check("estimators", "rolling_window_ridge", entry)

    # The two panel pins are complementary, NOT redundant: clusterFalse is
    # the only panel golden on the HAC SE path (clusterTrue overwrites
    # se/t/p/CI with the CR1 block). Keep both.
    @pytest.mark.parametrize("cluster_se", [False, True])
    def test_panel_dml_ridge(self, cluster_se: bool) -> None:
        Y, T, X, individual_id, time_id = _panel(8, 30, 12)
        est = PanelDML(
            fixed_effects="individual",
            cluster_se=cluster_se,
            model_y="ridge",
            model_t="ridge",
            n_folds=5,
            random_state=0,
        )
        if cluster_se:
            # The trim disclosure is part of the golden contract for this
            # config (30 units x 12 periods: the CV prefix swallows ~5 units).
            with pytest.warns(RuntimeWarning, match="clusters have no observations"):
                r = est.fit(Y, T, X, individual_id, time_id)
        else:
            r = est.fit(Y, T, X, individual_id, time_id)
        entry = {
            "theta": r.theta,
            "se": r.se,
            "t_stat": r.t_stat,
            "p_value": r.p_value,
            "ci_lower": r.ci_lower,
            "ci_upper": r.ci_upper,
            "cv_strategy": r.cv_strategy,
        }
        _check("estimators", f"panel_ridge_cluster{cluster_se}", entry)

    def test_dynamic_g_panel_linear(self) -> None:
        Y, T, X, groups = _dynamic_panel(9, 200, 2)
        est = DynamicGEstimationDML(model_y="linear", model_t="linear", n_folds=5, random_state=0)
        r = est.fit(Y, T, X, groups=groups)
        entry = {
            "theta_t": r.theta_t,
            "se_t": r.se_t,
            "cumulative_effect": r.cumulative_effect,
            "cumulative_se": r.cumulative_se,
            "n_units": r.n_units,
            "n_periods": r.n_periods,
            "mode": r.mode,
        }
        _check("estimators", "dynamic_g_panel_linear", entry)

    def test_dynamic_g_series_linear(self) -> None:
        rng = np.random.default_rng(10)
        n = 400
        X = rng.normal(size=(n, 2))
        T = _ar1(rng, n, phi=0.4) + 0.3 * X[:, 0]
        Y = 1.0 * T + np.concatenate([[0.0], 0.5 * T[:-1]]) + X[:, 0] + rng.normal(size=n)
        est = DynamicGEstimationDML(
            n_periods=2, model_y="linear", model_t="linear", n_folds=5, random_state=0
        )
        r = est.fit(Y, T, X, mode="series")
        entry = {
            "theta_t": r.theta_t,
            "se_t": r.se_t,
            "cumulative_effect": r.cumulative_effect,
            "cumulative_se": r.cumulative_se,
            "n_periods": r.n_periods,
            "mode": r.mode,
        }
        _check("estimators", "dynamic_g_series_linear", entry)
