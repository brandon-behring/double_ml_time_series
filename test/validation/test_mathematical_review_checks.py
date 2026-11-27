"""Check the mathematical review's helpers against known answers and its archived draws."""

from __future__ import annotations

import doctest
import gzip
import hashlib
import importlib
import inspect
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from temporalcv import PurgedKFold

from dml_ts.dml import TemporalPLRDML
from dml_ts.dml._utils import ModelType, get_nuisance_model
from dml_ts.validation.dynamic_dgp import DynamicTreatmentDGP

checks = importlib.import_module("docs.audits.2026-11-22_mathematical_review.checks")
pytestmark = pytest.mark.tier2

ARCHIVE = Path(str(checks.__file__)).parent
EXAMPLES = [test for test in doctest.DocTestFinder().find(checks) if test.examples]


@pytest.fixture(scope="module")
def draws() -> pd.DataFrame:
    """Load the archived draws once.

    Examples
    --------
    ``def test_x(draws): assert not draws.empty``
    """
    return pd.read_csv(ARCHIVE / "draws.csv.gz")


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda test: test.name.split("checks.")[-1])
def test_documented_example_runs(example: doctest.DocTest) -> None:
    """Each driver example is executable, e.g. ``cosine_dof(415) == 22``."""
    report: list[str] = []
    result = doctest.DocTestRunner().run(example, out=report.append)
    assert result.failed == 0, "".join(report)


def test_every_driver_function_has_an_executed_example() -> None:
    """No function escapes the previous test, e.g. ``cosine_dof`` must appear in it."""
    defined = {
        name
        for name, value in vars(checks).items()
        if inspect.isfunction(value) and value.__module__ == checks.__name__
    }
    assert defined <= {test.name.rsplit(".", 1)[-1] for test in EXAMPLES}


def test_cosine_variance_recovers_known_ar1_long_run_variance() -> None:
    """AR(0.6) with unit innovations has variance 1 / (1 - 0.6)^2 = 6.25."""
    rng = np.random.default_rng(20261122)
    series = checks.innovation_ar1(rng, (60, 20000), 0.6)
    estimates = [checks.cosine_long_run_variance(row)[0] for row in series]
    # One estimate has relative SD sqrt(2 / nu), about 8%; the mean of 60 has about 1.1%.
    assert np.mean(estimates) == pytest.approx(1 / (1 - 0.6) ** 2, rel=0.05)


@pytest.mark.parametrize("seed", range(12))
def test_cosine_variance_ignores_shifts_and_scales_quadratically(seed: int) -> None:
    """For any series, Omega(c x + a) = c^2 Omega(x), e.g. c = 2 gives four times the variance."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(rng.integers(10, 600)))
    shift, scale = rng.uniform(-9, 9, size=2)
    omega, nu = checks.cosine_long_run_variance(x)
    moved, _ = checks.cosine_long_run_variance(scale * x + shift)
    assert omega >= 0 and 2 <= nu < len(x)
    assert moved == pytest.approx(scale**2 * omega, rel=1e-8)


@pytest.mark.parametrize("seed", range(20))
def test_two_sided_folds_respect_gap_and_score_every_row_once(seed: int) -> None:
    """Geometry holds for random sizes, e.g. n = 100, K = 5, gap = 2."""
    rng = np.random.default_rng(seed)
    n, k = int(rng.integers(60, 500)), int(rng.integers(2, 11))
    gap = int(rng.integers(0, 9))
    folds = checks.buffered_two_sided_splits(n, k, gap)
    tested = np.sort(np.concatenate([test for _, test in folds]))
    np.testing.assert_array_equal(tested, np.arange(n))
    for train, test in folds:
        distance = np.abs(train[:, None] - test[None, :]).min(axis=1)
        assert (distance > gap).all()
        omitted = np.setdiff1d(np.arange(n), train)
        assert omitted.min() == max(0, test[0] - gap)
        assert omitted.max() == min(n - 1, test[-1] + gap)


@pytest.mark.parametrize("gap", [0, 2, 10])
def test_two_sided_folds_equal_upstream_when_blocks_are_equal(gap: int) -> None:
    """For n = 500 and K = 5 the helper is ``PurgedKFold`` without an embargo."""
    expected = PurgedKFold(n_splits=5, purge_gap=gap, embargo_pct=0.0).split(np.zeros((500, 1)))
    actual = checks.buffered_two_sided_splits(500, 5, gap)
    for (train, test), (want_train, want_test) in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(train, want_train)
        np.testing.assert_array_equal(test, want_test)


@pytest.mark.parametrize("seed", range(5))
def test_own_date_recursion_recovers_noiseless_feedback_blips_exactly(seed: int) -> None:
    """With no state or outcome noise the blips are exactly (1, 1, 0.5)."""
    dgp = DynamicTreatmentDGP(
        n_periods=3,
        theta_t=np.array([1.0, 0.5, 0.25]),
        p=3,
        state_feedback=True,
        state_noise=0.0,
        outcome_noise=0.0,
        mode="series",
        series_length=600,
        random_state=seed,
    )
    data = dgp.generate()
    ends = np.arange(4, 600, 3)
    theta, _ = checks.own_date_recursion(data.Y, data.T, data.X, 3, ends, history=True)
    np.testing.assert_allclose(theta, [1.0, 1.0, 0.5], atol=1e-8)
    np.testing.assert_allclose(checks.series_total_blips(dgp), [1.0, 1.0, 0.5])


@pytest.mark.parametrize("history,first", [(False, 1), (True, 3)])
def test_own_date_recursion_refuses_dates_before_the_series(history: bool, first: int) -> None:
    """An early block end raises, e.g. end 1 with three lags would otherwise wrap around."""
    _, data = checks.feedback_series(100, True, 0)
    with pytest.raises(ValueError, match="block ends"):
        checks.own_date_recursion(
            data.Y, data.T, data.X, 3, np.arange(first, 100, 3), history=history
        )


@pytest.mark.parametrize("learner", ["ridge", "lasso"])
def test_cross_fit_replica_matches_the_package(learner: ModelType) -> None:
    """With forward folds the helper is ``TemporalPLRDML(n_lags=0)``."""
    Y, T, X = checks.review_data(4, 300, 0.8)
    fit = TemporalPLRDML(n_lags=0, model_y=learner, model_t=learner).fit(Y, T, X)
    replica = checks.cross_fit_plr(
        Y, T, X, checks.make_splits("forward", 300), get_nuisance_model(learner)
    )
    assert replica.theta == pytest.approx(fit.theta, rel=1e-12)
    assert checks.score_row({}, replica.theta, replica.scores)["se_nw"] == pytest.approx(
        fit.se, rel=1e-12
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: checks.cosine_long_run_variance(np.array([1.0, np.nan, 2.0])),
        lambda: checks.buffered_two_sided_splits(20, 2, 30),
        lambda: checks.make_splits("unknown", 100),
        lambda: checks.fixed_b_critical_value(1.5),
        lambda: checks.feedback_outcomes(*[np.ones((1, 3))] * 4, 2.0, 0.6),
        lambda: checks.run_cell("review_f3", (120,), 999, 1001),
        lambda: checks.write_draws([{"not_a_column": 1}], Path("never-written.gz")),
        lambda: checks.summarize(pd.DataFrame(columns=checks.COLUMNS)),
    ],
)
def test_invalid_input_raises(call: Callable[[], object]) -> None:
    """Bad input is refused, e.g. a NaN score."""
    with pytest.raises(ValueError):
        call()


def test_fixed_b_simulation_agrees_with_the_cubic() -> None:
    """At b = 0.15 both give a critical value near 2.41."""
    table = checks.fixed_b_table([checks.fixed_b_simulation(99, n=500, reps=20000)])
    row = next(row for row in table if row["b"] == 0.15)
    assert row["simulated"] == pytest.approx(row["recalled_polynomial"], abs=0.1)


@pytest.mark.parametrize("study", list(checks.STUDIES))
def test_two_seed_replay_matches_archived_draws(
    study: str, draws: pd.DataFrame, tmp_path: Path
) -> None:
    """The first two replications of each study reproduce, e.g. seeds 10000 and 10001."""
    spec = checks.STUDIES[study]
    stop = min(2, spec.replications)
    rows, _ = checks.run_cell(study, spec.cells[0], 0, stop)
    checks.write_draws(rows, tmp_path / "replay.gz")
    actual = pd.read_csv(tmp_path / "replay.gz", compression="gzip")
    first = actual.iloc[0]
    archived = draws[
        (draws.study == study)
        & (draws.design == first.design)
        & (draws.n == first.n)
        & (draws.replicate < stop)
        & draws.variant.isin(actual.variant)
    ]
    keys = ["variant", "component", "replicate"]
    pd.testing.assert_frame_equal(
        actual.sort_values(keys, ignore_index=True),
        archived.sort_values(keys, ignore_index=True),
        check_dtype=False,
        rtol=2e-5,
        atol=1e-9,
    )


def test_archived_summary_and_hashes_come_from_the_archived_draws(draws: pd.DataFrame) -> None:
    """``summary.csv`` is a function of ``draws.csv.gz``, and ``run.json`` hashes both."""
    record = json.loads((ARCHIVE / "run.json").read_text())
    for name, digest in record["outputs_sha256"].items():
        assert hashlib.sha256((ARCHIVE / name).read_bytes()).hexdigest() == digest
    saved = pd.read_csv(ARCHIVE / "summary.csv")
    pd.testing.assert_frame_equal(
        checks.summarize(draws), saved, check_dtype=False, rtol=2e-5, atol=1e-9
    )
    assert record["rows"] == len(draws)


def test_draw_files_are_byte_reproducible(tmp_path: Path) -> None:
    """Two writes are identical, with no timestamp in the gzip header."""
    rows, _ = checks.run_cell("review_f3", (120,), 0, 2)
    first, second = tmp_path / "a.gz", tmp_path / "b.gz"
    checks.write_draws(rows, first)
    checks.write_draws(rows, second)
    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first) as stream:
        stream.read()
        assert stream.mtime == 0
