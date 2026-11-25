"""Check which experimental data and uncertainty the legacy C2/C3 comparisons use."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dml_ts.validation import DGPGenerator
from scripts import test_issue_c2_binary_treatment as binary
from scripts import test_issue_c3_monte_carlo_comparability as ranking

pytestmark = pytest.mark.tier1


def test_c2_compares_same_samples_and_folds_and_asks_for_marginal_intervals(monkeypatch):
    calls = []

    class FakeDML:
        def __init__(self, discrete_treatment, cv, **kwargs):
            self.discrete = discrete_treatment
            self.folds = cv

        def fit(self, Y, T, X):
            calls.append((Y, T, X, self.folds))

        def ate(self, X):
            return 2 + self.discrete

        def ate_interval(self, X, alpha):
            assert alpha == 0.05
            return (1.5, 2.5) if self.discrete else (1, 3)

    monkeypatch.setitem(sys.modules, "econml.dml", SimpleNamespace(LinearDML=FakeDML))
    sample = DGPGenerator(50, 2, 2.0, random_state=5).generate()
    assert binary.paired_draw(sample, 42) == [(2, 1, 3), (3, 1.5, 2.5)]
    assert all(a is b for a, b in zip(calls[0], calls[1], strict=True))
    for train, test in calls[0][3]:
        assert not np.intersect1d(train, test).size


def test_c2_counts_paired_hits_and_reports_monte_carlo_precision():
    fits = np.asarray([[[3.0, 1.0, 4.0], [4.0, 3.0, 5.0]], [[1.0, 0.0, 1.5], [1.0, 0.0, 1.5]]])
    delta = binary.paired_differences(fits, 2.0)
    assert delta["signed_error_difference"] == pytest.approx(0.5)
    assert delta["signed_error_difference_mcse"] == pytest.approx(0.5)
    assert delta["coverage_difference"] == pytest.approx(-0.5)
    assert binary.summarize(*fits[:, 0, :].T, truth=2.0)["covered"] == 1


def test_c3_matched_helper_replays_public_validators_including_their_intervals():
    data_seen = []

    class Validator:
        n_simulations = 2

        def validate(self, dgp):
            data_seen.append(np.concatenate([dgp.generate().Y for _ in range(self.n_simulations)]))
            return SimpleNamespace(bias=-0.1, mse=0.9, coverage=0.5)

    results = ranking.run_comparison_corrected(
        {"n": 50, "p": 2, "true_effect": 2.0, "random_state": 23},
        {"first": Validator(), "second": Validator()},
        2,
    )
    np.testing.assert_array_equal(*data_seen)
    assert (
        results["first"]
        == results["second"]
        == {"bias": -0.1, "abs_bias": 0.1, "mse": 0.9, "coverage": 0.5}
    )
