"""Check the exact assignment reference and numeric Table 1 reanalysis."""

import importlib
import itertools
import os
from pathlib import Path

import numpy as np
import pytest

MODULE = "docs.audits.2026-11-19_time_series_dml.external"
daycare = importlib.import_module(f"{MODULE}.questions.daycare")


@pytest.mark.tier1
def test_exact_interval_against_direct_null_randomization():
    # Use a sufficiently large assignment set for a finite 95% interval.
    changes = np.array([-0.4, 1.0, 1.2, 1.9, 0.7, 0.4, -0.1, 0.2, 0.19, 0.3])
    assigned = np.arange(len(changes)) < 6
    result = daycare.randomization(changes, assigned)
    assignments = list(itertools.combinations(range(10), 6))

    def direct_p(tau):
        adjusted = changes - tau * assigned
        threshold = abs(adjusted[assigned].mean() - adjusted[~assigned].mean())
        counts = 0
        for centers in assignments:
            z = np.isin(np.arange(10), centers)
            counts += abs(adjusted[z].mean() - adjusted[~z].mean()) >= threshold - 1e-11
        return counts / len(assignments)

    assert result["two_sided_p"] == direct_p(0)
    lower, upper = result["constant_effect_interval"]
    for endpoint, direction in [(lower, -1), (upper, 1)]:
        assert direct_p(endpoint) >= 0.05
        assert direct_p(endpoint + direction * 1e-5) < 0.05
    shifted = daycare.randomization(changes[::-1] + 3 * assigned[::-1], assigned[::-1])
    np.testing.assert_allclose(shifted["constant_effect_interval"], np.array([lower, upper]) + 3)


@pytest.mark.tier1
def test_count_table_requires_all_twenty_reported_weeks():
    with pytest.raises(ValueError, match="ten centers"):
        daycare.published_counts("Test group:\n1 37 8 8 7 6\nThe columns under")


@pytest.mark.network
@pytest.mark.tier4
@pytest.mark.skipif(os.environ.get("DML_TS_EXTERNAL_REPLAY") != "1", reason="opt-in archive replay")
def test_published_fine_experiment(tmp_path):
    results = daycare.study(Path(os.environ.get("DML_TS_BENCHMARK_CACHE", tmp_path)))
    assert results["center_1_checks"] == {"pre": 7.25, "first_four_fine_weeks": 9.5, "after": 15.25}
    assert results["during"]["assignments"] == 210
    assert results["after_repeal"]["as_extreme"] == 1
    assert results["exclude_short_week_11"]["did"] == pytest.approx(8.98485, abs=0.00001)
