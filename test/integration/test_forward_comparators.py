"""Timing, recoverability, and held-out-location contracts for forward benchmarks."""

import importlib
import os
from pathlib import Path

import numpy as np
import pytest

PREFIX = "docs.audits.2026-11-19_time_series_dml.external.forward"
fertilizer = importlib.import_module(f"{PREFIX}.fertilizer")
microcredit = importlib.import_module(f"{PREFIX}.microcredit")
progresa = importlib.import_module(f"{PREFIX}.progresa")


@pytest.mark.tier1
def test_missing_or_late_fertilizer_measurement():
    season = "In season 2 (the 2004 short rains)"
    with pytest.raises(ValueError, match="survey year"):
        fertilizer.assess(season)
    phrase = "post-treatment adoption data in {} (two seasons after"
    assert fertilizer.assess(season + " " + phrase.format(2005))[
        "adoption_definitely_unavailable_at_offer"
    ]
    assert not fertilizer.assess(season + " " + phrase.format(2004))[
        "adoption_definitely_unavailable_at_offer"
    ]


@pytest.mark.tier1
def test_holdout_is_unchanged_by_held_out_study_and_propagates_both_standard_errors():
    effects = [1, -1, 2]
    row = microcredit.holdout(effects, [2, 2, 1], 2)
    assert row["forecast"] == 0
    assert row["forecast_minus_local"] == -2
    np.testing.assert_allclose(
        row["fixed_sites_error_ci95"], -2 + np.array([-1, 1]) * 1.96 * np.sqrt(3)
    )
    assert microcredit.holdout([1, -1, 999], [2, 2, 400], 2)["forecast"] == row["forecast"]


@pytest.mark.tier1
def test_rate_rounding_rejects_nonunique_and_impossible_counts():
    assert progresa.count_from_rate(74.4, 223) == 166
    with pytest.raises(ValueError, match="unique"):
        progresa.count_from_rate(50, 10000)
    with pytest.raises(ValueError, match="unique"):
        progresa.count_from_rate(50, 3)


@pytest.mark.network
@pytest.mark.tier4
@pytest.mark.skipif(os.environ.get("DML_TS_EXTERNAL_REPLAY") != "1", reason="opt-in source replay")
@pytest.mark.parametrize("name", ["microcredit", "progresa", "fertilizer"])
def test_archived_comparator_results(name, tmp_path):
    cache = Path(os.environ.get("DML_TS_BENCHMARK_CACHE", tmp_path))
    if name == "microcredit":
        result = microcredit.run(cache)
        assert result["mean_absolute_error"] == pytest.approx(18.70559, abs=1e-5)
        assert result["sites"][0]["assigned_profit_rows"] == 16560
    elif name == "progresa":
        girls = progresa.run(cache)["girls"]
        assert girls["attendees_control"] == 117
        assert girls["experimental_effect_pp"] == pytest.approx(7.96219, abs=1e-5)
    else:
        assert fertilizer.run(cache)["adoption_definitely_unavailable_at_offer"]
