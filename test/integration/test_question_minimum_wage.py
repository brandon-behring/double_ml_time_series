"""Replication and falsification checks for the restaurant minimum-wage case."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docs/audits/2026-11-19_time_series_dml/external/questions/minimum_wage.py"
spec = importlib.util.spec_from_file_location("wage_case", SCRIPT)
case = importlib.util.module_from_spec(spec)
spec.loader.exec_module(case)


@pytest.mark.tier1
def test_geographic_clusters_ignore_cz_identifier_width():
    pairs = pd.Series(["620001-620047", "10001-10047", "620005-620048"])
    first, second = np.array([1, 1, 5]), np.array([47, 47, 48])
    geo = case.crosswalk_segment(first, second)
    legacy = case.legacy_segment(pairs)
    assert geo[0] == geo[1] and geo[0] != geo[2]
    assert legacy[0] != legacy[1] and legacy[0] == legacy[2]
    np.testing.assert_array_equal(geo, case.crosswalk_segment(second, first))


@pytest.mark.tier1
def test_pair_contrast_lags_and_dates_are_stable_across_duplicates():
    rows = []
    for pair, neighbors in [("ab", [(1, 11), (2, 22)]), ("bc", [(2, 22), (3, 33)])]:
        for state, county in neighbors:
            for t in range(1, 28):
                # Arbitrarily large changes shared by a pair cancel in the contrast.
                shock = 100 * t**2
                rows.append(
                    {
                        "pair_id": pair,
                        "state_fips": state,
                        "county": county,
                        "period": t,
                        "nonmissing_rest_both": 66,
                        "lnMW": state * t**2 + shock,
                        "lnemp_rest_both": 2 * state * t**2 + shock,
                        "lnAWW_rest_both": state * t + shock,
                        "lnpop": county + shock,
                    }
                )
    raw = pd.DataFrame(rows)
    changes = case.pair_changes(raw)
    pd.testing.assert_frame_equal(changes, case.pair_changes(raw.sample(frac=1, random_state=3)))
    np.testing.assert_allclose(changes.wage, 2 * changes.period - 1)
    np.testing.assert_allclose(changes.pre_employment, 2 * (2 * (changes.period - 1) - 1))
    np.testing.assert_allclose(changes.lag2_lnMW, 2 * (changes.period - 2) - 1)
    for train, test in case.forward_folds(changes.period.to_numpy()):
        train_dates = changes.loc[train, "period"]
        test_dates = changes.loc[test, "period"]
        assert train_dates.max() + 2 < test_dates.min()
        assert not set(train_dates) & set(test_dates)
        assert changes.loc[test].groupby("period").pair_id.nunique().eq(2).all()
    with pytest.raises(ValueError, match="time gap"):
        case.pair_changes(raw.loc[raw.period != 8])


@pytest.mark.tier4
@pytest.mark.network
@pytest.mark.skipif(
    os.getenv("DML_TS_EXTERNAL_REPLAY") != "1", reason="full study downloads are opt-in"
)
def test_wage_public_studies_and_falsification(tmp_path):
    directory = os.environ.get("DML_TS_BENCHMARK_CACHE", str(tmp_path))
    data = json.loads(
        subprocess.check_output([sys.executable, str(SCRIPT), "--cache-dir", directory], text=True)
    )
    card = data["card_krueger"]
    assert (card["unique_sheet_numbers"], card["archive_stores"]) == (409, 410)
    assert (card["matched"]["pa"], card["matched"]["nj"]) == (75, 309)
    assert card["matched_store_att"]["constant_nuisances"] == pytest.approx(2.75)
    assert 0 < card["matched_store_att"]["effective_control_stores"] <= 75
    models = data["dlr_jnr"]
    published, geographic = models["cz_private"], models["cz_geographic_clusters"]
    assert published["estimate"] == pytest.approx(geographic["estimate"], abs=1e-12)
    assert (published["segments"], geographic["segments"]) == (33, 48)
    assert (
        published["n"],
        models["border_private"]["n"],
        models["border_complete_private"]["n"],
    ) == (9634, 70582, 41676)
    assert geographic["se"] == pytest.approx(0.139130, abs=2e-6)
    trends = data["dlr_table3_pretrends"]
    assert (trends["national"]["n"], trends["border"]["n"]) == (82800, 64200)
    for place, pairs in [("border", 316), ("commuting_zone", 73)]:
        comparison = data["short_run"][place]
        assert (comparison["pairs"], comparison["quarters"]) == (pairs, 50)
        assert 0 < comparison["observed_wage_changes"] < comparison["n"]
        for model in ["linear", "forest"]:
            assert np.isfinite(comparison[model]["pre_employment"]["estimate"])
            assert comparison[model]["earnings"]["date_hac_se"] > 0
