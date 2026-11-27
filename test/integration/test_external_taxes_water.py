"""Offline published water-trial oracles and opt-in full raw-data replays."""

import json
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

replay = import_module("docs.audits.2026-11-19_time_series_dml.external.taxes_water.replay")


@pytest.mark.tier2
def test_water_trial_matches_published_treatment_effects():
    moments = json.loads(replay.FIXTURE.read_text())
    fitted = replay.fit_water_moments(moments)
    # Ferraro & Price (2013) Table 2, model A. Values are printed to two decimals.
    assert moments["n"] == 106669
    assert moments["arm_counts"] == {"control": 71643, "tips": 11675, "weak": 11675, "peer": 11676}
    for arm, published in [
        ("tips", [-0.24, 0.19]),
        ("weak", [-0.99, 0.17]),
        ("peer", [-1.74, 0.17]),
    ]:
        np.testing.assert_allclose(
            [fitted[arm]["slope"], fitted[arm]["HC1_SE"]], published, atol=0.005, rtol=0
        )
    np.testing.assert_allclose(fitted["peer"]["homoskedastic_SE"], 0.17854395841730455)


@pytest.mark.tier2
def test_changing_peer_treatment_units_rescales_slope_and_uncertainty():
    moments = json.loads(replay.FIXTURE.read_text())
    original = replay.fit_water_moments(moments)
    rescaled = replay.fit_water_moments(moments, peer_scale=10.0)
    for arm in ["tips", "weak", "peer"]:
        for field in ["slope", "HC1_SE", "homoskedastic_SE"]:
            np.testing.assert_allclose(
                rescaled[arm][field], original[arm][field] / (10 if arm == "peer" else 1)
            )


@pytest.mark.tier2
def test_gallons_rescale_water_slopes_and_both_standard_errors():
    moments = json.loads(replay.FIXTURE.read_text())
    in_thousands = replay.fit_water_moments(moments)
    units = np.ones(len(moments["columns"]))
    units[moments["columns"].index("volume")] = 1000
    # Each occurrence of the outcome in a cross-product changes its units.
    # The OLS score contains one residual, so its cross-product gains 1000².
    gallons = {
        **moments,
        "crossproducts": np.asarray(moments["crossproducts"]) * np.outer(units, units),
        "score_crossproducts": np.asarray(moments["score_crossproducts"]) * 1000**2,
    }
    in_gallons = replay.fit_water_moments(gallons)
    for arm in ["tips", "weak", "peer"]:
        for field in ["slope", "HC1_SE", "homoskedastic_SE"]:
            np.testing.assert_allclose(in_gallons[arm][field], 1000 * in_thousands[arm][field])


@pytest.mark.tier4
@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("DML_TS_EXTERNAL_REPLAY") != "1",
    reason="Set DML_TS_EXTERNAL_REPLAY=1 to fetch the primary archives",
)
@pytest.mark.parametrize("study", ["beer", "water"])
def test_full_replay(study, tmp_path):
    cache = os.environ.get("DML_TS_BENCHMARK_CACHE", str(tmp_path))
    result = subprocess.run(
        [sys.executable, str(Path(replay.__file__)), study, "--cache-dir", cache],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    report = json.loads(result.stdout)
    if study == "beer":
        assert report["Table 6 column 1"]["state_years"] == 1607
    else:
        extension = report["two_arm_DML_extension"]
        assert extension["n"] == 83319
        assert np.isfinite([extension["slope"], extension["IF_SE"]]).all()
        assert extension["IF_SE"] > 0
