"""Independent oracles for assignment clocks, repeated surveys and held-out randomized effects."""

import importlib
import os
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

controls = importlib.import_module(
    "docs.audits.2026-11-19_time_series_dml.external.forward.rolling_controls"
)
retail = importlib.import_module("forward.retail_assignments")
sierra = importlib.import_module("forward.sierra_indices")


@pytest.mark.tier1
def test_permutation_expectation_respects_hour_boundaries_and_sizes():
    # The same label on either side of the hour boundary is never an adjacent test pair.
    origin = pd.Timestamp("2019-11-25", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": [origin + pd.Timedelta(seconds=d) for d in (0, 1, 2, 3600, 3601)],
            "item": [1, 2, 2, 2, 2],
        }
    )
    result = retail.repetition(frame.sample(frac=1, random_state=1), "item")
    repeats = [sum(a == b for a, b in zip(p[:-1], p[1:])) for p in permutations([1, 2, 2])]
    assert result["pairs"] == 3 and result["repeats"] == 2
    assert result["permuted_expectation"] == pytest.approx((np.mean(repeats) + 1) / 3)
    with pytest.raises(ValueError, match="No adjacent"):
        retail.repetition(frame.iloc[[0, 3]], "item")


@pytest.mark.tier1
def test_freeze_nudge_forecast_before_scoring_and_weight_each_trial_once():
    arms = pd.DataFrame(
        {
            "year": [1, 1, 1, 2, 3],
            "trialnumber": [1, 1, 2, 3, 4],
            "treatmenteffect": [0, 4, 10, 8, 0],
        }
    )
    result = controls.predict_trials(arms)
    forecasts = [row for row in result["schedule"] if row["method"] == "historical"]
    assert forecasts[0]["forecast_pp"] == 6
    assert forecasts[0]["advantage_vs_zero_pp2"] == 60
    assert forecasts[1]["forecast_pp"] == pytest.approx(20 / 3)
    arms.loc[arms.year == 2, "treatmenteffect"] = -20
    later = controls.predict_trials(arms)
    assert later["schedule"][1]["forecast_pp"] == forecasts[0]["forecast_pp"]
    assert later["schedule"][1]["advantage_vs_zero_pp2"] != forecasts[0]["advantage_vs_zero_pp2"]
    arms.loc[arms.year == 3, "trialnumber"] = 1
    with pytest.raises(ValueError, match="crosses"):
        controls.predict_trials(arms)


@pytest.mark.tier1
def test_village_index_standardizes_on_controls_and_pairs_ids_after_shuffling():
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(
        {
            "t1": [0, 1] * 10,
            "tothhs": rng.normal(size=20),
            "road": rng.normal(size=20),
            "ward_1": [1] * 10 + [0] * 10,
            "ward_2": [0] * 10 + [1] * 10,
            "item": rng.normal(size=20),
        },
        index=np.arange(30, 50),
    )
    frame.loc[31, "item"] = np.nan
    later = frame.assign(item=frame.item + 2 * frame.t1).sample(frac=1, random_state=29)
    result = sierra.wave_comparison(frame, later, ["item", "item"], ["item"])
    control_sd = frame.loc[frame.t1 == 0, "item"].std(ddof=1)
    assert result["paired_change"]["effect"] == pytest.approx(2 / control_sd)
    assert result["paired_change"]["se"] < 1e-12
    assert result["components"] == [1, 1]
    with pytest.raises(ValueError, match="same villages"):
        sierra.wave_comparison(frame, later.iloc[:-1], ["item"], ["item"])


@pytest.mark.tier4
@pytest.mark.network
@pytest.mark.skipif(os.environ.get("DML_TS_EXTERNAL_REPLAY") != "1", reason="opt-in source replay")
def test_external_control_references(tmp_path):
    out = controls.run(Path(os.environ.get("DML_TS_BENCHMARK_CACHE", tmp_path)))
    assert out["effort"]["rank_correlation"] == pytest.approx(0.9071428571)
    assert out["sierra_leone"]["institutions_common_items"]["paired_change"][
        "effect"
    ] == pytest.approx(-0.0127158, abs=1e-6)
    assert out["nudge_units"]["pooled_historical_advantage_pp2"] == pytest.approx(-0.20732743)
    market = out["retail"]
    assert market["repetition"]["2"]["uniform_item"]["permuted_expectation"] == pytest.approx(
        0.03620, abs=0.00002
    )
    assert market["repetition"]["2"]["uniform_item"]["repeats"] == 58645
    assert market["repetition"]["2"]["policy_arm"]["observed"] == pytest.approx(0.8200, abs=0.00005)
    assert [(r["rows"], r["position"]) for r in market["late_weight_over_1000_cells"]] == [
        (15, 2),
        (19, 2),
        (7, 1),
    ]
