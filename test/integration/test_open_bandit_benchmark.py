"""A randomized future benchmark and tests of leakage and item propensities."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/audits/2026-11-19_time_series_dml/external/forward/open_bandit.py"
spec = importlib.util.spec_from_file_location("open_bandit", SOURCE)
case = importlib.util.module_from_spec(spec)
spec.loader.exec_module(case)


def example(arm):
    # Each hour supplies both policies at all positions.
    n = len(case.HOURS) * 3
    frame = pd.DataFrame(
        {
            "timestamp": case.HOURS.repeat(3).astype(str),
            "position": np.tile([1, 2, 3], n // 3),
            "item_id": np.arange(n) % 34,
            "click": (np.arange(n) % (4 if arm == "bts" else 7) == 0).astype(int),
            "propensity_score": (0.2 if arm == "bts" else 1 / 34),
        }
    )
    return frame


@pytest.mark.tier1
def test_frozen_estimate_and_off_policy_model_do_not_train_on_later_clicks():
    frames = {arm: example(arm) for arm in ["random", "bts"]}
    cells = {arm: case.summarize_rows(rows, arm).reset_index() for arm, rows in frames.items()}
    before = case.compare(cells)
    early, late = case.periods(24)
    assert early[-1] == case.CUTOFF - pd.Timedelta(hours=25) and late[0] == case.CUTOFF

    # Replace all uniform clicks from the intervening (embargo) day onwards.
    random = frames["random"].copy()
    random.loc[pd.to_datetime(random.timestamp) > early[-1], "click"] = 1
    cells["random"] = case.summarize_rows(random, "random").reset_index()
    after = case.compare(cells)
    a, b = (x["reporting_lags_hours"]["24"] for x in (before, after))
    assert a["early_pp"] == b["early_pp"] and a["later_pp"] != b["later_pp"]
    assert (
        before["reporting_lags_hours"]["0"]["early_pp"]
        != after["reporting_lags_hours"]["0"]["early_pp"]
    )
    for method in ["ipw", "doubly_robust"]:
        assert (
            before["later_off_policy"][method]["ctr_pct"]
            == after["later_off_policy"][method]["ctr_pct"]
        )

    thompson = frames["bts"].copy()
    thompson.loc[pd.to_datetime(thompson.timestamp) >= case.CUTOFF, "click"] = 0
    cells["bts"] = case.summarize_rows(thompson, "bts").reset_index()
    perturbed = case.compare(cells)
    assert perturbed["reporting_lags_hours"]["24"]["early_pp"] == a["early_pp"]
    assert perturbed["later_off_policy"]["ipw"]["ctr_pct"] == 0
    assert (
        perturbed["later_off_policy"]["augmentation_terms_pct"]
        == before["later_off_policy"]["augmentation_terms_pct"]
    )


@pytest.mark.tier1
def test_item_weights_use_the_published_per_slot_design():
    frame = example("bts").iloc[:2].copy()
    frame["propensity_score"] = [0.5, 0.01]
    frame["click"] = [1, 0]
    cells = case.summarize_rows(frame, "bts")
    assert cells.weight.sum() == pytest.approx((2 + 100) / 34)
    assert cells.weight_click.sum() == pytest.approx(2 / 34)
    with pytest.raises(ValueError, match="uniformly"):
        case.summarize_rows(frame, "random")
    frame.iloc[0, frame.columns.get_loc("propensity_score")] = 0
    with pytest.raises(ValueError, match="positive logged item"):
        case.summarize_rows(frame, "bts")


@pytest.mark.tier1
def test_off_policy_score_integrates_all_possible_items():
    early, late = case.periods(24)
    # The two early items fit to 0.75 and 0.25, while unseen items pool to 0.50.
    # Only item 0 is observed in the evaluation log.
    train = pd.DataFrame(
        [
            (early[0], pos, item, 100, clicks)
            for pos in (1, 2, 3)
            for item, clicks in [(0, 100), (1, 0)]
        ],
        columns=["hour", "position", "item_id", "n", "clicks"],
    )
    all_cells = case.summarize_rows(example("bts"), "bts").reset_index()
    test = all_cells.loc[all_cells.hour.isin(late)].assign(item_id=0)
    hourly = test.groupby("hour")[["n", "clicks"]].sum()
    out = case.off_policy(pd.concat([train, test]), {"random": hourly}, early, late)
    weights = (1 / 34) / 0.2
    expected = out["ipw"]["ctr_pct"] + 50 - 75 * weights
    assert out["doubly_robust"]["ctr_pct"] == pytest.approx(expected)


@pytest.mark.tier4
@pytest.mark.network
@pytest.mark.skipif(os.environ.get("DML_TS_EXTERNAL_REPLAY") != "1", reason="opt-in source replay")
def test_full_retail_policy_replay(tmp_path):
    result = case.run(Path(os.environ.get("DML_TS_BENCHMARK_CACHE", tmp_path)))
    assert result["full_sample"]["random"]["clicks"] == 2321
    assert result["full_sample"]["bts"]["clicks"] == 27495
    score = result["reporting_lags_hours"]["24"]["signed_error"]
    assert score["pp"] == pytest.approx(0.0237753719)
    assert score["ci95_pp_by_span_hours"]["6"] == pytest.approx([-0.05271915, 0.10026989])
    test = result["later_off_policy"]
    assert test["uniform_arm_ctr_pct"] == pytest.approx(0.54173542)
    assert test["ipw"]["ctr_pct"] == pytest.approx(0.52837758)
    assert test["doubly_robust"]["ctr_pct"] == pytest.approx(0.36085315)
    assert test["weights"]["effective_rows"] == pytest.approx(362.374883)
    assert test["weights"]["rows_above_1000"] == 41
    assert test["largest_weight_block"]["weight_mass_pct"] == pytest.approx(20.8786231)
    assert test["largest_weight_block"]["dr_excluding_hour_pct"] == pytest.approx(0.52282479)
