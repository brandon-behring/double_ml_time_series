"""Check timing, policy estimands, and chain covariance for the Ban the Box case."""

import importlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

case = importlib.import_module("docs.audits.2026-11-19_time_series_dml.external.forward.ban_box")


@pytest.mark.tier1
def test_only_closed_followup_and_consistent_dates_enter_a_prediction():
    dates = ("2015-01-01", "2015-03-01", "2015-03-02", "2015-03-15", "2015-04-20")
    days = np.array(
        [
            case.stata_day(s)
            for s in ["2015-01-01", "2015-02-26", "2015-01-05", "2015-03-16", "2025-02-01"]
        ]
    )
    frame = pd.DataFrame(
        {
            "app_date_d": days,
            "response_date_d": days + [10, 0, -1, 5, 2],
            "response": [1, 0, 1, 1, 1],
            "post": [0, 0, 0, 1, 0],
        }
    )
    kept, audit = case.eligible_waves(frame, dates)
    assert kept.index.tolist() == [0, 3]
    assert audit == {
        "outside_paper_dates": 1,
        "positive_with_unusable_date": 1,
        "early_not_mature_at_post": 1,
        "early_mature_by_law": 1,
    }


@pytest.mark.tier1
def test_chain_difference_cancels_shared_arm_shock_and_keeps_early_forecast_frozen():
    data = pd.DataFrame(
        [
            (c, p, w, 0.2 * c + w * (0.1 * c + 0.1 * p))
            for c in range(1, 5)
            for p in (0, 1)
            for w in (0, 1)
        ],
        columns=["chain_id", "post", "white", "response"],
    )
    result = case.comparison(data)
    assert result["early_minus_later"]["estimate"] == pytest.approx(-10.0)
    assert result["early_minus_later"]["ci95"] == pytest.approx([-10, -10], abs=2e-6)
    data.loc[data.post.eq(1) & data.white.eq(1), "response"] += 0.2
    assert case.comparison(data)["early"]["estimate"] == pytest.approx(result["early"]["estimate"])


@pytest.mark.network
@pytest.mark.tier4
@pytest.mark.skipif(os.getenv("DML_TS_EXTERNAL_REPLAY") != "1", reason="opt-in archive replay")
def test_ban_the_box_source_replay(tmp_path):
    result = case.run(Path(os.getenv("DML_TS_BENCHMARK_CACHE", tmp_path)))
    assert result["table_v"]["effect_pp"] == pytest.approx(3.89819, abs=0.00001)
    assert result["waves"]["New Jersey"]["chronology"]["early_mature_by_law"] == 0
    ny = result["waves"]["New York City"]["documented_dates"]
    assert ny["early_minus_later"]["estimate"] == pytest.approx(-0.7024112, abs=1e-6)
    assert ny["early_minus_later"]["ci95"] == pytest.approx([-2.5373881, 1.1325656], abs=1e-6)
