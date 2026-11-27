"""Timing, physical units and full-data regression checks for the two demand studies."""

import json
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

STUDIES = "docs.audits.2026-11-19_time_series_dml.external.questions"
retail = import_module(f"{STUDIES}.retail")
energy = import_module(f"{STUDIES}.energy")
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.tier2
def test_retail_cut_uses_current_and_prior_prices_and_respects_store_calendar():
    raw = pd.DataFrame(
        [
            (s, b, w, 10.0, np.log(float(10 * w + b)), 0)
            for s in [1, 2]
            for b in [1, 2, 3]
            for w in range(1, 8)
            if (s, w) != (2, 4)
        ],
        columns=["store", "brand", "week", "price", "logmove", "feat"],
    )
    raw.loc[(raw.store == 1) & (raw.week == 5) & (raw.brand < 3), "price"] = 9.0
    base = retail.prepare(raw)
    future = raw.copy()
    future.loc[future.week > 5, "price"] = 4.0
    future.loc[future.week > 5, "logmove"] = 0.0
    changed = retail.prepare(future)
    anchor = (base.store == 1) & (base.brand == 1) & (base.week == 5)
    row = base.loc[anchor].iloc[0]
    other = changed.loc[anchor].iloc[0]
    assert row.sale == other.sale == 1 and row.other_cuts == other.other_cuts == 1
    assert other.own_next != pytest.approx(row.own_next)
    assert row.own_current == pytest.approx(10)
    assert row.own_next == pytest.approx(20)
    assert row.category_three_week_total == pytest.approx(
        row.own_three_week_total + row.rivals_three_week_total
    )
    assert base.loc[(base.store == 2) & (base.week == 5), "sale"].isna().all()
    assert base.loc[(base.store == 2) & (base.week == 3), "own_next"].isna().all()


@pytest.mark.tier2
def test_retail_origin_folds_hold_out_whole_weeks_and_all_future_labels():
    weeks = np.repeat(np.arange(1, 40), 6)
    scored = []
    for train, test in retail.forward_splits(weeks):
        assert weeks[train].max() + 2 < weeks[test].min()
        np.testing.assert_array_equal(test, np.flatnonzero(np.isin(weeks, np.unique(weeks[test]))))
        scored.extend(test)
    assert len(scored) == len(set(scored))


@pytest.mark.tier2
def test_followup_excludes_adjacent_events_and_new_announcements():
    days = pd.date_range("2020-01-01", periods=20)
    events = days[[2, 3, 6, 9, 11, 16, 19]]
    assert energy.isolated_starts(events, days) == list(days[[6, 11, 16]])


@pytest.mark.tier2
def test_energy_events_are_aggregated_within_each_household_by_calendar_date():
    days = pd.date_range("2020-01-01", periods=5)
    raw = pd.DataFrame(
        [(h, d, float(10 * h + i), float(h)) for h in [1, 2] for i, d in enumerate(days)],
        columns=["household", "day", "peak", "offpeak"],
    )
    dates = days[[0, 3]]
    windows = energy.household_windows(raw.sample(frac=1, random_state=9), dates, 48)
    np.testing.assert_allclose(windows.loc[1, ["peak", "offpeak", "total"]], [24, 2, 26])
    np.testing.assert_allclose(windows.loc[2, ["peak", "offpeak", "total"]], [44, 4, 48])
    missing = raw.loc[~((raw.household == 2) & (raw.day == days[1]))]
    with pytest.raises(ValueError, match="Missing"):
        energy.household_windows(missing, dates, 48)


@pytest.mark.tier4
@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("DML_TS_EXTERNAL_REPLAY") != "1",
    reason="Set DML_TS_EXTERNAL_REPLAY=1 to replay original sources",
)
@pytest.mark.parametrize("name", ["retail", "energy"])
def test_source_reanalysis(name, tmp_path):
    module = retail if name == "retail" else energy
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT), env.get("PYTHONPATH"))))
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            module.__file__,
            "--cache-dir",
            os.environ.get("DML_TS_BENCHMARK_CACHE", str(tmp_path)),
        ],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )
    result = json.loads(completed.stdout)
    if name == "retail":
        primary = result["all"]
        assert (primary["rows"], primary["cuts"], primary["simultaneous"]) == (20436, 4424, 1733)
        methods = primary["methods"]
        assert methods["difference"]["outcomes"]["own_next"]["slope"] == pytest.approx(
            9412.6, abs=0.1
        )
        assert methods["forward_ridge_PLR"]["outcomes"]["own_next"]["slope"] == pytest.approx(
            1348.2, abs=0.1
        )
        assert result["no_simultaneous_rival_cut"]["simultaneous"] == 0
        assert result["no_simultaneous_rival_cut"]["cuts"] == 2691
        spells = result["retrospective_one_week_spells"]
        assert (spells["completed_spells"], spells["complete_five_week_spells"]) == (2441, 2185)
        for series, expected in [("own", -0.3187348), ("category", -0.0210182)]:
            assert spells["mean_log_change_from_prior_week"][series]["1"] == pytest.approx(
                expected, abs=1e-7
            )
    else:
        assert result["households"] == {"assigned": 201, "control": 204} and result["events"] == 21
        full = result["all_24h"]["contrasts"]
        assert full["peak"]["change_from_baseline"]["effect"] == pytest.approx(-1.077, abs=0.001)
        assert full["total"]["change_from_baseline"]["effect"] == pytest.approx(-1.997, abs=0.001)
        assert result["isolated_48h"]["event_count"] == 7
        next_day = result["isolated_next_24h"]["contrasts"]["total"]["change_from_baseline"]
        assert next_day["effect"] == pytest.approx(-2.0838, abs=0.0001)
        np.testing.assert_allclose(next_day["ci95"], [-3.652, -0.515], atol=0.001, rtol=0)
        event_day = result["isolated_24h"]["contrasts"]["total"]["change_from_baseline"]["effect"]
        both_days = result["isolated_48h"]["contrasts"]["total"]["change_from_baseline"]["effect"]
        assert both_days == pytest.approx(event_day + next_day["effect"])
