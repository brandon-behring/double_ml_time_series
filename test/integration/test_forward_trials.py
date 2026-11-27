"""Scientific and calendar checks for transporting two randomized effects to later people."""

import importlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = "docs.audits.2026-11-19_time_series_dml.external.forward"
reminders = importlib.import_module(f"{ROOT}.reminders")
canvassing = importlib.import_module(f"{ROOT}.canvassing")


@pytest.mark.tier1
def test_earlier_cohort_cannot_depend_on_later_outcomes_or_incomplete_followup():
    data = pd.DataFrame(
        {
            reminders.ID: ["early", "too_late", "new", "new"],
            reminders.DAY: [14, 14, 14, 97],
            reminders.ENROLL: pd.to_datetime(
                ["2022-06-01", "2022-12-01", "2023-02-01", "2023-02-01"]
            ),
            "log_steps": [1.0, 1.0, 1.0, 1.0],
        }
    )
    early, late = reminders.cohorts(data, 97)
    assert early[reminders.ID].tolist() == ["early"]
    assert late[reminders.ID].tolist() == ["new", "new"]
    data.loc[data[reminders.ID] == "new", "log_steps"] = 999
    pd.testing.assert_frame_equal(early, reminders.cohorts(data, 97)[0])


@pytest.mark.tier1
def test_known_probability_scores_recover_randomized_effect_and_age_mix():
    # Every age stratum has exactly 1 activity, 6 no prompts and 1 diet assignment.
    ages = np.array([False, True])
    early = np.array([[1, 6, 12, 60, 8], [1, 6, 23, 120, 8]], dtype=float)
    late = np.array([[1, 6, 13, 60, 8], [1, 6, 24, 120, 8]], dtype=float)
    eweights = np.array([[1, 1]], dtype=float)
    lweights = np.array([[1, 3]], dtype=float)
    primary, age_informed = reminders.forecast(early, ages, eweights, ages, lweights)
    np.testing.assert_allclose(primary, 2.5)
    np.testing.assert_allclose(age_informed, 2.75)
    one_step = reminders.later_augmented(early, late, ages, ages, eweights, lweights)
    np.testing.assert_allclose(one_step, 3.75)
    np.testing.assert_allclose(one_step, reminders.effect(lweights @ late))
    # Alter only later outcomes. Both earlier predictions are functions of early outcomes.
    late[:, 2:4] += 250
    np.testing.assert_allclose(
        reminders.forecast(early, ages, eweights, ages, lweights), [[2.5], [2.75]]
    )


@pytest.mark.tier1
def test_city_estimator_respects_walk_lists_and_negative_control():
    data = pd.DataFrame(
        {
            "treatmen": [0, 1, 0, 1, 1, 1],
            "turf": [1, 1, 2, 2, 3, 3],
            "city": "A",
            "turnout": [0.1, 0.3, 0.6, 0.8, 0.25, 0.25],
            "placebo": [0.1, 0.1, 0.6, 0.6, 0.25, 0.25],
        }
    )
    assert canvassing.site_contrast(data, "turnout")["estimate"] == pytest.approx(20.0)
    assert canvassing.site_contrast(data, "turnout")["supported_lists"] == 2
    assert canvassing.site_contrast(data, "placebo")["estimate"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="No within-list"):
        canvassing.site_contrast(data[data.turf.eq(3)], "turnout")


@pytest.mark.network
@pytest.mark.tier4
@pytest.mark.skipif(os.getenv("DML_TS_EXTERNAL_REPLAY") != "1", reason="opt-in archive replay")
@pytest.mark.parametrize("name", ["reminders", "canvassing"])
def test_archived_forward_experiments(name, tmp_path):
    cache = Path(os.getenv("DML_TS_BENCHMARK_CACHE", tmp_path))
    if name == "reminders":
        result = reminders.run(cache)["days14_97"]
        assert result["frozen"]["estimate"] == pytest.approx(0.04241620, abs=1e-8)
        assert result["frozen_minus_later"]["ci95"] == pytest.approx(
            [-0.02791750, 0.14688615], abs=1e-8
        )
    else:
        result = canvassing.run(cache)
        assert result["cities_pp"]["All sites"]["later"]["rows"] == 18933
        assert result["cities_pp"]["St Paul"]["later"]["estimate"] == pytest.approx(4.64, abs=0.005)
        assert result["corrected_forecast_pp"]["estimate"] == pytest.approx(2.8275848, abs=1e-6)
