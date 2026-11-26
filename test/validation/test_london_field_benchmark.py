"""Regression checks on event timing, allocation support and the level of randomization."""

import importlib

import numpy as np
import pandas as pd
import pytest

london = importlib.import_module("docs.audits.2026-11-19_time_series_dml.field.london.replay")
pytestmark = pytest.mark.tier2


def test_summer_overnight_window_uses_gmt_end_of_halfhour_and_matched_dates():
    events = pd.DataFrame(
        {
            "local_start": pd.to_datetime(["2013-02-01 23:00", "2013-06-01 23:00"]),
        }
    )
    winter, summer = london.meter_times(events, "gmt_end")
    assert (winter[0], winter[-1]) == ("2013-02-01 23:30:00.0000000", "2013-02-02 02:00:00.0000000")
    assert (summer[0], summer[-1]) == ("2013-06-01 22:30:00.0000000", "2013-06-02 01:00:00.0000000")

    observed = pd.read_csv(london.HERE / "events.csv")
    full, *quarters = london.support(observed.sample(frac=1, random_state=42))
    assert full["high"] == full["low"] == 24
    assert [len(q["matched_hours"]) for q in quarters] == [3, 4, 4, 0]
    assert quarters[-1]["high"] == 2
    assert quarters[-1]["low"] == 7


def test_date_variance_uses_arm_sizes_and_refuses_unsupported_comparison():
    dates = np.array([1.0, 3.0, 4.0, 8.0, 5.0, 7.0, 8.0, 12.0])
    arms = np.tile([False, False, True, True], 2)
    slots = np.repeat([2, 20], 4)
    effect, se, df = london.stratified_contrast(dates, arms, slots)
    assert effect == pytest.approx(4)
    assert se == pytest.approx(np.sqrt(2.5))
    assert df == pytest.approx(100 / 34)
    with pytest.raises(ValueError, match="two assignment dates"):
        london.stratified_contrast(dates[:3], arms[:3], slots[:3])


def test_planted_price_shifts_yield_same_uncertainty_and_bias_on_paired_assignments():
    rng = np.random.default_rng(42)
    events = pd.DataFrame({"hour": np.repeat(np.arange(8), 6)})
    # All meters share a large date-level shock.
    matrix = rng.lognormal(size=(1024, 48)) + rng.uniform(0, 25, size=48)
    draws, _ = london.planted(events, matrix, 2)
    null = draws.loc[draws.truth == 0].reset_index(drop=True)
    shift = draws.loc[draws.truth == -0.18].reset_index(drop=True)
    np.testing.assert_allclose(null.estimate - shift.estimate, 0.18)
    np.testing.assert_array_equal(null.se, shift.se)
    large = null.loc[null.households == 1024].pivot(
        index=["rep", "days"], columns="inference", values="se"
    )
    assert (large.assignment_day > 3 * large.household_date_iid).all()
