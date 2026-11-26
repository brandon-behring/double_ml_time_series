import importlib

import numpy as np
import pandas as pd
import pytest

calib = importlib.import_module(
    "docs.audits.2026-11-19_time_series_dml.external.forward.rolling_calibration"
)


def sample_calendar():
    frame = []
    for quarter in range(7):
        for i in range(12):
            origin = calib.ORIGINS[quarter] + pd.Timedelta(days=i)
            for day in (14, 15, 19, 35, 50, 97):
                frame.append(
                    {
                        calib.ID: f"{quarter}-{i}",
                        calib.ENROLL: origin,
                        calib.DAY: day,
                        calib.AGE: 55 if i % 2 else 75,
                        "base_step": 100.0,
                        calib.PRIOR_STEPS: day,
                    }
                )
    return calib.calendar(pd.DataFrame(frame))


@pytest.mark.tier1
def test_change_in_randomization_flips_native_estimand_without_subgroup_drift():
    cal = sample_calendar()
    p, tau = calib.randomization(cal, "changing_assignment")
    # Deterministic fractional arm counts provide exact population sufficient statistics.
    n = np.full(len(p), 100.0)
    x = np.ones((len(n), 1))
    counts = n[:, None] * p
    arm_y = np.column_stack((counts[:, 0] * (1 + tau), counts[:, 1]))
    m = calib.Moments(n, n[:, None] * x, counts[:, :, None, None], arm_y[:, :, None], arm_y)
    for mask, sign in ((cal.enroll < calib.BREAK, 1), (cal.enroll >= calib.BREAK, -1)):
        fits = calib.fits(cal, m, np.flatnonzero(mask), p, tau)
        expected = sign * 0.2 * (0.25 - 0.09) / (0.25 + 0.09)
        assert fits["native_plr"].value == pytest.approx(expected)
        assert fits["native_plr"].truth == pytest.approx(expected)
        for method in ("aipw", "historical", "fixed_mix_aipw"):
            assert fits[method].value == pytest.approx(0, abs=1e-11)


@pytest.mark.tier2
def test_replay_all_dates_including_warmup_and_deterministic_parallelism():
    cal = sample_calendar()
    serial = calib.study(cal, 2, 1)
    parallel = calib.study(cal, 2, 2)
    pd.testing.assert_frame_equal(serial, parallel, check_exact=True)
    assert len(serial) == 2 * 7 * 4 * 5
    warmup = serial[(serial.origin == "2022-01-01") & (serial.method == "aipw")]
    assert warmup.forecast.isna().all() and warmup.reason.str.contains("history").all()
    summary = calib.summarize(serial)
    assert len(summary) == 4 * 7 * 5
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        calib.summarize(serial.iloc[1:])


@pytest.mark.tier1
def test_absent_treatment_and_shared_week_shocks():
    cal = sample_calendar()
    p, tau = calib.randomization(cal, "stable")
    n_cells = len(cal.n)
    uniforms = np.full(len(cal.decision_cells), 0.20)
    input_noise = (np.zeros((n_cells, 2)), np.zeros(len(cal.people)))
    no_shock = calib.moments(cal, p, tau, uniforms, *input_noise, np.zeros(cal.week.max() + 1))
    with_shock = calib.moments(cal, p, tau, uniforms, *input_noise, np.ones(cal.week.max() + 1))
    np.testing.assert_allclose(with_shock.y[:, 1] - no_shock.y[:, 1], no_shock.n)
    assert calib.fits(cal, no_shock, np.arange(len(p)), p, tau) == {}
