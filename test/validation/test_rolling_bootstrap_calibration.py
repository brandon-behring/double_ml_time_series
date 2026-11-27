"""Check the refitted percentile calibration against direct draws from the decline scenario."""

import importlib

import numpy as np
import pandas as pd
import pytest

bootstrap = importlib.import_module(
    "docs.audits.2026-11-19_time_series_dml.external.forward.reminder_bootstrap_calibration"
)
calibration = bootstrap.calibration
reminders = bootstrap.reminders


def calendar():
    return calibration.calendar(
        pd.DataFrame(
            [
                {
                    reminders.ID: f"{origin}-{person}",
                    reminders.ENROLL: origin + pd.Timedelta(days=person),
                    reminders.DAY: day,
                    reminders.AGE: person + 60,
                    "base_step": 1000 + person,
                    reminders.PRIOR_STEPS: day * 2,
                }
                for origin in reminders.ORIGINS
                for person in range(10)
                for day in range(14, 98, 3)
            ]
        )
    )


@pytest.mark.tier1
def test_errors_and_refitted_percentiles_use_the_same_synthetic_realizations():
    cal = calendar()
    rep, seed, draws = 3, 11, 39
    results = pd.DataFrame(bootstrap.evaluate(cal, rep, seed, draws))
    others = pd.DataFrame(calibration.run_replication(cal, rep, seed))
    reference = others.loc[others.method.eq("aipw")].merge(results, on=["scenario", "origin"])
    np.testing.assert_allclose(reference.error, reference.signed_error, atol=1e-10)
    assert len(reference) == 8

    # Directly regenerate a genuinely shifted arm and refit all nuisance models.
    rng = np.random.default_rng(np.random.SeedSequence([seed, rep]))
    p, tau = calibration.randomization(cal, "decline")
    m = calibration.moments(
        cal,
        p,
        tau,
        rng.random(len(cal.decision_cells)),
        rng.normal(size=(len(cal.n), 2)),
        rng.normal(0, 0.7, len(cal.people)),
        rng.normal(0, 0.25, cal.week.max() + 1),
    )
    w = reminders.joint_person_weights(cal.enroll, draws, np.random.SeedSequence([seed, rep, 6863]))
    early = np.flatnonzero(cal.enroll + np.timedelta64(98, "D") < np.datetime64("2023-01-01"))
    later = np.flatnonzero(
        (cal.enroll >= np.datetime64("2023-01-01")) & (cal.enroll < np.datetime64("2023-04-01"))
    )
    f = reminders.crossfit(bootstrap.take(m, cal.people, early), w[:, early])
    z = reminders.arm_difference(bootstrap.take(m, cal.people, later), w[:, later])
    lo, hi = np.quantile(f - z, [0.025, 0.975])
    row = results.query("scenario=='decline' and origin=='2023-01-01'").iloc[0]
    assert row.width == pytest.approx(hi - lo)
    assert row.covered == float(lo <= 0.10 <= hi)
    assert row.alert == float(lo > 0 or hi < 0)


@pytest.mark.tier2
def test_serial_and_parallel_refit_calibration_are_identical():
    cal = calendar()
    a = bootstrap.study(cal, repetitions=3, draws=19, workers=1)
    b = bootstrap.study(cal, repetitions=3, draws=19, workers=2)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert len(a) == 8 and (a.replications == 3).all() and (a.intervals == 3).all()


@pytest.mark.tier1
def test_joint_alarm_keeps_shared_panel_and_excludes_incomplete_repetition():
    # Row two is a genuine quiet panel. Row three alerts before disappearing.
    # Treating the missing quarter as a non-alert would change the denominator.
    patterns = [[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, np.nan, 0], [0, 1, 0, 0]]
    rows = pd.DataFrame(
        [
            {
                "rep": rep,
                "origin": dt.date().isoformat(),
                "scenario": "stable",
                "true_error": 0.0,
                "signed_error": 0.0,
                "covered": 1 - alert,
                "alert": alert,
                "width": 0.1,
            }
            for rep, pattern in enumerate(patterns)
            for dt, alert in zip(bootstrap.ORIGINS, pattern, strict=True)
        ]
    )
    out = bootstrap.summarize(rows)
    end = out.iloc[-1]
    assert end.complete_panels == 3
    assert end.any_alert_four_dates == pytest.approx(2 / 3)
    assert end.any_alert_two_postbreak_dates == pytest.approx(1 / 3)
    assert end.any_alert_four_dates_mcse == pytest.approx(1 / 3)
    assert out.iloc[:-1].any_alert_four_dates.isna().all()
