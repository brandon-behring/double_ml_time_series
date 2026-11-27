"""Offline checks of the commuter-level treatment, timing and uncertainty conventions."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

pytestmark = pytest.mark.tier2

FOLDER = (
    Path(__file__).resolve().parents[2] / "docs/audits/2026-11-19_time_series_dml/field/bangalore"
)
SPEC = importlib.util.spec_from_file_location("bangalore_field_audit", FOLDER / "replay.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_within_fit_matches_explicit_commuter_dummies_and_cr1():
    rng = np.random.default_rng(54)
    records = []
    for person in range(32):
        arm = audit.ARMS[person % 4]
        for day in range(6 - person % 2):
            post = int(day >= 3)
            cycle = 0 if not post else day - 2
            y = 10 + person + 2.5 * (arm == "high") * post + rng.normal()
            records.append((person, arm, cycle, post, y, True))
    panel = pd.DataFrame(records, columns=["uidp", "arm", "study_cycle", "post", "y", "observed"])
    design = pd.concat(
        [
            pd.get_dummies(panel.uidp, dtype=float),
            pd.get_dummies(panel.study_cycle, prefix="cycle", drop_first=True, dtype=float),
            pd.DataFrame({arm: panel.arm.eq(arm) * panel.post for arm in audit.ARMS[:-1]}),
        ],
        axis=1,
    )
    fit = sm.OLS(panel.y, design).fit(cov_type="cluster", cov_kwds={"groups": panel.uidp})
    for data in [panel, panel.sample(frac=1, random_state=9)]:
        result = audit.fixed_effects(data)
        np.testing.assert_allclose(
            result["arm_coefficients"]["high"], fit.params["high"], atol=1e-10
        )
        np.testing.assert_allclose(result["cluster_se"]["high"], fit.bse["high"], atol=1e-10)


def test_histories_exclude_assignment_day_and_later_records():
    date = pd.date_range("2017-03-01", periods=4)
    panel = pd.DataFrame(
        {
            "uidp": "p",
            "date": date,
            "assignment_date": date[2],
            "observed": True,
            "post": [0, 0, 0, 1],
            "y": [10, 20, 900, 800],
            "trips": [1, 3, 90, 80],
        }
    )
    features = audit.baseline_histories(panel)
    assert features.mean_rate.eq(15).all() and features.pre_days.eq(2).all()
    assert features.mean_trips.eq(2).all()


def test_published_targets_and_forward_support_receipt():
    receipt = json.loads((FOLDER / "results.json").read_text())
    fit = receipt["published_specification"]
    assert (fit["n"], fit["commuters"]) == (15_574, 497)
    assert round(fit["arm_coefficients"]["high"], 2) == -12.94
    assert round(fit["cluster_se"]["high"], 2) == 6.04
    assert round(fit["control_mean"], 2) == 96.88
    for row in receipt["rolling"]:
        if row["status"] == "scored":
            assert row["last_training_date"] < row["month"] + "-01"
            assert min(row["test_high"], row["test_ctrl"]) >= 2
    assert receipt["rolling"][-1]["unadjusted"]["effect"] is None
