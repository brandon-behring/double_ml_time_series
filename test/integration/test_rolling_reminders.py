"""Calendar, estimand, leakage and shared-resampling checks for the quarterly reminder study."""

import importlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

rolling = importlib.import_module(
    "docs.audits.2026-11-19_time_series_dml.external.forward.reminder_rolling"
)


def trial():
    rows = []
    for number, entry in enumerate(["2022-01-03"] * 15 + ["2022-07-03"] * 7 + ["2022-10-10"]):
        for occasion in range(16):
            activity = int(occasion % 8 == 0)
            none = int(occasion % 8 >= 2)
            age, day = 48 + number, 17 + occasion
            y = age * 0.1 + day * 0.3 + number * 0.02 + activity * (number / 4 + 1)
            rows.append(
                {
                    rolling.ID: f"P{number}",
                    rolling.DAY: day,
                    rolling.AGE: age,
                    rolling.ENROLL: pd.Timestamp(entry),
                    rolling.PRIOR_STEPS: occasion * 10,
                    "activity": activity,
                    "none": none,
                    "base_step": 600 + number,
                    "y": y,
                }
            )
    data = pd.DataFrame(rows)
    return data.drop(columns="y"), data.y


@pytest.mark.tier1
def test_grouped_weighted_ridge_scores_match_direct_row_models():
    meta, y = trial()
    sample = rolling.summarize(meta, y)
    weights = np.random.default_rng(12).uniform(0.5, 2, len(sample.people))
    folds = rolling.fold_membership(sample.people)
    people = sample.people.get_indexer(meta[rolling.ID])
    x = np.column_stack(
        (
            (meta[rolling.AGE] - 60) / 10,
            (meta[rolling.DAY] - 55) / 30,
            np.log1p(meta.base_step),
            np.log1p(meta[rolling.PRIOR_STEPS]),
        )
    )
    correction = np.zeros(len(meta))
    for fold in np.unique(folds):
        test = folds[people] == fold
        for arm, prob, sign in [("activity", 0.125, 1), ("none", 0.75, -1)]:
            train = ~test & meta[arm].eq(1)
            model = Ridge(alpha=1).fit(x[train], y[train], sample_weight=weights[people[train]])
            pred = model.predict(x[test])
            correction[test] += sign * (pred + meta.loc[test, arm] * (y[test] - pred) / prob)
    expected = np.average(correction, weights=weights[people])
    assert rolling.crossfit(sample, weights[None, :])[0] == pytest.approx(expected, abs=1e-10)


@pytest.mark.tier1
def test_outcome_reader_sees_the_freeze_before_the_holdout(tmp_path):
    meta, y = trial()
    origin = pd.Timestamp("2022-07-01")
    new = meta[rolling.ENROLL] >= origin
    changed = y.copy()
    changed.loc[new] += meta.loc[new, "activity"] * 1.7
    hashes, later, joint = [], [], []
    for i, outcomes in enumerate((y, changed)):
        dest = tmp_path / str(i)

        def read(ix, dest=dest, outcomes=outcomes):
            if new.loc[ix].any():
                assert (dest / "2022-07-01.json").exists()
            return outcomes.loc[ix]

        if i == 1:
            meta.loc[new, rolling.PRIOR_STEPS] = 99_999
        rows, summary = rolling.replay(meta, read, dest, [origin], draws=19)
        hashes.append(summary["freeze_sha256"])
        later.append(rows[0]["later_effect"])
        joint.append([r["forecast"] for r in rows])
    assert hashes[0] == hashes[1]
    assert joint[0] == joint[1]
    assert later[1] - later[0] == pytest.approx(1.7)


@pytest.mark.tier1
def test_empty_training_missing_outcomes_and_one_new_person_retained(tmp_path):
    meta, y = trial()
    incomplete = meta.index[meta.activity.eq(1)][0]
    y.loc[incomplete] = np.nan
    rows, summary = rolling.replay(
        meta,
        lambda ix: y.loc[ix],
        tmp_path,
        pd.to_datetime(["2022-01-01", "2022-07-01", "2022-10-01"]),
        draws=19,
    )
    assert len(rows) == 9
    zero, historical = rows[:2]
    assert zero["forecast"] == 0 and zero["later_missing_activity"] == 1
    assert historical["forecast"] is None and "insufficient" in historical["reason"]
    assert summary["full_comparison_origins"] == ["2022-07-01"]
    assert rows[-1]["signed_error"] is not None and "ci95_low" not in rows[-1]
    assert rows[-1]["reason"] == "point only: one later participant"


@pytest.mark.tier1
def test_missing_assignment_arm_returns_unavailable(tmp_path):
    meta, y = trial()
    meta.loc[meta[rolling.ENROLL] > "2022-07-01", "activity"] = 0
    rows, _ = rolling.replay(
        meta, lambda ix: y.loc[ix], tmp_path, [pd.Timestamp("2022-07-01")], draws=19
    )
    assert all(row["signed_error"] is None for row in rows)
    assert "evaluation arms absent" in rows[1]["reason"]


@pytest.mark.tier1
@pytest.mark.skipif(
    os.environ.get("DML_TS_EXTERNAL_REPLAY") != "1", reason="external pinned inputs"
)
def test_all_seven_quarters_reconcile_with_pinned_sources(tmp_path):
    cache = Path(os.environ.get("DML_TS_BENCHMARK_CACHE", Path.home() / ".cache/dml-ts/external"))
    result = rolling.run(cache, tmp_path)
    rows = result["origins"][1::3]
    assert [(r["train_people"], r["later_people"]) for r in rows] == [
        (0, 33),
        (0, 52),
        (28, 48),
        (83, 39),
        (131, 58),
        (172, 56),
        (226, 1),
    ]
    assert rows[4]["forecast"] == pytest.approx(0.0424, abs=0.00005)
    assert (tmp_path / "mybp-origins.csv").is_file()
    assert len(result["summary"]["full_comparison_origins"]) == 4


@pytest.mark.tier1
def test_last_training_enrollment_is_99_days_before_origin(tmp_path):
    meta, y = trial()
    origin = pd.Timestamp("2022-07-01")
    meta.loc[meta[rolling.ID] == "P0", rolling.ENROLL] = origin - pd.Timedelta(days=98)
    meta.loc[meta[rolling.ID] == "P1", rolling.ENROLL] = origin - pd.Timedelta(days=99)
    read_ids = []

    def read(ix):
        read_ids.append(set(meta.loc[ix, rolling.ID]))
        return y.loc[ix]

    rolling.replay(meta, read, tmp_path, [origin], draws=19)
    assert "P1" in read_ids[0] and "P0" not in read_ids[0]
    assert not read_ids[0].intersection(read_ids[1])
