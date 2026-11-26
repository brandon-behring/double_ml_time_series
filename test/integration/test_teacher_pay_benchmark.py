"""Check temporal isolation, shared-school covariance and the published Indonesian trial."""

import importlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

case = importlib.import_module(
    "docs.audits.2026-11-19_time_series_dml.external.forward.teacher_pay"
)


@pytest.fixture
def followups():
    rows = []
    for year in (2, 3):
        for school in range(12):
            for student in range(3):
                treated = int(school % 3 == 0)
                # School outcome shocks persist identically across the waves.
                score = (school % 5) + (1 + 0.1 * school) * student + treated * year
                rows.append(
                    [school, school // 3, school * 3 + student, year, treated, student + 1, score]
                )
    return pd.DataFrame(rows, columns=[*case.IDS, "treatment", "weight", "ITT_score"])


@pytest.mark.tier1
def test_weighted_fixed_effects_and_school_uncertainty_match_dummy_regression(followups):
    wave = followups.loc[followups.year.eq(2)].copy()
    # A covariate already explained by the triplet fixed effects can be absorbed.
    wave["triplet_control"] = wave.triplet_id % 2
    measured, _ = case.fit_wave(wave, ["triplet_control"])
    indicators = pd.get_dummies(wave.triplet_id, dtype=float)
    ref = sm.WLS(
        wave.ITT_score, pd.concat([wave.treatment, indicators], axis=1), weights=wave.weight
    ).fit(cov_type="cluster", cov_kwds={"groups": wave.school_id})
    assert measured["effect_sd"] == pytest.approx(ref.params.iloc[0])
    assert measured["se_sd"] == pytest.approx(ref.bse.iloc[0])
    assert measured["r_squared"] == pytest.approx(ref.rsquared)


@pytest.mark.tier1
def test_future_outcomes_and_reused_school_scores(followups):
    result = case.compare_waves(followups, [])
    assert result["difference_se_sd"] == pytest.approx(0, abs=1e-12)
    assert result["early_minus_later_sd"] == pytest.approx(-1)
    changed = followups.copy()
    changed.loc[changed.year.eq(3), "ITT_score"] += (
        100 * changed.loc[changed.year.eq(3), "treatment"]
    )
    changed_result = case.compare_waves(changed.sample(frac=1, random_state=42), [])
    assert changed_result["year_2"]["effect_sd"] == pytest.approx(result["year_2"]["effect_sd"])
    assert changed_result["early_minus_later_sd"] == pytest.approx(-101)
    # Relabel later schools to remove all covariance while preserving each wave's model.
    changed.loc[changed.year.eq(3), "school_id"] += 100
    independent = case.compare_waves(changed, [])
    assert independent["covariance_sd2"] == 0
    assert independent["difference_se_sd"] == pytest.approx(
        np.hypot(result["year_2"]["se_sd"], result["year_3"]["se_sd"])
    )
    followups.loc[0, "treatment"] = 1 - followups.loc[0, "treatment"]
    with pytest.raises(ValueError, match="one randomized assignment"):
        case.compare_waves(followups, [])


@pytest.mark.network
@pytest.mark.tier4
@pytest.mark.skipif(os.getenv("DML_TS_EXTERNAL_REPLAY") != "1", reason="opt-in archive replay")
def test_original_teacher_pay_file_and_joint_followup(tmp_path):
    result = case.run(Path(os.getenv("DML_TS_BENCHMARK_CACHE", tmp_path)))
    primary = result["table_v"]
    assert primary["shared_schools"] == 348
    assert primary["shared_students"] == 53504
    assert primary["early_minus_later_sd"] == pytest.approx(-0.0158113622)
    assert primary["difference_ci95_sd"] == pytest.approx([-0.044803792, 0.013181068])
    assert result["common_school_sensitivity"]["year_2"]["schools"] == 348
    assert result["no_baseline_sensitivity"]["early_minus_later_sd"] == pytest.approx(-0.0269747263)
