"""Validate the Drink Less early/late contrast and its participant-level uncertainty."""

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docs/audits/2026-11-19_time_series_dml/external/questions/reminders.py"


@pytest.fixture
def functions(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    return runpy.run_path(str(SCRIPT))


@pytest.mark.tier1
def test_known_probability_score_recovers_risk_difference(functions):
    # Averaging over the exact assignment proportions eliminates nuisance error.
    assignment = np.array([1, 1, 1, 0, 0])
    outcome = np.where(assignment == 1, 0.7, 0.2)
    for mu0, mu1 in [(0.0, 0.0), (0.2, 0.7), (0.35, 0.4)]:
        score = functions["augmented_score"](assignment, outcome, mu0, mu1)
        assert score.mean() == pytest.approx(0.5)


@pytest.mark.tier1
def test_paired_se_is_standard_error_of_within_participant_differences(functions):
    scores = np.array([[0.0, 0.1], [0.3, 0.2], [0.9, 0.7], [0.4, 0.3]])
    differences = scores[:, 1] - scores[:, 0]
    summary = functions["paired_summary"](scores)
    contrast = summary["late_minus_early"]
    assert contrast["risk_difference"] == pytest.approx(differences.mean())
    assert contrast["se"] == pytest.approx(differences.std(ddof=1) / np.sqrt(len(scores)))
    assert contrast["se"] < np.hypot(summary["early"]["se"], summary["late"]["se"])


@pytest.mark.tier4
@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("DML_TS_EXTERNAL_REPLAY") != "1",
    reason="set DML_TS_EXTERNAL_REPLAY=1 to replay the public trial",
)
def test_drink_less_week_contrast(tmp_path):
    cache = os.environ.get("DML_TS_BENCHMARK_CACHE", str(tmp_path))
    output = subprocess.check_output(
        [sys.executable, str(SCRIPT), "--cache-dir", cache], cwd=ROOT, text=True
    )
    result = json.loads(output)
    assert (result["participants"], result["scored_decisions"]) == (349, 4886)
    assert result["paper_table2_relative_risk"]["ratio"] == pytest.approx(3.523, abs=0.0005)
    assert result["counts"]["early"]["any"]["opens"] == 278
    assert result["counts"]["late"]["none"]["opens"] == 26
    for method, early, late, paired_se in [
        ("ipw", 0.14258425, 0.05525993, 0.01669381),
        ("aipw", 0.14752681, 0.05979606, 0.01582006),
    ]:
        fit = result["estimates"][method]
        assert fit["early"]["risk_difference"] == pytest.approx(early, abs=1e-5)
        assert fit["late"]["risk_difference"] == pytest.approx(late, abs=1e-5)
        assert fit["late_minus_early"]["risk_difference"] == pytest.approx(late - early, abs=1e-5)
        assert fit["late_minus_early"]["se"] == pytest.approx(paired_se, abs=1e-5)
