"""Regression checks for the method ranking and the scientific audit inventory."""

import importlib
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from scripts.comprehensive_method_comparison import best_baseline_comparison, rank_methods

pytestmark = pytest.mark.tier2
AUDIT = "docs.audits.2026-11-21_evaluation_methods.reproduction"
ROOT = Path(__file__).resolve().parents[2]


def test_winners_average_all_configurations_and_measure_coverage_calibration():
    # A wins one cell on bias but is much worse than B on average.
    t = pd.DataFrame(
        {
            "Method": ["A", "A", "B", "B"],
            "Abs_Bias": [0.0, 1.0, 0.2, 0.2],
            "MSE": [0.01, 9.0, 0.4, 0.4],
            "Coverage": [1.0, 1.0, 0.94, 0.96],
        }
    )
    leaders = rank_methods(t)
    assert leaders["lowest_mean_bias"]["Method"] == leaders["lowest_mse"]["Method"] == "B"
    assert leaders["highest_coverage"]["Method"] == "A"
    assert leaders["closest_to_nominal_coverage"]["Method"] == "B"
    scores = {k: {"mean_abs_bias": v} for k, v in {"A": 0.5, "B": 0.2, "DML": 0.3}.items()}
    best, gain = best_baseline_comparison(scores)
    assert best == "B" and gain == pytest.approx(-50)
    assert best_baseline_comparison({**scores, "XGBoost": {"mean_abs_bias": 0.1}})[0] == "XGBoost"
    scores["B"]["mean_abs_bias"] = 0
    assert best_baseline_comparison(scores) == ("B", None)


def test_missingness_label_maturity_and_registry_crosswalk():
    p = importlib.import_module(AUDIT + ".design_probes").evaluate()
    assert p["attrition"] == {
        "true_effect": 0,
        "eligible": 400,
        "retained_by_arm": [0.5, 0.5],
        "complete_case_effect": 2.0,
    }
    clock = p["clock"]
    assert clock["regular"] == clock["gap_of_300_days"]
    assert clock["training_labels_unavailable_at_origin"] == {
        "horizon": 4,
        "without_gap": 4,
        "with_horizon_gap": 0,
    }
    module = importlib.import_module(AUDIT + ".inventory_check")
    rows, studies = module.crosswalk()
    saved = pd.read_csv(module.HERE / "registry-crosswalk.csv")
    assert list(saved.itertuples(index=False, name=None)) == rows
    assert len(studies) == 40


def test_chapter_and_active_bibliography_trigger_incremental_build(tmp_path):
    import os
    import shutil

    if not shutil.which("make"):
        pytest.skip("GNU make unavailable")
    shutil.copyfile(ROOT / "Makefile", tmp_path / "Makefile")
    for name in (
        "main.tex",
        "bibliography.bib",
        "chapters/chapter_09.tex",
        "chapters/appendix_a.tex",
        "shared/dml-preamble-tufte.sty",
        "main.pdf",
    ):
        f = tmp_path / name
        f.parent.mkdir(exist_ok=True)
        f.touch()
        os.utime(f, (20, 20) if name == "main.pdf" else (10, 10))
    assert subprocess.run(["make", "-q", "main.pdf"], cwd=tmp_path).returncode == 0
    for name in (
        "chapters/chapter_09.tex",
        "chapters/appendix_a.tex",
        "bibliography.bib",
        "shared/dml-preamble-tufte.sty",
    ):
        assert subprocess.run(["make", "-q", "-W", name, "main.pdf"], cwd=tmp_path).returncode == 1
