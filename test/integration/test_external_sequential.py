"""Opt-in tests of three published results using fingerprinted, external archives."""

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "docs/audits/2026-11-19_time_series_dml/external/sequential"


def replay(study, tmp_path):
    cache = os.environ.get("DML_TS_BENCHMARK_CACHE", str(tmp_path))
    output = subprocess.check_output(
        [sys.executable, str(SCRIPTS / f"{study}.py"), "--cache-dir", cache], cwd=ROOT, text=True
    )
    return json.loads(output)


@pytest.mark.tier1
def test_calendar_gap_breaks_feature_history_and_future_label(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    calendar_support = runpy.run_path(str(SCRIPTS / "italy.py"))["calendar_support"]
    quarters = pd.period_range("2016Q1", "2021Q4", freq="Q")
    quarters = quarters[quarters != pd.Period("2018Q3", freq="Q")]
    origins = pd.PeriodIndex(["2017Q4", "2018Q1", "2018Q2", "2018Q4", "2019Q3", "2019Q4"], freq="Q")
    actual = calendar_support(origins.to_timestamp(), quarters.to_timestamp(), horizon=2)
    assert actual.tolist() == [True, False, False, False, False, True]


@pytest.mark.tier4
@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("DML_TS_EXTERNAL_REPLAY") != "1",
    reason="set DML_TS_EXTERNAL_REPLAY=1 to download and replay the studies",
)
class TestExternalReplays:
    def test_italian_raw_to_figure_and_forward_support(self, tmp_path):
        result = replay("italy", tmp_path)
        assert result["cells_compared"] == 225 and result["max_abs_error"] < 1e-8
        # These two GDP values belong to the revised paper and the author's 50bp scaling.
        assert result["GDP_h2"]["Coefficient"] == pytest.approx(-0.143807374, abs=1e-8)
        assert result["GDP_h2"]["CI_Upper_95"] == pytest.approx(-0.043357697, abs=1e-8)
        assert result["excluded_origins"] == ["2019-09-30", "2019-12-31", "2022-03-31"]
        for key, count, theta, se in [
            ("published_rows", 39, -0.0178660, 0.0962703),
            ("continuous_calendar", 36, -0.1027573, 0.0715767),
        ]:
            native = result["forward_h2"][key]
            assert native["scored"] == count
            assert native["theta"] == pytest.approx(theta, abs=1e-5)
            assert native["se"] == pytest.approx(se, abs=1e-5)
            assert native["lower"] < 0 < native["upper"]
            assert native["outcome_r2"] < 0

    def test_drink_less_published_and_design_probability(self, tmp_path):
        estimates = replay("drink_less", tmp_path)["estimates"]
        assert estimates["one_hour"]["ratio"] == pytest.approx(3.523, abs=0.0005)
        assert estimates["24_hours"]["ratio"] == pytest.approx(1.260, abs=0.0005)
        assert estimates["one_hour_design_probability"]["ratio"] == pytest.approx(
            3.524335, abs=1e-6
        )
        assert estimates["one_hour"]["log_se"] > estimates["one_hour"]["independent_row_log_se"]
        assert estimates["one_hour_aipw"]["risk_difference"] == pytest.approx(0.090995, abs=5e-6)

    def test_mybp_clustered_activity_outcome(self, tmp_path):
        result = replay("mybp", tmp_path)
        assert (result["participants"], result["randomizations"], result["available"]) == (
            287,
            187517,
            153983,
        )
        estimate = result["activity"]
        assert estimate["ratio"] == pytest.approx(1.01, abs=0.005)
        assert estimate["p"] == pytest.approx(0.40, abs=0.005)
        assert result["all_complete_activity"]["ratio"] == pytest.approx(1.022405, abs=1e-6)
