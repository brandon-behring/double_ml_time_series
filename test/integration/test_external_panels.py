"""Opt-in replays of independently published estimates from author-hosted data."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docs/audits/2026-11-19_time_series_dml/external/panels/replay.py"
pytestmark = [
    pytest.mark.tier4,
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("DML_TS_EXTERNAL_REPLAY") != "1",
        reason="set DML_TS_EXTERNAL_REPLAY=1 to replay the authors' archives",
    ),
]


@pytest.mark.parametrize("study", ["wolfers", "dahl"])
def test_external_panel_replay(study, tmp_path):
    cache = os.environ.get("DML_TS_BENCHMARK_CACHE", str(tmp_path))
    output = subprocess.check_output(
        [sys.executable, str(SCRIPT), study, "--cache-dir", cache],
        cwd=ROOT,
        text=True,
    )
    result = json.loads(output)
    # Run through the CLI so its paper comparisons, checksum checks and native-estimator
    # comparisons are exercised together. Require the expected specification coverage.
    if study == "wolfers":
        assert [len(model["estimate"]) for model in result["table_2"]] == [8, 8, 8]
    else:
        assert set(result["table_2"]) == {"1", "6", "7"}
        assert [len(model["se"]) for model in result["table_3"].values()] == [3, 3, 3, 3]
