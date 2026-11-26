"""Regression oracles for the exploratory historical half-effect replay."""

import importlib
import os
from pathlib import Path

import pytest

replay = importlib.import_module(
    "docs.audits.2026-11-19_time_series_dml.external.forward.shrinkage_replay"
)


@pytest.mark.tier1
def test_existing_frozen_record_supplies_signed_half_or_missing_history():
    import json

    refs = json.loads((replay.ROLLING / "mybp-summary.json").read_text())["freeze_sha256"]
    for date, expected_empty in [("2022-01-01", True), ("2022-07-01", False)]:
        path = replay.ROLLING / "mybp-freezes" / f"{date}.json"
        raw = path.read_bytes()
        values = replay.frozen_values(path, refs[date])
        if expected_empty:
            assert values["half_history"] is None
        else:
            assert values["half_history"] == values["historical"] / 2 < 0
        assert path.read_bytes() == raw
        with pytest.raises(ValueError, match="SHA-256"):
            replay.frozen_values(path, refs[date][::-1])


@pytest.mark.tier1
def test_jtpa_separate_source_versions_on_same_holdout_and_midpoint():
    rows = replay.jtpa_replay()
    assert len(rows) == 4
    assert len({r["estimate"] for r in rows}) == 1
    primary = next(
        r
        for r in rows
        if r["source_variant"] == "post_unblinding" and r["reporting_lag_quarters"] == 1
    )
    values = {r["method"]: r for r in primary["forecasts"]}
    assert values["half_history"]["forecast"] == values["historical"]["forecast"] * 0.5
    comparison = primary["dml_vs_half_fixed"]
    assert comparison["gain"] == pytest.approx(
        values["half_history"]["squared_error"] - values["dml"]["squared_error"]
    )
    assert comparison["status"] == "unresolved"
    assert comparison["gain_ci95"] == pytest.approx([-34559.34236, 688.57519])
    assert primary["dml_move_from_history"] == "away_from_zero"


@pytest.mark.tier2
@pytest.mark.skipif(os.environ.get("DML_TS_EXTERNAL_REPLAY") != "1", reason="pinned trial source")
def test_all_reminder_origin_statuses_and_paired_refit_interval():
    cache = Path(os.environ.get("DML_TS_BENCHMARK_CACHE", Path.home() / ".cache/dml-ts/external"))
    result = replay.reminder_replay(cache)
    assert [
        None if r["dml_vs_half_fixed"] is None else r["dml_vs_half_fixed"]["status"]
        for r in result["schedule"]
    ] == [
        None,
        None,
        "comparator_closer",
        "unresolved",
        "unresolved",
        "unresolved",
        "point_only",
    ]
    assert len(result["full_comparison_origins"]) == 4
    assert result["mse"]["half_history"] == pytest.approx(0.0034983689515363766)
    assert result["dml_gains"]["half_history"]["ci95_refit"] == pytest.approx(
        [-0.008428286787685533, 0.0007481245180817184]
    )
