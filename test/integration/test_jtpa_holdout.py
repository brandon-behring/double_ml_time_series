"""Outcome-isolation, source coding and estimand checks for the blinded JTPA replay."""

import hashlib
import importlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

jtpa = importlib.import_module(
    "docs.audits.2026-11-19_time_series_dml.external.forward.jtpa_holdout"
)


def fixture():
    dates = pd.to_datetime(np.repeat(["1988-03-31", "1988-06-30", "1988-07-01", "1989-01-01"], 24))
    ids = [f"{i:04d}" for i in range(len(dates))]
    treated = np.arange(len(dates)) % 2
    design = pd.DataFrame(
        {
            "recid": ids,
            "assigned_at": dates,
            "quarter": dates.to_period("Q"),
            "assigned": treated,
            "adultf": treated.astype(str),
            "site": "AA",
            "age": 31.0,
        }
    )

    class Wages:
        index = pd.DataFrame(
            {"recid": ids, "row": np.arange(len(ids)), "ardsite": "AA", "rejflag": "0"}
        )

        def __init__(self):
            self.earnings = 100 + 45.0 * treated
            self.requested = []

        def read_wages(self, selected, **kwargs):
            self.requested.extend(selected.astype(int).tolist())
            return self.earnings[selected.to_numpy(dtype=int)]

    return design, Wages()


@pytest.mark.tier1
def test_issuance_reads_only_mature_participants_and_never_changes_on_later_outcomes(tmp_path):
    baseline, wages = fixture()
    issue = jtpa.issue(baseline, wages, tmp_path)
    assert max(wages.requested) == 47
    assert issue["1"]["latest_assignment"] == "1988-06-30"
    assert issue["2"]["latest_assignment"] == "1988-03-31"
    records = json.loads((tmp_path / "jtpa-corrected-lag1-frozen.json").read_text())["forecasts"]
    assert [(r["method"], r["value"]) for r in records] == [
        ("zero", 0),
        ("historical", 45),
        ("dml", 45),
    ]
    first = jtpa.evaluate(baseline, wages, tmp_path, issue)
    assert first["estimate"] == 45
    wages.earnings[73::2] += 200
    assert jtpa.issue(baseline, wages, tmp_path) == issue
    assert jtpa.evaluate(baseline, wages, tmp_path, issue)["estimate"] == 245


@pytest.mark.tier1
def test_issuance_verification_precedes_opening_the_holdout(tmp_path):
    baseline, wages = fixture()
    issued = jtpa.issue(baseline, wages, tmp_path)
    path = tmp_path / "jtpa-corrected-lag2-frozen.json"
    path.write_text(path.read_text() + " ")
    wages.requested.clear()
    with pytest.raises(ValueError, match="digest mismatch"):
        jtpa.evaluate(baseline, wages, tmp_path, issued)
    assert wages.requested == []


@pytest.mark.tier1
def test_original_and_corrected_source_coding_and_ambiguous_fields(monkeypatch):
    archive = object.__new__(jtpa.WageArchive)
    fields = {"uearn01": [0, 7830, 99999], "uempl01": [0, 1, 99], "anflg01": [2, 2, np.nan]}
    monkeypatch.setattr(
        archive,
        "read_field",
        lambda rows, name: np.array(fields[name], dtype=float)[np.asarray(rows)],
    )
    np.testing.assert_equal(archive.read_wages([0, 1, 2]), [0, 7830, np.nan])
    np.testing.assert_equal(archive.read_wages([2], reproduce_original_issuance=True), [99999])
    fields["uempl01"][2] = 1
    with pytest.raises(ValueError, match="Ambiguous"):
        archive.read_wages([2])


@pytest.mark.tier1
def test_score_counts_missing_linkage_and_keeps_recorded_zero():
    baseline, wages = fixture()
    selected = jtpa.select_cohort(baseline, jtpa.ORIGIN)
    wages.earnings[72] = 0
    wages.earnings[73] = np.nan
    wages.index = wages.index.copy()
    wages.index.loc[74, "rejflag"] = "1"
    frame, counts = jtpa.prepare(selected, wages)
    assert counts["observed"] == {"offer": 11, "control": 11}
    assert counts["observed_zeros"]["control"] == 1
    assert 74 not in wages.requested
    assert len(frame) == 22


@pytest.mark.tier1
def test_frozen_records_and_source_correction_remain_separate():
    published = json.loads((jtpa.ROLLING / "jtpa-issuance.json").read_text())
    assert jtpa.protocol()["origins"] == [jtpa.ORIGIN]
    for lag in (1, 2):
        raw = jtpa.frozen_path(jtpa.ROLLING, lag, True).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == published[str(lag)]["sha256"]
    sealed = json.loads((jtpa.ROLLING / "jtpa-results.json").read_text())
    adjusted = json.loads((jtpa.ROLLING / "jtpa-corrected-results.json").read_text())
    assert sealed["estimate"] == adjusted["estimate"]
    assert sealed["scores"][1]["forecast"] != adjusted["scores"][1]["forecast"]
    assert len(sealed["scores"]) == len(adjusted["scores"]) == 6


@pytest.mark.tier1
@pytest.mark.skipif(
    os.getenv("DML_TS_EXTERNAL_REPLAY") != "1", reason="opt-in pinned public archive"
)
def test_checksum_pinned_replay_recovers_all_scores_and_source_check(tmp_path):
    cache = Path(os.getenv("DML_TS_BENCHMARK_CACHE", str(Path.home() / ".cache/dml-ts/external")))
    replay = jtpa.run(cache, tmp_path, "replay")
    for result, filename in (
        ("sealed", "jtpa-results"),
        ("post_unblinding_correction", "jtpa-corrected-results"),
        ("source_audit", "jtpa-source-audit"),
    ):
        assert replay[result] == json.loads((jtpa.ROLLING / f"{filename}.json").read_text())
    # Independently verify the byte-level selective decoder against pandas only
    # after the full fixed-issuance replay has been scored.
    archive = jtpa.WageArchive(cache / "jtpa-national-evaluation.zip")
    names = ["uearn01", "uempl01", "anflg01"]
    full = pd.read_stata(io.BytesIO(archive.data), columns=names, convert_categoricals=False)
    selected = pd.Series(np.arange(0, len(full), 17))
    for name in names:
        np.testing.assert_equal(
            archive.read_field(selected, name), full[name].iloc[selected].to_numpy()
        )
