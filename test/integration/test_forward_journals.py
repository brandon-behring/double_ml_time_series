"""Regression checks for the 2018 print roster, evidence coding and denominators."""

import importlib

import pytest

journal = importlib.import_module(
    "docs.audits.2026-11-19_time_series_dml.external.forward.journals"
)
pytestmark = pytest.mark.tier1


def test_all_issues_and_independent_denominators():
    rows = journal.read_ledger()
    result = journal.summary(rows)
    assert (result["issued_items"], result["non_front_matter"], result["empirical"]) == (
        215,
        200,
        158,
    )
    assert {
        j: (v["issued_items"], v["issues"], v["empirical"]) for j, v in result["by_journal"].items()
    } == {"AER": (125, 11, 78), "AEJ Applied": (50, 4, 46), "QJE": (40, 4, 34)}
    assert result["temporal_target"]["effect projection"] == 3
    assert result["temporal_target"]["outcome prediction"] == 5
    assert sum(result["data_access"].values()) == 158
    assert result["data_access"]["acquisition route unverified"] > 0
    assert sum(r["journal"] == "QJE" and r["published_online_year"] == "2017" for r in rows) == 22


def test_invalid_exclusions_and_missing_findings_are_detected():
    empirical = next(r for r in journal.read_ledger() if r["eligible"] == "yes")
    with pytest.raises(ValueError, match="eligibility conflicts"):
        journal.validate([dict(empirical, eligible="no")])
    with pytest.raises(ValueError, match="missing published_finding"):
        journal.validate([dict(empirical, published_finding="")])
    with pytest.raises(ValueError, match="duplicate"):
        journal.validate([empirical, empirical])


def test_roster_comparison_checks_issue_and_multiplicity():
    row = journal.read_ledger()[0]
    member = (row["doi"], row["journal"], row["volume"], row["issue"])
    journal.compare_rosters([row], [member])
    for bad in ([], [member, member], [(*member[:-1], "wrong")]):
        with pytest.raises(ValueError, match="Print rosters differ"):
            journal.compare_rosters([row], bad)


def test_crossref_retries_and_reuses_complete_response(monkeypatch, tmp_path):
    import io
    import json
    from urllib.error import HTTPError

    body = {
        "message": {"total-results": 1, "items": [{"DOI": "10.1/Q", "volume": "133", "issue": "1"}]}
    }
    calls = []

    def once_limited(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise HTTPError(request.full_url, 429, "rate limited", {"Retry-After": "0"}, None)
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(journal, "urlopen", once_limited)
    expected = [("10.1/q", "QJE", "133", "1")]
    assert journal.fetch_volume("QJE", tmp_path) == expected
    assert journal.fetch_volume("QJE", tmp_path) == expected
    assert len(calls) == 2
    assert "from-print-pub-date" in calls[0]
    body["message"]["total-results"] = 3
    with pytest.raises(ValueError, match="incomplete"):
        journal.fetch_volume("QJE", tmp_path / "incomplete")
