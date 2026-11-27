"""Regression oracles for the prospective innings-choice effect and fixture ledger."""

import importlib
import json
import os
from datetime import UTC, date, datetime
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

live = importlib.import_module(
    "docs.audits.2026-11-19_time_series_dml.external.forward.prospective_cricket"
)
cricket = live.cricket


def info(day, outcome=None, toss_winner="Zimbabwe", event="Tour"):
    return {
        "dates": [day],
        "team_type": "international",
        "match_type": "T20",
        "gender": "female",
        "season": "2026/27",
        "event": {"name": event},
        "teams": ["Zimbabwe", "England"],
        "toss": {"winner": toss_winner, "decision": "bat"} if toss_winner else {},
        "outcome": {"winner": "Zimbabwe"} if outcome is None else outcome,
    }


def future_fixture(tmp):
    # Entirely artificial future dates for checking the calendar gate.
    day1, day2 = "2026-12-03", "2026-12-09"
    path = tmp / "synthetic.zip"
    with ZipFile(path, "w") as z:
        z.writestr(
            "README.txt",
            "\n".join(
                f"{d} - international - T20 - female - {i} - Zimbabwe vs England"
                for i, d in (("13", day1), ("14", day2), ("15", "2027-04-22"))
            ),
        )
        z.writestr("13.json", json.dumps({"info": info(day1)}))
        z.writestr("14.json", json.dumps({"info": info(day2, {"winner": "England"}, event="Cup")}))
        # This would raise if an unfinished quarter's individual match were opened.
        z.writestr("15.json", "invalid")
    rows = []
    for ident, match, day, status in (
        ("icc:A", "13", day1, "tossed"),
        ("icc:B", "14", day2, "tossed"),
        ("icc:C", "icc:C", "2026-12-12", "cancelled_before_toss"),
    ):
        rows.append(
            {
                "fixture_id": ident,
                "match_id": match,
                "date": day,
                "gender": "female",
                "team1": "England",
                "team2": "Zimbabwe",
                "status": status,
                "source": "https://example.org/fixture",
            }
        )
    return path, rows


@pytest.mark.tier1
@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        ({"winner": "England"}, 1),
        ({"result": "tie", "eliminator": "Zimbabwe"}, 0),
        ({"result": "tie", "bowl_out": "England"}, 1),
        ({"result": "no result"}, 0.5),
        ({"result": "tie"}, 0.5),
    ],
)
def test_toss_right_orientation_and_tiebreak(outcome, expected):
    day = "2010-03-03"
    data = info(day, outcome)
    row = cricket.match_record("9", day, data)
    assert (row["focal"], row["focal_toss"], row["focal_credit"]) == ("England", 0, expected)
    assert cricket.score_contributions(pd.DataFrame([row]))[0] == 100 * (1 - 2 * expected)
    data["teams"].reverse()
    data["toss"]["decision"] = "field"
    assert cricket.match_record("9", day, data) == row


@pytest.mark.tier1
def test_toss_and_outcome_classification_and_preassignment_features():
    day = "2010-03-03"
    no = info(day, toss_winner=None)
    assert cricket.match_record("1", day, no)["status"] == "no_toss"
    data = info(day)
    reference = cricket.match_record("1", day, data)
    data["toss"]["winner"] = "England"
    data["outcome"] = {"winner": "England"}
    other = cricket.match_record("1", day, data)
    assert cricket.features(pd.DataFrame([reference])) == cricket.features(pd.DataFrame([other]))
    data["toss"]["uncontested"] = True
    assert cricket.match_record("1", day, data)["status"] == "uncontested"
    data["teams"][0] = "Afghanistan"
    assert cricket.match_record("1", day, data)["status"] == "coverage_exclusion"
    data["teams"][0] = "Zimbabwe Under-19"
    assert cricket.match_record("1", day, data)["status"] == "youth"
    with pytest.raises(ValueError, match="Unrecognized outcome"):
        cricket.match_record("1", day, info(day, {}))


@pytest.mark.tier1
def test_clock_seals_and_unavailable_result_never_open(tmp_path):
    registry = live.read_issuance(live.DEST, live.ISSUANCE_SHA)
    for before_close in (date(2026, 11, 21), date(2027, 3, 1)):
        result = live.evaluation(live.DEST, registry, before_close, tmp_path / "absent", tmp_path)
        assert all(r["status"] == "pending" for r in result["windows"].values())
        assert all(r["dml_vs_half"]["gain"] is None for r in result["windows"].values())
    assert live.periods(datetime(2026, 12, 2, tzinfo=UTC))["annual"][0] == "2027-01-01"
    with pytest.raises(ValueError, match="registry SHA"):
        live.read_issuance(live.DEST, "changed")
    with pytest.raises(ValueError, match="issuance digest"):
        from copy import deepcopy

        mutated = deepcopy(registry)
        mutated["windows"]["q1"]["sha256"] = "changed"
        live.evaluation(live.DEST, mutated, date(2027, 3, 2))


def score_fixture(tmp_path, archive, rows, complete=date(2027, 2, 28)):
    path = tmp_path / "roster.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return live.evaluation(
        live.DEST,
        live.read_issuance(live.DEST, live.ISSUANCE_SHA),
        date(2027, 3, 2),
        archive,
        path,
        complete,
    )


@pytest.mark.tier1
def test_score_only_closed_reconciled_quarter_using_design_ci(tmp_path):
    archive, rows = future_fixture(tmp_path)
    result = score_fixture(tmp_path, archive, rows)
    q = result["windows"]["q1"]
    assert q["status"] == "scored"
    assert q["benchmark"]["matches"] == 2 and q["benchmark"]["estimate"] == 0
    assert q["benchmark"]["series"]["clusters"] == 2
    assert q["dml_vs_half"]["gain_ci95"][0] < 0 < q["dml_vs_half"]["gain_ci95"][1]
    assert all(r["status"] == "pending" for k, r in result["windows"].items() if k != "q1")
    assert (
        score_fixture(tmp_path, archive, rows, complete=None)["windows"]["q1"]["status"]
        == "pending"
    )
    assert (
        score_fixture(tmp_path, archive, rows, complete=date(2027, 2, 27))["windows"]["q1"][
            "status"
        ]
        == "pending"
    )
    with pytest.raises(ValueError, match="beyond"):
        score_fixture(tmp_path, archive, rows, complete=date(2027, 4, 1))


@pytest.mark.tier1
def test_bounds_for_missing_match_and_supplement_only_observed_post_toss_results(tmp_path):
    archive, rows = future_fixture(tmp_path)
    missing = dict(
        rows[-1], fixture_id="icc:D", match_id="icc:D", date="2026-12-11", status="tossed"
    )
    q = score_fixture(tmp_path, archive, rows + [missing])["windows"]["q1"]
    assert q["status"] == "pending" and q["scores"][0]["estimate"] is None
    assert q["coverage"]["roster_bounds_pp"] == pytest.approx([-100 / 3, 100 / 3])
    supplemental = dict(
        missing, status="tossed_no_result", season="2026/27", event="Tour", toss_winner="England"
    )
    q = score_fixture(tmp_path, archive, rows + [supplemental])["windows"]["q1"]
    assert q["status"] == "scored" and q["benchmark"]["matches"] == 3
    assert q["coverage"]["supplemented_official_ids"] == ["icc:D"]
    with pytest.raises(ValueError, match="Official post-toss"):
        score_fixture(tmp_path, archive, rows + [dict(supplemental, event="")])
    with pytest.raises(ValueError, match="Unknown toss winner"):
        score_fixture(tmp_path, archive, rows + [dict(supplemental, toss_winner="Pakistan")])
    official = dict(supplemental, status="tossed_official_result", final_result="England")
    q = score_fixture(tmp_path, archive, rows + [official])["windows"]["q1"]
    assert q["status"] == "scored" and q["benchmark"]["estimate"] == pytest.approx(100 / 3)
    with pytest.raises(ValueError, match="final winner"):
        score_fixture(tmp_path, archive, rows + [dict(official, final_result="")])
    with pytest.raises(ValueError, match="Unknown match winner"):
        score_fixture(tmp_path, archive, rows + [dict(official, final_result="Pakistan")])
    disguised_duplicate = dict(supplemental, date=rows[0]["date"])
    assert (
        score_fixture(tmp_path, archive, rows + [disguised_duplicate])["windows"]["q1"]["status"]
        == "pending"
    )


@pytest.mark.tier1
def test_independent_crosswalk_is_validated_before_filtering(tmp_path):
    archive, rows = future_fixture(tmp_path)
    for change in (
        {"date": ""},
        {"date": "20269998"},
        {"date": "20261203"},
        {"match_id": ""},
        {"source": "https://"},
        {"status": "lost"},
    ):
        with pytest.raises(ValueError):
            score_fixture(tmp_path, archive, [*rows[:2], dict(rows[-1], **change)])
    with pytest.raises(ValueError, match="unique"):
        score_fixture(tmp_path, archive, [*rows, rows[0]])
    for change in ({"team1": "Scotland"}, {"status": "no_toss"}, {"gender": "male"}):
        q = score_fixture(tmp_path, archive, [dict(rows[0], **change), *rows[1:]])["windows"]["q1"]
        assert q["status"] == "pending" and q["coverage"]["mismatched_ids"] == ["13"]
    q = score_fixture(tmp_path, archive, rows[1:])["windows"]["q1"]
    assert q["coverage"]["absent_from_fixture_ledger"] == ["13"]


@pytest.mark.tier2
@pytest.mark.skipif(
    os.getenv("DML_TS_EXTERNAL_REPLAY") != "1", reason="requires pinned Cricsheet source ZIP"
)
def test_full_archive_extract_has_archived_digest():
    cache = Path(os.environ.get("DML_TS_BENCHMARK_CACHE", Path.home() / ".cache/dml-ts/external"))
    path = live.fetch(
        cricket.URL, cache, f"cricsheet-t20s-{cricket.SOURCE_SHA}.zip", cricket.SOURCE_SHA
    )
    live.verify_extract(path)


@pytest.mark.tier1
def test_archived_world_cup_toss_without_a_ball_is_an_eligible_public_result():
    """ICC fixture 268112 reports a post-toss washout absent from the pinned archive."""
    day = "2026-02-21"
    world_cup = live.history()
    world_cup = world_cup.loc[world_cup.series == "male | 2025/26 | ICC Men's T20 World Cup"]
    assert len(world_cup) == 49
    assert world_cup.loc[
        (world_cup.date == day)
        & (world_cup.focal == "New Zealand")
        & (world_cup.opponent == "Pakistan")
    ].empty
    ledger = pd.DataFrame(
        [
            {
                "date": day,
                "fixture_id": "icc:268112",
                "match_id": "icc:268112",
                "gender": "male",
                "team1": "New Zealand",
                "team2": "Pakistan",
                "status": "tossed_no_result",
                "event": "ICC Men's T20 World Cup",
                "season": "2025/26",
                "toss_winner": "Pakistan",
                "source": "https://www.icc-cricket.com/tournaments/mens-t20-world-cup-2026/news/live-pakistan-take-on-new-zealand-in-super-eight-opener",
            }
        ]
    )
    live.validate_roster(ledger)
    coverage, with_rainout = live.reconcile(world_cup, ledger)
    assert not coverage["complete"]  # The one-fixture ledger cannot attest the tournament.
    assert coverage["supplemented_official_ids"] == ["icc:268112"]
    bench = cricket.benchmark(world_cup)
    supplement = cricket.benchmark(with_rainout)
    assert supplement["matches"] == bench["matches"] + 1
    assert supplement["estimate"] == pytest.approx(bench["estimate"] * 49 / 50)
