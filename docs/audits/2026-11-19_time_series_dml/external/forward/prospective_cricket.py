"""Issue, reproduce, and later score the frozen one-year cricket coin-toss forecasts."""

import argparse
import hashlib
import json
import sys
from calendar import monthrange
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import pandas as pd

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[5])]
from benchmark_io import fetch  # noqa: E402
from forward import cricket_data as cricket  # noqa: E402

from dml_ts.validation.effect_forecast import freeze, ranking_precision, score  # noqa: E402

DEST = HERE.parents[2] / "prospective"
HISTORY = DEST / "cricket-history.csv.gz"
HISTORY_SHA = "6ed9c19769f0a8430fbd3fd5d6762990e5ace58b66220809b4f811f6e20d876c"
ISSUANCE_SHA = "079a1bb1c38fcb31f4d2b50e2e381a99485b4702913d2abf8c051d6e30f22885"
TARGET = (
    "ITT of winning the right to choose innings order on the prespecified focal team's win credit"
)
UNIT = "percentage points of win credit"
LEDGER_STATUSES = {
    "tossed",
    "tossed_no_result",
    "tossed_official_result",
    "no_toss",
    "cancelled_before_toss",
    "uncontested",
    "coverage_exclusion",
    "youth",
    "unresolved",
}


def add_months(d, n):
    index = d.month - 1 + n
    year, month = d.year + index // 12, 1 + index % 12
    return date(year, month, min(d.day, monthrange(year, month)[1]))


def periods(issued):
    """Move a missed opening to the first untouched full local calendar month."""
    opening = date(2026, 12, 1)
    # Allow all time zones and the preceding pre-match toss.
    while issued >= datetime(opening.year, opening.month, 1, tzinfo=UTC) - timedelta(days=1):
        opening = add_months(opening, 1)
    windows = {"annual": (opening, add_months(opening, 12))}
    windows.update(
        {
            f"q{i + 1}": (add_months(opening, 3 * i), add_months(opening, 3 * i + 3))
            for i in range(4)
        }
    )
    return {name: (start.isoformat(), end.isoformat()) for name, (start, end) in windows.items()}


def history():
    if cricket.digest(HISTORY) != HISTORY_SHA:
        raise ValueError("Cricket historical extract SHA-256 mismatch")
    return pd.read_csv(HISTORY, dtype={"match_id": str, "date": str})


def issue(output, issued):
    output.mkdir(parents=True, exist_ok=True)
    timestamp = issued.astimezone(UTC).isoformat(timespec="seconds")
    origin = timestamp[:10]
    hist = history()
    if hist.date.max() >= origin:
        raise ValueError("Historical match on or after issuance")
    f = cricket.forecasts(hist)
    rows = [{"origin": origin, "method": name, "value": f[name]} for name in cricket.METHODS]
    entries = {}
    for name, (start, end) in periods(issued).items():
        protocol = {
            "target": TARGET,
            "unit": UNIT,
            "horizon": f"match-local start dates in [{start}, {end})",
            "intervention": "fair toss awards choice of innings order to focal team",
            "comparator": "opponent wins right to choose",
            "issued_utc": timestamp,
            "origins": [origin],
            "methods": cricket.METHODS,
            "comparators": ["zero", "historical", "half_history"],
            "window": name,
            "start": start,
            "end_exclusive": end,
            "role": "primary" if name == "annual" else "descriptive",
            "focal_team": "alphabetically first of the two archived team names",
            "eligibility": "senior men's and women's international T20 matches with a contested toss; exclude Afghanistan coverage, youth teams and pre-toss cancellations",
            "credit": "one for final win including eliminator or bowl-out, half for unresolved tie/draw or post-toss no-result, zero for loss",
            "assignment_probability": 0.5,
            "history_sha256": HISTORY_SHA,
            "archive_sha256": cricket.SOURCE_SHA,
            "fit": "five whole-event/season folds; arm-specific ridge alpha 1 with intercept; gender-specific focal and opponent, venue, gender, month, numeric year; predictions clipped to [0,1]; known-p AIPW",
            "score": "mean centered randomized win-credit contrast; series-clustered t interval; fixed 28-day calendar-block sensitivity; later eligible matches receive equal weights",
            "ranking": "DML against half-history using paired squared error and transformed two-sided 95% target interval; quarterly estimates descriptive",
            "missing": "require reconciliation with independent full fixture roster before final score; report provisional covered-sample estimate and worst-case bounds when matches unresolved",
        }
        path = output / f"cricket-{name}-frozen.json"
        h = freeze(path, protocol, rows)
        entries[name] = {"file": path.name, "sha256": h, "start": start, "end_exclusive": end}
    return {
        "issued_utc": timestamp,
        "source": cricket.URL,
        "source_sha256": cricket.SOURCE_SHA,
        "history_sha256": HISTORY_SHA,
        "historical_status": hist.status.value_counts().to_dict(),
        "first_match": hist.date.min(),
        "latest_match": hist.date.max(),
        "forecasts": f,
        "windows": entries,
    }


def verify_records(records, issuance):
    records_by_window = {}
    for label, entry in issuance["windows"].items():
        path = records / entry["file"]
        if cricket.digest(path) != entry["sha256"]:
            raise ValueError("Frozen cricket file differs from issuance digest")
        p = json.loads(path.read_bytes())["protocol"]
        if (p["window"], p["start"], p["end_exclusive"], p["issued_utc"]) != (
            label,
            entry["start"],
            entry["end_exclusive"],
            issuance["issued_utc"],
        ):
            raise ValueError("Cricket issuance window mismatch")
        records_by_window[label] = (path, entry["sha256"], p)
    return records_by_window


def validate_roster(roster):
    """Validate before date selection so undated or malformed fixtures cannot disappear."""
    required = {"fixture_id", "match_id", "date", "gender", "team1", "team2", "status", "source"}
    if not required.issubset(roster) or roster.empty:
        raise ValueError("Closing a window requires its independently sourced fixture roster")
    for col in ("fixture_id", "match_id"):
        ids = roster[col].fillna("")
        if ids.duplicated().any() or not ids.str.fullmatch(r"\S+").all():
            raise ValueError("Fixture IDs must be unique and present")
    if (
        not roster.gender.isin(("female", "male")).all()
        or not (
            roster.team1.fillna("").str.strip().ne("")
            & roster.team2.fillna("").str.strip().ne("")
            & roster.team1.ne(roster.team2)
        ).all()
    ):
        raise ValueError("Fixture crosswalk requires gender and two distinct teams")
    for value in roster.date.fillna(""):
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError("Expected canonical ISO fixture dates")
    urls = [urlsplit(str(url)) for url in roster.source.fillna("")]
    if not roster.status.isin(LEDGER_STATUSES).all() or not all(
        url.scheme == "https" and url.hostname for url in urls
    ):
        raise ValueError("Provide a fixture status and public source URL for every row")


def official_row(ident, row):
    """Code an independently sourced scorecard using the frozen archive outcome rules."""
    fields = ["event", "season", "toss_winner"]
    if any(pd.isna(row.get(key)) or not row[key].strip() for key in fields):
        raise ValueError("Official post-toss record requires teams, toss, gender, event and season")
    outcome = {"result": "no result"}
    if row.status == "tossed_official_result":
        final = row.get("final_result")
        if pd.isna(final) or not final.strip():
            raise ValueError("Official scorecard requires the final winner or unresolved tie/draw")
        outcome = {"result": final} if final in ("tie", "draw") else {"winner": final}
    result = cricket.match_record(
        ident,
        row.date,
        {
            "dates": [row.date],
            "team_type": "international",
            "match_type": "T20",
            "teams": [row.team1, row.team2],
            "gender": row.gender,
            "event": {"name": row.event},
            "season": row.season,
            "toss": {"winner": row.toss_winner},
            "outcome": outcome,
        },
    )
    if result["status"] != "eligible":
        raise ValueError("Official scorecard must meet randomized senior-T20 eligibility")
    return result


def reconcile(matches, roster):
    archive = matches.set_index("match_id")
    ledger = roster.set_index("match_id")
    missing = []
    conflicts = []
    supplemental = []
    for ident, r in ledger.iterrows():
        teams = sorted([r.team1, r.team2])
        if ident in archive.index:
            src = archive.loc[ident]
            if (src.date, src.gender, src.focal, src.opponent) != (r.date, r.gender, *teams):
                conflicts.append(ident)
                continue
        else:
            same = matches.loc[(matches.date == r.date) & (matches.gender == r.gender)]
            if ((same.focal == teams[0]) & (same.opponent == teams[1])).any():
                conflicts.append(ident)
                continue
        if r.status == "unresolved":
            missing.append(ident)
        elif r.status in ("tossed_no_result", "tossed_official_result"):
            supplemental_row = official_row(ident, r)
            if ident in archive.index:
                fields = [
                    "date",
                    "gender",
                    "focal",
                    "opponent",
                    "series",
                    "focal_toss",
                    "focal_credit",
                ]
                if any(archive.at[ident, k] != supplemental_row[k] for k in fields):
                    conflicts.append(ident)
            else:
                supplemental.append(supplemental_row)
        elif ident not in archive.index:
            if r.status == "tossed":
                missing.append(ident)
        else:
            expected = {"tossed": "eligible", "cancelled_before_toss": "no_toss"}.get(
                r.status, r.status
            )
            if src.status != expected:
                conflicts.append(ident)
    foreign = sorted(set(archive.index) - set(ledger.index))
    measured = (
        pd.concat([matches, pd.DataFrame(supplemental)], ignore_index=True)
        if supplemental
        else matches
    )
    eligible = measured[measured.status == "eligible"]
    out = {
        "fixture_rows": len(roster),
        "fixture_status": roster.status.value_counts().to_dict(),
        "archive_status": matches.status.value_counts().to_dict(),
        "unresolved_ids": missing,
        "mismatched_ids": conflicts,
        "absent_from_fixture_ledger": foreign,
        "eligible_on_file": int((matches.status == "eligible").sum()),
        "supplemented_official_ids": [r["match_id"] for r in supplemental],
        "complete": not (missing or conflicts or foreign),
    }
    # Extreme completions for all potentially eligible unknown outcomes conditional on this roster.
    if not foreign and not conflicts:
        known = eligible.loc[~eligible.match_id.isin(missing)]
        n = len(known)
        m = len(missing)
        total = cricket.score_contributions(known).sum()
        if n + m:
            out["roster_bounds_pp"] = [
                float((total - 100 * m) / (n + m)),
                float((total + 100 * m) / (n + m)),
            ]
    elif len(matches) or len(roster):
        out["roster_bounds_pp"] = [-100.0, 100.0]
    return out, measured


def evaluation(records, issuance, as_of, archive=None, roster_path=None, complete_through=None):
    """Verify seals before reading outcomes and never open unfinished windows."""
    files = verify_records(records, issuance)
    if complete_through is not None and complete_through > as_of:
        raise ValueError("Cannot attest fixture completeness beyond the evaluation date")
    output = {"evaluated_as_of_utc": as_of.isoformat(), "windows": {}}
    roster = None
    for label, (path, sha, p) in files.items():
        origin = p["origins"][0]
        result = {
            "role": p["role"],
            "period_start": p["start"],
            "period_end_exclusive": p["end_exclusive"],
        }
        # The extra UTC day lets the final local match date finish in every time zone.
        closed = as_of >= date.fromisoformat(p["end_exclusive"]) + timedelta(days=1)
        reason = "window has not closed"
        bench = None
        if closed and (archive is None or roster_path is None):
            reason = "awaiting result snapshot and reconciled fixture roster"
        elif closed:
            if roster is None:
                roster = pd.read_csv(roster_path, dtype=str)
                validate_roster(roster)
                output.update(
                    archive_sha256=cricket.digest(archive),
                    roster_sha256=cricket.digest(roster_path),
                    fixture_coverage_attested_through=(
                        complete_through.isoformat() if complete_through else None
                    ),
                )
            subset = roster.loc[(roster.date >= p["start"]) & (roster.date < p["end_exclusive"])]
            if subset.empty:
                reason = "no independent fixture roster for this window"
            else:
                matches = cricket.read_archive(archive, p["start"], p["end_exclusive"])
                coverage, measured = reconcile(matches, subset)
                last_day = date.fromisoformat(p["end_exclusive"]) - timedelta(days=1)
                attested = complete_through is not None and complete_through >= last_day
                coverage["all_fixtures_attested"] = attested
                coverage["complete"] &= attested
                result["coverage"] = coverage
                benchmark = cricket.benchmark(measured)
                if coverage["complete"] and benchmark:
                    bench = benchmark
                    reason = None
                else:
                    reason = (
                        "missing independent attestation of complete fixture coverage"
                        if not attested
                        else "incomplete source reconciliation or no randomized matches"
                    )
                    result["provisional_covered_sample"] = benchmark
        obs = {key: p[key] for key in ["target", "unit", "horizon", "intervention", "comparator"]}
        obs.update(origin=origin, estimate=bench["estimate"] if bench else None, reason=reason)
        obs["ci95"] = bench["series"]["ci95"] if bench else None
        result["scores"] = score(path, sha, [obs])
        dml = next(row for row in result["scores"] if row["method"] == "dml")
        result["dml_vs_half"] = dml["comparisons"]["half_history"]
        result.update(status="scored" if bench else "pending", reason=reason, benchmark=bench)
        output["windows"][label] = result
    return output


def verify_extract(raw):
    if cricket.digest(raw) != cricket.SOURCE_SHA:
        raise ValueError("Original source snapshot SHA-256 mismatch")
    frame = cricket.read_archive(raw, "1900-01-01", "2026-11-21")
    if hashlib.sha256(cricket.archive_extract(frame)).hexdigest() != HISTORY_SHA:
        raise ValueError("Reconstructed cricket extract differs")


def read_issuance(directory, expected_sha):
    path = directory / "cricket-issuance.json"
    if cricket.digest(path) != expected_sha:
        raise ValueError("Unexpected cricket issuance registry SHA-256")
    return json.loads(path.read_bytes())


def precision():
    f = read_issuance(DEST, ISSUANCE_SHA)["forecasts"]
    form = ranking_precision(f["dml"], f["half_history"])
    hist = history()
    bench = cricket.benchmark(hist.loc[hist.date.str.startswith("2025")])
    return {
        "dml_vs_half": form,
        "historical_precision_2025": bench,
        "two_sided_80pct_mde_pp_using_2025_series_se": 2.80158521811297 * bench["series"]["se"],
        "independent_946_matches_upper_sd_mde_pp": 2.80158521811297 * 100 / 946**0.5,
    }


def replay():
    saved = read_issuance(DEST, ISSUANCE_SHA)

    with TemporaryDirectory() as tmp:
        result = issue(Path(tmp), datetime.fromisoformat(saved["issued_utc"]))
        if result != saved:
            raise ValueError("Replayed cricket forecasts differ from the archived issuance")
    verify_records(DEST, saved)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--stage",
        choices=["replay", "verify-source", "issue", "score", "precision"],
        default="replay",
    )
    parser.add_argument("--output-dir", type=Path, default=DEST)
    parser.add_argument(
        "--archive",
        type=Path,
        help="Original source ZIP for verification, or later results ZIP for scoring",
    )
    parser.add_argument("--fixtures", type=Path, help="Independent completed-fixture ledger CSV")
    parser.add_argument(
        "--fixtures-complete-through",
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="Attest the independent ledger covers every in-scope fixture since opening through this inclusive local date, including cancellations",
    )
    parser.add_argument(
        "--issuance-sha256", default=ISSUANCE_SHA, help="Retained registry digest for scoring"
    )
    args = parser.parse_args()
    if args.stage == "verify-source":
        if not args.archive and not args.cache_dir:
            parser.error("verify-source requires --archive or --cache-dir")
        raw = args.archive or fetch(
            cricket.URL,
            args.cache_dir,
            f"cricsheet-t20s-{cricket.SOURCE_SHA}.zip",
            cricket.SOURCE_SHA,
        )
        verify_extract(raw)
        result = {"source_sha256": cricket.SOURCE_SHA, "matches_extract_sha256": HISTORY_SHA}
    elif args.stage == "issue":
        issuance_path = args.output_dir / "cricket-issuance.json"
        if issuance_path.exists():
            prev = json.loads(issuance_path.read_bytes())
            result = issue(args.output_dir, datetime.fromisoformat(prev["issued_utc"]))
            if result != prev:
                raise ValueError("Cricket issuance conflict")
        else:
            result = issue(args.output_dir, datetime.now(UTC))
            with issuance_path.open("x") as f:
                json.dump(result, f, indent=2, allow_nan=False)
                f.write("\n")
    elif args.stage == "score":
        if bool(args.archive) != bool(args.fixtures):
            parser.error("Supply both --archive and --fixtures for completed windows")
        saved = read_issuance(args.output_dir, args.issuance_sha256)
        result = evaluation(
            args.output_dir,
            saved,
            datetime.now(UTC).date(),
            args.archive,
            args.fixtures,
            args.fixtures_complete_through,
        )
    elif args.stage == "precision":
        result = precision()
    else:
        result = replay()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
