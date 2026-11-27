"""Recount the hand-coded audit and optionally verify DOI and print-issue membership."""

import argparse
import csv
import json
import tempfile
import time
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

LEDGER = Path(__file__).resolve().parents[2] / "forward" / "journals-2018.csv"
JOURNALS = {
    "AER": ("0002-8282", "108"),
    "AEJ Applied": ("1945-7782", "10"),
    "QJE": ("0033-5533", "133"),
}
EXCLUSIONS = {
    "front matter",
    "comment or reply",
    "editorial or lecture",
    "theory",
    "quantitative calibration",
    "methods",
}
DESIGNS = {
    "estimated structural model",
    "observational",
    "measurement",
    "randomized",
    "mixed randomized and observational",
    "quasi-experimental",
    "predictive",
}
TARGETS = {
    "historical",
    "follow-up of original exposure",
    "model counterfactual",
    "external applicability",
    "outcome prediction",
    "effect projection",
}


def read_ledger(path=LEDGER):
    with Path(path).open(encoding="utf-8", newline="") as infile:
        return list(csv.DictReader(infile))


def validate(rows):
    """Check codebook, DOI uniqueness, required evidence and eligibility independent of access."""
    seen = set()
    for row in rows:
        doi = row["doi"]
        if not doi or doi != doi.lower() or doi in seen:
            raise ValueError(f"Missing, duplicate or noncanonical DOI: {doi!r}")
        seen.add(doi)
        if row["journal"] not in JOURNALS or row["volume"] != JOURNALS[row["journal"]][1]:
            raise ValueError(f"{doi}: unexpected journal or print volume")
        for key in ("issue", "title", "headline_claim_and_design", "screen", "data_access"):
            if not row[key].strip():
                raise ValueError(f"{doi}: missing {key}")
        for key in ("issue_url", "claim_source"):
            if not row[key].startswith("https://"):
                raise ValueError(f"{doi}: missing {key}")
        if row["screen"] not in EXCLUSIONS | DESIGNS:
            raise ValueError(f"{doi}: unknown study design")
        eligible = row["screen"] in DESIGNS
        if row["eligible"] != ("yes" if eligible else "no"):
            raise ValueError(f"{doi}: eligibility conflicts with design")
        if eligible:
            if row["temporal_target"] not in TARGETS:
                raise ValueError(f"{doi}: invalid temporal target")
            for key in (
                "published_finding",
                "reported_check",
                "later_comparison",
                "interpretation",
                "access_notes",
            ):
                if not row[key].strip():
                    raise ValueError(f"{doi}: missing {key}")
            if not row["access_evidence"].startswith("https://"):
                raise ValueError(f"{doi}: missing access evidence")
            if row["data_access"] != "acquisition route unverified" and not row[
                "archive_url"
            ].startswith("https://"):
                raise ValueError(f"{doi}: missing archive URL")
        elif row["temporal_target"] != "outside empirical denominator" or row["published_finding"]:
            raise ValueError(f"{doi}: excluded item included in results")


def summary(rows):
    validate(rows)
    empirical = [r for r in rows if r["eligible"] == "yes"]
    by_journal = {}
    for journal in JOURNALS:
        group = [r for r in rows if r["journal"] == journal]
        elig = [r for r in group if r["eligible"] == "yes"]
        by_journal[journal] = {
            "issued_items": len(group),
            "issues": len({r["issue"] for r in group}),
            "front_matter": sum(r["screen"] == "front matter" for r in group),
            "empirical": len(elig),
            "targets": Counter(r["temporal_target"] for r in elig),
        }
    return {
        "issued_items": len(rows),
        "non_front_matter": sum(r["screen"] != "front matter" for r in rows),
        "empirical": len(empirical),
        "by_journal": by_journal,
        "screen": Counter(r["screen"] for r in rows),
        "temporal_target": Counter(r["temporal_target"] for r in empirical),
        "data_access": Counter(r["data_access"] for r in empirical),
    }


def fetch_volume(journal, cache_dir=None):
    issn, volume = JOURNALS[journal]
    query = urlencode(
        {
            "filter": "from-print-pub-date:2018-01-01,until-print-pub-date:2018-12-31",
            "select": "DOI,volume,issue",
            "rows": 250,
        }
    )
    req = Request(
        f"https://api.crossref.org/journals/{issn}/works?{query}",
        headers={"User-Agent": "dml-ts-replication/1.0"},
    )
    target = Path(cache_dir) / f"crossref-print-2018-{issn}.json" if cache_dir is not None else None
    cached = target is not None and target.exists()
    if cached:
        raw = target.read_bytes()
    else:
        for attempt in range(4):
            try:
                with urlopen(req, timeout=60) as response:
                    raw = response.read()
                break
            except HTTPError as exc:
                if exc.code not in (429, 502, 503, 504) or attempt == 3:
                    raise
                wait = exc.headers.get("Retry-After", "")
                time.sleep(int(wait) if wait.isdigit() else 2 ** (attempt + 1))
    message = json.loads(raw)["message"]
    if len(message["items"]) != message["total-results"]:
        raise ValueError(f"{journal}: incomplete Crossref response")
    if target is not None and not cached:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
            temporary.write(raw)
        Path(temporary.name).replace(target)
    return [
        (r["DOI"].lower(), journal, r["volume"], r["issue"])
        for r in message["items"]
        if r["volume"] == volume
    ]


def compare_rosters(rows, published):
    """Compare multisets so that dropped, repeated and misassigned items are visible."""
    expected = Counter((r["doi"], r["journal"], r["volume"], r["issue"]) for r in rows)
    observed = Counter(published)
    if expected != observed:
        raise ValueError(
            f"Print rosters differ. Ledger only: {list((expected - observed).elements())}; "
            f"metadata only: {list((observed - expected).elements())}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument(
        "--verify-roster",
        action="store_true",
        help="retrieve the three publisher-deposited Crossref print rosters",
    )
    parser.add_argument(
        "--cache-dir", type=Path, help="reuse the three Crossref responses for --verify-roster"
    )
    args = parser.parse_args()
    rows = read_ledger(args.ledger)
    result = summary(rows)
    if args.verify_roster:
        # Crossref enforces a low burst limit on this anonymous endpoint.
        volumes = [fetch_volume(journal, args.cache_dir) for journal in JOURNALS]
        compare_rosters(rows, [record for volume in volumes for record in volume])
        result["verified_print_metadata"] = True
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
