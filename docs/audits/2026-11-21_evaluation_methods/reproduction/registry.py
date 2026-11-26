"""Compare fresh registry stdout and generated artifacts with the historical replay."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = Path("docs/audits/2026-11-19_time_series_dml")
SUITES = {
    "replications": "italy water turnout wolfers beer dahl drink-less mybp sicily ramey",
    "questions": "retail-timing electricity-shifting wage-employment daycare-fines reminder-habituation",
    "forward": "reminder-cohorts gdp-vintages pennsylvania-bonus canvassing-elections progresa-schooling fertilizer-timing microcredit-sites ban-the-box retail-recommendations bracero-wages teacher-pay weatherization",
    "rolling": "rolling-reminders rolling-calibration rolling-bootstrap rolling-controls jtpa-holdout",
    "prospective": "shrinkage-replay cricket-toss gss-forms",
    "method-validation": "classical-econometrics classical-impulse paired-time-splits bangalore-pricing london-tariffs",
}


def parse(log):
    """Keep stdout from the registered subprocess, including arrays and multi-line tables."""
    pieces = re.split(r"^\[([\w-]+)\] (PASS|FAIL)\s*$", log, flags=re.M)
    out = {}
    for name, state, body in zip(pieces[1::3], pieces[2::3], pieces[3::3]):
        body = re.sub(r"^\d+/\d+ studies passed\s*$", "", body, flags=re.M).strip()
        if state == "FAIL":
            out[name] = {"state": state, "error": body.splitlines()[-1]}
            continue
        value = body
        for start in re.finditer(r"^[\[{]", body, flags=re.M):
            try:
                value, _ = json.JSONDecoder().raw_decode(body[start.start() :])
            except ValueError:
                continue
            break
        out[name] = {"state": state, "value": value}
    return out


def flatten(data, path=()):
    if isinstance(data, dict):
        for key, value in data.items():
            yield from flatten(value, (*path, key))
    elif isinstance(data, list):
        for i, value in enumerate(data):
            yield from flatten(value, (*path, str(i)))
    else:
        yield path, data


def compare(expected, actual):
    before, after = dict(flatten(expected)), dict(flatten(actual))
    delta = []
    max_abs = 0.0
    n_numeric = 0
    for key in sorted(before.keys() | after.keys()):
        if key not in before or key not in after:
            delta.append({"path": list(key), "kind": "missing field"})
            continue
        old, new = before[key], after[key]
        if (
            not isinstance(old, bool)
            and isinstance(old, (float, int))
            and not isinstance(new, bool)
            and isinstance(new, (float, int))
        ):
            n_numeric += 1
            max_abs = max(max_abs, abs(old - new))
            if math.isclose(old, new, rel_tol=1e-9, abs_tol=1e-9):
                continue
        elif old == new:
            continue
        delta.append({"path": list(key), "old": old, "new": new})
    return n_numeric, max_abs, delta


def payload(path):
    return gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()


def file_digest(path):
    return hashlib.sha256(payload(path)).hexdigest()


def write_gzip(path, obj):
    path.write_bytes(
        gzip.compress(
            json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(),
            mtime=0,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--baseline", type=Path, default=HERE / "historical-stdout.json.gz")
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    old_outputs = json.loads(payload(args.baseline))
    run, failures = {}, {}
    for path in args.logs:
        for name, record in parse(path.read_text()).items():
            if record["state"] == "FAIL":
                failures.setdefault(name, []).append(record["error"])
            run[name] = record
    args.output.mkdir(exist_ok=True, parents=True)
    rows, diff = [], {}
    source = HERE.parents[3] / OLD
    for suite, studies in SUITES.items():
        for study in studies.split():
            record = run.get(study, {"state": "pending"})
            row = {
                "suite": suite,
                "study": study,
                "entrypoint": record["state"],
                "comparison": "pending",
                "numeric_fields": "",
                "max_abs_diff": "",
                "discrepancies": "",
            }
            if record["state"] == "PASS":
                expected = old_outputs[study]["value"]
                actual = record["value"]
                if study == "paired-time-splits":
                    # CPU allocation and wall-clock duration are execution metadata.
                    expected = {
                        k: v for k, v in expected.items() if k not in {"workers", "seconds"}
                    }
                    actual = {k: v for k, v in actual.items() if k not in {"workers", "seconds"}}
                n, m, ds = compare(expected, actual)
                row.update(
                    numeric_fields=n,
                    max_abs_diff=m,
                    discrepancies=len(ds),
                    comparison="different"
                    if ds
                    else "identical"
                    if expected == actual
                    else "numerical tolerance",
                )
                if ds:
                    diff[study] = ds
                    if study == "rolling-bootstrap":
                        # The historic stdout predates the across-origin fields in the checked-in table.
                        typed = []
                        with (source / "rolling/bootstrap-calibration.csv").open(newline="") as fp:
                            for archived in csv.DictReader(fp):
                                typed.append(
                                    {
                                        k: v
                                        if k in ("scenario", "origin")
                                        else (float(v) if v else None)
                                        for k, v in archived.items()
                                    }
                                )
                        if not compare(typed, actual)[2]:
                            row["comparison"] = "matches updated archived table"
            rows.append(row)
    with (args.output / "registry.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    write_gzip(args.output / "fresh-stdout.json.gz", run)
    # The checkout includes research drivers that were uncommitted at launch.
    # Pin their exact contents, as well as the local estimator and dependency lock.
    code = sorted(
        p
        for p in args.snapshot.rglob("*")
        if p.is_file() and (p.suffix in {".py", ".R"} or p.name in {"pyproject.toml", "uv.lock"})
    )
    write_gzip(
        args.output / "execution-code-sha256.json.gz",
        {
            str(p.relative_to(args.snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in code
        },
    )
    (args.output / "registry-differences.json").write_text(
        json.dumps({"fields": diff, "retry_errors": failures}, indent=2) + "\n"
    )
    copied = args.snapshot / OLD
    files = sorted(
        [
            *source.glob("prospective/*-frozen.json"),
            *source.glob("prospective/*-issuance.json"),
            *source.glob("rolling/jtpa*-frozen.json"),
            *source.glob("rolling/jtpa-issuance.json"),
            *source.glob("classical/impulse*csv*"),
            *source.glob("paired/*csv*"),
            source / "paired/folds.json",
            *source.glob("field/london/*csv*"),
            source / "field/london/run.json",
        ]
    )
    check = []
    for file in files:
        rel = file.relative_to(source)
        fresh = (copied / rel).stat().st_mtime_ns > (
            args.snapshot.parent / "launch.json"
        ).stat().st_mtime_ns
        mode = (
            "preserved original"
            if rel.parts[0] in {"rolling", "prospective"}
            else ("fresh output" if fresh else "awaiting regeneration")
        )
        check.append(
            {
                "file": str(rel),
                "role": mode,
                "archived": file_digest(file),
                "snapshot": file_digest(copied / rel),
            }
        )
    (args.output / "file-comparison.json").write_text(json.dumps(check, indent=2) + "\n")
    print("Fresh passes", sum(r["entrypoint"] == "PASS" for r in rows), "of", len(rows))
    print("Historical transcript differences", list(diff))
    print("Differing artifacts", [i["file"] for i in check if i["archived"] != i["snapshot"]])

    if args.require_complete:
        incomplete = [
            r["study"] for r in rows if r["entrypoint"] != "PASS" or r["comparison"] == "different"
        ]
        inconsistent = [
            i["file"]
            for i in check
            if i["role"] == "awaiting regeneration" or i["archived"] != i["snapshot"]
        ]
        if incomplete or inconsistent:
            raise SystemExit(f"Unresolved studies {incomplete}; artifacts {inconsistent}")


if __name__ == "__main__":
    main()
