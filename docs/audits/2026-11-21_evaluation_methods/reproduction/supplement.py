"""Rerun source verifications and materialize historical rolling benchmark outputs."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from registry import OLD, compare, file_digest, parse, payload

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--italy-python", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    snap = args.snapshot.resolve()
    cache = args.cache.resolve()
    out = snap.parent / "supplement"
    roll = out / "rolling"
    roll.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(exist_ok=True, parents=True)
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(snap),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        DML_N_JOBS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    steps = {
        "italy-authors": ("sequential/italy.py", ["--author-python", str(args.italy_python)]),
        "gdp-survey-verification": ("forward/gdp.py", ["--verify-survey"]),
        "cricket-original-roster": ("forward/prospective_cricket.py", ["--stage", "verify-source"]),
        "cricket-precision": ("forward/prospective_cricket.py", ["--stage", "precision"]),
        "rolling-reminder-origins": ("forward/reminder_rolling.py", ["--output-dir", str(roll)]),
        "rolling-reminder-calibration": (
            "forward/rolling_calibration.py",
            ["--workers", "2", "--output", str(roll)],
        ),
        "rolling-reminder-bootstrap": (
            "forward/reminder_bootstrap_calibration.py",
            [
                "--repetitions",
                "500",
                "--draws",
                "999",
                "--workers",
                "2",
                "--output",
                str(roll / "bootstrap-calibration.csv"),
            ],
        ),
    }

    def run(name):
        script, cli = steps[name]
        cmd = [
            sys.executable,
            str(snap / OLD / "external" / script),
            "--cache-dir",
            str(cache),
            *cli,
        ]
        log = out / f"{name}.log"
        with log.open("wb") as f:
            rc = subprocess.run(
                cmd, env=env, cwd=snap, stdout=f, stderr=subprocess.STDOUT, check=False
            ).returncode
        return name, {
            "rc": rc,
            "stdout_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
            "error": log.read_text().strip().splitlines()[-1] if rc else None,
        }

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        result = dict(pool.map(run, steps))
    archived = snap / OLD / "rolling"
    data = [p for p in roll.rglob("*") if p.is_file() and p.suffix != ".md"]
    files = []
    for file in sorted(data):
        rel = file.relative_to(roll)
        ref = archived / rel
        files.append(
            {
                "file": str(rel),
                "regenerated_sha256": file_digest(file),
                "archived_sha256": file_digest(ref) if ref.exists() else None,
            }
        )
    historical = json.loads(payload(HERE / "historical-stdout.json.gz"))
    comparisons = {}
    for name, reference in [
        ("gdp-survey-verification", "gdp-vintages"),
        ("italy-authors", "italy"),
        ("cricket-precision", "cricket-precision"),
    ]:
        if result[name]["rc"]:
            continue
        current = parse("[sample] PASS\n" + (out / f"{name}.log").read_text())["sample"]["value"]
        expected = (
            historical[reference]["value"]
            if reference in historical
            else json.loads((snap / OLD / "prospective/cricket-precision.json").read_text())
        )
        n, delta, fields = compare(expected, current)
        comparisons[name] = {
            "numeric_fields": n,
            "max_abs_difference": delta,
            "field_differences": fields,
        }
    receipt = {"steps": result, "rolling_files": files, "comparisons": comparisons}
    (args.output / "supplement.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return int(
        any(r["rc"] for r in result.values())
        or any(r["field_differences"] for r in comparisons.values())
        or any(f["archived_sha256"] != f["regenerated_sha256"] for f in files)
    )


if __name__ == "__main__":
    raise SystemExit(main())
