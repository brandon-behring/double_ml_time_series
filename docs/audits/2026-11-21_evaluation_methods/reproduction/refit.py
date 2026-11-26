"""Compare complete, freshly emitted 401(k) bundles with the earlier seeded fit."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parents[1] / "2026-11-19_time_series_dml/401k-draws.csv.gz"
METHODS = {"RLasso", "Trees", "Forest", "Boosting", "Nnet", "Ensemble", "best"}
PANELS = ("plinear", "interactive", "LATE")


def key(row):
    return (row["panel"], int(row["folds"]), int(row["partition"]), row["method"])


def publish(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        f.write(data)
        f.flush()
        os.fchmod(f.fileno(), 0o644)
    os.replace(f.name, path)


def final_table(current, archive, output):
    """Rebuild all 42 published table cells from this run's 100 fitted draws."""
    table = io.StringIO(newline="")
    fields = (
        "panel",
        "folds",
        "method",
        "partitions",
        "estimate",
        "median_se",
        "adjusted_se",
        "published_estimate",
        "published_median_se",
        "published_adjusted_se",
        "previous_estimate",
        "previous_median_se",
        "previous_adjusted_se",
    )
    writer = csv.DictWriter(table, fieldnames=fields)
    writer.writeheader()
    with archive.with_name("401k-summary.csv").open(newline="") as f:
        historical = list(csv.DictReader(f))
    if len(historical) != 42:
        raise ValueError("Expected 42 historical cell summaries")
    for old in historical:
        panel, folds, method = old["panel"], int(old["folds"]), old["method"]
        rows = [current[panel, folds, rep, method] for rep in range(1, 101)]
        theta = median(float(row["estimate"]) for row in rows)
        variances = [float(row["se"]) ** 2 + (float(row["estimate"]) - theta) ** 2 for row in rows]
        values = {
            "estimate": theta,
            "median_se": median(float(row["se"]) for row in rows),
            "adjusted_se": math.sqrt(median(variances)),
        }
        writer.writerow(
            {
                "panel": panel,
                "folds": folds,
                "method": method,
                "partitions": len(rows),
                **values,
                **{f"published_{key}": old[f"published_{key}"] for key in values},
                **{f"previous_{key}": old[key] for key in values},
            }
        )
        if not all(
            math.isclose(val, float(old[k]), rel_tol=1e-9, abs_tol=1e-7)
            for k, val in values.items()
        ):
            raise ValueError(f"Historical table aggregate mismatch for {panel}, {folds}, {method}")
    publish(output / "401k-final-table.csv", table.getvalue().encode())


def resource_sample(run, status):
    """Append the dispatcher's actual admission accounting to the local run log."""
    p = run / "resource-samples.csv"
    usage = status["resources"]
    row = {
        "utc": status["updated_utc"],
        "complete_bundles": status["valid_joint_jobs"],
        "active": len(status["active"]),
        "launched_in_sample": status["launches_this_sample"],
        "host_cpu_target": status["target_busy_cores"],
        **{k: v for k, v in usage.items() if k != "parallel_growth_reserved_gib"},
    }
    empty = not p.exists() or not p.stat().st_size
    with p.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row)
        if empty:
            writer.writeheader()
        writer.writerow(row)


def compare_run(args):
    args.output.mkdir(exist_ok=True, parents=True)
    with gzip.open(args.archive, "rt", newline="") as f:
        history = {key(row): row for row in csv.DictReader(f) if row["run"] == "joint"}
    if len(history) != 4200:
        raise ValueError("Historical comparison must contain seven methods on 600 distinct jobs")
    status = json.loads((args.run / "dispatcher/status.json").read_text())
    current = {}
    counts = defaultdict(int)
    for panel in PANELS:
        for folds in (2, 5):
            for rep in range(1, 101):
                fn = args.run / f"checkpoints/all-{panel}-k{folds}-r{rep:03d}.csv"
                if not fn.is_file():
                    continue
                with fn.open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                if len(rows) != 7 or {key(r) for r in rows} != {
                    (panel, folds, rep, m) for m in METHODS
                }:
                    raise ValueError(f"Malformed bundle {fn.name}")
                for row in rows:
                    theta, se = float(row["estimate"]), float(row["se"])
                    if not math.isfinite(theta) or not math.isfinite(se) or se < 0:
                        raise ValueError(f"Invalid numerical result {fn.name}")
                    current[key(row)] = row
                counts[f"{panel}-k{folds}"] += 1
    receipts = []
    exact = within = 0
    max_theta = max_se = 0.0
    for k, row in sorted(current.items()):
        historical = history[k]
        th, se = float(row["estimate"]), float(row["se"])
        oth, ose = float(historical["estimate"]), float(historical["se"])
        max_theta, max_se = max(max_theta, abs(th - oth)), max(max_se, abs(se - ose))
        exact += int(th == oth and se == ose)
        within += int(
            all(math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-7) for x, y in ((th, oth), (se, ose)))
        )
        receipts.append([*k, row["estimate"], row["se"], historical["estimate"], historical["se"]])
    table = io.StringIO(newline="")
    writer = csv.writer(table)
    writer.writerow(
        [
            "panel",
            "folds",
            "partition",
            "method",
            "estimate",
            "se",
            "previous_estimate",
            "previous_se",
        ]
    )
    writer.writerows(receipts)
    publish(
        args.output / "401k-comparison.csv.gz", gzip.compress(table.getvalue().encode(), mtime=0)
    )
    summary = {
        "as_of_utc": status["updated_utc"],
        "receipt_basis": "final seven-method R bundles only",
        "panels": counts,
        "completed_bundles": len(receipts) // 7,
        "total_bundles": 600,
        "finite_rows": len(receipts),
        "exact_estimate_and_se": exact,
        "within_numerical_tolerance": within,
        "max_abs_estimate_difference": max_theta if receipts else None,
        "max_abs_se_difference": max_se if receipts else None,
        "active_at_last_dispatch_poll": len(status["active"]),
        "failure_counts": status["failure_counts"],
        "retry_exhausted": status["retry_exhausted"],
        "dispatcher_complete": status["complete"],
        "admission_resources": status["resources"],
        "high_water": status["high_water"],
        "previous_archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
    }
    publish(
        args.output / "401k-status.json",
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(
        f"{summary['completed_bundles']}/600; {exact}/{len(receipts)} rows match both previous numbers exactly"
    )
    display = [
        "# 401(k) fresh refit progress",
        "",
        f"At **{status['updated_utc']}**, **{len(receipts) // 7} of 600** freshly fitted partition bundles were complete. "
        f"These bundles contain **{len(receipts)}** paired results. "
        f"The estimate and standard error matched the earlier run exactly in **{exact}** cases, "
        f"and both matched to numerical tolerance in **{within}** cases.",
        "",
        "| Target | Folds | Completed partitions | Requested |",
        "|---|---:|---:|---:|",
    ]
    for panel in PANELS:
        for fold in (2, 5):
            display.append(f"| {panel} | {fold} | {counts[f'{panel}-k{fold}']} | 100 |")
    display.extend(
        [
            "",
            f"At the latest dispatcher sample, {len(status['active'])} single-thread R jobs were active. "
            f"The maximum number of simultaneous fresh jobs observed so far was {status['high_water']['dispatcher_workers']}. "
            f"There were {sum(status['failure_counts'].values())} recorded failures. "
            "Counts include final seven-method bundles only. "
            "[Joined numerical rows](401k-comparison.csv.gz) · [Execution receipt](401k-status.json) · "
            "[Method and full registry replay](../historical-reproduction.md)",
            "",
        ]
    )
    if len(receipts) == 4200 and within != 4200:
        publish(args.output / "401k-progress.md", "\n".join(display).encode())
        raise SystemExit("One or more matched results differ beyond numerical tolerance")
    if len(receipts) == 4200:
        final_table(current, args.archive, args.output)
        display.extend(
            [
                "The 42 complete table cells reproduce the historical median estimate, median SE and split-adjusted SE. "
                "Each cell also retains the corresponding printed paper values. [Fresh 42-cell table](401k-final-table.csv)",
                "",
            ]
        )
    publish(args.output / "401k-progress.md", "\n".join(display).encode())

    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run", type=Path, help="Fresh dispatcher root containing checkpoints and dispatcher/"
    )
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument(
        "--follow", action="store_true", help="Update receipts until the dispatcher finishes"
    )
    args = parser.parse_args()
    while True:
        status = compare_run(args)
        if args.follow:
            resource_sample(args.run, status)
        if not args.follow or status["complete"]:
            return
        if status["retry_exhausted"] or not Path(f"/proc/{status['pid']}").exists():
            raise SystemExit(
                "Dispatcher stopped or a partition exhausted its retries; inspect the status receipt"
            )
        time.sleep(60)


if __name__ == "__main__":
    main()
