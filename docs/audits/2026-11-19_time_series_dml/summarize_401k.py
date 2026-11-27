"""Aggregate fresh 401(k) draws and update the audit's numerical section.

Usage: python summarize_401k.py CHECKPOINT_DIRECTORY
During a run this snapshots atomically completed R rows. Without an argument,
rebuild the tables from the published local 401k-draws.csv.gz artifact.
"""
from __future__ import annotations

import csv
import gzip
import io
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parent
METHODS = ("RLasso", "Trees", "Forest", "Boosting", "Nnet", "Ensemble", "best")
# Published Table 2A, 2B, 3. Each triple is estimate, median SE, split-adjusted SE.
TARGETS = {
    ("interactive", 2): [(6830,1282,1530),(7713,1208,1271),(7770,1276,1363),(7806,1159,1202),(7764,1328,1468),(7702,1149,1170),(7546,1360,1533)],
    ("interactive", 5): [(7170,1201,1398),(7993,1198,1236),(8105,1242,1299),(7713,1155,1177),(7788,1238,1293),(7839,1134,1148),(7753,1237,1294)],
    ("plinear", 2): [(7717,1346,1749),(8709,1363,1427),(9116,1302,1377),(8759,1339,1382),(8950,1335,1408),(9010,1309,1344),(9125,1304,1357)],
    ("plinear", 5): [(8187,1298,1558),(8871,1358,1418),(9247,1295,1328),(9110,1314,1328),(9038,1322,1355),(9166,1299,1310),(9215,1294,1312)],
    ("LATE", 2): [(8978,2192,3014),(11073,1749,1849),(11384,1832,1993),(11329,1666,1718),(11094,1903,2098),(11119,1653,1689),(10952,1657,1699)],
    ("LATE", 5): [(8944,2259,3307),(11459,1717,1786),(11764,1788,1893),(11133,1661,1710),(11186,1795,1890),(11173,1641,1678),(11113,1645,1675)],
}
PANEL_NAMES = {"interactive": "2A · Eligibility, interactive", "plinear": "2B · Eligibility, PLR",
               "LATE": "3 · Participation, LATE"}
DRAW_FIELDS = ("run", "panel", "folds", "partition", "method", "estimate", "se", "seconds", "origin")
METHOD_NAMES = dict(zip(METHODS, ("Lasso", "Tree", "Forest", "Boosting", "Neural net", "Ensemble", "Best")))


def write_atomic(path: Path, payload: bytes) -> None:
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(payload)
        name = Path(tmp.name)
    name.chmod(0o644)
    name.replace(path)


def get_draws(raw_dir: Path | None) -> list[dict]:
    archive = ROOT / "401k-draws.csv.gz"
    if raw_dir is None:
        with gzip.open(archive, "rt", newline="") as stream:
            return list(csv.DictReader(stream))
    from dispatch_401k import JOBS, valid_joint

    if any(raw_dir.glob("partial-all-*")):
        subprocess.run(["Rscript", str(ROOT / "score_401k_checkpoints.R"), str(raw_dir),
                        str(ROOT.parents[2] / "test/fixtures/401k_data.csv")], check=True)
    result = []
    joint = {}
    for path in sorted(raw_dir.glob("*.csv")):
        bundle = path.name.startswith("all-")
        if bundle and (path.stem not in JOBS or not valid_joint(path, JOBS[path.stem])):
            continue
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if not {"panel", "folds", "partition", "method", "estimate", "se", "seconds"} <= row.keys():
                    continue
                if path.name.startswith(("all-", "component-")):
                    row["run"] = "joint"
                    row["origin"] = "bundle" if bundle else "component"
                    key = tuple(row[k] for k in ("panel", "folds", "partition", "method"))
                    if key in joint:
                        for field in ("estimate", "se"):
                            old, new = joint[key][field], row[field]
                            if finite(old) != finite(new) or (finite(old) and not math.isclose(
                                    float(old), float(new), rel_tol=1e-9, abs_tol=1e-7)):
                                raise ValueError(f"Checkpoint and final row disagree for {key} {field}")
                    if path.name.startswith("all-") or key not in joint:
                        joint[key] = row
                else:
                    row["run"] = "single"
                    row["origin"] = "single"
                    result.append(row)
    result.extend(joint.values())
    if not result:
        raise ValueError("No completed draw files")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, DRAW_FIELDS)
    writer.writeheader()
    writer.writerows(result)
    write_atomic(archive, gzip.compress(buffer.getvalue().encode(), mtime=0))
    return result


def finite(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def summarize(draws: list[dict]) -> list[dict]:
    groups = defaultdict(dict)
    for row in draws:
        key = (row["run"], row["panel"], int(row["folds"]), row["method"])
        partition = int(row["partition"])
        if partition in groups[key]:
            raise ValueError(f"Duplicate partition {partition} in {key}")
        groups[key][partition] = row
    summary = []
    for (panel, folds), targets in TARGETS.items():
        for method, target in zip(METHODS, targets):
            candidates = {}
            for run in ("single", "joint"):
                group = groups[run, panel, folds, method]
                candidates[run] = [r for r in group.values() if finite(r["estimate"]) and finite(r["se"])]
            # Keep run families separate. Joint fits win ties.
            run = max(("joint", "single"), key=lambda name: len(candidates[name]))
            rows = candidates[run]
            item = {"panel": panel, "folds": folds, "method": method, "source": run if rows else "",
                        "completed": len(groups[run, panel, folds, method]), "finite": len(rows),
                        "published_estimate": target[0], "published_median_se": target[1],
                        "published_adjusted_se": target[2], "estimate": "", "median_se": "", "adjusted_se": "", "median_adjusted_root_se": "", "difference": ""}
            if rows:
                theta = median(float(r["estimate"]) for r in rows)
                standard_error = median(float(r["se"]) for r in rows)
                variances = [float(r["se"])**2 + (float(r["estimate"]) - theta)**2 for r in rows]
                adjusted = math.sqrt(median(variances))
                item.update(estimate=theta, median_se=standard_error, adjusted_se=adjusted,
                            median_adjusted_root_se=median(math.sqrt(v) for v in variances),
                            difference=theta-target[0])
            summary.append(item)
    return summary


def bundle_counts(draws: list[dict]) -> dict[tuple, int]:
    groups = defaultdict(set)
    for r in draws:
        if r["run"] != "joint" or r.get("origin") not in (None, "bundle"):
            continue
        # Legacy snapshots predate 'origin'. Only final bundles recorded finite
        # timings; recovered constituent fits carried seconds=NA.
        if not all(finite(r[k]) for k in ("estimate", "se", "seconds")):
            continue
        if float(r["se"]) < 0 or float(r["seconds"]) < 0:
            continue
        panel, folds, partition = r["panel"], int(r["folds"]), int(r["partition"])
        if (panel, folds) in TARGETS and 1 <= partition <= 100:
            groups[panel, folds, partition].add(r["method"])
    counts = defaultdict(int)
    for (panel, folds, _), members in groups.items():
        if members == set(METHODS):
            counts[panel, folds] += 1
    return counts


def comparison_table(summary: list[dict], field: str) -> str:
    lines = ["| Panel and folds | Lasso | Tree | Forest | Boosting | Neural net | Ensemble | Best |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for panel, folds in TARGETS:
        values = []
        for r in summary:
            if (r["panel"], r["folds"]) == (panel, folds):
                fresh = f'{r[field]:,.0f}' if r["finite"] else "—"
                values.append(f'{fresh} / {r["published_" + field]:,}')
        lines.append(f"| {PANEL_NAMES[panel]}, {folds} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def comparison_text(full: list[dict]) -> str:
    noun = "estimate" if len(full) == 1 else "estimates"
    median_difference = median(abs(r["difference"]) for r in full)
    largest = max(full, key=lambda r: abs(r["difference"]))
    biggest_se = max(full, key=lambda r: abs(r["adjusted_se"] - r["published_adjusted_se"]))

    def label(r):
        return f'{PANEL_NAMES[r["panel"]]} ({r["folds"]} folds, {METHOD_NAMES[r["method"]]}, {r["source"]})'

    return (f'Among the {len(full)} point {noun} with 100 usable partitions, the median absolute '
            f'difference from publication was ${median_difference:,.2f}. The largest was '
            f'${abs(largest["difference"]):,.2f} for {label(largest)}, with '
            f'${largest["estimate"]:,.2f} fresh and ${largest["published_estimate"]:,} published. '
            f'The largest absolute split-adjusted SE difference was '
            f'${abs(biggest_se["adjusted_se"] - biggest_se["published_adjusted_se"]):,.2f} '
            f'for {label(biggest_se)}, with ${biggest_se["adjusted_se"]:,.2f} fresh and '
            f'${biggest_se["published_adjusted_se"]:,} published.')


def render(summary: list[dict], bundles: dict[tuple, int]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, summary[0])
    writer.writeheader()
    writer.writerows(summary)
    write_atomic(ROOT / "401k-summary.csv", stream.getvalue().encode())

    full = [r for r in summary if r["finite"] == 100]
    joint_cells = sum(r["source"] == "joint" for r in full)
    measured = sum(r["finite"] > 0 for r in summary)
    blocks = [
        f'**{sum(bundles.values())} of 600 joint jobs returned all seven finite estimates '
        f'and standard errors.** Of the 42 cells, {measured} have a finite refit and {len(full)} '
        f'have 100 usable partitions ({joint_cells} from joint runs and '
        f'{len(full) - joint_cells} from single-learner runs).'
    ]
    if full:
        blocks.append(comparison_text(full))
    coverage_note = ("**Coverage by panel and run family.** The first numeric column counts completed "
                     "seven-method jobs. Each learner column gives usable partitions out of 100 and "
                     "the run family used for its median.")
    if sum(bundles.values()) < 600:
        coverage_note += " Single-learner results are provisional comparisons while the joint runs finish."
    blocks.append(coverage_note)
    coverage = ["| Panel and folds | Joint jobs | Lasso | Tree | Forest | Boosting | Neural net | Ensemble | Best |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for panel, folds in TARGETS:
        cells = [r for r in summary if (r["panel"], r["folds"]) == (panel, folds)]
        labels = [f'{r["finite"]} {r["source"]}' if r["finite"] else "0" for r in cells]
        coverage.append(f'| {PANEL_NAMES[panel]}, {folds} | {bundles.get((panel, folds), 0)} / 100 | '
                        + " | ".join(labels) + " |")
    blocks += ["\n".join(coverage),
               "**Point estimates in dollars**, fresh median / published median. "
               "A dash marks a cell awaiting a finite fit.",
               comparison_table(summary, "estimate"),
               "<details><summary>Median within-partition standard errors, fresh / published ($)</summary>",
               comparison_table(summary, "median_se"), "</details>",
               "<details><summary>Split-adjusted standard errors, fresh / published ($)</summary>",
               comparison_table(summary, "adjusted_se"), "</details>"]
    path = ROOT / "07-401k-replication.md"
    before, remainder = path.read_text().split("<!-- begin numeric output -->")
    _, after = remainder.split("<!-- end numeric output -->")
    write_atomic(path, (before + "<!-- begin numeric output -->\n\n" + "\n\n".join(blocks)
                        + "\n\n<!-- end numeric output -->" + after).encode())


if __name__ == "__main__":
    values = get_draws(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
    render(summarize(values), bundle_counts(values))
