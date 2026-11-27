"""Compare fresh standalone outputs with the untouched snapshot or archived pilot receipts."""

import argparse
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import shapiro

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import compare, file_digest, payload, write_gzip  # noqa: E402

OLD = Path("docs/audits/2026-11-19_time_series_dml")
RUN_FIELDS = {"wall_seconds", "workers"}
META_FIELDS = {"start_time", "end_time", "duration_seconds", "n_jobs", "timestamp"}
# Hashes of earlier independent pilot outputs, checked against the original local
# November 19 files, to allow verification beyond their rounded manuscript tables.
PILOTS = {
    "chapter5.log": "550ae8b36ee95a7e4847f5d7b204aa92abc938f39b37b2a23be2207c6f861104",
    "bike.log": "3653108a8b864ee367d53d98a97453b4443c3fb3629cf29bc0419a368596b5f2",
    "california-reference.log": "60c749a71ab0ab6ad39441d90bbdf4a7c39496d6914182026dfeeb39edb25bce",
    "iflex_daily.csv": "e9daa2a789568bd2633b229c4dd750023518a750fd9671428e58057ff8ab3c2b",
}


def json_comparison(expected, actual, omitted):
    before, after = [json.loads(p.read_text()) for p in (expected, actual)]
    metadata = {
        k: [before.get(k), after.get(k)] for k in sorted(omitted) if k in before or k in after
    }
    for k in metadata:
        before.pop(k, None)
        after.pop(k, None)
    clocks = {"seconds", "median_seconds"}

    def omit_clocks(obj):
        if isinstance(obj, dict):
            return {k: omit_clocks(v) for k, v in obj.items() if k not in clocks}
        if isinstance(obj, list):
            return [omit_clocks(v) for v in obj]
        return obj

    _, _, all_diffs = compare(before, after)
    timing = [d for d in all_diffs if d["path"][-1] in clocks]
    count, max_abs, diffs = compare(omit_clocks(before), omit_clocks(after))
    return {
        "numeric_fields": count,
        "different_elapsed_time_fields": len(timing),
        "largest_absolute_difference": max_abs,
        "different_fields": len(diffs),
        "first_differences": diffs[:8],
        "operational_fields": metadata,
        "status": "changed" if diffs else "matched",
    }


def table_comparison(expected, actual, keys):
    def read_table(path):
        source = io.BytesIO(payload(path))
        if path.name == "method_summary_statistics.csv":
            # pandas writes the grouped metric and aggregation in two rows, with
            # the actual method index name in a third header row.
            frame = pd.read_csv(
                source, header=[0, 1], index_col=0, dtype=str, keep_default_na=False
            )
            frame.columns = [".".join(column) for column in frame.columns]
            return frame.rename_axis("Method").reset_index()
        return pd.read_csv(source, dtype=str, keep_default_na=False)

    a, b = read_table(expected), read_table(actual)
    result = {
        "historical_rows": len(a),
        "fresh_rows": len(b),
        "payload_bytes_identical": file_digest(expected) == file_digest(actual),
    }
    if "Configuration" in keys and expected.name in {
        "configuration_summaries.csv",
        "configuration_performance.csv",
        "method_comparison_detailed.csv",
    }:
        # The November 2025 archive omitted the seed in displayed keys. Both replay scripts
        # explicitly use 42 in all six designs. Preserve the missing label as provenance.
        seed_labels = {}
        for label, frame in [("historical", a), ("fresh", b)]:
            config = frame["Configuration"]
            seeds = config.str.extract(r"(?:^|, )random_state=([^,]+)(?:,|$)", expand=False)
            seed_labels[label] = sorted(seeds.dropna().unique().tolist())
            if seeds.isna().any():
                seed_labels[label].append(None)
            if any(seed not in {None, "42"} for seed in seed_labels[label]):
                raise ValueError(f"{expected.name}: unexpected seed in fixed-seed replay")
            frame["Configuration"] = config.str.replace(r", random_state=42$", "", regex=True)
        result["displayed_seed_labels"] = seed_labels
    # Historical timestamps change filenames. Stable scientific keys handle output reordering.
    if keys:
        if not all(c in a and c in b for c in keys):
            raise ValueError(f"{actual.name}: missing row key")
        a, b = [frame.set_index(keys).sort_index() for frame in (a, b)]
        if not a.index.is_unique or not b.index.is_unique:
            raise ValueError(f"{actual.name}: duplicate row identifiers")
    shared = a.index.intersection(b.index)
    result.update(
        {
            "rows_only_in_historical": len(a.index.difference(b.index)),
            "rows_only_in_fresh": len(b.index.difference(a.index)),
            "columns_only_in_historical": sorted(set(a) - set(b)),
            "columns_only_in_fresh": sorted(set(b) - set(a)),
        }
    )
    changed, compared, max_abs = {}, 0, 0.0
    by_method = (
        {method: {} for method in shared.get_level_values("Method").unique()}
        if expected.name == "method_comparison_detailed.csv" and "Method" in keys
        else {}
    )
    for col in a.columns.intersection(b.columns):
        left, right = a.loc[shared, col], b.loc[shared, col]
        differing = 0
        example = None
        for ix, old, new in zip(shared, left, right, strict=True):
            compared += 1
            if old == new:
                continue
            try:
                x, y = float(old), float(new)
            except ValueError:
                close = False
            else:
                if (
                    col not in {"seconds", "median_seconds"}
                    and math.isfinite(x)
                    and math.isfinite(y)
                ):
                    max_abs = max(max_abs, abs(x - y))
                close = (
                    math.isclose(x, y, abs_tol=1e-9, rel_tol=1e-9)
                    or math.isnan(x)
                    and math.isnan(y)
                )
            if not close:
                differing += 1
                if by_method:
                    method = ix[keys.index("Method")]
                    by_method[method][col] = by_method[method].get(col, 0) + 1
                if example is None:
                    example = {"row": str(ix), "historical": old, "fresh": new}
        if differing:
            changed[col] = {"count": differing, "first": example}
    timing = {col: changed.pop(col) for col in ["seconds", "median_seconds"] if col in changed}
    if by_method:
        result["disagreements_by_method"] = by_method
    result.update(
        {
            "elapsed_time_columns": timing,
            "common_cells_compared": compared,
            "columns_with_differences": changed,
            "largest_absolute_difference": max_abs,
        }
    )
    result["status"] = (
        "changed"
        if changed
        or any(
            result[k]
            for k in [
                "rows_only_in_historical",
                "rows_only_in_fresh",
                "columns_only_in_historical",
                "columns_only_in_fresh",
            ]
        )
        else "matched"
    )
    return result


def method_summary_checks(directory, *, coverage_rule="closest_to_nominal"):
    """Check configuration summaries against the detailed rows saved by the same invocation."""
    details = pd.read_csv(directory / "method_comparison_detailed.csv")
    summaries = pd.read_csv(directory / "configuration_summaries.csv").set_index("Configuration")
    if not summaries.index.is_unique or details.duplicated(["Configuration", "Method"]).any():
        raise ValueError(f"Duplicate method-comparison identifiers in {directory}")
    groups = details.groupby("Configuration")
    if set(groups.groups) != set(summaries.index):
        raise ValueError(f"Summary and detailed configurations differ in {directory}")
    mismatches = {}
    for config, frame in groups:
        aggregate = summaries.loc[config]
        by_method = frame.set_index("Method")
        checks = {}
        for key, column in [
            ("Mean_Absolute_Bias", "Abs_Bias"),
            ("Mean_MSE", "MSE"),
            ("Mean_Coverage", "Coverage"),
        ]:
            observed, calculated = float(aggregate[key]), float(by_method[column].mean())
            if not math.isclose(observed, calculated, rel_tol=1e-9, abs_tol=1e-9):
                checks[key] = {"reported": observed, "from_detail": calculated}
        if coverage_rule == "closest_to_nominal":
            coverage_metric, coverage_extreme = (by_method["Coverage"] - 0.95).abs(), min
        elif coverage_rule == "highest":
            coverage_metric, coverage_extreme = by_method["Coverage"], max
        else:
            raise ValueError(f"Unknown coverage winner rule {coverage_rule}")
        for key, metric, extreme in [
            ("Best_Bias_Method", by_method["Abs_Bias"], min),
            ("Worst_Bias_Method", by_method["Abs_Bias"], max),
            ("Best_MSE_Method", by_method["MSE"], min),
            ("Best_Coverage_Method", coverage_metric, coverage_extreme),
        ]:
            best = extreme(metric)
            tied = sorted(
                method
                for method, value in metric.items()
                if math.isclose(value, best, rel_tol=1e-9, abs_tol=1e-9)
            )
            if aggregate[key] not in tied:
                checks[key] = {"reported": aggregate[key], "eligible_from_detail": tied}
        if checks:
            mismatches[config] = checks
    return {
        "status": "changed" if mismatches else "matched",
        "configuration_count": len(summaries),
        "methods_per_configuration": sorted(groups.Method.nunique().unique().tolist()),
        "detailed_rows": len(details),
        "coverage_target": 0.95,
        "coverage_winner_rule": coverage_rule,
        "disagreements": mismatches,
    }


def pilot_comparisons(stored, fresh):
    """Compare the display-precision pilot tables to fresh stdout, with half-unit rounding tolerance."""

    def cells(line):
        return re.findall(r"[-+]?(?:\d+\.\d+|\d+)", line.replace("−", "-").replace(",", ""))

    def check(target, values, wanted=None):
        if wanted:
            values = [values[i] for i in wanted]
        if len(values) != len(target):
            raise ValueError(f"Different numbers of cells in {target} and {values}")
        return sum(
            abs(float(v) - float(t)) > 0.50001 * 10 ** (-len(t.partition(".")[2]))
            for t, v in zip(target, values, strict=True)
        )

    def table(doc, prefix, first=1):
        line = next(
            line
            for line in (stored / "pilots" / doc).read_text().splitlines()
            if line.startswith("| " + prefix + " |")
        )
        return cells(" ".join(x.strip() for x in line.split("|")[first + 1 : -1]))

    result = {}
    # HeartSteps table includes eligible decisions as the first numeric column.
    cases = [
        (
            "heartsteps",
            "heartsteps.md",
            [
                (
                    "Published centered regression, recorded delivery",
                    "published_delivery_definition",
                ),
                ("Same regression, recorded assignment draw", "recorded_assignment_draw"),
                ("Known-propensity orthogonal score, five participant folds", "draw_person_folds"),
                ("Centered regression, recorded draw, days 7–41", "draw_WCLS_after_warmup"),
                (
                    "Known-propensity orthogonal score, expanding study-week folds",
                    "draw_forward_study_weeks",
                ),
            ],
        ),
        (
            "california-reference",
            "california.md",
            [
                ("Ordinary difference in differences", "did_estimate:"),
                ("Synthetic control, outcome history", "sc_estimate:"),
                ("Synthetic difference in differences", "synthdid_estimate:"),
            ],
        ),
        (
            "iflex-reference",
            "iflex.md",
            [
                ("Pre-announcement baseline", "Pre-announcement baseline "),
                ("First event", "First event "),
                ("First event minus baseline", "First event minus baseline "),
                ("Average event day", "All event days"),
                ("Average event day minus baseline", "All events minus baseline"),
            ],
        ),
    ]
    for job, doc, items in cases:
        path = fresh / (job + ".log")
        if not path.is_file():
            result[job] = {"status": "pending"}
            continue
        lines = path.read_text().splitlines()
        checks = {}
        for label, match in items:
            target = table(doc, label)
            i = next(i for i, line in enumerate(lines) if line.startswith(match))
            vals = cells(lines[i][len(match) :])
            if job == "heartsteps":
                count = (
                    int(
                        cells(
                            next(line for line in lines if line.startswith("forward-scored rows"))
                        )[0]
                    )
                    if match in ["draw_WCLS_after_warmup", "draw_forward_study_weeks"]
                    else int(next(line for line in lines if line.startswith("[1]")).split()[6])
                )
                vals = [str(count), *vals]
            elif job == "iflex-reference":
                vals = cells(lines[i + 1].removeprefix("[1]"))
                vals = [vals[j] for j in (0, 1, 2, 4, 5)]
            checks[label] = {"cells": len(target), "mismatches": check(target, vals)}
        result[job] = {
            "rows": checks,
            "status": "changed" if any(v["mismatches"] for v in checks.values()) else "matched",
        }
    path = fresh / "iflex-temporal.log"
    if path.is_file() and (fresh / "receipts/iflex-temporal.json").is_file():
        lines = path.read_text().splitlines()
        checks = {}
        for gap, bw in [(2, 7), (2, 14), (7, 7)]:
            line = next(line for line in lines if line.startswith(f"{gap} {bw} "))
            target = table("iflex.md", f"{gap} | {bw}", first=2)
            checks[f"gap {gap}, bandwidth {bw}"] = {
                "cells": len(target),
                "mismatches": check(target, cells(line)[3:7]),
            }
        result["iflex-temporal"] = {
            "rows": checks,
            "status": "changed" if any(v["mismatches"] for v in checks.values()) else "matched",
        }
    else:
        result["iflex-temporal"] = {"status": "pending"}
    path = fresh / "california.log"
    if path.is_file():
        lines = [
            line for line in path.read_text().splitlines() if line.startswith("Train, test years")
        ]
        checks = {}
        for line, span in zip(lines, ["1970–1978", "1970–1985", "1970–1992"], strict=True):
            target = table("california.md", span)
            vals = cells(line)
            checks[span] = {
                "cells": len(target) + 2,
                "mismatches": check(cells(span) + target, vals),
            }
        result["california"] = {
            "rows": checks,
            "status": "changed" if any(v["mismatches"] for v in checks.values()) else "matched",
        }
    return result


def lasso_fields(text):
    """Collect comparable displayed measurements despite the two reports' different captions."""

    def numbers(line):
        return [
            float(v)
            for v in re.findall(r"[-+]?\d+(?:\.\d+)?", line.split(":", 1)[1].replace(",", ""))
        ]

    fields = {}
    section, parameter = None, None
    for raw in text.splitlines():
        line = raw.strip()
        if line in {"BOOTSTRAP DISTRIBUTION:", "HYPERPARAMETER SENSITIVITY:", "SEED SENSITIVITY:"}:
            section = line.split()[0].lower()
        elif line.startswith(("ROOT CAUSE ANALYSIS", "DIAGNOSTIC FINDINGS", "RECOMMENDATIONS")):
            section = None
        elif section == "bootstrap":
            for labels, key in [
                (("Mean ATE:",), "mean_and_sd"),
                (("95% CI:", "95% pairs-bootstrap percentile interval:"), "interval"),
                (("Normality:", "Shapiro-Wilk p:"), "shapiro_p"),
                (("Outliers:",), "outliers_and_draws"),
            ]:
                if line.startswith(labels):
                    fields[f"bootstrap/{key}"] = numbers(line)
        elif section == "hyperparameter":
            if line in {"cv_folds:", "max_iter:"}:
                parameter = line[:-1]
            elif line.startswith(
                ("Sensitivity:", "Recommended:", "Value with smallest fitted ATE SE:", "ATE Range:")
            ):
                label = (
                    "range"
                    if line.startswith("ATE Range")
                    else ("score" if line.startswith("Sensitivity:") else "selected_setting")
                )
                fields[f"{parameter}/{label}"] = numbers(line)
        elif section == "seed" and line.startswith(("Mean ATE:", "Range:", "CV:")):
            fields["seed/" + line.split(":")[0]] = numbers(line)
    return fields


def lasso_conclusions(report):
    lines = [line.strip() for line in report.splitlines()]
    flags = [
        line
        for line in lines
        if line.startswith(
            ("Normality:", "Converged:", "Shapiro-Wilk p:", "Running-mean stability:")
        )
    ]
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if line.startswith(("ROOT CAUSE ANALYSIS", "DIAGNOSTIC FINDINGS"))
        ),
        len(lines),
    )
    return {"diagnostic_flags": flags, "interpretation": [line for line in lines[start:] if line]}


def lasso_admission(path):
    if not path.exists():
        return None
    with path.open() as log:
        rows = [json.loads(line) for line in log if line.strip()]
    if not rows:
        return None
    return {
        "sample_count": len(rows),
        "first_utc": rows[0]["utc"],
        "last_utc": rows[-1]["utc"],
        "minimum_runnable": min(row["selected"] for row in rows),
        "maximum_runnable": max(row["selected"] for row in rows),
        "samples_parked": sum(row["selected"] == 0 for row in rows),
        "minimum_memory_available_gib": min(row["available_gib"] for row in rows),
        "maximum_reserved_r_growth_gib": max(row["r_future_growth_gib"] for row in rows),
        "maximum_observed_host_busy_cores": max(row["host_busy_cores"] for row in rows),
        "system_reserve_gib": sorted({row["system_reserve_gib"] for row in rows}),
        "host_targets": sorted({row["host_cpu_target"] for row in rows}),
        "maximum_worker_vmhwm_gib": max(row.get("worker_vmhwm_gib", 0) for row in rows),
        "maximum_python_incremental_reservation_gib": max(
            row.get("requested_python_growth_gib", 0) for row in rows
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    snap, work, fresh = [
        args.run / d for d in ["snapshot", "standalone-work", "standalone-results"]
    ]
    stored = snap / OLD
    dest = args.destination
    dest.mkdir(exist_ok=True, parents=True)
    receipts = {p.stem: json.loads(p.read_text()) for p in (fresh / "receipts").glob("*.json")}
    artifacts = {}
    fresh_legacy_outputs = {}

    def add(name, expected, actual, *, keys=None, omitted=None, json_format=False):
        if not actual or not actual.is_file():
            artifacts[name] = {"status": "pending"}
        elif json_format or expected.suffix == ".json":
            artifacts[name] = json_comparison(expected, actual, omitted or set())
        else:
            artifacts[name] = table_comparison(expected, actual, keys or [])
        if name.startswith("legacy/") and actual and actual.is_file():
            fresh_legacy_outputs[name] = {"filename": actual.name, "content": actual.read_text()}

    for name in [
        "chapter5-draws.csv",
        "panel-draws.csv",
        "insurance-panel-draws.csv",
        "insurance-panel-summary.csv",
        "insurance-rolling.csv",
        "retail-results.csv",
        "retail-folds.csv",
        "retail-baselines.json",
        "stress-draws.csv.gz",
        "stress-summary.csv",
    ]:
        add(name, stored / name, fresh / name)
    for name, log in [
        ("chapter10-pipeline.csv", "chapter10_pipeline.log"),
        ("bias-archive-rescored.csv", "rescore_bias_archive.log"),
    ]:
        add(name, stored / name, fresh / log)
    for p in sorted((stored / "adversarial").glob("simulation-*")):
        if p.is_file() and p.suffix in {".gz", ".csv", ".json"}:
            add("adversarial/" + p.name, p, fresh / "adversarial" / p.name, omitted=RUN_FIELDS)
    for case in ["empirical", "failures", "transport"]:
        add(
            "adversarial/" + case + ".json",
            stored / "adversarial" / (case + ".json"),
            fresh / ("adversarial-transport.log" if case == "transport" else case + ".json"),
        )

    for name, digest in PILOTS.items():
        path = fresh / name
        matches = path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
        artifacts["archived-stdout-or-file/" + name] = {
            "sha256_reference": digest,
            "status": "matched" if matches else "changed",
        }

    def latest_new(subdir, pattern):
        files = sorted(
            p
            for p in (work / "results" / subdir).glob(pattern)
            if not (snap / "results" / subdir / p.name).exists()
        )
        return files[-1] if files else None

    for case, key in [
        (
            "c2",
            [
                "configuration",
                "discrete_treatment",
                "dgp_n",
                "dgp_p",
                "dgp_true_effect",
                "dgp_confounding",
            ],
        ),
        ("c3", ["scenario", "run", "method"]),
        ("c5", ["n_covariates"]),
    ]:
        subdir = "impact_analysis"
        pat = f"issue_{case}_impact_*.csv"
        (old,) = (snap / "results" / subdir).glob(pat)
        add("legacy/" + case, old, latest_new(subdir, pat), keys=key)
    for kind, suff in [("results", "csv"), ("metadata", "json")]:
        pat = f"BIAS_VALIDATION_{kind}_*.{suff}"
        (old,) = (snap / "results/simulations").glob(pat)
        add(
            "legacy/bias-grid-" + kind,
            old,
            latest_new("simulations", pat),
            keys=["n", "p", "true_effect", "confounding_strength"],
            omitted=META_FIELDS,
        )
    subdir = Path("results/method_comparison")
    for p in sorted((snap / subdir).glob("*")):
        if p.suffix in {".json", ".csv"}:
            # These files already exist in the snapshot. A passing receipt certifies a new run.
            actual = (
                work / subdir / p.name
                if receipts.get("comprehensive_method_comparison", {}).get("exit") == 0
                else None
            )
            keys = {
                "method_comparison_detailed.csv": ["Configuration", "Method"],
                "method_summary_statistics.csv": ["Method"],
                "configuration_summaries.csv": ["Configuration"],
                "configuration_performance.csv": ["Configuration"],
            }.get(p.name)
            add("legacy/method-comparison/" + p.name, p, actual, keys=keys, omitted=META_FIELDS)
    method_reconciliations = {
        "archived_release": method_summary_checks(snap / subdir, coverage_rule="highest"),
        "frozen_source_refit": (
            method_summary_checks(work / subdir)
            if receipts.get("comprehensive_method_comparison", {}).get("exit") == 0
            else {"status": "pending"}
        ),
    }
    corrected_dir = args.run / "corrected-method-comparison"
    if (corrected_dir / "configuration_summaries.csv").is_file():
        method_reconciliations["corrected_refit"] = method_summary_checks(corrected_dir)
    original_done = receipts.get("comprehensive_method_comparison", {}).get("exit") == 0
    method_file = "method_comparison_detailed.csv"
    refit_comparison = (
        table_comparison(
            work / subdir / method_file, corrected_dir / method_file, ["Configuration", "Method"]
        )
        if original_done and (corrected_dir / method_file).is_file()
        else {"status": "pending"}
    )
    coverage_run = latest_new("coverage_stress", "*.csv")
    if coverage_run:
        fresh_legacy_outputs["coverage_stress"] = {
            "filename": coverage_run.name,
            "content": coverage_run.read_text(),
        }
        revised_path = (
            Path(__file__).resolve().parent.parent / "experiments/legacy-coverage-stress.csv"
        )
        prior = pd.read_csv(coverage_run).set_index("scenario_num")
        revised = pd.read_csv(revised_path).set_index("scenario_num")
        if not prior.index.is_unique or not revised.index.is_unique:
            raise ValueError("Duplicate scenario numbers in coverage grid")
        fields = {}
        for col in ["n", "p", "confounding", "n_simulations", "actual_coverage", "bias", "mse"]:
            rows = {}
            for ix in prior.index.intersection(revised.index):
                old, new = float(prior.loc[ix, col]), float(revised.loc[ix, col])
                if not math.isclose(old, new, rel_tol=1e-9, abs_tol=1e-9):
                    rows[str(ix)] = [old, new]
            if rows:
                fields[col] = rows
        artifacts["legacy/coverage-stress-vs-corrected"] = {
            "historical_rows": len(prior),
            "corrected_rows": len(revised),
            "scientific_differences": fields,
            "status": "changed" if fields or set(prior.index) != set(revised.index) else "matched",
        }
    else:
        artifacts["legacy/coverage-stress-vs-corrected"] = {"status": "pending"}
    lasso_report = latest_new("lasso_diagnostic", "*.txt")
    if lasso_report:
        (archive,) = (snap / "results/lasso_diagnostic").glob("*.txt")
        old, new = [lasso_fields(p.read_text()) for p in (archive, lasso_report)]
        if not old or old.keys() != new.keys():
            raise ValueError(f"Lasso diagnostic report fields changed: {old.keys() ^ new.keys()}")
        n, max_abs, delta = compare(old, new)
        artifacts["legacy/lasso-diagnostic"] = {
            "status": "changed" if delta else "matched",
            "numeric_fields": n,
            "largest_absolute_difference": max_abs,
            "historical": old,
            "fresh": new,
            "different_fields": len(delta),
            "historical_conclusions": lasso_conclusions(archive.read_text()),
            "fresh_conclusions": lasso_conclusions(lasso_report.read_text()),
        }
        fresh_legacy_outputs["lasso_diagnostic"] = {
            "filename": lasso_report.name,
            "content": lasso_report.read_text(),
        }
    else:
        artifacts["legacy/lasso-diagnostic"] = {"status": "pending"}
    write_gzip(dest / "standalone-legacy-outputs.json.gz", fresh_legacy_outputs)
    logs = {}
    for name, record in sorted(receipts.items()):
        log = fresh / (name + ".log")
        if log.exists():
            if hashlib.sha256(log.read_bytes()).hexdigest() != record["log_sha256"]:
                raise ValueError(f"Log changed after job {name}")
            logs[name] = log.read_text()
    attempts = {}
    for path in sorted((fresh / "attempts").glob("*.json")):
        record = json.loads(path.read_text())
        attempts[path.stem] = record
        attempt_log = path.with_suffix(".log")
        if attempt_log.is_file():
            if hashlib.sha256(attempt_log.read_bytes()).hexdigest() != record["log_sha256"]:
                raise ValueError(f"Attempt log changed: {attempt_log.name}")
            logs[path.stem] = attempt_log.read_text()
    lasso = args.run / "standalone-lasso"
    if (lasso / "verification.json").is_file():
        probe = {
            name: json.loads((lasso / (name + ".json")).read_text())
            for name in ["manifest", "verification", "serial-peak"]
            if (lasso / (name + ".json")).is_file()
        }
        (dest / "standalone-lasso-probe.json").write_text(json.dumps(probe, indent=2) + "\n")
    if (lasso / "complete.json").is_file():
        ordered = {}
        for f in lasso.glob("chunk-*.json"):
            chunk = json.loads(f.read_text())
            for offset, value in enumerate(chunk["values"]):
                index = chunk["first_draw"] + offset
                if index in ordered:
                    raise ValueError("Duplicate Lasso bootstrap estimate")
                ordered[index] = value
        if set(ordered) != set(range(1000)):
            raise ValueError("Expected all historical Lasso bootstrap fits")
        preserved = {
            k: json.loads((lasso / (k + ".json")).read_text())
            for k in ["manifest", "verification", "complete", "cv_folds", "max_iter", "seeds"]
        }
        preserved["resource_admission"] = lasso_admission(lasso / "admission.jsonl")
        if (lasso / "serial-peak.json").is_file():
            preserved["serial_peak_recheck"] = json.loads((lasso / "serial-peak.json").read_text())
        values = np.asarray([ordered[i] for i in range(1000)])
        if not np.isfinite(values).all() or lasso_report is None:
            raise ValueError("Expected all finite historical estimates and their original report")
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        checks = {
            "bootstrap/mean_and_sd": [mean, sd],
            "bootstrap/interval": np.quantile(values, [0.025, 0.975]).tolist(),
            "bootstrap/shapiro_p": [float(shapiro(values).pvalue)],
            "bootstrap/outliers_and_draws": [int((abs(values - mean) > 3 * sd).sum()), len(values)],
        }
        displayed = lasso_fields(lasso_report.read_text())
        for key, calculated in checks.items():
            tolerance = {"bootstrap/shapiro_p": 0.00005, "bootstrap/outliers_and_draws": 0}.get(
                key, 0.005
            )
            np.testing.assert_allclose(calculated, displayed[key], atol=tolerance, rtol=0)
        preserved["independent_summary_from_ordered_draws"] = checks
        preserved["bootstrap_estimates_by_zero_based_draw"] = values.tolist()
        write_gzip(dest / "standalone-lasso.json.gz", preserved)
    write_gzip(dest / "standalone-stdout.json.gz", logs)
    compact_receipts = {
        name: {k: r[k] for k in ["command", "exit", "seconds", "log_sha256"] if k in r}
        for name, r in sorted(receipts.items())
    }
    supervisor_attempts = []
    initial_log = args.run / "standalone.log"
    if initial_log.exists():
        for line in initial_log.read_text().splitlines():
            match = re.fullmatch(r"([\w-]+) (-?\d+) (\d+(?:\.\d+)?)", line.strip())
            if match:
                name, code, duration = match.groups()
                supervisor_attempts.append(
                    {"study": name, "exit": int(code), "seconds": float(duration)}
                )
    outcome = {
        "tolerance": {"relative": 1e-9, "absolute": 1e-9},
        "commands": compact_receipts,
        "initial_supervisor_attempts": supervisor_attempts,
        "additional_lasso_attempts": attempts,
        "pilot_tables": pilot_comparisons(stored, fresh),
        "artifacts": artifacts,
        "within_run_method_summaries_against_detailed_rows": method_reconciliations,
        "frozen_vs_corrected_method_detail": refit_comparison,
    }
    (dest / "standalone-comparison.json").write_text(
        json.dumps(outcome, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
