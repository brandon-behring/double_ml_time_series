"""Recover bias tests from the archived mean error, mean squared error and run count.

Run from anywhere. The row-level CSV is written to stdout.
"""

import csv
import json
import sys
from math import isclose, sqrt
from pathlib import Path

from scipy.stats import binomtest, t

from dml_ts.validation.bias_validation import classify_bias_and_coverage

RESULTS = Path(__file__).resolve().parents[3] / "results"


def bias_test(bias, mse, n_runs):
    # MSE is the mean squared error around truth, with denominator n_runs.
    mcse = sqrt((mse - bias**2) / (n_runs - 1))
    return mcse, float(2 * t.sf(abs(bias) / mcse, n_runs - 1))


def method_comparison():
    directory = RESULTS / "method_comparison"
    n_runs = json.loads((directory / "findings.json").read_text())["simulations_per_method"]
    with (directory / "method_comparison_detailed.csv").open() as source:
        for row in csv.DictReader(source):
            bias = float(row["Bias"])
            mcse, p_bias = bias_test(bias, float(row["MSE"]), n_runs)
            p_coverage = None
            p_for_status = p_bias
            if row["Method"] == "DML":
                # Bonferroni across the historical DML bias and coverage tests.
                n_cover = round(float(row["Coverage"]) * n_runs)
                assert isclose(n_cover / n_runs, float(row["Coverage"]))
                p_coverage = binomtest(n_cover, n_runs, 0.95).pvalue
                p_for_status = min(1.0, 2 * min(p_bias, p_coverage))
            if p_for_status < 0.01:
                verdict = "FAIL"
            elif p_for_status < 0.05:
                verdict = "WARNING"
            else:
                verdict = "PASS"
            if p_coverage is not None:
                component = classify_bias_and_coverage(
                    bias, p_bias, float(row["Coverage"]), 0.95, p_coverage
                )
                current = component.overall
                bias_label, coverage_label = component
            else:
                current, bias_label, coverage_label = verdict, verdict, None
            yield (
                "method_comparison",
                row["Configuration"],
                row["Method"],
                n_runs,
                row["Status"],
                row["P_Value"],
                bias,
                mcse,
                p_bias,
                p_coverage,
                verdict,
                None,
                current,
                bias_label,
                coverage_label,
            )


def earlier_sweep():
    directory = RESULTS / "simulations"
    meta = json.loads((directory / "BIAS_VALIDATION_metadata_20251114_103701.json").read_text())
    n_runs = meta["n_simulations"]
    possible_coverage = [(k / n_runs, binomtest(k, n_runs, 0.95).pvalue) for k in range(n_runs + 1)]
    with (directory / "BIAS_VALIDATION_results_20251114_103701.csv").open() as source:
        for row in csv.DictReader(source):
            assert int(row["n_simulations"]) == n_runs
            bias = float(row["bias"])
            mcse, p_bias = bias_test(bias, float(row["mse"]), n_runs)
            settings = ("n", "p", "true_effect", "confounding_strength")
            config = ", ".join(f"{k}={row[k]}" for k in settings)
            # This archive used absolute bias cutoffs and placeholder intervals.
            # Check the original verdict. A modern coverage p-value is unavailable.
            old_verdict = "FAIL" if abs(bias) > 0.2 else "WARNING" if abs(bias) >= 0.1 else "PASS"
            assert row["status"] == old_verdict
            components = [
                classify_bias_and_coverage(bias, p_bias, coverage, 0.95, p)
                for coverage, p in possible_coverage
            ]
            possible = {verdict.overall for verdict in components}
            yield (
                "earlier_sweep",
                config,
                "DML",
                n_runs,
                old_verdict,
                None,
                bias,
                mcse,
                p_bias,
                None,
                None,
                "|".join(label for label in ("PASS", "WARNING", "FAIL") if label in possible),
                None,
                components[0].bias,
                None,
            )


if __name__ == "__main__":
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        (
            "archive",
            "configuration",
            "method",
            "runs",
            "archived_status",
            "archived_bias_p",
            "mean_error",
            "mcse",
            "corrected_bias_p",
            "coverage_p_used",
            "rescored_status",
            "possible_current_dml_status_without_coverage",
            "current_status",
            "current_bias_status",
            "current_coverage_status",
        )
    )
    writer.writerows(method_comparison())
    writer.writerows(earlier_sweep())
