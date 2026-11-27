"""Independent count and partition checks on an IID DGP and a 401(k) subsample."""

import argparse
import io
import json
import subprocess
import sys
import tempfile
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import binom, binomtest
from sklearn.exceptions import ConvergenceWarning

from dml_ts.validation.bias_validation import BiasValidation
from dml_ts.validation.dgp_generator import DGPGenerator
from dml_ts.validation.lasso_diagnostic import LassoDiagnostic

ROOT = Path(__file__).resolve().parents[4]


def coverage():
    """Analyze one frozen configuration with 50 draws and inspect the per-fit ATE bounds."""
    dgp = DGPGenerator(n=200, p=5, true_effect=2, confounding_strength=0.5, random_state=42)
    val = BiasValidation(n_simulations=50, random_state=42).validate(dgp)
    lower, upper = val.ci_estimates.T
    hits = int(np.count_nonzero((lower <= 2.0) & (2.0 <= upper)))
    assert val.coverage == hits / 50
    assert val.coverage_p_value == binomtest(hits, 50, 0.95).pvalue
    assert np.isclose(val.bias, np.mean(val.point_estimates - 2.0))
    interval = binomtest(hits, 50).proportion_ci()
    return {
        "n": 200,
        "p": 5,
        "seed": 42,
        "draws": 50,
        "hits": hits,
        "coverage": val.coverage,
        "coverage_mc_ci": [interval.low, interval.high],
        "bias": val.bias,
        "mse": val.mse,
        "bias_mc_ci": [val.ci_lower, val.ci_upper],
        "bias_p_value": val.bias_p_value,
        "coverage_p_value": val.coverage_p_value,
        "mean_interval_width": float(np.mean(upper - lower)),
        "theoretical_p_all_50": float(binom.pmf(50, 50, 0.95)),
        "theoretical_p_at_most_44": float(binom.cdf(44, 50, 0.95)),
    }


def lasso(full_sample=False):
    """Exercise two fivefold fits on the full fixture or a fixed random subsample."""
    from econml import __version__ as econml_version

    diagnostic = LassoDiagnostic(random_state=42, verbose=False)
    path = ROOT / "test/fixtures/401k_data.csv"
    sample = pd.read_csv(path)
    if not full_sample:
        sample = sample.sample(n=250, random_state=142).reset_index(drop=True)
    diagnostic._data = sample
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", ConvergenceWarning)
        result = diagnostic.analyze_seed_sensitivity(2)
    return {
        "sample_size": len(sample),
        "sample_seed": None if full_sample else 142,
        "econml": econml_version,
        "outer_partition_seeds": result.random_states.tolist(),
        "learner_seed": result.metadata["learner_seed"],
        "estimates": result.ate_estimates.tolist(),
        "sd_across_partitions": result.std_ate,
        "convergence_warnings": sum(issubclass(w.category, ConvergenceWarning) for w in recorded),
    }


def scripted_validator(runs, counts):
    sequence = iter(counts)

    def scripted_validate(self, dgp):
        hits = next(sequence)
        if hits is None:
            raise RuntimeError("scripted fit error")
        return SimpleNamespace(
            coverage=hits / runs,
            bias=0.0,
            mse=0.03,
            status="PASS",
            bias_status="PASS",
            coverage_status="PASS",
            n_simulations=runs,
            bias_p_value=0.72,
            coverage_p_value=binomtest(hits, runs, 0.95).pvalue,
        )

    return scripted_validate


def decisions():
    """Run the historical CLI and current driver against identical scripted validator results."""
    from scripts.comprehensive_coverage_test import coverage_exit_code, run_coverage_stress_test

    commit = "3a919dc"
    path = "scripts/comprehensive_coverage_test.py"
    original = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT, text=True)
    cases = {
        "one_47_of_50_and_14_fit_errors": (50, [47] + [None] * 14),
        "one_42_of_50_and_14_at_48": (50, [42] + [48] * 14),
        "all_50_of_50": (50, [50] * 15),
        "one_500_of_500_and_14_at_475": (500, [500] + [475] * 14),
    }
    first_step = 0.05 / 15
    reject_counts = [k for k in range(51) if binomtest(k, 50, 0.95).pvalue <= first_step]
    outcomes = {
        "source_commit": commit,
        "nominal_coverage": 0.95,
        "first_holm_step": {
            "threshold": first_step,
            "max_covered_for_rejection_at_50": max(reject_counts),
            "crossing_probability": {
                str(p): float(binom.pmf(reject_counts, 50, p).sum())
                for p in [0.80, 0.85, 0.90, 1.00]
            },
        },
        "cases": {},
    }
    for name, (runs, counts) in cases.items():
        with (
            tempfile.TemporaryDirectory() as tmp,
            redirect_stdout(io.StringIO()),
            patch.object(BiasValidation, "validate", scripted_validator(runs, counts)),
        ):
            with patch.object(sys, "argv", [path, str(runs), tmp]):
                try:
                    exec(compile(original, path, "exec"), {"__name__": "__main__"})
                except SystemExit as exit_code:
                    before = int(exit_code.code)
            with patch.object(BiasValidation, "validate", scripted_validator(runs, counts)):
                current = run_coverage_stress_test(tmp, runs)
        outcomes["cases"][name] = {
            "historical_exit_code": before,
            "current_exit_code": coverage_exit_code(current),
            "current_counts": current["suite_coverage_status"].value_counts().to_dict(),
            "first_bias_p": current.loc[0, "bias_p_value"],
            "first_coverage_p": current.loc[0, "coverage_p_value"],
            "first_coverage_holm_p": current.loc[0, "coverage_p_value_holm"],
        }
    return outcomes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", choices=["coverage", "lasso", "full-lasso", "decisions"])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    measured = {
        "coverage": coverage,
        "lasso": lasso,
        "full-lasso": lambda: lasso(full_sample=True),
        "decisions": decisions,
    }[args.probe]()
    args.output.write_text(json.dumps(measured, indent=2, allow_nan=False) + "\n")
    print(args.output.read_text())
