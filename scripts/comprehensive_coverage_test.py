"""Test nominal ATE-interval coverage across 15 frozen IID simulation scenarios.

Sample sizes span 50–5,000, covariate counts 5–50 and confounding strengths
0.5–4. Historical scenario ranges are descriptive annotations. The suite
tests the nominal level with two-sided exact binomial tests and controls the
familywise error rate across its coverage hypotheses with Holm's procedure.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

from dml_ts.validation.bias_validation import BiasValidation
from dml_ts.validation.dgp_generator import DGPGenerator


def create_test_scenarios() -> list[dict[str, Any]]:
    """
    Create 15 test scenarios varying from easy to extremely challenging.

    Returns:
        List of scenario dictionaries
    """
    scenarios = []

    # Easy scenarios (should work well)
    scenarios.append(
        {
            "name": "Easy: Large n, Low dim, Weak confounding",
            "n": 2000,
            "p": 5,
            "confounding_strength": 0.5,
            "expected_coverage": "95-98%",
        }
    )

    scenarios.append(
        {
            "name": "Moderate: Medium n, Low dim, Medium confounding",
            "n": 1000,
            "p": 5,
            "confounding_strength": 1.0,
            "expected_coverage": "93-97%",
        }
    )

    # Challenging scenarios (sample size)
    scenarios.append(
        {
            "name": "Hard: Small n, Low dim, Weak confounding",
            "n": 200,
            "p": 5,
            "confounding_strength": 0.5,
            "expected_coverage": "90-95%",
        }
    )

    scenarios.append(
        {
            "name": "Very Hard: Very small n, Low dim, Medium confounding",
            "n": 100,
            "p": 5,
            "confounding_strength": 1.0,
            "expected_coverage": "85-93%",
        }
    )

    # Challenging scenarios (dimensionality)
    scenarios.append(
        {
            "name": "Hard: Medium n, Medium dim, Weak confounding",
            "n": 500,
            "p": 15,
            "confounding_strength": 0.5,
            "expected_coverage": "90-96%",
        }
    )

    scenarios.append(
        {
            "name": "Very Hard: Medium n, High dim, Medium confounding",
            "n": 500,
            "p": 30,
            "confounding_strength": 1.0,
            "expected_coverage": "85-94%",
        }
    )

    # Challenging scenarios (confounding)
    scenarios.append(
        {
            "name": "Hard: Medium n, Low dim, Strong confounding",
            "n": 1000,
            "p": 5,
            "confounding_strength": 2.0,
            "expected_coverage": "91-96%",
        }
    )

    scenarios.append(
        {
            "name": "Very Hard: Medium n, Medium dim, Very strong confounding",
            "n": 500,
            "p": 10,
            "confounding_strength": 3.0,
            "expected_coverage": "88-95%",
        }
    )

    # Extremely challenging scenarios (multiple factors)
    scenarios.append(
        {
            "name": "Extreme: Small n, Medium dim, Strong confounding",
            "n": 200,
            "p": 15,
            "confounding_strength": 2.0,
            "expected_coverage": "85-93%",
        }
    )

    scenarios.append(
        {
            "name": "Extreme: Very small n, 20 covariates for 100 observations, Medium confounding",
            "n": 100,
            "p": 20,
            "confounding_strength": 1.0,
            "expected_coverage": "80-92%",
        }
    )

    # Edge cases
    scenarios.append(
        {
            "name": "Edge: Tiny n, 30 covariates for 50 observations, Strong confounding",
            "n": 50,
            "p": 30,
            "confounding_strength": 2.0,
            "expected_coverage": "75-90%",
        }
    )

    scenarios.append(
        {
            "name": "Robustness: Large n, Very high dim, Weak confounding",
            "n": 3000,
            "p": 50,
            "confounding_strength": 0.5,
            "expected_coverage": "92-97%",
        }
    )

    # Additional stress tests
    scenarios.append(
        {
            "name": "Stress: Medium n, Low dim, Extreme confounding",
            "n": 500,
            "p": 5,
            "confounding_strength": 4.0,
            "expected_coverage": "87-94%",
        }
    )

    scenarios.append(
        {
            "name": "Balanced: Large n, Medium dim, Medium confounding",
            "n": 2000,
            "p": 15,
            "confounding_strength": 1.0,
            "expected_coverage": "93-98%",
        }
    )

    scenarios.append(
        {
            "name": "Ultimate: Massive n, Low dim, Weak confounding",
            "n": 5000,
            "p": 5,
            "confounding_strength": 0.5,
            "expected_coverage": "94-99%",
        }
    )

    return scenarios


def coverage_exit_code(results: pd.DataFrame) -> int:
    """Return 1 for incomplete, 3 for undercoverage, 2 for excess, or 0 otherwise."""
    if results.empty or results["suite_coverage_status"].eq("INCOMPLETE").any():
        return 1
    if results["suite_coverage_status"].eq("UNDERCOVERAGE").any():
        return 3
    if results["suite_coverage_status"].eq("OVERCOVERAGE").any():
        return 2
    return 0


def run_coverage_stress_test(
    output_dir: str = "results/coverage_stress", n_simulations: int = 50, n_jobs: int = 1
) -> pd.DataFrame:
    """Save one row per planned scenario, including errors and Monte Carlo uncertainty."""
    if n_simulations < 1:
        raise ValueError("n_simulations must be positive")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    scenarios = create_test_scenarios()
    print(f"Testing {len(scenarios)} scenarios with {n_simulations} simulations each")
    results = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"[{i}/{len(scenarios)}] {scenario['name']}", flush=True)
        validator = BiasValidation(n_simulations=n_simulations, random_state=42, n_jobs=n_jobs)
        row = {
            "scenario_num": i,
            "scenario_name": scenario["name"],
            "n": scenario["n"],
            "p": scenario["p"],
            "confounding": scenario["confounding_strength"],
            "expected_coverage": scenario["expected_coverage"],  # Historical annotation
            "nominal_coverage": 1 - validator.alpha,
            "n_simulations": n_simulations,
            "n_covered": np.nan,
            "actual_coverage": np.nan,
            "coverage_pct": "ERROR",
            "coverage_mc_ci_lower": np.nan,
            "coverage_mc_ci_upper": np.nan,
            "bias": np.nan,
            "mse": np.nan,
            "status": "ERROR",
            "bias_status": "ERROR",
            "coverage_status": "ERROR",
            "bias_p_value": np.nan,
            "coverage_p_value": np.nan,
            "error": "",
        }
        try:
            dgp = DGPGenerator(
                n=scenario["n"],
                p=scenario["p"],
                true_effect=2.0,
                confounding_strength=scenario["confounding_strength"],
                random_state=42,
            )
            result = validator.validate(dgp)
            if result.n_simulations != n_simulations:
                raise ValueError("Incomplete simulation count")
            count = round(result.coverage * n_simulations)
            interval = binomtest(count, n_simulations).proportion_ci(0.95, method="exact")
            row.update(
                n_covered=count,
                actual_coverage=result.coverage,
                coverage_pct=f"{result.coverage:.1%}",
                coverage_mc_ci_lower=interval.low,
                coverage_mc_ci_upper=interval.high,
                bias=result.bias,
                mse=result.mse,
                status=result.status,
                bias_status=result.bias_status,
                coverage_status=result.coverage_status,
                bias_p_value=result.bias_p_value,
                coverage_p_value=result.coverage_p_value,
            )
        except Exception as err:
            row["error"] = str(err)
            print(f"  Fit error: {err}")
        results.append(row)

    df = pd.DataFrame(results)
    valid = df["coverage_p_value"].notna() & df["status"].ne("ERROR")
    # A missing fit still occupies its place in the planned family.
    adjusted = multipletests(df["coverage_p_value"].fillna(1.0), method="holm")[1]
    df["coverage_p_value_holm"] = np.where(valid, adjusted, np.nan)
    reject = valid & (df["coverage_p_value_holm"] <= 0.05)
    df["suite_coverage_status"] = "NO_REJECTION"
    df.loc[~valid, "suite_coverage_status"] = "INCOMPLETE"
    df.loc[reject & (df["actual_coverage"] < df["nominal_coverage"]), "suite_coverage_status"] = (
        "UNDERCOVERAGE"
    )
    df.loc[reject & (df["actual_coverage"] > df["nominal_coverage"]), "suite_coverage_status"] = (
        "OVERCOVERAGE"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_path / f"coverage_stress_test_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(
        df[
            [
                "scenario_name",
                "coverage_pct",
                "bias_p_value",
                "coverage_p_value",
                "coverage_p_value_holm",
                "suite_coverage_status",
            ]
        ].to_string(index=False)
    )
    print(f"Completed {valid.sum()}/{len(df)} scenarios. Results in {csv_path}")
    return df


if __name__ == "__main__":
    import sys

    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    output = sys.argv[2] if len(sys.argv) > 2 else "results/coverage_stress"
    workers = int(os.environ.get("DML_VALIDATION_DRAW_JOBS", "1"))
    sys.exit(coverage_exit_code(run_coverage_stress_test(output, n_sims, n_jobs=workers)))
