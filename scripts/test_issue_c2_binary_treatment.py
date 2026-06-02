"""
Test script for Issue C2: Binary Treatment Mis-specification Impact Analysis

This script empirically tests whether using discrete_treatment=False with RandomForestRegressor
vs discrete_treatment=True with RandomForestClassifier has a material impact on:
- Bias
- MSE
- Coverage

Decision Criteria:
- Bias difference >5% → Fix is critical
- Coverage difference >10% → Fix is critical
- Otherwise → Document as limitation

Runtime: ~10-15 minutes (100 simulations × 2 configurations × 5 scenarios)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Literal

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dml_ts.validation.dgp_generator import DGPGenerator
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def run_dml_with_config(
    dgp: DGPGenerator,
    n_simulations: int,
    discrete_treatment: bool,
    config_name: str,
    random_state: int = 42,
) -> dict:
    """
    Run DML validation with specified configuration.

    Args:
        dgp: Data generating process
        n_simulations: Number of Monte Carlo simulations
        discrete_treatment: Whether to use discrete_treatment=True
        config_name: Name for this configuration
        random_state: Random seed

    Returns:
        Dictionary with bias, mse, coverage, and metadata
    """
    rng = np.random.RandomState(random_state)
    estimates = np.zeros(n_simulations)
    ci_bounds = np.zeros((n_simulations, 2))

    for i in range(n_simulations):
        # Generate data
        data = dgp.generate()

        # Configure models based on discrete_treatment setting
        model_y = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=10,
            random_state=random_state,
        )

        if discrete_treatment:
            # CORRECT: Use classifier for binary treatment
            model_t = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=10,
                random_state=random_state,
            )
        else:
            # CURRENT: Use regressor (incorrect for binary treatment)
            model_t = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=10,
                random_state=random_state,
            )

        # Fit DML
        dml = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=discrete_treatment,
            cv=5,
            random_state=random_state,
        )

        dml.fit(Y=data.Y, T=data.T, X=data.X)

        # Get estimate and CI
        ate = dml.effect(X=data.X).mean()
        ci_lower, ci_upper = dml.effect_interval(X=data.X, alpha=0.05)

        estimates[i] = ate
        ci_bounds[i, 0] = ci_lower.mean()
        ci_bounds[i, 1] = ci_upper.mean()

    # Calculate metrics
    bias = float(np.mean(estimates) - dgp.true_effect)
    mse = float(np.mean((estimates - dgp.true_effect) ** 2))
    rmse = float(np.sqrt(mse))

    # Coverage: proportion of CIs containing true effect
    covers = (ci_bounds[:, 0] <= dgp.true_effect) & (dgp.true_effect <= ci_bounds[:, 1])
    coverage = float(np.mean(covers))

    return {
        "configuration": config_name,
        "discrete_treatment": discrete_treatment,
        "bias": bias,
        "abs_bias": abs(bias),
        "mse": mse,
        "rmse": rmse,
        "coverage": coverage,
        "n_simulations": n_simulations,
        "dgp_n": dgp.n,
        "dgp_p": dgp.p,
        "dgp_true_effect": dgp.true_effect,
        "dgp_confounding": dgp.confounding_strength,
    }


def main():
    """Run comprehensive comparison across multiple scenarios."""

    print("=" * 80)
    print("ISSUE C2: BINARY TREATMENT MIS-SPECIFICATION IMPACT ANALYSIS")
    print("=" * 80)
    print()
    print("Testing whether discrete_treatment setting materially affects results...")
    print()

    # Define test scenarios (smaller n_simulations for speed)
    scenarios = [
        # (n, p, true_effect, confounding_strength, description)
        (500, 5, 2.0, 0.5, "Easy: Large n, few confounders, weak confounding"),
        (500, 10, 2.0, 1.0, "Moderate: Medium confounding, more covariates"),
        (200, 10, 2.0, 2.0, "Challenging: Smaller n, strong confounding"),
        (500, 5, 0.5, 1.0, "Small effect: Harder to detect"),
        (1000, 5, 2.0, 0.5, "Large sample: Should reduce variance"),
    ]

    n_simulations = 100  # Reduced for speed (full analysis would use 1000)
    random_state = 42

    results = []

    for i, (n, p, true_effect, confounding, description) in enumerate(scenarios, 1):
        print(f"\n{'─' * 80}")
        print(f"Scenario {i}/{len(scenarios)}: {description}")
        print(f"  n={n}, p={p}, true_effect={true_effect}, confounding={confounding}")
        print(f"{'─' * 80}")

        # Create DGP
        dgp = DGPGenerator(
            n=n,
            p=p,
            true_effect=true_effect,
            confounding_strength=confounding,
            random_state=random_state,
        )

        # Test CURRENT configuration (discrete_treatment=False + Regressor)
        print("  Testing CURRENT config (discrete_treatment=False + Regressor)...")
        current_result = run_dml_with_config(
            dgp,
            n_simulations,
            discrete_treatment=False,
            config_name="CURRENT",
            random_state=random_state,
        )
        results.append(current_result)
        print(f"    Bias: {current_result['bias']:.4f}, Coverage: {current_result['coverage']:.2%}")

        # Test CORRECTED configuration (discrete_treatment=True + Classifier)
        print("  Testing CORRECTED config (discrete_treatment=True + Classifier)...")
        corrected_result = run_dml_with_config(
            dgp,
            n_simulations,
            discrete_treatment=True,
            config_name="CORRECTED",
            random_state=random_state,
        )
        results.append(corrected_result)
        print(
            f"    Bias: {corrected_result['bias']:.4f}, Coverage: {corrected_result['coverage']:.2%}"
        )

        # Calculate differences
        bias_diff = abs(corrected_result["bias"] - current_result["bias"])
        bias_pct_change = (
            (bias_diff / abs(current_result["bias"])) * 100 if current_result["bias"] != 0 else 0
        )
        coverage_diff = abs(corrected_result["coverage"] - current_result["coverage"])
        coverage_pct_diff = coverage_diff * 100  # Already in [0,1], convert to percentage points

        print(f"  Differences:")
        print(f"    Bias: {bias_diff:.4f} (change: {bias_pct_change:.1f}%)")
        print(f"    Coverage: {coverage_diff:.2%} ({coverage_pct_diff:.1f} percentage points)")

        # Decision criteria
        bias_critical = bias_pct_change > 5.0
        coverage_critical = coverage_pct_diff > 10.0

        if bias_critical or coverage_critical:
            print(f"  ⚠️  CRITICAL: Fix is needed (bias>{5}% or coverage>{10}pp)")
        else:
            print(f"  ✓ Difference within acceptable range")

    # Save results
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(__file__).parent.parent / "results" / "impact_analysis"
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / f"issue_c2_impact_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 80}")
    print("SUMMARY & DECISION")
    print(f"{'=' * 80}\n")

    # Group by configuration and calculate averages
    summary = (
        df.groupby("configuration")
        .agg(
            {
                "bias": "mean",
                "abs_bias": "mean",
                "coverage": "mean",
                "mse": "mean",
            }
        )
        .round(4)
    )

    print("Average metrics across all scenarios:")
    print(summary)
    print()

    # Overall decision
    current_avg = summary.loc["CURRENT"]
    corrected_avg = summary.loc["CORRECTED"]

    overall_bias_diff = abs(corrected_avg["bias"] - current_avg["bias"])
    overall_coverage_diff = abs(corrected_avg["coverage"] - current_avg["coverage"])

    print(f"Overall differences:")
    print(f"  Average bias difference: {overall_bias_diff:.4f}")
    print(f"  Average coverage difference: {overall_coverage_diff:.2%}")
    print()

    # Decision
    if overall_bias_diff / abs(current_avg["bias"]) > 0.05 or overall_coverage_diff > 0.10:
        decision = "FIX REQUIRED"
        priority = "HIGH"
        reasoning = "Differences exceed decision criteria (>5% bias or >10% coverage)"
    else:
        decision = "DOCUMENT AS LIMITATION"
        priority = "LOW"
        reasoning = "Differences within acceptable range for teaching purposes"

    print(f"DECISION: {decision}")
    print(f"PRIORITY: {priority}")
    print(f"REASONING: {reasoning}")
    print()
    print(f"Results saved to: {csv_path}")
    print()

    return df, decision, priority


if __name__ == "__main__":
    results_df, decision, priority = main()
