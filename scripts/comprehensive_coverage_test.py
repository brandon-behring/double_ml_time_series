"""
Comprehensive coverage stress testing across challenging scenarios.

Tests coverage rate across 15 scenarios varying in:
- Sample size (n=100 to 5000)
- Dimensionality (p=5 to 50)
- Confounding strength (0.5 to 3.0)

Expected: Coverage should be 92-98%, NOT 100%.
If coverage is 100% in hard scenarios, CIs are too wide (overconservative).
If coverage is <90%, CIs are too narrow (under-coverage).
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from typing import Any
from src.validation.dgp_generator import DGPGenerator
from src.validation.bias_validation import BiasValidation


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
            "name": "Extreme: Very small n, High dim (p>n), Medium confounding",
            "n": 100,
            "p": 20,
            "confounding_strength": 1.0,
            "expected_coverage": "80-92%",
        }
    )

    # Edge cases
    scenarios.append(
        {
            "name": "Edge: Tiny n, High dim (p>>n), Strong confounding",
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


def run_coverage_stress_test(
    output_dir: str = "results/coverage_stress", n_simulations: int = 50
) -> pd.DataFrame:
    """
    Run comprehensive coverage stress test across all scenarios.

    Args:
        output_dir: Where to save results
        n_simulations: Number of MC runs per scenario (default: 50)

    Returns:
        DataFrame with results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("COMPREHENSIVE COVERAGE STRESS TEST")
    print("=" * 100)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Simulations per scenario: {n_simulations}")
    print(f"Total scenarios: 15")
    print(f"Total DML fits: 15 × {n_simulations} = {15 * n_simulations}")
    print("\n" + "=" * 100)

    scenarios = create_test_scenarios()
    results = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/15] Testing: {scenario['name']}")
        print(
            f"  Config: n={scenario['n']}, p={scenario['p']}, confounding={scenario['confounding_strength']}"
        )
        print(f"  Expected coverage: {scenario['expected_coverage']}")

        # Create DGP
        dgp = DGPGenerator(
            n=scenario["n"],
            p=scenario["p"],
            true_effect=2.0,
            confounding_strength=scenario["confounding_strength"],
            random_state=42,
        )

        # Run validation
        validator = BiasValidation(n_simulations=n_simulations, random_state=42)

        try:
            result = validator.validate(dgp)

            # Record results
            results.append(
                {
                    "scenario_num": i,
                    "scenario_name": scenario["name"],
                    "n": scenario["n"],
                    "p": scenario["p"],
                    "confounding": scenario["confounding_strength"],
                    "expected_coverage": scenario["expected_coverage"],
                    "actual_coverage": result.coverage,
                    "coverage_pct": f"{result.coverage:.1%}",
                    "bias": result.bias,
                    "mse": result.mse,
                    "status": result.status,
                    "p_value": result.bias_p_value,
                    "n_simulations": n_simulations,
                }
            )

            # Interpret coverage
            coverage_pct = result.coverage * 100
            if coverage_pct > 98:
                interpretation = "⚠️  OVERCONSERVATIVE (CIs too wide)"
            elif coverage_pct < 90:
                interpretation = "❌ UNDERCOVERAGE (CIs too narrow)"
            else:
                interpretation = "✅ GOOD"

            print(f"  Actual coverage: {result.coverage:.1%} {interpretation}")
            print(f"  Bias: {result.bias:.4f}, MSE: {result.mse:.4f}, Status: {result.status}")

        except Exception as e:
            print(f"  ❌ ERROR: {str(e)}")
            results.append(
                {
                    "scenario_num": i,
                    "scenario_name": scenario["name"],
                    "n": scenario["n"],
                    "p": scenario["p"],
                    "confounding": scenario["confounding_strength"],
                    "expected_coverage": scenario["expected_coverage"],
                    "actual_coverage": np.nan,
                    "coverage_pct": "ERROR",
                    "bias": np.nan,
                    "mse": np.nan,
                    "status": "ERROR",
                    "p_value": np.nan,
                    "n_simulations": n_simulations,
                    "error": str(e),
                }
            )

    # Create results DataFrame
    df = pd.DataFrame(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_path / f"coverage_stress_test_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    # Generate summary report
    print("\n" + "=" * 100)
    print("SUMMARY REPORT")
    print("=" * 100)

    valid_results = df[df["actual_coverage"].notna()]

    if len(valid_results) > 0:
        print(f"\nCoverage Statistics:")
        print(f"  Mean coverage: {valid_results['actual_coverage'].mean():.1%}")
        print(f"  Median coverage: {valid_results['actual_coverage'].median():.1%}")
        print(f"  Min coverage: {valid_results['actual_coverage'].min():.1%}")
        print(f"  Max coverage: {valid_results['actual_coverage'].max():.1%}")
        print(f"  Std coverage: {valid_results['actual_coverage'].std():.1%}")

        # Categorize scenarios
        good = (valid_results["actual_coverage"] >= 0.90) & (
            valid_results["actual_coverage"] <= 0.98
        )
        overconservative = valid_results["actual_coverage"] > 0.98
        undercoverage = valid_results["actual_coverage"] < 0.90

        print(f"\nCoverage Quality:")
        print(f"  ✅ Good (90-98%): {good.sum()}/{len(valid_results)} scenarios")
        print(
            f"  ⚠️  Overconservative (>98%): {overconservative.sum()}/{len(valid_results)} scenarios"
        )
        print(f"  ❌ Undercoverage (<90%): {undercoverage.sum()}/{len(valid_results)} scenarios")

        if overconservative.any():
            print(f"\nOverconservative scenarios:")
            for _, row in valid_results[overconservative].iterrows():
                print(f"  - {row['scenario_name']}: {row['coverage_pct']}")

        if undercoverage.any():
            print(f"\nUndercoverage scenarios:")
            for _, row in valid_results[undercoverage].iterrows():
                print(f"  - {row['scenario_name']}: {row['coverage_pct']}")

        # Overall assessment
        print(f"\n" + "=" * 100)
        print("OVERALL ASSESSMENT")
        print("=" * 100)

        if all(good):
            print("✅ EXCELLENT: All scenarios have appropriate coverage (90-98%)")
        elif overconservative.sum() > len(valid_results) / 2:
            print("⚠️  WARNING: More than half of scenarios show overconservative coverage")
            print("   This suggests CIs are too wide. Investigate CI calculation.")
        elif undercoverage.sum() > len(valid_results) / 3:
            print("❌ CRITICAL: More than 1/3 of scenarios show undercoverage")
            print("   This suggests CIs are too narrow. DML assumptions may be violated.")
        else:
            print("✅ ACCEPTABLE: Most scenarios have reasonable coverage")
            print("   Some deviations expected in extreme scenarios")

    print(f"\n✅ Results saved to: {csv_path}")
    print("=" * 100)

    return df


if __name__ == "__main__":
    import sys

    # Allow command-line arguments
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    output = sys.argv[2] if len(sys.argv) > 2 else "results/coverage_stress"

    results = run_coverage_stress_test(output_dir=output, n_simulations=n_sims)

    # Exit with code based on results
    valid = results[results["actual_coverage"].notna()]
    if len(valid) == 0:
        sys.exit(1)  # All scenarios failed

    overconservative = (valid["actual_coverage"] > 0.98).sum()
    undercoverage = (valid["actual_coverage"] < 0.90).sum()

    if overconservative > len(valid) / 2:
        sys.exit(2)  # Too many overconservative
    elif undercoverage > len(valid) / 3:
        sys.exit(3)  # Too many undercoverage
    else:
        sys.exit(0)  # Acceptable
