"""
Bias Validation Simulation Study.

Evaluates DML estimator bias across varying sample sizes, confounder counts,
true effects, and confounding strengths. Uses BiasValidation class directly.

Usage:
    python scripts/bias_validation_simulation.py

Results saved to: results/simulations/BIAS_VALIDATION_summary_*.csv
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import json

from src.validation.dgp_generator import DGPGenerator
from src.validation.bias_validation import BiasValidation


# =============================================================================
# Simulation Parameters
# =============================================================================

SIMULATION_NAME = "BIAS_VALIDATION"

# Monte Carlo settings
N_SIMULATIONS = 100  # Runs per combination (use 1000+ for final results)
RANDOM_STATE = 42  # Global random seed

# DGP parameters to vary
SAMPLE_SIZES = [500, 1000, 2000]  # Sample sizes
CONFOUNDER_COUNTS = [5, 10, 20]  # Number of confounders
TRUE_EFFECTS = [1.0, 2.0, 3.0]  # True treatment effects
CONFOUNDING_STRENGTHS = [0.5, 1.0, 2.0]  # Confounding strength

# Parallel execution (64-core Threadripper optimization)
N_JOBS = 48  # Leave 16 cores for system
USE_PARALLEL = True  # Set False for debugging

# Output settings
OUTPUT_DIR = Path("results/simulations")
SAVE_INDIVIDUAL_RUNS = False  # Save each Monte Carlo run
SAVE_SUMMARY_ONLY = True  # Save aggregated results only


# =============================================================================
# Simulation Run (Using BiasValidation)
# =============================================================================


def run_parameter_combination(
    n: int, p: int, true_effect: float, confounding_strength: float
) -> Dict[str, Any]:
    """
    Run BiasValidation for a single parameter combination.

    Args:
        n: Sample size
        p: Number of confounders
        true_effect: True treatment effect
        confounding_strength: Confounding strength

    Returns:
        Dictionary with validation results
    """
    # Create DGP
    dgp = DGPGenerator(
        n=n,
        p=p,
        true_effect=true_effect,
        confounding_strength=confounding_strength,
        random_state=RANDOM_STATE,
    )

    # Run validation
    validator = BiasValidation(n_simulations=N_SIMULATIONS, random_state=RANDOM_STATE)
    result = validator.validate(dgp)

    # Extract metrics
    return {
        "n": n,
        "p": p,
        "true_effect": true_effect,
        "confounding_strength": confounding_strength,
        "bias": result.bias,
        "mse": result.mse,
        "rmse": np.sqrt(result.mse),
        "coverage": result.coverage,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "status": result.status,
        "n_simulations": result.n_simulations,
    }


# =============================================================================
# Parameter Sweep
# =============================================================================


def run_parameter_sweep() -> pd.DataFrame:
    """
    Run BiasValidation across all parameter combinations.

    Returns:
        DataFrame with validation results for each combination
    """
    print(f"Starting {SIMULATION_NAME}")
    total_combos = (
        len(SAMPLE_SIZES) * len(CONFOUNDER_COUNTS) * len(TRUE_EFFECTS) * len(CONFOUNDING_STRENGTHS)
    )
    print(f"Total combinations: {total_combos}")
    print(f"Monte Carlo runs per combination: {N_SIMULATIONS}")
    total_sims = total_combos * N_SIMULATIONS
    print(f"Total DML fits: {total_sims:,}")
    print()

    all_results = []

    # Loop over parameter combinations
    for n in SAMPLE_SIZES:
        for p in CONFOUNDER_COUNTS:
            for true_effect in TRUE_EFFECTS:
                for confounding in CONFOUNDING_STRENGTHS:
                    print(
                        f"Running: n={n}, p={p}, τ={true_effect}, confounding={confounding}",
                        flush=True,
                    )

                    # Run validation for this combination
                    result = run_parameter_combination(n, p, true_effect, confounding)
                    all_results.append(result)

                    # Print status
                    status_symbol = (
                        "✅"
                        if result["status"] == "PASS"
                        else "⚠️"
                        if result["status"] == "WARNING"
                        else "❌"
                    )
                    print(
                        f"  {status_symbol} Bias={result['bias']:.4f}, RMSE={result['rmse']:.4f}, Status={result['status']}"
                    )

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    return df


# =============================================================================
# Results Summary
# =============================================================================


def summarize_results(df: pd.DataFrame) -> None:
    """
    Print summary statistics of validation results.

    Args:
        df: DataFrame with validation results
    """
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    # Overall statistics
    print(f"\nTotal combinations tested: {len(df)}")
    print(f"Total DML fits: {df['n_simulations'].sum():,}")

    # Status breakdown
    print("\nStatus Breakdown:")
    status_counts = df["status"].value_counts()
    for status, count in status_counts.items():
        pct = 100 * count / len(df)
        print(f"  {status}: {count} ({pct:.1f}%)")

    # Bias statistics
    print(f"\nBias Statistics:")
    print(f"  Mean bias: {df['bias'].mean():.6f}")
    print(f"  Median bias: {df['bias'].median():.6f}")
    print(f"  Max |bias|: {df['bias'].abs().max():.6f}")
    print(f"  Min |bias|: {df['bias'].abs().min():.6f}")

    # RMSE statistics
    print(f"\nRMSE Statistics:")
    print(f"  Mean RMSE: {df['rmse'].mean():.6f}")
    print(f"  Median RMSE: {df['rmse'].median():.6f}")

    print("\n" + "=" * 80)


# =============================================================================
# Main Execution
# =============================================================================


def main() -> None:
    """Run complete simulation study."""
    start_time = datetime.now()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run simulations
    print("=" * 80)
    print(f"SIMULATION: {SIMULATION_NAME}")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    df_results = run_parameter_sweep()

    # Print summary
    summarize_results(df_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = OUTPUT_DIR / f"{SIMULATION_NAME}_results_{timestamp}.csv"
    df_results.to_csv(summary_path, index=False)
    print(f"\n✅ Saved results: {summary_path}")

    # Save metadata
    metadata = {
        "simulation_name": SIMULATION_NAME,
        "n_simulations": N_SIMULATIONS,
        "sample_sizes": SAMPLE_SIZES,
        "confounder_counts": CONFOUNDER_COUNTS,
        "true_effects": TRUE_EFFECTS,
        "confounding_strengths": CONFOUNDING_STRENGTHS,
        "n_jobs": N_JOBS if USE_PARALLEL else 1,
        "random_state": RANDOM_STATE,
        "start_time": start_time.isoformat(),
        "end_time": datetime.now().isoformat(),
        "duration_seconds": (datetime.now() - start_time).total_seconds(),
    }

    metadata_path = OUTPUT_DIR / f"{SIMULATION_NAME}_metadata_{timestamp}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved metadata: {metadata_path}")

    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time

    print()
    print("=" * 80)
    print("✅ SIMULATION COMPLETE")
    print("=" * 80)
    print(f"Duration: {duration}")
    print(f"Parameter combinations: {len(df_results)}")
    print(f"Total DML fits: {df_results['n_simulations'].sum():,}")
    print()
    print(f"Results saved to: {summary_path}")
    print(f"Metadata saved to: {metadata_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
