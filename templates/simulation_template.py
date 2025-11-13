"""
Template for Monte Carlo simulation scripts.

Use this template for standalone simulation studies or sensitivity analyses.
Optimized for parallel execution on 64-core Threadripper.

Usage:
    1. Copy this file: cp templates/simulation_template.py scripts/my_simulation.py
    2. Customize simulation parameters
    3. Implement your estimator in estimate_effect()
    4. Run: python scripts/my_simulation.py
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import json

from src.validation.dgp_generator import DGPGenerator


# =============================================================================
# Simulation Parameters
# =============================================================================

SIMULATION_NAME = "TEMPLATE_SIMULATION"  # TODO: Replace with your simulation name

# Monte Carlo settings
N_SIMULATIONS = 1000  # Number of Monte Carlo runs
RANDOM_STATE = 42  # Global random seed

# DGP parameters to vary (for sensitivity analysis)
SAMPLE_SIZES = [500, 1000, 2000, 5000]  # TODO: Customize
CONFOUNDER_COUNTS = [5, 10, 20]  # TODO: Customize
TRUE_EFFECTS = [1.0, 2.0, 3.0]  # TODO: Customize
CONFOUNDING_STRENGTHS = [0.0, 0.5, 1.0, 2.0]  # TODO: Customize

# Parallel execution (64-core Threadripper optimization)
N_JOBS = 48  # Leave 16 cores for system
USE_PARALLEL = True  # Set False for debugging

# Output settings
OUTPUT_DIR = Path("results/simulations")
SAVE_INDIVIDUAL_RUNS = False  # Save each Monte Carlo run (use for debugging)
SAVE_SUMMARY_ONLY = True  # Save aggregated results only


# =============================================================================
# Estimator Implementation
# =============================================================================


def estimate_effect(Y: np.ndarray, T: np.ndarray, X: np.ndarray) -> Dict[str, float]:
    """
    Estimate treatment effect from data.

    TODO: Implement your estimator here (DML, IPW, OLS, etc.)

    Args:
        Y: Outcome variable (n,)
        T: Treatment variable (n,)
        X: Confounders (n, p)

    Returns:
        Dictionary with:
        - 'estimate': Point estimate
        - 'se': Standard error
        - 'ci_lower': Lower CI bound
        - 'ci_upper': Upper CI bound
    """
    # TODO: Replace with your estimator
    # Example options:
    # 1. DML with EconML
    # 2. Naive OLS
    # 3. IPW
    # 4. DR-learner

    from sklearn.linear_model import LinearRegression

    # Placeholder: Simple OLS with controls
    model = LinearRegression()
    X_with_T = np.column_stack([T, X])
    model.fit(X_with_T, Y)

    estimate = model.coef_[0]

    # Placeholder SE (replace with proper calculation)
    residuals = Y - model.predict(X_with_T)
    mse = np.mean(residuals**2)
    se = np.sqrt(mse / len(Y))

    return {
        "estimate": estimate,
        "se": se,
        "ci_lower": estimate - 1.96 * se,
        "ci_upper": estimate + 1.96 * se,
    }


# =============================================================================
# Single Simulation Run
# =============================================================================


def run_single_simulation(
    n: int, p: int, true_effect: float, confounding_strength: float, sim_id: int
) -> Dict[str, Any]:
    """
    Run single Monte Carlo simulation.

    Args:
        n: Sample size
        p: Number of confounders
        true_effect: True treatment effect
        confounding_strength: Confounding strength
        sim_id: Simulation ID for reproducibility

    Returns:
        Dictionary with simulation results
    """
    # Generate data
    dgp = DGPGenerator(
        n=n,
        p=p,
        true_effect=true_effect,
        confounding_strength=confounding_strength,
        random_state=RANDOM_STATE + sim_id,
    )

    data = dgp.generate()

    # Estimate effect
    result = estimate_effect(data.Y, data.T, data.X)

    # Calculate metrics
    bias = result["estimate"] - true_effect
    covers = result["ci_lower"] <= true_effect <= result["ci_upper"]

    return {
        "sim_id": sim_id,
        "n": n,
        "p": p,
        "true_effect": true_effect,
        "confounding_strength": confounding_strength,
        "estimate": result["estimate"],
        "se": result["se"],
        "ci_lower": result["ci_lower"],
        "ci_upper": result["ci_upper"],
        "bias": bias,
        "squared_error": bias**2,
        "covers": covers,
    }


# =============================================================================
# Parameter Sweep
# =============================================================================


def run_parameter_sweep() -> pd.DataFrame:
    """
    Run simulations across all parameter combinations.

    Returns:
        DataFrame with all simulation results
    """
    print(f"Starting {SIMULATION_NAME}")
    total_combos = (
        len(SAMPLE_SIZES) * len(CONFOUNDER_COUNTS) * len(TRUE_EFFECTS) * len(CONFOUNDING_STRENGTHS)
    )
    print(f"Total combinations: {total_combos}")
    print(f"Monte Carlo runs per combination: {N_SIMULATIONS}")
    total_sims = total_combos * N_SIMULATIONS
    print(f"Total simulations: {total_sims:,}")
    print(f"Parallel: {USE_PARALLEL} (n_jobs={N_JOBS if USE_PARALLEL else 1})")
    print()

    all_results = []

    # Loop over parameter combinations
    for n in SAMPLE_SIZES:
        for p in CONFOUNDER_COUNTS:
            for true_effect in TRUE_EFFECTS:
                for confounding in CONFOUNDING_STRENGTHS:
                    print(f"Running: n={n}, p={p}, τ={true_effect}, " f"confounding={confounding}")

                    if USE_PARALLEL:
                        # Parallel execution
                        from joblib import Parallel, delayed

                        results = Parallel(n_jobs=N_JOBS)(
                            delayed(run_single_simulation)(n, p, true_effect, confounding, sim_id)
                            for sim_id in range(N_SIMULATIONS)
                        )
                    else:
                        # Sequential execution
                        results = [
                            run_single_simulation(n, p, true_effect, confounding, sim_id)
                            for sim_id in range(N_SIMULATIONS)
                        ]

                    all_results.extend(results)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    return df


# =============================================================================
# Results Aggregation
# =============================================================================


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate simulation results by parameter combination.

    Args:
        df: Individual simulation results

    Returns:
        Aggregated summary statistics
    """
    summary = (
        df.groupby(["n", "p", "true_effect", "confounding_strength"])
        .agg(
            mean_estimate=("estimate", "mean"),
            median_estimate=("estimate", "median"),
            sd_estimate=("estimate", "std"),
            mean_bias=("bias", "mean"),
            median_bias=("bias", "median"),
            rmse=("squared_error", lambda x: np.sqrt(np.mean(x))),
            coverage=("covers", "mean"),
            n_sims=("sim_id", "count"),
        )
        .reset_index()
    )

    return summary


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

    # Aggregate results
    print("\nAggregating results...")
    df_summary = aggregate_results(df_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if SAVE_INDIVIDUAL_RUNS:
        individual_path = OUTPUT_DIR / f"{SIMULATION_NAME}_individual_{timestamp}.csv"
        df_results.to_csv(individual_path, index=False)
        print(f"Saved individual runs: {individual_path}")

    if SAVE_SUMMARY_ONLY:
        summary_path = OUTPUT_DIR / f"{SIMULATION_NAME}_summary_{timestamp}.csv"
        df_summary.to_csv(summary_path, index=False)
        print(f"Saved summary: {summary_path}")

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

    print(f"Saved metadata: {metadata_path}")

    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time

    print()
    print("=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    print(f"Duration: {duration}")
    print(f"Total simulations: {len(df_results):,}")
    print(f"Parameter combinations: {len(df_summary)}")
    print()
    print("Summary statistics:")
    print(df_summary.to_string(index=False))
    print()


if __name__ == "__main__":
    main()
