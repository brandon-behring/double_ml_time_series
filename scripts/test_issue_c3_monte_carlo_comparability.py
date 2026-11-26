"""Measure rank variability across seeds when five estimators receive separate iid draws.

The CLI replays the original unmatched design. It cannot isolate how much rank
variation is attributable to the data-sharing rule. BaselineComparison.compare
supports a matched-draw comparison for that separate question.
"""

import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd

# Run directly from the repository root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dml_ts.validation.bias_validation import BiasValidation
from dml_ts.validation.dgp_generator import DGPGenerator
from dml_ts.validation.ipw_baseline import AugmentedIPW, IPWEstimator
from dml_ts.validation.ols_baseline import NaiveOLS, OLSWithControls


def run_comparison_current(dgp: DGPGenerator, methods: dict, n_simulations: int) -> dict:
    """
    Historical approach: Each method calls dgp.generate() independently.
    This means each method sees DIFFERENT random draws.
    """
    results = {}
    for name, method in methods.items():
        if method.n_simulations != n_simulations:
            raise ValueError(f"{name} must use {n_simulations} simulations")
        result = method.validate(dgp)
        results[name] = {
            "bias": result.bias,
            "abs_bias": abs(result.bias),
            "mse": result.mse,
            "coverage": result.coverage,
        }
    return results


def run_comparison_corrected(dgp_config: dict, methods: dict, n_simulations: int) -> dict:
    """
    Run each public validator on a fresh copy of one generator state.
    """
    # Each copy starts with the same generator state. The public validator computes
    # per-fit bounds for each supported method, preserving its coverage definition.
    generator = DGPGenerator(**dgp_config)
    results = {}
    for name, method in methods.items():
        if method.n_simulations != n_simulations:
            raise ValueError(f"{name} must use {n_simulations} simulations")
        fit = method.validate(deepcopy(generator))
        results[name] = {
            "bias": fit.bias,
            "abs_bias": abs(fit.bias),
            "mse": fit.mse,
            "coverage": fit.coverage,
        }

    return results


def compare_rankings(current: dict, corrected: dict) -> tuple:
    """
    Compare method rankings by bias between two approaches.

    Returns:
        (rankings_match, max_rank_change, relative_bias_changes)
    """
    # Get rankings by absolute bias
    current_ranking = sorted(current.items(), key=lambda x: x[1]["abs_bias"])
    corrected_ranking = sorted(corrected.items(), key=lambda x: x[1]["abs_bias"])

    current_names = [name for name, _ in current_ranking]
    corrected_names = [name for name, _ in corrected_ranking]

    # Calculate rank changes
    rank_changes = []
    for name in current_names:
        if name in corrected_names:
            current_rank = current_names.index(name)
            corrected_rank = corrected_names.index(name)
            rank_changes.append(abs(current_rank - corrected_rank))

    max_rank_change = max(rank_changes) if rank_changes else 0
    rankings_match = current_names == corrected_names

    # Calculate relative bias differences
    relative_diffs = {}
    for name in current.keys():
        if name in corrected:
            current_bias = current[name]["bias"]
            corrected_bias = corrected[name]["bias"]
            if abs(current_bias) > 1e-10:
                rel_diff = abs((corrected_bias - current_bias) / current_bias) * 100
            else:
                rel_diff = 0.0
            relative_diffs[name] = rel_diff

    return rankings_match, max_rank_change, relative_diffs


def main():
    """Run comparison stability analysis across multiple scenarios."""

    print("=" * 80)
    print("ISSUE C3: MONTE CARLO COMPARABILITY IMPACT ANALYSIS")
    print("=" * 80)
    print()
    print("Measuring rank variation across seeds under the historical unmatched-draw design.")
    print()

    # Test scenarios
    scenarios = [
        {
            "n": 500,
            "p": 5,
            "true_effect": 2.0,
            "confounding_strength": 1.0,
            "random_state": 42,
            "description": "Moderate confounding",
        },
        {
            "n": 200,
            "p": 10,
            "true_effect": 2.0,
            "confounding_strength": 2.0,
            "random_state": 42,
            "description": "Strong confounding, smaller sample",
        },
        {
            "n": 1000,
            "p": 5,
            "true_effect": 1.0,
            "confounding_strength": 0.5,
            "random_state": 42,
            "description": "Weak confounding, large sample",
        },
    ]

    n_simulations = 100
    n_runs = 10  # Run multiple times to see if rankings are stable

    all_results = []

    for scenario_idx, scenario_config in enumerate(scenarios, 1):
        print(f"\n{'─' * 80}")
        print(f"Scenario {scenario_idx}/{len(scenarios)}: {scenario_config['description']}")
        print(
            f"  n={scenario_config['n']}, p={scenario_config['p']}, "
            f"confounding={scenario_config['confounding_strength']}"
        )
        print(f"{'─' * 80}\n")

        # Initialize methods (with different random states for each run)
        for run in range(n_runs):
            run_seed = scenario_config["random_state"] + run

            methods = {
                "NaiveOLS": NaiveOLS(n_simulations=n_simulations, random_state=run_seed),
                "OLSWithControls": OLSWithControls(
                    n_simulations=n_simulations, random_state=run_seed
                ),
                "IPWEstimator": IPWEstimator(n_simulations=n_simulations, random_state=run_seed),
                "AugmentedIPW": AugmentedIPW(n_simulations=n_simulations, random_state=run_seed),
                "DML": BiasValidation(n_simulations=n_simulations, random_state=run_seed),
            }

            # Start the historical unmatched design at the specified seed.
            dgp_current = DGPGenerator(
                n=scenario_config["n"],
                p=scenario_config["p"],
                true_effect=scenario_config["true_effect"],
                confounding_strength=scenario_config["confounding_strength"],
                random_state=run_seed,
            )

            # Each validator advances the common generator.
            current_results = run_comparison_current(dgp_current, methods, n_simulations)

            # Get rankings
            current_ranking = sorted(current_results.items(), key=lambda x: x[1]["abs_bias"])

            print(f"  Run {run + 1}/{n_runs}: Unmatched-draw ranking (by absolute bias)")
            for rank, (name, metrics) in enumerate(current_ranking, 1):
                print(
                    f"    {rank}. {name:20s} bias={metrics['bias']:+.4f}, mse={metrics['mse']:.4f}"
                )

            # Store results
            for rank, (name, metrics) in enumerate(current_ranking, 1):
                all_results.append(
                    {
                        "scenario": scenario_config["description"],
                        "run": run + 1,
                        "method": name,
                        "rank": rank,
                        "bias": metrics["bias"],
                        "abs_bias": metrics["abs_bias"],
                        "mse": metrics["mse"],
                        "coverage": metrics["coverage"],
                    }
                )

        print()

    # Analyze ranking stability across runs
    df = pd.DataFrame(all_results)

    print(f"{'=' * 80}")
    print("RANKING STABILITY ANALYSIS")
    print(f"{'=' * 80}\n")

    for scenario_desc in df["scenario"].unique():
        scenario_df = df[df["scenario"] == scenario_desc]

        print(f"Scenario: {scenario_desc}")
        print(f"{'─' * 40}")

        # Calculate rank variance for each method
        rank_stats = scenario_df.groupby("method")["rank"].agg(["mean", "std", "min", "max"])
        rank_stats = rank_stats.sort_values("mean")

        print("Method rankings across runs (mean ± std):")
        for method, stats in rank_stats.iterrows():
            print(
                f"  {method:20s}: {stats['mean']:.1f} ± {stats['std']:.2f} "
                f"(range: {int(stats['min'])}-{int(stats['max'])})"
            )

        # Check if any method's rank changed by >1
        max_rank_sd = rank_stats["std"].max()
        max_rank_range = (rank_stats["max"] - rank_stats["min"]).max()

        if max_rank_sd > 1.0 or max_rank_range > 2:
            print(f"\n  ⚠️  HIGH INSTABILITY: Rankings vary by up to {max_rank_range:.0f} positions")
        else:
            print(f"\n  ✓ Rankings stable across runs (maximum rank SD: {max_rank_sd:.2f})")

        print()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(__file__).parent.parent / "results" / "impact_analysis"
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / f"issue_c3_impact_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    print(f"{'=' * 80}")
    print("UNMATCHED-DRAW RANK SUMMARY")
    print(f"{'=' * 80}\n")

    # Descriptive rank spread across runs.
    overall_rank_std = df.groupby(["scenario", "method"])["rank"].std().max()
    overall_rank_range = (
        df.groupby(["scenario", "method"])["rank"].max()
        - df.groupby(["scenario", "method"])["rank"].min()
    ).max()

    print("Overall ranking stability:")
    print(f"  Max std deviation in ranks: {overall_rank_std:.2f}")
    print(f"  Max rank range: {overall_rank_range:.0f} positions")
    print()

    description = (
        "large rank variation"
        if overall_rank_std > 1.0 or overall_rank_range > 2
        else "small rank variation"
    )
    print(f"Historical unmatched-draw design: {description}")
    print("These runs do not quantify the change in rank variability from matching the draws.")
    print(f"Results saved to: {csv_path}")
    return df, description, "descriptive"


if __name__ == "__main__":
    main()
