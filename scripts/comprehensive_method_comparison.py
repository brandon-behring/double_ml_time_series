"""
Comprehensive comparison analysis of all baseline methods and DML.

Generates:
1. Summary comparison tables
2. Method performance statistics
3. Comparison across multiple DGP configurations
4. Visual insights and key findings
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from dml_ts.validation.dgp_generator import DGPGenerator
from dml_ts.validation.baseline_comparison import BaselineComparison


def run_comprehensive_comparison(
    output_dir: str = "results/method_comparison",
    n_simulations: int = 20,
    include_ml: bool = True,
    include_dml: bool = True,
    random_state: int = 42,
):
    """
    Run comprehensive comparison of all methods.

    Args:
        output_dir: Directory to save results
        n_simulations: Number of Monte Carlo simulations per method
        include_ml: Whether to include ML baselines (RF, XGBoost)
        include_dml: Whether to include DML method
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with summary results and statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COMPREHENSIVE METHOD COMPARISON: DML vs Baselines")
    print("=" * 80)

    # Create comparison framework
    comp = BaselineComparison(
        n_simulations=n_simulations,
        random_state=random_state,
        include_ml=include_ml,
        include_dml=include_dml,
    )

    # Test configurations: vary confounding and sample size
    configs = [
        {
            "n": 500,
            "p": 5,
            "true_effect": 2.0,
            "confounding_strength": 0.5,
            "random_state": random_state,
        },
        {
            "n": 500,
            "p": 5,
            "true_effect": 2.0,
            "confounding_strength": 1.0,
            "random_state": random_state,
        },
        {
            "n": 500,
            "p": 5,
            "true_effect": 2.0,
            "confounding_strength": 2.0,
            "random_state": random_state,
        },
        {
            "n": 1000,
            "p": 5,
            "true_effect": 2.0,
            "confounding_strength": 0.5,
            "random_state": random_state,
        },
        {
            "n": 1000,
            "p": 5,
            "true_effect": 2.0,
            "confounding_strength": 1.0,
            "random_state": random_state,
        },
        {
            "n": 1000,
            "p": 5,
            "true_effect": 2.0,
            "confounding_strength": 2.0,
            "random_state": random_state,
        },
    ]

    print(f"\n📊 Configuration Sweep:")
    print(f"  - Configurations: {len(configs)}")
    print(f"  - Methods: {len(comp.methods)}")
    print(f"  - Simulations per method/config: {n_simulations}")
    print(f"  - Total method evaluations: {len(comp.methods) * len(configs) * n_simulations}")

    all_results = {}
    all_summaries = {}

    # Run comparisons across configurations
    for i, config in enumerate(configs, 1):
        config_str = comp._config_to_string(config)
        print(f"\n[{i}/{len(configs)}] Configuration: {config_str}")

        dgp = DGPGenerator(**config)

        # Compare all methods
        results = comp.compare(dgp)
        all_results[config_str] = results

        # Get summary statistics
        summary = comp.generate_summary_statistics(dgp)
        all_summaries[config_str] = summary

        # Print quick summary
        print(f"  Best bias method: {summary['best_bias_method']}")
        print(f"  Worst bias method: {summary['worst_bias_method']}")
        print(f"  Mean absolute bias: {summary['mean_absolute_bias']:.4f}")
        print(f"  Mean MSE: {summary['mean_mse']:.4f}")
        print(f"  Mean coverage: {summary['mean_coverage']:.1%}")

    # Aggregate results
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS")
    print("=" * 80)

    # Create comprehensive comparison table
    comparison_data = []
    for config_str, results in all_results.items():
        for method_name, result in results.items():
            comparison_data.append(
                {
                    "Configuration": config_str,
                    "Method": method_name,
                    "Bias": result.bias,
                    "Abs_Bias": abs(result.bias),
                    "MSE": result.mse,
                    "RMSE": np.sqrt(result.mse),
                    "Coverage": result.coverage,
                    "Status": result.status,
                    "P_Value": result.bias_p_value,
                }
            )

    comparison_table = pd.DataFrame(comparison_data)

    # Save comparison table
    comparison_table.to_csv(output_path / "method_comparison_detailed.csv", index=False)
    print(f"\n✅ Saved detailed comparison to: method_comparison_detailed.csv")

    # Summary statistics by method (across all configurations)
    method_summary = (
        comparison_table.groupby("Method")
        .agg(
            {
                "Bias": ["mean", "std"],
                "Abs_Bias": ["mean", "min", "max"],
                "MSE": ["mean", "std"],
                "Coverage": ["mean", "std"],
                "Status": lambda x: pd.Series(x).value_counts().to_dict(),
            }
        )
        .round(4)
    )

    print("\n" + "=" * 80)
    print("METHOD SUMMARY STATISTICS (across all configurations)")
    print("=" * 80)
    print(method_summary)

    method_summary.to_csv(output_path / "method_summary_statistics.csv")
    print(f"\n✅ Saved summary statistics to: method_summary_statistics.csv")

    # Performance by configuration
    config_performance = (
        comparison_table.groupby("Configuration")
        .agg(
            {
                "Bias": "mean",
                "Abs_Bias": "mean",
                "MSE": "mean",
                "Coverage": "mean",
            }
        )
        .round(4)
    )

    print("\n" + "=" * 80)
    print("PERFORMANCE BY CONFIGURATION")
    print("=" * 80)
    print(config_performance)

    config_performance.to_csv(output_path / "configuration_performance.csv")

    # Key findings
    findings = {
        "timestamp": datetime.now().isoformat(),
        "configurations": len(configs),
        "methods_compared": list(comp.methods.keys()),
        "simulations_per_method": n_simulations,
        "best_methods_by_metric": {
            "lowest_mean_bias": comparison_table.loc[
                comparison_table["Abs_Bias"].idxmin()
            ].to_dict(),
            "lowest_mse": comparison_table.loc[comparison_table["MSE"].idxmin()].to_dict(),
            "highest_coverage": comparison_table.loc[
                comparison_table["Coverage"].idxmax()
            ].to_dict(),
        },
        "method_statistics": {
            method: {
                "mean_abs_bias": comparison_table[comparison_table["Method"] == method][
                    "Abs_Bias"
                ].mean(),
                "mean_mse": comparison_table[comparison_table["Method"] == method]["MSE"].mean(),
                "mean_coverage": comparison_table[comparison_table["Method"] == method][
                    "Coverage"
                ].mean(),
                "pass_rate": (
                    (
                        comparison_table[comparison_table["Method"] == method]["Status"] == "PASS"
                    ).sum()
                    / len(comparison_table[comparison_table["Method"] == method])
                    * 100
                ),
            }
            for method in comparison_table["Method"].unique()
        },
    }

    # Save findings
    with open(output_path / "findings.json", "w") as f:
        json.dump(findings, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print(f"\n📊 Methods Compared: {', '.join(findings['methods_compared'])}")
    print(f"\n🏆 Best by Metric:")
    print(f"  - Lowest bias: {findings['best_methods_by_metric']['lowest_mean_bias']['Method']}")
    print(f"  - Lowest MSE: {findings['best_methods_by_metric']['lowest_mse']['Method']}")
    print(
        f"  - Highest coverage: {findings['best_methods_by_metric']['highest_coverage']['Method']}"
    )

    print(f"\n📈 Method Performance Summary:")
    for method, stats in findings["method_statistics"].items():
        print(f"\n  {method}:")
        print(f"    - Mean absolute bias: {stats['mean_abs_bias']:.4f}")
        print(f"    - Mean MSE: {stats['mean_mse']:.4f}")
        print(f"    - Mean coverage: {stats['mean_coverage']:.1%}")
        print(f"    - Pass rate: {stats['pass_rate']:.1f}%")

    # Save all summaries
    summaries_data = []
    for config_str, summary in all_summaries.items():
        summaries_data.append(
            {
                "Configuration": config_str,
                "Best_Bias_Method": summary["best_bias_method"],
                "Worst_Bias_Method": summary["worst_bias_method"],
                "Best_MSE_Method": summary["best_mse_method"],
                "Best_Coverage_Method": summary["best_coverage_method"],
                "Mean_Absolute_Bias": summary["mean_absolute_bias"],
                "Mean_MSE": summary["mean_mse"],
                "Mean_Coverage": summary["mean_coverage"],
            }
        )

    summaries_table = pd.DataFrame(summaries_data)
    summaries_table.to_csv(output_path / "configuration_summaries.csv", index=False)

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # Analyze patterns
    dml_included = "DML" in findings["methods_compared"]
    if dml_included:
        dml_stats = findings["method_statistics"]["DML"]
        print(f"\n✅ DML Performance Summary:")
        print(f"   - Mean absolute bias: {dml_stats['mean_abs_bias']:.4f}")
        print(f"   - Mean MSE: {dml_stats['mean_mse']:.4f}")
        print(f"   - Pass rate: {dml_stats['pass_rate']:.1f}%")

        # Compare DML vs best baseline
        baseline_methods = [
            m for m in findings["methods_compared"] if m not in ["DML", "RandomForest", "XGBoost"]
        ]
        if baseline_methods:
            best_baseline = max(
                baseline_methods,
                key=lambda m: findings["method_statistics"][m]["mean_abs_bias"],
            )
            best_baseline_bias = findings["method_statistics"][best_baseline]["mean_abs_bias"]
            dml_advantage = (
                (best_baseline_bias - dml_stats["mean_abs_bias"]) / best_baseline_bias * 100
            )
            print(
                f"\n   - vs Best Baseline ({best_baseline}): "
                f"{dml_advantage:+.1f}% bias reduction"
            )

    print("\n✅ Results saved to:", output_path)
    print("=" * 80)

    return {
        "comparison_table": comparison_table,
        "method_summary": method_summary,
        "findings": findings,
        "output_directory": str(output_path),
    }


if __name__ == "__main__":
    results = run_comprehensive_comparison(
        output_dir="results/method_comparison",
        n_simulations=20,
        include_ml=True,
        include_dml=True,
        random_state=42,
    )
