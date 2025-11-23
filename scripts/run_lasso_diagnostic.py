"""
Execute comprehensive Lasso diagnostic for 401(k) mismatch investigation.

Days 2-4 of comprehensive remediation plan: Diagnose why Lasso ATE
is -44.4% different from published estimate.

Problem:
    - Published ATE: $9,580
    - Our ATE: $5,330 (-44.4% difference)
    - SE: $6,930 (130% of point estimate!)
    - CI: $-8,253 to $18,913 (spans $27,167)

This script runs all diagnostic analyses and generates a comprehensive report.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.lasso_diagnostic import LassoDiagnostic


def main():
    """Run comprehensive Lasso diagnostic and display results."""
    print("=" * 80)
    print("401(k) LASSO DIAGNOSTIC ANALYSIS")
    print("DAYS 2-4: Investigating -44.4% Mismatch")
    print("=" * 80)
    print()
    print("This diagnostic will take approximately 30-45 minutes due to:")
    print("  - 1000 bootstrap samples (~20 minutes)")
    print("  - Hyperparameter sensitivity analysis (~15 minutes)")
    print("  - Cross-fitting seed sensitivity (~10 minutes)")
    print()

    # Initialize diagnostic
    diagnostic = LassoDiagnostic(random_state=42, verbose=True)

    # Run comprehensive analysis
    results = diagnostic.run_comprehensive_diagnostic(
        n_bootstrap=1000,  # 1000 bootstrap samples for robust distribution
        n_seeds=20,  # 20 different random seeds for cross-fitting
    )

    # Display summary
    print("=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print()

    # Bootstrap results
    boot = results.bootstrap_diagnostic
    print("BOOTSTRAP DISTRIBUTION:")
    print(f"  Mean ATE: ${boot.mean_ate:,.2f} ± ${boot.std_ate:,.2f}")
    print(f"  95% CI: (${boot.ci_lower:,.2f}, ${boot.ci_upper:,.2f})")
    print(f"  Normality: {'PASS' if boot.is_normal else 'FAIL'} (p={boot.normality_pvalue:.4f})")
    print(
        f"  Outliers: {boot.n_outliers}/{boot.n_bootstrap} ({100*boot.n_outliers/boot.n_bootstrap:.1f}%)"
    )
    print(f"  Converged: {'YES' if boot.converged else 'NO'}")
    print()

    # Hyperparameter sensitivity
    print("HYPERPARAMETER SENSITIVITY:")
    for param_name, param_result in results.hyperparameter_sensitivity.items():
        print(f"  {param_name}:")
        print(
            f"    Sensitivity: {param_result.sensitivity_score:.3f} ({'HIGH' if param_result.is_sensitive else 'LOW'})"
        )
        print(f"    Recommended: {param_result.recommended_value}")
        print(
            f"    ATE Range: ${param_result.metadata['min_ate']:,.2f} to ${param_result.metadata['max_ate']:,.2f}"
        )
    print()

    # Seed sensitivity
    seed = results.seed_sensitivity
    print("CROSS-FITTING SEED SENSITIVITY:")
    print(f"  Mean ATE: ${seed.mean_ate:,.2f} ± ${seed.std_ate:,.2f}")
    print(f"  Range: ${seed.min_ate:,.2f} to ${seed.max_ate:,.2f} (span: ${seed.range_ate:,.2f})")
    print(f"  CV: {seed.cv_ate:.3f} ({'STABLE' if seed.is_stable else 'UNSTABLE'})")
    print()

    # Root cause
    print("ROOT CAUSE ANALYSIS:")
    print(f"  {results.root_cause_analysis}")
    print()

    # Recommendations
    print("RECOMMENDATIONS:")
    for i, rec in enumerate(results.recommendations, 1):
        print(f"  {i}. {rec}")
    print()

    # Save results to file
    output_path = (
        Path("results")
        / "lasso_diagnostic"
        / f"diagnostic_results_{results.timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("401(k) LASSO DIAGNOSTIC ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {results.timestamp}\n\n")

        f.write("BOOTSTRAP DISTRIBUTION:\n")
        f.write(f"  Mean ATE: ${boot.mean_ate:,.2f} ± ${boot.std_ate:,.2f}\n")
        f.write(f"  95% CI: (${boot.ci_lower:,.2f}, ${boot.ci_upper:,.2f})\n")
        f.write(
            f"  Normality: {'PASS' if boot.is_normal else 'FAIL'} (p={boot.normality_pvalue:.4f})\n"
        )
        f.write(f"  Outliers: {boot.n_outliers}/{boot.n_bootstrap}\n")
        f.write(f"  Converged: {'YES' if boot.converged else 'NO'}\n\n")

        f.write("HYPERPARAMETER SENSITIVITY:\n")
        for param_name, param_result in results.hyperparameter_sensitivity.items():
            f.write(f"  {param_name}:\n")
            f.write(f"    Sensitivity: {param_result.sensitivity_score:.3f}\n")
            f.write(f"    Recommended: {param_result.recommended_value}\n")
            f.write(
                f"    ATE Range: ${param_result.metadata['min_ate']:,.2f} to ${param_result.metadata['max_ate']:,.2f}\n"
            )
        f.write("\n")

        f.write("SEED SENSITIVITY:\n")
        f.write(f"  Mean ATE: ${seed.mean_ate:,.2f} ± ${seed.std_ate:,.2f}\n")
        f.write(f"  Range: ${seed.min_ate:,.2f} to ${seed.max_ate:,.2f}\n")
        f.write(f"  CV: {seed.cv_ate:.3f}\n\n")

        f.write("ROOT CAUSE ANALYSIS:\n")
        f.write(f"  {results.root_cause_analysis}\n\n")

        f.write("RECOMMENDATIONS:\n")
        for i, rec in enumerate(results.recommendations, 1):
            f.write(f"  {i}. {rec}\n")

    print(f"Results saved to: {output_path}")
    print()

    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = main()
