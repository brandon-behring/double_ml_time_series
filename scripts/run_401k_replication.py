"""
Run 401(k) DML estimation and compare with published results.

Executes empirical replication of Chernozhukov et al. (2018) and
displays detailed comparison with published ATE estimates.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.empirical_replication import FourZeroOneKReplication


def main():
    """Run 401(k) replication and display results."""
    print("=" * 80)
    print("401(k) DML Replication - Chernozhukov et al. (2018)")
    print("=" * 80)
    print()

    # Initialize replicator with fixed random_state for reproducibility
    print("Initializing replication with random_state=42...")
    replicator = FourZeroOneKReplication(random_state=42, tolerance=0.15)
    print()

    # Load data
    print("Loading 401(k) dataset from doubleml...")
    df = replicator.load_data()
    print(f"  ✓ Dataset loaded: {df.shape[0]:,} observations, {df.shape[1]} variables")
    print()

    # Replicate PLR with Random Forest (PRIMARY)
    print("-" * 80)
    print("REPLICATION 1: Partially Linear Regression with Random Forest")
    print("-" * 80)
    print("Estimating DML with Random Forest (n_estimators=500)...")
    print("This may take 10-15 minutes due to 5-fold cross-fitting...")
    print()

    result_rf = replicator.replicate_plr_rf()

    print(f"✓ Estimation complete!")
    print()
    print("RESULTS:")
    print(f"  Published ATE (Chernozhukov et al. 2018): ${result_rf.published_ate:,.2f}")
    print(f"  Our ATE Estimate:                          ${result_rf.ate_estimate:,.2f}")
    print(f"  Standard Error:                            ${result_rf.std_error:,.2f}")
    print(
        f"  95% Confidence Interval:                   (${result_rf.ci_lower:,.2f}, ${result_rf.ci_upper:,.2f})"
    )
    print()
    print("COMPARISON:")
    print(f"  Absolute Difference:                       ${result_rf.difference:,.2f}")
    print(f"  Relative Difference:                       {result_rf.rel_difference:.1%}")
    print(f"  P-value (two-sided test):                  {result_rf.p_value:.4f}")
    print(f"  Replication Status:                        {result_rf.status}")
    print(f"  Tolerance threshold:                       ±{result_rf.tolerance:.0%}")
    print()

    if result_rf.status == "MATCH":
        print("✅ SUCCESS: Our estimate matches published results within tolerance!")
    else:
        print("⚠️  WARNING: Our estimate differs from published results by more than tolerance.")
        print("   This may be due to:")
        print("   - Different Random Forest implementations")
        print("   - Different cross-fitting random splits")
        print("   - Different hyperparameter settings")
    print()

    # Replicate PLR with Lasso (SECONDARY)
    print("-" * 80)
    print("REPLICATION 2: Partially Linear Regression with Lasso")
    print("-" * 80)
    print("Estimating DML with Lasso (LassoCV with 5-fold CV)...")
    print("This may take 5-10 minutes...")
    print()

    result_lasso = replicator.replicate_plr_lasso()

    print(f"✓ Estimation complete!")
    print()
    print("RESULTS:")
    print(f"  Published ATE (Chernozhukov et al. 2018): ${result_lasso.published_ate:,.2f}")
    print(f"  Our ATE Estimate:                          ${result_lasso.ate_estimate:,.2f}")
    print(f"  Standard Error:                            ${result_lasso.std_error:,.2f}")
    print(
        f"  95% Confidence Interval:                   (${result_lasso.ci_lower:,.2f}, ${result_lasso.ci_upper:,.2f})"
    )
    print()
    print("COMPARISON:")
    print(f"  Absolute Difference:                       ${result_lasso.difference:,.2f}")
    print(f"  Relative Difference:                       {result_lasso.rel_difference:.1%}")
    print(f"  P-value (two-sided test):                  {result_lasso.p_value:.4f}")
    print(f"  Replication Status:                        {result_lasso.status}")
    print()

    if result_lasso.status == "MATCH":
        print("✅ SUCCESS: Our estimate matches published results within tolerance!")
    else:
        print("⚠️  WARNING: Our estimate differs from published results.")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Published Results (Chernozhukov et al. 2018, Table 1):")
    print(f"  PLR Random Forest: ${result_rf.published_ate:,.2f}")
    print(f"  PLR Lasso:         ${result_lasso.published_ate:,.2f}")
    print()
    print("Our Results:")
    print(f"  PLR Random Forest: ${result_rf.ate_estimate:,.2f} ({result_rf.status})")
    print(f"  PLR Lasso:         ${result_lasso.ate_estimate:,.2f} ({result_lasso.status})")
    print()
    print("Dataset:")
    print(f"  n = {result_rf.metadata['n_obs']:,} observations")
    print(f"  Treatment: {result_rf.metadata['treatment']}")
    print(f"  Controls: {result_rf.metadata['n_controls']} variables")
    print()
    print("=" * 80)

    # Return results for potential programmatic use
    return {
        "PLR_RF": result_rf,
        "PLR_Lasso": result_lasso,
    }


if __name__ == "__main__":
    results = main()
