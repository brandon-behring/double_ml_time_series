"""
Test script for Issue C5: 401(k) Covariate Mismatch Impact Analysis

This script tests whether using 9 covariates (CURRENT, excluding 'nifa' and 'tw')
vs 11 covariates (CORRECTED, including all) materially affects the 401(k) replication
estimate and its match to published results.

Issue: Current preprocessing drops 'nifa' and 'tw', leaving 9 controls instead of
the 11 used in Chernozhukov et al. (2018).

Published Results (Table 1):
- PLR Random Forest: $9,127 (95% CI: $7,723 - $10,531)
- PLR Lasso: $9,580

Decision Criteria:
- If 11-covariate estimate is >$500 closer to published → Fix valuable
- If conclusions about replication quality change → Fix critical
- Otherwise → Document as limitation

Runtime: ~5-10 minutes (2 configurations × RF/Lasso methods)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_401k_data():
    """Load 401(k) dataset from doubleml package."""
    try:
        from doubleml.datasets import fetch_401K

        return fetch_401K(return_type="DataFrame")
    except ImportError:
        raise ImportError("doubleml package required. Install with: pip install doubleml")


def preprocess_401k(
    df: pd.DataFrame, treatment: str = "e401", include_all_covariates: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess 401(k) data with either 9 or 11 covariates.

    Args:
        df: Raw 401(k) DataFrame
        treatment: Treatment variable ("e401" or "p401")
        include_all_covariates: If True, use all 11 covariates. If False, use 9 (drop nifa, tw)

    Returns:
        Tuple of (Y, T, X)
    """
    # Outcome
    Y = df["net_tfa"].values

    # Treatment
    T = df[treatment].values

    # Controls
    if include_all_covariates:
        # CORRECTED: Use all 11 covariates (match published design)
        control_vars = [col for col in df.columns if col not in ["net_tfa", "e401", "p401"]]
    else:
        # CURRENT: Exclude nifa and tw (only 9 covariates)
        control_vars = [
            col for col in df.columns if col not in ["net_tfa", "e401", "p401", "nifa", "tw"]
        ]

    X = df[control_vars].values

    return Y, T, X


def run_plr_rf(Y: np.ndarray, T: np.ndarray, X: np.ndarray, random_state: int = 42) -> dict:
    """Run PLR with Random Forest."""
    from econml.dml import LinearDML
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    # Random Forest nuisance models
    model_y = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1,
    )

    # CORRECTED: Use classifier for binary treatment
    model_t = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1,
    )

    # Fit DML (with corrected discrete_treatment=True)
    dml = LinearDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        cv=5,
        random_state=random_state,
    )

    dml.fit(Y=Y, T=T, X=X)

    # Get estimate and CI
    ate = dml.effect(X=X).mean()
    ci_lower, ci_upper = dml.effect_interval(X=X, alpha=0.05)

    return {
        "ate": float(ate),
        "ci_lower": float(ci_lower.mean()),
        "ci_upper": float(ci_upper.mean()),
        "std_error": float((ci_upper.mean() - ci_lower.mean()) / (2 * 1.96)),
    }


def run_plr_lasso(Y: np.ndarray, T: np.ndarray, X: np.ndarray, random_state: int = 42) -> dict:
    """Run PLR with Lasso."""
    from econml.dml import LinearDML
    from sklearn.linear_model import LassoCV, LogisticRegressionCV

    # Lasso nuisance models
    model_y = LassoCV(cv=5, random_state=random_state, n_jobs=-1)

    # CORRECTED: Use logistic regression for binary treatment
    model_t = LogisticRegressionCV(
        cv=5, penalty="l1", solver="saga", random_state=random_state, n_jobs=-1, max_iter=1000
    )

    # Fit DML (with corrected discrete_treatment=True)
    dml = LinearDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        cv=5,
        random_state=random_state,
    )

    dml.fit(Y=Y, T=T, X=X)

    # Get estimate and CI
    ate = dml.effect(X=X).mean()
    ci_lower, ci_upper = dml.effect_interval(X=X, alpha=0.05)

    return {
        "ate": float(ate),
        "ci_lower": float(ci_lower.mean()),
        "ci_upper": float(ci_upper.mean()),
        "std_error": float((ci_upper.mean() - ci_lower.mean()) / (2 * 1.96)),
    }


def main():
    """Run covariate comparison analysis."""

    print("=" * 80)
    print("ISSUE C5: 401(K) COVARIATE MISMATCH IMPACT ANALYSIS")
    print("=" * 80)
    print()
    print("Testing whether using 9 vs 11 covariates affects replication quality...")
    print()

    # Published results
    published_rf = 9127
    published_lasso = 9580

    # Load data
    print("Loading 401(k) dataset from doubleml package...")
    df = load_401k_data()
    print(f"  Dataset shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print()

    results = []
    random_state = 42

    # Test both configurations
    for include_all in [False, True]:
        config_name = "CORRECTED (11 covariates)" if include_all else "CURRENT (9 covariates)"
        print(f"{'─' * 80}")
        print(f"Configuration: {config_name}")
        print(f"{'─' * 80}")

        # Preprocess data
        Y, T, X = preprocess_401k(df, treatment="e401", include_all_covariates=include_all)
        print(f"  Data shape: Y={Y.shape}, T={T.shape}, X={X.shape}")
        print(f"  Number of covariates: {X.shape[1]}")

        # Run Random Forest
        print("  Running PLR Random Forest...")
        rf_result = run_plr_rf(Y, T, X, random_state=random_state)
        rf_ate = rf_result["ate"]
        rf_diff = abs(rf_ate - published_rf)
        rf_pct_diff = (rf_diff / published_rf) * 100

        print(f"    ATE: ${rf_ate:,.2f}")
        print(f"    95% CI: [${rf_result['ci_lower']:,.2f}, ${rf_result['ci_upper']:,.2f}]")
        print(f"    Difference from published ($9,127): ${rf_diff:,.2f} ({rf_pct_diff:.1f}%)")

        # Run Lasso
        print("  Running PLR Lasso...")
        lasso_result = run_plr_lasso(Y, T, X, random_state=random_state)
        lasso_ate = lasso_result["ate"]
        lasso_diff = abs(lasso_ate - published_lasso)
        lasso_pct_diff = (lasso_diff / published_lasso) * 100

        print(f"    ATE: ${lasso_ate:,.2f}")
        print(f"    95% CI: [${lasso_result['ci_lower']:,.2f}, ${lasso_result['ci_upper']:,.2f}]")
        print(f"    Difference from published ($9,580): ${lasso_diff:,.2f} ({lasso_pct_diff:.1f}%)")
        print()

        # Store results
        results.append(
            {
                "configuration": config_name,
                "n_covariates": X.shape[1],
                "rf_ate": rf_ate,
                "rf_ci_lower": rf_result["ci_lower"],
                "rf_ci_upper": rf_result["ci_upper"],
                "rf_diff_from_published": rf_diff,
                "rf_pct_diff": rf_pct_diff,
                "lasso_ate": lasso_ate,
                "lasso_ci_lower": lasso_result["ci_lower"],
                "lasso_ci_upper": lasso_result["ci_upper"],
                "lasso_diff_from_published": lasso_diff,
                "lasso_pct_diff": lasso_pct_diff,
            }
        )

    # Compare configurations
    current = results[0]
    corrected = results[1]

    print(f"{'=' * 80}")
    print("COMPARISON & DECISION")
    print(f"{'=' * 80}\n")

    # Random Forest comparison
    rf_improvement = current["rf_diff_from_published"] - corrected["rf_diff_from_published"]
    print(f"Random Forest:")
    print(
        f"  CURRENT (9 cov): ${current['rf_ate']:,.2f} (diff: ${current['rf_diff_from_published']:,.2f})"
    )
    print(
        f"  CORRECTED (11 cov): ${corrected['rf_ate']:,.2f} (diff: ${corrected['rf_diff_from_published']:,.2f})"
    )
    print(f"  Improvement: ${rf_improvement:,.2f} {'closer' if rf_improvement > 0 else 'farther'}")
    print()

    # Lasso comparison
    lasso_improvement = (
        current["lasso_diff_from_published"] - corrected["lasso_diff_from_published"]
    )
    print(f"Lasso:")
    print(
        f"  CURRENT (9 cov): ${current['lasso_ate']:,.2f} (diff: ${current['lasso_diff_from_published']:,.2f})"
    )
    print(
        f"  CORRECTED (11 cov): ${corrected['lasso_ate']:,.2f} (diff: ${corrected['lasso_diff_from_published']:,.2f})"
    )
    print(
        f"  Improvement: ${lasso_improvement:,.2f} {'closer' if lasso_improvement > 0 else 'farther'}"
    )
    print()

    # Decision
    rf_critical = abs(rf_improvement) > 500
    lasso_critical = abs(lasso_improvement) > 500

    if rf_critical or lasso_critical:
        decision = "FIX RECOMMENDED"
        priority = "MEDIUM-HIGH"
        reasoning = f"Adding missing covariates improves match to published results by >${500}"
    else:
        decision = "DOCUMENT AS LIMITATION"
        priority = "LOW"
        reasoning = f"Improvement <$500, not critical for teaching purposes"

    print(f"DECISION: {decision}")
    print(f"PRIORITY: {priority}")
    print(f"REASONING: {reasoning}")
    print()

    # Save results
    df_results = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(__file__).parent.parent / "results" / "impact_analysis"
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / f"issue_c5_impact_{timestamp}.csv"
    df_results.to_csv(csv_path, index=False)

    print(f"Results saved to: {csv_path}")
    print()

    return df_results, decision, priority


if __name__ == "__main__":
    results_df, decision, priority = main()
