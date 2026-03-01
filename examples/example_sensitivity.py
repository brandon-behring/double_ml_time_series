"""Rosenbaum Bounds Sensitivity Analysis.

Demonstrates sensitivity analysis for DML treatment effect estimates.
Quantifies how strong unmeasured confounding would need to be to
overturn causal conclusions.

Interpretation:
    - Gamma > 2.0: Robust — strong confounding needed to overturn
    - Gamma 1.5-2.0: Moderately robust
    - Gamma 1.2-1.5: Sensitive — modest confounding could alter results
    - Gamma < 1.2: Fragile — very sensitive to unmeasured confounding

Usage:
    python examples/example_sensitivity.py
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.dml import double_ml
from src.sensitivity import RosenbaumBounds


def main() -> None:
    """Run sensitivity analysis on a DML estimate."""
    # ── Generate Data with Strong Signal ─────────────────────────────────
    print("=" * 60)
    print("Step 1: DML Estimation")
    print("=" * 60)

    np.random.seed(42)
    n = 1000
    X = np.random.randn(n, 5)
    T = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n) * 0.5
    Y = 2.0 * T + np.cos(X[:, 0]) + X[:, 1] ** 2 + np.random.randn(n) * 0.5

    result = double_ml(
        Y,
        T,
        X,
        outcome_model=RandomForestRegressor(n_estimators=200, random_state=42),
        treatment_model=RandomForestRegressor(n_estimators=200, random_state=42),
        n_folds=5,
    )

    print(f"  θ = {result.theta:.3f} ± {result.se:.3f}")
    print(f"  95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
    print(f"  p-value: {result.p_value:.4f}")
    print()

    # ── Sensitivity Analysis ─────────────────────────────────────────────
    print("=" * 60)
    print("Step 2: Rosenbaum Bounds Sensitivity Analysis")
    print("=" * 60)

    bounds = RosenbaumBounds(gamma_max=3.0, gamma_step=0.25)

    # Approximate treated/control split (for continuous treatment,
    # use above/below median as a heuristic)
    median_t = np.median(T)
    n_treated = int(np.sum(T > median_t))
    n_control = n - n_treated

    sensitivity = bounds.analyze(
        theta=result.theta,
        se=result.se,
        n_treated=n_treated,
        n_control=n_control,
    )

    print(sensitivity.summary())
    print()

    # ── Interpretation ───────────────────────────────────────────────────
    print("=" * 60)
    print("Interpretation")
    print("=" * 60)
    gamma_crit = sensitivity.gamma_critical
    print(f"  Critical Gamma: {gamma_crit:.2f}")
    if gamma_crit > 2.0:
        print("  → ROBUST: Very strong unmeasured confounding needed to overturn")
    elif gamma_crit > 1.5:
        print("  → MODERATELY ROBUST: Moderate confounding could affect conclusions")
    elif gamma_crit > 1.2:
        print("  → SENSITIVE: Modest confounding could alter results")
    else:
        print("  → FRAGILE: Very sensitive to unmeasured confounding")


if __name__ == "__main__":
    main()
