"""FWL → Robinson → DML Progression.

Demonstrates the pedagogical progression from Frisch-Waugh-Lovell through
Robinson's semiparametric estimator to Double Machine Learning. Each step
resolves a limitation of the previous approach.

Key takeaways:
    1. FWL is exact when confounding is linear
    2. FWL is biased when confounding is nonlinear
    3. Robinson handles nonlinearity but overfits (in-sample nuisance)
    4. DML eliminates overfitting bias via cross-fitting

Usage:
    python examples/example_fwl_to_dml.py
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from dml_ts.dml import double_ml
from dml_ts.dml.fwl import fwl_estimate, fwl_vs_ols_comparison
from dml_ts.dml.robinson import robinson_estimator


def main() -> None:
    """Run the FWL → Robinson → DML progression."""
    TRUE_THETA = 2.0
    np.random.seed(42)

    # ── Part 1: FWL on a Linear DGP ──────────────────────────────────────
    print("=" * 60)
    print("Part 1: FWL on Linear DGP (should be exact)")
    print("=" * 60)

    n = 1000
    X = np.random.randn(n, 3)
    T = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n)
    Y = TRUE_THETA * T + X @ np.array([1.0, -0.5, 0.3]) + np.random.randn(n)

    result_fwl_linear = fwl_estimate(Y, T, X)
    comparison = fwl_vs_ols_comparison(Y, T, X)

    print(f"  FWL estimate:  {result_fwl_linear.theta:.4f} (true: {TRUE_THETA})")
    print(f"  OLS-FWL diff:  {comparison['difference']:.2e} (should be ~1e-14)")
    print(f"  Match: {comparison['match']}")
    print()

    # ── Part 2: FWL Fails on Nonlinear DGP ───────────────────────────────
    print("=" * 60)
    print("Part 2: Nonlinear DGP — FWL vs Robinson vs DML")
    print("=" * 60)

    n = 2000
    X = np.random.randn(n, 5)
    T = np.sin(X[:, 0]) + 0.5 * X[:, 1] + np.random.randn(n) * 0.5
    Y = TRUE_THETA * T + np.cos(X[:, 0]) + X[:, 1] ** 2 + np.random.randn(n) * 0.5

    # FWL — biased (can't handle nonlinearity)
    r_fwl = fwl_estimate(Y, T, X)
    print(f"  FWL:      θ = {r_fwl.theta:.3f}  (bias: {r_fwl.theta - TRUE_THETA:+.3f})")

    # Robinson — less biased but overfits in-sample
    r_rob = robinson_estimator(
        Y,
        T,
        X,
        model=RandomForestRegressor(n_estimators=200, random_state=42),
    )
    print(f"  Robinson: θ = {r_rob.theta:.3f}  (bias: {r_rob.theta - TRUE_THETA:+.3f})")

    # DML — unbiased with valid inference via cross-fitting
    r_dml = double_ml(
        Y,
        T,
        X,
        outcome_model=RandomForestRegressor(n_estimators=200, random_state=42),
        treatment_model=RandomForestRegressor(n_estimators=200, random_state=42),
        n_folds=5,
    )
    print(f"  DML:      θ = {r_dml.theta:.3f}  (bias: {r_dml.theta - TRUE_THETA:+.3f})")
    print(f"            SE = {r_dml.se:.3f}")
    print(f"            CI = [{r_dml.ci_lower:.3f}, {r_dml.ci_upper:.3f}]")
    covers = r_dml.ci_lower <= TRUE_THETA <= r_dml.ci_upper
    print(f"            Covers true θ: {covers}")
    print()

    # ── Part 3: Diagnostics ──────────────────────────────────────────────
    print("=" * 60)
    print("Part 3: DML Diagnostics")
    print("=" * 60)
    print(f"  Outcome R² (CV):   {r_dml.outcome_r2_cv:.3f}")
    print(f"  Treatment R² (CV): {r_dml.treatment_r2_cv:.3f}")
    print(f"  t-statistic:       {r_dml.t_stat:.3f}")
    print(f"  p-value:           {r_dml.p_value:.4f}")
    print(f"  n_folds:           {r_dml.n_folds}")


if __name__ == "__main__":
    main()
