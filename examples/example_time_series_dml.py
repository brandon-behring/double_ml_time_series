"""Time Series DML with Temporal Cross-Validation and HAC Standard Errors.

Demonstrates the full time series DML workflow:
    1. Generate autocorrelated data with a known treatment effect
    2. Show temporal cross-validation fold structure
    3. Estimate treatment effect with DML
    4. Compare HAC vs naive standard errors

Usage:
    python examples/example_time_series_dml.py
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.dml import double_ml
from src.dml.cross_fitting import TimeSeriesCrossValidator
from src.dml.hac import newey_west_se
from src.validation import TimeSeriesDGPGenerator


def main() -> None:
    """Run the time series DML pipeline."""
    TRUE_THETA = 1.5

    # ── Part 1: Generate Autocorrelated Data ─────────────────────────────
    print("=" * 60)
    print("Part 1: Time Series DGP (AR confounding)")
    print("=" * 60)

    dgp = TimeSeriesDGPGenerator(
        n_periods=500,
        p=5,
        true_effect=TRUE_THETA,
        treatment_ar_order=1,
        confounder_var_coef=0.7,
        random_state=42,
    )
    data = dgp.generate()
    Y, T, X = data.Y, data.T, data.X
    print(f"  n_periods: {len(Y)}")
    print(f"  n_features: {X.shape[1]}")
    print(f"  AR confounder persistence: 0.7")
    print(f"  True effect: {TRUE_THETA}")
    print()

    # ── Part 2: Temporal Cross-Validation ────────────────────────────────
    print("=" * 60)
    print("Part 2: TimeSeriesCrossValidator")
    print("=" * 60)

    cv = TimeSeriesCrossValidator(
        n_splits=5,
        test_size=50,
        gap=10,
        purge_length=5,
    )

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        print(
            f"  Fold {fold_idx}: "
            f"train=[{train_idx[0]}:{train_idx[-1]}] ({len(train_idx)}), "
            f"test=[{test_idx[0]}:{test_idx[-1]}] ({len(test_idx)})"
        )
    print()

    # ── Part 3: DML Estimation ───────────────────────────────────────────
    print("=" * 60)
    print("Part 3: DML with Temporal CV")
    print("=" * 60)

    result = double_ml(
        Y,
        T,
        X,
        outcome_model=RandomForestRegressor(n_estimators=100, random_state=42),
        treatment_model=RandomForestRegressor(n_estimators=100, random_state=42),
        n_folds=5,
    )

    print(f"  DML estimate: {result.theta:.3f} (true: {TRUE_THETA})")
    print(f"  Naive SE:     {result.se:.4f}")
    print(f"  95% CI:       [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
    print()

    # ── Part 4: HAC Standard Errors ──────────────────────────────────────
    print("=" * 60)
    print("Part 4: HAC vs Naive Standard Errors")
    print("=" * 60)

    # Use influence scores as residuals for HAC
    se_hac = newey_west_se(result.influence_scores, bandwidth="auto")

    print(f"  Naive SE: {result.se:.4f}")
    print(f"  HAC SE:   {se_hac:.4f}")
    ratio = se_hac / result.se if result.se > 0 else float("inf")
    print(f"  Ratio:    {ratio:.2f}x")
    print()

    if ratio > 1.2:
        print("  -> HAC SE substantially larger: autocorrelation matters here")
    else:
        print("  -> HAC and naive SE are similar: autocorrelation is mild")


if __name__ == "__main__":
    main()
