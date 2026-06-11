"""Temporal PLR DML with temporal cross-validation and HAC standard errors.

Demonstrates the current time-series companion workflow:
    1. Generate autocorrelated data with a known treatment effect
    2. Show temporal cross-validation fold structure
    3. Estimate a scalar temporal PLR treatment effect
    4. Report dropped initial rows and HAC inference

Usage:
    venv/bin/python examples/example_time_series_dml.py
"""

import numpy as np
from temporalcv import TimeSeriesCrossValidator

from dml_ts.dml import TemporalPLRDML
from dml_ts.validation import TimeSeriesDGPGenerator


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
    print("  AR confounder persistence: 0.7")
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

    # ── Part 3: Temporal PLR DML Estimation ──────────────────────────────
    print("=" * 60)
    print("Part 3: TemporalPLRDML")
    print("=" * 60)

    model = TemporalPLRDML(
        n_lags=2,
        model_y="ridge",
        model_t="ridge",
        n_splits=5,
        gap=10,
        hac_bandwidth=8,
        random_state=42,
    )
    result = model.fit(Y, T, X, time_index=np.arange(len(Y)))

    print(f"  Temporal PLR estimate: {result.theta:.3f} (true: {TRUE_THETA})")
    print(f"  HAC SE:                {result.se:.4f}")
    print(f"  95% CI:                [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
    print(f"  Lag rows dropped:      {result.lagged_rows_dropped}")
    print(f"  CV rows dropped:       {result.dropped_initial_rows}")
    print()

    # ── Part 4: Diagnostics ──────────────────────────────────────────────
    print("=" * 60)
    print("Part 4: Nuisance Diagnostics")
    print("=" * 60)

    print(f"  Outcome R2 (temporal CV):   {result.outcome_r2_cv:.3f}")
    print(f"  Treatment R2 (temporal CV): {result.treatment_r2_cv:.3f}")
    print(f"  HAC bandwidth:              {result.hac_bandwidth}")


if __name__ == "__main__":
    main()
