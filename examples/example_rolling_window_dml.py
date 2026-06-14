"""Rolling-window DML: tracing a treatment effect over time.

Demonstrates ``RollingWindowDML``, which refits a partially-linear DML estimate
on a sliding window of the series to show how the scalar treatment effect moves
across the sample.

Usage:
    venv/bin/python examples/example_rolling_window_dml.py
"""

import numpy as np

from dml_ts.dml import RollingWindowDML


def main() -> None:
    """Run the rolling-window DML pipeline on synthetic data."""
    TRUE_THETA = 1.5

    # ── Synthetic series with a known constant effect ────────────────────
    print("=" * 60)
    print("Rolling-Window DML")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n = 400
    X = rng.standard_normal((n, 3))
    T = 0.6 * X[:, 0] + rng.standard_normal(n)
    Y = TRUE_THETA * T + X[:, 1] - 0.5 * X[:, 2] + rng.standard_normal(n)
    print(f"  n_periods:   {n}")
    print(f"  n_features:  {X.shape[1]}")
    print(f"  True effect: {TRUE_THETA}")
    print()

    # ── Fit a sliding-window DML ─────────────────────────────────────────
    model = RollingWindowDML(
        window_size=150,
        step_size=50,
        model_y="ridge",
        model_t="ridge",
        n_folds=3,
        random_state=42,
    )
    model.fit(Y, T, X)
    centers, thetas, ses = model.get_effects()

    print(f"  Windows fit: {len(centers)}")
    print(f"  Mean theta:  {thetas.mean():.3f} (true: {TRUE_THETA})")
    print()
    print("  Per-window estimates:")
    for center, theta, se in zip(centers, thetas, ses):
        print(f"    center={int(center):4d}  theta={theta:+.3f}  se={se:.3f}")


if __name__ == "__main__":
    main()
