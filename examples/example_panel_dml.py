"""Panel DML with fixed effects and cluster-robust inference.

Demonstrates ``PanelDML`` on a synthetic panel: a within (individual
fixed-effects) transformation removes unit-level confounding, DML residualizes
the remaining controls, and the treatment effect is reported with a
cluster-robust standard error.

Usage:
    venv/bin/python examples/example_panel_dml.py
"""

import warnings

import numpy as np

from dml_ts.dml import PanelDML


def main() -> None:
    """Run the panel DML pipeline on synthetic panel data."""
    TRUE_THETA = 2.0

    print("=" * 60)
    print("Panel DML (individual fixed effects)")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n_units, m = 60, 6
    n_obs = n_units * m
    individual_id = np.repeat(np.arange(n_units), m)
    time_id = np.tile(np.arange(m), n_units)

    # Unit-level confounder (the fixed effect) plus idiosyncratic controls.
    unit_effect = np.repeat(rng.standard_normal(n_units), m)
    X = rng.standard_normal((n_obs, 2))
    T = 0.5 * X[:, 0] + unit_effect + rng.standard_normal(n_obs)
    Y = TRUE_THETA * T + X[:, 1] + 3.0 * unit_effect + rng.standard_normal(n_obs)

    print(f"  units:        {n_units}")
    print(f"  periods:      {m}")
    print(f"  observations: {n_obs}")
    print(f"  True effect:  {TRUE_THETA}")
    print()

    model = PanelDML(
        fixed_effects="individual",
        cluster_se=True,
        model_y="ridge",
        model_t="ridge",
        n_folds=3,
        random_state=42,
    )
    # Short per-unit panels leave a few early temporal-CV rows uncovered.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = model.fit(Y, T, X, individual_id, time_id)

    print(f"  Panel DML estimate: {result.theta:.3f} (true: {TRUE_THETA})")
    print(f"  Cluster-robust SE:  {result.se:.4f}")
    print(f"  95% CI:             [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")


if __name__ == "__main__":
    main()
