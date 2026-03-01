"""Production Pipeline: Model Registry + Causal Monitoring.

Demonstrates the production deployment workflow for DML models:
    1. Generate insurance pricing data
    2. Fit DML model
    3. Register model version in the registry
    4. Monitor causal assumptions on new data

Usage:
    python examples/example_production_pipeline.py
"""

import tempfile

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from src.dml import double_ml
from src.production import (
    CausalMonitor,
    DMLModelRegistry,
    DMLModelVersion,
)
from src.validation import create_insurance_dgp


def main() -> None:
    """Run the production pipeline example."""
    # ── Step 1: Generate Training Data ───────────────────────────────────
    print("=" * 60)
    print("Step 1: Generate Insurance Pricing Data")
    print("=" * 60)

    train_data = create_insurance_dgp(
        realism="moderate",
        n_periods=120,
        n_products=10,
        true_tau=-0.8,
        seed=42,
    )

    print(f"  Observations: {len(train_data.Y)}")
    print(f"  True effect:  {train_data.true_params.tau}")
    print()

    # ── Step 2: Estimate Treatment Effect ────────────────────────────────
    print("=" * 60)
    print("Step 2: DML Estimation")
    print("=" * 60)

    result = double_ml(
        train_data.Y,
        train_data.T,
        train_data.X,
        outcome_model=GradientBoostingRegressor(n_estimators=100, random_state=42),
        treatment_model=GradientBoostingRegressor(n_estimators=100, random_state=42),
        n_folds=5,
    )

    print(f"  theta = {result.theta:.3f} +/- {result.se:.3f}")
    print(f"  95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
    print(f"  Outcome R2 (CV):   {result.outcome_r2_cv:.3f}")
    print(f"  Treatment R2 (CV): {result.treatment_r2_cv:.3f}")
    print()

    # ── Step 3: Register Model ───────────────────────────────────────────
    print("=" * 60)
    print("Step 3: Model Registry")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = DMLModelRegistry(base_path=tmpdir)

        # Create dummy nuisance models (in production, these come from DML folds)
        from sklearn.linear_model import Ridge

        dummy_propensity = Ridge().fit(train_data.X[:100], train_data.T[:100])
        dummy_outcome = Ridge().fit(train_data.X[:100], train_data.Y[:100])
        nuisance = {i: (dummy_propensity, dummy_outcome) for i in range(5)}

        version = DMLModelVersion.create(
            model_type="double_ml",
            nuisance_models=nuisance,
            feature_names=[f"X{i}" for i in range(train_data.X.shape[1])],
            treatment_name="competitor_rate_change",
            outcome_name="sales_volume",
            hyperparameters={
                "n_folds": result.n_folds,
                "outcome_model": "GradientBoostingRegressor",
                "treatment_model": "GradientBoostingRegressor",
            },
            metrics={
                "theta": result.theta,
                "se": result.se,
                "outcome_r2_cv": result.outcome_r2_cv,
                "treatment_r2_cv": result.treatment_r2_cv,
            },
            metadata={
                "n_obs": len(train_data.Y),
                "true_effect": train_data.true_params.tau,
            },
        )

        version_id = registry.register(version)
        print(f"  Registered version: {version_id}")
        print(f"  Model type: {version.model_type}")
        print(f"  Created: {version.created_at}")
        print()

    # ── Step 4: Monitor on New Data ──────────────────────────────────────
    print("=" * 60)
    print("Step 4: Causal Monitoring")
    print("=" * 60)

    # Generate new data (simulating production scoring)
    new_data = create_insurance_dgp(
        realism="moderate",
        n_periods=120,
        n_products=10,
        true_tau=-0.8,
        seed=123,  # Different seed = different realization
    )

    monitor = CausalMonitor()

    # Check key diagnostics
    print(f"  New data observations: {len(new_data.Y)}")
    print(f"  Treatment mean (train): {train_data.T.mean():.3f}")
    print(f"  Treatment mean (new):   {new_data.T.mean():.3f}")
    print(f"  Treatment std (train):  {train_data.T.std():.3f}")
    print(f"  Treatment std (new):    {new_data.T.std():.3f}")
    print()

    # Basic stability check
    t_shift = abs(train_data.T.mean() - new_data.T.mean()) / train_data.T.std()
    print(f"  Treatment distribution shift (standardized): {t_shift:.3f}")
    if t_shift < 0.5:
        print("  -> Treatment distribution stable")
    else:
        print("  -> WARNING: Treatment distribution may have shifted")


if __name__ == "__main__":
    main()
