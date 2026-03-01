"""Insurance Pricing DGP at Multiple Realism Levels.

Demonstrates the parameterized insurance DGP for competitor pricing
causal inference. Shows how DGP complexity affects estimation difficulty.

Realism levels:
    - simple: Y = tau*T + X'beta + eps (linear, no dynamics)
    - moderate: + macro confounding, AR errors, product fixed effects
    - full: + seasonal patterns, regime change, heterogeneous effects

Usage:
    python examples/example_insurance_dgp.py
"""

from typing import Literal

import numpy as np

from src.validation import create_insurance_dgp

RealismLevel = Literal["simple", "moderate", "full"]


def main() -> None:
    """Generate and compare insurance DGPs at different realism levels."""
    TRUE_TAU = -0.8
    levels: tuple[RealismLevel, ...] = ("simple", "moderate", "full")

    for realism in levels:
        print("=" * 60)
        print(f"Insurance DGP: realism='{realism}'")
        print("=" * 60)

        dgp_result = create_insurance_dgp(
            realism=realism,
            n_periods=120,
            n_products=10,
            true_tau=TRUE_TAU,
            seed=42,
        )

        print(f"  Observations: {len(dgp_result.Y)}")
        print(f"  Time periods: {len(np.unique(dgp_result.time_index))}")
        print(f"  Products:     {len(np.unique(dgp_result.product_index))}")
        print(f"  Controls (X): {dgp_result.X.shape[1]} features")
        print(f"  Macro (X_m):  {dgp_result.X_macro.shape[1]} features")
        print()

        # True parameters
        params = dgp_result.true_params
        print(f"  True τ:        {params.tau:.2f}")
        print(f"  AR coef:       {params.ar_coef:.2f}")
        print(f"  Seasonal amp:  {params.seasonal_amplitude:.2f}")
        print(f"  Regime shift:  {params.regime_shift:.2f}")
        print()

        # Summary statistics
        print(f"  Y: mean={dgp_result.Y.mean():.2f}, std={dgp_result.Y.std():.2f}")
        print(f"  T: mean={dgp_result.T.mean():.2f}, std={dgp_result.T.std():.2f}")
        print()

        print(f"  Description: {dgp_result.description}")
        print()


if __name__ == "__main__":
    main()
