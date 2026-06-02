"""
Quick verification script to demonstrate the coverage bug fix.

Before Fix:
  - Coverage ~0.09-0.24 (using Monte Carlo SE)

After Fix:
  - Coverage ~0.90-1.00 (using DML's actual CIs from effect_interval())

Usage:
    python scripts/verify_coverage_fix.py
"""

from dml_ts.validation.dgp_generator import DGPGenerator
from dml_ts.validation.bias_validation import BiasValidation

print("=" * 80)
print("COVERAGE BUG FIX VERIFICATION")
print("=" * 80)
print()

# Create DGP with moderate settings for stable coverage estimate
dgp = DGPGenerator(
    n=2000,  # Large sample → stable DML estimates
    p=5,  # Moderate dimensionality
    true_effect=2.0,
    confounding_strength=0.5,  # Moderate confounding
    random_state=42,
)

print("DGP Configuration:")
print(f"  n = {dgp.n}")
print(f"  p = {dgp.p}")
print(f"  true_effect = {dgp.true_effect}")
print(f"  confounding_strength = {dgp.confounding_strength}")
print()

# Run validation with moderate number of simulations
validator = BiasValidation(n_simulations=100, alpha=0.05, random_state=42)

print(f"Running validation ({validator.n_simulations} simulations)...")
print("This will take ~3-5 minutes...")
print()

result = validator.validate(dgp)

print("=" * 80)
print("RESULTS")
print("=" * 80)
print()
print(f"Status: {result.status}")
print(f"Bias: {result.bias:.4f} (95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}])")
print(f"MSE: {result.mse:.4f}")
print(f"RMSE: {result.mse**0.5:.4f}")
print()
print(f"Coverage: {result.coverage:.2%}")
print()

# Interpretation
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print()

if 0.85 <= result.coverage <= 1.00:
    print("✅ COVERAGE FIX SUCCESSFUL!")
    print()
    print(f"Coverage rate of {result.coverage:.2%} is approximately nominal (expected ~95%)")
    print("This confirms DML's confidence intervals are now calculated correctly.")
    print()
    print("How it works:")
    print(
        "  1. _estimate_effect() returns (estimate, ci_lower, ci_upper) from dml.effect_interval()"
    )
    print("  2. _run_simulations() collects CI bounds for each simulation")
    print("  3. _calculate_coverage() checks: (ci_lower ≤ true_effect ≤ ci_upper)")
    print("  4. Coverage = proportion of CIs containing true effect")
else:
    print("❌ COVERAGE STILL BROKEN!")
    print()
    print(f"Coverage rate of {result.coverage:.2%} is far below nominal 95%")
    print("Expected coverage to be in range [0.85, 1.00]")
    print()
    print("Possible issues:")
    print("  - DML's effect_interval() not working correctly")
    print("  - CI bounds not being stored properly")
    print("  - Coverage calculation logic error")

print()
print("=" * 80)
