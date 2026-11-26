"""Check that reported coverage counts per-fit EconML ate_interval hits.

Run from the repository root with python scripts/verify_coverage_fix.py.
"""

import os

from dml_ts.validation.bias_validation import BiasValidation
from dml_ts.validation.dgp_generator import DGPGenerator

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
validator = BiasValidation(
    n_simulations=100,
    alpha=0.05,
    random_state=42,
    n_jobs=int(os.environ.get("DML_VALIDATION_DRAW_JOBS", "1")),
)

print(f"Running validation ({validator.n_simulations} simulations)...")
print(f"Independent fit workers: {validator.n_jobs}")
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

# Cross-check the saved marginal ATE intervals against the reported hit rate.
bounds = result.ci_estimates
hits = int(((bounds[:, 0] <= dgp.true_effect) & (dgp.true_effect <= bounds[:, 1])).sum())
expected_count = result.n_simulations
if len(bounds) != expected_count or hits / expected_count != result.coverage:
    raise AssertionError("Reported coverage disagrees with the stored per-fit ATE intervals")
print(f"Marginal ATE interval hits: {hits}/{expected_count}")
print(
    f"Exact binomial p for nominal {1 - validator.alpha:.0%} coverage: {result.coverage_p_value:.6g}"
)
print(f"Within-scenario coverage verdict: {result.coverage_status}")
