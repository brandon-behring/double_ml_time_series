# Phase 0: Building Scientific Foundations for DML Validation

**Created**: 2025-11-14 20:00
**Updated**: 2025-11-14 20:00
**STATUS**: IN_PROGRESS
**Objective**: Fix critical bugs, add baseline comparisons, build research-quality validation framework

---

## Executive Summary

**Problem**: Current validation has fundamental flaws preventing publishable research:
- Coverage calculation completely broken (0.09-0.24 instead of ~0.95)
- No baseline methods for comparison
- Arbitrary validation thresholds without statistical justification

**Solution**: Pause Methods 2-7, fix foundations, add baselines, create honest comparative analysis

**Target**: Internal research quality with exploratory freedom
**Timeline**: 3 weeks
**MC Runs**: 100 (fast iteration, increase later if needed)

---

## Current State Assessment

### What's Working ✅
- Infrastructure: Test harness, DGP generator, parallel execution (48 cores)
- BiasValidation class: Runs Monte Carlo simulations correctly
- Test suite: 26 tests, 25 passing (96% pass rate)
- Simulation framework: 81 parameter combinations in 27 minutes

### What's Broken 🔴
1. **Coverage calculation** (`bias_validation.py:204-231`):
   ```python
   # WRONG - Uses Monte Carlo standard error
   se = np.std(estimates) / np.sqrt(len(estimates))
   ci_lower = estimate - 1.96 * se

   # Should use DML's own confidence intervals
   ci_lower, ci_upper = dml.effect_interval(X=data.X, alpha=0.05)
   ```
   - **Result**: Coverage 0.09-0.24 (should be ~0.95)
   - **Impact**: Blocks Method 2 entirely, invalidates all coverage results

2. **No baseline comparisons**:
   - Can't claim DML is better without comparing to OLS, IPW, ML methods
   - Missing oracle estimator (theoretical best case)

3. **Arbitrary thresholds**:
   - Current: |bias| < 0.1 → PASS
   - No statistical justification
   - Should use hypothesis tests

### Current Results (From BIAS_VALIDATION_results_20251114_103701.csv)
- 81 parameter combinations tested
- PASS: 43/81 (53.1%), WARNING: 25/81 (30.9%), FAIL: 13/81 (16.0%)
- Mean bias: -0.11 (moderate underestimation)
- Mean RMSE: 0.55
- Coverage: 0.09-0.24 (BROKEN - this is the critical bug)

---

## Phase 0 Plan (3 Weeks)

### Week 1: Critical Fixes + OLS/IPW Baselines

#### Day 1: Coverage Bug Fix (CRITICAL) - ✅ IN PROGRESS
- [x] **Morning**: Present technical implementation plan
  - Showed exact code changes (4 method modifications)
  - Explained test-first approach
  - Got approval to proceed

- [x] **Afternoon**: Implement fix (COMPLETED)
  - Updated `ValidationResult` dataclass (added `ci_estimates`, `point_estimates`)
  - Modified `_estimate_effect()` to return `(estimate, ci_lower, ci_upper)`
  - Modified `_run_simulations()` to collect CI bounds
  - Replaced `_calculate_coverage()` with correct vectorized implementation
  - Updated `validate()` to use actual DML CIs

- [ ] **Verification**: Test suite running (27 tests), then rerun simulation

#### Day 2: Statistical Hypothesis Tests
- [ ] Replace arbitrary |bias| < 0.1 with t-test for H₀: E[bias] = 0
- [ ] Add binomial test for H₀: coverage = 0.95
- [ ] Update ValidationResult to include p-values
- [ ] Document statistical methodology

#### Day 3: OLS Baselines
- [ ] Implement `NaiveOLS` (Y ~ T only, no controls)
- [ ] Implement `OLSWithControls` (Y ~ T + X)
- [ ] Test on existing DGPs
- [ ] Create first comparison table (DML vs OLS variants)

#### Days 4-5: IPW Methods
- [ ] Implement `IPWEstimator` (inverse propensity weighting)
- [ ] Implement `AugmentedIPW` (doubly robust)
- [ ] Test overlap violations
- [ ] Compare robustness to DML

### Week 2: ML Baselines + Realistic Testing

#### Days 6-7: ML Baselines
- [ ] Implement `RandomForestATE`
- [ ] Implement `XGBoostATE`
- [ ] Performance comparison

#### Day 8: Unified Framework
- [ ] Create `BaselineComparison` class
- [ ] Run all 7 methods on same data
- [ ] Generate comparison table

#### Days 9-10: Model Misspecification
- [ ] Test: True exponential, fit linear
- [ ] Test: True interactions, fit main effects
- [ ] Test: Unmeasured confounders
- [ ] Test: Heterogeneous treatment effects

### Week 3: Analysis & Insights

#### Days 11-12: Complex DGPs
- [ ] Non-linear: Y = exp(X·β) + T·sin(X·γ)
- [ ] High-dimensional: p=100, only 5 matter
- [ ] Sparse confounding
- [ ] Time-varying (time series prep)

#### Days 13-14: Comprehensive Analysis
- [ ] Generate comparison table (all methods × all scenarios)
- [ ] Identify when DML wins/loses
- [ ] Document computational costs
- [ ] Create honest assessment

#### Day 15: Final Deliverables
- [ ] Research notebook with all comparisons
- [ ] "Best Practices" guide
- [ ] "When DML Fails" section
- [ ] Recommendations for different use cases

---

## Success Metrics

### Phase 0 Complete When:
✅ Coverage shows 0.90-1.00 (not 0.09-0.24)
✅ Statistical tests replace arbitrary thresholds
✅ All 26 tests passing
✅ 6+ baseline methods implemented
✅ Comparison table complete
✅ Clear understanding of DML's relative performance

### Key Research Questions Answered:
1. Does DML's orthogonality actually help? (vs OLS/IPW)
2. When does DML outperform simpler methods?
3. When does DML fail? (sample size, dimensionality, misspecification)
4. What's the computational cost tradeoff?

---

## What's Paused (Deferred to Phase 1+)

❌ Method 2: Coverage Validation (original roadmap) - blocked by bug
❌ Methods 3-7: Power, Sensitivity, MSE, Efficiency, Robustness
❌ LaLonde/IHDP benchmarks (defer if needed)
❌ 5000+ MC runs (keeping at 100 for now)
❌ Theoretical verification (Neyman orthogonality, √n-consistency)

---

## Methodological Decisions & Lessons Learned

### Decision 1: Fix Foundations Before Advancing
- **Rationale**: Can't build on broken coverage calculation
- **Alternative considered**: Parallel development (fix + continue Methods 2-7)
- **Why rejected**: Coverage bug invalidates everything downstream

### Decision 2: Keep MC Runs at 100
- **Rationale**: Fast iteration for development, increase later for final results
- **Trade-off**: Less precise estimates, but adequate for method comparison
- **When to increase**: Final publication-ready results (if needed)

### Decision 3: Internal Research Standard
- **Rationale**: Exploratory freedom without publication pressure
- **What this means**: Can be more experimental, less polish required
- **Still maintain**: Scientific rigor, honest assessment, statistical validity

### Decision 4: Prioritized Batch for Baselines
- **Order**: OLS (Days 3-4) → IPW (Days 4-5) → ML (Days 6-7)
- **Rationale**: OLS is fastest to implement, builds foundation
- **IPW is standard** in causal inference literature
- **ML methods** provide modern comparison

---

## Technical Implementation Notes

### Coverage Fix Details
**File**: `src/validation/bias_validation.py`

**Current (WRONG)**:
```python
def _calculate_coverage(self, estimates: np.ndarray, true_effect: float) -> float:
    n_covers = 0
    for estimate in estimates:
        # ❌ This is SE of Monte Carlo mean, NOT per-estimate CI!
        se = np.std(estimates) / np.sqrt(len(estimates))
        ci_lower = estimate - 1.96 * se
        ci_upper = estimate + 1.96 * se
        if ci_lower <= true_effect <= ci_upper:
            n_covers += 1
    return n_covers / len(estimates)
```

**Correct (TO IMPLEMENT)**:
```python
def _calculate_coverage(self, ci_bounds: np.ndarray, true_effect: float) -> float:
    """Calculate coverage using DML's actual confidence intervals.

    Args:
        ci_bounds: CI bounds from each DML fit (n_simulations, 2)
        true_effect: True treatment effect

    Returns:
        Coverage rate (proportion of CIs containing true effect)
    """
    # Vectorized check: does each CI contain truth?
    covers = (ci_bounds[:, 0] <= true_effect) & (true_effect <= ci_bounds[:, 1])
    return float(np.mean(covers))
```

**Required changes**:
1. `_estimate_effect()`: Return `(estimate, ci_lower, ci_upper)` using `dml.effect_interval()`
2. `_run_simulations()`: Store CI bounds for each simulation
3. `_calculate_coverage()`: Use actual CI bounds, not reconstructed
4. `ValidationResult`: Add fields for `ci_estimates` and `point_estimates`

---

## Timeline

**Week 1** (Nov 14-20): Fix bugs + OLS/IPW
**Week 2** (Nov 21-27): ML baselines + realistic testing
**Week 3** (Nov 28-Dec 4): Analysis + insights

**Estimated completion**: December 4, 2025

---

## Next Steps

1. ✅ Create this plan document
2. ⏳ Present technical coverage fix plan
3. ⏳ Get approval for implementation approach
4. ⏳ Begin Day 1 work (coverage bug fix)

---

**Last Updated**: 2025-11-14 20:00
**Status**: Ready to present coverage fix technical plan
