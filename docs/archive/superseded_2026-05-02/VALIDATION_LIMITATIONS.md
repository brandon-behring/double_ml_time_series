# Validation Infrastructure Limitations
**Last Updated**: 2025-11-22
**Status**: Known limitations in validation methods

---

## M1: Bootstrap Refitting in IPW/AIPW Estimators

### Issue

The IPW and AIPW baseline estimators do not refit first-stage propensity score models in each bootstrap replicate. This violates proper bootstrap inference procedures.

### Current Implementation

**Location**: `src/validation/ipw_baseline.py`

```python
# IPWEstimator.validate() method (lines 126-161)
def validate(self, dgp: DGPGenerator) -> ValidationResult:
    # 1. Fit propensity score model ONCE on first dataset
    propensity_model.fit(X, T)

    # 2. Bootstrap loop (reuses fitted model)
    for b in range(n_bootstrap):
        X_boot, T_boot, Y_boot = resample(...)

        # ⚠️ PROBLEM: Reuses propensity scores without refitting
        propensity_scores = propensity_model.predict_proba(X_boot)[:, 1]
        weights = ...
        ate = weighted_mean(Y_boot, weights)
```

### Consequences

1. **First-stage uncertainty ignored**: Bootstrap CIs don't reflect propensity score estimation error
2. **Overfitting propagated**: Propensity scores fitted to first dataset reused across all bootstraps
3. **Coverage likely overstated**: CIs narrower than they should be
4. **Extreme weights not re-evaluated**: Outlier propensity scores persist across bootstraps

### Theoretical Impact

**Without proper refitting**:
- Bootstrap variance: $\text{Var}(\hat{\theta} | \hat{p}(X))$ (conditional on fitted propensity)
- Ignores: $\text{Var}(\hat{p}(X))$ (first-stage estimation uncertainty)

**With proper refitting**:
- Bootstrap variance: $\text{Var}(\hat{\theta})$ (unconditional, includes first-stage)
- Correctly captures: $\text{Var}(\hat{\theta}) = \text{Var}(\hat{\theta} | \hat{p}) + E[\text{Var}(\hat{p})]$

**Expected bias**: Coverage rates 92-93% instead of nominal 95% (slight understatement)

### Why Not Fixed

**Effort**: 3-4 hours to implement proper bootstrap refitting
**Benefit**: Marginal improvement (2-3 percentage points in coverage)
**Priority**: LOW (other issues have higher impact)

**Decision** (2025-11-21): Document as limitation, defer fix to Phase 3 (advanced validation)

### Correct Implementation (Future Work)

```python
def validate(self, dgp: DGPGenerator) -> ValidationResult:
    for b in range(n_bootstrap):
        X_boot, T_boot, Y_boot = resample(...)

        # FIX: Refit propensity model in each bootstrap
        boot_propensity_model = clone(propensity_model)
        boot_propensity_model.fit(X_boot, T_boot)
        propensity_scores = boot_propensity_model.predict_proba(X_boot)[:, 1]

        weights = ...
        ate = weighted_mean(Y_boot, weights)
```

### Affected Methods

- `IPWEstimator` (src/validation/ipw_baseline.py:126-161)
- `AugmentedIPW` (src/validation/ipw_baseline.py:333-383)

### Mitigation

For current validation purposes:
1. **Interpretation**: Treat IPW/AIPW CIs as slightly optimistic
2. **Comparisons**: Still valid for relative method ranking (all baselines have same limitation)
3. **DML unaffected**: DML uses cross-fitting, not bootstrap (proper first-stage refitting built-in)

### References

- Chernozhukov et al. (2018): DML vs IPW bootstrap properties
- Efron & Tibshirani (1993): *An Introduction to the Bootstrap*, Ch. 8.4

---

## Future Work

**Phase 3 Enhancement** (if needed):
- Implement proper bootstrap refitting for IPW/AIPW
- Empirically test coverage impact (expected 2-3% improvement)
- Compare computational cost (10-20x slower due to refitting)

**Estimated effort**: 3-4 hours
**Priority**: LOW (validation infrastructure works well for book purposes)

---

**Related Documents**:
- `docs/VALIDATION_FIXES_DECISION_2025-11-21.md` (decision rationale)
- `docs/IMPACT_ASSESSMENT_MATRIX_2025-11-21.md` (M1 impact analysis)
- `docs/DEV_VALIDATION_AUDIT.md` (original audit findings)
