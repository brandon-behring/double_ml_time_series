# Deep Investigation Progress: Validation Issues Impact Analysis

**Date**: 2025-11-21
**Purpose**: Empirically test whether documented validation issues actually matter
**Status**: IN PROGRESS

## Motivation

User requested deeper investigation to validate that the 5 documented issues in DEV_VALIDATION_AUDIT.md are:
1. Actually problems (not theoretical concerns)
2. Worth fixing (material impact on results)
3. Critical for academic/research credibility

This investigation runs empirical tests to measure impact before committing 2-3 days to fixes.

## Investigation Plan

### Phase 1: Empirical Testing of Documented Issues (~9 hours total)

#### Issue C2: Binary Treatment Mis-specification ✅ IN PROGRESS
- **Current**: BiasValidation uses `discrete_treatment=False` + RandomForestRegressor for binary treatment
- **Should be**: `discrete_treatment=True` + RandomForestClassifier
- **Test**: Run 100 simulations across 5 scenarios comparing both configurations
- **Decision criteria**:
  - Bias difference >5% → Fix is critical
  - Coverage difference >10% → Fix is critical
- **Script**: `scripts/test_issue_c2_binary_treatment.py`
- **Status**: Running in background (est. 10-15 min)

#### Issue C3: Monte Carlo Comparability ⏳ PREPARED
- **Current**: BaselineComparison reuses DGPGenerator, methods see different random draws
- **Should be**: Clone DGP state or pre-generate datasets
- **Test**: Compare ranking stability across 10 runs with 5 methods
- **Decision criteria**:
  - Rankings change >2 positions → Fix is critical
  - Relative bias differs >10% → Fix is critical
- **Script**: `scripts/test_issue_c3_monte_carlo_comparability.py`
- **Status**: Script ready, will run after C2 completes

#### Issue C5: 401(k) Covariate Mismatch ⏳ PENDING
- **Current**: Uses 9 controls instead of published 11
- **Should be**: Include all 11 covariates (marital status, two-earner status)
- **Test**: Compare estimates with 9 vs 11 covariates to published results
- **Decision criteria**:
  - Estimate moves >$500 closer to published → Fix valuable
  - Different conclusions → Fix critical
- **Script**: To be created
- **Status**: Not started

#### Issue M1: Bootstrap Refits ⏳ PENDING
- **Current**: IPW/AIPW don't refit first-stage models in bootstrap
- **Should be**: Refit propensity and outcome models in each replicate
- **Test**: Compare CI widths and coverage with/without refitting
- **Decision criteria**:
  - Coverage changes >5% → Fix needed
  - CI widths materially different → Fix needed
- **Script**: To be created
- **Status**: Not started

#### Issue M2: CI Reporting ⏳ PENDING
- **Current**: Mixes bias percentiles with effect CIs in reporting
- **Should be**: Separate and clearly label different CI types
- **Test**: This is documentation, not methodology
- **Decision**: Just fix it (~1 hour)
- **Status**: Not started

### Phase 2: Unidentified Issues (~3 hours)

- Sensitivity analysis across wider parameter ranges
- Code review for statistical correctness
- Literature comparison with reference implementations
- Edge case testing

### Phase 3: Meta-Analysis (~2 hours)

- **Health check on Brandon**: Phantom crisis pattern recognition
- **Health check on Claude**: Understanding accuracy verification
- **Impact assessment matrix**: Data-driven prioritization
- **Decision document**: Which issues to fix vs. document

## Key Questions Being Answered

1. **Is the discrete_treatment issue actually causing bias?**
   - Or is it a theoretical concern that doesn't matter in practice?

2. **Do unstable random draws actually change method rankings?**
   - Or do comparisons remain valid despite different seeds?

3. **Does the 401(k) covariate mismatch explain the Lasso divergence?**
   - Or are there other factors at play?

4. **Do bootstrap refits actually improve coverage?**
   - Or is the current approach good enough for teaching purposes?

5. **Most importantly: Are we solving real problems or creating work?**

## Evidence-Based Decision Making

Instead of assuming fixes are needed based on theory, we're:
- Running actual simulations
- Measuring actual impact
- Comparing to decision criteria
- Making data-driven choices

This aligns with:
- Brandon's "reality check before solution" principle
- Scientific rigor in validation work
- Avoiding phantom crisis pattern

## Expected Outcomes

### Best Case
All issues have <5% impact → Document limitations, continue with book

### Worst Case
All issues have >10% impact → Systematic fixes needed (2-3 days)

### Most Likely
Mixed results → Fix critical issues (1-2 days), document others

## Timeline

- **Phase 1 Day 1** (today): C2, C3, M2 testing
- **Phase 1 Day 2**: C5, M1 testing + sensitivity analysis
- **Phase 2 Day 2**: Code review + literature comparison
- **Phase 3 Day 2**: Meta-analysis + decision document

**Total**: 2 days investigation → Evidence-based decision on fixes

## Meta-Learning

This investigation is itself a test of:
- Brandon's ADHD-optimized workflows (reality check first)
- Claude's understanding accuracy (are we solving real problems?)
- Project health assessment methods (phantom vs. real crisis)

The investigation process is as valuable as the results - it teaches us how to evaluate technical debt scientifically rather than reactively.

---

**Updates will be appended below as tests complete...**

## Test Results

### Issue C2: Binary Treatment Mis-specification ✅ COMPLETED

**Status**: ✅ **CRITICAL FIX REQUIRED**
**Completed**: 2025-11-21 23:27 UTC
**Runtime**: ~55 minutes (100 simulations × 2 configs × 5 scenarios)

**Results Summary**:

| Metric | CURRENT (discrete_treatment=False) | CORRECTED (discrete_treatment=True) | Difference |
|--------|-----------------------------------|-------------------------------------|------------|
| Avg Bias | -0.0761 | -0.0103 | **86.5% reduction** |
| Avg Coverage | 99.6% | 100.0% | +0.4pp |

**Critical Findings**:
1. **Bias differences range from 47.8% to 113.9%** across scenarios
2. **Scenario 3** (challenging: n=200, strong confounding):
   - CURRENT bias: -0.1930 (9.65% of true effect)
   - CORRECTED bias: -0.0015 (0.08% of true effect)
   - **99.2% bias reduction!**
3. **ALL 5 scenarios** exceeded the 5% decision threshold

**Decision**: **FIX REQUIRED** - Priority: **HIGH**

**Reasoning**: The binary treatment mis-specification causes substantial bias, particularly in challenging scenarios (small n, strong confounding). This is NOT a theoretical concern - it materially affects estimates across all tested configurations.

**Impact on Book**: Current validation results using `discrete_treatment=False` are systematically biased. Fixing this will:
- Improve bias from -7.6% to -1.0% average
- Strengthen academic credibility
- Align with DML theory (orthogonal scores for binary treatment)

**Effort Estimate**: 4-6 hours (update BiasValidation, empirical_replication, lasso_diagnostic)

**CSV**: `results/impact_analysis/issue_c2_impact_20251121_173022.csv`

---

### Issue C3: Monte Carlo Comparability ✅ COMPLETED

**Status**: ✅ **CRITICAL FIX REQUIRED**
**Completed**: 2025-11-22 02:03 UTC
**Runtime**: ~2.5 hours (100 simulations × 5 methods × 10 runs × 3 scenarios)

**Results Summary**:

| Scenario | Max Rank Variance | Max Rank Range | Stability |
|----------|-------------------|----------------|-----------|
| Moderate confounding | 1.25 std dev | 3 positions | ⚠️ HIGH INSTABILITY |
| Strong confounding | 1.84 std dev | **4 positions** | ⚠️ HIGH INSTABILITY |
| Weak confounding | 1.84 std dev | **4 positions** | ⚠️ HIGH INSTABILITY |

**Critical Findings**:
1. **Rankings unstable across all scenarios** - methods seeing different random draws leads to inconsistent comparisons
2. **NaiveOLS in Scenario 3**: Ranks from #1 to #5 across runs (complete instability!)
3. **OLSWithControls**: More stable but still varies 1-4 positions
4. **Max std deviation: 1.84** (threshold was 1.0 for acceptable stability)
5. **ALL 3 scenarios** showed HIGH INSTABILITY warning

**Decision**: **FIX REQUIRED** - Priority: **HIGH**

**Reasoning**: Method rankings are not reproducible when comparing methods that see different random draws. This makes baseline comparisons scientifically invalid - you cannot determine which method is actually better because rankings depend on which random seed was used.

**Impact on Book**: Current BaselineComparison results are unreliable for determining method superiority. Tables showing "DML is better than IPW" might reverse with different random draws.

**Fix Approach**: Clone DGP state before each method or pre-generate datasets, ensuring all methods evaluate on identical simulations.

**Effort Estimate**: 2-3 hours (modify BaselineComparison.compare() to use deterministic data sharing)

**CSV**: `results/impact_analysis/issue_c3_impact_20251121_193852.csv`

---

### Issue C5: 401(k) Covariate Mismatch ✅ COMPLETED

**Status**: ✅ **FIX RECOMMENDED**
**Completed**: 2025-11-22 02:10 UTC
**Runtime**: ~5 minutes (2 configurations × 2 methods on n=9,915 dataset)

**Results Summary**:

| Method | CURRENT (9 cov) | CORRECTED (11 cov) | Improvement | Published Target |
|--------|-----------------|---------------------|-------------|------------------|
| Random Forest | $8,064 ($1,063 diff) | $7,025 ($2,102 diff) | **-$1,039 WORSE** | $9,127 |
| Lasso | $3,868 ($5,712 diff) | $6,705 ($2,875 diff) | **+$2,837 BETTER** | $9,580 |

**Critical Findings**:
1. **Lasso dramatically improves**: Error reduces from 59.6% to 30.0% with full covariates
2. **Random Forest gets worse**: Error increases from 11.6% to 23.0% (counterintuitive!)
3. **Missing covariates** (marr, twoearn) explain Lasso diagnostic issues from SESSION_2025-11-15
4. **Method-specific impact**: Not all estimators benefit equally from additional controls

**Decision**: **FIX RECOMMENDED** - Priority: **MEDIUM-HIGH**

**Reasoning**: While RF performs worse, Lasso improvement of $2,837 (exceeding $500 threshold) is substantial and explains previous diagnostic failures. For academic credibility, should match published covariate specification.

**Impact on Book**: Current 401(k) replication shows:
- Lasso performing very poorly (59.6% error) - misleading about Lasso quality
- Missing key demographic variables (marital status, two-earner households)
- Fix will show Lasso is actually reasonable (30% error vs 60%)

**Fix Approach**: Include all 11 published covariates in preprocessing (lines 191-195 in empirical_replication.py)

**Effort Estimate**: 1-2 hours (update preprocessing, regenerate 401k results, update documentation)

**CSV**: `results/impact_analysis/issue_c5_impact_20251121_210521.csv`

**Note**: RF getting worse is likely due to overfitting with more features on small effective sample after treatment/propensity matching.

