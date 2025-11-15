# Action Plan: Addressing Critical Issues

**Date**: 2025-11-14
**Priority**: CRITICAL - Must fix before proceeding
**Timeline**: 2-3 weeks to address fundamental issues

---

## PRIORITY 1: Fix Coverage Calculation (BLOCKING)

### The Problem
Coverage is showing 100% when it should be ~95%. This means either:
1. Confidence intervals are too wide (overconservative)
2. The test is flawed (only testing easy cases)
3. The calculation is still wrong

### Investigation Steps

#### Step 1: Test Coverage Across Multiple Scenarios
```python
# Create test script: scripts/test_coverage_scenarios.py

scenarios = [
    {"n": 500, "p": 5, "confounding": 0.5},   # Small sample
    {"n": 500, "p": 5, "confounding": 2.0},   # Strong confounding
    {"n": 1000, "p": 10, "confounding": 1.0}, # Many covariates
    {"n": 2000, "p": 20, "confounding": 1.5}, # Large p
    {"n": 5000, "p": 5, "confounding": 0.5},  # Large n
]

for scenario in scenarios:
    dgp = DGPGenerator(**scenario)
    validator = BiasValidation(n_simulations=200)
    result = validator.validate(dgp)
    print(f"{scenario}: Coverage = {result.coverage:.2%}")
```

Expected: Coverage should be 92-98% across scenarios, centering on 95%

#### Step 2: Examine CI Width
```python
# Are the CIs too wide?
def analyze_ci_width(dgp, n_runs=100):
    widths = []
    for _ in range(n_runs):
        data = dgp.generate()
        dml = LinearDML(...)
        dml.fit(Y=data.Y, T=data.T, X=data.X)
        ci_lower, ci_upper = dml.effect_interval(X=data.X, alpha=0.05)
        width = ci_upper.mean() - ci_lower.mean()
        widths.append(width)

    return np.mean(widths), np.std(widths)
```

#### Step 3: Compare CI Methods
```python
# Compare different CI construction methods
def compare_ci_methods(dgp):
    results = {}

    # Method 1: DML built-in CIs
    results['dml_builtin'] = get_dml_coverage(dgp)

    # Method 2: Bootstrap percentile
    results['bootstrap_percentile'] = get_bootstrap_coverage(dgp)

    # Method 3: Normal approximation
    results['normal'] = get_normal_coverage(dgp)

    # Method 4: Bootstrap BCa
    results['bootstrap_bca'] = get_bca_coverage(dgp)

    return results
```

### Fix Implementation

Create `scripts/fix_coverage_calculation.py`:
```python
"""
Fix the coverage calculation to properly achieve ~95% coverage
"""

def validate_coverage_calculation():
    """
    Ensure coverage calculation is correct:
    1. CI should contain true effect 95% of the time
    2. Not 100%, not 9%, but ~95%
    """
    # Implementation here
    pass

def implement_proper_ci_calculation():
    """
    Use the most appropriate CI method based on:
    - Sample size
    - Distribution of estimates
    - Bootstrap diagnostics
    """
    pass
```

---

## PRIORITY 2: Fix DGP Generator Realism

### Current Issues
- Only linear relationships
- No time series structure
- No heterogeneous effects
- Unrealistic error distributions

### Enhanced DGP Implementation

Create `src/validation/enhanced_dgp.py`:
```python
class EnhancedDGPGenerator:
    """
    Realistic data generating process with:
    - Non-linear confounding
    - Heterogeneous treatment effects
    - Time series structure
    - Model misspecification scenarios
    """

    def __init__(self,
                 n: int,
                 p: int,
                 true_effect: float,
                 scenario: str = "linear",  # linear, nonlinear, heterogeneous, temporal
                 **kwargs):
        self.scenario = scenario
        # ...

    def generate_nonlinear(self):
        """Non-linear confounding and outcomes"""
        X = self.generate_covariates()

        # Non-linear propensity score
        ps = 1 / (1 + np.exp(-(X[:, 0]**2 + np.sin(X[:, 1]) + X[:, 2])))
        T = np.random.binomial(1, ps)

        # Non-linear outcome with interactions
        Y = (self.true_effect * T * (1 + 0.5 * X[:, 0]) +  # Heterogeneous effect
             X[:, 0]**2 + np.sqrt(np.abs(X[:, 1])) +        # Non-linear main effects
             X[:, 2] * X[:, 3] +                             # Interactions
             np.random.standard_t(df=5, size=self.n))        # Heavy-tailed errors

        return DGPResult(X, T, Y)

    def generate_temporal(self):
        """Time series structure with autocorrelation"""
        # AR(1) errors, time-varying treatment effects, etc.
        pass
```

---

## PRIORITY 3: Statistical Testing Corrections

### Fix Multiple Testing Problem

Create `src/validation/statistical_testing.py`:
```python
class StatisticalTester:
    """
    Proper statistical testing with:
    - Multiple comparison correction
    - Power analysis
    - Effect size calculation
    """

    def test_bias_with_correction(self, results_dict, alpha=0.05):
        """
        Test bias for multiple methods with Bonferroni correction
        """
        n_tests = len(results_dict)
        corrected_alpha = alpha / n_tests

        test_results = {}
        for method, result in results_dict.items():
            p_value = self._calculate_p_value(result.bias_samples)
            test_results[method] = {
                'p_value': p_value,
                'corrected_alpha': corrected_alpha,
                'significant': p_value < corrected_alpha,
                'effect_size': self._calculate_effect_size(result)
            }

        return test_results

    def calculate_required_sample_size(self, effect_size, power=0.8, alpha=0.05):
        """
        Determine sample size needed for desired power
        """
        # Use statsmodels power analysis
        from statsmodels.stats.power import tt_solve_power
        n = tt_solve_power(effect_size=effect_size,
                          alpha=alpha,
                          power=power)
        return int(np.ceil(n))
```

---

## PRIORITY 4: Complete Validation Suite

### Implement All 7 Validation Methods

#### Method 1: Published Results Replication
```python
# scripts/validate_published_results.py
def replicate_chernozhukov_2018():
    """
    Replicate 401(k) analysis from Chernozhukov et al. (2018)
    Expected ATE: ~7000-8000
    """
    # Load data
    # Run DML
    # Compare results
    pass
```

#### Method 2: Synthetic Monte Carlo (FIX EXISTING)
```python
# Fix the coverage issue first!
```

#### Method 3: Cross-Implementation Validation
```python
# scripts/cross_implementation_validation.py
def compare_implementations():
    """
    Compare our implementation with:
    - EconML (Microsoft)
    - DoubleML (R package)
    - CausalML (Uber)
    """
    results = {}

    # Our implementation
    results['ours'] = run_our_dml(data)

    # EconML
    from econml.dml import LinearDML as EconMLDML
    results['econml'] = run_econml(data)

    # Compare
    return compare_results(results)
```

#### Method 4: Diagnostics Suite
```python
# scripts/diagnostic_validation.py
def run_diagnostics(dml_model):
    """
    Comprehensive diagnostics:
    - First-stage R²
    - Residual plots
    - Sensitivity analysis
    - Overlap checks
    """
    diagnostics = {}

    # First-stage performance
    diagnostics['first_stage_r2'] = {
        'outcome_model': dml.model_y.score(X, Y),
        'treatment_model': dml.model_t.score(X, T)
    }

    # Residual analysis
    diagnostics['residuals'] = analyze_residuals(dml)

    # Sensitivity to unmeasured confounding
    diagnostics['sensitivity'] = sensitivity_analysis(dml)

    return diagnostics
```

---

## WEEK-BY-WEEK EXECUTION PLAN

### Week 1: Fix Fundamentals
**Days 1-3: Coverage Fix**
- [ ] Investigate why coverage is 100%
- [ ] Test across multiple scenarios
- [ ] Implement proper CI calculation
- [ ] Verify ~95% coverage achieved

**Days 4-5: Statistical Testing**
- [ ] Implement multiple testing correction
- [ ] Add power analysis
- [ ] Fix hypothesis testing approach

### Week 2: Enhance Realism
**Days 6-8: DGP Enhancement**
- [ ] Add non-linear scenarios
- [ ] Add heterogeneous effects
- [ ] Add time series structure
- [ ] Validate DGP properties

**Days 9-10: Baseline Improvements**
- [ ] Fix bootstrap implementation
- [ ] Standardize CI methods
- [ ] Add diagnostics

### Week 3: Complete Validation
**Days 11-12: Cross-Implementation**
- [ ] Compare with EconML
- [ ] Compare with DoubleML (R)
- [ ] Document discrepancies

**Days 13-14: Published Replication**
- [ ] Replicate Chernozhukov et al.
- [ ] Replicate other published results
- [ ] Verify accuracy

**Day 15: Integration & Documentation**
- [ ] Run full validation suite
- [ ] Document all findings
- [ ] Update roadmap based on results

---

## SUCCESS CRITERIA

Before proceeding past this fix phase:

1. **Coverage**: Consistently 93-97% across scenarios
2. **Bias**: Properly detected with corrected p-values
3. **DGP**: Realistic scenarios validated
4. **Validation**: At least 5/7 methods implemented
5. **Documentation**: All assumptions clearly stated

---

## DO NOT PROCEED TO PHASE 1 UNTIL

- [ ] Coverage is properly ~95% (not 100%, not 9%)
- [ ] Statistical tests use proper corrections
- [ ] DGP includes realistic scenarios
- [ ] Cross-implementation validation complete
- [ ] Published results successfully replicated

---

## Files to Create/Modify

Priority order:
1. `scripts/test_coverage_scenarios.py` - Test coverage across scenarios
2. `scripts/fix_coverage_calculation.py` - Fix the coverage issue
3. `src/validation/enhanced_dgp.py` - Realistic DGP generator
4. `src/validation/statistical_testing.py` - Proper statistical tests
5. `scripts/cross_implementation_validation.py` - Compare with other packages
6. `scripts/diagnostic_validation.py` - Comprehensive diagnostics
7. `scripts/validate_published_results.py` - Replicate published work

---

**This plan addresses the critical issues identified in the review. DO NOT skip steps or proceed without fixing the fundamentals.**