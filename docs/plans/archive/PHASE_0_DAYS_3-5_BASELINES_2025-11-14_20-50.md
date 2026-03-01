# Phase 0 Days 3-5: Baseline Methods (OLS + IPW)

**Created**: 2025-11-14 20:50
**Updated**: 2025-11-14 20:50
**STATUS**: IN_PROGRESS
**Objective**: Implement OLS and IPW baseline estimators for comparative analysis

---

## Executive Summary

**Current State**: Statistical hypothesis tests complete (Day 2). BiasValidation alone doesn't prove DML superiority.

**Solution**: Build OLS and IPW baselines to answer: *Is DML actually better than simpler methods?*

**Baselines to Implement**:
1. **NaiveOLS**: Y ~ T only (no controls) - simplest possible
2. **OLSWithControls**: Y ~ T + X - standard econometrics
3. **IPWEstimator**: Inverse propensity weighting
4. **AugmentedIPW**: Doubly robust (combines outcome modeling + propensity)

**Timeline**: 3 days (Days 3-5)
**MC Runs**: 100 (same as DML for comparison)

---

## Architecture

### File Structure

```
src/validation/
├── bias_validation.py        (existing: DML estimator)
├── ols_baseline.py           (NEW: OLS baselines)
├── ipw_baseline.py           (NEW: IPW baselines)
├── baseline_comparison.py     (NEW: unified runner)
└── validation_result.py       (existing: result format)

test/validation/
├── test_bias_validation.py    (existing: DML tests)
├── test_ols_baseline.py       (NEW: OLS tests)
├── test_ipw_baseline.py       (NEW: IPW tests)
└── test_baseline_comparison.py (NEW: runner tests)
```

### BaselineEstimator Interface

All estimators implement common interface:

```python
class BaselineEstimator:
    def __init__(self, **kwargs):
        pass

    def validate(self, dgp: DGPGenerator) -> ValidationResult:
        """Run estimation on DGP and return results"""
        # Monte Carlo simulation
        # Returns: ValidationResult with status, bias, mse, coverage
```

Returns same `ValidationResult` format as BiasValidation for direct comparison.

---

## Days 3-5 Implementation Plan

### Day 3: OLS Baselines (8-10 hours)

#### 3.1: NaiveOLS Estimator (2 hours)

**File**: `src/validation/ols_baseline.py`

```python
class NaiveOLS:
    """Y ~ T only (no confounders)"""

    def __init__(self, n_simulations=100, alpha=0.05, random_state=None):
        self.n_simulations = n_simulations
        self.alpha = alpha
        self._rng = np.random.RandomState(random_state)

    def validate(self, dgp: DGPGenerator) -> ValidationResult:
        """Run MC simulations"""
        estimates = []
        ci_bounds = []

        for _ in range(self.n_simulations):
            data = dgp.generate()

            # OLS: Y ~ T (no X controls)
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(data.T.reshape(-1, 1), data.Y)

            ate = model.coef_[0]
            # CI: from standard errors
            se = ...  # calculated from residuals
            ci_lower, ci_upper = ate - 1.96*se, ate + 1.96*se

            estimates.append(ate)
            ci_bounds.append((ci_lower, ci_upper))

        # Calculate metrics
        bias = np.mean(estimates) - dgp.true_effect
        mse = np.mean((estimates - dgp.true_effect)**2)
        coverage = ...  # proportion CIs contain truth

        # Return ValidationResult
        return ValidationResult(
            method="NaiveOLS",
            status=...,  # determined by hypothesis tests
            bias=bias,
            mse=mse,
            coverage=coverage,
            ...
        )
```

**Key differences from DML**:
- No cross-fitting
- No nuisance parameter estimation
- Suffers from bias from confounding (since X not controlled)

**Tests** (3-4 tests):
- Instantiation
- Systematic bias detection (should FAIL with confounding)
- Reproducibility
- Parameter validation

#### 3.2: OLSWithControls Estimator (2 hours)

**File**: `src/validation/ols_baseline.py`

```python
class OLSWithControls:
    """Y ~ T + X (controls for X)"""

    def validate(self, dgp):
        """Run MC simulations"""
        estimates = []

        for _ in range(self.n_simulations):
            data = dgp.generate()

            # OLS: Y ~ T + X (controls included)
            X_with_T = np.column_stack([data.T, data.X])
            model = LinearRegression()
            model.fit(X_with_T, data.Y)

            # ATE = coefficient on T
            ate = model.coef_[0]
            estimates.append(ate)

        # Calculate metrics
        bias = np.mean(estimates) - dgp.true_effect
        mse = np.mean((estimates - dgp.true_effect)**2)

        # Should show less bias than NaiveOLS (confounding removed)
        # But less efficient than DML (no double robustness)

        return ValidationResult(...)
```

**Key properties**:
- Less biased than NaiveOLS (X confounding controlled)
- But inefficient (no debiasing)
- Still vulnerable to model misspecification (linear assumed)

**Tests** (3-4 tests):
- Less bias than NaiveOLS
- Still biased vs true effect
- Reproducibility
- Parameter validation

#### 3.3: Unified OLS Module + Tests (2 hours)

Combine NaiveOLS and OLSWithControls in single file with:
- Shared helper functions
- Unified testing across both classes
- Type hints and docstrings
- ValidationResult consistency

**Test file**: `test/validation/test_ols_baseline.py`

```python
class TestNaiveOLS:
    def test_instantiation(self): ...
    def test_detects_confounding_bias(self): ...
    def test_reproducibility(self): ...
    def test_parameter_validation(self): ...

class TestOLSWithControls:
    def test_instantiation(self): ...
    def test_less_biased_than_naive(self): ...
    def test_reproducibility(self): ...
    def test_parameter_validation(self): ...
```

#### 3.4: OLS vs DML Comparison (2 hours)

Create simple comparison script:

```python
# scripts/compare_ols_vs_dml.py
from src.validation.dgp_generator import DGPGenerator
from src.validation.bias_validation import BiasValidation
from src.validation.ols_baseline import NaiveOLS, OLSWithControls

# Create test DGP
dgp = DGPGenerator(n=1000, p=5, true_effect=2.0,
                   confounding_strength=1.0, random_state=42)

# Run all three methods
dml = BiasValidation(n_simulations=100)
naive_ols = NaiveOLS(n_simulations=100)
ols_controls = OLSWithControls(n_simulations=100)

dml_result = dml.validate(dgp)
naive_result = naive_ols.validate(dgp)
controls_result = ols_controls.validate(dgp)

# Compare results
print(f"DML Bias: {dml_result.bias:.4f}")
print(f"NaiveOLS Bias: {naive_result.bias:.4f}")
print(f"OLSWithControls Bias: {controls_result.bias:.4f}")
```

**Expected output**:
- DML bias < OLS with controls bias < Naive OLS bias
- DML shows PASS, OLS with controls WARNING/FAIL, Naive FAIL

---

### Day 4: IPW Estimator (6-8 hours)

#### 4.1: IPWEstimator (Inverse Propensity Weighting) (3-4 hours)

**File**: `src/validation/ipw_baseline.py`

```python
class IPWEstimator:
    """Inverse propensity weighting estimator"""

    def __init__(self, n_simulations=100, alpha=0.05, random_state=None):
        self.n_simulations = n_simulations
        self.alpha = alpha
        self._rng = np.random.RandomState(random_state)

    def validate(self, dgp: DGPGenerator) -> ValidationResult:
        estimates = []
        ci_bounds = []

        for _ in range(self.n_simulations):
            data = dgp.generate()

            # Step 1: Estimate propensity score P(T=1|X)
            from sklearn.linear_model import LogisticRegression
            ps_model = LogisticRegression(max_iter=1000)
            ps_model.fit(data.X, data.T)
            propensity_scores = ps_model.predict_proba(data.X)[:, 1]

            # Step 2: IPW estimator
            # E[Y * T / P(T=1|X)] - E[Y * (1-T) / (1-P(T=1|X))]
            weights_t = data.T / (propensity_scores + 1e-6)
            weights_0 = (1 - data.T) / (1 - propensity_scores + 1e-6)

            ate = np.mean(data.Y * weights_t) - np.mean(data.Y * weights_0)

            # Approximate CI (needs proper variance estimation)
            se = ... # bootstrap or analytic formula
            ci_lower, ci_upper = ate - 1.96*se, ate + 1.96*se

            estimates.append(ate)
            ci_bounds.append((ci_lower, ci_upper))

        # Calculate metrics
        bias = np.mean(estimates) - dgp.true_effect
        mse = np.mean((estimates - dgp.true_effect)**2)
        coverage = ...

        return ValidationResult(...)
```

**Key properties**:
- Doubly robust property: consistent if either outcome or propensity model correct
- Sensitive to positivity/overlap violations (extreme propensity scores)
- More efficient than OLS on some problems
- Can be unstable with poor overlap

**Tests** (4-5 tests):
- Instantiation
- Consistency with simpler models
- Overlap violation detection (propensity scores near 0 or 1)
- Reproducibility
- Parameter validation

#### 4.2: AugmentedIPW (Doubly Robust) (2-3 hours)

```python
class AugmentedIPW:
    """Augmented inverse propensity weighting (doubly robust)"""

    def validate(self, dgp: DGPGenerator) -> ValidationResult:
        estimates = []

        for _ in range(self.n_simulations):
            data = dgp.generate()

            # Step 1: Estimate outcome models
            y1_model = LinearRegression()
            y0_model = LinearRegression()

            y1_model.fit(data.X[data.T==1], data.Y[data.T==1])
            y0_model.fit(data.X[data.T==0], data.Y[data.T==0])

            y1_pred = y1_model.predict(data.X)
            y0_pred = y0_model.predict(data.X)

            # Step 2: Estimate propensity score
            ps_model = LogisticRegression(max_iter=1000)
            ps_model.fit(data.X, data.T)
            propensity_scores = ps_model.predict_proba(data.X)[:, 1]

            # Step 3: Doubly robust combination
            # Averages outcome predictions + IPW residuals
            ate = np.mean(y1_pred - y0_pred) + \
                  np.mean(data.T * (data.Y - y1_pred) / (propensity_scores + 1e-6)) - \
                  np.mean((1 - data.T) * (data.Y - y0_pred) / (1 - propensity_scores + 1e-6))

            estimates.append(ate)

        # Calculate metrics
        bias = np.mean(estimates) - dgp.true_effect
        mse = np.mean((estimates - dgp.true_effect)**2)

        return ValidationResult(...)
```

**Key properties**:
- Combines outcome regression + IPW
- Consistent if EITHER outcome or propensity model correct
- More robust than pure IPW or pure outcome regression
- Should outperform both on misspecified models

**Tests** (4-5 tests):
- Instantiation
- Robustness to outcome model misspecification
- Robustness to propensity model misspecification
- Reproducibility
- Parameter validation

---

### Day 5: Unified Framework + Comparison (6-8 hours)

#### 5.1: BaselineComparison Class (3-4 hours)

**File**: `src/validation/baseline_comparison.py`

```python
class BaselineComparison:
    """Run all baseline methods on same data"""

    def __init__(self, n_simulations=100, random_state=None):
        self.methods = {
            'DML': BiasValidation(n_simulations),
            'NaiveOLS': NaiveOLS(n_simulations),
            'OLSWithControls': OLSWithControls(n_simulations),
            'IPW': IPWEstimator(n_simulations),
            'AugmentedIPW': AugmentedIPW(n_simulations),
        }
        self.random_state = random_state

    def compare(self, dgp: DGPGenerator) -> Dict[str, ValidationResult]:
        """Run all methods, return results dict"""
        results = {}
        for name, method in self.methods.items():
            results[name] = method.validate(dgp)
        return results

    def create_comparison_table(self, dgp: DGPGenerator) -> pd.DataFrame:
        """Create summary table"""
        results = self.compare(dgp)

        data = []
        for name, result in results.items():
            data.append({
                'Method': name,
                'Bias': result.bias,
                'Bias 95% CI': f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]",
                'MSE': result.mse,
                'RMSE': np.sqrt(result.mse),
                'Coverage': f"{result.coverage:.1%}",
                'Status': result.status,
                'P-value (Bias)': result.bias_p_value if hasattr(result, 'bias_p_value') else np.nan,
            })

        return pd.DataFrame(data)
```

**Output**: Summary table comparing all methods on single DGP

#### 5.2: Comprehensive Comparison Tests (2-3 hours)

**File**: `test/validation/test_baseline_comparison.py`

```python
class TestBaselineComparison:
    def test_all_methods_return_results(self): ...
    def test_dml_outperforms_naive_ols(self): ...
    def test_ipw_better_than_ols_controls(self): ...
    def test_augmented_ipw_most_robust(self): ...
    def test_comparison_table_format(self): ...
```

#### 5.3: Generate Comparison Output (1-2 hours)

**Script**: `scripts/comprehensive_baseline_comparison.py`

```python
# Run across multiple DGPs
from src.validation.baseline_comparison import BaselineComparison

configurations = [
    {'n': 500, 'p': 5, 'confounding': 0.5},
    {'n': 1000, 'p': 5, 'confounding': 1.0},
    {'n': 2000, 'p': 10, 'confounding': 2.0},
]

for config in configurations:
    dgp = DGPGenerator(**config, random_state=42)
    comparison = BaselineComparison(n_simulations=100)
    table = comparison.create_comparison_table(dgp)

    print(f"\nConfiguration: {config}")
    print(table.to_string())
    print(f"\nWinner: {table.loc[table['Bias'].abs().idxmin(), 'Method']}")
```

**Expected output**:
- DML superior bias for strong confounding
- OLS with controls similar bias for weak confounding
- IPW unstable with extreme propensity scores
- AugmentedIPW robust across scenarios

---

## Success Criteria

### Day 3 (OLS) ✓
- [ ] NaiveOLS implemented and tested (4 tests)
- [ ] OLSWithControls implemented and tested (4 tests)
- [ ] Simple comparison shows DML > OLS with controls > NaiveOLS
- [ ] All tests passing

### Day 4 (IPW) ✓
- [ ] IPWEstimator implemented and tested (5 tests)
- [ ] AugmentedIPW implemented and tested (5 tests)
- [ ] Overlap detection working (warns on poor overlap)
- [ ] All tests passing

### Day 5 (Framework) ✓
- [ ] BaselineComparison class working
- [ ] Comparison table generation working
- [ ] Comprehensive comparison script producing output
- [ ] All 20+ tests passing

**Total Tests Added**: 20-25 new test methods

---

## Technical Notes

### Type Safety
All methods must:
- Accept DGPGenerator, return ValidationResult
- Have full type hints
- Pass mypy checks

### Variance Estimation
For CI calculation:
- OLS: analytical standard errors from sklearn
- IPW: bootstrap or analytical formulas
- AugmentedIPW: bootstrap variance

### Reproducibility
All methods use `random_state` parameter for reproducibility.

---

## Decisions Made

### Decision 1: Include Both OLS Variants
- Rationale: Shows importance of controlling confounding
- NaiveOLS demonstrates confounding bias clearly
- OLSWithControls represents standard econometrics practice

### Decision 2: Implement Doubly Robust (AugmentedIPW)
- Rationale: More robust than pure IPW or regression
- Modern best practice in causal inference
- Good test case for model misspecification

### Decision 3: Simple Propensity Score (LogisticRegression)
- Rationale: Fast, interpretable, sufficient for baseline
- Could upgrade to Random Forest later if needed
- Demonstrates overlap/positivity issues clearly

---

## Next Phase

After Day 5 complete:
- **Days 6-7**: ML baselines (RandomForest, XGBoost)
- **Day 8**: Unified comparison of all 5+ methods
- **Week 3**: Analysis, recommendations, publication-ready insights

---

**Status**: Ready to implement Days 3-5
**Estimated Completion**: 18-24 hours across 3 days
