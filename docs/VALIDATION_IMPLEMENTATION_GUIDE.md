# Validation Implementation Guide

**Phase 1B Infrastructure - Complete Implementation Guide**

This guide explains how to implement new validation methods using the infrastructure built in Phase 1B.1 (Days 1-5).

## Table of Contents

1. [Quick Start](#quick-start)
2. [Infrastructure Overview](#infrastructure-overview)
3. [Implementing a Validation Method](#implementing-a-validation-method)
4. [Running Simulations](#running-simulations)
5. [Testing Your Implementation](#testing-your-implementation)
6. [Generating Reports](#generating-reports)
7. [Best Practices](#best-practices)

---

## Quick Start

**Implement a new validation method in 5 steps:**

```bash
# 1. Copy validation method template
cp templates/validation_method_template.py src/validation/bias_validation.py

# 2. Replace "TEMPLATE" with your method name
sed -i 's/TEMPLATE/Bias/g' src/validation/bias_validation.py

# 3. Implement your estimator in _estimate_effect()
# Edit src/validation/bias_validation.py

# 4. Copy test template and implement tests
cp templates/test_harness_template.py test/validation/test_bias_validation.py

# 5. Run tests
python -m pytest test/validation/test_bias_validation.py -v
```

---

## Infrastructure Overview

### Core Components

**Built in Phase 1B.1** (Days 1-5):

| Component | Purpose | Coverage | Tests |
|-----------|---------|----------|-------|
| `dgp_generator.py` | Generate synthetic datasets | 99% | 27 tests |
| `validation_result.py` | Standardized result format | 100% | 16 tests |
| `parallel.py` | 48-core parallel execution | 95% | 33 tests |
| `plotting.py` | Professional plots | N/A | TBD |
| `storage.py` | Result caching | N/A | TBD |

**Templates**:
- `validation_method_template.py` - Copy-paste ready method implementation
- `simulation_template.py` - Parameter sweep simulations
- `test_harness_template.py` - Comprehensive test structure

**Tools**:
- `notebook_to_latex.py` - Convert Jupyter notebooks to LaTeX reports

---

## Implementing a Validation Method

### Step 1: Copy the Template

```bash
cp templates/validation_method_template.py src/validation/my_validation.py
```

### Step 2: Rename the Class

Replace `TEMPLATE` with your validation method name (e.g., `Bias`, `Coverage`, `Power`):

```python
# Change this:
class TEMPLATEValidation:
    ...

# To this:
class BiasValidation:
    ...
```

### Step 3: Implement Your Estimator

Find the `_estimate_effect()` method and replace the placeholder with your estimator:

```python
def _estimate_effect(self, data: DGPResult) -> float:
    """
    Estimate treatment effect from data.

    TODO: Implement your estimator here.
    """
    # Example: DML estimator
    from econml.dml import LinearDML
    from sklearn.ensemble import RandomForestRegressor

    model_y = RandomForestRegressor(n_estimators=100, random_state=42)
    model_t = RandomForestRegressor(n_estimators=100, random_state=42)

    dml = LinearDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=False
    )

    dml.fit(Y=data.Y, T=data.T, X=data.X)
    return dml.effect(X=data.X).mean()
```

### Step 4: Customize Validation Thresholds

Adjust thresholds in `_determine_status()`:

```python
def _determine_status(self, bias: float, mse: float, coverage: float) -> str:
    # Customize these thresholds for your validation criteria
    BIAS_THRESHOLD = 0.1  # Max acceptable bias
    MSE_THRESHOLD = 0.5   # Max acceptable MSE
    COVERAGE_LOWER = 0.93  # Min coverage (95% ± 2%)
    COVERAGE_UPPER = 0.97  # Max coverage

    # ... validation logic ...
```

### Step 5: Update Metadata

Update the `validate()` method to include relevant metadata:

```python
result = ValidationResult(
    method="BiasValidation",  # Your method name
    status=status,
    bias=bias,
    mse=mse,
    coverage=coverage,
    ci_lower=ci_lower,
    ci_upper=ci_upper,
    n_simulations=self.n_simulations,
    timestamp=datetime.now(),
    metadata={
        "dgp_n": dgp.n,
        "dgp_p": dgp.p,
        "dgp_true_effect": dgp.true_effect,
        "dgp_confounding": dgp.confounding_strength,
        "alpha": self.alpha,
        # Add custom metadata
        "estimator": "LinearDML",
        "model_y": "RandomForestRegressor",
        "model_t": "RandomForestRegressor",
    },
)
```

---

## Running Simulations

### Option 1: Use Validation Method Directly

```python
from src.validation import DGPGenerator, BiasValidation

# Create DGP
dgp = DGPGenerator(
    n=1000,
    p=5,
    true_effect=2.0,
    confounding_strength=1.0,
    random_state=42
)

# Run validation
validator = BiasValidation(n_simulations=1000, random_state=42)
result = validator.validate(dgp)

print(result.summary())
# Bias Validation: PASS
# Bias: 0.023 (95% CI: [-0.045, 0.091])
# MSE: 0.156
# Coverage: 0.947
# Simulations: 1000
```

### Option 2: Use Simulation Template for Parameter Sweep

```bash
# Copy simulation template
cp templates/simulation_template.py scripts/bias_simulation.py

# Edit parameters
# SAMPLE_SIZES = [500, 1000, 2000, 5000]
# CONFOUNDER_COUNTS = [5, 10, 20]
# TRUE_EFFECTS = [1.0, 2.0, 3.0]

# Run simulation
python scripts/bias_simulation.py
```

**Parallel execution** (automatic with 48-core Threadripper):

```python
from src.validation import parallel_monte_carlo

results = parallel_monte_carlo(
    simulation_func=run_single_simulation,
    n_simulations=1000,
    n_jobs=48,  # Use 48 cores
    show_progress=True,
    random_state=42,
    # Pass additional kwargs
    n=1000,
    p=5,
    true_effect=2.0,
)
```

---

## Testing Your Implementation

### Step 1: Copy Test Template

```bash
cp templates/test_harness_template.py test/validation/test_my_validation.py
```

### Step 2: Update Test Class Names

Replace `TEMPLATE` with your method name:

```python
# Change:
class TestTEMPLATEBasicFunctionality:
    ...

# To:
class TestBiasValidationBasicFunctionality:
    ...
```

### Step 3: Implement Test Cases

Fill in the TODOs in each test class:

```python
class TestBiasValidationBasicFunctionality:
    def test_basic_execution(self):
        """Should run validation successfully."""
        dgp = DGPGenerator(n=100, p=5, true_effect=2.0, random_state=42)
        validator = BiasValidation(n_simulations=10, random_state=42)

        result = validator.validate(dgp)

        assert result.method == "BiasValidation"
        assert result.status in ["PASS", "FAIL", "WARNING"]
        assert isinstance(result.bias, float)
```

### Step 4: Run Tests

```bash
# Run tests for your validation method
python -m pytest test/validation/test_my_validation.py -v

# Run with coverage
python -m pytest test/validation/test_my_validation.py --cov=src/validation/my_validation

# Run all validation tests
python -m pytest test/validation/ -v
```

**Pre-commit hooks will automatically**:
- Run all tests before commit (blocks if fail)
- Check coverage ≥80% before push (blocks if below)
- Validate template syntax

---

## Generating Reports

### Option 1: Create Jupyter Notebook

```python
# In notebook cell
from src.validation import DGPGenerator, BiasValidation, plotting
import numpy as np

# Run validation
dgp = DGPGenerator(n=1000, p=5, true_effect=2.0, random_state=42)
validator = BiasValidation(n_simulations=1000, random_state=42)
result = validator.validate(dgp)

# Get bias estimates
# (You'll need to modify validator to return estimates)
estimates = validator._last_estimates  # Store in validate()

# Plot bias distribution
fig = plotting.plot_bias_distribution(
    biases=estimates - dgp.true_effect,
    true_effect=dgp.true_effect,
    title="Bias Distribution: DML Estimator"
)
```

### Option 2: Convert Notebook to LaTeX

```bash
# Convert notebook to LaTeX
python scripts/documents/notebook_to_latex.py \
    validation_results.ipynb \
    validation_report.tex \
    --style=article

# Compile to PDF
pdflatex validation_report.tex
pdflatex validation_report.tex  # Second pass for references
```

### Option 3: Use Storage System

```python
from src.validation import ResultStorage

# Save results
storage = ResultStorage(base_dir="results/validation")
filepath = storage.save_result(result, method="bias", metadata={
    "experiment": "main_validation",
    "date": "2025-11-13"
})

# Load results later
results = storage.load_results(method="bias", status="PASS", limit=10)

# Generate summary
summary = storage.get_summary()
print(f"Total results: {summary['total_results']}")
print(f"By method: {summary['methods']}")
```

---

## Best Practices

### 1. Test-First Development

**Always write tests before implementation**:

```bash
# ✅ Correct workflow
cp templates/test_harness_template.py test/validation/test_bias.py
# Write tests first
python -m pytest test/validation/test_bias.py  # Should fail
# Implement method
python -m pytest test/validation/test_bias.py  # Should pass

# ❌ Wrong workflow
# Write implementation first → hard to test
```

### 2. Use Parallel Execution

**For large simulations, always use parallel utilities**:

```python
# ✅ Good: Parallel execution
from src.validation import parallel_monte_carlo

results = parallel_monte_carlo(
    simulation_func,
    n_simulations=10000,
    n_jobs=48
)

# ❌ Bad: Sequential loop
results = [simulation_func(i) for i in range(10000)]  # Very slow!
```

### 3. Reproducibility

**Always set random seeds**:

```python
# ✅ Good: Reproducible
dgp = DGPGenerator(..., random_state=42)
validator = MyValidation(..., random_state=42)

# ❌ Bad: Non-reproducible
dgp = DGPGenerator(...)  # Different results each run
```

### 4. Informative Metadata

**Include all relevant information**:

```python
# ✅ Good: Rich metadata
metadata = {
    "dgp_n": dgp.n,
    "dgp_p": dgp.p,
    "estimator": "LinearDML",
    "model_y": "RandomForestRegressor(n_estimators=100)",
    "model_t": "RandomForestRegressor(n_estimators=100)",
    "cv_folds": 5,
}

# ❌ Bad: Minimal metadata
metadata = {"dgp_n": dgp.n}  # Can't reproduce later
```

### 5. Coverage Targets

**Aim for high test coverage**:

- Modules: 80%+ required (enforced by pre-commit hook)
- Validation methods: 90%+ recommended
- Templates: 100% (ensure all paths work)

---

## Example Workflow

**Complete workflow for implementing Method 1 (Bias Validation)**:

```bash
# Day 1: Setup
cp templates/validation_method_template.py src/validation/bias_validation.py
cp templates/test_harness_template.py test/validation/test_bias_validation.py

# Day 2: Tests (test-first!)
# Edit test/validation/test_bias_validation.py
python -m pytest test/validation/test_bias_validation.py  # Should fail

# Day 3: Implementation
# Edit src/validation/bias_validation.py
python -m pytest test/validation/test_bias_validation.py  # Should pass

# Day 4: Simulation
cp templates/simulation_template.py scripts/bias_simulation.py
python scripts/bias_simulation.py  # Run parameter sweep

# Day 5: Documentation
# Create Jupyter notebook with results
# Convert to LaTeX report
python scripts/documents/notebook_to_latex.py bias_results.ipynb bias_report.tex

# Commit (hooks run automatically)
git add src/validation/bias_validation.py test/validation/test_bias_validation.py
git commit -m "feat(validation): Implement bias validation (Method 1)"
```

---

## Troubleshooting

### Tests Failing?

```bash
# Run with verbose output
python -m pytest test/validation/ -v --tb=long

# Run specific test
python -m pytest test/validation/test_my_validation.py::TestClass::test_method -v
```

### Coverage Too Low?

```bash
# See which lines are missing coverage
python -m pytest test/validation/ --cov=src/validation --cov-report=html
# Open htmlcov/index.html in browser
```

### Parallel Execution Slow?

```python
# Check optimal n_jobs for your workload
from src.validation.parallel import get_optimal_n_jobs

n_jobs = get_optimal_n_jobs(n_tasks=1000, n_jobs=48)
print(f"Optimal jobs: {n_jobs}")

# Try different backends
results = parallel_map(func, items, backend='loky')  # Default
results = parallel_map(func, items, backend='multiprocessing')
results = parallel_map(func, items, backend='threading')
```

### Template Syntax Errors?

```bash
# Validate template syntax
python -m py_compile templates/validation_method_template.py

# Check for common issues
python -c "import ast; ast.parse(open('templates/validation_method_template.py').read())"
```

---

## Next Steps

1. **Implement Methods 1-7** using this guide
2. **Run validation simulations** with parameter sweeps
3. **Generate LaTeX reports** for documentation
4. **Proceed to Phase 1B.2** (Methods implementation)

---

**Phase 1B.1 Complete**: Infrastructure ready for Methods 1-7 implementation.

**Created**: 2025-11-13
**Author**: Brandon Behring
**Project**: Double ML Validation Framework
