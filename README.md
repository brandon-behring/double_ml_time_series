# Double Machine Learning for Time Series

**Production-grade DML methodology for time series causal inference, with insurance/annuity pricing applications and macroeconomic controls.**

<!-- Status badges (placeholder) -->
<!-- ![Tests](https://img.shields.io/badge/tests-796-green) -->
<!-- ![Python](https://img.shields.io/badge/python-3.11+-blue) -->
<!-- ![LaTeX](https://img.shields.io/badge/book-180_pages-orange) -->

---

## Overview

A complete reference book (180 pages, 10 chapters + Julia appendix) paired with a validated Python library implementing Double Machine Learning for time series causal inference. The progression FWL → Robinson → DML is pedagogically structured: each estimator builds on the last, from linear residualization to cross-fitted debiased inference.

**Book**: Professional LaTeX typesetting (scrbook + Tufte-style), zero compilation errors, full theorem environments with proofs, executable code examples throughout.

**Library**: 14,068 lines of Python (2,453 production module), 796 tests across a 4-tier system, validated against published results and Monte Carlo simulations.

---

## Features

### DML Estimators
- **FWL** — Frisch-Waugh-Lovell linear residualization baseline
- **Robinson** — Nonparametric partially linear model with ML nuisances
- **Double ML** — Cross-fitted Robinson with Neyman-orthogonal scores
- **DynamicDML** — Sequential g-estimation for time-varying treatment effects
- **RollingWindowDML** — Adaptive window estimation for non-stationary series
- **PanelDML** — Entity fixed effects with temporal blocking

### Time Series Infrastructure
- **TimeSeriesCrossValidator** — Temporal blocking, purged/embargoed cross-validation
- **HAC Standard Errors** — Newey-West with Bartlett kernel, automatic bandwidth
- **Stationarity Tests** — ADF, KPSS, Phillips-Perron (from mathematical primitives, verified against statsmodels)
- **FREDLoader** — Macroeconomic controls from Federal Reserve Economic Data

### Validation Suite (7 methods)
- Published results replication (Chernozhukov 2018, Facure 2022, 401(k) dataset)
- Synthetic Monte Carlo (1000 runs, 95% coverage verification)
- Cross-implementation comparison (manual vs EconML vs R DoubleML)
- First-stage diagnostics (R², residual analysis, sensitivity)
- Real-world known outcomes
- Public dataset benchmarks
- Synthetic DGP generators (cross-sectional, time series, insurance)

### Production Pipeline
- **DMLModelRegistry** — Model versioning and promotion (staging → production)
- **CausalMonitor** — Overlap, treatment distribution, nuisance quality monitoring
- **RetrainScheduler** — Causal-specific retraining triggers
- **InsuranceDMLPipeline** — End-to-end pipeline for insurance pricing

---

## Installation

**Requirements**: Python 3.11+

```bash
# Clone and install
git clone <repo-url>
cd double_ml_time_series

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Verify
python -c "from src.dml import double_ml, DynamicDML, RollingWindowDML; print('OK')"
```

---

## Quick Start

### Basic DML Estimation

```python
import numpy as np
from src.dml import double_ml

# Generate sample data
np.random.seed(42)
n = 500
X = np.random.randn(n, 5)           # Confounders
T = X @ np.random.randn(5) + np.random.randn(n)  # Treatment
Y = 2.0 * T + X @ np.random.randn(5) + np.random.randn(n)  # Outcome

# Estimate causal effect (true ATE = 2.0)
result = double_ml(Y, T, X, n_folds=5)
print(f"ATE: {result.theta:.3f} ± {result.se:.3f}")
print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
```

### Time-Varying Treatment Effects

```python
from src.dml import DynamicDML

# DynamicDML estimates effects across time periods
dml = DynamicDML(n_periods=10, n_splits=3)
result = dml.fit(Y, D, X, time_index=time_idx)
print(f"Period effects: {result.period_effects}")
```

### Time Series Cross-Validation

```python
from src.dml import TimeSeriesCrossValidator

# Temporal blocking prevents look-ahead bias
cv = TimeSeriesCrossValidator(n_splits=5, gap=10)
for train_idx, test_idx in cv.split(X):
    # Train always precedes test with gap
    assert train_idx[-1] + 10 <= test_idx[0]
```

### Macroeconomic Controls

```python
from src.data import FREDLoader, STANDARD_MACRO_SERIES

# Load FRED macro data (GDP, CPI, unemployment, etc.)
loader = FREDLoader()
macro = loader.load_controls(
    series=STANDARD_MACRO_SERIES,
    start_date="2010-01-01",
    end_date="2023-12-31",
)
print(f"Loaded {len(macro.series_names)} macro controls")
```

---

## Project Structure

```
double_ml_time_series/
├── src/
│   ├── dml/                    # DML estimators
│   │   ├── fwl.py              # Frisch-Waugh-Lovell (409 lines)
│   │   ├── robinson.py         # Robinson estimator (365 lines)
│   │   ├── double_ml.py        # DML with cross-fitting (554 lines)
│   │   ├── cross_fitting.py    # TimeSeriesCrossValidator (590 lines)
│   │   ├── hac.py              # HAC/Newey-West SE (729 lines)
│   │   └── dynamic_dml.py      # Dynamic/Rolling/Panel DML (1,045 lines)
│   ├── data/
│   │   ├── fred_loader.py      # FRED macro data (705 lines)
│   │   └── oj_loader.py        # Orange Juice benchmark dataset
│   ├── validation/
│   │   ├── dgp_generator.py    # Cross-sectional DGP
│   │   ├── dgp_generator_ts.py # Time series DGP (714 lines)
│   │   ├── stationarity.py     # ADF, KPSS, PP tests (920 lines)
│   │   ├── insurance_dgp.py    # Insurance DGP (667 lines)
│   │   └── bias_validation.py  # Bias diagnostics
│   └── production/
│       ├── model_registry.py   # Model versioning
│       ├── causal_monitor.py   # Causal monitoring
│       ├── retrain_pipeline.py # Retraining triggers
│       └── dml_pipeline.py     # End-to-end pipeline
├── chapters/                   # LaTeX book (10 chapters + appendix)
│   ├── chapter_01.tex          # Potential Outcomes + FWL
│   ├── chapter_02.tex          # Neyman Orthogonality + DML
│   ├── chapter_03.tex          # Validation Framework
│   ├── chapter_04.tex          # Cross-Sectional Application
│   ├── chapter_05.tex          # Dynamic Treatment Effects
│   ├── chapter_06.tex          # Panel DML + Rolling Window
│   ├── chapter_07.tex          # FRED Integration
│   ├── chapter_08.tex          # Competitor Pricing (957 lines)
│   ├── chapter_09.tex          # Heterogeneity Analysis (680 lines)
│   ├── chapter_10.tex          # Production Pipeline (863 lines)
│   └── appendix_a.tex          # Julia DML.jl Roadmap (586 lines)
├── test/                       # 796 tests (4-tier system)
├── main.tex                    # Book entry point (LuaLaTeX)
├── docs/                       # State tracking and plans
└── requirements.txt            # Python dependencies
```

---

## Book Build

The book uses LuaLaTeX (scrbook + Tufte-style). **Not pdflatex.**

```bash
# Full build (two passes + bibliography)
lualatex -shell-escape main.tex && biber main && lualatex -shell-escape main.tex

# Output: main.pdf (180 pages)
```

**Zero-error compilation requirement** — the build must produce no errors and no warnings.

---

## Testing

Four-tier system with enforced timeouts (`pytest-timeout`):

| Tier | Purpose | Tests | Time | Command |
|------|---------|-------|------|---------|
| tier1 | Unit — pure logic, no estimation | 285 | ~12s | `pytest -m tier1` |
| tier2 | Integration — light estimation | 316 | ~2min | `pytest -m "tier1 or tier2"` |
| tier3 | Validation — Monte Carlo/bootstrap | 161 | ~30min | `pytest -m "tier1 or tier2 or tier3"` |
| tier4 | Full replication + stress | 34 | ~2h | `pytest` |

```bash
# Quick smoke test
pytest -m tier1

# Pre-push gate
pytest -m "tier1 or tier2"

# Full suite
pytest
```

Timeouts per tier: 10s / 60s / 300s / 1800s. Every test has a `@pytest.mark.tierN` marker — zero unmarked tests.

---

## API Reference

### Estimators (`src.dml`)

| Class/Function | Description |
|----------------|-------------|
| `fwl_estimate(Y, D, X)` | Linear FWL residualization |
| `robinson_estimator(Y, D, X)` | Nonparametric partially linear model |
| `double_ml(Y, D, X)` | Cross-fitted DML with Neyman-orthogonal scores |
| `DynamicDML` | Time-varying treatment effects |
| `RollingWindowDML` | Adaptive window DML estimation |
| `PanelDML` | Entity fixed effects DML |
| `TimeSeriesCrossValidator` | Temporal blocking cross-validation |
| `HACEstimator` | HAC covariance estimation |
| `newey_west_se(residuals)` | Newey-West standard errors |

### Data (`src.data`)

| Class/Function | Description |
|----------------|-------------|
| `FREDLoader` | Load FRED macroeconomic series |
| `OJDataLoader` | Load Orange Juice benchmark dataset |
| `create_synthetic_fred_data()` | Synthetic FRED data for testing |

### Validation (`src.validation`)

| Class/Function | Description |
|----------------|-------------|
| `DGPGenerator` | Cross-sectional data generating process |
| `TimeSeriesDGPGenerator` | Time series DGP with autocorrelation |
| `create_insurance_dgp()` | Parameterized insurance pricing DGP |
| `StationarityDiagnostic` | ADF, KPSS, Phillips-Perron tests |
| `BiasValidation` | Bias diagnostics and coverage checks |

### Production (`src.production`)

| Class/Function | Description |
|----------------|-------------|
| `DMLModelRegistry` | Model versioning and promotion |
| `CausalMonitor` | Causal-specific monitoring |
| `RetrainScheduler` | Intelligent retraining triggers |
| `InsuranceDMLPipeline` | End-to-end insurance pricing pipeline |

---

## Hardware

Optimized for 64-core AMD Threadripper:

```python
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Prevent nested parallelism
os.environ['MKL_NUM_THREADS'] = '1'
N_JOBS = 48  # Leave 16 cores for system
```

Monte Carlo simulations, cross-validation, and bootstrap all parallelize across 48 cores.

---

## Author

Brandon Behring

## License

Personal research project — not for distribution.
