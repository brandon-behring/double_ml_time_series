# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Shared Foundation (Hub Reference)

**This project uses shared patterns from lever_of_archimedes**:

See: `~/Claude/lever_of_archimedes/patterns/` for:
- `testing.md` - 6-layer validation architecture
- `sessions.md` - Session workflow (CURRENT_WORK.md, ROADMAP.md)
- `git.md` - Commit format and workflow
- `style/python_style.yaml` - Black 100-char, strict mypy

**Core principles** (from hub):
1. NEVER fail silently - explicit errors always
2. Simplicity over complexity - 20-50 line functions
3. Immutability by default
4. Fail fast with diagnostics

---

## Project Overview

**Double Machine Learning for Time Series** - Academic reference book on DML methodology with focus on time series causal inference for insurance/annuity competitor pricing with macroeconomic controls.

**Current Status** (2026-01-30):
- PROJECT COMPLETE - All 10 chapters + Julia appendix
- Book: 180 pages, 10 chapters + 1 appendix, zero LaTeX errors
- Code: 14,068 lines Python (incl. 2,453 production module)
- Tests: 763 total

**Build System**: LuaLaTeX (scrbook + Tufte-style) with zero-error compilation requirement.

---

## Quick Reference Commands

```bash
# Build complete book (NOTE: lualatex, NOT pdflatex)
lualatex -shell-escape main.tex && biber main && lualatex -shell-escape main.tex

# Run tests by tier (from repo root)
pytest -m tier1                      # ~12s smoke test (285 tests)
pytest -m "tier1 or tier2"           # ~2min pre-push (588 tests)
pytest -m "tier1 or tier2 or tier3"  # ~30min nightly (750 tests)
pytest                               # Full suite including tier4 (~2h)

# Quick DML verification (includes time series)
python -c "from src.dml import double_ml, DynamicDML, RollingWindowDML; print('OK')"

# Verify FRED loader
python -c "from src.data import FREDLoader; print('OK')"

# Install in development mode (required for imports)
pip install -e .
```

---

## Project Structure

```
src/
├── dml/
│   ├── fwl.py           # Frisch-Waugh-Lovell theorem
│   ├── robinson.py      # Robinson estimator
│   ├── double_ml.py     # DML with cross-fitting
│   ├── cross_fitting.py # TimeSeriesCrossValidator (590 lines)
│   ├── hac.py           # HAC/Newey-West standard errors (729 lines)
│   └── dynamic_dml.py   # DynamicDML, RollingWindow, Panel (1,045 lines)
├── data/
│   └── fred_loader.py   # FRED macroeconomic data (705 lines)
└── validation/
    ├── dgp_generator.py    # Cross-sectional DGP
    ├── dgp_generator_ts.py # Time series DGP (714 lines)
    ├── stationarity.py     # ADF, KPSS, PP tests (920 lines)
    ├── insurance_dgp.py    # Insurance DGP (667 lines)
    └── ...                 # Baseline comparisons

test/
└── validation/          # 652 tests

chapters/
├── chapter_01.tex       # Potential Outcomes + FWL
├── chapter_02.tex       # Neyman Orthogonality + DML
├── chapter_03.tex       # Validation Framework
├── chapter_04.tex       # Cross-Sectional Application
├── chapter_05.tex       # Dynamic Treatment Effects
├── chapter_06.tex       # Panel DML + Rolling Window
├── chapter_07.tex       # FRED Integration
├── chapter_08.tex       # Competitor Pricing (957 lines)
├── chapter_09.tex       # Heterogeneity Analysis (680 lines)
├── chapter_10.tex       # Production Pipeline (863 lines)
└── appendix_a.tex       # Julia Roadmap (586 lines)
```

---

## Key Principles

### LaTeX Compilation
- **Zero errors, zero warnings** - compilation must be clean
- Build: `lualatex -shell-escape main.tex && biber main && lualatex -shell-escape main.tex`
- Uses biber (not bibtex) for bibliography

### Validation (7-Method Suite)
1. Published results replication
2. Synthetic Monte Carlo (1000 runs)
3. Cross-implementation (Manual vs EconML vs R)
4. First-stage diagnostics
5. Real-world known outcomes
6. Public dataset benchmarks
7. Synthetic DGP generator

### Test Tiers (4-tier system)

| Tier | Purpose | Speed | Tests | Command |
|------|---------|-------|-------|---------|
| tier1 | Unit — no estimation, pure logic | ~12s | 285 | `pytest -m tier1` |
| tier2 | Integration — light estimation | ~2min | 316 | `pytest -m "tier1 or tier2"` |
| tier3 | Validation — moderate MC/bootstrap | ~30min | 161 | `pytest -m "tier1 or tier2 or tier3"` |
| tier4 | Full replication + stress | ~2h | 40 | `pytest` |

- Timeouts: 10s / 60s / 300s / 1800s per tier (enforced by pytest-timeout via `test/conftest.py`)
- Tiered bootstrap: `BootstrapConfig.tier2()` / `.tier3()` factory methods
- Cached 401(k) data: `test/fixtures/401k_data.csv` (offline testing)
- Zero unmarked tests — every test has a `@pytest.mark.tierN` marker

### Code Style
- **Formatter**: Black with 100-character lines
- **Type hints**: Required everywhere
- **Parallelization**: n_jobs=48 (64-core Threadripper)

---

## State Files

- `docs/CURRENT_WORK.md` - Current task (30-sec resume)
- `docs/MASTER_ROADMAP_2025-11-21.md` - Master 11-chapter plan
- `docs/IMPLEMENTATION_STRATEGY_REPORT.md` - Time series DML roadmap

---

## Current Gaps (Document as you work)

**None** - All chapters complete (2026-01-30)

---

## Time Series Capabilities (IMPLEMENTED)

| Component | Description | Location |
|-----------|-------------|----------|
| TimeSeriesCrossValidator | Temporal blocking, proper train/test splits | `src/dml/cross_fitting.py` |
| HAC Standard Errors | Newey-West, Bartlett kernel | `src/dml/hac.py` |
| DynamicDML | Time-varying treatment effects | `src/dml/dynamic_dml.py` |
| RollingWindowDML | Adaptive window estimation | `src/dml/dynamic_dml.py` |
| PanelDML | Fixed effects, entity-time structure | `src/dml/dynamic_dml.py` |
| FREDLoader | Macroeconomic controls from FRED | `src/data/fred_loader.py` |
| Time Series DGP | Autocorrelated confounders | `src/validation/dgp_generator_ts.py` |
| Stationarity Tests | ADF, KPSS, Phillips-Perron | `src/validation/stationarity.py` |
| Insurance DGP | Parameterized complexity DGP | `src/validation/insurance_dgp.py` |
| DMLModelRegistry | Model versioning and promotion | `src/production/model_registry.py` |
| CausalMonitor | Overlap/treatment/nuisance monitoring | `src/production/causal_monitor.py` |
| RetrainScheduler | Causal-specific retraining triggers | `src/production/retrain_pipeline.py` |
| InsuranceDMLPipeline | End-to-end production pipeline | `src/production/dml_pipeline.py` |

---

## Hardware

64-core Threadripper configuration:
```python
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
N_JOBS = 48
```
