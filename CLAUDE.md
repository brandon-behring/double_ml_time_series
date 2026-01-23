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

**Current Status** (2026-01-23):
- Phase 1A: Complete - Chapters 1-2 content restructured
- Book infrastructure: Rebuilt with amsbook + natbib
- Code: Static DML implementations verified (FWL, Robinson, DML)
- Time Series: NOT YET IMPLEMENTED (planned Phase 2)

**Build System**: Native LaTeX (amsbook class) with zero-error compilation requirement.

---

## Quick Reference Commands

```bash
# Build complete book
pdflatex -shell-escape main.tex && bibtex main && pdflatex -shell-escape main.tex

# Run tests (from repo root)
pytest test/validation/ -v

# Quick DML verification
python -c "from src.dml import double_ml, fwl_estimate, robinson_estimator; print('OK')"

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
│   └── double_ml.py     # DML with cross-fitting
└── validation/
    ├── dgp_generator.py # Synthetic data generation
    └── ...              # Baseline comparisons

test/
└── validation/          # 27+ tests pass

chapters/
├── chapter_01.tex       # Potential Outcomes + FWL
├── chapter_02.tex       # Neyman Orthogonality + DML
└── chapter_03.tex       # Validation (skeleton)
```

---

## Key Principles

### LaTeX Compilation
- **Zero errors, zero warnings** - compilation must be clean
- 3-pass build: pdflatex → bibtex → pdflatex → pdflatex
- Use `-shell-escape` for minted code blocks

### Validation (7-Method Suite)
1. Published results replication
2. Synthetic Monte Carlo (1000 runs)
3. Cross-implementation (Manual vs EconML vs R)
4. First-stage diagnostics
5. Real-world known outcomes
6. Public dataset benchmarks
7. Synthetic DGP generator

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

## Known Limitations (Document as you work)

1. **Not Time Series Yet**: Despite repo name, current code is i.i.d. only
2. **Chapter 3 Incomplete**: Only outline exists, needs full implementation
3. **No Dynamic DML**: Sequential g-estimation planned for Phase 2

---

## Hardware

64-core Threadripper configuration:
```python
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
N_JOBS = 48
```
