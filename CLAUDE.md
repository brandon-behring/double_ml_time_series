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

**Status**: Phase 1A Complete - Chapters 1-2 (13,213 words, 43 pages)

**Build System**: Native LaTeX (amsbook class) with zero-error compilation requirement.

---

## Quick Reference Commands

```bash
# Build complete book
make

# Build and open PDF
make view

# Run tests
pytest tests/ --cov=src -v

# Check LaTeX errors
make check-errors

# Clean build artifacts
make clean
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
- `docs/plans/active/CHAPTER_STATUS.md` - Granular progress
- `docs/plans/active/DOUBLE_ML_VOL2_2025-11-13.md` - Master plan

---

## Hardware

64-core Threadripper configuration:
```python
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
N_JOBS = 48
```
