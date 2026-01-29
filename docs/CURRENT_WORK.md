# Current Work

**Last Updated**: 2026-01-29

---

## Right Now

**Gate 1 Verification Complete** ✅

Chapter 3 written and infrastructure optimized:
- ✅ Book: 77 pages (main.tex compiles 0 errors)
- ✅ Tests: 331 total (43 marked @slow, 288 fast for pre-commit)
- ✅ Pre-commit hooks optimized to run ~100 fast tests
- ✅ Chapter 3 complete (Validation Framework)

**Code Status**:
- FWL, Robinson, DML implementations verified working
- 7-method validation suite implemented
- Static (i.i.d.) DML only - NOT time series yet

**Book Status**:
- main.tex: 77 pages, 0 errors
- Chapter 1: Potential Outcomes + FWL (complete)
- Chapter 2: Neyman Orthogonality + DML (complete)
- Chapter 3: Validation Framework (complete)

---

## Why

Building comprehensive DML reference book with rigorous validation:
1. Verified existing implementations work
2. Fixed infrastructure issues
3. Documented actual state (not claimed state)
4. Ready for Phase 2 (time series)

---

## Next Steps

**Immediate** (choose one):
1. Begin Phase 2 (Dynamic DML) - see `docs/IMPLEMENTATION_STRATEGY_REPORT.md`
2. Complete Chapter 4 (Sensitivity Analysis)
3. Run full validation battery report

**Phase 2 (Dynamic DML) Roadmap**:
- Rolling-window DML for time series
- Sequential g-estimation
- Macroeconomic control integration
- See `docs/IMPLEMENTATION_STRATEGY_REPORT.md`

---

## Context for Return

- **Build**: `pdflatex -shell-escape main.tex` (0 errors, 77 pages)
- **Tests**: `pytest test/validation/ -v` (331 tests, 43 slow)
- **Fast tests**: `pytest test/validation/ -m "not slow"` (~288 tests, <2 min)
- **Install**: `pip install -e .` (required for imports)
- **Master plan**: `docs/MASTER_ROADMAP_2025-11-21.md`
- **Time series roadmap**: `docs/IMPLEMENTATION_STRATEGY_REPORT.md`
- **Hardware**: 64-core Threadripper, n_jobs=48

---

## Known Issues

1. **Not Time Series**: Current code is i.i.d. only despite repo name
2. **Stale logs/results**: `results/` and `logs/` may be outdated
