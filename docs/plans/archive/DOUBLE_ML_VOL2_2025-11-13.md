# Double ML Volume 2: Time Series Causal Inference - Master Plan

**Created**: 2025-11-13
**Status**: Phase 1A - IN_PROGRESS
**Current**: Infrastructure setup

---

## Ultimate Goal (ANCHOR - Never Changes)

Production-ready Double Machine Learning book for time series causal inference with:
- **Rigorous mathematical foundations** (all proofs verified against source papers)
- **Comprehensive 7-method validation** (published, synthetic MC, cross-impl, diagnostics, real-world, public datasets, parametric DGP)
- **Work-ready applications** (insurance/annuity competitor pricing with FRED macro controls)
- **Hardware optimized** (64-core Threadripper parallelization: n_jobs=48, batched validation)
- **Production template** (ready to swap synthetic → real data for actual work)
- **Future Julia roadmap** (Python as correctness benchmark for DML.jl implementation)

---

## Current Phase

**Phase**: 1
**Milestone**: 1A (Core Theory)
**Status**: IN_PROGRESS
**Working on**: Infrastructure setup → Chapter 1

---

## Complete Roadmap

### Phase 1: Foundation (40-50 hours)
- **1A: Core Theory** (Chapters 1-2)
  - ⏳ Chapter 1: Potential Outcomes + FWL (NOT_STARTED)
  - ⏳ Chapter 2: Neyman Orthogonality + DML (NOT_STARTED)
- **1B: Validation Battery** (Chapter 3) ⚠️ CRITICAL GATE
  - ⏳ 7-method validation suite (NOT_STARTED)
  - ⏳ Synthetic DGP module + unit tests (NOT_STARTED)
- **1C: First Application** (Chapter 4)
  - ⏳ Cross-sectional DML example (NOT_STARTED)

### Phase 2: Time Series Extension (40-50 hours)
- **2A: Temporal Methods** (Chapters 5-7)
  - ⏳ Chapter 5: Dynamic Treatment Effects (NOT_STARTED)
  - ⏳ Chapter 6: DynamicDML + Panel (NOT_STARTED)
  - ⏳ Chapter 7: FRED Integration (NOT_STARTED)
- **2B: Work Application** (Chapter 8) ⚠️ PRODUCTION TEMPLATE
  - ⏳ Insurance/annuity competitor pricing (NOT_STARTED)

### Phase 3: Advanced Topics (30-40 hours)
- **3A: Extensions** (Chapters 9-10)
  - ⏳ Chapter 9: Heterogeneity Analysis (NOT_STARTED)
  - ⏳ Chapter 10: Production Pipeline (NOT_STARTED)
- **3B: Julia Roadmap** (Appendix)
  - ⏳ DML.jl implementation roadmap (NOT_STARTED)

---

## Decision Log

### 2025-11-13 (Session 1): Initial Planning
**Decision**: Complete project plan with all best practices integrated
**Why**: Ensure methodological soundness, prevent mid-thought context issues, optimize for hardware
**Impact**:
- 7-method validation suite (comprehensive rigor)
- Hybrid test-first (unit tests for infrastructure, notebooks validate DML)
- main+dev git workflow (stable vs WIP)
- Context management at 70% (chapter-aware stopping)
- ProjectRegistry integration (morning reports, backups, RAG)
- Pre-commit hooks (Black 100-char, mypy, large commit warning)
- Parallelization (n_jobs=48, batched validation 3-4 concurrent)

**Alternatives considered**:
- Simpler validation (rejected - production use requires rigor)
- Single branch git (rejected - need stable main for reviews)
- No archimedes integration (rejected - consistency with other projects)

---

## Deferred Items

*(Items moved to later phases or future work)*

**From Planning**:
- Causal discovery methods → Separate volume (out of scope)
- Healthcare case studies → Future enhancement
- Julia implementation → Post-Phase 3 (Appendix provides roadmap)
- Interactive dashboards → Future enhancement
- Additional domain examples → Future work

**Review at Each Milestone**: Check if any deferred items should be integrated into next phase

---

## Validation Status

### Chapter Validation
- Chapter 1: ⏳ NOT_STARTED
- Chapter 2: ⏳ NOT_STARTED
- Chapter 3: ⏳ NOT_STARTED (validation chapter itself)
- Chapter 4: ⏳ NOT_STARTED
- Chapter 5: ⏳ NOT_STARTED
- Chapter 6: ⏳ NOT_STARTED
- Chapter 7: ⏳ NOT_STARTED
- Chapter 8: ⏳ NOT_STARTED
- Chapter 9: ⏳ NOT_STARTED
- Chapter 10: ⏳ NOT_STARTED
- Appendix: ⏳ NOT_STARTED

### 7-Method Validation Suite (Phase 1B)
1. **Published Results Replication**: ⏳ NOT_RUN
   - Chernozhukov et al. (2018) 401(k): ⏳ NOT_RUN
   - Facure (2022) Ch22 synthetic: ⏳ NOT_RUN

2. **Synthetic Monte Carlo**: ⏳ NOT_RUN
   - 1000 runs, known τ=3.0: ⏳ NOT_RUN
   - 95% coverage check: ⏳ NOT_RUN

3. **Cross-Implementation**: ⏳ NOT_RUN
   - Manual vs EconML: ⏳ NOT_RUN
   - EconML vs R DoubleML: ⏳ NOT_RUN

4. **Diagnostics Suite**: ⏳ NOT_RUN
   - First-stage R²: ⏳ NOT_RUN
   - Residual checks: ⏳ NOT_RUN
   - Sensitivity analysis: ⏳ NOT_RUN

5. **Real-World Known Outcomes**: ⏳ NOT_RUN

6. **Public Dataset Benchmarks**: ⏳ NOT_RUN

7. **Synthetic DGP Generator**: ⏳ NOT_BUILT
   - Unit tests: ⏳ NOT_WRITTEN

---

## Review Checkpoints

### Milestone 1B: Validation (CRITICAL)
- **Status**: ⏳ PENDING
- **Required**: All 7 validation methods pass
- **Your review time**: 1-2 hours
- **Blocking**: Cannot proceed to Phase 1C until approved

### Milestone 1C: First Application
- **Status**: ⏳ PENDING
- **Your review time**: 1-2 hours

### Milestone 2B: Work Application (CRITICAL)
- **Status**: ⏳ PENDING
- **Required**: Production-ready insurance pricing template
- **Your review time**: 1-2 hours

### Milestone 3B: Final Review
- **Status**: ⏳ PENDING
- **Your review time**: 1-2 hours

---

## Session Log

### 2025-11-13 (Session 1): Infrastructure Setup
**Time**: Started
**Tasks**:
- Created project directory structure
- Initialized git repository (dev branch)
- Creating state tracking files
- Next: Python environment, pre-commit hooks, ProjectRegistry registration

**Progress**: Infrastructure in progress
**Next session**: Complete infrastructure → Begin Chapter 1
