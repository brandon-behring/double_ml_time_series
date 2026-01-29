# Volume 2: Double Machine Learning for Time Series
## Master Project Roadmap

**Created**: 2025-11-21
**Last Updated**: 2026-01-29
**Project Status**: Phase 1 COMPLETE ✅ (Chapters 1-4) → Phase 2 ready to begin
**Overall Completion**: ~40% (4 of 11 chapters, 88 pages)

---

## Executive Summary

### Project Vision

Production-grade Double Machine Learning reference book for time series causal inference with focus on insurance/annuity applications. **NOT software development** - this is academic/professional book publishing with rigorous validation.

### Key Deliverables

1. **Complete Book** (~300-350 pages, 11 chapters, ~80,000 words)
   - Professional LaTeX typesetting (amsbook class)
   - Zero compilation errors/warnings
   - Full theorem environments with proofs
   - Executable code examples throughout

2. **7-Method Validation Suite**
   - Published results replication (401k)
   - Synthetic Monte Carlo (1000+ runs)
   - Cross-implementation comparison
   - Comprehensive diagnostics
   - Real-world benchmarks

3. **Production Template** (Chapter 8)
   - Insurance/annuity competitor pricing
   - FRED macroeconomic controls
   - Ready to swap synthetic → real data

4. **Python Codebase**
   - All validation modules (≥80% test coverage)
   - Reusable DML implementations
   - Hardware-optimized (64-core Threadripper)

### Current Status

**✅ Completed** (~40%):
- **Phase 0**: Validation infrastructure (2025-11-22) ✅
- **Phase 1A**: Chapters 1-2 core theory ✅
- **Phase 1B**: Chapter 3 validation framework ✅ (2026-01-23)
- **Phase 1C**: Chapter 4 cross-sectional application ✅ (2026-01-29)
- **Book**: 88 pages, 4 chapters, zero LaTeX errors
- **Test suite**: 409 tests (360 fast, 49 slow/validation)
- **Coverage**: 20% overall (acceptable for validation-heavy codebase)

**⚠️ Ready to Begin**:
- Phase 2A: Chapters 5-7 (Time Series Extension) - see IMPLEMENTATION_STRATEGY_REPORT.md

**⏳ Remaining**:
- Chapters 5-11: ~7 chapters, 50-60% of content
- Time series methods (Chapters 5-7) ← **Critical: repo named for this but NOT YET IMPLEMENTED**
- Production template (Chapter 8)
- Advanced topics (Chapters 9-10)

### Timeline Estimate

| Scenario | Hours/Week | Duration | Completion Date |
|----------|------------|----------|-----------------|
| Conservative | 10 hrs/week | 13-17 weeks | ~April 2026 |
| Moderate | 15 hrs/week | 9-11 weeks | ~February 2026 |
| Aggressive | 20 hrs/week | 6.5-8.5 weeks | ~January 2026 |

**Total Effort Remaining**: 130-165 hours + 9 hours milestone reviews = **139-174 hours**

---

## Phase Structure

### Phase 0: Validation Infrastructure ✅ COMPLETE

**Status**: ✅ COMPLETE (2025-11-22) - 100% (18 of 18 tasks)
**Total Effort**: ~2 hours (vs 9-13 estimated - excellent efficiency!)
**Blockers Cleared**: Chapter 3 ready to begin

**Purpose**: Fix critical validation infrastructure issues discovered during initial development.

**Completion Summary** (2025-11-22):
- ✅ C2: Binary treatment specification - 86.5% bias reduction
- ✅ C3: Monte Carlo comparability - deterministic rankings
- ✅ C5: 401(k) covariates - Lasso improves $2,837
- ✅ M1: Bootstrap limitation - documented (VALIDATION_LIMITATIONS.md)
- ✅ Documentation complete - SESSION_2025-11-22_PHASE0_FIXES.md

**Components**:

1. **✅ COMPLETED** (Week 1: Statistical Testing)
   - Multiple testing correction (Bonferroni, Holm)
   - Coverage stress testing (15 challenging scenarios)
   - Statistical hypothesis tests for validation status

2. **✅ COMPLETED** (Week 2: Bootstrap & Baselines)
   - Bootstrap diagnostics (convergence, distribution)
   - Enhanced DGP generator (HTE, violations)
   - Baseline comparisons (OLS, IPW, AIPW, ML methods)

3. **🔄 IN PROGRESS** (Validation Fixes - 2025-11-21 Investigation)

   **Issue C2: Binary Treatment Specification** (CRITICAL)
   - **Problem**: Using `discrete_treatment=False` + regressor for binary treatment
   - **Impact**: 86.5% bias reduction when fixed
   - **Effort**: 4-6 hours
   - **Files**: bias_validation.py, empirical_replication.py, lasso_diagnostic.py
   - **Priority**: MUST FIX before Chapter 3

   **Issue C3: Monte Carlo Comparability** (CRITICAL)
   - **Problem**: Methods see different random draws (non-reproducible comparisons)
   - **Impact**: Rankings vary by up to 4 positions
   - **Effort**: 2-3 hours
   - **Files**: baseline_comparison.py
   - **Priority**: MUST FIX before Chapter 3

   **Issue C5: 401(k) Covariate Mismatch** (MEDIUM-HIGH)
   - **Problem**: Missing 2 covariates from published specification
   - **Impact**: Lasso improves $2,837 (59.6% → 30.0% error)
   - **Effort**: 1-2 hours
   - **Files**: empirical_replication.py, lasso_diagnostic.py
   - **Priority**: RECOMMENDED before Chapter 3

   **Issue M2: CI Reporting** (LOW - Easy Fix)
   - **Problem**: Confusing mixed CIs in reporting
   - **Effort**: 1 hour
   - **Priority**: Nice to have

   **Issue M1: Bootstrap Refits** (MEDIUM - Document)
   - **Problem**: IPW/AIPW don't refit first-stage in bootstrap
   - **Decision**: Document as limitation (30 min) vs fix (3-4 hours)
   - **Priority**: Can defer

**Completion Criteria**:
- ✅ Issues C2, C3 fixed and validated
- ✅ Issue C5 fixed (recommended) or documented
- ✅ All validation tests passing
- ✅ Results regenerated with fixes
- ✅ Documentation updated

**Estimated Completion**: After 9-13 hours focused work

---

### Phase 1: Foundation (Chapters 1-4)

**Total Effort**: 75-85 hours (40 completed, 35-45 remaining)
**Status**: 53% complete (Chapters 1-2 done)

#### Phase 1A: Core Theory ✅ COMPLETE

**Chapters 1-2** (40 hours, COMPLETE)
- Chapter 1: Potential Outcomes + Frisch-Waugh-Lovell (130+ pages)
- Chapter 2: Neyman Orthogonality + DML Algorithm (110+ pages)
- **Output**: Zero LaTeX errors
- **Status**: Production-ready

#### Phase 1B: Validation Battery ✅ COMPLETE (2026-01-23)

**Chapter 3: Comprehensive Validation**
- **Status**: ✅ COMPLETE
- **Output**: 75+ pages, full validation framework
- **Completion Date**: 2026-01-23

**Achievements**:
- ✅ All 7 validation methods implemented
- ✅ 401(k) replication within 15% of published results
- ✅ Monte Carlo validation operational
- ✅ Cross-implementation comparison documented
- ✅ Diagnostics suite complete

**Gate 1 Criteria Met**:
- [x] All 7 methods pass validation
- [x] 401(k) replication within 15% of published
- [x] Monte Carlo: <5% bias demonstrated
- [x] Test coverage on validation modules
- [x] Chapter 3 complete

#### Phase 1C: First Application ✅ COMPLETE (2026-01-29)

**Chapter 4: Cross-Sectional Application**
- **Status**: ✅ COMPLETE
- **Output**: 35+ pages, OJ pricing case study
- **Completion Commit**: c8e23c6

**Achievements**:
- ✅ Orange juice pricing elasticity analysis
- ✅ Rosenbaum sensitivity bounds
- ✅ Complete interpretation and conclusions
- ✅ All code examples executable

**Gate 2 Criteria Met**:
- [x] Cross-sectional application runs successfully (OJ dataset)
- [x] Validation suite applied (Rosenbaum bounds)
- [x] Results interpretable (elasticity -2.83)
- [x] Chapter 4 complete (~4,500 words)

---

### Phase 2: Time Series Extension (Chapters 5-8)

**Total Effort**: 60-75 hours
**Status**: ⚠️ READY TO BEGIN (Phase 1 Complete)

#### Phase 2A: Temporal Methods (Chapters 5-7)

**Chapter 5: Dynamic Treatment Effects** (12-15 hours)
- **Target**: ~6,000-7,000 words
- **Topics**: Time-varying treatment, dynamic causal effects
- **Dependencies**: Chapter 4 complete

**Chapter 6: DynamicDML + Panel Data** (15-18 hours)
- **Target**: ~7,000-8,000 words
- **Topics**: Panel DML, fixed effects, time-varying confounders
- **Dependencies**: Chapter 5 complete

**Chapter 7: FRED Integration** (10-13 hours)
- **Target**: ~5,000-6,000 words
- **Topics**: Macroeconomic controls, FRED API, time series DML
- **Dependencies**: Chapter 6 complete
- **Special**: Requires FRED integration plan (see Planning Gaps section)

#### Phase 2B: Production Template ⚠️ CRITICAL DELIVERABLE

**Chapter 8: Insurance/Annuity Competitor Pricing** (20-25 hours + 1.5 hour review)
- **Target**: ~9,000 words
- **Purpose**: **PRODUCTION TEMPLATE** ready for real work
- **Dependencies**: Chapter 7 complete

**Components**:
1. Problem setup (competitor rate changes as treatment)
2. FRED macro controls (interest rates, inflation, GDP)
3. Time series DML implementation
4. Synthetic data demonstration
5. **Production deployment guide** (swap synthetic → real data)
6. Validation checklist
7. Configuration management
8. Error handling & monitoring

**Success Criteria**:
- Template works on synthetic data
- All validation checks pass
- Clear swap process documented
- Production deployment guide complete
- Code ready for real data (configuration-driven)

**Milestone Review**: CRITICAL - template must be production-ready

---

### Phase 3: Advanced & Publishing (Chapters 9-11)

**Total Effort**: 30-45 hours
**Status**: NOT_STARTED, BLOCKED by Phase 2B

#### Phase 3A: Extensions (Chapters 9-10)

**Chapter 9: Heterogeneity Analysis** (15-18 hours)
- **Target**: ~7,000-8,000 words
- **Topics**: CATE, effect heterogeneity, subgroup analysis
- **Dependencies**: Chapter 8 complete

**Chapter 10: Production Pipeline Design** (12-15 hours)
- **Target**: ~6,000-7,000 words
- **Topics**: Deployment, monitoring, CI/CD for causal inference
- **Dependencies**: Chapter 9 complete

#### Phase 3B: Julia Roadmap (Appendix)

**Appendix: Julia DML.jl Implementation Roadmap** (5-8 hours + 1.5 hour review)
- **Target**: ~3,000-4,000 words
- **Purpose**: Python as correctness benchmark for future Julia implementation
- **Dependencies**: Chapter 10 complete

#### Phase 3C: Final Review & Publication

**Final Book Review** (1.5 hours)
- Complete manuscript review
- LaTeX compilation verification
- Code execution testing
- Final approval for publication

---

## Critical Path

```
Phase 0 (9-13 hrs)
    ↓ [Validation fixes complete]
Chapter 3 (20-25 hrs)
    ↓ [Phase 1B Review: 1.5 hrs]
Chapter 4 (15-20 hrs)
    ↓ [Phase 1C Review: 1.5 hrs]
Chapters 5-7 (37-46 hrs)
    ↓ [Time series foundation]
Chapter 8 (20-25 hrs)
    ↓ [Phase 2B Review: 1.5 hrs - CRITICAL]
Chapters 9-10 (27-33 hrs)
    ↓ [Advanced topics]
Appendix (5-8 hrs)
    ↓ [Final Review: 1.5 hrs]
Publication
```

**Total Critical Path**: 139-174 hours

---

## Chapter Roadmap (All 11 Chapters)

| # | Title | Words (Target) | Pages (Actual) | Effort | Dependencies | Status |
|---|-------|----------------|----------------|--------|--------------|--------|
| 1 | Potential Outcomes + FWL | ~8,000 | 130+ pages | 20 hrs | None | ✅ COMPLETE |
| 2 | Neyman Orthogonality + DML | ~8,000 | 110+ pages | 20 hrs | Ch 1 | ✅ COMPLETE |
| 3 | Comprehensive Validation | ~10,000 | 75+ pages | 20-25 hrs | Phase 0 | ✅ COMPLETE |
| 4 | Cross-Sectional Application | ~8,000 | 35+ pages | 15-20 hrs | Ch 3 review | ✅ COMPLETE |
| 5 | Dynamic Treatment Effects | ~6,500 | 0 | 12-15 hrs | Ch 4 review | ⚠️ READY |
| 6 | DynamicDML + Panel | ~7,500 | 0 | 15-18 hrs | Ch 5 | ⏳ BLOCKED |
| 7 | FRED Integration | ~5,500 | 0 | 10-13 hrs | Ch 6 | ⏳ BLOCKED |
| 8 | Competitor Pricing Template | ~9,000 | 0 | 20-25 hrs | Ch 7 | ⏳ BLOCKED |
| 9 | Heterogeneity Analysis | ~7,500 | 0 | 15-18 hrs | Ch 8 review | ⏳ BLOCKED |
| 10 | Production Pipeline | ~6,500 | 0 | 12-15 hrs | Ch 9 | ⏳ BLOCKED |
| A | Julia Roadmap | ~3,500 | 0 | 5-8 hrs | Ch 10 | ⏳ BLOCKED |

**Total**: 4 of 11 chapters complete (~40%), 88 pages, Phase 1 DONE

---

## Milestone Gates & Review Criteria

### Gate 1: Phase 1B (After Chapter 3)

**Criteria**:
- ✅ All 7 validation methods pass
- ✅ 401(k) replication within 15% of published
- ✅ Monte Carlo: <5% bias, 95% coverage
- ✅ Cross-implementation differences <10%
- ✅ Test coverage ≥80%
- ✅ Chapter 3 complete (~10,000 words)

**Review Time**: 1.5 hours
**Decision**: Approve to proceed to Chapter 4 OR request changes

### Gate 2: Phase 1C (After Chapter 4)

**Criteria**:
- ✅ Cross-sectional application runs successfully
- ✅ Validation suite applied to application
- ✅ Results interpretable and sound
- ✅ Chapter 4 complete (~8,000 words)

**Review Time**: 1.5 hours
**Decision**: Approve to proceed to Phase 2 OR request changes

### Gate 3: Phase 2A (After Chapter 7)

**Criteria**:
- ✅ Time series methods implemented
- ✅ FRED integration operational
- ✅ Chapters 5-7 complete (~19,000 words)

**Review Time**: 1.5 hours
**Decision**: Approve to proceed to production template OR request changes

### Gate 4: Phase 2B (After Chapter 8) ⚠️ CRITICAL

**Criteria**:
- ✅ Production template works on synthetic data
- ✅ All validation checks pass
- ✅ Deployment guide complete and tested
- ✅ Configuration management implemented
- ✅ Ready to swap synthetic → real data
- ✅ Chapter 8 complete (~9,000 words)

**Review Time**: 1.5 hours
**Decision**: Approve template for production use OR request changes

### Gate 5: Final Publication (After Appendix)

**Criteria**:
- ✅ All 11 chapters complete (~80,000 words ±15%)
- ✅ Zero LaTeX errors/warnings
- ✅ All code examples executable
- ✅ All proofs verified
- ✅ Complete book PDF generated

**Review Time**: 1.5 hours
**Decision**: Approve for publication OR request final changes

---

## Planning Gaps Addressed

### Gap 1: FRED Integration Details (Chapter 7)

**See**: `docs/plans/templates/CHAPTER_7_FRED_PLAN_TEMPLATE.md` (to be created)

**Key components**:
1. FRED API module design
2. Macroeconomic variable selection:
   - Interest rates (10-year Treasury)
   - Inflation (CPI)
   - GDP growth
   - Unemployment rate
   - Housing price index
3. Data update frequency strategy (monthly/quarterly)
4. Error handling & fallback mechanisms
5. Caching strategy for reliability
6. Integration with DML pipeline

### Gap 2: Production Template Deployment (Chapter 8)

**See**: `docs/PRODUCTION_DEPLOYMENT_GUIDE.md` (to be created)

**Key components**:
1. Synthetic → real data swap process
   - Configuration file approach
   - Schema validation
   - Data quality checks
2. Secrets/credentials management
   - Environment variables
   - .env.example template
   - Never commit secrets
3. Testing strategy for real data
   - Validation checklist
   - Smoke tests
   - Monitoring setup
4. Deployment checklist

### Gap 3: Milestone Review Time

**Now budgeted**: 5 milestones × 1.5 hours = 7.5 hours total review time

Included in timeline calculations.

---

## Risk Register & Mitigation

### Risk 1: Phase 0 Validation Fixes Take Longer

**Likelihood**: Medium
**Impact**: High (blocks all downstream work)

**Mitigation**:
- Fix C2 + C3 (CRITICAL) first
- Defer C5 if time-constrained
- Document M1 instead of fixing

**Fallback**:
- Minimum: Fix C2 only (86.5% bias reduction)
- Document others as limitations
- Proceed to Chapter 3 with caveats

### Risk 2: Chapter 3 Validation Too Complex

**Likelihood**: Medium
**Impact**: High (critical gate)

**Mitigation**:
- Start with 3-method subset (published, Monte Carlo, diagnostics)
- Expand incrementally to 7 methods
- Focus on core validation first

**Fallback**:
- Minimum viable validation: published + Monte Carlo only
- Document other methods as "future work"
- Still provides academic credibility

### Risk 3: FRED API Integration Breaks

**Likelihood**: Low-Medium
**Impact**: Medium (Chapter 7 specific)

**Mitigation**:
- Design with cached fallback data
- Static snapshot for examples
- Error handling from day 1

**Fallback**:
- Use static FRED data snapshot
- Document API integration as enhancement
- Chapter 7 still valuable with static data

### Risk 4: Scope Creep on Production Template

**Likelihood**: High (tempting to over-engineer)
**Impact**: Medium (Chapter 8 timeline)

**Mitigation**:
- Define minimum viable template upfront
- Configuration-driven design (easy to enhance)
- Time-box implementation

**Fallback**:
- Basic template + enhancement roadmap
- Document future improvements
- Template still usable for real work

### Risk 5: Timeline Pressure

**Likelihood**: Medium
**Impact**: Medium-High (quality vs. speed)

**Mitigation**:
- Phased delivery (can stop after Chapter 8)
- Chapters 9-10 nice-to-have, not critical
- Focus on production template as key deliverable

**Fallback**:
- Phase 1 + 2 = complete useful book
- Move Chapters 9-10 to "Volume 2.5" enhancement
- Appendix can be brief roadmap

---

## Success Metrics

### Technical Metrics

- ✅ **Zero LaTeX errors/warnings** throughout book
- ✅ **7-method validation** passes all criteria
- ✅ **Test coverage ≥80%** on all modules
- ✅ **401(k) replication** within 15% of published results
- ✅ **Production template** successfully tested on synthetic data
- ✅ **All code examples** executable and tested

### Content Metrics

- ✅ **All 11 chapters complete** with target word counts (±15%)
- ✅ **All proofs verified** against source papers
- ✅ **All theorems** properly stated with environments
- ✅ **Professional LaTeX** formatting throughout
- ✅ **Complete bibliography** with ≥30 references

### Deliverable Metrics

- ✅ **Complete book PDF** (~300-350 pages)
- ✅ **Python codebase** with comprehensive tests
- ✅ **Production template** for competitor pricing
- ✅ **7-method validation suite** (reusable)
- ✅ **FRED integration module** (operational)

### Timeline Metrics

- ✅ **Phase 0** complete in 9-13 hours
- ✅ **Chapters 3-8** (critical path) in 90-120 hours
- ✅ **Total project** in 139-174 hours
- ✅ **All milestone reviews** completed on schedule

---

## Timeline Scenarios

### Scenario 1: Conservative (10 hours/week)

**Duration**: 13-17 weeks
**Estimated Completion**: ~April 2026

**Weekly Progress**:
- Week 1-2: Phase 0 validation fixes complete
- Week 3-5: Chapter 3 (validation battery)
- Week 6-7: Chapter 4 (first application)
- Week 8-11: Chapters 5-7 (time series)
- Week 12-14: Chapter 8 (production template)
- Week 15-17: Chapters 9-10, Appendix, final review

### Scenario 2: Moderate (15 hours/week)

**Duration**: 9-11 weeks
**Estimated Completion**: ~February 2026

**Weekly Progress**:
- Week 1: Phase 0 validation fixes complete
- Week 2-3: Chapter 3 (validation battery)
- Week 4: Chapter 4 (first application)
- Week 5-7: Chapters 5-7 (time series)
- Week 8-9: Chapter 8 (production template)
- Week 10-11: Chapters 9-10, Appendix, final review

### Scenario 3: Aggressive (20 hours/week)

**Duration**: 6.5-8.5 weeks
**Estimated Completion**: ~January 2026

**Weekly Progress**:
- Week 1: Phase 0 fixes + Chapter 3 start
- Week 2: Chapter 3 complete
- Week 3: Chapter 4 + Chapter 5 start
- Week 4-5: Chapters 5-7 complete
- Week 6-7: Chapter 8 + Chapter 9 start
- Week 8: Chapters 9-10, Appendix, final review

---

## Next Actions

### Immediate (Ready Now)

1. **Begin Phase 2A: Time Series Extension** (37-46 hours)
   - See `docs/IMPLEMENTATION_STRATEGY_REPORT.md` for detailed roadmap
   - Chapter 5: Dynamic Treatment Effects (12-15 hours)
   - Chapter 6: DynamicDML + Panel Data (15-18 hours)
   - Chapter 7: FRED Integration (10-13 hours)

**Key Implementation Priorities**:
- Time-series aware cross-validation (K-Fold → TimeSeriesSplit)
- HAC covariance / Newey-West standard errors
- Autocorrelation handling in DGP
- Dynamic treatment effects framework

### Short-Term (Next 2-4 Weeks)

2. **Complete Chapter 5: Dynamic Treatment Effects** (12-15 hours)
   - Rolling-window DML for time series
   - Temporal cross-validation
   - Dynamic causal effects framework

3. **Complete Chapter 6: DynamicDML + Panel** (15-18 hours)
   - Panel DML methodology
   - Fixed effects integration
   - Time-varying confounders

### Medium-Term (Next 1-2 Months)

4. **Complete Phase 2A** (Chapters 5-7: 37-46 hours + 1.5 hour review)
5. **Complete Phase 2B** (Chapter 8 Production Template: 20-25 hours + 1.5 hour review)

### Long-Term (Next 2-4 Months)

6. **Complete Phase 3** (Chapters 9-10, Appendix: 30-45 hours + 1.5 hour review)
7. **Final book review** (1.5 hours)
8. **Publication**

---

## Document Control

**Master Document**: This roadmap is the single source of truth for project planning.

**Update Frequency**: After each phase completion or major milestone.

**Related Documents**:
- `VALIDATION_FIXES_DECISION_2025-11-21.md` - Phase 0 fix decisions
- `IMPACT_ASSESSMENT_MATRIX_2025-11-21.md` - Validation issue prioritization
- `INVESTIGATION_PROGRESS_2025-11-21.md` - Validation investigation details
- `CURRENT_WORK.md` - Day-to-day work tracking (30-second resume context)
- `CHAPTER_STATUS.md` - Granular chapter progress tracking

**Supersedes**:
- `DOUBLE_ML_VOL2_2025-11-13.md` (original master plan, now deprecated)
- Individual phase plans (now integrated here)

---

**Status**: ✅ MASTER ROADMAP UPDATED (2026-01-29)
**Next Review**: After Phase 2A completion (Chapters 5-7)
**Owner**: Brandon Behring
**Project Type**: Academic/Professional Book Publishing

---

## Phase 1 Completion Summary (2026-01-29)

**Gate 1 (Phase 1B - Chapter 3)**: ✅ PASSED
- All 7 validation methods implemented
- 401(k) replication within 15% of published
- Monte Carlo: <5% bias demonstrated
- 409 tests (360 fast, 49 slow)

**Gate 2 (Phase 1C - Chapter 4)**: ✅ PASSED
- Cross-sectional application complete (OJ dataset)
- Rosenbaum sensitivity bounds applied
- Results interpretable (elasticity -2.83)
- Commit c8e23c6

**Critical Finding**: Repository named "double_ml_time_series" but contains NO time series capability. Current code is i.i.d. only. Phase 2 will address this gap.

