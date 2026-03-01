# Deferred Ideas & Future Enhancements

**Purpose**: Track scope additions and enhancements for future phases or separate work

**Review Frequency**: At each milestone review, assess if any deferred items should be integrated into next phase

---

## From Initial Planning (2025-11-13)

### Out of Scope (Separate Projects)
- **Causal Discovery Methods** (PC, FCI, constraint-based algorithms)
  - **Why deferred**: Separate methodology domain, would double book size
  - **Future**: Could be Volume 3 or separate causal discovery book
  - **Status**: OUT_OF_SCOPE

- **Healthcare Case Studies**
  - **Why deferred**: Focus on insurance/annuity pricing for work relevance
  - **Future**: Could add as Chapter 11 or separate healthcare-focused volume
  - **Status**: FUTURE_ENHANCEMENT

- **Additional Domain Examples** (climate, policy, marketing beyond insurance)
  - **Why deferred**: Keep focused on time series pricing
  - **Future**: Expand applications chapter or separate domain guides
  - **Status**: FUTURE_ENHANCEMENT

### Post-Phase 3 (After Book Complete)
- **Julia DML.jl Implementation**
  - **Why deferred**: Appendix provides roadmap, actual implementation is separate project
  - **Future**: Use Python code as benchmark for Julia package development
  - **Status**: POST_PHASE_3 (Appendix in Phase 3B documents this)

- **Interactive Dashboards** (Streamlit/Dash for DML)
  - **Why deferred**: Production pipeline chapter covers concepts, full implementation out of scope
  - **Future**: Build dashboards for specific work applications
  - **Status**: FUTURE_ENHANCEMENT

- **Flashcard Deck** (Anki cards for DML concepts)
  - **Why deferred**: Not critical for production use
  - **Future**: Could generate after book complete for study/retention
  - **Status**: FUTURE_ENHANCEMENT

---

## Open Issues (Logged 2026-02-28)

Identified from pre-completion analysis reports. All actionable findings either resolved or logged here.

### BiasValidation ATE Interval Logic
- **Item**: `src/validation/bias_validation.py:194-197` used `effect_interval(X)` (CATE interval) averaged as ATE confidence interval. Methodologically incorrect.
- **Severity**: MEDIUM-HIGH
- **Resolution**: Replaced with `dml.ate()` + `dml.ate_interval()` which uses delta method SE for correct marginal ATE CI construction.
- **Status**: RESOLVED (2026-03-01)

### IRM_RF Replication Incomplete
- **Item**: `src/validation/empirical_replication.py` defined `IRM_RF: 8202.0` in `PUBLISHED_ATES` but no `replicate_irm_rf()` method existed. Dead constant.
- **Severity**: LOW-MEDIUM
- **Resolution**: Removed dead constant. IRM replication is out of scope for PLR-focused validation; deferred to Appendix A Julia roadmap.
- **Status**: RESOLVED (2026-03-01)

### pyproject.toml Missing Runtime Dependencies
- **Item**: No `[project.dependencies]` section in `pyproject.toml`.
- **Severity**: MEDIUM
- **Resolution**: Already resolved — `pyproject.toml` has `[project.dependencies]` with all 8 runtime packages.
- **Status**: RESOLVED (pre-existing)

---

## Scope Adjustments During Development

*(Items added here when we decide to defer something mid-project)*

---

## Review History

### Milestone 1B Review (Planned)
- Review all deferred items
- Assess: Should any be integrated into Phase 2?
- Document decisions

### Milestone 2B Review (Planned)
- Review all deferred items
- Assess: Should any be integrated into Phase 3?
- Document decisions

### Final Review (Planned)
- Prioritize deferred items for post-book work
- Create roadmap for enhancements
