# Double ML Time Series Roadmap

**Status**: Phase 1A Complete | Phase 1B Pending
**Total Target**: ~77,000-83,000 words

---

## Current Sprint

- [ ] Chapter 3: Comprehensive Validation Battery (~10,000 words)
  - [ ] 3.1 Published Results Replication (Chernozhukov 2018, Facure 2022)
  - [ ] 3.2 Synthetic Monte Carlo (1000 runs, 95% coverage)
  - [ ] 3.3 Cross-Implementation Validation (Manual vs EconML vs R)
  - [ ] 3.4 First-Stage Diagnostics
  - [ ] 3.5 Real-World Known Outcomes
  - [ ] 3.6 Public Dataset Benchmarks
  - [ ] 3.7 Synthetic DGP Generator
- [ ] Hub integration validation

---

## Backlog

### Phase 1C (Chapter 4)
- [ ] Chapter 4: First-Stage ML Methods (~8,000 words)

### Phase 2A (Chapters 5-7)
- [ ] Chapter 5: Heterogeneous Treatment Effects
- [ ] Chapter 6: Time Series Extensions
- [ ] Chapter 7: Sensitivity Analysis

### Phase 2B (Chapter 8)
- [ ] Chapter 8: Production Implementation

### Phase 3A (Chapters 9-10)
- [ ] Chapter 9: Insurance Applications
- [ ] Chapter 10: Advanced Topics

### Phase 3B (Appendix)
- [ ] Appendix: Mathematical Foundations

---

## Completed

- [x] Phase 1A: Chapters 1-2 (13,213 words, 43 pages) (2025-11-13)
  - [x] Chapter 1: Potential Outcomes + FWL (6,898 words)
  - [x] Chapter 2: Neyman Orthogonality + DML (6,315 words)
- [x] LaTeX migration from AsciiDoc (2025-11-13)
- [x] Build system (Makefile, 3-pass compilation) (2025-11-13)
- [x] Python environment (3.13, 21 packages) (2025-11-13)
- [x] Pre-commit hooks (Black 100-char, mypy) (2025-11-13)
- [x] Hub integration setup (2025-11-26)
  - [x] .claude/settings.json with hooks
  - [x] Pattern symlink
  - [x] Session workflow structure
  - [x] Test layer directories

---

## Phase Summary

| Phase | Chapters | Status | Words | Est. Hours |
|-------|----------|--------|-------|-----------|
| **1A** | 1-2 | ✅ Complete | 13,213 | 40-50h |
| **1B** | 3 | ⏳ Pending | ~10,000 | 20-25h |
| **1C** | 4 | Not started | ~8,000 | 15-20h |
| **2A** | 5-7 | Not started | ~18-21K | 40-50h |
| **2B** | 8 | Not started | ~9,000 | 15-20h |
| **3A** | 9-10 | Not started | ~13-15K | 30-40h |
| **3B** | Appendix | Not started | ~3-4K | 8-10h |
