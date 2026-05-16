# Content Expansion Roadmap

**Created**: 2026-03-01
**Objective**: Expand book from 180 pages to ~285 pages (thin chapters + worked examples)
**Status**: Deferred as of 2026-05-02

> Truth-cleanup note: this content-expansion roadmap is not current execution
> guidance. Do not expand manuscript claims until README/Sphinx/examples/tests/book
> gates pass and `docs/CURRENT_STATUS.md` says expansion is unblocked. References
> below to DynamicDML, production readiness, and chapter completion are historical.

---

## Motivation

The book is complete at 180 pages but well under the 300-350 page target for an academic
reference. Five chapters are thin (9-14 pages each), and the dense chapters lack worked examples.
This plan addresses both gaps.

---

## Phase 1: Thin Chapter Expansion (5 sessions, ~3-4h each)

### Priority Order (thinnest first, highest pedagogical impact)

| # | Chapter | Current | Target | Growth | Session |
|---|---------|---------|--------|--------|---------|
| 1 | Ch4: Cross-Sectional Application | 464L (9p) | 900-1000L (18-20p) | +100% | 1 |
| 2 | Ch6: Panel DML + Rolling Window | 514L (10p) | 900-1100L (18-22p) | +75% | 2 |
| 3 | Ch5: Dynamic Treatment Effects | 629L (13p) | 1100-1300L (22-26p) | +75% | 3 |
| 4 | Ch9: Heterogeneity Analysis | 680L (14p) | 1100-1300L (22-26p) | +60% | 4 |
| 5 | Ch7: FRED Integration | 716L (14p) | 1150-1350L (23-27p) | +60% | 5 |

---

### Session 1: Chapter 4 — Cross-Sectional Application (464 -> 900-1000L)

**Current**: OJ dataset pipeline -> DML elasticity -> Rosenbaum bounds. Single application,
informal theory, limited diagnostics.

**New/Deepened Sections**:

1. **Sensitivity Analysis Theory** (~200L)
   - Formal theorem: Rosenbaum bounds for general DML (not just OLS)
   - Proof sketch of critical Gamma existence and uniqueness
   - Imbens partial identification bounds as alternative
   - E-value framework (VanderWeele & Ding 2017)

2. **Overlap Diagnostics** (~150L)
   - Propensity score distribution plots (treated vs control)
   - Near-violation detection (extreme propensity scores)
   - Trimming/weighting strategies to enforce overlap
   - Application to OJ data with visualization

3. **Contrastive Application: Gasoline Elasticity** (~150L)
   - Higher-dimensional confounders
   - Compare OJ results (robust) vs gasoline (fragile sensitivity)
   - Illustrate what a fragile result looks like

4. **Heterogeneous Sensitivity** (~100L)
   - Does Gamma_crit vary by brand/store?
   - Subgroup-specific robustness assessment

5. **Diagnostic Visualization Code** (~100L)
   - Propensity score overlap plots
   - Sensitivity surface contours (Gamma vs p-value)
   - Residual Q-Q plots

6. **Reporting Checklist (expanded)** (~50L)
   - What to include in a causal inference paper
   - Table templates for sensitivity analysis

---

### Session 2: Chapter 6 — Panel DML + Rolling Window (514 -> 900-1100L)

**Current**: Fixed effects -> PanelDML -> cluster-robust SE -> rolling window. Bare sketches.

**New/Deepened Sections**:

1. **Two-Way Fixed Effects (deep dive)** (~150L)
   - Detailed mechanics of two-way demeaning
   - Recent TWFE critique (Goodman-Bacon 2021, de Chaisemartin & D'Haultfoeuille)
   - How DML helps in staggered treatment settings

2. **Multi-Way Clustering** (~130L)
   - Cameron, Gelbach, Miller (2011) multi-way approach
   - Implementation in Python
   - Insurance example: firm x product x time

3. **Formal Structural Break Testing** (~140L)
   - Chow test (known break date)
   - Bai-Perron sequential procedure (unknown breaks)
   - Comparison with rolling window visual diagnostics

4. **Time-Varying Heterogeneity** (~120L)
   - Rolling window + CATE estimation combined
   - Heatmap: CATE by subgroup x time period

5. **Full Worked Applications** (~200L)
   - Insurance pricing: 50 states x 10 years panel
   - Macro policy: QE effects across countries
   - Side-by-side with naive (no FE) comparison

6. **Panel Diagnostics** (~100L)
   - Checking FE assumptions
   - Cluster structure validation
   - Pre-treatment balance testing

---

### Session 3: Chapter 5 — Dynamic Treatment Effects (629 -> 1100-1300L)

**Current**: TS cross-validation -> HAC SEs -> DynamicDML -> Monte Carlo. Synthetic only.

**New Sections**:

1. **Real Application: Monetary Policy & Growth** (~200L)
   - FRED data: interest rates -> GDP, with inflation controls
   - Compare with VAR/LP literature estimates

2. **Treatment Lag Structure** (~150L)
   - Formal lag selection: AIC, BIC for CATE models
   - Wald tests on lag coefficients

3. **Stationarity Diagnostics** (~100L)
   - ADF/KPSS workflow for Y, T, X
   - When to difference vs level
   - Over-differencing pitfalls

4. **Impulse Response Analysis** (~150L)
   - One-time shock to T_t -> time path of Y
   - Cumulative long-run effects
   - Comparison with traditional VAR IRFs

5. **Convergence & Stability Testing** (~80L)
   - Does the DML estimate stabilize with sample size?
   - Rolling subsample estimates

6. **Troubleshooting Guide** (~100L)
   - Large HAC SE -> bandwidth instability?
   - Residual autocorrelation beyond HAC correction
   - When to suspect model misspecification

---

### Session 4: Chapter 9 — Heterogeneity Analysis (680 -> 1100-1300L)

**Current**: CATE theory -> R-learner -> CausalForest -> BLP -> policy trees. Synthetic only.

**New Sections**:

1. **Inference for CATE** (~150L)
   - Pointwise vs uniform confidence intervals
   - Honest inference (Athey & Wager 2019)
   - Multiplicity corrections (Bonferroni, FDR)

2. **Method Comparison** (~140L)
   - S-learner, T-learner, X-learner (Kunzel et al. 2019)
   - When each is preferred
   - Simulation comparison on OJ data

3. **Effect Modifier Selection** (~120L)
   - How to choose X (effect modifiers) vs W (controls)?
   - Data-driven variable importance

4. **CATE Extrapolation & Generalization** (~100L)
   - External validity
   - Insurance: CATE predictions for new product entry

5. **Expanded Insurance Application** (~150L)
   - High-fidelity synthetic insurance panel
   - Policy tree: which products to price-adjust
   - Cost-benefit analysis

6. **Advanced Topics** (~110L)
   - Local linear regression for CATE
   - Sensitivity of CATE to unobserved confounding
   - Multiple treatment valuations

---

### Session 5: Chapter 7 — FRED Integration (716 -> 1150-1350L)

**Current**: Macro confounding theory -> FREDLoader architecture -> alignment. Synthetic only.

**New Sections**:

1. **Real FRED API Integration** (~150L)
   - Actual API call code with error handling, rate limiting
   - Authentication setup guide
   - Caching pipeline demonstration

2. **Stationarity & Cointegration** (~130L)
   - ADF/KPSS tests for standard FRED series
   - When to use levels vs differences in DML

3. **Macro Variable Selection** (~140L)
   - Lasso-based selection of macro controls
   - Double selection / partialling-out

4. **Time-Alignment Challenges** (~100L)
   - Real-world data gaps, release calendar effects
   - GDP revision vintages

5. **Production Data Validation** (~110L)
   - Data quality checks, version control for frozen datasets
   - Backtesting sensitivity to data version

6. **Insurance-Specific Macro Discussion** (~100L)
   - Which FRED series matter for insurance pricing?
   - Product-line specific macro sensitivities

---

## Phase 2: Worked Examples for Dense Chapters (Session 6)

Add 25-45 pages of worked examples across Ch1-3, 8, 10, Appendix A.
Details TBD after Phase 1 completion.

---

## Phase 3 (Deferred): New Methodology Chapters

- Local Projections for causal inference
- Synthetic Control Methods
- Regression Discontinuity with DML
- Difference-in-Differences with heterogeneous effects

---

## Page Budget

| Source | Current | Target | Delta |
|--------|---------|--------|-------|
| Ch4-7, 9 (thin chapters) | ~60p | 103-120p | +43-60p |
| Ch1-3, 8, 10, App (worked examples) | ~120p | 145-165p | +25-45p |
| New methodology chapters (deferred) | 0p | 50-70p | +50-70p |
| **Total** | **180p** | **298-355p** | **+118-175p** |

Phase 1 target: ~250-285 pages.

---

## Decisions Made

- **Priority order**: Thinnest first (Ch4 at 464L), not by importance. Rationale: each line
  added to a thin chapter has more marginal impact on perceived completeness.
- **Real vs synthetic data**: Each expanded chapter should include at least one real-data
  application, not just synthetic. This is the primary pedagogical gap.
- **Deferred Phase 3**: New methodology chapters saved for later. ROI is lower than expanding
  existing thin chapters.

---

## Execution Protocol

Each session:
1. Read chapter completely
2. Identify insertion points
3. Write new sections (LaTeX + any supporting Python)
4. Compile: `lualatex -shell-escape main.tex && biber main && lualatex -shell-escape main.tex`
5. Verify zero errors
6. Run affected tests
7. Commit with `feat: Expand chapter N — [description]`
