# PanelDML design gate

Date: 2026-11-27. Status: **decided by the owner on 2026-11-27** (§7). The protocol of §5 is frozen
in [`2026-11-27_paneldml_protocol.md`](2026-11-27_paneldml_protocol.md). Part II, step 1 of the backlog-landing plan (`docs/plans/active/2026-11-24_backlog_landing.md`). The step is Addendum 5's step 7, which is in the session plan of 2026-11-26 (a copy is in the landing's local backup folder, `plans/`), not in the record.

Nothing was simulated for this note. Only single fits were timed, and no estimate was recorded
(`part2-design/timing/` in the landing's local backup folder, not in the repository). Every number below is quoted
from a paper or from an earlier review row, which is named with it.

## 1. What is wrong with `PanelDML` today

On `main` (`dml_ts/dml/temporal_plr_dml.py` at `c84ce72`):

1. **The controls are demeaned before the learners** (`:997-1013`). This is the "approximate"
   within-group procedure of Clarke and Polselli (Section 5.1.2 a, eqs 5.1-5.2) and the "early
   demeaning" of Fuhr and Papies (eq 3). Under confounding nonlinear in X it keeps nearly all of
   OLS's bias: in Clarke and Polselli's Table 1, panel C, the bias is 0.977 (Lasso), 0.972
   (forest), 0.918 (boosting) and 0.993 for OLS, for a true effect of 0.5. Filed as #84 (P1):
   default forest, confounder `x**2`, bias +1.07 for a slope of 2 and 0% coverage in 40
   replications.
2. **Identifiers are cast** to `int64` (`:991-992`). Units 1.00 to 1.29 become one unit; text that spells integers is cast silently, and other text raises.
3. **The folds depend on row order.** The demeaned data go to
   `TemporalPLRDML(n_lags=0, cv_strategy="time_series_split")` (`:1027-1038`), which splits by
   position. With unit-major rows the folds are blocks of units and the first block is never
   scored. With date-major rows they are blocks of dates. Nothing in the signature says which.
4. **Two-way demeaning is one pass** of each mean (`:1006-1013`). That is the two-way projection
   only for a balanced panel.
5. **The input boundary does not apply** (`np.asarray(..., dtype=np.float64)` at `:988-990`):
   unequal pandas indexes are not refused. `PanelDML` is the one estimator not yet under the
   input contract of #79 and #82.

The uncommitted redesign in the snapshot (`dml_ts/dml/_panel.py`, 133 lines) repairs points 2
(by `group_codes`), 3 (an explicit `fold_design`: `whole_unit` or `forward_date`) and 4
(`absorb_twoway` iterates to convergence; `transform_panel_fold` checks connectivity). It adds three inference options in `panel_standard_error`, two of them checked to 1e-12 against statsmodels and `date_hac` against a written-out Bartlett formula, and `PanelDMLResult`.
**It does not repair point 1:** `transform_panel_fold` still demeans X before the learner. The
second audit measured that defect on the redesign itself (row F03b: bias +1.0589 at R = 500).

Two further facts about the snapshot's code, found for this note:

- `group_codes` calls `np.asarray(labels)` before `pd.factorize`. A list with the unit `1` and
  the unit `"1"` becomes the strings `'1'` and `'1'`, so two units are merged into one.
  Checked in the locked environment (`[1, "1", 2]` gives codes `[0, 0, 1]`). Identifiers
  "compared as given" need an object-preserving path (for example `pd.factorize` on a
  `pd.Index` or on an object array built without coercion).
- With `fold_design="forward_date"` and individual effects, the held-out rows subtract the
  unit's **training-period** means (`_panel.py:72`). Under feedback this is a Nickell-type bias
  (review row M14). Neither paper studies this design.

## 2. What the two papers say

Sources: Clarke and Polselli, "Double Machine Learning for Static Panel Models with Fixed
Effects", The Econometrics Journal 29(1), 69-86 (2026), doi:10.1093/ectj/utaf011, read as arXiv
2312.08174v5 and checked against the authors' accepted manuscript of 27 February 2025 (Essex
repository 40496), which agrees on every point used here. Fuhr and Papies, "Double Machine
Learning meets Panel Data - Promises, Pitfalls, and Potential Solutions", arXiv 2409.01266v1.
Section-level notes: `part2-design/notes-*.md` in the same local folder.

| Question | Clarke and Polselli | Fuhr and Papies |
|---|---|---|
| Model | Partially linear, static: no feedback to X (Ass. 3.1), no dynamics (3.2), additive unit effect (3.5), homogeneous effect (3.4) | Same structure, simulated only; three causal structures, (C) "the most plausible": the unit effect also enters X |
| Demeaning X before the learner | Biased under nonlinear confounding (Table 1, panel C) | Biased under u-shaped confounding in every structure (Figure 4) |
| Correlated random effects (Mundlak means as predictors) | Accurate (Table 1, panel A: Lasso 0.009, boosting −0.007); needs the treatment's unit mean, D̄_i (Section 5.1.1, steps 1-3), and the Mundlak model to be right (Section 4.1) | "Only DML with correlated random effects consistently delivers estimates that are close to the true effect"; degrades as confounders grow, about like pooled DML by J = 10 at N = 500, T = 10 (4.5.2) |
| First difference, exact | Learn ΔY and ΔD on (X_it, X_it−1) (5.1.2 b). **Their recommendation for practice** (Section 8): fewest assumptions about the unit effects. Trees poor (Table 1, panel B: forest 0.752) | Not studied |
| Hybrid | CRE fit on levels, then transform the predictions (5.1.2 c); needs the CRE conditions | Not studied |
| Cross-fitting | By unit (block-k-fold, Algorithm 5.1); the variance (5.6) clusters by unit and relies on it | Do **not** split by unit (Section 5, recommendation 3): bias for the dummy estimator, weakly for CRE with very few units (N = 10, T = 500; none at N = 250, T = 20, A.1.2) |
| Inference | Cluster-robust by unit (5.6), DML2, repeated splits averaged (Step 4) | "We discourage interpreting standard errors or confidence intervals" (Section 5) |
| Learners | Lasso with a dictionary of cubic terms and interactions best; trees poor in bias, SD and normality; use an ensemble | XGBoost throughout; "other flexible ML methods ... lead to very similar results" |
| Two-way effects | Not in the model; wave dummies as controls in the application | CRE adds period means too and is "virtually unaffected" (4.5.4) |
| Unbalanced panels | Extend "naturally" if non-response is conditionally independent (Section 8); not simulated | Not studied |

Where the two papers disagree, on splitting by unit, only Clarke and Polselli study intervals,
and their interval needs it. The protocol below measures it in a few-units cell.

## 3. Candidates

Each candidate is a way for the nuisance learners to see X. Every one keeps the partialling-out
score with the treatment residual transformed by the fixed effects, and the absorbed-treatment
refusal of `main` (`:1017-1024`).

| | A. Basis first, then within | B. Correlated random effects | C. First difference, exact | D. Hybrid | Z. Within first (`main`, #84) |
|---|---|---|---|---|---|
| What the learner sees | A dictionary B(X_it) (splines, interactions) built on levels, then transformed with Y and T; a linear or sparse fit | (X_it, X̄_i) on levels, any learner; treatment residual uses D̄_i (C&P 5.1.1) | ΔY, ΔD on (X_it, X_it−1), any learner | (X_it, X̄_i) on levels, predictions then demeaned | Demeaned X |
| Identifies θ when | The nuisances lie in the span of B (exact: Q(B'ψ) = Q(B)'ψ) | The unit effects follow the Mundlak form E[α_i \| X_i] = ω(X̄_i) | Assumptions 3.1-3.5 only | As B | The nuisances are linear in X |
| Nonlinear confounding in X | Yes, within the basis | Yes | Yes | Yes | **No** |
| Unit effects correlated with X | Yes (exact transform) | Yes, through X̄_i, if Mundlak holds | Yes (differenced away) | Yes, if Mundlak holds | Yes, but see the first row |
| Two-way effects | `absorb_twoway` on the basis (iterated, exact) | Add period means (F&P 4.5.4) or period dummies | Period dummies in the differenced learners | Period means, then two-way transform | One-pass demeaning (wrong if unbalanced) |
| Unbalanced panel | Exact projection | X̄_i over observed waves (Wooldridge 2019 caveats) | Only consecutive pairs; gaps drop rows | As B | As above |
| Feedback (T or X responds to past Y) | Violates every candidate's assumptions (C&P 3.1-3.2): a violation cell, reported apart | Same | Same | Same | Same |
| Few units | Cluster SE with few clusters; basis dimension limited by N·T | Doubles the predictors; F&P: needs N large beside the number of confounders | Loses one period per unit | As B | n/a |
| Treatment absorbed by the effects | Raises | Raises (the treatment residual is a within residual: V̂ = (D − D̄) − (m̂ − m̂̄)) | Raises (ΔD ≈ 0) | Raises | Raises (`main`) |
| Learner restriction | Linear in the basis (ridge, lasso); no trees | None | None, but trees poor in C&P | None | None |
| Evidence in this repository | F03c: bias +0.024, coverage 0.919 at R = 1,000, quadratic design (basis after demeaning: +1.040) | F03e/S106 tested a CRE **without D̄_i**: +0.663 on the quadratic design with boosting(40), and its linear control failed (+0.0526). The paper's procedure is untested here | F03e/S106: FD with boosting(40) +0.497 (quadratic), +0.0193 (linear control, inside its band of 3 SE + 0.02; that row's control failed on the CRE arm) | None | F03a, #84 |

Z is not a candidate. It stays in the protocol as a known-defective arm, a negative control: a
study in which it passes the quadratic design is broken.

## 4. Choices that cut across the candidates

1. **The nuisance treatment is an explicit argument** (`nuisance="basis_within" | "cre" |
   "first_difference" | "hybrid"`), never a silent default. Options: (a) required, no default;
   (b) a default set to the candidate that passes most cells. Recommendation: (a) until the
   study is in, then decide.
2. **Folds.** `whole_unit` (block-k-fold) is what Clarke and Polselli's variance needs.
   `forward_date` is in neither paper and carries M14. Recommendation: release `whole_unit`
   only; keep `forward_date` out unless it passes the protocol in its own cell.
3. **Inference.** `unit_cluster` (C&P 5.6) as the default; `two_way_cluster` (Cameron, Gelbach
   and Miller) offered with two-way effects; `date_hac` kept behind the protocol. Fuhr and
   Papies's warning is the reason every coverage figure is measured, not assumed.
4. **Repeated cross-fitting** (C&P Algorithm 5.1, Step 4): an option, `n_repeats`, at S times the
   cost. Default 1 to keep the book's numbers comparable.
5. **The input boundary.** `PanelDML` comes under `as_validated_arrays`: Y and T read by value,
   X converted under its refusals (the estimator computes on X), unequal pandas indexes refused,
   and `unit` and `date` passed through `also_labelled` so their indexes are checked too.
   Identifiers compared as given, never cast; the `group_codes` merge above is repaired and gets
   a test (`[1, "1"]` is two units; `1.00 ... 1.29` are thirty units).
6. **A frame-based entry** (option): `PanelDML.fit_frame(df, y="...", t="...", x=[...],
   unit="...", date="...")`, so that the identifiers cannot be passed in another order than the
   rows. Recommendation: yes, as a thin wrapper over `fit`, after the estimator passes.
7. **Landing `panel_standard_error` early** (option). With `group_codes` (repaired) and
   `_cluster_variance` it is a pure function with a statsmodels check. On its own it releases
   three held audit scripts whose other inputs are on `main`:
   `docs/audits/2026-11-19_time_series_dml/retail_check.py`,
   `docs/audits/2026-11-19_time_series_dml/external/questions/minimum_wage.py` and
   `docs/audits/2026-11-21_evaluation_methods/reproduction/design_probes.py`, with the test at lines 824-831 of the snapshot's `test/dml/test_temporal_plr_dml.py`. Their receipts are checked against the held-file
   list before they land. `replay.py` and `method_checks.py` also need other parts of the redesign (`absorb_twoway`, `group_means`, and for `method_checks.py` the redesigned `PanelDML`), and `reproduce_audit_probes.py` needs `transform_panel_fold`; they stay with the redesign. Recommendation: yes, as one small pull
   request before the protocol, so the study's variance code is the landed one.

## 5. Draft protocol (frozen and hashed before any run)

**Designs**, from `second_audit_checks/area1.py` (`_panel_data`: α_i, μ_i = 1.2 α_i + N(0, 0.5²),
X = μ_i + N(0, 1), true effect 2), m = 6 periods:

| Cell | Change | N (units) | Gates the verdict |
|---|---|---|---|
| linear | g = X | 80, 160 | yes (control) |
| quadratic | g = X² | 80, 160 | yes |
| interactive | T = α X + e, Y = 2T + α X + e | 80, 160 | yes |
| uncorrelated effects | μ_i independent of α_i | 160 | yes |
| two-way | add a period effect to X, T and Y (F&P eqs 13-15) | 80, 160 | yes |
| unbalanced | drop 20% of unit-periods at random (conditionally independent) | 160 | yes |
| few units | quadratic | 20 | reported, not gating |
| feedback | T_it depends on Y_i,t−1 | 160 | violation, reported apart |
| absorbed | T constant within unit | 80 | every arm must raise |

**Arms:** Z (`main`'s `PanelDML`, known defective, #84); the snapshot's `PanelDML`; A, B, C, D,
each with ridge or lasso on the dictionary, gradient boosting and the default forest where the
candidate admits it; and the true-nuisance arm (true l1, m1, same transform and folds), which is
the reference. Inputs pass the boundary in every arm.

**Pass rule** (from the review's protocol, §11, and its lesson on control bands):

- An arm **covers** in a cell if its 95% interval covers in at least 92.5% of replications, and
  **fails** below 90% (protocol §11). With R = 1,000 the Monte Carlo SE of a 95% coverage is
  0.69 points.
- Bias is judged against the sampling SD: |bias| ≤ 0.2 SD. Bands of a few Monte Carlo SEs around zero failed controls in the second audit for estimators whose own finite-sample bias was small (`second-audit-review.md:285`): F03b's linear control, −0.0165 against 3 SE = 0.0075, and the CRE arm of F03e and S106, +0.0526 against 3 SE + 0.02 = 0.028. A bias of 0.2 SD alone lowers a 95% interval's coverage
  to 94.5%.
- A candidate **passes the gate** if it covers and meets the bias rule in every gating cell at
  both sizes, raises in the absorbed cell, and the true-nuisance arm covers everywhere (else the
  study is broken). Z must fail the quadratic cell.

**Replications:** 1,000 per cell (protocol §2.5), seeds from `SeedSequence` per (design, N,
replication), draws stored and a second run byte-identical (§2.6).

**Cost.** Timed single fits (locked environment, one thread): `main`'s `PanelDML` with the
default forest 0.70 s (N = 80) and 0.76 s (N = 160); with ridge 0.01 s. One boosting(40) fit
0.02-0.03 s, one forest(100) 0.12-0.20 s, RidgeCV 0.001 s. 12 simulated cell-sizes × 1,000
replications × about 15 arm-learner pairs ≈ 180,000 fits. If the average fit took 0.76 s, the time of one whole-`PanelDML` fit with the forest, that is about 38 CPU hours; the slowest arms take about 1 s a fit (8 nuisance fits of a forest at N = 160, timed by the claims check of 2026-11-27), so the pilot of the protocol's §6 decides. *(As drafted, this line said "about 36 minutes on 64 workers". At the 1.2 GiB per process of the out-of-memory report, 64 workers need about 77 GiB of the machine's 123 GiB, which other work shares: on 2026-11-26 the machine
was powered off after an out-of-memory event while 71 Python processes ran with 85 GiB resident in all (about 1.2 GiB each),
attributed by inference to this repository's review agents. Recomputed under the cap that now
applies to every run:
at most 16 workers in a 40 GiB memory scope, so about 2.4 hours at an average of 0.76 s a fit.)* That is under the protocol's
four-hour budget (§2.7). The cost is recomputed from the frozen list before the run.

## 6. Recommendation

1. Implement **B (correlated random effects with the treatment-mean correction)** and
   **C (first difference, exact)** as the two primary candidates, as both papers support B and
   Clarke and Polselli recommend C. Implement **A (basis first)** beside them, because it is the
   one repair this repository has measured. D waits: it needs B's conditions and adds no
   identification.
2. `whole_unit` folds, `unit_cluster` inference by default, the nuisance treatment a required
   argument, `PanelDML` under the input boundary with identifiers compared as given.
3. Land `panel_standard_error` (with a repaired `group_codes`) first, as its own small pull
   request.
4. Freeze the protocol above (after your changes), then implement, then run.
5. If no candidate passes: report, and you decide between a restricted, documented scope (for
   example A only, with linear-in-basis learners) and holding `PanelDML`.

## 7. Decisions for the owner

1. Which candidates are implemented: B and C, with A (recommended); or another set.
2. The nuisance argument: required (recommended) or with a default.
3. `forward_date`: out of the first release (recommended), or tested in its own cell.
4. `fit_frame`: yes after the estimator passes (recommended), now, or not at all.
5. Landing `panel_standard_error` early, with `group_codes` repaired: yes (recommended) or no.
6. The protocol's designs, arms, pass rule (|bias| ≤ 0.2 SD in place of three Monte Carlo SEs)
   and cost, as drafted or changed.

**Decided on 2026-11-27** (the owner's choices):

| Decision | Choice |
|---|---|
| Candidates | B and C, with A; D waits |
| The nuisance argument | Required, no default, until the study is in |
| `forward_date` | Out of the first release |
| `fit_frame` | After the estimator passes |
| `panel_standard_error` | Lands early, in its own pull request, with `group_codes` repaired |
| The protocol | As drafted, with the cost recomputed under a memory cap (§5). Frozen in [`2026-11-27_paneldml_protocol.md`](2026-11-27_paneldml_protocol.md); details the draft left open are marked there |

## Not checked for this note

- The typeset pages of The Econometrics Journal (69-86); the authors' accepted manuscript of
  February 2025 was compared instead.
- Clarke and Polselli's Online Supplement (S1-S4): DGP details, the tuning strategy and the
  derivations S1.2-S1.3 were taken from the main text's references to them.
- No candidate was implemented or simulated. The review rows quoted (F03a, F03b, F03c, F03e, S106, M14)
  are from the snapshot's `docs/audits/2026-11-22_end_to_end_review/second-audit-checks.csv`,
  which has not landed.
