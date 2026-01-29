# Validation methodology audit (dev branch)

Scope: reviewed `src/validation`, 401(k) replication/diagnostics, and comparison plumbing to check mathematical correctness and cross-method consistency.

---

## Resolution Status (Phase 0 - 2025-11-22)

**CRITICAL ISSUES RESOLVED**:
- ✅ **C2: Binary treatment mis-specification** (2025-11-22)
  - Fixed: All DML implementations now use classifiers + `discrete_treatment=True`
  - Impact: 86.5% bias reduction confirmed empirically
  - Files: bias_validation.py, empirical_replication.py, lasso_diagnostic.py

- ✅ **C3: Monte Carlo comparability** (2025-11-22)
  - Fixed: BaselineComparison now resets DGP state for deterministic comparisons
  - Impact: Rankings stable (previously varied 4 positions)
  - Files: baseline_comparison.py

- ✅ **C5: 401(k) covariate mismatch** (2025-11-22)
  - Fixed: Now includes all 11 published covariates (removed 'nifa', 'tw' from exclusion)
  - Impact: Lasso improves $2,837 (59.6% → 30.0% error)
  - Files: empirical_replication.py, lasso_diagnostic.py

**DOCUMENTED LIMITATIONS**:
- ℹ️ **M1: Bootstrap refits** (2025-11-22)
  - Status: Documented as known limitation (see VALIDATION_LIMITATIONS.md)
  - Decision: Defer fix to Phase 3 (30 min documentation vs 3-4 hour fix)
  - Impact: Minimal (coverage 92-93% vs nominal 95%)

**DEFERRED TO PHASE 3**:
- ⏳ **M2: CI reporting clarity** (low priority)
- ⏳ **C4: Lasso hyperparameter alignment** (requires broader refactoring)
- ⏳ **Bootstrap diagnostics coverage** (minor issue)

**Related Documents**:
- Investigation: `INVESTIGATION_PROGRESS_2025-11-21.md`
- Impact analysis: `IMPACT_ASSESSMENT_MATRIX_2025-11-21.md`
- Decision rationale: `VALIDATION_FIXES_DECISION_2025-11-21.md`
- Session summary: `SESSION_2025-11-22_PHASE0_FIXES.md`

---

## Critical issues
- Monte Carlo comparability is broken: `BaselineComparison.compare` reuses a single `DGPGenerator` for all methods (`src/validation/baseline_comparison.py:91`), so each estimator consumes a different sequence of draws instead of evaluating on the same simulated datasets. Cross-method tables are not like-for-like and bias/mse rankings can flip arbitrarily. Fix by cloning the DGP state per method (or pre-generating datasets and replaying them) so every estimator sees identical simulations.
- Binary-treatment DML is mis-specified throughout: `BiasValidation` uses `LinearDML` with `discrete_treatment=False` and a regressor for the treatment model (`src/validation/bias_validation.py:166-181`). The 401(k) replications do the same with RandomForestRegressor/LassoCV for a binary `T` (`src/validation/empirical_replication.py:230-248` and `:303-320`), and the Lasso diagnostic inherits the same setup (`src/validation/lasso_diagnostic.py:279-285`, `:401-407`, `:486-492`). This violates the orthogonal score for binary treatments and can bias effects and inference. Switch to classifiers (e.g., `RandomForestClassifier`, `LogisticRegressionCV`/`Lasso` with binomial link) and set `discrete_treatment=True`.
- 401(k) covariate set does not match the published design: preprocessing drops `nifa` and `tw`, leaving 9 controls instead of the 11 used in Chernozhukov et al. (`src/validation/empirical_replication.py:191-195` and `src/validation/lasso_diagnostic.py:224-228`). This changes nuisance fits and helps explain the Lasso mismatch. Align controls with the paper (include the full set of covariates used in Table 1).
- Lasso replication parameters are ignored/misaligned: `replicate_plr_lasso` accepts `alpha` but always runs `LassoCV` with its own tuning (`src/validation/empirical_replication.py:280-337`), and the diagnostic never varies alpha (`src/validation/lasso_diagnostic.py:364-399`). Treatment is also modeled with linear Lasso on a binary target plus `discrete_treatment=False`. This makes the “Lasso” comparison not reflective of the published spec. Honor the provided alpha or expose a proper grid, use a classifier for `T`, and mark the comparison as exploratory until fixed.
- Bootstrap inference for IPW/AIPW omits first-stage refits: confidence intervals reuse propensity scores and outcome predictions from the original sample inside each bootstrap draw (`src/validation/ipw_baseline.py:126-161` and `:333-383`). First-stage uncertainty and overfitting are ignored, so coverage is overstated and weights can be pathologically reused. Refit the propensity and outcome models inside each bootstrap replicate (or use typical influence-function variance formulas) and add truncation diagnostics for extreme weights.
- Reported CIs in results mix quantities: baseline estimators store percentiles of bootstrap *bias* in `ci_lower/ci_upper` (`src/validation/ipw_baseline.py:104-124`, `src/validation/ols_baseline.py:112-141`, `src/validation/ml_baseline.py:96-126`), while coverage is computed from per-simulation effect CIs. Tables therefore show bias intervals next to coverage of effect CIs, which is misleading. Store effect-interval summaries (e.g., mean lower/upper) alongside bias intervals with clear labels.
- Bootstrap diagnostics coverage check is not informative: `diagnose_convergence` computes coverage from a single CI per replication (`src/validation/bootstrap_diagnostics.py:178-182`) because `_bootstrap_confidence_intervals` returns only one interval (`:387-399`). The resulting coverage estimates are 0/1 indicators, so convergence recommendations on coverage are unreliable. Generate many CIs per replication (or reuse the bootstrap draws directly) before summarizing coverage.

## Additional observations
- IPW/AIPW and ML baselines fit on the same sample they evaluate, without cross-fitting or sample-splitting. That is fine for simple baselines but should be documented as “plug-in” estimators to avoid over-claiming causal robustness.
- Heavy defaults (`n_simulations=1000`, large bootstraps) will be very slow; consider “fast” configs in code paths used by CLI scripts.

## Recommended next actions
- Align DML setup for binary treatments (classifiers + `discrete_treatment=True`) across BiasValidation, empirical replication, and diagnostics; rerun the 401(k) benchmarks.
- Make BaselineComparison deterministic and comparable by replaying the same simulated datasets for every method.
- Fix bootstrap refits and CI reporting so coverage diagnostics reflect the estimand rather than bootstrap bias.
- Reconcile 401(k) covariates and Lasso hyperparameters with the published specification; document any remaining gaps.
