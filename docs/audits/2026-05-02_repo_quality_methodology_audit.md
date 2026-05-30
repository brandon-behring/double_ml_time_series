# Repo Quality And Methodology Audit

> **SUPERSEDED (2026-05-30)** by
> [`2026-05-30_repo_quality_methodology_audit.md`](2026-05-30_repo_quality_methodology_audit.md).
> Kept in place as historical evidence — its path is referenced by web Chapter 1
> provenance, so it is intentionally not moved. Of the 16 findings below, 13 are now
> CLOSED, F10 is accepted within scope, and F14 / F15-coverage remain open. See the
> 2026-05-30 audit for current, live-verified status.

Date: 2026-05-02

Repo: `double_ml_time_series`

Audit standard: book-grade companion repo, not deployable production software.
Executable behavior, live signatures, tests, logs, and primary methodology
sources outrank README/docs/roadmaps.

## Executive Verdict

Verdict: partial restart.

Do not restart from a blank repo. The repository has substantial salvageable
assets: 796 collected tests, green tier1 and tier1+tier2 gates, working core
FWL/Robinson/DML examples, usable HAC and time-series CV utilities, synthetic
DGPs, and a 205-page compiled book artifact.

But this is not a simple cleanup. The public story is materially ahead of the
implementation. The repo needs a partial restart of:

- The source-of-truth system: README, `CLAUDE.md`, current work, roadmap,
  Sphinx docs, active plans, and archive status need one canonical hierarchy.
- The time-series methodology contract: current `DynamicDML` is a partially
  linear residualization estimator with lagged features and HAC inference, not
  Lewis-Syrgkanis dynamic DML via recursive g-estimation.
- The public API/docs contract: README and Sphinx examples must be executable
  or removed.
- The build and verification gates: the book build, Sphinx docs, examples, and
  public snippets need automated checks matched to the book claims.

Best path: keep the repo, quarantine stale docs, rename or rewrite the dynamic
methodology surface, and rebuild the book-companion contract around tested
snippets and explicit assumptions. A full restart would waste working tests and
pedagogical code. A light cleanup would leave false methodology claims in place.

## Methodology Sources Used

The audit checked implementation claims against these anchors:

- Chernozhukov et al. (2018), "Double/debiased machine learning for treatment
  and structural parameters." The paper emphasizes Neyman-orthogonal scores and
  cross-fitting as the core bias-reduction ingredients.
  Source: https://academic.oup.com/ectj/article/21/1/C1/5056401
- Lewis and Syrgkanis (2021), "Double/Debiased Machine Learning for Dynamic
  Treatment Effects via g-Estimation." The paper describes multiple treatments
  over time, structural nested mean models, and a recursive peeling process.
  Source: https://arxiv.org/abs/2002.07285
- Robinson (1988), "Root-N-Consistent Semiparametric Regression." The paper is
  the root reference for partially linear residualization with nonparametric
  nuisance estimates.
  Source: https://tesnewdev.econometricsociety.org/publications/econometrica/1988/07/01/root-n-consistent-semiparametric-regression
- Newey and West (1987), HAC covariance estimation.
  Source: https://faculty.utrgv.edu/diego.escobari/teaching/Econ8370/Papers/NeweyWest%281987%29Econometrica-Juan.pdf
- Dickey and Fuller (1979), unit-root testing.
  Source: https://www.tandfonline.com/doi/abs/10.1080/01621459.1979.10482531
- Engle and Granger (1987), cointegration and error-correction.
  Source: https://www.econometricsociety.org/publications/econometrica/1987/03/01/co-integration-and-error-correction-representation-estimation
- Bergmeir, Hyndman, and Koo (2018), time-series cross-validation nuance.
  Source: https://robjhyndman.com/publications/cv-time-series/

## Verification Summary

| Check | Result | Audit implication |
|---|---:|---|
| `git status --short --branch` | `main...origin/main [ahead 1]`, tracked tree clean before report creation | No pre-existing tracked dirt observed during audit |
| `venv/bin/python -m pytest --collect-only -q` | 796 tests collected | README/CLAUDE total test count is currently true |
| `venv/bin/python -m pytest -m tier1 --no-cov -q` | 314 passed, 482 deselected, 3.53s | Tier1 gate is healthy |
| `venv/bin/python -m pytest -m "tier1 or tier2" --no-cov -q` | 615 passed, 181 deselected, 16 warnings, 64.54s | Main pre-push gate is healthy but documented counts are stale |
| Marker counts | tier1 314, tier1+tier2 615, tier1+tier2+tier3 760, network 20, not network 776 | `CLAUDE.md` says 617 and 762; README tier table says 285/316/161/34 |
| `venv/bin/python -m black --check src/ test/ examples/` | Pass, 81 files unchanged | Formatting is good |
| `venv/bin/python -m mypy src/ ...` | Pass, 37 source files | Type check gate is good under the permissive command used |
| `venv/bin/python -m sphinx.cmd.build ...` | Fails: `No module named 'sphinx'` | Current venv cannot build docs without docs extras |
| All `examples/*.py` | All five examples execute | Examples are runnable, but two are methodologically misleading |
| `pdfinfo main.pdf` | 205 pages, LuaTeX-1.18.0, created 2026-03-06 | `CLAUDE.md`/`CURRENT_WORK` page count is true; README 180-page badge is stale |
| `main.log` grep | 0 direct LaTeX warnings, 0 direct errors, 254 overfull hboxes, 12 underfull hboxes | "Zero warnings" is not an honest quality claim if bad boxes are counted |
| Optional import check | `econml`, `doubleml`, `fredapi`, `xgboost`, `lightgbm`, `dowhy`, `causalml`, `linearmodels`, `sphinx` absent in current venv | Several book/Sphinx claims need extras or guards |
| Bare `pytest test/ --collect-only -q` | Fails with `ModuleNotFoundError: No module named 'src'` | Public commands should use the venv or installed package consistently |
| Pre-commit template hook command | Fails: `python: command not found` | Hook uses bare `python` outside the venv |

## Critical Findings

Severity scale: blocker means false public contract or methodology; critical
means likely to mislead a reader or break a core workflow; high means serious
maintainability or correctness risk; medium means cleanup risk.

| ID | Severity | Claim/source | Evidence | Fix |
|---|---|---|---|---|
| F1 | Blocker | `README.md` and `src/dml/dynamic_dml.py` claim `DynamicDML` is sequential g-estimation for time-varying treatment effects. | `DynamicDML.fit` creates lagged features, cross-fits `E[Y|X_aug]` and `E[T|X_aug]`, then computes one scalar `theta = sum(Y_tilde*T_tilde)/sum(T_tilde^2)`. Result fields are scalar only. No `theta_t`, `period_effects`, lag-effect vector, or recursive peeling. | Either rename current class to something honest like `TemporalPLRDML`, or implement a separate true `DynamicGEstimationDML` with stage-wise lag effects and tests against a known dynamic DGP. |
| F2 | Blocker | README quickstarts are executable examples. | `DynamicDML(n_periods=10, n_splits=3)` raises unexpected `n_periods`; `result.period_effects` does not exist; `FREDLoader.load_controls` does not exist. | Add snippet tests for README/Sphinx. Fix snippets to use `n_lags`, `get_macro_controls`, and real result fields. |
| F3 | Blocker | Sphinx quickstart/user guide is current API documentation. | Sphinx snippets use `double_ml(..., model_y=..., model_t=...)`, `TimeSeriesCrossValidator(..., purge=5)`, `create_synthetic_fred_data(n_periods=120)`, `InsuranceDGPParams(n_obs=...)`, `DynamicDML(..., n_folds=5, cv_type="temporal")`, and `result.theta_t`; these do not match live signatures. | Treat Sphinx examples as tests. Use `outcome_model`/`treatment_model`, `purge_length`, date-based synthetic FRED args, and actual DGP constructors. |
| F4 | Blocker | Roadmap/docs say project is complete and source of truth is clear. | `CLAUDE.md` says "PROJECT COMPLETE"; `docs/CURRENT_WORK.md` says complete; `docs/MASTER_ROADMAP_2025-11-21.md` says 100 percent complete and "single source of truth"; active plan `2026-03-01_22-00_content_expansion.md` says expansion from 180 pages to 298-355 pages is active. | Create one current status file and archive/supersede old reports. Keep this `docs/audits/README.md` pattern. |
| F5 | Critical | Time-series CV always respects temporal order in DynamicDML. | `_cross_fit_nuisance_time_series` fills missing early predictions by fitting on later `available_idx` and predicting backwards. That violates the temporal training-before-test story for those rows. | Drop uncovered early observations, add an explicit `initial_window_policy`, or train early points only on past data when past data exists. Add a leakage test for every predicted index. |
| F6 | Critical | Validation suite includes cross-implementation comparison/manual vs EconML vs R DoubleML. | `docs/CROSS_IMPLEMENTATION_DECISION.md` explicitly says that module was removed because it compared EconML with itself. README still advertises cross-implementation comparison. | Remove cross-implementation claims or implement an actual independent benchmark behind optional extras. |
| F7 | Critical | Heterogeneity chapter/status says `CausalForestDML`, BLP, and policy trees are implemented. | `econml` is absent in current venv; `src/` has no CATE/BLP/policy-tree implementation; claims live in chapter text only. | Reclassify as book discussion requiring `full` extras, not implemented repo feature, or add runnable optional examples and tests. |
| F8 | Critical | Build system is LuaLaTeX/biber with zero-error requirement. | `main.tex`, README, CLAUDE, and roadmap say LuaLaTeX+biber. `Makefile` uses `pdflatex` and `bibtex`. CI has no LaTeX book build. | Replace Makefile book targets with LuaLaTeX+biber or `latexmk -lualatex`; add CI/manual gate for the book log. |
| F9 | Critical | Sphinx docs are part of the public docs surface. | Current venv cannot run Sphinx; generated `_build` exists locally but is ignored; docs examples are stale. | Add `docs` optional extra or include Sphinx in dev env; build docs in CI and run doctested snippets. |
| F10 | High | Production pipeline is "end-to-end DML with proper time-series cross-fitting." | `InsuranceDMLPipeline.fit` implements its own simplified AIPW/PLR logic and does not delegate to `double_ml` or `DynamicDML`; time-series CV is used only if `time_column` is set, but `fit(X,T,Y)` has no time column data. | Rename it as a pedagogical production template, or refactor it to use the same estimator objects and carry fold indices, time indices, and assumptions. |
| F11 | High | Stationarity/cointegration guardrails are part of methodology. | Stationarity diagnostics exist, but the estimator and production pipeline do not enforce ADF/KPSS checks, cointegration checks, VECM sanity checks, low-variance treatment gates, or auto-differencing plans documented in remediation reports. | Add explicit pre-estimation diagnostics, or document that users must run them manually. Do not present them as pipeline behavior. |
| F12 | High | `double_ml` is appropriate for time-series examples. | `double_ml` uses shuffled KFold internally. `examples/example_time_series_dml.py` Part 3 says "DML with Temporal CV" but calls `double_ml` without temporal CV support. | Either add a `cv` parameter to `double_ml`, or change the example text and use `DynamicDML`/a temporal PLR estimator for temporal sections. |
| F13 | High | Dependency setup is coherent. | `requirements.txt` lists heavy optional packages not in current venv. `pyproject.toml` makes most of them optional under `full`, but README says `pip install -r requirements.txt`. | Make README install path use `pip install -e ".[dev]"`; document `.[full]`, `.[fred]`, and `-r requirements-docs.txt` separately. |
| F14 | Medium | Generated artifacts are ignored. | `.gitignore` ignores `*.bbl` and `results/`, but `main.bbl` and multiple `results/...` files are tracked. | Decide whether these are frozen evidence artifacts. If yes, unignore with comments. If no, remove from tracking in a cleanup commit. |
| F15 | Medium | Tests are fully aligned with docs. | Tests pass, but docs have stale tier counts; coverage threshold is only 30; Sphinx/README snippets are not tested. | Add contract tests for docs snippets and key chapter code blocks. Raise coverage only after removing false gates. |
| F16 | Medium | Local command UX is clear. | Bare `python` is absent; bare global `pytest` fails to import `src`; README and roadmap use bare commands. | Standardize on `venv/bin/python -m ...`, `uv`, or package install instructions. |

## Source-Of-Truth Map

| Surface | Status | Reason |
|---|---|---|
| Runtime code in `src/` | Current, but claims must be checked | Live signatures and tests are authoritative for API behavior. Some docstrings overclaim. |
| Tests under `test/` | Current for exercised behavior | 796 collect and tier1/tier2 pass. They do not validate README/Sphinx snippets or all book claims. |
| `README.md` | Conflicting | Correct total tests, stale page badge, broken quickstarts, overclaims validation and DynamicDML. |
| `CLAUDE.md` | Partly current, partly stale | Page count and total tests mostly current; tier counts stale; "no gaps" and zero-warning claims are too strong. |
| `docs/CURRENT_WORK.md` | Conflicting | Says project complete but known issues include page-count gap; active plan says expansion is still active. |
| `docs/MASTER_ROADMAP_2025-11-21.md` | Stale as source of truth | It says it is the single source of truth, but later current work and active plan disagree. |
| `docs/plans/active/2026-03-01_22-00_content_expansion.md` | Current as open book-content plan unless explicitly closed | It documents thin chapters and real-data gaps that contradict "complete." |
| Old audit/remediation docs in `docs/` | Historical only | Some identify real risks, but several include claims now false or never implemented, such as `temporalcv` integration. |
| Sphinx docs | Broken public contract | Build deps missing locally and many snippets do not match live APIs. |
| Chapters | Valuable but not fully executable | Several chapter examples depend on missing optional packages or use stale API patterns. |
| Examples under `examples/` | Runnable but require labeling fixes | All run; at least one says temporal CV while using shuffled KFold. |
| CI workflows | Partly current | Tests and Sphinx workflows exist. No book build CI; Sphinx workflow may catch docs build but not snippet truth. |
| Makefile | Stale | Uses pdflatex/bibtex and old test markers, unlike current docs and pytest tier system. |

## Methodology Audit

### FWL

The FWL material and implementation are salvageable. The example verifies
algebraic equivalence to OLS to numerical precision. This is book-grade if the
chapter code is kept executable.

### Robinson and Basic DML

The FWL -> Robinson -> DML progression matches the main pedagogical arc. The
basic `double_ml` function implements a partially linear residualization
estimator with cross-fitting, scalar `theta`, influence-score standard error,
and nuisance R2 diagnostics. This is consistent with the core DML story for
cross-sectional PLR-style examples.

Limit: `double_ml` always uses shuffled KFold. It should not be used as a
time-series estimator unless the docs clearly explain the assumptions or it
accepts a temporal splitter.

### Time-Series Cross-Validation

`TimeSeriesCrossValidator` provides expanding/sliding split behavior with
`gap`, `purge_length`, `test_size`, and `min_train_size`. The implementation is
useful as a teaching splitter. It does not use `time_index` in the basic
implementation, and the public docs use the wrong keyword `purge`.

The Hyndman/Bergmeir/Koo reference also means the repo should avoid a blanket
"standard K-fold always fails for time series" statement. The safer statement is:
standard K-fold is often inappropriate for causal nuisance estimation under
serial dependence unless the dependence/forecasting error conditions justify it.

### HAC

The HAC/Newey-West component is a reasonable book-companion implementation.
The live API returns an estimator from `HACEstimator.fit`; Sphinx examples that
expect `result.se`, `result.bandwidth`, and `result.effective_df` from `fit`
are false.

### DynamicDML

This is the largest methodology gap. Lewis-Syrgkanis dynamic DML is about
dynamic treatment effects with multiple treatments over time, structural nested
mean models, and recursive peeling. The current class:

1. Adds lagged treatment and covariates to `X`.
2. Cross-fits nuisance predictions.
3. Computes one scalar PLR coefficient.
4. Computes HAC SE on influence scores.

That can be a useful "temporal PLR DML" estimator. It should not be sold as
period-specific dynamic g-estimation. The current class also backfills early
fold predictions using future data, weakening the time-ordering claim.

### RollingWindowDML and PanelDML

These are useful teaching wrappers, but docs should frame them as simplified
estimators. They are not a full panel causal inference framework with formal
cluster/multiway inference, staggered-treatment guarantees, or dynamic CATE.

### Stationarity and Cointegration

The stationarity diagnostic module is useful and tested. The claims around
stationarity enforcement, auto-differencing, VECM sanity checks, low-variance
treatment gates, and cointegration are not implemented. For a book companion,
that is acceptable only if the docs say "diagnostic tools provided; enforcement
is manual."

### Production Pipeline

The production module is better described as a pedagogical template. It is not
production-grade software even under a modest interpretation: it lacks a real
data contract, time-index propagation, estimator reuse, operational deployment,
security, alert routing, rollback validation, and tested monitoring dashboards.
For the user's intended "book-grade only" standard, it can stay if renamed and
scoped honestly.

## Docs/API Truthfulness Findings

Published snippets that fail against live APIs:

| Snippet source | Failing API | Runtime failure |
|---|---|---|
| README time-varying treatment effects | `DynamicDML(n_periods=10, n_splits=3)` | Unexpected keyword `n_periods` |
| README time-varying treatment effects | `result.period_effects` | Attribute absent |
| README FRED controls | `FREDLoader().load_controls(...)` | Attribute absent |
| Sphinx quickstart basic DML | `double_ml(..., model_y=..., model_t=...)` | Unexpected keyword `model_y`; live names are `outcome_model` and `treatment_model` |
| Sphinx time-series CV | `TimeSeriesCrossValidator(..., purge=5)` | Unexpected keyword `purge`; live name is `purge_length` |
| Sphinx synthetic FRED | `create_synthetic_fred_data(n_periods=120)` | Live function takes date range/frequency and returns `MacroControlsResult` |
| Sphinx insurance DGP | `InsuranceDGPParams(n_obs=...)` | Live dataclass has different fields; public constructor is `create_insurance_dgp(...)` |
| Sphinx DynamicDML | `n_folds`, `cv_type`, `result.theta_t`, `result.se_t` | Constructor/result fields do not exist |
| Sphinx HAC | `result = hac.fit(...); result.se` | `fit` returns `HACEstimator`; no `se` field |

Examples under `examples/`:

- `example_time_series_dml.py`: runs, but Part 3 says "DML with Temporal CV"
  while it calls `double_ml`, which uses shuffled KFold.
- `example_fwl_to_dml.py`: runs, but its DML confidence interval missed the
  true value in this audit run (`[1.866, 1.989]` for true 2.0). That is not a
  bug by itself, but a deterministic teaching example should not present a
  non-covering draw without explaining sampling variability.
- `example_sensitivity.py`: runs.
- `example_production_pipeline.py`: runs, but treatment R2 was negative in the
  audit run (`-0.049`), so the demo should explain weak nuisance diagnostics.
- `example_insurance_dgp.py`: runs.

## Architecture Assessment

What is good:

- The package has clear broad folders: `src/dml`, `src/validation`, `src/data`,
  `src/production`, `src/sensitivity`.
- Tests are broad, fast enough for tier1/tier2, and currently green.
- Core educational flow exists: FWL, Robinson, DML, HAC, temporal CV, DGPs.
- `pyproject.toml` has a saner optional-dependency split than `requirements.txt`.
- Existing `post_transformers` repo suggests a workable pattern: one canonical
  audit index, one roadmap, explicit status taxonomy, and Makefile gates that
  use the project venv.

What is risky:

- The installed Python package is literally `src`, because setuptools includes
  `src*` from repo root. This works locally but is a weak public package shape.
- Public APIs are inconsistent: strings vs sklearn estimator objects, `model`
  vs `model_y`/`model_t`, `purge` vs `purge_length`, scalar result vs dynamic
  result names.
- Docs are sprawling: 58 active markdown/rst docs excluding Sphinx build output,
  24 root-level docs under `docs/`, 19 archived docs, and 2 active plans. There
  is no enforced current/stale boundary.
- Makefile, README, CLAUDE, CI, and pyproject do not agree on build/test commands.
- Tracked generated artifacts conflict with `.gitignore`.
- Book build is central but absent from CI.

## Cleanup Vs Restart Decision

Partial restart is the correct call.

Keep:

- Core `src/dml/fwl.py`, `src/dml/robinson.py`, `src/dml/double_ml.py` for the
  cross-sectional book arc.
- `src/dml/hac.py` after API/docs cleanup.
- `src/dml/cross_fitting.py` as a teaching splitter after keyword/docs fixes and
  clearer scope.
- Synthetic DGPs, stationarity diagnostics, and current tiered tests.
- LaTeX chapters as a draft manuscript, but with code-snippet truth checks.

Restart or heavily rewrite:

- The DynamicDML public contract and chapter narrative.
- Sphinx docs and README quickstarts.
- Source-of-truth documentation hierarchy.
- Makefile and book/doc verification gates.
- "Production-grade" language and production pipeline scope.

Only full-restart if the target changes to "publish a real Lewis-Syrgkanis
dynamic treatment effects library." For the current target, a book companion,
the repo is worth cleaning.

## Minimum Viable Remediation Plan

1. Freeze source of truth.
   - Keep `docs/audits/README.md` as the audit index.
   - Create or update one `docs/CURRENT_STATUS.md`.
   - Mark `MASTER_ROADMAP`, old audit reports, and old plans as superseded or
     historical.

2. Make public language honest.
   - Replace "production-grade" with "book companion" except where discussing
     conceptual production templates.
   - Change DynamicDML descriptions to "temporal partially linear DML with lagged
     controls and HAC inference" unless true g-estimation is implemented.
   - Remove cross-implementation, CausalForest, BLP, and policy-tree claims from
     "implemented" sections unless optional examples are made runnable.

3. Repair API contract.
   - Update README and Sphinx snippets to live signatures.
   - Add a snippet test script that executes fenced Python examples selected
     from README, Sphinx quickstart, and core chapters.
   - Standardize estimator constructor conventions.

4. Fix time-series leakage.
   - Remove future backfill from `_cross_fit_nuisance_time_series`.
   - Add tests asserting every prediction is trained only on allowed indices.
   - Decide whether early rows are dropped, left missing, or estimated with an
     explicitly labeled fallback.

5. Align build/test tooling.
   - Replace Makefile pdflatex/bibtex with LuaLaTeX/biber or latexmk.
   - Use `venv/bin/python -m pytest` or a declared environment tool everywhere.
   - Add book build/log check to CI or a documented release gate.
   - Fix the pre-commit template hook to use the venv.

6. Split dependency tiers.
   - README default: `pip install -e ".[dev]"`.
   - Optional: `.[fred]`, `.[full]`, and docs requirements.
   - Chapter 9 and FRED-live examples should clearly require extras.

7. Reclassify examples.
   - `examples/` should be deterministic and labeled by chapter.
   - Each example should say whether it is cross-sectional, temporal PLR, or
     synthetic/optional.
   - Add CI that runs examples with core deps.

## Deeper Redesign Option

If you want a cleaner architecture after the minimum cleanup:

```
src/
  dml/
    plr.py                     # cross-sectional PLR DML
    temporal_plr.py            # current DynamicDML, renamed
    dynamic_g_estimation.py    # true recursive dynamic DML, if built
    cv.py                      # temporal splitters
    inference.py               # HAC, influence functions
  diagnostics/
    stationarity.py
    cointegration.py
    overlap.py
  data/
    fred.py
    synthetic.py
  examples/
    contracts.py               # shared code used by docs snippets
docs/
  CURRENT_STATUS.md
  roadmap.md
  audits/
  specs/
    public_api.md
    chapter_claims.md
```

The key architectural move is to separate estimators from diagnostics,
diagnostics from validation studies, and book prose from tested executable
contracts.

## Product And Production-Grade Questions To Resolve

These are the questions I would require before declaring the cleaned repo
book-grade:

1. Is the book teaching "DML for time series intuition" or claiming a valid
   implementation of Lewis-Syrgkanis dynamic treatment effects?
2. Is `DynamicDML` allowed to be renamed if that is the most honest fix?
3. Should the book include a true dynamic g-estimation implementation, or should
   it explicitly stop at temporal PLR with lags?
4. What is the primary estimand for the insurance/pricing chapters: scalar
   elasticity, lag-specific impulse response, period-specific effect, or CATE?
5. Are real FRED/API examples required to run in CI, or should they remain
   optional/network examples?
6. Should Chapter 9 be an EconML optional chapter or a repo-native implemented
   feature chapter?
7. Is "production pipeline" meant as conceptual architecture, reusable template,
   or actually deployable software?
8. What is the minimum acceptable build gate for the book: no TeX errors only,
   no warnings, or no overfull/underfull boxes above a threshold?
9. Do you want generated evidence files in `results/` tracked as frozen audit
   artifacts?
10. Should `main.pdf` be tracked even though PDFs are ignored?
11. Should the package remain importable as `src.*`, or should it move to a real
    package namespace like `double_ml_time_series.*`?
12. Should docs examples use only core dependencies, or can they depend on
    `.[full]`?
13. What is the acceptance criterion for "validated against published results"?
14. Do you want independent implementation comparison, or is empirical
    replication enough?
15. Should stationarity checks block estimation or only warn?
16. Should cointegration checks be added before any differencing advice appears
    in chapters?
17. How much non-stationarity is the book allowed to handle before recommending
    classical time-series methods instead of DML?
18. Should low-variance treatment gates be part of code or just chapter guidance?
19. Should temporal CV support label horizons and embargo intervals explicitly?
20. Should `double_ml` accept a `cv` object to avoid duplicating PLR logic?
21. Should all chapter code blocks be executable, or only blocks tagged as
    runnable?
22. Should examples use fixed seeds that guarantee pedagogically clean CIs?
23. What docs are allowed to remain at `docs/` root after cleanup?
24. Who owns updating `CURRENT_STATUS.md` after future changes?
25. Should CI run tier3/nightly on every push to main, or only schedule/manual?
26. Should `requirements.txt` be removed in favor of `pyproject` extras?
27. Should `post_transformers` style be copied only for audits/status, or also
    for guide authoring standards?
28. What is the release bar for version 1.0.0: code tests only, book build, docs
    build, examples, or all of them?
29. Are private/real insurance data workflows in scope, or should the repo stay
    fully public/synthetic?
30. Should the final product optimize for reader trust over feature breadth, even
    if that means cutting advanced claims?

## Bottom Line

This repo should be saved, not restarted wholesale. But the cleanup must be
truth-first and fairly aggressive: remove or rename false dynamic-DML claims,
make every public snippet executable, collapse source-of-truth sprawl, and align
the build/test/dependency story. The existing code can support a strong
book-companion repo if the book stops claiming more than the code actually does.
