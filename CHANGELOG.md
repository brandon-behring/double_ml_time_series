# Changelog

All notable changes to the `dml_ts` book-companion package.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning policy: see [CONTRIBUTING.md](CONTRIBUTING.md) (pre-PyPI; book listings
are the compatibility contract).

## [Unreleased]

### Changed

- `RollingWindowDML` fits complete windows of exactly `window_size` consecutive rows. Starts
  advance by `step_size` from row 0, and the reported center is `start + window_size // 2`.
  Before, windows were cut around centers: an odd `window_size` gave windows of
  `window_size - 1` rows, and with an even one the window that ends at the last row was dropped
  when `n - window_size` is a multiple of `step_size`. `window_size` and `step_size` must be
  positive integers, and the lengths of `Y`, `T`, `X` and `time_index` are checked. (#42)
- A classifier passed as a nuisance model supplies its conditional mean through `predict_proba`
  (numeric class values weighted by their probabilities) in `double_ml`, `robinson_estimator`
  and native `DynamicGEstimationDML`. Before, `predict` returned the most likely class. A
  classifier without `predict_proba` raises `ValueError`. `robinson_estimator` fits one `model`
  to both the outcome and the treatment, so a classifier works there only when both are
  categorical.
- The R² helpers return 0.0 only when the outcome is constant up to rounding: the norm of the
  centered outcome is at most `norm(y) * n * eps`. The cutoffs were absolute (`1e-10` and
  `1e-12` on the sum of squares) and returned 0.0 for a varying outcome in small units.
- `PanelDML` reports a treatment that its fixed effects absorb before it fits a nuisance model,
  with the message "no variation after removing ... fixed effects". The same input raised
  before, from the residual check of the inner estimator.
- Docstrings of `double_ml` and `TemporalPLRDML` describe the estimate as the slope of the
  partially linear model. They called it the ATE; with a binary treatment and effects that vary
  with X the slope weights each effect by `Var(T|X)` and can differ from the ATE.
- `dml_ts.sensitivity` describes its curve as what it is: an assumed log-gamma bias function
  applied to a two-sided normal test of a scalar estimate. It is not the Rosenbaum bound for
  matched pairs, and the module's docstrings and API page say so. The class keeps its name,
  `RosenbaumBounds`, and its summary is headed "Illustrative Rosenbaum Bounds Sensitivity
  Analysis"; the example still prints the old name. The result says whether the tested grid
  crossed the significance level and labels its threshold as a threshold or as a lower bound;
  for a grid without a crossing, `interpretation` categorizes the grid's upper end. (#44)
- `InsuranceDMLPipeline` fits nuisance preprocessing within folds and reports intervals from
  held-out scores: the partially linear score for a continuous treatment, the augmented
  inverse-probability score for a binary one. A binary treatment needs a model with
  `predict_proba` and a continuous one a regressor; folds or scored samples without both
  treatment arms raise. The pipeline hands `CausalMonitor` the held-out R² of both nuisance
  fits; for a binary treatment that is the R² of the held-out probabilities, where it was
  classification accuracy. The monitor's default thresholds (0.50 and 0.30) are unchanged and
  the pipeline has no setting for them, so a randomized binary treatment, whose best
  propensity R² is zero, is reported as critical and `evaluate_retrain_need` returns a
  trigger; both case studies of Chapter 10 are reported as critical. The monitor's alerts
  remain as #64 describes them. (#60, #64)
- `create_insurance_dgp` returns every variable in one product-major order, so an outcome, its
  treatment and its controls belong to the same product and month, and
  `InsuranceDGPResult.treatment_history(n_lags)` returns the treatment of the same product at
  the preceding months. Results computed from the generator's earlier output change. Chapter 8
  is not regenerated yet: its `PanelDML` listing prints -0.8553 with a standard error of 0.0330
  and an interval that covers the true -0.8; on the aligned rows the same call returns -0.8531
  with 0.0175 and an interval that excludes it, and the recovery study of its exercise covers
  the truth in 21% of 100 draws (56% on the earlier rows; the chapter prints 93.0%). The
  chapter is recomputed with the `PanelDML` redesign.
- `OJDataLoader` checks its download against a pinned SHA-256 digest; a cached file that does
  not match is not served, and a failed download says so. `FREDLoader.get_macro_controls`
  computes year-over-year changes at the target frequency: 52 weeks, 12 months, 4 quarters or
  one year back, and for daily data the same calendar date a year earlier. A frequency without
  a fixed number of rows per year (business days, two weeks) raises for that transform. Not
  repaired: in a daily request of a series without weekend rows the filled rows repeat the
  previous row's change, and a weekly series requested daily returns no rows. (#62)
- `DMLModelRegistry` and `ResultStorage` write complete JSON documents atomically under a file
  lock; the registry keeps a deployment history, and a failed archive leaves the prior index and
  payload in place. A store or registry opened with a relative path keeps meaning the same
  directory after the working directory changes, a store that has been moved or renamed still
  loads its results, and published files have the permissions of an ordinary file. `filelock`
  is a declared dependency.

### Fixed

- The DML bias validator and six baselines now test mean error across independent
  Monte Carlo draws. Bootstrap resamples supply the bias interval. Validation
  intervals and the DML coverage null use the configured confidence level.
  [Effect on archived validation verdicts](docs/audits/2026-11-19_time_series_dml/08-bias-test-correction.md)
- The six validation baselines (`NaiveOLS`, `OLSWithControls`, `IPWEstimator`, `AugmentedIPW`,
  `RandomForestEstimator`, `XGBoostEstimator`) judge interval coverage with the exact binomial
  test the DML validator uses, and their status is the worse of the bias and coverage verdicts.
  They classified the bias p-value alone, so `OLSWithControls.validate()` returned PASS with 76%
  of its 95% intervals covering the truth, and `NaiveOLS` with 34%. Their bias verdict follows
  the DML validator's rule as well: FAIL needs a bias above 0.15 (with no test), or of at least
  0.1 with a rejection at the 1% level; any other rejection at the 5% level is a WARNING.
  `OLSWithControls` now fails wherever it is run, because its standard error leaves out the
  fitted intercept (#61, not repaired here). The forest and XGBoost baselines report bias, MSE
  and coverage for the simulations that had both treatment arms, and warn when they leave one
  out. A simulation without noise, whose errors are at the rounding level of the true effect,
  is refused instead of judged. (#41)
- The 401(k) comparison uses the fivefold Table 2B eligibility benchmarks and
  reports uncertainty for the mean fitted effect. Its status records whether a
  descriptive gap meets the caller's tolerance. Direct participation fits have
  no Table 3 comparator, and the comparison p-value is unavailable.
  [Published and refitted results](docs/audits/2026-11-19_time_series_dml/07-401k-replication.md)
- `BootstrapDiagnostics` and `LassoDiagnostic` cut their cross-fitting folds on the original
  observations and map them onto each bootstrap draw, so the copies of one observation stay on
  one side of every split. Folds cut after the draw scored a nuisance model on rows it was
  trained on.
- `TimeSeriesDGPGenerator` draws its first treatment lags from the joint stationary
  distribution of the treatment and the confounder signal, and warns when the error
  innovation variance is floored. `EnhancedDGPGenerator` applies `omitted_var_strength` and
  `polynomial_degree` as documented and draws measurement noise last, so runs that differ only
  in `measurement_error_std` share their latent draws. `omitted_var_strength` was ignored
  before and defaults to 0, so `p_omitted > 0` without it now gives covariates that confound
  nothing; the generator warns in that case.
- The "no treatment variation" check is relative to the size of the treatment. `fwl_estimate`,
  `robinson_estimator`, `double_ml`, `TemporalPLRDML` and panel-mode `DynamicGEstimationDML`
  raise when the norm of the treatment residuals is at most the larger of two bounds:
  `1e-6 * norm(T - mean(T))` (the controls explain all but `1e-12` of the treatment's variance)
  and `sqrt(eps) * norm(T)` (the residuals are below `1.5e-8` of the treatment's level, as for a
  constant treatment). The absolute cutoff `sum(T̃²) < 1e-10` refused healthy data in small
  units and accepted an exact fit in large ones. The error message reports the residual norm and
  the bound. A treatment whose variation is below `1.5e-8` of its level is refused; subtract a
  constant from it first. The weak-overlap warning of `TemporalPLRDML` compares `mean(T̃²)` with
  `1e-6 * var(T)` for the same reason. (#42)
- `fwl_estimate(method="ols")` takes its degrees of freedom from the rank of the controls, so a
  redundant control column no longer changes the standard error. (#42)
- Series-mode `DynamicGEstimationDML` residualizes the treatments with `model_t`. It used
  `model_y` for both the outcome and the treatments and ignored `model_t`. The outcome and the
  treatments are now fitted separately, so series-mode estimates change with the default
  learners too; with ridge for both they are unchanged. (#42)
- Panel-mode `DynamicGEstimationDML` draws one set of folds per fit and uses it for the outcome
  and treatment residuals at every stage. With `random_state=None` each residualization was
  shuffled separately.
- `TemporalPLRDML.fit` and `effect_interval` raise on `alpha` outside (0, 1), and
  `effect_interval` returns ordered bounds when `T1 < T0`.
- `fwl_estimate`, `double_ml`, `robinson_estimator`, `TemporalPLRDML` and `RollingWindowDML`
  raise `ValueError` when the outcome or the treatment is not one-dimensional. An array of shape
  `(n, 1)`, as from `df[["t"]].to_numpy()`, broadcast against `(n,)` residuals to an `(n, n)`
  product, and the result was wrong with no error: for a true slope of 2, `fwl_estimate` returned
  a slope of order 1e-16 and `double_ml` returned 0.0005, `robinson_estimator` returned -0.002
  with a standard error of 0.004, and `RollingWindowDML` returned no window. The input is
  refused, not flattened; pass `np.asarray(T).ravel()`. `fwl_estimate` with both arguments as
  columns returned the right slope and is refused as well. (#72)
- Series-mode `DynamicGEstimationDML` refuses lag effects that are not identified on the scored
  rows, before it solves for them: a lag that is constant there; a lag whose residuals are
  within the bound the other estimators use; lags, or lag residuals, that are linearly
  dependent; and a fit with no more scored rows than lags. Before, a treatment that the
  controls reproduce exactly gave a lag-0 effect of order 1e13 to 1e15 under the linear
  learner, a constant treatment ended in `LinAlgError: Singular matrix`, and an on/off
  treatment (lag 1 is a constant minus lag 0) returned two lag effects, significant at the 5%
  level under gradient boosting. As in the other estimators, a treatment whose variation is
  below `1.5e-8` of its level is refused; subtract the level first. Not covered: a learner
  that only approximates a treatment the controls determine leaves residuals far above the
  bound, so for `T = 0.5 * x0` the ridge learner, the default random forest and gradient
  boosting still return an estimate. (#42)

## [1.1.4] - 2026-06-14

### Added

- **Global doctest gate.** A dedicated `doctest` CI job runs `pytest --doctest-modules dml_ts/`
  (`.[dev,full]`, `MPLBACKEND=Agg`); every docstring example across the package is now runnable
  (50 executed, 19 `# doctest: +SKIP` for network / real-401(k) / multi-thousand-bootstrap
  examples). Folded in the previously-missing `Examples` for the public result/estimator classes
  (`FWLResult`, `RobinsonResult`, `DMLResult`, `TemporalPLRDMLResult`, `RollingWindowDML`,
  `PanelDML`, `DynamicGEstimationDML`/`Result`). (#36)
- **Sphinx API coverage** for the 12 previously-undocumented `dml_ts.validation` modules
  (OLS/IPW/ML baselines, baseline comparison, bootstrap config/diagnostics, enhanced DGP,
  empirical replication, lasso diagnostic, validation result, storage, plotting) — the API
  reference now documents 18/18 validation modules. (#35)
- **Runnable examples** `example_rolling_window_dml.py` and `example_panel_dml.py`; the
  `examples/` gate now covers 8/8 scripts.

### Changed

- `DynamicGEstimationDML` accessors (`effect_by_period`, `cumulative_effect`) now raise
  `ValueError` (was `RuntimeError`) when called before `fit()`, matching the `TemporalPLRDML`
  not-fitted convention.

### Fixed

- Doc-vs-code mismatches surfaced by the new doctest gate (docstring-only): `lasso_diagnostic`
  advertised an unsupported `alpha` hyperparameter (the code raises `ValueError`) → docs reduced
  to `(cv_folds, max_iter)`; the `insurance_dgp` example unpacked the `InsuranceDGPResult`
  dataclass as a tuple; the `causal_monitor` example used stale kwarg names. (#36)

Behavior-neutral except the `RuntimeError`->`ValueError` accessor change; no public estimation
result changes. No PyPI publication (pre-PyPI per CONTRIBUTING).

## [1.1.3] - 2026-06-13

### Changed

- Bumped the locked `temporalcv` to **2.2.0** (the `>=2.0.0,<3` pin already admitted it).
  Relative to the last released lock (2.0.1), 2.2.0 hardens the purged-splitter family to
  raise on under-provisioned configs (temporalcv #35/#36), makes `validators.psd`
  scale-aware (temporalcv #33), adds a lossless serialization shape and a
  `BaseCrossValidator` base for the purged splitters (temporalcv #21/#25), and — in 2.2.0
  itself — fixes a `CombinatorialPurgedCV` embargo-leakage bug by making the embargo
  one-sided per contiguous test run (temporalcv #38), deprecates `PurgedKFold(shuffle=True)`
  (temporalcv #39), and tightens `estimate_purge_gap`/`compute_label_overlap` validation
  (temporalcv #40). The temporalcv #38 embargo change is **golden-neutral for dml_ts**,
  which uses `embargo_pct=0` on the forward-only `PurgedWalkForward` everywhere: goldens
  stay 71/71 byte-identical and the full tier1+2 suite (666 tests) passes against 2.2.0.
  The `DynamicGEstimation` covariance symmetrize is retained as defense-in-depth. No
  public-behavior change here.

## [1.1.2] - 2026-06-12

### Fixed

- **`bias_validation` 'holm' now performs real Holm step-down** (#14): the
  branch labeled 'holm' applied plain Bonferroni thresholds (the sorted
  p-values were never used). Per-hypothesis rejections now delegate to
  `statsmodels.stats.multitest.multipletests` for both 'bonferroni' and
  'holm'. Truth-in-labeling fix with no observable status change: with k=2
  tests and the any-rejection status rule, Holm's first step IS the
  Bonferroni threshold, so the tri-state status coincides under both
  methods — pinned by tests, alongside the distinguishing per-hypothesis
  case the old code got wrong (`p=[0.02, 0.04]`, α=0.05: Holm rejects
  both, Bonferroni one). Degenerate inputs (NaN, empty) now raise instead
  of inheriting statsmodels' divergent silent NaN handling.

### Changed

- **temporalcv 2.0.1** (lock refresh within the `>=2.0.0,<3` pin):
  `PurgedWalkForward` now raises on under-provisioned configs instead of
  silently dropping folds (temporalcv#32) — the failure mode dml_ts's
  cross-fit shortfall warning was working around. The warning stays as
  splitter-agnostic defense-in-depth; its comment no longer cites #32.
  Golden parity 71/71 byte-identical across the bump.

## [1.1.1] - 2026-06-12

### Changed

- **`temporalcv` now resolves from PyPI** (`temporalcv>=2.0.0,<3`) instead of
  the git tag pin. temporalcv 2.0.0 reached PyPI on 2026-06-12 after repairing
  the trusted-publisher config, which had been severed by that repo's
  mirror→canonical migration (the v1.0.1 and v2.0.0 publish runs both failed
  with `invalid-publisher`; PyPI was serving a stale 1.0.0). Before the flip,
  the PyPI wheel was proven behaviorally identical to the git tag: 71/71
  golden-parity tests and the full tier1 suite (339 tests) pass against a
  PyPI-only install. Installing no longer requires git access, and removing
  the direct-URL dependency unblocks a future dml_ts PyPI upload.
  `test_temporalcv_pin_resolves` now asserts the declared range rather than
  an exact version — exact behavioral agreement remains the golden suite's
  job.

## [1.1.0] - 2026-06-11

dml_ts is now a thin causal consumer of
[temporalcv](https://github.com/brandon-behring/temporalcv) v2.0.0: the
generic time-series infrastructure (CV splitters, HAC, stationarity
diagnostics) is consumed from the pinned upstream, each retirement gated on
golden snapshots captured after the 1.0.0 inference fixes. Net effect:
~4,000 lines retired, one temporal-leakage bug fixed (`purged_cv`), every
silent numeric fallback replaced with a raise or a warning, and the
manuscript corrected wherever it taught something the code never did.

### Removed

- **`dml_ts/dml/cross_fitting.py` retired onto temporalcv v2.0.0** (Track B):
  `TimeSeriesCrossValidator` and `BlockedTimeSeriesCV` import from temporalcv
  under the same names (they were ported there verbatim with bit-exact golden
  parity); `CVFold`/`get_fold_info()` are gone (temporalcv exposes
  `SplitInfo`/`get_split_info()`). `create_time_series_cv` survives as a thin
  factory (`dml_ts/dml/cv_factory.py`) returning temporalcv objects.
- **`PurgedGroupTimeSeriesCV` removed — it was leaky.** The retired splitter
  was a bidirectional purged K-fold: it trained on observations AFTER each
  test fold, i.e. nuisances saw the future. `TemporalPLRDML
  (cv_strategy="purged_cv")` and the factory's `"purged"` strategy now use
  the forward-only `temporalcv.PurgedWalkForward` (`gap` maps directly to the
  absolute `purge_gap`). **Previous `purged_cv` results were produced with
  temporal leakage**; the purged golden-snapshot keys were regenerated with
  provenance, and a no-lookahead invariant now gates every pinned splitter.
  `gap` maps EXACTLY to `purge_gap` (`embargo_pct=0.0` is set explicitly).
  Known inherited limitation: PurgedWalkForward silently skips
  under-provisioned folds (temporalcv#32) — the estimator now WARNS when the
  CV yields fewer folds than requested.

- **`dml_ts/validation/stationarity.py` retired onto temporalcv v2.0.0**
  (Track B): `StationarityDiagnostic`, `StationarityResult`, and
  `ComprehensiveStationarityResult` are gone from `dml_ts.validation` — use
  `from temporalcv import adf_test, kpss_test, pp_test, check_stationarity`
  (results: `StationarityTestResult` / `JointStationarityResult`). The module
  had no internal consumers; its capability is tested upstream.
- **`dml_ts/dml/hac.py` retired onto temporalcv v2.0.0** (Track B):
  `HACEstimator`, `newey_west_se`, and `newey_west_covariance` are gone from
  `dml_ts.dml` — use `from temporalcv import newey_west_se,
  newey_west_covariance, optimal_bandwidth`. Golden-parity gated: every
  estimator snapshot byte-identical (the temporalcv port was proven bit-exact
  against this module at its capture time).

### Added

- `dml_ts/dml/inference.py`: the causal-layer `hac_inference` survivor, now
  raising on non-positive/non-finite `se_hac` (previously reported t=inf,
  p=0.0 — "infinitely significant" from degenerate input).

### Changed

- **Numeric hard-fails at the arithmetic boundary (B3, closes #11).** Every
  result object's `_validate()` hook (the B1 seam) now rejects degenerate
  numerics at construction via temporalcv validators: non-finite theta,
  non-positive/non-finite SEs, inverted CIs, p-values outside [0,1],
  non-PSD covariance. The four inventoried silent-fallback sites are gone:
  TemporalPLRDML and PanelDML no longer fabricate t=0/p=1.0 from degenerate
  SEs (they raise); DynamicGEstimationDML no longer clips negative variances
  to zero nor launders NaN through `max(0, .)` in the cumulative SE (it
  raises); the demo pipeline now REFUSES to run with `use_hac=True` when HAC
  components are unavailable instead of silently reporting naive iid SEs,
  and refuses the silent shuffled-KFold downgrade when `time_column` is set
  without temporal CV available.
- **RollingWindowDML no longer silently skips windows (closes #10):** a
  failed or undersized window previously vanished AND shifted every later
  estimate onto the wrong time center (the center list was truncated, not
  filtered). Skips now warn with per-window reasons, and centers are kept
  aligned with the successful fits. DynamicG's covariance is symmetrized
  before validation (float roundoff at large scales is not asymmetry).

- `TemporalPLRDML` validates `hac_bandwidth` at construction (None or
  non-negative int, parameter-named error) and raises with context on
  non-finite influence scores. `hac_bandwidth=0` now means zero lags
  (heteroskedasticity-only SEs) instead of silently selecting automatic
  bandwidth (a falsy-check quirk). Invalid bandwidths that the retired module
  silently clamped/truncated (negative, fractional) now raise.
- The demo pipeline (`dml_ts/production`) maps `hac_bandwidth=None` to
  Andrews selection explicitly (the retired module did so via an else-branch
  accident); temporalcv's Andrews uses the literature alpha(1) constant for
  Bartlett, so SEs on this demo path shift slightly. The pipeline now WARNS
  loudly when HAC components are unavailable instead of silently degrading
  to naive iid SEs.
- Chapter 5's HAC listing teaches the temporalcv function API and the
  `HACResult` long-run-variance/variance/se split; the example output block
  is regenerated from a seeded, reproducible run (the previous block was not
  producible by any shipped code); the bandwidth definition now matches the
  implemented floor(n^(1/3)) rule.

## [1.0.0] - 2026-06-10

First stable release of the causal-layer API: the headline estimators
`double_ml`, `TemporalPLRDML`, `RollingWindowDML`, `PanelDML`, and
`DynamicGEstimationDML`.

### Fixed

- **TemporalPLRDML HAC standard errors were understated by a factor of sqrt(n)**
  ([#7](https://github.com/brandon-behring/double_ml_time_series/issues/7), PR #8):
  `HACEstimator.get_variance()` already returns the variance of the mean (long-run
  variance / n); the estimator divided by n again before the square root. t-stats,
  p-values, and confidence intervals were wrong by the same factor (empirical CI
  coverage ~0.10 at nominal 0.95). `RollingWindowDML` inherited the bug via its
  per-window fits. The vacuous coverage assertion that masked it
  (`assert coverage_rate >= 0.0`) was replaced with a real >= 0.80 gate plus an
  rtol-1e-12 SE consistency pin against the hac module.
- **PanelDML cluster-robust SEs were overstated by ~the effective cluster size**
  ([#9](https://github.com/brandon-behring/double_ml_time_series/issues/9), PR #13):
  cluster-level influence sums were treated as cluster means. Replaced with CR1
  (`G_eff/(G_eff-1) * sum_g S_g^2 / n_eff^2`) with effective-cluster accounting:
  clusters falling entirely inside the CV-dropped prefix no longer count toward G
  (disclosed via RuntimeWarning) and fewer than 2 retained clusters raises instead
  of silently returning se~0 / t=0 / p=1.0.

### Added

- `temporalcv` as a core dependency, pinned to the v2.0.0 release tag via git
  (`temporalcv @ git+https://github.com/brandon-behring/temporalcv@v2.0.0`) until
  both packages debut on PyPI. This is the Track B migration target for the
  generic time-series infrastructure.
- `dml_ts.__version__`, plus a tier1 gate (`test/test_version.py`) pinning version
  agreement across `pyproject.toml`, `docs/sphinx/conf.py`, `CITATION.cff`, and the
  package, and smoke-testing the temporalcv pin.
- Governance files: `CHANGELOG.md`, `CITATION.cff`, `CONTRIBUTING.md`,
  `ASSUMPTIONS.md`, `STYLE.md`.
- `uv.lock` for reproducible development environments (replaces the descriptive
  `requirements.txt`, now removed).

### Changed

- Lint/format tooling migrated black -> ruff (`ruff==0.15.16` pinned exactly in
  the dev extra; CI and pre-commit both run that pin), aligning with the
  temporalcv ecosystem. Includes a mechanical modernization pass (~630 autofixes:
  PEP 604/585 annotations, import sorting, formatting) plus targeted residue
  fixes (exception chaining via `raise ... from e`, a previously masked rollback
  error message in the model registry, silently dropped OJ rows now warned).

## [0.1.0] - 2026-06-01

Baseline as of the `src.*` -> `dml_ts` namespace migration (development began
2025-11; the 0.1.0 version string predates this date). Initial book-companion
baseline: FWL, Robinson, and cross-sectional `double_ml`;
`TemporalPLRDML` with `RollingWindowDML`/`PanelDML` companions;
`DynamicGEstimationDML` (recursive dynamic g-estimation, Lewis-Syrgkanis);
time-series CV helpers; HAC/Newey-West inference; FRED and orange-juice loaders;
synthetic DGPs and validation suite; LaTeX manuscript (`chapters/`) with the Astro
web companion (`web/`); 4-tier test system.
