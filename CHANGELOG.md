# Changelog

All notable changes to the `dml_ts` book-companion package.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning policy: see [CONTRIBUTING.md](CONTRIBUTING.md) (pre-PyPI; book listings
are the compatibility contract).

## [Unreleased]

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
