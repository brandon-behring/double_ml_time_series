# Changelog

All notable changes to the `dml_ts` book-companion package.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning policy: see [CONTRIBUTING.md](CONTRIBUTING.md) (pre-PyPI; book listings
are the compatibility contract).

## [Unreleased]

No unreleased changes.

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

- Lint/format tooling migrated black -> ruff (format + lint, one pinned version
  across dev extras, CI, and pre-commit), aligning with the temporalcv ecosystem.
  Includes a mechanical modernization pass (PEP 604/585 annotations, import
  sorting, exception chaining via `raise ... from e`).

## [0.1.0] - 2026-06-01

Initial book-companion baseline: FWL, Robinson, and cross-sectional `double_ml`;
`TemporalPLRDML` with `RollingWindowDML`/`PanelDML` companions;
`DynamicGEstimationDML` (recursive dynamic g-estimation, Lewis-Syrgkanis);
time-series CV helpers; HAC/Newey-West inference; FRED and orange-juice loaders;
synthetic DGPs and validation suite; LaTeX manuscript (`chapters/`) with the Astro
web companion (`web/`); 4-tier test system.
