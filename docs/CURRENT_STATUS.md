# Current Status

Last updated: 2026-11-27

This file is the canonical project-status source: one baseline and one roadmap. Historical
reports and roadmaps are in `docs/archive/`. The current audits are listed in
[`docs/audits/README.md`](audits/README.md); the first to read is the
[time-series DML audit of 2026-11-19](audits/2026-11-19_time_series_dml/README.md).

The repository is a book-grade companion to a Double Machine Learning manuscript. It is not
deployable production software and not a stable public package.

## Where the work stands

The November 2026 work was held in one uncommitted working copy and has been landed in
reviewed pull requests ([plan](plans/active/2026-11-24_backlog_landing.md)). Part I of that
plan lands what does not depend on the panel estimator: nine of its pull requests and #79 are
merged, and two are open (#82, #89). Its conservation check passed on 2026-11-27: every path in
which the working copy differs from `main` is landed, in an open pull request, or recorded as
held (snapshot `refs/snapshots/backlog-2026-11-24`). Part II, the panel estimator, has started;
Part III lands what waits for it.

### On `main`

| Subject | State |
|---|---|
| Core numerics (#69, #74) | The "no treatment variation" check is relative to the treatment's size; a column vector as outcome or treatment is refused by `fwl_estimate`, `robinson_estimator`, `double_ml`, `TemporalPLRDML`, `RollingWindowDML` and `PanelDML` (`DynamicGEstimationDML` flattens it); series-mode `DynamicGEstimationDML` refuses lags that are not identified |
| Validation inference (#68, #78) | The bias test of the seven validators tests mean error across independent simulations; the six baselines also give a coverage verdict; the 401(k) comparison uses the published fivefold benchmarks |
| Applications (#80) | The insurance pipeline scores held-out rows in the treatment's units; the insurance generator returns its rows in one product-major order; the sensitivity module's docstrings call its curve a log-gamma bias curve (its printed summary is still headed "Rosenbaum Bounds"); the orange-juice loader checks a pinned digest; the registry and the result store write atomically |
| Input contract, first half (#79) | `fwl_estimate`, `robinson_estimator` and `double_ml` read inputs by value and pair them by position, refuse by name what a conversion would misread, and refuse two pandas inputs with different indexes (`as_validated_arrays`, `dml_ts/dml/_utils.py`) |
| Continuous integration (#73) | Installs from `uv.lock`; a weekly job with unpinned dependencies reports drift and blocks nothing |
| Audits and Chapters 2 and 5 (#83, #87) | The four November audit folders with their tests, the effect-forecast module, both editions of Chapters 2 and 5, and the public-data experiments of 2026-11-21 |

### Open, and what follows from it

- **The second half of the input contract** (#82, `fix/one-input-boundary`):
  `fwl_residualize`, `fwl_vs_ols_comparison`, `TemporalPLRDML`, `RollingWindowDML` and
  `DynamicGEstimationDML`. It is held for the owner's decision on one regression its final
  review found: a one-field structured float array as Y or T, which `main` reads, is refused by
  the three temporal estimators. Until it lands, those estimators read labelled
  inputs by position without comparing their labels: for a treatment that carries the
  outcome's labels in another order, `TemporalPLRDML` returns -0.18 where the arrays give
  2.09 (#75, #77).
- **`CausalMonitor` reports a randomized binary treatment as "critical"** through the
  pipeline, and judges an effect's change by its relative size alone (#64). The repair, #89,
  waits for a go-ahead.
- **`PanelDML` is the estimator of June, and it is biased when confounding is nonlinear in the
  controls.** It removes the fixed effects from the controls before a learner sees them: with
  40 units, 15 periods and a confounder `x**2`, the default forest gives a bias of +1.07 for a
  slope of 2 and 0% coverage in 40 replications (#84). It also casts unit and period
  identifiers to integers and reads labelled inputs by position (#77). Part II replaces it.
  The design was chosen on 2026-11-27
  ([design gate](plans/active/2026-11-27_paneldml_design_gate.md)) and is tested under a
  protocol frozen before any run ([protocol](plans/active/2026-11-27_paneldml_protocol.md)).
  Until then:
  - Chapter 6 and its listings describe the June estimator.
  - **Chapter 8's printed numbers do not reproduce.** They were computed before the insurance
    generator's rows were aligned: its `PanelDML` listing prints -0.8553 with a standard error
    of 0.0330, and `main` returns -0.8531 with 0.0175, an interval that excludes the true -0.8.
  - Chapter 4 cites `retail_check.py`, which is held.
- **Held for Part III**, recorded file by file in the plan: the redesigned panel module and the
  28 audit files that call it or report its numbers, the end-to-end review of 2026-11-22 and
  the second audit it reviews, nine test files that need a held file, the Monte Carlo scripts
  of Chapters 6 and 8, and the nightly workflow's November changes. The audit reports that
  landed refer to some of them; where they say Chapters 6 and 8 report corrected panel
  designs, that is not true of `main`.
- **Held for their licence:** the rows of the Survey of Professional Forecasters and two
  receipts that embed them ([`DATA_SOURCES.md`](audits/DATA_SOURCES.md)).

### Known limits of what is on `main`

- `PanelDML` under confounding nonlinear in the controls (#84), above.
- **The archived cricket forecast replays only on some machines:** `GroupKFold` orders groups
  of equal size with an unstable sort. The issued forecast stands; the two tests that replay
  it are held (#86).
- The default interval of `TemporalPLRDML` is far too short when few scored rows carry the
  treatment's variation (#81), and under persistent scores (#54).
- The default learners are biased under persistent covariates (#51); with the lasso learner the
  estimate depends on the treatment's units. The validation gates cannot detect these failures
  (#56).
- Series-mode g-estimation returns direct effects, not blips, under feedback from treatment to
  state (#50).
- `FREDLoader.get_macro_controls` returns no rows for the default monthly call under pandas 3
  (#62). The 401(k) default control list includes post-treatment wealth (#63).
- `fwl_residualize` reports an R² of NaN for data scaled to 1e-170 or 1e155, where the right
  value is about 0.51: the sums of squares underflow or overflow (#90). `double_ml` and
  `robinson_estimator` refuse at those scales.
- Tier 3 and tier 4 run only in the `Nightly Tests` workflow, whose schedule has not fired since
  2026-10-19. A run dispatched by hand on 2026-11-27 passed tier 3; tier 4 runs only on the Sunday
  schedule and cannot be dispatched (#88).
- Open findings on the manuscript: #49, #57, #59, #65, #85.

## Baseline

The gate numbers of this commit are added in the next commit, which changes only this section.

Not run: tier 3 and tier 4 (nightly only); the test cases that replay external archives
(`DML_TS_EXTERNAL_REPLAY=1`); `npm run build`.

## Roadmap

| Stage | Content | Gate |
|---|---|---|
| **Part II** | `PanelDML` rebuilt with the nuisance treatment a required argument: A (basis first), B (correlated random effects with the treatment-mean correction) and C (exact first difference); `whole_unit` folds, unit-clustered inference by default, identifiers compared as given. `panel_standard_error` lands first, on its own. The study runs under the frozen protocol, then an independent challenge and an adversarial review | the owner decides from the study |
| Alongside Part II | The manuscript errata that do not depend on `PanelDML`: #65, then #85, #57, #59 (the Chapter 7 parts), #49; the web scaffold to `^4.26.x` (#46); FRED under pandas 3 (#62); the comments of `main.tex` | one pull request each |
| **Part III** | `PanelDML` landed; the insurance panel and the falsification study; the held audits and tests; the end-to-end review; Chapters 6 and 8 recomputed | as planned |
| Close | A final conservation check against the snapshot; the mirror backups and the retired working copy removed; whether to release 1.2.0 | the owner |
| Methodology | #51, #81 and #54, #50, #56, #52, #53, #55; then #61, #63, #44, #41, #42, #58, #66, #21, #70, #71, #86 | P1 first, then by what printed numbers depend on |
| Deferred | EconML heterogeneity examples as tested features; blocking stationarity, cointegration and overlap diagnostics, opt-in and built on temporalcv; heterogeneous `theta_t(X)`; book content expansion from a new baseline | after the methodology stage |

## History

From May to June 2026 the repository went through a truth-first remediation (2026-05-02:
overclaims removed, toolchains gated in CI), the web companion (`web/`, all ten chapters ported
by 2026-06-01, drift-guarded), the `dml_ts` namespace (2026-06-01), `DynamicGEstimationDML`
(2026-06-04) and the move onto temporalcv 2.2.0 (golden-neutral, 1.1.3 and 1.1.4). Its
baselines and decisions are in the
[2026-05-30 audit](audits/2026-05-30_repo_quality_methodology_audit.md), the CHANGELOG
and `docs/archive/`. The web companion's working notes are in `web/CLAUDE.md` and
`web/UPSTREAM_ISSUES.md`.

## Canonical References

- Current audits: the November 2026 folders listed in `docs/audits/README.md`
- Data sources, licences and held files: `docs/audits/DATA_SOURCES.md`
- Superseded audits: `docs/audits/2026-05-30_repo_quality_methodology_audit.md`,
  `docs/audits/2026-05-02_repo_quality_methodology_audit.md`
- Historical status, reports and plans: `docs/archive/`
- Active plans: `docs/plans/active/`
- Web companion: `web/CLAUDE.md`, `web/UPSTREAM_ISSUES.md`

## Working Rules

- Runtime behavior, tests, examples, build logs and primary literature outrank old roadmaps or
  status files.
- Public snippets in README, Sphinx, examples and chapters either execute under documented
  dependencies or are labeled as conceptual.
- The current temporal estimator is scalar temporal PLR DML, not recursive dynamic
  g-estimation.
- "Production" language means book-companion production concepts unless code is explicitly
  tested as deployable software.
- LaTeX `chapters/*.tex` is canonical book content; the `web/` MDX is a parallel port,
  drift-guarded against it.
