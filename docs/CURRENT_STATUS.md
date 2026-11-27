# Current Status

Last updated: 2026-11-27

This file is the canonical project-status source. Historical reports and roadmaps
were moved to `docs/archive/superseded_2026-05-02/` during the 2026-05-02 truth-first
remediation. The current audits are listed in [`docs/audits/README.md`](audits/README.md);
the first to read is the
[time-series DML audit of 2026-11-19](audits/2026-11-19_time_series_dml/README.md).

The repository is a book-grade companion to a Double Machine Learning manuscript. It is not
deployable production software and not a stable public package.

## November 2026: where the landing of the November work stands

Work done in November 2026 was held in one uncommitted working copy and is being landed in
reviewed pull requests ([plan](plans/active/2026-11-24_backlog_landing.md)). This section says
what is on `main`, what is not, and what a reader must not rely on meanwhile.

### On `main`

| Subject | State |
|---|---|
| Core numerics (#69, #74) | The "no treatment variation" check is relative to the treatment's size; a column vector as outcome or treatment is refused by `fwl_estimate`, `robinson_estimator`, `double_ml`, `TemporalPLRDML`, `RollingWindowDML` and `PanelDML` (`DynamicGEstimationDML` flattens it); series-mode `DynamicGEstimationDML` refuses lags that are not identified |
| Validation inference (#68, #78) | The bias test of the seven validators tests mean error across independent simulations; the six baselines also give a coverage verdict; the 401(k) comparison uses the published fivefold benchmarks |
| Applications (#80) | The insurance pipeline scores held-out rows in the treatment's units; the insurance generator returns its rows in one product-major order; the sensitivity module's docstrings call its curve a log-gamma bias curve (its printed summary is still headed "Rosenbaum Bounds"); the orange-juice loader checks a pinned digest; the registry and the result store write atomically |
| Continuous integration (#73) | Installs from `uv.lock`; a weekly job with unpinned dependencies reports drift and blocks nothing |
| Panel standard errors (this change) | `panel_standard_error` and `group_codes` in the private module `dml_ts/dml/_panel.py`, which `PanelDML` does not use yet; three audit scripts that need only them, with two test files |
| Audits and Chapters 2 and 5 (#83) | The four November audit folders with 31 test files, the effect-forecast module with its test, and both editions of Chapters 2 and 5 |
| Public-data experiments | `docs/audits/2026-11-21_public_data_extensions/`: four experiments on public data (bicycle demand, household electricity offers, ride counts, a repeated cooperation game) with their scripts and reference outputs. Its comparison of `TemporalPLRDML` on matched bike hours reproduces its stored output byte for byte on `main`'s code |

### Not on `main`, and what follows from it

- **`PanelDML` is the estimator of June, and it is biased when confounding is nonlinear in the
  controls.** It removes the fixed effects from the controls before a learner sees them, so a
  learner cannot represent a confounder that depends on the controls' levels. On `main`, with
  40 units, 15 periods and a confounder `x**2`, the default forest gives a bias of +1.07 for a
  slope of 2 and 0% coverage in 40 replications; with a confounder linear in `x` it gives
  +0.01 and 90% (#84). A November redesign has the same property (the end-to-end review, which
  is not landed, measured about +1.04 and 0.3% for it) and is not landed. A new design is
  chosen and validated under a protocol fixed in advance before any change to `PanelDML`
  lands. Until then:
  - Chapter 6 and its listings describe the June estimator.
  - **Chapter 8's printed numbers do not reproduce.** They were computed before the insurance
    generator's rows were aligned. The chapter's `PanelDML` listing prints -0.8553 with a
    standard error of 0.0330; on `main` the same call returns -0.8531 with 0.0175, and that
    interval excludes the true -0.8. Its exercise prints 93.0% coverage and reaches 21% in 100
    draws. The chapter is recomputed when the panel estimator is settled.
- **21 files of the audit folders are held with it**, because they call the redesigned
  estimator or report numbers from the rejected design. The audit reports refer to them, and
  also to the fold transformations of `dml_ts/dml/_panel.py` (of that module only the
  standard errors are landed), to `scripts/validate_falsification.py`
  and to two of the held tests. The reports were written against the November working
  copy: where they say that Chapters 6 and 8 report corrected panel designs, that is not true
  of `main`. Two landed receipts of the 2026-11-21 reproduction,
  `standalone-stdout.json.gz` and `execution-code-sha256.json.gz`, contain the printed output
  and the digests of held scripts.
  - Scripts of `audits/2026-11-19_time_series_dml/`: `method_checks.py`, `classical/replay.py`, `external/turnout/replay.py`, `adversarial/failures.py`, `adversarial/empirical.py`.
  - Artifacts: `panel-draws.csv`, `insurance-panel-draws.csv`, `insurance-panel-summary.csv`, `insurance-rolling.csv`, `stress-draws.csv.gz`, `stress-summary.csv`, `adversarial/failures.json`, `adversarial/empirical.json`, `classical/results.json`, `external/turnout/household_counts.csv`.
  - Reports: `09-panel-retail-insurance.md`, `panel-inference.md`, `10-property-tests-and-falsification.md`, `adversarial/failures.md`, `adversarial/empirical.md`, `questions/minimum-wage.md`.
- **3 further files are held for their licence**: the rows of the Survey of Professional
  Forecasters and two receipts that embed them; see
  [`docs/audits/DATA_SOURCES.md`](audits/DATA_SOURCES.md). Four files whose sources state no
  licence that grants redistribution were held with them and landed by the owner's decision
  of 2026-11-26: three result files of the ACIC study, which carry the organizers' true
  effect for each contest file, and six announced cricket fixtures.
- **Not landed either:** the end-to-end review of 2026-11-22 and the second audit it reviews
  (their folders and tests), seven test files that need a held file, among them the test that
  checks the numbers printed in the chapters, one test of the minimum-wage case that expects a
  zero date-HAC standard error to be returned rather than refused, two tests that replay the cricket forecast (see
  the known limits), the Monte Carlo scripts of Chapters 6 and 8, and the nightly workflow's
  November changes. The `Nightly Tests` workflow last ran on 2026-10-19;
  tier 3 and tier 4 have not run in CI since then. It is run once by hand when this stage of
  the landing ends.
- **The insurance monitor reports a randomized binary treatment as "critical"** when it is
  called through the pipeline (#64).

### The input contract of the estimators

The contract: inputs are read by value and paired by position; what a conversion would misread
is refused by the argument's name, and two pandas inputs with different indexes are refused. It
arrives with the two changes below, in the docstring of `as_validated_arrays`
(`dml_ts/dml/_utils.py`). No estimator on `main` follows it yet.

| Estimator | Follows the contract |
|---|---|
| `fwl_estimate`, `robinson_estimator`, `double_ml` | From #79 on (open on 2026-11-26) |
| `fwl_residualize`, `fwl_vs_ols_comparison`, `TemporalPLRDML`, `RollingWindowDML`, `DynamicGEstimationDML` | From the pull request of branch `fix/one-input-boundary` on (open on 2026-11-26; refs #77) |
| `PanelDML` | Not yet: it casts unit and period identifiers to integers and reads labelled inputs by position without comparing them (#77). Its inputs are settled with its redesign |

Before those two changes, every estimator but `double_ml` returns a wrong slope with no error
for a treatment that carries the outcome's labels in another order: `fwl_estimate` and
`robinson_estimator` give 0.90 and 0.91 where the arrays give 2.03, `TemporalPLRDML` -0.18
where they give 2.09, and `PanelDML` -0.18 where they give 2.16; `RollingWindowDML` and both
modes of `DynamicGEstimationDML` are wrong in the same way. `double_ml` raises `KeyError`
(#75, #77).

### Known limits of what is on `main`

- `PanelDML` under confounding that is nonlinear in the controls: see above (#84).
- **The archived cricket forecast replays only on some machines.** Its cross-fitting assigns
  series to folds with scikit-learn's `GroupKFold`, which orders groups of equal size with an
  unstable sort; 1,069 of the 1,078 series share their size with another. The RMSE of the
  DML forecast is 12.5298 on the machine that issued it, 12.4767 there with a stable sort,
  and 12.5793 on the CI runner. The issued forecast stands as issued; the two tests that
  replay it are held (#86). The other four forecasts of that study use no folds and agree.
- The default interval of `TemporalPLRDML` is far too short when few scored rows carry the
  treatment's variation (one treated scored row: 7.5% coverage in 400 draws, #81), and under
  persistent scores (#54).
- The default learners are biased under persistent covariates (#51). With the lasso learner
  the estimate depends on the treatment's units (1.97, 1.11 and 1.97 for a slope of 2 when the
  treatment is multiplied by 1, 0.01 and 100); the default forest does not. The validation
  gates cannot detect these failures (#56).
- Series-mode g-estimation returns direct effects, not blips, under feedback from treatment to
  state (#50).
- `FREDLoader.get_macro_controls` returns no rows for the default monthly call under pandas 3
  (#62). The 401(k) default control list includes post-treatment wealth (#63).
- Open findings on the manuscript: #49, #57, #59, #65, #85.

### Verified on 2026-11-26

The gate was run on this change as built on `main` at `8feb54c`, in the locked environment
(`uv sync --locked --all-extras`, Python 3.13):

| Gate | Result |
| --- | --- |
| `ruff check`, `ruff format --check` | pass (143 files) |
| `mypy dml_ts/` | no issues in 45 source files |
| Test collection | 1,520 collected |
| Tier 1 | 586 passed, 2 skipped |
| Tier 1 + Tier 2, coverage | 1,304 passed, 6 skipped; coverage 85.50% (gate 70%) |
| Examples | 8 of 8 run |
| Doctests | 54 passed, 12 skipped |
| Sphinx `-W` | build succeeded |
| Book (`make`) | 190 pages |
| Web (`npm run validate`) | 10 chapters, no errors |
| Drift guard | every ported chapter matches its `.tex` |

Not run: tier 3 and tier 4; the 31 test cases that replay external archives
(`DML_TS_EXTERNAL_REPLAY=1`); `npm run build`.

## June 2026 status

The repository is a book-grade companion repo for Double Machine Learning pedagogy.
It is not a deployable production software project.

The June verdict was **healthy book companion.** The 2026-05-02 remediation closed the
major methodology overclaims (false dynamic-g-estimation / CausalForest /
production-grade language, broken snippets, time-series leakage, source-of-truth
sprawl). Build, test, and docs toolchains now agree and are CI-gated. Active work has
shifted to a `web/` Astro companion (see below).

Stable core:

- Cross-sectional FWL, Robinson, and `double_ml` (i.i.d.-style) teaching path.
- `TemporalPLRDML`: scalar temporal PLR DML with lagged controls, temporal
  cross-fitting, and HAC inference (not recursive dynamic g-estimation).
- `RollingWindowDML` and `PanelDML` companion estimators.
- HAC inference (consumed from temporalcv — the estimators call `newey_west_se` directly), stationarity diagnostics (retired; available via `from temporalcv import adf_test, check_stationarity`), time-series splitters, synthetic DGPs, test suite.
- LaTeX manuscript (10 chapters, 205-page PDF) as the canonical book.
- `web/` Astro companion, live at `dml.brandon-behring.dev` (pilot).

## Web Companion

`web/` is a `@brandon_m_behring/book-scaffold-astro@^4.8.0` academic consumer,
deployed to Cloudflare Workers at `dml.brandon-behring.dev` (PR previews enabled via
`deploy-web.yml`). Pilot tracked in GitHub issue #1.

- **Canonical content stays LaTeX.** `chapters/*.tex` + `main.tex` are the source of
  truth during the migration window; the MDX under `web/src/content/chapters/` is a
  parallel hand-port.
- **Bibliography is single-source:** `web` reads `../bibliography.bib` via
  `BOOK_BIB_PATH`, so LaTeX and web cite from the same file (no drift).
- **Port coverage: 10 of 10 chapters (2026-06-01).** All chapters are hand-ported to MDX
  (`chapter01`–`chapter10`). `npm run validate` checks 10 chapters with 0 errors; `npm run
  build` indexes 15 pages; the drift guard passes (exit 0) with every chapter's
  `source_sha256` matching its `chapters/chapter_NN.tex`. Parts use the scaffold's existing
  generic academic slots — `foundations` (1–3), `integration` (4–7), `synthesis` (8–10) —
  rather than modifying the in-flight `book-scaffold-astro` release; a consumer-configurable
  `parts` option is a deferred nicety (re-stampable in one line per chapter if it ships).
- **Drift guard (W3) — enforced:** each ported chapter's MDX records `source_file` +
  `source_sha256` of its LaTeX source; `scripts/check_tex_mdx_drift.py` recomputes and
  fails on mismatch, wired into pre-commit and the `tests.yml` CI lint job. Re-stamp
  after an intentional re-port with `--update`.
- **Known web risk (W4):** open upstream `book-scaffold-astro#69` keeps the
  `web/src/pages/chapters/[...slug].astro` route shim load-bearing. See `web/UPSTREAM_ISSUES.md`.

## Verified Baseline (live, 2026-06-14; temporalcv 2.2.0; v1.1.4 pending)

dml_ts is a thin causal consumer of temporalcv, now locked at **2.2.0**
(splitters/HAC/stationarity consumed upstream, golden-snapshot gated; the Track B
migration onto temporalcv v2.0.0 landed in 1.1.0 — see CHANGELOG 1.0.0/1.1.0 for the
inference fixes and the purged_cv leakage correction). The 2.2.0 lock is **golden-neutral
for dml_ts** (see "temporalcv 2.2.0 consumption" below). Environment
`pip install -e ".[dev,docs,full]"`, Python 3.13. The v1.1.4 delta (the global doctest
gate, two new examples, full validation API docs, and the `RuntimeError`->`ValueError`
accessor change) re-ran ruff / mypy / tier1+2 / examples / doctest / Sphinx live on
2026-06-14; the full-suite, book, and web rows carry from the 2026-06-13 baseline
(untouched by v1.1.4's docstring/docs-only changes):

| Gate | Command | Result |
| --- | --- | --- |
| Test collection | `pytest --collect-only -q` | **851** collected |
| Tier 1 (unit) | `pytest -m tier1` | **349** passed |
| Tier 1 + Tier 2 (dev lane) | `pytest -m "tier1 or tier2"` | **666** passed, 185 deselected |
| Full suite (non-network) | `pytest -m "not network"` | **810** passed, 41 net deselected (incl. tier3/tier4 econml MC + stress)* |
| Coverage (tier1+2) | `pytest -m "tier1 or tier2" --cov=dml_ts` | **75%** (gate `fail_under=70`) |
| Ruff check + format | `ruff check` / `ruff format --check` | pass (94 files) |
| Mypy | `mypy dml_ts/ --ignore-missing-imports --no-strict-optional --explicit-package-bases` | **0 issues, 41 source files** |
| Examples | `for f in examples/*.py; do … "$f"; done` | **8/8** execute |
| Doctest | `pytest --doctest-modules dml_ts/` | **50** passed, 19 `+SKIP` (~15s) |
| Sphinx | `sphinx -b html -W --keep-going docs/sphinx` | build succeeded; validation API now **18/18** modules |
| Book | `make` (LuaLaTeX×4 + biber) | **209 pages**, 0 fatal |
| Web | `npm run validate && npm run build` | validate ✓ 10 chapters (academic); build ✓ 15 pages |
| Drift guard | `scripts/check_tex_mdx_drift.py` | exit 0 (all ported chapters match their `.tex`) |

\* The tier3/tier4 validation + stress lane (167 + 40 tests) requires the `[full]` extra
(EconML) and runs real Monte-Carlo estimation; the full non-network run took **1h31m**
(810 passed, 0 failures) and also runs in `nightly.yml` CI. Without EconML these tests
raise `ModuleNotFoundError` rather than skip-gating (a pre-existing test-quality nit, not a
logic failure).

### temporalcv 2.2.0 consumption

temporalcv 2.2.0 (locked here; shipping as dml_ts 1.1.3) fixed a `CombinatorialPurgedCV`
embargo-leakage bug — the embargo is now one-sided per contiguous test run (temporalcv #38)
— deprecated `PurgedKFold(shuffle=True)` (#39), and tightened
`estimate_purge_gap`/`compute_label_overlap` validation (#40). It is **golden-neutral for
dml_ts**, which sets `embargo_pct=0` on the forward-only `PurgedWalkForward` everywhere
(`_create_cv`, the cv factory, and the golden snapshots): the 71/71 golden-parity snapshots
and the full tier1+2 suite (666) are byte-identical against 2.2.0, verified pre-release. See
the temporalcv CHANGELOG 2.2.0.

CI gates the full stack: `tests.yml` (ruff + mypy + collect + tier1 + tier1/2 +
an examples job), `book.yml` (LuaLaTeX+biber), `docs.yml` (Sphinx `-W` + Pages),
`nightly.yml` (tier3/tier4 + coverage), `deploy-web.yml` (web validate/build/deploy).

The 2026-05-02 pre/post-remediation baselines (796/314/615 → 802/320/621) are
recorded in the superseded audit; the numbers above are the current live values.

## Roadmap

Three tracks. Track 3 stays deferred until tracks 1–2 are comfortable.

### Track 1 — Truth & gates tail — RECONCILED 2026-05-30

Closed in the 2026-05-30 reconciliation (see the audit's Reconciliation section):
F14 (`results/` frozen in `.gitignore`), F5 (leakage regression test added), F15
(coverage gate `30 → 70`), R1 (toolchain pinned to mypy 2.1.0 / Python 3.13).
Landed in **1.1.4**: a global `pytest --doctest-modules dml_ts/` doctest gate (dedicated
CI `doctest` job; 50 runnable examples, 19 `+SKIP`), full Sphinx API coverage (18/18
validation modules), and two new runnable examples (`rolling_window`, `panel`).

### Track 2 — Web pilot

- ~~Port chapters 2–10 from LaTeX to MDX.~~ **Done 2026-06-01** — all 10 chapters hand-ported
  (validate ✓ 10 chapters, build ✓ 15 pages, drift guard exit 0, 7 W3 tier1 tests pass).
- ~~Add an *enforcing* LaTeX↔MDX drift guard (W3).~~ **Done 2026-05-30** —
  `scripts/check_tex_mdx_drift.py` fails when a `chapters/*.tex` changes without its MDX
  `source_sha256` being re-stamped (`--update`); enforced in pre-commit + the CI lint job.
- Resolve or keep the `#69` route shim per upstream; keep the scaffold current.

### Track 3 — Deferred methodology (post-gates)

- ~~True Lewis-Syrgkanis recursive dynamic g-estimation (separate, honestly-named class).~~
  **Done 2026-06-02, merged to `main` 2026-06-04 (PR #4, `6ef1519`)** — `DynamicGEstimationDML` recovers period-specific blips for a linear
  SNMM via recursive peeling (panel + single-series), with a joint coupled-stage sandwich
  variance and a gated EconML `DynamicDML` cross-check; validated against a known-blip DGP
  (`DynamicTreatmentDGP`). Heterogeneous `theta_t(X)` remains future work.
- ~~Public package namespace migration away from `src.*`.~~ **Done 2026-06-01** — code
  installs and imports as the `dml_ts` package (`from dml_ts import TemporalPLRDML`); the
  four headline estimators are hoisted to the package root, loaders/DGPs/sensitivity/
  production stay namespaced under their subpackages.
- Optional EconML heterogeneity examples as runnable, tested features.
- Blocking stationarity, cointegration, and overlap gates (currently diagnostics only).
- Book content expansion (see `docs/plans/active/2026-03-01_22-00_content_expansion.md`,
  ~180 → ~285 pages) — deferred until tracks 1–2 are stable.

## Canonical References

- Current audits: the November 2026 folders listed in `docs/audits/README.md`
- Data sources, licences and held files: `docs/audits/DATA_SOURCES.md`
- Superseded audits: `docs/audits/2026-05-30_repo_quality_methodology_audit.md`,
  `docs/audits/2026-05-02_repo_quality_methodology_audit.md`
- Audit index: `docs/audits/README.md`
- Historical status/reports: `docs/archive/superseded_2026-05-02/`
- Active plans: `docs/plans/active/`
- Web companion guide: `web/CLAUDE.md`, `web/UPSTREAM_ISSUES.md`

## Working Rules

- Runtime behavior, tests, examples, build logs, and primary literature outrank old
  roadmaps or status files.
- Public snippets in README, Sphinx, examples, and chapters must either execute under
  documented dependencies or be clearly labeled as conceptual.
- The current temporal estimator is scalar temporal PLR DML, not recursive dynamic
  g-estimation.
- "Production" language means book-companion production concepts unless code is
  explicitly tested as deployable software.
- LaTeX `chapters/*.tex` is canonical book content; the `web/` MDX is a parallel port
  and must be reconciled against the LaTeX before being treated as authoritative.
