# Current Status

Last updated: 2026-06-14

This file is the canonical project-status source. Historical reports and roadmaps
were moved to `docs/archive/superseded_2026-05-02/` during the 2026-05-02 truth-first
remediation. The current audit is
`docs/audits/2026-05-30_repo_quality_methodology_audit.md` (all gates re-run live).

## Status

The repository is a book-grade companion repo for Double Machine Learning pedagogy.
It is not a deployable production software project.

Current verdict: **healthy book companion.** The 2026-05-02 remediation closed the
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

- Current audit: `docs/audits/2026-05-30_repo_quality_methodology_audit.md`
- Superseded audit: `docs/audits/2026-05-02_repo_quality_methodology_audit.md`
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
