# Current Status

Last updated: 2026-05-30

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
- HAC, time-series splitters, stationarity diagnostics, synthetic DGPs, test suite.
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
- **Port coverage: 1 of 10 chapters.** Only Chapter 1 (Potential Outcomes + FWL) is
  ported. A 2026-05-30 parity check confirmed the web Ch1 is structurally faithful to
  `chapters/chapter_01.tex` (1:1 section coverage) — no drift yet.
- **Known web risks:** chapter content has no automated LaTeX↔MDX drift guard (W3);
  open upstream `book-scaffold-astro#69` keeps the `web/src/pages/chapters/[...slug].astro`
  route shim load-bearing (W4). See `web/UPSTREAM_ISSUES.md`.

## Verified Baseline (live, 2026-05-30)

Every gate below was re-run in this pass (Python via repo venv, 3.13.5):

- Test collection: **802 collected**.
- Tier 1: **320 passed**, 482 deselected.
- Tier 1 + Tier 2: **621 passed**, 181 deselected, 23 warnings.
- Black: pass, 82 files unchanged.
- Mypy: **pass** under the pinned toolchain (pre-commit `mirrors-mypy` v1.7.0 + CI
  `.[dev]`). A locally-drifted mypy 2.1.0 flags the `fred_loader.py:555` `type: ignore`
  as unused — finding R1, **local venv drift, not a repo defect; no code change**.
- Examples: all 5 `examples/*.py` execute.
- Sphinx: `-W --keep-going` build succeeded (Sphinx 9.1.0).
- Book: forced clean rebuild → **205 pages, 0 fatal errors**, 257 overfull /
  12 underfull boxes (LuaTeX 1.22.0). Boxes are non-blocking report items.
- Web: `npm run validate` ✓ 0 errors (academic profile); `npm run build` ✓.

CI gates the full stack: `tests.yml` (black + mypy + collect + tier1 + tier1/2 +
an examples job), `book.yml` (LuaLaTeX+biber), `docs.yml` (Sphinx `-W` + Pages),
`nightly.yml` (tier3/tier4 + coverage), `deploy-web.yml` (web validate/build/deploy).

The 2026-05-02 pre/post-remediation baselines (796/314/615 → 802/320/621) are
recorded in the superseded audit; the numbers above are the current live values.

## Roadmap

Three tracks. Track 3 stays deferred until tracks 1–2 are comfortable.

### Track 1 — Truth & gates tail (small, finite)

- **F14:** decide freeze-vs-untrack for the 11 tracked `results/*` artifacts
  (`git ls-files results/` = 11); `main.bbl` is already untracked.
- **F5 residual:** add an explicit leakage regression test asserting no temporal
  prediction is trained on future indices (the fix is in code; the test is missing).
- **F15:** decide whether to raise coverage above `fail_under = 30`; optionally add
  a doctest gate for README/Sphinx fenced snippets beyond the in-suite contract tests.

### Track 2 — Web pilot

- Port chapters 2–10 from LaTeX to MDX (currently 1/10).
- Add a LaTeX↔MDX drift guard (W3) — e.g. a source-hash in provenance frontmatter or
  a CI warning when a `chapters/*.tex` changes without its MDX.
- Resolve or keep the `#69` route shim per upstream; keep the scaffold current.

### Track 3 — Deferred methodology (post-gates)

- True Lewis-Syrgkanis recursive dynamic g-estimation (separate, honestly-named class).
- Public package namespace migration away from `src.*`.
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
