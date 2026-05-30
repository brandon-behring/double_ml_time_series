# Repo Quality And Methodology Audit

Date: 2026-05-30

Repo: `double_ml_time_series`

Supersedes: `docs/audits/2026-05-02_repo_quality_methodology_audit.md` (kept in place
as historical evidence; its path is referenced by web Ch1 provenance).

Audit standard: book-grade companion repo, not deployable production software.
Executable behavior, live signatures, tests, logs, and primary methodology sources
outrank README/docs/roadmaps. Every gate in the verification table below was re-run
live in this pass.

## Executive Verdict

Verdict: **healthy book-companion; one source-of-truth gap fixed in this pass.**

The 2026-05-02 truth-first remediation worked. The methodology layer is now honest:
the headline estimator is named and documented as scalar temporal PLR DML, the false
dynamic-g-estimation / CausalForest / production-grade claims are gone, the
time-series leakage path is fixed, and the build/test/docs toolchains agree and are
gated in CI. Of the 16 findings in the prior audit, **13 are CLOSED, 1 is accepted
within scope (F10), and 2 remain open as known/deferred cleanup (F14, F15-coverage)**.

The new problem is not methodology — it is that the canonical status surface went
stale. Since 2026-05-02, ~20 of the last ~23 commits built a **`web/` Astro book
companion** (deployed to Cloudflare at `dml.brandon-behring.dev`), and **no status
document mentioned it**. This audit records that workstream (W1–W5) and the refresh
of `docs/CURRENT_STATUS.md` closes the source-of-truth gap (W1).

The repo is in materially better shape than the prior audit's "partial restart"
framing. No restart is warranted; the remaining work is a finite cleanup tail plus
the web-port pilot.

## Live Verification Summary

All commands run 2026-05-30 from repo root. Python via repo venv (Python 3.13.5).

| Gate | Command | Result |
|---|---|---|
| Test collection | `pytest --collect-only -q` | **802 collected** |
| Tier 1 | `pytest -m tier1 --no-cov -q` | **320 passed**, 482 deselected, 3.73s |
| Tier 1 + Tier 2 | `pytest -m "tier1 or tier2" --no-cov -q` | **621 passed**, 181 deselected, 23 warnings, 61.6s |
| Black | `black --check src/ test/ examples/` | Pass, 82 files unchanged |
| Mypy | documented permissive command | **Pass under the pinned toolchain** (pre-commit `mirrors-mypy` v1.7.0 + CI `.[dev]`), confirmed via `pre-commit run mypy`. A locally-drifted mypy 2.1.0 reports `src/data/fred_loader.py:555` `type: ignore` as unused — local skew, see R1. |
| Examples | run all `examples/*.py` | **5/5 execute** (fwl_to_dml, insurance_dgp, production_pipeline, sensitivity, time_series_dml) |
| Sphinx | `sphinx -b html -W --keep-going docs/sphinx ...` | **build succeeded** (Sphinx 9.1.0) |
| Book | `make` after removing `main.pdf` (forced clean rebuild) | **205 pages, 0 fatal errors**, 257 overfull / 12 underfull hboxes (LuaTeX 1.22.0) |
| Web validate | `cd web && npm run validate` | **✓ 0 errors** (profile=academic, 1 chapter, 34 bib entries, 15 ids) |
| Web build | `cd web && npm run build` | **✓** astro build + pagefind, 6 pages / 2069 words indexed |

Toolchain present and working: LuaHBTeX 1.22.0, biber 2.21, Sphinx 9.1.0, Node
v22.20.0, npm 11.7.0. Sphinx and web `node_modules` are installed locally.

### CI inventory (this is a strength, and corrects prior F8/F9)

`.github/workflows/` now gates the full stack:

| Workflow | Gates |
|---|---|
| `tests.yml` | black + mypy + collect (lint job), tier1, tier1+2 with `--cov-fail-under=30`, **and an `examples:` job that runs every `examples/*.py`** |
| `book.yml` | clean LuaLaTeX+biber build via `make`; reports overfull/underfull boxes |
| `docs.yml` | Sphinx `-W --keep-going` build + GitHub Pages deploy |
| `nightly.yml` | tier3 daily, tier4 weekly, coverage |
| `deploy-web.yml` | web `validate` + `build` + Cloudflare deploy (prod + PR preview) |

## Disposition Of 2026-05-02 Findings (F1–F16)

| ID | Prior severity | Status | Current evidence |
|---|---|---|---|
| F1 | Blocker | **CLOSED** | Class renamed `TemporalPLRDML`; `temporal_plr_dml.py:5-6` explicitly disavows Lewis-Syrgkanis; result is scalar `theta` (`:547,583`). |
| F2 | Blocker | **CLOSED** | README quickstarts use live API; public-snippet contract tests are in the suite (collection rose 796→802). |
| F3 | Blocker | **CLOSED** | Sphinx builds under `-W`; `docs.yml` gates it on every PR/push. |
| F4 | Blocker | **CLOSED** (but drifted → W1) | `docs/CURRENT_STATUS.md` canonical; 25 docs archived under `superseded_2026-05-02`. Gap: it omitted `web/` until this pass. |
| F5 | Critical | **CLOSED in code** | No future backfill: `temporal_plr_dml.py:287-289` leaves early rows `NaN`; `:494-507` drop them via `valid_oof_mask`. Residual: an explicit *leakage regression test* is not confirmed (roadmap). |
| F6 | Critical | **CLOSED** | Cross-implementation comparison claim removed from public surface. |
| F7 | Critical | **CLOSED** | CausalForest/BLP/policy-tree reclassified deferred; absent anywhere in `src/`. |
| F8 | Critical | **CLOSED** | `Makefile:12-13` uses `lualatex`+`biber`; `book.yml` gates a clean book build. (Prior "no book CI" no longer holds.) |
| F9 | Critical | **CLOSED** | `docs.yml` builds Sphinx with `-W` and deploys Pages; build verified locally. |
| F10 | High | **ACCEPTED (in scope)** | `src/production/` relabeled research/demo in docs; it still implements its own PLR logic rather than delegating. Acceptable for a book companion; not deployment software. |
| F11 | High | **CLOSED** | Docs state stationarity/cointegration/overlap are diagnostics with manual enforcement; code confirms (no blocking gates). |
| F12 | High | **CLOSED** | `examples/example_time_series_dml.py:67,70` Part 3 now labeled "TemporalPLRDML" and uses `TemporalPLRDML`, not shuffled `double_ml`. |
| F13 | High | **CLOSED** | `README.md:40,46` use `pip install -e ".[dev]"` / `".[dev,docs]"`. |
| F14 | Medium | **OPEN (deferred)** | `git ls-files results/` = **11** tracked artifacts. `main.bbl` is no longer tracked (that half closed). Freeze-or-untrack decision still pending. |
| F15 | Medium | **PARTIAL** | Tier counts now current; snippet contract tests exist. Coverage gate remains `fail_under = 30` (`pyproject.toml:130`, enforced in `tests.yml`/`nightly.yml`) by design. |
| F16 | Medium | **CLOSED** | Docs use `venv/bin/python -m ...`; CI installs the package and uses `python -m ...`. |

### New residual finding

| ID | Severity | Finding | Fix |
|---|---|---|---|
| R1 | Info | **Local toolchain drift, not a repo defect.** The repo pins `mypy>=1.7`; the pre-commit hook (`mirrors-mypy` v1.7.0) and CI (`.[dev]`) both *require* the `# type: ignore[misc]` on `fred_loader.py:555` (a `[misc]` "slice index" error on `df.loc[start_date:end_date]`) and pass. A locally-drifted mypy **2.1.0** (venv is Python 3.13.5 vs the pinned 3.11) reports that ignore as unused under `warn_unused_ignores`. | **No code change** — removing the ignore breaks the pinned pre-commit/CI gate (verified: the commit hook rejected exactly that). Align the local venv to the pinned toolchain instead (rebuild via `uv pip install -e ".[dev,docs]"`). |

## Web Companion Findings (W1–W5)

The `web/` directory is a `@brandon_m_behring/book-scaffold-astro@^4.8.0` academic
consumer, deployed to Cloudflare Workers at `dml.brandon-behring.dev` (pilot tracked
in GitHub issue #1).

| ID | Severity | Finding | Disposition |
|---|---|---|---|
| W1 | Blocker (source-of-truth) | `docs/CURRENT_STATUS.md` and root `CLAUDE.md` never mentioned `web/`, the most active workstream. | **Fixed this pass** — `CURRENT_STATUS.md` gains a Web companion section; `CLAUDE.md` gains a `web/` pointer. |
| W2 | Status | Chapter port coverage is **1 of 10**: only `web/src/content/chapters/chapter01-fwl-potential-outcomes.mdx` exists; LaTeX has `chapters/chapter_01..10.tex`. | Expected — pilot. Tracked in roadmap track 2. |
| W3 | Medium (latent process) | LaTeX and MDX chapter bodies are independent hand-maintained copies with **no sync guard**; provenance frontmatter records an audit *date*, not a source hash. Only `bibliography.bib` is single-source (`BOOK_BIB_PATH=../bibliography.bib`). | **Parity verified now** (below): no current divergence. Risk is future drift; a guard is a roadmap item. |
| W4 | Medium | Open upstream `book-scaffold-astro#69`: the academic `/chapters/` index links to per-chapter routes the scaffold does not ship, so the consumer-side shim `web/src/pages/chapters/[...slug].astro` is load-bearing. | Logged in `web/UPSTREAM_ISSUES.md`; keep shim until upstream ships the route. |
| W5 | Low | Prior audit implied weak CI; in reality CI is comprehensive (table above). Residual: no dedicated doctest job for README/Sphinx fenced snippets beyond the in-suite contract tests; `web/` has only `validate` (no component tests). | Minor; optional roadmap hardening. |

### W3 parity check — Chapter 1 LaTeX vs MDX (verified 2026-05-30)

Structural diff of `chapters/chapter_01.tex` (1605 lines) vs
`web/src/content/chapters/chapter01-fwl-potential-outcomes.mdx` (1352 lines):

- **Section coverage is 1:1.** Every LaTeX `\section`/`\subsection` maps to an
  identical MDX `##`/`###` heading, in the same order: Introduction → Potential
  Outcomes → Fundamental Problem → ATE → Identification (CIA, Overlap, Result) →
  Insurance deep-dive → Healthcare example → Understanding Overlap → PS Python →
  FWL theorem (regression setup … computational) → FWL Python → Summary →
  Concluding Remarks (Roadmap) → Exercises (Conceptual, Mathematical, Computational).
- Component vs environment counts differ by convention, not content: MDX uses 20
  generic `<Theorem>` callouts; LaTeX uses 3 `theorem` + 3 `definition` + 1
  `example`. The line-count gap reflects LaTeX preamble/labels and exercise verbosity.
- **Verdict: the ported web Ch1 is faithful — no drift has occurred yet.** W3 is a
  latent process risk (no guard), not a realized inconsistency.

## Source-Of-Truth Map (updated)

| Surface | Status | Note |
|---|---|---|
| Runtime code in `src/` | Current | Live signatures/tests authoritative; docstrings now honest. |
| Tests under `test/` | Current, green | 802 collect; tier1 320, tier1+2 621 re-verified live. |
| `README.md` | Current | Live API, correct install path, honest scope. |
| `CLAUDE.md` (root) | Current after this pass | Gains a `web/` pointer; historical baseline numbers left as labeled history. |
| `docs/CURRENT_STATUS.md` | Current after this pass | Gains Web companion + Roadmap; repoints to this audit. |
| `docs/audits/2026-05-02_...md` | Superseded (kept in place) | Banner added; path preserved for web Ch1 provenance backlink. |
| `web/` Astro companion | Current, live, pilot | 1/10 chapters; scaffold ^4.8.0; open upstream #69. |
| `chapters/` LaTeX | Canonical book content | 10 chapters, 205-page PDF, 0 fatal errors. |
| CI workflows | Current, comprehensive | tests/book/docs/nightly/deploy-web all gate. |

## Bottom Line

Keep going as-is. The methodology contract is honest, the gates are green and
CI-enforced, and the only true documentation-vs-reality gap — silence about the web
companion — is closed by this pass. Remaining work is a small, well-scoped tail:
the `results/` artifact decision (F14), an explicit leakage regression test (F5
residual), and the web-port pilot with a drift guard (W2/W3). The mypy finding (R1)
is local toolchain skew, not a repo change. The consolidated roadmap lives in
`docs/CURRENT_STATUS.md`.
