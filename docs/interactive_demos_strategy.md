# Interactive Demos for the DML Book — Strategy & Architecture

*Status: living doc. Demo #1 (the "confounding lens") is built; this maps the series and the
reusable structure behind it.*

## Vision
Each chapter of the DML book earns a small **interactive teaching figure** — readers manipulate the
mechanism, not just read it. All figures are **Tier-1 (no backend)**: the model/DGP runs either
*in the browser* (cheap closed-form: OLS, residualization, simulation) or *offline once* into a
committed JSON trace. Render with the scaffold's design tokens (Warm Tol), Preact islands, SVG +
canvas, full a11y and `prefers-reduced-motion`. Hosted on the academic `book-scaffold-astro` site
(dml.brandon-behring.dev); deployable as a static asset, $0 inference.

## Chapter → demo map
| Ch | Topic | Interactive figure |
|----|-------|--------------------|
| 1 | FWL / potential outcomes | "partial-out" reveal: residualize X out, watch the slope correct |
| **2** | **Neyman orthogonality + DML** | **the confounding lens — demo #1 (built)** |
| 3 | Validation framework | coverage/bias dashboard over the committed `results/method_comparison/` |
| 4 | Cross-sectional elasticity | demand curve: confounded vs DML elasticity |
| 5 | Temporal PLR DML | naïve-over-time vs DML; HAC bands |
| 6 | Panel + rolling window | rolling-window DML / temporal-CV split animation |
| 7 | FRED macro controls | real-data control selection |
| 8 | Insurance pricing | pricing / lapse sensitivity |
| 9 | Heterogeneity | CATE varying by X |
| 10 | Research pipeline | the DAG / pipeline |

## Architecture (repo-local kit → graduate to the scaffold)
- **Kernels** — generalize `src/components/dml-core.ts` into shared, framework-free causal/stats
  utilities: DGP builders, `olsSlope`/`olsFit`/`residualize`, sampling-distribution sim, metrics.
- **Primitives** — factor `src/components/demo/`: `Slider`, `ScatterFit`, `SamplingDistribution`
  (canvas), `StatCards`, `DemoFrame` — themed, a11y + reduced-motion, no D3.
- **Each chapter demo** = its DGP/data + a thin island composing primitives + the chapter MDX embed.
- **Graduation:** once 2–3 demos here prove the pattern, promote the generic kernels + primitives up
  into `@brandon_m_behring/book-scaffold-astro` so every book gets them (see upstream issues below).
  This repo is the proving ground.

## The proven pattern (from demo #1)
`in-browser DGP/compute → typed Preact island → SVG scatter + canvas dot-strip`, design tokens,
slider-release resampling for heavy panels, committed real numbers as anchors (e.g. the book's
naïve-29% / DML-100% coverage). Embed in a chapter MDX **body** only (the W3 LaTeX↔MDX drift guard
hashes the `.tex`, so body-only edits are safe; never touch the `.tex`).

## Demo #1 status + deferred polish
Built on `draft/dml-confounding-lens` (scratch page). Independent audit: math/seeding/perf pass.
Deferred polish (next round): dark-mode theme-aware canvas + redraw-on-toggle; deterministic + wider
jitter + lower alpha; per-cloud mean labels; reduced-motion fade-in; dedupe `mean`; "~95% (MC)" note.

## Proposed sequencing (you decide)
#1 polish → **Ch3 validation dashboard** (reuses the sampling-distribution panel over real
`method_comparison` data — cheapest #2) → temporal/panel (Ch5–6) → heterogeneity (Ch9) → applied
(Ch4/8). Adjust freely.

## Open decisions
Which chapters to prioritize; when to graduate the kit to the scaffold; per-chapter in-browser-compute
vs committed-trace.

## Upstream (book-scaffold-astro)
- [Issue #102](https://github.com/brandon-behring/book-scaffold-astro/issues/102) — fontsource dev-server SSR fix.
- [Issue #103](https://github.com/brandon-behring/book-scaffold-astro/issues/103) — native interactive-demo support (this doc is the linked design).
