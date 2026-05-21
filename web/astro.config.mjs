// @ts-check
/**
 * astro.config.mjs — double-ml-time-series-web, a consumer of
 * @brandon_m_behring/book-scaffold-astro (academic profile).
 *
 * Explicit profile: .env is gitignored, so the env-driven fallback inside
 * resolveProfile picks 'minimal' in Cloudflare's build container, which
 * dispatches to the tools schema — invalid for this academic book.
 * Hardcoding makes the deploy deterministic. (Same reasoning as
 * post_transformers/guides/web/astro.config.mjs.)
 *
 * katexMacros: causal/DML notation the LaTeX preamble defines via
 * \newcommand. Merges on top of the scaffold's ssmMacros (closes
 * brandon-behring/book-scaffold-astro#22). Per the upstream-first pilot
 * strategy, the list grows as the port surfaces real uses; do not
 * speculatively add macros from the LaTeX preamble that no MDX uses yet.
 */
import { defineBookConfig } from '@brandon_m_behring/book-scaffold-astro';

export default await defineBookConfig({
  site: 'https://double-ml-time-series.brandon-m-behring.workers.dev',
  profile: 'academic',
  // Note on `routes.chapters`: the academic profile leaves it off by
  // default. Enabling it surfaces the upstream bug filed at
  // brandon-behring/book-scaffold-astro#24 — the shipped pages/chapters.astro
  // is hardcoded to the tools schema (volatility, tools_compared,
  // last_verified, chapter) and crashes on academic frontmatter. Use
  // /print as the chapter access point until #24 ships an academic
  // chapter-listing page (or until this book ships its own).
  katexMacros: {
    // From shared/dml-preamble-tufte.sty:410-411. Used in Chapter 1
    // (variance/covariance of estimators, residualized treatment).
    '\\Var': '\\mathrm{Var}',
    '\\Cov': '\\mathrm{Cov}',
  },
});
