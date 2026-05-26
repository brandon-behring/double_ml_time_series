// @ts-check
/**
 * astro.config.mjs — double-ml-time-series-web, a consumer of
 * @brandon_m_behring/book-scaffold-astro (academic style, v4.2.0+).
 *
 * styles: explicit v4 composition (was profile:'academic' in v3.x).
 * Hardcoded — Cloudflare's build container does not inherit .env, so the
 * env-driven fallback would pick 'minimal' and dispatch to the tools schema,
 * invalid for this academic book.
 *
 * routes.chapters: enabled. The v3.7.0 release shipped a schema-aware
 * academicChaptersRenderer that closes upstream issue #24 (visual baseline
 * AE=0); the old workaround of relying on /print is no longer needed.
 *
 * site: Pass 1 of the two-pass staging — points at the new person-prefixed
 * Worker subdomain so the first deploy validates against a URL that exists
 * at push-time. Pass 2 (after the dml.brandon-behring.dev custom domain is
 * bound in the Cloudflare dashboard) flips this to https://dml.brandon-behring.dev.
 *
 * katexMacros: causal/DML notation the LaTeX preamble defines via
 * \newcommand. Merges on top of the scaffold's ssmMacros (consumer API
 * shipped in v3.6.0, closes #22). Per the upstream-first pilot strategy,
 * the list grows as the port surfaces real uses; do not speculatively add
 * macros from the LaTeX preamble that no MDX uses yet.
 */
import { defineBookConfig, academicStyle } from '@brandon_m_behring/book-scaffold-astro';

export default await defineBookConfig({
  site: 'https://brandon-behring-double-ml-time-series.brandon-m-behring.workers.dev',
  // v4.5.0: title + description feed the auto-injected `/` landing's H1 + lead.
  // Voice matches the projects.json summary on brandon-behring.dev so the
  // landing reads consistently with the portfolio entry. Portfolio backlink
  // (footer "Part of brandon-behring.dev") is inherited from the scaffold's
  // BRANDON_PORTFOLIO_DEFAULT — no `portfolio:` override needed here.
  title: 'Double Machine Learning for Time Series',
  description: 'Companion code + manuscript-style material for temporal partially-linear DML, cross-fitting, HAC inference, and synthetic examples.',
  styles: [academicStyle],
  output: 'static',
  routes: { chapters: true },
  katexMacros: {
    // From shared/dml-preamble-tufte.sty:410-411. Used in Chapter 1
    // (variance/covariance of estimators, residualized treatment).
    '\\Var': '\\mathrm{Var}',
    '\\Cov': '\\mathrm{Cov}',
  },
});
