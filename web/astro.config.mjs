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
 */
import { defineBookConfig } from '@brandon_m_behring/book-scaffold-astro';

export default await defineBookConfig({
  site: 'https://double-ml-time-series.brandon-m-behring.workers.dev',
  profile: 'academic',
});
