/**
 * src/content.config.ts — Content collections for double-ml-time-series-web.
 *
 * Explicit profile (same reasoning as astro.config.mjs): .env is gitignored,
 * so the env-driven fallback inside resolveProfile picks 'minimal' in
 * Cloudflare's build container, which dispatches to the tools schema —
 * invalid for this academic book. Hardcoding makes the deploy deterministic.
 */
import { defineBookSchemas } from '@brandon_m_behring/book-scaffold-astro/schemas';

export const { collections } = defineBookSchemas({ profile: 'academic' });
