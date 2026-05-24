# Upstream issues filed during the double_ml_time_series pilot

Running log of friction surfaced while porting LaTeX chapters to the Astro `book-scaffold-astro` academic style. Per [the pilot plan](https://github.com/brandon-behring/double_ml_time_series/issues/1) (R2.Q1 + R3.Q2), each entry follows the inline-upstream-PR loop:

1. Stop the port on friction.
2. File issue at `brandon-behring/book-scaffold-astro` with receipt (LaTeX snippet + scaffold gap + proposed API).
3. Implement PR upstream; bump scaffold version; update `CHANGELOG.md`.
4. Update `web/package.json` to consume the fix (typically via `file:` dependency until npm publish).
5. Append entry below with issue link, PR link, version bumped, and the original receipt.
6. Resume the port; delete any consumer-side workaround the upstream fix superseded.

## Status — 2026-05-22

All filed issues are **RESOLVED**. The consumer is on the published `@brandon_m_behring/book-scaffold-astro@^4.2.0` from npm (no more `file:` dependency).

The v4.0.0 release introduced a breaking `preset:` → `styles:[…]` API migration; this consumer migrated in the same change that promoted the dependency from `file:` to npm. See `MIGRATION-v3-to-v4.md` in the scaffold for the 2-line shape.

---

## Closed



### Issue #20 — `book-scaffold validate` ignores `.env BOOK_PROFILE`; defaults to minimal

**Status:** **RESOLVED** in v3.5.2.

**Surfaced during:** Commit 2 (bootstrap + bib smoke test) — running `npm run validate` after `npm install` and `npm run build:bib`.

**Receipt:** With `.env` containing `BOOK_PROFILE=academic` and explicit `profile: 'academic'` in both `astro.config.mjs` and `src/content.config.ts`, `npm run validate` reported `profile=minimal` and silently passed (skipping academic-only `<Cite key=...>` validation). The scaffold's own demo chapter `week01-hello-world.mdx:29` cites a non-existent `example-key2024` — which the minimal profile silently accepts. Setting `BOOK_PROFILE=academic` in the npm script env makes the CLI correctly surface the broken cite.

**Root cause:** `package/scripts/validate.mjs:71` rolled its own preset resolution chain that skipped the `.env` fallback that `resolvePreset()` in `package/src/types.ts` already implements. SKILL.md and `types.ts:126-131` both promise `.env` is read; the CLI didn't deliver.

**Severity:** `kind:bug` + `kind:doc-drift`. Silent fallback to minimal is worse than a hard error because academic-profile errors disappear from CI without warning.

**Links:**
- Issue: https://github.com/brandon-behring/book-scaffold-astro/issues/20
- PR: https://github.com/brandon-behring/book-scaffold-astro/pull/21
- Branch: `fix/validate-env-reading` (off `v3.0`)
- Commit (scaffold): `0ce1bc5`
- Version: 3.5.1 → 3.5.2 (patch)

**Consumer-side state:**
- `web/package.json` consumes the fix via `file:../../book-scaffold-astro/package` while the PR is in review. Switch back to a registry version (`^3.5.2`) once the PR merges and 3.5.2 publishes to npm.

**Verification (post-fix, consumer-side):**
```
$ npm run validate
> book-scaffold validate
validate: ✗ 1 error(s) in 1 chapter(s) (profile=academic):
  week01-hello-world.mdx:29  Unknown bibkey "example-key2024" — not in references.json
```

The remaining error is the scaffolded demo's broken cite, which is expected to be deleted when Chapter 1 is ported in Commit 3.

**Tests added upstream:**
- `package/tests/validate-root.test.mjs`: `.env BOOK_PROFILE is honored when no env or flag is set (closes #20)`.
- `package/tests/validate-root.test.mjs`: `BOOK_PROFILE env still wins over .env (closes #20)` (priority preserved).

---

### Issue #22 — `defineBookConfig` doesn't accept consumer KaTeX macros

**Status:** **RESOLVED** in v3.6.0; shipped on npm.

**Surfaced during:** Commit 3, before the first line of Ch1 MDX was written. The scaffold's strict KaTeX mode (`strict: 'error'`) means a build referencing `\Var` or `\Cov` fails — and `ssmMacros` (the only macro set the scaffold ships) is SSM-focused, not causal-inference.

**Receipt:** Chapter 1 of the manuscript uses `\Var(\cdot)` and `\Cov(\cdot, \cdot)` in dense math (variance reduction theorem, residualized covariance, orthogonal decomposition). Defined in `shared/dml-preamble-tufte.sty:410-411` for LaTeX. The scaffold's `defineBookConfig` (`package/src/config.ts:45`) hardcoded `macros: ssmMacros` with no consumer extension point. `markdown.rehypePlugins` wasn't a viable workaround: it would have meant reconstructing the whole rehype-katex registration internally.

**Root cause:** Missing extension point in `BookConfigOptions`. Every non-SSM academic book hits this (Bertsekas RL would want `\argmin`, `\bellman`; calculus-of-variations would want `\Lag`, `\ham`).

**Severity:** `kind:api-friction`. Not a bug (the scaffold works as designed) but a missing API every non-SSM academic consumer trips over.

**Links:**
- Issue: https://github.com/brandon-behring/book-scaffold-astro/issues/22
- PR: https://github.com/brandon-behring/book-scaffold-astro/pull/23 (stacked on PR #21)
- Branch: `feat/katex-macros-option` (off `fix/validate-env-reading`, which is off `v3.0`)
- Commit (scaffold): `a281a77`
- Version: 3.5.2 → 3.6.0 (minor — additive API)

**Consumer-side state:**
- `web/astro.config.mjs` uses the `katexMacros` option to add `\Var → \mathrm{Var}` and `\Cov → \mathrm{Cov}`.
- `web/package.json` consumes `@brandon_m_behring/book-scaffold-astro@^4.2.0` from npm (was `file:` during PR review).

**Verification (post-fix, consumer-side):**
```
$ cd web && npm run dev
[...]
 astro  v6.3.6 ready in 1892 ms
┃ Local    http://localhost:4321/
```

Astro dev server starts cleanly with `katexMacros` wired in.

**Tests added upstream:**
- `package/tests/katex-macros.test.mjs` (4 tests):
  - omitting `katexMacros` yields exactly `ssmMacros` (backward compat)
  - consumer macros merged onto `ssmMacros` (the closes-#22 path)
  - consumer wins on key collision (override semantics)
  - tools profile does NOT register rehype-katex even when `katexMacros` is supplied (no leak)

**Follow-up signalled upstream:** Extend `ssmMacros` to a registry of named presets (`causalMacros`, `dynamicalSystemsMacros`, etc.) the scaffold ships out of the box, composable by consumers. Out of scope for #22; flagged in the issue.

---

### Issue #24 — `routes.chapters=true` crashes on academic profile

**Status:** **RESOLVED** in v3.7.0 via per-profile `chaptersRenderer` dispatch (academic / tools / fallback). Visual-regression baseline `fixture-academic-chapters` (5 chapters across 4 parts, 4 viewports = 16 PNGs) passes at AE=0.

**Surfaced during:** Commit 3, after Chapter 1 MDX was written + Astro build was first attempted. The chapter rendered correctly via the `/print` aggregate page, but enabling `routes: { chapters: true }` to get per-chapter routing crashed the build.

**Receipt:** `package/pages/chapters.astro` is hardcoded to the tools-profile schema. It reads `c.data.volatility`, `c.data.tools_compared`, `c.data.last_verified`, `c.data.chapter` — none of which exist on academic chapters (which use `week`/`part-enum`/no-volatility). When the page tries to iterate academic chapters, `getFreshness(undefined, undefined)` returns null and `.status` throws.

```
generating static routes
├─ /chapters/index.html [ERROR] TypeError: Cannot read properties of null (reading 'status')
    at .../chapters_Bgpge6sw.mjs:193:39
```

**Severity:** `kind:bug` + `kind:api-friction`. The documented option crashes; the academic profile has no working chapter listing.

**Links:**
- Issue: https://github.com/brandon-behring/book-scaffold-astro/issues/24
- Resolved in scaffold v3.7.0 (per-profile `chaptersRenderer`).

**Consumer-side state:**
- `web/astro.config.mjs` now sets `routes: { chapters: true }`. Per-chapter URLs (e.g., `/chapters/01-fwl-potential-outcomes/`) are the canonical access pattern; `/print` remains available as the aggregate route.
- The earlier workaround that pointed Cloudflare deploys at `/print/` is removed.

**Tests added upstream:** visual-regression `fixture-academic-chapters` (16 baseline PNGs at AE=0).

---
