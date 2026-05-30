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

One new issue is open from the v4.2.0 deploy work; see Open section.

---

## Open

### Issue #69 — academic style: `/chapters/` index links to per-chapter routes the scaffold doesn't ship

**Status:** OPEN. Filed during Phase 1 of the Cloudflare deploy wiring.

**Surfaced during:** First post-migration `npm run build` against `^4.2.0` with `routes: { chapters: true }`. The scaffold-shipped `/chapters/` index rendered cleanly, but every chapter card linked to `/chapters/<slug>/` — a route the scaffold doesn't generate. All links 404.

**Root cause:** The scaffold ships `pages/chapters.astro` (the index, with the v3.7.0 `academicChaptersRenderer` fix from #24) but no `pages/chapters/[...slug].astro`. Compare with the `frontmatter` collection, which ships both halves: `pages/frontmatter/[...slug].astro` exists, providing dynamic per-slug routes for any frontmatter MDX the consumer drops.

**Severity:** `kind:api-friction`. The scaffold's own `CLAUDE.md` "Add a new chapter" section instructs authors to preview at `/chapters/<slug>/` — which only works if the consumer has already added the dynamic route. Every academic consumer hits this and writes the same shim.

**Consumer-side workaround (in place):**
- `web/src/pages/chapters/[...slug].astro` mirrors the canonical pattern from `post_transformers/guides/web/`: `getCollection('chapters', !draft)`, `<Chapter entry={entry} headings={headings}><Content /></Chapter>` using the scaffold's `Chapter` layout.

**Proposed fixes** (in the issue):
- (a) Ship `package/pages/chapters/[...slug].astro` in the scaffold, gated on `routes.chapters: true`.
- (b) Document the consumer responsibility explicitly (CLAUDE.md + a recipe) and have `book-scaffold validate` warn when the shim is missing.

**Links:**
- Issue: https://github.com/brandon-behring/book-scaffold-astro/issues/69
- Consumer-side workaround commit (this repo): `deae877` (`feat(web): migrate scaffold to v4.2.0 + add per-chapter route`).

---

### Provenance schema: `.strict()` rejects `source_file` / `source_sha256` drift keys

**Status:** OPEN — to file. Consumer-side resolution already in place; non-blocking.

**Surfaced during:** W3 drift-guard work (2026-05-30). The LaTeX↔MDX drift guard
(`scripts/check_tex_mdx_drift.py`) records, per ported chapter, the canonical `.tex`
path and its `sha256`. The Ch1 reconciliation first placed `source_file` +
`source_sha256` under the frontmatter `provenance:` block. `npm run validate` accepted
it, but `npm run build` (Astro content-sync Zod parse) failed:

```
[InvalidContentEntryDataError] provenance: Unrecognized keys: "source_file", "source_sha256"
```

**Root cause:** `provenanceObject` is declared `.strict()` (`package/src/schemas.ts`), so
any key outside `{ai_tools, prompts_archive, decisions_log, audit_history,
citation_backstop}` hard-fails the build. The outer `academicChapterSchema` is *not*
strict (it strips unknown top-level keys), so the friction is specific to the nested
`provenance` block. Net effect: there is no schema-blessed place to record a
source-of-truth hash for LaTeX→MDX drift tracking.

**Severity:** `kind:api-friction`. Recording which canonical file an MDX was ported
from, at what hash, is exactly the "process-as-artifact" the v4.8.0 provenance feature
is for — yet the strict schema has no field for it.

**Consumer-side resolution (in place):** moved `source_file` + `source_sha256` to the
**top level** of the Ch1 frontmatter (Astro strips unknown top-level keys; `build`
passes). The drift guard reads them by frontmatter regex, so nesting is irrelevant to
enforcement. An inline comment in the MDX documents why; future ported chapters follow
the same top-level placement.

**Proposed upstream fix:** add an optional `source` object to `provenanceObject`
(e.g. `source: { file: string, sha256: string }`), or relax provenance to passthrough,
so consumers can record port provenance in the semantically correct block — and have
`Provenance.astro` render a small "ported from `<file>` @ `<sha8>`" line.

**Links:**
- Issue: _to file_ at `brandon-behring/book-scaffold-astro` (`consumer:double-ml-time-series` + `kind:api-friction`).
- Consumer resolution commit (this repo): _pending_ — W3 drift-guard pass.

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
