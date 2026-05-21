# Upstream issues filed during the double_ml_time_series pilot

Running log of friction surfaced while porting LaTeX chapters to the Astro `book-scaffold-astro` academic profile. Per [the pilot plan](https://github.com/brandon-behring/double_ml_time_series/issues/1) (R2.Q1 + R3.Q2), each entry follows the inline-upstream-PR loop:

1. Stop the port on friction.
2. File issue at `brandon-behring/book-scaffold-astro` with receipt (LaTeX snippet + scaffold gap + proposed API).
3. Implement PR upstream; bump scaffold version; update `CHANGELOG.md`.
4. Update `web/package.json` to consume the fix (typically via `file:` dependency until npm publish).
5. Append entry below with issue link, PR link, version bumped, and the original receipt.
6. Resume the port; delete any consumer-side workaround the upstream fix superseded.

---

## Issue #20 — `book-scaffold validate` ignores `.env BOOK_PROFILE`; defaults to minimal

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
