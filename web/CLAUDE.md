# double-ml-time-series-web — AI authoring guide

Web companion to the Double Machine Learning for Time Series manuscript. Built with `@brandon_m_behring/book-scaffold-astro` (academic style, v4.2.0+ from npm).

**Project context:**

This is a `web/` subdirectory inside the LaTeX repo at `~/Claude/double_ml_time_series/`. The LaTeX manuscript (`main.tex` + `chapters/*.tex`) is canonical during the migration window; this Astro consumer builds in parallel. See the parent `CLAUDE.md` for the methodology rules of the host repo.

The pilot tracks GitHub issue [#1](https://github.com/brandon-behring/double_ml_time_series/issues/1).

**Where things live:**

- Chapters: `src/content/chapters/*.mdx` — frontmatter follows the academic schema
- Bibliography: pointed at `../bibliography.bib` via `BOOK_BIB_PATH` override in `package.json` (single source of truth; LaTeX and Astro read the same file)
- Cross-references: ids on `<Theorem>` / `<Figure>` → `src/data/labels.json` via `npm run build:labels`
- KaTeX macro bridge: `src/lib/dml-katex-macros.ts` (consumer-side; delete when upstream `causalMacros` preset ships)
- Upstream issue log: `UPSTREAM_ISSUES.md` (running log of friction surfaced during the port + scaffold PRs)

**Upstream-first protocol:** when porting LaTeX friction surfaces (missing macro, schema mismatch, component gap), pause the port, file the issue at `brandon-behring/book-scaffold-astro` with label `consumer:double-ml-time-series` + `kind:<enhancement|api-friction|bug>`, implement the PR upstream, bump scaffold version, update `package.json`, log in `UPSTREAM_ISSUES.md`, then resume. See `~/.claude/projects/-home-brandon-behring-Claude-double-ml-time-series/memory/book_scaffold_strategy.md`.

**Toolkit reference:** [PACKAGE_DESIGN.md](https://github.com/brandon-behring/book-scaffold-astro/blob/v3.0/PACKAGE_DESIGN.md) — single source of truth for the API.
