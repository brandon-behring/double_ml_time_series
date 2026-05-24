# web

Web companion to the Double Machine Learning for Time Series manuscript. Built with [`@brandon_m_behring/book-scaffold-astro`](https://github.com/brandon-behring/book-scaffold-astro) (academic style, v4.2.0+ from npm).

Canonical URL: **https://dml.brandon-behring.dev** (bound to the Cloudflare Worker `brandon-behring-double-ml-time-series`). Pre-binding, the same content also serves at `https://brandon-behring-double-ml-time-series.brandon-m-behring.workers.dev`.

## Getting started

```bash
npm install
npm run dev    # http://localhost:4321
```

## Authoring

Chapters live under `src/content/chapters/*.mdx`. `chapter01-fwl-potential-outcomes.mdx` is the working template for the academic schema (frontmatter shape, component usage, citation keys, anchor IDs).

Available components are documented in the toolkit's [PACKAGE_DESIGN.md](https://github.com/brandon-behring/book-scaffold-astro/blob/main/PACKAGE_DESIGN.md).

## Build + preview

```bash
npm run validate    # strict — fails on broken cite key, duplicate anchor, schema mismatch
npm run build       # → dist/
npm run preview     # serves dist/ on localhost:4321
```

## Deploy

Production and preview deploys are wired through `.github/workflows/deploy-web.yml` at the repo root:

- **Push to `main`** → production deploy via the reusable workflow at `brandon-behring/deploy-workflows`. Lands at `dml.brandon-behring.dev`.
- **Pull request** → preview deploy via `wrangler versions upload`; the workflow posts the preview URL as a PR comment.

Both jobs run `npm run validate` ahead of the build, so a broken `<Cite>` key or duplicate anchor fails CI.

For the one-time Cloudflare dashboard setup (project connect, repo secrets, custom-domain binding), see `~/Claude/brandon-behring.dev/docs/cloudflare-setup.md`. Worker config lives in `wrangler.jsonc`.
