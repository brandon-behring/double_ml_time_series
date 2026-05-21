# web

Built with [`@brandon_m_behring/book-scaffold-astro`](https://github.com/brandon-behring/book-scaffold-astro) (academic profile, v3.5.0).

## Getting started

```bash
npm install
npm run dev    # http://localhost:4321
```

## Authoring

Chapters live under `src/content/chapters/*.mdx`. The starter `week01-hello-world.mdx` shows the frontmatter shape and basic component usage.

Available components are documented in the toolkit's [PACKAGE_DESIGN.md §10](https://github.com/brandon-behring/book-scaffold-astro/blob/v3.0/PACKAGE_DESIGN.md#10-mdx-import-patterns).

## Build + deploy

```bash
npm run validate    # pre-flight content checks
npm run build       # → dist/
npx wrangler deploy # Cloudflare Workers + Static Assets
```

See `wrangler.toml` for deploy config.
