# Cloudflare deployment — one-time setup

Walks through every manual click + command needed to take this repo from "5 local commits ready" → "Chapter 1 live at `https://dml.brandon-behring.dev`".

All work in this document is **dashboard + secrets work only**. The code, workflow, and `wrangler.jsonc` are already on `main`. Verified locally on 2026-05-22 (build green, 5 routes, KaTeX + bibliography render).

---

## Prerequisites

- A Cloudflare account that owns the `brandon-behring.dev` zone (the apex is already deployed via Workers, so this account exists).
- Repo admin access on `github.com/brandon-behring/double_ml_time_series` (for secrets).
- Optional but recommended: the `gh` CLI authenticated (`gh auth status`).
- **Wait for `brandon-behring/deploy-workflows#1` to merge before pushing anything.** Until then, `git push origin main` will fail at the production job because the reusable workflow doesn't yet recognize the `working-directory` and `validate-command` inputs.

---

## Step 1 — Get the Cloudflare credentials

You need two values: a scoped API token and your Account ID. The API token gets used by the GitHub Actions runner; never paste it into the repo, only into GitHub secrets.

### 1a. Account ID

1. Open the [Cloudflare dashboard](https://dash.cloudflare.com).
2. Pick any site in the sidebar (e.g. `brandon-behring.dev`).
3. The right-side overview pane lists the **Account ID** under "API". Copy it. It looks like `9a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d`.

### 1b. API token

1. Top-right user menu → **My Profile** → **API Tokens** → **Create Token**.
2. Pick the **"Edit Cloudflare Workers"** template (it includes the right scopes by default).
3. On the next screen, scope:
   - **Account resources**: include your account only (not "All accounts").
   - **Zone resources**: include `brandon-behring.dev` only (this is what lets the Worker bind a custom domain on this zone).
4. Click **Continue to summary** → **Create Token**.
5. **Copy the token immediately** — Cloudflare shows it once. Treat it like a password.

### 1c. Sanity check (optional)

```bash
curl -s -H "Authorization: Bearer <PASTE-TOKEN-HERE>" \
  https://api.cloudflare.com/client/v4/user/tokens/verify \
  | jq .result.status
```

Should print `"active"`.

---

## Step 2 — Set the two GitHub repo secrets

The reusable workflow at `brandon-behring/deploy-workflows` reads these two secrets from the calling repo.

Via the dashboard:

1. Open `https://github.com/brandon-behring/double_ml_time_series/settings/secrets/actions`.
2. **New repository secret**:
   - Name: `CLOUDFLARE_API_TOKEN`
   - Value: the token from Step 1b.
3. **New repository secret**:
   - Name: `CLOUDFLARE_ACCOUNT_ID`
   - Value: the Account ID from Step 1a.

Or via `gh` from the repo root:

```bash
gh secret set CLOUDFLARE_API_TOKEN
# (paste the token at the prompt, then Enter)
gh secret set CLOUDFLARE_ACCOUNT_ID
# (paste the account ID at the prompt, then Enter)
```

Verify both exist:

```bash
gh secret list
# Expect:
#   CLOUDFLARE_ACCOUNT_ID  Updated YYYY-MM-DD
#   CLOUDFLARE_API_TOKEN   Updated YYYY-MM-DD
```

---

## Step 3 — Confirm `deploy-workflows#1` has merged

The DML `deploy-web.yml` references the reusable workflow at `@main`. The reusable must already have the `working-directory` and `validate-command` inputs.

```bash
gh pr view 1 --repo brandon-behring/deploy-workflows --json state,mergedAt
# Expect: { "state": "MERGED", "mergedAt": "..." }
```

If it's still open, review and merge it first:

```bash
gh pr view 1 --repo brandon-behring/deploy-workflows --web   # open in browser
# or merge from the terminal:
gh pr merge 1 --repo brandon-behring/deploy-workflows --squash --delete-branch
```

---

## Step 4 — First deploy (Pass 1 — workers.dev URL)

From the DML repo root:

```bash
cd /home/brandon_behring/Claude/double_ml_time_series

# Confirm 5 unpushed commits on main
git log origin/main..main --oneline
# Expect: 5e55ce7, a6b0288, d25d18c, d8b5f70, deae877

# Push
git push origin main

# Follow the run
gh run watch
```

The workflow does:

1. `actions/checkout`
2. `actions/setup-node` (Node 22, cache `web/package-lock.json`)
3. `cd web && npm ci`
4. `cd web && npm run validate` (academic schema + cite + anchor)
5. `cd web && npm run build` (`astro build` + `pagefind`)
6. `cloudflare/wrangler-action@v3` deploys `web/dist/` to the Worker `brandon-behring-double-ml-time-series`

Expected outcome: the workflow completes green. The site is live at:

```
https://brandon-behring-double-ml-time-series.brandon-m-behring.workers.dev
```

Smoke test:

```bash
URL=https://brandon-behring-double-ml-time-series.brandon-m-behring.workers.dev
for p in / /chapters/ /chapters/chapter01-fwl-potential-outcomes/ /references/ /search/ /print/; do
  printf "%-50s %s\n" "$p" "$(curl -s -o /dev/null -w "%{http_code}" $URL$p)"
done
```

All should return `200`.

---

## Step 5 — Bind the custom domain `dml.brandon-behring.dev`

This step is **dashboard-only**. Cloudflare Workers' custom-domain binding writes the DNS CNAME automatically; you don't touch DNS records by hand.

1. Cloudflare dashboard → **Workers & Pages**.
2. Click the Worker `brandon-behring-double-ml-time-series`.
3. **Settings** tab → **Domains & Routes** section → **Add → Custom Domain**.
4. Enter `dml.brandon-behring.dev`.
5. Cloudflare detects the zone (you own it), validates, and offers **Add Custom Domain**. Click it.
6. Cloudflare creates a CNAME record `dml` → the Worker, scoped to the `brandon-behring.dev` zone. The orange-cloud proxy is on by default — leave it.

Propagation is usually under 60 seconds because the zone already lives at Cloudflare. Verify:

```bash
curl -I https://dml.brandon-behring.dev/
# Expect: HTTP/2 200, with cf-ray + server: cloudflare headers
```

---

## Step 6 — Pass 2: flip the site URL to the custom domain

Once `dml.brandon-behring.dev` serves the site, the static build's canonical URL, sitemap, and Open Graph metadata should match.

```bash
# Edit web/astro.config.mjs — change one line:
#   site: 'https://brandon-behring-double-ml-time-series.brandon-m-behring.workers.dev',
#   →
#   site: 'https://dml.brandon-behring.dev',

# Then:
git add web/astro.config.mjs
git commit -m "feat(web): Pass 2 — flip site URL to dml.brandon-behring.dev

The custom domain was bound in the Cloudflare dashboard; canonical URLs,
sitemap, and OG metadata now settle on the production hostname.

Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
git push origin main
gh run watch
```

After the second deploy lands, both URLs continue to work, but the canonical inside `<head>` is `dml.brandon-behring.dev`.

---

## Step 7 — Verify end-to-end

```bash
URL=https://dml.brandon-behring.dev

# Routes
for p in / /chapters/ /chapters/chapter01-fwl-potential-outcomes/ /references/ /search/ /print/; do
  printf "%-50s %s\n" "$p" "$(curl -s -o /dev/null -w "%{http_code}" $URL$p)"
done
# All 200.

# Content checks
curl -s $URL/chapters/chapter01-fwl-potential-outcomes/ | grep -c "Frisch-Waugh-Lovell"
# >= 13

curl -s $URL/references/ | grep -ic "chernozhukov"
# >= 4

# Canonical URL settled on custom domain
curl -s $URL/ | grep -oE 'rel="canonical"[^>]*'
# Should contain dml.brandon-behring.dev, not workers.dev
```

If all three checks pass, Phase 1 is shipped. Chapters 2–10 porting is the next phase.

---

## Troubleshooting

### Workflow fails at "Validate workflow_call inputs"

The reusable workflow at `brandon-behring/deploy-workflows@main` doesn't yet have the `working-directory` or `validate-command` inputs. Confirm `deploy-workflows#1` has merged (Step 3).

If you need to push *before* the PR merges, temporarily pin the reusable to the PR branch in `.github/workflows/deploy-web.yml`:

```yaml
uses: brandon-behring/deploy-workflows/.github/workflows/deploy-astro-worker.yml@feat/working-directory-and-validate-inputs
```

Don't commit this — keep it as a local-only patch — and flip it back to `@main` after the PR merges.

### Workflow fails at "npm ci" with "Lockfile not found"

The `working-directory: web` setting in the reusable wraps `npm ci` so it runs in `web/`. If you see this, double-check that:

- `web/package-lock.json` is committed (it should be — `deae877` commits 431 insertions including the lockfile)
- The `cache-dependency-path` in the reusable points at `${{ inputs.working-directory }}/package-lock.json` (the PR adds this)

### Worker deploys but routes 404

This means the build succeeded but `dist/` is structured wrong. Astro static output uses trailing-slash directory routes (`/chapters/index.html`). Cloudflare's `[assets]` block in `wrangler.jsonc` handles this by default. If you're seeing 404s on trailing-slash URLs that work locally with `npm run preview`, check the **Settings → Resources → Assets** panel in the Cloudflare dashboard and confirm the asset binding is intact.

### Custom domain returns 522 (Connection timed out)

The custom domain was bound before any Worker version existed. Push to `main` to trigger a deploy; the 522 resolves once the Worker has a version.

### Custom domain returns SSL error / 525

Cloudflare needs ~30–60s after binding to provision the edge certificate. If errors persist after 5 minutes, check the Workers domain settings — there should be no warning banner about pending SSL.

### `gh run watch` says "no recent runs"

The `paths:` filter on `deploy-web.yml` only triggers on changes under `web/**`, `bibliography.bib`, or the workflow file itself. If you pushed commits that only touch LaTeX / Python files, the deploy workflow correctly skips. Trigger one manually for testing:

```bash
gh workflow run deploy-web.yml --ref main
```

---

## What to expect operationally going forward

- **Every push to main that touches `web/**` triggers a production deploy**. Build time is ~25 seconds (Astro + Pagefind + Wrangler upload).
- **Every PR that touches `web/**` triggers a preview deploy** and posts the preview URL as a PR comment via `actions/github-script`.
- **Strict validation runs on both paths**. A broken `<Cite key="...">` (bibkey not in `references.json`) or a duplicate anchor ID will fail CI before the build runs. Fix the offending MDX before merging.
- **The `wrangler.jsonc` Worker name is `brandon-behring-double-ml-time-series`**; renaming it would invalidate the custom-domain binding. Avoid.

---

## Reference

- Reusable workflow: `~/Claude/deploy-workflows/.github/workflows/deploy-astro-worker.yml`
- Sibling consumer (production proof): `~/Claude/brandon-behring.dev/.github/workflows/deploy.yml`
- Cloudflare docs: <https://developers.cloudflare.com/workers/static-assets/>
- Custom-domain docs: <https://developers.cloudflare.com/workers/configuration/routing/custom-domains/>
