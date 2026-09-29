# 29 · Code mode: preview origin on a second domain

**Phase:** code mode · **Status:** in progress (repo side done; deploy step open)

## Why
Code mode runs the agent's app in a sandboxed frame. Without a second site, that frame lives in the room tab's own process, so an app that loops forever can freeze the room, and the agent cannot run code snippets at all (`run_js` needs a separate process). The code already supports a relay on another site: a page with `<meta name="preview-origin" content="https://...">` runs previews in `/harness/preview-relay.html` there. On localhost it uses `127.0.0.1` for this. pooled.run has no second domain yet.

## Design
- Register a second domain (a different registrable domain, not a subdomain of pooled.run, so the browser puts it in its own process) and deploy the same static site to it, or only `harness/`.
- Add the `preview-origin` meta to the room page on pooled.run and staging.
- Keep `/harness/preview-relay.html` frameable from pooled.run: no `X-Frame-Options`, and no `frame-ancestors` that blocks it (see 17).
- Update SECURITY.md's Code mode section once it is live.

## In the repo
- `preview-host/vercel.json`: a Vercel project that serves only `/harness/preview-relay.html` (the build copies it into `public/`). It rebuilds only when the relay or `preview-host/` changed since the last deployed commit.
- The relay is served with a CSP: the preview document's own policy (`harness/preview-build.js` `CSP`, which the app's srcdoc document inherits from the relay), plus `form-action 'none'` and `frame-ancestors` for pooled.run, pooled-dev.vercel.app and loopback. No `X-Frame-Options`. `vercel.json` sends the same headers for pooled.run's own copy, so a second domain added to the main project works too.
- The `preview-origin` meta takes one origin, or `host=origin` entries, so one static `p2p.html` can point production and staging at different preview sites. It refuses plain http (except loopback from an http page) and any origin on the page's own site (the same host, a subdomain or a sibling).
- `tests/unit/preview_host_test.js` checks the parsing and that both configs carry the same relay headers. `tests/e2e/preview_browser.mjs` serves the relay with those headers, so the hang and `run_js` checks run under the real policy.

## Deploy step (owner)
Nothing below is done by the repo; it needs Vercel access.
1. Create a second Vercel project from this repo with **Root Directory** `preview-host` (framework: Other). Keep "Include files outside the root directory in the Build Step" on (the build copies `../harness/preview-relay.html`).
2. Its production domain is the preview site. `*.vercel.app` is on the Public Suffix List, so `pooled-preview.vercel.app` is already a different registrable domain from pooled.run and pooled-dev.vercel.app; a separate purchased domain also works. Do not use a subdomain of pooled.run.
3. Check it: `curl -I https://<preview-site>/harness/preview-relay.html` shows the `Content-Security-Policy` with `frame-ancestors https://pooled.run ...` and no `X-Frame-Options`.
4. If staging should use its own relay, alias a second deployment (e.g. `pooled-preview-dev.vercel.app`). If staging uses the production relay, one project is enough.
5. Fill the meta in `p2p.html`, e.g. `content="pooled.run=https://pooled-preview.vercel.app pooled-dev.vercel.app=https://pooled-preview.vercel.app"`, and push to staging first. In a Code mode room on staging, the preview's "sandbox" tip should say "on a separate site", and a `while (true) {}` app should show "hung" while the room keeps working.
6. If the room page later gets a CSP (roadmap 17), its `frame-src` must include the preview site. If a new host must frame the relay, add it to `frame-ancestors` in both `vercel.json` files.
7. After production is checked, update SECURITY.md's Code mode section to say pooled.run runs previews on the preview site.

## Done when
- On pooled.run, a preview that runs `while (true) {}` is reported as hung and the room tab keeps working.
- The agent's `run_js` tool works on pooled.run.
- `tests/e2e/preview_browser.mjs` still passes, and a staging check confirms the relay loads from the second domain.
