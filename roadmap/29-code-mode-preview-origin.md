# 29 · Code mode: preview origin on a second domain

**Phase:** code mode · **Status:** planned

## Why
Code mode runs the agent's app in a sandboxed frame. Without a second site, that frame lives in the room tab's own process, so an app that loops forever can freeze the room, and the agent cannot run code snippets at all (`run_js` needs a separate process). The code already supports a relay on another site: a page with `<meta name="preview-origin" content="https://...">` runs previews in `/harness/preview-relay.html` there. On localhost it uses `127.0.0.1` for this. pooled.run has no second domain yet.

## Design
- Register a second domain (a different registrable domain, not a subdomain of pooled.run, so the browser puts it in its own process) and deploy the same static site to it, or only `harness/`.
- Add the `preview-origin` meta to the room page on pooled.run and staging.
- The page checks the meta before using it (`relayProblem` in `harness/preview-frame.js`): a subdomain of the page's site or plain http from https is ignored with a console warning, so a wrong value leaves run_js off instead of running snippets in the room's process. With a valid value, run_js is offered to the agent with no other change (`room/code.js`).
- Keep `/harness/preview-relay.html` frameable from pooled.run: no `X-Frame-Options`, and no `frame-ancestors` that blocks it (see 17).
- Update SECURITY.md's Code mode section once it is live.

## Done when
- On pooled.run, a preview that runs `while (true) {}` is reported as hung and the room tab keeps working.
- The agent's `run_js` tool works on pooled.run.
- `tests/e2e/preview_browser.mjs` still passes, and a staging check confirms the relay loads from the second domain.
