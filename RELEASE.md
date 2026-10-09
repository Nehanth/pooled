# Releases

Pooled deploys through Vercel's git integration. Git tags mark milestones people can cite.

| Push to | Deploys to |
|---|---|
| any branch | nothing by default: previews are opt-in (`ignoreCommand` in `vercel.json`). A commit whose message contains `[preview]` builds a public preview, `https://<project>-git-<branch>-nehanths-projects.vercel.app` (PRs get the link as a bot comment). The first part is the Vercel project's name, so it changes if the project is renamed. |
| `feat/engine-opt` | also `https://pooled-dev.vercel.app` (staging) |
| `main` | production, [pooled.run](https://pooled.run) |

So every merge to `main` is a production release. Check the change on the branch preview or on staging first, and merge with a pull request.

`scripts/deploy-staging.sh` points the staging alias at a manual deploy. It is a fallback for when the git integration is down.

## Old domain

swarmllm.ai and www.swarmllm.ai stay attached to the same Vercel project. `vercel.json` sends every request on them to the same path on pooled.run with a permanent redirect, and query strings carry over, so old join links like `swarmllm.ai/r/ABCD?signal=...` keep working. www.pooled.run redirects to pooled.run the same way. Keep these rules in place: people have the old links saved.

Browser caches are per site. A device that downloaded weights on swarmllm.ai downloads them again the first time it opens pooled.run.

## Cutting a release

1. All GPU tests pass on the maintainer's hardware (`npm run test:gpu`, `npm run test:q38`), and the no-GPU checks pass (`npm test`, `npm run e2e:code`, `node tests/e2e/preview_browser.mjs`, `npm run check:preview`).
2. [docs/bench-log.md](docs/bench-log.md) has rows for every performance change since the last tag.
3. Update [CHANGELOG.md](CHANGELOG.md): move *Unreleased* into a dated section, and bump `version` in `package.json` and `CITATION.cff`.
4. Tag: `git tag -a v0.X.0 -m "..." && git push --tags`.
5. Merge to `main`. Vercel deploys production. A manual `npx vercel deploy --prod` is only for emergencies.

Versioning is `0.MINOR.PATCH` until the peer protocol is declared stable. A protocol change bumps MINOR.

## Isolated production previews

Before releasing Code mode on pooled.run, deploy a separate Vercel project with `preview-host` as its root directory. Its configuration serves only `/harness/preview-relay.html`. Use an HTTPS address on a different registrable site: `preview.pooled.run` shares the room's site and cannot isolate an infinite loop from the room tab.

Set `<meta name="preview-origin">` in `p2p.html` to the deployed origin, or use host mappings such as `pooled.run=https://<deployed-preview-host>` with a separate entry for staging if needed. Each mapped room host must be allowed by `frame-ancestors` in `preview-host/vercel.json`. Run `npm run check:preview` before merging a production release. For another production host, run `npm run check:preview -- p2p.html room.example.com preview-host/vercel.json`.

The checker validates the static configuration; it does not deploy the relay or verify that it responds. Check the deployed room's preview, an infinite-loop timeout and a subsequent healthy preview before releasing. An empty meta deliberately fails this production check until the relay has been deployed and configured. Self-hosted and development rooms can still leave it empty for local previews.
