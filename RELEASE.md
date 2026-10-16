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

1. All GPU tests pass on the maintainer's hardware (`npm run test:gpu`, `npm run test:q38`), and the no-GPU checks pass (`npm test`, `npm run e2e:code`, `node tests/e2e/preview_browser.mjs`).
2. [docs/bench-log.md](docs/bench-log.md) has rows for every performance change since the last tag.
3. Update [CHANGELOG.md](CHANGELOG.md): move *Unreleased* into a dated section, and bump `version` in `package.json` and `CITATION.cff`.
4. Write the release notes at `docs/releases/<MAJOR.MINOR>.md` (headline, what's new by area, compatibility and upgrade notes, known limits); they become the GitHub release's text. If `@pooled/cli` or `@pooled/openclaw` changed, bump and publish them as their own READMEs say.
5. Tag: `git tag -a vX.Y.Z -m "..." && git push --tags`.
6. Merge to `main`. Vercel deploys production. A manual `npx vercel deploy --prod` is only for emergencies.

Versioning: the app has been `MAJOR.MINOR.PATCH` since 1.0.0 (2026-09-27). The peer protocol has its own number (`PROTOCOL` in `room/transport.js`, 4 since before 1.0.0), and `@pooled/cli` and `@pooled/openclaw` have their own versions.
