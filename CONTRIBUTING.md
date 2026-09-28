# Contributing to Pooled

Thanks for helping. Pooled (formerly SwarmLLM) runs one open model across the devices in a room, in their browser tabs. It is our own WebGPU engine, a peer-to-peer runtime over WebRTC, and Code mode, an in-browser coding agent on top. Useful contributions include kernels, model support, the room protocol, Code mode, the site, docs, and benchmark reports from hardware we do not have.

## Before you start

- Read [GOVERNANCE.md](GOVERNANCE.md) for what needs a design note first, and [SECURITY.md](SECURITY.md) for the trust model.
- Search existing issues. For anything beyond a small fix, open an issue before writing code so the approach can be agreed on.
- One change per pull request. Keep refactors separate from behavior changes.

## AI usage policy

Contributions written with AI assistance are welcome. You are responsible for the code you submit: read it, run it, and be able to explain it in review. Pull requests that show no sign of having been run (failing tests, invented APIs, unverified performance claims) will be closed. Do not paste raw model output into issues or reviews.

## Development setup

The engine runs in browsers and, for testing, in [Deno](https://deno.com) (which ships WebGPU). You need a GPU with WebGPU support to run most tests.

```bash
git clone https://github.com/Nehanth/pooled && cd pooled
curl -fsSL https://deno.land/install.sh | sh          # Deno 2.x
npm test                                               # quick unit tests (no GPU)
```

Model files are not in the repository. Put GGUFs under `models/` (see [docs/models.md](docs/models.md)); the full test suite and benchmarks expect `models/qwen/model.gguf` (Qwen3 0.6B, Q8_0) and `models/q38/model.gguf` (Qwen 3.8 27B, Q4_0). `tests/test_moe.js` also needs `models/q36moe/` (Qwen 3.6 35B MoE), and Code mode's small model is `models/qwen17/` (Qwen3 1.7B).

```bash
npm run test:gpu          # golden tests on the small model
npm run test:q38          # 27B tests (needs ~16 GB of GPU memory)
npm run bench             # decode/prefill tok/s
npm run e2e -- --phone    # room emulator: host, worker and a phone-shaped tab in headless Chromium on this machine
```

The emulator (`tests/e2e/room.mjs`) forms a real room through PeerJS signaling and WebRTC, splits the small model across the tabs, runs prompts, and reports stats, wire channel counters and console errors. `--wire off|slice|stripeN` picks the hidden-state transport, `--phone` adds a tab with a mobile user agent and a 4x CPU throttle so the phone code paths run. It needs `npm install` and a Chromium with WebGPU on the machine; it is never run automatically.

**No GPU?** Headless Chromium runs WebGPU on SwiftShader, so these run on any machine (they need `npm install` for Playwright and the `peer` server):

```bash
npm run e2e:code                          # Code mode end to end: two tabs, a real room, a scripted model builds and fixes a Tetris game
node tests/e2e/preview_browser.mjs        # the sandboxed preview on its own
node tests/e2e/room_synth.mjs             # the real engine and room on a tiny synthetic model
```

`harness_tetris.mjs` and `room_synth.mjs` take `--port` and `--signal-port` if the default ports are busy.

To run the site locally, serve the repository root with `npm run serve` (or `npx serve .`, which reads `serve.json` for the `/room` and `/r/:code` rewrites) and open `/room`.

## Pull requests

**Preparing your PR**

- Kernel and engine changes must keep the golden tests passing **bit-exactly**. The speculative decoding path must produce a stream identical to plain decoding (`tests/test_mtp.js`). If a change is intentionally approximate, it needs a switch that defaults to off, and a note in the PR.
- Performance changes must add a row to [docs/bench-log.md](docs/bench-log.md) with the hardware, commit, and before/after numbers, including neutral results. "Faster" without a number is not reviewable. Room measurements name the devices and one of the standard prompts listed in the bench log, so rows stay comparable across PRs.
- Protocol changes must keep older peers failing loudly rather than silently (bump the message version).
- Add or update a test. GPU tests live in `tests/`, unit tests that need no GPU in `tests/unit/`, browser tests on SwiftShader in `tests/e2e/`.
- Code mode changes (`harness/`, the Code pane in the room) must keep `npm run e2e:code` and `tests/e2e/preview_browser.mjs` passing.
- Keep the diff readable: no reformatting of untouched code, no drive-by renames.

**Commit messages**: imperative subject line under 72 characters, a body that says *why*, and the measured effect for performance work.

**After submitting**: CI runs syntax checks and the unit tests, and Vercel posts a preview URL for your branch on the PR (every branch gets `<project>-git-<branch>-nehanths-projects.vercel.app`; see [RELEASE.md](RELEASE.md)). GPU tests run on the maintainer's hardware before merge; say which tests you ran locally and on what GPU. Merging to `main` deploys production.

## Coding guidelines

- Plain modern JavaScript (ES modules), no build step, no framework. Serve the repo with any static server that applies the `serve.json` rewrites (`npm run serve`); the room uses ES modules and the landing page uses absolute `/site/...` paths, so opening the HTML files directly does not work.
- WGSL kernels live in `engine/wgsl/*.js` (`base.js`, `coop.js`, `qwen35.js`, `gemm.js`, `moe.js`) as JavaScript templates, generated per shape when they vary. Keep generated code deterministic and readable.
- Kernel rules learned the hard way (see [docs/kernels.md](docs/kernels.md)): never dynamically index a local array (it spills), keep `workgroupBarrier()` outside conditionals, accumulate in f32, no subgroup or pointer-parameter features without a probe and a fallback, and dispatch tall matvecs in 2-D (a single dimension over 65,535 workgroups is silently dropped).
- Everything must run on Chrome, Safari 26, and Deno. Feature-detect at runtime; never assume an extension.
- Prefer measuring to reasoning: `tests/prof_ts.js` gives per-kernel GPU time with timestamp queries, and `tests/bench/chrome_bench.mjs` gives the Chrome numbers we quote. (`benchmarks/bench_breakdown.js` still has kernel family lists from before the fused kernels.)

## Naming guidelines

- Files: engine files are short lowercase names; room and harness modules use kebab-case (`code-ui.js`, `engine-model.js`). GPU tests are `tests/test_<what>.js`, unit tests `tests/unit/<what>_test.js`, benchmarks `bench_<what>.js`.
- Kernels: `<op>[_<quant>][_<variant>]`, e.g. `matvec_q4_coop_b`, `dn_delta_mc`. Suffixes: `_coop` cooperative rows, `_b` batched columns, `_acc` accumulate into output, `_mc` multi-column glue, `_gu` fused gate/up.
- Protocol messages: `ai-<verb>` with a `t` field.

## Documentation

User-facing behavior changes update the relevant page in `docs/`. New models go in [docs/models.md](docs/models.md).

## Resources

- [docs/architecture.md](docs/architecture.md): how a token flows through the system.
- [docs/kernels.md](docs/kernels.md): the WebGPU engine.
- [docs/protocol.md](docs/protocol.md): the room protocol.
- [docs/design/harness-app.md](docs/design/harness-app.md): Code mode.
- [docs/bench-log.md](docs/bench-log.md): every performance change with its numbers.
