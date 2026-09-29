# Working on Pooled with an AI agent

Instructions for coding agents (Claude Code, Codex, Cursor, …). Humans: see [CONTRIBUTING.md](../CONTRIBUTING.md).

## What this is
A from-scratch WebGPU inference engine (`engine/`), a browser room runtime (`p2p.html` + `room.js` + `room/`) that splits one LLM's layers across devices over WebRTC, and Code mode (`harness/`), a coding agent on the room's model. No build step, no framework, ES modules only. Read [architecture.md](architecture.md) first, then [kernels.md](kernels.md), and [design/harness-app.md](design/harness-app.md) for Code mode. For any UI change, follow [design/language.md](design/language.md).

## Non-negotiables (from GOVERNANCE.md)
- Output must stay **bit-exact**: golden tests must pass, and the speculative stream must equal plain decoding (`tests/test_mtp.js`). Approximations need a default-off switch.
- Every performance claim gets a row in `docs/bench-log.md` with hardware + commit, including neutral results.
- Easy to use (a browser link always works; native/headless peers welcome), no accounts/tokens/ads, honest privacy claims (`SECURITY.md`).

## How to run things
```bash
npm test                              # unit tests, no GPU (CI runs these)
cd tests && deno run --unstable-webgpu --allow-read --allow-env test_qwen.js   # one GPU test
tests/run.sh quick | q38 | all        # suites; GPU tests read models from ../models/
cd benchmarks && MODEL=q38 deno run --unstable-webgpu --allow-read --allow-env bench.js
npm run e2e -- --phone                # room emulator: host + worker + phone-shaped tab, real WebRTC + GPU (manual only)
```

No GPU needed (headless Chromium on SwiftShader; needs `npm install` or `NODE_PATH` pointing at `playwright`, `peer` and `peerjs`):
```bash
node tests/e2e/harness_tetris.mjs --port <free port> --signal-port <free port>   # Code mode end to end, scripted model
node tests/e2e/preview_browser.mjs                                               # the preview sandbox (port 18986)
node tests/e2e/room_synth.mjs                                                    # the real room on a synthetic model
```
Model files live in `models/` (git-ignored). The 27B loads in ~2 minutes; plan runs accordingly.

## Rules learned the hard way (do not relearn them)
- **Never run two GPU jobs at once** (tests, benches, or a CUDA build): the GB10 shares memory bandwidth and every number becomes garbage. If another agent or workflow is using the GPU, stick to the no-GPU tests above.
- Do not edit `engine/*.js` while a queued test chain is still starting new processes; each process loads the files from disk when it starts.
- A shell loop that waits with `pgrep -f "<name>"` matches its own command line and waits forever. Wait on a completion line in a log instead.
- `dispatchWorkgroups(n)` with n > 65,535 is silently dropped; use the 2-D form (`_dop` does this).
- Dynamically indexed local arrays in WGSL spill (3× slower); generate unrolled code with literal indices. Barriers stay outside conditionals. Accumulate in f32.
- Feature-detect (`probeUnpack`, subgroup probes); Safari 26 has no subgroups and no storage-pointer function parameters.
- Profile by skipping kernel families (`benchmarks/bench_breakdown.js`); timestamp queries cost 3× just by being on.
- Vercel refuses deploys whose commits have an email that isn't on GitHub; `git config user.email` must match.
- The room page is served at `/room`; `curl` the alias's `/room`, not `/p2p.html`, to verify a deploy.

## Layout
`engine/engine.js` is a barrel; implementation is in `engine/{dense,qwen35,gguf,tokenizer,sampling,quant,autotune,selftest,safetensors}.js` and `engine/wgsl/*.js`. The room is `room.js` + `room/*.js`; Code mode is `harness/*.js`; the landing page is `index.html` + `site/`. Tests in `tests/` (goldens in `tests/golden`, CPU references in `tests/reference`, no-GPU tests in `tests/unit`). Benchmarks in `benchmarks/`. Docs in `docs/`.

## Style
Plain modern JS, two-space indent, no reformatting of untouched code, descriptive commit subjects with the measured effect for performance work.
