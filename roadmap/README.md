# Roadmap

What we are working on next, grouped by the part of Pooled it touches. Each line is one item with its priority and its GitHub issue; the pinned [Roadmap: Pooled](https://github.com/Nehanth/pooled/issues/83) issue has the same list as a checklist. Longer design notes live in the numbered files in this folder; an item links its file when it has one. File names never change, because issues and links point at them.

The direction since 0.4: rooms of your own machines (a Mac and a PC, from the terminal or the OpenClaw plugin) running agents through `pooled serve`, and MoE models bigger than any one of them. The browser room stays; work that only matters to strangers' rooms or phones holding layers is lower now.

Priorities:

- **P0**: now. Broken, wrong, or the biggest thing holding rooms back.
- **P1**: next. Planned and ready to build.
- **P2**: later, or needs a design note first.

When an issue is filed, it gets the `area:` label of the section it sits in here. Last checked against the code on 2026-10-11 (main at 2b39a86): every line below was verified in code, not only in its issue.

## Big models

- **P1** SSD expert offload: stream a MoE's routed experts from NVMe when VRAM and RAM together can't hold them, below the RAM tier `--ram` already has (#319, #320). The next model size up on a Mac + PC. [32](32-ssd-expert-offload.md)
- **P1** 122B in Mac + PC rooms: a split-vs-solo correctness check with experts offloaded, encode-ahead and decode-ahead under offload (off today), and taking the 122B out of `?dev=1`.
- **P1** Converted-weight cache for the room node: CLI rooms convert and repack every load (the 122B: ~133 s from disk). The browser has one (`room/convertedcache.js`, #250); the room node doesn't.
- **P2** Expert-split MoE across a room: devices hold different experts of the same layer instead of different layers. [10](10-moe-expert-split.md) ([#10](https://github.com/Nehanth/pooled/issues/10))

## Kernels

- **P1** Prefill, what's left: the prefill kernels landed (GEMM tiles, flash-attention prefill, MoE grouping; chunked DeltaNet was dropped as slower). Left: the gap to llama.cpp (about 3x on the MoE at 2K, attention at 8K+), Chrome MoE rows at 1K/8K, Mac prefill. [02](02-prefill-gemm.md) ([#2](https://github.com/Nehanth/pooled/issues/2), [#217](https://github.com/Nehanth/pooled/issues/217))
- **P1** Apple MoE decode: the MoE reads about 45% of the Mac's bandwidth (MLX is ~25% faster). Try the coop GEMV 64/2 shape the autotuner never tries (+8.5% on an M5 Max), and fix Chrome-on-Metal wide-prefill drift (relDiff 0.17-0.38). [24](24-close-the-mac-gap.md) ([#24](https://github.com/Nehanth/pooled/issues/24), [#229](https://github.com/Nehanth/pooled/issues/229))
- **P2** Autotune validation: check each GEMV candidate against a CPU reference before timing it (the ROWS=8 bug itself was fixed in ca0b310). In review as part of #334.
- **P2** Prefill GEMM for the dense 1.7B: the MoE uses the wide GEMM; Qwen3 dense still falls back to batched GEMV, and agent prompts (~12K tokens) are prefill-bound on it. ([#66](https://github.com/Nehanth/pooled/issues/66))
- **P2** Freeze per-GPU MoE bit fingerprints, so a kernel regression shows up without llama.cpp. ([#67](https://github.com/Nehanth/pooled/issues/67), [#218](https://github.com/Nehanth/pooled/issues/218))
- **P2** Phone WebGPU limits: windowed bindings for expert stacks over 256 MB, and a first Android room. ([#65](https://github.com/Nehanth/pooled/issues/65))

## Inference

- **P1** Batched multi-session decode: OpenClaw and coding agents with subagents send requests at the same time, and they queue behind each other on one room today. One batched pass, bit-exact per session. ([#72](https://github.com/Nehanth/pooled/issues/72))
- **P1** Split MoE correctness in CI: `test_moe_split` (lavapipe, SYNTH=1) and `room_synth --compare` exist but run by hand. Put them in CI, then close the old 200-token bug. [31](31-moe-split-correctness.md) ([#68](https://github.com/Nehanth/pooled/issues/68))
- **P2** Pipelined speculative windows across devices, so a cross-network room keeps more than one lap in flight. Becomes P1 if cross-network laps turn out to be round-trip bound. [09](09-lap-overlap.md) ([#9](https://github.com/Nehanth/pooled/issues/9))
- **P2** Speculative decoding: draft trees, better prose drafts, adaptive depth. ([#220](https://github.com/Nehanth/pooled/issues/220))

## Agents (`pooled serve`, OpenClaw)

- **P1** Agent eval through `pooled serve`: the eval suite (`tests/eval`) on the MoE and the 122B, solo and on a Mac + PC room, with success rate, steps, reused tokens and wall time. ([#80](https://github.com/Nehanth/pooled/issues/80))
- **P2** Re-deal without re-reading the conversation: a re-deal re-prefills the whole context, which is long for OpenClaw on the 35B/122B; move the KV for ranges that moved. ([#3](https://github.com/Nehanth/pooled/issues/3))

## Harness (Code mode)

- **P2** Agent robustness, what's left: the grammar, loop detection and first-error feedback landed (#239); left are e2e cases per call format and a real-model run. [28](28-code-mode-agent-robustness.md) ([#74](https://github.com/Nehanth/pooled/issues/74))
- **P2** Turn on `run_js` in production: the code landed (#247, #248); left is the owner deploy of the preview origin on a second domain, then the meta in `p2p.html`. [29](29-code-mode-preview-origin.md) ([#79](https://github.com/Nehanth/pooled/issues/79))
- **P2** Code mode on the room models. ([#224](https://github.com/Nehanth/pooled/issues/224))

## Loading

- **P1** Pinned model revisions: every URL is still `resolve/main`. #334 pins them and checks ranges; after it, call `navigator.storage.persist()` before big downloads. [19](19-pinned-revisions-and-cache.md) ([#19](https://github.com/Nehanth/pooled/issues/19))
- **P1** Download retries in the browser and `pooled pull`: the room node already retries with backoff (`packages/room-node/source.js`); port it, then a pinned mirror. [18](18-download-path.md) ([#18](https://github.com/Nehanth/pooled/issues/18))
- **P2** Room-node devices share downloaded weights: browsers do (`ai-wget`), the room node answers "miss", so each machine in a CLI room downloads from Hugging Face.
- **P2** One bench-log row for a real browser converted-weight cache hit, then close. ([#77](https://github.com/Nehanth/pooled/issues/77))

## Networking

- **P1** Self-hosted signaling, part 1: vendor PeerJS (or SRI), and a scheduled canary for 0.peerjs.com and `/api/turn` with a status line. A broker of our own drops to P2: #325 binds auth to the DTLS fingerprints. [15](15-self-hosted-signaling-and-status.md) ([#15](https://github.com/Nehanth/pooled/issues/15))
- **P1** Cross-network numbers: the loss/latency matrix for the MoE at 2-4 devices, and one run across real separate networks (home to cellular) through the relay. Per-hop telemetry is not built yet; it goes with the benchmark work in #21. [25](25-cross-network-quick-wins.md) ([#25](https://github.com/Nehanth/pooled/issues/25))
- **P2** Show direct or relayed on each device card (only the room log and `pooledDebug()` say today). The relay itself is done, in the browser (#259, #264) and the CLI (#335). [01](01-relay-fallback.md) ([#1](https://github.com/Nehanth/pooled/issues/1))
- **P2** LAN rooms without pooled.run: `pooled host --lan` with its own signaling, building on #311. [05](05-offline-rooms.md) ([#5](https://github.com/Nehanth/pooled/issues/5))
- **P2** Lossy Wi-Fi: measure real p50/p90 hop times and extend duplicate sends to verify and prefill frames. [25](25-cross-network-quick-wins.md) ([#34](https://github.com/Nehanth/pooled/issues/34))
- **P2** Chain order by measured round trip, and one readback per hop (fewest hops and speed-aware placement landed). [27](27-placement-and-chain-order.md) ([#27](https://github.com/Nehanth/pooled/issues/27))

## Room

- **P1** Host-authoritative room page: only the host may send `ai-load`, and only the room's creator `ai-start-req` (any member can repoint the host today, `room.js` FROM_HOST). The room node already checks. Then field validation and per-peer quotas. [17](17-security-sweep.md) ([#17](https://github.com/Nehanth/pooled/issues/17), [#216](https://github.com/Nehanth/pooled/issues/216))
- **P1** Strict join auth by default: hosts still take the raw invite key from pre-#325 devices for one release. Make `POOLED_LEGACY_AUTH=0` / `?legacyauth=0` the default in the next release.
- **P1** A "copy diagnostic report" on the page and in `pooled host` / `join` (build, protocol, GPU limits, relay, per-hop stats), and the remaining raw load errors. [16](16-actionable-errors-and-versioning.md) ([#16](https://github.com/Nehanth/pooled/issues/16))
- **P2** Re-deal on join without a click, and skip the 60 s grace after a clean leave. [03](03-swarm-recovery.md) ([#3](https://github.com/Nehanth/pooled/issues/3))
- **P2** Kick a device (page and `pooled host`); rejoin dedupe by name already works. ([#56](https://github.com/Nehanth/pooled/issues/56))
- **P2** Hand the host role to another device when the host leaves. [27](27-placement-and-chain-order.md) ([#58](https://github.com/Nehanth/pooled/issues/58))
- **P2** Keep shrinking room.js: session handling and message dispatch next. The headless room is done (the room node and the shared pipeline, #293). [22](22-extract-the-runtime.md) ([#22](https://github.com/Nehanth/pooled/issues/22))
- **P2** A desktop Chrome worker backgrounded for 10 minutes, measured. ([#44](https://github.com/Nehanth/pooled/issues/44))

## UI

- **P2** MB/s on the load bar, and speed and time left on each device card (this device's time left is done). [18](18-download-path.md) ([#57](https://github.com/Nehanth/pooled/issues/57))
- **P2** Record the phone acceptance run for "Lend this device" (the code shipped in #236), then close. ([#78](https://github.com/Nehanth/pooled/issues/78))

## Tests and benchmarks

- **P1** Reproducible benchmarks: a protocol, a runner with one result format (model revision in every row) and a device matrix. A contributor is taking the first slice (3090, GTX 1060s, Galaxy S21) in #21. [21](21-reproducible-benchmarks.md) ([#21](https://github.com/Nehanth/pooled/issues/21))
- **P2** The synthetic ROWS 4/8 dense check (`engine_dense_synth.mjs`) in browser CI.

## Docs

- **P2** Roadmap and #83 in sync, and the old `swarmllm` links in issue bodies (they redirect). [23](23-contributor-on-ramp.md) ([#23](https://github.com/Nehanth/pooled/issues/23))

## Done

| # | Item | Status |
|---|---|---|
| 04, 11 | [`pooled serve`](04-local-endpoint.md) and [a native peer for headless GPUs](11-native-peer.md) | shipped in `@pooled/cli` 0.4 (`pooled serve`, `host`, `join`, `chat`) and the OpenClaw plugin |
| - | Qwen3.5 122B and MoE experts in system RAM (`--ram`) | shipped in CLI 0.4.0 (#316, #319, #320, #323) |
| - | Channel-bound join auth: keys and passes proved, never sent | landed (#325); strict-by-default is under Room |
| 01 | [TURN relay](01-relay-fallback.md) | browser (#259, #264) and CLI (#335); the card badge is under Networking |
| 12 | [Stop, fail fast, re-deal: the room survives launch day](12-room-survives-launch-day.md) | landed; automatic re-deal on leave too |
| 13 | [Multi-turn conversation and an honest context limit](13-multi-turn-context.md) | landed |
| 14 | [Pre-flight check, join links, and a model ladder](14-preflight-and-join-links.md) | landed |
| 06 | [More models: Qwen 3.8 27B, Qwen 3.6 35B MoE, Qwen3 1.7B](06-more-models.md) | shipped |
| 30 | [Checkpoints on disk, prefix reuse across agent turns](30-long-context-and-sessions.md) | landed (#242, #251, #313); int8 KV stays off by default (#252) |
| 26 | [Host-side overhead](26-host-side-overhead.md): GPU sampling, encode-ahead | landed; follow-ups in #218 and #220 |
| - | Tokenizer matches llama.cpp's Qwen split (#243), GPU tests fail on a printed FAIL (#241), one shared engine preset (#249), delete one model's cache (#234), harness correctness fixes (#238) | landed |
| - | [Rename to Pooled at pooled.run](../docs/rename-pooled.md) | done |

Research notes with no scheduled work yet: [07 persistent rooms and stats](07-rooms-and-stats.md), [08 audited compute](08-verification.md) ([#289](https://github.com/Nehanth/pooled/issues/289)). Older reasoning is in [docs/archive/master-plan.md](../docs/archive/master-plan.md) and the [roadmap gap review](../docs/archive/roadmap-review.md), which keep the old name.

**Explicitly not on the roadmap** (see [GOVERNANCE.md](../GOVERNANCE.md)): accounts, tokens or ads, rooms open to strangers by default, distributed training, noise or permutation "privacy" features, frontier-scale models that cannot fit a room.
