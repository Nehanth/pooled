# 32 · SSD expert offload

**Phase:** big models · **Status:** PR 1 in progress · **First target:** the 122B in a browser tab on the Mac Studio alone · **Builds on:** #319 (ExpertStore), #320 (offload in rooms, `--ram`), #323

## Why

Expert offload today has two tiers: a per-layer LRU cache of experts in VRAM, and every routed expert of an offloaded layer parked in system RAM (MAP_WRITE upload-heap buffers, 51 GB/s to VRAM on the RTX 5070 PC). A room can only hold a MoE whose experts fit in its devices' VRAM plus RAM: a Mac Studio and the 5070 PC hold about 80-85 GB, with the Spark about 190 GB.

The next models up don't fit. Qwen3.5-397B-A17B is about 230 GB at Q4_0 (60 layers, 512 experts, 10 routed + 1 shared, ~7.1 MB per expert). The 122B (72.5 GB) doesn't fit on the 5070 PC alone either. A third tier, experts read from the GGUF on NVMe when they miss, makes both possible. Outputs stay exact: the same bytes land in the same slots, and no kernel changes.

## Design

Three tiers below the GPU, all room-node only:

- **VRAM pool** (as today): the same number of slots for every offloaded layer, per-layer LRU.
- **Parked layers** (as today): sealed MAP_WRITE buffers, within the device's `--ram` promise.
- **Disk layers** (new): the last layers of a device's offloaded range. Their experts are never parked; the loader records where each expert's three parts are in the GGUF. A miss is served from a RAM cache of per-expert upload slots (T1), or else read from the file by worker threads (one fd per shard, ~1 MiB chunks, converted in the worker), copied into a mapped slot and on to the GPU. When no mapped slot is free, it falls back to `queue.writeBuffer` after the cut's submit.

Prefill on a disk layer submits what's queued first, then fills the region asynchronously (`_cutResolve`, `_regionLoad`, the `_prefillWide` loop): a `writeBuffer` in the middle of an unsubmitted encoder would land before the kernels that came before it.

**Planner:** disk is a last resort. A room that fits without it gets exactly the deal it gets today. Otherwise devices that can (discrete GPU, `--disk`, the model on local disk) take parked + disk layers, with a minimum RAM for T1.

**Later, gated on measurements:** router lookahead (top-M experts for the next 2-3 layers, computed in the same readback, prefetched into T1 only); a smaller prefill region (R slots instead of every expert of a layer: 1.64 GiB for the 122B, too much for the 5070 alone); a unified-memory disk tier for the Spark and the Mac; layer-major prefill so each disk layer streams once per prompt.

## The cut (per-layer readback) and how to shrink it

Decode on an offloaded layer stops at `moe_route`: the encoder is submitted, the 9 routed ids are read back, `plan()` copies the misses into the layer's slot pool, and the expert kernels run on the CPU's slot ids. That round trip happens once per offloaded layer per token. PR #350 found it, not the disk, capping a tab: 8.1-8.6 tok/s on the 35B with every expert in memory, against 58-60 resident. Measured on the Spark, `feat/offload-cut-latency`, numbers in `docs/bench-log.md`:

- **Where the time went in Chrome:** a `mapAsync` whose GPU work was still running when the request reached the GPU process resolved on that process's next timed check, about 2 ms later. A readback behind 80 us of GPU work took 2.9 ms median, while one with nothing left to run took 0.1-0.5 ms. Dawn in Node took 0.03 ms of overhead (0.095 ms against 0.080 ms of GPU work). CPU planning, submit and encoding were 0.05-0.1 ms per cut in both runtimes.
- **Map kick (`engine/map_kick.js`, on for offloaded readbacks, by default only in Chrome; Dawn gained nothing and may lose 0-2 tok/s).** While a readback of the cut, the region load or an offloaded token is pending, the engine keeps one `mapAsync` of an idle 16-byte buffer in flight. Every request the GPU process handles makes it check the device too, so the real map resolves within one IPC round trip of the GPU finishing: 0.4 ms instead of 2.9 ms in the microbenchmark. Nothing reads the kick buffer. Resident decode does not kick: a review's interleaved Chrome A/B measured it 2-6 tok/s slower with the kick on the token readback, where decode-ahead already hides the wait. The CPU and power cost of the kick stream is not measured. If a kick map fails, the engine stops kicking and waits on the real map.
- **Encode while waiting (`cutPipe`, opt-in; one-token decode).** For one token the pool path is certain (`nCols * K <= P`), so the cut can encode the layer's expert kernels before waiting, while the caller encodes the next layer and the selection comes back. `plan()`'s copies are submitted ahead of that part. The GPU receives the same commands in the same order. Its gain was within noise in Chrome and on Dawn, so it is off by default.
- **Speculative windows (`cutSpec`, off by default; one-token decode).** Each layer keeps a moving average of whether it missed. A layer below `cutSpecTheta` does not cut. Instead, `moe_remap` maps its ids to slots through a GPU copy of the store's table (`ExpertStore.gpuTables`, rewritten by `plan()` and `clear()`), and its experts run straight away. The ids and a miss flag go to a log, which the next real cut reads back in the same map. If every speculated layer hit, `plan()` is replayed for each one in order (stamps only), so the LRU state is exactly what a cut per layer would leave. If one missed, the token rolls back to that layer's FFN:
  - its misses are copied into the pool;
  - DeltaNet S and the conv windows of every later layer are restored from the copy taken at the token's start;
  - x is restored from `moe_remap`'s copy of the layer's FFN input;
  - the FFN reruns from its norm (same route, same ids) on the CPU's slot ids;
  - encoding continues from the next layer. KV rows the window wrote are written again before anything reads them.

  The last offloaded layer always cuts. Gate: `OFFLOAD_SPEC=always` (every layer speculates on every token, so most windows roll back: 932 rollbacks in one run) gives main's hashes and main's exact LRU counters. Only the `always` runs exercise rollback; with `OFFLOAD_SPEC=1` the 16-slot run never speculates.
- **What still costs.** After the kick, a cut in Chrome costs about 0.5 ms (vs 0.2-0.3 ms on Dawn). Speculation removes it only on layers that hit everything, which needs a large cache per layer: P(all 8 hit) is 0.43 at 90% hits and 0.17 at 80%. With a cache that small it is neutral, because it seldom speculates. A tab should still offload as few layers as its memory allows; a planner cost model can use about 0.5 ms per offloaded layer in Chrome and 0.25 ms in Dawn.
- **Considered and not built:** overlapping the next layer's attention (it reads this layer's output; only the shared expert is independent, about 1/9 of the expert FLOPs, too small to matter); poisoning every later kernel instead of rolling back (it touches every decode kernel or turns them all into indirect dispatches); a flag-only readback skipped on all-hit layers (the CPU still has to wait for the flag to know whether to encode copies, which is the same round trip).

## Measurements it starts from

From the 122B work (scratchpad feasibility study on the 5070 PC, SK hynix P41):

- Parked expert to VRAM: 51 GB/s, 0.121 ms per 5.5 MB expert. `writeBuffer`: 3.8-10 GB/s. Copies don't overlap compute on Dawn's D3D12 queue; CPU work does (25-69%).
- NVMe cold random 4 MB reads: 1.8 GB/s at QD 1, 5.2 at QD 4, about 6.2 at QD 8 and up. So reads must be many and in flight at once; one synchronous handle serializes them.
- mmap'd experts in llama.cpp (GLM-5 on the Spark and the PC) stalled at about 1 GB/s: page faults are latency-bound. Explicit parallel reads, not mmap.

## First target: the 122B in a browser tab on a Mac Studio

The Mac Studio (36 GB unified memory) can't hold the 122B (72.5 GB) today, alone or in a tab. With a disk tier it can: the non-expert weights and the hottest experts stay in memory (about 25 GB lent), and each token's misses come from the model's files in the browser's origin-private file system (OPFS), read in a worker with sync access handles. On unified memory there is no separate VRAM, so the prefill region (1.64 GiB) fits without the region shrink the 5070 needs. A token touches 48 layers x 8 experts x 5.5 MB, about 2.1 GB if nothing is cached; with most of that hitting the in-memory cache, the rest is a few hundred MB from an SSD that reads 5-7 GB/s raw (less through OPFS). Estimate: 5-10 tok/s, to be measured. Order for it: PR 1, then the unified-memory disk tier (8) with an OPFS reader, then the room-node reader and T1 (2, 3), the planner and `--disk` (4, 5), lookahead (6).

## Other targets (estimates, to be replaced by measurements)

| Room | Model | Expected |
|---|---|---|
| Mac Studio alone, in a browser tab | 122B Q4_0 | 5-10 tok/s |
| Mac (host) + 5070 PC with 19 PC layers on disk | 122B Q4_0 | 10.7-13 tok/s; 15-17 with lookahead (17 all in RAM today) |
| 5070 PC alone (`pooled host --here --disk`), after the region shrink | 122B Q4_0 | 10-11 tok/s; 14-15 with lookahead. Impossible today |
| Mac + Spark + PC | 397B Q4_0 | 4-5 tok/s; 8-10 with lookahead and a Spark disk tier |

## Plan

1. **Engine disk tier, correctness first.** `engine/expert_disk.js` (per-expert reads and conversion, injected reader), ExpertStore `attach` / `prepare` / async region loads, the three engine hooks, `OFFLOAD_DISK` in the test loader. Gate: `test_q38_bits` on the 35B with every offloaded layer on disk prints the same hashes as main. A first tok/s baseline in bench-log.
2. Room-node reader: worker threads, per-worker fds, a staging ring, and a disk benchmark (GB/s, p50/p99 per expert).
3. T1: a RAM LRU of per-expert MAP_WRITE slots, plus a 3-tier cache simulator over recorded routing traces to size it.
4. Planner: `meta.disk`, mPark + mDisk splits, deal fields, fuzz tests (rooms that fit without disk deal exactly as before).
5. CLI and room node: `--disk` on `host`, `join` and `--here`; docs. First real room: the 122B on Mac + PC with part of the PC on disk.
6. Router lookahead prefetch.
7. Region shrink, so the 122B runs on the PC alone.
8. Unified-memory disk tier (Mac and Spark) with an OPFS reader for browser tabs. **Next after 1**, for the first target.
9. Layer-major prefill for disk layers.
10. Qwen3.5-397B-A17B: model entry, 512 experts and K = 10 in the kernels, llama.cpp goldens.
11. Optional, each only if measured to help: an engine-layout pack file, a cost model for disk placement, a native IO addon.

## Non-goals

Lossy shortcuts: fewer routed experts than the model uses, a predicted or replaced router. Every output stays exact. Dense models' weights stay in memory.
