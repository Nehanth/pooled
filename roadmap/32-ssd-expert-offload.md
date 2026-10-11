# 32 · SSD expert offload

**Phase:** big models · **Status:** PR 1 in progress · **Builds on:** #319 (ExpertStore), #320 (offload in rooms, `--ram`), #323

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

## Measurements it starts from

From the 122B work (scratchpad feasibility study on the 5070 PC, SK hynix P41):

- Parked expert to VRAM: 51 GB/s, 0.121 ms per 5.5 MB expert. `writeBuffer`: 3.8-10 GB/s. Copies don't overlap compute on Dawn's D3D12 queue; CPU work does (25-69%).
- NVMe cold random 4 MB reads: 1.8 GB/s at QD 1, 5.2 at QD 4, about 6.2 at QD 8 and up. So reads must be many and in flight at once; one synchronous handle serializes them.
- mmap'd experts in llama.cpp (GLM-5 on the Spark and the PC) stalled at about 1 GB/s: page faults are latency-bound. Explicit parallel reads, not mmap.

## Targets (estimates, to be replaced by measurements)

| Room | Model | Expected |
|---|---|---|
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
8. Unified-memory disk tier (Spark, then Mac).
9. Layer-major prefill for disk layers.
10. Qwen3.5-397B-A17B: model entry, 512 experts and K = 10 in the kernels, llama.cpp goldens.
11. Optional, each only if measured to help: an engine-layout pack file, a cost model for disk placement, a native IO addon.

## Non-goals

Lossy shortcuts: fewer routed experts than the model uses, a predicted or replaced router. Every output stays exact. Dense models' weights stay in memory.
