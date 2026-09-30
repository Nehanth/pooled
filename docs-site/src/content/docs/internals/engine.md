---
title: Engine and kernels
description: The WebGPU engine under Pooled, its kernel families, the tricks that made it fast with what each one bought, and what runs differently on Apple GPUs.
eyebrow: How it works
sidebar:
  label: Engine and kernels
  order: 3
---


The engine is our own. It is not WebLLM, MLC or llama.cpp. Every kernel is hand-written WGSL, generated from JavaScript templates so that thread counts, rows per workgroup and batch widths can change per device from one source.

Owning the kernels is what makes the room possible: no ML compiler we looked at could split a model mid-way, run the hybrid DeltaNet architecture, or use our wire format.

:::tip[Same output, always]
Every trick below is either bit-identical to what it replaced or checked against a reference by a test. Speculative decoding produces the same stream as plain decoding, and a split room produces the same tokens as one device running the whole model (on the same kind of GPU).
:::

## What it runs

| Model | Architecture | Engine |
|---|---|---|
| Qwen 3.8 27B | Hybrid: 48 Gated DeltaNet layers + 16 full-attention layers, draft layer built in | `Qwen35Engine` |
| Qwen 3.6 35B MoE | Same hybrid, 40 layers, each FFN is 256 experts (8 active per token) plus a shared expert | `Qwen35Engine` |
| Qwen3 1.7B (and 0.6B, 4B) | Dense transformer | `DenseEngine` |

Weights are GGUF, in Q4_0 or Q8_0 blocks: 32 weights plus one f16 scale. A few tensors in the public MoE file are Q4_1, Q5_0 or K-quants; they are requantized to Q8_0 on the way to the GPU, a few rows at a time.

## Kernel families

| Family | Role |
|---|---|
| Cooperative GEMV | Decode. 64 threads sweep one weight row together so every read is a contiguous stripe. |
| Batched GEMV (`_b8`, `_b4` twins) | Speculative verify and short batches. A twin is picked by how many columns are live. |
| Fused gate/up | Both FFN projections in one sweep of the weights, SiLU applied before writing. |
| Prefill GEMM | 16-column prefill passes: a row-stationary Q4_0 / Q8_0 GEMM with the weight tile kept packed in shared memory. |
| DeltaNet | The gated delta-rule recurrence, with snapshot slots for speculative rollback. |
| Attention | Split-K flash attention over an f16 (or int8) KV cache. |
| MoE | Router, top-k, expert gate/up and down, and the weighted combine, all on the GPU. Routing needs no readback. |
| Sampling | Greedy argmax and top-k on the GPU, so a token reads back 16 bytes instead of the 1 MB logits vector. |

## What each trick bought

All numbers are from a DGX Spark (GB10) through Deno unless stated. Each row has a commit and a measurement in the [bench log](https://github.com/Nehanth/pooled/blob/main/docs/bench-log.md).

### Reading weights

| Trick | Effect |
|---|---|
| **Cooperative rows.** 64 threads sweep a row with interleaved 32-weight blocks, so the GPU coalesces every read. | 2.2× on the 27B |
| **Named scalar accumulators.** Four totals in an array indexed by a variable spilled to scratch memory and made the "fast" kernel 3× slower than the naive one. | Fixed a 3× loss |
| **Dequantize in registers.** Nibbles are unpacked from `u32` words and scaled in registers. Full-precision weights never exist in memory. | |
| **f16 block scales end to end.** Scales stay packed two per `u32`. | ~10% fewer bytes |
| **Load-time repacking** into separate nibble and scale arrays, so stripes are contiguous. | Layout, not compression |
| **Device autotune** of workgroup size and rows per workgroup, about 1 s at load, with a 3% noise guard. | GB10 picks (64, 4) |

### Fewer passes

| Trick | Effect |
|---|---|
| **One command submit per token.** All ~900 dispatches of a decode token go in one command buffer. | |
| **Accumulate into the residual** (`y += W·x`) in the output, DeltaNet out and FFN down projections. | 128 fewer dispatches per token; neutral on Vulkan |
| **Fused norm + router and norm + DeltaNet gates** on the MoE (#246). | 502 → 432 dispatches per MoE token, +4 to 5% decode |
| **GPU argmax and top-k.** Lowest-index tie-breaking matches the CPU loop exactly. | 16 B read back per greedy token instead of 1 MB |

### Batching and prefill

| Trick | Effect |
|---|---|
| **Batched prefill.** Prompt tokens go through as columns, so each weight block is read once for all of them. | 3 → 39 tok/s prefill on the 27B |
| **Multi-column glue kernels.** One dispatch covers every live column. | 2× on batched passes |
| **Row-stationary prefill GEMM** at 16 columns, split-K pinned per shape so every device computes identical hidden states. | 1.64× on the whole-model pass (34.6 → 56.7 tok/s) |
| **Width-dependent rows per workgroup** for the verify twins, so every batched kernel keeps 16 accumulators per thread. | Recovered verify from 12.8 back to 15.4 tok/s |

### Speculation

| Trick | Effect |
|---|---|
| **Multi-token prediction with exact rollback.** The model's `nextn` layer drafts up to 7 tokens; one batched pass verifies them; DeltaNet states are snapshotted inside the recurrence kernel after every column. | 9 → 16 tok/s on the 27B |
| **Draft over the first 65,536 vocabulary rows.** The head is the biggest matrix a draft reads, and only 1 to 2.5% of English and code tokens sit above that row. Other scripts fall back to the full head automatically. Drafts only: output never changes. | |
| **Draft depth by measured throughput.** The room tries depths 3, 5 and 7 and keeps the fastest. | Fixed a Mac-hosted room locked at 1.5 tok/s |

### Correctness and portability

- **2-D dispatch for tall matrices.** WebGPU silently drops a dispatch over 65,535 workgroups in one dimension, and the LM head needs 124,160. Kernels compute the row from two dimensions.
- **No f16 accumulation.** It corrupted output. f16 is used only for block scales, the KV cache and the wire.
- **No timestamp queries in production.** Merely enabling them cost 3× on the GB10. Profiling skips kernel families and re-times instead.
- **No subgroups required.** Safari has none and Deno hides them. They were measured at 0 to 8% anyway.
- **Feature probes at load.** Optional WGSL builtins are compiled as one-line test shaders, with a fallback where they are missing.
- **No storage pointers as function parameters.** They are unsafe on Safari, so helper functions are generated per buffer instead.

## Apple GPUs (Metal)

Apple Silicon runs the same kernels through WebGPU on Metal. A few things differ:

- **Wide fused MoE expert kernels** are the default on Apple adapters in Chrome and Safari, and the legacy layout everywhere else (#209). On an M5 Max in Chrome, MoE decode went up 5.6% plain and 11% speculative.
- **Wide prefill submits every 8 layers** instead of once per chunk. One large submit lost the device on Metal under Deno.
- **Per-component writes into a workgroup `vec4` array** lose neighbouring threads' writes on Metal. The tiled MoE prefill kernel gave garbage from the first token on Apple GPUs until that store was rewritten (#104).

For scale: an M5 Max decodes the MoE at 80 to 90 tok/s plain in Chrome, and the 27B at about 21.6 tok/s plain and 45 tok/s speculative. It moves memory about twice as fast as a GB10, and decode follows memory bandwidth.

## Tried and dropped

| Idea | Why not |
|---|---|
| Q4 KV cache | −92.5% prefill, and it hurts long documents and tool calls |
| An external 0.6B draft model | Different vocabulary (151,936 vs 248,320), so it can't be verified |
| RMSNorm fused into the GEMV | 0.91× on Metal |
| Tree or Medusa drafting | No trained heads, and a DeltaNet state would be needed per branch |
| Subgroup reductions | 0 to 8%; the shared-memory tree was never the bottleneck |
| 4×4-per-thread GEMM tiles | Slower than 2×4: occupancy beat reuse |

## What is still open

- **Prefill speed.** It is the biggest gap to native llama.cpp: about 1.0 s to the first token of a 172-token prompt on the MoE, against about 0.07 s native. Pinned GEMM shapes for the MoE and the 1.7B are next.
- **A register-resident DeltaNet update.** About 4% of a pass, bit-identical. Spec: [`docs/deltanet-prefill-spec.md`](https://github.com/Nehanth/pooled/blob/main/docs/deltanet-prefill-spec.md).
- **Expert-parallel splits.** Rooms split MoE models by layers today, like dense ones.

The full list, with each item's rationale, is in [`docs/kernels.md`](https://github.com/Nehanth/pooled/blob/main/docs/kernels.md).
