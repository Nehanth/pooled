# Long-context decode: split-K decode attention (attnDecode "v2")

September 29, 2026. GB10 (DGX Spark), Deno WebGPU (Vulkan), no subgroups. Branch `perf/kf-long-context`.

## What was slow

Decode attention (`attn_flash` + `attn_combine`, engine/wgsl/qwen35.js) already splits over the sequence,
but two things hurt at long context:

1. **The split length comes from `maxSeq`, not from the position.** `faSplit = max(256, up64(maxSeq / 128))`,
   so a room opened at 64K context uses 576-position splits from token one: at 4K that is 8 splits x 2 KV heads
   = 16 workgroups on the MoE, a fraction of the GPU. At the room default (MoE 32K) it is 256.
2. **The inner loop is latency bound.** Every thread does a serial 256-long dot product over u32 loads of its
   own K row, then G threads run the softmax serially, then every thread walks the V rows one load at a time.
   Measured: 39 ms of attention per MoE token at 64K for ~1.3 GB of K/V, far below the GB10's bandwidth.

Attention share of one decode token's GPU time, v1 (timestamp queries, `tests/bench_attn_ctx.js`):
MoE at 64K 39.6 of 58 ms (68 %); 27B at 16K 26 of 120 ms (22 %).

## What v2 does (engine/wgsl/attn_dec.js)

Same outer shape (one workgroup per (split, column, KV head), per-split partials, combine pass), new inside:

- **Split length from the column's own length:** `splitLen(s) = max(64, up64(ceil(s / S)))`, S = 256 / nKV
  by default (`attnDecodeSplits`). Short contexts still fill the GPU, long ones get longer splits, and a
  column's splits depend only on its position, so a verify column equals a decode step at the same position.
- 128-position chunks; scores by thread pairs over interleaved 16-byte `vec4<u32>` loads (one 32-byte sector per
  pair per load, 8 loads in flight), q as vec4s in workgroup memory.
- Softmax with 32 threads per head; V with 32 threads per row (a warp reads a whole 512-byte row), 8 row groups,
  4 rows in flight, summed in a fixed order.

Every reduction has a fixed order, so the kernel is deterministic. It is **not bit-identical** to attn_flash
(different split boundaries and summation order). It started opt-in and is **the default since September 30**
(engine option `attnDecode`, default `"v2"`; runtime toggle `engine.attnDecode = "v1" | "v2"`; `?attndecode=v1`
for a room, `ATTN_DECODE=v1` for tests through `tests/load_model.js`). The 27B goldens were re-baselined (see
"Default on" below).
Prefill full-width passes keep the tiled prefill kernel; partial prefill passes and verify passes use v2 when on.
Limits: headDim 256, G <= 8, f16 KV (not `kvQ8`); otherwise it stays on v1.

## Correctness

- `tests/test_attn_dec.js` (kernel vs float64 reference, MoE and 27B shapes, 1 .. 8K in QUICK mode):
  relDiff <= 5e-7; a 4-column verify pass is bit-identical to four 1-column passes (verify == decode). PASS.
- `tests/test_attn_dec_model.js`, real models, same prefilled KV, decode with v1 vs v2:
  - MoE, 1000 / 8000 / 30000-token prompts: logits relDiff 3.4e-6 / 1.2e-6 / 2.0e-6, argmax equal, 24 greedy
    tokens identical, v2 speculative (MTP, K=3) identical to v2 plain. PASS.
  - 27B, 1000 / 8000: relDiff 3.7e-6 / 8.0e-6, greedy 16 identical, spec == plain. PASS.
- While it was opt-in, the default path was unchanged: 27B bits `BITS plain 8a532ef5 hidden 52f2ae10` (the reference),
  `test_moe` MATCH llama.cpp on both prompts with spec identical to plain, unit tests 433 passed.

## Speed (tests/bench_attn_ctx.js, plain decode, wall tok/s; attention = kv_store + attention + combine GPU ms)

The bench sets the position without prefilling (attention reads every row regardless), so the numbers isolate
decode at that fill. The Spark's GPU queue was shared with other jobs; rows marked * ran while another GPU job
was active and are only good as v1/v2 ratios (A/B alternates within one process).

MoE (35B-A3B Q4_0), maxSeq 65792, two reps each (clean run):

| fill | v1 tok/s | v2 tok/s | speedup | attention v1 -> v2 (ms) |
|---:|---:|---:|---:|---:|
| 4096 | 14.5 | 20.5 | 1.41x | 25.8 -> 1.22 |
| 16384 | 20.4 † | 29.9 | 1.47x | 18.6 -> 2.77 |
| 32768 | 19.0 | 26.9 | 1.42x | 21.7 -> 5.56 |
| 65536 | 14.0 | 24.7 | 1.76x | 39.6 -> 9.40 |

† one v1 rep at 16K measured 2.4 tok/s (a GPU stall from another job); the other rep is shown.

MoE at the room default (maxSeq 32768; some contention in the 16K / 32K rows): 1K 15.5 -> 19.0, 4K 18.6 -> 22.5,
32.5K 13.3 -> 17.7 tok/s; attention 13 -> 0.63 ms (1K), 9-16 -> 1.26 ms (4K), 29-32 -> 8.9-16 ms (32.5K).

27B (Q4_0) at the room default (maxSeq 16384, clean):

| fill | v1 tok/s | v2 tok/s | speedup | attention v1 -> v2 (ms) |
|---:|---:|---:|---:|---:|
| 1024 | 8.87 | 9.51 | 1.07x | 9.0 -> 1.08 |
| 4096 | 8.64 | 9.34 | 1.08x | 11.6 -> 2.52 |
| 16128 | 7.78 | 8.96 | 1.15x | 26.4 -> 8.06 |

27B at maxSeq 65792*: attention 19.5 -> 3.9 ms (4K), 48 -> 16 (16K), 117 -> 35 (32K), 152 -> 49 (64K);
tok/s 2.9 -> 4.2 at 64K under contention (the dense weights, ~90 ms per token, dominate a clean 27B run).

## Takeaways

- 3-20x less attention time per token; decode +41-76 % on the MoE at 4K-64K and +7-15 % on the 27B.
- Most of the short-context win on large-maxSeq rooms is point 1 (split length from maxSeq); most of the
  long-context win is point 2 (v2 at 64K streams ~140 GB/s of K/V vs ~35 GB/s for v1).
- Done since: the combine is parallel over splits (4 workgroups per head, fixed-order tree reductions),
  0.83 -> 0.17 ms per MoE token at 64K. Still open: subgroup reductions (where available) for the softmax.

Reproduce:

```
cd tests
QUICK=1 deno run --unstable-webgpu --allow-read --allow-env test_attn_dec.js
MODEL=moe LENS=1000,8000,30000 deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights test_attn_dec_model.js
MODEL=moe FILLS=4096,16384,32768,65536 ATTN_DECODE=both deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights bench_attn_ctx.js
```
