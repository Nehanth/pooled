# q8_0 KV cache (kvQ8): memory, speed and quality against f16

September 29, 2026. DGX Spark (GB10), Deno/wgpu. Branch `perf/kf-kv-quant`.

## What changed

`kvQ8: true` (room `?kv=q8`) already existed on main as int8 with one **f32** scale per 32 values
(56% of f16). This branch makes it llama.cpp's **q8_0** layout and lets the batched and prefill
attention paths run on it:

- `kv_store_q8`: d = max|x| / 127 per 32 values, `roundf(x / d)` (half away from zero, as
  `quantize_row_q8_0`; WGSL `round` is half-to-even), d stored as **f16** (2 per word). One thread per
  pair of blocks, so each thread writes whole scale words. 34 bytes per 32 values: **53.1% of f16**.
- `attn_flash_q8`: the score sums each block in integers-as-f32 and applies the block scale once
  (`sum_b d_b * sum_i q_i * x_i`), as llama.cpp's q8_0 dot does. V is dequantised value by value.
- `attn_flash_t2_q8` (new; f16 `attn_flash_t2` now comes from the same template and keeps its bits):
  two columns per workgroup for verify / batched passes, term for term the same as `attn_flash_q8`,
  so spec == plain holds with q8_0 KV. Main turned the 2-column kernel off under kvQ8.
- The tiled prefill attention (`engine/wgsl/attn_tile.js`, `Q8` config) dequantises q8_0 while it
  stages K/V tiles. Main turned the tile off under kvQ8, so long prompts fell back to per-column
  `attn_flash_q8`.
- Saved states carry `kvQ8: "q8_0"` in their signature, so old int8/f32-scale states are refused
  instead of misread.

Off by default. f16 paths give the same bits as main (`test_q38_bits.js`: 8a532ef5 / 52f2ae10).

## Memory

| Model | f16 KV per token | q8_0 per token | at 32K |
|---|---|---|---|
| Qwen 3.6 35B-A3B (MoE, 10 full-attention layers, kvDim 512) | 20.0 KB | 10.6 KB | 0.63 GB -> 0.33 GB |
| Qwen 3.8 27B (16 full-attention layers, kvDim 1024) | 64 KB | 34 KB | 2.1 GB -> 1.1 GB |

KV is a small part of the MoE's footprint at 32K (weights ~19 GB), so the memory win matters on
phones / laptops holding the 27B at 16-32K, or a room slice that holds only attention layers.

## Quality (`tests/kv_quant_eval.js`)

MoE, `CORPUS=code` (this repo's source), engines A = f16 (reference), B = q8_0, C = f16 with the
tiled prefill off (a noise floor: same KV format, different prefill summation order). Per fill: T = 48
teacher-forced positions (KL(A||X), top-1, max |dlogit|, perplexity of the true next token), then
G = 96 greedy tokens from the same state.

| Fill | Engine | KL mean | KL p99 | top-1 | mean max\|dlogit\| | greedy same until | greedy same positions | ppl |
|---|---|---|---|---|---|---|---|---|
| 1024 (T 16, G 32) | q8_0 | 0.0018 | 0.0040 | 100% | 0.40 | 32/32 | 100% | 26.54 (f16 26.66) |
| 1024 | noise | 0.0006 | 0.0020 | 100% | 0.29 | 32/32 | 100% | 26.56 |
| 4096 | q8_0 | 0.0045 | 0.087 | 100% | 0.71 | 1/96 | 8% | 3.168 (f16 3.134) |
| 4096 | noise | 0.0132 | 0.583 | 97.9% | 0.37 | 51/96 | 53% | 3.221 |
| 32768 | q8_0 | **0.0405** | **1.88** | 97.9% | 1.24 | 46/96 | 57% | **1.253** (f16 1.182) |
| 32768 | noise | 0.0003 | 0.0049 | 100% | 0.50 | 34/96 | 37% | 1.184 |

Reading:

- Up to 4K the q8_0 error is at the noise floor of merely reordering prefill sums (KL 0.0045 vs
  0.013). Greedy continuations of code fork early for both B and C: they are chaotic at near-ties and
  are not a useful agreement metric on their own; teacher-forced top-1 is.
- At 32K q8_0 is clearly above the floor: mean KL 0.04 (about 100x the reorder noise), one position in
  48 changes its top-1, and perplexity on the true text rises 6% (1.18 -> 1.25). The 32K window here is
  the source corpus repeated, so the model is copying from ~16K tokens back: exactly the long-range
  retrieval that K precision affects. The mean KL matches the published q8_0 figure for Qwen 3.6
  (KL < 0.04, docs/research/tabby-2026-09.md §3).
- Verdict: **keep opt-in**. Agreement is very high below ~8K and acceptable at 32K (the greedy answer
  at 32K is the same text in different words), but not high enough to turn it on by default for
  retrieval-heavy long contexts. The earlier plan to default it on for the MoE at >= 32K
  (kernels-next-2026-10.md) should not go ahead on this evidence: the MoE's KV is small, so the
  saving there does not pay for a 6% perplexity hit.

## Speed

`tests/bench_ctx.js`, MoE, CTX 33000, back to back on the Spark. Absolute numbers are below earlier bench-log rows for this model
(the machine was busy with a long queue); compare within the table only:

| Fill | KV | prefill tok/s | plain decode tok/s | spec decode tok/s (K 3) | spec == plain |
|---|---|---|---|---|---|
| 8192 | f16 | 204.6 | 15.79 | 22.89 | identical |
| 8192 | q8_0 | 198.9 (-3%) | 16.47 (+4%) | 23.49 (+3%) | identical |
| 32000 | f16 | 142.0 | 11.45 | 13.22 | identical |
| 32000 | q8_0 | 138.2 (-3%) | **12.29 (+7%)** | **15.29 (+16%)** | identical |

Decode at long context gains what the halved K/V reads predict (attention is the part of MoE decode
that grows with context). Prefill is 3% slower: the tile's K/V staging now unpacks int8 and a scale per
value, and prefill attention is compute-, not bandwidth-bound on GB10. Before this branch kvQ8 prefill
ran without the tile at all (per-column `attn_flash_q8`). The greedy-decode tok/s in the quality table
include a 1 MB logits readback per token and are not a speed measurement.

## Gates

- `deno test` unit suite: 436 passed (attn_tile q8_0 cases: the JS mirror of the tile kernel matches
  exact float64 attention over the dequantised K/V to 1e-15).
- `test_q38_bits.js` (27B, f16 KV): 8a532ef5 / 52f2ae10, the golden bits; spec == plain; GPU sampling ==
  logits path.
- `test_moe.js` (MoE, f16 KV): MOE PASS, every prompt matches llama.cpp, spec K=3 identical to plain.
- q8_0 KV: `bench_ctx.js KVQ8=1` spec == plain at 8K and 32K (uses `attn_flash_t2_q8` for verify).

## Reproduce

```
cd tests
MODEL=moe FILLS=4096,32768 T=48 G=96 deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights kv_quant_eval.js
MODEL=moe CTX=33000 FILLS=8192,32000 KVQ8=1 deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights bench_ctx.js
```

Not done: the 27B quality run (`MODEL=27b`), a prose corpus (`CORPUS=docs`), and Chrome / Metal
timings (`?kv=q8`).
