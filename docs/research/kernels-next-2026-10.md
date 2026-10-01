# Next kernel and engine levers, October 2026: decode and prefill for the 27B and the 35B-A3B MoE

Written 2026-09-27. CPU-only work: nothing here was run on a GPU. It builds on `kernels-2026-09.md` and `kernels-next-2026-09.md` and skips everything those documents list that has since landed. Tags:
- **[M]**: measured in this repo. The source is named each time, mostly `docs/bench-log.md`, 2026-09-26 entries.
- **[E]**: my estimate, with the arithmetic shown.
- **[W]**: an outside source, linked.

Exactness uses three labels:
- **bit-identical**: today's bits for every column of every pass. All goldens pass unchanged.
- **tolerance (prefill-only)**: the change applies only to full-width (NC = 16) prefill passes. Those already differ from one-token passes within tolerance (the GEMM path), and verify passes never reach them.
- **numerics change**: new bits for every pass width alike. Spec == plain and split == solo still hold, because every width runs the same per-column arithmetic. The goldens are re-baselined. For the MoE the goldens are llama.cpp text, not bits (`engine/wgsl/moe.js` header), so the bar there is lower.

## 1. What already exists (so it is not recommended again)

| Area | In the tree today |
|---|---|
| Decode GEMV | Cooperative GEMV ladder (`coop.js`): 4 threads per Q4 block, 4-byte weight loads, `unpack4x` dequant, `_acc` residual folding, fused gate/up + SiLU. Merged projections (`fuseProj`: `[qkv\|z]`, `[beta\|alpha]`, `[k\|v]`, router + shared gate), bit-identical. |
| Batched GEMV | `_b8` / `_b4` twins at width-dependent rows per workgroup, byte-identical to the native width (`test_twins.js`). |
| Prefill GEMM | Row-stationary Q4_0 and Q8_0 GEMM at 16 columns with pinned split-K. **Only the 27B's 5120-wide shapes are pinned in `GEMM_S`** (`gemm.js`). The MoE's 2048-wide shapes have no entry, so MoE prefill runs entirely on the batched GEMV. |
| MoE | Fused FFN (default): router GEMV with the shared gate as row 256; `moe_route` (parallel softmax, top-8 by rank counting); `moe_gus` (gate/up + SiLU for the 8 routed slots plus the shared expert as slot 8, one launch); `moe_dnc` (down for all 9 slots, then the weighted combine and residual add in the epilogue). 5 dispatches per MoE layer. Grid y = (column, slot): **no expert grouping** across columns. |
| DeltaNet | Register-resident `dn_delta` / `dn_delta_mc`, fused `dn_delta_gn` for decode, fused `dn_pre` (gates + q/k L2), batched draft-cache fill, replay rollback (`S_pre` + replay of the accepted columns with `dn_delta_mc`). |
| Attention | f16 KV plus split-K `attn_flash` / `attn_combine`. Splits of 256 positions (larger past 32K, so at most 128 splits). The G q-heads of a KV group share each K/V read. `attn_flash_t2` handles 2 columns per workgroup and is bit-identical. `kvQ8` (int8 with an f32 scale per 32 values) is opt-in. |
| Speculation | MTP draft chain in one submit, `specFuse` (drafts + verify + head in one submit), `draftvocab` with auto-fallback, prompt-lookup drafts of up to 15 tokens, room depth probing 3/5/7 by measured tok/s. Solo decoding is always K = 3 (`room.js` `pickK`). |

## 2. The profiles this is based on

**MoE, GB10, one decode token, timestamp queries** [M: bench-log 2026-09-26, "combined optimizations"]:
- GPU total 20.46–20.63 ms over **502 dispatches**.
- Top kernels:

| Kernel | Total | Calls | Per call |
|---|---|---|---|
| `matvec_q8_coop` | 3.08 ms | 15 | |
| `moe_gus_q4_q8` | 3.05 ms | 40 | 76 µs |
| `matvec_q4_coop` | 2.97 ms | 36 | |
| `moe_dnc_q4_q8` | 1.87 ms | 35 | 54 µs |
| `dn_pre` | 1.32 ms | 30 | 44 µs |
| `dn_delta_gn` | 1.28 ms | 30 | 43 µs |
| `moe_route` | 1.25 ms | 40 | 31 µs |
| `matvec_coop` (f32) | 1.11 ms | 70 | |
| `rmsnorm` | 1.08 ms | 81 | 13 µs |

Other MoE measurements [M: same bench-log sections]:
- Chrome decode: 40–45 tok/s plain and 65–84 tok/s speculative with K = 3 (22–25 ms per token plain).
- Syncs: Chrome 0.76 ms, Deno 11.7 ms.
- CPU encode: 4.5 ms for 732 dispatches, about 6 µs each. At today's 502 dispatches that is about 3 ms [E].
- llama.cpp CUDA on the same file and GPU: **85.5 tok/s (11.7 ms)**, pp512 2520 tok/s.

**MoE bytes per token** [E, from the shapes in `moe-2026-09.md`, bartowski Q4_0 with Q8 shared experts and a Q8 LM head]:

| Weights | MB |
|---|---|
| Routed experts: 40 × 8 × 3 × 0.59 MB | 566 |
| Shared experts (Q8) | 134 |
| f32 routers | 84 |
| DeltaNet projections: 30 × 18.9 MB | 567 |
| Attention: 10 × 15.3 MB | 153 |
| Q8 LM head | 540 |
| **Total** | **≈ 2.04 GB** |

At the repo's 184 GB/s WebGPU streaming probe, 2.04 GB is **11.1 ms**. The GPU takes 20.5 ms. **About 9.4 ms per token is not bandwidth**: it is small-kernel latency and below-roofline kernels.

**Per-kernel effective bandwidth** [E, from the table above]:
- `moe_gus`: 9.44 MB routed + 2.23 MB shared in 76 µs = **154 GB/s**, 84% of the probe.
- `moe_dnc`: 4.72 + 1.11 MB in 54 µs = **108 GB/s**, 59% of the probe.

**Dense 27B.**
- 15 GB of weights, 82 ms of GEMVs at 183 GB/s [M: bench-log Sep 1].
- Chrome decode: 10.1–10.8 tok/s plain, 20–27 tok/s speculative [M: 2026-09-26].
- **llama.cpp CUDA on the same GB10 is now 13.8 tok/s** [M: bench-log 2026-09-26 reference line]. That means native code streams the 27B at about 15 GB / 72 ms ≈ **208 GB/s**, 13% above the WebGPU probe; the GB10's peak is 273 GB/s.

This is the most important new fact for the dense model: **the 184 GB/s probe is a property of how we load, not a ceiling of the machine.** On 2026-09-01 this engine beat llama.cpp (9.0 vs 8.0 tok/s). It no longer does.

**Measurement caveat.** `tests/prof_ts.js` puts each dispatch in its own compute pass with begin/end timestamps. Per-call numbers therefore include pass overhead and exclude any overlap between dispatches. For the ~13–44 µs kernels this overhead may be a large share of the number. Before trusting the small-kernel rows, time an empty dispatch the same way and subtract it (item S0). In Chrome, timestamp queries are quantized to 100 µs unless "WebGPU Developer Features" is on ([W: Chrome developer features](https://developer.chrome.com/docs/web-platform/webgpu/developer-features)). Per-kernel Chrome profiles need that flag; Deno is not quantized. `docs/kernels.md` still says "no timestamp queries (3× slowdown when merely enabled)". That line predates `prof_ts.js` and should be re-checked.

## 3. MoE decode

### M-a. Serial small kernels: latency fixes that keep the bits (MoE and 27B)

Four kernels account for 4.9 ms of the MoE's 20.5 ms, and each is a latency problem, not a bandwidth one:
- **`dn_pre`, 44 µs × 30.** It is one workgroup. Each of the 32 q/k-head threads walks 128 values of global memory serially (`pp_l2`: a load-then-FMA loop that is not unrolled, so there is about one L2 round trip per element), then writes 128 values.
  - Fix: all 128 threads first stage the conv output's q/k region into workgroup memory with coalesced loads. Then each head thread runs **the same serial `ss += v*v` chain** over workgroup memory (about 30 cycles per step instead of about 400), and the normalized values are written back cooperatively.
  - This is the same arithmetic in the same order, so it is **bit-identical**. `attn_glue` already sums from workgroup memory this way.
- **`moe_route`, 31 µs × 40.** The rank count `r += select(...)` over 256 experts is an integer sum, so it can be reordered freely without changing any bit.
  - Bake `n = 256` as a literal, unroll by 8 with independent counters, and read `rt_p` as `vec4`.
  - The softmax max/sum trees and the final thread-0 renormalization keep their order. **Bit-identical.**
- **`rmsnorm`, 13 µs × 81.** One workgroup of 256 threads runs a strided loop with a dynamic bound, which leaves 8 dependent load round trips at dim 2048.
  - Emit the loop unrolled for the model's literal `dim`, with all loads issued first and then summed in the same `ss += v*v` order per thread. The tree is unchanged. **Bit-identical.**
- **`dn_delta_gn`, 43 µs × 30.** There are only 32 workgroups (one per value head) on a 48-SM GPU, and each moves 64 KB of state in and out: 2 MB × 2 per layer, which is 120 MB per token, **0.65 ms at the probe versus 1.28 ms measured**.
  - Splitting a head's 128 state columns over 2–4 workgroups is exact, because columns are independent. It doubles the number of SMs streaming the state.
  - The catch is the fused gated norm, which needs all 128 outputs of a head. Either keep 128-column workgroups for `dn_delta_gn` and only split the unfused `dn_delta` (then run `dn_gatenorm`: +1 dispatch, bit-identical per `dnFuse`), or A/B both.
  - Expected −0.3 to −0.5 ms per token on the MoE [E]. On the 27B (48 heads) the effect is smaller.
  - Measured 2026-10-01 (bench-log, branch perf/gdn-head-split): bit-identical but slower. The split `dn_delta` takes 44 µs against 49 µs unsplit and 40 µs fused, so with `dn_gatenorm` it loses ~11 µs per layer. Not kept.

Expected total: **−2 to −3 ms per MoE token, about +10–15% plain [E]**, and about −2 to −3 ms on the 27B (48 × `dn_pre` plus 129 × `rmsnorm`), about +2–3%.
- Effort: low. These are generator edits in `engine/wgsl/qwen35.js`, `base.js` and `moe.js`.
- Risk: low.
- Measure with `prof_ts.js` after the S0 calibration.

### M-b. Wide loads in the fused expert kernels; gate/up/down fusion across the top-8

`moe_gus` and `moe_dnc` already fuse everything that can be fused without a cross-workgroup dependency:
- `moe_gus` does gate + up + SiLU for all 8 slots plus the shared expert.
- `moe_dnc` does down for all 9 slots, the fixed-order weighted sum and the residual add.

A single gate→up→down kernel per expert would need either:
- one workgroup to own a whole expert's 512 h-rows, which is 9 workgroups per token and unusable, or
- split-K on down with cross-workgroup accumulation. WGSL has no float atomics, a CAS loop has nondeterministic order (it breaks spec == plain), and fixed-order partial buffers cost about 9 MB of extra traffic, as much as the weights.

**Do not build it** (§10). The remaining lever is the per-thread layout.

The fused kernels still use the first coop layout: 4 threads per block with 4-byte loads, `moe_dnc` at R = 1 with 64 threads. The unfused kernels have the GB10-tuned `MOE_DEFAULT` (16 B whole-block loads, input staged in workgroup memory), measured at `moe_dn` 43.0 → 38.4 µs and `moe_gu` 69.8 → 60.7 µs [M: `docs/kernels.md`].

**Build.** Port the `expertKernel` `{WG, TPR, R, U, wide, xsh}` generator into `gusKernel` / `dncKernel`, keeping slot K for the shared expert and the combine epilogue. Sweep the layouts with `tests/bench/moe_kernel_sweep.js`.

**Target.** `moe_dnc` from 108 GB/s to ≥150 GB/s, i.e. 54 → about 39 µs (−0.6 ms per token). `moe_gus` 76 → about 66 µs (−0.4 ms).

**Also exact: interleave gate and up rows per expert at load** (the `ffn_gate_up_exps` layout, [W: `moe-2026-09.md` §3]), so one 16 B load stream feeds both accumulators.

**Exactness.** Numerics change (a different reduction order per layout). Batched == single and spec == plain hold for any layout, because each (column, slot) runs the same code at every width. The MoE goldens are llama.cpp text.

### M-c. Fewer bytes, exactly: BF16 routers, native Q6_K head

- **Routers as BF16.** The file stores the routers as BF16, and the loader expands them to f32 (exact). Keep them as BF16 on the GPU and expand in the kernel with `bitcast<f32>(w << 16u)`: the same f32 values in the same order. **Bit-identical**, 84 → 42 MB per token, about −0.25 ms [E]. `matvec_coop` (f32) is 1.11 ms for 70 calls today, against a ~0.46 ms roofline for the routers alone.
- **LM head in native Q6_K.** The head is 540 MB of Q8, 26% of the MoE's bytes. The file stores it as Q6_K (417 MB), and requantizing Q6_K → Q8_0 is lossy (one f16 scale per 32 cannot represent Q6_K's per-16 int8 sub-scales).
  - A Q6_K coop GEMV would save 123 MB per token (−0.67 ms) and move the logits *closer* to llama.cpp's.
  - Numerics change. It is worth doing together with M-b's re-baseline.
  - The same argument applies to the Q4_1 `ffn_down_exps` in the first layers.
  - Shared experts in Q5_0 → Q8_0 are exactly representable (`q8 = q5 − 16`, same scale), so keep those.

## 4. MoE verify and prefill: expert grouping (the largest MoE lever)

**Why.** Every (column, slot) pair streams its own expert rows, and the shared expert is re-read for every column.
- A K = 3 verify (4 columns) reads 4 × (566 + 134) = 2.8 GB of FFN weights, against 0.7 GB for one token.
- A 16-column prefill chunk reads 11.2 GB, 61 ms at the probe.

**Real routing traces** [M: `offload-2026-09.md`]:
- Adjacent tokens share 34–46% of a layer's 8 experts.
- 64 tokens touch 86–109 of 256 experts.
- 700 tokens touch 202–211.

| Pass | (col, slot) pairs per layer | Unique experts [E] | Routed bytes saved | Shared bytes saved |
|---|---|---|---|---|
| K = 3 verify, 4 columns | 32 | ≈ 22 (8 + 3 × 4.8) | ≈ 31% | 75% |
| 16-column prefill | 128 | ≈ 45–60 | ≈ 55–65% | 94% |
| 64-column prefill | 512 | 86–109 | ≈ 80% | 98% |

**Build** (`moe.js`, `qwen35.js` `_encodeFFN` batched path):
1. A `moe_group` kernel: one workgroup per layer-pass. It stable-sorts the (column, slot) pairs by expert id (at most 16 × 8 pairs) and writes, for each unique expert, the list of columns (and the slot each came from) plus a count. It can also count the union size as telemetry.
2. `moe_gus_g`: grid x = row block, y = unique-expert index. The workgroup loads the expert's rows once and keeps **per-column accumulators in the same per-thread layout and block order as the single-column kernel**. This is the `_b4` twin technique, which is already byte-identical to native width. Each column's h is written to its own (column, slot) slot. The shared expert becomes one "expert" used by every column.
3. `moe_dnc_g`: grouping is awkward here, because `moe_dnc` does the combine per column. Split it: a grouped down pass writes `y[col][slot]`, then an epilogue kernel does the fixed-order combine and residual add. That is the old `moe_combine`, which is bit-identical to the fused epilogue if `y` is.
4. Use a fixed maximum number of columns per group (e.g. 16) with a tail loop so all shapes are static, or dispatch indirectly (`dispatchWorkgroupsIndirect`) with the count written by `moe_group`.

**Expected gain.**
- **Verify:** −27% of a K = 3 verify's bytes (4.14 → 3.03 GB [E]). Wide verifies are what makes MoE speculation weak today (1.6–1.9× over plain, against 2–2.7× on the 27B [M]), so **spec +10–20% [E]**.
- **Prefill at 16 columns:** the FFN drops from 11.2 to about 3.7 GB per chunk, so **prefill 2–3× [E]**.
- **Larger MoE chunks** (32–64 columns) compound this. They then need the chunked DeltaNet (§8) and more activation memory.

**Also, cheaply: pin `GEMM_S` entries for the MoE's 2048-wide shapes** in `engine/wgsl/gemm.js`: `12288x2048` (merged qkv|z), `8192x2048` (q + gate), `4096x2048`, `2048x4096`, `512x2048`, `2048x512`. With dIn = 2048, `(dIn/32/2) % S == 0` allows S ∈ {2, 4, 8, 16}; benchmark once and pin. Today MoE prefill runs its dense projections on the 16-column batched GEMV, which `gemm.js` measures at ~26 GB/s against 2.5× that for the GEMM. Tolerance (prefill-only); low effort.

**Risk and prerequisite.** `moe_synth`'s "batched prefill == one token" check **still fails with 202 logits** [M: bench-log 2026-09-26], and the cause is open. Grouping relies on that property, so fix or explain it first.
- Effort: 1–2 weeks.
- Exactness: bit-identical if the per-column order is kept (verify-safe). A GEMM-style grouped kernel is tolerance-only and must be restricted to NC = 16 prefill.

## 5. Dense 27B decode: bytes in flight

**The 16-byte coop GEMV** (`kernels-next-2026-09.md` D4, specified but not built; `coop.js` still issues one `u32` per thread per row per step). Each physical thread loads a whole Q4 block as one `vec4<u32>`, or a Q8 half-block as two. It keeps four logical-thread accumulators, `(bl, qt)` for qt = 0..3, each built in exactly today's `sc*(dot(lo,xlo)+dot(hi,xhi))` form and block order. It then writes them to the tree slots those logical threads own. The tree is unchanged.

- **Evidence.** llama.cpp streams the same 27B file at ~208 GB/s on the same machine [M]. The expert kernels' wide layout gained 10–13% [M]. MMVQ ([W: llama.cpp `ggml-cuda/mmvq.cu`](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda)) loads whole quant blocks per thread.
- **Expected gain.** GEMVs from 82 ms to 72–76 ms per token (−6 to −10 ms, **+6–11% plain**). The verify GEMVs gain in proportion, so spec gains about the same.
- **Exactness.** **Bit-identical by construction**, provided no backend compiler re-contracts the FMAs differently. Verify with the golden suite on Vulkan, Metal and D3D12. If any backend differs, the change becomes a flag.
- **Effort.** Medium: `coop.js` covers `_coop`, `_coop_acc`, `_gu` and the b4/b8 twins.
- **First step.** Add a 16 B/thread variant to the streaming probe. If the probe itself rises from 184 GB/s toward 208, the gain is real.

## 6. Long context: flash-decoding and KV quantization

**What exists.** Split-K by sequence (`attn_flash`), a fixed-order `attn_combine`, GQA sharing, f16 or int8 KV. Room defaults are **27B 16K (max 32K), MoE 32K (max 64K)** (`room/models.js`). Nothing at long context has been timed. `tests/bench_ctx.js` exists, but no row is in the bench log.

**KV bytes streamed per decode token** [E]:

| Context | 27B f16 (65.5 KB per token) | 27B q8 | MoE f16 (20 KB per token) | MoE q8 |
|---|---|---|---|---|
| 16K | 1.07 GB = 5.8 ms | 3.3 ms | 0.34 GB = 1.8 ms | 1.0 ms |
| 32K | 2.1 GB = 11.7 ms | 6.6 ms | 0.67 GB = 3.6 ms | 2.1 ms |
| 64K | — | — | 1.34 GB = 7.3 ms | 4.1 ms |

At the MoE's default 32K, attention at roofline adds 18% to a 20.5 ms token. At 64K it adds 36%.

**What is missing in `attn_flash`** (`engine/wgsl/qwen35.js`):
1. **Too few workgroups at short and medium context.** The split length is fixed at 256 up to 32K, so the grid is `splits × nKV`: **8 workgroups at 1K on the MoE**, 32 at 4K, and 16 at 1K on the 27B, against 48 SMs.
   - Choose the split length as a deterministic function of `seqLen`, aiming for ≥ 2–4 workgroups per SM.
   - Spec == plain still holds, because every pass at a given position picks the same splits. Split rooms are also unaffected, since only K/V rows cross devices.
2. **Uncoalesced K reads.** Thread t reads its own K row serially (`u32` by `u32`, 128 iterations), so each warp-wide load touches 32 cache lines. Use 8 lanes per K row with `vec4<u32>` loads and a fixed 3-level shuffle or shared tree.
3. **Serial online softmax.** Only G threads (6 or 8 of 256) do each chunk's max, exp and sum, serially over 64 positions, between barriers. Give each head 32 lanes with a tree max and sum.
4. **Prefill re-reads KV.** `attn_flash_t2` reads K/V once per 2 columns, so a 16-column chunk reads the whole cache 8 times.
   - For a 32K MoE prompt, the KV re-reads alone are ≈ 8 × Σ_c(16c × 20 KB) ≈ 5.5 TB, **about 30 s at the probe**. For a 16K prompt on the 27B it is about 24 s [E].
   - A `t4` tile (the 32 KB `maxComputeWorkgroupStorageSize` most desktop adapters expose, with q kept in f32) cuts that in half and is still bit-identical to one column. A full 16-column tile needs q in registers and is prefill-only.

**Exactness.**
- Items 2–3 change the order of the dot products and softmax sums. That is a numerics change, applied uniformly to decode, verify and prefill, so spec == plain and split == solo hold.
- Item 1 alone also changes bits relative to today (different split boundaries).
- Item 4's `t4` tile is bit-identical.
- Bundle 1–3 into one re-baseline.

**KV quantization.** `kvQ8` saves 44% of KV bytes and memory; `q8_0`'s f16 scale instead of f32 would make it 47%. It is not bit-identical to f16, but it is consistent across pass widths. Published quality for Qwen 3.6 [W: tabby-2026-09.md §3]: q8_0 KL < 0.04; q4 hurts long documents (KL 0.58) and tool calls.
- Recommendation: after `bench_ctx.js` and a long-document KL check, **turn on `kvQ8` by default for the MoE at ≥ 32K** (−1.5 to −3 ms per token at 32–64K). Keep f16 below that.
- Do not use q4 K.

**Expected gain.** Unknown until measured. If today's kernel reaches half of roofline at 32K (plausible given items 2–3), the rewrite is worth −3 to −4 ms per MoE token at 32K (+15%) and a large share of long-prompt prefill time.
- Effort: medium.
- **Measure first.** Run `bench_ctx.js` at `FILLS=1024,4096,16384,32768`, plus `prof_ts` on `attn_flash`.

## 7. Overlapping CPU encoding with the GPU

WebGPU has no reusable compute command buffers. Render bundles are render-only, and a `GPUCommandBuffer` can be submitted once. So the options are:
- **Encode-ahead** (D3 of the previous document, not built; `forwardToken` encodes each token from scratch). Encode token N+1's command buffer while the GPU runs token N.
  - Every per-token input is already a buffer (`frame` via `writeBuffer`, which is queue-ordered, and the embedding row). Bind groups are prebuilt.
  - The only shape that changes is `attn_flash`'s x-dimension, `ceil(seqLen / split)`. It is known one token ahead, or can come from `dispatchWorkgroupsIndirect`, or the dispatch can always be the maximum (idle workgroups return at once).
  - Invalidate on reset, rollback, cancel or session switch.
  - [M]: −7.45 ms per token on the 27B in Deno (`bench_pipe_ab2.js`).
  - In Chrome, the CPU encode it hides is about 3 ms per MoE token (502 × ~6 µs) and about 5 ms on the 27B (898 dispatches): **MoE plain +10–13%, 27B +4–5% [E]**.
  - The same applies to the specFuse step: pre-encode the K = 3 verify step while the previous one runs. The step shape is fixed for solo K = 3; lookup steps fall back to encoding in line.
  - **Bit-identical.** Effort: low to medium.
- **Chunked submit** (Chrome only, a hypothesis to A/B). Dawn records the backend command buffer when `submit` is called. A 500-dispatch submit may therefore delay the GPU's start by the GPU process's recording time. Submitting a token as 2–4 command buffers (per 10–20 layers) lets recording overlap execution.
  - Test by comparing Chrome wall time against GPU time with 1 vs 4 submits. Bit-identical.
- **Fewer dispatches.** M-a removes none, but the norm-into-consumer fold does (flagged; see §10 for why it is not recommended). MoE: 502 dispatches today.
- **Sync cost.** Chrome takes 0.76 ms per sync and Deno 11.7 ms [M]. Chrome 145's experimental synchronous buffer mapping in workers ([W](https://developer.chrome.com/blog/new-in-webgpu-145)) could shave part of Chrome's 0.76 ms. Low priority.
- **GPU-resident greedy loop** (larger, exact, greedy only). Compute acceptance on the GPU: compare the verify argmaxes with the drafts, write `a` and the new position into the frame and replay uniforms, and dispatch the refill and replay indirectly. The CPU can then queue step N+1 before reading step N's tokens.
  - This removes the per-step sync and the idle gap: about −1 to −2 ms per step in Chrome (5%) and −12 ms per step in Deno [E].
  - Effort: high. It pays mostly in Deno and in Code mode (greedy).

## 8. Prefill: DeltaNet chunked form, DP4a, subgroup matrix

**Chunked DeltaNet** (spec: `docs/deltanet-prefill-spec.md`). `dn_delta_mc` was about 11% of a 16-column 27B pass before the register-resident rewrite [M: kernels-2026-09.md], which made it 2.03× faster at 16 columns [M]. That leaves about 6% of the pass, so the chunked form buys ≤ 5% at NC = 16.
- llama.cpp's chunked GDN kernel measured +5% to +19% end to end for Qwen3.x-27B prefill ([W: PR #26001](https://github.com/ggml-org/llama.cpp/pull/26001), [#29353](https://github.com/ggml-org/llama.cpp/pull/29353)), at ubatch 512, where the serial recurrence hurts more.
- **Build it only together with larger MoE prefill chunks (§4)**, where a 64-column serial recurrence becomes the bottleneck.
- Tolerance (prefill-only).

**DP4a with Q8_1 activations in the prefill GEMM.** `dot4I8Packed` has been in Chrome since 123. It is the broadly available integer path, emulated on Apple. As in llama.cpp's MMQ, activations are quantized to Q8_1 per 32 and dotted with `dot4I8Packed`.
- The earlier estimate is ×1.3–1.4 on the prefill matmuls on NVIDIA, AMD and Intel.
- Prefill-only (tolerance). **Never in decode or verify**: it would break spec == plain unless decode used it too, and decode is bandwidth-bound anyway.

**Subgroup matrix.** Status as of September 2026:
- The `subgroup-matrix` proposal is still a **Draft** in gpuweb ([W: proposal](https://github.com/gpuweb/gpuweb/blob/main/proposals/subgroup-matrix.md)); in Chrome it is `chromium-experimental-subgroup-matrix`, behind a flag.
- Backends:
  - Metal maps it to `simdgroup_matrix` 8×8 in f16 and f32, Apple7+ only.
  - Vulkan maps it to `VK_KHR_cooperative_matrix`, with configurations queried through `adapter.info.subgroupMatrixConfigs`.
  - The D3D12 mapping was specified only this month ([W: gpuweb PR #10964](https://github.com/gpuweb/gpuweb/pull/10964), 2026-09-22).
- ONNX Runtime's WebGPU EP ships f16 16×16×16 subgroup-matrix kernels, extended to AMD RDNA3+ on D3D12 on 2026-09-17 ([W: ORT PR #32667](https://github.com/microsoft/onnxruntime/pull/32667)).

Related Chrome features:
- `subgroups` has shipped since 134 ([W](https://developer.chrome.com/blog/new-in-webgpu-134)).
- `subgroup-size-control` (`@subgroup_size`) shipped in 151–152 ([W](https://developer.chrome.com/blog/new-in-webgpu-151-152)).
- `shader-f16` has shipped since 120 ([W](https://developer.chrome.com/blog/new-in-webgpu-120)).
- "Immediates" (push constants) shipped in 149–150 ([W](https://developer.chrome.com/blog/new-in-webgpu-149-150)).
- WebGPU on Linux NVIDIA arrived in 147–148 ([W: news index](https://developer.chrome.com/docs/web-platform/webgpu/news)).
- Safari 26 ships WebGPU on macOS and iOS ([W](https://webkit.org/blog/17333/webkit-features-in-safari-26-0/)). Feature databases report `shader-f16`, `subgroups` and `timestamp-query` on Mac Safari; confirm this on the device, since `docs/kernels.md` still says subgroups are absent on Safari.
- Deno/wgpu on the GB10 exposes no subgroups.

**Recommendation.**
- Build a prefill-only subgroup-matrix Q4 GEMM (dequantize a tile to f16 in workgroup memory, then 8×8 on Apple or 16×16 elsewhere, with f32 accumulation where the configuration allows). Feature-detect it, keep it off by default, and place it after §4 and DP4a.
- It is the route to llama.cpp-class prefill (2520 tok/s pp512 for the MoE [M]), but only on Chrome-with-flag today, and every device in a room must agree on the setting.
- Tolerance (prefill-only). It is irrelevant to decode: N = 1 wastes 7/8 of an 8×8 tile, and decode is bandwidth-bound.

**`shader-f16` for storage and math.** f16 accumulation corrupted output [M: `docs/kernels.md`], and f16 activation storage measured 0% [M]. Only f16 *loads* of K/V (already f16) and f16 tiles inside a prefill GEMM make sense.

## 9. Speculative decoding

Measured [M: bench-log 2026-09-26]:
- Acceptance: MoE 28/33 and 28/39; 27B 28/33 and 26/39.
- Speedup over plain: MoE 1.6–1.9×; 27B 2–2.7×.

On the 27B a wider verify is nearly free, because the batched GEMV reads each weight once. On the MoE every verify column costs about 0.7 GB of experts until §4 lands.

| Idea | Interaction with DeltaNet and replay | Verdict |
|---|---|---|
| **Adaptive draft length per step** | Drafts only; the replay rollback already handles any accepted prefix. | **Build.** The draft-chain argmax kernel also writes the top-1/top-2 logit margin of each draft; the chain stops, or the verify drops columns, when the margin falls below a threshold. The CPU learns the drafts at the one existing readback. Pays most on the MoE (expensive columns): +5–10% spec [E]. Output **bit-identical** (drafts only). Low effort. |
| Suffix / n-gram drafts chained after MTP drafts | None beyond today's lookup path | Already on the roadmap (long-context doc "Next" #6); keep it there. |
| Tree drafts (top-2 at depth 1–2) | Each branch needs its own recurrent state. Workable form: the GEMVs run once per tree node (the projections are per token), and `dn_delta` runs **per root-to-leaf path** from the shared S (paths in parallel workgroups; a node's output is owned by its first path). The conv takes parent pointers. Attention needs a tree mask and **K/V written to scratch rows, gathered in logical-position order** so the split and sum order equals plain decoding. Replay generalizes to "replay this column list". | Exact is possible, but it touches conv, attention, KV and replay. With MTP at about 85% per-token acceptance at K = 3, a tree adds maybe +10–15% tokens per step [E]. **Later**: only worth it on the 27B, after encode-ahead. |
| Multi-MTP | Qwen ships one MTP layer (`mtp_num_hidden_layers = 1`); more heads need training. | No. |
| Lookahead / Jacobi | Every Jacobi iteration advances DeltaNet state speculatively and needs rollback. Its n-gram pool overlaps prompt lookup. | No. |
| Cost-aware drafts (EcoSpec) | Only with top-k drafts or trees | After §4, if trees ever land. |

## 10. What llama.cpp does differently, and what transfers

| llama.cpp (CUDA / Metal) | Here | Transfer? |
|---|---|---|
| MMVQ: whole quant blocks per thread, x pre-quantized to Q8_1, int8 dot (`dp4a`) ([W](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda)) | f32 x, 4 B loads, f32 FMA | Wide loads yes (§5, exact). Q8_1 x no for decode (breaks spec == plain and does not help a bandwidth-bound GEMV); yes for prefill (§8). |
| `mul_mat_id`: batch tokens grouped by expert, each expert's rows read once | Per (column, slot) | Yes: §4. |
| topk-moe fusion (softmax + top-k + normalization in one kernel) ([W](https://am17an.bearblog.dev/new-post/)) | Already `moe_route`, also parallel | Done; §3 M-a makes it faster. |
| Fused weighted expert reduction ([W: PR #25952](https://github.com/ggml-org/llama.cpp/pull/25952)) | `moe_dnc` epilogue | Done. |
| rms_norm + mul (+ add) fusion, rms_norm + scale ([W: PR #29393](https://github.com/ggml-org/llama.cpp/pull/29393)) | `rmsnorm` already multiplies by the weight; the add is folded into `_acc` GEMVs | Done in substance. Folding the norm into the consuming GEMV was 0.91× on Metal [M]. |
| GEMV + GLU fusion in MMVQ | `_gu` kernels, `moe_gus` | Done. |
| CUDA graphs plus concurrent streams for Q/K/V ([W](https://am17an.bearblog.dev/new-post/); +17–27% combined on gpt-oss) | One submit per token; no reuse possible in WebGPU | Encode-ahead and chunked submit (§7) are the WebGPU equivalents. Q/K/V are already one merged GEMV (`fuseProj`). |
| Flash-attn vec kernels: split-K over the sequence with a stream-K fix-up, one warp per KV block, half2 dots | `attn_flash` with fixed splits, serial softmax, uncoalesced K | Yes: §6. |
| GDN op: warp per state column with rows split across lanes ([W: PR #19504](https://github.com/ggml-org/llama.cpp/pull/19504)); chunked prefill | Register-resident, one thread per column | Lane-split changes the sum order (flag, D7 of the previous document). Chunked prefill: §8, later. |

## 11. What not to do (low expected value)

- **One kernel for gate→up→down across the top-8.** It needs cross-workgroup accumulation, which is either nondeterministic (CAS atomics) or costs as much traffic as the weights (§3 M-b).
- **Subgroup reductions in the decode GEMV tree.** Measured 0–8% [M]. The bits differ between Chrome and Deno/Safari, so every device in a room would need the same setting.
- **Q8_1 or DP4a activations in decode or verify.** Decode is bandwidth-bound, and it breaks spec == plain.
- **Subgroup matrix for decode.** N = 1.
- **More `moe_gus` tuning.** It is already at about 84% of the probe. `moe_dnc` (59%) and the small kernels are where the time is.
- **Norm folded into the consuming GEMV** (0.91× on Metal [M]), f16 accumulation (corrupts output [M]), and f16 activation storage (0% [M]).
- **q4 KV** (quality on long documents and tool calls), multi-MTP, lookahead/Jacobi, EAGLE heads (no trained heads), expert parallelism across devices for decode (`offload-2026-09.md`: 80 network crossings per token).
- **Render-bundle-style tricks for compute.** They do not exist in WebGPU; pre-encoding is the substitute.

## 12. Measurement protocol additions

- **S0:** calibrate `prof_ts.js` by timing an empty dispatch in its own pass and subtracting it. Profile a whole specFuse step, not only `forwardToken`. In Chrome, profile with the WebGPU Developer Features flag.
- Log a `bench_ctx.js` row (MoE and 27B, fills 1K / 4K / 16K / 32K) **before** touching attention.
- Log the per-layer expert union size from `moe_group` (§4) on the `japan` prompt and on a code prompt.
- For every bit-identical item: `tests/run.sh quick`, `q38`, `test_twins.js`, `tests/test_moe.js` (spec == plain, also with `DRAFTCHAIN=0`, `SPECFUSE=0`).

## Top 5, in priority order

1. **Exact latency fixes for the serial small kernels** (`dn_pre` staging, `moe_route` unrolled integer ranking, literal-unrolled `rmsnorm`, column-split `dn_delta`) after calibrating `prof_ts`. MoE −2 to −3 ms per token (+10–15% plain); 27B +2–3%. Bit-identical.
2. **Expert-grouped MoE verify and prefill** (`moe_group` + grouped gus/down with per-column accumulators in today's order), plus pinned `GEMM_S` for the MoE's 2048-wide shapes. MoE prefill 2–3× at 16 columns; MoE spec +10–20%. Bit-identical if the per-column order is kept; the GEMM entries are tolerance (prefill-only). Fix the open `moe_synth` batched == single failure first.
3. **Encode-ahead (plus the chunked-submit A/B) for plain decode and the specFuse step.** Chrome: MoE plain +10–13%, 27B +4–5%; more in Deno (−7.45 ms per token measured). Bit-identical.
4. **16-byte whole-block loads**: the dense coop GEMV ladder (D4, exact form) and the wide layout in the fused `moe_gus` / `moe_dnc`. 27B plain and spec +6–11% (llama.cpp streams the same file at ~208 GB/s against our 184); MoE −1 ms per token. Bit-identical for the dense ladder, subject to a per-backend golden check; numerics change for the MoE kernels (MoE goldens are text).
5. **Long-context attention rework, after measuring with `bench_ctx.js`**: split count by `seqLen`, coalesced vec4 K reads, a parallel softmax, and a bit-identical `t4` prefill tile, plus `kvQ8` by default for the MoE at ≥ 32K after a KL check. MoE at 32K −3 to −4 ms per token (+15%); long-prompt prefill KV reads halved. Numerics change (spec == plain and split == solo preserved); the `t4` tile is bit-identical.

Next in line:
- Adaptive draft length from the draft margin (spec +5–10% on the MoE; bit-identical output).
- BF16 routers (bit-identical, −0.25 ms) and a native Q6_K LM head (−0.67 ms; numerics move closer to llama.cpp).
- DP4a prefill GEMM, then subgroup matrix behind a flag.
