# The WebGPU engine: kernels, tricks and hacks

Everything here was measured on a DGX Spark (GB10, 184 GB/s achievable through WebGPU) and the maintainer's MacBook. Every row of the table below has a commit and a number in [bench-log.md](bench-log.md). Output is bit-identical to the reference at every step unless a row says otherwise.

## Where the time goes

A generated token on the 27B = ~111 ms on the GB10: **82 ms streaming 15 GB of weights through the matvec kernels** (183 GB/s: at the roofline) and ~30 ms of everything else (small kernels, command encoding, submit, logits readback). Prefill passes cost ~139 ms for 4 tokens. Measured with `benchmarks/bench_breakdown.js`, which re-times passes with one kernel family skipped at a time. These numbers predate the prefill GEMM, flash attention and the fused kernels, and the profiler's family lists need rebuilding before they are re-measured.

## Kernel families

| Family | Entry points | Role |
|---|---|---|
| Cooperative GEMV | `matvec[_q8|_q4]_coop[_acc]` | decode: `ROWS` rows per workgroup of `WG` threads; 64 threads sweep one row together |
| Batched GEMV | `matvec*_coop_b[8|4][_acc]` | speculative verify and prefill tails: `_b8`/`_b4` twins for ≤8 / ≤4 live columns, each with rows-per-workgroup chosen to keep 16 accumulators per thread |
| Fused gate/up | `matvec*_gu[_b[8|4]]` | FFN gate and up in one weight sweep, SiLU applied in the epilogue (GEMV path only) |
| Prefill GEMM | `gemm_q4_<dIn>_s<S>`, `gemm_red_s<S>[_acc]`, `gemm_xpose` | row-stationary Q4_0 GEMM for 16-column prefill passes: packed-nibble weight tile in shared memory, broadcast activations, pinned split-K, fixed-order reduce |
| Wide prefill GEMM | `gemm_w_q4|q8[_acc]`, `silu_mul_w` | tiled f32 GEMM over a whole 256-token chunk (engine/wgsl/gemm_wide.js); default for any engine holding the embedding. `silu_mul_w` is also the 16-column pass's one-launch SiLU |
| DP4a prefill GEMM | `quant_q8_w`, `gemm_d_q4|q8[_acc]` | the wide chunk's projections as `dot4I8Packed` on activations quantized per 32 values (llama.cpp's MMQ numerics); the dense default on NVIDIA (`DP4A_DEFAULT.dense`; ~1.8x the f32 wide GEMM on the 27B, GB10) since it passed the pre-registered accuracy evaluation `tests/eval_dp4a.js` (KL / top-1 vs llama.cpp, downstream); the MoE default on NVIDIA too since it passed the same evaluation (`DP4A_DEFAULT.moe`, with the dp4a expert kernels below), `prefillDp4a: false` / `?dp4a=0` turns it off; never Apple |
| Expert-grouped MoE prefill | `moe_gsort`, `moe_gusg|moe_dng_<fmt>` (f32 tiled), `moe_qx` + `moe_gusq|moe_dnq_<fmt>` (dp4a), `moe_combw` | the ubatch's (token, slot) pairs sorted by expert (one workgroup, atomic histogram + per-column bitmask placement, stable), each expert's rows read once per chunk of up to 8 of its pairs; inside wide chunks the gate/up and down run as `dot4I8Packed` on activations quantized per 32 values with the weight blocks kept as packed i8 in registers (`moeGroupDp4a`, follows `DP4A_DEFAULT.moe`; 1.9x the f32 tiled kernels); the wide chunk (ubatch 512 on the MoE) is the expert group |
| Glue | `rmsnorm`, `qsplit`, `head_norm`, `rope_part`, `attn_scores/softmax/out`, `sigmoid_mul`, `silu_mul`, `add_res`, `argmax` + `_mc` multi-column forms | normalization, attention, activations, sampling prep |
| DeltaNet | `dn_pre` (gates + L2), `dn_conv`, `dn_delta`, `dn_gatenorm` + `_mc` forms | the gated delta-rule recurrence, with snapshot slots for speculative rollback |
| QSA (Qwen3.8-Flash-Next) | `qsa_proj`, `qsa_prep`, `qsa_pool`, `qsa_score[_c4]`, `qsa_select`, `attn_dec_qsa` + `_combine[_g]` | sparse attention for the 12 full-attention layers: indexer, pooled block keys, exact top-512 block select, attention over the selected KV rows ([QSA](#qsa-qwen-sparse-attention) below) |
| Legacy | `matvec[_q8|_q4]` | one thread per row; kept as the correctness reference and fallback |

## The tricks, with what each one bought

### Reading weights
1. **Cooperative rows (2.2× on the 27B).** One thread per row gives scattered memory traffic. Instead 64 threads sweep a row with interleaved 32-weight blocks, so every cycle the group touches one contiguous stripe and the GPU coalesces it. Partials merge in a shared-memory halving tree with `workgroupBarrier()` *outside* the `if`.
2. **Scalar accumulators (the 3× bug).** Four running totals in a `array<f32, 4>` indexed by a variable spilled to scratch memory and made the "fast" kernel 3× slower than the naive one. Named scalars `acc0..acc3` fixed it. Corollary: all unrolled code is generated with literal indices; a WGSL `for` over a constant bound does not unroll on wgpu.
3. **Dequantize in registers.** Nibbles are extracted from packed `u32` words and `(nibble − 8) × scale` happens in registers; full-precision weights never exist in memory. The scale multiplies each block's partial sum once, not each weight.
4. **f16 block scales end to end (~10% fewer bytes).** GGUF's native f16 scales stay packed two per `u32` and are decoded with `unpack2x16float`; requantized tensors round scales to f16 first, matching the spec.
5. **Load-time repacking.** The file interleaves `[scale][nibbles]`; at load they are split into a nibble array and a scale array so trick 1's stripes are contiguous. Layout, not compression.
6. **`unpack4xU8` / `unpack4xI8` dequant (neutral on NVIDIA, kept for Metal/Android).** One instruction per four packed bytes instead of shift/mask/convert per element; a runtime probe compiles a one-line shader and falls back to the shift form where the builtins are missing.
7. **vec4 aliased binding views.** The same buffer is declared as `array<u32>` for one entry point and `array<vec4<f32>>` for another; legal as long as no single entry point references both views.
8. **Device autotune (~1 s at load).** Times `(WG, ROWS)` candidates on the actual GPU and keeps the winner with a 3% noise guard. The GB10 prefers (64, 4).

### Fewer passes, fewer bytes moved
9. **One command submit per token.** All ~900 dispatches of a decode token (1,232 for an 8-column verify pass) are encoded into a single command buffer.
10. **Fused gate/up + SiLU.** Both FFN projections read the same input, so one kernel computes both and applies `silu(g)·u` before writing. Per-buffer helper functions are generated instead of storage-pointer parameters, which are unsafe on Safari. Neutral on the GB10 (bandwidth-bound), aimed at Metal's dispatch cost.
11. **Accumulate-into-residual matvecs (`_acc`).** `y[row] += W·x` folds the residual add into the o-proj, DeltaNet out-proj and FFN down-proj: 128 fewer dispatches per token (two `add_res` sites per layer). Measured neutral on Vulkan, which taught us small dispatches are nearly free there.
12. **Fused DeltaNet pre-pass (`dn_pre`).** Gates (sigmoid β, decay) and the per-head L2 norms of q and k run in one dispatch after the causal conv.
13. **GPU argmax.** A single-workgroup reduction over the 248,320 logits with lowest-index tie-breaking (matches the CPU loop exactly) so the draft chain reads back 8 bytes instead of 1 MB per draft.

### Batching
14. **Batched prefill (3 → 39 tok/s).** 4 (now up to 8) prompt tokens per GPU pass as columns; each weight block is read once for all columns. Causality inside a pass: recurrent kernels process columns strictly in order; K/V are appended before attending; per-column position uniforms.
15. **Multi-column glue kernels (`_mc`, 2× on batched passes).** One dispatch covers all live columns instead of one dispatch per column.
16. **4-column twin kernels.** With 8 batch columns, a 4-token verify would pay for 4 garbage columns of ALU; a second kernel set generated for 4 columns is chosen per dispatch by live column count.
17. **8-wide columns were neutral for prefill** (an 8-column pass costs 2× a 4-column pass), which is how we learned prefill is not weight-streaming-bound and needs a real GEMM.

### Speculation
18. **Multi-token prediction with exact rollback (9 → 16 tok/s).** The model's `nextn` layer drafts up to 7 tokens; one batched trunk pass verifies them; DeltaNet states are snapshotted after every non-final column into per-layer shadow slots (7 of them), written inside the recurrence kernel; a rejection copies the right slot back. KV caches need no rollback (position-capped). Any sampler stays lossless because the trunk decides.
19. **Snapshot slot base packed into the frame uniform.** `frame.snap` carries `(total << 8) | (base + 1)` so an 8-column verify split into chunks on any device writes global slot indices.
20. **Draft depth by measured throughput.** A lap-time threshold locked a Mac-hosted room at depth 7 (1.5 tok/s); the room now probes depths 3/5/7 and keeps the fastest measured tokens per second.

30. **Row-stationary prefill GEMM (1.64× on the whole-model pass, 34.6 → 56.7 tok/s).** The batched GEMV is column-stationary and re-reads the activation tile per row group; at 16 columns it is worse than at 8. The GEMM gives each thread R rows × all 16 columns, keeps the weight tile as packed nibbles in shared memory (4 KB, never dequantized there), broadcast-reads activations from a transposed staging copy, and prefetches the next block pair into registers. Split-K (pinned per shape, so every peer computes identical hidden states) keeps tall tiles occupied; partials reduce in fixed order. It fires only at the full batch width, so decode and speculative verify never touch it. `ffn_down` and `ssm_out` are Q8_0 in the 27B and stay on the GEMV: a Q8 variant is the next step.
31. **Width-dependent rows per workgroup for the twins.** With 16 batch columns the narrow twins would inherit one row per workgroup and speculative decode fell 15.5 → 12.8 tok/s; each twin now uses `16 / C` rows so every batched kernel carries 16 accumulators per thread. A twin is byte-identical to the kernel generated natively at its width (checked in `tests/test_twins.js`).

### Correctness and portability hacks
21. **2-D dispatch for tall matvecs.** WebGPU silently drops a dispatch over 65,535 workgroups in one dimension; the LM head at 2 rows per workgroup needs 124,160. Kernels derive `row0` from `(wg.y * 32768 + wg.x)`.
22. **No f16 accumulation** (corrupted output); **no timestamp queries** (3× slowdown when merely enabled); **no subgroups** (absent on Safari, hidden on Deno; measured worth 0–8% anyway).
23. **Feature probes at load** for `unpack4x` and the WGSL extensions, compiled as one-line shaders with `getCompilationInfo()`.
24. **Generated helper functions per buffer** instead of `ptr<storage>` parameters (Safari).
25. **Bank-conflict padding** in the GEMM prototype's shared tile (row stride 17 vec4s instead of 16): the only change that moved that kernel (1.25×).

### On the wire (the room)
26. **Binary frames (−33%) and f16 activations (−50%)** over WebRTC data channels; decoders accept f32 from older peers.
27. **16 prompt tokens per network round** (4 GPU passes per round) so prefill pays one round-trip per 16 tokens.
28. **Cache API with size stamps**: the cache refuses 206 responses, so range responses are stored as 200 with an `x-swarm-len` stamp and validated on read (the header keeps its old name so caches from before the rename stay valid).
29. **Fire-and-forget prefill dispatches** with periodic `onSubmittedWorkDone()` syncs to keep the queue from growing unbounded.

## QSA (Qwen Sparse Attention)

Qwen3.8-Flash-Next's 12 full-attention layers (and the MTP block) attend to a selected subset of the KV cache. Kernels: `engine/wgsl/qsa.js` and the gather mode of `engine/wgsl/attn_dec.js`; host side (sizes, pass order, dispatch, bind groups): `engine/qsa.js`; CPU reference written from llama.cpp (file:line citations): `tests/reference/qsa_ref.mjs`; tests: `tests/test_qsa.js` (GPU), `tests/unit/qsa_ref_test.js`; timings: `tests/bench_qsa.js`. Not wired into `engine/qwen35.js` yet.

**What it computes.** For the token at position p (0-based within the sequence), on the layer input x:
1. raw indexer key `kr[p] = index_k_proj · x` (128), cached with no norm and no rope;
2. blocks are positions 4b .. 4b+3. When the last one is written, the block's pooled key is `P[b] = rope(rmsnorm(((kr[4b] + kr[4b+1]) + kr[4b+2]) + kr[4b+3]) * 0.25) * k_norm, 4b)`, cached once (rope: NeoX, first 64 of 128 dims, theta 1e7, at the block's first position);
3. indexer query `qi = rope(rmsnorm_head(index_q_proj · x) * q_norm, p)`, 4 heads x 128;
4. for the nv = floor((p+1)/4) complete blocks: `s[b] = Σ_h relu(qi_h · P[b]) / sqrt(128)`;
5. keep the top min(nv, 512) blocks by (score descending, block index ascending), plus the tail rows 4nv .. p (the token's own unfinished block, itself included): at most 2051 rows;
6. the layer's ordinary GQA attention (24 q heads, 2 kv heads, hd 256, scale 1/16, then the sigmoid gate) over those rows only.

With p + 1 ≤ 2051 every block is kept, so QSA *is* dense causal attention; past that it differs (llama.cpp's PR #27742 reports 3% of positions diverging at 8K under dense). A prefill token selects exactly what a decode step at its position would. The GGUF ships `indexer.{q,k}_proj` as BF16 and `indexer.{q,k}_norm` already folded to (1 + w) (llama.cpp's converter adds the 1; the bartowski Q4_0 file's norms average ~0.96).

**Ties.** llama.cpp leaves them undefined: `ggml_top_k` is "unordered" (the CPU op is a `std::partial_sort`, CUDA's radix/CUB paths are explicitly non-deterministic). They are common: a block whose four head dots are all ≤ 0 scores exactly 0. We keep the lower block index, as `engine/topk.js` and ds4/vmlx do, in the reference and on the GPU.

**State per sequence and QSA layer** (besides the K / V cache): `rawK` [maxSeq][128] f16 and `pooled` [maxSeq / 4][128] f16, 256 B plus 64 B = 320 B per position against the layer's 2,048 B of f16 K + V (2 kv heads x 256 x 2): +15.6% (+12.5% of it is `rawK`, which a ring would cut to ~3%, open item 4). llama.cpp keeps both in its F16 idx cache (`type_k`), so the f16 rounding matches it. Rollback needs nothing extra: rewinding to position r and re-decoding rewrites `rawK` from r and re-pools a block when its last position is written again; a pooled key of an unfinished block is never read (nv only counts complete blocks). Scratch per pass, shared by all layers: `proj` (cols x 640), `qi` (cols x 512), `scores` (cols x maxSeq / 4 f32), `sel` (cols x 2052 u32: count, then rows). `scores` is the one that grows: 512 columns at 262,144 positions is exactly 128 MiB, WebGPU's default `maxStorageBufferBindingSize`. There is no column-group path (every kernel covers `frame.nCols` columns), so the engine caps the pass width at `qsaMaxCols({ maxSeq })` or requests the adapter's limit. `qsaSizes` also refuses maxSeq above `QSA_MAX_SEQ` (411,000), where the rope reduction stops being exact (open item 3).

**Kernels and pass order** (one pass = nCols consecutive positions from `frame.pos`; group 0 is the engine's `[cfg, frame]`):

| Kernel | Dispatch | Does |
|---|---|---|
| `qsa_proj` (optional) | (80, cols) | `proj = [index_q_proj ; index_k_proj] · x`, BF16 weights as u32 pairs. The engine may instead append these 640 rows to its own fused projection |
| `qsa_prep` | (5, cols) | q heads: rmsnorm · q_norm, rope → `qi`; head 4: raw key → `rawK[p]` (f16) |
| `qsa_pool` | (blocks touched, 1) | pooled key of every block this pass completes |
| `qsa_score` / `qsa_score_c4` | (nv/256, cols / CW) | scores of the visible blocks; skipped while no column sees more than 512 blocks. Same bits for CW 1 and 4 |
| `qsa_select` | (cols) | per column, one workgroup: 11/11/10-bit radix select of the 512th key (scores ≥ 0, so their bits order them), then a block-order compaction with a workgroup scan; writes the ascending row list. Exact, deterministic, no sort |
| (engine) | | store this pass's K / V rows, as for dense attention |
| `attn_dec_qsa` + `attn_dec_qsa_combine[_g]` | (splits, cols, nKV·R), (nH·4, cols) | `attn_dec` with list position t reading KV row `list[t]`; split lengths come from the list length |

Integration points for the Flash path: per QSA layer call `prep` and `pool` on every pass from position 0 (the state must be complete before the context passes 2051); `score`, `select` and `attn` replace the layer's attention. Below 2052 positions the engine can keep its dense kernels: `attn_dec_qsa` with the full list gives `attn_dec`'s bits (tested), and a dense prefill chunk (`attn_tile`) is the same math. Decode uses the engine's `adCfg` (S = 64 splits, R = 2 for 24 / 2 heads); for wide prefill chunks pass S = 4 in the FD uniform so the split partials stay at cols x 24 x 4 x 256 floats (50 MB at 512 columns). `qsaDispatch` gives every size from (pos, nCols). MTP: the paper reuses draft step 0's selection for later draft steps; `sel` can simply be kept.

**Tests** (`tests/test_qsa.js`, GB10): `qsa_select` equals the reference rule exactly on 28 score sets (random with ~50% zeros, five-value ties, all equal, all zero, a tie run across the boundary, 1-ulp steps and -0, every tail length, up to 65,536 blocks). End to end at 512, 2051, 2052, 4096, 8192 and 32768 positions (decode and a prefill chunk ending there, random data, plus real blk.3 indexer weights at 4100): row lists equal to the CPU reference's, the GPU list equals the reference rule applied to the GPU's own scores, attention within 3e-7 of float64; ≤ 2051: bit-equal to `attn_dec` (decode and multi-column); a prefill column is bit-equal to a decode step at its position; two runs give the same bits; all-zero scores pick blocks 0 .. 511; a rewind re-pools correctly. Scores agree with the reference to ~5e-5 relative (f16 pooled keys round differently in ~0.05% of values), so a GPU list could differ from a CPU one on a near tie; none did. `rawK` and the pooled keys are rounded to f16 by hand, to nearest even (`engine/wgsl/f16.js`), not with `pack2x16float`: WGSL leaves its rounding to the backend, and D3D12 (Dawn/Tint and wgpu/naga, RTX 5070) picks the other neighbour for ~30% of inputs, which moved 59% of pooled values, scores by 5e-4 and near-threshold choices in 3-4 of 32 prefill columns. On Vulkan / Metal the hand rounding gives the same bits as the builtin. The test's `f16` section checks the helpers on the GPU and reports how the backend's builtins differ. Passes on GB10 Vulkan and on the RTX 5070 under Dawn D3D12 and wgpu dx12.

**Timings** per layer (`tests/bench_qsa.js`, GB10, random data; µs). "Stage sum" adds the five stages, each timed alone; "one pass" runs them in order as the engine will, so it also pays the gaps between dependent dispatches (15-30% more here). One pass is the figure to compare with dense. The stage columns are from a quiet GPU; the one-pass comparison was rerun while another process held the GPU at ~95%, which inflates every absolute time 2-3x, so read its ratio rather than its µs.

| context | decode stage sum (prep, pool, score, select, attn), quiet GPU | dense `attn_dec`, quiet GPU | shared GPU: one pass / stage sum / dense | one pass vs dense | prefill, per token (512-col chunk; 64 above 32K), quiet GPU |
|---|---|---|---|---|---|
| 4K | 92 (6, 6, 9, 21, 49) | 70 | 199 / 168 / 149 | 0.75x (slower) | 17.9 |
| 8K | 94 (6, 6, 10, 23, 50) | 94 | 204 / 174 / 205 | 1.0x | 18.3 |
| 32K | 106 (6, 6, 11, 35, 48) | 779 | 242 / 185 / 1,632 | 6.7x | 20.8 |
| 64K | 138 | 1,435 | 299 / 240 / 2,949 | 9.9x | 25.7 |
| 128K | 194 | 2,811 | 439 / 368 / 5,646 | 12.9x | 32.8 |
| 256K | 300 (7, 7, 53, 183, 51) | 5,594 | 779 / 607 / 9,994 | 12.8x | 46.9 |

Run as one pass, decode is slower than dense at 4K, breaks even at ~8K, and is ~7x faster at 32K and ~13x at 128K-256K (the stage sums alone suggested 7x and 19x). Below ~8K the engine should keep dense attention for decode (it is exact there up to 2,051 positions anyway). Prefill is dominated by the attention (8.8 ms per 512 columns, ~17 µs per token): each column runs the decode kernel over its own ~2051 rows, so K / V rows are re-read per column (~4 GB of L2 traffic per chunk). At long context a 512-column chunk turns score-bound: at 256K on the shared GPU one chunk was 75 ms, of which score 37 ms and select 9 ms (attention 30 ms), ~147 µs per token; scoring grows with columns x blocks, so at 256K a 64-column chunk (the 46.9 above) or `qsa_score` work shared across columns is the better shape. The indexer projection through `qsa_proj` costs 17 µs per decode token and 2 ms per 512-column chunk; folding it into the engine's prefill GEMM is the cheaper route.

**Open items.** (1) Prefill attention: a tiled kernel over the union of a few adjacent columns' lists (per-column masks) would share K / V tiles as `attn_tile` does. (2) Decode select is one workgroup per column (183 µs at 256K); a multi-workgroup histogram, or ds4's tile-maximum prefilter fused into `qsa_score`, would cut it. (3) The engine's own `rope_part` takes `cos(p · pow(...))` directly, which drifts from ggml by ~1e-3 rad at p ~ 1e5; the QSA kernels use ggml's f32 theta product and a Cody-Waite reduction, the main attention's q / k rope does not. That reduction (`qsa_cs`, 2 pi in three parts) is exact only for k = round(p / 2 pi) < 2^16, i.e. p < ~411K (`QSA_MAX_SEQ`, enforced by `qsaSizes`; the GGUF's context is 262,144), and only if the backend keeps the three subtractions in order: Metal's fast-math may reassociate them, and only Vulkan has been tested, so a Metal run must include `tests/test_qsa.js`. (4) `rawK` only needs the last few positions once their block is pooled (vLLM keeps a ring of `4·ceil((4 + n_draft)/4)`); a ring would save 256 B per position per layer. (5) Agreement with llama.cpp itself: checked on the Flash path (next section). The 1943-token dense golden (`ctx2k_b`) and the 4K one take another token at step 6, a backend near tie (see "The qwen4exp goldens' pass rule" below): not the selection (`ctx2k_b` never selects, and on real layer-3 activations the GPU's pooled keys, scores and row lists match this reference past 2051 cells), and not a position-dependent engine error. Record which backend produced the goldens: ggml-cpu's BF16 mul_mat rounds the activations to BF16 before `index_q_proj` / `index_k_proj` (its vec_dot_type is BF16), while `qsa_proj` keeps f32 activations, more accurate but not bit-faithful. Against CPU goldens expect near-tie block swaps from that alone; tolerate them, or emulate the BF16 activation rounding for the comparison.

### QSA on the Flash path, and its batched passes

`engine/qwen4exp_qsa.js` wires the kernels above into the Qwen3.8-Flash-Next engine (`engine/qwen4exp.js`, decode) and its batched passes (`engine/qwen4exp_batch.js`, prefill and verify). Both share each QSA layer's `rawK` / `pooled` state. `qsa_proj`, `qsa_prep` and `qsa_pool` run on every pass from position 0. A pass in which every column has at most 2051 cells keeps the engine's dense attention, because QSA selects every cell there. A pass with any column past 2051 cells runs `qsa_score` (only when some column sees more than 512 blocks), then `qsa_select`, then a gathered attention. In the engine's op lists these are ops with `dyn(pos, nCols)`: `_encode` asks each one for its workgroups at the pass's position, or skips it. The gathered attention is `attnDecMcWGSL({ gather })` in `engine/wgsl/qwen4exp_batch.js`, which reads the Flash engine's own K / V cache (f32, or f16 with `kv: "f16"`, below). `attn_dec_qsa` from above does not fit here, because it reads the Qwen3.5 engine's split-K layout. Over the full row list the gathered kernel gives exactly the dense kernel's bits. `engine.qsaAlways = true` forces every pass onto the gathered path, which is how `tests/test_q38flash_batch.js` checks that QSA equals dense below 2052 cells. The room loader still loads without the indexer (`qsa: false`): dense, context <= 2048, and a room checkpoint does not carry `rawK` / `pooled` yet.

The batched kernels are the engine's fused decode kernels plus a column index, and the batched op lists are laid out as `forwardToken`'s (`_halfOps`: each HC combine folded into the next site; on the engine's own weight buffers, `st.w` for the ones it joined or fused). Each column does the same arithmetic in the same order, so a pass of n columns leaves the engine where n decode steps would, bit for bit. `tests/test_q38flash_mc.js` checks this kernel by kernel on random data; `tests/test_q38flash_batch.js` checks it on the model's residual. The kernels with a dependency across tokens walk the columns in order inside one dispatch, with the decode kernel's workgroup shape:
- `dn_conv` (conv + l2) keeps each channel's conv history in registers; `dn_delta` (delta rule + gated norm) keeps each thread's 64 entries of S in registers.
- `ple` runs one workgroup per stream, as decode. Each thread owns its channels in every column, so a column's ring write is seen by the later columns' taps in program order, with no barrier, and a pass longer than the 10-slot ring works. The gated value stays next to the residual add exactly as in the decode kernel (the old split kernel of #348 had to route it through memory to keep the bits).
- `gemv_mc` shares each weight block's load and dequant across 4 columns, with the decode GEMV's lanes per row (`gemvLanes`). The decode GEMV's 4-rows-per-lane-group variant only changes which lanes load x, not any row's sum, so it needs no counterpart; the up projection + mix keeps its 4 rows per lane group, one column per workgroup.
- With the head, the head's site (which folds the last layer's combine) runs on every column; a prefill pass then copies the last column's normed residual to column 0 and runs the rest of the head on that column only.

Verify passes copy the recurrent state (conv, S, PLE ring) before the pass. A partial accept restores that copy and re-runs the accepted columns. K / V and `rawK` are simply rewritten, and a pooled key is re-pooled when its block completes again. That re-run is the whole trunk for the accepted columns (every layer, the MoE expert reads, attention and QSA), so a partial accept costs about one more pass of n columns, plus copying ~115 MB of state before every verify pass. It is correct and simple, and nothing calls `verify` yet. Deferred (#348 review finding 3): before MTP drafting is built on it, port the Qwen3.5 engine's replay rollback (`engine/qwen35.js`, `dn_delta_mc`): each verify column saves its DeltaNet inputs (conv input, beta, decay), the pass keeps S from before it, and an accept replays only the delta step and the conv history up to column n, with a small per-column ring for the PLE taps. Nothing else in the trunk is recurrent, so that removes the re-run.

Expert offload (`engine/expert_store.js`, #364): with the engine created with `expertRegion: true`, an offloaded layer's batched pass splits after its router (`cut`). A wide pass (n x K >= half the experts) copies the whole layer into the store's region (`ExpertStore.loadLayer`, no readback); a narrow one, such as a verify pass, reads its n x K ids back and copies only those experts (`loadExperts`). The expert kernels then run on `regionEntry()` with the original expert ids: the same kernels on the same bytes, so an offloaded pass gives the resident bits (`OFFLOAD=0-3 tests/test_q38flash_batch.js`, its BITS line). Without the region, `Qwen4ExpBatch.create` refuses offloaded layers, and the engine's `_wq` refuses parked entries (`notParked`).

GB10, first 4 layers (Vulkan, Deno), 4300-token prompt: 64-column prefill 303-399 tok/s against decode 44.8-57.8 tok/s, 5.6-8.9x over four runs (the ratio moves with both runs and with whatever else shares the GPU: 7.7x and 8.9x in the PR's first runs, 5.6x in the #365 review with the GPU shared (303 vs 54.0), 6.8x after the review fixes (391 vs 57.8)). The full model (48 layers + head) prefills at 34-40 tok/s against 10-14 tok/s decode, about 3x: the MoE runs per column per expert. With `OFFLOAD=0-3` (16 pool slots per layer) the 4-layer prefill runs at 264-279 tok/s against 13.3-14.1 tok/s offloaded decode (three runs), with the resident run's bits. The bit-equality relies on the shader compiler treating the decode and batched kernels' expressions alike. It holds on the GB10's Vulkan driver; run `tests/test_q38flash_mc.js` once on Metal (Chrome) and D3D12 (the PC) before relying on it there.

Bit-equality with decode is the test oracle for every batched kernel. It made the batched path easy to trust, but it also rules out what a faster prefill needs: an expert-grouped MoE GEMM and tiled attention sum in a different order than the per-column decode kernels. Plan the switch together with the grouped MoE work: the batched path against decode within a tolerance (relative residual error, as `test_qsa.js` does for attention), plus the llama.cpp golden ids, and keep the bit-equal kernels as the reference implementation.

### Flash long context: KV and QSA memory

Two engine options shrink the per-sequence state of `engine/qwen4exp.js`:

- **`kv: "f16"`.** The 12 attention layers keep K and V as packed f16 pairs (the low half is the even element) instead of f32. The store rounds to nearest even by hand (`pack2x16_rne`, `engine/wgsl/f16.js`), so D3D12 writes the same bits as Vulkan and Metal. The read is exact (`unpack2x16_exact`, no subnormal flush). The decode, batched and gathered attention kernels do the f32 arithmetic in the same order on those values, so decode, prefill and verify still agree bit for bit with each other. `tests/test_q38flash_mc.js` checks that each stored value is the f32 run's value rounded to nearest even. The default is still `"f32"`.
- **QSA raw keys as a ring (`qsaRing`, default 1024 rows).** `qsa_pool` only reads the rows of the blocks a pass completes, `4 * floor(pos / 4) .. pos + nCols - 1`. That is at most nCols + 3 rows, so `rawK[p]` lives in slot `p & (ring - 1)` (`qsaWGSL({ rawRing })`). `Qwen4ExpBatch.create` refuses passes wider than ring - 3 columns. A partial verify accept re-runs from the pass start, and the rows of the block that was open before the pass are not overwritten, since nCols + 3 <= ring. `qsaRing: 0` keeps one row per position. The pooled keys stay whole, because `qsa_score` reads every block.

Accounting: `qwen4expMemory(cfg, { ctx, kv, rawRing, layerRange })` (`engine/qwen4exp_mem.js`, no GPU) gives the bytes per context for a set of layers. `qwen4expMaxCtx` inverts it for a memory budget, for the planner. The engine's `stateBytes()` is checked against it (`tests/test_q38flash_ckpt.js`). `_stateParts(pos)` is what a checkpoint copies: K / V rows `[0, pos)` in the engine's format, the whole raw-key ring, the pooled keys of the complete blocks, and the DeltaNet S, conv windows and PLE ring. `stateSignature()` names the format. The checkpoint test restores a QSA state at 2302 (the open block's rows come back from the restored ring), bit for bit, in both formats.

Full model (48 layers), state per sequence:

| context | | K / V | rawK | pooled | DeltaNet + PLE | state | + QSA scores (decode + 64-col pass) |
|---|---|---|---|---|---|---|---|
| 8K | before (f32, rawK whole) | 384.0 MiB | 24.0 MiB | 6.0 MiB | 112.6 MiB | **526.6 MiB** | 0.5 MiB |
| 8K | after (f16, ring 1024) | 192.0 MiB | 3.0 MiB | 6.0 MiB | 112.6 MiB | **313.6 MiB** | 0.5 MiB |
| 32K | before (f32, rawK whole) | 1.50 GiB | 96.0 MiB | 24.0 MiB | 112.6 MiB | **1.73 GiB** | 2.0 MiB |
| 32K | after (f16, ring 1024) | 768.0 MiB | 3.0 MiB | 24.0 MiB | 112.6 MiB | **907.6 MiB** | 2.0 MiB |
| 128K | before (f32, rawK whole) | 6.00 GiB | 384.0 MiB | 96.0 MiB | 112.6 MiB | **6.58 GiB** | 8.1 MiB |
| 128K | after (f16, ring 1024) | 3.00 GiB | 3.0 MiB | 96.0 MiB | 112.6 MiB | **3.21 GiB** | 8.1 MiB |
| 256K | before (f32, rawK whole) | 12.00 GiB | 768.0 MiB | 192.0 MiB | 112.6 MiB | **13.05 GiB** | 16.3 MiB |
| 256K | after (f16, ring 1024) | 6.00 GiB | 3.0 MiB | 192.0 MiB | 112.6 MiB | **6.30 GiB** | 16.3 MiB |

Per token: 52,992 B before (f32 K / V 49,152 + rawK 3,072 + pooled 768), 25,344 B after (f16 K / V 24,576 + pooled 768).

Not counted above: the verify snapshot (~115 MB, fixed), the fixed per-pass scratch, and the weights. Each attention layer's K and V are one buffer each, bound whole. At 256K that is 512 MiB per buffer in f32 and 256 MiB in f16, past a browser's default 128 MiB binding limit (65,536 positions in f32, 131,072 in f16). The engine refuses a context whose buffer exceeds the device's limit, and a room must ask for the adapter's limit.

Checks on the GB10 (Vulkan, Deno), first 4 layers (`tests/test_q38flash_batch.js`, 4300 tokens of the ctx8k prompt, QSA on, raw-key ring 1024). Every check passes in both formats: prefill == decode bit for bit (mixed passes and across 2051 / 2052), interleave, QSA == dense, verify / accept, and QSA against the CPU reference (raw keys collected from the ring as decode runs, every pooled key re-pooled). The 64-column prefill hash is `d6a6d01a` for f32, the same as main before this change, so the ring changes no bit. It is `37561b6a` for f16. Each hash is the same with `OFFLOAD=0-3 OFFLOAD_SLOTS=16`. `tests/test_q38flash_ckpt.js` restores the state bit for bit in f32 and f16, with and without QSA.

Full model, judged by the golden rule (`FULL=1 CASES=ctx2k,ctx8k,ctx32k MAXSEQ=32256`, with the long goldens from the long-context goldens branch). Batched prefill, then decode, then a verify pass:

| case (prompt tokens) | KV | prefill tok/s | decode tok/s | vs llama.cpp | verify == decode |
|---|---|---|---|---|---|
| ctx2k (1887) | f32 | 41 | 11.78 | near tie at 26 (our gap 0.225) | yes |
| ctx2k (1887) | f16 | 37 | 11.27 | near tie at 26 (our gap 0.231) | yes |
| ctx8k (7663) | f32 | 37 | 11.39 | near tie at 6 (our gap 0.262) | yes |
| ctx8k (7663) | f16 | 37 | 11.34 | near tie at 6 (our gap 0.219) | yes |
| ctx32k (31958) | f32 | 37 | 11.31 | near tie at 0 (our gap 0.407) | yes |
| ctx32k (31958) | f16 | 37 | 11.23 | **rule FAIL at 0** (our gap 0.649 > 0.5) | yes |

The engine allocated 1651 MiB of state at maxSeq 32256 with f32 and 895 MiB with f16. Speed is the same in both formats. At ctx32k step 0, both formats pick 1192, which is llama.cpp's top in one of its three runs. llama.cpp's own margin between 8839 and 1192 moves across its reruns from +0.49 (the golden) to +0.05 to -0.56 (1192 ahead). Our f16 margin is -0.65 and f32's is -0.41. The two formats agree on the first 8 ids (the verify pass) and on the answer: note 32, "SSD expert offload". By the rule as written, f16 still fails this case, so `kv` stays `"f32"` by default until it passes. The options are a rule that accepts our token when it is some llama.cpp run's top, or more 32K goldens to see how often f16 lands outside the window.

### The qwen4exp goldens' pass rule

`tests/q38flash_rule.js` judges the Flash goldens (`tests/golden/q38flash_llama_greedy.json`) for `tests/test_q38flash.js` and `tests/test_q38flash_batch.js` alike. Ids are compared with llama.cpp's step by step. At the first step where they differ, the case passes as "differs at a backend near tie" when both hold: our token is in llama.cpp's top-5 at that step within 0.5 nats of llama.cpp's top, and our own top-2 gap there is under 0.5. Comparing stops there. Anything else is a mismatch, including a difference after a tie we matched through. A golden without top-k data falls back to the old rule: llama.cpp's own top-2 gap at that step under 0.25.

Why 0.5 nats. The step-6 differences on `ctx2k_b`, `ctx4k` and `ctx8k` ("are" vs "describe") looked like position-dependent drift, so they were traced (October 2026):
- llama.cpp's own CPU backend (f32 KV, no flash attention) moves `ctx2k_b`'s step-6 top-2 gap from the CUDA golden's 0.382 to 0.140, already a near tie by the old rule. Our engine moves it by 0.42, the same spread we show at 23 tokens.
- Between llama.cpp backends the gap moves by |d| 0.25-0.6 on most steps, max 0.99.
- Per-layer errors do not grow with position, and the rope is at least as accurate as CUDA's.
- The CUDA llama-server (unsloth b10840, `-fa on`) is not reproducible past ~2K prompt tokens. Repeating the same request on `ctx4k` gives step-6 gaps of 0.04-0.48; one run takes another token at step 6 itself. `ctx8k`'s step-6 gap was 0.073 in the old golden and 0.23-0.52 on reruns. The ids of `ctx4k`, `ctx8k` and `ctx2k1` change from run to run at steps 6-54. The cases up to 1943 prompt tokens gave identical ids and logprobs in 3 of 3 runs.

So a 0.5-nat window on both sides covers backend noise.

What it still catches. Perturbing the engine in a scratch run (full model; `two-sum`, `hash-map`, `bash`, `eos-mid`, plus `ctx2k_b` for the first three rows):

| Perturbation (`ssm_norm`, the DeltaNet output scale) | This rule | Old rule (llama.cpp gap < 0.25) |
|---|---|---|
| none | PASS (`ctx2k_b` at 6, `ctx2k` at 26, `hash-map` at 12: near ties) | FAIL (`ctx2k_b` at 6, llama.cpp gap 0.352) |
| layer 20 x 1.03, x 1.25 | PASS | PASS |
| layer 20 x 2 | PASS | FAIL (`hash-map` at 26, llama.cpp gap 0.34) |
| all 36 DeltaNet layers x 1.1 | PASS | PASS |
| layer 20 x -1 | FAIL (`bash` at 12: our gap 0.56; `eos-mid` at 0: 1.14 nats down) | FAIL (`eos-mid` only) |

Greedy ids are a coarse oracle under either rule: the model absorbs a scaled layer, and its ids then move only at near ties. The x 2 run's `hash-map` step looks just like the unperturbed engine's `ctx2k_b` step (llama.cpp 0.34 vs ours 0.02), so no ids-only rule can excuse one and flag the other. Numeric errors are the per-layer check's job (`LAYERS=48 tests/test_q38flash.js`, relative L2 per tensor against llama.cpp's dumps); the goldens guard against gross breakage and wrong tokens at confident steps. Because of the non-reproducibility, `make_q38flash_llama.js` asks each case REPEAT (3) times. The first run is the golden; when another run's ids differ, the golden keeps that run's ids and top-5 ("runs"). The rule may take llama.cpp's top-5 from any run whose ids agree with the golden's up to the step. Without that, `ctx4k` would hinge on which run became the golden: 7276 is 0.505 below the top in the stored run, 0.447 and 0.054 in the two reruns. The maker reuses a case's stored prompt ids (the long prompts are cut from `roadmap/*.md`, which changes), so regenerating does not move a prompt.

## Things tried and rejected (measured)
- Q4 KV cache: −92.5% prefill.
- External 0.6B draft model: different vocabulary (151,936 vs 248,320), can't verify.
- RMSNorm fused into the GEMV: 0.91× on Metal.
- Tree/Medusa drafting: no trained heads, and a DeltaNet state per branch.
- Subgroup reductions: 0–8% (the shared-memory tree was never the bottleneck).
- 4×4-per-thread GEMM tiles: slower than 2×4 (occupancy beat reuse).

## Open items
- Prefill GEMM: shipped in `engine/wgsl/gemm.js` (row-stationary Q4_0 and Q8_0 at 16 columns, split-K pinned per 27B shape; bench log, Sep 4). `benchmarks/bench_gemm.js` is the original prototype. Open: pinned shapes for the MoE and the 1.7B, and the prefill/* branches (roadmap 02).
- Register-resident `dn_delta` (see [deltanet-prefill-spec.md](deltanet-prefill-spec.md)): ~4% of a pass, bit-identical.
- MoE expert GEMV layout (`engine/wgsl/moe.js`, engine option `moeKernel`): the expert kernels are generated from
  `{ WG, TPR, R, U, wide, xsh }` per kernel (threads per workgroup, threads per row group, rows per group,
  unroll, 16 B whole-block loads, input staged transposed in workgroup memory). `MOE_DEFAULT` (wide 16 B loads,
  staged x; gate/up 128 threads, 32 per row, 4 rows per workgroup; down 128 threads, 8 per row, 16 rows) was
  picked by the sweep on the GB10: in-model moe_gu_q4 69.8 -> 60.7 µs, moe_dn_q4 43.0 -> 38.4 µs (~0.45 ms of
  22.9 ms GPU per token); Chrome decode tok/s is within run-to-run noise of legacy;
  the tuned layout is therefore opt-in (`moeKernel: "default"`); leaving `moeKernel` unset gives `MOE_LEGACY`, the
  first coop build's bits exactly. It only affects the unfused `moe_gu` / `moe_dn` kernels (`moeFuse: false`); the
  fused MoE FFN (the default) has its own kernels. Sweep with `tests/bench/moe_kernel_sweep.js`,
  select at runtime with `MOE_KERNEL=<preset|JSON>` (Deno tests and `chrome_bench.mjs`).
  Index math is checked on the CPU for ~400 layouts by `tests/unit/moe_kernels_test.js`.
