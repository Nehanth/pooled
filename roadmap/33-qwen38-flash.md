# 33 · Qwen3.8-Flash-Next (qwen4exp)

**Phase:** big models · **Status:** in progress (loader #340, reference #342, n-gram rows #339, QSA kernels #344) · **First target:** greedy ids equal to llama.cpp at ctx <= 2048 on the GB10, then the Mac + 5070 PC room · **Builds on:** 31 (MoE split correctness), 32 (SSD expert offload), the 122B work (#319, #320)

Sources are cited inline. "llama.cpp" means ggml-org/llama.cpp master at `f2cc728` (2026-10-11), which contains PR #27742 "model: add Qwen3.8-Flash-Next (qwen4exp)" (merged 2026-08-27 as `6c84c7d`). `qwen4exp.cpp` is `src/models/qwen4exp.cpp`, the converter is `conversion/qwen4exp.py`. The model card and config are at huggingface.co/Qwen/Qwen3.8-Flash-Next (`README.md`, `config.json`). GGUF headers were read over HTTP range requests (tensor tables, not weights), with a small parser in the scratchpad.

## Updates since this note (2026-10-11 prior-art review)

- llama.cpp MTP did land: #28243 was replaced by **#29761** (merged; `--spec-type draft-mtp`).
- All RMSNorms are zero-centred `(1+w)` with zero-init weights, the GDN gated output norm excepted. Check that the GGUF folds the indexer and PLE norms too; the QSA work found the indexer q/k norms folded (means ~0.96).
- The n-gram hash seed (`1234 + 10007·layer`) is not in config.json; load the GGUF's `layer_multipliers`, never recompute them.
- QSA indexer scores must be fp32 with ties broken lowest index first (#344 does).
- On Apple unified memory a big in-tab expert cache can starve the page cache the disk reads go through; size the Mac cache by measurement, starting around 18-20 GB.
- Cutting top-k (10 to 5) is a possible opt-in lossy mode, not a default.

## Why

It is the best open model that fits our rooms on paper: about 6B active parameters per token, and almost everything that isn't active (512 experts per layer, a 51B n-gram table) can live off the GPU. llama.cpp already runs it at 18-20 tok/s on the 5070 PC with experts in RAM (UD-Q4_K_XL, our own measurement). But it is not a Qwen3.5 variant. Three things are new: a 4-stream residual (hyper-connections) that replaces every RMSNorm, sparse attention (QSA), and an n-gram hash embedding. It also has 512 experts and top-10 routing. So this is a new engine mode, not a `room/models.js` entry.

## 1. Architecture, exactly

### Headline numbers vs claims

| Claim | Verified | Source |
|---|---|---|
| ~177-180B total | 177B in the main file: BF16 GGUF is 354.0 GB. Card: "125B with 6B activated, plus 51B n-gram embedding and 4B MTP" (MTP ships as separate files) | unsloth BF16 file sizes; card "Model Overview" |
| ~6B active | Card says 6B. My count from tensor shapes is ~6.6B with the LM head and the HC weights: routed 2.36B, shared 0.24B, DeltaNet 2.08B, attention 0.62B, HC 0.63B, router 0.06B, head 0.64B | GGUF shapes below |
| 512 experts, 10 routed + 1 shared | Yes. `num_experts 512`, `num_experts_per_tok 10`, `moe_intermediate_size 640`, `shared_expert_intermediate_size 640` | config.json |
| "Qwen Sparse Attention, micro-block" | Yes. A micro-block is 4 tokens. The budget is 512 blocks (2048 tokens), scored by a small MQA indexer | config `indexer_*`; qwen4exp.cpp:742-871 |
| n-gram table "20M" | The 20M is per hash head (`ngram_vocab_size_base 20000000`). There are 16 heads, so the table has 320,001,446 rows (padded to 320,001,536) of 160 values = 51.2B params | GGUF `ple.head_vocab_sizes`, tensor shape |
| "at layer 2" | 1-based. `ple_layer_ids: [2]` becomes `blk.1` (0-based). The converter subtracts 1 | conversion/qwen4exp.py:91-94; GGUF `qwen4exp.ple.layers = [1]` |

Not in the card's headline: **every layer norm is replaced by hyper-connections** (`hc_count 4`, `hc_lowrank 320`). The card calls this "Gated Residual: 4 branches, bottleneck rank 320". Also there is **no final output norm**: the last hyper-connection mixer acts as the norm (qwen4exp.cpp:184, 511-514).

### Layer layout

- 48 layers, hidden size 2560, vocab 248,320 (untied `token_embd` and `output`).
- `layer_types`: 3 x `linear_attention` then 1 x `full_attention`, repeated 12 times. Layers 3, 7, ..., 47 are QSA. The other 36 are Gated DeltaNet. llama.cpp derives this from `full_attention_interval = 4` (qwen4exp.cpp:148-156), and the GGUF writes `attention.compress_ratios = [0,0,0,4,...]`.
- Every layer's FFN is MoE: 512 experts, top-10, plus one gated shared expert.
- PLE (n-gram) injection on layer 1 only, before that layer's first HC mixer (qwen4exp.cpp:452-454). It must be a DeltaNet layer, because its conv history is kept in the recurrent cache (qwen4exp.cpp:158-163).
- MTP: 1 extra block (`blk.48`), a QSA layer plus MoE, shipped in separate GGUFs.
- Vision tower: a stock Qwen3-VL ViT, in a separate mmproj file. Out of scope here.

### Hyper-connections (the residual)

The residual is `R[4][2560]` per token, and it starts as 4 copies of the token embedding (qwen4exp.cpp:441-447). Each layer runs two "mix, block, combine" rounds: one around the token mixer and one around the MoE (qwen4exp.cpp:449-498).

`hc_mix(R; norm[4][2560], down[320 x 10240], up[10240 x 320], inject[4 x 10240])` (qwen4exp.cpp:313-367):
1. `xn[c] = rmsnorm(R[c]) * norm[c]`: a separate RMS norm for each stream, eps 1e-6. The converter already folded the gammas to (1 + w).
2. `lo = silu(down · flat(xn) / 4)`, giving `[320]`.
3. `gate = up · lo`, giving `[10240]`.
4. `mixed = mean over c of (xn[c] * sigmoid(gate[c]))`, giving `[2560]`. This is the block's input, with no further norm.
5. `inj = inject · flat(xn)`, giving `[4]` (one scalar per stream).

`hc_combine(R, out, inj)` (qwen4exp.cpp:369-397): `R[c] += 2 * sigmoid(inj[c] / 4) * out`.

The final head mixer is `hc_mix` with `output_hc_{norm,down,up}` and no inject. Its `mixed` goes straight into `output.weight` (qwen4exp.cpp:511-521). Weights per layer: `hc_{attn,ffn}_{norm [10240], down [10240->320], up [320->10240], inject [10240->4]}`.

### Gated DeltaNet layers (36)

These are unchanged from Qwen3.5 except one function: the output gated norm uses **sigmoid(z), not silu(z)** (qwen4exp.cpp:638-648, "the one numerical difference from Qwen3.5's GDN"). The shapes are 16 QK heads and 48 V heads, head dim 128, so `ssm.inner_size 6144`. The conv kernel is 4 and `conv_dim = 2*16*128 + 6144 = 10240`. Tensors are `attn_qkv [2560->10240]`, `attn_gate (z) [2560->6144]`, `ssm_alpha/beta [2560->48]`, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d [4 x 10240]`, `ssm_norm [128]` and `ssm_out [6144->2560]` (qwen4exp.cpp:254-264, 1019-1144). The V-head order is the Qwen3.5 converter's (`_LinearAttentionVReorderBase`), so it matches our `kh = h % nKH` (engine/wgsl/qwen35.js:37). The head ratio is 3, between the 35B's 2 and the 122B's 4.

### QSA layers (12, plus MTP)

The attention core is Qwen3.5 gated attention: 24 Q heads, 2 KV heads, head dim 256. `attn_q [2560 -> 12288]` holds [q | gate] interleaved per head. There are per-head `attn_q_norm` / `attn_k_norm` (RMS over 256), IMRoPE on the first 64 dims (`partial_rotary_factor 0.25`, sections [11,11,10], theta 1e7), a scale of 1/16, and `out = attn * sigmoid(gate)` followed by `attn_output [6144->2560]` (qwen4exp.cpp:932-1017). The only difference from our full-attention layer is which cells are attended.

Selection (qwen4exp.cpp:742-871, cache logic in `src/llama-memory-hybrid-idx.cpp:737-980`). Per token, on the layer input `mixed`:
1. Raw indexer key `k_raw = index_k_proj · x` with shape `[128]` (BF16 weight). It is cached per position with no norm and no RoPE.
2. Blocks are groups of 4 consecutive tokens in sequence order (`indexer_compress_ratio 4`). When a block completes, its pooled key is `rope(rmsnorm(mean of the 4 raw keys) * k_norm)`, rotated at the block's first position (64 rotary dims of 128). It is cached once per block.
3. Indexer query: `q = rope(rmsnorm_head(index_q_proj · x) * q_norm)`, 4 heads x 128.
4. Score per visible complete block p: `s_p = sum over h of relu(q_h · K_p) / sqrt(128)`. This is `ggml_lightning_indexer` with every head weight set to 1/sqrt(128) (qwen4exp.cpp:806-810; CPU op ggml/src/ggml-cpu/ops.cpp:12427). A token sees only blocks up to its own position.
5. Keep the top `min(n_pool, 2048/4 = 512)` blocks, plus the "tail": the up to 3 cells of the token's incomplete block, its own cell included. When the token completes its block, the tail is empty and its own block competes like any other. Up to 512*4 + 3 = 2051 cells are selected.
6. Run dense GQA attention over the selected cells only. This is a mask over the normal KV cache (qwen4exp.cpp:874-930).

Consequence: **with 2051 or fewer cached tokens, QSA selects everything and is exactly dense causal attention.** The PR reports bit-identical logits on BF16/F32 below the budget, and 3% of positions diverging at 8K when dense is used instead (PR #27742 body). On quantized files it reports small non-bit-identical deltas across the boundary (max 2.8e-3 logit on UD-IQ1_S). Indexer tensors per QSA layer are `indexer.{q_proj [2560->512], k_proj [2560->128]}` (BF16) and `indexer.{q_norm, k_norm} [128]`. The HF checkpoint fuses these as `index_qk_proj`, and the converter splits it (conversion/qwen4exp.py:154-161).

### MoE

`build_moe_ffn(..., LLM_FFN_SILU, norm_w = true, ..., LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX)` (qwen4exp.cpp:1146-1164). This is a softmax over all 512 logits, top-10, renormalised to sum 1, with SwiGLU experts (2560 -> 640 -> 2560). The shared expert is SwiGLU 640 times `sigmoid(ffn_gate_inp_shexp · x)` (qwen4exp.cpp:1166-1188). That is the same math as the 35B/122B, with 512 experts and K = 10. The router is `ffn_gate_inp [2560->512]`, F32 in every file I checked.

### N-gram embedding (PLE)

Table: `per_layer_token_embd.weight`, ggml shape `[160, 320001536]`. That is 16 heads, each owning a contiguous row range `[head_offsets[h], head_offsets[h] + head_vocab_sizes[h])`, with vocab sizes 20,000,003 ... 20,000,171 (primes). Heads 0-7 hash bigrams and heads 8-15 hash trigrams (`ngram_size 3`, `heads_per_ngram 8`).

Index computation (host side in llama.cpp: qwen4exp.cpp:1196-1295):
```
ctx[0] = token at p; ctx[s] = token at p - s (s = 1, 2)
  a missing predecessor (before the sequence start) or any EOS (248044) at or before it -> EOS for it and all older
  (the token's own EOS does not cut its context); image positions use image_token_id 248056
for n in 2..3:
  mixed = (u64)ctx[0]*m[0] ^ (u64)ctx[1]*m[1] [^ (u64)ctx[2]*m[2]]      m = [23703573157769, 20109073645365, 8052911324071]
  for g in 0..7: row[(n-2)*8 + g] = mixed % vocab[(n-2)*8+g] + offset[(n-2)*8+g]
```
The multipliers are 45-bit and token ids are below 2^18, so each product fits in 63 bits. JS needs BigInt or exact 64-bit emulation. Doubles are wrong here. The GGUF keys are `qwen4exp.ple.{layers, ngram_size, heads_per_ngram, conv_kernel, eos_token_id, image_token_id, layer_multipliers, head_offsets, head_vocab_sizes}` and `embedding_length_per_layer_input = 160`.

Combination at layer 1, before its first `hc_mix`, on the 4-stream residual `R` (qwen4exp.cpp:1353-1461):
```
emb   = concat of the 16 gathered rows                    [2560]
key   = ple_key · emb  -> [10240], rms-normed per stream * ple_norm_key[c]
query = rmsnorm(R[c]) * ple_norm_query[c]
gate[c] = sigmoid(sign(s) * sqrt(max(|s|, 1e-6))),  s = dot(key[c], query[c]) / sqrt(2560)
value = ple_value · emb  -> [2560]
g[c]  = value * gate[c]                                   (4 x 2560)
n     = rmsnorm per stream of g * ple_norm_conv[c]        (10240)
conv  = silu(depthwise causal conv over time of n: kernel 4, dilation 3, taps at t-9, t-6, t-3, t; weight ple_conv1d [4 x 10240])
R[c] += g[c] + conv[c]
```
The conv keeps a history of 9 positions x 10240 values per sequence, and it must be rolled back with speculation (llama.cpp keeps it in the recurrent cache: qwen4exp.cpp:1297-1351, 1413-1454). The "embedding" is therefore not added to the token embedding. It is a gated, per-stream residual injection at layer 1.

### MTP

One block, `blk.48`, in separate files (ggml-org `mtp-Qwen3.8-Flash-Next-Q4_0.gguf` 2.2 GB; unsloth `MTP/mtp-...-shared-Q8_0.gguf` 2.8 GB, which borrows `token_embd` and `output`). Input: `R_mtp[c] = eh_proj · [rmsnorm(embed(t_next)) * enorm ; rmsnorm(h[c]) * hnorm[c]]` for each of the 4 trunk streams. The trunk's `h` is the hc-wide residual **before** the final mixer, 10240 floats. Then come one QSA + MoE layer with its own HC mixers, then its own head mixer `nextn.hc_head_*` and the shared `output` (qwen4exp.cpp:500-503, 526-620). `eh_proj [5120 -> 2560]` is `[fc_embedding | fc_hidden]` concatenated by the converter (conversion/qwen4exp.py:145-151). On mainline status: master has `graph_mtp`, but PR #28243 ("Qwen3.8-Flash-Next MTP") is closed unmerged and the unsloth MTP README says mainline has no MTP graph. That README looks stale. The local unsloth build has MTP.

### Norms, RoPE, context

- RMS eps 1e-6 everywhere. The only RMSNorms outside HC are q/k head norms, the indexer norms, the DeltaNet gated norm, the PLE norms and the MTP enorm/hnorm.
- RoPE: IMRoPE, theta 1e7, 64 of 256 dims, `rope_type "default"` in config.json. The 1M-token YaRN factor 4 is a README override, not the shipped config. One search summary conflated the two.
- Context: 262,144 native.

## 2. Diff against engine/qwen35.js

### Reusable as is (or with a size bump)

- DeltaNet core: `dn_conv`, `dn_pre`, `dn_delta(_gn,_gnp)`, snapshots and rollback (`engine/wgsl/qwen35.js`). The gate function is the only change; see below.
- Gated attention: `attn_glue` (q/gate split, head norms, IMRoPE `rope_part`), `attn_dec` / flash / tiled prefill, KV cache f16/q8 (qwen35.js:1563-1630). 24/2 = 12 Q heads per KV head is already supported (`dims.attnR`, added for the 122B's 16).
- MoE math: `moe_route` with norm (softmax top-K renormalised), `moe_gus` / `moe_dnc` / grouped prefill kernels, shared expert with sigmoid gate (qwen35.js:1657-1704, engine/wgsl/moe.js, moe_group.js).
- Expert offload: `engine/expert_store.js` park, pool and cut machinery, roadmap 32's disk tier, `tests/trace_moe.js`.
- Loader plumbing: split GGUF merge (`room/models.js mergeSplitHeaders`), streamed load, converted-weights cache. `convertEntry` (engine/gguf.js:227-244) already handles every type in the target file.
- MTP speculation driver (`specStep`, `_verifyFused`, draft chain), given a new MTP input stage.

### Needs a change

| Where | Change |
|---|---|
| engine/gguf.js:119-122 | The `^qwen3[5-9]` alias does not match `qwen4exp`. Add an explicit `qwen4exp -> qwen35.*` alias for the shared keys, and read the new keys (`hyper_connection.*`, `attention.indexer.*`, `ple.*`, `compress_ratios`) under their own names |
| engine/gguf.js:496-589 `qwen35LayerNames` / `qwen35Weights` | New name map: no `attn_norm` / `post_attention_norm`; add `hc_{attn,ffn}_{norm,down,up,inject}`, `indexer.*`, and `ple_*` on layer 1; head: `output_hc_{norm,down,up}` instead of `output_norm`; MTP: `nextn.{eh_proj, enorm, hnorm, hc_head_*}` with no `shared_head_norm`, loaded from a second file |
| engine/gguf.js `ggufEntry` | Never convert `per_layer_token_embd` (28.8 GB). It needs a row-reader path (new, below) |
| engine/wgsl/qwen35.js:344 (`dn_gatenorm`), `dn_delta_gn`, `dn_delta_gnp`, `dn_gatenorm_mc` | `silu(z)` becomes a template switch to `sigmoid(z)` |
| engine/qwen35.js:274-275, 328-341 (`dims`) | Add `hc = 4`, `hcRank = 320` and `hDim = hc * dim` (the width of the residual and of what crosses devices) |
| engine/wgsl/moe.js:249-262 `routeKernel`, `gusKernel`, moe_group | K = 10 and nExp = 512. `H = 1024 + CH` already covers 512 ids. K-sized arrays are template parameters. `moeGroupWGSL` allows K <= 16. Roadmap 32 item 10 calls this out for the 397B. Needs a test sweep, not a design |
| engine/expert_store.js | The prefill region becomes all 512 experts of a layer (1.42 GB at Q4_0, 1.84 GB on the Q4_1 layers) |
| room/plan.js:118 `OFFLOAD_EXPERTS = 256` | Per model (512) |
| room/models.js `stateBytesPerLayer`, `kvBytesPerLayerPos`, `SHAPE`, `expertSlices` | Read `qwen4exp` keys; add the PLE conv history (9 x 10240 f32 = 369 KB per sequence) and the 4x hidden |
| room/pipeline.js:210, 287, 339, 386 | Pass `dims.hDim` (10240 floats, 40 KB per token in f32) between devices, not `dims.dim` |
| Every fused "norm in the GEMV" path | `layerFuse.norm` (`matvec_*_coop_n`), `dn_nba`, `moe_nrt`, `mvKVn`: the block input is `mixed`, already normalised, so these run in their no-norm form or are off for this arch |
| `add_res` / `.acc` epilogues on `attn_output`, `ssm_out`, `moe_dnc` | They become `hc_combine`: 4 strided writes, each scaled by `2*sigmoid(inj[c]/4)` |

### New code

1. **HC kernels.** These are on the critical path; there are 2 per layer, plus the head:
   - `hc_norm_down`: grouped RMSNorm of `R` (4 x 2560), then a Q8/Q4 GEMV 10240 -> 320 with `silu(x/4)` in the epilogue. Also the 4-row `inject` GEMV (F32 or BF16 in the files) on the same `xn`.
   - `hc_up_mix`: GEMV 320 -> 10240, with rows ordered so a workgroup owns rows `{i, 2560+i, 5120+i, 7680+i}`. The epilogue writes `mixed[i] = mean of xn[c][i] * sigmoid(gate[c][i])`. `xn` is needed again here; keep it in a scratch buffer (40 KB).
   - `hc_combine`, fused into the out-projection, `ssm_out` and `moe_dnc` epilogues.
   - Multi-column variants (batch, wide, grouped prefill, verify), since every prefill path changes.
2. **PLE block (layer 1 only).** A CPU row gather (hash, then read 16 x 90-byte Q4_0 rows, then dequantize to 2560 f32, a 10 KB upload per token), `ple_key` / `ple_value` GEMVs, per-stream norms and dots, the signed-sqrt gate, and the dilated causal conv with its own state slot. The state slot covers export/import, sessions, checkpoints and speculative snapshots. Predecessor tokens for the hash must also persist across turns and rollbacks.
3. **N-gram row source** (`engine/ngram.js`). The interface is `rows(ids) -> Promise<Float32Array>`. Backends: in memory (tests), a file range (room-node `pread`), OPFS (the browser; one file per hash head, so 16 sync handles can read in parallel), and an HTTP range (a cold start only). Indices depend only on tokens, so the read can start before the forward pass. For decode, issue it as soon as the token is sampled; layer 0 runs while it lands.
4. **QSA (phase 2).** A per-QSA-layer indexer cache: raw key `[128]` per position and pooled key `[128]` per 4-token block. There is a pooled-key kernel that runs on block completion, an indexer score kernel (4 heads x n_pool, relu, sum), a top-512 select over up to 65,536 blocks at 262K, and a gathered-cell variant of `attn_dec` and of the tiled prefill attention. The select needs a new kernel: `engine/topk.js` is sized for k <= 64 (`TOPK_MAX`). A radix or threshold select is the obvious design. Below 2051 tokens none of this changes outputs, so phase 1 runs dense and caps the context at 2048.
5. **MTP input stage.** Four `eh_proj` GEMVs (one per stream, same weights) on `[enorm(e); hnorm(h[c])]`. The trunk must expose `R` before the head mixer. `headFromHidden` / `setHidden` carry `hDim`.

The cleanest code shape is a mode flag on `Qwen35Engine` (`this.hc`), as the MoE is today, not a fork. Most of the 3,948 lines (spec, sessions, offload, room plumbing) stay shared, and the layer encoders branch where the norm and residual are touched. The cost is that every encoder (`_encodeLayerR`, `_encodeLayerBatch`, `_encodeLayerWide`, `_prefillGrouped`, MTP) gets an HC branch. A subclass would duplicate them instead.

## 3. Quantization: which GGUF

The loader handles Q4_0 and Q8_0 natively. Q4_1, Q5_0, Q5_K and Q6_K are requantized to Q8 (`convertEntry`, `requantQ8Streaming`), and F32, F16 and BF16 become f32. `ggmlTypeBytes` returns -1 for Q4_K, IQ4_NL, IQ4_XS and the rest of the IQ/K family (engine/gguf.js:305-321).

Candidate files, from their tensor tables:

| File | Size | Experts | Dense | Head | n-gram table | Loads today? |
|---|---|---|---|---|---|---|
| **bartowski Q4_0** (3 shards) | 100.6 GB | Q4_0; down Q4_1 on layers 0-5 | mostly Q4_0; 6 of 12 `attn_q` Q6_K; half of `ssm_out` Q8_0; shared experts Q8_0/Q5_0; HC down/up Q4_0, inject BF16; indexer BF16; routers F32 | Q6_K (-> Q8 on load); embed Q4_0 | Q4_0 (28.8 GB) | **Yes, every type is handled.** Same recipe family as the 35B/122B files we match today |
| nitinpanj Q4_0-Q8out-v3 | 102.6 GB | Q4_0; down Q4_1 on layers 0-5 | all Q8_0; HC inject F32; output_hc BF16 | Q8_0; embed Q8_0 | Q4_0 | Yes. Same imatrix path as bartowski in its metadata, so likely derived from it. Its README claims MTP and vision inside, but the header has neither (1224 tensors, no `blk.48`) |
| unsloth UD-Q4_K_XL | 111.3 GB | Q4_K / Q5_K gate-up, Q5_1 / Q8_0 down | Q8_0 | Q8_0 | IQ4_NL | No: Q4_K, Q5_1 and IQ4_NL are not handled; K-quant experts would widen to Q8 (2x bytes) |
| unsloth / bartowski / ggml-org Q8_0 | 188 GB | Q8_0 | Q8_0 | Q8_0 | Q8_0 (54.4 GB) | Yes, but 2x the expert bytes |
| ggml-org IQ4_NL | 102 GB | IQ4_NL | | | | No |

**Target: bartowski `Qwen3.8-Flash-Next-Q4_0`, plus ggml-org `mtp-Qwen3.8-Flash-Next-Q4_0.gguf` (2.2 GB) for MTP.** It needs no new dequant code. The parts that widen on load are what the 122B already does and still matched llama.cpp's ids: Q4_1 down experts on layers 0-5 parked as Q8, the Q6_K head and 6 `attn_q`, and the Q5_0 shared experts. nitinpanj is the fallback if we want all-Q8 dense weights. That is +1.4 GB read per token for a likely small quality gain.

Conversion work:
- None for the weights we keep.
- `per_layer_token_embd` gets a split, not a conversion: copy each head's row range out of shard 1 into 16 per-head files (1.8 GB each, 90 B per row). This is for OPFS parallelism and for keeping the table out of the converted-weights cache.
- Later and optional: a native Q4_1 expert kernel (saves 0.82 MB per expert on layers 0-5), as noted in the bench log for the 122B.

## 4. Memory, disk and tok/s

### Bytes per token (bartowski Q4_0, as loaded, decode, batch 1)

| Part | Bytes per token | Arithmetic |
|---|---|---|
| DeltaNet projections | 1.31 GB | qkv 0.53 + gate (z) 0.32 + ssm_out 0.46 (36 layers) |
| QSA projections | 0.45 GB | q 0.31 (Q6_K half widened to Q8) + k/v 0.03 + o 0.11; indexer 0.04 only once QSA is on |
| HC mixers | 0.37 GB | 4 x (10240 x 320) Q4_0 + inject, per layer, 48 layers, plus the head mixer |
| Router + shared expert | 0.50 GB | F32 routers 0.25; shared 0.25 (Q5_0 widened) |
| LM head | 0.68 GB | Q6_K widened to Q8 |
| PLE key / value | 0.02 GB | layer 1 |
| **Dense total** | **~3.4 GB** | resident, read every token |
| DeltaNet state | ~0.23 GB | 36 x 48 x 128 x 128 x 4 B = 113 MB, read and written |
| KV at 2048 tokens | ~0.05 GB | 12 layers x 2 KV x 256 x 2 (K, V) x 2 B = 24 KB per position |
| **Routed experts** | **1.38 GB** | one expert is 3 x 2560 x 640 weights: 2.77 MB at Q4_0, 3.59 MB on layers 0-5 (down widened to Q8). 6 x 10 x 3.59 + 42 x 10 x 2.77 MB |
| n-gram rows | 1.4 KB | 16 rows x 90 B, but 16 random reads |

Total about 5.0 GB per token if everything were resident. That is close to the 122B's (~3.8 GB dense + 2.04 GB experts). The experts are the smaller share, so a cache miss costs less here than on the 122B.

Resident footprint: dense 3.4 GB, embed 0.36 GB (JS memory), state/KV/work ~0.5 GB, and a prefill region of 1.84 GB. **Experts total 70.5 GB** (42 layers x 512 x 2.77 MB + 6 x 512 x 3.59 MB). The **n-gram table is 28.8 GB**, entirely disk-resident by design.

### What lives on disk

- **n-gram table: always on disk** (OPFS on the Mac, NVMe on a room node). A token needs 16 small rows, known one token ahead, so the read is off the critical path. On a cold SSD that is 16 x ~80 µs random 4K reads, under ~0.3 ms with 16 handles, and it overlaps layer 0. Hot n-grams stay in the page cache or a small LRU of rows. Prefill: 2048 tokens x 16 rows = 32K random reads, ~0.3-0.6 s, issued before the first layer. Only the device holding layer 1 needs the table.
- **Experts: roadmap 32's tiers unchanged.** VRAM pool, then parked RAM, then disk. Per-expert byte ranges come from the GGUF. The unit is 1/512 of a layer instead of 1/256, which roughly halves the bytes per miss compared with the 122B (2.77 vs 5.3 MB). That is good for random reads at queue depth 3-4. `OFFLOAD_EXPERTS` and the region size change. Nothing else in the tier design does.
- Dense weights, HC, routers, KV and state stay in memory (roadmap 32's non-goal applies).

### Estimate: Mac Studio (M5 Max, 36 GB) alone, in a Chrome tab

Inputs: Chrome streams 443 GB/s raw on this Mac. Our MoE kernels reach ~45% of that and the 27B's ~84% (bench-log 2026-09-28). OPFS cold reads were 1.2 GB/s per worker and 4.8 GB/s with 4 workers on the GB10 (docs/research/offload-2026-09.md). Chrome's `mapAsync` round trip is 0.27 ms. The memory lent is ~25 GB, as in roadmap 32.

- Memory left for the expert cache: 25 - (3.4 dense + 0.4 embed + 0.5 work + 1.8 region) ≈ 19 GB. That is 27% of the 70.5 GB, ~140 experts per layer out of 512.
- Hit rate: unknown for this model. The 35B hit 76-87% with a 25% cache, and the 122B 69-71% with 64 of 256. I assume 65-80%; it must be measured with `tests/trace_moe.js`.
- Compute: 3.4 GB dense at ~260 GB/s ≈ 13 ms. Hit experts ~1 GB at ~200 GB/s ≈ 5 ms. State and dispatches ≈ 2 ms. Total ≈ 20 ms.
- Cuts: 48 layers x ~0.4 ms (readback round trip plus plan and submit) ≈ 20 ms.
- Misses: 1.38 GB x (1 - hit). Each layer's misses are known only after its router, so reads run at queue depth ~3, about 2.5-4 GB/s through OPFS.

| Hit rate | Miss bytes per token | IO at 2.5 / 4 GB/s | Per token (40 ms + IO) | tok/s |
|---|---|---|---|---|
| 65% | 0.48 GB | 193 / 121 ms | 233 / 161 ms | 4.3 / 6.2 |
| 72% | 0.39 GB | 155 / 97 ms | 195 / 137 ms | 5.1 / 7.3 |
| 80% | 0.28 GB | 110 / 69 ms | 150 / 109 ms | 6.7 / 9.2 |

**Estimate: 4-9 tok/s decode, ~6 central.** Router lookahead (roadmap 32 item 6: 77-87% next-layer recall on the 35B) overlaps IO with compute and is the biggest lever. Speculation helps less here, because a verify takes the union of 4 tokens' experts.

**Prefill is the real problem on a lone Mac.** Today a prefill frame copies each offloaded layer's whole expert set into the region. From disk that is ~70 GB per frame, so 15-28 s per 512-token frame. Layer-major prefill (roadmap 32 item 9) is required: read each layer once per prompt, giving a TTFT of about 70 GB / 3-5 GB/s ≈ 15-25 s for any prompt over ~32 tokens. That is still bad for agents. A short-prompt path (only the chosen experts) covers chat follow-ups.

Storage: the whole 100.6 GB file must be in OPFS (a ~100 GB download plus per-origin quota; the research note flagged a ~70 GB quota as a risk on the GB10 profile).

### Estimate: Mac (host) + 5070 PC room

Inputs: PC RTX 5070 (12 GB, ~10 GB lent) with up to 48 GB of parked RAM (`--ram` default: 64 GB total less 16). Parked-to-VRAM copies run at 51 GB/s. The 35B with 40 cuts on the PC spent ~0.2 ms of JS per cut. The measured 122B room on Mac + PC with all experts in RAM runs at 17 tok/s (roadmap 32 table).

- Split: Mac layers 0-13 fully resident. That is 6 x 1.90 GB + 8 x 1.47 GB (experts + dense per layer) = 23.2 GB, plus head and embed 1.0 GB, about 24 GB. The Mac also holds the n-gram table in OPFS, since it has layer 1. PC layers 14-47 (34 layers): dense 1.9 GB in VRAM and experts 34 x 1.42 = 48.1 GB parked. That is exactly at its RAM promise, so expect 1-3 PC layers on the disk tier, or the Mac lending 1-2 GB more.
- PC per token: dense 1.9 GB at ~450 GB/s = 4 ms. Pool ≈ 10 - 1.9 - 1.4 region - 0.5 = 6 GB, 64 slots per layer (12.5%), hits ~50-60% by the 122B's curve. Misses: 34 x 10 x 0.45 x 2.77 MB = 0.42 GB at 51 GB/s = 8 ms. Cuts 34 x ~0.4 ms = 14 ms. Expert compute ~2 ms. Total ≈ 28 ms.
- Mac per token: 14 layers' dense 0.8 GB, experts 14 x ~30 MB = 0.4 GB, and head 0.7 GB, about 1.9 GB at ~250 GB/s ≈ 8 ms. No cuts.
- Two LAN hops of 40 KB each: ~3 ms.
- Serial decode: 28 + 8 + 3 ≈ 39 ms, so ~25 tok/s on paper. The same arithmetic on the 122B predicts ~22 against a measured 17, so I scale by 0.75: **~15-19 tok/s plain**, and **12-16** with a few PC layers on disk. MTP speculation adds something on top (the 122B went from 25 to 30-35 resident on the GB10), less with offload. For comparison, llama.cpp does 18-20 on the PC alone.

These are estimates to be replaced by measurements, as in roadmap 32.

## 5. Correctness plan

Reference: llama.cpp with qwen4exp, on the same bartowski Q4_0 file, greedy. That matches what `tests/test_moe.js` does for the 35B/122B: text plus token ids, with goldens in `tests/golden/*_llama_greedy.json` produced by `tests/golden/make_q122_llama.js` against `llama-server`.

What is on the Spark (this box: GB10, 121 GB RAM, 2.0 TB free):
- `/home/nehanth/llm/llama-unsloth`: the unsloth prebuilt `b10840-mix-d5c17a0` (source `unslothai/llama.cpp` 58670d1, CUDA 13, arm64, built 2026-09-07). **It supports qwen4exp**: `libllama.so` contains `llama_model_qwen4exp`, `graph_mtp`, the QSA/kpool cache and the `ple.*` keys. It has `llama-server`, `llama-cli`, `llama-completion` and `llama-tokenize`, but **no `llama-eval-callback`**.
- `/home/nehanth/llama.cpp`: an upstream checkout at `749f688`, the one used for the 122B goldens. It has a CUDA build with `llama-eval-callback`, but **it predates qwen4exp**. `src/models/qwen4exp.cpp` is missing, and #27742 merged later as `6c84c7d`.

Needed (not done here; read-only):
1. Download bartowski `Qwen3.8-Flash-Next-Q4_0` (3 shards, 100.6 GB) and the ggml-org MTP Q4_0 (2.2 GB) to `models/q38flash/`. It fits in RAM with mmap: ~74 GB hot (experts plus dense), and the 28.8 GB table stays mostly unread (`-ot "per_layer_token_embd=CPU"`). The box shows 33 GB used right now, so stop other big jobs first.
2. Greedy goldens now, with the unsloth `llama-server` (`--temp 0`, `-c 4096`, `-n 64`, `n_probs` for near-tie detection). Use the 3 existing chat prompts plus one with an EOS mid-context to test the n-gram reset. Write `tests/golden/q38flash_llama_greedy.json` with a `make_q38flash_llama.js` clone of the 122B script. Include one 4K-token and one 8K-token prompt now, for the QSA phase later.
3. Per-layer goldens: update `/home/nehanth/llama.cpp` to master (≥ `6c84c7d`) and rebuild `llama-eval-callback` (CUDA), or build `unslothai/llama.cpp` 58670d1 with examples on. Dump the named tensors `model.input_embed`, `ple_embd`, `ple_gate`, `ple_conv_out`, `hc_mixed-N`, `hc_inject-N`, `hc_combine-N`, `l_last-N`, `attn_output-N`, `linear_attn_out-N`, `ffn_moe_out-N`, `result_norm` and `result_output` for a 6-token prompt. eval-callback prints sums and edge values. For full vectors, use a ~40-line callback that writes `.f32` files, as the 27B work did with eval-callback dumps.
4. `tests/reference/ref_q38flash.mjs`: a CPU JS reference (style of `ref_q38.mjs`) for layers 0..N, validated against the dumps above to ~1e-4 relative. HC and PLE are new math, so they get an oracle before any WGSL. Unit tests for the n-gram hash: indices from the JS port vs indices from the llama.cpp `set_input` logic. Use a tiny C++ harness or the reference's own port on fixed token sequences, covering EOS resets and sequence starts.
5. Engine gates, in order:
   - Per-layer hidden vs the JS reference.
   - Greedy ids vs llama.cpp (MATCH 3/3; text and ids).
   - A `test_q38_bits`-style hash: prefill == decode, spec == plain, offloaded == resident.
   - Above 2051 tokens (QSA phase): the 4K/8K goldens, plus an internal check that our QSA equals our dense below the budget (the PR's "free oracle").

   Expect logit-level, not bit-level, agreement with llama.cpp. The PR reports small QSA-path deltas on quantized files even below the budget, so near-ties are excluded from goldens as before.

## 6. PR plan

Each PR is testable on its own. Effort is rough, for one person with agents.

1. **Loader and metadata** (2 days). `qwen4exp` keys in `parseGGUFHeader`; `qwen4expLayerNames` / `qwen4expWeights` (HC, indexer, PLE, head mixer, MTP from a second file); `per_layer_token_embd` excluded from conversion. Test: header fixtures from the three bartowski shards (tensor tables only), with names, shapes and byte sums matching this note; no GPU.
2. **JS reference and goldens** (3-4 days). `ref_q38flash.mjs`, the hash port with unit tests, and greedy plus per-layer goldens from the Spark (§5). Test: the reference matches the eval-callback dumps; the hash matches llama.cpp indices.
3. **N-gram row source** (2 days). `engine/ngram.js`: per-head split files, readers (memory, `pread`, OPFS, HTTP range), Q4_0 row dequant, predecessor-token state with export/import/rollback. Test: rows equal `dequantF32` of the same rows; OPFS reader in a headless Chrome test.
4. **Decode path, resident, ctx <= 2048** (2 weeks). The `hc` mode in `Qwen35Engine`: HC mix/combine kernels, the PLE block and its conv state, the sigmoid gate in the DN norm kernels, K = 10 / 512 experts, the head mixer. Norm fusions are off. Test: on the GB10 (engine ~75 GB GPU, streamed load as for the 122B), per-layer vs the reference, and `MODEL=q38flash test_moe` MATCH 3/3. This PR carries most of the risk.
5. **Batched and prefill paths** (1-1.5 weeks). Multi-column HC/PLE for `_encodeLayerBatch`, `_encodeLayerWide`, `_prefillGrouped` and verify. Re-enable the safe fusions. Test: prefill == decode hidden hashes; MATCH with defaults.
6. **MTP** (1 week). Load the MTP file, 4-stream `eh_proj`, rollback of the PLE conv and n-gram predecessors. Test: spec == plain on the goldens; acceptance logged.
7. **Offload at 512 experts** (3-4 days). `OFFLOAD_EXPERTS` per model, region sizing, `expertSlices` for 512, routing traces. Test: offloaded == resident hashes (all layers, 16 slots); trace hit rates in bench-log. These numbers replace §4's assumed 65-80%.
8. **Room** (1 week). `MODELS` / `SHAPE` / `FILES` / `NEED_GB` / `CTX` (max 2048 until 9), `hDim` across the pipeline, n-gram table placement (the device holding layer 1 needs 28.8 GB of disk; planner field), and the session/checkpoint state for PLE. Test: Spark + PC room MATCH; Mac + PC room tok/s.
9. **QSA** (2-3 weeks). Indexer cache, pooled keys, score kernel, top-512 select, gathered-cell attention for decode and prefill; context cap lifted. Test: QSA == dense below 2051 (bit-equal on our side); 4K/8K goldens vs llama.cpp.
10. **Mac browser tab on disk** (after roadmap 32 items 1 and 8, and ideally 6 and 9). Test: tok/s and TTFT in bench-log against §4's table.

PRs 1-3 can run in parallel. PR 4 needs 1 and 2, and PR 9 needs only 4 and 5. Total ≈ 8-10 weeks to a long-context room. A short-context, exact, resident version on the GB10 comes after PRs 1-5, about 4 weeks.

## Risks and unknowns

- **HC touches the residual contract everywhere.** The residual becomes 4x wider. That reaches every layer encoder, the fused norm GEMVs, room hidden transfers, checkpoints, prefix/session state and MTP. It is the largest code delta and the most likely source of quiet bugs. Mitigation: the JS reference first, and per-layer gates.
- **Context capped at 2048 until QSA lands.** Thinking-mode chat and agent prompts (OpenClaw is ~12K) need PR 9. Without it the model is a demo. Above 2051 tokens, dense attention gives different outputs than the model was trained with (3% of positions at 8K per the PR).
- **The prefill-from-disk TTFT** (§4) makes a lone Mac poor for long prompts even with layer-major prefill.
- **Hit rates with 512 experts / top-10 are unmeasured.** All tok/s numbers hinge on them.
- **Browser limits.** ~25 GB of GPU memory in one tab is untested on the Mac. We have seen a 13 GB browser-process abort in an off-the-record context. OPFS needs a 100 GB quota and `persist()`.
- **PC RAM is tight** for 34 layers of parked experts (48.1 vs 48 GB). The plan needs the disk tier or a slightly different split.
- **Source gaps.** The HF `transformers` implementation (via #27739, huggingface/transformers PR 48337) and the arXiv design paper (2608.30320) were not read; llama.cpp's port is the de facto spec. The PR text gives the table as "97.7 GiB", but the GGUF shape gives 95.4 GiB at 16 bits; minor, unexplained. MTP mainline status is contradictory (above). nitinpanj's README overstates its contents. The 6B-active figure is the card's; my count is ~6.6B.
- **Quant provenance.** bartowski's Q4_0 mixes Q6_K/Q5_0/Q4_1 into a "Q4_0" file. We widen those as with the 122B, which matched llama.cpp, but the head and attention-q widening means our bytes are not the file's bytes. That is fine for goldens on ids, not for bit-level comparison with llama.cpp.
- **JS 64-bit hashing.** One wrong `Number` multiply silently changes every n-gram row and degrades output without obviously breaking it. The hash gets unit tests against llama.cpp indices before anything else.

## Non-goals

Vision (mmproj), YaRN past 262K, lossy shortcuts (fewer experts, a predicted router, a truncated n-gram table), and the K-quant/IQ files.
