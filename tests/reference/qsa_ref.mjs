// Qwen Sparse Attention (QSA) CPU reference for one Qwen3.8-Flash-Next attention layer: decode and prefill.
//
// Written from llama.cpp master f2cc7282 (2026-10-11), the code that PR #27742 added:
//   Q = src/models/qwen4exp.cpp          (graph: build_inp_kpool, build_qsa_sel, build_attn_qsa, build_layer_attn)
//   H = src/llama-memory-hybrid-idx.cpp  (indexer cache, pool layout, per-ubatch inputs: set_input_kpool)
//   O = ggml/src/ggml-cpu/ops.cpp        (CPU ops: lightning_indexer, top_k, rms_norm, rope)
// Citations below are file:line in that commit.
//
// What one QSA layer computes for a token at sequence position p (0-based, the sequence's first cell is 0):
//   1. raw indexer key  kr[p] = index_k_proj . x[p]               [128], cached, no norm, no rope  (Q:761-766)
//   2. blocks: cells 4b .. 4b+3 in sequence order ("by order", Q:82, H:441-445). Block b is complete once
//      cell 4b+3 exists; its pooled key is computed once, when it completes (H:627-631, H:695-701):
//        P[b] = rope(rmsnorm(((kr[4b] + kr[4b+1]) + kr[4b+2]) + kr[4b+3]) * 0.25) * k_norm, pos 4b)  (Q:772-790)
//      rotated at its first member's position (H:871-878: npos = [pos, y, x, pos] with x = y = pos for text,
//      src/llama-kv-cache.cpp ext_set, src/llama-batch.cpp "[p, p, p, 0]").
//   3. indexer query  qi[p] = rope(rmsnorm_head(index_q_proj . x[p]) * q_norm, p)   4 heads x 128     (Q:798-803)
//   4. score of every visible block b (complete, 4b+3 <= p: nv = floor((p+1)/4) blocks, H:930-932)
//        s[b] = sum_h relu(qi[p]_h . P[b]) * (1/sqrt(128))     (Q:805-809, O:12420-12430: per-head weight, summed
//      in head order, mask 0 / -inf added after)
//   5. keep the top min(nv, 512) blocks (Q:813-814: n_top_pool = min(n_pool, top_k / kpool) with n_pool >= nv + 1
//      padded entries scored -inf, H:371-374, so the -inf ones only fill a short selection and are then masked dead,
//      Q:843-858) plus the tail: the (p+1) % 4 cells 4nv .. p of the token's own incomplete block, its own cell
//      included (H:950-980). <= 512*4 + 3 = 2051 cells (Q:727).
//   6. ordinary causal GQA attention (24 q heads, 2 kv heads, hd 256, scale 1/16) over the selected cells only:
//      a -inf mask over the dense KV cache (Q:828-869, Q:874-930). The selection mask "already carries the causal
//      mask" (Q:911) and every selected cell is <= p.
// So with p + 1 <= 2051 every visible block is kept and QSA is exactly dense causal attention.
//
// Ties (step 5): llama.cpp does not define an order. ggml_top_k's output is "unordered" (Q:814); the CPU op is
// std::partial_sort with a strict `>` comparator (O:8778-8815: which equal scores survive at the boundary is an
// artifact of libstdc++'s heap select), CUDA uses radix / bitonic / CUB DeviceTopK with
// determinism::not_guaranteed (ggml/src/ggml-cuda/top-k.cu). Equal scores are common in practice: a block whose
// four head dots are all <= 0 scores exactly 0. We define: score descending, block index ascending (the earliest
// block wins a tie), the same rule as engine/topk.js. The GPU kernels implement the same rule.
//
// Prefill vs decode (H:627-721): llama.cpp re-pools exactly the blocks a ubatch completes (decode: one every 4
// tokens), before scoring, and per token masks out blocks that end after it (pool_mask, H:915-944) and builds the
// tail from its own position. So a prefill token selects exactly what a decode step at that position would; the
// two paths differ only in shapes. This reference runs a chunk of nCols consecutive tokens; nCols = 1 is decode.
//
// Cache state per sequence and QSA layer (what step() mutates):
//   rawK   [maxSeq][128]   raw indexer key per position (llama.cpp keeps it in the idx K cache row, Q:763-766)
//   pooled [maxSeq/4][128] pooled key per complete block (llama.cpp: the pooled half of the block's last row, H:857-859)
//   plus the usual K / V cache of the attention layer.
// The idx cache uses the context's type_k (H:52-73 passes type_k; F16 by default), so llama.cpp rounds raw and pooled keys to
// f16 on store. f16Cache = true (default) does the same here.

export const QSA = Object.freeze({
  KPOOL: 4,          // indexer_compress_ratio, hparams.indexer_kpool (Q:65-83)
  TOP_K: 2048,       // indexer_top_k (Q:58): cells, so 512 blocks
  N_TOP: 512,        // TOP_K / KPOOL
  MAX_SEL: 2051,     // N_TOP * KPOOL + KPOOL - 1 (Q:727)
  IH: 4,             // indexer_n_head (Q:56)
  ID: 128,           // indexer_head_size (Q:57)
  NROT: 64,          // n_rot of the model (partial_rotary_factor 0.25 of 256), shared by the indexer (Q:786, Q:802)
  THETA: 1e7,        // rope freq_base
  EPS: 1e-6,         // f_norm_rms_eps (Q:29)
});

const fr = Math.fround;

// f32 -> nearest f16 value (round to nearest even), as ggml_fp32_to_fp16 does on store
const _f32 = new Float32Array(1), _u32 = new Uint32Array(_f32.buffer);
export function f16bits(x) {
  _f32[0] = x; const i = _u32[0];
  const s = (i >>> 16) & 0x8000; let e = ((i >>> 23) & 0xff) - 127 + 15; const m = i & 0x7fffff;
  if (((i >>> 23) & 0xff) === 0xff) return s | 0x7c00 | (m ? 0x200 : 0);
  if (e >= 31) return s | 0x7c00;
  if (e <= 0) {   // subnormal f16
    if (e < -10) return s;
    const mm = m | 0x800000, sh = 14 - e;
    let h = mm >> sh; const rem = mm & ((1 << sh) - 1), half = 1 << (sh - 1);
    if (rem > half || (rem === half && (h & 1))) h++;
    return s | h;
  }
  let h = s | (e << 10) | (m >> 13);
  if ((m & 0x1fff) > 0x1000 || ((m & 0x1fff) === 0x1000 && (h & 1))) h++;
  return h;
}
export function h2f(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  if (e === 31) return m ? NaN : s * Infinity;
  return e === 0 ? s * m * 2 ** -24 : s * (1 + m / 1024) * 2 ** (e - 15);
}
export const f16round = (x) => h2f(f16bits(x));

// ggml rms_norm (O:4140-4150: sum of squares in double, scale = 1/sqrtf(mean + eps)) then ggml_mul by the weight
export function rmsnorm(x, off, n, w, eps, out, oOff) {
  let sum = 0;
  for (let i = 0; i < n; i++) sum += fr(x[off + i] * x[off + i]);
  const mean = fr(sum / n);
  const scale = fr(1 / fr(Math.sqrt(fr(mean + eps))));
  for (let i = 0; i < n; i++) out[oOff + i] = fr(fr(x[off + i] * scale) * w[i]);
}

// ggml rope_multi, IMROPE, text positions [p, p, p, *]: every rotary pair uses position p, so it is NeoX rope
// (pairs i, i + nRot/2) with theta_i = p * theta_scale^i accumulated in f32 (O:6219-6270, theta_scale O:6371),
// cos / sin of that f32 value (rope_yarn with ext_factor 0, freq_scale 1, mscale 1: O:6186-6201)
export function ropeThetaScale(nRot = QSA.NROT, base = QSA.THETA) { return fr(Math.pow(base, -2 / nRot)); }
export function rope(v, off, pos, nRot = QSA.NROT, base = QSA.THETA) {
  const half = nRot / 2, ts = ropeThetaScale(nRot, base);
  let theta = fr(pos);
  for (let i = 0; i < half; i++) {
    const c = fr(Math.cos(theta)), s = fr(Math.sin(theta));
    const a = v[off + i], b = v[off + i + half];
    v[off + i] = fr(fr(a * c) - fr(b * s));
    v[off + i + half] = fr(fr(a * s) + fr(b * c));
    theta = fr(theta * ts);
  }
}

// the cache of one sequence, one QSA layer
export class QsaRefCache {
  constructor({ maxSeq, f16Cache = true }) {
    this.maxSeq = maxSeq;
    this.f16 = f16Cache;
    this.rawK = new Float32Array(maxSeq * QSA.ID);
    this.pooled = new Float32Array(Math.ceil(maxSeq / QSA.KPOOL) * QSA.ID);
    this.n = 0;   // positions written
  }
  store(x) { return this.f16 ? f16round(x) : fr(x); }
  // rewind to n positions (a rolled-back verify pass): later raw keys are overwritten when re-decoded; the pooled
  // key of a block is recomputed when the block completes again (H:633-660: an edit stales the pools after it)
  rewind(n) { this.n = Math.min(this.n, n); }
}

// step 2: pool block b from the cached raw keys (Q:772-790)
export function poolBlock(cache, b, kNorm) {
  const { ID, KPOOL, EPS } = QSA;
  const sum = new Float32Array(ID);
  for (let i = 0; i < KPOOL; i++) {   // Q:776-780: ((r0 + r1) + r2) + r3
    const r = (KPOOL * b + i) * ID;
    for (let d = 0; d < ID; d++) sum[d] = i === 0 ? cache.rawK[r + d] : fr(sum[d] + cache.rawK[r + d]);
  }
  for (let d = 0; d < ID; d++) sum[d] = fr(sum[d] * fr(1 / KPOOL));   // Q:781 ggml_scale
  const pk = new Float32Array(ID);
  rmsnorm(sum, 0, ID, kNorm, EPS, pk, 0);                              // Q:782
  rope(pk, 0, KPOOL * b);                                              // Q:785-787, position of the first member
  for (let d = 0; d < ID; d++) cache.pooled[b * ID + d] = cache.store(pk[d]);   // Q:793 set_rows into the F16 cache
}

// step 3: the indexer query of one token from its projection qRaw[512] (Q:798-803)
export function indexerQuery(qRaw, off, pos, qNorm) {
  const { IH, ID, EPS } = QSA;
  const q = new Float32Array(IH * ID);
  for (let h = 0; h < IH; h++) { rmsnorm(qRaw, off + h * ID, ID, qNorm, EPS, q, h * ID); rope(q, h * ID, pos); }
  return q;
}

// step 4: scores of blocks 0 .. nv-1 (Q:805-809, O:12420-12430)
export function scoreBlocks(cache, q, nv) {
  const { IH, ID } = QSA;
  const w = fr(1 / fr(Math.sqrt(ID)));
  const s = new Float32Array(nv);
  for (let b = 0; b < nv; b++) {
    let score = 0;
    for (let h = 0; h < IH; h++) {
      let qk = 0;   // ggml_vec_dot_f32 accumulates in f32 (SIMD lanes; the lane order is not reproduced)
      for (let d = 0; d < ID; d++) qk = fr(qk + fr(q[h * ID + d] * cache.pooled[b * ID + d]));
      score = fr(score + fr(Math.max(qk, 0) * w));
    }
    s[b] = score;
  }
  return s;
}

// step 5: the top min(nv, nTop) of s[0 .. nv) by (score desc, index asc), returned in ascending block order
export function selectBlocks(s, nv, nTop = QSA.N_TOP) {
  if (nv <= nTop) return Int32Array.from({ length: nv }, (_, i) => i);
  const idx = Int32Array.from({ length: nv }, (_, i) => i);
  idx.sort((a, b) => (s[b] - s[a]) || (a - b));
  return idx.subarray(0, nTop).sort();
}

// the selected cells of the token at position p: the chosen blocks' members in order, then the tail (H:950-980)
export function selectedCells(blocks, p) {
  const K = QSA.KPOOL, nv = Math.floor((p + 1) / K), tail = (p + 1) % K;
  const cells = new Int32Array(blocks.length * K + tail);
  let j = 0;
  for (const b of blocks) for (let i = 0; i < K; i++) cells[j++] = K * b + i;
  for (let t = 0; t < tail; t++) cells[j++] = K * nv + t;
  return cells;
}

// step 6: GQA attention (float64) of one token over the given cells. q [nH*hd] (normed, roped), K / V [pos][nKV*hd]
// (f32 values; the engine caches f16), scale 1/sqrt(hd) (Q:996). Returns the pre-gate output [nH*hd].
export function attendCells(q, qOff, K, V, cells, { nH, nKV, hd }) {
  const G = nH / nKV, kvDim = nKV * hd, out = new Float64Array(nH * hd), n = cells.length;
  const sc = new Float64Array(n), rs = Math.sqrt(hd);
  for (let h = 0; h < nH; h++) {
    const g = Math.floor(h / G);
    let M = -Infinity;
    for (let j = 0; j < n; j++) {
      const kr = cells[j] * kvDim + g * hd;
      let s = 0; for (let d = 0; d < hd; d++) s += q[qOff + h * hd + d] * K[kr + d];
      sc[j] = s / rs; if (sc[j] > M) M = sc[j];
    }
    let L = 0; for (let j = 0; j < n; j++) { sc[j] = Math.exp(sc[j] - M); L += sc[j]; }
    for (let d = 0; d < hd; d++) {
      let o = 0; for (let j = 0; j < n; j++) o += sc[j] * V[cells[j] * kvDim + g * hd + d];
      out[h * hd + d] = o / L;
    }
  }
  return out;
}

// One pass of nCols consecutive tokens at positions pos .. pos + nCols - 1 (nCols = 1: decode; more: a prefill
// chunk or verify pass). In:
//   qIdxRaw [nCols][512]  index_q_proj . x     kIdxRaw [nCols][128]  index_k_proj . x
//   qNorm, kNorm [128]    the indexer norms
//   q [nCols][nH*hd]      attention q after q_norm and rope (null: selection only)
//   K, V [>= pos+nCols][nKV*hd]  the layer's KV cache with this pass's rows already stored (Q:903-909 stores
//                         before attending)
// Out: per column { blocks, cells, scores (null when nv <= 512: nothing to rank), out (pre-gate attention) }.
// steps 1-2 alone: cache the pass's raw keys and pool the blocks it completes (H:690-713, Q:761-793). Pooling runs
// after every raw key of the pass is stored and before any scoring, as in the graph.
export function ingest(cache, { pos, nCols, kIdxRaw, kNorm }) {
  const { ID, KPOOL } = QSA;
  for (let c = 0; c < nCols; c++) {
    for (let d = 0; d < ID; d++) cache.rawK[(pos + c) * ID + d] = cache.store(kIdxRaw[c * ID + d]);
  }
  cache.n = Math.max(cache.n, pos + nCols);
  const end = pos + nCols;
  for (let b = Math.floor(pos / KPOOL); KPOOL * b + KPOOL - 1 < end; b++) {
    if (KPOOL * b + KPOOL - 1 >= pos) poolBlock(cache, b, kNorm);
  }
}

export function qsaStep(cache, { pos, nCols, qIdxRaw, kIdxRaw, qNorm, kNorm, q = null, K = null, V = null, shape = null }) {
  const { ID, KPOOL, N_TOP } = QSA;
  ingest(cache, { pos, nCols, kIdxRaw, kNorm });
  // 3-6 per token
  const cols = [];
  for (let c = 0; c < nCols; c++) {
    const p = pos + c, nv = Math.floor((p + 1) / KPOOL);
    const qi = indexerQuery(qIdxRaw, c * QSA.IH * ID, p, qNorm);
    const scores = nv > N_TOP ? scoreBlocks(cache, qi, nv) : null;   // nv <= 512: everything is kept, ranks moot
    const blocks = selectBlocks(scores, nv);
    const cells = selectedCells(blocks, p);
    const out = q ? attendCells(q, c * shape.nH * shape.hd, K, V, cells, shape) : null;
    cols.push({ p, nv, qi, scores, blocks, cells, out });
  }
  return cols;
}

// all scores of a column even when nv <= 512 (tests compare the GPU score kernel on short contexts too)
export function scoresFor(cache, qIdxRaw, off, p, qNorm) {
  const nv = Math.floor((p + 1) / QSA.KPOOL);
  return scoreBlocks(cache, indexerQuery(qIdxRaw, off, p, qNorm), nv);
}

// dense causal attention of the token at p (the oracle for p + 1 <= 2051)
export function denseCells(p) { return Int32Array.from({ length: p + 1 }, (_, i) => i); }
