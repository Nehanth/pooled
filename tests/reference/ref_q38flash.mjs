// Qwen3.8-Flash-Next (llama.cpp arch "qwen4exp") CPU reference: the trunk (48 layers) and the head, for a batch
// of tokens of one sequence. Ported from llama.cpp src/models/qwen4exp.cpp at master f2cc728 (line numbers below)
// and checked against llama.cpp's own tensors, dumped by tests/reference/q38dump (llama.cpp CPU backend, f32 KV,
// no flash attention, no weight repacking).
//
// The math is f32/f64 like the existing refs, with one difference: by default every matmul quantizes its
// activations the way llama.cpp's CPU backend does before it multiplies a quantized weight (Q4_0/Q5_0/Q8_0 ->
// Q8_0 blocks, Q4_1 -> Q8_1, Q6_K -> Q8_K, BF16 -> bf16, F16 -> f16), then takes the integer dot per block.
// That is what lets it match llama.cpp per op at 1e-5. The engine's decode GEMVs keep f32 activations, so
// `--act f32` (constructor { act: "f32" }) does that instead; it is ~1e-2 from llama.cpp's tensors (below).
// Weight types handled: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q8_0, Q6_K (bartowski Q4_0 uses all but F16 in
// matmuls); any other type (K-quants other than Q6_K, IQ*) throws. The F16 path rounds activations to f16 as
// ggml does but sums in its own order, so it is not checked to 1e-5 (no F16 matmul in the file it was built on).
//
// Agreement (2026-10-11, 4 sequences, 29 tokens, all 48 layers and the head; 6781 tensors compared):
//  - teacher-forced: 94% of tensors within 1e-5 relative (L2), 99.1% within 1e-4, all within 1e-3 except hc_gate
//    (judged at 5e-3; worst 1.35e-3); the same expert sets everywhere, n-gram rows exact, the same argmax at
//    every token
//  - teacher-forced, --act f32: matmul outputs land around 1e-2 (worst 3.3e-2, hc_gate 1.6e-2, logits
//    0.9e-2..3.3e-2; 89% of tensors within 1e-2); still the same expert sets and argmax at every token
//  - free-running (our own tensors chained): Q8 activation rounding is discontinuous, so small differences grow:
//    with llama.cpp's expert ids forced, layer 47 ends 1.8e-2 off and the logits 2.6e-2, every argmax equal;
//    left alone, a near-tied router flips an expert from layer 1 on, layer 47 ends ~1e-1 off, and one argmax
//    of six flips (a 0.24-logit near tie). So a per-layer gate for the engine should be teacher-forced too.
//    With --act f32 (sequence 0): layer 47 ends 3.6e-2 off with llama.cpp's experts and 9e-2 without; argmax 6/6.
//
// What is new against Qwen3.5 (the Qwen3.8-Flash-Next design note, roadmap/33-qwen38-flash.md, section 1):
//  - hyper-connections: the residual is 4 streams R[4][2560]; every layer norm is a "mix" (per-stream RMS norm,
//    low-rank sigmoid gate, mean over streams) and every residual add is a "combine" (scaled by 2*sigmoid(inj/4))
//  - the DeltaNet output gate is sigmoid(z), not silu(z)
//  - the n-gram embedding (PLE) on layer 1, a gated per-stream injection with a dilated causal conv
//  - 512 experts, top-10; no output_norm (the head mixer is the norm)
// QSA (sparse attention) selects every cell below 2051 tokens, so this reference attends densely, as llama.cpp
// does at these lengths.
//
// usage:
//   node tests/reference/ref_q38flash.mjs <seqDir> [--layers N] [--free [--route]] [--act q8|f32] [--tol 1e-3] [--gate-tol 5e-3] [--quiet]
//     seqDir: one sequence of q38dump output, chunk dirs c0, c1, ... (shapes.tsv, <name>.<k>.f32, tokens.txt),
//             decoded in order on one state (n-gram predecessors, PLE conv history, DeltaNet state, KV).
//             The dumps behind the numbers above are ~/llm/models/q38flash/dumps/s0..s3 (689 MB, not committed);
//             tests/reference/q38dump/q38dump.cpp has the exact command that writes them.
//     --layers N: run layers 0..N-1 (default: all 48, then the head)
//     --free: chain our own tensors (default: teacher-forced, every tapped tensor is replaced by llama.cpp's
//             after it is compared, so each op is checked on llama.cpp's exact inputs)
//     --route: with --free, still take llama.cpp's expert ids (separates routing flips from drift)
//     --act f32: f32 activations in every matmul, as the engine's decode GEMVs (default q8: llama.cpp's CPU rounding)
//     --tol: worst relative L2 allowed per tensor; --gate-tol: the same for hc_gate (see the tap below)
//     Exit 1 when a tensor is over its tolerance, n-gram rows differ, or (teacher-forced or --route) a token's
//     expert set differs from llama.cpp's.
//     --quiet: print only tensors off by more than 1e-5
//   Q38F_DIR=<dir with the 3 GGUF shards> (default models/q38flash)
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { parseGGUFHeader } from "../../engine/gguf.js";
import { pleParams, ngramRows } from "./q38flash_ngram.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const Q38F_DIR = process.env.Q38F_DIR || path.join(HERE, "../../models/q38flash");

// ---- number formats ----

// f16 bits -> value, as a table
const F16 = new Float32Array(65536);
for (let h = 0; h < 65536; h++) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  F16[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
}
const _f64 = new Float64Array(1), _u32 = new Uint32Array(_f64.buffer);
// round to the nearest f16 value, ties to even (the ARM vcvt that GGML_CPU_FP32_TO_FP16 uses)
export function f16round(x) {
  const a = Math.abs(x);
  if (a === 0 || !isFinite(a)) return x;
  if (a >= 65520) return Math.sign(x) * Infinity;
  _f64[0] = a;
  const e = ((_u32[1] >>> 20) & 0x7ff) - 1023;
  const q = 2 ** (Math.max(e, -14) - 10);
  const r = a / q;
  let f = Math.floor(r);
  const diff = r - f;
  if (diff > 0.5 || (diff === 0.5 && (f & 1))) f++;
  return Math.sign(x) * f * q;
}
const _f32 = new Float32Array(1), _b32 = new Uint32Array(_f32.buffer);
// round to bf16, ties to even (ggml_compute_fp32_to_bf16)
export function bf16round(x) {
  _f32[0] = x;
  const u = _b32[0];
  _b32[0] = (((u + (0x7fff + ((u >>> 16) & 1))) >>> 16) << 16) >>> 0;
  return _f32[0];
}
const fr = Math.fround;
// vcvtnq_s32_f32: round half to even
const rne = (v) => { const f = Math.floor(v), d = v - f; return d > 0.5 || (d === 0.5 && (f & 1)) ? f + 1 : f; };
// ggml nearest_int (the q8_K quantizer): also ties to even, via the 1.5*2^23 trick
const nearestInt = (v) => rne(fr(v));

// ---- model file ----

const T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q4_1 = 3, T_Q5_0 = 6, T_Q8_0 = 8, T_Q6_K = 14, T_BF16 = 30;
const ROW_BYTES = { [T_F32]: (n) => n * 4, [T_F16]: (n) => n * 2, [T_BF16]: (n) => n * 2, [T_Q4_0]: (n) => n / 32 * 18,
  [T_Q4_1]: (n) => n / 32 * 20, [T_Q5_0]: (n) => n / 32 * 22, [T_Q8_0]: (n) => n / 32 * 34, [T_Q6_K]: (n) => n / 256 * 210 };

// Open the split GGUF: metadata from shard 1, each tensor's offset within its own shard.
export function openModel(dir = Q38F_DIR) {
  const files = fs.readdirSync(dir).filter((f) => /^Qwen3\.8-Flash-Next-.*-\d{5}-of-\d{5}\.gguf$/.test(f)).sort();
  if (!files.length) throw new Error("no Qwen3.8-Flash-Next shards in " + dir);
  const fds = files.map((f) => fs.openSync(path.join(dir, f), "r"));
  const readAt = (fd, off, len) => {
    const b = Buffer.alloc(len);
    let o = 0;
    while (o < len) { const n = fs.readSync(fd, b, o, Math.min(len - o, 1 << 30), off + o); if (n <= 0) throw new Error("short read"); o += n; }
    return new Uint8Array(b.buffer, b.byteOffset, len);
  };
  let meta = null;
  const tensors = {};
  fds.forEach((fd, i) => {
    const h = readAt(fd, 0, Math.min(64 << 20, fs.fstatSync(fd).size));
    const G = parseGGUFHeader(h.buffer.slice(h.byteOffset, h.byteOffset + h.byteLength), { skipTokenizer: i > 0 });
    if (i === 0) meta = G.meta;
    for (const [n, t] of Object.entries(G.tensors)) tensors[n] = { ...t, fd };
  });
  // a weight [rows x cols] (torch order: shape [.., rows, cols]); `slice` picks one matrix of a 3-D tensor
  const weight = (name, slice = -1) => {
    const t = tensors[name];
    if (!t) throw new Error("missing tensor " + name);
    const cols = t.shape[t.shape.length - 1], rows = t.shape.length > 1 ? t.shape[t.shape.length - 2] : 1;
    if (!ROW_BYTES[t.ggmlType]) throw new Error(`${name}: type ${t.ggmlType} not handled`);
    const rowBytes = ROW_BYTES[t.ggmlType](cols);
    const off = t.byteOffset + (slice >= 0 ? slice * rows * rowBytes : 0);
    return { name, type: t.ggmlType, rows, cols, rowBytes, bytes: readAt(t.fd, off, rows * rowBytes) };
  };
  // row r of a 2-D tensor, as a one-row weight (embedding lookups, the n-gram table)
  const rowOf = (name, r) => {
    const t = tensors[name], cols = t.shape[t.shape.length - 1], rowBytes = ROW_BYTES[t.ggmlType](cols);
    return { name, type: t.ggmlType, rows: 1, cols, rowBytes, bytes: readAt(t.fd, t.byteOffset + r * rowBytes, rowBytes) };
  };
  return { meta, tensors, weight, rowOf, close: () => fds.forEach((fd) => fs.closeSync(fd)) };
}

// Dequantize row r of a weight into out (length cols).
function dequantRow(w, r, out) {
  const b = w.bytes, ro = r * w.rowBytes, n = w.cols;
  switch (w.type) {
    case T_F32: out.set(new Float32Array(b.buffer.slice(b.byteOffset + ro, b.byteOffset + ro + n * 4))); return out;
    case T_F16: for (let i = 0; i < n; i++) out[i] = F16[b[ro + 2 * i] | (b[ro + 2 * i + 1] << 8)]; return out;
    case T_BF16: for (let i = 0; i < n; i++) { _b32[0] = (b[ro + 2 * i] | (b[ro + 2 * i + 1] << 8)) << 16; out[i] = _f32[0]; } return out;
    case T_Q4_0:
      for (let k = 0; k < n / 32; k++) {
        const o = ro + k * 18, d = F16[b[o] | (b[o + 1] << 8)];
        for (let l = 0; l < 16; l++) { const q = b[o + 2 + l]; out[k * 32 + l] = d * ((q & 15) - 8); out[k * 32 + l + 16] = d * ((q >> 4) - 8); }
      }
      return out;
    case T_Q4_1:
      for (let k = 0; k < n / 32; k++) {
        const o = ro + k * 20, d = F16[b[o] | (b[o + 1] << 8)], mn = F16[b[o + 2] | (b[o + 3] << 8)];
        for (let l = 0; l < 16; l++) { const q = b[o + 4 + l]; out[k * 32 + l] = d * (q & 15) + mn; out[k * 32 + l + 16] = d * (q >> 4) + mn; }
      }
      return out;
    case T_Q5_0:
      for (let k = 0; k < n / 32; k++) {
        const o = ro + k * 22, d = F16[b[o] | (b[o + 1] << 8)];
        const qh = (b[o + 2] | (b[o + 3] << 8) | (b[o + 4] << 16) | (b[o + 5] << 24)) >>> 0;
        for (let j = 0; j < 16; j++) {
          const v = b[o + 6 + j], h0 = ((qh >>> j) << 4) & 0x10, h1 = (qh >>> (j + 12)) & 0x10;
          out[k * 32 + j] = d * (((v & 15) | h0) - 16); out[k * 32 + j + 16] = d * (((v >> 4) | h1) - 16);
        }
      }
      return out;
    case T_Q8_0:
      for (let k = 0; k < n / 32; k++) {
        const o = ro + k * 34, d = F16[b[o] | (b[o + 1] << 8)];
        for (let j = 0; j < 32; j++) { const v = b[o + 2 + j]; out[k * 32 + j] = d * (v > 127 ? v - 256 : v); }
      }
      return out;
    case T_Q6_K:                        // dequantize_row_q6_K
      for (let k = 0; k < n / 256; k++) {
        const o = ro + k * 210, d = F16[b[o + 208] | (b[o + 209] << 8)];
        for (let half = 0; half < 2; half++) {
          const ql = o + half * 64, qh = o + 128 + half * 32, sc = o + 192 + half * 8, y = k * 256 + half * 128;
          const s = (i) => { const v = b[sc + i]; return v > 127 ? v - 256 : v; };
          for (let l = 0; l < 32; l++) {
            const is = l >> 4, h = b[qh + l];
            out[y + l] = d * s(is) * (((b[ql + l] & 15) | ((h & 3) << 4)) - 32);
            out[y + l + 32] = d * s(is + 2) * (((b[ql + l + 32] & 15) | (((h >> 2) & 3) << 4)) - 32);
            out[y + l + 64] = d * s(is + 4) * (((b[ql + l] >> 4) | (((h >> 4) & 3) << 4)) - 32);
            out[y + l + 96] = d * s(is + 6) * (((b[ql + l + 32] >> 4) | (((h >> 6) & 3) << 4)) - 32);
          }
        }
      }
      return out;
  }
  throw new Error(`${w.name}: dequant type ${w.type} not handled`);
}

// Dequantize a whole weight (small tensors: norms, conv kernels, embedding rows).
export function dequant(w) {
  const out = new Float32Array(w.rows * w.cols);
  for (let r = 0; r < w.rows; r++) dequantRow(w, r, out.subarray(r * w.cols, (r + 1) * w.cols));
  return out;
}

// ---- matmul with llama.cpp's CPU activation quantization ----

// Quantize one activation row for the given weight type (ggml-cpu type_traits_cpu[type].vec_dot_type).
function quantAct(x, type) {
  const n = x.length;
  if (type === T_F32) return x;
  if (type === T_F16) { const y = new Float32Array(n); for (let i = 0; i < n; i++) y[i] = f16round(x[i]); return y; }
  if (type === T_BF16) { const y = new Float32Array(n); for (let i = 0; i < n; i++) y[i] = bf16round(x[i]); return y; }
  if (type === T_Q6_K) {               // quantize_row_q8_K_ref: d = max/-127 with the signed max, kept in f32
    const nb = n / 256, q = new Int8Array(n), d = new Float32Array(nb);
    for (let k = 0; k < nb; k++) {
      let amax = 0, max = 0;
      for (let j = 0; j < 256; j++) { const ax = Math.abs(x[k * 256 + j]); if (ax > amax) { amax = ax; max = x[k * 256 + j]; } }
      if (!amax) continue;
      const iscale = fr(-127 / max);
      for (let j = 0; j < 256; j++) q[k * 256 + j] = Math.min(127, nearestInt(fr(iscale * x[k * 256 + j])));
      d[k] = fr(1 / iscale);
    }
    return { q, d };
  }
  if (type !== T_Q4_0 && type !== T_Q4_1 && type !== T_Q5_0 && type !== T_Q8_0) throw new Error("quantAct: type " + type);
  // Q8_0 / Q8_1 (arch/arm quantize_row_q8_0/1): d = amax/127 stored as f16, values scaled by the unrounded 1/d
  const nb = n / 32, q = new Int8Array(n), d = new Float32Array(nb), s = new Float32Array(nb);
  for (let k = 0; k < nb; k++) {
    let amax = 0;
    for (let j = 0; j < 32; j++) amax = Math.max(amax, Math.abs(x[k * 32 + j]));
    const dd = fr(amax / 127), id = dd ? fr(1 / dd) : 0;
    let sum = 0;
    for (let j = 0; j < 32; j++) { const v = rne(fr(x[k * 32 + j] * id)); q[k * 32 + j] = v; sum += v; }
    d[k] = f16round(dd);
    s[k] = f16round(fr(dd * sum));
  }
  return { q, d, s };
}

// y[r] = W[r,:] . x for one row r, W's raw bytes against the quantized activation
function dotRow(w, r, a) {
  const b = w.bytes, ro = r * w.rowBytes, nb32 = w.cols / 32;
  let acc = 0;
  switch (w.type) {
    case T_F32: { const f = new Float32Array(b.buffer, b.byteOffset + ro, w.cols); for (let i = 0; i < w.cols; i++) acc += f[i] * a[i]; return acc; }
    // ggml_vec_dot_f16: f16 x f16 products (exact in f32), summed in f32 lanes; the sum order is not emulated
    case T_F16: { for (let i = 0; i < w.cols; i++) acc += F16[b[ro + 2 * i] | (b[ro + 2 * i + 1] << 8)] * a[i]; return acc; }
    case T_BF16: { for (let i = 0; i < w.cols; i++) { _b32[0] = (b[ro + 2 * i] | (b[ro + 2 * i + 1] << 8)) << 16; acc += _f32[0] * a[i]; } return acc; }
    case T_Q4_0: {
      const q = a.q, d = a.d;
      for (let k = 0; k < nb32; k++) {
        const o = ro + k * 18, qo = k * 32;
        let si = 0;
        for (let l = 0; l < 16; l++) { const v = b[o + 2 + l]; si += ((v & 15) - 8) * q[qo + l] + ((v >> 4) - 8) * q[qo + l + 16]; }
        acc += si * F16[b[o] | (b[o + 1] << 8)] * d[k];
      }
      return acc;
    }
    case T_Q4_1: {
      const q = a.q, d = a.d, s = a.s;
      for (let k = 0; k < nb32; k++) {
        const o = ro + k * 20, qo = k * 32;
        let si = 0;
        for (let l = 0; l < 16; l++) { const v = b[o + 4 + l]; si += (v & 15) * q[qo + l] + (v >> 4) * q[qo + l + 16]; }
        acc += si * F16[b[o] | (b[o + 1] << 8)] * d[k] + F16[b[o + 2] | (b[o + 3] << 8)] * s[k];
      }
      return acc;
    }
    case T_Q5_0: {
      const q = a.q, d = a.d;
      for (let k = 0; k < nb32; k++) {
        const o = ro + k * 22, qo = k * 32;
        const qh = (b[o + 2] | (b[o + 3] << 8) | (b[o + 4] << 16) | (b[o + 5] << 24)) >>> 0;
        let si = 0;
        for (let j = 0; j < 16; j++) {
          const v = b[o + 6 + j], h0 = ((qh >>> j) << 4) & 0x10, h1 = (qh >>> (j + 12)) & 0x10;
          si += (((v & 15) | h0) - 16) * q[qo + j] + (((v >> 4) | h1) - 16) * q[qo + j + 16];
        }
        acc += si * F16[b[o] | (b[o + 1] << 8)] * d[k];
      }
      return acc;
    }
    case T_Q8_0: {
      const q = a.q, d = a.d;
      for (let k = 0; k < nb32; k++) {
        const o = ro + k * 34, qo = k * 32;
        let si = 0;
        for (let j = 0; j < 32; j++) { const v = b[o + 2 + j]; si += (v > 127 ? v - 256 : v) * q[qo + j]; }
        acc += si * F16[b[o] | (b[o + 1] << 8)] * d[k];
      }
      return acc;
    }
    case T_Q6_K: {                      // ggml_vec_dot_q6_K_q8_K: 16 groups of 16 with an int8 scale each
      const q = a.q, d = a.d;
      for (let k = 0; k < w.cols / 256; k++) {
        const o = ro + k * 210, qo = k * 256;
        let si = 0;
        for (let half = 0; half < 2; half++) {
          const ql = o + half * 64, qh = o + 128 + half * 32, sc = o + 192 + half * 8, y = qo + half * 128;
          for (let l = 0; l < 32; l++) {
            const is = l >> 4, h = b[qh + l];
            const s = (i) => { const v = b[sc + is + i]; return v > 127 ? v - 256 : v; };
            si += s(0) * (((b[ql + l] & 15) | ((h & 3) << 4)) - 32) * q[y + l];
            si += s(2) * (((b[ql + l + 32] & 15) | (((h >> 2) & 3) << 4)) - 32) * q[y + l + 32];
            si += s(4) * (((b[ql + l] >> 4) | (((h >> 4) & 3) << 4)) - 32) * q[y + l + 64];
            si += s(6) * (((b[ql + l + 32] >> 4) | (((h >> 6) & 3) << 4)) - 32) * q[y + l + 96];
          }
        }
        acc += F16[b[o + 208] | (b[o + 209] << 8)] * d[k] * si;
      }
      return acc;
    }
  }
  throw new Error(`${w.name}: matmul type ${w.type} not handled`);
}

// Y [T x rows] = X [T x cols] W^T.
// act "q8" (default): each activation row is quantized for the weight's type as llama.cpp's CPU backend does.
// act "f32": f32 activations against the dequantized weight row, as the engine's decode GEMVs do (only its dp4a
// prefill quantizes activations, Q8_1-style). Against the q38dump tensors this lands near 1e-2, not 1e-5.
export function matmul(w, X, T, act = "q8") {
  const Y = new Float32Array(T * w.rows);
  if (act === "f32") {
    const row = new Float32Array(w.cols);
    for (let r = 0; r < w.rows; r++) {
      dequantRow(w, r, row);
      for (let t = 0; t < T; t++) {
        let acc = 0;
        for (let i = 0, o = t * w.cols; i < w.cols; i++) acc += row[i] * X[o + i];
        Y[t * w.rows + r] = acc;
      }
    }
    return Y;
  }
  if (act !== "q8") throw new Error("matmul: act must be q8 or f32, not " + act);
  for (let t = 0; t < T; t++) {
    const a = quantAct(X.subarray(t * w.cols, (t + 1) * w.cols), w.type);
    for (let r = 0; r < w.rows; r++) Y[t * w.rows + r] = dotRow(w, r, a);
  }
  return Y;
}

// ---- elementwise ----

const sigmoid = (v) => 1 / (1 + Math.exp(-v));
const silu = (v) => v / (1 + Math.exp(-v));
const softplus = (v) => (v > 20 ? v : Math.log1p(Math.exp(v)));

// ggml_rms_norm over runs of n (f64 sum, f32 scale), times w (length n, or a [groups x n] weight cycled per run)
function rmsnorm(x, n, eps, w = null) {
  const y = new Float32Array(x.length), wl = w ? w.length : 0;
  for (let o = 0; o < x.length; o += n) {
    let ss = 0;
    for (let i = 0; i < n; i++) ss += x[o + i] * x[o + i];
    const sc = fr(1 / Math.sqrt(fr(ss / n + eps)));
    const wo = w ? o % wl : 0;
    for (let i = 0; i < n; i++) y[o + i] = w ? fr(x[o + i] * sc) * w[wo + i] : x[o + i] * sc;
  }
  return y;
}

// ---- the model ----

export class Q38FlashRef {
  // tap(name, tensor) is called for every named tensor (llama.cpp's cb() names); it may return a replacement
  // (teacher forcing) or nothing. act: "q8" (llama.cpp CPU activation rounding, the default) or "f32" (see matmul).
  constructor(model, tap = () => null, { act = "q8" } = {}) {
    this.m = model;
    this.tap = tap;
    this.act = act;
    const M = model.meta, A = (k) => M["qwen4exp." + k];
    this.hp = {
      nLayer: A("block_count"), d: A("embedding_length"), eps: A("attention.layer_norm_rms_epsilon"),
      hc: A("hyper_connection.count"), hcRank: A("hyper_connection.low_rank"),
      nH: A("attention.head_count"), nKV: A("attention.head_count_kv"), hd: A("attention.key_length"),
      nRot: A("rope.dimension_count"), ropeBase: A("rope.freq_base"),
      dConv: A("ssm.conv_kernel"), dState: A("ssm.state_size"), nKH: A("ssm.group_count"), nVH: A("ssm.time_step_rank"),
      dInner: A("ssm.inner_size"), interval: A("full_attention_interval"),
      nExp: A("expert_count"), nUsed: A("expert_used_count"),
    };
    this.P = pleParams(M);
    this.vocab = model.tensors["output.weight"].shape[0];
  }
  mm(w, X, T) { return matmul(w, X, T, this.act); }
  _t(name, x) { const r = this.tap(name, x); return r || x; }
  isRecurrent(il) { return (il + 1) % this.hp.interval !== 0; }

  // State for one sequence: DeltaNet conv/state, KV, PLE conv history, n-gram predecessors.
  newState() {
    const { nLayer, dState, nKH, nVH, dInner, dConv, d, hc } = this.hp;
    const convDim = 2 * nKH * dState + dInner;
    return {
      pos: 0, prev: [],
      layers: Array.from({ length: nLayer }, (_, il) => this.isRecurrent(il)
        ? { conv: new Float32Array(convDim * (dConv - 1)), S: new Float32Array(nVH * dState * dState) }
        : { k: [], v: [] }),
      // the PLE conv's input history: (kernel-1)*dilation positions of hc*d values, zero at the sequence start
      pleHist: new Float32Array((this.P.conv - 1) * this.P.ngram * hc * d),
    };
  }

  // hc_mix (qwen4exp.cpp:313-367). R: [T][hc][d]. Returns { mixed [T][d], inject [T][hc] | null }.
  hcMix(R, T, wNorm, wDown, wUp, wInject, il, tag) {
    const { d, hc, eps } = this.hp, HD = hc * d;
    const xn = rmsnorm(R, d, eps, wNorm);
    let lo = this.mm(wDown, xn, T);
    for (let i = 0; i < lo.length; i++) lo[i] = fr(silu(fr(lo[i] / hc)));
    const gate = this._t("hc_gate" + tag, this.mm(wUp, lo, T));
    let mixed = new Float32Array(T * d);
    for (let t = 0; t < T; t++)
      for (let i = 0; i < d; i++) {
        let s = 0;
        for (let c = 0; c < hc; c++) { const j = t * HD + c * d + i; s += fr(xn[j] * fr(sigmoid(gate[j]))); }
        mixed[t * d + i] = fr(s / hc);
      }
    mixed = this._t(il < 0 ? "result_norm" : "hc_mixed" + tag, mixed);
    const inject = wInject ? this._t("hc_inject" + tag, this.mm(wInject, xn, T)) : null;
    return { mixed, inject };
  }

  // hc_combine (qwen4exp.cpp:369-397): R[c] += 2*sigmoid(inj[c]/hc) * out
  hcCombine(R, out, inj, T, name) {
    const { d, hc } = this.hp, y = new Float32Array(R.length);
    for (let t = 0; t < T; t++)
      for (let c = 0; c < hc; c++) {
        const w = fr(2 * fr(sigmoid(fr(inj[t * hc + c] / hc))));
        for (let i = 0; i < d; i++) { const j = (t * hc + c) * d + i; y[j] = fr(R[j] + fr(out[t * d + i] * w)); }
      }
    return this._t(name, y);
  }

  // n-gram embedding rows for this chunk, gathered and dequantized: [T][16*160]
  pleEmbd(st, ids) {
    const P = this.P, rows = ngramRows(P, ids, st.prev.slice(-(P.ngram - 1)));
    this.tap("ple_rows", rows);
    const emb = new Float32Array(ids.length * P.nHeads * P.headDim);
    for (let i = 0; i < rows.length; i++) emb.set(dequant(this.m.rowOf("per_layer_token_embd.weight", rows[i])), i * P.headDim);
    return this._t("ple_embd", emb);
  }

  // build_ple (qwen4exp.cpp:1373-1461): the gated, per-stream n-gram injection with its dilated causal conv
  ple(st, il, emb, R, T) {
    const { d, hc, eps } = this.hp, HD = hc * d, m = this.m, p = `blk.${il}.`;
    const nk = dequant(m.weight(p + "ple_norm_key.weight")), nq = dequant(m.weight(p + "ple_norm_query.weight"));
    const nc = dequant(m.weight(p + "ple_norm_conv.weight")), cw = dequant(m.weight(p + "ple_conv1d.weight")); // [HD][kern]
    const key = rmsnorm(this.mm(m.weight(p + "ple_key.weight"), emb, T), d, eps, nk);
    const value = this.mm(m.weight(p + "ple_value.weight"), emb, T);
    const query = rmsnorm(R, d, eps, nq);
    let gate = new Float32Array(T * hc);
    for (let t = 0; t < T; t++)
      for (let c = 0; c < hc; c++) {
        let s = 0;
        for (let i = 0; i < d; i++) s += fr(key[(t * hc + c) * d + i] * query[(t * hc + c) * d + i]);
        s = fr(fr(s) / Math.sqrt(d));
        gate[t * hc + c] = fr(sigmoid(Math.sign(s) * fr(Math.sqrt(Math.max(Math.abs(s), 1e-6)))));
      }
    gate = this._t(`ple_gate-${il}`, gate);
    let gated = new Float32Array(T * HD);
    for (let t = 0; t < T; t++) for (let c = 0; c < hc; c++) for (let i = 0; i < d; i++)
      gated[(t * hc + c) * d + i] = fr(value[t * d + i] * gate[t * hc + c]);
    gated = this._t(`ple_gated_value-${il}`, gated);
    const nrm = rmsnorm(gated, d, eps, nc);
    // depthwise causal conv over time, kernel K, dilation = ngram size: taps at t-(K-1)*dil .. t
    const K = cw.length / HD, dil = this.P.ngram, hist = (K - 1) * dil;
    const H = st.pleHist;                     // the last `hist` normalized inputs, oldest first: [hist][HD]
    const seq = new Float32Array((hist + T) * HD);
    seq.set(H, 0);
    seq.set(nrm, hist * HD);
    let conv = new Float32Array(T * HD);
    for (let t = 0; t < T; t++)
      for (let ch = 0; ch < HD; ch++) {
        let acc = 0;
        for (let k = 0; k < K; k++) acc = fr(acc + fr(seq[(hist + t - (K - 1 - k) * dil) * HD + ch] * cw[ch * K + k]));
        conv[t * HD + ch] = fr(silu(acc));
      }
    st.pleHist = seq.slice(T * HD);
    conv = this._t(`ple_conv_out-${il}`, conv);
    const y = new Float32Array(R.length);
    for (let j = 0; j < R.length; j++) y[j] = fr(R[j] + fr(gated[j] + conv[j]));
    return y;
  }

  // Gated DeltaNet (qwen4exp.cpp:1019-1144), token by token
  deltaNet(st, il, x, T) {
    const { eps, dConv, dState, nKH, nVH, dInner } = this.hp, m = this.m, p = `blk.${il}.`;
    const keyDim = dState * nKH, convDim = 2 * keyDim + dInner, K1 = dConv - 1;
    const qkv = this.mm(m.weight(p + "attn_qkv.weight"), x, T);
    const z = this._t(`z-${il}`, this.mm(m.weight(p + "attn_gate.weight"), x, T));
    const beta = this.mm(m.weight(p + "ssm_beta.weight"), x, T), alpha = this.mm(m.weight(p + "ssm_alpha.weight"), x, T);
    const dtB = dequant(m.weight(p + "ssm_dt.bias")), A = dequant(m.weight(p + "ssm_a"));
    const cw = dequant(m.weight(p + "ssm_conv1d.weight")), nw = dequant(m.weight(p + "ssm_norm.weight"));
    const S = st.S, conv = st.conv, scale = 1 / Math.sqrt(dState);
    let out = new Float32Array(T * dInner);
    for (let t = 0; t < T; t++) {
      const xq = qkv.subarray(t * convDim, (t + 1) * convDim);
      const co = new Float32Array(convDim);
      for (let c = 0; c < convDim; c++) {
        let acc = cw[c * dConv + K1] * xq[c];
        for (let j = 0; j < K1; j++) acc += cw[c * dConv + j] * conv[c * K1 + j];
        co[c] = silu(acc);
        for (let j = 0; j < K1 - 1; j++) conv[c * K1 + j] = conv[c * K1 + j + 1];
        conv[c * K1 + K1 - 1] = xq[c];
      }
      const l2 = (o, n) => { let ss = 0; for (let i = 0; i < n; i++) ss += co[o + i] * co[o + i]; const inv = 1 / Math.max(Math.sqrt(ss), eps); for (let i = 0; i < n; i++) co[o + i] *= inv; };
      for (let h = 0; h < nKH; h++) { l2(h * dState, dState); l2(keyDim + h * dState, dState); }
      for (let h = 0; h < nVH; h++) {
        const kh = h % nKH, qo = kh * dState, ko = keyDim + kh * dState, vo = 2 * keyDim + h * dState, Sb = h * dState * dState;
        const g = softplus(alpha[t * nVH + h] + dtB[h]) * A[h], b = sigmoid(beta[t * nVH + h]), decay = Math.exp(g);
        for (let e = 0; e < dState * dState; e++) S[Sb + e] *= decay;
        const vhat = new Float64Array(dState);
        for (let i = 0; i < dState; i++) { const ki = co[ko + i], row = Sb + i * dState; for (let j = 0; j < dState; j++) vhat[j] += S[row + j] * ki; }
        for (let j = 0; j < dState; j++) vhat[j] = (co[vo + j] - vhat[j]) * b;
        for (let i = 0; i < dState; i++) { const ki = co[ko + i], row = Sb + i * dState; for (let j = 0; j < dState; j++) S[row + j] += ki * vhat[j]; }
        const o = t * dInner + h * dState;
        for (let i = 0; i < dState; i++) { const qi = co[qo + i] * scale, row = Sb + i * dState; for (let j = 0; j < dState; j++) out[o + j] += S[row + j] * qi; }
      }
    }
    out = this._t(`attn_output-${il}`, out);   // llama.cpp names the delta-rule output "attn_output" too
    // gated norm: rmsnorm(o_h) * ssm_norm * sigmoid(z), the one change from Qwen3.5's silu(z) (qwen4exp.cpp:638-648)
    out = rmsnorm(out, dState, eps, nw);
    for (let i = 0; i < out.length; i++) out[i] = fr(out[i] * fr(sigmoid(z[i])));
    out = this._t(`final_output-${il}`, out);
    return this._t(`linear_attn_out-${il}`, this.mm(m.weight(p + "ssm_out.weight"), out, T));
  }

  // RoPE on the first nRot dims of each head, NeoX pairs (i, i + nRot/2); text positions are equal in every
  // IMRoPE section, so this is the 1-D rotation. theta follows ggml's f32 recurrence theta *= base^(-2/nRot).
  rope(v, heads, pos) {
    const { hd, nRot, ropeBase } = this.hp, half = nRot / 2, ts = fr(Math.pow(ropeBase, -2 / nRot));
    let th = fr(pos);
    for (let i = 0; i < half; i++) {
      const c = fr(Math.cos(th)), s = fr(Math.sin(th));
      for (let h = 0; h < heads; h++) {
        const o = h * hd, a = v[o + i], b = v[o + i + half];
        v[o + i] = fr(a * c - b * s); v[o + i + half] = fr(a * s + b * c);
      }
      th = fr(th * ts);
    }
  }

  // Gated full attention (qwen4exp.cpp:932-1017), dense causal (== QSA below 2051 cells)
  attention(st, il, x, T) {
    const { eps, nH, nKV, hd } = this.hp, m = this.m, p = `blk.${il}.`;
    const qf = this.mm(m.weight(p + "attn_q.weight"), x, T), kr = this.mm(m.weight(p + "attn_k.weight"), x, T);
    const vr = this.mm(m.weight(p + "attn_v.weight"), x, T);
    const qn = dequant(m.weight(p + "attn_q_norm.weight")), kn = dequant(m.weight(p + "attn_k_norm.weight"));
    let out = new Float32Array(T * nH * hd);
    for (let t = 0; t < T; t++) {
      const q = new Float32Array(nH * hd), g = new Float32Array(nH * hd);
      for (let h = 0; h < nH; h++) {
        q.set(qf.subarray(t * nH * 2 * hd + h * 2 * hd, t * nH * 2 * hd + h * 2 * hd + hd), h * hd);
        g.set(qf.subarray(t * nH * 2 * hd + h * 2 * hd + hd, t * nH * 2 * hd + (h + 1) * 2 * hd), h * hd);
      }
      const qq = rmsnorm(q, hd, eps, qn), kk = rmsnorm(kr.subarray(t * nKV * hd, (t + 1) * nKV * hd), hd, eps, kn);
      const pos = st.pos + t;
      this.rope(qq, nH, pos); this.rope(kk, nKV, pos);
      st.layers[il].k.push(kk); st.layers[il].v.push(Float32Array.from(vr.subarray(t * nKV * hd, (t + 1) * nKV * hd)));
      const K = st.layers[il].k, V = st.layers[il].v, n = K.length, sc = new Float64Array(n), grp = nH / nKV;
      for (let h = 0; h < nH; h++) {
        const kv = Math.floor(h / grp);
        let mx = -Infinity;
        for (let j = 0; j < n; j++) { let a = 0; for (let i = 0; i < hd; i++) a += qq[h * hd + i] * K[j][kv * hd + i]; sc[j] = fr(a) / Math.sqrt(hd); mx = Math.max(mx, sc[j]); }
        let sum = 0;
        for (let j = 0; j < n; j++) { sc[j] = Math.exp(sc[j] - mx); sum += sc[j]; }
        for (let j = 0; j < n; j++) { const w = sc[j] / sum; for (let i = 0; i < hd; i++) out[t * nH * hd + h * hd + i] += w * V[j][kv * hd + i]; }
      }
      for (let i = 0; i < nH * hd; i++) out[t * nH * hd + i] = fr(out[t * nH * hd + i] * fr(sigmoid(g[i])));
    }
    out = this._t(`attn_gated-${il}`, out);
    return this._t(`attn_output-${il}`, this.mm(m.weight(p + "attn_output.weight"), out, T));
  }

  // MoE (qwen4exp.cpp:1146-1194): softmax over all experts, top-K renormalised, SwiGLU experts, plus the
  // shared expert times sigmoid(gate_inp_shexp . x)
  moe(il, x, T) {
    const { d, nExp, nUsed } = this.hp, m = this.m, p = `blk.${il}.`;
    const logits = this.mm(m.weight(p + "ffn_gate_inp.weight"), x, T);
    let top = new Int32Array(T * nUsed), wts = new Float32Array(T * nUsed);
    for (let t = 0; t < T; t++) {
      const lg = logits.subarray(t * nExp, (t + 1) * nExp);
      let mx = -Infinity; for (const v of lg) mx = Math.max(mx, v);
      let sum = 0; const pr = new Float64Array(nExp);
      for (let e = 0; e < nExp; e++) { pr[e] = Math.exp(lg[e] - mx); sum += pr[e]; }
      for (let e = 0; e < nExp; e++) pr[e] = fr(pr[e] / sum);
      const order = Array.from(pr.keys()).sort((a, b) => pr[b] - pr[a] || a - b).slice(0, nUsed);
      top.set(order, t * nUsed);
    }
    top = this._t(`ffn_moe_topk-${il}`, top);
    for (let t = 0; t < T; t++) {
      const lg = logits.subarray(t * nExp, (t + 1) * nExp);
      let mx = -Infinity; for (const v of lg) mx = Math.max(mx, v);
      let sum = 0; for (let e = 0; e < nExp; e++) sum += Math.exp(lg[e] - mx);
      let ws = 0;
      for (let k = 0; k < nUsed; k++) { wts[t * nUsed + k] = fr(Math.exp(lg[top[t * nUsed + k]] - mx) / sum); ws += wts[t * nUsed + k]; }
      ws = Math.max(fr(ws), 6.103515625e-5);
      for (let k = 0; k < nUsed; k++) wts[t * nUsed + k] = fr(wts[t * nUsed + k] / ws);
    }
    wts = this._t(`ffn_moe_weights_norm-${il}`, wts);
    let moe = new Float32Array(T * d);
    const cache = new Map();
    const expert = (e) => {
      if (!cache.has(e)) cache.set(e, [m.weight(p + "ffn_gate_exps.weight", e), m.weight(p + "ffn_up_exps.weight", e), m.weight(p + "ffn_down_exps.weight", e)]);
      return cache.get(e);
    };
    const swiglu = (wg, wu, wd, xt) => {
      const g = this.mm(wg, xt, 1), u = this.mm(wu, xt, 1), h = new Float32Array(g.length);
      for (let i = 0; i < g.length; i++) h[i] = fr(fr(silu(g[i])) * u[i]);
      return this.mm(wd, h, 1);
    };
    for (let t = 0; t < T; t++) {
      const xt = x.subarray(t * d, (t + 1) * d);
      for (let k = 0; k < nUsed; k++) {
        const [wg, wu, wd] = expert(top[t * nUsed + k]), y = swiglu(wg, wu, wd, xt), w = wts[t * nUsed + k];
        for (let i = 0; i < d; i++) moe[t * d + i] = fr(moe[t * d + i] + fr(y[i] * w));
      }
    }
    moe = this._t(`ffn_moe_out-${il}`, moe);
    const sg = this.mm(m.weight(p + "ffn_gate_inp_shexp.weight"), x, T);
    const [wg, wu, wd] = [m.weight(p + "ffn_gate_shexp.weight"), m.weight(p + "ffn_up_shexp.weight"), m.weight(p + "ffn_down_shexp.weight")];
    let sh = new Float32Array(T * d);
    for (let t = 0; t < T; t++) {
      const y = swiglu(wg, wu, wd, x.subarray(t * d, (t + 1) * d)), gt = fr(sigmoid(sg[t]));
      for (let i = 0; i < d; i++) sh[t * d + i] = fr(y[i] * gt);
    }
    sh = this._t(`ffn_shexp_gated-${il}`, sh);
    const out = new Float32Array(T * d);
    for (let i = 0; i < out.length; i++) out[i] = fr(moe[i] + sh[i]);
    return this._t(`ffn_out-${il}`, out);
  }

  // One chunk of tokens through layers 0..nLayers-1, then the head if all layers ran.
  // Returns { R, logits } (logits null when the head did not run).
  forward(st, ids, nLayers = this.hp.nLayer) {
    const { d, hc } = this.hp, m = this.m, T = ids.length;
    let embd = new Float32Array(T * d);
    for (let t = 0; t < T; t++) embd.set(dequant(m.rowOf("token_embd.weight", ids[t])), t * d);
    embd = this._t("model.input_embed", embd);
    const emb = this.P.layers && this.P.layers.length ? this.pleEmbd(st, ids) : null;
    let R = new Float32Array(T * hc * d);
    for (let t = 0; t < T; t++) for (let c = 0; c < hc; c++) R.set(embd.subarray(t * d, (t + 1) * d), (t * hc + c) * d);
    R = this._t("hc_init", R);
    for (let il = 0; il < nLayers; il++) {
      const p = `blk.${il}.`, w = (s) => m.weight(p + s);
      if (this.P.layers.includes(il)) R = this.ple(st, il, emb, R, T);
      let { mixed, inject } = this.hcMix(R, T, dequant(w("hc_attn_norm.weight")), w("hc_attn_down.weight"), w("hc_attn_up.weight"), w("hc_attn_inject.weight"), il, `-${il}`);
      const out = this.isRecurrent(il) ? this.deltaNet(st.layers[il], il, mixed, T) : this.attention(st, il, mixed, T);
      R = this.hcCombine(R, out, inject, T, `hc_combine-${il}`);
      ({ mixed, inject } = this.hcMix(R, T, dequant(w("hc_ffn_norm.weight")), w("hc_ffn_down.weight"), w("hc_ffn_up.weight"), w("hc_ffn_inject.weight"), il, `-${il}`));
      R = this.hcCombine(R, this.moe(il, mixed, T), inject, T, `l_last-${il}`);
    }
    st.prev.push(...ids);
    st.pos += T;
    if (nLayers < this.hp.nLayer) return { R, logits: null };
    const { mixed } = this.hcMix(R, T, dequant(m.weight("output_hc_norm.weight")), m.weight("output_hc_down.weight"), m.weight("output_hc_up.weight"), null, -1, "");
    return { R, logits: this._t("result_output", this.mm(m.weight("output.weight"), mixed, T)) };
  }
}

// ---- llama.cpp dump comparison (CLI) ----

// q38dump's chunk directory: shapes.tsv lists "<name>.<k>  op  ne0 ne1 ne2 ne3  type"
export function loadDump(dir) {
  const out = new Map();
  for (const line of fs.readFileSync(path.join(dir, "shapes.tsv"), "utf8").trim().split("\n")) {
    const [key, , , , , , type] = line.split("\t");
    const f = path.join(dir, key + (type === "i32" ? ".i32" : ".f32"));
    const b = fs.readFileSync(f);
    out.set(key, type === "i32" ? new Int32Array(b.buffer, b.byteOffset, b.length / 4) : new Float32Array(b.buffer, b.byteOffset, b.length / 4));
  }
  return out;
}

export function relErr(a, b) {
  let num = 0, den = 0, mx = 0;
  for (let i = 0; i < b.length; i++) { const e = a[i] - b[i]; num += e * e; den += b[i] * b[i]; mx = Math.max(mx, Math.abs(e)); }
  return { rel: Math.sqrt(num / (den || 1)), max: mx };
}

// One sequence of q38dump output: <seqDir>/c0, c1, ... decoded in order on one state; tokens in each chunk's tokens.txt
async function main() {
  const args = process.argv.slice(2), dir = args[0];
  const opt = (k, dflt) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : dflt; };
  if (!dir) { console.error("usage: node tests/reference/ref_q38flash.mjs <seqDir> [--layers N] [--free [--route]] [--act q8|f32] [--tol 1e-3] [--gate-tol 5e-3] [--quiet]"); process.exit(2); }
  const nLayers = +opt("--layers", 48), free = args.includes("--free"), route = args.includes("--route"), tol = +opt("--tol", 1e-3);
  const gateTol = +opt("--gate-tol", 5e-3), act = opt("--act", "q8"), quiet = args.includes("--quiet");
  // expert sets are judged whenever llama.cpp's inputs are fed back (teacher-forced, or --route); left free,
  // a near-tied router may legitimately pick another expert
  const judgeExperts = !free || route;
  const chunks = fs.readdirSync(dir).filter((f) => /^c\d+$/.test(f)).sort((a, b) => +a.slice(1) - +b.slice(1));
  let dump = null, seen = null, rows = null, worst = 0, worstGate = 0, expertMiss = 0, ids = null;
  const tap = (name, x) => {
    const k = seen.get(name) || 0;
    seen.set(name, k + 1);
    const key = `${name}.${k}`, ref = dump.get(key);
    if (!ref) return null;
    if (ref instanceof Int32Array) {
      // expert ids: the same set per token (the order of equal probabilities may differ); n-gram rows: exact
      let bad = 0;
      if (name.startsWith("ffn_moe_topk")) {
        const kk = ref.length / ids.length;
        for (let t = 0; t < ids.length; t++) { const a = new Set(x.subarray(t * kk, (t + 1) * kk)); for (let j = 0; j < kk; j++) if (!a.has(ref[t * kk + j])) bad++; }
      } else for (let i = 0; i < ref.length; i++) if (ref[i] !== x[i]) bad++;
      rows.push([key, bad ? `${bad} differ` : "exact"]);
      if (bad && name.startsWith("ffn_moe_topk")) expertMiss += bad;
      else if (bad) worst = Infinity;
      return free && !(route && name.startsWith("ffn_moe_topk")) ? null : ref;
    }
    if (ref.length !== x.length) { rows.push([key, `size ${x.length} vs ${ref.length}`]); worst = Infinity; return null; }
    const { rel, max } = relErr(x, ref);
    rows.push([key, `rel ${rel.toExponential(2)}  max|d| ${max.toExponential(2)}`]);
    // hc_gate (the mixer's pre-sigmoid gate) gets its own, looser tolerance: one Q8 rounding flip in its 320-wide
    // input moves it by up to ~1.4e-3 while the hc_mixed it feeds stays at ~1e-7
    if (name.startsWith("hc_gate")) worstGate = Math.max(worstGate, rel);
    else worst = Math.max(worst, rel);
    return free ? null : Float32Array.from(ref);
  };
  const ref = new Q38FlashRef(openModel(), tap, { act });
  const st = ref.newState();
  const t0 = performance.now();
  for (const c of chunks) {
    const cd = path.join(dir, c);
    ids = fs.readFileSync(path.join(cd, "tokens.txt"), "utf8").trim().split(/[\s,]+/).map(Number);
    dump = loadDump(cd); seen = new Map(); rows = [];
    const { logits } = ref.forward(st, ids, nLayers);
    console.log(`-- ${c}: tokens ${ids.join(",")}`);
    for (const [k, v] of rows) if (!quiet || (v !== "exact" && !(+(/rel (\S+)/.exec(v) || [])[1] <= 1e-5))) console.log("  " + k.padEnd(28), v);
    if (logits) {
      const V = ref.vocab, d = dump.get("result_output.0");
      const am = (a, t) => { let b = 0; for (let i = 1; i < V; i++) if (a[t * V + i] > a[t * V + b]) b = i; return b; };
      for (let t = 0; t < ids.length; t++) console.log(`  token ${t}: argmax ours ${am(logits, t)} llama.cpp ${d ? am(d, t) : "?"}`);
    }
  }
  const mode = (free ? (route ? "free-running, llama.cpp's experts" : "free-running") : "teacher-forced") + `, ${act} activations`;
  console.log(`${mode}, layers 0..${nLayers - 1}: worst rel ${worst.toExponential(2)} (tol ${tol}), hc_gate ${worstGate.toExponential(2)} ` +
    `(tol ${gateTol}), expert ids not in llama.cpp's set: ${expertMiss}${judgeExperts ? "" : " (not judged)"} in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  process.exit(worst <= tol && worstGate <= gateTol && !(judgeExperts && expertMiss) ? 0 : 1);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) main();
