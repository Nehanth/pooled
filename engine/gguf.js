// GGUF loading: header and tensor parsing, tokenizer extraction, Q4_0/Q8_0
// repacking into GPU-friendly layouts (packed nibbles + f16 scales), and
// streaming upload straight into GPU buffers. See docs/models.md.
// Parses GGUF v2/v3 and handles Q4_0, Q4_1, Q5_0, Q5_K, Q6_K, Q8_0, F16, BF16
// and F32 tensors, plus the qwen35 loaders. The parser returns tensor byte
// ranges so browser peers can range-fetch only their layers.

const T_U8 = 0, T_I8 = 1, T_U16 = 2, T_I16 = 3, T_U32 = 4, T_I32 = 5, T_F32 = 6,
  T_BOOL = 7, T_STR = 8, T_ARR = 9, T_U64 = 10, T_I64 = 11, T_F64 = 12;

// ggml tensor types we support
export const GGML_F32 = 0, GGML_F16 = 1, GGML_Q8_0 = 8;
export const QK8_0 = 32;                 // elems per Q8_0 block
export const Q8_0_BLOCK_BYTES = 34;      // f16 scale + 32 int8

const _f16buf = new Float32Array(1), _f16u32 = new Uint32Array(_f16buf.buffer);
export function f32ToF16(v) {
  // IEEE f32 -> f16 bits, round to nearest with ties rounded up (away from
  // zero in magnitude), not ties-to-even like the GPU's pack2x16float. The scratch views are hoisted:
  // this is called once per element of every wire frame and every quantized block.
  _f16buf[0] = v;
  const x = _f16u32[0];
  const sign = (x >>> 16) & 0x8000;
  let e = (x >>> 23) & 0xff, m = x & 0x7fffff;
  if (e === 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);
  e = e - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    m = (m | 0x800000) >> (1 - e);
    return sign | ((m + 0x1000) >> 13);
  }
  // '+' not '|': a mantissa that rounds up to 0x400 must CARRY into the
  // exponent. With '|' the carry was dropped whenever the exponent's low bit
  // was already set, silently halving ~0.03% of all values (e.g. -1.99999 -> -1).
  return sign | ((e << 10) + ((m + 0x1000) >> 13));
}

export function f16ToF32(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

// Parse the GGUF header from an ArrayBuffer that contains AT LEAST the full
// header (metadata + tensor infos). Tensor data may be absent (range-fetch).
// opts.skipTokenizer: walk past tokenizer.* arrays without building 500k JS
// strings (100+ MB of heap a worker shard never uses; phones cannot afford it)
export function parseGGUFHeader(buf, opts = {}) {
  const dv = new DataView(buf);
  let off = 0;
  const u32 = () => { const v = dv.getUint32(off, true); off += 4; return v; };
  const u64 = () => { const v = Number(dv.getBigUint64(off, true)); off += 8; return v; };
  const i64 = () => { const v = Number(dv.getBigInt64(off, true)); off += 8; return v; };
  const f32v = () => { const v = dv.getFloat32(off, true); off += 4; return v; };
  const f64v = () => { const v = dv.getFloat64(off, true); off += 8; return v; };
  const str = (skip) => {
    const n = u64();
    const s = skip ? undefined : new TextDecoder().decode(new Uint8Array(buf, off, n));
    off += n;
    return s;
  };
  const value = (t, skip = false) => {
    switch (t) {
      case T_U8: return dv.getUint8(off++);
      case T_I8: return dv.getInt8(off++);
      case T_U16: { const v = dv.getUint16(off, true); off += 2; return v; }
      case T_I16: { const v = dv.getInt16(off, true); off += 2; return v; }
      case T_U32: return u32();
      case T_I32: { const v = dv.getInt32(off, true); off += 4; return v; }
      case T_F32: return f32v();
      case T_BOOL: return !!dv.getUint8(off++);
      case T_STR: return str(skip);
      case T_U64: return u64();
      case T_I64: return i64();
      case T_F64: return f64v();
      case T_ARR: {
        const et = u32(), n = u64(), out = skip ? undefined : new Array(n);
        for (let i = 0; i < n; i++) { const v = value(et, skip); if (!skip) out[i] = v; }
        return out;
      }
      default: throw new Error("bad gguf value type " + t);
    }
  };

  if (u32() !== 0x46554747) throw new Error("not a GGUF file");
  const version = u32();
  if (version < 2) throw new Error("gguf v1 unsupported");
  const nTensors = u64();
  const nKV = u64();
  const meta = {};
  for (let i = 0; i < nKV; i++) {
    const key = str();
    const skip = !!opts.skipTokenizer && key.startsWith("tokenizer.");
    const v = value(u32(), skip);
    if (!skip) meta[key] = v;
  }
  const infos = [];
  for (let i = 0; i < nTensors; i++) {
    const name = str();
    const nd = u32();
    const dims = [];
    for (let d = 0; d < nd; d++) dims.push(u64());
    const ggmlType = u32();
    const offset = u64();
    // ggml dims are [ne0(=innermost/cols), ne1(rows), ...]; torch order is reversed
    infos.push({ name, shape: dims.slice().reverse(), ggmlType, offset });
  }
  const align = meta["general.alignment"] || 32;
  const dataStart = Math.ceil(off / align) * align;
  const tensors = {};
  for (const t of infos) {
    const n = t.shape.reduce((a, b) => a * b, 1);
    const bytes = ggmlTypeBytes(t.ggmlType, n); // -1 for unsupported types
    tensors[t.name] = { ...t, nElems: n, byteOffset: dataStart + t.offset, byteLength: bytes };
  }
  // the Qwen3.5+ MoE files use their own architecture name (keys like "qwen35moe.block_count");
  // alias them under "qwen35." so the one engine reads both, dense and MoE
  const arch = meta["general.architecture"];
  if (typeof arch === "string" && arch !== "qwen35" && /^qwen3[5-9]/.test(arch))
    for (const k of Object.keys(meta)) if (k.startsWith(arch + ".")) { const a = "qwen35." + k.slice(arch.length + 1); if (!(a in meta)) meta[a] = meta[k]; }
  return { meta, tensors, dataStart, headerBytes: off };
}

// Dequant a tensor's raw bytes to Float32Array.
export function ggufToF32(info, bytes /* Uint8Array of the tensor's data */) {
  const n = info.nElems;
  if (info.ggmlType === GGML_F32)
    return new Float32Array(bytes.buffer, bytes.byteOffset, n);
  if (info.ggmlType === GGML_F16) {
    const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, n);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = f16ToF32(u16[i]);
    return out;
  }
  if (info.ggmlType === GGML_Q8_0) {
    const out = new Float32Array(n);
    const nb = n / QK8_0;
    for (let b = 0; b < nb; b++) {
      const base = b * Q8_0_BLOCK_BYTES;
      const scale = f16ToF32(bytes[base] | (bytes[base + 1] << 8));
      for (let i = 0; i < QK8_0; i++) {
        const q = bytes[base + 2 + i];
        out[b * QK8_0 + i] = scale * (q > 127 ? q - 256 : q);
      }
    }
    return out;
  }
  throw new Error("unsupported ggml type " + info.ggmlType);
}

// Repack Q8_0 for the GPU: {qs: Int8Array (as Uint32-aligned buffer), scales: Float32Array}
export function q8Repack(info, bytes) {
  const n = info.nElems;
  const nb = n / QK8_0;
  const qs = new Uint8Array(n);
  const scales = new Uint32Array(Math.ceil(nb / 2));   // raw f16 scales, 2 per word
  const sc16 = new Uint16Array(scales.buffer);
  if ((bytes.byteOffset & 1) === 0) {
    // 34-byte blocks keep 2-byte alignment: move u16 words (3x faster than subarray+set per block)
    const src = new Uint16Array(bytes.buffer, bytes.byteOffset, nb * 17), dst = new Uint16Array(qs.buffer);
    for (let b = 0, s = 0, d = 0; b < nb; b++, s += 17, d += 16) {
      sc16[b] = src[s];
      for (let j = 0; j < 16; j++) dst[d + j] = src[s + 1 + j];
    }
    return { qs, scales };
  }
  for (let b = 0; b < nb; b++) {
    const base = b * Q8_0_BLOCK_BYTES;
    sc16[b] = bytes[base] | (bytes[base + 1] << 8);
    qs.set(bytes.subarray(base + 2, base + 2 + QK8_0), b * QK8_0);
  }
  return { qs, scales };
}

// map ggml tensor names -> the engine's internal names for llama/qwen-family models
export function ggmlLayerNames(i) {
  const p = `blk.${i}.`;
  return {
    inNorm: p + "attn_norm.weight",
    q: p + "attn_q.weight", k: p + "attn_k.weight", v: p + "attn_v.weight",
    o: p + "attn_output.weight",
    qNorm: p + "attn_q_norm.weight", kNorm: p + "attn_k_norm.weight", // qwen3 only
    postNorm: p + "ffn_norm.weight",
    gate: p + "ffn_gate.weight", up: p + "ffn_up.weight", down: p + "ffn_down.weight",
  };
}
export const GGML_EMBED = "token_embd.weight";
export const GGML_FINAL_NORM = "output_norm.weight";
export const GGML_OUTPUT = "output.weight"; // absent when embeddings are tied

// One tensor's engine entry (repacked for the GPU) from a parsed GGUF header.
// bytesOf: async (info) => Uint8Array of that tensor's data (local slice or
// HTTP range fetch, same contract as the safetensors shard path).
// cpuOnly: never through G.streamEntry (an offloaded expert tensor is parked by the caller, not uploaded)
export async function ggufEntry(G, bytesOf, name, optional, onBytes = () => {}, cpuOnly = false) {
  const info0 = G.tensors[name];
  if (!info0) {
    if (optional) return null;
    throw new Error("missing tensor " + name);
  }
  // stacked MoE experts [nExp][dOut][dIn] are an ordinary matrix of nExp * dOut rows
  const info = info0.shape.length === 3 ? { ...info0, shape: [info0.shape[0] * info0.shape[1], info0.shape[2]] } : info0;
  // converted-weights cache (tests/weight_cache.js in Deno/Node, room/convertedcache.js in the browser): a hit returns the exact
  // bytes the conversion below would produce, without reading or converting the tensor
  if (G.entryCache) {
    const hit = await G.entryCache.get(info);
    if (hit) { onBytes(info.byteLength); return hit; }
  }
  // the embedding stays on the CPU too (per-token row lookups), so it takes the normal path
  if (G.streamEntry && !cpuOnly && name !== GGML_EMBED && info.shape.length === 2 && streamable(info.ggmlType)) {
    const e = await G.streamEntry(info);
    if (e) { onBytes(info.byteLength); return e; }
  }
  const bytes = await bytesOf(info);
  onBytes(info.byteLength);
  // G.convert: the same conversion elsewhere (packages/room-node/convert.js: worker threads, off the event loop)
  const e = G.convert ? await G.convert(info, bytes) : convertEntry(info, bytes);
  if (G.entryCache) await G.entryCache.put(info, e);
  return e;
}

// The CPU conversion from a tensor's GGUF bytes to the entry the engine uploads: Q4_0/Q8_0
// repacked (nibbles/int8 + split f16 scales), other quants requantized to Q8, the rest f32.
// Pure and deterministic: the weight cache stores exactly what this returns.
export function convertEntry(info, bytes) {
  if (info.shape.length === 2) {
    if (info.ggmlType === GGML_Q8_0) {
      const { qs, scales } = q8Repack(info, bytes);
      return { kind: "q8", qs, scales, shape: info.shape };
    }
    if (info.ggmlType === GGML_Q4_0) {
      const { qs, scales } = q4Repack(info, bytes);
      return { kind: "q4", qs, scales, shape: info.shape };
    }
    if (info.ggmlType !== GGML_F32 && info.ggmlType !== GGML_F16 && info.ggmlType !== GGML_BF16) {
      const { qs, scales } = requantQ8Streaming(info, bytes);
      return { kind: "q8", qs, scales, shape: info.shape };
    }
  }
  return { kind: "f32", data: dequantF32(info, bytes) };
}

// K-quant -> Q8 without materializing the whole tensor in f32 (a 27B lm_head
// would be a 5 GB allocation, which browser tabs refuse). Chunks of rows.
export function requantQ8Streaming(info, bytes) {
  const [rows, cols] = info.shape;
  const rowBytes = ggmlTypeBytes(info.ggmlType, cols);
  const n = rows * cols;
  const qs = new Uint8Array(n);
  const scales = new Uint32Array(Math.ceil(n / 32 / 2));
  const sc16 = new Uint16Array(scales.buffer);
  const CH = Math.max(1, Math.floor((8 * 2 ** 20) / (cols * 4))); // ~8MB f32 per chunk
  for (let r0 = 0; r0 < rows; r0 += CH) {
    const rc = Math.min(CH, rows - r0);
    const sub = { ggmlType: info.ggmlType, nElems: rc * cols, shape: [rc, cols] };
    const view = bytes.subarray(r0 * rowBytes, (r0 + rc) * rowBytes);
    const f = dequantF32(sub, view);
    const q = quantizeQ8(f);
    qs.set(q.qs, r0 * cols);
    const nbChunk = (rc * cols) / 32;
    sc16.set(new Uint16Array(q.scales.buffer, 0, nbChunk), (r0 * cols) / 32);
  }
  return { qs, scales };
}

export async function ggufWeights(G, bytesOf, { lo, hi, hasEmbed, hasHead }, onProgress = () => {}, onEntry = null) {
  let fetched = 0;
  const entry = async (name, optional) => {
    const e = await ggufEntry(G, bytesOf, name, optional, (b) => { fetched += b; onProgress(fetched); });
    if (e && onEntry) onEntry(e, name);
    return e;
  };
  const layers = [];
  for (let i = lo; i < hi; i++) {
    const N = ggmlLayerNames(i);
    layers.push({
      inNorm: await entry(N.inNorm), postNorm: await entry(N.postNorm),
      q: await entry(N.q), k: await entry(N.k), v: await entry(N.v), o: await entry(N.o),
      qNorm: await entry(N.qNorm, true), kNorm: await entry(N.kNorm, true),
      gate: await entry(N.gate), up: await entry(N.up), down: await entry(N.down),
    });
  }
  const out = { layers };
  if (hasEmbed || hasHead) out.embed = await entry(GGML_EMBED);
  if (hasHead) {
    out.finalNorm = await entry(GGML_FINAL_NORM);
    out.head = await entry(GGML_OUTPUT, true); // null when tied
  }
  return out;
}

// Total bytes a shard will download (for progress bars / pledge checks)
export function ggufShardBytes(G, { lo, hi, hasEmbed, hasHead }) {
  let total = 0;
  const add = (n) => { if (G.tensors[n]) total += G.tensors[n].byteLength; };
  for (let i = lo; i < hi; i++) Object.values(ggmlLayerNames(i)).forEach(add);
  if (hasEmbed || hasHead) add(GGML_EMBED);
  if (hasHead) { add(GGML_FINAL_NORM); add(GGML_OUTPUT); }
  return total;
}

// ---------- extended quant formats (Qwen3.8 / Q4_0 file family) ----------
export const GGML_Q4_0 = 2, GGML_Q4_1 = 3, GGML_Q5_0 = 6, GGML_Q5_K = 13, GGML_Q6_K = 14, GGML_BF16 = 30;
export const QK_K = 256;

export function ggmlTypeBytes(type, n) {
  switch (type) {
    case GGML_F32: return n * 4;
    case GGML_F16: return n * 2;
    case GGML_Q4_0: return (n / 32) * 18;
    case GGML_Q4_1: return (n / 32) * 20;
    case GGML_Q5_0: return (n / 32) * 22;
    case GGML_BF16: return n * 2;
    case GGML_Q8_0: return (n / 32) * 34;
    case GGML_Q5_K: return (n / QK_K) * 176;
    case GGML_Q6_K: return (n / QK_K) * 210;
    default: return -1;
  }
}

function getScaleMinK4(j, scales) {
  // ported from ggml get_scale_min_k4
  if (j < 4) return [scales[j] & 63, scales[j + 4] & 63];
  return [
    (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
    (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
  ];
}

// Dequantize any supported ggml type to Float32Array (CPU).
export function dequantF32(info, bytes) {
  const n = info.nElems, T = info.ggmlType;
  if (T === GGML_F32 || T === GGML_F16 || T === GGML_Q8_0) return ggufToF32(info, bytes);
  const out = new Float32Array(n);
  if (T === GGML_BF16) {   // bf16 is the top half of an f32: exact
    const u = new Uint32Array(out.buffer);
    for (let i = 0; i < n; i++) u[i] = (bytes[2 * i] | (bytes[2 * i + 1] << 8)) << 16;
    return out;
  }
  if (T === GGML_Q5_0) {   // ggml dequantize_row_q5_0: d f16, qh u32 (5th bits), 16 bytes of nibbles
    const nb = n / 32;
    for (let b = 0; b < nb; b++) {
      const base = b * 22, d = f16ToF32(bytes[base] | (bytes[base + 1] << 8));
      const qh = (bytes[base + 2] | (bytes[base + 3] << 8) | (bytes[base + 4] << 16) | (bytes[base + 5] << 24)) >>> 0;
      for (let j = 0; j < 16; j++) {
        const q = bytes[base + 6 + j], h0 = ((qh >>> j) << 4) & 0x10, h1 = (qh >>> (j + 12)) & 0x10;
        out[b * 32 + j] = d * (((q & 0xF) | h0) - 16);
        out[b * 32 + j + 16] = d * (((q >> 4) | h1) - 16);
      }
    }
    return out;
  }
  if (T === GGML_Q4_0) {
    const nb = n / 32;
    for (let b = 0; b < nb; b++) {
      const base = b * 18;
      const d = f16ToF32(bytes[base] | (bytes[base + 1] << 8));
      for (let l = 0; l < 16; l++) {
        const q = bytes[base + 2 + l];
        out[b * 32 + l] = d * ((q & 0xF) - 8);
        out[b * 32 + l + 16] = d * ((q >> 4) - 8);
      }
    }
    return out;
  }
  if (T === GGML_Q4_1) {
    const nb = n / 32;
    for (let b = 0; b < nb; b++) {
      const base = b * 20;
      const d = f16ToF32(bytes[base] | (bytes[base + 1] << 8));
      const m = f16ToF32(bytes[base + 2] | (bytes[base + 3] << 8));
      for (let l = 0; l < 16; l++) {
        const q = bytes[base + 4 + l];
        out[b * 32 + l] = d * (q & 0xF) + m;
        out[b * 32 + l + 16] = d * (q >> 4) + m;
      }
    }
    return out;
  }
  if (T === GGML_Q5_K) {
    const nb = n / QK_K;
    for (let i = 0; i < nb; i++) {
      const base = i * 176;
      const d = f16ToF32(bytes[base] | (bytes[base + 1] << 8));
      const min = f16ToF32(bytes[base + 2] | (bytes[base + 3] << 8));
      const scales = bytes.subarray(base + 4, base + 16);
      const qh = bytes.subarray(base + 16, base + 48);
      let ql = base + 48;
      let y = i * QK_K, is = 0, u1 = 1, u2 = 2;
      for (let j = 0; j < QK_K; j += 64) {
        const [sc1, m1q] = getScaleMinK4(is + 0, scales);
        const [sc2, m2q] = getScaleMinK4(is + 1, scales);
        const d1 = d * sc1, m1 = min * m1q;
        const d2 = d * sc2, m2 = min * m2q;
        for (let l = 0; l < 32; l++) out[y++] = d1 * ((bytes[ql + l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
        for (let l = 0; l < 32; l++) out[y++] = d2 * ((bytes[ql + l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
        ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
      }
    }
    return out;
  }
  if (T === GGML_Q6_K) {
    const nb = n / QK_K;
    for (let i = 0; i < nb; i++) {
      const base = i * 210;
      const d = f16ToF32(bytes[base + 208] | (bytes[base + 209] << 8));
      let ql = base, qh = base + 128, sc = base + 192;
      let y = i * QK_K;
      for (let nblk = 0; nblk < QK_K; nblk += 128) {
        for (let l = 0; l < 32; l++) {
          const is = (l / 16) | 0;
          const s = (v) => v > 127 ? v - 256 : v; // int8 scales
          const q1 = ((bytes[ql + l] & 0xF) | (((bytes[qh + l] >> 0) & 3) << 4)) - 32;
          const q2 = ((bytes[ql + l + 32] & 0xF) | (((bytes[qh + l] >> 2) & 3) << 4)) - 32;
          const q3 = ((bytes[ql + l] >> 4) | (((bytes[qh + l] >> 4) & 3) << 4)) - 32;
          const q4 = ((bytes[ql + l + 32] >> 4) | (((bytes[qh + l] >> 6) & 3) << 4)) - 32;
          out[y + l] = d * s(bytes[sc + is]) * q1;
          out[y + l + 32] = d * s(bytes[sc + is + 2]) * q2;
          out[y + l + 64] = d * s(bytes[sc + is + 4]) * q3;
          out[y + l + 96] = d * s(bytes[sc + is + 6]) * q4;
        }
        y += 128; ql += 64; qh += 32; sc += 8;
      }
    }
    return out;
  }
  throw new Error("dequantF32: unsupported type " + T);
}

// Tokenizer straight from GGUF metadata (tokens + merges arrays); returns a
// tokenizer.json-shaped object to pass to makeTokenizer.
export function tokenizerFromGGUF(meta) {
  const tokens = meta["tokenizer.ggml.tokens"];
  const merges = meta["tokenizer.ggml.merges"];
  const tj = {
    model: {
      vocab: Object.fromEntries(tokens.map((t, i) => [t, i])),
      merges,
    },
    // which pre-tokenizer split the model uses (qwen2, qwen35, ...); see preSplitter in tokenizer.js
    pre: meta["tokenizer.ggml.pre"],
  };
  return tj; // caller passes through makeTokenizer-compatible builder
}

// Repack Q4_0 for the GPU: nibbles stay packed (16 bytes/block), scales split out.
export function q4Repack(info, bytes) {
  const n = info.nElems;
  const nb = n / 32;
  const qs = new Uint8Array(n / 2);
  const scales = new Uint32Array(Math.ceil(nb / 2));   // raw f16 scales, 2 per word
  const sc16 = new Uint16Array(scales.buffer);
  if ((bytes.byteOffset & 1) === 0) {
    // 18-byte blocks keep 2-byte alignment: move u16 words (3x faster than subarray+set per block;
    // this loop is most of a cold 27B load)
    const src = new Uint16Array(bytes.buffer, bytes.byteOffset, nb * 9), dst = new Uint16Array(qs.buffer);
    for (let b = 0, s = 0, d = 0; b < nb; b++, s += 9, d += 8) {
      sc16[b] = src[s];
      dst[d] = src[s + 1]; dst[d + 1] = src[s + 2]; dst[d + 2] = src[s + 3]; dst[d + 3] = src[s + 4];
      dst[d + 4] = src[s + 5]; dst[d + 5] = src[s + 6]; dst[d + 6] = src[s + 7]; dst[d + 7] = src[s + 8];
    }
    return { qs, scales };
  }
  for (let b = 0; b < nb; b++) {
    const base = b * 18;
    sc16[b] = bytes[base] | (bytes[base + 1] << 8);
    qs.set(bytes.subarray(base + 2, base + 18), b * 16);
  }
  return { qs, scales };
}

// Quantize f32 -> Q8_0-style GPU entry (for K-quant tensors we don't have GPU
// kernels for: dequant on CPU, requant to q8 — tiny extra error, big memory win)
export function quantizeQ8(data) {
  const n = data.length;
  const nb = Math.ceil(n / 32);
  const qs = new Uint8Array(nb * 32);
  const scales = new Uint32Array(Math.ceil(nb / 2));   // packed f16, like the GGUF format itself
  const sc16 = new Uint16Array(scales.buffer);
  for (let b = 0; b < nb; b++) {
    let amax = 0;
    for (let i = b * 32; i < Math.min(n, b * 32 + 32); i++) amax = Math.max(amax, Math.abs(data[i]));
    const h = f32ToF16(amax / 127 || 1);
    sc16[b] = h;
    const d = f16ToF32(h) || 1;
    for (let i = b * 32; i < Math.min(n, b * 32 + 32); i++)
      qs[i] = Math.max(-127, Math.min(127, Math.round(data[i] / d))) & 0xFF;
  }
  return { qs, scales };
}


// ---------- qwen3.5/3.8 (hybrid delta-net) shard loader ----------
export function qwen35LayerNames(i, forceFull = false, moe = false, interval = 4) {
  const p = `blk.${i}.`;
  const isFull = forceFull || i % interval === interval - 1;
  const shared = {
    attnNorm: p + "attn_norm.weight",
    postNorm: p + "post_attention_norm.weight",
    ...(moe ? {
      // routed experts (stacked [nExp][dOut][dIn]) + router, and the always-on shared expert
      router: p + "ffn_gate_inp.weight",
      expGate: p + "ffn_gate_exps.weight", expUp: p + "ffn_up_exps.weight", expDown: p + "ffn_down_exps.weight",
      shGate: p + "ffn_gate_shexp.weight", shUp: p + "ffn_up_shexp.weight", shDown: p + "ffn_down_shexp.weight",
      shRouter: p + "ffn_gate_inp_shexp.weight",
    } : { ffnGate: p + "ffn_gate.weight", ffnUp: p + "ffn_up.weight", ffnDown: p + "ffn_down.weight" }),
  };
  if (isFull) return { ...shared, isFull,
    wq: p + "attn_q.weight", wk: p + "attn_k.weight", wv: p + "attn_v.weight",
    wo: p + "attn_output.weight",
    qNorm: p + "attn_q_norm.weight", kNorm: p + "attn_k_norm.weight" };
  return { ...shared, isFull,
    wqkv: p + "attn_qkv.weight", wz: p + "attn_gate.weight",
    wBeta: p + "ssm_beta.weight", wAlpha: p + "ssm_alpha.weight",
    dtBias: p + "ssm_dt.bias", ssmA: p + "ssm_a",
    conv: p + "ssm_conv1d.weight", ssmNorm: p + "ssm_norm.weight",
    wOut: p + "ssm_out.weight" };
}

// experts: an ExpertStore (engine/expert_store.js): the routed experts of the layers it holds (experts.has(i)) are
// parked in its buffers instead of returned for upload; the layer then carries its placeholder entries.
export async function qwen35Weights(G, bytesOf, { lo, hi, hasEmbed, hasHead, mtp = false, experts = null }, onProgress = () => {}, onEntry = null) {
  let fetched = 0;
  const entry = async (name, optional) => {
    const e = await ggufEntry(G, bytesOf, name, optional, (b) => { fetched += b; onProgress(fetched); });
    if (e && onEntry) onEntry(e, name);
    return e;
  };
  const moe = (G.meta["qwen35.expert_count"] || 0) > 0;
  const interval = G.meta["qwen35.full_attention_interval"] || 4;
  const loadLayer = async (i, forceFull = false) => {
    const N = qwen35LayerNames(i, forceFull, moe, interval);
    const L = { isFull: N.isFull,
      attnNorm: await entry(N.attnNorm), postNorm: await entry(N.postNorm) };
    if (moe) {
      L.moe = true;
      L.router = await entry(N.router);
      if (experts && !forceFull && experts.has(i)) {   // offloaded: converted on the CPU, parked, never uploaded
        const nExp = G.meta["qwen35.expert_count"];
        for (const [k, part] of [["expGate", "gate"], ["expUp", "up"], ["expDown", "down"]])
          L[k] = experts.park(i, part, await ggufEntry(G, bytesOf, N[k], false, (b) => { fetched += b; onProgress(fetched); }, true), nExp);
      } else { L.expGate = await entry(N.expGate); L.expUp = await entry(N.expUp); L.expDown = await entry(N.expDown); }
      L.shGate = await entry(N.shGate, true); L.shUp = await entry(N.shUp, true); L.shDown = await entry(N.shDown, true);
      L.shRouter = await entry(N.shRouter, true);
    } else {
      L.ffnGate = await entry(N.ffnGate); L.ffnUp = await entry(N.ffnUp); L.ffnDown = await entry(N.ffnDown);
    }
    if (N.isFull) {
      L.wq = await entry(N.wq); L.wk = await entry(N.wk); L.wv = await entry(N.wv);
      L.wo = await entry(N.wo);
      L.qNorm = await entry(N.qNorm); L.kNorm = await entry(N.kNorm);
    } else {
      L.wqkv = await entry(N.wqkv); L.wz = await entry(N.wz);
      L.wBeta = await entry(N.wBeta); L.wAlpha = await entry(N.wAlpha);
      L.dtBias = await entry(N.dtBias); L.ssmA = await entry(N.ssmA);
      L.conv = await entry(N.conv); L.ssmNorm = await entry(N.ssmNorm);
      L.wOut = await entry(N.wOut);
    }
    return L;
  };
  const layers = [];
  for (let i = lo; i < hi; i++) layers.push(await loadLayer(i));
  const out = { layers };
  if (hasEmbed) out.embed = await entry(GGML_EMBED);
  if (hasHead) {
    if (!out.embed) out.embed = await entry(GGML_EMBED);
    out.finalNorm = await entry(GGML_FINAL_NORM);
    out.head = await entry(GGML_OUTPUT, true);
  }
  // multi-token-prediction block (the "nextn" layer after the trunk): a normal
  // full-attention layer plus eh_proj / enorm / hnorm / shared_head_norm.
  if (mtp && hasHead) {
    const N = G.meta["qwen35.block_count"] - 1;
    const p = `blk.${N}.nextn.`;
    if (G.tensors[p + "eh_proj.weight"]) {
      out.mtp = {
        layer: await loadLayer(N, true),
        ehProj: await entry(p + "eh_proj.weight"),
        enorm: await entry(p + "enorm.weight"),
        hnorm: await entry(p + "hnorm.weight"),
        sharedHeadNorm: await entry(p + "shared_head_norm.weight"),
      };
    }
  }
  return out;
}

// qwen35LayerNames for this file (dense or MoE, its full-attention interval)
export function qwen35NamesFor(G, i, forceFull = false) {
  return qwen35LayerNames(i, forceFull, (G.meta["qwen35.expert_count"] || 0) > 0, G.meta["qwen35.full_attention_interval"] || 4);
}
export function qwen35ShardBytes(G, { lo, hi, hasEmbed, hasHead, mtp = false }) {
  let total = 0;
  const add = (n) => { if (G.tensors[n]) total += G.tensors[n].byteLength; };
  for (let i = lo; i < hi; i++) Object.values(qwen35NamesFor(G, i)).forEach((v) => { if (typeof v === "string") add(v); });
  if (hasEmbed || hasHead) add(GGML_EMBED);
  if (hasHead) { add(GGML_FINAL_NORM); add(GGML_OUTPUT); }
  if (mtp && hasHead) total += qwen35MtpBytes(G);
  return total;
}
// bytes of the multi-token-prediction block (host only)
export function qwen35MtpBytes(G) {
  const N = G.meta["qwen35.block_count"] - 1;
  const p = `blk.${N}.nextn.`;
  if (!G.tensors[p + "eh_proj.weight"]) return 0;
  let total = 0;
  const add = (n) => { if (G.tensors[n]) total += G.tensors[n].byteLength; };
  Object.values(qwen35NamesFor(G, N, true)).forEach((v) => { if (typeof v === "string") add(v); });
  ["eh_proj", "enorm", "hnorm", "shared_head_norm"].forEach((n) => add(p + n + ".weight"));
  return total;
}


// Upload one weight entry to the GPU right away and drop its CPU copy. Used on
// memory-starved devices (phones): peak RAM becomes one tensor instead of the
// whole shard. Engines pick up `e.gpu` instead of re-uploading.
export function gpuUploadEntry(device, e, keepCpu = false) {
  if (e.gpu) return e.gpu;
  // only the big quantized matrices; f32 tensors (norms, conv, biases) are tiny and the
  // engines read their CPU copy directly when building layers
  if (e.kind !== "q8" && e.kind !== "q4") return null;
  const mk = (data, usage) => {
    // writeBuffer in slices: no full-size mapped staging copy (matters on iOS,
    // where the tab's memory budget is a fraction of the device's)
    const src = ArrayBuffer.isView(data) ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
    const size = Math.ceil(src.byteLength / 4) * 4;
    const buf = device.createBuffer({ size, usage: usage | GPUBufferUsage.COPY_DST });
    const STEP = 16 * 2 ** 20;
    for (let o = 0; o < src.byteLength; o += STEP) {
      const n = Math.min(STEP, src.byteLength - o);
      const n4 = Math.ceil(n / 4) * 4;
      if (n4 === n) device.queue.writeBuffer(buf, o, src, o, n);
      else { const tail = new Uint8Array(n4); tail.set(src.subarray(o, o + n)); device.queue.writeBuffer(buf, o, tail); }
    }
    return buf;
  };
  if (e.kind === "q8" || e.kind === "q4")
    // COPY_SRC: engines may row-concatenate projections on the GPU at load (Qwen35Engine fuseProj), and
    // the MoE engine repacks the shared expert's gate/up on the GPU (engine/qwen35.js packGU)
    e.gpu = { kind: e.kind, qs: mk(e.qs, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC), sc: mk(e.scales, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC) };
  else e.gpu = { kind: "f32", buf: mk(e.data, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC) };
  if (!keepCpu) e.qs = e.scales = e.data = null;
  return e.gpu;
}


// Types streamEntryToGPU takes: Q4_0 / Q8_0 are repacked as they arrive, the other quants
// (Q4_1, Q5_0, Q5_K, Q6_K) are requantized to Q8 a few rows at a time (streamRequantToGPU).
// F32 / F16 / BF16 stay on the CPU path: they are the small tensors (norms, biases, conv).
export const streamable = (t) => t === GGML_Q8_0 || t === GGML_Q4_0 || t === GGML_Q4_1 || t === GGML_Q5_0 || t === GGML_Q5_K || t === GGML_Q6_K;

// Staging memory for the streamed uploads, reused from tensor to tensor (a shard streams hundreds of
// tensors one after another: without this each one left a few MB of garbage behind). A stream
// borrows one and gives it back when done; concurrent streams simply get their own.
const stagePool = [];
function borrowStage(bytes) {
  const i = stagePool.findIndex((b) => b.byteLength >= bytes);
  return i >= 0 ? stagePool.splice(i, 1)[0] : new ArrayBuffer(bytes);
}
function giveStage(buf) { if (stagePool.length < 2) stagePool.push(buf); }

// Stream a Q4_0 / Q8_0 tensor straight from the network into GPU buffers.
// Nothing tensor-sized ever exists in JS: chunks arrive, whole blocks are
// repacked into a small reused staging area and written out, the rest waits
// for the next chunk. Peak CPU memory ~ one network chunk + staging (a few
// MB) instead of 3x the tensor. This is what keeps an iPhone tab alive.
export async function streamEntryToGPU(device, info, openRange, { pace = 0, staging = 4 * 2 ** 20 } = {}) {
  if (info.ggmlType !== GGML_Q4_0 && info.ggmlType !== GGML_Q8_0) return streamRequantToGPU(device, info, openRange, { pace, staging });
  const q4 = info.ggmlType === GGML_Q4_0;
  const BLK = q4 ? 18 : Q8_0_BLOCK_BYTES;      // bytes per block in the file
  const QSB = q4 ? 16 : QK8_0;                 // quant bytes per block on the GPU
  const nb = info.nElems / 32;
  device.pushErrorScope("out-of-memory");
  const qsBuf = device.createBuffer({ size: Math.ceil(nb * QSB / 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const scBuf = device.createBuffer({ size: Math.ceil(nb / 2) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const oom = await device.popErrorScope();
  if (oom) throw new Error(`GPU out of memory while allocating ${info.name} (${(info.byteLength / 2 ** 20).toFixed(0)} MB): this device pledged more than its GPU can hold`);
  const blocksPerFlush = Math.max(1, Math.floor(staging / QSB));
  const stage = borrowStage(blocksPerFlush * (QSB + 2)), scAt = blocksPerFlush * QSB;
  try {
  const qsStage = new Uint8Array(stage, 0, scAt), qsStage16 = new Uint16Array(stage, 0, scAt / 2);
  const scStage = new Uint16Array(stage, scAt, blocksPerFlush);   // raw f16 scales
  let staged = 0, block = 0;
  const flush = async () => {
    if (!staged) return;
    const first = block - staged;
    // byte views only: WebKit and Chrome disagree on element-vs-byte sizes for typed arrays
    device.queue.writeBuffer(qsBuf, first * QSB, stage, 0, staged * QSB);
    device.queue.writeBuffer(scBuf, first * 2, stage, scAt, staged * 2);
    staged = 0;
    await new Promise((r) => setTimeout(r, 0));      // let WebKit hand the copy to the GPU process before we make more
  };
  const r = await openRange(info);
  if (!r.ok && r.status !== 206) throw new Error("range fetch failed for " + info.name);
  const reader = r.body.getReader();
  let carry = new Uint8Array(0);
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    let buf = value;
    if (carry.length) { const m = new Uint8Array(carry.length + value.length); m.set(carry); m.set(value, carry.length); buf = m; carry = new Uint8Array(0); }
    const whole = Math.floor(buf.length / BLK);
    // blocks are an even number of bytes: when the chunk starts 2-aligned, move u16 words
    const src16 = (buf.byteOffset & 1) === 0 ? new Uint16Array(buf.buffer, buf.byteOffset, whole * BLK / 2) : null;
    for (let b = 0; b < whole; b++) {
      const base = b * BLK;
      if (src16) {
        const s = base >> 1, d = staged * QSB / 2;
        scStage[staged] = src16[s];
        for (let j = 0; j < QSB / 2; j++) qsStage16[d + j] = src16[s + 1 + j];
      } else {
        scStage[staged] = buf[base] | (buf[base + 1] << 8);
        qsStage.set(buf.subarray(base + 2, base + BLK), staged * QSB);
      }
      staged++; block++;
      if (staged === blocksPerFlush) await flush();
    }
    const rest = buf.length - whole * BLK;
    if (rest) carry = buf.slice(whole * BLK);
  }
  await flush();
  if (block !== nb) throw new Error(`short tensor ${info.name}: ${block}/${nb} blocks`);
  } finally { giveStage(stage); }
  if (pace) await new Promise((r) => setTimeout(r, pace));   // give the allocator time to return pages
  return { kind: q4 ? "q4" : "q8", shape: info.shape, gpu: { kind: q4 ? "q4" : "q8", qs: qsBuf, sc: scBuf } };
}

// A K-quant (or Q4_1 / Q5_0) tensor, requantized to Q8 on its way to the GPU. The CPU path
// (convertEntry -> requantQ8Streaming) holds the whole tensor's bytes plus the whole Q8 output
// (a MoE expert-down tensor: 160 MB in, 256 MiB out, both alive at once) and this is what killed
// iPhone tabs on layers 0-4 (#207). Here only one chunk of rows is ever in JS: the rows arrive,
// are dequantized and quantized exactly as requantQ8Streaming does (Q8 blocks are 32 elements
// within a row, so the row chunking never changes a bit), and are written to the GPU buffers.
export async function streamRequantToGPU(device, info, openRange, { pace = 0, staging = 4 * 2 ** 20 } = {}) {
  const [rows, cols] = info.shape;
  const rowBytes = ggmlTypeBytes(info.ggmlType, cols);
  if (!(rowBytes > 0) || cols % 32) throw new Error(`can't stream ${info.name}: type ${info.ggmlType}, ${cols} columns`);
  const n = rows * cols, nb = n / 32;
  device.pushErrorScope("out-of-memory");
  const qsBuf = device.createBuffer({ size: Math.ceil(n / 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const scBuf = device.createBuffer({ size: Math.ceil(nb / 2) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const oom = await device.popErrorScope();
  if (oom) throw new Error(`GPU out of memory while allocating ${info.name} (${(info.byteLength / 2 ** 20).toFixed(0)} MB): this device pledged more than its GPU can hold`);
  // rows per chunk: about `staging` bytes of f32 while converting, an even count so every chunk's
  // scales start on a 4-byte boundary of the scale buffer
  const CH = Math.max(2, Math.floor(staging / (cols * 4)) & ~1);
  const stage = borrowStage(CH * rowBytes);
  const rowBuf = new Uint8Array(stage, 0, CH * rowBytes);   // reused for every chunk
  let fill = 0, row = 0;
  try {
  const flush = async () => {
    const rc = fill / rowBytes;
    if (!rc) return;
    const sub = { ggmlType: info.ggmlType, nElems: rc * cols, shape: [rc, cols] };
    const q = quantizeQ8(dequantF32(sub, rowBuf.subarray(0, fill)));
    device.queue.writeBuffer(qsBuf, row * cols, q.qs.buffer, 0, rc * cols);
    const scBytes = (rc * cols / 32) * 2;           // odd block count (the last chunk): q.scales is padded to a word
    device.queue.writeBuffer(scBuf, (row * cols / 32) * 2, q.scales.buffer, 0, Math.ceil(scBytes / 4) * 4);
    row += rc; fill = 0;
    await new Promise((r) => setTimeout(r, 0));
  };
  const r = await openRange(info);
  if (!r.ok && r.status !== 206) throw new Error("range fetch failed for " + info.name);
  const reader = r.body.getReader();
  let got = 0;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    for (let o = 0; o < value.length;) {
      const take = Math.min(value.length - o, rowBuf.length - fill, info.byteLength - got);
      if (take <= 0) break;
      rowBuf.set(value.subarray(o, o + take), fill);
      fill += take; o += take; got += take;
      if (fill === rowBuf.length) await flush();
    }
  }
  if (got !== info.byteLength || fill % rowBytes) throw new Error(`short tensor ${info.name}: ${got}/${info.byteLength} bytes`);
  await flush();
  } finally { giveStage(stage); }
  if (pace) await new Promise((r) => setTimeout(r, pace));
  return { kind: "q8", shape: info.shape, gpu: { kind: "q8", qs: qsBuf, sc: scBuf } };
}
