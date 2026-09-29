// Tiled causal flash attention for full-width prefill passes (docs/research/prefill-profile-2026-09.md,
// candidate E). attn_flash / attn_flash_t2 run one workgroup per (split, column[s], kv head) and every
// thread does serial 256-long dot products, so a 16-column pass re-reads each K/V row 8 to 16 times and
// runs at about 0.3 TFLOPS. Here one 256-thread workgroup takes a (split, kv head, column group): up to
// 64 query rows (CW columns x G heads) share every K/V tile, which is staged once in workgroup memory
// as f32.
//
// Layout: thread tid owns row r = tid / 4 and a quarter of the head dims (ds = tid % 4: dims
// i*16 + ds*4 + 0..3, i < HD/16), so q and the output accumulator live in registers (2 x HD/16 vec4)
// for the whole split. Per tile of TK positions:
//   1. K tile -> at_kv (f16 pairs unpacked to f32 vec4s, positions past the split end zeroed)
//   2. partial scores q . k over the thread's dims -> at_sp[t * 256 + tid]
//   3. V tile -> at_kv; each of the row's 4 threads sums the 4 partials in the same fixed order and runs
//      the same online-softmax update (max / sum / rescale), so the copies agree bit for bit
//   4. o = o * alpha + sum_t p[t] * v[t] over the thread's dims
// Workgroup memory: TK * HD * 4 (tile) + TK * 256 * 4 (partials) bytes; TK = 8 fits the 16 KB default at
// HD = 256, TK = 16 needs 32 KB (picked only when the device grants it).
//
// Split length: the output uses attn_flash's partial layout (faO / faML: maxSplits slots per (column,
// head)), so ceil(seqEnd / splitLen) <= maxSplits must hold. splitLen is derived from the pass itself
// (frame.seqLen, frame.nCols; tileSplitLen below is the JavaScript mirror), aiming at TARGET splits so
// short contexts still fill the GPU, never longer than attn_flash's faSplit. attn_combine_tile is
// attn_combine with that split length.
//
// Numerics: the scores, softmax and output are accumulated in a different order than attn_flash (4-way
// split dot products, TK-position softmax steps, different split boundaries), so a prefill column's
// attention output differs at float32 rounding level from the decode / verify path. The engine uses
// it only for full-width prefill passes (the same gate as the prefill GEMM), behind attnPrefillTile.
//
// js = true emits the kernel bodies as JavaScript generator bodies with the same control flow and index
// math (tests/unit/attn_tile_test.js runs them on the CPU, one generator per thread, barriers as yields).

import { wgslToJs } from "./moe.js";

// { HD, G, CW, TK, FASPLIT, FASPLITS, TARGET } or null when the shape is not supported
export function attnTileConfig({ hd, G, faSplit, faSplits, wgMem = 16384, target = 32, tk = 0, q8 = false }) {
  if (q8 && hd % 32 !== 0) return null;
  if (!(hd % 16 === 0 && hd >= 16 && hd <= 256 && G >= 1 && G <= 64)) return null;
  if (faSplit % 64 !== 0 || faSplits < 1) return null;
  const bytes = (TK) => TK * hd * 4 + TK * 256 * 4;
  const TK = tk || [16, 8, 4].find((t) => bytes(t) <= wgMem);
  if (!TK || bytes(TK) > wgMem) return null;
  return { HD: hd, G, CW: Math.floor(64 / G), TK, FASPLIT: faSplit, FASPLITS: faSplits, TARGET: Math.max(1, target | 0), Q8: !!q8 };
}

// JavaScript mirror of the kernel's split length for a pass whose last column sees seqEnd positions
export function tileSplitLen(seqEnd, c) {
  const up64 = (x) => Math.ceil(x / 64) * 64;
  return Math.min(c.FASPLIT, Math.max(64, up64(Math.ceil(seqEnd / c.TARGET)), up64(Math.ceil(seqEnd / c.FASPLITS))));
}

function emit(js) {
  return {
    div: (a, b) => js ? `Math.floor((${a}) / (${b}))` : `((${a}) / (${b}))`,
    vzero: js ? `[0, 0, 0, 0]` : `vec4<f32>(0.0)`,
    v4of2: (a, b) => js ? `[...unpack2x16float(${a}), ...unpack2x16float(${b})]` : `vec4<f32>(unpack2x16float(${a}), unpack2x16float(${b}))`,
    // four int8 of a q8_0 word times the block scale (the same f32 products attn_flash_q8 forms)
    q8v4: (w, sc) => js ? `vmad([0, 0, 0, 0], ${sc}, i8x4(${w}))` : `(vec4<f32>(f32(bitcast<i32>(${w} << 24u) >> 24u), f32(bitcast<i32>(${w} << 16u) >> 24u), f32(bitcast<i32>(${w} << 8u) >> 24u), f32(bitcast<i32>(${w}) >> 24u)) * ${sc})`,
    v4: (a, b, c, d) => js ? `[${a}, ${b}, ${c}, ${d}]` : `vec4<f32>(${a}, ${b}, ${c}, ${d})`,
    vmad: (acc, s, v) => js ? `${acc} = vmad(${acc}, ${s}, ${v});` : `${acc} += ${s} * ${v};`,
    vscale: (acc, s) => js ? `${acc} = vmad([0, 0, 0, 0], ${s}, ${acc});` : `${acc} *= ${s};`,
    vdecl: (n) => js ? `let ${n} = [0, 0, 0, 0];` : `var ${n} = vec4<f32>(0.0);`,
    varr: (n, len) => js ? `let ${n} = Array.from({ length: ${len} }, () => [0, 0, 0, 0]);` : `var ${n}: array<vec4<f32>, ${len}>;`,
    farr: (n, len) => js ? `let ${n} = new Array(${len}).fill(0);` : `var ${n}: array<f32, ${len}>;`,
  };
}

// the pass's split length (uniform: frame values and constants only)
function splitExpr(c, E) {
  const e = `(frame.seqLen + max(frame.nCols, 1u) - 1u)`;
  const up = (n) => `(${E.div(`${E.div(`${e} + ${n - 1}u`, `${n}u`)} + 63u`, "64u")} * 64u)`;
  return `min(${c.FASPLIT}u, max(64u, max(${up(c.TARGET)}, ${up(c.FASPLITS)})))`;
}

export function attnTileBodies(c, js = false) {
  const E = emit(js);
  const { HD, G, CW, TK } = c;
  const NQ = HD / 16, HD4 = HD / 4, HW = HD / 2, RR = CW * G;
  const I = Array.from({ length: NQ }, (_, i) => i), T = Array.from({ length: TK }, (_, t) => t);
  // K or V tile -> at_kv as f32 vec4s: f16 pairs, or (Q8) llama.cpp q8_0 words (4 int8) times the
  // block's f16 scale (${src}s: 2 per word, one per 32 values)
  const tileLoad = (src) => `for (var w: u32 = tid; w < ${TK * HD4}u; w += 256u) {
      let t = ${E.div("w", `${HD4}u`)}; let d4 = w % ${HD4}u;
      ${E.vdecl("kv")}
      if (c0 + t < t1) {${c.Q8 ? `
        let si = (c0 + t) * ${E.div("cfg.kvDim", "32u")} + g * ${HD / 32}u + ${E.div("d4", "8u")};
        let sc = unpack2x16float(${src}s[${E.div("si", "2u")}])[si % 2u];
        kv = ${E.q8v4(`${src}[(c0 + t) * ${E.div("cfg.kvDim", "4u")} + g * ${HD4}u + d4]`, "sc")};` : `
        let kb = (c0 + t) * kvw + g * ${HW}u + d4 * 2u;
        kv = ${E.v4of2(`${src}[kb]`, `${src}[kb + 1u]`)};`}
      }
      at_kv[w] = kv;
    }`;
  const flash = `
  let sp = wg.x; let g = wg.y; let cg = wg.z; let tid = lid.x;
  let nc = max(frame.nCols, 1u);
  let sl = ${splitExpr(c, E)};
  let t0 = sp * sl;
  let gEnd = frame.seqLen + min(nc, cg * ${CW}u + ${CW}u) - 1u;
  if (t0 >= gEnd || cg * ${CW}u >= nc) { return; }
  let t1 = min(t0 + sl, gEnd);
  let r = ${E.div("tid", "4u")}; let ds = tid % 4u;
  let col = cg * ${CW}u + ${E.div("r", `${G}u`)}; let h = r % ${G}u;
  let rowEnd = select(0u, min(t1, frame.seqLen + col), r < ${RR}u && col < nc);
  let hasRow = rowEnd > t0;
  let kvw = ${E.div("cfg.kvDim", "2u")};
  let rs = sqrt(f32(${HD}u));
  ${E.varr("q", NQ)}
  ${E.varr("o", NQ)}
  ${E.farr("p", TK)}
  if (hasRow) {
    let qb = col * atu.s0 + (g * ${G}u + h) * ${HD}u + ds * 4u;
    ${I.map((i) => `q[${i}u] = ${E.v4(...[0, 1, 2, 3].map((j) => `at_q[qb + ${i * 16 + j}u]`))};`).join("\n    ")}
  }
  var m: f32 = -3.0e38;
  var l: f32 = 0.0;
  for (var c0: u32 = t0; c0 < t1; c0 += ${TK}u) {
    ${tileLoad("at_k")}
    workgroupBarrier();
    let act = c0 < rowEnd;
    if (act) {
      ${T.map((t) => `{
        ${E.vdecl("a")}
        ${I.map((i) => E.vmad("a", `q[${i}u]`, `at_kv[${t * HD4 + i * 4}u + ds]`)).join(" ")}
        at_sp[${t * 256}u + tid] = (a[0u] + a[1u]) + (a[2u] + a[3u]);
      }`).join("\n      ")}
    }
    workgroupBarrier();
    ${tileLoad("at_v")}
    var alpha: f32 = 1.0;
    if (act) {
      let n = min(${TK}u, rowEnd - c0);
      let sb = r * 4u;
      var cm: f32 = m;
      ${T.map((t) => `p[${t}u] = (((at_sp[${t * 256}u + sb] + at_sp[${t * 256 + 1}u + sb]) + at_sp[${t * 256 + 2}u + sb]) + at_sp[${t * 256 + 3}u + sb]) / rs;
      if (${t}u < n) { cm = max(cm, p[${t}u]); }`).join("\n      ")}
      alpha = exp(m - cm);
      l = l * alpha;
      ${T.map((t) => `p[${t}u] = select(0.0, exp(p[${t}u] - cm), ${t}u < n); l += p[${t}u];`).join("\n      ")}
      m = cm;
    }
    workgroupBarrier();
    if (act) {
      ${I.map((i) => E.vscale(`o[${i}u]`, "alpha")).join(" ")}
      ${T.map((t) => `{
        let pt = p[${t}u];
        ${I.map((i) => E.vmad(`o[${i}u]`, "pt", `at_kv[${t * HD4 + i * 4}u + ds]`)).join(" ")}
      }`).join("\n      ")}
    }
    workgroupBarrier();
  }
  if (hasRow) {
    let hb = (col * cfg.nH + g * ${G}u + h) * atu.maxSplits + sp;
    let ob = hb * ${HD}u + ds * 4u;
    ${I.map((i) => [0, 1, 2, 3].map((j) => `at_o[ob + ${i * 16 + j}u] = o[${i}u][${j}u];`).join(" ")).join("\n    ")}
    if (ds == 0u) { at_ml[hb * 2u] = m; at_ml[hb * 2u + 1u] = l; }
  }`;
  const combine = `
  let qh = wg.x; let col = wg.y; let i = lid.x;
  if (qh >= cfg.nH || i >= ${HD}u) { return; }
  let sl = ${splitExpr(c, E)};
  let seqLen = frame.seqLen + col;
  let ns = ${E.div("seqLen + sl - 1u", "sl")};
  let b0 = (col * cfg.nH + qh) * atc.maxSplits;
  var M: f32 = -3.0e38;
  for (var s: u32 = 0u; s < ns; s++) { M = max(M, atc_ml[(b0 + s) * 2u]); }
  var L: f32 = 0.0;
  var O: f32 = 0.0;
  for (var s: u32 = 0u; s < ns; s++) {
    let w = exp(atc_ml[(b0 + s) * 2u] - M);
    L += atc_ml[(b0 + s) * 2u + 1u] * w;
    O += atc_o[(b0 + s) * ${HD}u + i] * w;
  }
  atc_out[col * atc.s1 + qh * ${HD}u + i] = O / L;`;
  return js ? { flash: wgslToJs(flash), combine: wgslToJs(combine) } : { flash, combine };
}

// A standalone module (compiled only when attnPrefillTile is on, so a problem here cannot take down
// the main module). Bindings match attn_flash_t2 / attn_combine (Q8: attn_flash_q8), so the engine binds
// the same buffers.
export function attnTileWGSL(c) {
  const { flash, combine } = attnTileBodies(c);
  return /* wgsl */ `
struct Config {
  dim: u32, kvDim: u32, nH: u32, nKV: u32,
  headDim: u32, inter: u32, vocab: u32, maxSeq: u32,
  eps: f32, theta: f32, qDim: u32,
};
struct Frame { pos: u32, seqLen: u32, nCols: u32, snap: u32 };
struct FA { s0: u32, s1: u32, splitLen: u32, maxSplits: u32 };   // q col stride, out col stride, (unused), slots
@group(0) @binding(0) var<uniform> cfg: Config;
@group(0) @binding(1) var<uniform> frame: Frame;

@group(1) @binding(0) var<storage, read> at_q: array<f32>;
${c.Q8 ? `@group(1) @binding(1) var<storage, read> at_k: array<u32>;      // q8_0: 4 int8 per word
@group(1) @binding(2) var<storage, read> at_v: array<u32>;
@group(1) @binding(3) var<storage, read> at_ks: array<u32>;     // f16 scales, 2 per word
@group(1) @binding(4) var<storage, read> at_vs: array<u32>;
@group(1) @binding(5) var<storage, read_write> at_o: array<f32>;
@group(1) @binding(6) var<storage, read_write> at_ml: array<f32>;
@group(1) @binding(7) var<uniform> atu: FA;` : `@group(1) @binding(1) var<storage, read> at_k: array<u32>;      // f16 pairs
@group(1) @binding(2) var<storage, read> at_v: array<u32>;
@group(1) @binding(3) var<storage, read_write> at_o: array<f32>;
@group(1) @binding(4) var<storage, read_write> at_ml: array<f32>;
@group(1) @binding(5) var<uniform> atu: FA;`}
var<workgroup> at_kv: array<vec4<f32>, ${c.TK * c.HD / 4}>;
var<workgroup> at_sp: array<f32, ${c.TK * 256}>;
@compute @workgroup_size(256)
fn attn_flash_tile(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {${flash}
}

@group(1) @binding(0) var<storage, read> atc_o: array<f32>;
@group(1) @binding(1) var<storage, read> atc_ml: array<f32>;
@group(1) @binding(2) var<storage, read_write> atc_out: array<f32>;
@group(1) @binding(3) var<uniform> atc: FA;
@compute @workgroup_size(256)
fn attn_combine_tile(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {${combine}
}
`;
}
