// Multi-column WGSL for Qwen3.8-Flash-Next (qwen4exp): the batched prefill and verify passes of
// engine/qwen4exp_batch.js, and the gathered (QSA) attention both paths use. Roadmap 33 PR plan items 5 and 9.
//
// One pass = nCols consecutive tokens at positions pos .. pos + nCols - 1. Every kernel here is the decode kernel of
// engine/wgsl/qwen4exp.js with a column index added: per column it does the same arithmetic in the same order, so a
// prefill column gives the decode step's bits (tests/test_q38flash_batch.js checks the residual bit for bit). The
// only kernels that change shape are the ones with a sequential dependency across tokens:
//   - dn_conv / dn_delta walk the columns in order inside one dispatch (the conv history and the DeltaNet state
//     S stay in registers across the columns);
//   - the PLE block splits into ple_gate (per column and stream: the gate and the normed value n) and ple_conv
//     (the dilated causal conv: positions inside the pass read n from the pass's own buffer, older ones from the
//     ring), then ple_ring stores the pass's n values into the ring. The decode kernel writes its ring slot before
//     the conv reads; here the ring is written after, because a pass of more columns than ring slots would
//     overwrite slots its earlier columns still need.
// The weight GEMVs (gemv_mc) take C columns per workgroup: each weight block is read once for those columns, and
// each column keeps its own lane sums and reduction tree, as the decode GEMV does.
//
// Column strides are in the uniforms (xs / ys: floats, or vec4s where the decode kernel reads vec4s).
// Bindings stay within WebGPU's default of 8 storage buffers per stage.

const DQ = /* wgsl */ `
fn q4lo(w: u32) -> vec4<f32> { return vec4<f32>(f32(w & 15u), f32((w >> 8u) & 15u), f32((w >> 16u) & 15u), f32((w >> 24u) & 15u)) - vec4<f32>(8.0); }
fn q4hi(w: u32) -> vec4<f32> { return vec4<f32>(f32((w >> 4u) & 15u), f32((w >> 12u) & 15u), f32((w >> 20u) & 15u), f32((w >> 28u) & 15u)) - vec4<f32>(8.0); }
fn i8x4(u: u32) -> vec4<f32> { let v = bitcast<i32>(u); return vec4<f32>(f32((v << 24u) >> 24u), f32((v << 16u) >> 24u), f32((v << 8u) >> 24u), f32(v >> 24u)); }
fn sigm(v: f32) -> f32 { return 1.0 / (1.0 + exp(-v)); }
fn silu(v: f32) -> f32 { return v / (1.0 + exp(-v)); }
`;

// The decode kernels' 32-wide block dot (engine/wgsl/qwen4exp.js blockDot), written to `d`; the same expression.
const blockDot = (kind, QS, SC, X) => kind === "q4" ? `
  {
    let s = unpack2x16float(${SC}[blk >> 1u])[blk & 1u];
    let wo = blk * 4u;
    let w0 = ${QS}[wo]; let w1 = ${QS}[wo + 1u]; let w2 = ${QS}[wo + 2u]; let w3 = ${QS}[wo + 3u];
    d = s * (dot(q4lo(w0), ${X}[xb4]) + dot(q4lo(w1), ${X}[xb4 + 1u]) + dot(q4lo(w2), ${X}[xb4 + 2u]) + dot(q4lo(w3), ${X}[xb4 + 3u])
           + dot(q4hi(w0), ${X}[xb4 + 4u]) + dot(q4hi(w1), ${X}[xb4 + 5u]) + dot(q4hi(w2), ${X}[xb4 + 6u]) + dot(q4hi(w3), ${X}[xb4 + 7u]));
  }` : `
  {
    let s = unpack2x16float(${SC}[blk >> 1u])[blk & 1u];
    let wo = blk * 8u;
    var a = 0.0;
    for (var j = 0u; j < 8u; j++) { a += dot(i8x4(${QS}[wo + j]), ${X}[xb4 + j]); }
    d = s * a;
  }`;

// The same dot with the weight words already loaded and dequantized (shared by the C columns of gemv_mc):
// the q4 form keeps the decode expression's grouping, s * (sum of the 8 dots in order).
const blockDotPre = (kind, X) => kind === "q4" ? `d = s * (dot(l0, ${X}[xb4]) + dot(l1, ${X}[xb4 + 1u]) + dot(l2, ${X}[xb4 + 2u]) + dot(l3, ${X}[xb4 + 3u])
           + dot(h0, ${X}[xb4 + 4u]) + dot(h1, ${X}[xb4 + 5u]) + dot(h2, ${X}[xb4 + 6u]) + dot(h3, ${X}[xb4 + 7u]));`
  : `{ var a = 0.0; for (var j = 0u; j < 8u; j++) { a += dot(wv[j], ${X}[xb4 + j]); } d = s * a; }`;

export const GEMV_MC_COLS = 4;   // columns per gemv_mc workgroup

// GEMV over nCols columns: y[c * ys + yOff + r] = epilogue(W[rowOff + r, :] . x[c * xs4 * 4 + xOff4 * 4 ..]).
// 8 rows x 32 lanes per workgroup as gemvWGSL; workgroup y = column group of C. mode 1: silu(acc * scale).
export function gemvMcWGSL(kind, C = GEMV_MC_COLS) {
  const cols = [...Array(C).keys()];
  const W = kind === "f32" ? `@group(0) @binding(0) var<storage, read> fx_w: array<vec4<f32>>;` : `
@group(0) @binding(0) var<storage, read> fx_qs: array<u32>;
@group(0) @binding(1) var<storage, read> fx_sc: array<u32>;`;
  const xb = (c) => `fx_P.xOff4 + cc${c} * fx_P.xs4`;
  const body = kind === "f32" ? `
    let n4 = fx_P.dIn / 4u;
    let ro = (fx_P.rowOff + r) * n4;
    for (var i = lane; i < n4; i += 32u) {
      let wv = fx_w[ro + i];
      ${cols.map((c) => `acc${c} += dot(wv, fx_x[${xb(c)} + i]);`).join("\n      ")}
    }` : `
    let nb = fx_P.dIn / 32u;
    let rb = (fx_P.rowOff + r) * nb;
    for (var b = lane; b < nb; b += 32u) {
      let blk = rb + b;
      let s = unpack2x16float(fx_sc[blk >> 1u])[blk & 1u];
      ${kind === "q4" ? `let wo = blk * 4u;
      let w0 = fx_qs[wo]; let w1 = fx_qs[wo + 1u]; let w2 = fx_qs[wo + 2u]; let w3 = fx_qs[wo + 3u];
      let l0 = q4lo(w0); let l1 = q4lo(w1); let l2 = q4lo(w2); let l3 = q4lo(w3);
      let h0 = q4hi(w0); let h1 = q4hi(w1); let h2 = q4hi(w2); let h3 = q4hi(w3);` : `let wo = blk * 8u;
      var wv: array<vec4<f32>, 8>;
      for (var j = 0u; j < 8u; j++) { wv[j] = i8x4(fx_qs[wo + j]); }`}
      var d = 0.0;
      ${cols.map((c) => `{ let xb4 = ${xb(c)} + b * 8u; ${blockDotPre(kind, "fx_x")} acc${c} += d; }`).join("\n      ")}
    }`;
  return DQ + `
struct GP { dOut: u32, dIn: u32, rowOff: u32, xOff4: u32, yOff: u32, mode: u32, scale: f32, xs4: u32, ys: u32, p0: u32, p1: u32, p2: u32 };
struct McF { pos: u32, nCols: u32, p0: u32, p1: u32 };
${W}
@group(0) @binding(2) var<storage, read> fx_x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> fx_y: array<f32>;
@group(0) @binding(4) var<uniform> fx_P: GP;
@group(0) @binding(5) var<uniform> fx_F: McF;
var<workgroup> fx_red: array<f32, ${256 * C}>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let lid = l.x; let lane = lid & 31u;
  let r = wg.x * 8u + (lid >> 5u);
  let c0 = wg.y * ${C}u;
  ${cols.map((c) => `let cc${c} = min(c0 + ${c}u, fx_F.nCols - 1u); var acc${c} = 0.0;`).join("\n  ")}
  if (r < fx_P.dOut) {${body}
  }
  ${cols.map((c) => `fx_red[${c * 256}u + lid] = acc${c};`).join(" ")}
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) { ${cols.map((c) => `fx_red[${c * 256}u + lid] += fx_red[${c * 256}u + lid + s];`).join(" ")} }
    workgroupBarrier();
  }
  if (lane == 0u && r < fx_P.dOut) {
    ${cols.map((c) => `if (c0 + ${c}u < fx_F.nCols) {
      var v = fx_red[${c * 256}u + lid];
      if (fx_P.mode == 1u) { v = silu(v * fx_P.scale); }
      fx_y[(c0 + ${c}u) * fx_P.ys + fx_P.yOff + r] = v;
    }`).join("\n    ")}
  }
}`;
}

// Routed (or shared) experts, gate and up, per column: workgroup (row group, k, column).
// h[(col * K + k) * dOut + r] = silu(Wg[e] x_col)[r] * (Wu[e] x_col)[r], e = ids[col * K + k]. x column stride xs4.
export function moeGuMcWGSL(kind) {
  return DQ + `
struct MP { dOut: u32, dIn: u32, K: u32, xs4: u32 };
@group(0) @binding(0) var<storage, read> fx_gq: array<u32>;
@group(0) @binding(1) var<storage, read> fx_gs: array<u32>;
@group(0) @binding(2) var<storage, read> fx_uq: array<u32>;
@group(0) @binding(3) var<storage, read> fx_us: array<u32>;
@group(0) @binding(4) var<storage, read> fx_x: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> fx_h: array<f32>;
@group(0) @binding(6) var<storage, read> fx_ids: array<u32>;
@group(0) @binding(7) var<uniform> fx_P: MP;
var<workgroup> fx_red: array<f32, 256>;
var<workgroup> fx_red2: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let lid = l.x; let lane = lid & 31u;
  let r = wg.x * 8u + (lid >> 5u);
  let k = wg.y; let col = wg.z;
  var ag = 0.0; var au = 0.0;
  if (r < fx_P.dOut) {
    let e = fx_ids[col * fx_P.K + k];
    let nb = fx_P.dIn / 32u;
    let rb = (e * fx_P.dOut + r) * nb;
    for (var b = lane; b < nb; b += 32u) {
      let blk = rb + b;
      let xb4 = col * fx_P.xs4 + b * 8u;
      var d = 0.0;
      ${blockDot(kind, "fx_gq", "fx_gs", "fx_x")}
      ag += d;
      ${blockDot(kind, "fx_uq", "fx_us", "fx_x")}
      au += d;
    }
  }
  fx_red[lid] = ag; fx_red2[lid] = au;
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) { fx_red[lid] += fx_red[lid + s]; fx_red2[lid] += fx_red2[lid + s]; }
    workgroupBarrier();
  }
  if (lane == 0u && r < fx_P.dOut) { fx_h[(col * fx_P.K + k) * fx_P.dOut + r] = silu(fx_red[lid]) * fx_red2[lid]; }
}`;
}

// Expert down projections summed with their weights, per column (workgroup z):
// y[col * dOut + r] = sum_k w * (Wd[ids[col*K+k]] h[col][k])[r]; w = wts[col*K+k] (sigmoid of it when sig = 1).
export function moeDnMcWGSL(kind) {
  return DQ + `
struct MP { dOut: u32, dIn: u32, K: u32, sig: u32 };
@group(0) @binding(0) var<storage, read> fx_qs: array<u32>;
@group(0) @binding(1) var<storage, read> fx_sc: array<u32>;
@group(0) @binding(2) var<storage, read> fx_x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> fx_y: array<f32>;
@group(0) @binding(4) var<storage, read> fx_ids: array<u32>;
@group(0) @binding(5) var<storage, read> fx_wts: array<f32>;
@group(0) @binding(6) var<uniform> fx_P: MP;
var<workgroup> fx_red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let lid = l.x; let lane = lid & 31u;
  let r = wg.x * 8u + (lid >> 5u);
  let col = wg.z;
  var acc = 0.0;
  if (r < fx_P.dOut) {
    let nb = fx_P.dIn / 32u;
    let cb4 = col * ((fx_P.K * fx_P.dIn) / 4u);
    for (var p = lane; p < fx_P.K * nb; p += 32u) {
      let k = p / nb; let b = p % nb;
      var wk = fx_wts[col * fx_P.K + k];
      if (fx_P.sig == 1u) { wk = sigm(wk); }
      let blk = (fx_ids[col * fx_P.K + k] * fx_P.dOut + r) * nb + b;
      let xb4 = cb4 + (k * fx_P.dIn) / 4u + b * 8u;
      var d = 0.0;
      ${blockDot(kind, "fx_qs", "fx_sc", "fx_x")}
      acc += wk * d;
    }
  }
  fx_red[lid] = acc;
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) { fx_red[lid] += fx_red[lid + s]; }
    workgroupBarrier();
  }
  if (lane == 0u && r < fx_P.dOut) { fx_y[col * fx_P.dOut + r] = fx_red[lid]; }
}`;
}

// Router per column (one workgroup each): MOE_ROUTE_WGSL with the column's logits / ids / weights.
export const MOE_ROUTE_MC_WGSL = DQ + `
struct RP { nExp: u32, K: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> fx_logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> fx_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> fx_wts: array<f32>;
@group(0) @binding(3) var<uniform> fx_P: RP;
var<workgroup> fx_rv: array<f32, 256>;
var<workgroup> fx_ri: array<u32, 256>;
var<workgroup> fx_taken: array<u32, 1024>;
var<workgroup> fx_pk: array<f32, 32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let t = l.x; let lo = wg.x * fx_P.nExp; let ko = wg.x * fx_P.K;
  var m = -3.0e38;
  for (var e = t; e < fx_P.nExp; e += 256u) { m = max(m, fx_logits[lo + e]); fx_taken[e] = 0u; }
  fx_rv[t] = m;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { fx_rv[t] = max(fx_rv[t], fx_rv[t + s]); } workgroupBarrier(); }
  let mx = fx_rv[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var e = t; e < fx_P.nExp; e += 256u) { sum += exp(fx_logits[lo + e] - mx); }
  fx_rv[t] = sum;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { fx_rv[t] += fx_rv[t + s]; } workgroupBarrier(); }
  let tot = fx_rv[0];
  workgroupBarrier();
  for (var k = 0u; k < fx_P.K; k++) {
    var bv = -3.0e38; var bi = 0xffffffffu;
    for (var e = t; e < fx_P.nExp; e += 256u) {
      if (fx_taken[e] == 0u && fx_logits[lo + e] > bv) { bv = fx_logits[lo + e]; bi = e; }
    }
    fx_rv[t] = bv; fx_ri[t] = bi;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
      if (t < s) {
        let ov = fx_rv[t + s]; let oi = fx_ri[t + s];
        if (ov > fx_rv[t] || (ov == fx_rv[t] && oi < fx_ri[t])) { fx_rv[t] = ov; fx_ri[t] = oi; }
      }
      workgroupBarrier();
    }
    if (t == 0u) { let e = fx_ri[0]; fx_ids[ko + k] = e; fx_taken[e] = 1u; fx_pk[k] = exp(fx_logits[lo + e] - mx) / tot; }
    workgroupBarrier();
  }
  if (t == 0u) {
    var ws = 0.0;
    for (var k = 0u; k < fx_P.K; k++) { ws += fx_pk[k]; }
    ws = max(ws, 6.103515625e-5);
    for (var k = 0u; k < fx_P.K; k++) { fx_wts[ko + k] = fx_pk[k] / ws; }
  }
}`;

// HC mixer part 1 per column: workgroup (stream, column). R / xn column stride hc * d.
export const HC_NORM_MC_WGSL = `
struct NP { d: u32, eps: f32, hc: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> fx_R: array<f32>;
@group(0) @binding(1) var<storage, read> fx_w: array<f32>;
@group(0) @binding(2) var<storage, read_write> fx_xn: array<f32>;
@group(0) @binding(3) var<uniform> fx_P: NP;
var<workgroup> fx_red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let t = l.x; let o = wg.x * fx_P.d; let co = wg.y * fx_P.d * fx_P.hc + o;
  var ss = 0.0;
  for (var i = t; i < fx_P.d; i += 256u) { let v = fx_R[co + i]; ss += v * v; }
  fx_red[t] = ss;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { fx_red[t] += fx_red[t + s]; } workgroupBarrier(); }
  let inv = 1.0 / sqrt(fx_red[0] / f32(fx_P.d) + fx_P.eps);
  for (var i = t; i < fx_P.d; i += 256u) { fx_xn[co + i] = (fx_R[co + i] * inv) * fx_w[o + i]; }
}`;

// HC mixer part 2 per column (global y = column): mixed[col][i] = mean_c xn[col][c][i] * sigmoid(gate[col][c][i]).
export const HC_MIX_MC_WGSL = DQ + `
struct XP { d: u32, hc: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> fx_xn: array<f32>;
@group(0) @binding(1) var<storage, read> fx_gate: array<f32>;
@group(0) @binding(2) var<storage, read_write> fx_mixed: array<f32>;
@group(0) @binding(3) var<uniform> fx_P: XP;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; let col = g.y;
  if (i >= fx_P.d) { return; }
  let cb = col * fx_P.d * fx_P.hc;
  var s = 0.0;
  for (var c = 0u; c < fx_P.hc; c++) { let j = cb + c * fx_P.d + i; s += fx_xn[j] * sigm(fx_gate[j]); }
  fx_mixed[col * fx_P.d + i] = s / f32(fx_P.hc);
}`;

// hc_combine per column (global y = column): R[col][c][i] += (a[col][i] + b[col][i]) * 2 sigmoid(inj[col][c] / hc).
export const HC_COMBINE_MC_WGSL = DQ + `
struct CP { d: u32, hc: u32, useB: u32, p1: u32 };
@group(0) @binding(0) var<storage, read_write> fx_R: array<f32>;
@group(0) @binding(1) var<storage, read> fx_a: array<f32>;
@group(0) @binding(2) var<storage, read> fx_b: array<f32>;
@group(0) @binding(3) var<storage, read> fx_inj: array<f32>;
@group(0) @binding(4) var<uniform> fx_P: CP;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let j = g.x; let col = g.y;
  if (j >= fx_P.d * fx_P.hc) { return; }
  let c = j / fx_P.d; let i = j % fx_P.d;
  var v = fx_a[col * fx_P.d + i];
  if (fx_P.useB == 1u) { v += fx_b[col * fx_P.d + i]; }
  let o = col * fx_P.d * fx_P.hc + j;
  fx_R[o] += v * (2.0 * sigm(fx_inj[col * fx_P.hc + c] / f32(fx_P.hc)));
}`;

// DeltaNet conv over the pass's columns in order (one thread per channel; the history stays in registers).
export const DN_CONV_MC_WGSL = DQ + `
struct DP { n: u32, K: u32, p0: u32, p1: u32 };
struct McF { pos: u32, nCols: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> fx_xin: array<f32>;
@group(0) @binding(1) var<storage, read_write> fx_st: array<f32>;
@group(0) @binding(2) var<storage, read> fx_w: array<f32>;
@group(0) @binding(3) var<storage, read_write> fx_co: array<f32>;
@group(0) @binding(4) var<uniform> fx_P: DP;
@group(0) @binding(5) var<uniform> fx_F: McF;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let c = g.x;
  if (c >= fx_P.n) { return; }
  let K1 = fx_P.K - 1u;
  var h: array<f32, 8>;   // convK - 1 <= 8
  for (var j = 0u; j < K1; j++) { h[j] = fx_st[c * K1 + j]; }
  for (var col = 0u; col < fx_F.nCols; col++) {
    let xv = fx_xin[col * fx_P.n + c];
    var acc = fx_w[c * fx_P.K + K1] * xv;
    for (var j = 0u; j < K1; j++) { acc += fx_w[c * fx_P.K + j] * h[j]; }
    fx_co[col * fx_P.n + c] = silu(acc);
    for (var j = 0u; j + 1u < K1; j++) { h[j] = h[j + 1u]; }
    h[K1 - 1u] = xv;
  }
  for (var j = 0u; j < K1; j++) { fx_st[c * K1 + j] = h[j]; }
}`;

// DeltaNet l2 norm per column: workgroup (q/k head, column), co column stride n.
export const DN_L2_MC_WGSL = `
struct LP { dS: u32, eps: f32, n: u32, p1: u32 };
@group(0) @binding(0) var<storage, read_write> fx_co: array<f32>;
@group(0) @binding(1) var<uniform> fx_P: LP;
var<workgroup> fx_red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let t = l.x; let o = wg.y * fx_P.n + wg.x * fx_P.dS;
  let v = fx_co[o + t];
  fx_red[t] = v * v;
  workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) { if (t < s) { fx_red[t] += fx_red[t + s]; } workgroupBarrier(); }
  fx_co[o + t] = v * (1.0 / max(sqrt(fx_red[0]), fx_P.eps));
}`;

// The delta rule over the pass's columns in order: one workgroup per V head, thread j owns column j of S, held in
// registers across the columns (DN_DELTA_WGSL's arithmetic per column). co / alpha / beta / o column strides in P.
export const DN_DELTA_MC_WGSL = DQ + `
struct EP { nKH: u32, dS: u32, keyDim: u32, n: u32, nV: u32, inner: u32, p0: u32, p1: u32 };
struct McF { pos: u32, nCols: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> fx_co: array<f32>;
@group(0) @binding(1) var<storage, read_write> fx_S: array<f32>;
@group(0) @binding(2) var<storage, read> fx_alpha: array<f32>;
@group(0) @binding(3) var<storage, read> fx_beta: array<f32>;
@group(0) @binding(4) var<storage, read> fx_dtb: array<f32>;
@group(0) @binding(5) var<storage, read> fx_A: array<f32>;
@group(0) @binding(6) var<storage, read_write> fx_o: array<f32>;
@group(0) @binding(7) var<uniform> fx_P: EP;
@group(0) @binding(8) var<uniform> fx_F: McF;
var<workgroup> fx_kq: array<f32, 256>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let h = wg.x; let j = l.x; let dS = fx_P.dS;
  let kh = h % fx_P.nKH;
  let base = h * dS * dS;
  var S: array<f32, 128>;
  for (var i = 0u; i < dS; i++) { S[i] = fx_S[base + i * dS + j]; }
  let scale = 1.0 / sqrt(f32(dS));
  for (var col = 0u; col < fx_F.nCols; col++) {
    let cb = col * fx_P.n;
    workgroupBarrier();
    fx_kq[j] = fx_co[cb + fx_P.keyDim + kh * dS + j];
    fx_kq[128u + j] = fx_co[cb + kh * dS + j];
    workgroupBarrier();
    let araw = fx_alpha[col * fx_P.nV + h] + fx_dtb[h];
    var sp = araw;
    if (araw <= 20.0) { sp = log(1.0 + exp(araw)); }
    let decay = exp(sp * fx_A[h]);
    let b = sigm(fx_beta[col * fx_P.nV + h]);
    let v = fx_co[cb + 2u * fx_P.keyDim + h * dS + j];
    var vh = 0.0;
    for (var i = 0u; i < dS; i++) {
      let s = S[i] * decay;
      S[i] = s;
      vh += s * fx_kq[i];
    }
    vh = (v - vh) * b;
    var acc = 0.0;
    for (var i = 0u; i < dS; i++) {
      let s = S[i] + fx_kq[i] * vh;
      S[i] = s;
      acc += s * fx_kq[128u + i];
    }
    fx_o[col * fx_P.inner + h * dS + j] = acc * scale;
  }
  for (var i = 0u; i < dS; i++) { fx_S[base + i * dS + j] = S[i]; }
}`;

// DeltaNet gated output norm per column: workgroup (V head, column); o / z / y column stride inner.
export const DN_GNORM_MC_WGSL = DQ + `
struct GN { dS: u32, eps: f32, inner: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> fx_o: array<f32>;
@group(0) @binding(1) var<storage, read> fx_z: array<f32>;
@group(0) @binding(2) var<storage, read> fx_nw: array<f32>;
@group(0) @binding(3) var<storage, read_write> fx_y: array<f32>;
@group(0) @binding(4) var<uniform> fx_P: GN;
var<workgroup> fx_red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let t = l.x; let i = wg.y * fx_P.inner + wg.x * fx_P.dS + t;
  let v = fx_o[i];
  fx_red[t] = v * v;
  workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) { if (t < s) { fx_red[t] += fx_red[t + s]; } workgroupBarrier(); }
  let inv = 1.0 / sqrt(fx_red[0] / f32(fx_P.dS) + fx_P.eps);
  fx_y[i] = (v * inv) * fx_nw[t] * sigm(fx_z[i]);
}`;

// The pass frame: first position and column count, then per column the rope (cos, sin, 0, 0) of each rotary pair
// (the decode frame's table, one per column; 64 pairs at most).
export const MC_FRAME_HEAD = 4, MC_FRAME_COL = 64 * 4;   // floats

// Attention prep per column: workgroup (head, column); ATTN_PREP_WGSL with the column's rope and position. The two
// head norms share one buffer (8 storage buffers per stage).
export const ATTN_PREP_MC_WGSL = `
struct AP { nH: u32, nKV: u32, hd: u32, nRot: u32, eps: f32, p0: u32, p1: u32, p2: u32 };
struct McF { pos: u32, nCols: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> fx_qf: array<f32>;
@group(0) @binding(1) var<storage, read> fx_kin: array<f32>;
@group(0) @binding(2) var<storage, read> fx_vin: array<f32>;
@group(0) @binding(3) var<storage, read_write> fx_qn: array<f32>;
@group(0) @binding(4) var<storage, read_write> fx_kc: array<f32>;
@group(0) @binding(5) var<storage, read_write> fx_vc: array<f32>;
@group(0) @binding(6) var<storage, read> fx_qkw: array<f32>;          // q norm (hd) then k norm (hd)
@group(0) @binding(7) var<storage, read> fx_cs: array<vec4<f32>>;   // [col][64] (cos, sin, 0, 0)
@group(0) @binding(8) var<uniform> fx_P: AP;
@group(0) @binding(9) var<uniform> fx_F: McF;
var<workgroup> fx_red: array<f32, 256>;
var<workgroup> fx_hv: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let t = l.x; let g = wg.x; let col = wg.y; let hd = fx_P.hd;
  let qfs = 2u * fx_P.nH * hd; let kvs = fx_P.nKV * hd;
  let isQ = g < fx_P.nH;
  var x = 0.0;
  if (t < hd) {
    if (isQ) { x = fx_qf[col * qfs + g * 2u * hd + t]; } else { x = fx_kin[col * kvs + (g - fx_P.nH) * hd + t]; }
  }
  fx_red[t] = x * x;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { fx_red[t] += fx_red[t + s]; } workgroupBarrier(); }
  let inv = 1.0 / sqrt(fx_red[0] / f32(hd) + fx_P.eps);
  var nv = 0.0;
  if (t < hd) { if (isQ) { nv = (x * inv) * fx_qkw[t]; } else { nv = (x * inv) * fx_qkw[hd + t]; } }
  fx_hv[t] = nv;
  workgroupBarrier();
  let hr = fx_P.nRot / 2u;
  var r = nv;
  if (t < hr) { let c = fx_cs[col * 64u + t]; r = nv * c.x - fx_hv[t + hr] * c.y; }
  else if (t < fx_P.nRot) { let c = fx_cs[col * 64u + t - hr]; r = fx_hv[t - hr] * c.y + nv * c.x; }
  if (t < hd) {
    if (isQ) { fx_qn[col * fx_P.nH * hd + g * hd + t] = r; }
    else {
      let h = g - fx_P.nH;
      let o = (fx_F.pos + col) * kvs + h * hd + t;
      fx_kc[o] = r;
      fx_vc[o] = fx_vin[col * kvs + h * hd + t];
    }
  }
}`;

// Attention per column: workgroup (q head, column). gather = 0: the dense causal ATTN_DEC_WGSL over cells
// 0 .. pos + col; gather = stride: over the column's row list from qsa_select (sel[col * stride] = n, then n rows
// ascending). With the list 0 .. p the gathered kernel does the dense arithmetic in the same order: the same bits.
// At most 2051 cells per column either way (dense runs only while every column has <= 2051), so the scores fit in
// workgroup memory. Output: the gated attention [col][nH * hd]. The gathered variant does not read the frame
// (binding 6 is left out of its layout).
export const ATTN_MAX_CELLS = 2052;
export function attnDecMcWGSL({ gather = 0 } = {}) {
  const row = (p) => gather ? `fx_sel[ib + ${p}]` : `${p}`;
  return DQ + `
struct AP { nH: u32, nKV: u32, hd: u32, nRot: u32, eps: f32, p0: u32, p1: u32, p2: u32 };
struct McF { pos: u32, nCols: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> fx_qn: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> fx_kc: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> fx_vc: array<f32>;
@group(0) @binding(3) var<storage, read> fx_qf: array<f32>;
@group(0) @binding(4) var<storage, read_write> fx_out: array<f32>;
@group(0) @binding(5) var<uniform> fx_P: AP;
@group(0) @binding(6) var<uniform> fx_F: McF;${gather ? `
@group(0) @binding(7) var<storage, read> fx_sel: array<u32>;` : ""}
var<workgroup> fx_sc: array<f32, ${ATTN_MAX_CELLS}>;
var<workgroup> fx_red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let t = l.x; let h = wg.x; let col = wg.y; let hd = fx_P.hd; let hd4 = hd / 4u;
  let kvh = h / (fx_P.nH / fx_P.nKV);
  ${gather ? `let ib = col * ${gather}u + 1u; let n = min(fx_sel[col * ${gather}u], ${ATTN_MAX_CELLS}u);` : `let n = min(fx_F.pos + col + 1u, ${ATTN_MAX_CELLS}u);`}
  let qb = col * fx_P.nH * hd4 + h * hd4;
  let scale = 1.0 / sqrt(f32(hd));
  var m = -3.0e38;
  for (var p = t; p < n; p += 256u) {
    let ko = (${row("p")} * fx_P.nKV + kvh) * hd4;
    var a = 0.0;
    for (var i = 0u; i < hd4; i++) { a += dot(fx_qn[qb + i], fx_kc[ko + i]); }
    a *= scale;
    fx_sc[p] = a;
    m = max(m, a);
  }
  fx_red[t] = m;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { fx_red[t] = max(fx_red[t], fx_red[t + s]); } workgroupBarrier(); }
  let mx = fx_red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var p = t; p < n; p += 256u) { let e = exp(fx_sc[p] - mx); fx_sc[p] = e; sum += e; }
  fx_red[t] = sum;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { fx_red[t] += fx_red[t + s]; } workgroupBarrier(); }
  let tot = fx_red[0];
  if (t < hd) {
    var acc = 0.0;
    for (var p = 0u; p < n; p++) { acc += fx_sc[p] * fx_vc[(${row("p")} * fx_P.nKV + kvh) * hd + t]; }
    fx_out[col * fx_P.nH * hd + h * hd + t] = (acc / tot) * sigm(fx_qf[col * 2u * fx_P.nH * hd + h * 2u * hd + hd + t]);
  }
}`;
}

// PLE (n-gram injection) part 1, per column and stream (workgroup (stream, column)): PLE_WGSL's gate and normed
// value; writes g = pv * gate to gb[col][c][i] and n = rmsnorm(g) * nc to nb[col][c][i]. pk / R column stride
// hc * d, pv column stride d. g goes through memory, as in the decode kernel (which also stores it for its debug
// taps): computed next to the R update, the compiler may fuse pv * gate into the add and round differently.
export const PLE_GATE_MC_WGSL = DQ + `
struct PP { d: u32, K: u32, dil: u32, slots: u32, eps: f32, hc: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> fx_R: array<f32>;
@group(0) @binding(1) var<storage, read> fx_pk: array<f32>;
@group(0) @binding(2) var<storage, read> fx_pv: array<f32>;
@group(0) @binding(3) var<storage, read> fx_pw: array<f32>;
@group(0) @binding(4) var<storage, read_write> fx_nb: array<f32>;
@group(0) @binding(5) var<storage, read_write> fx_gb: array<f32>;
@group(0) @binding(6) var<uniform> fx_P: PP;
var<workgroup> fx_r1: array<f32, 256>;
var<workgroup> fx_r2: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let t = l.x; let c = wg.x; let col = wg.y; let d = fx_P.d; let o = c * d;
  let HD = d * fx_P.hc; let cb = col * HD; let vb = col * d;
  let NQ = HD; let NC = 2u * HD;
  var sk = 0.0; var sq = 0.0;
  for (var i = t; i < d; i += 256u) { let a = fx_pk[cb + o + i]; let b = fx_R[cb + o + i]; sk += a * a; sq += b * b; }
  fx_r1[t] = sk; fx_r2[t] = sq;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { fx_r1[t] += fx_r1[t + s]; fx_r2[t] += fx_r2[t + s]; } workgroupBarrier(); }
  let ik = 1.0 / sqrt(fx_r1[0] / f32(d) + fx_P.eps);
  let iq = 1.0 / sqrt(fx_r2[0] / f32(d) + fx_P.eps);
  workgroupBarrier();
  var dt = 0.0;
  for (var i = t; i < d; i += 256u) { dt += ((fx_pk[cb + o + i] * ik) * fx_pw[o + i]) * ((fx_R[cb + o + i] * iq) * fx_pw[NQ + o + i]); }
  fx_r1[t] = dt;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { fx_r1[t] += fx_r1[t + s]; } workgroupBarrier(); }
  let sv = fx_r1[0] / sqrt(f32(d));
  let gate = sigm(sign(sv) * sqrt(max(abs(sv), 1e-6)));
  workgroupBarrier();
  var sg = 0.0;
  for (var i = t; i < d; i += 256u) { let g = fx_pv[vb + i] * gate; sg += g * g; }
  fx_r1[t] = sg;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { fx_r1[t] += fx_r1[t + s]; } workgroupBarrier(); }
  let ig = 1.0 / sqrt(fx_r1[0] / f32(d) + fx_P.eps);
  for (var i = t; i < d; i += 256u) {
    let g = fx_pv[vb + i] * gate;
    fx_nb[cb + o + i] = (g * ig) * fx_pw[NC + o + i];
    fx_gb[cb + o + i] = g;
  }
}`;

// PLE part 2, one thread per (channel, column): the dilated causal conv (taps inside the pass from nb, older ones
// from the ring), silu, R += g + conv. Part 3 (ple_ring, same dispatch shape) then stores the pass's n into the
// ring, for the columns whose slot a later position can still read.
const PLE_CONV_HEAD = DQ + `
struct PP { d: u32, K: u32, dil: u32, slots: u32, eps: f32, hc: u32, p1: u32, p2: u32 };
struct McF { pos: u32, nCols: u32, p0: u32, p1: u32 };`;
export const PLE_CONV_MC_WGSL = PLE_CONV_HEAD + `
@group(0) @binding(0) var<storage, read_write> fx_R: array<f32>;
@group(0) @binding(1) var<storage, read> fx_gb: array<f32>;
@group(0) @binding(2) var<storage, read> fx_pw: array<f32>;
@group(0) @binding(3) var<storage, read> fx_ring: array<f32>;
@group(0) @binding(4) var<storage, read> fx_nb: array<f32>;
@group(0) @binding(5) var<uniform> fx_P: PP;
@group(0) @binding(6) var<uniform> fx_F: McF;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g3: vec3<u32>) {
  let d = fx_P.d; let HD = d * fx_P.hc; let CW = 3u * HD;
  let ch = g3.x; let col = g3.y;
  if (ch >= HD) { return; }
  let pos = fx_F.pos + col;
  let g = fx_gb[col * HD + ch];
  let nv = fx_nb[col * HD + ch];
  var acc = fx_pw[CW + ch * fx_P.K + fx_P.K - 1u] * nv;
  for (var k = 0u; k + 1u < fx_P.K; k++) {
    let back = (fx_P.K - 1u - k) * fx_P.dil;
    if (pos >= back) {
      let q = pos - back;
      var x = 0.0;
      if (q >= fx_F.pos) { x = fx_nb[(q - fx_F.pos) * HD + ch]; } else { x = fx_ring[(q % fx_P.slots) * HD + ch]; }
      acc += fx_pw[CW + ch * fx_P.K + k] * x;
    }
  }
  let cv = silu(acc);
  let o = col * HD + ch;
  fx_R[o] = fx_R[o] + (g + cv);
}`;
export const PLE_RING_MC_WGSL = PLE_CONV_HEAD + `
@group(0) @binding(0) var<storage, read_write> fx_ring: array<f32>;
@group(0) @binding(1) var<storage, read> fx_nb: array<f32>;
@group(0) @binding(2) var<uniform> fx_P: PP;
@group(0) @binding(3) var<uniform> fx_F: McF;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g3: vec3<u32>) {
  let HD = fx_P.d * fx_P.hc;
  let ch = g3.x; let col = g3.y;
  if (ch >= HD || col + fx_P.slots < fx_F.nCols) { return; }
  fx_ring[((fx_F.pos + col) % fx_P.slots) * HD + ch] = fx_nb[col * HD + ch];
}`;

// Greedy head per column: ARGMAX1 / ARGMAX2 with workgroup y (stage 1) or x (stage 2) = column.
export const ARGMAX1_MC_WGSL = `
struct AM { n: u32, per: u32, nPart: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> fx_lg: array<f32>;
@group(0) @binding(1) var<storage, read_write> fx_part: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> fx_P: AM;
var<workgroup> fx_rv: array<f32, 256>;
var<workgroup> fx_ri: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let t = l.x; let lo = wg.x * fx_P.per; let hi = min(lo + fx_P.per, fx_P.n); let cb = wg.y * fx_P.n;
  var bv = -3.0e38; var bi = 0xffffffffu;
  for (var i = lo + t; i < hi; i += 256u) { let v = fx_lg[cb + i]; if (v > bv) { bv = v; bi = i; } }
  fx_rv[t] = bv; fx_ri[t] = bi;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (t < s) { let ov = fx_rv[t + s]; let oi = fx_ri[t + s]; if (ov > fx_rv[t] || (ov == fx_rv[t] && oi < fx_ri[t])) { fx_rv[t] = ov; fx_ri[t] = oi; } }
    workgroupBarrier();
  }
  if (t == 0u) { fx_part[wg.y * fx_P.nPart + wg.x] = vec2<f32>(fx_rv[0], bitcast<f32>(fx_ri[0])); }
}`;
export const ARGMAX2_MC_WGSL = `
struct AM { n: u32, per: u32, nPart: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> fx_part: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> fx_outv: array<u32>;
@group(0) @binding(2) var<uniform> fx_P: AM;
var<workgroup> fx_rv: array<f32, 256>;
var<workgroup> fx_ri: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let t = l.x; let pb = wg.x * fx_P.nPart;
  var bv = -3.0e38; var bi = 0xffffffffu;
  for (var i = t; i < fx_P.nPart; i += 256u) { let p = fx_part[pb + i]; let pi = bitcast<u32>(p.y); if (p.x > bv || (p.x == bv && pi < bi)) { bv = p.x; bi = pi; } }
  fx_rv[t] = bv; fx_ri[t] = bi;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (t < s) { let ov = fx_rv[t + s]; let oi = fx_ri[t + s]; if (ov > fx_rv[t] || (ov == fx_rv[t] && oi < fx_ri[t])) { fx_rv[t] = ov; fx_ri[t] = oi; } }
    workgroupBarrier();
  }
  if (t == 0u) { fx_outv[2u * wg.x] = fx_ri[0]; fx_outv[2u * wg.x + 1u] = bitcast<u32>(fx_rv[0]); }
}`;
