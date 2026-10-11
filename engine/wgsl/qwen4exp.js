// WGSL for the Qwen3.8-Flash-Next (llama.cpp arch "qwen4exp") decode path, engine/qwen4exp.js.
// Roadmap 33 section 2 "New code" and PR plan item 4. Every kernel is its own module with its bindings in group 0,
// so a pipeline's "auto" layout is exactly its bindings (none are left unused).
//
// Weight formats are the loader's (engine/gguf.js convertEntry): "q4" = Q4_0 repacked (16 nibble bytes per 32-block,
// byte j holds element j low and element j + 16 high; f16 scales two per word), "q8" = Q8_0 repacked (32 int8 per
// block, four per word; f16 scales two per word), "f32" = row-major floats. Activations are f32 throughout (the
// reference's --act f32 mode, tests/reference/ref_q38flash.mjs).

// Dequant helpers: 4 elements of a q4 word (low or high nibbles) and of a q8 word, as vec4<f32>.
const DQ = /* wgsl */ `
fn q4lo(w: u32) -> vec4<f32> { return vec4<f32>(f32(w & 15u), f32((w >> 8u) & 15u), f32((w >> 16u) & 15u), f32((w >> 24u) & 15u)) - vec4<f32>(8.0); }
fn q4hi(w: u32) -> vec4<f32> { return vec4<f32>(f32((w >> 4u) & 15u), f32((w >> 12u) & 15u), f32((w >> 20u) & 15u), f32((w >> 28u) & 15u)) - vec4<f32>(8.0); }
fn i8x4(u: u32) -> vec4<f32> { let v = bitcast<i32>(u); return vec4<f32>(f32((v << 24u) >> 24u), f32((v << 16u) >> 24u), f32((v << 8u) >> 24u), f32(v >> 24u)); }
fn sigm(v: f32) -> f32 { return 1.0 / (1.0 + exp(-v)); }
fn silu(v: f32) -> f32 { return v / (1.0 + exp(-v)); }
`;

// The 32-wide block dot for one row block `blk` (row * nb + b) against x[xb4 .. xb4 + 8) (vec4 units).
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

// Lane reduction: 8 rows x 32 lanes per 256-thread workgroup, red[lid] holds each lane's partial sum.
const LANE_REDUCE = (R = "fx_red") => `
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) { ${R}[lid] += ${R}[lid + s]; }
    workgroupBarrier();
  }`;

// GEMV y[yOff + r] = epilogue(W[rowOff + r, :] . x[xOff4*4 ..]), r < dOut. 8 rows per workgroup, 32 lanes per row.
// mode 0: store; 1: silu(acc * scale) (the HC down projection: silu(x / hc)).
// Workgroups: x = ceil(dOut / 8) up to 32768, then y.
export function gemvWGSL(kind) {
  const W = kind === "f32" ? `@group(0) @binding(0) var<storage, read> fx_w: array<vec4<f32>>;` : `
@group(0) @binding(0) var<storage, read> fx_qs: array<u32>;
@group(0) @binding(1) var<storage, read> fx_sc: array<u32>;`;
  const body = kind === "f32" ? `
    let n4 = fx_P.dIn / 4u;
    let ro = (fx_P.rowOff + r) * n4;
    for (var i = lane; i < n4; i += 32u) { acc += dot(fx_w[ro + i], fx_x[fx_P.xOff4 + i]); }` : `
    let nb = fx_P.dIn / 32u;
    let rb = (fx_P.rowOff + r) * nb;
    for (var b = lane; b < nb; b += 32u) {
      let blk = rb + b;
      let xb4 = fx_P.xOff4 + b * 8u;
      var d = 0.0;
      ${blockDot(kind, "fx_qs", "fx_sc", "fx_x")}
      acc += d;
    }`;
  return DQ + `
struct GP { dOut: u32, dIn: u32, rowOff: u32, xOff4: u32, yOff: u32, mode: u32, scale: f32, pad: u32 };
${W}
@group(0) @binding(2) var<storage, read> fx_x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> fx_y: array<f32>;
@group(0) @binding(4) var<uniform> fx_P: GP;
var<workgroup> fx_red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let lid = l.x; let lane = lid & 31u;
  let r = (wg.y * 32768u + wg.x) * 8u + (lid >> 5u);
  var acc = 0.0;
  if (r < fx_P.dOut) {${body}
  }
  fx_red[lid] = acc;
  ${LANE_REDUCE()}
  if (lane == 0u && r < fx_P.dOut) {
    var v = fx_red[lid];
    if (fx_P.mode == 1u) { v = silu(v * fx_P.scale); }
    fx_y[fx_P.yOff + r] = v;
  }
}`;
}

// Routed (or shared) experts, gate and up together: h[k * dOut + r] = silu(Wg[e] x)[r] * (Wu[e] x)[r], e = ids[k];
// workgroup y = k. The shared expert runs it with ids = [0] and K = 1.
export function moeGuWGSL(kind) {
  return DQ + `
struct MP { dOut: u32, dIn: u32, K: u32, sig: u32 };
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
  let k = wg.y;
  var ag = 0.0; var au = 0.0;
  if (r < fx_P.dOut) {
    let e = fx_ids[k];
    let nb = fx_P.dIn / 32u;
    let rb = (e * fx_P.dOut + r) * nb;
    for (var b = lane; b < nb; b += 32u) {
      let blk = rb + b;
      let xb4 = b * 8u;
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
  if (lane == 0u && r < fx_P.dOut) { fx_h[k * fx_P.dOut + r] = silu(fx_red[lid]) * fx_red2[lid]; }
}`;
}

// Expert down projections summed with their weights: y[r] = sum_k w_k * (Wd[ids[k]] h[k])[r]; w_k = wts[k], or
// sigmoid(wts[k]) when sig = 1 (the shared expert's gate, sigmoid(ffn_gate_inp_shexp . x)).
export function moeDnWGSL(kind) {
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
  var acc = 0.0;
  if (r < fx_P.dOut) {
    let nb = fx_P.dIn / 32u;
    for (var p = lane; p < fx_P.K * nb; p += 32u) {
      let k = p / nb; let b = p % nb;
      var wk = fx_wts[k];
      if (fx_P.sig == 1u) { wk = sigm(wk); }
      let blk = (fx_ids[k] * fx_P.dOut + r) * nb + b;
      let xb4 = (k * fx_P.dIn) / 4u + b * 8u;
      var d = 0.0;
      ${blockDot(kind, "fx_qs", "fx_sc", "fx_x")}
      acc += wk * d;
    }
  }
  fx_red[lid] = acc;
  ${LANE_REDUCE()}
  if (lane == 0u && r < fx_P.dOut) { fx_y[r] = fx_red[lid]; }
}`;
}

// Router: softmax over all nExp logits, top-K (equal values: lowest index first), weights renormalised to sum 1
// (the sum clamped at 6.103515625e-5 as llama.cpp's build_moe_ffn). One workgroup.
export const MOE_ROUTE_WGSL = DQ + `
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
fn main(@builtin(local_invocation_id) l: vec3<u32>) {
  let t = l.x;
  var m = -3.0e38;
  for (var e = t; e < fx_P.nExp; e += 256u) { m = max(m, fx_logits[e]); fx_taken[e] = 0u; }
  fx_rv[t] = m;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { fx_rv[t] = max(fx_rv[t], fx_rv[t + s]); } workgroupBarrier(); }
  let mx = fx_rv[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var e = t; e < fx_P.nExp; e += 256u) { sum += exp(fx_logits[e] - mx); }
  fx_rv[t] = sum;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { fx_rv[t] += fx_rv[t + s]; } workgroupBarrier(); }
  let tot = fx_rv[0];
  workgroupBarrier();
  for (var k = 0u; k < fx_P.K; k++) {
    var bv = -3.0e38; var bi = 0xffffffffu;
    for (var e = t; e < fx_P.nExp; e += 256u) {
      if (fx_taken[e] == 0u && fx_logits[e] > bv) { bv = fx_logits[e]; bi = e; }
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
    if (t == 0u) { let e = fx_ri[0]; fx_ids[k] = e; fx_taken[e] = 1u; fx_pk[k] = exp(fx_logits[e] - mx) / tot; }
    workgroupBarrier();
  }
  if (t == 0u) {
    var ws = 0.0;
    for (var k = 0u; k < fx_P.K; k++) { ws += fx_pk[k]; }
    ws = max(ws, 6.103515625e-5);
    for (var k = 0u; k < fx_P.K; k++) { fx_wts[k] = fx_pk[k] / ws; }
  }
}`;

// HC mixer, part 1: a separate RMS norm per stream, xn[c] = rmsnorm(R[c]) * w[c] (the converter folded 1 + w).
// One workgroup per stream.
export const HC_NORM_WGSL = `
struct NP { d: u32, eps: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> fx_R: array<f32>;
@group(0) @binding(1) var<storage, read> fx_w: array<f32>;
@group(0) @binding(2) var<storage, read_write> fx_xn: array<f32>;
@group(0) @binding(3) var<uniform> fx_P: NP;
var<workgroup> fx_red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let t = l.x; let o = wg.x * fx_P.d;
  var ss = 0.0;
  for (var i = t; i < fx_P.d; i += 256u) { let v = fx_R[o + i]; ss += v * v; }
  fx_red[t] = ss;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { fx_red[t] += fx_red[t + s]; } workgroupBarrier(); }
  let inv = 1.0 / sqrt(fx_red[0] / f32(fx_P.d) + fx_P.eps);
  for (var i = t; i < fx_P.d; i += 256u) { fx_xn[o + i] = (fx_R[o + i] * inv) * fx_w[o + i]; }
}`;

// HC mixer, part 2: mixed[i] = mean over c of xn[c][i] * sigmoid(gate[c][i]).
export const HC_MIX_WGSL = DQ + `
struct XP { d: u32, hc: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> fx_xn: array<f32>;
@group(0) @binding(1) var<storage, read> fx_gate: array<f32>;
@group(0) @binding(2) var<storage, read_write> fx_mixed: array<f32>;
@group(0) @binding(3) var<uniform> fx_P: XP;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= fx_P.d) { return; }
  var s = 0.0;
  for (var c = 0u; c < fx_P.hc; c++) { let j = c * fx_P.d + i; s += fx_xn[j] * sigm(fx_gate[j]); }
  fx_mixed[i] = s / f32(fx_P.hc);
}`;

// hc_combine: R[c][i] += (a[i] + b[i]) * 2 * sigmoid(inj[c] / hc) (b only when useB = 1: the MoE's shared expert).
export const HC_COMBINE_WGSL = DQ + `
struct CP { d: u32, hc: u32, useB: u32, p1: u32 };
@group(0) @binding(0) var<storage, read_write> fx_R: array<f32>;
@group(0) @binding(1) var<storage, read> fx_a: array<f32>;
@group(0) @binding(2) var<storage, read> fx_b: array<f32>;
@group(0) @binding(3) var<storage, read> fx_inj: array<f32>;
@group(0) @binding(4) var<uniform> fx_P: CP;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let j = g.x;
  if (j >= fx_P.d * fx_P.hc) { return; }
  let c = j / fx_P.d; let i = j % fx_P.d;
  var v = fx_a[i];
  if (fx_P.useB == 1u) { v += fx_b[i]; }
  fx_R[j] += v * (2.0 * sigm(fx_inj[c] / f32(fx_P.hc)));
}`;

// Gated DeltaNet, step 1: the causal conv over the last convK inputs of each channel (state: the convK - 1 older
// inputs, oldest first), then silu. w is [ch][convK], the newest tap last.
export const DN_CONV_WGSL = DQ + `
struct DP { n: u32, K: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> fx_xin: array<f32>;
@group(0) @binding(1) var<storage, read_write> fx_st: array<f32>;
@group(0) @binding(2) var<storage, read> fx_w: array<f32>;
@group(0) @binding(3) var<storage, read_write> fx_co: array<f32>;
@group(0) @binding(4) var<uniform> fx_P: DP;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let c = g.x;
  if (c >= fx_P.n) { return; }
  let K1 = fx_P.K - 1u;
  let xv = fx_xin[c];
  var acc = fx_w[c * fx_P.K + K1] * xv;
  for (var j = 0u; j < K1; j++) { acc += fx_w[c * fx_P.K + j] * fx_st[c * K1 + j]; }
  fx_co[c] = silu(acc);
  for (var j = 0u; j + 1u < K1; j++) { fx_st[c * K1 + j] = fx_st[c * K1 + j + 1u]; }
  fx_st[c * K1 + K1 - 1u] = xv;
}`;

// Step 2: l2-normalise each q and k head in place (2 * nKH workgroups of dS threads): v / max(|v|, eps).
export const DN_L2_WGSL = `
struct LP { dS: u32, eps: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read_write> fx_co: array<f32>;
@group(0) @binding(1) var<uniform> fx_P: LP;
var<workgroup> fx_red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let t = l.x; let o = wg.x * fx_P.dS;
  let v = fx_co[o + t];
  fx_red[t] = v * v;
  workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) { if (t < s) { fx_red[t] += fx_red[t + s]; } workgroupBarrier(); }
  fx_co[o + t] = v * (1.0 / max(sqrt(fx_red[0]), fx_P.eps));
}`;

// Step 3: the delta rule, one workgroup per V head, thread j owns column j of the head's state S[i][j]
// (dS x dS). g = softplus(alpha + dt_bias) * A, decay = exp(g), beta = sigmoid(beta_raw):
//   S *= decay; vhat = (v - S^T k) * beta; S += k vhat^T; o = S^T q / sqrt(dS).
// The V-head order is the Qwen3.5 converter's: V head h reads QK head h % nKH.
export const DN_DELTA_WGSL = DQ + `
struct EP { nKH: u32, dS: u32, keyDim: u32, p0: u32 };
@group(0) @binding(0) var<storage, read> fx_co: array<f32>;
@group(0) @binding(1) var<storage, read_write> fx_S: array<f32>;
@group(0) @binding(2) var<storage, read> fx_alpha: array<f32>;
@group(0) @binding(3) var<storage, read> fx_beta: array<f32>;
@group(0) @binding(4) var<storage, read> fx_dtb: array<f32>;
@group(0) @binding(5) var<storage, read> fx_A: array<f32>;
@group(0) @binding(6) var<storage, read_write> fx_o: array<f32>;
@group(0) @binding(7) var<uniform> fx_P: EP;
var<workgroup> fx_kq: array<f32, 256>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let h = wg.x; let j = l.x; let dS = fx_P.dS;
  let kh = h % fx_P.nKH;
  fx_kq[j] = fx_co[fx_P.keyDim + kh * dS + j];          // k
  fx_kq[128u + j] = fx_co[kh * dS + j];               // q
  workgroupBarrier();
  let araw = fx_alpha[h] + fx_dtb[h];
  var sp = araw;
  if (araw <= 20.0) { sp = log(1.0 + exp(araw)); }
  let decay = exp(sp * fx_A[h]);
  let b = sigm(fx_beta[h]);
  let v = fx_co[2u * fx_P.keyDim + h * dS + j];
  let base = h * dS * dS;
  var vh = 0.0;
  for (var i = 0u; i < dS; i++) {
    let s = fx_S[base + i * dS + j] * decay;
    fx_S[base + i * dS + j] = s;
    vh += s * fx_kq[i];
  }
  vh = (v - vh) * b;
  let scale = 1.0 / sqrt(f32(dS));
  var acc = 0.0;
  for (var i = 0u; i < dS; i++) {
    let s = fx_S[base + i * dS + j] + fx_kq[i] * vh;
    fx_S[base + i * dS + j] = s;
    acc += s * fx_kq[128u + i];
  }
  fx_o[h * dS + j] = acc * scale;
}`;

// Step 4: the gated output norm, per V head: rmsnorm(o_h) * ssm_norm * sigmoid(z) (qwen4exp: sigmoid, where
// Qwen3.5 has silu).
export const DN_GNORM_WGSL = DQ + `
struct GN { dS: u32, eps: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> fx_o: array<f32>;
@group(0) @binding(1) var<storage, read> fx_z: array<f32>;
@group(0) @binding(2) var<storage, read> fx_nw: array<f32>;
@group(0) @binding(3) var<storage, read_write> fx_y: array<f32>;
@group(0) @binding(4) var<uniform> fx_P: GN;
var<workgroup> fx_red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let t = l.x; let i = wg.x * fx_P.dS + t;
  let v = fx_o[i];
  fx_red[t] = v * v;
  workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) { if (t < s) { fx_red[t] += fx_red[t + s]; } workgroupBarrier(); }
  let inv = 1.0 / sqrt(fx_red[0] / f32(fx_P.dS) + fx_P.eps);
  fx_y[i] = (v * inv) * fx_nw[t] * sigm(fx_z[i]);
}`;

// Per-token frame: the position and the RoPE cos/sin of each rotary pair at that position (computed on the CPU
// with the reference's f32 recurrence, theta_i = pos * base^(-2i/nRot)).
const FRAME = `struct Frame { pos: u32, seqLen: u32, p0: u32, p1: u32, cs: array<vec4<f32>, 64> };`;
export const FRAME_FLOATS = 4 + 64 * 4;   // pos, seqLen, pad, pad, then (cos, sin, 0, 0) per rotary pair (<= 64 pairs)

// Gated attention, step 1 (one workgroup per Q head, then one per KV head; hd threads): per-head RMS norm,
// partial NeoX RoPE (pairs i, i + nRot/2 for i < nRot/2), then the K/V cache write at frame.pos. q and the gate are
// interleaved per head in qf: [q_h (hd) | gate_h (hd)].
export const ATTN_PREP_WGSL = `
${FRAME}
struct AP { nH: u32, nKV: u32, hd: u32, nRot: u32, eps: f32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> fx_qf: array<f32>;
@group(0) @binding(1) var<storage, read> fx_kin: array<f32>;
@group(0) @binding(2) var<storage, read> fx_vin: array<f32>;
@group(0) @binding(3) var<storage, read_write> fx_qn: array<f32>;
@group(0) @binding(4) var<storage, read_write> fx_kc: array<f32>;
@group(0) @binding(5) var<storage, read_write> fx_vc: array<f32>;
@group(0) @binding(6) var<storage, read> fx_qw: array<f32>;
@group(0) @binding(7) var<storage, read> fx_kw: array<f32>;
@group(0) @binding(8) var<uniform> fx_P: AP;
@group(0) @binding(9) var<uniform> fx_F: Frame;
var<workgroup> fx_red: array<f32, 256>;
var<workgroup> fx_hv: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let t = l.x; let g = wg.x; let hd = fx_P.hd;
  let isQ = g < fx_P.nH;
  var x = 0.0;
  if (t < hd) {
    if (isQ) { x = fx_qf[g * 2u * hd + t]; } else { x = fx_kin[(g - fx_P.nH) * hd + t]; }
  }
  fx_red[t] = x * x;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { fx_red[t] += fx_red[t + s]; } workgroupBarrier(); }
  let inv = 1.0 / sqrt(fx_red[0] / f32(hd) + fx_P.eps);
  var nv = 0.0;
  if (t < hd) { if (isQ) { nv = (x * inv) * fx_qw[t]; } else { nv = (x * inv) * fx_kw[t]; } }
  fx_hv[t] = nv;
  workgroupBarrier();
  let hr = fx_P.nRot / 2u;
  var r = nv;
  if (t < hr) { let c = fx_F.cs[t]; r = nv * c.x - fx_hv[t + hr] * c.y; }
  else if (t < fx_P.nRot) { let c = fx_F.cs[t - hr]; r = fx_hv[t - hr] * c.y + nv * c.x; }
  if (t < hd) {
    if (isQ) { fx_qn[g * hd + t] = r; }
    else {
      let h = g - fx_P.nH;
      let o = fx_F.pos * fx_P.nKV * hd + h * hd + t;
      fx_kc[o] = r;
      fx_vc[o] = fx_vin[h * hd + t];
    }
  }
}`;

// Step 2: dense causal attention for one decode token over cache cells 0..frame.pos, one workgroup per Q head
// (hd = 256 threads), then out = attn * sigmoid(gate). Scores live in workgroup memory: seqLen <= 2048.
// QSA (roadmap 33 PR 9) replaces the cell range with the selected cells; below 2051 cached tokens it selects all.
export const ATTN_DEC_WGSL = DQ + `
${FRAME}
struct AP { nH: u32, nKV: u32, hd: u32, nRot: u32, eps: f32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> fx_qn: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> fx_kc: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> fx_vc: array<f32>;
@group(0) @binding(3) var<storage, read> fx_qf: array<f32>;
@group(0) @binding(4) var<storage, read_write> fx_out: array<f32>;
@group(0) @binding(5) var<uniform> fx_P: AP;
@group(0) @binding(6) var<uniform> fx_F: Frame;
var<workgroup> fx_sc: array<f32, 2048>;
var<workgroup> fx_red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let t = l.x; let h = wg.x; let hd = fx_P.hd; let hd4 = hd / 4u;
  let kvh = h / (fx_P.nH / fx_P.nKV);
  let n = fx_F.pos + 1u;
  let scale = 1.0 / sqrt(f32(hd));
  var m = -3.0e38;
  for (var p = t; p < n; p += 256u) {
    let ko = (p * fx_P.nKV + kvh) * hd4;
    var a = 0.0;
    for (var i = 0u; i < hd4; i++) { a += dot(fx_qn[h * hd4 + i], fx_kc[ko + i]); }
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
    for (var p = 0u; p < n; p++) { acc += fx_sc[p] * fx_vc[(p * fx_P.nKV + kvh) * hd + t]; }
    fx_out[h * hd + t] = (acc / tot) * sigm(fx_qf[h * 2u * hd + hd + t]);
  }
}`;

// The n-gram (PLE) injection at its layer (roadmap 33 "N-gram embedding (PLE)"), one workgroup per stream c:
//   key[c] = rmsnorm(pk[c]) * nk[c]; query[c] = rmsnorm(R[c]) * nq[c]; s = key[c] . query[c] / sqrt(d)
//   gate[c] = sigmoid(sign(s) * sqrt(max(|s|, 1e-6))); g[c] = pv * gate[c]; n[c] = rmsnorm(g[c]) * nc[c]
//   conv[c] = silu(sum over k of cw[ch][k] * n at t - (K-1-k) * dil)  (depthwise, causal, dilated; a ring of
//   (K-1)*dil + 1 positions, positions before the sequence start read as 0)
//   R[c] += g[c] + conv[c]
// dbg keeps g, conv and gate for the per-layer checks.
export const PLE_WGSL = DQ + `
${FRAME}
struct PP { d: u32, K: u32, dil: u32, slots: u32, eps: f32, hc: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read_write> fx_R: array<f32>;
@group(0) @binding(1) var<storage, read> fx_pk: array<f32>;
@group(0) @binding(2) var<storage, read> fx_pv: array<f32>;
@group(0) @binding(3) var<storage, read> fx_pw: array<f32>;            // [nk | nq | nc] (hc*d each) then cw [hc*d][K]
@group(0) @binding(4) var<storage, read_write> fx_ring: array<f32>;
@group(0) @binding(5) var<storage, read_write> fx_dbg: array<f32>;     // [g (hc*d) | conv (hc*d) | gate (hc)]
@group(0) @binding(6) var<uniform> fx_P: PP;
@group(0) @binding(7) var<uniform> fx_F: Frame;
var<workgroup> fx_r1: array<f32, 256>;
var<workgroup> fx_r2: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let t = l.x; let c = wg.x; let d = fx_P.d; let o = c * d;
  let HD = d * fx_P.hc;   // ring row: every stream's channels
  let NQ = HD; let NC = 2u * HD; let CW = 3u * HD;
  var sk = 0.0; var sq = 0.0;
  for (var i = t; i < d; i += 256u) { let a = fx_pk[o + i]; let b = fx_R[o + i]; sk += a * a; sq += b * b; }
  fx_r1[t] = sk; fx_r2[t] = sq;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { fx_r1[t] += fx_r1[t + s]; fx_r2[t] += fx_r2[t + s]; } workgroupBarrier(); }
  let ik = 1.0 / sqrt(fx_r1[0] / f32(d) + fx_P.eps);
  let iq = 1.0 / sqrt(fx_r2[0] / f32(d) + fx_P.eps);
  workgroupBarrier();
  var dt = 0.0;
  for (var i = t; i < d; i += 256u) { dt += ((fx_pk[o + i] * ik) * fx_pw[o + i]) * ((fx_R[o + i] * iq) * fx_pw[NQ + o + i]); }
  fx_r1[t] = dt;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { fx_r1[t] += fx_r1[t + s]; } workgroupBarrier(); }
  let sv = fx_r1[0] / sqrt(f32(d));
  let gate = sigm(sign(sv) * sqrt(max(abs(sv), 1e-6)));
  workgroupBarrier();
  var sg = 0.0;
  for (var i = t; i < d; i += 256u) { let g = fx_pv[i] * gate; sg += g * g; }
  fx_r1[t] = sg;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { fx_r1[t] += fx_r1[t + s]; } workgroupBarrier(); }
  let ig = 1.0 / sqrt(fx_r1[0] / f32(d) + fx_P.eps);
  if (t == 0u) { fx_dbg[2u * HD + c] = gate; }
  let slot = fx_F.pos % fx_P.slots;
  for (var i = t; i < d; i += 256u) {
    let ch = o + i;
    let g = fx_pv[i] * gate;
    let nv = (g * ig) * fx_pw[NC + ch];
    fx_ring[slot * HD + ch] = nv;
    var acc = fx_pw[CW + ch * fx_P.K + fx_P.K - 1u] * nv;
    for (var k = 0u; k + 1u < fx_P.K; k++) {
      let back = (fx_P.K - 1u - k) * fx_P.dil;
      if (fx_F.pos >= back) { acc += fx_pw[CW + ch * fx_P.K + k] * fx_ring[((fx_F.pos - back) % fx_P.slots) * HD + ch]; }
    }
    let cv = silu(acc);
    fx_dbg[ch] = g; fx_dbg[HD + ch] = cv;
    fx_R[ch] = fx_R[ch] + (g + cv);
  }
}`;

// Greedy head, stage 1: per workgroup (256 threads over a strided slice), the largest logit, lowest index on ties.
export const ARGMAX1_WGSL = `
struct AM { n: u32, per: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> fx_lg: array<f32>;
@group(0) @binding(1) var<storage, read_write> fx_part: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> fx_P: AM;
var<workgroup> fx_rv: array<f32, 256>;
var<workgroup> fx_ri: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let t = l.x; let lo = wg.x * fx_P.per; let hi = min(lo + fx_P.per, fx_P.n);
  var bv = -3.0e38; var bi = 0xffffffffu;
  for (var i = lo + t; i < hi; i += 256u) { let v = fx_lg[i]; if (v > bv) { bv = v; bi = i; } }
  fx_rv[t] = bv; fx_ri[t] = bi;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (t < s) { let ov = fx_rv[t + s]; let oi = fx_ri[t + s]; if (ov > fx_rv[t] || (ov == fx_rv[t] && oi < fx_ri[t])) { fx_rv[t] = ov; fx_ri[t] = oi; } }
    workgroupBarrier();
  }
  if (t == 0u) { fx_part[wg.x] = vec2<f32>(fx_rv[0], bitcast<f32>(fx_ri[0])); }
}`;
// Stage 2: one workgroup over the stage-1 partials -> out[0] = index, out[1] = bits of the value.
export const ARGMAX2_WGSL = `
struct AM { n: u32, per: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> fx_part: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> fx_outv: array<u32>;
@group(0) @binding(2) var<uniform> fx_P: AM;
var<workgroup> fx_rv: array<f32, 256>;
var<workgroup> fx_ri: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>) {
  let t = l.x;
  var bv = -3.0e38; var bi = 0xffffffffu;
  for (var i = t; i < fx_P.n; i += 256u) { let p = fx_part[i]; let pi = bitcast<u32>(p.y); if (p.x > bv || (p.x == bv && pi < bi)) { bv = p.x; bi = pi; } }
  fx_rv[t] = bv; fx_ri[t] = bi;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (t < s) { let ov = fx_rv[t + s]; let oi = fx_ri[t + s]; if (ov > fx_rv[t] || (ov == fx_rv[t] && oi < fx_ri[t])) { fx_rv[t] = ov; fx_ri[t] = oi; } }
    workgroupBarrier();
  }
  if (t == 0u) { fx_outv[0] = fx_ri[0]; fx_outv[1] = bitcast<u32>(fx_rv[0]); }
}`;
