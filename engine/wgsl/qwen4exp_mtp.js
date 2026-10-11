// WGSL for the Qwen3.8-Flash-Next MTP draft block (engine/qwen4exp_mtp.js). Roadmap 33 PR plan item 6.
// Its own file so the decode kernels (engine/wgsl/qwen4exp.js) stay as they are; the block itself runs on those.

// The block's input, before eh_proj: for each HC stream s, x_s = [enorm(e) ; hnorm(h_s)] (2 * d values), where e is
// the token's embedding row and h the trunk's 4-stream residual before its head mixer (llama.cpp graph_mtp:
// ggml_concat(e_norm repeated per stream, h_norm, 0)). Both are plain RMSNorms times the file's gammas (enorm: d,
// hnorm: d per stream), the same arithmetic as HC_NORM_WGSL. eh_proj then maps each x_s to stream s of the block's
// residual (one GEMV per stream). One workgroup per stream; each one also normalizes e (d values, a few us).
export const MTP_IN_WGSL = `
struct MP { d: u32, eps: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> fx_e: array<f32>;
@group(0) @binding(1) var<storage, read> fx_h: array<f32>;
@group(0) @binding(2) var<storage, read> fx_ew: array<f32>;
@group(0) @binding(3) var<storage, read> fx_hw: array<f32>;
@group(0) @binding(4) var<storage, read_write> fx_x: array<f32>;
@group(0) @binding(5) var<uniform> fx_P: MP;
var<workgroup> fx_red: array<f32, 256>;
var<workgroup> fx_red2: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let t = l.x; let d = fx_P.d; let o = wg.x * d; let xo = wg.x * 2u * d;
  var se = 0.0; var sh = 0.0;
  for (var i = t; i < d; i += 256u) { let a = fx_e[i]; se += a * a; let b = fx_h[o + i]; sh += b * b; }
  fx_red[t] = se; fx_red2[t] = sh;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { fx_red[t] += fx_red[t + s]; fx_red2[t] += fx_red2[t + s]; } workgroupBarrier(); }
  let ie = 1.0 / sqrt(fx_red[0] / f32(d) + fx_P.eps);
  let ih = 1.0 / sqrt(fx_red2[0] / f32(d) + fx_P.eps);
  for (var i = t; i < d; i += 256u) {
    fx_x[xo + i] = (fx_e[i] * ie) * fx_ew[i];
    fx_x[xo + d + i] = (fx_h[o + i] * ih) * fx_hw[o + i];
  }
}`;
