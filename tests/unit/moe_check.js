// CPU check of the MoE expert GEMV kernels (engine/wgsl/moe.js) for every layout option: the generator
// emits the same kernel body as JavaScript (expertKernel(..., js = true)), which runs here with one
// generator per thread and workgroupBarrier() as a yield, in float64. It checks the index math and the
// reduction (every output row written once with the right value, nothing written outside it, tails and
// clamped rows handled), not float32 rounding. No GPU. Shared by moe_kernels_test.js and the
// moe_sweep_wg*_test.js shards (separate files so deno test --parallel runs the sweep on several cores).
import { moeKernelConfig, expertKernel } from "../../engine/wgsl/moe.js";
import { f16ToF32, f32ToF16 } from "../../engine/gguf.js";

const H = {
  q4lo: (w) => [0, 8, 16, 24].map((s) => ((w >>> s) & 15) - 8),
  q4hi: (w) => [4, 12, 20, 28].map((s) => ((w >>> s) & 15) - 8),
  i8x4: (w) => [0, 8, 16, 24].map((s) => (((w >>> s) & 255) << 24) >> 24),
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3],
  unpack2x16float: (w) => [f16ToF32(w & 0xffff), f16ToF32(w >>> 16)],
  select: (f, t, c) => (c ? t : f), min: Math.min, exp: Math.exp,
};
let seed = 777;
const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
// [rows][dIn] quantized in the engine's layout
function quant(fmt, rows, dIn) {
  const nb = dIn / 32, W = fmt === "q4" ? 4 : 8;
  const qs = new Uint32Array(rows * nb * W), sch = new Uint16Array(rows * nb + ((rows * nb) % 2));
  for (let i = 0; i < qs.length; i++) qs[i] = (rnd() * 4294967296) >>> 0;
  for (let i = 0; i < rows * nb; i++) sch[i] = f32ToF16((rnd() - 0.5) * 0.05);
  const deq = (r, j) => {
    const b = Math.floor(j / 32), k = j % 32, s = f16ToF32(sch[r * nb + b]);
    if (fmt === "q4") { const w = qs[(r * nb + b) * 4 + ((k % 16) >> 2)]; return s * (((w >>> (8 * (k % 4) + (k >= 16 ? 4 : 0))) & 15) - 8); }
    const w = qs[(r * nb + b) * 8 + (k >> 2)]; let q = (w >>> (8 * (k % 4))) & 255; if (q > 127) q -= 256; return s * q;
  };
  return { qs, sc: new Uint32Array(sch.buffer), deq };
}
const vec4s = (u32) => Array.from({ length: u32.length / 4 }, (_, i) => Array.from(u32.subarray(i * 4, i * 4 + 4)));
const vecF = (f) => Array.from({ length: f.length / 4 }, (_, i) => Array.from(f.subarray(i * 4, i * 4 + 4)));

// run one kernel over a (gx, gy) grid; bufs maps binding names (without the P_ prefix) to values
function run(k, c, bufs, gx, gy) {
  const hn = Object.keys(H), bn = Object.keys(bufs).map((n) => `${k.P}_${n}`);
  const make = new Function(...hn, ...bn, `${k.P}_red`, `${k.P}_xs`, `return function* (wg, lid) {${k.body}};`);
  const NA = k.name.startsWith("moe_gu") ? 2 : 1;
  for (let y = 0; y < gy; y++) for (let x = 0; x < gx; x++) {
    const red = new Float64Array(NA * c.R * c.WG).fill(NaN), xs = new Array(8 * (c.dIn / 32 + 1)).fill(null);
    const fn = make(...hn.map((n) => H[n]), ...Object.values(bufs), red, xs);
    const th = Array.from({ length: c.WG }, (_, t) => fn({ x, y }, { x: t }));
    for (;;) {
      const done = th.map((g) => g.next().done);
      if (done.every(Boolean)) break;
      if (done.some(Boolean)) throw new Error(`${k.name}: threads disagree at a barrier`);
    }
  }
}

// one gate/up + down pass over C columns x K slots; returns the max relative error of each kernel vs float64
export function check(fmtG, fmtD, cfg, { dim, inter, nExp = 5, K = 3, C = 2 }) {
  const xsF = dim + 8, ysH = inter + 4, ysY = dim + 4;   // padded strides: gaps must stay untouched
  const Wg = quant(fmtG, nExp * inter, dim), Wu = quant(fmtG, nExp * inter, dim), Wd = quant(fmtD, nExp * dim, inter);
  const x = Float32Array.from({ length: C * xsF }, () => rnd() - 0.5);
  const sel = Uint32Array.from({ length: C * K }, () => Math.floor(rnd() * nExp));
  const hOut = new Float64Array(C * K * ysH + 16).fill(NaN), yOut = new Float64Array(C * K * ysY + 16).fill(NaN);
  const q = (fmt, W, wide) => (wide ? vec4s(W.qs) : W.qs);
  const kg = expertKernel("gu", fmtG, cfg.gu, true), kd = expertKernel("dn", fmtD, cfg.dn, true);
  run(kg, cfg.gu, { gq: q(fmtG, Wg, cfg.gu.wide), gs: Wg.sc, uq: q(fmtG, Wu, cfg.gu.wide), us: Wu.sc, x: vecF(x), h: hOut, sel,
    s: { dOut: inter, dIn: dim, K, nExp, xs: xsF, ys: ysH } }, Math.ceil(inter / cfg.gu.rows), C * K);
  // the down kernel reads h as vec4s with stride ys (a multiple of 4)
  const hIn = new Float32Array(C * K * ysH); for (let i = 0; i < C * K * ysH; i++) hIn[i] = Number.isNaN(hOut[i]) ? 0 : hOut[i];
  run(kd, cfg.dn, { q: q(fmtD, Wd, cfg.dn.wide), sc: Wd.sc, x: vecF(hIn), y: yOut, sel, s: { dOut: dim, dIn: inter, K, nExp, xs: ysH, ys: ysY } },
    Math.ceil(dim / cfg.dn.rows), C * K);
  let eh = 0, ey = 0, mh = 0, my = 0;
  for (let cs = 0; cs < C * K; cs++) {
    const col = Math.floor(cs / K), e = sel[cs];
    for (let r = 0; r < inter; r++) {
      let g = 0, u = 0;
      for (let j = 0; j < dim; j++) { const xv = x[col * xsF + j]; g += Wg.deq(e * inter + r, j) * xv; u += Wu.deq(e * inter + r, j) * xv; }
      const ref = (g / (1 + Math.exp(-g))) * u, got = hOut[cs * ysH + r];
      if (Number.isNaN(got)) throw new Error(`h[${cs}][${r}] not written`);
      eh = Math.max(eh, Math.abs(got - ref)); mh = Math.max(mh, Math.abs(ref));
    }
    for (let r = inter; r < ysH; r++) if (!Number.isNaN(hOut[cs * ysH + r])) throw new Error(`h[${cs}] gap ${r} written`);
    for (let r = 0; r < dim; r++) {
      let y = 0; for (let j = 0; j < inter; j++) y += Wd.deq(e * dim + r, j) * hIn[cs * ysH + j];
      const got = yOut[cs * ysY + r];
      if (Number.isNaN(got)) throw new Error(`y[${cs}][${r}] not written`);
      ey = Math.max(ey, Math.abs(got - y)); my = Math.max(my, Math.abs(y));
    }
    for (let r = dim; r < ysY; r++) if (!Number.isNaN(yOut[cs * ysY + r])) throw new Error(`y[${cs}] gap ${r} written`);
  }
  for (const [b, n] of [[hOut, C * K * ysH], [yOut, C * K * ysY]]) for (let i = n; i < b.length; i++) if (!Number.isNaN(b[i])) throw new Error("write past the end");
  return [eh / mh, ey / my];
}
export const ok = ([a, b], what) => { if (!(a < 1e-12 && b < 1e-12)) throw new Error(`${what}: relative error gate/up ${a}, down ${b}`); };


// the layout sweep at dim 256 / expert width 96 (tails on both kernels), restricted to the given
// workgroup sizes; n counts over the full sweep so each layout gets the same q4/q8 pairing as unsplit
export function sweep(WGs) {
  const dims = { dim: 256, inter: 96 };
  let n = 0, checked = 0;
  for (const WG of [32, 64, 128]) for (const TPR of [1, 2, 4, 8, 16, 32]) for (const R of [1, 2, 3]) for (const U of [1, 2, 3])
    for (const wide of [true, false]) for (const xsh of [true, false]) {
      if (TPR > WG || R > TPR || (!wide && TPR < 4)) continue;
      const fg = n % 2 ? "q4" : "q8", fd = n % 2 ? "q8" : "q4";
      n++;
      if (!WGs.includes(WG)) continue;
      const v = { WG, TPR, R, U, wide, xsh }, cfg = moeKernelConfig({ gu: v, dn: v }, dims);
      try { ok(check(fg, fd, cfg, dims), JSON.stringify(v)); } catch (e) { throw new Error(`${JSON.stringify(v)} ${fg}/${fd}: ${e.message}`); }
      checked++;
    }
  return checked;
}
