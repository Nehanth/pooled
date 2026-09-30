// CPU check of the wide prefill GEMM (engine/wgsl/gemm_wide.js). The generator emits the kernel body
// as JavaScript (wideKernel(..., { js: true })) with the same control flow and index math as the WGSL;
// it runs here with one generator per thread and workgroupBarrier() as a yield, in float64. Checks:
// every output written once with the right value (Q4_0 and Q8_0, = and +=), clamped row tails,
// padded strides left untouched, no read of an unstaged shared-memory slot, several tile shapes.
// A second test compares float32 summation orders (the wide kernel's vs the 16-column GEMM's) against
// float64 on a real 27B shape: the wide order is as accurate, just different. No GPU.
//   deno test --no-check tests/unit/gemm_wide_test.js
import { wideKernel, wideTileConfig, gemmWideWGSL, WIDE_TILE_DEFAULT, SILU_MUL_W_WGSL } from "../../engine/wgsl/gemm_wide.js";
import { f16ToF32, f32ToF16 } from "../../engine/gguf.js";

const H = {
  q4lo: (w) => [0, 8, 16, 24].map((s) => ((w >>> s) & 15) - 8),
  q4hi: (w) => [4, 12, 20, 28].map((s) => ((w >>> s) & 15) - 8),
  i8x4: (w) => [0, 8, 16, 24].map((s) => (((w >>> s) & 255) << 24) >> 24),
  mul4: (v, s) => { if (v.some((e) => e == null) || s == null || Number.isNaN(s)) throw new Error("dequant of an unloaded value"); return v.map((e) => e * s); },
  fma4: (a, s, x) => {
    if (s == null || x == null || Number.isNaN(s) || x.some((e) => e == null || Number.isNaN(e))) throw new Error("read of an unstaged shared-memory slot");
    return a.map((e, i) => e + s * x[i]);
  },
  unpack2x16float: (w) => [f16ToF32(w & 0xffff), f16ToF32(w >>> 16)],
  min: Math.min,
};
let seed = 12345;
const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
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
const vec4s = (a) => Array.from({ length: a.length / 4 }, (_, i) => Array.from(a.subarray(i * 4, i * 4 + 4)));

function run(k, c, bufs, gx, gy) {
  const hn = Object.keys(H), bn = Object.keys(bufs).map((n) => `gw_${n}`);
  const make = new Function(...hn, ...bn, "gw_W", "gw_X", `return function* (wg, lid) {${k.body}};`);
  for (let y = 0; y < gy; y++) for (let x = 0; x < gx; x++) {
    const W = new Array(c.KS * c.BM / 4).fill(null), X = new Array(c.KS * c.BN / 4).fill(null);
    const fn = make(...hn.map((n) => H[n]), ...Object.values(bufs), W, X);
    const th = Array.from({ length: c.T }, (_, t) => fn({ x, y }, { x: t }));
    for (let stage = 0; ; stage++) {
      const done = th.map((g) => g.next().done);
      if (done.every(Boolean)) break;
      if (done.some(Boolean)) throw new Error(`${k.name}: threads disagree at a barrier`);
      // poison the slots before each staging round (odd yields are the one before staging)
      if (stage % 2 === 0) { W.fill(null); X.fill(null); }
    }
  }
}

// y[col][row] (+)= W x[col] for N columns; padded strides, rows not a multiple of BM
function check(fmt, acc, c, { dOut, dIn, N }) {
  const xs4 = dIn / 4 + 3, ys = dOut + 5;
  const Wq = quant(fmt, dOut, dIn);
  const x = new Float32Array(N * xs4 * 4); for (let i = 0; i < x.length; i++) x[i] = rnd() - 0.5;
  const y0 = Float64Array.from({ length: N * ys + 8 }, () => (acc ? rnd() - 0.5 : NaN));
  const y = Float64Array.from(y0);
  const k = wideKernel(fmt, acc, c, { js: true });
  run(k, c, { q: vec4s(Wq.qs), sc: Wq.sc, x: vec4s(x), y, s: { dOut, dIn, xs4, ys } }, Math.ceil(dOut / c.BM), N / c.BN);
  let err = 0, mx = 0;
  for (let col = 0; col < N; col++) {
    for (let r = 0; r < dOut; r++) {
      let ref = acc ? y0[col * ys + r] : 0;
      for (let j = 0; j < dIn; j++) ref += Wq.deq(r, j) * x[col * xs4 * 4 + j];
      const got = y[col * ys + r];
      if (Number.isNaN(got)) throw new Error(`y[${col}][${r}] not written`);
      err = Math.max(err, Math.abs(got - ref)); mx = Math.max(mx, Math.abs(ref));
    }
    for (let r = dOut; r < ys; r++) if (!Object.is(y[col * ys + r], y0[col * ys + r])) throw new Error(`y[${col}] gap row ${r} written`);
  }
  for (let i = N * ys; i < y.length; i++) if (!Object.is(y[i], y0[i])) throw new Error("write past the last column");
  if (!(err / mx < 1e-12)) throw new Error(`${k.name} ${JSON.stringify(c)}: relative error ${err / mx}`);
}

Deno.test("wide gemm: default tile, Q4_0 / Q8_0, = and +=, KB 1 and 2, clamped row tail", () => {
  for (const [wgMem, KB] of [[16384, undefined], [32768, undefined], [32768, 2]]) {
    const c = wideTileConfig(KB ? { KB } : {}, wgMem);
    if (c.KB !== (KB ?? 1)) throw new Error(`KB ${c.KB} for ${wgMem} B`);
    for (const fmt of ["q4", "q8"]) for (const acc of [false, true]) check(fmt, acc, c, { dOut: 72, dIn: 128, N: 128 });
  }
});

Deno.test("wide gemm: tile shape sweep", () => {
  const shapes = [{ BM: 128, BN: 64, TM: 8, TN: 4, KB: 2 }, { BM: 64, BN: 128, TM: 4, TN: 8, KB: 2 }, { BM: 32, BN: 32, TM: 4, TN: 4 },
    { BM: 128, BN: 128, TM: 8, TN: 8, KB: 1 }, { BM: 32, BN: 64, TM: 4, TN: 8 }];
  let n = 0;
  for (const s of shapes) {
    const c = wideTileConfig(s, 49152);
    for (const fmt of ["q4", "q8"]) { check(fmt, n % 2 === 1, c, { dOut: c.BM + 20, dIn: 64 * c.KB, N: c.BN }); n++; }
  }
});

Deno.test("wide gemm: config resolution and WGSL emission", () => {
  const d = wideTileConfig(undefined, 16384);
  if (d.BM !== WIDE_TILE_DEFAULT.BM || d.KB !== 1 || d.T !== 256 || d.smem !== 16384) throw new Error("default config changed: " + JSON.stringify(d));
  for (const bad of [{ TM: 3 }, { BM: 512, TM: 4 }, { BN: 8, TN: 4 }, { BM: 256, BN: 256, TM: 4, TN: 4 }, { KB: 3 }]) {
    let threw = false; try { wideTileConfig(bad, 49152); } catch { threw = true; }
    if (!threw) throw new Error("accepted " + JSON.stringify(bad));
  }
  let threw = false; try { wideTileConfig({ BM: 128, BN: 128, TM: 8, TN: 8 }, 16384); } catch { threw = true; }
  if (!threw) throw new Error("a 128x128 tile cannot fit 16 KB");
  const w = gemmWideWGSL(d);
  for (const f of ["fn gemm_w_q4(", "fn gemm_w_q4_acc(", "fn gemm_w_q8(", "fn gemm_w_q8_acc(", "array<vec4<f32>, 512>"])
    if (!w.includes(f)) throw new Error("WGSL is missing " + f);
  // the multi-column SiLU is its own snippet (in every engine module: the wide chunk and the batched pass use it)
  if (w.includes("fn silu_mul_w(") || !SILU_MUL_W_WGSL.includes("fn silu_mul_w(")) throw new Error("silu_mul_w belongs to SILU_MUL_W_WGSL only");
  if (gemmWideWGSL(d, { UNPACK: false }).includes("unpack4x")) throw new Error("UNPACK false still uses unpack4x");
});

// float32 accuracy of the two prefill summation orders, emulated with Math.fround (no FMA contraction):
//   wide: one accumulator per output, k ascending;  gemm16: engine/wgsl/gemm.js (split-K S, lo/hi nibble
//   interleave inside a block, partials summed in split order). Both against float64.
Deno.test("wide gemm: f32 summation order is as accurate as the 16-column GEMM (27B ffn_down shape)", () => {
  const f = Math.fround, dIn = 17408, rows = 24, S = 2, nb = dIn / 32;
  const Wq = quant("q4", rows, dIn);
  const W = Array.from({ length: rows }, (_, r) => Float32Array.from({ length: dIn }, (_, j) => Wq.deq(r, j)));
  let eW = 0, eG = 0, dWG = 0, mx = 0;
  for (let col = 0; col < 4; col++) {
    const x = Float32Array.from({ length: dIn }, () => (rnd() - 0.5) * 4);
    for (let r = 0; r < rows; r++) {
      let t = 0; for (let j = 0; j < dIn; j++) t += W[r][j] * x[j];
      let a = 0; for (let j = 0; j < dIn; j++) a = f(a + f(W[r][j] * x[j]));
      let g = 0;
      for (let s = 0; s < S; s++) {
        let p = 0;
        for (let b = s * nb / S; b < (s + 1) * nb / S; b++) for (let jj = 0; jj < 4; jj++) for (let i = 0; i < 4; i++) {
          const kl = 32 * b + 4 * jj + i, kh = kl + 16;
          p = f(p + f(W[r][kl] * x[kl])); p = f(p + f(W[r][kh] * x[kh]));
        }
        g = f(g + p);
      }
      eW = Math.max(eW, Math.abs(a - t)); eG = Math.max(eG, Math.abs(g - t)); dWG = Math.max(dWG, Math.abs(a - g)); mx = Math.max(mx, Math.abs(t));
    }
  }
  console.log(`  f32 vs f64 max rel: wide ${(eW / mx).toExponential(2)}, 16-col GEMM ${(eG / mx).toExponential(2)}; wide vs 16-col ${(dWG / mx).toExponential(2)}`);
  if (!(eW / mx < 1e-4) || eW > 4 * eG + 1e-6 * mx) throw new Error("wide summation order is less accurate than expected");
});
