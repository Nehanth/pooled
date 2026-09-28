// CPU check of the wide-layout fused MoE kernels (engine/wgsl/moe.js gusKernelWide / dncKernelWide) against the
// legacy fused kernels (gusKernel / dncKernel), both generated with JavaScript op spellings and run with one
// generator per thread (workgroupBarrier() as a yield), in float64. The two layouts sum in different orders, so
// they agree only up to rounding (~1e-15 in float64); what must hold exactly (===) is that a column's h and x do
// not depend on how many other columns share the pass (the batched verify / prefill path == the one-token path).
// It also checks that every output row is written, nothing outside the outputs is, and the layout validation.
// No GPU.   deno test --allow-read tests/unit/moe_fused_wide_test.js
import { FOPS_JS, gusKernel, dncKernel, gusKernelWide, dncKernelWide, wgslToJs, moeFusedLayout, moeFusedWGSL, MOEF_WIDE } from "../../engine/wgsl/moe.js";
import { f16ToF32, f32ToF16 } from "../../engine/gguf.js";

const H = {
  q4lo: (w) => [0, 8, 16, 24].map((s) => ((w >>> s) & 15) - 8),
  q4hi: (w) => [4, 12, 20, 28].map((s) => ((w >>> s) & 15) - 8),
  i8x4: (w) => [0, 8, 16, 24].map((s) => (((w >>> s) & 255) << 24) >> 24),
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3],
  unpack2x16float: (w) => [f16ToF32(w & 0xffff), f16ToF32(w >>> 16)],
  select: (f, t, c) => (c ? t : f), min: Math.min, max: Math.max, exp: Math.exp,
};
let seed = 4242;
const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);

function bodyJs(src, name) {
  const i = src.indexOf(`fn ${name}(`), j = src.indexOf("{\n", i), k = src.indexOf("\n}", j) + 1;
  if (i < 0) throw new Error(`${name} not in source`);
  return wgslToJs(src.slice(j + 2, k));
}
function run(src, name, WG, bufs, wgv, gx, gy) {
  const hn = Object.keys(H), bn = Object.keys(bufs), wn = Object.keys(wgv);
  const make = new Function(...hn, ...bn, ...wn, `return function* (wg, lid) {${bodyJs(src, name)}};`);
  for (let y = 0; y < gy; y++) for (let x = 0; x < gx; x++) {
    const fn = make(...hn.map((n) => H[n]), ...bn.map((n) => bufs[n]), ...wn.map((n) => wgv[n]()));
    const th = Array.from({ length: WG }, (_, t) => fn({ x, y }, { x: t }));
    for (;;) {
      const done = th.map((g) => g.next().done);
      if (done.every(Boolean)) break;
      if (done.some(Boolean)) throw new Error(`${name}: threads disagree at a barrier`);
    }
  }
}
function quant(fmt, rows, dIn) {
  const nb = dIn / 32, W = fmt === "q4" ? 4 : 8;
  const qs = new Uint32Array(rows * nb * W), sch = new Uint16Array(rows * nb + ((rows * nb) % 2));
  for (let i = 0; i < qs.length; i++) qs[i] = (rnd() * 4294967296) >>> 0;
  for (let i = 0; i < rows * nb; i++) sch[i] = f32ToF16((rnd() - 0.5) * 0.05);
  return { qs, sc: new Uint32Array(sch.buffer) };
}
const vec4s = (u32) => Array.from({ length: Math.ceil(u32.length / 4) }, (_, i) => [0, 1, 2, 3].map((j) => u32[i * 4 + j] ?? 0));
const vecF = (f) => Array.from({ length: f.length / 4 }, (_, i) => Array.from(f.subarray(i * 4, i * 4 + 4)));
// engine/qwen35.js packGU: gate qs | up qs | gate scales | up scales (words)
function packShared(g, u) {
  const oUq = g.qs.length, oGs = oUq + u.qs.length, oUs = oGs + g.sc.length;
  const b = new Uint32Array(oUs + u.sc.length);
  b.set(g.qs, 0); b.set(u.qs, oUq); b.set(g.sc, oGs); b.set(u.sc, oUs);
  return { buf: b, oUq, oGs, oUs };
}

// A model: routed experts + shared expert weights, the columns' inputs, residuals and routing.
function model({ fmt, sfmt, dim, ei, sDim, nExp, K, C }) {
  const KS = K + 1, xs = dim + 8;
  const Wg = quant(fmt, nExp * ei, dim), Wu = quant(fmt, nExp * ei, dim), Wd = quant(fmt, nExp * dim, ei);
  const Sg = quant(sfmt, sDim, dim), Su = quant(sfmt, sDim, dim), Sd = quant(sfmt, dim, sDim), pk = packShared(Sg, Su);
  const xn = Float64Array.from({ length: C * xs }, () => rnd() - 0.5);
  const x0 = Float64Array.from({ length: C * xs }, () => (rnd() - 0.5) * 1e-6);
  const sel = new Uint32Array(C * KS), w = new Float64Array(C * KS);
  for (let c = 0; c < C; c++) {
    const pick = new Set(); while (pick.size < K) pick.add(Math.floor(rnd() * nExp));
    [...pick].forEach((e, k) => { sel[c * KS + k] = e; w[c * KS + k] = rnd(); });
    sel[c * KS + K] = 0; w[c * KS + K] = rnd() * 1e-3;
  }
  return { fmt, sfmt, dim, ei, sDim, nExp, K, C, KS, xs, hs: Math.max(ei, sDim), Wg, Wu, Wd, Sd, pk, xn, x0, sel, w };
}
// Run the fused FFN on columns cols of model m (as one pass of cols.length columns), legacy (lay null) or wide.
function ffn(m, lay, cols) {
  const { fmt, sfmt, dim, ei, sDim, nExp, K, KS, xs, hs, Wg, Wu, Wd, Sd, pk } = m, C = cols.length;
  const pick = (a, n) => { const o = new a.constructor(C * n); cols.forEach((c, i) => o.set(a.subarray(c * n, c * n + n), i * n)); return o; };
  const xn = pick(m.xn, xs), sel = pick(m.sel, KS), w = pick(m.w, KS), xA = Array.from(pick(m.x0, xs));
  const uG = { dOut: ei, dIn: dim, sDim, nExp, xs, ys: hs, norm: 1, shOff: 1, oUq: pk.oUq, oGs: pk.oGs, oUs: pk.oUs, pad: 0 };
  const uD = { dOut: dim, dIn: ei, sDim, nExp, xs, ys: hs, norm: 1, shOff: 1, oUq: 0, oGs: 0, oUs: 0, pad: 0 };
  const P = `gs${fmt}${sfmt}`, Q = `dc${fmt}${sfmt}`, V = lay ? vec4s : (a) => a;
  const srcG = lay ? gusKernelWide(fmt, sfmt, K, lay.gu, FOPS_JS) : gusKernel(fmt, sfmt, K, 256, FOPS_JS);
  const srcD = lay ? dncKernelWide(fmt, sfmt, K, lay.dn, FOPS_JS) : dncKernel(fmt, sfmt, K, 1, 64, FOPS_JS);
  const WGg = lay ? lay.gu.WG : 256, WGd = lay ? lay.dn.WG : 64, NRg = lay ? 2 * lay.gu.R : 8, NRd = KS * (lay ? lay.dn.R : 1);
  const hA = new Array(C * KS * hs + 8).fill(NaN);
  run(srcG, `moe_gus_${fmt}_${sfmt}`, WGg, { [`${P}_gq`]: V(Wg.qs), [`${P}_gs`]: Wg.sc, [`${P}_uq`]: V(Wu.qs), [`${P}_us`]: Wu.sc, [`${P}_x`]: vecF(xn),
    [`${P}_h`]: hA, [`${P}_sel`]: sel, [`${P}_sh`]: V(pk.buf), [`${P}_s`]: uG }, { [`${P}_red`]: () => new Array(NRg * WGg).fill(NaN) },
    Math.ceil(hs / (lay ? lay.gu.rows : 4)), C * KS);
  for (let i = C * KS * hs; i < hA.length; i++) if (!Number.isNaN(hA[i])) throw new Error("h written past the end");
  const hIn = Float64Array.from(hA.slice(0, C * KS * hs), (v) => (Number.isNaN(v) ? 0 : v));
  run(srcD, `moe_dnc_${fmt}_${sfmt}`, WGd, { [`${Q}_q`]: V(Wd.qs), [`${Q}_sc`]: Wd.sc, [`${Q}_h`]: vecF(hIn), [`${Q}_x`]: xA, [`${Q}_sel`]: sel, [`${Q}_w`]: w,
    [`${Q}_sq`]: V(Sd.qs), [`${Q}_ss`]: Sd.sc, [`${Q}_s`]: uD }, { [`${Q}_red`]: () => new Array(NRd * WGd).fill(NaN) },
    Math.ceil(dim / (lay ? lay.dn.rows : 1)), C);
  return { h: hA, x: xA };
}
function check(cfg, lay) {
  const m = model(cfg), { C, KS, K, hs, xs, ei, sDim, dim } = m, all = Array.from({ length: C }, (_, c) => c);
  const A = ffn(m, null, all), B = ffn(m, lay, all);
  const close = (a, b) => Math.abs(a - b) <= 1e-9 * (Math.abs(a) + 1e-3);
  for (let cs = 0; cs < C * KS; cs++) {
    const rows = cs % KS === K ? sDim : ei;
    for (let r = 0; r < hs; r++) {
      const a = A.h[cs * hs + r], b = B.h[cs * hs + r];
      if (r < rows ? !close(a, b) : !Number.isNaN(b)) throw new Error(`wide h[${cs}][${r}]: ${b} vs legacy ${a}`);
    }
  }
  for (let c = 0; c < C; c++) for (let i = 0; i < xs; i++) {
    const a = A.x[c * xs + i], b = B.x[c * xs + i], x0 = m.x0[c * xs + i];
    if (i >= dim ? b !== x0 : !close(a, b) || b === x0) throw new Error(`wide x[${c}][${i}]: ${b} vs legacy ${a}`);
  }
  // one column alone gives exactly the bits it gets in the C-column pass
  for (const c of [0, C - 1]) {
    const S = ffn(m, lay, [c]);
    for (let i = 0; i < KS * hs; i++) if (!Object.is(S.h[i], B.h[c * KS * hs + i])) throw new Error(`column ${c} alone: h[${i}] ${S.h[i]} != ${B.h[c * KS * hs + i]}`);
    for (let i = 0; i < xs; i++) if (!Object.is(S.x[i], B.x[c * xs + i])) throw new Error(`column ${c} alone: x[${i}] ${S.x[i]} != ${B.x[c * xs + i]}`);
  }
}

Deno.test("moe fused wide: close to legacy, batched === one column (q4 routed / q8 shared, default layout)", () => {
  check({ fmt: "q4", sfmt: "q8", dim: 256, ei: 64, sDim: 96, nExp: 5, K: 3, C: 3 }, moeFusedLayout("wide", 3));
});
Deno.test("moe fused wide: layout sweep with tails (q8 / q4, dOut not a multiple of the rows per workgroup)", () => {
  let n = 0;
  for (const [g, d] of [[{ WG: 32, TPR: 32, R: 1 }, { WG: 16, TPR: 4, R: 3 }], [{ WG: 64, TPR: 8, R: 3 }, { WG: 32, TPR: 2, R: 2 }],
    [{ WG: 16, TPR: 4, R: 4 }, { WG: 64, TPR: 16, R: 1 }], [{ WG: 8, TPR: 1, R: 1 }, { WG: 8, TPR: 8, R: 4 }], [{ WG: 32, TPR: 2, R: 1 }, { WG: 16, TPR: 1, R: 1 }]]) {
    const K = 2 + (n % 2);
    check({ fmt: n % 2 ? "q4" : "q8", sfmt: n % 2 ? "q8" : "q4", dim: 160, ei: 96, sDim: 32, nExp: 4, K, C: 2 }, moeFusedLayout({ gu: g, dn: d }, K));
    n++;
  }
});
Deno.test("moe fused wide: several blocks per thread (dim 2048, expert width 512, q4 / q8)", () => {
  check({ fmt: "q4", sfmt: "q8", dim: 2048, ei: 64, sDim: 64, nExp: 3, K: 2, C: 2 }, moeFusedLayout("wide", 2));
});
Deno.test("moe fused wide: layout resolution and WGSL", () => {
  if (moeFusedLayout(undefined) !== null || moeFusedLayout("legacy") !== null) throw new Error("legacy is null");
  const w = moeFusedLayout("wide");
  if (w.gu.rows !== (MOEF_WIDE.gu.WG / MOEF_WIDE.gu.TPR) * MOEF_WIDE.gu.R || w.dn.TPR !== MOEF_WIDE.dn.TPR) throw new Error(JSON.stringify(w));
  const o = moeFusedLayout({ dn: { TPR: 2 } });
  if (o.dn.TPR !== 2 || o.dn.WG !== MOEF_WIDE.dn.WG || o.gu.TPR !== MOEF_WIDE.gu.TPR) throw new Error("partial override");
  for (const bad of [{ gu: { WG: 96 } }, { gu: { TPR: 512 } }, { dn: { R: 5 } }, { dn: { TPR: 2, R: 3 } }, { dn: { WG: 256, R: 4 } }, "fast"]) {
    let threw = false; try { moeFusedLayout(bad, 8); } catch { threw = true; }
    if (!threw) throw new Error(`accepted ${JSON.stringify(bad)}`);
  }
  const src = moeFusedWGSL({ K: 8, R: 1, layout: w, gu: [["q4", "q8"]], dn: [["q4", "q8"]] });
  for (const f of ["moe_route", "moe_gus_q4_q8", "moe_dnc_q4_q8"]) if (!src.includes(`fn ${f}(`)) throw new Error(`missing ${f}`);
  if (/undefined|NaN|\$\{/.test(src)) throw new Error("template hole in the generated WGSL");
  const bad = src.match(/[-+*] [\w.]+ \/ \d+u/);
  if (bad) throw new Error(`unparenthesized division in the WGSL: "${bad[0]}"`);
  // the legacy layout's WGSL is unchanged by the option
  if (moeFusedWGSL({ K: 8, R: 1, gu: [["q4", "q8"]], dn: [["q4", "q8"]] }) !== moeFusedWGSL({ K: 8, R: 1, layout: null, gu: [["q4", "q8"]], dn: [["q4", "q8"]] })) throw new Error("legacy changed");
});
