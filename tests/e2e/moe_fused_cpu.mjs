// The fused MoE kernels (engine/wgsl/moe.js moeFusedWGSL) run on the CPU through wgsl_reflect's WGSL
// interpreter (detectRaces: honours workgroupBarrier, and reports shared-memory races). No GPU.
//   npm i --prefix /tmp/wr wgsl_reflect
//   WGSL_REFLECT=/tmp/wr/node_modules/wgsl_reflect/wgsl_reflect.module.js node tests/e2e/moe_fused_cpu.mjs
// Checks, on small random Q4_0 / Q8_0 experts:
//   1. moe_route gives the same ids and weights as moe_router (rank top-K == K argmax rounds), and
//      writes sigmoid(shared-gate logit) to slot K
//   2. route -> gus -> dnc equals the unfused chain moe_router -> moe_gu -> moe_dn (routed slots),
//      moe_gu / moe_dn over the shared expert as a 1-expert stack, then moe_combine: the same values
//      (per-slot terms, reductions and combine order are the same code), for every format pair and R
//   3. an N-column launch gives each column the same values as that column launched alone
//   4. both are close to a float64 reference, and the interpreter reports no races
// Checks 3 and 4 run again with the wide fused layout (moeFusedLayout: engine/wgsl/moe.js gusKernelWide /
// dncKernelWide), whose sums are in another order (check 2 does not apply to it).
// The interpreter's float rounding is not a GPU's, so "same" here means the same expression tree;
// the GPU-side bit checks are tests/test_moe.js (speculative == plain, llama.cpp text).
import { moeWGSL, moeFusedWGSL, moeKernelConfig, moeFusedLayout } from "../../engine/wgsl/moe.js";
import { f16ToF32, f32ToF16 } from "../../engine/gguf.js";
const WR = process.env.WGSL_REFLECT;
if (!WR) { console.log("set WGSL_REFLECT to wgsl_reflect.module.js (see the header); skipping"); process.exit(0); }
const { detectRaces } = await import(WR);

const HEAD = `struct Config { dim: u32 };\n@group(0) @binding(0) var<uniform> cfg: Config;\n`;
let seed = 777; const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
const quant = (fmt, rows, dIn) => {
  const nb = dIn / 32, W = fmt === "q4" ? 4 : 8;
  const qs = new Uint32Array(rows * nb * W), sch = new Uint16Array(rows * nb + (rows * nb) % 2);
  for (let i = 0; i < qs.length; i++) qs[i] = (rnd() * 4294967296) >>> 0;
  for (let i = 0; i < rows * nb; i++) sch[i] = f32ToF16((rnd() - 0.5) * 0.05);
  const sc = new Uint32Array(sch.buffer);
  const deq = (r, j) => {
    const b = Math.floor(j / 32), k = j % 32, s = f16ToF32(sch[r * nb + b]);
    if (fmt === "q4") { const qt = (k % 16) >> 2, i = k % 4, w = qs[(r * nb + b) * 4 + qt]; return s * (((w >>> (8 * i + (k >= 16 ? 4 : 0))) & 15) - 8); }
    const w = qs[(r * nb + b) * 8 + (k >> 2)]; let q = (w >>> (8 * (k % 4))) & 255; if (q > 127) q -= 256; return s * q;
  };
  return { qs, sc, deq };
};
const out = [], res = [];
const check = (n, ok, d = "") => { res.push(ok); out.push(`${ok ? "PASS" : "FAIL"} ${n}${d ? "  " + d : ""}`); console.log(out[out.length - 1]); };
let races = 0;
const run = (code, kernel, grid, bufs) => {
  const g = {}; bufs.forEach((b, i) => { g[i] = b; });
  const r = detectRaces(code, kernel, grid, { 0: { 0: new Uint32Array(4) }, 1: g }, { stepBudget: 1e9 });
  if (r.races.length || r.errors.length) { races++; console.log(kernel, "races", r.races.slice(0, 2), r.errors.slice(0, 2)); }
};
const same = (a, b) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]));

const dim = 64, ei = 32, nExp = 12, K = +(process.env.MOE_K || 3), KS = K + 1, C = 2;   // MOE_K=8 for the real top-k (slower)
const WIDE = [null, moeFusedLayout("wide", K), moeFusedLayout({ gu: { WG: 16, TPR: 2, R: 2 }, dn: { WG: 16, TPR: 4, R: 3 } }, K)];
for (const [gf, sgf, df, sdf, R, sDim, lay] of [["q4", "q8", "q4", "q8", 2, 32, 0], ["q8", "q8", "q8", "q8", 4, 32, 0], ["q4", "q4", "q8", "q4", 1, 64, 0],
  ["q4", "q8", "q4", "q8", 1, 32, 1], ["q8", "q4", "q8", "q4", 1, 64, 2]]) {
  const hs = Math.max(ei, sDim), layout = WIDE[lay];
  // the unfused chain in the legacy layout: the legacy fused kernels have its per-slot terms and reductions
  const code = HEAD + moeWGSL(moeKernelConfig("legacy", { dim, inter: ei })) + moeFusedWGSL({ K, R, gu: [[gf, sgf]], dn: [[df, sdf]], layout });
  const Wg = quant(gf, nExp * ei, dim), Wu = quant(gf, nExp * ei, dim), Wd = quant(df, nExp * dim, ei);
  const Sg = quant(sgf, sDim, dim), Su = quant(sgf, sDim, dim), Sd = quant(sdf, dim, sDim);
  const logits = Float32Array.from({ length: C * (nExp + 1) }, () => (rnd() - 0.5) * 4);
  logits[3] = logits[5];   // a tie: the lower index must win
  const x = Float32Array.from({ length: C * dim }, () => rnd() - 0.5);
  const x0 = Float32Array.from({ length: C * dim }, () => rnd() - 0.5);
  // shared gate/up packed as the engine packs them: [gate qs | up qs | gate sc | up sc]
  const pk = new Uint32Array(Math.ceil((Sg.qs.length + Su.qs.length + Sg.sc.length + Su.sc.length) / 4) * 4);   // whole vec4s, as the engine rounds it
  pk.set(Sg.qs, 0); pk.set(Su.qs, Sg.qs.length); pk.set(Sg.sc, Sg.qs.length + Su.qs.length); pk.set(Su.sc, Sg.qs.length + Su.qs.length + Sg.sc.length);
  const oUq = Sg.qs.length, oGs = oUq + Su.qs.length, oUs = oGs + Sg.sc.length;
  const U = (xs, dOut, dIn, ys, p) => new Uint32Array([dOut, dIn, sDim, nExp, xs, ys, 1, 1, p ? oUq : 0, p ? oGs : 0, p ? oUs : 0, 0]);

  const fused = (c0, n) => {
    const lg = logits.slice(c0 * (nExp + 1), (c0 + n) * (nExp + 1)), sel = new Uint32Array(n * KS), w = new Float32Array(n * KS);
    const h = new Float32Array(n * KS * hs), xo = x0.slice(c0 * dim, (c0 + n) * dim);
    run(code, "moe_route", [n, 1, 1], [lg, sel, w, U(nExp + 1, 0, 0, 0)]);
    run(code, `moe_gus_${gf}_${sgf}`, [Math.ceil(hs / (layout ? layout.gu.rows : 4)), n * KS, 1], [Wg.qs, Wg.sc, Wu.qs, Wu.sc, x.slice(c0 * dim, (c0 + n) * dim), h, sel, pk, U(dim, ei, dim, hs, true)]);
    run(code, `moe_dnc_${df}_${sdf}`, [Math.ceil(dim / (layout ? layout.dn.rows : R)), n, 1], [Wd.qs, Wd.sc, h, xo, sel, w, Sd.qs, Sd.sc, U(dim, dim, ei, hs)]);
    return { sel, w, xo };
  };
  const all = fused(0, C);
  let colsOk = true;
  for (let c = 0; c < C; c++) { const one = fused(c, 1); if (!same(one.xo, all.xo.subarray(c * dim, (c + 1) * dim)) || !same(one.sel, all.sel.subarray(c * KS, (c + 1) * KS))) colsOk = false; }
  const lname = layout ? `wide ${JSON.stringify(layout)}` : `R ${R}`;
  check(`${gf}/${sgf} gate-up, ${df}/${sdf} down, ${lname}, sDim ${sDim}: ${C}-column launch == one column at a time`, colsOk);

  // the unfused chain on the same inputs
  const MOE = (a) => new Uint32Array(a);
  const lgR = new Float32Array(C * nExp); for (let c = 0; c < C; c++) lgR.set(logits.subarray(c * (nExp + 1), c * (nExp + 1) + nExp), c * nExp);
  const selR = new Uint32Array(C * K), wR = new Float32Array(C * K);
  run(code, "moe_router", [C, 1, 1], [lgR, selR, wR, MOE([0, 0, K, nExp, nExp, 0, 1, 0])]);
  let routeOk = true;
  for (let c = 0; c < C; c++) for (let k = 0; k < K; k++)
    if (selR[c * K + k] !== all.sel[c * KS + k] || !Object.is(wR[c * K + k], all.w[c * KS + k])) routeOk = false;
  for (let c = 0; c < C; c++) if (!Object.is(all.w[c * KS + K], Math.fround(1 / (1 + Math.exp(-logits[c * (nExp + 1) + nExp]))))) {
    // the interpreter's exp may round differently from Math.exp: compare loosely
    if (Math.abs(all.w[c * KS + K] - 1 / (1 + Math.exp(-logits[c * (nExp + 1) + nExp]))) > 1e-6) routeOk = false;
  }
  check(`moe_route == moe_router (ids, weights), shared gate in slot K`, routeOk, `sel ${Array.from(all.sel.subarray(0, KS))}`);
  const hR = new Float32Array(C * K * ei), yR = new Float32Array(C * K * dim);
  run(code, `moe_gu_${gf}`, [Math.ceil(ei / 4), C * K, 1], [Wg.qs, Wg.sc, Wu.qs, Wu.sc, x, hR, selR, MOE([ei, dim, K, nExp, dim, ei, 0, 0])]);
  run(code, `moe_dn_${df}`, [Math.ceil(dim / 4), C * K, 1], [Wd.qs, Wd.sc, hR, yR, selR, MOE([dim, ei, K, nExp, ei, dim, 0, 0])]);
  const sel0 = new Uint32Array(C), hS = new Float32Array(C * sDim), yS = new Float32Array(C * dim);
  run(code, `moe_gu_${sgf}`, [Math.ceil(sDim / 4), C, 1], [Sg.qs, Sg.sc, Su.qs, Su.sc, x, hS, sel0, MOE([sDim, dim, 1, 1, dim, sDim, 0, 0])]);
  run(code, `moe_dn_${sdf}`, [Math.ceil(dim / 4), C, 1], [Sd.qs, Sd.sc, hS, yS, sel0, MOE([dim, sDim, 1, 1, sDim, dim, 0, 0])]);
  const sg = new Float32Array(C); for (let c = 0; c < C; c++) sg[c] = logits[c * (nExp + 1) + nExp];
  const xu = x0.slice();
  run(code, "moe_combine", [Math.ceil(dim / 64), C, 1], [xu, yR, wR, yS, sg, MOE([dim, 0, K, dim, dim, dim, 1, 1])]);
  let maxD = 0; for (let i = 0; i < xu.length; i++) maxD = Math.max(maxD, Math.abs(xu[i] - all.xo[i]));
  if (!layout) check(`fused chain == unfused chain (router, gu, dn, shared as a 1-expert stack, combine)`, same(xu, all.xo), `max |diff| ${maxD.toExponential(2)}`);

  // float64 reference
  let maxErr = 0, maxRef = 0;
  for (let c = 0; c < C; c++) {
    const l = Array.from(logits.subarray(c * (nExp + 1), c * (nExp + 1) + nExp)), m = Math.max(...l);
    const pr = l.map((v) => Math.exp(v - m)), z = pr.reduce((a, b) => a + b, 0);
    const order = pr.map((v, i) => [v / z, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, K), tot = order.reduce((a, b) => a + b[0], 0);
    const o = new Float64Array(dim);
    const ffn = (G, Uw, D, base, dOut, wt) => {
      const hh = new Float64Array(dOut);
      for (let r = 0; r < dOut; r++) { let g = 0, u = 0; for (let j = 0; j < dim; j++) { g += G.deq(base * dOut + r, j) * x[c * dim + j]; u += Uw.deq(base * dOut + r, j) * x[c * dim + j]; } hh[r] = g / (1 + Math.exp(-g)) * u; }
      for (let r = 0; r < dim; r++) { let y = 0; for (let j = 0; j < dOut; j++) y += D.deq(base * dim + r, j) * hh[j]; o[r] += wt * y; }
    };
    for (const [pw, e] of order) ffn(Wg, Wu, Wd, e, ei, pw / tot);
    ffn(Sg, Su, Sd, 0, sDim, 1 / (1 + Math.exp(-logits[c * (nExp + 1) + nExp])));
    for (let i = 0; i < dim; i++) { const ref = x0[c * dim + i] + o[i]; maxErr = Math.max(maxErr, Math.abs(ref - all.xo[c * dim + i])); maxRef = Math.max(maxRef, Math.abs(ref)); }
  }
  check(`fused chain vs float64 reference`, maxErr / maxRef < 1e-4, `max err ${maxErr.toExponential(2)} (${(maxErr / maxRef).toExponential(1)} of max)`);
}
check("no shared-memory races or interpreter errors", races === 0);
console.log(res.every(Boolean) ? "MOE FUSED CPU PASS" : "MOE FUSED CPU FAIL");
process.exit(res.every(Boolean) ? 0 : 1);
