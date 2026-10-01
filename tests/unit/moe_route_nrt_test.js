// CPU check of moe_route (the merge-network top-K) and normRouterKernel (moe_nrt / dn_nba: RMSNorm + GEMV
// in one launch), engine/wgsl/moe.js. The kernel bodies run as JavaScript (wgslToJs, one generator per thread,
// workgroupBarrier() as a yield), in float64 with float32 storage. The GPU-side bit checks are tests/test_moe.js.
// No GPU.   deno test --no-check tests/unit/moe_route_nrt_test.js
import { routeKernel, normRouterKernel, moeFusedWGSL, wgslToJs } from "../../engine/wgsl/moe.js";

const f32u = new Float32Array(1), u32f = new Uint32Array(f32u.buffer);
const H = {
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3],
  V4: (a, b, c, d) => [a, b, c, d], mul4: (a, b) => a.map((v, i) => v * b[i]),
  bitcastF: (u) => { u32f[0] = u >>> 0; return f32u[0]; }, bitcastU: (f) => { f32u[0] = f; return u32f[0]; },
  inverseSqrt: (v) => 1 / Math.sqrt(v), f32: (v) => v,
  select: (f, t, c) => (c ? t : f), min: Math.min, max: Math.max, exp: Math.exp,
};
let seed = 9001;
const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);

// the kernel body as JavaScript: wgslToJs plus the few spellings these two kernels use that it leaves alone
function bodyJs(src, name) {
  const i = src.indexOf(`fn ${name}(`), j = src.indexOf("{\n", i), k = src.indexOf("\n}", j) + 1;
  if (i < 0) throw new Error(`${name} not in source`);
  return wgslToJs(src.slice(j + 2, k))
    .replace(/= ([^;=]+?) \/ (\d+);/g, "= Math.floor(($1) / $2);")   // u32 division by a constant
    .replace(/let d4 = n \/ 4;/, "let d4 = Math.floor(n / 4);")
    .replace(/let xw = x4 \* vec4<f32>\(([^;]*)\);/, "let xw = mul4(x4, V4($1));")
    .replace(/vec4<f32>\(/g, "V4(").replace(/bitcast<f32>\(/g, "bitcastF(").replace(/bitcast<u32>\(/g, "bitcastU(");
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

// moe_route on C columns of nExp logits (+ the shared-gate logit), column stride xs
function route(K, nExp, lg, xs, C, norm) {
  const KS = K + 1, sel = new Uint32Array(C * KS).fill(0xdead), w = new Float32Array(C * KS).fill(NaN);
  run(routeKernel(K), "moe_route", 256, { rt_l: lg, rt_sel: sel, rt_w: w, rt_s: { nExp, xs, norm } },
    { rt_k: () => new Uint32Array(1024 + K), rt_ix: () => new Uint32Array(2 * (1024 + K)),
      rt_v: () => new Float32Array(256), rt_ki: () => new Uint32Array(K), rt_kv: () => new Float32Array(K) }, C, 1);
  return { sel, w };
}
// reference: softmax over the nExp logits, top-K by (probability desc, index asc), renormalized when norm
function routeRef(K, l, norm) {
  const m = Math.max(...l), e = l.map((v) => Math.exp(v - m)), z = e.reduce((a, b) => a + b, 0);
  const top = l.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, K), tot = top.reduce((a, [, i]) => a + e[i] / z, 0);
  return top.map(([, i]) => [i, norm ? e[i] / z / tot : e[i] / z]);
}
function checkRoute(K, nExp, C, norm, make) {
  const KS = K + 1, xs = nExp + 3, lg = new Float32Array(C * xs).fill(NaN);
  for (let c = 0; c < C; c++) for (let i = 0; i <= nExp; i++) lg[c * xs + i] = make(c, i);
  const { sel, w } = route(K, nExp, lg, xs, C, norm);
  for (let c = 0; c < C; c++) {
    const ref = routeRef(K, Array.from(lg.subarray(c * xs, c * xs + nExp)), norm);
    for (let k = 0; k < K; k++) {
      const [id, wt] = ref[k], got = sel[c * KS + k], gw = w[c * KS + k];
      if (got !== id) throw new Error(`K ${K} nExp ${nExp} norm ${norm} column ${c} slot ${k}: expert ${got}, want ${id}`);
      if (!(Math.abs(gw - wt) <= 1e-6 * wt + 1e-12)) throw new Error(`K ${K} nExp ${nExp} norm ${norm} column ${c} slot ${k}: weight ${gw}, want ${wt}`);
    }
    const g = lg[c * xs + nExp];
    if (sel[c * KS + K] !== 0 || Math.abs(w[c * KS + K] - 1 / (1 + Math.exp(-g))) > 1e-7) throw new Error(`column ${c}: shared gate slot`);
  }
}

Deno.test("moe_route: top-K ids and weights match the full sort (random logits, ties, uneven groups)", () => {
  for (const norm of [1, 0]) {
    for (const [K, nExp] of [[8, 256], [3, 12], [3, 13], [8, 60]]) {
      checkRoute(K, nExp, 2, norm, () => (rnd() - 0.5) * 6);
      // ties inside and across groups: the lower index wins, as in moe_router's argmax rounds
      checkRoute(K, nExp, 2, norm, () => Math.floor(rnd() * 5) - 2);
    }
    // fewer experts than K groups can fill (nExp 12, K 8: groups of 2, the last two empty) and nExp == K
    checkRoute(8, 12, 1, norm, () => (rnd() - 0.5) * 6);
    checkRoute(4, 4, 1, norm, () => (rnd() - 0.5) * 6);
    // +0 and -0 tie (the sort keys map -0 to +0), the lower index first; and nExp at the 1024 limit, K 16
    checkRoute(8, 256, 2, norm, () => (rnd() < 0.5 ? 0 : -0));
    checkRoute(16, 1024, 1, norm, () => Math.floor(rnd() * 9) - 4);
    // every expert tied: ids 0 .. K - 1, equal weights
    checkRoute(8, 256, 1, norm, (c, i) => (i < 256 ? 0.25 : 1));
    // all the top experts in one group (the other groups' maxima are low; the threshold still admits them)
    checkRoute(8, 256, 1, norm, (c, i) => (i < 256 ? (i < 32 ? 5 + i * 0.01 : (rnd() - 0.5)) : 0));
  }
});

// normRouterKernel on C columns: x [C x xs], norm weight nw [dim], W [dOut x dim] (bf16: packed pairs), eps
function nrt({ dim, dOut, C, bf16, ROWS = 4, WG = 256 }) {
  const xs = dim + 8, ls = dOut + 3, xns = dim + 4, eps = 1e-6;
  const x = Float32Array.from({ length: C * xs }, () => rnd() - 0.5), nw = Float32Array.from({ length: dim }, () => 0.5 + rnd());
  const Wf = Float32Array.from({ length: dOut * dim }, () => (rnd() - 0.5) * 0.2);
  if (bf16) { const u = new Uint32Array(Wf.buffer); for (let i = 0; i < u.length; i++) u[i] &= 0xffff0000; }   // exactly BF16, as the engine requires
  const u = new Uint32Array(Wf.buffer);
  // the engine's layouts: f32 rows as vec4s; BF16 as vec2<u32> (two values per word, low half first)
  const W = bf16 ? Array.from({ length: dOut * dim / 4 }, (_, i) => ({ x: (u[4 * i] >>> 16) | (u[4 * i + 1] & 0xffff0000), y: (u[4 * i + 2] >>> 16) | (u[4 * i + 3] & 0xffff0000) }))
    : Array.from({ length: dOut * dim / 4 }, (_, i) => Array.from(Wf.subarray(4 * i, 4 * i + 4)));
  const xn = new Float32Array(C * xns).fill(NaN), lg = new Float32Array(C * ls).fill(NaN);
  const src = normRouterKernel({ ROWS, bf16, WG });
  run(src, "moe_nrt", WG, { cfg: { eps }, nrt_x: x, nrt_nw: nw, nrt_W: W, nrt_xn: xn, nrt_lg: lg, nrt_s: { dim, dOut, xs, ls, xns } },
    { nrt_red: () => new Float32Array((ROWS + 1) * WG) }, Math.ceil(dOut / ROWS), C);
  for (let c = 0; c < C; c++) {
    let ss = 0; for (let j = 0; j < dim; j++) ss += x[c * xs + j] ** 2;
    const inv = 1 / Math.sqrt(ss / dim + eps);
    for (let j = 0; j < xns; j++) {
      const got = xn[c * xns + j];
      if (j >= dim) { if (!Number.isNaN(got)) throw new Error(`xn[${c}][${j}] written past dim`); continue; }
      const ref = x[c * xs + j] * inv * nw[j];
      if (!(Math.abs(got - ref) <= 1e-5 * Math.abs(ref) + 1e-7)) throw new Error(`xn[${c}][${j}]: ${got}, want ${ref}`);
    }
    for (let r = 0; r < ls; r++) {
      const got = lg[c * ls + r];
      if (r >= dOut) { if (!Number.isNaN(got)) throw new Error(`logit[${c}][${r}] written past dOut`); continue; }
      let a = 0; for (let j = 0; j < dim; j++) a += Wf[r * dim + j] * x[c * xs + j] * nw[j] * inv;
      if (!(Math.abs(got - a) <= 1e-5 * (Math.abs(a) + 1e-2))) throw new Error(`logit[${c}][${r}] (bf16 ${bf16}): ${got}, want ${a}`);
    }
  }
}

Deno.test("moe_nrt / dn_nba: norm + GEMV equals rmsnorm then the GEMV (f32 and BF16 rows, dOut not a multiple of ROWS)", () => {
  nrt({ dim: 64, dOut: 13, C: 2, bf16: true });
  nrt({ dim: 64, dOut: 13, C: 2, bf16: false });
  nrt({ dim: 520, dOut: 9, C: 1, bf16: true, WG: 32 });   // several vec4s per thread, the tree at another width
  nrt({ dim: 36, dOut: 8, C: 3, bf16: false, ROWS: 2, WG: 16 });
});

Deno.test("normRouterKernel: entry names, bindings and the NRT struct declared once", () => {
  const nba = normRouterKernel({ ROWS: 4, bf16: false, name: "dn_nba", P: "nba", struct: false });
  if (!nba.includes("fn dn_nba(") || nba.includes("struct NRT") || !nba.includes("var<storage, read> nba_W: array<vec4<f32>>")) throw new Error("dn_nba source");
  if (!normRouterKernel().includes("nrt_W: array<vec2<u32>>")) throw new Error("BF16 router binding");
  const on = moeFusedWGSL({ K: 8, nrt: { ROWS: 4, bf16: true } }), off = moeFusedWGSL({ K: 8 });
  if (!on.includes("fn moe_nrt(") || (on.match(/struct NRT/g) || []).length !== 1 || off.includes("moe_nrt")) throw new Error("moeFusedWGSL nrt");
});
