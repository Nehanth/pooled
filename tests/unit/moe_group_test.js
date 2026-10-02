// CPU check of the expert-grouped MoE prefill kernels (engine/wgsl/moe_group.js) against the fused per-pair
// kernels they replace (engine/wgsl/moe.js moe_gus / moe_dnc). Both are generated with JavaScript op
// spellings and run here with one generator per thread (workgroupBarrier() as a yield), in float64.
// Float64 cannot show float32 rounding, but it does show the order of operations: the grouped path must
// give exactly the same numbers (===) as the per-pair path, which holds only if every pair's terms,
// accumulation order, reduction tree and epilogue are the same. It also checks the sort (every pair once,
// grouped by expert, stable, chunk sizes, indirect args) and that nothing outside the outputs is written.
// No GPU.   deno test --allow-read tests/unit/moe_group_test.js
import { FOPS_JS, gusKernel, dncKernel, wgslToJs, moeFusedWGSL } from "../../engine/wgsl/moe.js";
import { moeGroupWGSL, moeGroupSizes, gusGroupKernel, dnGroupKernel, combKernel, tiledGroupWGSL, tileRows } from "../../engine/wgsl/moe_group.js";
import { f16ToF32, f32ToF16 } from "../../engine/gguf.js";

const H = {
  q4lo: (w) => [0, 8, 16, 24].map((s) => ((w >>> s) & 15) - 8),
  q4hi: (w) => [4, 12, 20, 28].map((s) => ((w >>> s) & 15) - 8),
  i8x4: (w) => [0, 8, 16, 24].map((s) => (((w >>> s) & 255) << 24) >> 24),
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3],
  unpack2x16float: (w) => [f16ToF32(w & 0xffff), f16ToF32(w >>> 16)],
  select: (f, t, c) => (c ? t : f), min: Math.min, max: Math.max, exp: Math.exp,
  countOneBits: (x) => { let n = 0; for (x >>>= 0; x; x &= x - 1) n++; return n; },
};
let seed = 12345;
const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);

// the body of kernel `name` in WGSL source `src`, as a JavaScript generator body
function bodyJs(src, name) {
  const i = src.indexOf(`fn ${name}(`), j = src.indexOf("{\n", i), k = src.indexOf("\n}", j) + 1;
  if (i < 0) throw new Error(`${name} not in source`);
  return wgslToJs(src.slice(j + 2, k))
    .replace(/let (\w+) = workgroupUniformLoad\(&(\w+)\);/g, "yield; let $1 = $2;")
    .replace(/var (\w+): array<u32, (\d+)>;/g, "let $1 = new Array($2).fill(0);")
    // workgroup atomics (moe_gsort): the threads run one at a time between barriers, so plain updates are atomic
    .replace(/atomicAdd\(&(\w+\[[^\]]+\]), ([^)]+)\)/g, "(($1 += $2) - $2)").replace(/atomicOr\(&(\w+\[[^\]]+\]), ([^;]+)\);/g, "$1 = ($1 | $2) >>> 0;")
    .replace(/atomicStore\(&(\w+\[[^\]]+\]), ([^)]+)\)/g, "($1 = $2)").replace(/atomicLoad\(&(\w+\[[^\]]+\])\)/g, "($1)");
}
// Run kernel `name` over a (gx, gy) grid of WG threads. bufs: binding name -> array / object (shared by all
// workgroups); wgv: workgroup variable name -> () => fresh value (per workgroup).
function run(src, name, WG, bufs, wgv, gx, gy) {
  const hn = Object.keys(H), bn = Object.keys(bufs), wn = Object.keys(wgv);
  const make = new Function(...hn, ...bn, ...wn, `return function* (wg, lid, gid) {${bodyJs(src, name)}};`);
  for (let y = 0; y < gy; y++) for (let x = 0; x < gx; x++) {
    const fn = make(...hn.map((n) => H[n]), ...bn.map((n) => bufs[n]), ...wn.map((n) => wgv[n]()));
    const th = Array.from({ length: WG }, (_, t) => fn({ x, y }, { x: t }, { x: x * WG + t, y }));
    for (;;) {
      const done = th.map((g) => g.next().done);
      if (done.every(Boolean)) break;
      if (done.some(Boolean)) throw new Error(`${name}: threads disagree at a barrier`);
    }
  }
}
// [rows][dIn] quantized in the engine's layout (u32 words, f16 scales packed two per u32)
function quant(fmt, rows, dIn) {
  const nb = dIn / 32, W = fmt === "q4" ? 4 : 8;
  const qs = new Uint32Array(rows * nb * W), sch = new Uint16Array(rows * nb + ((rows * nb) % 2));
  for (let i = 0; i < qs.length; i++) qs[i] = (rnd() * 4294967296) >>> 0;
  for (let i = 0; i < rows * nb; i++) sch[i] = f32ToF16((rnd() - 0.5) * 0.05);
  return { qs, sc: new Uint32Array(sch.buffer) };
}
const vecF = (f) => Array.from({ length: f.length / 4 }, (_, i) => Array.from(f.subarray(i * 4, i * 4 + 4)));
// the shared gate/up packed as engine/qwen35.js packGU does: gate qs | up qs | gate scales | up scales (words)
function packShared(g, u) {
  const oUq = g.qs.length, oGs = oUq + u.qs.length, oUs = oGs + g.sc.length;
  const b = new Uint32Array(oUs + u.sc.length);
  b.set(g.qs, 0); b.set(u.qs, oUq); b.set(g.sc, oGs); b.set(u.sc, oUs);
  return { buf: b, oUq, oGs, oUs };
}

// routing for C columns: K distinct experts per column (slot K = the shared expert, sel 0 as moe_route writes)
function route(C, K, nExp, skew = 0) {
  const KS = K + 1, sel = new Uint32Array(C * KS), w = new Float64Array(C * KS);
  for (let c = 0; c < C; c++) {
    const pick = new Set();
    while (pick.size < K) pick.add(skew && rnd() < skew ? Math.floor(rnd() * Math.min(4, nExp)) : Math.floor(rnd() * nExp));
    [...pick].forEach((e, k) => { sel[c * KS + k] = e; w[c * KS + k] = rnd(); });
    sel[c * KS + K] = 0; w[c * KS + K] = rnd() * 1e-3;   // small, so it does not absorb the routed sum's rounding
  }
  return { sel, w };
}

// JS reference of moe_gsort's output: pairs grouped by expert (shared = nExp), ascending, stable
function sortRef(sel, K, nExp, UC) {
  const KS = K + 1, by = Array.from({ length: nExp + 1 }, () => []);
  for (let i = 0; i < sel.length; i++) by[i % KS === K ? nExp : Math.min(sel[i], nExp - 1)].push(i);
  const chunks = [], list = [];
  for (let e = 0; e <= nExp; e++) {
    for (let k = 0; k < by[e].length; k += UC) chunks.push({ start: list.length + k, n: Math.min(UC, by[e].length - k), e });
    list.push(...by[e]);
  }
  return { chunks, list, uniq: by.slice(0, nExp).filter((a) => a.length).length };
}
function runSort(sel, { U, K, nExp, UC, gx = 7, dx = 9 }) {
  const { words, CO, pairs } = moeGroupSizes({ U, K, nExp, UC });
  const src = moeGroupWGSL({ K, UC, U, nExp }, FOPS_JS);
  const grp = new Array(words).fill(-1), ind = new Array(8).fill(-1);
  run(src, "moe_gsort", 256, { gso_sel: sel, gso_grp: grp, gso_ind: ind, gso_s: { n: pairs, nExp, gx, dx } },
    { gso_h: () => new Array(Math.ceil((nExp + 1) / 256) * 256).fill(-7), gso_m: () => new Array(Math.ceil((nExp + 1) / 256) * 256).fill(-7), gso_b: () => new Array(Math.ceil((nExp + 1) / 256) * 256).fill(-7),
      gso_r: () => new Array(Math.ceil((nExp + 1) / 256) * 256).fill(-7), gso_sc: () => new Array(256).fill(-7), gso_sp: () => new Array(256).fill(-7), gso_su: () => new Array(256).fill(-7) }, 1, 1);
  return { grp, ind, CO, pairs };
}
function checkSort(sel, cfg) {
  const { grp, ind, CO, pairs } = runSort(sel, cfg), ref = sortRef(sel, cfg.K, cfg.nExp, cfg.UC);
  const exp = [cfg.gx ?? 7, ref.chunks.length, 1, cfg.dx ?? 9, ref.chunks.length, 1, ref.uniq, pairs];
  if (ind.join() !== exp.join()) throw new Error(`indirect args ${ind} != ${exp}`);
  ref.chunks.forEach((c, i) => { const g = grp.slice(4 * i, 4 * i + 4).join(), r = [c.start, c.n, c.e, 0].join(); if (g !== r) throw new Error(`chunk ${i}: ${g} != ${r}`); });
  if (grp.slice(CO, CO + pairs).join() !== ref.list.join()) throw new Error("pair list differs from the stable sort");
  for (let i = 4 * ref.chunks.length; i < CO; i++) if (grp[i] !== -1) throw new Error(`chunk area written past the last chunk (${i})`);
  return ref;
}

// One MoE FFN for C columns, per-pair (moe_gus + moe_dnc) and grouped (moe_gsort + moe_gusg + moe_dng +
// moe_combw); returns both { h, x } results.
// R: rows per moe_dnc workgroup (per pair), RG: rows per moe_dng workgroup (grouped); a row's value must not depend on either
function ffnBoth({ fmt, sfmt, dim, ei, sDim, nExp, K, C, R = 1, RG = R, UC, skew = 0, tiled = false }) {
  const KS = K + 1, hs = Math.max(ei, sDim), xs = dim + 8;   // padded column stride: the gap must stay untouched
  const Wg = quant(fmt, nExp * ei, dim), Wu = quant(fmt, nExp * ei, dim), Wd = quant(fmt, nExp * dim, ei);
  const Sg = quant(sfmt, sDim, dim), Su = quant(sfmt, sDim, dim), Sd = quant(sfmt, dim, sDim), pk = packShared(Sg, Su);
  const xn = Float64Array.from({ length: C * xs }, () => rnd() - 0.5);
  // a small residual, so a change in the combine's summation order is not absorbed by the final add
  const x0 = Float64Array.from({ length: C * xs }, () => (rnd() - 0.5) * 1e-6);
  const { sel, w } = route(C, K, nExp, skew);
  const uG = { dOut: ei, dIn: dim, sDim, nExp, xs, ys: hs, norm: 1, shOff: 1, oUq: pk.oUq, oGs: pk.oGs, oUs: pk.oUs, pad: 0 };
  const uD = { dOut: dim, dIn: ei, sDim, nExp, xs, ys: hs, norm: 1, shOff: 1, oUq: 0, oGs: 0, oUs: 0, pad: 0 };
  const gx = Math.ceil(hs / 4), dx = Math.ceil(dim / R), dxG = Math.ceil(dim / RG);
  // per pair
  const srcG = gusKernel(fmt, sfmt, K, 256, FOPS_JS), srcD = dncKernel(fmt, sfmt, K, R, 64, FOPS_JS);
  const P = `gs${fmt}${sfmt}`, Q = `dc${fmt}${sfmt}`;
  const hA = new Array(C * KS * hs).fill(NaN), xA = Array.from(x0);
  run(srcG, `moe_gus_${fmt}_${sfmt}`, 256, { [`${P}_gq`]: Wg.qs, [`${P}_gs`]: Wg.sc, [`${P}_uq`]: Wu.qs, [`${P}_us`]: Wu.sc, [`${P}_x`]: vecF(xn),
    [`${P}_h`]: hA, [`${P}_sel`]: sel, [`${P}_sh`]: pk.buf, [`${P}_s`]: uG }, { [`${P}_red`]: () => new Array(8 * 256).fill(NaN) }, gx, C * KS);
  const hIn = Float64Array.from(hA, (v) => (Number.isNaN(v) ? 0 : v));
  run(srcD, `moe_dnc_${fmt}_${sfmt}`, 64, { [`${Q}_q`]: Wd.qs, [`${Q}_sc`]: Wd.sc, [`${Q}_h`]: vecF(hIn), [`${Q}_x`]: xA, [`${Q}_sel`]: sel, [`${Q}_w`]: w,
    [`${Q}_sq`]: Sd.qs, [`${Q}_ss`]: Sd.sc, [`${Q}_s`]: uD }, { [`${Q}_red`]: () => new Array(KS * R * 64).fill(NaN) }, dx, C);
  // grouped
  if (tiled) {
    const gxT = Math.ceil(hs / tileRows("gu")), dxT = Math.ceil(dim / tileRows("dn"));
    const { grp, ind } = runSort(sel, { U: C, K, nExp, UC, gx: gxT, dx: dxT });
    const nCh = ind[1], src = tiledGroupWGSL({ K, UC, U: C, nExp, gu: [[fmt, sfmt]], dn: [[fmt, sfmt]] }, FOPS_JS);
    const G = `tg${fmt}${sfmt}`, E = `td${fmt}${sfmt}`, XT = Math.max(UC * 64, 512);
    const wg = (P) => ({ [`${P}_xt`]: () => new Array(4 * XT).fill(NaN), [`${P}_xo`]: () => new Array(UC).fill(NaN),
      [`${P}_cs`]: () => new Array(UC).fill(NaN), [`${P}_n`]: () => NaN });
    const hB = new Array(C * KS * hs).fill(NaN), yB = new Array(C * KS * dim).fill(NaN), xB = Array.from(x0);
    // the kernels copy vec4s into the workgroup tile and later write floats into it: hand out copies, as a GPU load does
    const vecC = (f) => new Proxy(vecF(f), { get: (a, k) => (Array.isArray(a[k]) ? a[k].slice() : a[k]) });
    run(src, `moe_gusg_${fmt}_${sfmt}`, 256, { [`${G}_gq`]: Wg.qs, [`${G}_gs`]: Wg.sc, [`${G}_uq`]: Wu.qs, [`${G}_us`]: Wu.sc, [`${G}_x`]: vecC(xn),
      [`${G}_h`]: hB, [`${G}_grp`]: grp, [`${G}_sh`]: pk.buf, [`${G}_s`]: uG }, wg(G), gxT, nCh);
    const hInB = Float64Array.from(hB, (v) => (Number.isNaN(v) ? 0 : v));
    run(src, `moe_dng_${fmt}_${sfmt}`, 256, { [`${E}_q`]: Wd.qs, [`${E}_sc`]: Wd.sc, [`${E}_x`]: vecC(hInB), [`${E}_y`]: yB, [`${E}_grp`]: grp,
      [`${E}_sq`]: Sd.qs, [`${E}_ss`]: Sd.sc, [`${E}_s`]: uD }, wg(E), dxT, nCh);
    run(src, "moe_combw", 64, { gcb_x: xB, gcb_y: yB, gcb_w: w, gcb_s: uD }, {}, Math.ceil(dim / 64), C);
    return { hA, hB, xA, xB, x0, KS, hs, xs, nCh, sel };
  }
  const { grp, ind, CO } = runSort(sel, { U: C, K, nExp, UC, gx, dx: dxG });
  const nCh = ind[1];
  const srcGG = gusGroupKernel(fmt, sfmt, K, UC, CO, 256, FOPS_JS), srcDG = dnGroupKernel(fmt, sfmt, K, RG, UC, CO, 64, FOPS_JS), srcC = combKernel(K);
  const G = `gg${fmt}${sfmt}`, E = `dg${fmt}${sfmt}`;
  const hB = new Array(C * KS * hs).fill(NaN), yB = new Array(C * KS * dim).fill(NaN), xB = Array.from(x0);
  const PPg = srcGG.match(new RegExp(`${G}_red: array<f32, (\\d+)>`))[1];
  run(srcGG, `moe_gusg_${fmt}_${sfmt}`, 256, { [`${G}_gq`]: Wg.qs, [`${G}_gs`]: Wg.sc, [`${G}_uq`]: Wu.qs, [`${G}_us`]: Wu.sc, [`${G}_x`]: vecF(xn),
    [`${G}_h`]: hB, [`${G}_grp`]: grp, [`${G}_sh`]: pk.buf, [`${G}_s`]: uG }, { [`${G}_red`]: () => new Array(+PPg).fill(NaN), [`${G}_n`]: () => NaN }, gx, nCh);
  const hInB = Float64Array.from(hB, (v) => (Number.isNaN(v) ? 0 : v));
  run(srcDG, `moe_dng_${fmt}_${sfmt}`, 64, { [`${E}_q`]: Wd.qs, [`${E}_sc`]: Wd.sc, [`${E}_h`]: vecF(hInB), [`${E}_y`]: yB, [`${E}_grp`]: grp,
    [`${E}_sq`]: Sd.qs, [`${E}_ss`]: Sd.sc, [`${E}_s`]: uD }, { [`${E}_red`]: () => new Array(UC * RG * 64).fill(NaN), [`${E}_n`]: () => NaN }, dxG, nCh);
  run(srcC, "moe_combw", 64, { gcb_x: xB, gcb_y: yB, gcb_w: w, gcb_s: uD }, {}, Math.ceil(dim / 64), C);
  return { hA, hB, xA, xB, x0, KS, hs, xs, nCh, sel };
}
function checkFfn(cfg) {
  const { hA, hB, xA, xB, x0, KS, hs, xs, nCh } = ffnBoth(cfg), { C, dim, ei, sDim, K } = cfg;
  for (let cs = 0; cs < C * KS; cs++) {
    const rows = cs % KS === K ? sDim : ei;
    for (let r = 0; r < hs; r++) {
      const a = hA[cs * hs + r], b = hB[cs * hs + r];
      if (r < rows && Number.isNaN(a)) throw new Error(`per-pair h[${cs}][${r}] not written`);
      if (!(Object.is(a, b))) throw new Error(`h[${cs}][${r}]: grouped ${b} != per-pair ${a}`);
    }
  }
  for (let c = 0; c < C; c++) for (let i = 0; i < xs; i++) {
    const a = xA[c * xs + i], b = xB[c * xs + i];
    if (!Object.is(a, b)) throw new Error(`x[${c}][${i}]: grouped ${b} != per-pair ${a}`);
    if (i >= dim && a !== x0[c * xs + i]) throw new Error(`x gap [${c}][${i}] written`);
    if (i < dim && a === x0[c * xs + i]) throw new Error(`x[${c}][${i}] unchanged`);
  }
  return nCh;
}

Deno.test("moe group: sort (stable, chunked, indirect args), small and over one 1024-pair tile", () => {
  for (const [U, K, nExp, UC] of [[3, 2, 5, 2], [6, 3, 7, 4], [8, 8, 256, 8], [300, 8, 256, 8], [130, 8, 256, 16], [40, 4, 511, 1]]) {
    const { sel } = route(U, K, nExp, U > 100 ? 0.3 : 0);
    const ref = checkSort(sel, { U, K, nExp, UC });
    if (U === 300 && ref.chunks.length >= U * (K + 1) / 2) throw new Error("skewed 300-token ubatch should share experts");
  }
});

Deno.test("moe group: grouped FFN === per-pair fused FFN (q4 routed / q8 shared, R 1 vs 4, UC 4)", () => {
  const n = checkFfn({ fmt: "q4", sfmt: "q8", dim: 64, ei: 32, sDim: 64, nExp: 5, K: 3, C: 6, R: 1, RG: 4, UC: 4, skew: 0.5 });
  if (n >= 6 * 4) throw new Error(`expected shared chunks, got ${n}`);
});
Deno.test("moe group: grouped FFN === per-pair fused FFN (q8 / q4, R 2 vs 4, UC 2, tails: sDim < ei)", () => {
  checkFfn({ fmt: "q8", sfmt: "q4", dim: 96, ei: 64, sDim: 32, nExp: 4, K: 2, C: 5, R: 2, RG: 4, UC: 2 });
});
Deno.test("moe group: grouped FFN === per-pair fused FFN (q4 / q4, R 4 vs 2, UC 8, one chunk holds all)", () => {
  checkFfn({ fmt: "q4", sfmt: "q4", dim: 64, ei: 32, sDim: 32, nExp: 2, K: 1, C: 9, R: 4, RG: 2, UC: 8 });
});

Deno.test("moe group: WGSL generation at the Qwen3.6-35B-A3B shape", () => {
  const src = moeGroupWGSL({ K: 8, R: 1, UC: 8, U: 256, nExp: 256, gu: [["q4", "q8"]], dn: [["q4", "q8"]] });
  for (const f of ["moe_gsort", "moe_gusg_q4_q8", "moe_dng_q4_q8", "moe_combw"]) if (!src.includes(`fn ${f}(`)) throw new Error(`missing ${f}`);
  if (/undefined|NaN|\$\{/.test(src)) throw new Error("template hole in the generated WGSL");
  // the CPU runs use JavaScript spellings (Math.floor((a) / (b))), which cannot show a WGSL precedence slip such as
  // "c + 7u / 8u": no u32 literal division may follow an additive / multiplicative operator without parentheses
  for (const UC of [1, 2, 4, 8, 16]) {
    const w = moeGroupWGSL({ K: 8, R: UC <= 8 ? 4 : 2, UC, U: 64, nExp: 256, gu: [["q4", "q8"], ["q8", "q4"]], dn: [["q4", "q8"], ["q8", "q4"]] });
    const bad = w.match(/[-+*] [\w.]+ \/ \d+u/);
    if (bad) throw new Error(`unparenthesized division in the WGSL: "${bad[0]}"`);
  }
  // the gate/up reduction scratch: one pair of 8 x 256 floats at a time (8 KB), within the 16 KB default
  if (!src.includes("_red: array<f32, 2048>")) throw new Error("gusg scratch size changed");
  const fused = moeFusedWGSL({ K: 8, R: 1, gu: [["q4", "q8"]], dn: [["q4", "q8"]] });
  if (!fused.includes("struct MOEF")) throw new Error("grouped kernels need the fused module's MOEF");
  for (const bad of [{ UC: 3 }, { R: 3 }]) { let threw = false; try { moeGroupWGSL({ K: 8, U: 32, nExp: 256, ...bad }); } catch { threw = true; } if (!threw) throw new Error(`accepted ${JSON.stringify(bad)}`); }
});
Deno.test("moe group: grouped FFN === per-pair fused FFN (several blocks per thread: dim 4096, expert width 2048)", () => {
  checkFfn({ fmt: "q4", sfmt: "q8", dim: 4096, ei: 2048, sDim: 32, nExp: 3, K: 2, C: 2, R: 1, RG: 4, UC: 2 });
});

// tiled kernels (moeGroupTiled): a different summation order, so equal only up to rounding (float64 here: ~1e-15)
function checkTiled(cfg) {
  const { hA, hB, xA, xB, x0, KS, hs, xs } = ffnBoth({ ...cfg, tiled: true }), { C, dim, ei, sDim, K } = cfg;
  const close = (a, b) => Math.abs(a - b) <= 1e-9 * (Math.abs(a) + 1e-3);
  for (let cs = 0; cs < C * KS; cs++) {
    const rows = cs % KS === K ? sDim : ei;
    for (let r = 0; r < hs; r++) {
      const a = hA[cs * hs + r], b = hB[cs * hs + r];
      if (r < rows ? !close(a, b) : !Number.isNaN(b)) throw new Error(`tiled h[${cs}][${r}]: ${b} vs per-pair ${a}`);
    }
  }
  for (let c = 0; c < C; c++) for (let i = 0; i < xs; i++) {
    const a = xA[c * xs + i], b = xB[c * xs + i];
    if (i >= dim ? b !== x0[c * xs + i] : !close(a, b)) throw new Error(`tiled x[${c}][${i}]: ${b} vs per-pair ${a}`);
  }
}
Deno.test("moe group tiled: close to per-pair (q4 / q8, UC 4, skewed, ragged tiles)", () => {
  checkTiled({ fmt: "q4", sfmt: "q8", dim: 320, ei: 48, sDim: 64, nExp: 5, K: 3, C: 6, UC: 4, skew: 0.5 });
});
Deno.test("moe group tiled: close to per-pair (q8 / q4, UC 8, sDim < ei, several k tiles)", () => {
  checkTiled({ fmt: "q8", sfmt: "q4", dim: 512, ei: 288, sDim: 32, nExp: 4, K: 2, C: 9, UC: 8 });
});
Deno.test("moe group tiled: close to per-pair (q4 / q4, UC 2)", () => {
  checkTiled({ fmt: "q4", sfmt: "q4", dim: 256, ei: 64, sDim: 64, nExp: 3, K: 2, C: 5, UC: 2 });
});
