// CPU check of the tiled prefill attention (engine/wgsl/attn_tile.js): the generator emits the kernel
// bodies as JavaScript (attnTileBodies(c, true)), which run here with one generator per thread and
// workgroupBarrier() as a yield, in float64, against exact causal attention over the same f16 K/V.
// Checks the index math, the causal mask per column, the split / tile tails, that no K/V row past a
// column's own position is ever used (those rows are NaN here), that every output of the pass is
// written and nothing outside it, and that the JS split length mirrors the kernel's. No GPU.
//   deno test --no-check tests/unit/attn_tile_test.js
import { attnTileConfig, attnTileBodies, attnTileWGSL, tileSplitLen } from "../../engine/wgsl/attn_tile.js";
import { f16ToF32, f32ToF16 } from "../../engine/gguf.js";

const H = {
  unpack2x16float: (w) => [f16ToF32(w & 0xffff), f16ToF32(w >>> 16)],
  vmad: (a, s, v) => typeof s === "number" ? a.map((x, j) => x + s * v[j]) : a.map((x, j) => x + s[j] * v[j]),
  select: (f, t, c) => (c ? t : f), min: Math.min, max: Math.max, exp: Math.exp, sqrt: Math.sqrt, f32: (x) => x,
};
let seed = 12345;
const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
// engine/qwen35.js split-K sizing
const faSizing = (maxSeq) => { const faSplit = Math.max(256, Math.ceil(maxSeq / 128 / 64) * 64); return { faSplit, faSplits: Math.ceil(maxSeq / faSplit) }; };

function runGrid(body, names, args, gx, gy, gz, wgArrays) {
  const hn = Object.keys(H);
  const make = new Function(...hn, ...names, "at_kv", "at_sp", `return function* (wg, lid) {${body}};`);
  for (let z = 0; z < gz; z++) for (let y = 0; y < gy; y++) for (let x = 0; x < gx; x++) {
    const [kv, sp] = wgArrays();
    const fn = make(...hn.map((n) => H[n]), ...args, kv, sp);
    const th = Array.from({ length: 256 }, (_, t) => fn({ x, y, z }, { x: t }));
    for (;;) {
      const done = th.map((g) => g.next().done);
      if (done.every(Boolean)) break;
      if (done.some(Boolean)) throw new Error("threads disagree at a barrier");
    }
  }
}

// one prefill pass of nc columns at basePos; returns max relative error vs exact attention
function check({ hd, nH, nKV, nc = 16, basePos, maxSeq, target = 32, tk = 0, wgMem = 16384, kvh = false, pf = false, nr = false }) {
  const G = nH / nKV, kvDim = nKV * hd;
  const { faSplit, faSplits } = faSizing(maxSeq);
  const c = attnTileConfig({ hd, G, faSplit, faSplits, wgMem, target, tk, kvh, pf, nr });
  if (!c) throw new Error("config rejected");
  const seqEnd = basePos + nc;
  // K/V caches (f16 pairs); rows at or past seqEnd hold NaN: the kernel must never use them
  const kc = new Uint32Array(maxSeq * kvDim / 2), vc = new Uint32Array(maxSeq * kvDim / 2);
  const K = new Float64Array(maxSeq * kvDim), V = new Float64Array(maxSeq * kvDim);
  for (let p = 0; p < maxSeq; p++) for (let w = 0; w < kvDim / 2; w++) {
    const pk = (i, a, f) => { const h = p < seqEnd ? f32ToF16(f) : 0x7e00; a[p * kvDim + 2 * w + i] = p < seqEnd ? f16ToF32(h) : NaN; return h; };
    kc[p * kvDim / 2 + w] = pk(0, K, (rnd() - 0.5) * 2) | (pk(1, K, (rnd() - 0.5) * 2) << 16);
    vc[p * kvDim / 2 + w] = pk(0, V, rnd() - 0.5) | (pk(1, V, rnd() - 0.5) << 16);
  }
  const s0 = nH * hd + 12, s1 = nH * hd + 20;   // padded column strides
  const q = new Float32Array(nc * s0).map(() => (rnd() - 0.5) * 3);
  const faO = new Float64Array(nc * nH * faSplits * hd).fill(NaN), faML = new Float64Array(nc * nH * faSplits * 2).fill(NaN);
  const out = new Float64Array(nc * s1).fill(NaN);
  const cfg = { kvDim, nH }, frame = { pos: basePos, seqLen: basePos + 1, nCols: nc, snap: 0 };
  const U = { s0, s1, splitLen: faSplit, maxSplits: faSplits };
  const js = attnTileBodies(c, true);
  const wgA = () => [Array.from({ length: c.TK * hd / 4 }, () => [NaN, NaN, NaN, NaN]), new Float64Array(c.TK * 256).fill(NaN)];
  runGrid(js.flash, ["cfg", "frame", "at_q", "at_k", "at_v", "at_o", "at_ml", "atu"], [cfg, frame, q, kc, vc, faO, faML, U],
    faSplits, nKV, Math.ceil(nc / c.CW), wgA);
  runGrid(js.combine, ["cfg", "frame", "atc_o", "atc_ml", "atc_out", "atc"], [cfg, frame, faO, faML, out, U], nH, nc, 1, wgA);
  let maxRel = 0;
  for (let col = 0; col < nc; col++) for (let qh = 0; qh < nH; qh++) {
    const g = Math.floor(qh / G), n = basePos + col + 1;
    const sc = new Float64Array(n);
    let mx = -Infinity;
    for (let p = 0; p < n; p++) {
      let s = 0; for (let d = 0; d < hd; d++) s += q[col * s0 + qh * hd + d] * K[p * kvDim + g * hd + d];
      sc[p] = s / Math.sqrt(hd); mx = Math.max(mx, sc[p]);
    }
    let L = 0; for (let p = 0; p < n; p++) { sc[p] = Math.exp(sc[p] - mx); L += sc[p]; }
    for (let d = 0; d < hd; d++) {
      let o = 0, a = 0; for (let p = 0; p < n; p++) { o += sc[p] * V[p * kvDim + g * hd + d]; a += sc[p] * Math.abs(V[p * kvDim + g * hd + d]); }
      const got = out[col * s1 + qh * hd + d];
      const rel = Math.abs(got - o / L) / (a / L + 1e-30);
      if (!(rel <= maxRel)) maxRel = Number.isNaN(rel) ? Infinity : Math.max(maxRel, rel);
    }
  }
  // padding between columns untouched
  for (let col = 0; col < nc; col++) for (let i = nH * hd; i < s1; i++) if (!Number.isNaN(out[col * s1 + i])) throw new Error("wrote into padding");
  return { maxRel, c, splits: Math.ceil(seqEnd / tileSplitLen(seqEnd, c)), out };
}

Deno.test("attn_tile: split length mirror and bounds", () => {
  for (const maxSeq of [256, 512, 2048, 4096, 16384, 32768, 65536, 262144]) {
    const { faSplit, faSplits } = faSizing(maxSeq);
    for (const target of [1, 8, 32, 64]) {
      const c = attnTileConfig({ hd: 256, G: 8, faSplit, faSplits, target });
      const E = new Function("min", "max", "frame", `return ${attnTileBodies(c, true).combine.match(/let sl = (.*);/)[1]};`);
      for (let seqEnd = 1; seqEnd <= maxSeq; seqEnd += Math.max(1, Math.floor(maxSeq / 997))) {
        const tl = tileSplitLen(seqEnd, c);
        if (tl % 64 || tl < 64 || tl > faSplit || Math.ceil(seqEnd / tl) > faSplits) throw new Error(`bad split ${tl} at ${seqEnd}/${maxSeq}`);
        for (const nc of [1, 16]) if (seqEnd >= nc) {
          const k = E(Math.min, Math.max, { seqLen: seqEnd - nc + 1, nCols: nc });
          if (k !== tl) throw new Error(`kernel split ${k} != js ${tl} at ${seqEnd}`);
        }
      }
    }
  }
});

Deno.test("attn_tile: WGSL shape", () => {
  for (const [hd, G, wgMem] of [[256, 6, 16384], [256, 8, 16384], [256, 8, 32768], [128, 4, 16384], [64, 2, 16384]]) {
    const c = attnTileConfig({ hd, G, faSplit: 256, faSplits: 64, wgMem });
    const src = attnTileWGSL(c);
    const bytes = c.TK * hd * 4 + c.TK * 256 * 4;
    if (bytes > wgMem) throw new Error("workgroup memory over the limit");
    if (/undefined|NaN|\$\{/.test(src)) throw new Error("template leak");
    if ((src.match(/{/g) || []).length !== (src.match(/}/g) || []).length) throw new Error("unbalanced braces");
    for (const e of ["fn attn_flash_tile", "fn attn_combine_tile"]) if (!src.includes(e)) throw new Error("missing " + e);
    if (c.CW * G > 64) throw new Error("too many rows");
  }
  if (attnTileConfig({ hd: 256, G: 8, faSplit: 256, faSplits: 64, wgMem: 32768 }).TK !== 16) throw new Error("TK 16 expected at 32 KB");
  if (attnTileConfig({ hd: 256, G: 8, faSplit: 256, faSplits: 64, wgMem: 16384 }).TK !== 8) throw new Error("TK 8 expected at 16 KB");
});

// MoE-like (G = 8, CW = 8: two column groups) and 27B-like (G = 6, CW = 10: groups of 10 and 6, 4 padded rows)
const cases = [
  { name: "MoE shape, first pass", hd: 256, nH: 16, nKV: 2, basePos: 0, maxSeq: 512 },
  { name: "MoE shape, 5 splits, partial tile", hd: 256, nH: 16, nKV: 2, basePos: 301, maxSeq: 4096 },
  { name: "27B shape, 2 big splits", hd: 256, nH: 24, nKV: 4, basePos: 173, maxSeq: 512 },
  { name: "27B shape, TK 16", hd: 256, nH: 24, nKV: 4, basePos: 90, maxSeq: 4096, wgMem: 32768 },
  { name: "small head, 1 split target", hd: 64, nH: 8, nKV: 2, basePos: 219, maxSeq: 2048, target: 1 },
  { name: "8-column batch", hd: 128, nH: 12, nKV: 2, nc: 8, basePos: 77, maxSeq: 1024 },
];
for (const t of cases) Deno.test(`attn_tile: ${t.name}`, () => {
  const { maxRel, c, splits } = check(t);
  console.log(`  ${t.name}: TK ${c.TK}, CW ${c.CW}, ${splits} splits, max rel err ${maxRel.toExponential(2)} (float64 run, f16 K/V)`);
  if (!(maxRel < 1e-9)) throw new Error(`max rel err ${maxRel}`);
});

// kvh (K / V staged as f16 pairs, widened at each read): the same values in the same order, so the same output
Deno.test("attn_tile: kvh gives the identical output", () => {
  for (const t of [cases[1], cases[3], cases[5]]) {
    const s0 = seed;
    const a = check(t); seed = s0;
    const b = check({ ...t, kvh: true });
    if (a.out.length !== b.out.length || a.out.some((v, i) => !Object.is(v, b.out[i]))) throw new Error(`${t.name}: kvh output differs`);
    if (!/at_kv: array<vec2<u32>/.test(attnTileWGSL(b.c))) throw new Error("kvh: the tile is not packed");
  }
});

// pf (the next tile prefetched into registers), alone and with kvh: the same output
Deno.test("attn_tile: pf, pf + kvh, nr, nr + pf give the identical output", () => {
  for (const t of [cases[0], cases[1], cases[3], cases[5]]) {
    const s0 = seed;
    const a = check(t);
    for (const o of [{ pf: true }, { pf: true, kvh: true }, { nr: true }, { nr: true, pf: true }]) {
      seed = s0;
      const b = check({ ...t, ...o });
      if (a.out.some((v, i) => !Object.is(v, b.out[i]))) throw new Error(`${t.name}: ${JSON.stringify(o)} output differs`);
    }
    seed = s0; check(t);
  }
});

Deno.test("attn_tile: nr leaves no private array", () => {
  const c = attnTileConfig({ hd: 256, G: 8, faSplit: 256, faSplits: 64, nr: true, pf: true });
  const src = attnTileWGSL(c);
  if (/var (q|o|p): array/.test(src) || /\b[qop]\[\d+u\]/.test(src)) throw new Error("a private array is left");
  if (!/var q15 = vec4<f32>\(0\.0\);/.test(src) || !/var p7 = 0\.0;/.test(src)) throw new Error("named variables missing");
});
