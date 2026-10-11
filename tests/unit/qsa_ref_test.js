// QSA CPU reference (tests/reference/qsa_ref.mjs) and host-side helpers (engine/qsa.js), no GPU:
//   - selection rule: top 512 by (score desc, index asc), ascending output; ties decided by block index
//   - <= 2051 positions select everything (QSA == dense); 2052 drops exactly one block
//   - a prefill chunk selects and attends exactly as token-by-token decode
//   - qsa_cs's 2 pi split and the dispatch bounds
import { QSA, QsaRefCache, qsaStep, selectBlocks, selectedCells, denseCells, f16round } from "../reference/qsa_ref.mjs";
import { QSA_TWO_PI_SPLIT, QSA_MAXSEL } from "../../engine/wgsl/qsa.js";
import { qsaDispatch, qsaAttnConfig, qsaMaxRows, qsaNeedsScore } from "../../engine/qsa.js";

const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
let seed = 99;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 2 ** 32 - 0.5; };

Deno.test("select: top 512 by score desc, index asc, returned ascending", () => {
  for (const nv of [513, 600, 5000]) {
    for (const mode of ["random", "ties", "zeros"]) {
      const s = new Float32Array(nv).map(() => mode === "random" ? Math.max(0, rnd()) : mode === "ties" ? [0, 1, 2][Math.floor((rnd() + 0.5) * 3)] : 0);
      const got = selectBlocks(s, nv);
      ok(got.length === 512, "512 blocks");
      for (let i = 1; i < got.length; i++) ok(got[i] > got[i - 1], "ascending");
      // brute force: the 512th value T, every block above T, and the lowest-index blocks equal to T
      const T = Float32Array.from(s).sort().reverse()[511];
      const above = [...s.keys()].filter((b) => s[b] > T), eq = [...s.keys()].filter((b) => s[b] === T).slice(0, 512 - above.length);
      const want = [...above, ...eq].sort((a, b) => a - b);
      ok(want.every((b, i) => b === got[i]), `${mode} nv ${nv}`);
    }
  }
  ok(selectBlocks(new Float32Array(1000), 1000).every((b, i) => b === i), "all equal: the first 512 blocks");
});

Deno.test("<= 2051 positions is dense; 2052 drops one block", () => {
  for (const p of [0, 1, 2, 3, 4, 100, 2047, 2048, 2049, 2050]) {
    const nv = Math.floor((p + 1) / 4);
    const cells = selectedCells(selectBlocks(new Float32Array(nv).map(() => Math.abs(rnd())), nv), p);
    ok(cells.length === p + 1 && cells.every((c, i) => c === i), `p ${p}: rows 0 .. p`);
  }
  const s = new Float32Array(513).map(() => Math.abs(rnd()));
  const cells = selectedCells(selectBlocks(s, 513), 2051);
  ok(cells.length === 2048 && QSA_MAXSEL === 2051, "p 2051: 512 whole blocks, no tail");
  const lo = s.indexOf(Math.min(...s));
  ok(!cells.includes(4 * lo) && denseCells(2051).length === 2052, "the lowest-scoring block is the one dropped");
});

Deno.test("prefill chunk == token-by-token decode (selection and attention)", () => {
  const ctx = 2100, nH = 4, nKV = 2, hd = 16, kvDim = nKV * hd;
  const qIdxRaw = new Float32Array(ctx * 512).map(() => rnd()), kIdxRaw = new Float32Array(ctx * 128).map(() => rnd() + 0.3);
  const qNorm = new Float32Array(128).map(() => 1 + rnd()), kNorm = new Float32Array(128).map(() => 1 + rnd());
  const K = new Float32Array(ctx * kvDim).map(() => f16round(rnd())), V = new Float32Array(ctx * kvDim).map(() => f16round(rnd()));
  const q = new Float32Array(ctx * nH * hd).map(() => rnd() * 3);
  const shape = { nH, nKV, hd };
  const run = (chunks) => {
    const cache = new QsaRefCache({ maxSeq: ctx });
    const out = [];
    let pos = 0;
    for (const nc of chunks) {
      out.push(...qsaStep(cache, { pos, nCols: nc, qIdxRaw: qIdxRaw.subarray(pos * 512), kIdxRaw: kIdxRaw.subarray(pos * 128), qNorm, kNorm,
        q: q.subarray(pos * nH * hd), K, V, shape }));
      pos += nc;
    }
    return out;
  };
  const tail = 2100 - 2040;
  const a = run([2040, ...Array(tail).fill(1)]), b = run([2040, 17, 3, 40]);
  for (let i = 2040; i < ctx; i++) {
    ok(a[i].cells.length === b[i].cells.length && a[i].cells.every((c, j) => c === b[i].cells[j]), `rows at ${i}`);
    ok(a[i].out.every((v, j) => v === b[i].out[j]), `output at ${i}`);
  }
  ok(a[2099].cells.length === 2048 + (2100 % 4), "sparse at the end");
});

Deno.test("qsa_cs: 2 pi in three parts, the first two with 8 significant bits", () => {
  const [c1, c2, c3] = QSA_TWO_PI_SPLIT;
  const bits = (x) => { let m = Math.abs(x); while (m !== Math.floor(m)) m *= 2; return m.toString(2).replace(/0+$/, "").length; };
  ok(bits(c1) <= 8 && bits(c2) <= 8, "8-bit parts");
  ok(Math.abs(c1 + c2 + c3 - 2 * Math.PI) < 1e-12 && Math.fround(c3) === c3, "sum");
});

Deno.test("dispatch bounds", () => {
  const adCfg = qsaAttnConfig({ hd: 256, nH: 24, nKV: 2 });
  ok(adCfg && adCfg.G === 6 && adCfg.R === 2 && adCfg.nKVa === 4, "24 / 2 heads: two workgroups of 6 per kv head");
  ok(!qsaNeedsScore(2050, 1) && qsaNeedsScore(2051, 1), "scoring starts at 513 visible blocks");
  ok(qsaMaxRows(100, 1) === 101 && qsaMaxRows(9000, 64) === 2051, "attention rows");
  const d = qsaDispatch({ pos: 4095, nCols: 1, adCfg, nH: 24, nKVa: 4 });
  ok(d.pool[0] === 1 && d.score.entry === "qsa_score" && d.select[0] === 1 && d.attn[2] === 4, "decode sizes");
  const p = qsaDispatch({ pos: 5, nCols: 10, adCfg, nH: 24, nKVa: 4 });
  ok(p.pool[0] === 3 && p.score === null, "prefill: candidate blocks 1 .. 3 (3 incomplete), nothing to score");
  ok(QSA.MAX_SEL === QSA_MAXSEL, "constants agree");
});
