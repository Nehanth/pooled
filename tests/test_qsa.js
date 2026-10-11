// QSA kernels (engine/wgsl/qsa.js, attn_dec.js gather mode, host side engine/qsa.js) against the CPU reference
// (tests/reference/qsa_ref.mjs), at Qwen3.8-Flash-Next's shape (24 q heads, 2 kv heads, hd 256, indexer 4 x 128):
//   proj     qsa_proj (BF16 GEMV) vs float64
//   select   qsa_select on given scores vs the reference rule (score desc, index asc), exactly: random, heavy ties,
//            all equal, every tail length, up to 65536 blocks (262K context)
//   e2e      contexts 512, 2051, 2052, 4096, 8192, 32768, decode and a prefill chunk ending there: pooled keys,
//            scores (tolerance), the row lists (exact, or only near-tie swaps, reported), attention (tolerance)
//   dense    <= 2051 positions: QSA == attn_dec bit for bit (decode and multi-column)
//   ties     q_norm = 0: every score is 0, the lowest 512 blocks win, GPU == CPU exactly
//   p == d   a prefill column's list and output == a decode step at that position, bit for bit
//   determinism  the same pass twice gives the same bits
//   cd tests && deno run --unstable-webgpu --allow-read test_qsa.js       (QUICK=1: contexts up to 8K)
import { qsaPipelines, qsaBindGroups, qsaDispatch, qsaUniform, qsaSizes, encodeQsa, QSA_SEL_STRIDE, QSA_PROJ } from "../engine/qsa.js";
import { attnDecWGSL, decSplits, DEC_CQ } from "../engine/wgsl/attn_dec.js";
import { QSA, QsaRefCache, ingest, qsaStep, selectBlocks, selectedCells, attendCells, f16bits, h2f } from "./reference/qsa_ref.mjs";
import { parseGGUFHeader } from "../engine/gguf.js";

const quick = Deno.env.get("QUICK") === "1";
const ad = await navigator.gpu.requestAdapter();
const device = await ad.requestDevice({ requiredLimits: { maxBufferSize: ad.limits.maxBufferSize, maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize } });
let gpuErr = 0;
device.addEventListener?.("uncapturederror", (e) => { if (gpuErr++ < 4) console.error("GPU ERROR:", e.error?.message?.slice(0, 400)); });
const S_ = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const U_ = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
const mk = (size, usage = S_) => device.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage });
const buf = (a, usage = S_) => { const b = mk(a.byteLength, usage); device.queue.writeBuffer(b, 0, a); return b; };
async function readU32(b, n, off = 0) {
  const st = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const e = device.createCommandEncoder(); e.copyBufferToBuffer(b, off * 4, st, 0, n * 4); device.queue.submit([e.finish()]);
  await st.mapAsync(GPUMapMode.READ); const out = new Uint32Array(st.getMappedRange().slice(0)); st.unmap(); st.destroy(); return out;
}
const readF32 = async (b, n, off = 0) => new Float32Array((await readU32(b, n, off)).buffer);

let seed = 20261011;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 2 ** 32 - 0.5; };
let fails = 0;
const check = (ok, msg) => { console.log(`${ok ? "ok  " : "FAIL"} ${msg}`); if (!ok) fails++; };
const bitsEq = (a, b) => { if (a.length !== b.length) return false; const x = new Uint32Array(a.buffer, a.byteOffset, a.length), y = new Uint32Array(b.buffer, b.byteOffset, b.length); for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false; return true; };
const relDiff = (g, r) => { let md = 0, mr = 0; for (let i = 0; i < r.length; i++) { md = Math.max(md, Math.abs(g[i] - r[i])); mr = Math.max(mr, Math.abs(r[i])); } return md / mr; };

// ---- shapes and pipelines ----
const nH = 24, nKV = 2, hd = 256, dim = 2560, qDim = nH * hd, kvDim = nKV * hd;
const K_ = await qsaPipelines(device, { hd, nH, nKV });
const { pipes, adCfg, nKVa } = K_;
// dense attn_dec at the same shape (the <= 2051 oracle)
const denseMod = device.createShaderModule({ code: attnDecWGSL(adCfg) });
const denseLayout1 = device.createBindGroupLayout({ entries: ["read-only-storage", "read-only-storage", "read-only-storage", "storage", "storage", "uniform"].map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
const denseCLayout1 = device.createBindGroupLayout({ entries: ["read-only-storage", "read-only-storage", "storage", "uniform"].map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
const dpl = (l1, e) => device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [K_.layout0, l1] }), compute: { module: denseMod, entryPoint: e } });
const pDense = dpl(denseLayout1, "attn_dec"), pDenseC = dpl(denseCLayout1, "attn_dec_combine");

const MAXSEQ = quick ? 8192 + 64 : 32768 + 64, MAXCOLS = 512;
const sz = qsaSizes({ maxSeq: MAXSEQ, maxCols: MAXCOLS });
const cfgU = new ArrayBuffer(48); { const u = new Uint32Array(cfgU); u.set([dim, kvDim, nH, nKV, hd, 0, 0, MAXSEQ]); new Float32Array(cfgU, 32, 1)[0] = 1e-6; new Uint32Array(cfgU, 40, 1)[0] = qDim; }
const B = {
  cfg: buf(new Uint8Array(cfgU), U_), frame: mk(16, U_), fd: mk(16, U_),
  qs: buf(qsaUniform({ xStride: dim, scoreStride: sz.maxBlocks }), U_),
  proj: mk(sz.proj), qi: mk(sz.qi), rawK: mk(sz.rawK), pooled: mk(sz.pooled), scores: mk(sz.scores), sel: mk(sz.sel),
  qNorm: mk(QSA.ID * 4), kNorm: mk(QSA.ID * 4),
  q: mk(MAXCOLS * qDim * 4), kCache: mk(MAXSEQ * kvDim * 2), vCache: mk(MAXSEQ * kvDim * 2),
  faO: mk(MAXCOLS * nH * adCfg.S * hd * 4), faML: mk(MAXCOLS * nH * adCfg.S * 2 * 4), out: mk(MAXCOLS * qDim * 4),
};
const G = qsaBindGroups(device, K_, B);
const g0 = G.qsa_prep[0];
const gDense = device.createBindGroup({ layout: denseLayout1, entries: [B.q, B.kCache, B.vCache, B.faO, B.faML, B.fd].map((b, binding) => ({ binding, resource: { buffer: b } })) });
const gDenseC = device.createBindGroup({ layout: denseCLayout1, entries: [B.faO, B.faML, B.out, B.fd].map((b, binding) => ({ binding, resource: { buffer: b } })) });

const setFrame = (pos, nCols) => device.queue.writeBuffer(B.frame, 0, new Uint32Array([pos, pos + 1, nCols, 0]));
const setFD = (S = adCfg.S) => device.queue.writeBuffer(B.fd, 0, new Uint32Array([qDim, qDim, S, adCfg.S]));
function submit(fn) { const e = device.createCommandEncoder(), p = e.beginComputePass(); fn(p); p.end(); device.queue.submit([e.finish()]); }

// ---- 1. qsa_proj vs float64 ----
{
  const nc = 3;
  const W = new Uint32Array(QSA_PROJ * dim / 2), Wf = new Float32Array(QSA_PROJ * dim);
  for (let i = 0; i < W.length; i++) {
    const lo = new Uint32Array(new Float32Array([rnd() * 0.05]).buffer)[0] >>> 16, hi = new Uint32Array(new Float32Array([rnd() * 0.05]).buffer)[0] >>> 16;
    W[i] = lo | (hi << 16); Wf[2 * i] = new Float32Array(new Uint32Array([lo << 16]).buffer)[0]; Wf[2 * i + 1] = new Float32Array(new Uint32Array([hi << 16]).buffer)[0];
  }
  const x = new Float32Array(nc * dim); for (let i = 0; i < x.length; i++) x[i] = rnd() * 2;
  const xb = buf(x), wb = buf(W);
  const Gp = qsaBindGroups(device, K_, { ...B, x: xb, w: wb });
  setFrame(0, nc);
  submit((p) => encodeQsa(p, pipes, Gp, qsaDispatch({ pos: 0, nCols: nc, adCfg, nH, nKVa }), { stages: ["proj"] }));
  const y = await readF32(B.proj, nc * QSA_PROJ);
  const r = new Float64Array(nc * QSA_PROJ);
  for (let c = 0; c < nc; c++) for (let o = 0; o < QSA_PROJ; o++) { let s = 0; for (let d = 0; d < dim; d++) s += Wf[o * dim + d] * x[c * dim + d]; r[c * QSA_PROJ + o] = s; }
  const rd = relDiff(y, r);
  check(rd < 1e-5, `proj   BF16 [640 x 2560] x 3 columns: rel diff vs f64 ${rd.toExponential(2)}`);
  xb.destroy(); wb.destroy();
}

// ---- 2. qsa_select on given scores ----
async function selectCase(name, scores, nv, tail) {
  const p = 4 * nv + tail - 1;
  device.queue.writeBuffer(B.scores, 0, scores);
  setFrame(p, 1);
  submit((pass) => encodeQsa(pass, pipes, G, qsaDispatch({ pos: p, nCols: 1, adCfg, nH, nKVa }), { stages: ["select"] }));
  const n = (await readU32(B.sel, 1))[0];
  const got = await readU32(B.sel, n, 1);
  const want = selectedCells(selectBlocks(scores, nv), p);
  const ok = n === want.length && got.every((v, i) => v === want[i]);
  check(ok, `select ${name}: nv ${nv} tail ${tail}: ${n} rows${ok ? "" : ` (want ${want.length}; first diff at ${got.findIndex((v, i) => v !== want[i])})`}`);
}
{
  const sizes = [1, 100, 512, 513, 777, 2048, 8192, 65536];
  for (const nv of sizes) {
    const s = new Float32Array(nv);
    for (let i = 0; i < nv; i++) s[i] = Math.max(0, rnd() * 3);                       // ~half exact zeros
    await selectCase("random+zeros", s, nv, nv % 4);
    const t = new Float32Array(nv); for (let i = 0; i < nv; i++) t[i] = [0, 0.5, 1, 1, 2][Math.floor((rnd() + 0.5) * 5)];   // heavy ties
    await selectCase("5 values", t, nv, (nv + 1) % 4);
  }
  for (const nv of [513, 4096, 65536]) {
    await selectCase("all equal", new Float32Array(nv).fill(0.25), nv, 3);
    await selectCase("all zero", new Float32Array(nv), nv, 0);
    // the boundary inside a run of equal values: 300 distinct large ones, then 1000 ties spread over the range
    const s = new Float32Array(nv); for (let i = 0; i < nv; i++) s[i] = Math.abs(rnd()) * 0.1;
    for (let i = 0; i < 300; i++) s[Math.floor((rnd() + 0.5) * nv)] = 10 + i;
    for (let i = 0; i < 1000; i++) s[Math.floor((rnd() + 0.5) * nv)] = 5;
    await selectCase("boundary tie run", s, nv, 2);
    // values that differ only in the last bit, -0 and 0
    const u = new Float32Array(nv); const ub = new Uint32Array(u.buffer);
    for (let i = 0; i < nv; i++) ub[i] = 0x3f800000 + (i % 3);
    for (let i = 0; i < nv; i += 7) u[i] = -0;
    await selectCase("1-ulp steps, -0", u, nv, 1);
  }
}

// ---- 3. end to end ----
// a random layer state for ctx positions: indexer projections, K / V (f16), attention q per column
function makeLayer(ctx, { zeroQ = false } = {}) {
  const qIdxRaw = new Float32Array(ctx * 512), kIdxRaw = new Float32Array(ctx * 128);
  for (let i = 0; i < qIdxRaw.length; i++) qIdxRaw[i] = rnd() * 2;
  // raw keys share a slowly varying component so pooled keys (and scores) have structure, not pure noise
  const drift = new Float32Array(128).map(() => rnd());
  for (let t = 0; t < ctx; t++) { if (t % 64 === 0) for (let d = 0; d < 128; d++) drift[d] = 0.8 * drift[d] + 0.4 * rnd(); for (let d = 0; d < 128; d++) kIdxRaw[t * 128 + d] = drift[d] + rnd(); }
  const qNorm = new Float32Array(128).map(() => zeroQ ? 0 : 1 + rnd() * 0.5), kNorm = new Float32Array(128).map(() => 1 + rnd() * 0.5);
  const Kh = new Uint16Array(ctx * kvDim), Vh = new Uint16Array(ctx * kvDim);
  for (let i = 0; i < Kh.length; i++) { Kh[i] = f16bits(rnd() * 0.6); Vh[i] = f16bits(rnd() * 2); }
  return { ctx, qIdxRaw, kIdxRaw, qNorm, kNorm, Kh, Vh, K: Float32Array.from(Kh, h2f), V: Float32Array.from(Vh, h2f) };
}
const projOf = (L, pos, nCols) => {
  const pr = new Float32Array(nCols * QSA_PROJ);
  for (let c = 0; c < nCols; c++) { pr.set(L.qIdxRaw.subarray((pos + c) * 512, (pos + c + 1) * 512), c * QSA_PROJ); pr.set(L.kIdxRaw.subarray((pos + c) * 128, (pos + c + 1) * 128), c * QSA_PROJ + 512); }
  return pr;
};
// upload K / V, run prep + pool over positions [0, upTo) in 512-column chunks on the GPU, and ingest on the CPU
async function prime(L, upTo) {
  device.queue.writeBuffer(B.kCache, 0, L.Kh); device.queue.writeBuffer(B.vCache, 0, L.Vh);
  device.queue.writeBuffer(B.qNorm, 0, L.qNorm); device.queue.writeBuffer(B.kNorm, 0, L.kNorm);
  const ref = new QsaRefCache({ maxSeq: MAXSEQ });
  for (let pos = 0; pos < upTo; pos += MAXCOLS) {
    const nc = Math.min(MAXCOLS, upTo - pos);
    device.queue.writeBuffer(B.proj, 0, projOf(L, pos, nc));
    setFrame(pos, nc);
    submit((p) => encodeQsa(p, pipes, G, qsaDispatch({ pos, nCols: nc, adCfg, nH, nKVa }), { stages: ["prep", "pool"] }));
    await device.queue.onSubmittedWorkDone();
    ingest(ref, { pos, nCols: nc, kIdxRaw: L.kIdxRaw.subarray(pos * 128, (pos + nc) * 128), kNorm: L.kNorm });
  }
  return ref;
}
// one QSA pass on the GPU: columns pos .. pos + nc - 1 -> per column { rows, out }; S: attn_dec target splits
async function gpuPass(L, q, pos, nc, S = adCfg.S) {
  device.queue.writeBuffer(B.proj, 0, projOf(L, pos, nc));
  device.queue.writeBuffer(B.q, 0, q);
  setFrame(pos, nc); setFD(S);
  const d = qsaDispatch({ pos, nCols: nc, adCfg: { ...adCfg, S }, nH, nKVa });
  submit((p) => encodeQsa(p, pipes, G, d, { stages: ["prep", "pool", "score", "select", "attn"] }));
  const sel = await readU32(B.sel, nc * QSA_SEL_STRIDE);
  const out = await readF32(B.out, nc * qDim);
  const cols = [];
  for (let c = 0; c < nc; c++) { const n = sel[c * QSA_SEL_STRIDE]; cols.push({ rows: sel.slice(c * QSA_SEL_STRIDE + 1, c * QSA_SEL_STRIDE + 1 + n), out: out.slice(c * qDim, (c + 1) * qDim) }); }
  return cols;
}
async function densePass(q, pos, nc) {
  device.queue.writeBuffer(B.q, 0, q); setFrame(pos, nc); setFD();
  let ns = 1; for (let c = 0; c < nc; c++) ns = Math.max(ns, decSplits(pos + c + 1, adCfg.S));
  submit((p) => {
    p.setPipeline(pDense); p.setBindGroup(0, g0); p.setBindGroup(1, gDense); p.dispatchWorkgroups(ns, nc, nKVa);
    p.setPipeline(pDenseC); p.setBindGroup(0, g0); p.setBindGroup(1, gDenseC); p.dispatchWorkgroups(nH * DEC_CQ, nc, 1);
  });
  return await readF32(B.out, nc * qDim);
}
const randQ = (nc) => { const q = new Float32Array(nc * qDim); for (let i = 0; i < q.length; i++) q[i] = rnd() * 4; return q; };

// compare one GPU column with the reference; returns a summary string, counts failures
function compareCol(tag, g, r, ref, L, qAll, c, scoresGpu) {
  const want = r.cells, got = g.rows;
  const same = got.length === want.length && got.every((v, i) => v === want[i]);
  let note = "";
  if (!same) {   // allowed only as swaps among near-tied scores at the selection boundary
    const gs = new Set(), ws = new Set();
    for (let i = 0; i < got.length; i += 4) gs.add(got[i] / 4); for (let i = 0; i < want.length; i += 4) ws.add(want[i] / 4);
    const onlyG = [...gs].filter((b) => !ws.has(b) && b < r.nv), onlyW = [...ws].filter((b) => !gs.has(b) && b < r.nv);
    const sorted = Float32Array.from(r.scores).sort().reverse(), T = sorted[QSA.N_TOP - 1];
    const gap = Math.max(...onlyG.concat(onlyW).map((b) => Math.abs(r.scores[b] - T) / Math.max(1e-30, Math.abs(T))));
    const nearTie = onlyG.length === onlyW.length && gap < 1e-4;
    note = ` lists differ by ${onlyG.length} block swap(s) at relative score gap ${gap.toExponential(1)} from the threshold${nearTie ? " (near tie)" : ""}`;
    if (!nearTie) fails++;
  }
  // attention: the GPU output vs float64 attention over the GPU's own rows (kernel accuracy), and vs the reference's
  const own = attendCells(qAll, c * qDim, L.K, L.V, Int32Array.from(got), { nH, nKV, hd });
  const rdOwn = relDiff(g.out, own), rdRef = relDiff(g.out, r.out);
  let sdiff = "";
  if (scoresGpu && r.scores) { sdiff = ` scores rel ${relDiff(scoresGpu, r.scores).toExponential(1)}`; }
  const ok = rdOwn < 2e-5 && (same ? rdRef < 2e-5 : true) && Number.isFinite(rdOwn);
  if (!ok) fails++;
  return `${ok ? "ok  " : "FAIL"} ${tag}: ${got.length} rows, lists ${same ? "equal" : "DIFFER"}${note}; attn rel ${rdOwn.toExponential(1)} (own rows) ${rdRef.toExponential(1)} (ref)${sdiff}`;
}

async function e2e(ctx, { prefillCols, layer = null }) {
  const L = layer || makeLayer(ctx);
  // decode: positions 0 .. ctx-2 cached, the token at ctx-1 decodes
  let ref = await prime(L, ctx - 1);
  {
    const pos = ctx - 1, q = randQ(1);
    // pooled keys: f16 GPU vs f16 CPU
    const nb = Math.floor(ctx / 4);
    const pk = await readU32(B.pooled, Math.max(1, Math.floor((ctx - 1) / 4)) * 64);
    let nd = 0, mu = 0;   // pooled keys: f16 values that differ from the reference's, and the largest difference
    for (let i = 0; i < pk.length; i++) for (const [h, r] of [[pk[i] & 0xffff, ref.pooled[2 * i]], [pk[i] >>> 16, ref.pooled[2 * i + 1]]]) {
      if (h !== f16bits(r)) { nd++; mu = Math.max(mu, Math.abs(h2f(h) - r)); }
    }
    if (mu > 2e-3) fails++;   // rounding flips only: an f16 ulp at 1 is 9.8e-4
    const [g] = await gpuPass(L, q, pos, 1);
    const scoresGpu = nb > QSA.N_TOP ? await readF32(B.scores, nb) : null;
    const [r] = qsaStep(ref, { pos, nCols: 1, qIdxRaw: L.qIdxRaw.subarray(pos * 512), kIdxRaw: L.kIdxRaw.subarray(pos * 128), qNorm: L.qNorm, kNorm: L.kNorm, q, K: L.K, V: L.V, shape: { nH, nKV, hd } });
    const older = Array.from(g.rows).filter((x, i) => i % 4 === 0 && x / 4 < nb - QSA.N_TOP).length;
    console.log(compareCol(`e2e decode  ctx ${String(ctx).padStart(5)}${L.tag || ""}`, g, r, ref, L, q, 0, scoresGpu) +
      `; pooled f16 ${nd}/${pk.length * 2} differ (max |d| ${mu.toExponential(1)})` + (nb > QSA.N_TOP ? `; ${older}/512 chosen blocks older than the newest 512` : ""));
    // the GPU's list must be exactly the reference rule applied to the GPU's own scores
    if (scoresGpu) { const w = selectedCells(selectBlocks(scoresGpu, nb), pos); check(w.length === g.rows.length && w.every((v, i) => v === g.rows[i]), `       ctx ${ctx}: GPU list == reference rule on GPU scores`); }
    if (ctx <= QSA.MAX_SEL) {
      const d = await densePass(q, pos, 1);
      check(bitsEq(d, g.out), `dense  ctx ${ctx} decode: QSA == attn_dec bit for bit (${g.rows.length} rows == 0 .. ${pos})`);
    }
    // determinism: the same decode again (prep / pool rewrite the same bits)
    const [g2] = await gpuPass(L, q, pos, 1);
    check(bitsEq(g2.out, g.out) && bitsEq(g2.rows, g.rows), `determ ctx ${ctx} decode: second run same bits`);
  }
  // prefill: a chunk of prefillCols columns ending at ctx-1, on a fresh state primed to its start
  {
    const nc = Math.min(prefillCols, ctx), pos = ctx - nc;
    ref = await prime(L, pos);
    const q = randQ(nc);
    const gs = await gpuPass(L, q, pos, nc);
    const rs = qsaStep(ref, { pos, nCols: nc, qIdxRaw: L.qIdxRaw.subarray(pos * 512), kIdxRaw: L.kIdxRaw.subarray(pos * 128), qNorm: L.qNorm, kNorm: L.kNorm, q, K: L.K, V: L.V, shape: { nH, nKV, hd } });
    let worst = "", nDiff = 0, okAll = true;
    for (let c = 0; c < nc; c++) {
      const before = fails;
      const line = compareCol(`e2e prefill ctx ${ctx} col ${c}`, gs[c], rs[c], ref, L, q, c, null);
      if (fails > before) { okAll = false; console.log(line); }
      if (!line.includes("lists equal")) { nDiff++; worst = line; }
    }
    console.log(`${okAll ? "ok  " : "FAIL"} e2e prefill ctx ${String(ctx).padStart(5)}: ${nc} columns at ${pos} .. ${ctx - 1}, lists equal in ${nc - nDiff}/${nc}${nDiff ? "; e.g. " + worst.replace(/^ok {2}/, "") : ""}`);
    // prefill column == decode step at that position: replay the last column as a decode (state already has its keys)
    const last = nc - 1, pl = pos + last;
    const [gd] = await gpuPass(L, q.subarray(last * qDim), pl, 1);
    check(bitsEq(gd.out, gs[last].out) && bitsEq(gd.rows, gs[last].rows), `p == d ctx ${ctx}: prefill column ${last} == decode at ${pl} bit for bit`);
    if (ctx <= QSA.MAX_SEL) {
      const d = await densePass(q, pos, nc);
      let eq = true; for (let c = 0; c < nc; c++) eq = eq && bitsEq(d.subarray(c * qDim, (c + 1) * qDim), gs[c].out);
      check(eq, `dense  ctx ${ctx} prefill: ${nc} QSA columns == attn_dec multi-column bit for bit`);
    }
    // the prefill attention split (S = 4: fewer split slots per column, engine/qsa.js): same rows, tolerance
    const g4 = await gpuPass(L, q, pos, nc, 4);
    let r4 = 0, rows4 = true;
    for (let c = 0; c < nc; c++) { rows4 = rows4 && bitsEq(g4[c].rows, gs[c].rows); r4 = Math.max(r4, relDiff(g4[c].out, attendCells(q, c * qDim, L.K, L.V, Int32Array.from(g4[c].rows), { nH, nKV, hd }))); }
    check(rows4 && r4 < 2e-5, `split4 ctx ${ctx} prefill with S = 4: same rows, attn rel ${r4.toExponential(1)} vs f64`);
    const gs2 = await gpuPass(L, q, pos, nc);
    check(gs2.every((g, c) => bitsEq(g.out, gs[c].out) && bitsEq(g.rows, gs[c].rows)), `determ ctx ${ctx} prefill: second run same bits`);
  }
}

const ctxs = quick ? [512, 2051, 2052, 4096, 8192] : [512, 2051, 2052, 4096, 8192, 32768];
for (const ctx of ctxs) await e2e(ctx, { prefillCols: ctx >= 32768 ? 16 : 32 });

// ---- 4. ties: q_norm = 0 makes every indexer score exactly 0; the lowest 512 blocks + the tail win ----
{
  const ctx = 4099, L = makeLayer(ctx, { zeroQ: true });
  const ref = await prime(L, ctx - 3);
  const q = randQ(3), pos = ctx - 3;
  const gs = await gpuPass(L, q, pos, 3);
  const rs = qsaStep(ref, { pos, nCols: 3, qIdxRaw: L.qIdxRaw.subarray(pos * 512), kIdxRaw: L.kIdxRaw.subarray(pos * 128), qNorm: L.qNorm, kNorm: L.kNorm, q, K: L.K, V: L.V, shape: { nH, nKV, hd } });
  for (let c = 0; c < 3; c++) {
    const p = pos + c, want = selectedCells(Int32Array.from({ length: 512 }, (_, i) => i), p);
    const ok = gs[c].rows.length === want.length && gs[c].rows.every((v, i) => v === want[i]) && rs[c].cells.every((v, i) => v === want[i]);
    check(ok && relDiff(gs[c].out, rs[c].out) < 2e-5, `ties   all scores 0 at p ${p}: rows 0 .. 2047 + tail ${(p + 1) % 4} (GPU and CPU)`);
  }
}

// ---- 5. rewind: a verify pass completes block 1024, is rolled back to 4098 and re-decoded with other tokens ----
{
  const ctx = 4100, L = makeLayer(ctx);
  await prime(L, 4096);
  await gpuPass(L, randQ(4), 4096, 4);
  const L2 = { ...L, kIdxRaw: Float32Array.from(L.kIdxRaw), qIdxRaw: Float32Array.from(L.qIdxRaw) };
  for (let i = 4098 * 128; i < 4100 * 128; i++) L2.kIdxRaw[i] = rnd() * 3;
  for (let i = 4098 * 512; i < 4100 * 512; i++) L2.qIdxRaw[i] = rnd() * 2;
  const q = randQ(2);
  const gs = await gpuPass(L2, q, 4098, 2);
  const ref = new QsaRefCache({ maxSeq: MAXSEQ });
  ingest(ref, { pos: 0, nCols: 4098, kIdxRaw: L2.kIdxRaw, kNorm: L2.kNorm });
  const rs = qsaStep(ref, { pos: 4098, nCols: 2, qIdxRaw: L2.qIdxRaw.subarray(4098 * 512), kIdxRaw: L2.kIdxRaw.subarray(4098 * 128), qNorm: L2.qNorm, kNorm: L2.kNorm, q, K: L2.K, V: L2.V, shape: { nH, nKV, hd } });
  const pk = await readU32(B.pooled, 64, 1024 * 64);
  let nd = 0; for (let i = 0; i < 64; i++) { if ((pk[i] & 0xffff) !== f16bits(ref.pooled[1024 * 128 + 2 * i])) nd++; if ((pk[i] >>> 16) !== f16bits(ref.pooled[1024 * 128 + 2 * i + 1])) nd++; }
  const same = gs.every((g, c) => g.rows.length === rs[c].cells.length && g.rows.every((v, i) => v === rs[c].cells[i]));
  check(nd <= 2 && same && gs.every((g, c) => relDiff(g.out, rs[c].out) < 2e-5), `rewind block 1024 re-pooled from the new keys (${nd}/128 f16 values off by rounding), rows and attention match`);
}

// ---- 6. real indexer weights (blk.3 of the Q4_0 GGUF, when present): qsa_proj on random x, then the e2e checks ----
{
  const dir = Deno.env.get("Q38FLASH_DIR") || `${Deno.env.get("HOME")}/llm/models/q38flash`;
  let file = null;
  try { file = [...Deno.readDirSync(dir)].map((e) => e.name).find((n) => /Q4_0-00001-of-/.test(n)); } catch { /* no model dir */ }
  if (!file) console.log(`real weights: no GGUF under ${dir}, not run`);
  else {
    const fh = Deno.openSync(`${dir}/${file}`, { read: true });
    const readAt = (off, len) => { const b = new Uint8Array(len); fh.seekSync(off, Deno.SeekMode.Start); let n = 0; while (n < len) n += fh.readSync(b.subarray(n)); return b; };
    const Ghead = parseGGUFHeader(readAt(0, 64 << 20).buffer);
    const T = (n) => { const t = Ghead.tensors[n]; return readAt(t.byteOffset, t.byteLength); };
    const qp = T("blk.3.indexer.q_proj.weight"), kp = T("blk.3.indexer.k_proj.weight");
    const W = new Uint8Array(qp.length + kp.length); W.set(qp); W.set(kp, qp.length);
    const qNorm = new Float32Array(T("blk.3.indexer.q_norm.weight").buffer), kNorm = new Float32Array(T("blk.3.indexer.k_norm.weight").buffer);
    fh.close();
    const ctx = 4100, xs = new Float32Array(ctx * dim);
    for (let i = 0; i < xs.length; i++) xs[i] = (rnd() + rnd() + rnd()) * 1.2;
    const wb = buf(W), xb = mk(MAXCOLS * dim * 4);
    const Gp = qsaBindGroups(device, K_, { ...B, x: xb, w: wb });
    const proj = new Float32Array(ctx * QSA_PROJ);
    for (let pos = 0; pos < ctx; pos += MAXCOLS) {
      const nc = Math.min(MAXCOLS, ctx - pos);
      device.queue.writeBuffer(xb, 0, xs.subarray(pos * dim, (pos + nc) * dim)); setFrame(pos, nc);
      submit((p) => encodeQsa(p, pipes, Gp, qsaDispatch({ pos, nCols: nc, adCfg, nH, nKVa }), { stages: ["proj"] }));
      proj.set(await readF32(B.proj, nc * QSA_PROJ), pos * QSA_PROJ);
    }
    const Wf = new Float32Array(QSA_PROJ * dim); { const u = new Uint16Array(W.buffer); for (let i = 0; i < Wf.length; i++) Wf[i] = new Float32Array(new Uint32Array([u[i] << 16]).buffer)[0]; }
    let rd = 0;
    for (const c of [0, 4099]) { const r = new Float64Array(QSA_PROJ); for (let o = 0; o < QSA_PROJ; o++) { let a = 0; for (let d = 0; d < dim; d++) a += Wf[o * dim + d] * xs[c * dim + d]; r[o] = a; } rd = Math.max(rd, relDiff(proj.subarray(c * QSA_PROJ, (c + 1) * QSA_PROJ), r)); }
    check(rd < 1e-5, `real   blk.3 indexer q_proj | k_proj (BF16) via qsa_proj: rel diff vs f64 ${rd.toExponential(2)}; q_norm mean ${(qNorm.reduce((a, b) => a + b) / 128).toFixed(3)} (folded 1 + w)`);
    const L = makeLayer(ctx);
    for (let t = 0; t < ctx; t++) { L.qIdxRaw.set(proj.subarray(t * QSA_PROJ, t * QSA_PROJ + 512), t * 512); L.kIdxRaw.set(proj.subarray(t * QSA_PROJ + 512, (t + 1) * QSA_PROJ), t * 128); }
    L.qNorm = qNorm; L.kNorm = kNorm; L.tag = " (blk.3 weights)";
    await e2e(ctx, { prefillCols: 32, layer: L });
    wb.destroy(); xb.destroy();
  }
}

console.log(fails || gpuErr ? `QSA FAIL (${fails} failures, ${gpuErr} GPU errors)` : "QSA PASS");
if (fails || gpuErr) Deno.exit(1);
