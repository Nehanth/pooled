// QSA kernel timings per attention layer (GPU timestamp queries), Qwen3.8-Flash-Next shape (24 q heads, 2 kv
// heads, hd 256, indexer 4 x 128), random data, at context lengths CTX:
//   decode:  prep, pool, score, select, attn (attn_dec_qsa + combine), their sum, and dense attn_dec at the same
//            position for comparison (what the layer would cost without QSA)
//   prefill: a COLS-column chunk ending at the context length (score c4, select per column, attention with S = 4)
// Each stage runs REPS times inside one timed compute pass; the figure is the per-run mean of the best of 3.
// The indexer projection (qsa_proj, 640 x 2560 BF16) is timed once (decode and prefill) since it does not depend
// on the context.
//   cd tests && deno run --unstable-webgpu --allow-read --allow-env bench_qsa.js
//   CTX=4096,8192,32768,131072 COLS=512 REPS=50
import { qsaPipelines, qsaBindGroups, qsaDispatch, qsaUniform, qsaSizes, encodeQsa, QSA_PROJ } from "../engine/qsa.js";
import { attnDecWGSL, decSplits, DEC_CQ } from "../engine/wgsl/attn_dec.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const CTXS = env("CTX", "4096,8192,32768").split(",").map(Number);
const COLS = +env("COLS", 512), REPS = +env("REPS", 50), PSPLIT = 4;
const ad = await navigator.gpu.requestAdapter();
const device = await ad.requestDevice({ requiredFeatures: ["timestamp-query"],
  requiredLimits: { maxBufferSize: ad.limits.maxBufferSize, maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize } });
let gpuErr = 0;
device.addEventListener?.("uncapturederror", (e) => { if (gpuErr++ < 4) console.error("GPU ERROR:", e.error?.message?.slice(0, 300)); });
const S_ = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, U_ = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
const mk = (size, usage = S_) => device.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage });
let seed = 7;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 2 ** 32 - 0.5; };
const fill = (b, n, f) => { const CH = 1 << 22; for (let o = 0; o < n; o += CH) { const m = Math.min(CH, n - o); device.queue.writeBuffer(b, o * 4, f(m)); } };
const randF = (n, s = 1) => { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = rnd() * s; return a; };
const randH = (n) => { const a = new Uint32Array(n); for (let i = 0; i < n; i++) a[i] = 0x3000 + ((rnd() + 0.5) * 0x1800 | 0) | ((0x3000 + ((rnd() + 0.5) * 0x1800 | 0)) << 16) | (rnd() < 0 ? 0x8000 : 0); return a; };

const nH = 24, nKV = 2, hd = 256, dim = 2560, qDim = nH * hd, kvDim = nKV * hd;
const K_ = await qsaPipelines(device, { hd, nH, nKV });
const { pipes, adCfg, nKVa } = K_;
const MAXSEQ = Math.max(...CTXS) + 16, MAXC = Math.max(COLS, 1);
const sz = qsaSizes({ maxSeq: MAXSEQ, maxCols: MAXC });
const slotsP = PSPLIT * 2;
const cfgU = new ArrayBuffer(48); { const u = new Uint32Array(cfgU); u.set([dim, kvDim, nH, nKV, hd, 0, 0, MAXSEQ]); new Float32Array(cfgU, 32, 1)[0] = 1e-6; new Uint32Array(cfgU, 40, 1)[0] = qDim; }
const B = {
  cfg: mk(48, U_), frame: mk(16, U_), fd: mk(16, U_), qs: mk(32, U_),
  x: mk(MAXC * dim * 4), w: mk(QSA_PROJ * dim * 2),
  proj: mk(sz.proj), qi: mk(sz.qi), rawK: mk(sz.rawK), pooled: mk(sz.pooled), scores: mk(sz.scores), sel: mk(sz.sel),
  qNorm: mk(512), kNorm: mk(512), q: mk(MAXC * qDim * 4), kCache: mk(MAXSEQ * kvDim * 2), vCache: mk(MAXSEQ * kvDim * 2),
  faO: mk(Math.max(adCfg.S, MAXC * slotsP) * nH * hd * 4), faML: mk(Math.max(adCfg.S, MAXC * slotsP) * nH * 2 * 4), out: mk(MAXC * qDim * 4),
};
device.queue.writeBuffer(B.cfg, 0, new Uint8Array(cfgU));
device.queue.writeBuffer(B.qs, 0, qsaUniform({ xStride: dim, scoreStride: sz.maxBlocks }));
fill(B.x, MAXC * dim, randF); fill(B.w, QSA_PROJ * dim / 2, randH); fill(B.proj, MAXC * QSA_PROJ, (n) => randF(n, 2));
fill(B.q, MAXC * qDim, (n) => randF(n, 4)); fill(B.kCache, MAXSEQ * kvDim / 2, randH); fill(B.vCache, MAXSEQ * kvDim / 2, randH);
fill(B.qNorm, 128, (n) => randF(n).map((v) => 1 + v)); fill(B.kNorm, 128, (n) => randF(n).map((v) => 1 + v));
const G = qsaBindGroups(device, K_, B);
const denseMod = device.createShaderModule({ code: attnDecWGSL(adCfg) });
const l1 = (types) => device.createBindGroupLayout({ entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
const dl = l1(["read-only-storage", "read-only-storage", "read-only-storage", "storage", "storage", "uniform"]), dcl = l1(["read-only-storage", "read-only-storage", "storage", "uniform"]);
const pD = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [K_.layout0, dl] }), compute: { module: denseMod, entryPoint: "attn_dec" } });
const pDC = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [K_.layout0, dcl] }), compute: { module: denseMod, entryPoint: "attn_dec_combine" } });
const gD = device.createBindGroup({ layout: dl, entries: [B.q, B.kCache, B.vCache, B.faO, B.faML, B.fd].map((b, binding) => ({ binding, resource: { buffer: b } })) });
const gDC = device.createBindGroup({ layout: dcl, entries: [B.faO, B.faML, B.out, B.fd].map((b, binding) => ({ binding, resource: { buffer: b } })) });
const g0 = G.qsa_prep[0];

const qset = device.createQuerySet({ type: "timestamp", count: 2 });
const qres = mk(16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC);
const qread = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
// ms per run of fn(pass), REPS runs per timed pass, best of 3
async function time(fn) {
  let best = Infinity;
  for (let k = 0; k < 3; k++) {
    const e = device.createCommandEncoder();
    const p = e.beginComputePass({ timestampWrites: { querySet: qset, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
    for (let r = 0; r < REPS; r++) fn(p);
    p.end(); e.resolveQuerySet(qset, 0, 2, qres, 0); e.copyBufferToBuffer(qres, 0, qread, 0, 16);
    device.queue.submit([e.finish()]);
    await qread.mapAsync(GPUMapMode.READ); const t = new BigUint64Array(qread.getMappedRange().slice(0)); qread.unmap();
    best = Math.min(best, Number(t[1] - t[0]) / 1e6 / REPS);
  }
  return best;
}
const setup = (pos, nc, S, slots) => {
  device.queue.writeBuffer(B.frame, 0, new Uint32Array([pos, pos + 1, nc, 0]));
  device.queue.writeBuffer(B.fd, 0, new Uint32Array([qDim, qDim, S, slots]));
};
const f = (x) => (x * 1000).toFixed(0).padStart(5) + " us";
// pool / prep over the whole context once so the pooled keys are realistic
async function prime(ctx) {
  for (let pos = 0; pos < ctx; pos += MAXC) {
    const nc = Math.min(MAXC, ctx - pos); setup(pos, nc, adCfg.S, adCfg.S);
    const e = device.createCommandEncoder(), p = e.beginComputePass();
    encodeQsa(p, pipes, G, qsaDispatch({ pos, nCols: nc, adCfg, nH, nKVa }), { stages: ["prep", "pool"] });
    p.end(); device.queue.submit([e.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
}
const stageRun = (d, s) => (p) => encodeQsa(p, pipes, G, d, { stages: [s] });

console.log(`QSA per layer, ${ad.info?.description || ad.info?.device || "GPU"}; REPS ${REPS}`);
for (const nc of [1]) {
  setup(0, nc, adCfg.S, adCfg.S);
  const d = qsaDispatch({ pos: 0, nCols: nc, adCfg, nH, nKVa });
  console.log(`indexer projection qsa_proj (640 x 2560 BF16), 1 column: ${f(await time(stageRun(d, "proj")))}`);
}
{
  setup(0, COLS, PSPLIT, slotsP);
  const d = qsaDispatch({ pos: 0, nCols: COLS, adCfg, nH, nKVa });
  console.log(`indexer projection qsa_proj, ${COLS} columns: ${f(await time(stageRun(d, "proj")))}`);
}
for (const ctx of CTXS) {
  await prime(ctx);
  // decode at position ctx - 1
  const pos = ctx - 1;
  setup(pos, 1, adCfg.S, adCfg.S);
  const d = qsaDispatch({ pos, nCols: 1, adCfg, nH, nKVa });
  const t = {};
  for (const s of ["prep", "pool", "score", "select", "attn"]) t[s] = await time(stageRun(d, s));
  const tot = Object.values(t).reduce((a, b) => a + b, 0);
  const dense = await time((p) => {
    p.setPipeline(pD); p.setBindGroup(0, g0); p.setBindGroup(1, gD); p.dispatchWorkgroups(decSplits(ctx, adCfg.S), 1, nKVa);
    p.setPipeline(pDC); p.setBindGroup(0, g0); p.setBindGroup(1, gDC); p.dispatchWorkgroups(nH * DEC_CQ, 1, 1);
  });
  const all = await time((p) => encodeQsa(p, pipes, G, d, { stages: ["prep", "pool", "score", "select", "attn"] }));
  console.log(`decode  ctx ${String(ctx).padStart(6)}: prep ${f(t.prep)} pool ${f(t.pool)} score ${f(t.score)} select ${f(t.select)} attn ${f(t.attn)} | sum ${f(tot)}, one pass ${f(all)} | dense attn_dec ${f(dense)}`);
  // prefill chunk of COLS columns ending at ctx
  if (ctx >= COLS) {
    const pp = ctx - COLS;
    setup(pp, COLS, PSPLIT, slotsP);
    const dp = qsaDispatch({ pos: pp, nCols: COLS, adCfg: { ...adCfg, S: PSPLIT }, nH, nKVa });
    const tp = {};
    for (const s of ["prep", "pool", "score", "select", "attn"]) tp[s] = await time(stageRun(dp, s));
    const totp = Object.values(tp).reduce((a, b) => a + b, 0);
    console.log(`prefill ctx ${String(ctx).padStart(6)}, ${COLS} cols: prep ${f(tp.prep)} pool ${f(tp.pool)} score ${f(tp.score)} select ${f(tp.select)} attn ${f(tp.attn)} | sum ${f(totp)} = ${(totp * 1000 / COLS).toFixed(1)} us/token`);
  }
}
if (gpuErr) { console.log("BENCH FAIL: GPU errors"); Deno.exit(1); }
