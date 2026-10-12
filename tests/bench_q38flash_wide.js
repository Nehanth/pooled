// Timings of the wide Qwen3.8-Flash-Next prefill kernels (engine/wgsl/qwen4exp_wide.js) on random data at the
// model's shapes, one 512-column pass: the tiled GEMM per projection shape and the grouped MoE (top-10 of 512).
//   deno run --unstable-webgpu --allow-read --allow-env tests/bench_q38flash_wide.js      [N=512 REPS=10]
// DP4A=1: the dp4a GEMM (with its activation quant) and MoE kernels; TILE / DTILE: GEMM tiles (JSON); ONLY=gemm|moe|attn
import * as X from "../engine/wgsl/qwen4exp_wide.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const ad = await navigator.gpu.requestAdapter();
const device = await ad.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize, maxBufferSize: ad.limits.maxBufferSize } });
device.addEventListener?.("uncapturederror", (e) => console.error("GPU ERROR:", e.error?.message?.slice(0, 300)));
const SU = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const mk = (bytes) => device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: SU });
const rndBuf = (bytes) => { const b = mk(bytes), a = new Uint32Array(Math.ceil(bytes / 4)); for (let i = 0; i < a.length; i++) a[i] = (Math.random() * 2 ** 32) >>> 0; device.queue.writeBuffer(b, 0, a); return b; };
const fBuf = (n) => { const b = mk(n * 4), a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = Math.random() - 0.5; device.queue.writeBuffer(b, 0, a); return b; };
const scBuf = (nBlk) => { const b = mk(nBlk * 2), a = new Uint16Array(Math.ceil(nBlk / 2) * 2).fill(0x2000); device.queue.writeBuffer(b, 0, a); return b; };
const uni = (fields) => {
  const ab = new ArrayBuffer(Math.max(16, Math.ceil(fields.length / 4) * 16)), dv = new DataView(ab);
  fields.forEach((v, i) => (typeof v === "number" ? dv.setUint32(i * 4, v, true) : dv.setFloat32(i * 4, v.f, true)));
  const b = device.createBuffer({ size: ab.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, ab); return b;
};
const pipe = (code) => device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
const bgOf = (p, list) => device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: list.map((b, i) => (b ? { binding: i, resource: { buffer: b } } : null)).filter(Boolean) });
async function time(jobs, reps = +env("REPS", 10)) {   // jobs: [{ p, bg, wg }] -> ms per run (all jobs)
  const go = (k) => { const e = device.createCommandEncoder(); for (let r = 0; r < k; r++) { const pass = e.beginComputePass(); for (const j of jobs) { pass.setPipeline(j.p); pass.setBindGroup(0, j.bg); pass.dispatchWorkgroups(...j.wg); } pass.end(); } device.queue.submit([e.finish()]); return device.queue.onSubmittedWorkDone(); };
  await go(1);
  const t = performance.now(); await go(reps); return (performance.now() - t) / reps;
}
const N = +env("N", 512), d = 2560, F = uni([0, N, 0, 0]);
const TL = env("TILE") ? { ...X.WIDE_T, ...JSON.parse(env("TILE")) } : X.WIDE_T;   // TILE='{"BM":128,"BN":128,"TM":8,"TN":8,"KH":1}'
const ONLY = env("ONLY");   // gemm | moe | attn
const rows = [];
// projections: [name, kind, dOut, dIn, launches per layer-average]
for (const [name, kind, dOut, dIn] of [["qkvz", "q4", 16384, 2560], ["ssm_out", "q8", 2560, 6144], ["attn_q+k+v", "q8", 13312, 2560], ["attn_o", "q4", 2560, 6144],
  ["hc_down", "q4", 320, 10240], ["hc_up", "q4", 10240, 320], ["router", "f32", 513, 2560], ["shexp_gate", "q8", 640, 2560], ["shexp_down", "q8", 2560, 640]]) {
  const nb = dOut * dIn / 32, w = kind === "f32" ? [fBuf(dOut * dIn), null] : [rndBuf(nb * (kind === "q4" ? 16 : 32)), scBuf(nb)];
  const x = fBuf(N * dIn), y = mk(N * dOut * 4), aux = mk(16);
  if (ONLY && ONLY !== "gemm") break;
  const u = uni([dOut, dIn, 0, dIn / 4, 0, dOut, 0, { f: 0 }, 1 << 30, 0, 0, 0]);
  let ms;
  if (env("DP4A") === "1" && kind !== "f32") {   // the dp4a GEMM (+ the activation quant, timed with it)
    const DT = env("DTILE") ? { ...X.DP4A_T, ...JSON.parse(env("DTILE")) } : X.DP4A_T;
    const pq = pipe(X.QUANT_X_WGSL), p = pipe(X.gemmDWGSL(kind, DT)), xq = mk(N * dIn), xd = mk(N * dIn / 8);
    ms = await time([{ p: pq, bg: bgOf(pq, [x, xq, xd, uni([dIn, 0, dIn / 4, 0]), F]), wg: [Math.ceil(dIn / 1024), N, 1] },
      { p, bg: bgOf(p, [...w, xq, xd, y, u, F, aux]), wg: [Math.ceil(dOut / DT.BM), Math.ceil(N / DT.BN), 1] }]);
  } else {
    const p = pipe(X.gemmTWGSL(kind, TL));
    ms = await time([{ p, bg: bgOf(p, [...w, x, y, u, F, aux]), wg: [Math.ceil(dOut / TL.BM), Math.ceil(N / TL.BN), 1] }]);
  }
  const mac = dOut * dIn * N;
  rows.push(`  gemm_t ${name.padEnd(12)} ${kind} ${String(dOut).padStart(5)} x ${String(dIn).padStart(5)}: ${ms.toFixed(3).padStart(7)} ms  ${(mac / ms / 1e9).toFixed(2)} TMAC/s`);
}
// grouped MoE: N columns x top-10 of 512 (uniform random picks), gate/up q4, down q8
if (!ONLY || ONLY === "moe") {
  const K = 10, nExp = 512, ff = 640, S = X.moeSortWSizes({ maxCols: N, K, nExp });
  const ids = new Uint32Array(N * K);
  for (let c = 0; c < N; c++) { const s = new Set(); while (s.size < K) s.add(Math.floor(Math.random() * nExp)); ids.set([...s], c * K); }
  const idb = mk(ids.byteLength); device.queue.writeBuffer(idb, 0, ids);
  const nbG = nExp * ff * d / 32, g = [rndBuf(nbG * 16), scBuf(nbG)], u = [rndBuf(nbG * 16), scBuf(nbG)], dn = [rndBuf(nbG * 32), scBuf(nbG)];
  const grp = mk(S.words * 4), x = fBuf(N * d), h = mk(N * K * ff * 4), y = mk(N * K * d * 4);
  const D = env("DP4A") === "1", KBG = +env("KBG", 4), KBD = +env("KBD", 4);
  const RWG = D ? X.MOE_RC : X.MOE_RW;
  const ps = pipe(X.moeSortWWGSL()), pg = pipe(D ? X.moeGuCWGSL("q4", KBG) : X.moeGuWWGSL("q4")), pd = pipe(D ? X.moeDnCWGSL("q8", KBD) : X.moeDnWWGSL("q8"));
  const tiles = Math.ceil(N * K / X.MOE_BP) + Math.min(nExp, N * K);
  const xq = mk(N * d), xd = mk(N * d / 8), hq = mk(N * K * ff), hd = mk(N * K * ff / 8);
  const js = { p: ps, bg: bgOf(ps, [idb, grp, uni([nExp, K, S.PO, 0]), F]), wg: [1, 1, 1] };
  const jg = { p: pg, bg: bgOf(pg, D ? [...g, ...u, xq, xd, h, grp, uni([ff, d, K, S.PO, ff, 0, 0, 0])] : [...g, ...u, x, h, grp, uni([ff, d, K, S.PO, d / 4, ff, 0, 0])]), wg: [Math.ceil(ff / RWG), tiles, 1] };
  const jd = { p: pd, bg: bgOf(pd, D ? [...dn, hq, hd, y, grp, uni([d, ff, K, S.PO, 0, 0, 0, 0])] : [...dn, h, y, grp, uni([d, ff, K, S.PO, 0, 0, 0, 0])]), wg: [Math.ceil(d / RWG), tiles, 1] };
  await time([js]);
  const tg = await time([jg]), td = await time([jd]), mac = N * K * ff * d;
  const nT = new Uint32Array(1); { const st = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }); const e = device.createCommandEncoder(); e.copyBufferToBuffer(grp, 0, st, 0, 16); device.queue.submit([e.finish()]); await st.mapAsync(GPUMapMode.READ); nT[0] = new Uint32Array(st.getMappedRange())[0]; st.unmap(); }
  const gb = (bytes) => (bytes * nT[0] / 1e9);   // weight bytes streamed: one expert's matrices per tile
  rows.push(`  moe gate+up q4 (grouped${D ? ", dp4a" : ""}, ${nT[0]} tiles):  ${tg.toFixed(3).padStart(7)} ms  ${(2 * mac / tg / 1e9).toFixed(2)} TMAC/s  ${(gb(2 * ff * d * 18 / 32) / tg * 1e3).toFixed(0)} GB/s of weights`);
  rows.push(`  moe down q8 (grouped${D ? ", dp4a" : ""}):     ${td.toFixed(3).padStart(7)} ms  ${(mac / td / 1e9).toFixed(2)} TMAC/s  ${(gb(ff * d * 34 / 32) / td * 1e3).toFixed(0)} GB/s of weights`);
}
// attention: one 512-column pass of one layer, dense at pos 1536 (cells up to 2048) and gathered (2048 rows per column)
if (!ONLY || ONLY === "attn") {
  const nH = 24, nKV = 2, hd = 256, qs = 2 * nH * hd + 2 * nKV * hd, stride = 2049, pos = 1536, maxSeq = 32768;
  const kc = fBuf(maxSeq * nKV * hd), vc = fBuf(maxSeq * nKV * hd), qn = fBuf(N * nH * hd), qkv = fBuf(N * qs), out = mk(N * nH * hd * 4);
  const sel = new Uint32Array(N * stride);
  for (let c = 0; c < N; c++) { sel[c * stride] = 2048; for (let i = 0; i < 2048; i++) sel[c * stride + 1 + i] = 20000 + 4 * i + (c & 3); }
  const selb = mk(sel.byteLength); device.queue.writeBuffer(selb, 0, sel);
  const AP = uni([nH, nKV, hd, 64, { f: 1e-6 }, 0, 0, 0]), MS = uni([qs, 2 * nH * hd, 2 * nH * hd + nKV * hd, 0]), Fp = uni([pos, N, 0, 0]);
  const pd = pipe(X.attnWWGSL({ G: 12 })), pg = pipe(X.attnWWGSL({ gather: stride, G: 12 }));
  const tD = await time([{ p: pd, bg: bgOf(pd, [qn, kc, vc, qkv, out, AP, Fp, MS]), wg: [nKV, Math.ceil(N / X.ATTN_W.dense.CW), 1] }]);
  const tG = await time([{ p: pg, bg: bgOf(pg, [qn, kc, vc, qkv, out, AP, Fp, MS, selb]), wg: [nKV, N, 1] }]);
  const macD = N * nH * hd * 2 * (pos + N / 2), macG = N * nH * hd * 2 * 2048;
  rows.push(`  attn_w dense (pos ${pos}):    ${tD.toFixed(3).padStart(7)} ms  ${(macD / tD / 1e9).toFixed(2)} TMAC/s`);
  rows.push(`  attn_w gathered (2048 rows): ${tG.toFixed(3).padStart(7)} ms  ${(macG / tG / 1e9).toFixed(2)} TMAC/s`);
}
console.log(`wide kernels, ${N} columns:\n${rows.join("\n")}`);
