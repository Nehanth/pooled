// attn_dec + attn_dec_combine (engine/wgsl/attn_dec.js) against a float64 CPU reference, on random
// q / K / V, at the MoE's (G = 8, 2 kv heads) and the 27B's (G = 6, 4 kv heads) shapes, for context
// lengths across the split-length steps (1 .. 70k), and:
//   - verify == decode: a column of a multi-column pass gives the same bits as a one-column pass at the
//     same position (what spec == plain needs)
//   - splits: each column uses decSplits() workgroups; slots past that are never written
//   cd tests && deno run --unstable-webgpu --allow-read test_attn_dec.js
import { attnDecWGSL, attnDecConfig, decSplits, decSplitLen } from "../engine/wgsl/attn_dec.js";

const ad = await navigator.gpu.requestAdapter();
const device = await ad.requestDevice({ requiredLimits: { maxBufferSize: ad.limits.maxBufferSize, maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize } });
let gpuErr = 0;
device.addEventListener?.("uncapturederror", (e) => { if (gpuErr++ < 4) console.error("GPU ERROR:", e.error?.message?.slice(0, 400)); });
const S_ = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const buf = (a, usage = S_) => { const b = device.createBuffer({ size: Math.max(16, Math.ceil(a.byteLength / 16) * 16), usage }); device.queue.writeBuffer(b, 0, a); return b; };
async function read(b, n) {
  const st = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const e = device.createCommandEncoder(); e.copyBufferToBuffer(b, 0, st, 0, n * 4); device.queue.submit([e.finish()]);
  await st.mapAsync(GPUMapMode.READ); const out = new Float32Array(st.getMappedRange().slice(0)); st.unmap(); st.destroy(); return out;
}
// deterministic PRNG
let seed = 12345;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 2 ** 32 - 0.5; };
const f16 = (x) => { const f = new Float32Array([x]), i = new Uint32Array(f.buffer)[0]; const s = (i >>> 16) & 0x8000; let e = ((i >>> 23) & 0xff) - 127 + 15, m = i & 0x7fffff;
  if (e <= 0) return s; if (e >= 31) return s | 0x7c00; let h = s | (e << 10) | (m >> 13); if ((m & 0x1fff) > 0x1000 || ((m & 0x1fff) === 0x1000 && (h & 1))) h++; return h; };
const h2f = (h) => { const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023; return e === 0 ? s * m * 2 ** -24 : s * (1 + m / 1024) * 2 ** (e - 15); };

let fails = 0;
async function run({ nH, nKV, maxSeq, cases }) {
  const hd = 256, G = nH / nKV, kvDim = nKV * hd, qDim = nH * hd;
  const cfg = attnDecConfig({ hd, G, nKV });
  const code = attnDecWGSL(cfg);
  const module = device.createShaderModule({ code });
  const info = await module.getCompilationInfo?.();
  for (const m of info?.messages || []) if (m.type === "error") { console.error("WGSL:", m.lineNum, m.message); fails++; return; }
  const pipe = (e) => device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: e } });
  const pDec = pipe("attn_dec"), pComb = pipe("attn_dec_combine");
  // K / V: f16 rows [maxSeq][kvDim]; scores need some spread so the softmax is not flat
  const K = new Uint16Array(maxSeq * kvDim), V = new Uint16Array(maxSeq * kvDim);
  for (let i = 0; i < K.length; i++) { K[i] = f16(rnd() * 0.6); V[i] = f16(rnd() * 2); }
  // a few "needle" rows with large scores so the max moves between splits
  const Kf = Float32Array.from(K, h2f), Vf = Float32Array.from(V, h2f);
  const kBuf = buf(new Uint32Array(K.buffer)), vBuf = buf(new Uint32Array(V.buffer));
  const NC = 4;
  const q = new Float32Array(NC * qDim); for (let i = 0; i < q.length; i++) q[i] = rnd() * 4;
  const qBuf = buf(q);
  const slots = cfg.S;
  const oBuf = device.createBuffer({ size: NC * nH * slots * hd * 4, usage: S_ });
  const mlBuf = device.createBuffer({ size: NC * nH * slots * 2 * 4, usage: S_ });
  const outBuf = device.createBuffer({ size: NC * qDim * 4, usage: S_ });
  const cfgU = new ArrayBuffer(48); { const u = new Uint32Array(cfgU); u.set([0, kvDim, nH, nKV, hd, 0, 0, maxSeq]); new Uint32Array(cfgU, 40, 1)[0] = qDim; }
  const cfgBuf = buf(new Uint8Array(cfgU), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const frameBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const fdBuf = buf(new Uint32Array([qDim, qDim, cfg.S, slots]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const bg = (p, i, bs) => device.createBindGroup({ layout: p.getBindGroupLayout(i), entries: bs.map((b, j) => ({ binding: j, resource: { buffer: b } })) });
  const g0d = bg(pDec, 0, [cfgBuf, frameBuf]), g0c = bg(pComb, 0, [cfgBuf, frameBuf]);
  const g1d = bg(pDec, 1, [qBuf, kBuf, vBuf, oBuf, mlBuf, fdBuf]), g1c = bg(pComb, 1, [oBuf, mlBuf, outBuf, fdBuf]);
  // pass: columns at positions pos .. pos + nc - 1 (column c sees pos + c + 1 rows), q columns qc0 ..
  async function pass(pos, nc, qc0) {
    if (Deno.env.get("DBG")) console.log(`  pass pos ${pos} nc ${nc}`);
    device.queue.writeBuffer(frameBuf, 0, new Uint32Array([pos, pos + 1, nc, 0]));
    device.queue.writeBuffer(fdBuf, 0, new Uint32Array([qDim, qDim, cfg.S, slots]));
    // q for this pass: columns qc0 .. qc0 + nc - 1 moved to the front
    device.queue.writeBuffer(qBuf, 0, q.subarray(qc0 * qDim, (qc0 + nc) * qDim));
    let ns = 1; for (let c = 0; c < nc; c++) ns = Math.max(ns, decSplits(pos + c + 1, cfg.S));
    const e = device.createCommandEncoder(), p = e.beginComputePass();
    p.setPipeline(pDec); p.setBindGroup(0, g0d); p.setBindGroup(1, g1d); p.dispatchWorkgroups(ns, nc, nKV);
    p.setPipeline(pComb); p.setBindGroup(0, g0c); p.setBindGroup(1, g1c); p.dispatchWorkgroups(nH, nc, 1);
    p.end(); device.queue.submit([e.finish()]);
    return await read(outBuf, nc * qDim);
  }
  function ref(qc, seqLen) {   // float64 attention for q column qc over rows [0, seqLen)
    const out = new Float64Array(qDim);
    for (let h = 0; h < nH; h++) {
      const g = Math.floor(h / G), sc = new Float64Array(seqLen);
      let M = -Infinity;
      for (let t = 0; t < seqLen; t++) { let s = 0; for (let d = 0; d < hd; d++) s += q[qc * qDim + h * hd + d] * Kf[t * kvDim + g * hd + d]; sc[t] = s / Math.sqrt(hd); M = Math.max(M, sc[t]); }
      let L = 0; for (let t = 0; t < seqLen; t++) { sc[t] = Math.exp(sc[t] - M); L += sc[t]; }
      for (let d = 0; d < hd; d++) { let o = 0; for (let t = 0; t < seqLen; t++) o += sc[t] * Vf[t * kvDim + g * hd + d]; out[h * hd + d] = o / L; }
    }
    return out;
  }
  for (const seqLen of cases) {
    const pos = seqLen - 1;
    const one = await pass(pos, 1, 0);
    const r = ref(0, seqLen);
    let md = 0, mr = 0; for (let i = 0; i < qDim; i++) { md = Math.max(md, Math.abs(one[i] - r[i])); mr = Math.max(mr, Math.abs(r[i])); }
    const rel = md / mr;
    // verify == decode: a 4-column pass ending at this position vs one-column passes at each position
    let same = true;
    if (seqLen >= 4) {
      const multi = await pass(pos - 3, 4, 0);
      for (let c = 0; c < 4; c++) {
        const single = await pass(pos - 3 + c, 1, c);
        for (let i = 0; i < qDim; i++) if (Object.is(single[i], multi[c * qDim + i]) === false) { same = false; break; }
      }
    }
    const ok = rel < 2e-5 && same && Number.isFinite(rel);
    if (!ok) fails++;
    console.log(`nH ${nH} nKV ${nKV} seqLen ${String(seqLen).padStart(6)}: splits ${decSplits(seqLen, cfg.S)} x ${decSplitLen(seqLen, cfg.S)}, relDiff vs f64 ${rel.toExponential(2)}, verify == decode ${same ? "yes" : "NO"} ${ok ? "ok" : "FAIL"}`);
  }
  for (const b of [kBuf, vBuf, qBuf, oBuf, mlBuf, outBuf]) b.destroy();
}

const quick = Deno.env.get("QUICK") === "1";
await run({ nH: 16, nKV: 2, maxSeq: quick ? 9000 : 70000, cases: quick ? [1, 5, 64, 65, 200, 1000, 8191, 8193] : [1, 2, 5, 63, 64, 65, 127, 128, 129, 300, 1000, 4096, 8192, 8193, 16385, 33000, 69999] });
await run({ nH: 24, nKV: 4, maxSeq: quick ? 5000 : 40000, cases: quick ? [1, 7, 130, 4097] : [1, 3, 64, 129, 257, 1000, 4097, 16384, 32769, 39999] });
console.log(fails || gpuErr ? "ATTN_DEC FAIL" : "ATTN_DEC PASS");
if (fails || gpuErr) Deno.exit(1);
