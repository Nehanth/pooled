// Tiled prefill attention (engine/wgsl/attn_tile.js) alone at the MoE's shape (hd 256, 16 query heads, 2 KV heads):
// one 16-column prefill pass at position POS of a synthetic f16 KV cache, timed over REPS launches per variant,
// variants interleaved round by round. Variants: attnTileConfig flags (kvh, pf, nr). No model.
//   cd tests && deno run --unstable-webgpu --allow-read --allow-env bench/attn_tile_bench.js
//   env: POS=8192,16384 VARIANTS='[{},{"pf":true},{"nr":true},{"pf":true,"nr":true}]' ROUNDS=5 REPS=20 NC=16
import { attnTileConfig, attnTileWGSL } from "../../engine/wgsl/attn_tile.js";
const env = (k, d) => Deno.env.get(k) ?? d;
const hd = 256, nH = 16, nKV = 2, G = nH / nKV, kvDim = nKV * hd, NC = +env("NC", 16);
const POS = env("POS", "8192,16384").split(",").map(Number), ROUNDS = +env("ROUNDS", 5), REPS = +env("REPS", 20);
const VARIANTS = JSON.parse(env("VARIANTS", '[{},{"pf":true},{"nr":true},{"pf":true,"nr":true}]'));
const maxSeq = Math.max(...POS) + NC + 64;
const faSplit = Math.max(256, Math.ceil(maxSeq / 128 / 64) * 64), faSplits = Math.ceil(maxSeq / faSplit);   // engine/qwen35.js sizing
const ad = await navigator.gpu.requestAdapter();
const device = await ad.requestDevice({ requiredLimits: { maxBufferSize: ad.limits.maxBufferSize, maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize } });
device.addEventListener?.("uncapturederror", (e) => console.error("GPU ERROR:", e.error?.message));
const BU = GPUBufferUsage;
let seed = 7; const rnd = () => { seed = (Math.imul(seed, 1103515245) + 12345) >>> 0; return (seed >>> 8) / 16777216; };
const buf = (n, f, usage = BU.STORAGE | BU.COPY_DST) => { const b = device.createBuffer({ size: Math.ceil(n * 4 / 16) * 16, usage }); if (f) { const a = new Uint32Array(n); for (let i = 0; i < n; i++) a[i] = f(i); device.queue.writeBuffer(b, 0, a); } return b; };
const f32bits = (x) => new Uint32Array(new Float32Array([x]).buffer)[0];
const h16 = () => 0x3000 + (rnd() * 0x800 | 0) | (rnd() < 0.5 ? 0x8000 : 0);   // small f16 values
const kc = buf(maxSeq * kvDim / 2, () => h16() | (h16() << 16)), vc = buf(maxSeq * kvDim / 2, () => h16() | (h16() << 16));
const s0 = nH * hd, q = buf(NC * s0, () => f32bits((rnd() - 0.5) * 2));
const faO = buf(NC * nH * faSplits * hd), faML = buf(NC * nH * faSplits * 2), out = buf(NC * s0, null, BU.STORAGE | BU.COPY_SRC);
const uni = (a, size = 16) => { const b = device.createBuffer({ size, usage: BU.UNIFORM | BU.COPY_DST }); device.queue.writeBuffer(b, 0, new Uint32Array(a)); return b; };
const cfgU = uni([2048, kvDim, nH, nKV, hd, 512, 248320, maxSeq, f32bits(1e-6), f32bits(1e7), nH * hd, 0], 48);
const faU = uni([s0, s0, faSplit, faSplits]);
const frames = Object.fromEntries(POS.map((p) => [p, uni([p, p + 1, NC, 0])]));
const vs = VARIANTS.map((o) => {
  const c = attnTileConfig({ hd, G, faSplit, faSplits, ...o });
  device.pushErrorScope("validation");
  const mod = device.createShaderModule({ code: attnTileWGSL(c) });
  const pf = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "attn_flash_tile" } });
  const pc = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "attn_combine_tile" } });
  const bg = (p, g, rs) => device.createBindGroup({ layout: p.getBindGroupLayout(g), entries: rs.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
  return { name: JSON.stringify(o), c, pf, pc, g0f: Object.fromEntries(POS.map((p) => [p, bg(pf, 0, [cfgU, frames[p]])])), g0c: Object.fromEntries(POS.map((p) => [p, bg(pc, 0, [cfgU, frames[p]])])),
    g1f: bg(pf, 1, [q, kc, vc, faO, faML, faU]), g1c: bg(pc, 1, [faO, faML, out, faU]), t: {}, err: device.popErrorScope() };
});
for (const v of vs) { const e = await v.err; if (e) throw new Error(`${v.name}: ${e.message}`); }
const time = async (v, p) => {
  const enc = device.createCommandEncoder(), ps = enc.beginComputePass();
  for (let r = 0; r < REPS; r++) {
    ps.setPipeline(v.pf); ps.setBindGroup(0, v.g0f[p]); ps.setBindGroup(1, v.g1f); ps.dispatchWorkgroups(v.c.FASPLITS, nKV, Math.ceil(NC / v.c.CW));
    ps.setPipeline(v.pc); ps.setBindGroup(0, v.g0c[p]); ps.setBindGroup(1, v.g1c); ps.dispatchWorkgroups(nH, NC);
  }
  ps.end(); await device.queue.onSubmittedWorkDone();
  const t0 = performance.now(); device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone();
  return (performance.now() - t0) / REPS;
};
// outputs must agree bit for bit between variants (same numerics by construction)
const rd = device.createBuffer({ size: NC * s0 * 4, usage: BU.COPY_DST | BU.MAP_READ });
const outs = [];
for (const v of vs) { await time(v, POS[0]); const e = device.createCommandEncoder(); e.copyBufferToBuffer(out, 0, rd, 0, NC * s0 * 4); device.queue.submit([e.finish()]); await rd.mapAsync(GPUMapMode.READ); outs.push(new Uint32Array(rd.getMappedRange().slice(0))); rd.unmap(); }
for (let i = 1; i < vs.length; i++) console.log(`${vs[i].name} output ${outs[i].every((x, j) => x === outs[0][j]) ? "== " : "!= "}${vs[0].name}`);
for (let r = 0; r < ROUNDS; r++) for (const p of POS) for (const v of vs) (v.t[p] ||= []).push(await time(v, p));
const med = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];
for (const v of vs) console.log(`${v.name.padEnd(28)} TK ${v.c.TK} ` + POS.map((p) => `pos ${p}: ${(med(v.t[p]) * 1000).toFixed(0)} µs`).join(" · "));
