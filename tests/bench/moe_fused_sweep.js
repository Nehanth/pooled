// Fused MoE kernel A/B (engine/wgsl/moe.js moeFusedWGSL: moe_route, moe_gus, moe_dnc) at the Qwen3.6-35B-A3B shape:
// synthetic Q4_0 routed experts (256 x [512][2048] gate and up, 256 x [2048][512] down), Q8_0 shared expert (512 wide),
// top-8. One compute pass per (kernel, layout, round) runs LAYERS launches of that kernel, each on its own random
// routing row (8 distinct experts), so the expert reads come from DRAM as in decode; timestamps at the pass
// boundaries, so the pass overhead is spread over LAYERS launches. Layouts are interleaved round by round, so a
// busy or throttling GPU slows every layout alike. Prints the median µs per launch and the effective GB/s.
//   cd tests && deno run --unstable-webgpu --allow-read --allow-env bench/moe_fused_sweep.js
//   env: LAYOUTS='["legacy","wide",{"gu":{"TPR":8}}]' (moeFusedLayout values; default legacy + wide), ROUNDS (15), LAYERS (40),
//        REF=<path to another engine/wgsl/moe.js> (adds its legacy kernels as "ref", e.g. origin/main's moe_route)
import { moeFusedWGSL, moeFusedLayout } from "../../engine/wgsl/moe.js";
const env = (k, d) => Deno.env.get(k) ?? d;
const dim = 2048, ei = 512, sDim = 512, nExp = 256, K = 8, KS = K + 1, hs = Math.max(ei, sDim);
const ROUNDS = +env("ROUNDS", 15), LAYERS = +env("LAYERS", 40);
const ad = await navigator.gpu.requestAdapter();
const device = await ad.requestDevice({ requiredFeatures: ["timestamp-query"],
  requiredLimits: { maxBufferSize: ad.limits.maxBufferSize, maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize } });
device.addEventListener?.("uncapturederror", (e) => console.error("GPU ERROR:", e.error?.message));
const U = GPUBufferUsage;
let seed = 99;
const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0);
// random words, or a constant f16 pair (scales: small, so nothing overflows over many launches)
const fill = (bytes, scale) => {
  const b = device.createBuffer({ size: Math.ceil(bytes / 16) * 16, usage: U.STORAGE | U.COPY_DST });
  for (let off = 0; off < bytes; off += 1 << 24) {
    const a = new Uint32Array(Math.min(1 << 24, bytes - off) / 4);
    if (scale) a.fill(0x14001400); else for (let i = 0; i < a.length; i++) a[i] = next();
    device.queue.writeBuffer(b, off, a);
  }
  return b;
};
const q4 = (rows, dIn) => rows * (dIn / 32) * 16, q8 = (rows, dIn) => rows * (dIn / 32) * 32, sc = (rows, dIn) => rows * (dIn / 32) * 2;
const W = { gq: fill(q4(nExp * ei, dim)), gs: fill(sc(nExp * ei, dim), true), uq: fill(q4(nExp * ei, dim)), us: fill(sc(nExp * ei, dim), true),
  dq: fill(q4(nExp * dim, ei)), ds: fill(sc(nExp * dim, ei), true), sd: fill(q8(dim, sDim)), sds: fill(sc(dim, sDim), true) };
// shared gate/up packed as engine/qwen35.js packGU: gate qs | up qs | gate scales | up scales
const oUq = q8(sDim, dim) / 4, oGs = oUq * 2, oUs = oGs + sc(sDim, dim) / 4;
const sh = device.createBuffer({ size: (oUs + sc(sDim, dim) / 4) * 4, usage: U.STORAGE | U.COPY_DST });
{ const a = new Uint32Array(oUs + sc(sDim, dim) / 4); for (let i = 0; i < oGs; i++) a[i] = next(); a.fill(0x14001400, oGs); device.queue.writeBuffer(sh, 0, a); }
const f32 = (n, f, usage = U.STORAGE | U.COPY_DST) => { const b = device.createBuffer({ size: n * 4, usage }); device.queue.writeBuffer(b, 0, Float32Array.from({ length: n }, (_, i) => f(i))); return b; };
const x = f32(dim, (i) => Math.sin(i) * 0.5), h = f32(KS * hs, (i) => Math.cos(i) * 0.5), xr = f32(dim, () => 0);
const logits = f32(nExp + 1, (i) => Math.sin(i * 7.1) * 3), rsel = device.createBuffer({ size: 256, usage: U.STORAGE }), rw = device.createBuffer({ size: 256, usage: U.STORAGE });
// LAYERS routing rows (K distinct experts + the shared slot) and their weights, each in its own 256-byte slot
const sel = device.createBuffer({ size: LAYERS * 256, usage: U.STORAGE | U.COPY_DST }), wts = f32(LAYERS * 64, () => 0.1);
{ const a = new Uint32Array(LAYERS * 64); for (let s = 0; s < LAYERS; s++) { const pick = new Set(); while (pick.size < K) pick.add(next() % nExp); [...pick].forEach((e, k) => a[s * 64 + k] = e); } device.queue.writeBuffer(sel, 0, a); }
const uni = (a) => { const b = device.createBuffer({ size: 48, usage: U.UNIFORM | U.COPY_DST }); device.queue.writeBuffer(b, 0, new Uint32Array(a)); return b; };
const uG = uni([ei, dim, sDim, nExp, dim, hs, 1, 1, oUq, oGs, oUs, 0]), uD = uni([dim, ei, sDim, nExp, dim, hs, 1, 1, 0, 0, 0, 0]), uR = uni([0, 0, 0, nExp, nExp + 1, 0, 1, 1, 0, 0, 0, 0]);

const REF = env("REF", "");
const refMod = REF ? await import(new URL(REF, `file://${Deno.cwd()}/`).href) : null;
const variants = [...JSON.parse(env("LAYOUTS", '["legacy","wide"]')).map((l) => ({ name: typeof l === "string" ? l : JSON.stringify(l), lay: moeFusedLayout(l, K), gen: moeFusedWGSL })),
  ...(refMod ? [{ name: "ref", lay: null, gen: refMod.moeFusedWGSL }] : [])];
for (const v of variants) {
  device.pushErrorScope("validation");
  const mod = device.createShaderModule({ code: v.gen({ K, R: 1, layout: v.lay, gu: [["q4", "q8"]], dn: [["q4", "q8"]] }) });
  const pipe = (e) => device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: e } });
  const pr = pipe("moe_route"), pg = pipe("moe_gus_q4_q8"), pd = pipe("moe_dnc_q4_q8");
  const err = await device.popErrorScope();
  if (err) throw new Error(`${v.name}: ${err.message}`);
  const bg = (p, bufs) => device.createBindGroup({ layout: p.getBindGroupLayout(1), entries: bufs.map((b, i) => ({ binding: i, resource: b.buffer ? b : { buffer: b } })) });
  const S = (l) => ({ buffer: sel, offset: l * 256, size: KS * 4 }), Wt = (l) => ({ buffer: wts, offset: l * 256, size: KS * 4 });
  const gRows = v.lay ? v.lay.gu.rows : 4, dRows = v.lay ? v.lay.dn.rows : 1;
  v.kernels = {
    route: { pipe: pr, bgs: [bg(pr, [logits, rsel, rw, uR])], grid: [1, 1], bytes: 0 },
    gus: { pipe: pg, bgs: Array.from({ length: LAYERS }, (_, l) => bg(pg, [W.gq, W.gs, W.uq, W.us, x, h, S(l), sh, uG])), grid: [Math.ceil(hs / gRows), KS],
      bytes: K * 2 * ei * (dim / 32) * 18 + 2 * sDim * (dim / 32) * 34 },
    dnc: { pipe: pd, bgs: Array.from({ length: LAYERS }, (_, l) => bg(pd, [W.dq, W.ds, h, xr, S(l), Wt(l), W.sd, W.sds, uD])), grid: [Math.ceil(dim / dRows), 1],
      bytes: K * dim * (ei / 32) * 18 + dim * (sDim / 32) * 34 },
  };
}
const kinds = ["gus", "dnc", "route"], NQ = 2 * ROUNDS * variants.length * kinds.length;
const qs = device.createQuerySet({ type: "timestamp", count: NQ });
const res = device.createBuffer({ size: NQ * 8, usage: U.QUERY_RESOLVE | U.COPY_SRC }), rd = device.createBuffer({ size: NQ * 8, usage: U.COPY_DST | U.MAP_READ });
const slots = [];
const enc = device.createCommandEncoder();
for (let r = -2; r < ROUNDS; r++) for (const v of variants) for (const k of kinds) {   // rounds -2, -1: warm-up (untimed)
  const q = r >= 0 ? slots.length * 2 : -1;
  if (q >= 0) slots.push({ v: v.name, k });
  const kk = v.kernels[k], p = enc.beginComputePass(q >= 0 ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: q, endOfPassWriteIndex: q + 1 } } : undefined);
  p.setPipeline(kk.pipe);
  for (let l = 0; l < LAYERS; l++) { p.setBindGroup(1, kk.bgs[l % kk.bgs.length]); p.dispatchWorkgroups(...kk.grid); }
  p.end();
}
enc.resolveQuerySet(qs, 0, NQ, res, 0); enc.copyBufferToBuffer(res, 0, rd, 0, NQ * 8);
device.queue.submit([enc.finish()]);
await rd.mapAsync(GPUMapMode.READ);
const t = new BigUint64Array(rd.getMappedRange().slice(0));
const med = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };
console.log(`fused MoE kernels, K ${K}, ${LAYERS} launches per pass, ${ROUNDS} interleaved rounds: median µs per launch (min), effective GB/s`);
for (const k of kinds) for (const v of variants) {
  const us = slots.map((s, i) => s.v === v.name && s.k === k ? Number(t[2 * i + 1] - t[2 * i]) / 1e3 / LAYERS : null).filter((x) => x !== null);
  const m = med(us), b = v.kernels[k].bytes;
  console.log(`  ${k.padEnd(5)} ${m.toFixed(2).padStart(7)} (${Math.min(...us).toFixed(2).padStart(7)})${b ? `  ${(b / (m * 1e3)).toFixed(0).padStart(4)} GB/s` : "           "}  ${v.name}`);
}
