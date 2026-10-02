// Expert-grouped MoE prefill kernels (engine/wgsl/moe_group.js) at the Qwen3.6-35B-A3B shape, without the model:
// synthetic Q4_0 routed experts (256 x [512][2048] gate and up, 256 x [2048][512] down), Q8_0 shared expert (512 wide),
// top-8 routing of U tokens (uniform, or SKEW > 0: a share of the picks from a few hot experts). Sorts once, then
// times REPS launches of each kernel (gate/up, down) per variant with onSubmittedWorkDone, variants interleaved per
// round. Prints the median ms per launch.
//   cd tests && deno run --unstable-webgpu --allow-read --allow-env bench/moe_group_sweep.js
//   env: U (256), UCS (8: the chunk sizes to try, comma list), ROUNDS (5), REPS (20), SKEW (0)
import { moeFusedWGSL } from "../../engine/wgsl/moe.js";
import { tiledGroupWGSL, moeGroupSizes, tileRows, dp4aGroupWGSL, dp4aRows } from "../../engine/wgsl/moe_group.js";
const env = (k, d) => Deno.env.get(k) ?? d;
const dim = 2048, ei = 512, sDim = 512, nExp = 256, K = 8, KS = K + 1, hs = Math.max(ei, sDim);
const U = +env("U", 256), ROUNDS = +env("ROUNDS", 5), REPS = +env("REPS", 20), SKEW = +env("SKEW", 0);
const ad = await navigator.gpu.requestAdapter();
const device = await ad.requestDevice({ requiredLimits: { maxBufferSize: ad.limits.maxBufferSize, maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize } });
device.addEventListener?.("uncapturederror", (e) => console.error("GPU ERROR:", e.error?.message));
const BU = GPUBufferUsage;
let seed = 99;
const next = () => { seed = (Math.imul(seed, 1103515245) + 12345) >>> 0; return seed >>> 8; };
const fill = (bytes, scale) => {
  const b = device.createBuffer({ size: Math.ceil(bytes / 16) * 16, usage: BU.STORAGE | BU.COPY_DST });
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
const oUq = q8(sDim, dim) / 4, oGs = oUq * 2, oUs = oGs + sc(sDim, dim) / 4;
const sh = device.createBuffer({ size: (oUs + sc(sDim, dim) / 4) * 4, usage: BU.STORAGE | BU.COPY_DST });
{ const a = new Uint32Array(oUs + sc(sDim, dim) / 4); for (let i = 0; i < oGs; i++) a[i] = next(); a.fill(0x14001400, oGs); device.queue.writeBuffer(sh, 0, a); }
const f32 = (n, f) => { const b = device.createBuffer({ size: n * 4, usage: BU.STORAGE | BU.COPY_DST | BU.COPY_SRC }); device.queue.writeBuffer(b, 0, Float32Array.from({ length: n }, (_, i) => f(i))); return b; };
const XN = f32(U * dim, (i) => Math.sin(i) * 0.5), H = f32(U * KS * hs, (i) => Math.cos(i) * 0.5), Y = f32(U * KS * dim, () => 0);
// routing: K distinct experts per token, slot K = shared (sel 0)
const selA = new Uint32Array(U * KS);
for (let c = 0; c < U; c++) { const pick = new Set(); while (pick.size < K) pick.add(SKEW && (next() % 1000) < SKEW * 1000 ? next() % 16 : next() % nExp); [...pick].forEach((e, k) => selA[c * KS + k] = e); }
const sel = device.createBuffer({ size: selA.byteLength, usage: BU.STORAGE | BU.COPY_DST }); device.queue.writeBuffer(sel, 0, selA);
{ const cnt = new Array(nExp).fill(0); for (let i = 0; i < selA.length; i++) if (i % KS !== K) cnt[selA[i]]++;
  const used = cnt.filter((c) => c).length; console.log(`U ${U}: ${used} experts used, max ${Math.max(...cnt)} pairs, mean ${(U * K / used).toFixed(1)}`); }
const uni = (a, size = 48) => { const b = device.createBuffer({ size, usage: BU.UNIFORM | BU.COPY_DST }); device.queue.writeBuffer(b, 0, new Uint32Array(a)); return b; };
const uG = uni([ei, dim, sDim, nExp, dim, hs, 1, 1, oUq, oGs, oUs, 0]), uD = uni([dim, ei, sDim, nExp, dim, hs, 1, 1, 0, 0, 0, 0]);
const bg = (p, bufs) => device.createBindGroup({ layout: p.getBindGroupLayout(1), entries: bufs.map((b, i) => ({ binding: i, resource: b.buffer ? b : { buffer: b } })) });
const MOEF = "struct MOEF { dOut: u32, dIn: u32, sDim: u32, nExp: u32, xs: u32, ys: u32, norm: u32, shOff: u32, oUq: u32, oGs: u32, oUs: u32, pad: u32 };\n";

// a variant: { name, UC, code, gx, dx } with entry points moe_gsort, moe_gusg_q4_q8, moe_dng_q4_q8 and the tiled kernels' bindings
const variants = [];
for (const UC of env("UCS", "8").split(",").map(Number))
  variants.push({ name: `tiled UC${UC}`, UC, code: MOEF + tiledGroupWGSL({ K, UC, U, nExp, gu: [["q4", "q8"]], dn: [["q4", "q8"]] }),
    gx: Math.ceil(hs / tileRows("gu")), dx: Math.ceil(dim / tileRows("dn")) });
for (const UC of env("DUCS", "").split(",").filter(Boolean).map(Number))
  variants.push({ name: `dp4a UC${UC}`, UC, dp4a: true, code: MOEF + dp4aGroupWGSL({ K, UC, U, nExp, gu: [["q4", "q8"]], dn: [["q4", "q8"]] }),
    gx: Math.ceil(hs / dp4aRows("gu")), dx: Math.ceil(dim / dp4aRows("dn")) });
if (env("REF")) {   // REF=<path to another moe_group.js> (e.g. origin/main's): its tiled kernels at UC 8
  const R = await import(new URL(env("REF"), `file://${Deno.cwd()}/`).href);
  variants.push({ name: "ref tiled UC8", UC: 8, code: MOEF + R.tiledGroupWGSL({ K, UC: 8, U, nExp, gu: [["q4", "q8"]], dn: [["q4", "q8"]] }),
    gx: Math.ceil(hs / R.tileRows("gu")), dx: Math.ceil(dim / R.tileRows("dn")) });
}
for (const v of variants) {
  device.pushErrorScope("validation");
  const mod = device.createShaderModule({ code: v.code });
  const pipe = (e) => device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: e } });
  v.ps = pipe("moe_gsort"); v.pg = pipe(v.dp4a ? "moe_gusq_q4_q8" : "moe_gusg_q4_q8"); v.pd = pipe(v.dp4a ? "moe_dnq_q4_q8" : "moe_dng_q4_q8");
  if (v.dp4a) v.pq = pipe("moe_qx");
  const err = await device.popErrorScope();
  if (err) throw new Error(`${v.name}: ${err.message}`);
  const sz = moeGroupSizes({ U, K, nExp, UC: v.UC });
  v.grp = device.createBuffer({ size: sz.words * 4, usage: BU.STORAGE });
  v.ind = device.createBuffer({ size: 32, usage: BU.STORAGE | BU.INDIRECT | BU.COPY_SRC });
  v.bs = bg(v.ps, [sel, v.grp, v.ind, uni([U * KS, nExp, v.gx, v.dx], 16)]);
  if (v.dp4a) {
    const qb = (n) => device.createBuffer({ size: n, usage: BU.STORAGE });
    v.xq = qb(U * dim * 9 / 8); v.hq = qb(U * KS * hs * 9 / 8);
    v.bqx = bg(v.pq, [XN, v.xq, uni([dim, dim / 4, U * dim / 4, 0], 16)]); v.bqh = bg(v.pq, [H, v.hq, uni([U * KS * hs, 0, U * KS * hs / 4, 0], 16)]);
    v.bgG = bg(v.pg, [W.gq, W.gs, W.uq, W.us, v.xq, H, v.grp, sh, uG]);
    v.bgD = bg(v.pd, [W.dq, W.ds, v.hq, Y, v.grp, W.sd, W.sds, uD]);
    const e2 = device.createCommandEncoder(), p2 = e2.beginComputePass();
    p2.setPipeline(v.pq); p2.setBindGroup(1, v.bqx); p2.dispatchWorkgroups(Math.ceil(dim / 32 / 32), U);
    p2.setBindGroup(1, v.bqh); p2.dispatchWorkgroups(Math.ceil(U * KS * hs / 1024)); p2.end(); device.queue.submit([e2.finish()]);
  } else {
    v.bgG = bg(v.pg, [W.gq, W.gs, W.uq, W.us, XN, H, v.grp, sh, uG]);
    v.bgD = bg(v.pd, [W.dq, W.ds, H, Y, v.grp, W.sd, W.sds, uD]);
  }
  const enc = device.createCommandEncoder(), p = enc.beginComputePass();
  p.setPipeline(v.ps); p.setBindGroup(1, v.bs); p.dispatchWorkgroups(1); p.end();
  const rd = device.createBuffer({ size: 32, usage: BU.COPY_DST | BU.MAP_READ }); enc.copyBufferToBuffer(v.ind, 0, rd, 0, 32);
  device.queue.submit([enc.finish()]); await rd.mapAsync(GPUMapMode.READ); v.chunks = new Uint32Array(rd.getMappedRange())[1]; rd.unmap();
  console.log(`${v.name}: ${v.chunks} chunks`);
  v.t = { sort: [], gu: [], dn: [], q: [] };
}
const time = async (fn) => {
  const enc = device.createCommandEncoder(), p = enc.beginComputePass();
  for (let r = 0; r < REPS; r++) fn(p);
  p.end(); await device.queue.onSubmittedWorkDone();
  const t0 = performance.now(); device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone();
  return (performance.now() - t0) / REPS;
};
for (let r = 0; r < ROUNDS; r++) for (const v of variants) {
  v.t.sort.push(await time((p) => { p.setPipeline(v.ps); p.setBindGroup(1, v.bs); p.dispatchWorkgroups(1); }));
  v.t.gu.push(await time((p) => { p.setPipeline(v.pg); p.setBindGroup(1, v.bgG); p.dispatchWorkgroupsIndirect(v.ind, 0); }));
  v.t.dn.push(await time((p) => { p.setPipeline(v.pd); p.setBindGroup(1, v.bgD); p.dispatchWorkgroupsIndirect(v.ind, 12); }));
  if (v.dp4a) v.t.q.push(await time((p) => { p.setPipeline(v.pq); p.setBindGroup(1, v.bqx); p.dispatchWorkgroups(Math.ceil(dim / 32 / 32), U); p.setBindGroup(1, v.bqh); p.dispatchWorkgroups(Math.ceil(U * KS * hs / 1024)); }));
}
const med = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];
for (const v of variants) { const q = v.t.q.length ? med(v.t.q) : 0;
  console.log(`${v.name.padEnd(28)} sort ${med(v.t.sort).toFixed(3)} ms · gate/up ${med(v.t.gu).toFixed(3)} ms · down ${med(v.t.dn).toFixed(3)} ms · quant ${q.toFixed(3)} · sum ${(med(v.t.gu) + med(v.t.dn) + q).toFixed(3)} ms per ${U} tokens`); }
