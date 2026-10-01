// Does a speculative verify pass (C columns) re-read the shared expert from DRAM once per column?
// (kernel-research report item MOE-2: "read the shared expert once for all verify columns")
// Fused moe_gus / moe_dnc (engine/wgsl/moe.js) at the Qwen3.6-35B-A3B shape, synthetic Q4_0 routed experts and a
// Q8_0 shared expert, with a separate shared expert per launch (LAYERS of them), so nothing stays in L2 across launches.
// Cases per layout, interleaved round by round:
//   C1        one column (the plain decode token)
//   C<n>      n columns, each with its own 8 random experts (a verify pass)
//   C<n>same  n columns, all with the same 8 experts (every re-read is an L2 hit or a DRAM re-read)
//   C<n>sh1   n columns, distinct experts, the shared expert read by column 0 only (patched kernels; the bound for a
//             "shared expert once" twin; wrong numbers, timing only)
//   cd tests && deno run --unstable-webgpu --allow-read --allow-env bench/moe_verify_cols.js
//   env: LAYOUTS ('["legacy","wide"]'), C (4), ROUNDS (15), LAYERS (40)
import { moeFusedWGSL, moeFusedLayout } from "../../engine/wgsl/moe.js";
const env = (k, d) => Deno.env.get(k) ?? d;
const dim = 2048, ei = 512, sDim = 512, nExp = 256, K = 8, KS = K + 1, hs = Math.max(ei, sDim);
const ROUNDS = +env("ROUNDS", 15), LAYERS = +env("LAYERS", 40), C = +env("C", 4);
const ad = await navigator.gpu.requestAdapter();
const device = await ad.requestDevice({ requiredFeatures: ["timestamp-query"],
  requiredLimits: { maxBufferSize: ad.limits.maxBufferSize, maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize } });
device.addEventListener?.("uncapturederror", (e) => console.error("GPU ERROR:", e.error?.message));
const U = GPUBufferUsage;
let seed = 99;
const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0);
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
  dq: fill(q4(nExp * dim, ei)), ds: fill(sc(nExp * dim, ei), true) };
const oUq = q8(sDim, dim) / 4, oGs = oUq * 2, oUs = oGs + sc(sDim, dim) / 4;
// one shared expert per launch (gate/up packed as engine/qwen35.js packGU, down + scales)
const SH = Array.from({ length: LAYERS }, () => {
  const sh = device.createBuffer({ size: (oUs + sc(sDim, dim) / 4) * 4, usage: U.STORAGE | U.COPY_DST });
  const a = new Uint32Array(oUs + sc(sDim, dim) / 4); for (let i = 0; i < oGs; i++) a[i] = next(); a.fill(0x14001400, oGs); device.queue.writeBuffer(sh, 0, a);
  return { sh, sd: fill(q8(dim, sDim)), sds: fill(sc(dim, sDim), true) };
});
const f32 = (n, f, usage = U.STORAGE | U.COPY_DST) => { const b = device.createBuffer({ size: n * 4, usage }); device.queue.writeBuffer(b, 0, Float32Array.from({ length: n }, (_, i) => f(i))); return b; };
const x = f32(C * dim, (i) => Math.sin(i) * 0.5), h = f32(C * KS * hs, (i) => Math.cos(i) * 0.5), xr = f32(C * dim, () => 0);
const wts = f32(LAYERS * 64, () => 0.1);
// routing rows: per launch, C columns x KS slots in a 256-byte slot; same = every column copies column 0's experts
const mkSel = (same) => {
  const b = device.createBuffer({ size: LAYERS * 256, usage: U.STORAGE | U.COPY_DST }), a = new Uint32Array(LAYERS * 64);
  for (let s = 0; s < LAYERS; s++) for (let c = 0; c < C; c++) {
    if (same && c > 0) { for (let k = 0; k < KS; k++) a[s * 64 + c * KS + k] = a[s * 64 + k]; continue; }
    const pick = new Set(); while (pick.size < K) pick.add(next() % nExp); [...pick].forEach((e, k) => a[s * 64 + c * KS + k] = e);
  }
  device.queue.writeBuffer(b, 0, a); return b;
};
const selD = mkSel(false), selS = mkSel(true);
const uni = (a) => { const b = device.createBuffer({ size: 48, usage: U.UNIFORM | U.COPY_DST }); device.queue.writeBuffer(b, 0, new Uint32Array(a)); return b; };
const uG = uni([ei, dim, sDim, nExp, dim, hs, 1, 1, oUq, oGs, oUs, 0]), uD = uni([dim, ei, sDim, nExp, dim, hs, 1, 1, 0, 0, 0, 0]);
// sh1: skip the shared expert for columns > 0 (timing bound only)
const sh1 = (code) => {
  let n = 0;
  const out = code.replace(/\} else \{\n(\s*)dOut = S\.sDim;/g, (m, sp) => (n++, `} else if (col == 0u) {\n${sp}dOut = S.sDim;`))
    .replace(/b < nbs;/g, () => (n++, "b < select(0u, nbs, col == 0u);"));
  if (n < 2) throw new Error(`sh1 patch matched ${n} sites`);
  return out;
};
const layouts = JSON.parse(env("LAYOUTS", '["legacy","wide"]'));
const cases = [{ c: 1, sel: selD, n: "C1" }, { c: C, sel: selD, n: `C${C}` }, { c: C, sel: selS, n: `C${C}same` }, { c: C, sel: selD, n: `C${C}sh1`, patch: true }];
const variants = [];
for (const l of layouts) {
  const lay = moeFusedLayout(l, K), lname = typeof l === "string" ? l : JSON.stringify(l);
  const gRows = lay ? lay.gu.rows : 4, dRows = lay ? lay.dn.rows : 1;
  for (const cs of cases) {
    device.pushErrorScope("validation");
    let code = moeFusedWGSL({ K, R: 1, layout: lay, gu: [["q4", "q8"]], dn: [["q4", "q8"]] });
    if (cs.patch) code = sh1(code);
    const mod = device.createShaderModule({ code });
    const pipe = (e) => device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: e } });
    const pg = pipe("moe_gus_q4_q8"), pd = pipe("moe_dnc_q4_q8");
    const err = await device.popErrorScope();
    if (err) throw new Error(`${lname} ${cs.n}: ${err.message}`);
    const bg = (p, bufs) => device.createBindGroup({ layout: p.getBindGroupLayout(1), entries: bufs.map((b, i) => ({ binding: i, resource: b.buffer ? b : { buffer: b } })) });
    const S = (l) => ({ buffer: cs.sel, offset: l * 256, size: C * KS * 4 }), Wt = (l) => ({ buffer: wts, offset: l * 256, size: C * KS * 4 });
    variants.push({ name: `${lname.padEnd(7)} ${cs.n}`, kernels: {
      gus: { pipe: pg, bgs: Array.from({ length: LAYERS }, (_, l) => bg(pg, [W.gq, W.gs, W.uq, W.us, x, h, S(l), SH[l].sh, uG])), grid: [Math.ceil(hs / gRows), cs.c * KS] },
      dnc: { pipe: pd, bgs: Array.from({ length: LAYERS }, (_, l) => bg(pd, [W.dq, W.ds, h, xr, S(l), Wt(l), SH[l].sd, SH[l].sds, uD])), grid: [Math.ceil(dim / dRows), cs.c] },
    } });
  }
}
const kinds = ["gus", "dnc"], NQ = 2 * ROUNDS * variants.length * kinds.length;
const qs = device.createQuerySet({ type: "timestamp", count: NQ });
const res = device.createBuffer({ size: NQ * 8, usage: U.QUERY_RESOLVE | U.COPY_SRC }), rd = device.createBuffer({ size: NQ * 8, usage: U.COPY_DST | U.MAP_READ });
const slots = [];
const enc = device.createCommandEncoder();
for (let r = -2; r < ROUNDS; r++) for (const v of variants) for (const k of kinds) {
  const q = r >= 0 ? slots.length * 2 : -1;
  if (q >= 0) slots.push({ v: v.name, k });
  const kk = v.kernels[k], p = enc.beginComputePass(q >= 0 ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: q, endOfPassWriteIndex: q + 1 } } : undefined);
  p.setPipeline(kk.pipe);
  for (let l = 0; l < LAYERS; l++) { p.setBindGroup(1, kk.bgs[l]); p.dispatchWorkgroups(...kk.grid); }
  p.end();
}
enc.resolveQuerySet(qs, 0, NQ, res, 0); enc.copyBufferToBuffer(res, 0, rd, 0, NQ * 8);
device.queue.submit([enc.finish()]);
await rd.mapAsync(GPUMapMode.READ);
const t = new BigUint64Array(rd.getMappedRange().slice(0));
const med = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };
console.log(`${ad.info?.vendor || ""} ${ad.info?.architecture || ""} ${ad.info?.description || ""}`.trim());
console.log(`fused MoE gus+dnc, K ${K}, C ${C}, ${LAYERS} launches per pass, ${ROUNDS} rounds: median µs per launch (min); ratio vs C1 of the same layout`);
const out = {};
for (const v of variants) {
  const row = kinds.map((k) => { const us = slots.map((s, i) => s.v === v.name && s.k === k ? Number(t[2 * i + 1] - t[2 * i]) / 1e3 / LAYERS : null).filter((x) => x !== null); return [med(us), Math.min(...us)]; });
  out[v.name] = row;
  const base = out[v.name.split(" ")[0].padEnd(7) + " C1"];
  const tot = row[0][0] + row[1][0], btot = base[0][0] + base[1][0];
  console.log(`  ${v.name.padEnd(18)} gus ${row[0][0].toFixed(1).padStart(6)} (${row[0][1].toFixed(1).padStart(6)})  dnc ${row[1][0].toFixed(1).padStart(6)} (${row[1][1].toFixed(1).padStart(6)})  sum ${tot.toFixed(1).padStart(6)}  x${(tot / btot).toFixed(2)}`);
}
