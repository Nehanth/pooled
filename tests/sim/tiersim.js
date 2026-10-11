// 3-tier expert cache simulator (roadmap 32, step 3): replays recorded MoE routing traces through the tiers a device
// serves routed experts from, and turns hits and misses into disk bytes and ms per token with a cost model built
// from the numbers measured so far. No GPU, no model files: only traces.
//
//   VRAM pool  per offloaded layer, P slots, LRU. Same policy as engine/expert_store.js plan(): hits are stamped in
//              selection order, then the misses, so a step's own experts are never evicted by its misses.
//   T1         one global LRU of experts in system RAM, sized in bytes (roadmap 32: per-expert MAP_WRITE upload
//              slots on a room node). Parked layers are always in RAM and take no T1 space. A browser tab has no
//              T1 of its own, but the OS page cache under OPFS plays its part: give it a size with t1Bytes and the
//              tab profile reads it at warm-cache speed.
//   disk       everything else, read from the GGUF (room node: worker threads + pread; tab: OPFS sync handles).
//
// Optional router lookahead: at layer l's readback the top-M experts predicted for layer l + A are read in the
// background into T1 (into the pool on a tab, which has no T1). Traces do not record the lookahead router, so the
// predictor is modelled: each true expert is in the predicted set with probability `recall` (docs/research/
// offload-2026-09.md measured 77-87% top-8 one layer ahead on the 35B), and the rest of the M are wrong experts.
//
// Traces: tests/trace_moe.js writes <name>.sel.npy (uint8, or uint16 past 256 experts) [T, L, K], <name>.w.npy
// (float32) and <name>.json. Steps before WARM (32) run but are not counted: the cache starts cold, and in the
// engine a prefill would have filled it.
//
//   node tests/sim/tiersim.js <traceDir> [--model q122|q38flash] [--json out.json]
//   deno run --allow-read --allow-write tests/sim/tiersim.js <traceDir> ...
//
// With only a trace directory it prints the report (tables in markdown): the traced model's rooms, then (for 122B
// traces) the extrapolation to Qwen3.8-Flash-Next. Every Flash number from 122B traces is an EXTRAPOLATION: the
// traces are mapped onto 512 experts and top-10 by splitFlash() below, not recorded on Flash.

export const WARM = 32;
const MiB = 2 ** 20, GiB = 2 ** 30;

// ---------------------------------------------------------------- traces

// a .npy file (version 1-3, C order) -> { data, shape }. Only the dtypes the traces use.
export function parseNpy(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (u8[0] !== 0x93 || String.fromCharCode(...u8.subarray(1, 6)) !== "NUMPY") throw new Error("not a .npy file");
  const major = u8[6], dv = new DataView(u8.buffer, u8.byteOffset);
  const hlen = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true), hoff = major === 1 ? 10 : 12;
  const head = new TextDecoder().decode(u8.subarray(hoff, hoff + hlen));
  const descr = /'descr':\s*'([^']+)'/.exec(head)?.[1];
  if (/'fortran_order':\s*True/.test(head)) throw new Error("npy: fortran order not supported");
  const shape = /'shape':\s*\(([^)]*)\)/.exec(head)[1].split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  const n = shape.reduce((a, b) => a * b, 1), off = hoff + hlen;
  const body = u8.slice(off, off + n * ({ "|u1": 1, "<u2": 2, "<u4": 4, "<f4": 4 }[descr] || 0)).buffer;
  const T = { "|u1": Uint8Array, "<u2": Uint16Array, "<u4": Uint32Array, "<f4": Float32Array }[descr];
  if (!T) throw new Error(`npy: dtype ${descr} not supported`);
  return { data: new T(body, 0, n), shape };
}

const isDeno = typeof Deno !== "undefined";
async function readBytes(p) { return isDeno ? Deno.readFile(p) : new Uint8Array(await (await import("node:fs/promises")).readFile(p)); }
async function readText(p) { return new TextDecoder().decode(await readBytes(p)); }
async function listDir(p) { if (isDeno) { const r = []; for await (const e of Deno.readDir(p)) r.push(e.name); return r; } return (await import("node:fs/promises")).readdir(p); }

// trace = { name, T, L, K, nExp, sel (T*L*K expert ids), w (router weights or null), meta }
export async function loadTraces(dir) {
  const names = (await listDir(dir)).filter((f) => f.endsWith(".sel.npy")).map((f) => f.slice(0, -8)).sort();
  const out = [];
  for (const name of names) {
    const s = parseNpy(await readBytes(`${dir}/${name}.sel.npy`));
    let w = null, meta = {};
    try { w = parseNpy(await readBytes(`${dir}/${name}.w.npy`)).data; } catch { /* weights are optional */ }
    try { meta = JSON.parse(await readText(`${dir}/${name}.json`)); } catch { /* so is the json */ }
    const [T, L, K] = s.shape;
    out.push({ name, T, L, K, nExp: meta.experts || (s.data.BYTES_PER_ELEMENT === 1 ? 256 : 1 + Math.max(...s.data)), sel: s.data, w, meta });
  }
  return out;
}

// small deterministic hash -> [0, 1) (no Math.random: every run of the report gives the same numbers)
export function hash01(...xs) {
  let h = 0x9e3779b9;
  for (const x of xs) { h = Math.imul(h ^ (x >>> 0), 0x85ebca6b); h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35); h ^= h >>> 16; }
  return (h >>> 0) / 4294967296;
}

// EXTRAPOLATION: a 256-expert top-8 trace mapped onto 512 experts and top-10, Flash's routing shape.
// Each expert e becomes two children 2e and 2e+1. Each use picks one child, remembered per (layer, e); with
// probability q the choice flips at a use. Then the two highest-weight experts of the step also take their other
// child, giving 10 distinct ids. q = 0: a child keeps serving the same tokens (the 122B's locality survives the
// split, optimistic); q = 0.5: the child is a coin flip per use (half the reuse per expert, pessimistic).
export function splitFlash(tr, { q = 0.25, seed = 1 } = {}) {
  const { T, L, K, sel, w } = tr, K2 = K + 2, out = new Uint16Array(T * L * K2), child = new Uint8Array(L * tr.nExp * 2);
  for (let l = 0; l < L; l++) for (let e = 0; e < tr.nExp; e++) child[l * tr.nExp + e] = hash01(seed, 7, l, e) < 0.5 ? 0 : 1;
  for (let t = 0; t < T; t++) for (let l = 0; l < L; l++) {
    const b = (t * L + l) * K, o = (t * L + l) * K2;
    const order = [...Array(K).keys()];
    if (w) order.sort((i, j) => w[b + j] - w[b + i]);
    for (let k = 0; k < K; k++) {
      const e = sel[b + k], c = l * tr.nExp + e;
      if (hash01(seed, t, l, e) < q) child[c] ^= 1;
      out[o + k] = 2 * e + child[c];
    }
    for (let j = 0; j < 2; j++) { const e = sel[b + order[j]]; out[o + K + j] = 2 * e + (child[l * tr.nExp + e] ^ 1); }
  }
  return { ...tr, name: `${tr.name}~flash(q=${q})`, K: K2, nExp: tr.nExp * 2, sel: out, w: null };
}

// ---------------------------------------------------------------- model geometry
// expert(l) -> { file, engine } bytes of one routed expert (gate + up + down): file = the GGUF's bytes (what the
// disk tier reads), engine = what the pool, T1 and parked buffers hold (Q4_1 / Q5_0 / Q6_K widened to Q8_0).
// resident: a layer's other weights read per token (attention or DeltaNet, router, shared expert, norms; HC on
// Flash). head: LM head bytes (Q6_K widened). layerFixedMs: per-layer dispatch time beyond bandwidth.
const q40 = (n) => n / 32 * 18, q41 = (n) => n / 32 * 20, q80 = (n) => n / 32 * 34;
export const MODELS = {
  // Qwen3.5-122B-A10B bartowski Q4_0 (q122/traces/expert_bytes.json): dim 3072, expert width 1024
  q122: { name: "Qwen3.5-122B-A10B Q4_0", L: 48, nExp: 256, K: 8, dim: 3072,
    expert: (l) => { const m = 3072 * 1024; return l < 6 ? { file: 2 * q40(m) + q41(m), engine: 2 * q40(m) + q80(m) } : { file: 3 * q40(m), engine: 3 * q40(m) }; },
    resident: 3383488512 / 48, head: 810516480, layerFixedMs: 0, ngram: false },
  // Qwen3.8-Flash-Next bartowski Q4_0 (roadmap 33 design note, section 4): dim 2560, expert width 640. Dense bytes
  // per token ~3.4 GB including the 0.68 GB head, so ~56.7 MB per layer; +0.05 ms per layer for the hyper-connection
  // mixers' extra dispatches (an assumption, not measured).
  q38flash: { name: "Qwen3.8-Flash-Next Q4_0", L: 48, nExp: 512, K: 10, dim: 2560,
    expert: (l) => { const m = 2560 * 640; return l < 6 ? { file: 2 * q40(m) + q41(m), engine: 2 * q40(m) + q80(m) } : { file: 3 * q40(m), engine: 3 * q40(m) }; },
    resident: (3.4e9 - 0.68e9) / 48, head: 0.68e9, layerFixedMs: 0.05, ngram: true },
};
export const expertBytesLayers = (model, layers, kind = "engine") => layers.reduce((s, l) => s + model.expert(l)[kind], 0);
// equal slots for every offloaded layer from a byte budget (engine/expert_store.js), at least K
export const slotsFor = (model, layers, bytes) => Math.max(model.K, Math.floor(bytes / expertBytesLayers(model, layers)));
export const range = (lo, hi) => Array.from({ length: hi - lo + 1 }, (_, i) => lo + i);

// ---------------------------------------------------------------- the simulation

// cfg: layers (offloaded, in order), P (slots per layer), parked (Set of layers whose experts are all in RAM),
// t1Bytes (global RAM LRU for the other layers), prefetch ({ ahead, top, recall } or null), expert(l) bytes.
// t1Policy: "use" (an expert is recent in T1 when it was last read from T1 or disk: what a plain LRU, or the OS
// page cache, does) or "evict" (also when the pool evicts it: T1 then keeps what just left VRAM, a victim cache;
// a room node can do this, the page cache cannot).
// -> per counted step and offloaded layer: VRAM hits, T1 hits, disk reads (and their file / engine bytes), T1
// bytes copied, prefetched reads, plus totals.
export function simulate(tr, { layers, P, parked = new Set(), t1Bytes = 0, t1Policy = "use", prefetch = null, expert, warm = WARM, seed = 1 }) {
  const { T, L, K, nExp, sel } = tr, nl = layers.length, N = Math.max(0, T - warm);
  if (P < K) throw new Error(`simulate: ${P} slots per layer < K = ${K}`);
  const pools = layers.map(() => new Map());   // per layer: expert -> true, in LRU order (first = least recent)
  const t1 = new Map();                         // global: l * nExp + e -> engine bytes, LRU order
  let t1Used = 0;
  const slot = (l) => layers.indexOf(l);
  const z = () => new Float64Array(N * nl);
  const r = { N, nl, layers, hitV: z(), hitT1: z(), disk: z(), diskFile: z(), diskEng: z(), t1Eng: z(), pfN: z(), pfFile: z(), pfUsed: z(), pfWasted: 0 };
  const pend = new Set();                       // keys prefetched and not yet used
  const t1Put = (key, bytes) => {
    if (t1Bytes <= 0 || bytes > t1Bytes) return;
    while (t1Used + bytes > t1Bytes) { const [k, b] = t1.entries().next().value; t1.delete(k); t1Used -= b; if (pend.delete(k)) r.pfWasted++; }
    t1.set(key, bytes); t1Used += bytes;
  };
  const poolPut = (pool, l, e) => {
    if (pool.size >= P) {
      const v = pool.keys().next().value, key = l * nExp + v;
      pool.delete(v);
      if (t1Policy === "evict" && t1.has(key)) { const by = t1.get(key); t1.delete(key); t1.set(key, by); }
    }
    pool.set(e, true);
  };
  for (let t = 0; t < T; t++) {
    const counted = t >= warm, row = (t - warm) * nl;
    for (let li = 0; li < nl; li++) {
      const l = layers[li], pool = pools[li], b = (t * L + l) * K, x = row + li;
      const uniq = [];
      for (let k = 0; k < K; k++) { const e = sel[b + k]; if (!uniq.includes(e)) uniq.push(e); }
      const miss = [];
      for (const e of uniq) {
        if (pool.has(e)) { pool.delete(e); pool.set(e, true); if (counted) r.hitV[x]++; } else miss.push(e);
      }
      const eb = expert(l);
      for (const e of miss) {
        const key = l * nExp + e;
        if (pend.delete(key) && counted) r.pfUsed[x]++;
        if (parked.has(l)) { if (counted) { r.hitT1[x]++; r.t1Eng[x] += eb.engine; } }
        else if (t1.has(key)) { const by = t1.get(key); t1.delete(key); t1.set(key, by); if (counted) { r.hitT1[x]++; r.t1Eng[x] += eb.engine; } }
        else { if (counted) { r.disk[x]++; r.diskFile[x] += eb.file; r.diskEng[x] += eb.engine; } t1Put(key, eb.engine); }
        poolPut(pool, l, e);
      }
      // lookahead: predict layer l + A of this step from this layer's readback, read the predicted disk misses
      if (prefetch) {
        const lt = l + prefetch.ahead, lj = slot(lt);
        if (lj >= 0 && !parked.has(lt)) {
          const bt = (t * L + lt) * K, truth = new Set(), pred = [];
          for (let k = 0; k < K; k++) truth.add(sel[bt + k]);
          for (const e of truth) if (hash01(seed, 11, t, lt, e) < prefetch.recall) pred.push(e);
          for (let i = 0; pred.length < prefetch.top && i < 4 * nExp; i++) {
            const e = Math.floor(hash01(seed, 13, t, lt, i) * nExp);
            if (!truth.has(e) && !pred.includes(e)) pred.push(e);
          }
          const ebt = expert(lt);
          for (const e of pred) {
            const key = lt * nExp + e;
            if (pools[lj].has(e) || t1.has(key)) continue;
            if (counted) { r.pfN[x]++; r.pfFile[x] += ebt.file; }
            if (t1Bytes > 0) { t1Put(key, ebt.engine); pend.add(key); }
            else { poolPut(pools[lj], lt, e); if (!truth.has(e) && counted) r.pfWasted++; }   // a tab: straight into the pool
          }
        }
      }
    }
  }
  const sum = (a) => a.reduce((s, v) => s + v, 0);
  const uses = sum(r.hitV) + sum(r.hitT1) + sum(r.disk);
  r.tot = { uses, hitV: sum(r.hitV) / uses, hitT1: sum(r.hitT1) / uses, disk: sum(r.disk) / uses,
    diskFilePerTok: sum(r.diskFile) / N, t1EngPerTok: sum(r.t1Eng) / N, pfFilePerTok: sum(r.pfFile) / N, pfUsed: sum(r.pfUsed) / Math.max(1, sum(r.pfN)) };
  return r;
}

// ---------------------------------------------------------------- the cost model
// Three cases per number: lo (fast), mid, hi (slow). Sources: q122/feasibility.md (RTX 5070 PC, measured),
// docs/research/offload-2026-09.md (OPFS and Chrome upload, measured on the GB10), docs/bench-log.md 2026-10-02
// (offload phases 2a/2b), roadmap 33 (Flash bytes). Anything not measured on that device says so.
const C3 = (lo, mid, hi) => ({ lo, mid, hi });
// NVMe random reads (SK hynix P41, 4 MB reads, measured): GB/s at queue depth 1 / 4 / 8 / 16 / 32
const P41 = [[1, 1.8], [4, 5.2], [8, 6.2], [16, 6.2], [32, 6.3]];
const interp = (tab, q) => { if (q <= tab[0][0]) return tab[0][1]; for (let i = 1; i < tab.length; i++) if (q <= tab[i][0]) { const [a, x] = tab[i - 1], [b, y] = tab[i]; return x + (y - x) * (q - a) / (b - a); } return tab[tab.length - 1][1]; };
const CHUNK = MiB;   // roadmap 32: reads in ~1 MiB chunks, so one expert miss is several reads in flight

export const PROFILES = {
  // RTX 5070 PC as a room node (Node + Dawn D3D12). Compute: 0.23 ms per 35B layer (38.6 MB) measured; the
  // feasibility study's 0.45-0.80 ms per 122B layer is ~245 / 185 / 140 GB/s effective. Cut: 0.148 ms measured
  // microbenchmark sync, ~0.4 ms per cut seen end to end on the 35B (bench-log phase 2a). Parked/T1 -> VRAM copy
  // 45-51 GB/s (copies do not overlap compute). Disk: P41 table at the chunks in flight (8 worker threads assumed);
  // 4 MB reads were measured, 1 MiB chunks were not (mid 0.85x, hi 0.65x). Disk -> T1 slot memcpy 12 GB/s (CPU).
  pcCli: { name: "PC room node", bw: C3(245, 185, 140), cut: C3(0.15, 0.25, 0.4), copy: C3(51, 45, 40), stage: C3(14, 12, 10),
    disk: (q, c) => interp(P41, Math.min(q, 8)) * { lo: 1, mid: 0.85, hi: 0.65 }[c], diskPeak: (c) => 6.2 * { lo: 1, mid: 0.85, hi: 0.65 }[c],
    t1: "ram" },
  // RTX 5070 PC as a Chrome tab: no parked RAM, the OS page cache under OPFS is its only RAM tier. OPFS cold reads
  // 1.2 GB/s per worker, 4.8 GB/s with 4 workers (GB10 Chrome, measured; Windows not measured). Upload: a mapAsync
  // staging ring 9 GB/s, writeBuffer 3.8 GB/s (GB10 Chrome). Page-cache reads ~20 GB/s warm (GB10). Chrome's
  // per-cut readback is not measured with offload (mapAsync round trip 0.27 ms on the Mac).
  pcTab: { name: "PC tab", bw: C3(245, 185, 140), cut: C3(0.3, 0.4, 0.6), upload: C3(9, 6, 3.8), warm: C3(21, 15, 8),
    disk: (q, c) => 1.2 * Math.min(q, 4) * { lo: 1.25, mid: 1, hi: 0.75 }[c], diskPeak: (c) => 4.8 * { lo: 1.25, mid: 1, hi: 0.75 }[c], t1: "pagecache" },
  // Mac Studio M5 Max in Chrome (unified memory). Compute: the 35B-A3B decodes at ~155 GB/s effective; the
  // feasibility study's 16 ms for the 122B's embed/head + 17 layers is ~170 GB/s; the design note's 260 GB/s
  // dense is the fast case. OPFS on the Mac's SSD (5-7 GB/s raw) is not measured: the GB10's 1.2 GB/s per worker
  // is the mid, x1.25 the fast case. Upload and page cache as the PC tab.
  macTab: { name: "Mac tab", bw: C3(230, 170, 140), cut: C3(0.3, 0.4, 0.6), upload: C3(9, 6, 3.8), warm: C3(21, 15, 8),
    disk: (q, c) => 1.2 * Math.min(q, 4) * { lo: 1.25, mid: 1, hi: 0.75 }[c], diskPeak: (c) => 4.8 * { lo: 1.25, mid: 1, hi: 0.75 }[c], t1: "pagecache" },
};
// two LAN hops per token in a 2-device room (feasibility study, 122B: 12 KB of hidden per hop). Flash sends the
// 4-stream residual, 40 KB per hop; on a LAN that is well inside the same range (design note: ~3 ms)
export const NET = C3(4, 6, 10);
// Flash n-gram rows: 16 random 4 KiB reads per token on the device holding layer 1, known a token ahead. lo: fully
// hidden behind layer 0; mid: 4 OPFS workers, 16 x 0.08 ms / 4, not hidden; hi: one worker, serial.
export const NGRAM = { ms: C3(0, 0.32, 1.28), bytes: 16 * 4096 };

const layerMs = (model, prof, c) => (model.resident + model.K * model.expert(10).engine) / (prof.bw[c] * 1e9) * 1e3 + model.layerFixedMs;
const headMs = (model, prof, c) => model.head / (prof.bw[c] * 1e9) * 1e3;
// time of a batch of disk reads issued together: n experts of `bytes` total, split into ~1 MiB chunks
const diskMs = (prof, n, bytes, c) => (n ? bytes / (prof.disk(Math.max(1, Math.ceil(bytes / CHUNK)), c) * 1e9) * 1e3 : 0);

// ms per counted step of one device: resident layers (compute only), offloaded layers from a simulate() result.
// -> Float64Array per step for case c, and the mean breakdown
export function deviceMs(model, prof, { resident = 0, sim = null, host = false, ngram = false, prefetch = null }, c) {
  const base = resident * layerMs(model, prof, c) + (host ? headMs(model, prof, c) : 0) + (ngram ? NGRAM.ms[c] : 0);
  if (!sim) return { steps: null, mean: base, parts: { compute: base } };
  const { N, nl } = sim, out = new Float64Array(N), lm = layerMs(model, prof, c);
  const parts = { compute: 0, cuts: 0, t1: 0, disk: 0, prefetchStall: 0 };
  for (let s = 0; s < N; s++) {
    let crit = base, diskBusy = 0, stall = 0;
    for (let li = 0; li < nl; li++) {
      const x = s * nl + li;
      let t1 = 0, dk = 0;
      // a T1 hit: one copy to VRAM (room node), or a warm page-cache read plus an upload (tab)
      if (prof.t1 === "ram") t1 = sim.t1Eng[x] / (prof.copy[c] * 1e9) * 1e3;
      else t1 = sim.t1Eng[x] / 1e9 * (1 / prof.warm[c] + 1 / prof.upload[c]) * 1e3;
      // a disk miss: the reads (all of this layer's misses in flight together), then memcpy into a mapped T1 slot
      // and the copy to VRAM (room node), or an upload (tab). The layer waits for all of it.
      if (sim.disk[x]) {
        const rd = diskMs(prof, sim.disk[x], sim.diskFile[x], c);
        diskBusy += rd;
        dk = rd + sim.diskEng[x] / 1e9 * (prof.t1 === "ram" ? 1 / prof.stage[c] + 1 / prof.copy[c] : 1 / prof.upload[c]) * 1e3;
      }
      // a prefetch issued here must land within `ahead` layers of compute; the rest of its read is a stall. On a
      // tab the prefetched expert is also uploaded into the pool (CPU and queue time, counted with the read).
      if (sim.pfN[x]) {
        const rd = sim.pfFile[x] / (prof.diskPeak(c) * 1e9) * 1e3 + (prof.t1 === "ram" ? 0 : sim.pfFile[x] / 1e9 / prof.upload[c] * 1e3);
        diskBusy += rd;
        stall += Math.max(0, rd - (prefetch?.ahead || 1) * (lm + prof.cut[c]));
      }
      crit += lm + prof.cut[c] + t1 + dk;
      parts.compute += lm; parts.cuts += prof.cut[c]; parts.t1 += t1; parts.disk += dk;
    }
    // the disk is one resource: a token cannot take less than the demand and prefetch reads together
    out[s] = Math.max(crit + stall, diskBusy);
    parts.prefetchStall += out[s] - crit;
  }
  for (const k in parts) if (k !== "compute") parts[k] /= N;
  parts.compute = parts.compute / N + base;
  return { steps: out, mean: out.reduce((a, b) => a + b, 0) / N, parts };
}

// a room: devices in pipeline order; batch-1 decode is serial, so a token is the sum of the devices plus the net
// -> { ms: C3, tps: C3, sim (the offloading device's), parts (mid) }
export function roomRun(tr, model, devices, { net = NET } = {}) {
  const ms = {}, out = { sims: [] };
  const sims = devices.map((d) => d.offload ? simulate(tr, { expert: model.expert, ...d.offload }) : null);
  for (const c of ["lo", "mid", "hi"]) {
    let tot = (devices.length > 1 ? net[c] : 0);
    const parts = [];
    devices.forEach((d, i) => {
      const r = deviceMs(model, d.prof, { resident: d.resident || 0, sim: sims[i], host: d.host, ngram: d.ngram && model.ngram, prefetch: d.offload?.prefetch }, c);
      tot += r.mean; parts.push(r.parts);
    });
    ms[c] = tot;
    if (c === "mid") out.parts = parts;
  }
  out.ms = ms;
  out.tps = C3(1000 / ms.lo, 1000 / ms.mid, 1000 / ms.hi);
  out.sim = sims.find(Boolean)?.tot || null;
  return out;
}

// ---------------------------------------------------------------- the report

const pct = (x) => (100 * x).toFixed(1);
const mb = (x) => (x / 1e6).toFixed(0);
const f1 = (x) => x.toFixed(1);

// lookahead settings: recall from the 35B's measured next-layer router (docs/research/offload-2026-09.md: top-8 at
// 1 layer ahead 0.77-0.87, at 2 ahead 0.67-0.80; top-16 at 2 ahead 0.83-0.95), the middle of each range, applied
// to top-K and top-2K at Flash's K. Wrong picks are uniform over the layer's experts here; a real router's wrong
// picks are near misses, likelier to be used soon, so this undercounts their value.
// The last is an oracle (every true expert, nothing else): the most lookahead could give.
const LOOKAHEAD = (m) => [{ ahead: 1, top: m.K, recall: 0.82 }, { ahead: 2, top: m.K, recall: 0.74 }, { ahead: 2, top: 2 * m.K, recall: 0.89 }, { ahead: 2, top: m.K, recall: 1 }];
const pfLabel = (pf) => pf.recall === 1 ? `ORACLE lookahead A=${pf.ahead} (exact set, an upper bound)` : `lookahead A=${pf.ahead} top-${pf.top} (recall ${pf.recall})`;

// the rooms of the report, for a model. Parameters that are choices, not measurements, are in the row label.
export function rooms(model) {
  const L = model.L, flash = model === MODELS.q38flash;
  const R = [];
  // Mac alone in a tab: every layer offloaded, the expert cache in unified memory, misses from OPFS
  for (const gb of [11, 18, 25]) {
    const layers = range(0, L - 1), P = slotsFor(model, layers, gb * GiB);
    R.push({ group: "Mac alone (tab)", label: `${gb} GiB expert cache, ${P} slots/layer`, devices: [{ prof: PROFILES.macTab, host: true, ngram: true, offload: { layers, P } }] });
    if (gb === 18) {
      R.push({ group: "Mac alone (tab)", label: `18 GiB + 3 GiB OS page cache under OPFS`, devices: [{ prof: PROFILES.macTab, host: true, ngram: true, offload: { layers, P, t1Bytes: 3 * GiB } }] });
      for (const pf of LOOKAHEAD(model)) R.push({ group: "Mac alone (tab)", label: `18 GiB + ${pfLabel(pf)}`, devices: [{ prof: PROFILES.macTab, host: true, ngram: true, offload: { layers, P, prefetch: pf } }] });
    }
  }
  // Mac (host, resident) + PC. 122B: Mac 0-14 (as measured, #320/#323), PC 15-47. Flash: Mac 0-13, PC 14-47 (design note)
  const macN = flash ? 14 : 15, pcLayers = range(macN, L - 1);
  const mac = { prof: PROFILES.macTab, host: true, ngram: true, resident: macN };
  const pcP = flash ? slotsFor(model, pcLayers, 6 * GiB) : 26;   // 122B: 26 slots as measured in #320; Flash: 6 GiB pool (design note)
  const cli = (label, mDisk, t1GiB, prefetch = null, t1Policy = "use") => {
    const disk = pcLayers.slice(pcLayers.length - mDisk);
    const parked = new Set(pcLayers.filter((l) => !disk.includes(l)));
    const parkGB = expertBytesLayers(model, [...parked]) * model.nExp / GiB;
    R.push({ group: "Mac + PC (CLI room node)", label: `${label}: ${parked.size} parked (${parkGB.toFixed(1)} GiB RAM) + ${mDisk} disk layers, T1 ${t1GiB} GiB${t1Policy === "evict" ? " victim-ordered" : ""}, ${pcP} slots${prefetch ? ", " + pfLabel(prefetch) : ""}`,
      devices: [mac, { prof: PROFILES.pcCli, offload: { layers: pcLayers, P: pcP, parked, t1Bytes: t1GiB * GiB, t1Policy, prefetch } }] });
  };
  cli("RAM only", 0, 0);
  if (flash) { cli("--ram ~44", 3, 2); cli("--ram ~44", 3, 2, LOOKAHEAD(model)[0]); }
  const mD = flash ? 20 : 19;
  cli("--ram 24", mD, 4);
  cli("--ram 24", mD, 4, null, "evict");
  for (const pf of LOOKAHEAD(model)) cli("--ram 24", mD, 4, pf, "evict");
  // Mac + PC as a Chrome tab: a VRAM pool only, misses from OPFS; Windows' page cache can hold the share's file
  const tabP = slotsFor(model, pcLayers, 10 * GiB - pcLayers.length * model.resident - model.nExp * model.expert(L - 1).engine - 0.6e9);
  for (const pcGB of [0, 16, 40]) {
    R.push({ group: "Mac + PC (PC in a tab)", label: `${tabP} slots, OS page cache ${pcGB} GiB`, devices: [mac, { prof: PROFILES.pcTab, offload: { layers: pcLayers, P: tabP, t1Bytes: pcGB * GiB } }] });
  }
  for (const pf of LOOKAHEAD(model)) R.push({ group: "Mac + PC (PC in a tab)", label: `${tabP} slots, page cache 16 GiB, ${pfLabel(pf)}`, devices: [mac, { prof: PROFILES.pcTab, offload: { layers: pcLayers, P: tabP, t1Bytes: 16 * GiB, prefetch: pf } }] });
  return R;
}

// runs every room over every trace -> rows
export function runRooms(traces, model, R) {
  const rows = [];
  for (const room of R) for (const tr of traces) {
    const r = roomRun(tr, model, room.devices);
    rows.push({ group: room.group, label: room.label, trace: tr.name, ...r });
  }
  return rows;
}

// T1 sizing (what roadmap 32 step 3 asks of this tool): the Mac + PC CLI room with the PC at --ram 24, T1 swept
export function t1Sweep(traces, model, sizes = [0, 1, 2, 4, 8, 16]) {
  const flash = model === MODELS.q38flash, macN = flash ? 14 : 15, pcLayers = range(macN, model.L - 1), mD = flash ? 20 : 19;
  const P = flash ? slotsFor(model, pcLayers, 6 * GiB) : 26, disk = pcLayers.slice(pcLayers.length - mD);
  const parked = new Set(pcLayers.filter((l) => !disk.includes(l)));
  const poolGiB = expertBytesLayers(model, disk) * P / GiB;
  const out = [`T1 for ${mD} disk layers of PC layers ${macN}-${model.L - 1}, ${P} slots per layer (the disk layers' pool holds ${poolGiB.toFixed(1)} GiB)\n`,
    `| T1 GiB | policy | T1 / parked hits | disk | disk MB/token | tok/s mid [range] |`, `|---|---|---|---|---|---|`];
  for (const gb of sizes) for (const pol of gb ? ["use", "evict"] : ["use"]) {
    const rs = byRoom(runRooms(traces, model, [{ group: "sweep", label: `${gb}`, devices: [
      { prof: PROFILES.macTab, host: true, ngram: true, resident: macN },
      { prof: PROFILES.pcCli, offload: { layers: pcLayers, P, parked, t1Bytes: gb * GiB, t1Policy: pol } }] }]))[0];
    out.push(`| ${gb} | ${pol === "use" ? "LRU on use" : "victim-ordered"} | ${pct(rs.sim.hitT1)}% | ${pct(rs.sim.disk)}% | ${mb(rs.sim.diskFilePerTok)} | ${f1(rs.tps.mid)} [${f1(rs.span[0])}-${f1(rs.span[1])}] |`);
  }
  return out.join("\n");
}

// the planner's question: one --ram budget on the PC, split between parked layers and T1 (mDisk layers on disk,
// the rest parked, T1 = what is left)
export function ramSplit(traces, model, ramGiB = 24) {
  const flash = model === MODELS.q38flash, macN = flash ? 14 : 15, pcLayers = range(macN, model.L - 1);
  const P = flash ? slotsFor(model, pcLayers, 6 * GiB) : 26, n = pcLayers.length;
  const layerGiB = (l) => model.expert(l).engine * model.nExp / GiB;
  const out = [`PC --ram ${ramGiB} GiB over layers ${macN}-${model.L - 1} (${n} layers, ${layerGiB(model.L - 1).toFixed(2)} GiB of experts each), ${P} slots per layer\n`,
    `| disk layers | parked | T1 GiB | T1 / parked hits | disk | disk MB/token | tok/s mid [range] |`, `|---|---|---|---|---|---|---|`];
  for (let mD = 0; mD <= n; mD += 2) {
    const disk = pcLayers.slice(n - mD), parked = new Set(pcLayers.slice(0, n - mD));
    const t1 = ramGiB - [...parked].reduce((s, l) => s + layerGiB(l), 0);
    if (t1 < 0) continue;
    const rs = byRoom(runRooms(traces, model, [{ group: "split", label: `${mD}`, devices: [
      { prof: PROFILES.macTab, host: true, ngram: true, resident: macN },
      { prof: PROFILES.pcCli, offload: { layers: pcLayers, P, parked, t1Bytes: t1 * GiB, t1Policy: "evict" } }] }]))[0];
    out.push(`| ${mD} | ${parked.size} | ${t1.toFixed(1)} | ${pct(rs.sim.hitT1)}% | ${pct(rs.sim.disk)}% | ${mb(rs.sim.diskFilePerTok)} | ${f1(rs.tps.mid)} [${f1(rs.span[0])}-${f1(rs.span[1])}] |`);
    if (disk.length === n) break;
  }
  return out.join("\n");
}

function table(rows, { calib = null } = {}) {
  const out = [];
  out.push(`| room | trace | VRAM hits | T1 / parked hits | disk | disk MB/token | prefetch MB/token | ms/token (mid) | tok/s mid [fast-slow]${calib ? " | calibrated tok/s" : ""} |`);
  out.push(`|---|---|---|---|---|---|---|---|---|${calib ? "---|" : ""}`);
  for (const r of rows) {
    const s = r.sim;
    out.push(`| ${r.label} | ${r.trace} | ${pct(s.hitV)}% | ${pct(s.hitT1)}% | ${pct(s.disk)}% | ${mb(s.diskFilePerTok)} | ${mb(s.pfFilePerTok)} | ${f1(r.ms.mid)} | ${f1(r.tps.mid)} [${f1(r.tps.lo)}-${f1(r.tps.hi)}]${calib ? ` | ${f1(r.tps.mid * calib)}` : ""} |`);
  }
  return out.join("\n");
}

// rows -> one row per room: the mean over traces (hit rates averaged over traces, tok/s from the mean ms)
function byRoom(rows) {
  const m = new Map();
  for (const r of rows) { const k = r.group + "|" + r.label; if (!m.has(k)) m.set(k, []); m.get(k).push(r); }
  return [...m.values()].map((rs) => {
    const avg = (f) => rs.reduce((s, r) => s + f(r), 0) / rs.length;
    const ms = C3(avg((r) => r.ms.lo), avg((r) => r.ms.mid), avg((r) => r.ms.hi));
    const sim = {}; for (const k of ["hitV", "hitT1", "disk", "diskFilePerTok", "pfFilePerTok"]) sim[k] = avg((r) => r.sim[k]);
    const lo = Math.min(...rs.map((r) => r.tps.hi)), hi = Math.max(...rs.map((r) => r.tps.lo));
    const parts = rs[0].parts.map((p, i) => Object.fromEntries(Object.keys(p).map((k) => [k, avg((r) => r.parts[i][k])])));
    return { group: rs[0].group, label: rs[0].label, trace: `mean of ${rs.length}`, ms, tps: C3(1000 / ms.lo, 1000 / ms.mid, 1000 / ms.hi), span: [lo, hi], sim, parts };
  });
}

export async function report(dir, { modelKey = null, json = null, log = console.log } = {}) {
  const traces = await loadTraces(dir);
  if (!traces.length) throw new Error(`no traces (<name>.sel.npy) in ${dir}`);
  const key = modelKey || (traces[0].nExp === 512 ? "q38flash" : "q122");
  const model = MODELS[key], all = { model: key, rows: {} };
  log(`# tiersim: ${traces.map((t) => `${t.name} (${t.T} steps)`).join(", ")}; ${model.name}; first ${WARM} steps not counted\n`);

  // validation against measured hit rates (bench-log 2026-10-02, phase 2a: 122B layers 17-47, GB10)
  if (key === "q122") {
    log(`## Validation: the pool against measured hit rates (122B, layers 17-47, VRAM only)\n`);
    log(`| slots | measured | simulated per trace (${traces.map((t) => t.name).join(" / ")}) |\n|---|---|---|`);
    for (const [P, meas] of [[43, "72-77% (offload_bench / test_moe, 8 GB)"], [17, "45-51% (4 GB)"], [50, "78.9 / 80.8 / 75.8 / 79.8% (the store replayed on the traces)"]]) {
      const hs = traces.map((tr) => { const s = simulate(tr, { layers: range(17, 47), P, parked: new Set(range(17, 47)), expert: model.expert }); return pct(s.tot.hitV); });
      log(`| ${P} | ${meas} | ${hs.join(" / ")} |`);
    }
    log("");
  }

  // calibration: Mac + PC CLI room, 122B, RAM only, measured ~17 tok/s plain (#323; #320's Spark + PC 12-19)
  let calib = 1;
  if (key === "q122") {
    const R = rooms(model).filter((r) => r.label.startsWith("RAM only"));
    const rr = byRoom(runRooms(traces, model, R))[0];
    calib = 17 / rr.tps.mid;
    log(`## Calibration\n\nThe RAM-only Mac + PC room (122B, Mac 0-14, PC 15-47 parked, 26 slots) is modelled at ${f1(rr.tps.mid)} tok/s mid [${f1(rr.tps.hi)}-${f1(rr.tps.lo)}]; measured ~17 (PR #323 notes; #320 Spark + PC 12-19). Calibration factor ${calib.toFixed(2)}: "calibrated" columns multiply the mid by it.\n`);
  }

  const R = rooms(model);
  const rows = runRooms(traces, model, R);
  all.rows[key] = rows;
  log(`## ${model.name}: rooms (per trace)\n`);
  for (const g of [...new Set(R.map((r) => r.group))]) { log(`### ${g}\n`); log(table(rows.filter((r) => r.group === g), { calib: key === "q122" ? calib : null })); log(""); }
  log(`## ${model.name}: rooms (mean over traces; [] = slowest trace slow case - fastest trace fast case)\n`);
  log(meanTable(byRoom(rows), key === "q122" ? calib : null));
  log("");
  log(`## ${model.name}: T1 sizing\n`);
  log(t1Sweep(traces, model));
  log("");
  log(ramSplit(traces, model, 24));
  log("");

  // EXTRAPOLATION to Flash from 256-expert traces
  if (key === "q122") {
    const fl = MODELS.q38flash;
    log(`## EXTRAPOLATION: Qwen3.8-Flash-Next geometry from the 122B traces\n`);
    log(`Traces mapped by splitFlash (512 experts, top-10) at q = 0 (locality kept), 0.25, 0.5 (child is a coin flip). Flash bytes: expert ${mb(fl.expert(10).engine)} MB (${mb(fl.expert(0).engine)} MB on layers 0-5 in the engine), ${(fl.resident / 1e6).toFixed(1)} MB dense per layer, head ${f1(fl.head / 1e9)} GB, n-gram 16 rows per token. Calibration factor from the 122B applied in the last column.\n`);
    // hit rate against the fraction of a layer cached, the curve the design note guessed (65-80% at 27%)
    log(`### Pool hit rate vs fraction of a layer cached (all 48 layers, VRAM only, mean over traces)\n`);
    log(`| fraction cached | 122B (measured traces) | Flash q=0 | Flash q=0.25 | Flash q=0.5 |\n|---|---|---|---|---|`);
    const qs = [0, 0.25, 0.5], flashT = Object.fromEntries(qs.map((q) => [q, traces.map((t) => splitFlash(t, { q }))]));
    for (const frac of [0.0625, 0.125, 0.17, 0.25, 0.27, 0.375, 0.5]) {
      const h = (trs, m) => { const P = Math.max(m.K, Math.round(frac * m.nExp)); return pct(trs.reduce((s, tr) => s + simulate(tr, { layers: range(0, 47), P, parked: new Set(range(0, 47)), expert: m.expert }).tot.hitV, 0) / trs.length); };
      log(`| ${(frac * 100).toFixed(1)}% | ${h(traces, model)}% | ${qs.map((q) => h(flashT[q], fl) + "%").join(" | ")} |`);
    }
    log("");
    const FR = rooms(fl);
    for (const q of qs) {
      const fr = runRooms(flashT[q], fl, FR);
      all.rows[`q38flash_q${q}`] = fr;
      log(`### Flash rooms, q = ${q} (EXTRAPOLATION; mean over the 4 mapped traces)\n`);
      log(meanTable(byRoom(fr), calib));
      log("");
      if (q === 0.25) { log(`### Flash T1 sizing, q = 0.25 (EXTRAPOLATION)\n`); log(t1Sweep(flashT[q], fl)); log(""); log(ramSplit(flashT[q], fl, 24)); log(""); log(ramSplit(flashT[q], fl, 40)); log(""); }
    }
  }
  if (json) {
    const strip = (rows) => rows.map(({ group, label, trace, ms, tps, sim, parts }) => ({ group, label, trace, ms, tps, sim, parts }));
    const o = { model: all.model, rows: Object.fromEntries(Object.entries(all.rows).map(([k, v]) => [k, strip(v)])) };
    const s = JSON.stringify(o, null, 1);
    if (isDeno) Deno.writeTextFileSync(json, s); else (await import("node:fs")).writeFileSync(json, s);
  }
  return all;
}

// "Mac 21.3 + PC 9.8 / 8.5 / 1.2 / 3.0 / 0.0": a resident device's compute, then the offloading device's parts
const partsStr = (parts) => parts.map((p) => p.cuts === undefined ? f1(p.compute) : [p.compute, p.cuts, p.t1, p.disk, p.prefetchStall].map(f1).join(" / ")).join(" + ");

function meanTable(rs, calib) {
  const out = [`| room | config | VRAM hits | T1 / parked hits | disk | disk MB/token | prefetch MB/token | ms/token mid | tok/s mid [range] | mid ms: [Mac resident +] offloading device compute / cuts / T1 / disk / prefetch stall${calib ? " | calibrated tok/s" : ""} |`,
    `|---|---|---|---|---|---|---|---|---|---|${calib ? "---|" : ""}`];
  for (const r of rs) {
    const s = r.sim;
    out.push(`| ${r.group} | ${r.label} | ${pct(s.hitV)}% | ${pct(s.hitT1)}% | ${pct(s.disk)}% | ${mb(s.diskFilePerTok)} | ${mb(s.pfFilePerTok)} | ${f1(r.ms.mid)} | ${f1(r.tps.mid)} [${f1(r.span[0])}-${f1(r.span[1])}] | ${partsStr(r.parts)}${calib ? ` | ${f1(r.tps.mid * calib)}` : ""} |`);
  }
  return out.join("\n");
}

// ---------------------------------------------------------------- CLI
const isMain = isDeno ? import.meta.main : (typeof process !== "undefined" && process.argv[1] && (await import("node:url")).pathToFileURL(process.argv[1]).href === import.meta.url);
if (isMain) {
  const args = isDeno ? Deno.args : process.argv.slice(2);
  const opt = (k) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : null; };
  const dir = args.find((a, i) => !a.startsWith("--") && !(i > 0 && args[i - 1].startsWith("--")));
  if (!dir) { console.log("usage: tiersim.js <traceDir> [--model q122|q38flash] [--json out.json]"); } else await report(dir, { modelKey: opt("--model"), json: opt("--json") });
}
