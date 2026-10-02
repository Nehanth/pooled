// CPU check of the expert offload store (engine/expert_store.js): parking, the per-layer LRU and its slot table,
// the copies plan() encodes, the region fallback. A fake device records buffers and copies and replays the copies
// on byte arrays, so a test can read back what a slot or the region holds. The GPU checks are tests/test_moe.js and
// tests/test_q38_bits.js with OFFLOAD set (the offloaded engine must give the resident engine's bits).
// No GPU.   deno test --no-check tests/unit/expert_store_test.js
import { ExpertStore } from "../../engine/expert_store.js";

globalThis.GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
const assert = (c, m) => { if (!c) throw new Error(m); };

function fakeDevice() {
  const bufs = [];
  return {
    bufs, limits: { maxBufferSize: 2 ** 31 },
    createBuffer({ size, usage, mappedAtCreation }) {
      const b = { size, usage, bytes: new Uint8Array(size), mapped: !!mappedAtCreation,
        getMappedRange() { return this.bytes.buffer; }, unmap() { this.mapped = false; } };
      bufs.push(b);
      return b;
    },
    pushErrorScope() {}, popErrorScope: async () => null,
  };
}
function fakeEncoder() {
  const copies = [];
  return { copies, copyBufferToBuffer(s, so, d, dO, n) {
    assert(!s.mapped, "copy from a mapped buffer"); assert(so % 4 === 0 && dO % 4 === 0 && n % 4 === 0, "unaligned copy");
    assert(so + n <= s.size && dO + n <= d.size, "copy out of range");
    copies.push([s, so, d, dO, n]); d.bytes.set(s.bytes.subarray(so, so + n), dO);
  } };
}
// a converted entry whose every byte says which (layer, part, expert) it came from
function entry(layer, part, nExp, rows, cols, kind = "q4") {
  const qsB = rows * cols / (kind === "q4" ? 2 : 1), scB = rows * cols / 16;
  const qs = new Uint8Array(qsB * nExp), sc = new Uint8Array(scB * nExp);
  for (let e = 0; e < nExp; e++) { qs.fill((layer * 31 + part * 7 + e) & 255, e * qsB, (e + 1) * qsB); sc.fill((layer * 13 + part * 5 + e * 3) & 255, e * scB, (e + 1) * scB); }
  return { kind, qs, scales: new Uint32Array(sc.buffer), shape: [nExp * rows, cols] };
}
function makeStore({ layers = [2, 5], nExp = 32, slots = 6, rows = 64, cols = 128 } = {}) {
  const dev = fakeDevice(), st = new ExpertStore(dev, { layers, slots, parkBytes: 1 << 20 });
  for (const l of layers) ["gate", "up", "down"].forEach((p, i) => {
    const ph = st.park(l, p, i < 2 ? entry(l, i, nExp, rows, cols) : entry(l, i, nExp, cols, rows, "q8"), nExp);
    assert(ph.offload && ph.layer === l && ph.part === p, "placeholder");
  });
  st.build({ K: 4 });
  return { dev, st, nExp, rows, cols };
}
// what pool slot s / the region at expert e holds for (layer, part): the first byte of its qs slice
const poolByte = (st, l, part, s) => { const S = st.L.get(l), j = ["gate", "up", "down"].indexOf(part) * 2; return S.pool[j].bytes[s * S.parts[part].sl[0].per]; };
const regionByte = (st, l, part, e) => { const S = st.L.get(l), j = ["gate", "up", "down"].indexOf(part) * 2; return st.region.bufs[j].bytes[e * S.parts[part].sl[0].per]; };
const want = (l, part, e) => (l * 31 + ["gate", "up", "down"].indexOf(part) * 7 + e) & 255;
const sel1 = (ids, K = 4) => { const a = new Uint32Array(K + 1); a.set(ids); return a; };

Deno.test("park: sealed buffers, slices, a placeholder per tensor", () => {
  const { dev, st } = makeStore();
  assert(st.sealed && dev.bufs.filter((b) => b.usage & GPUBufferUsage.MAP_WRITE).every((b) => !b.mapped), "parked buffers unmapped");
  assert(st.P === 6 && st.nExp === 32, "slots");
  assert(st.region.bufs.length === 6 && st.L.get(2).pool.length === 6, "six arrays");
});

Deno.test("plan: misses fill slots, hits reuse them, the remap is the table", () => {
  const { st } = makeStore();
  let enc = fakeEncoder();
  const r = st.plan(enc, 2, sel1([7, 3, 9, 1]), 1, 5);
  assert(r.pool && enc.copies.length === 4 * 6, "4 misses x 6 slices");
  for (let k = 0; k < 4; k++) {
    const e = [7, 3, 9, 1][k], s = r.remap[k];
    for (const p of ["gate", "up", "down"]) assert(poolByte(st, 2, p, s) === want(2, p, e), `slot ${s} holds expert ${e} ${p}`);
  }
  enc = fakeEncoder();
  const r2 = st.plan(enc, 2, sel1([9, 7, 3, 1]), 1, 5);
  assert(enc.copies.length === 0 && st.stats.hits === 4, "all hits");
  assert([9, 7, 3, 1].every((e, k) => r2.remap[k] === r.remap[[7, 3, 9, 1].indexOf(e)]), "same slots");
  // the other layer is independent
  enc = fakeEncoder();
  st.plan(enc, 5, sel1([7, 3, 9, 1]), 1, 5);
  assert(enc.copies.length === 24 && poolByte(st, 5, "down", st.L.get(5).slotOf[9]) === want(5, "down", 9), "layer 5 own pool");
});

Deno.test("plan: least recently used is evicted, never an expert of the current step", () => {
  const { st } = makeStore();   // 6 slots, K = 4
  const run = (ids) => st.plan(fakeEncoder(), 2, sel1(ids), 1, 5);
  run([0, 1, 2, 3]);            // t1: 0 1 2 3
  run([4, 5, 0, 1]);            // t2: 4 5 hit 0 1 -> cache 0..5 full; LRU order 2, 3 (t1) < 0 1 4 5 (t2)
  const r = run([6, 7, 2, 4]);  // 2, 4 hit; 6, 7 must evict 3 and 0/1/5? (stamps: 3 -> t1, 0 1 4 5 -> t2) -> 3 then one of t2's
  const S = st.L.get(2);
  assert(S.slotOf[3] < 0, "3 (oldest) evicted");
  assert(S.slotOf[2] >= 0 && S.slotOf[4] >= 0 && S.slotOf[6] >= 0 && S.slotOf[7] >= 0, "current step all cached");
  assert([6, 7, 2, 4].every((e, k) => r.remap[k] === S.slotOf[e]), "remap");
  for (let s = 0; s < 6; s++) if (S.expOf[s] >= 0) assert(S.slotOf[S.expOf[s]] === s, "table and slots agree");
  // against a reference LRU (Map in recency order) over a random stream with repeats
  const { st: st2 } = makeStore({ slots: 12 });
  const ref = new Map(); let seed = 7, refMiss = 0;
  const rnd = (n) => ((seed = (seed * 1103515245 + 12345) >>> 0) % n);
  for (let t = 0; t < 400; t++) {
    const ids = []; while (ids.length < 4) { const e = rnd(rnd(3) ? 10 : 32); if (!ids.includes(e)) ids.push(e); }
    for (const e of ids) if (ref.has(e)) { ref.delete(e); ref.set(e, t); }
    for (const e of ids) if (!ref.has(e)) {
      refMiss++;
      if (ref.size >= 12) { for (const k of ref.keys()) if (!ids.includes(k)) { ref.delete(k); break; } }
      ref.set(e, t);
    }
    st2.plan(fakeEncoder(), 2, sel1(ids), 1, 5);
    const S2 = st2.L.get(2);
    assert([...ref.keys()].every((e) => S2.slotOf[e] >= 0) && [...S2.expOf].filter((e) => e >= 0).length === ref.size, `same cache contents at t ${t}`);
  }
  assert(st2.stats.misses === refMiss, `misses ${st2.stats.misses} vs reference ${refMiss}`);
});

Deno.test("plan: a batch over the pool's size goes through the region at the experts' own ids", () => {
  const { st } = makeStore();   // 6 slots
  const KS = 5, sel = new Uint32Array(2 * KS);
  sel.set([1, 2, 3, 4], 0); sel.set([5, 6, 7, 8], KS);   // 8 distinct > 6
  const enc = fakeEncoder(), r = st.plan(enc, 5, sel, 2, KS);
  assert(!r.pool && st.stats.regionCuts === 1 && enc.copies.length === 8 * 6, "region path");
  for (const e of [1, 2, 3, 4, 5, 6, 7, 8]) for (const p of ["gate", "up", "down"]) assert(regionByte(st, 5, p, e) === want(5, p, e), `region holds ${e}`);
  assert(st.region.layer === -1, "region marked partial");
  // a batch that fits: pool, the remap per column (slot K left 0)
  const sel2 = new Uint32Array(2 * KS); sel2.set([1, 2, 3, 4], 0); sel2.set([2, 1, 9, 3], KS);
  const r2 = st.plan(fakeEncoder(), 5, sel2, 2, KS), S = st.L.get(5);
  assert(r2.pool && r2.remap[KS + 0] === S.slotOf[2] && r2.remap[KS + 2] === S.slotOf[9] && r2.remap[4] === 0, "batch remap");
});

Deno.test("loadLayer: the whole layer into the region, skipped when it is already there", () => {
  const { st, nExp } = makeStore();
  const enc = fakeEncoder();
  st.loadLayer(enc, 2);
  // one copy per parked segment of each of the six arrays (a slice may span two parked buffers)
  const c1 = enc.copies.length, segs = (l) => ["gate", "up", "down"].reduce((a, p) => a + st.L.get(l).parts[p].sl.reduce((x, y) => x + y.segs.length, 0), 0);
  assert(c1 === segs(2) && c1 >= 6 && st.region.layer === 2, "whole-array copies");
  for (let e = 0; e < nExp; e += 5) for (const p of ["gate", "up", "down"]) assert(regionByte(st, 2, p, e) === want(2, p, e), "region contents");
  st.loadLayer(enc, 2);
  assert(enc.copies.length === c1, "skipped");
  st.loadLayer(enc, 5);
  assert(enc.copies.length === c1 + segs(5) && regionByte(st, 5, "up", 31) === want(5, "up", 31), "next layer");
  for (let e = 0; e < nExp; e++) for (const p of ["gate", "up", "down"]) assert(regionByte(st, 5, p, e) === want(5, p, e), "next layer's contents");
});

Deno.test("build: the budget sets the slots; under one token's experts it refuses", () => {
  const dev = fakeDevice(), st = new ExpertStore(dev, { layers: [0, 1], vramBytes: 0 });
  for (const l of [0, 1]) ["gate", "up", "down"].forEach((p, i) => st.park(l, p, entry(l, i, 16, 32, 64), 16));
  let threw = false;
  try { st.build({ K: 8 }); } catch { threw = true; }
  assert(threw, "too small a budget");
  const per = 3 * (32 * 64 / 2 + 32 * 64 / 16), region = 16 * per;
  const st2 = new ExpertStore(fakeDevice(), { layers: [0, 1], vramBytes: region + 2 * 10 * per + 100 });
  for (const l of [0, 1]) ["gate", "up", "down"].forEach((p, i) => st2.park(l, p, entry(l, i, 16, 32, 64), 16));
  st2.build({ K: 8 });
  assert(st2.P === 10, `slots ${st2.P}`);
});

Deno.test("park: a slice that doesn't fit what is left of a parked buffer fills it and goes on in the next, at an expert boundary", () => {
  // 5 experts of 4096 B (qs) per 16 KiB buffer: slices span buffers; no buffer is left mostly empty
  const nExp = 32, rows = 64, cols = 128, dev = fakeDevice(), st = new ExpertStore(dev, { layers: [3], slots: 6, parkBytes: 18 * 1024 });
  ["gate", "up", "down"].forEach((p, i) => st.park(3, p, entry(3, i, nExp, rows, cols), nExp));
  st.build({ K: 4 });
  const parked = dev.bufs.filter((b) => b.usage & GPUBufferUsage.MAP_WRITE);
  const need = 3 * nExp * (rows * cols / 2 + rows * cols / 16);
  assert(parked.length * 18 * 1024 <= need + 3 * 18 * 1024, `${parked.length} buffers of 18 KiB for ${need} B`);
  assert(st.L.get(3).parts.gate.sl[0].segs.length > 1, "the gate qs slice spans buffers");
  let enc = fakeEncoder();
  const r = st.plan(enc, 3, sel1([0, 4, 5, 31]), 1, 5);
  for (let k = 0; k < 4; k++) for (const p of ["gate", "up", "down"]) assert(poolByte(st, 3, p, r.remap[k]) === want(3, p, [0, 4, 5, 31][k]), `slot of expert ${[0, 4, 5, 31][k]} ${p}`);
  enc = fakeEncoder();
  st.loadLayer(enc, 3);
  for (let e = 0; e < nExp; e++) for (const p of ["gate", "up", "down"]) assert(regionByte(st, 3, p, e) === want(3, p, e), `region expert ${e} ${p}`);
});
