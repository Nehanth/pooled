// CPU check of the expert offload store (engine/expert_store.js): parking, the per-layer LRU and its slot table,
// the copies plan() encodes, the region fallback. A fake device records buffers and copies and replays the copies
// on byte arrays, so a test can read back what a slot or the region holds. The GPU checks are tests/test_moe.js and
// tests/test_q38_bits.js with OFFLOAD set (the offloaded engine must give the resident engine's bits).
// Disk layers (the disk tier): the same pools and region, filled by writeBuffer from a reader, must hold the same bytes
// as the parked path for the same selections.
// No GPU.   deno test --no-check tests/unit/expert_store_test.js
import { ExpertStore, notParked } from "../../engine/expert_store.js";
import { GGML_Q4_0, GGML_Q8_0 } from "../../engine/gguf.js";

globalThis.GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
const assert = (c, m) => { if (!c) throw new Error(m); };

function fakeDevice() {
  const bufs = [], writes = [];
  return {
    bufs, writes, limits: { maxBufferSize: 2 ** 31 },
    // queue.writeBuffer (the disk tier's uploads): applied at once, as it lands in queue order
    queue: { writeBuffer(b, off, data) {
      const d = data instanceof Uint8Array ? data : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      assert(off % 4 === 0 && d.byteLength % 4 === 0 && off + d.byteLength <= b.size, "writeBuffer range");
      writes.push([b, off, d.byteLength]); b.bytes.set(d, off);
    } },
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

Deno.test("build without a region (one-token engines, Flash): its bytes go to the pools; a batch over the pool refuses", () => {
  const per = 3 * (32 * 64 / 2 + 32 * 64 / 16), region = 16 * per;
  const st = new ExpertStore(fakeDevice(), { layers: [0, 1], vramBytes: region + 2 * 10 * per + 100 });
  for (const l of [0, 1]) ["gate", "up", "down"].forEach((p, i) => st.park(l, p, entry(l, i, 16, 32, 64), 16));
  st.build({ K: 4, region: false });
  assert(st.region === null && st.regionBytes === 0, "no region");
  assert(st.P === 16, `the region's bytes become slots (capped at nExp): ${st.P}`);
  const r = st.plan(fakeEncoder(), 0, sel1([1, 2, 3, 4]), 1, 5);
  assert(r.pool && r.remap.length === 5, "one token through the pool");
  const st2 = new ExpertStore(fakeDevice(), { layers: [0], slots: 4 });
  ["gate", "up", "down"].forEach((p, i) => st2.park(0, p, entry(0, i, 16, 32, 64), 16));
  st2.build({ K: 4, region: false });
  let threw = false;
  try { st2.plan(fakeEncoder(), 0, new Uint32Array([1, 2, 3, 4, 0, 5, 6, 7, 8, 0]), 2, 5); } catch (e) { threw = /no region/.test(e.message); }
  assert(threw, "8 experts over 4 slots and no region: a clear error");
  threw = false;
  try { st2.loadLayer(fakeEncoder(), 0); } catch (e) { threw = /without a region/.test(e.message); }
  assert(threw, "loadLayer refuses");
});

Deno.test("notParked: a batched engine binding a parked entry gets a readable error, not a missing buffer", () => {
  const st = new ExpertStore(fakeDevice(), { layers: [7], slots: 4 });
  const parked = st.park(7, "gate", entry(7, 0, 16, 32, 64), 16), resident = entry(7, 1, 16, 32, 64);
  assert(notParked(resident, "experts gate") === resident, "a resident entry passes");
  let msg = "";
  try { notParked(parked, "experts gate"); } catch (e) { msg = e.message; }
  assert(/layer 7's gate experts are offloaded/.test(msg) && /batched prefill \/ verify/.test(msg), msg || "no error");
  // a store built with its region (expertRegion: true) gives a batch engine what it needs per offloaded layer
  for (const [i, p] of [[1, "up"], [2, "down"]]) st.park(7, p, entry(7, i, 16, 32, 64), 16);
  st.build({ K: 4 });
  const enc = fakeEncoder();
  assert(st.loadLayer(enc, 7) > 0 && st.regionEntry(7, "gate").qs, "loadLayer + regionEntry with the region");
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

// ---- disk layers ----
// a reader whose expert e of (layer, part) is exactly what entry() parks there, after a random wait
function fakeReader({ nExp, rows, cols, delay = true }) {
  const calls = [];
  let seed = 3;
  const wait = () => new Promise((r) => setTimeout(r, delay ? ((seed = (seed * 1103515245 + 12345) >>> 0) % 4) : 0));
  return { calls, async readPart(info, e) {
    calls.push([info.layer, info.part, e]);
    await wait();
    const i = ["gate", "up", "down"].indexOf(info.part), q8 = i === 2;
    const E = q8 ? entry(info.layer, i, nExp, cols, rows, "q8") : entry(info.layer, i, nExp, rows, cols);
    const qsB = E.qs.length / nExp, sc = new Uint8Array(E.scales.buffer), scB = sc.length / nExp;
    return { qs: E.qs.slice(e * qsB, (e + 1) * qsB), sc: sc.slice(e * scB, (e + 1) * scB) };
  } };
}
// header entries as the loader passes them: stacked [nExp][rows][cols] (gate / up Q4_0, down Q8_0 with rows and cols swapped)
const tinfo = (layer, part, nExp, rows, cols) => part === "down"
  ? { layer, part, shape: [nExp, cols, rows], ggmlType: GGML_Q8_0, byteOffset: 0 }
  : { layer, part, shape: [nExp, rows, cols], ggmlType: GGML_Q4_0, byteOffset: 0 };
function makeDiskStore({ layers = [2, 5], disk = [2, 5], nExp = 32, slots = 6, rows = 64, cols = 128, delay = true } = {}) {
  const dev = fakeDevice(), reader = fakeReader({ nExp, rows, cols, delay });
  const st = new ExpertStore(dev, { layers, slots, parkBytes: 1 << 20, disk: { layers: disk, reader } });
  for (const l of layers) ["gate", "up", "down"].forEach((p, i) => {
    const ph = st.isDisk(l) ? st.attach(l, p, tinfo(l, p, nExp, rows, cols), nExp)
      : st.park(l, p, i < 2 ? entry(l, i, nExp, rows, cols) : entry(l, i, nExp, cols, rows, "q8"), nExp);
    assert(ph.offload && ph.layer === l && ph.part === p && ph.kind === (i < 2 ? "q4" : "q8"), "placeholder");
  });
  st.build({ K: 4 });
  return { dev, st, reader, nExp };
}
// one cut on a disk layer as the engine runs it: prepare, then plan
async function diskPlan(st, layer, sel, nCols, KS) { await st.prepare(layer, sel, nCols, KS); return st.plan(fakeEncoder(), layer, sel, nCols, KS); }
const poolBytes = (st, l) => st.L.get(l).pool.map((b) => b.bytes);
const sameBytes = (a, b) => a.length === b.length && a.every((x, i) => x.length === b[i].length && x.every((y, j) => y === b[i][j]));

Deno.test("disk: attach parks nothing, sizes the pools and region as parking would", () => {
  const { dev, st } = makeDiskStore();
  const { st: ref } = makeStore();
  assert(st.parkedBytes === 0 && dev.bufs.filter((b) => b.usage & GPUBufferUsage.MAP_WRITE).length === 0, "nothing parked");
  assert(st.P === ref.P && st.regionBytes === ref.regionBytes && st.poolBytes === ref.poolBytes, "same pools and region");
});

Deno.test("disk: prepare + plan put the same bytes in the same slots as the parked path, over a random stream", async () => {
  const { st } = makeDiskStore({ slots: 8 }), { st: ref } = makeStore({ slots: 8 });
  let seed = 11;
  const rnd = (n) => ((seed = (seed * 1103515245 + 12345) >>> 0) % n);
  for (let t = 0; t < 150; t++) {
    const l = rnd(2) ? 2 : 5, KS = 5, nCols = rnd(4) ? 1 : 2, sel = new Uint32Array(nCols * KS);
    for (let c = 0; c < nCols; c++) { const ids = []; while (ids.length < 4) { const e = rnd(rnd(3) ? 12 : 32); if (!ids.includes(e)) ids.push(e); } sel.set(ids, c * KS); }
    const a = await diskPlan(st, l, sel, nCols, KS), b = ref.plan(fakeEncoder(), l, sel, nCols, KS);
    assert(a.pool === b.pool && (!a.pool || a.remap.every((x, i) => x === b.remap[i])), `t ${t}: same plan`);
    assert(sameBytes(poolBytes(st, l), poolBytes(ref, l)), `t ${t}: same pool bytes`);
    if (!a.pool) for (let j = 0; j < 6; j++) assert(st.region.bufs[j].bytes.every((x, i) => x === ref.region.bufs[j].bytes[i]), `t ${t}: same region`);
  }
  assert(st.stats.misses === ref.stats.misses && st.stats.hits === ref.stats.hits && st.stats.regionCuts === ref.stats.regionCuts, "same counters");
  assert(st.stats.diskReads > 0 && st.stats.diskBytes > 0, "disk counters");
});

Deno.test("disk: only misses are read; a plan without prepare throws", async () => {
  const { st, reader } = makeDiskStore();
  await diskPlan(st, 2, sel1([7, 3, 9, 1]), 1, 5);
  assert(reader.calls.length === 4 * 3, `4 misses x 3 parts, got ${reader.calls.length}`);
  await diskPlan(st, 2, sel1([9, 7, 3, 1]), 1, 5);
  assert(reader.calls.length === 12, "all hits: nothing read");
  await diskPlan(st, 2, sel1([9, 7, 3, 20]), 1, 5);
  assert(reader.calls.length === 15 && reader.calls.slice(12).every((c) => c[2] === 20), "one miss read");
  let msg = "";
  try { st.plan(fakeEncoder(), 2, sel1([21, 22, 23, 24]), 1, 5); } catch (e) { msg = e.message; }
  assert(/not prepared/.test(msg), "unprepared plan throws");
});

Deno.test("disk: a batch over the pool's size reads every distinct expert and fills the region at their ids", async () => {
  const { st, reader } = makeDiskStore(), { st: ref } = makeStore();   // 6 slots
  const KS = 5, sel = new Uint32Array(2 * KS);
  sel.set([1, 2, 3, 4], 0); sel.set([5, 6, 7, 8], KS);
  const r = await diskPlan(st, 5, sel, 2, KS);
  ref.plan(fakeEncoder(), 5, sel, 2, KS);
  assert(!r.pool && reader.calls.length === 8 * 3 && st.region.layer === -1, "region path, 8 experts read");
  for (const e of [1, 2, 3, 4, 5, 6, 7, 8]) for (const p of ["gate", "up", "down"]) assert(regionByte(st, 5, p, e) === want(5, p, e), `region holds ${e}`);
});

Deno.test("disk: the whole layer into the region asynchronously; the sync calls refuse a disk layer", async () => {
  const { st, nExp, dev } = makeDiskStore({ delay: false }), { st: ref } = makeStore();
  for (const f of [() => st.loadLayer(fakeEncoder(), 2), () => st.loadExperts(fakeEncoder(), 2, sel1([1, 2, 3, 4]), 1, 5)]) {
    let threw = false;
    try { f(); } catch { threw = true; }
    assert(threw, "sync region fill on a disk layer throws");
  }
  const w0 = dev.writes.length;
  await st.loadLayerAsync(2, 7);   // runs of 7 experts
  ref.loadLayer(fakeEncoder(), 2);
  for (let j = 0; j < 6; j++) assert(st.region.bufs[j].bytes.every((x, i) => x === ref.region.bufs[j].bytes[i]), `region array ${j} equals the parked load`);
  assert(st.region.layer === 2 && st.loadLayer(fakeEncoder(), 2) === 0, "the sync call is a no-op once it is there");
  assert(dev.writes.length - w0 === 6 * Math.ceil(nExp / 7), "one write per slice per run");
  assert(await st.loadLayerAsync(2) === 0, "skipped when already there");
  // a short frame's experts
  const sel = new Uint32Array(10); sel.set([3, 9, 30, 9], 0); sel.set([4, 3, 1, 2], 5);
  const n = await st.loadExpertsAsync(5, sel, 2, 5);
  assert(n === 6 && st.region.layer === -1, "6 distinct experts, region partial");
  for (const e of [3, 9, 30, 4, 1, 2]) for (const p of ["gate", "up", "down"]) assert(regionByte(st, 5, p, e) === want(5, p, e), `frame expert ${e}`);
});

Deno.test("disk: a store with parked and disk layers never reads for the parked ones", async () => {
  const { st, reader } = makeDiskStore({ layers: [2, 5], disk: [5] });
  assert(!st.isDisk(2) && st.isDisk(5) && st.parkedBytes > 0, "layer 2 parked, 5 on disk");
  await diskPlan(st, 2, sel1([1, 2, 3, 4]), 1, 5);
  st.loadLayer(fakeEncoder(), 2);
  assert(reader.calls.length === 0, "parked layer: no reads");
  await diskPlan(st, 5, sel1([1, 2, 3, 4]), 1, 5);
  assert(reader.calls.length === 12 && reader.calls.every((c) => c[0] === 5), "disk layer: its reads only");
  assert(/disk layers: 12 expert reads|disk layers: 4 expert reads/.test(st.summary()), st.summary());
});

// a reader that counts the reads in flight (the most at once) and can be told to fail
function countingReader(base) {
  const r = { inflight: 0, most: 0, fail: 0, calls: base.calls, async readPart(info, e) {
    r.inflight++; r.most = Math.max(r.most, r.inflight);
    try {
      if (r.fail > 0) { r.fail--; await new Promise((ok) => setTimeout(ok, 1)); throw new Error("EIO: disk read failed"); }
      return await base.readPart(info, e);
    } finally { r.inflight--; }
  } };
  return r;
}

Deno.test("disk: a wide pass and a short frame read in runs, never every expert at once", async () => {
  const run0 = ExpertStore.REGION_RUN;
  ExpertStore.REGION_RUN = 4;
  try {
    const { st, reader } = makeDiskStore({ nExp: 64 }), { st: ref } = makeStore({ nExp: 64 });   // 6 slots
    const R = st.reader = countingReader(reader);
    const KS = 5, sel = new Uint32Array(5 * KS);
    for (let c = 0; c < 5; c++) sel.set([c * 4, c * 4 + 1, c * 4 + 2, 40 + c], c * KS);   // 20 distinct
    const r = await diskPlan(st, 5, sel, 5, KS);
    ref.plan(fakeEncoder(), 5, sel, 5, KS);
    assert(!r.pool && R.calls.length === 20 * 3, `region path, 20 experts read (${R.calls.length / 3})`);
    assert(R.most <= 4 * 3, `at most a run of 4 experts (12 parts) in flight, got ${R.most}`);
    for (let j = 0; j < 6; j++) for (const e of new Set(sel.filter((_, i) => i % KS < KS - 1))) {
      const per = st.L.get(5).parts[["gate", "up", "down"][j >> 1]].sl[j & 1].per;
      assert(st.region.bufs[j].bytes.subarray(e * per, (e + 1) * per).every((x, i) => x === ref.region.bufs[j].bytes[e * per + i]), `region slice ${j} expert ${e}`);
    }
    assert(st.stats.regionCuts === 1 && st.stats.regionBytes === ref.stats.regionBytes, "region counters as the parked path's");
    // a short prefill frame (engine _regionLoad, W * K < nExp)
    R.most = 0;
    const f = new Uint32Array(9 * KS);
    for (let c = 0; c < 9; c++) f.set([c, 20 + c, 40 + c, 63 - c], c * KS);
    const n = await st.loadExpertsAsync(2, f, 9, KS);
    assert(n === 36 && R.most <= 12 && st.region.layer === -1, `36 experts in runs (most ${R.most})`);
    for (const e of [0, 8, 20, 28, 40, 48, 63, 55]) for (const p of ["gate", "up", "down"]) assert(regionByte(st, 2, p, e) === want(2, p, e), `frame expert ${e}`);
  } finally { ExpertStore.REGION_RUN = run0; }
});

Deno.test("disk: park() refuses a disk layer (a loader without an attach branch)", () => {
  const dev = fakeDevice(), st = new ExpertStore(dev, { layers: [2, 5], slots: 6, disk: { layers: [5], reader: fakeReader({ nExp: 32, rows: 64, cols: 128 }) } });
  st.park(2, "gate", entry(2, 0, 32, 64, 128), 32);
  let msg = "";
  try { st.park(5, "gate", entry(5, 0, 32, 64, 128), 32); } catch (e) { msg = e.message; }
  assert(/disk layer/.test(msg), `park on a disk layer throws (${msg})`);
});

Deno.test("disk: a failed read rejects prepare and leaves the store usable: the next prepare + plan give the parked bytes", async () => {
  const { st, reader } = makeDiskStore({ slots: 8 }), { st: ref } = makeStore({ slots: 8 });
  const R = st.reader = countingReader(reader);
  await diskPlan(st, 2, sel1([1, 2, 3, 4]), 1, 5); ref.plan(fakeEncoder(), 2, sel1([1, 2, 3, 4]), 1, 5);
  R.fail = 1;
  let msg = "";
  try { await st.prepare(2, sel1([5, 6, 7, 8]), 1, 5); } catch (e) { msg = e.message; }
  assert(/EIO/.test(msg), `prepare rejects with the read's error (${msg})`);
  // the engine marks itself failed and plans nothing; once reset, the same cut again
  for (const s of [sel1([5, 6, 7, 8]), sel1([1, 9, 5, 2])]) {
    const a = await diskPlan(st, 2, s, 1, 5), b = ref.plan(fakeEncoder(), 2, s, 1, 5);
    assert(a.pool && a.remap.every((x, i) => x === b.remap[i]), "same plan after the failure");
    assert(sameBytes(poolBytes(st, 2), poolBytes(ref, 2)), "same pool bytes after the failure");
  }
  // a failed run of a region fill leaves the region marked partial
  R.fail = 1;
  const f = new Uint32Array(3 * 5); f.set([1, 2, 3, 4], 0); f.set([5, 6, 7, 8], 5); f.set([9, 10, 11, 12], 10);
  try { await st.loadExpertsAsync(5, f, 3, 5); msg = ""; } catch (e) { msg = e.message; }
  assert(/EIO/.test(msg) && st.region.layer === -1, "frame load rejects, region not marked as a whole layer");
});

Deno.test("disk: prepare({ flush: false }) leaves T1 alone (engine _specCut: one encoder, several plans)", async () => {
  const { st } = makeDiskStore();
  let flushes = 0;
  st.t1 = { flush() { flushes++; }, find: () => -1, wants: () => false };
  await st.prepare(2, sel1([1, 2, 3, 4]), 1, 5);
  await st.prepare(2, sel1([5, 6, 7, 8]), 1, 5, { flush: false });
  assert(flushes === 1, `one flush, got ${flushes}`);
  st.t1 = null;
});

Deno.test("gpuTables: each layer's GPU table mirrors its expert -> slot table through plan() and clear()", () => {
  const { dev, st, nExp } = makeStore({ slots: 6 });
  const writes = [];
  dev.queue = { writeBuffer(b, off, data) { writes.push(b); b.bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), off); } };
  st.gpuTables();
  const same = () => st.layers.every((l) => { const S = st.L.get(l), t = new Int32Array(S.tab.bytes.buffer); return S.slotOf.every((v, e) => t[e] === v); });
  assert(same(), "initial tables");
  let a = 7;
  const rnd = () => ((a = (Math.imul(a, 1103515245) + 12345) >>> 0) >>> 8) % nExp;
  for (let i = 0; i < 300; i++) {
    const l = st.layers[i % 2], ids = []; while (ids.length < 4) { const e = rnd() % 12; if (!ids.includes(e)) ids.push(e); }   // 12 hot experts, 6 slots: misses and hits
    const n0 = writes.length, m0 = st.stats.misses;
    st.plan(fakeEncoder(), l, sel1(ids), 1, 5);
    assert((writes.length > n0) === (st.stats.misses > m0), "a table is written exactly when plan() changes it");
    assert(same(), `tables after plan ${i}`);
  }
  st.clear();
  assert(same() && st.layers.every((l) => new Int32Array(st.L.get(l).tab.bytes.buffer).every((v) => v === -1)), "clear empties the tables");
});

// ---- deferred reader (room/flashdisk.js expertStoreDeferred: the tab's disk copy runs after the GPU step) ----
function attachAll(st, layers, nExp = 32, rows = 64, cols = 128) {
  for (const l of layers) ["gate", "up", "down"].forEach((p) => st.attach(l, p, tinfo(l, p, nExp, rows, cols), nExp));
}
Deno.test("disk deferred: attach, then setDisk, then build works and reads like a store made with its reader", async () => {
  const dev = fakeDevice(), reader = fakeReader({ nExp: 32, rows: 64, cols: 128, delay: false });
  const st = new ExpertStore(dev, { layers: [2, 5], slots: 6, parkBytes: 1 << 20, disk: { layers: [2, 5], reader: null, deferred: true } });
  attachAll(st, [2, 5]);
  st.setDisk({ reader, ring: null });
  st.build({ K: 4 });
  const { st: ref } = makeDiskStore({ delay: false });
  assert(st.P === ref.P && st.poolBytes === ref.poolBytes, "same pools as a store made with its reader");
  await diskPlan(st, 2, sel1([1, 2, 3, 4]), 1, 5); await diskPlan(ref, 2, sel1([1, 2, 3, 4]), 1, 5);
  assert(sameBytes(poolBytes(st, 2), poolBytes(ref, 2)), "same bytes in the pools");
  assert(st.reader === reader, "the reader set");
});
Deno.test("disk deferred: build without setDisk throws; setDisk after build throws; close with no reader is fine", async () => {
  const st = new ExpertStore(fakeDevice(), { layers: [2, 5], slots: 6, disk: { layers: [2, 5], reader: null, deferred: true } });
  attachAll(st, [2, 5]);
  let msg = "";
  try { st.build({ K: 4 }); } catch (e) { msg = e.message; }
  assert(/setDisk before build/.test(msg), `build without a reader throws (${msg})`);
  await st.close();
  // not deferred: the constructor still refuses disk layers without a reader
  msg = "";
  try { new ExpertStore(fakeDevice(), { layers: [2], disk: { layers: [2], reader: null } }); } catch (e) { msg = e.message; }
  assert(/need a reader/.test(msg), `no reader, not deferred: throws (${msg})`);
  const { st: built, reader } = makeDiskStore({ delay: false });
  msg = "";
  try { built.setDisk({ reader }); } catch (e) { msg = e.message; }
  assert(/after build/.test(msg), `setDisk after build throws (${msg})`);
});

Deno.test("prealloc: the parked buffers made up front hold the same bytes at the same places, and leftovers go", () => {
  const layers = [1, 2, 3, 4, 5, 6, 7, 8], nExp = 32, rows = 64, cols = 128;
  const fill = (st) => { for (const l of layers) ["gate", "up", "down"].forEach((p, i) => st.park(l, p, i < 2 ? entry(l, i, nExp, rows, cols) : entry(l, i, nExp, cols, rows, "q8"), nExp)); };
  const t = new ExpertStore(fakeDevice(), { layers, slots: 6, parkBytes: 1 << 20 });
  fill(t);
  const expect = t.parkedBytes;   // what the deal would say they park
  const a = new ExpertStore(fakeDevice(), { layers, slots: 6, parkBytes: 1 << 20, expectBytes: expect });
  fill(a);
  const dev = fakeDevice(), b = new ExpertStore(dev, { layers, slots: 6, parkBytes: 1 << 20, expectBytes: expect });
  const n = b.prealloc();
  assert(n >= 1 && dev.bufs.length === n, `made ${n} buffers up front`);
  fill(b);
  const lay = (st) => [...st.L.entries()].map(([l, S]) => [l, ["gate", "up", "down"].map((p) => S.parts[p].sl.map((x) => x.segs.map((g) => [st.park_.findIndex((q) => q.buf === g.buf), g.off, g.e0, g.n])))]);
  assert(JSON.stringify(lay(a)) === JSON.stringify(lay(b)), "the same slices in the same buffers at the same offsets");
  assert(a.park_.map((p) => p.buf.bytes).every((x, i) => x.slice(0, a.park_[i].used).every((v, j) => v === b.park_[i].buf.bytes[j])), "the same bytes");
  // more than the store needed: a buffer made up front that nothing parked in is destroyed when the store seals
  const d2 = fakeDevice(), c = new ExpertStore(d2, { layers: [2], slots: 6, parkBytes: 1 << 20, expectBytes: 8 << 20 });
  c.prealloc();
  const made = d2.bufs.length;
  let destroyed = 0;
  d2.bufs.forEach((x) => { x.destroy = () => { destroyed++; }; });
  ["gate", "up", "down"].forEach((p, i) => c.park(2, p, i < 2 ? entry(2, i, nExp, rows, cols) : entry(2, i, nExp, cols, rows, "q8"), nExp));
  c.seal();
  assert(destroyed === made - c.park_.length && c.bufBytes === c.park_.reduce((s, p) => s + p.buf.size, 0), `${destroyed} unused buffers destroyed, bufBytes counts only the kept ones`);
});
