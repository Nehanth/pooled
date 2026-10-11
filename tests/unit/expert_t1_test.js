// CPU check of the disk tier's RAM cache (engine/expert_t1.js, "T1") and of ExpertStore with it (engine/expert_store.js
// disk.ramBytes). A fake device keeps each buffer's map state the way WebGPU does (mapped at creation, mapAsync pending
// until the "GPU" is done, unmapped) and a fake queue checks every submitted command buffer: it may not use a buffer
// that is mapped or has a map pending. Over random decode streams the cached store must put exactly the parked store's
// bytes in the same slots, read the file less, and never take more RAM than it was given.
// No GPU.   deno test --no-check tests/unit/expert_t1_test.js
import { ExpertStore } from "../../engine/expert_store.js";
import { ExpertT1 } from "../../engine/expert_t1.js";
import { GGML_Q4_0, GGML_Q8_0 } from "../../engine/gguf.js";

globalThis.GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
globalThis.GPUMapMode ??= { READ: 1, WRITE: 2 };
const assert = (c, m) => { if (!c) throw new Error(m); };

// mapLatency: "tick" (a map resolves on the next macrotask, as once the GPU has finished), or "never" (maps never
// come back: every miss past the first slots must take the writeBuffer path)
function fakeDevice({ mapLatency = "tick" } = {}) {
  const bufs = [], writes = [], submits = [];
  const dev = {
    bufs, writes, submits, limits: { maxBufferSize: 2 ** 31 },
    queue: {
      writeBuffer(b, off, data) {
        const d = data instanceof Uint8Array ? data : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        assert(b.state === "unmapped", "writeBuffer into a mapped buffer");
        assert(off % 4 === 0 && d.byteLength % 4 === 0 && off + d.byteLength <= b.size, "writeBuffer range");
        writes.push([b, off, d.byteLength]); b.bytes.set(d, off);
      },
      // the invariant: nothing a submitted command buffer uses is mapped or being mapped; then its copies run
      submit(cbs) {
        for (const cb of cbs) {
          for (const [s, so, d, dO, n] of cb.copies) {
            assert(s.state === "unmapped" && d.state === "unmapped", `submit uses a buffer that is ${s.state === "unmapped" ? d.state : s.state}`);
            d.bytes.set(s.bytes.subarray(so, so + n), dO);
          }
          submits.push(cb);
        }
      },
    },
    createBuffer({ size, usage, mappedAtCreation }) {
      const b = { size, usage, bytes: new Uint8Array(size), state: mappedAtCreation ? "mapped" : "unmapped", destroyed: false,
        getMappedRange() { assert(this.state === "mapped", "getMappedRange on an unmapped buffer"); return this.bytes.buffer; },
        unmap() { this.state = "unmapped"; },
        mapAsync(mode) {
          assert(this.state === "unmapped", "mapAsync on a mapped buffer");
          assert(usage & GPUBufferUsage.MAP_WRITE && mode === GPUMapMode.WRITE, "map mode");
          this.state = "pending";
          if (mapLatency === "never") return new Promise(() => {});
          return new Promise((r) => setTimeout(() => { if (this.state === "pending") this.state = "mapped"; r(); }, 0));
        },
        destroy() { this.destroyed = true; } };
      bufs.push(b);
      return b;
    },
    pushErrorScope() {}, popErrorScope: async () => null,
  };
  return dev;
}
// an encoder whose copies run at submit (queue order)
function fakeEncoder() {
  const copies = [];
  return { copies, copyBufferToBuffer(s, so, d, dO, n) {
    assert(so % 4 === 0 && dO % 4 === 0 && n % 4 === 0, "unaligned copy");
    assert(so + n <= s.size && dO + n <= d.size, "copy out of range");
    copies.push([s, so, d, dO, n]);
  }, finish() { return { copies }; } };
}
// a converted entry whose every byte says which (layer, part, expert) it came from (expert_store_test.js's)
function entry(layer, part, nExp, rows, cols, kind = "q4") {
  const qsB = rows * cols / (kind === "q4" ? 2 : 1), scB = rows * cols / 16;
  const qs = new Uint8Array(qsB * nExp), sc = new Uint8Array(scB * nExp);
  for (let e = 0; e < nExp; e++) { qs.fill((layer * 31 + part * 7 + e) & 255, e * qsB, (e + 1) * qsB); sc.fill((layer * 13 + part * 5 + e * 3) & 255, e * scB, (e + 1) * scB); }
  return { kind, qs, scales: new Uint32Array(sc.buffer), shape: [nExp * rows, cols] };
}
function fakeReader({ nExp, rows, cols }) {
  const calls = [];
  return { calls, async readPart(info, e) {
    calls.push([info.layer, info.part, e]);
    await null;
    const i = ["gate", "up", "down"].indexOf(info.part), E = i === 2 ? entry(info.layer, i, nExp, cols, rows, "q8") : entry(info.layer, i, nExp, rows, cols);
    const qsB = E.qs.length / nExp, sc = new Uint8Array(E.scales.buffer), scB = sc.length / nExp;
    return { qs: E.qs.slice(e * qsB, (e + 1) * qsB), sc: sc.slice(e * scB, (e + 1) * scB) };
  }, closed: false, async close() { this.closed = true; } };
}
const tinfo = (layer, part, nExp, rows, cols) => part === "down"
  ? { layer, part, shape: [nExp, cols, rows], ggmlType: GGML_Q8_0, byteOffset: 0 }
  : { layer, part, shape: [nExp, rows, cols], ggmlType: GGML_Q4_0, byteOffset: 0 };
// layers 2 and 5 offloaded; disk: which of them stay in the file; t1: RAM for T1 in experts (slots), 0 for none
function makeStore({ disk = [], t1 = 0, slots = 6, nExp = 32, rows = 64, cols = 128, reserve, mapLatency } = {}) {
  const layers = [2, 5], dev = fakeDevice({ mapLatency }), reader = fakeReader({ nExp, rows, cols });
  const slotBytes = (rows * cols / 2 + rows * cols / 16) * 2 + rows * cols + rows * cols / 16;   // 6 slices, each a 256 multiple here
  const st = new ExpertStore(dev, { layers, slots, parkBytes: 1 << 20, disk: disk.length ? { layers: disk, reader, ramBytes: t1 * slotBytes, reserve } : null });
  for (const l of layers) ["gate", "up", "down"].forEach((p, i) => {
    if (st.isDisk(l)) st.attach(l, p, tinfo(l, p, nExp, rows, cols), nExp);
    else st.park(l, p, i < 2 ? entry(l, i, nExp, rows, cols) : entry(l, i, nExp, cols, rows, "q8"), nExp);
  });
  st.build({ K: 4 });
  return { dev, st, reader, slotBytes };
}
// one decode cut as the engine runs it: prepare (after the previous submit), plan into a new encoder, submit
async function cut(st, dev, layer, sel, nCols, KS) {
  if (st.isDisk(layer)) await st.prepare(layer, sel, nCols, KS);
  const enc = fakeEncoder(), r = st.plan(enc, layer, sel, nCols, KS);
  dev.queue.submit([enc.finish()]);
  return r;
}
const poolBytes = (st, l) => st.L.get(l).pool.map((b) => b.bytes);
const same = (a, b) => a.length === b.length && a.every((x, i) => x.length === b[i].length && x.every((y, j) => y === b[i][j]));
function stream(seed, n, { nExp = 32, hot = 10 } = {}) {
  const rnd = (m) => (((seed = (seed * 1103515245 + 12345) >>> 0) >>> 16) % m), out = [];
  for (let t = 0; t < n; t++) {
    const l = rnd(2) ? 2 : 5, KS = 5, nCols = rnd(5) ? 1 : 2, sel = new Uint32Array(nCols * KS);
    for (let c = 0; c < nCols; c++) { const ids = []; while (ids.length < 4) { const e = rnd(rnd(4) ? hot : nExp); if (!ids.includes(e)) ids.push(e); } sel.set(ids, c * KS); }
    out.push({ l, sel, nCols, KS });
  }
  return out;
}

// ---- ExpertT1 on its own ----
function t1(cap, { reserve = 1 } = {}) {
  const dev = fakeDevice();
  return { dev, T: new ExpertT1(dev, { bytes: cap * 1024, slotBytes: 1024, reserve }) };
}
const put = (T, key) => { const s = T.take(key); assert(s, `a free slot for ${key}`); s.view.fill(key & 255); T.commit(s.s, key); return s.s; };
const tick = () => new Promise((r) => setTimeout(r, 0));

Deno.test("T1: every miss is admitted while slots are left; then only a second miss is", () => {
  const { T } = t1(4);
  for (const k of [1, 2, 3, 4]) { assert(T.wants(k), `room left: ${k} admitted`); put(T, k); }
  assert(T.slots.length === 4 && T.full === 4, "4 slots made, all full");
  assert(!T.wants(9), "a first miss with the cache full is not admitted");
  assert(T.wants(9), "its second miss is");
  assert(T.stats.skips === 1, "one first miss counted");
});

Deno.test("T1: a hit in probation is protected; eviction takes probation's least recent first", async () => {
  const { T } = t1(5, { reserve: 1 });
  for (const k of [1, 2, 3, 4, 5]) { T.wants(k); put(T, k); }
  assert(T.find(1) >= 0 && T.find(2) >= 0, "hits");
  assert(T.prot.has(1) && T.prot.has(2) && !T.prob.has(1), "hits moved to protected");
  T.flush();   // reserve 1: one eviction, probation's least recent: 3
  assert(T.of.get(3) === undefined && T.of.get(1) !== undefined, "3 evicted, 1 kept");
  assert(T.counts().mapping === 1 && T.pending === 1, "its slot is mapping");
  assert(T.take(77) === null && T.stats.noSlot === 1, "no free slot until the map is back");
  await tick();
  assert(T.counts().free === 1 && T.freeIdx.length === 1, "mapped again: free");
  T.wants(77); T.wants(77); const s = put(T, 77);
  assert(T.find(77) === s, "the reused slot holds the new expert");
});

Deno.test("T1: protected overflowing demotes its least recent to probation", () => {
  const { T } = t1(5);   // protCap 4
  for (const k of [1, 2, 3, 4, 5]) { T.wants(k); put(T, k); }
  for (const k of [1, 2, 3, 4, 5]) T.find(k);
  assert(T.prot.size === 4 && T.prob.size === 1 && T.prob.has(1), `protected holds 4; 1 demoted (${[...T.prob.keys()]})`);
});

Deno.test("T1: never more slots than its bytes hold; a zero budget caches nothing", () => {
  const { T } = t1(3);
  for (let k = 0; k < 20; k++) if (T.wants(k)) { const s = T.take(k); if (s) T.commit(s.s, k); }
  assert(T.slots.length === 3 && T.bytes === 3 * 1024, "3 slots of 1 KiB");
  const z = new ExpertT1(fakeDevice(), { bytes: 1000, slotBytes: 1024 });
  assert(z.cap === 0 && !z.wants(1) && z.take(1) === null, "no room for one slot: nothing cached");
});

// ---- ExpertStore with T1 ----
Deno.test("T1 store: the same pool bytes as the parked store over random decode, fewer reads, no mapped buffer submitted", async () => {
  for (const [t1n, slots] of [[6, 6], [12, 8], [40, 6], [64, 16]]) {
    const A = makeStore({ disk: [2, 5], t1: t1n, slots }), B = makeStore({ slots }), C = makeStore({ disk: [2, 5], slots });
    for (const { l, sel, nCols, KS } of stream(7 + t1n, 300)) {
      const a = await cut(A.st, A.dev, l, sel, nCols, KS), b = await cut(B.st, B.dev, l, sel, nCols, KS);
      await cut(C.st, C.dev, l, sel, nCols, KS);
      assert(a.pool === b.pool && (!a.pool || a.remap.every((x, i) => x === b.remap[i])), "same plan");
      assert(same(poolBytes(A.st, l), poolBytes(B.st, l)), `T1 ${t1n}: same pool bytes as parked`);
      if (!a.pool) for (let j = 0; j < 6; j++) assert(A.st.region.bufs[j].bytes.every((x, i) => x === B.st.region.bufs[j].bytes[i]), "same region");
      await tick();
    }
    const T = A.st.t1;
    assert(T.slots.length <= t1n && T.bytes <= t1n * A.slotBytes, `T1 ${t1n}: within its RAM (${T.slots.length} slots)`);
    assert(T.stats.hits > 0 && A.reader.calls.length < C.reader.calls.length, `T1 ${t1n}: hits ${T.stats.hits}, reads ${A.reader.calls.length} < ${C.reader.calls.length} without T1`);
    assert(A.st.stats.hits === B.st.stats.hits && A.st.stats.misses === B.st.stats.misses, "the VRAM pool's counters are T1's business only");
    assert(/RAM cache \d+\/\d+ experts/.test(A.st.summary()), A.st.summary());
  }
});

Deno.test("T1 store: with maps that never come back, misses fall back to writeBuffer and stay exact", async () => {
  const A = makeStore({ disk: [2, 5], t1: 8, slots: 6, reserve: 2, mapLatency: "never" }), B = makeStore({ slots: 6 });
  for (const { l, sel, nCols, KS } of stream(99, 200)) {
    await cut(A.st, A.dev, l, sel, nCols, KS); await cut(B.st, B.dev, l, sel, nCols, KS);
    assert(same(poolBytes(A.st, l), poolBytes(B.st, l)), "same pool bytes");
  }
  const T = A.st.t1;
  assert(T.counts().mapping === 2 && T.stats.noSlot > 0 && A.st.stats.t1Writes > 0, `stuck maps: ${JSON.stringify(T.counts())}, ${T.stats.noSlot} misses with no slot`);
  assert(T.full === 6, "the cache keeps what it had less the two evicted");
});

Deno.test("T1 store: a mixed store (parked + disk) caches only disk layers; prefill frames and region batches don't admit", async () => {
  const A = makeStore({ disk: [5], t1: 16, slots: 6 }), B = makeStore({ slots: 6 });
  for (const { l, sel, nCols, KS } of stream(5, 120)) { await cut(A.st, A.dev, l, sel, nCols, KS); await cut(B.st, B.dev, l, sel, nCols, KS); }
  assert([...A.st.t1.of.keys()].every((k) => Math.floor(k / 32) === 5), "only layer 5's experts cached");
  const admits = A.st.t1.stats.admits;
  // a batch over the pool's size (region path) and a prefill frame read past T1
  const KS = 5, sel = new Uint32Array(2 * KS); sel.set([20, 21, 22, 23], 0); sel.set([24, 25, 26, 27], KS);
  const r = await cut(A.st, A.dev, 5, sel, 2, KS);
  assert(!r.pool && A.st.t1.stats.admits === admits, "region batch: nothing admitted");
  await A.st.loadExpertsAsync(5, sel, 2, KS);
  assert(A.st.t1.stats.admits === admits, "prefill frame: nothing admitted");
  for (const e of [20, 21, 22, 27]) assert(A.st.region.bufs[0].bytes[e * A.st.L.get(5).parts.gate.sl[0].per] === ((5 * 31 + e) & 255), `region expert ${e}`);
  const slotBufs = A.st.t1.slots.map((x) => x.buf);
  await A.st.close();
  assert(A.st.t1 === null && A.reader.closed && slotBufs.length > 0 && slotBufs.every((b) => b.destroyed), "close frees T1 and the reader");
});
