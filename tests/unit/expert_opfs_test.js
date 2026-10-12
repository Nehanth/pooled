// The disk tier in a tab (engine/expert_opfs.js, engine/stage_ring.js), on the CPU: the per-layer record layout, the
// resumable ingest (a crash mid-layer resumes from its checkpoint and the files come out byte for byte the same), the
// reader's record coalescing and its spread over workers (a fake worker pool; the real OPFS path is
// tests/e2e/expert_opfs.mjs in Chromium), and ExpertStore's staging ring: decode cuts through mapped slots and
// copyBufferToBuffer put the same bytes in the same slots as the writeBuffer path.
// No GPU, no browser.   deno test --no-check tests/unit/expert_opfs_test.js
import { expertLayout, expertModelKey, expertContentSig, expertPartOf, ingestExperts, ingested, ingestStorage, ingestQuotaError, OpfsExpertReader, diskCacheBytes, expertFileName } from "../../engine/expert_opfs.js";
import { expertDisk } from "../../engine/expert_disk.js";
import { StageRing } from "../../engine/stage_ring.js";
import { ExpertStore } from "../../engine/expert_store.js";
import { ggmlTypeBytes, GGML_Q4_0, GGML_Q8_0 } from "../../engine/gguf.js";

globalThis.GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
globalThis.GPUMapMode ??= { READ: 1, WRITE: 2 };
const assert = (c, m) => { if (!c) throw new Error(m); };
const same = (a, b) => a.byteLength === b.byteLength && a.every((x, i) => x === b[i]);

// A fake GGUF: layers 0..nL-1, each with stacked gate / up (Q4_0, [nExp][rows][cols]) and down (Q8_0, [nExp][cols][rows])
// expert tensors at odd offsets in one shard, deterministic contents with small finite f16 scales.
function synthModel({ nL = 3, nExp = 8, rows = 32, cols = 64, name = "synth" } = {}) {
  const tensors = {}, parts = [];
  let off = 4096 + 18;
  for (let l = 0; l < nL; l++) for (const [p, type, shape] of [["gate", GGML_Q4_0, [nExp, rows, cols]], ["up", GGML_Q4_0, [nExp, rows, cols]], ["down", GGML_Q8_0, [nExp, cols, rows]]]) {
    const len = ggmlTypeBytes(type, shape[0] * shape[1] * shape[2]);
    tensors[`blk.${l}.ffn_${p}_exps.weight`] = { shape, ggmlType: type, byteOffset: off, byteLength: len, shard: 0 };
    parts.push([off, len, type]);
    off += len + 32;
  }
  const file = new Uint8Array(off);
  let s = 777;
  for (let i = 0; i < file.length; i++) file[i] = (s = (s * 1103515245 + 12345) >>> 0) >>> 24;
  for (const [o0, len, type] of parts) { const blk = type === GGML_Q4_0 ? 18 : 34; for (let o = o0; o < o0 + len; o += blk) { file[o] = 0x00; file[o + 1] = 0x24; } }
  return { G: { meta: { "general.name": name }, tensors }, file, readAt: async (_shard, o, n) => file.slice(o, o + n) };
}
// what an expert's record must hold: expertDisk's conversion of each part, in SLICES order at the layout's offsets
async function wantRecord(m, L, e) {
  const R = expertDisk({ readAt: m.readAt }), rec = new Uint8Array(L.recBytes);
  for (const [i, p] of ["gate", "up", "down"].entries()) { const { qs, sc } = await R.readPart(L.parts[p], e); rec.set(qs, L.offs[2 * i]); rec.set(sc, L.offs[2 * i + 1]); }
  return rec;
}
// an in-memory store with the OPFS store's interface; failAfter: throw on that many-th write (a tab closed mid-ingest)
function memStore({ failAfter = Infinity } = {}) {
  const files = new Map(), log = [];
  let manifest = null;
  const grow = (f, n) => { if (f.length >= n) return f; const g = new Uint8Array(Math.max(n, f.length * 2)); g.set(f); return g; };
  return { files, log, sizes: new Map(), failAfter, writes: 0,
    manifest: async () => (manifest ? JSON.parse(manifest) : null),
    saveManifest: async (m) => { manifest = JSON.stringify(m); log.push(["manifest", JSON.parse(manifest)]); },
    async open(l, keep) { const sz = Math.min(this.sizes.get(l) || 0, keep); this.sizes.set(l, sz); files.set(l, files.get(l) || new Uint8Array(0)); log.push(["open", l, keep]); },
    async write(l, at, b) {
      if (++this.writes > this.failAfter) throw new Error("tab closed");
      const n = b.byteLength, f = grow(files.get(l), at + n); f.set(b, at); files.set(l, f);
      this.sizes.set(l, Math.max(this.sizes.get(l) || 0, at + n));
      structuredClone(b.buffer, { transfer: [b.buffer] });   // as the OPFS store does: the buffer goes to its worker
    },
    async flush(l) { log.push(["flush", l]); },
    async close(l) { log.push(["close", l]); },
    bytes(l) { return files.get(l).subarray(0, this.sizes.get(l)); },
  };
}

Deno.test("layout: a record per expert, slices in ExpertStore.SLICES order, 4 KiB aligned; the key follows the file", () => {
  const m = synthModel(), lay = expertLayout(m.G, [0, 2]);
  assert(Object.keys(lay.layers).join() === "0,2", "the layers asked for");
  const L = lay.layers[2], qsB = 32 * 64 / 2, scB = 32 * 64 / 16, q8B = 32 * 64;
  assert(L.slices.join() === [qsB, scB, qsB, scB, q8B, scB].join(), `slices ${L.slices}`);
  assert(L.offs[0] === 0 && L.offs[5] === L.slices.slice(0, 5).reduce((a, b) => a + b, 0), "offsets are prefix sums");
  assert(L.recBytes % 4096 === 0 && L.recBytes >= L.slices.reduce((a, b) => a + b, 0), "record padded to 4 KiB");
  assert(L.file === expertFileName(2) && L.file === "L002.bin" && L.nExp === 8, L.file);
  assert(lay.totalBytes === 2 * 8 * L.recBytes, "total");
  assert(expertModelKey(m.G) === expertModelKey(synthModel().G), "same file, same key");
  const moved = synthModel(); moved.G.tensors["blk.1.ffn_up_exps.weight"].byteOffset += 32;
  assert(expertModelKey(moved.G) !== expertModelKey(m.G) && expertModelKey(synthModel({ name: "other" }).G).startsWith("other-"), "another file, another key");
  assert(expertPartOf({ name: "blk.17.ffn_down_exps.weight" }).layer === 17 && expertPartOf({ name: "blk.17.ffn_down_exps.weight" }).part === "down", "part of a name");
  let threw = 0;
  for (const f of [() => expertLayout(m.G, [9]), () => expertPartOf({ name: "blk.1.attn_q.weight" })]) try { f(); } catch { threw++; }
  assert(threw === 2, "guards");
});

Deno.test("ingest: every record is the converted expert; a second run skips it all", async () => {
  const m = synthModel(), lay = expertLayout(m.G, [0, 1, 2]), st = memStore();
  const prog = [];
  const r = await ingestExperts({ layout: lay, readAt: m.readAt, store: st, chunkExperts: 3, onProgress: (p) => prog.push(p.doneBytes) });
  assert(r.wroteBytes === lay.totalBytes && r.skippedBytes === 0, JSON.stringify(r));
  for (const l of [0, 1, 2]) {
    const L = lay.layers[l], f = st.bytes(l);
    assert(f.byteLength === L.nExp * L.recBytes, `layer ${l} size ${f.byteLength}`);
    for (let e = 0; e < L.nExp; e++) assert(same(f.subarray(e * L.recBytes, (e + 1) * L.recBytes), await wantRecord(m, L, e)), `layer ${l} expert ${e}`);
  }
  assert(prog.at(-1) === lay.totalBytes && prog.every((x, i) => !i || x >= prog[i - 1]), "progress grows to the total");
  assert(ingested(await st.manifest(), lay), "manifest: all there");
  const r2 = await ingestExperts({ layout: lay, readAt: async () => { throw new Error("no reads"); }, store: st });
  assert(r2.wroteBytes === 0 && r2.skippedBytes === lay.totalBytes, "nothing to do the second time");
  // a layout with one more layer than the manifest: only that layer is new
  assert(!ingested(await st.manifest(), expertLayout(synthModel({ nL: 4 }).G, [0, 1, 2, 3])), "a layer missing is not ingested");
});

Deno.test("ingest: a crash mid-layer resumes from the last checkpoint and gives the same files", async () => {
  const m = synthModel({ nExp: 16 }), lay = expertLayout(m.G, [0, 1]), rec = lay.layers[0].recBytes;
  const ref = memStore();
  await ingestExperts({ layout: lay, readAt: m.readAt, store: ref, chunkExperts: 2 });
  // checkpoint every 4 experts (2 chunks), fail on the 11th write: layer 0 done (8 writes), layer 1 has 2 written, a
  // checkpoint after write 10 (4 experts), so the 11th write's experts are lost and redone
  const st = memStore({ failAfter: 11 });
  let err = null;
  try { await ingestExperts({ layout: lay, readAt: m.readAt, store: st, chunkExperts: 2, inflight: 0, checkpointBytes: 4 * rec }); } catch (e) { err = e; }
  assert(err && /tab closed/.test(err.message), "the first run fails");
  const man = await st.manifest();
  assert(man.layers[0].done === 16 && man.layers[1].done === 4, `checkpoint ${JSON.stringify(man.layers)}`);
  // the half-written tail past the checkpoint is cut on resume
  st.sizes.set(1, 5 * rec + 100);
  st.failAfter = Infinity;
  const r = await ingestExperts({ layout: lay, readAt: m.readAt, store: st, chunkExperts: 2, checkpointBytes: 4 * rec });
  assert(r.skippedBytes === 20 * rec && r.wroteBytes === 12 * rec, `resumed: ${JSON.stringify(r)}`);
  assert(st.log.some(([k, l, keep]) => k === "open" && l === 1 && keep === 4 * rec), "layer 1 reopened at its checkpoint");
  for (const l of [0, 1]) assert(same(st.bytes(l), ref.bytes(l)), `layer ${l}: same bytes as an uninterrupted ingest`);
});

// Chrome on a Mac (2026-10-11): a fresh origin's estimate().quota is ~10.7 GB with 159 GB free and grows as the origin
// writes, so a copy bigger than the reported quota must still be ingested; only a write that fails stops it
Deno.test("quota: a reported quota below the copy warns and the ingest goes on; persist() is asked first", async () => {
  const m = synthModel(), lay = expertLayout(m.G, [0, 1, 2]), calls = [];
  const storage = { persist: async () => { calls.push("persist"); return false; }, estimate: async () => { calls.push("estimate"); return { quota: lay.totalBytes / 2, usage: 0 }; } };
  const s = await ingestStorage(lay.totalBytes, { storage });
  assert(calls.join() === "persist,estimate" && s.persisted === false && s.quota === lay.totalBytes / 2, JSON.stringify(s));
  assert(s.warning && /going on/.test(s.warning), "a warning, not an error");
  const r = await ingestExperts({ layout: lay, readAt: m.readAt, store: memStore(), chunkExperts: 3, estimate: storage.estimate });
  assert(r.wroteBytes === lay.totalBytes, "the whole copy is written past the reported quota");
  // room enough: no warning; a storage that throws or is missing: no throw, nothing known
  assert((await ingestStorage(lay.totalBytes, { storage: { persist: async () => true, estimate: async () => ({ quota: 4 * lay.totalBytes, usage: 0 }) } })).warning === null, "room: no warning");
  const bad = await ingestStorage(lay.totalBytes, { storage: { persist: async () => { throw new Error("no"); }, estimate: async () => { throw new Error("no"); } } });
  assert(bad.persisted === false && bad.quota === null && bad.warning === null, JSON.stringify(bad));
  assert((await ingestStorage(1, { storage: undefined })).quota === null, "no storage at all");
});

Deno.test("quota: a write's QuotaExceededError is a clear error and nothing past the last checkpoint is marked done", async () => {
  const m = synthModel({ nExp: 16 }), lay = expertLayout(m.G, [0, 1]), rec = lay.layers[0].recBytes;
  // the disk fills on the 11th write (2 experts each): layer 0 done, layer 1 checkpointed at 4 experts
  const st = memStore(), write = st.write;
  st.write = async function (l, at, b) {
    if (this.writes >= 10) { const e = new Error("The operation failed because it would cause the application to exceed its storage quota."); e.name = "QuotaExceededError"; throw e; }
    return write.call(this, l, at, b);
  };
  let err = null;
  try { await ingestExperts({ layout: lay, readAt: m.readAt, store: st, chunkExperts: 2, inflight: 0, checkpointBytes: 4 * rec, estimate: async () => ({ quota: 10.7e9, usage: 20 * rec }) }); } catch (e) { err = e; }
  assert(err?.name === "QuotaExceededError", `quota error: ${err}`);
  assert(err.wroteBytes === 20 * rec && err.savedBytes === 20 * rec && err.totalBytes === lay.totalBytes && err.neededBytes === 12 * rec, JSON.stringify(err));
  assert(err.estimate.quota === 10.7e9 && /out of storage after writing/.test(err.message) && /still needed/.test(err.message) && /estimate\(\) quota 10\.0 GiB/.test(err.message) && err.cause?.name === "QuotaExceededError", err.message);
  const man = await st.manifest();
  assert(!ingested(man, lay), "a failed ingest is never complete");
  assert(man.layers[0].done === 16 && man.layers[1].done === 4, `checkpoint ${JSON.stringify(man.layers)}`);
  // the manifest write itself running out of room: the in-memory count goes back too, the error is the same kind
  const st2 = memStore();
  st2.saveManifest = async () => { throw Object.assign(new Error("quota"), { name: "QuotaExceededError" }); };
  try { await ingestExperts({ layout: lay, readAt: m.readAt, store: st2, chunkExperts: 2 }); err = null; } catch (e) { err = e; }
  assert(err?.name === "QuotaExceededError" && err.savedBytes === 0 && /estimate\(\) unknown/.test(err.message), `${err?.message}`);
  assert(await st2.manifest() === null, "nothing saved");
  // the message's numbers; any other error goes through unchanged
  const e3 = ingestQuotaError(new Error("x"), { wroteBytes: 2 ** 30, savedBytes: 2 ** 30, totalBytes: 3 * 2 ** 30 });
  assert(e3.neededBytes === 2 * 2 ** 30 && /1\.0 GiB of the 3\.0 GiB copy/.test(e3.message), e3.message);
  let other = null;
  try { await ingestExperts({ layout: lay, readAt: async () => { throw new Error("fetch failed"); }, store: memStore() }); } catch (e) { other = e; }
  assert(other?.message === "fetch failed", "a read error is not a quota error");
});

// a worker pool that answers "read" messages from in-memory files, after a delay, counting what it was sent
function fakeRpc(files, W) {
  const sent = [];
  return { sent, ws: Array.from({ length: W }, () => ({})), busy: new Array(W).fill(0),
    call(w, msg, _t, weight = 1) {
      sent.push([w, msg.reqs.length]);
      this.busy[w] += weight;
      return new Promise((r) => setTimeout(r, 1 + w)).then(() => {
        this.busy[w] -= weight;
        return { bufs: msg.reqs.map(([name, at, len]) => files.get(name).slice(at, at + len).buffer) };
      });
    },
    close: async () => {} };
}

Deno.test("reader: the three parts of an expert share one record read; a cut's misses spread over the workers", async () => {
  const m = synthModel(), lay = expertLayout(m.G, [0, 1, 2]), st = memStore();
  await ingestExperts({ layout: lay, readAt: m.readAt, store: st });
  const files = new Map([0, 1, 2].map((l) => [lay.layers[l].file, st.bytes(l)]));
  const rpc = fakeRpc(files, 4), R = new OpfsExpertReader(lay, rpc, [0, 1, 2].map((l) => lay.layers[l].file), true);
  const ref = expertDisk({ readAt: m.readAt });
  // as ExpertStore._readMany does it: every expert's three parts, all at once
  const ids = [7, 1, 4, 0, 6, 2, 5, 3], info = (p) => ({ name: `blk.1.ffn_${p}_exps.weight`, ...m.G.tensors[`blk.1.ffn_${p}_exps.weight`] });
  const got = await Promise.all(ids.map((e) => Promise.all(["gate", "up", "down"].map((p) => R.readPart(info(p), e)))));
  for (const [i, e] of ids.entries()) for (const [j, p] of ["gate", "up", "down"].entries()) {
    const w = await ref.readPart(info(p), e);
    assert(same(got[i][j].qs, w.qs) && same(got[i][j].sc, w.sc), `expert ${e} ${p}`);
  }
  assert(R.stats.reads === 8 && R.stats.batches === 1, `one record read per expert, one batch: ${JSON.stringify(R.stats)}`);
  assert(rpc.sent.length === 4 && rpc.sent.every(([, n]) => n === 2), `two reads to each of 4 workers: ${JSON.stringify(rpc.sent)}`);
  assert(R.inflight.size === 0, "nothing left in flight");
  // a lone miss (a decode cut) is striped over the idle workers, and comes back whole
  R.stripeBytes = 4096; rpc.sent.length = 0;
  const one = await Promise.all(["gate", "up", "down"].map((p) => R.readPart(info(p), 5)));
  for (const [j, p] of ["gate", "up", "down"].entries()) { const w = await ref.readPart(info(p), 5); assert(same(one[j].qs, w.qs) && same(one[j].sc, w.sc), `striped expert 5 ${p}`); }
  const k = Math.min(4, lay.layers[1].recBytes / 4096);   // 4 KiB stripes, at most one per worker
  assert(k > 1 && rpc.sent.length === k && rpc.sent.every(([, n]) => n === 1) && R.stats.reads === 9, `one record in ${k} stripes: ${JSON.stringify(rpc.sent)} ${R.stats.reads}`);
  // owned mode (no read-only handles): a file's reads go to its owner
  const rpc2 = fakeRpc(files, 2), R2 = new OpfsExpertReader(lay, rpc2, [0, 1, 2].map((l) => lay.layers[l].file), false);
  await Promise.all([0, 1, 2].map((l) => R2.readPart({ name: `blk.${l}.ffn_gate_exps.weight` }, 3)));
  assert(JSON.stringify(rpc2.sent) === JSON.stringify([[0, 2], [1, 1]]), `owners: ${JSON.stringify(rpc2.sent)}`);
  let threw = 0;
  for (const f of [() => R.readPart({ name: "blk.9.ffn_gate_exps.weight" }, 0), () => R.readPart(info("up"), 8)]) try { await f(); } catch { threw++; }
  assert(threw === 2, "a layer not in the copy, an expert out of range");
});

Deno.test("diskCacheBytes: a 36 GB Mac starts near 18-20 GB; less with a big resident part", () => {
  const G = 2 ** 30, mac = diskCacheBytes({ memoryBytes: 36 * G, residentBytes: 2 * G });
  assert(mac >= 18 * G && mac <= 20 * G, `36 GB: ${mac / G}`);
  assert(diskCacheBytes({ memoryBytes: 36 * G, residentBytes: 12 * G }) === 36 * 0.75 * G - 14 * G, "the working set caps it");
  assert(diskCacheBytes({ memoryBytes: 8 * G, residentBytes: 8 * G }) === 0, "never negative");
});

// ---- the staging ring in ExpertStore ----
// A device that enforces the ring's contract: no buffer is mapped again (mapAsync) while an encoder that copies from
// it is not yet submitted, and no submit copies from a buffer that is mapped or has a map pending (WebGPU rejects
// that submit). Encoders come from dev.createCommandEncoder() and are open until submitted.
function fakeDevice() {
  const bufs = [], writes = [], open = new Set();
  const dev = {
    bufs, writes, open, limits: { maxBufferSize: 2 ** 31 },
    queue: { writeBuffer(b, off, data) { const d = new Uint8Array(data.buffer, data.byteOffset, data.byteLength); writes.push([b, off, d.byteLength]); b.bytes.set(d, off); },
      submit(cbs) {
        for (const c of cbs) {
          assert(c.copies.every(([s]) => !s.mapped && !s.pending), "submit with a mapped or mapping source");
          open.delete(c);
        }
      } },
    createCommandEncoder() { const e = fakeEncoder(); e.finish = () => e; open.add(e); return e; },
    createBuffer({ size, usage, mappedAtCreation }) {
      const b = { size, usage, bytes: new Uint8Array(size), mapped: !!mappedAtCreation, pending: false, maps: 0,
        getMappedRange() { assert(this.mapped, "getMappedRange on an unmapped buffer"); return this.bytes.buffer; },
        unmap() { assert(this.mapped, "unmap twice"); this.mapped = false; },
        mapAsync() {
          assert(!this.mapped && !this.pending, "mapAsync on a mapped buffer");
          assert(![...open].some((e) => e.copies.some(([src]) => src === this)), "mapAsync on a buffer an unsubmitted encoder copies from");
          this.maps++; this.pending = true;
          return Promise.resolve().then(() => { this.pending = false; this.mapped = true; });
        },
        destroy() {} };
      bufs.push(b);
      return b;
    },
    pushErrorScope() {}, popErrorScope: async () => null,
  };
  return dev;
}
function fakeEncoder() {
  const copies = [];
  return { copies, copyBufferToBuffer(s, so, d, dO, n) {
    assert(!s.mapped, "copy from a mapped buffer"); assert(so % 4 === 0 && dO % 4 === 0 && n % 4 === 0, "unaligned copy");
    copies.push([s, so, d, dO, n]); d.bytes.set(s.bytes.subarray(so, so + n), dO);
  } };
}
function diskStore(ring, { nExp = 32, slots = 8 } = {}) {
  const m = synthModel({ nL: 2, nExp, rows: 64, cols: 128 }), dev = fakeDevice();
  const R = expertDisk({ readAt: m.readAt }), rg = ring ? new StageRing(dev, { slotBytes: expertLayout(m.G, [0]).layers[0].recBytes, slots: ring }) : null;
  const st = new ExpertStore(dev, { layers: [0, 1], slots, disk: { layers: [0, 1], reader: R, ring: rg } });
  for (const l of [0, 1]) for (const p of ["gate", "up", "down"]) st.attach(l, p, { name: `blk.${l}.ffn_${p}_exps.weight`, ...m.G.tensors[`blk.${l}.ffn_${p}_exps.weight`] }, nExp);
  st.build({ K: 4 });
  return { st, dev };
}

Deno.test("ring: decode cuts copy from mapped slots and give the writeBuffer path's bytes; a full ring falls back", async () => {
  const { st, dev } = diskStore(6), { st: ref } = diskStore(0);
  let seed = 5;
  const rnd = (n) => ((seed = (seed * 1103515245 + 12345) >>> 0) % n);
  for (let t = 0; t < 120; t++) {
    const l = rnd(2), KS = 5, nCols = rnd(5) ? 1 : 3, sel = new Uint32Array(nCols * KS);
    for (let c = 0; c < nCols; c++) { const ids = []; while (ids.length < 4) { const e = rnd(rnd(3) ? 12 : 32); if (!ids.includes(e)) ids.push(e); } sel.set(ids, c * KS); }
    await st.prepare(l, sel, nCols, KS); await ref.prepare(l, sel, nCols, KS);
    const enc = dev.createCommandEncoder(), a = st.plan(enc, l, sel, nCols, KS), b = ref.plan(fakeEncoder(), l, sel, nCols, KS);
    dev.queue.submit([enc.finish()]);   // as engine _cutResolve: the cut's encoder goes out before the next prepare()
    assert(a.pool === b.pool && (!a.pool || a.remap.every((x, i) => x === b.remap[i])), `t ${t}: same plan`);
    for (let j = 0; j < 6; j++) assert(same(st.L.get(l).pool[j].bytes, ref.L.get(l).pool[j].bytes), `t ${t}: pool slice ${j}`);
    if (!a.pool) for (let j = 0; j < 6; j++) assert(same(st.region.bufs[j].bytes, ref.region.bufs[j].bytes), `t ${t}: region`);
    await Promise.resolve(); await Promise.resolve();   // the GPU "finishes": recycled slots map again
  }
  // a 3-column verify with 12 distinct experts: over the pool, all 12 read, into the region by prepare() (_writeRegion):
  // the region fill streams through the 6-slot ring, waiting for slots instead of writeBuffer
  const big = Uint32Array.from([20, 21, 22, 23, 0, 24, 25, 26, 27, 0, 28, 29, 30, 31, 0]);
  await st.prepare(1, big, 3, 5); await ref.prepare(1, big, 3, 5);
  const encBig = dev.createCommandEncoder();
  assert(!st.plan(encBig, 1, big, 3, 5).pool && !ref.plan(fakeEncoder(), 1, big, 3, 5).pool, "region path");
  dev.queue.submit([encBig.finish()]);
  for (let j = 0; j < 6; j++) assert(same(st.region.bufs[j].bytes, ref.region.bufs[j].bytes), "region bytes with an overflowing ring");
  const rs = st.ring.stats;
  assert(rs.puts === st.stats.diskReads, `every read through the ring: ${JSON.stringify(rs)} of ${st.stats.diskReads} reads`);
  assert(rs.full > 0 && dev.writes.length === 0, `12 experts over 6 slots: waited for slots, ${dev.writes.length} writes`);
  assert(/through the staging ring/.test(st.summary()), st.summary());
  // a decode cut with more misses than the ring has slots: the overflow takes the writeBuffer path, same pool bytes
  const { st: s2, dev: d2 } = diskStore(2), { st: r2 } = diskStore(0), sel = Uint32Array.from([3, 9, 17, 25, 0]);
  await s2.prepare(0, sel, 1, 5); await r2.prepare(0, sel, 1, 5);
  const e2 = d2.createCommandEncoder();
  s2.plan(e2, 0, sel, 1, 5); r2.plan(fakeEncoder(), 0, sel, 1, 5);
  d2.queue.submit([e2.finish()]);
  for (let j = 0; j < 6; j++) assert(same(s2.L.get(0).pool[j].bytes, r2.L.get(0).pool[j].bytes), `overflow: pool slice ${j}`);
  const ringBufs = new Set(s2.ring.slots.map((s) => s.buf));
  assert(s2.ring.stats.puts === 2 && s2.ring.stats.full === 2 && d2.writes.length === 2 * 6, `2 through the ring, 2 written: ${JSON.stringify(s2.ring.stats)}, ${d2.writes.length} writes`);
  assert(d2.writes.every(([b]) => !ringBufs.has(b)), "never a writeBuffer into the ring");
});

Deno.test("ring: prefill region fills (a whole layer, a short frame) stream through a small ring with the writeBuffer path's bytes", async () => {
  const { st, dev } = diskStore(5), { st: ref } = diskStore(0);
  await st.loadLayerAsync(1, 7); await ref.loadLayerAsync(1, 7);
  for (let j = 0; j < 6; j++) assert(same(st.region.bufs[j].bytes, ref.region.bufs[j].bytes), `whole layer: region slice ${j}`);
  assert(st.region.layer === 1 && dev.writes.length === 0 && st.ring.stats.puts === 32, `all 32 experts through 5 slots: ${JSON.stringify(st.ring.stats)}, ${dev.writes.length} writes`);
  const sel = Uint32Array.from([3, 9, 30, 2, 0, 9, 17, 4, 5, 0]);
  await st.loadExpertsAsync(0, sel, 2, 5); await ref.loadExpertsAsync(0, sel, 2, 5);
  for (let j = 0; j < 6; j++) assert(same(st.region.bufs[j].bytes, ref.region.bufs[j].bytes), `frame: region slice ${j}`);
  assert(st.stats.regionCuts === 1 && st.region.layer === -1 && dev.writes.length === 0, "a frame: a region cut, no writeBuffer");
});

Deno.test("ring: slots cycle free -> filled -> sealed -> mapping -> free, never handed out while unmapped", async () => {
  const dev = fakeDevice(), ring = new StageRing(dev, { slotBytes: 64, slots: 2 });
  const a = ring.put([new Uint8Array(16).fill(1), new Uint8Array(8).fill(2)]);
  assert(a && a.offs.join() === "0,16" && a.buf.bytes[16] === 2, "packed");
  const b = ring.put([new Uint8Array(64)]);
  assert(b && ring.put([new Uint8Array(4)]) === null && ring.stats.full === 1, "two slots, the third put is refused");
  ring.seal();
  assert(!a.buf.mapped && !b.buf.mapped, "sealed: unmapped for the copies");
  ring.recycle();
  assert(ring.put([new Uint8Array(4)]) === null, "still mapping");
  await Promise.resolve(); await Promise.resolve();
  assert(ring.put([new Uint8Array(4)]) !== null, "mapped again");
  let threw = false;
  try { ring.put([new Uint8Array(65)]); } catch { threw = true; }
  assert(threw, "an expert bigger than a slot throws");
});

Deno.test("ring: a prefill layer load right after a cut that filled the ring still goes through the ring", async () => {
  const { st, dev } = diskStore(4), { st: ref } = diskStore(0);
  // a decode cut with 4 misses: every slot is sealed, its copies submitted, none mapped again yet
  const sel = Uint32Array.from([3, 9, 17, 25, 0]);
  await st.prepare(1, sel, 1, 5); await ref.prepare(1, sel, 1, 5);
  const enc = dev.createCommandEncoder();
  st.plan(enc, 1, sel, 1, 5); ref.plan(fakeEncoder(), 1, sel, 1, 5);
  dev.queue.submit([enc.finish()]);
  assert(st.ring.slots.every((x) => x.state === "sealed") && dev.writes.length === 0, "the cut filled the ring");
  await st.loadLayerAsync(0, 7); await ref.loadLayerAsync(0, 7);
  for (let j = 0; j < 6; j++) assert(same(st.region.bufs[j].bytes, ref.region.bufs[j].bytes), `region slice ${j}`);
  assert(dev.writes.length === 0 && st.ring.stats.full > 0, `the layer waited for slots instead of writeBuffer: ${dev.writes.length} writes, ${JSON.stringify(st.ring.stats)}`);
});

Deno.test("ring: a speculative window's later prepare (flush false) leaves the slots its open encoder copies from mapped out", async () => {
  // engine _specCut: several prepare() + plan() into one encoder before its submit; only the first may recycle
  const { st, dev } = diskStore(8), { st: ref } = diskStore(0);
  const a = Uint32Array.from([1, 2, 3, 4, 0]), b = Uint32Array.from([5, 6, 7, 8, 0]);
  const enc = dev.createCommandEncoder();
  await st.prepare(0, a, 1, 5); await ref.prepare(0, a, 1, 5);
  st.plan(enc, 0, a, 1, 5); ref.plan(fakeEncoder(), 0, a, 1, 5);
  await st.prepare(1, b, 1, 5, { flush: false }); await ref.prepare(1, b, 1, 5, { flush: false });   // throws if it recycled
  st.plan(enc, 1, b, 1, 5); ref.plan(fakeEncoder(), 1, b, 1, 5);
  dev.queue.submit([enc.finish()]);
  for (const l of [0, 1]) for (let j = 0; j < 6; j++) assert(same(st.L.get(l).pool[j].bytes, ref.L.get(l).pool[j].bytes), `layer ${l} slice ${j}`);
  assert(st.ring.stats.puts === 8 && dev.writes.length === 0, `8 experts through the ring: ${JSON.stringify(st.ring.stats)}`);
});

Deno.test("ring: the fake device catches a slot mapped again under an unsubmitted encoder", async () => {
  const dev = fakeDevice(), ring = new StageRing(dev, { slotBytes: 64, slots: 2 }), dst = dev.createBuffer({ size: 64 });
  const a = ring.put([new Uint8Array(64).fill(7)]);
  ring.seal();
  const enc = dev.createCommandEncoder();
  enc.copyBufferToBuffer(a.buf, 0, dst, 0, 64);
  let threw = null;
  try { ring.recycle(); } catch (e) { threw = e; }
  assert(threw && /unsubmitted encoder/.test(threw.message), `recycle before submit: ${threw}`);
  const dev2 = fakeDevice(), ring2 = new StageRing(dev2, { slotBytes: 64, slots: 2 }), dst2 = dev2.createBuffer({ size: 64 });
  const b = ring2.put([new Uint8Array(64)]);
  ring2.seal();
  const enc2 = dev2.createCommandEncoder();
  enc2.copyBufferToBuffer(b.buf, 0, dst2, 0, 64);
  dev2.queue.submit([enc2.finish()]);
  ring2.recycle();   // after the submit: fine
  await Promise.resolve(); await Promise.resolve();
  assert(ring2.put([new Uint8Array(4)]) !== null, "mapped again after the submit");
});

Deno.test("content signature: same layout, other weights -> another key; the OPFS reader counts record bytes", async () => {
  const m = synthModel(), m2 = synthModel();
  const t = m2.G.tensors["blk.2.ffn_down_exps.weight"];
  m2.file[t.byteOffset + t.byteLength - 5] ^= 0xff;   // a fix in the last expert of the last tensor, same offsets
  const s1 = await expertContentSig(m.G, m.readAt), s2 = await expertContentSig(m2.G, m2.readAt);
  assert(/^[0-9a-f]{8}$/.test(s1) && s1 === await expertContentSig(synthModel().G, synthModel().readAt), `stable: ${s1}`);
  assert(s1 !== s2, "other weights, other signature");
  assert(expertModelKey(m.G) === expertModelKey(m2.G), "the layout alone cannot tell them apart");
  const k1 = expertLayout(m.G, [0], { contentSig: s1 }).key, k2 = expertLayout(m2.G, [0], { contentSig: s2 }).key;
  assert(k1 !== k2 && k1.startsWith(expertModelKey(m.G)) && k1.endsWith(s1), `keys ${k1} ${k2}`);
  // ExpertStore's diskBytes counts what the reader reads (a 4 KiB-padded record), not the GGUF's bytes
  const lay = expertLayout(m.G, [0, 1, 2]), st = memStore();
  await ingestExperts({ layout: lay, readAt: m.readAt, store: st });
  const files = new Map([0, 1, 2].map((l) => [lay.layers[l].file, st.bytes(l)]));
  const R = new OpfsExpertReader(lay, fakeRpc(files, 2), [0, 1, 2].map((l) => lay.layers[l].file), true);
  const es = new ExpertStore(fakeDevice(), { layers: [1], slots: 4, disk: { layers: [1], reader: R } });
  for (const p of ["gate", "up", "down"]) es.attach(1, p, { name: `blk.1.ffn_${p}_exps.weight`, ...m.G.tensors[`blk.1.ffn_${p}_exps.weight`] }, 8);
  es.build({ K: 2 });
  await es.prepare(1, Uint32Array.from([2, 5, 0]), 1, 3);
  assert(es.stats.diskReads === 2 && es.stats.diskBytes === 2 * lay.layers[1].recBytes && es.stats.diskBytes === R.stats.bytes, `diskBytes ${es.stats.diskBytes} vs ${R.stats.bytes}`);
});
