// Qwen3.8-Flash-Next's disk in a browser tab: the two things a tab keeps in its origin-private file system (OPFS) so a
// share bigger than its memory runs from SSD.
//   experts   tabExpertTier(): the routed experts of the layers the deal offloads (room/plan.js offloadPlan, diskOnly:
//             the tab's GPU shares the computer's memory, so it never parks them in RAM) are copied out of the GGUF into
//             OPFS once (engine/expert_opfs.js ingestExperts: converted, one record per expert, resumable) and read
//             from there into the GPU's expert pools when a token picks them (OpfsExpertReader, the store's T1 cache,
//             a staging ring). The next load of the same layers finds them there and reads nothing from the network.
//   n-gram    ingestNgram(): the 28.8 GB n-gram (PLE) table, split per head (engine/ngram.js splitHeads) into
//             NGRAM_OPFS_DIR, so the device holding layer 1 reads its 16 rows a token from disk (NgramWorkers) instead
//             of 16 HTTP range requests a token (#356 gap D).
// Both write past Chrome's reported quota: it reads ~10 GB on a fresh origin and grows as the origin writes
// (engine/expert_opfs.js ingestStorage), so a low estimate is a warning, and only a real write error stops a copy.
// Browser-only at runtime (OPFS, workers); importing it touches neither.
import { expertLayout, expertContentSig, ingestExperts, opfsExpertStore, OpfsExpertReader, readManifest, ingested, ingestStorage, EXPERT_OPFS_DIR } from "../engine/expert_opfs.js";
import { ExpertStore } from "../engine/expert_store.js";
import { StageRing } from "../engine/stage_ring.js";
import { splitHeads, ngramFileName } from "../engine/ngram.js";

// Where a tab keeps the n-gram table split per head (engine/ngram.js ngramFileName), from the OPFS root
export const NGRAM_OPFS_DIR = ["models", "q38flash-ngram"];
const gb = (b) => (b / 2 ** 30).toFixed(1);

// The layers of a tab's ai-load offload ({ lo, hi, diskLo, vramBytes, diskRamBytes }, room/plan.js dealOffload) within
// its range -> { layers, disk } (disk: those whose experts stay on disk), or null when it offloads none. A tab only
// takes disk layers: a plan that parks experts in RAM (a room node's) is refused, never half done.
export function tabOffloadLayers(o, range) {
  if (!o) return null;
  const lo = Math.max(range[0], +o.lo || 0), hi = Math.min(range[1], +o.hi || 0);
  if (!(hi > lo) || !(o.vramBytes > 0)) return null;
  const layers = Array.from({ length: hi - lo }, (_, i) => lo + i);
  if (!Number.isInteger(o.diskLo) || o.diskLo > lo)
    throw new Error(`the host asked this tab to park experts of layers ${lo}-${(Number.isInteger(o.diskLo) ? o.diskLo : hi) - 1} in RAM; a browser tab only reads offloaded experts from its disk (OPFS)`);
  return { layers, disk: layers };
}

// The disk tier for `layers` of a parsed (merged) header G: their experts in OPFS (copied there first when they are
// not, from readAt), a reader over them, a staging ring, and the ExpertStore the loader attaches them to.
//   readAt(shard, byteOffset, byteLength) -> Promise<Uint8Array>   the GGUF, by HTTP range (never the weight cache:
//            70 GB of experts there would be a second copy)
//   offload: { vramBytes, diskRamBytes }   the deal's GPU cache (pools) and RAM cache (T1)
//   onStatus(text), onProgress(done, total) during the copy; log(text)
//   opts: { workers (OPFS reader workers, 4), ring (staging ring slots, 64), storage (navigator.storage, tests) }
// -> { store, reader, layout, ingest: { wroteBytes, skippedBytes, s } | null }
export async function tabExpertTier(device, G, layers, { readAt, offload, onStatus = () => {}, onProgress = () => {}, log = () => {}, workers = 4, ring = 64, storage = globalThis.navigator?.storage } = {}) {
  onStatus("checking this device's copy of the experts…");
  const layout = expertLayout(G, layers, { contentSig: await expertContentSig(G, readAt) });
  const dir = [EXPERT_OPFS_DIR, layout.key];
  let ingest = null;
  if (!ingested(await readManifest(dir), layout)) {
    const st = await ingestStorage(layout.totalBytes, { storage });
    if (st.warning) log(st.warning);
    const w = await opfsExpertStore(layout, { dir });
    const t0 = performance.now();
    let last = 0;
    try {
      const r = await ingestExperts({ layout, store: w, readAt, estimate: () => storage?.estimate?.(), onProgress: (p) => {
        onProgress(p.doneBytes, p.totalBytes);
        const t = performance.now();
        if (t - last > 1000) { last = t; onStatus(`copying the experts of ${layers.length} layers to this device's disk: ${gb(p.doneBytes)} of ${gb(p.totalBytes)} GB (once; the next load reads them from there)`); }
      } });
      ingest = { wroteBytes: r.wroteBytes, skippedBytes: r.skippedBytes, s: (performance.now() - t0) / 1000 };
      log(`expert disk tier: copied ${gb(r.wroteBytes)} GB of experts into OPFS in ${ingest.s.toFixed(0)} s (${gb(r.skippedBytes)} GB already there; persist() ${st.persisted || w.persisted})`);
    } finally { await w.shutdown(); }
  } else log(`expert disk tier: the experts of layers ${layers[0]}-${layers[layers.length - 1]} are in OPFS already (${gb(layout.totalBytes)} GB)`);
  const reader = await OpfsExpertReader.open(layout, { dir, workers });
  let sring = null;
  try {
    const slotBytes = Math.max(...Object.values(layout.layers).map((x) => x.slices.reduce((a, b) => a + b, 0)));
    sring = ring > 0 ? new StageRing(device, { slotBytes, slots: ring }) : null;
    const store = new ExpertStore(device, { layers, vramBytes: offload.vramBytes, disk: { layers, reader, ramBytes: +offload.diskRamBytes || 0, ring: sring } });
    return { store, reader, layout, ingest };
  } catch (e) { await reader.close().catch(() => {}); throw e; }
}

// The n-gram table into OPFS, split per head, unless every head file is there at its size. readAt as splitHeads'
// (file = the table's GGUF shard). Heads are copied `parallel` at a time, each through its own writable (written to a
// swap file, committed on close): a head is either all there or not, and a load that stops part way copies only the
// heads that are missing next time. -> bytes copied (0 when it was all there)
export async function ingestNgram(cfg, readAt, { dir = NGRAM_OPFS_DIR, parallel = 4, onStatus = () => {}, onProgress = () => {}, log = () => {}, storage = globalThis.navigator?.storage } = {}) {
  let d = await storage.getDirectory();
  for (const p of dir) d = await d.getDirectoryHandle(p, { create: true });
  const want = (h) => cfg.vocab[h] * cfg.rowBytes, missing = [];
  for (let h = 0; h < cfg.nHeads; h++) {
    let ok = false;
    try { ok = (await (await d.getFileHandle(ngramFileName(h))).getFile()).size === want(h); } catch { /* not there */ }
    if (!ok) missing.push(h);
  }
  if (!missing.length) return 0;
  const total = missing.reduce((a, h) => a + want(h), 0), t0 = performance.now();
  const st = await ingestStorage(total, { storage });
  if (st.warning) log(st.warning);
  const cur = new Map();   // head -> bytes copied so far (heads go in parallel)
  let last = 0;
  const one = async (h) => {
    const w = await (await d.getFileHandle(ngramFileName(h), { create: true })).createWritable({ keepExistingData: false });
    try {
      await splitHeads(cfg, readAt, (_h, bytes, at) => w.write({ type: "write", position: at, data: bytes }), { heads: [h], onProgress: (n) => {
        cur.set(h, n);
        const t = performance.now(), done = [...cur.values()].reduce((a, b) => a + b, 0);
        onProgress(done, total);
        if (t - last > 1000) { last = t; onStatus(`copying the n-gram table to this device's disk: ${gb(done)} of ${gb(total)} GB (once)`); }
      } });
      await w.close();
    } catch (e) { await w.abort?.().catch(() => {}); throw e; }
  };
  const queue = missing.slice();
  await Promise.all(Array.from({ length: Math.min(parallel, queue.length) }, async () => { while (queue.length) await one(queue.shift()); }));
  log(`n-gram table: copied ${gb(total)} GB (${missing.length} heads) into OPFS in ${((performance.now() - t0) / 1000).toFixed(0)} s`);
  return total;
}
