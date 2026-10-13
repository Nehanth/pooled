// Qwen3.8-Flash-Next's disk in a browser tab: the two things a tab keeps in its origin-private file system (OPFS) so a
// share bigger than its memory runs from SSD, and whether its user lets it.
//   consent   tabDiskChoice() / tabDiskMeta() / tabDiskGate(): nothing is written to this disk until the tab's own user
//             says yes (remembered per origin in localStorage, TAB_DISK_KEY; the room's settings change it). A tab
//             advertises the tier as meta.tabDisk (room/pledge.js tabDiskGB): with ask: true until its user answers,
//             without it once they said yes, not at all once they said no. The receiving side refuses an offload it
//             did not advertise or its user turned down, whatever the host asked (tabDiskGate).
//   experts   tabExpertTier(): the routed experts of the layers the deal offloads (room/plan.js offloadPlan, diskOnly:
//             the tab's GPU shares the computer's memory, so it never parks them in RAM) are copied out of the GGUF into
//             OPFS once (engine/expert_opfs.js ingestExperts: converted, one record per expert, resumable) and read
//             from there into the GPU's expert pools when a token picks them (OpfsExpertReader, the store's T1 cache,
//             a staging ring). The copy is per layer under one directory per model: a later load of those layers,
//             offloaded or resident (opfsResidentExperts), reads their experts from it instead of the network.
//   n-gram    ingestNgram(): the 28.8 GB n-gram (PLE) table, split per head (engine/ngram.js splitHeads) into
//             ngramDir(sig), so the device holding layer 1 reads its 16 rows a token from disk (NgramWorkers) instead
//             of 16 HTTP range requests a token (#356 gap D).
//   space     flashDiskBytes() / removeFlashCopy(): what both take on this origin, and the room's "free disk space".
// Both copies write past Chrome's reported quota: it reads ~10 GB on a fresh origin and grows as the origin writes
// (engine/expert_opfs.js ingestStorage), so a low estimate is a warning, and only a real write error stops a copy.
// Browser-only at runtime (OPFS, workers); importing it touches neither.
import { expertLayout, expertContentSig, expertModelKey, expertPartOf, ingestExperts, opfsExpertStore, OpfsExpertReader, readManifest, ingested,
  ingestStorage, removeExperts, EXPERT_OPFS_DIR } from "../engine/expert_opfs.js";
import { expertKind } from "../engine/expert_disk.js";
import { ExpertStore } from "../engine/expert_store.js";
import { StageRing } from "../engine/stage_ring.js";
import { splitHeads, ngramFileName } from "../engine/ngram.js";

// Where a tab keeps the n-gram table split per head (engine/ngram.js ngramFileName), from the OPFS root: one directory
// per table content (ngramDir: ngramContentSig), so a re-uploaded table of the same size never reads as the old copy
export const NGRAM_OPFS_DIR = ["models", "q38flash-ngram"];
export const ngramDir = (sig) => [...NGRAM_OPFS_DIR, sig];
const gb = (b) => (b / 2 ** 30).toFixed(1);
const G_ = /** @type {any} */ (globalThis);

// ------------------------------------------------------------------ consent

// localStorage key of this origin's answer ("allow" | "deny")
export const TAB_DISK_KEY = "pooled-tabdisk";
// the models a tab's disk tier serves, and the RAM it offers by default for the tier's spare and T1 (?tabram=G)
export const TAB_DISK_MODELS = ["qwen3.8-flash"];
export const TAB_RAM_GB = 3;
// This tab's answer: "allow", "deny" or null (not asked yet). ?tabdisk=0 says no for this page only (a kiosk, a test);
// the URL can never say yes: room links (/r/:code, the invite) keep their query, so a host could hand out one that
// skips the question. Only the user's own answer (stored, or the dialog) allows. storage: localStorage, or null
export function tabDiskChoice({ search = "", storage = null } = {}) {
  if (new URLSearchParams(search).get("tabdisk") === "0") return "deny";
  try { const v = storage?.getItem(TAB_DISK_KEY); return v === "allow" || v === "deny" ? v : null; } catch { return null; }
}
// remember (or with null, forget) the answer for this origin
export function saveTabDiskChoice(v, storage) {
  try { if (v === "allow" || v === "deny") storage?.setItem(TAB_DISK_KEY, v); else storage?.removeItem(TAB_DISK_KEY); } catch { /* private mode */ }
}
// The most RAM (GB) a tab offers its disk tier: room/pledge.js TAB_DISK_MAX_GB, and half the device's memory when the
// browser says it has less than 8 GB (navigator.deviceMemory is exact below its cap of 8)
export function tabRamCap(deviceMemory = null) {
  let cap = 16;
  if (deviceMemory > 0 && deviceMemory < 8) cap = Math.min(cap, deviceMemory / 2);
  return Math.floor(cap * 10) / 10;
}
// meta.tabDisk for this tab (room/pledge.js tabDiskGB), or null when it has no disk tier: a computer's tab with WebGPU
// and OPFS whose user did not say no. m: the page's probed meta ({ webgpu, phone }). ram: ?tabram=G (default
// TAB_RAM_GB; 0 turns the tier off), held to tabRamCap. Phones never: their tabs die long before a model this size.
export function tabDiskMeta(m, { search = "", choice = null, opfs = false, deviceMemory = null } = {}) {
  if (!m?.webgpu || m.phone || !opfs || choice === "deny") return null;
  const q = new URLSearchParams(search);
  const ram = Math.min(q.has("tabram") ? +q.get("tabram") : TAB_RAM_GB, tabRamCap(deviceMemory));
  if (!(ram > 0)) return null;
  return { ramGB: Math.floor(ram * 10) / 10, models: TAB_DISK_MODELS.slice(), ...(choice === "allow" ? {} : { ask: true }) };
}
// The receiving side of an ai-load's offload on a tab (room/model-loader.js, before anything is read or written):
// null when it may go ahead, "ask" when its user must say yes first, else why it refuses. tabDisk: what this tab
// advertises (tabDiskMeta); choice: its user's answer (tabDiskChoice).
export function tabDiskGate(offload, { model, tabDisk = null, choice = null } = {}) {
  if (!offload) return null;
  if (choice === "deny") return "this device's user did not allow models on its disk";
  if (!tabDisk || !Array.isArray(tabDisk.models) || !tabDisk.models.includes(model)) return `this device did not offer its disk for ${model}`;
  const ram = (+offload.diskRamBytes || 0) + (+offload.ramSpare || 0);
  if (ram > (+tabDisk.ramGB || 0) * 2 ** 30 + 2 ** 20) return `the host asked for ${gb(ram)} GB of RAM for the disk tier; this device offered ${tabDisk.ramGB} GB`;
  return choice === "allow" ? null : "ask";
}

// ------------------------------------------------------------------ experts

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
// What storing these layers' experts takes on disk (their OPFS records), bytes: the consent prompt's number
export const tabExpertBytes = (G, layers) => (layers.length ? expertLayout(G, layers).totalBytes : 0);

// The model's content signature for its OPFS copy (engine/expert_opfs.js expertContentSig: a few bytes of every expert
// tensor), once per parsed header: the first load pays its small reads, later loads in this page reuse it
const sigMemo = new WeakMap();
export async function flashExpertSig(G, readAt) {
  let p = sigMemo.get(G);
  if (!p) { p = expertContentSig(G, readAt); sigMemo.set(G, p); p.catch(() => sigMemo.delete(G)); }
  return p;
}

// The disk tier for `layers` of a parsed (merged) header G: their experts in OPFS (copied there first when they are
// not, from readAt), a reader over them, a staging ring, and the ExpertStore the loader attaches them to.
//   readAt(shard, byteOffset, byteLength) -> Promise<Uint8Array>   the GGUF, by HTTP range (never the weight cache:
//            70 GB of experts there would be a second copy)
//   offload: { vramBytes, diskRamBytes }   the deal's GPU cache (pools) and RAM cache (T1)
//   onStatus(text) during the copy; onProgress(done, total, text) (text: the line onStatus last said, so a progress
//   handler that also sets the status keeps it); log(text)
//   opts: { workers (OPFS reader workers, 4), ring (staging ring slots, 64), storage (navigator.storage, tests),
//   contentSig (flashExpertSig's) }
// -> { store, reader, layout, ingest: { wroteBytes, skippedBytes, s } | null }. store.close() also frees the ring.
export async function tabExpertTier(device, G, layers, { readAt, offload, onStatus = () => {}, onProgress = () => {}, log = () => {}, workers = 4, ring = 64, storage = G_.navigator?.storage, contentSig = null } = {}) {
  let line = "checking this device's copy of the experts…";
  onStatus(line);
  const layout = expertLayout(G, layers, { contentSig: contentSig ?? await flashExpertSig(G, readAt) });
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
        const t = performance.now();
        if (t - last > 1000) { last = t; line = `copying the experts of ${layers.length} layers to this device's disk: ${gb(p.doneBytes)} of ${gb(p.totalBytes)} GB (once; later loads read them from there)`; onStatus(line); }
        onProgress(p.doneBytes, p.totalBytes, line);
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
    // the ring is the tier's: it goes with the store (the engine's close, or the loader's on a failed load)
    const close0 = store.close.bind(store);
    store.close = async () => { try { await close0(); } finally { const r = sring; sring = null; r?.destroy(); } };
    return { store, reader, layout, ingest };
  } catch (e) {
    sring?.destroy();
    await reader.close().catch(() => {});
    throw e;
  }
}

// Resident layers' experts from the OPFS copy: when a deal makes layers resident whose experts this tab already copied
// (a re-deal after a device joins: the Mac's layers 0-11 were on disk, now on its GPU), their stacked expert tensors
// are read from the copy's records straight into GPU buffers instead of downloaded again (and kept a second time in
// the weight cache). The records hold exactly the bytes the GGUF's conversion gives (engine/expert_opfs.js layout), so
// the engine computes the same logits either way. layers: the resident layers of this load; contentSig as
// tabExpertTier's. -> an OpfsResident (entry(info), close(), layers), or null when the copy holds none of them.
// Never asks for consent: it only reads what this origin already holds.
export async function opfsResidentExperts(device, G, layers, { readAt, contentSig = null, workers = 4, storage = G_.navigator?.storage } = {}) {
  if (!layers.length || !storage?.getDirectory) return null;
  // without a copy of this model at all, skip the content signature's reads
  const prefix = expertModelKey(G) + "-";
  let any = false;
  try {
    const root = await (await storage.getDirectory()).getDirectoryHandle(EXPERT_OPFS_DIR);
    for await (const name of root.keys()) if (name.startsWith(prefix)) { any = true; break; }
  } catch { return null; }
  if (!any) return null;
  const sig = contentSig ?? await flashExpertSig(G, readAt);
  const key = expertModelKey(G, sig), dir = [EXPERT_OPFS_DIR, key];
  const man = await readManifest(dir);
  if (!man) return null;
  // whole layers only, and only slices writeBuffer can place (4-byte multiples: every Q4_0 / Q8 layout is)
  const have = layers.filter((l) => {
    const one = expertLayout(G, [l], { contentSig: sig });
    return ingested(man, one) && one.layers[l].slices.every((s) => s % 4 === 0) && one.layers[l].offs.every((o) => o % 4 === 0);
  });
  if (!have.length) return null;
  const layout = expertLayout(G, have, { contentSig: sig });
  return new OpfsResident(device, layout, await OpfsExpertReader.open(layout, { dir, workers }));
}
const PART_J = { gate: 0, up: 2, down: 4 };
class OpfsResident {
  constructor(device, layout, reader) {
    this.device = device; this.layout = layout; this.reader = reader;
    this.layers = Object.keys(layout.layers).map(Number);
    this.built = new Map();   // layer -> Promise<{ gate, up, down }> (a part handed out is set to null)
    this.bytes = 0;           // record bytes read
  }
  owns(info) {
    try { return !!this.layout.layers[expertPartOf(info).layer]; } catch { return false; }
  }
  // info: a stacked expert tensor's entry (engine/gguf.js ggufEntry's, 2-D) -> { kind, shape, gpu: { kind, qs, sc } }
  // (streamEntryToGPU's shape: gpuUploadEntry keeps it as it is), or null when it is not one of this copy's layers
  async entry(info) {
    if (!this.owns(info)) return null;
    const { layer, part } = expertPartOf(info);
    if (!this.built.has(layer)) this.built.set(layer, this._build(layer));
    const parts = await this.built.get(layer);
    const e = parts[part];
    if (!e) throw new Error(`expert OPFS: ${info.name} was read twice`);
    parts[part] = null;
    if (!parts.gate && !parts.up && !parts.down) this.built.delete(layer);
    return { ...e, shape: info.shape };
  }
  // one layer's three stacked tensors: every expert's record once, its six slices into six GPU buffers
  async _build(layer) {
    const L = this.layout.layers[layer], dev = this.device, nE = L.nExp, U = G_.GPUBufferUsage;
    dev.pushErrorScope("out-of-memory");
    const bufs = L.slices.map((s) => dev.createBuffer({ size: Math.ceil(s * nE / 4) * 4, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC }));
    if (await dev.popErrorScope()) { bufs.forEach((b) => b.destroy()); throw new Error(`GPU out of memory while allocating layer ${layer}'s experts`); }
    try {
      const BATCH = 16;
      for (let e0 = 0; e0 < nE; e0 += BATCH) {
        const ids = Array.from({ length: Math.min(BATCH, nE - e0) }, (_, k) => e0 + k);
        const recs = await Promise.all(ids.map((e) => this.reader.record(layer, e)));
        recs.forEach((rec, k) => {
          const e = ids[k];
          L.slices.forEach((s, j) => dev.queue.writeBuffer(bufs[j], e * s, rec.buffer, rec.byteOffset + L.offs[j], s));
          this.bytes += L.recBytes;
        });
      }
    } catch (err) { bufs.forEach((b) => b.destroy()); throw err; }
    const out = {};
    for (const [p, j] of Object.entries(PART_J)) {
      const kind = expertKind(L.parts[p].ggmlType);
      out[p] = { kind, gpu: { kind, qs: bufs[j], sc: bufs[j + 1] } };
    }
    return out;
  }
  close() { return this.reader.close(); }
}

// ------------------------------------------------------------------ n-gram table

// The n-gram table's content signature (its OPFS directory: ngramDir): the file it is in (its URL pins the model's
// revision), where and how big it is, and FNV-1a over its first and last 4 KiB (two small range reads). readAt as
// splitHeads'. -> 8 hex digits
export async function ngramContentSig(cfg, readAt, { url = "" } = {}) {
  const total = cfg.nRows * cfg.rowBytes, k = Math.min(4096, total);
  const [a, b] = await Promise.all([readAt(cfg.shard, cfg.byteOffset, k), readAt(cfg.shard, cfg.byteOffset + total - k, k)]);
  let x = 0x811c9dc5;
  const s = JSON.stringify([url, cfg.shard, cfg.byteOffset, cfg.nRows, cfg.rowBytes, cfg.ggmlType, cfg.offsets, cfg.vocab]);
  for (let i = 0; i < s.length; i++) x = Math.imul(x ^ s.charCodeAt(i), 0x01000193) >>> 0;
  for (const buf of [a, b]) for (let i = 0; i < buf.length; i++) x = Math.imul(x ^ buf[i], 0x01000193) >>> 0;
  return x.toString(16).padStart(8, "0");
}

// One ingest of a directory at a time: across this origin's tabs (Web Locks), else within this page
const dirLocks = new Map();
async function withDirLock(name, fn) {
  if (G_.navigator?.locks?.request) return G_.navigator.locks.request(`pooled-opfs:${name}`, fn);
  const prev = dirLocks.get(name) || Promise.resolve();
  const run = prev.catch(() => {}).then(fn);
  const tail = run.catch(() => {});
  dirLocks.set(name, tail);
  try { return await run; } finally { if (dirLocks.get(name) === tail) dirLocks.delete(name); }
}

// The n-gram table into OPFS (dir: ngramDir(ngramContentSig)), split per head, unless every head file is there at its
// size. readAt as splitHeads' (file = the table's GGUF shard; a 4th { signal } argument aborts a read). Heads are copied
// `parallel` at a time, each through its own writable (written to a swap file, committed on close): a head is either
// all there or not, and a load that stops part way copies only the heads that are missing next time. The first failure
// stops the other heads at once (their reads aborted, their writables discarded). Concurrent ingests of one directory
// run one after the other (the second finds the heads there). Other copies under NGRAM_OPFS_DIR (an older table, the
// unkeyed layout before ngramDir) are removed first: only this table's copy is ever read.
// -> bytes copied (0 when it was all there)
export async function ingestNgram(cfg, readAt, { dir, parallel = 2, onStatus = () => {}, onProgress = () => {}, log = () => {}, storage = G_.navigator?.storage } = {}) {
  if (!dir) throw new Error("ingestNgram: dir (ngramDir(sig)) is required");
  return withDirLock(dir.join("/"), async () => {
    let d = await storage.getDirectory();
    for (const p of dir) d = await d.getDirectoryHandle(p, { create: true });
    const want = (h) => cfg.vocab[h] * cfg.rowBytes, missing = [];
    for (let h = 0; h < cfg.nHeads; h++) {
      let ok = false;
      try { ok = (await (await d.getFileHandle(ngramFileName(h))).getFile()).size === want(h); } catch { /* not there */ }
      if (!ok) missing.push(h);
    }
    if (!missing.length) return 0;
    await pruneNgram(storage, dir[dir.length - 1], log);
    const total = missing.reduce((a, h) => a + want(h), 0), t0 = performance.now();
    const st = await ingestStorage(total, { storage });
    if (st.warning) log(st.warning);
    const cur = new Map();   // head -> bytes copied so far (heads go in parallel)
    const ac = new AbortController();
    let failed = null, last = 0;
    const read = (f, o, n) => { if (failed) throw failed; return readAt(f, o, n, { signal: ac.signal }); };
    const one = async (h) => {
      const w = await (await d.getFileHandle(ngramFileName(h), { create: true })).createWritable({ keepExistingData: false });
      try {
        await splitHeads(cfg, read, (_h, bytes, at) => { if (failed) throw failed; return w.write({ type: "write", position: at, data: bytes }); }, { heads: [h], onProgress: (n) => {
          cur.set(h, n);
          const t = performance.now(), done = [...cur.values()].reduce((a, b) => a + b, 0);
          onProgress(done, total);
          if (t - last > 1000) { last = t; onStatus(`copying the n-gram table to this device's disk: ${gb(done)} of ${gb(total)} GB (once)`); }
        } });
        if (failed) throw failed;
        await w.close();
      } catch (e) { await w.abort?.().catch(() => {}); throw e; }
    };
    const queue = missing.slice();
    // allSettled: the lock is held until every head has stopped (a failed one's siblings abort, then settle)
    await Promise.allSettled(Array.from({ length: Math.min(parallel, queue.length) }, async () => {
      while (queue.length && !failed) {
        try { await one(queue.shift()); } catch (e) { if (!failed) { failed = e; ac.abort(); } throw e; }
      }
    }));
    if (failed) throw failed;
    log(`n-gram table: copied ${gb(total)} GB (${missing.length} heads) into OPFS in ${((performance.now() - t0) / 1000).toFixed(0)} s`);
    return total;
  });
}
// every entry under NGRAM_OPFS_DIR but `keep` (the current table's directory)
async function pruneNgram(storage, keep, log) {
  let d;
  try { d = await storage.getDirectory(); for (const p of NGRAM_OPFS_DIR) d = await d.getDirectoryHandle(p); } catch { return; }
  const names = [];
  for await (const name of d.keys()) if (name !== keep) names.push(name);
  for (const name of names) {
    try { await d.removeEntry(name, { recursive: true }); } catch (e) { log(`n-gram table: could not remove the old copy ${name} (${e?.name || e})`); }
  }
  if (names.length) log(`n-gram table: removed ${names.length} entr${names.length === 1 ? "y" : "ies"} of an older copy`);
}

// ------------------------------------------------------------------ disk space

async function dirBytes(d) {
  let n = 0;
  for await (const [, h] of d.entries()) n += h.kind === "file" ? (await h.getFile()).size : await dirBytes(h);
  return n;
}
async function dirAt(storage, path) {
  let d = await storage.getDirectory();
  for (const p of path) d = await d.getDirectoryHandle(p);
  return d;
}
// What this origin's Flash disk copies take: { experts, ngram, total } in bytes
export async function flashDiskBytes({ storage = G_.navigator?.storage } = {}) {
  const at = async (path) => { try { return await dirBytes(await dirAt(storage, path)); } catch { return 0; } };
  const experts = await at([EXPERT_OPFS_DIR]), ngram = await at(NGRAM_OPFS_DIR);
  return { experts, ngram, total: experts + ngram };
}
// Delete this origin's Flash copies (every model's expert copy under EXPERT_OPFS_DIR, through removeExperts, and the
// n-gram table). A copy the running model is reading (open sync handles) can't go: stop the model first.
// -> { removed: bytes, kept: bytes still there }
export async function removeFlashCopy({ storage = G_.navigator?.storage } = {}) {
  const before = await flashDiskBytes({ storage });
  try {
    const d = await dirAt(storage, [EXPERT_OPFS_DIR]), keys = [];
    for await (const name of d.keys()) keys.push(name);
    for (const k of keys) await removeExperts(k);
  } catch { /* none */ }
  try { await (await dirAt(storage, NGRAM_OPFS_DIR.slice(0, -1))).removeEntry(NGRAM_OPFS_DIR[NGRAM_OPFS_DIR.length - 1], { recursive: true }); } catch { /* none, or in use */ }
  const after = await flashDiskBytes({ storage });
  return { removed: Math.max(0, before.total - after.total), kept: after.total };
}
