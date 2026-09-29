// Converted-weights cache for the browser (issue #77): the tensors the loader converts on the CPU
// (K-quant -> Q8 requant, BF16/F16/Q5_0 -> f32, the embedding's Q4_0/Q8_0 repack) are kept in the
// origin-private file system (OPFS), so a returning device skips the conversion. The raw byte
// ranges are cached separately (room.js rangeFetch, Cache API); this caches what comes out of
// engine/gguf.js convertEntry for them. The streamed Q4_0/Q8_0 matrices never pass through here.
//
// It plugs into the same hook as the CLI cache (tests/weight_cache.js): engine/gguf.js ggufEntry
// calls G.entryCache.get(info) before converting and G.entryCache.put(info, entry) after. Entries
// use the CLI's file format (64-byte header, then the payload), so the two stay easy to compare.
//
// Keying: one OPFS directory per (model URL, pinned revision, GGUF header fingerprint, converter
// version). The revision is the "/resolve/<rev>/" part of the URL; the fingerprint hashes the
// header's metadata sizes and tensor table, so a re-upload that moves or retypes a tensor gets a
// new directory even under "resolve/main"; the converter version hashes engine/gguf.js (the same
// string as the CLI's LOADER_VERSION), so any change to the conversion starts fresh. Opening a
// key deletes the other directories of the same model (same URL up to the revision): they can
// never be read again.
//
// Safety: an entry is written under a temp name and renamed into place; on read the header and
// the exact file size are checked against the tensor info, and anything off is dropped and
// converted fresh. Quota: an entry is only written while the origin keeps a reserve free (the raw
// download cache matters more than this one); a quota error makes the cache read-only for the load.
// Everything here is best effort: any failure means "convert as before", never a failed load.

export const WCACHE_EPOCH = 1;          // same meaning as LOADER_EPOCH in tests/weight_cache.js
export const WCACHE_DIR = "pooled-converted-weights";
export const MIN_ENTRY_BYTES = 256 * 1024;   // smaller tensors convert faster than a file open
const MAGIC = 0x43575753;               // "SWWC", the CLI cache's format
const FORMAT = 1;
export const HDR = 64;
const KIND = { f32: 0, q4: 1, q8: 2 }, KIND_NAME = ["f32", "q4", "q8"];
const GGML_F32 = 0;

const hex = (u8, n) => [...u8.slice(0, n)].map((b) => b.toString(16).padStart(2, "0")).join("");
export async function sha256Hex(data, nBytes = 32) {
  const u8 = typeof data === "string" ? new TextEncoder().encode(data) : data;
  return hex(new Uint8Array(await crypto.subtle.digest("SHA-256", u8)), nBytes);
}

// The converter version from engine/gguf.js's source bytes: "e<epoch>-<16 hex>", the same string
// tests/weight_cache.js computes as LOADER_VERSION.
export async function converterVersion(src) {
  return `e${WCACHE_EPOCH}-` + await sha256Hex(src, 8);
}

// The pinned revision in a Hugging Face style URL (".../resolve/<rev>/file"), "" when none.
export function revisionOf(url) {
  const m = /\/resolve\/([^/?#]+)\//.exec(String(url || ""));
  return m ? decodeURIComponent(m[1]) : "";
}

// The model a URL names, whatever its revision: ".../resolve/<rev>/file" -> ".../resolve/*/file".
// Directories of the same model under another revision are stale and get deleted.
export function modelOf(url) {
  return String(url || "").split(/[?#]/)[0].replace(/\/resolve\/[^/]+\//, "/resolve/*/");
}

// What identifies the GGUF's content short of hashing the weights: where the data starts, the
// metadata keys with a small stand-in for each value, and every tensor's name, type, shape and
// offset. The tokenizer keys are left out (a worker parses the header without them), so a header
// parsed with or without the tokenizer gives the same fingerprint.
export function headerFingerprint(G) {
  const meta = Object.keys(G.meta || {}).filter((k) => !k.startsWith("tokenizer.")).sort().map((k) => {
    const v = G.meta[k];
    return [k, Array.isArray(v) ? `[${v.length}]` : v];
  });
  const tensors = Object.values(G.tensors || {}).map((t) => [t.name, t.ggmlType, t.shape, t.offset ?? t.byteOffset])
    .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  return JSON.stringify({ dataStart: G.dataStart, meta, tensors });
}

// Directory name for one cache key: "<file stem>-<16 hex>".
export async function cacheDirName({ url, revision = "", fingerprint = "", converter, variant = "" }) {
  const file = String(url).split(/[?#]/)[0].split("/").pop() || "model";
  const stem = file.replace(/\.gguf$/i, "").replace(/[^A-Za-z0-9._-]/g, "_").slice(0, 48);
  const fp = await sha256Hex(fingerprint);
  return `${stem}-${await sha256Hex(JSON.stringify({ url, revision, fp, converter, variant }), 8)}`;
}

export const entryFile = (name) => name.replace(/[^A-Za-z0-9._-]/g, "_") + ".bin";

// Worth a file: converted (not a plain f32 copy) and big enough that the conversion costs more
// than opening a file.
export function worthCaching(info) {
  return info.ggmlType !== GGML_F32 && info.byteLength >= MIN_ENTRY_BYTES;
}

// Room left to write `bytes` while keeping a reserve free: max(1 GB, 15% of the quota).
// est: navigator.storage.estimate() output; unknown quota -> no write.
export function quotaAllows(est, bytes) {
  const quota = est?.quota, usage = est?.usage;
  if (!(quota > 0) || !(usage >= 0)) return false;
  const reserve = Math.max(2 ** 30, quota * 0.15);
  return usage + bytes <= quota - reserve;
}

// 32-bit multiply-xor hash of the payload (tests/weight_cache.js payloadHash, same value).
export function payloadHash(...parts) {
  let h = 0x811C9DC5, total = 0;
  for (const u8 of parts) {
    const n4 = u8.byteLength >>> 2;
    const w = (u8.byteOffset & 3) === 0 ? new Uint32Array(u8.buffer, u8.byteOffset, n4) : new Uint32Array(u8.slice(0, n4 * 4).buffer);
    for (let i = 0; i < n4; i++) h = Math.imul(h ^ w[i], 0x01000193);
    for (let i = n4 * 4; i < u8.byteLength; i++) h = Math.imul(h ^ u8[i], 0x01000193);
    total += u8.byteLength;
  }
  return (h ^ total) >>> 0;
}

// Byte layout of an entry for this tensor/kind: [qs | pad to 4 | scales] or [data]
export function entryLayout(info, kind) {
  const n = info.nElems, nb = Math.ceil(n / 32);
  if (kind === "f32") return { a: n * 4, pad: 0, b: 0 };
  const a = kind === "q4" ? n / 2 : nb * 32, pad = (4 - (a & 3)) & 3;
  return { a, pad, b: Math.ceil(nb / 2) * 4 };
}

const u8of = (v) => new Uint8Array(v.buffer, v.byteOffset, v.byteLength);

// A converted entry as file parts [header, ...payload], or null when it is not a layout we know.
export function encodeEntry(info, e) {
  if (!e || e.gpu || !(e.kind in KIND)) return null;
  const kind = e.kind, L = entryLayout(info, kind);
  const parts = kind === "f32" ? [u8of(e.data)] : [u8of(e.qs), u8of(e.scales)];
  if (parts[0].byteLength !== L.a || (parts[1]?.byteLength ?? 0) !== L.b) return null;
  const body = kind === "f32" ? parts : [parts[0], new Uint8Array(L.pad), parts[1]];
  const hdr = new Uint8Array(HDR), dv = new DataView(hdr.buffer);
  dv.setUint32(0, MAGIC, true); dv.setUint32(4, FORMAT, true); dv.setUint32(8, KIND[kind], true);
  dv.setUint32(12, info.ggmlType, true); dv.setFloat64(16, info.nElems, true);
  dv.setFloat64(24, L.a, true); dv.setFloat64(32, L.b, true); dv.setUint32(40, payloadHash(...body), true);
  return { parts: [hdr, ...body], bytes: HDR + L.a + L.pad + L.b };
}

// A whole entry file (ArrayBuffer) back to the entry the engine uploads; throws when the header or
// size does not match this tensor. verify also checks the payload hash.
export function decodeEntry(info, buf, verify = false) {
  if (buf.byteLength < HDR) throw new Error("short header");
  const dv = new DataView(buf, 0, HDR);
  if (dv.getUint32(0, true) !== MAGIC || dv.getUint32(4, true) !== FORMAT) throw new Error("bad magic");
  const kind = KIND_NAME[dv.getUint32(8, true)];
  if (!kind || dv.getUint32(12, true) !== info.ggmlType || dv.getFloat64(16, true) !== info.nElems) throw new Error("wrong tensor");
  const L = entryLayout(info, kind);
  if (dv.getFloat64(24, true) !== L.a || dv.getFloat64(32, true) !== L.b) throw new Error("wrong layout");
  const size = HDR + L.a + L.pad + L.b;
  if (buf.byteLength !== size) throw new Error(`size ${buf.byteLength} != ${size}`);
  if (verify && payloadHash(new Uint8Array(buf, HDR)) !== dv.getUint32(40, true)) throw new Error("checksum");
  if (kind === "f32") return { kind, data: new Float32Array(buf, HDR, info.nElems) };
  return { kind, qs: new Uint8Array(buf, HDR, L.a), scales: new Uint32Array(buf, HDR + L.a + L.pad, L.b / 4), shape: info.shape };
}

async function readJSON(dir, name) {
  try { return JSON.parse(await (await (await dir.getFileHandle(name)).getFile()).text()); } catch { return null; }
}
async function writeWhole(dir, name, parts) {
  const w = await (await dir.getFileHandle(name, { create: true })).createWritable();
  try { for (const p of parts) await w.write(p); await w.close(); }
  catch (err) { try { await w.abort?.(); } catch { /* already closed */ } throw err; }
}

export class BrowserWeightCache {
  // dir: this key's OPFS directory handle; have: entry file names already in it
  constructor(dir, meta, { have = new Set(), estimate = null, verify = false, policy = worthCaching } = {}) {
    this.dir = dir; this.meta = meta; this.have = have;
    this.estimate = estimate; this.verify = verify; this.policy = policy;
    this.readOnly = !estimate;
    this.stats = { hit: 0, miss: 0, bad: 0, write: 0, skip: 0, full: 0, hitBytes: 0, writeBytes: 0, readMs: 0, writeMs: 0 };
  }

  // Open (creating) the directory for this model under root, drop the model's other directories.
  // opts: { url, revision, G, converter, variant, estimate, verify, policy }. null when unusable.
  static async open(root, opts) {
    try {
      const top = await root.getDirectoryHandle(WCACHE_DIR, { create: true });
      const meta = { url: opts.url, revision: opts.revision ?? revisionOf(opts.url), converter: opts.converter, variant: opts.variant || "" };
      const name = await cacheDirName({ ...meta, fingerprint: headerFingerprint(opts.G) });
      meta.model = modelOf(opts.url);
      await pruneOthers(top, meta.model, name);
      const dir = await top.getDirectoryHandle(name, { create: true });
      if (!(await readJSON(dir, "meta.json"))) await writeWhole(dir, "meta.json", [new TextEncoder().encode(JSON.stringify(meta))]);
      const have = new Set();
      for await (const [n] of dir.entries()) {
        if (n.endsWith(".bin")) have.add(n);
        else if (n.includes(".tmp-")) dir.removeEntry(n).catch(() => {});   // a tab that died mid-write
      }
      const c = new BrowserWeightCache(dir, { ...meta, dir: name }, { have, estimate: opts.estimate, verify: opts.verify, policy: opts.policy });
      return c;
    } catch { return null; }
  }

  // The cached entry for this tensor, or null (not cached, or a bad file: then it is deleted).
  async get(info) {
    const f = entryFile(info.name);
    if (!this.have.has(f)) { this.stats.miss++; return null; }
    const t0 = performance.now();
    try {
      const buf = await (await (await this.dir.getFileHandle(f)).getFile()).arrayBuffer();
      const e = decodeEntry(info, buf, this.verify);
      this.stats.hit++; this.stats.hitBytes += buf.byteLength; this.stats.readMs += performance.now() - t0;
      return e;
    } catch {
      this.stats.bad++; this.have.delete(f);
      this.dir.removeEntry(f).catch(() => {});
      return null;
    }
  }

  // Store a freshly converted entry, when the policy wants it and the quota has room.
  async put(info, e) {
    if (this.readOnly || !this.policy(info)) { this.stats.skip++; return; }
    const enc = encodeEntry(info, e);
    if (!enc) { this.stats.skip++; return; }
    let est = null;
    try { est = await this.estimate(); } catch { /* unknown: no write */ }
    if (!quotaAllows(est, enc.bytes)) { this.stats.full++; return; }
    const f = entryFile(info.name), tmp = `${f}.tmp-${Math.random().toString(36).slice(2)}`;
    const t0 = performance.now();
    try {
      const h = await this.dir.getFileHandle(tmp, { create: true });
      const w = await h.createWritable();
      try { for (const p of enc.parts) await w.write(p); await w.close(); }
      catch (err) { try { await w.abort?.(); } catch { /* closed */ } throw err; }
      if (h.move) {
        try { await h.move(f); }
        catch (err) { await this.dir.removeEntry(tmp).catch(() => {}); throw err; }
      } else {
        // no rename here: write the real name directly (a torn file fails the size check on read)
        await this.dir.removeEntry(tmp).catch(() => {});
        await writeWhole(this.dir, f, enc.parts);
      }
      this.have.add(f);
      this.stats.write++; this.stats.writeBytes += enc.bytes; this.stats.writeMs += performance.now() - t0;
    } catch (err) {
      // out of space (QuotaExceededError) or no writable files in this browser: read-only from here
      this.readOnly = true;
      if (err?.name === "QuotaExceededError") this.stats.full++;
      await this.dir.removeEntry(tmp).catch(() => {});
    }
  }

  summary() {
    const s = this.stats, mb = (b) => (b / 2 ** 20).toFixed(0) + " MB";
    return `converted weights: ${s.hit} from this device (${mb(s.hitBytes)}, ${(s.readMs / 1000).toFixed(1)} s), ` +
      `${s.write} saved (${mb(s.writeBytes)}, ${(s.writeMs / 1000).toFixed(1)} s)` +
      (s.bad ? `, ${s.bad} bad dropped` : "") + (s.full ? `, ${s.full} not saved (storage nearly full)` : "");
  }
}

// Delete every directory under the cache root whose meta.json names this model, except keep.
async function pruneOthers(top, model, keep) {
  const drop = [];
  for await (const [n, h] of top.entries()) {
    if (n === keep || h.kind !== "directory") continue;
    const m = await readJSON(h, "meta.json");
    if (m && (m.model ?? modelOf(m.url)) === model) drop.push(n);
  }
  for (const n of drop) await top.removeEntry(n, { recursive: true }).catch(() => {});
  return drop;
}

// Bytes of converted weights kept on this device (all models).
export async function convertedBytes(root) {
  let n = 0;
  try {
    const top = await root.getDirectoryHandle(WCACHE_DIR);
    for await (const [, d] of top.entries()) {
      if (d.kind !== "directory") continue;
      for await (const [f, h] of d.entries()) if (f.endsWith(".bin")) n += (await h.getFile()).size;
    }
  } catch { /* none yet */ }
  return n;
}

// Forget all converted weights (the "Clear cached weights" menu item).
export async function clearConverted(root) {
  try { await root.removeEntry(WCACHE_DIR, { recursive: true }); } catch { /* none yet */ }
}

// The loader hook for one load: G.entryCache set to this model's cache, or cleared when the cache
// cannot be used here (no OPFS, private window, ?wcache=0). storage: navigator.storage; srcUrl:
// engine/gguf.js as served, hashed once per page for the converter version.
let converterMemo = null;
export async function attachBrowserWeightCache(G, url, { storage = globalThis.navigator?.storage, srcUrl, fetchFn = globalThis.fetch, verify = false } = {}) {
  G.entryCache = null;
  if (!storage?.getDirectory || !srcUrl) return null;
  try {
    const key = String(srcUrl);
    if (!converterMemo || converterMemo.key !== key) {
      const r = await fetchFn(key);
      if (!r.ok) return null;
      converterMemo = { key, v: await converterVersion(new Uint8Array(await r.arrayBuffer())) };
    }
    const c = await BrowserWeightCache.open(await storage.getDirectory(), {
      url, G, converter: converterMemo.v, verify,
      estimate: storage.estimate ? () => storage.estimate() : null,
    });
    G.entryCache = c;
    return c;
  } catch { return null; }
}
