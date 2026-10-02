// Pinned prefix checkpoints on disk, so a restarted host (an OpenClaw gateway restart, `pooled host`
// again) answers its first question from the system prompt + tools it already read, instead of
// prefilling ~16k tokens again. DOM-free, no GPU: plain files under the user's home.
//
// What is kept: the pinned checkpoints only (ckpt.js: a prompt's fixed start, the system prompt and
// tools, and an agent's cache boundary). Answer and turn checkpoints change every call and stay in
// GPU memory. Each device writes its own layers' part (engine exportSlot: its KV rows [0, n), its
// DeltaNet states and conv windows, as raw GPU bytes: what importState uploads with one writeBuffer
// per part) when the host pins a prefix, and reads it back when the room comes online again.
//
// When a copy may be used. A KV state is only valid for exactly the computation that made it, so
// every file is keyed by, and its header repeats in full:
//   - room: the room key (roomKey below): the model, the context, the KV mode and EVERY device's
//     model file id and engine state signature (engine stateSignature(): its layer range, KV format,
//     wide-prefill ubatch, dp4a, prefill math), in chain order. A worker's KV depends on the hidden
//     states the devices before it computed, so a change anywhere in the room changes every key;
//   - local: this device's own { model, file, sig, ctx, kv } (checked again against the engine);
//   - h: a SHA-256 of the model and the exact token ids of the prefix (the host keeps the ids too,
//     and checks they hash to h before it indexes them).
// Anything that does not match exactly, a damaged or truncated file, or another format version reads
// as missing (and is removed): a device without a matching copy says so, and the host prefills as it
// would have without the cache. Never a resume from a wrong state.
//
// Files: <dir>/<room key 16>.<local key 16>.<h 16>.pkv, dir ~/.pooled/cache/ckpt (POOLED_CKPT_DIR),
// mode 0700, files 0600: they hold prompt-derived state (the attention KV of the system prompt and
// tools; the prompt's text can be approximately recovered from it). Layout: [u32 magic][u32 header
// length][header JSON], zero padding to a 4096-byte boundary, then the parts, each padded to 4 bytes.
// Written to <name>.<pid>.<random>.tmp, fsync'd and renamed: a crash never leaves a half file under a
// real name; a temp file older than 10 minutes is a write that never finished and is removed.
// Least recently used copies go first when the cache would pass its cap (POOLED_CKPT_GB, default 4).
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";

export const DISK_FORMAT = 1;
const MAGIC = 0x504b5631;   // "PKV1"
const EXT = ".pkv";
const ALIGN = 4096;
export const DEFAULT_CAP_GB = 4;
export const defaultCkptDir = (env = process.env) => env.POOLED_CKPT_DIR || path.join(os.homedir(), ".pooled", "cache", "ckpt");

const sha = (s) => crypto.createHash("sha256").update(s).digest("hex");
// a stable JSON (sorted object keys), so two equal keys always hash the same
export function canon(v) {
  if (Array.isArray(v)) return "[" + v.map(canon).join(",") + "]";
  if (v && typeof v === "object") return "{" + Object.keys(v).sort().filter((k) => v[k] !== undefined).map((k) => JSON.stringify(k) + ":" + canon(v[k])).join(",") + "}";
  return JSON.stringify(v ?? null);
}
// the prefix: the model and its exact token ids -> 64 hex
export function prefixHash(model, ids) {
  const h = crypto.createHash("sha256").update(String(model) + "\0");
  const a = Uint32Array.from(ids);
  return h.update(new Uint8Array(a.buffer, 0, a.byteLength)).digest("hex");
}
// this device: { model, file, sig, ctx, kv } -> 64 hex
export const localKey = (local) => sha("local\0" + canon(local));
// the whole room: { model, ctx, kv, devices: [{ file, sig }] in chain order, host first } -> 64 hex
export const roomKey = ({ model, ctx, kv, devices }) => sha("room\0" + canon({ f: DISK_FORMAT, model, ctx, kv, devices }));

// A model file's identity, cheap enough for every load: its size and a hash of its first 4 MiB (the
// GGUF header: metadata, tensor names, types, shapes and offsets) and its last 1 MiB. src: source.js
// openModel(). -> 32 hex, or null when it cannot be read (the disk cache stays off for that device)
export async function modelFileId(src) {
  try {
    let size = null;
    if (src.local) size = (await fs.promises.stat(src.local)).size;
    const head = await src.readAt(0, 4 * 2 ** 20);
    const h = crypto.createHash("sha256").update(String(src.url || "") + "\0" + size + "\0").update(head);
    if (size && size > 5 * 2 ** 20) h.update(await src.readAt(size - 2 ** 20, 2 ** 20));
    return h.digest("hex").slice(0, 32);
  } catch { return null; }
}

export class CkptDiskError extends Error {}

export function encodeHeader(h) {
  const json = Buffer.from(JSON.stringify({ f: DISK_FORMAT, ...h }));
  const len = Math.ceil((8 + json.length) / ALIGN) * ALIGN;
  const out = Buffer.alloc(len);
  out.writeUInt32BE(MAGIC, 0); out.writeUInt32BE(json.length, 4); json.copy(out, 8);
  return out;
}
// the first bytes of a file -> { header, off (where the parts start) }, or throws
export function decodeHeader(buf) {
  if (buf.length < 8) throw new CkptDiskError("too short");
  if (buf.readUInt32BE(0) !== MAGIC) throw new CkptDiskError("not a checkpoint file");
  const n = buf.readUInt32BE(4);
  if (buf.length < 8 + n) throw new CkptDiskError("header truncated");
  let h;
  try { h = JSON.parse(buf.subarray(8, 8 + n).toString("utf8")); } catch { throw new CkptDiskError("header is not JSON"); }
  if (h?.f !== DISK_FORMAT) throw new CkptDiskError(`format ${h?.f}, this build reads ${DISK_FORMAT}`);
  if (!Array.isArray(h.sizes) || !Number.isInteger(h.pos) || typeof h.h !== "string" || typeof h.room !== "string") throw new CkptDiskError("header incomplete");
  return { header: h, off: Math.ceil((8 + n) / ALIGN) * ALIGN };
}
const pad4 = (n) => (n + 3) & ~3;
const bodyBytes = (sizes) => sizes.reduce((s, n) => s + pad4(n), 0);
// why a header cannot be used for this lookup, or null
export function mismatch(hd, { room, local, h }) {
  if (hd.room !== room) return "room";
  if (canon(hd.local) !== canon(local)) return "device";
  if (h != null && hd.h !== h) return "prefix";
  if (Array.isArray(hd.ids) && (hd.ids.length !== hd.pos || prefixHash(hd.local?.model, hd.ids) !== hd.h)) return "ids";
  return null;
}

export class CkptDisk {
  // dir: where the files go; capBytes: the most the cache keeps (least recently used out first);
  // tmpStaleMs: a temp file this old is a write that never finished
  constructor({ dir = defaultCkptDir(), capBytes = DEFAULT_CAP_GB * 2 ** 30, tmpStaleMs = 10 * 60e3, log = () => {} } = {}) {
    this.dir = dir; this.cap = capBytes; this.tmpStaleMs = tmpStaleMs; this.log = log;
    this.q = Promise.resolve();
    this.writes = 0; this.failures = 0;
  }
  // POOLED_CKPT_DISK=0 turns it off; POOLED_CKPT_GB its cap; POOLED_CKPT_DIR where it lives
  static fromEnv(env = process.env, opts = {}) {
    if (env.POOLED_CKPT_DISK === "0" || env.POOLED_CKPT_DISK === "false") return null;
    const gb = +(env.POOLED_CKPT_GB || DEFAULT_CAP_GB);
    return new CkptDisk({ dir: defaultCkptDir(env), capBytes: (gb > 0 ? gb : DEFAULT_CAP_GB) * 2 ** 30, ...opts });
  }
  // one operation at a time, in call order
  _run(fn) { const p = this.q.then(fn); this.q = p.catch(() => {}); return p; }
  _ensureDir() {
    fs.mkdirSync(this.dir, { recursive: true, mode: 0o700 });
    try { fs.chmodSync(this.dir, 0o700); } catch {}
  }
  name({ room, local, h }) { return `${room.slice(0, 16)}.${localKey(local).slice(0, 16)}.${h.slice(0, 16)}${EXT}`; }
  file(where) { return path.join(this.dir, this.name(where)); }

  // is there a copy for exactly this (room, device, prefix)? (header checked; touches it)
  has(where) { return this._run(async () => !!(await this._header(this.file(where), where, true))); }
  async _header(file, where, touch = false) {
    let fd;
    try { fd = await fs.promises.open(file, "r"); } catch { return null; }
    try {
      const st = await fd.stat();
      const first = Buffer.alloc(Math.min(st.size, 64 * 1024));
      await fd.read(first, 0, first.length, 0);
      let n = first.length >= 8 && first.readUInt32BE(0) === MAGIC ? first.readUInt32BE(4) : 0;
      let buf = first;
      if (8 + n > first.length && 8 + n <= st.size) { buf = Buffer.alloc(8 + n); await fd.read(buf, 0, buf.length, 0); }
      const { header, off } = decodeHeader(buf);
      if (mismatch(header, where) || st.size !== off + bodyBytes(header.sizes)) throw new CkptDiskError("mismatch");
      if (touch) { const now = new Date(); await fs.promises.utimes(file, now, now).catch(() => {}); }
      return { header, off, size: st.size, t: st.mtimeMs };
    } catch {
      await fd.close().catch(() => {}); fd = null;
      await fs.promises.rm(file, { force: true }).catch(() => {});
      return null;
    } finally { await fd?.close().catch(() => {}); }
  }

  // Write one prefix's state. where: { room, local, h }; extra: { ids } (the host's copy keeps the
  // token ids, to index the prefix again); state: { pos, parts: [ArrayBuffer | view] } or an async
  // function returning one (read when the write's turn comes; a throw means the slot is gone: skip).
  // -> true when written (or already there)
  put(where, state, extra = {}) { return this._run(() => this._put(where, state, extra)); }
  async _put(where, state, extra) {
    const file = this.file(where);
    if (await this._header(file, where, true)) return true;
    let st;
    try { st = typeof state === "function" ? await state() : state; } catch (e) { this.failures++; this.log(`checkpoint not saved to disk: ${e.message}`); return false; }
    if (!st?.parts) { this.failures++; return false; }
    const sizes = st.parts.map((p) => p.byteLength);
    const head = encodeHeader({ room: where.room, local: where.local, h: where.h, pos: st.pos, sizes, ...extra, t: Date.now() });
    const bytes = head.length + bodyBytes(sizes);
    if (bytes > this.cap) { this.failures++; return false; }
    let tmp = null, fd = null;
    try {
      this._ensureDir();
      await this._evict(this.cap - bytes);
      tmp = `${file}.${process.pid}.${crypto.randomBytes(4).toString("hex")}.tmp`;
      fd = await fs.promises.open(tmp, "wx", 0o600);
      await fd.write(head, 0, head.length, 0);
      let off = head.length;
      const zero = Buffer.alloc(4);
      for (const p of st.parts) {
        const u8 = p instanceof ArrayBuffer ? new Uint8Array(p) : new Uint8Array(p.buffer, p.byteOffset, p.byteLength);
        for (let o = 0; o < u8.length; o += 1 << 28) { const c = u8.subarray(o, Math.min(u8.length, o + (1 << 28))); await fd.write(c, 0, c.length, off + o); }
        off += u8.length;
        if (pad4(u8.length) > u8.length) { await fd.write(zero, 0, pad4(u8.length) - u8.length, off); off = pad4(off); }
      }
      await fd.sync();
      await fd.close(); fd = null;
      await fs.promises.rename(tmp, file); tmp = null;
      this.writes++;
      return true;
    } catch (e) {
      this.failures++;
      this.log(`checkpoint not saved to disk: ${e.message}`);
      return false;
    } finally {
      await fd?.close().catch(() => {});
      if (tmp) await fs.promises.rm(tmp, { force: true }).catch(() => {});
    }
  }

  // The copies this device can load in this room: [{ h, n, ids?, t }], most recently used first.
  // Reads headers only; a file of this device in this room that does not check out is removed.
  list(where) { return this._run(() => this._list(where)); }
  async _list({ room, local }) {
    const pre = `${room.slice(0, 16)}.${localKey(local).slice(0, 16)}.`, out = [];
    for (const f of await this._files()) {
      if (f.tmp || !f.name.startsWith(pre)) continue;
      const got = await this._header(f.path, { room, local });
      if (got) out.push({ h: got.header.h, n: got.header.pos, ids: got.header.ids || null, t: got.t });
    }
    return out.sort((a, b) => b.t - a.t);
  }
  // -> { pos, parts: [Uint8Array] (views of one buffer, 4-byte aligned), ids? } or null
  get(where) { return this._run(() => this._get(where)); }
  async _get(where) {
    const file = this.file(where);
    const got = await this._header(file, where, true);
    if (!got) return null;
    let buf;
    try { buf = await fs.promises.readFile(file); } catch { return null; }
    if (buf.length !== got.size) return null;   // changed under us
    let off = got.off;
    const parts = got.header.sizes.map((n) => { const p = new Uint8Array(buf.buffer, buf.byteOffset + off, n); off += pad4(n); return p; });
    return { pos: got.header.pos, parts, ids: got.header.ids || null };
  }
  drop(where) { return this._run(() => fs.promises.rm(this.file(where), { force: true }).catch(() => {})); }

  // every file in the dir: { name, path, bytes, t (last use), tmp }; temp files of writes that never
  // finished are removed here
  async _files() {
    let names;
    try { names = await fs.promises.readdir(this.dir); } catch { return []; }
    const out = [];
    for (const name of names) {
      const tmp = name.endsWith(".tmp");
      if (!name.endsWith(EXT) && !(tmp && name.includes(EXT + "."))) continue;
      const p = path.join(this.dir, name);
      let st;
      try { st = await fs.promises.stat(p); } catch { continue; }
      if (!st.isFile()) continue;
      if (tmp && Date.now() - st.mtimeMs > this.tmpStaleMs) { await fs.promises.rm(p, { force: true }).catch(() => {}); continue; }
      out.push({ name, path: p, bytes: st.size, t: st.mtimeMs, tmp });
    }
    return out;
  }
  // drop the least recently used copies until at most `limit` bytes are kept
  async _evict(limit) {
    const all = await this._files();
    let total = all.reduce((s, f) => s + f.bytes, 0);   // (writes in progress count, and stay)
    for (const f of all.filter((x) => !x.tmp).sort((a, b) => a.t - b.t)) {
      if (total <= limit) break;
      await fs.promises.rm(f.path, { force: true }).catch(() => {});
      total -= f.bytes;
    }
  }
  // bytes on disk now
  async size() { return (await this._files()).reduce((s, f) => s + f.bytes, 0); }
}
