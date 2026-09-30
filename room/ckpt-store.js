// Room checkpoints on disk (the browser's origin-private file system, OPFS), so a room's state
// survives a reload (roadmap 30, issue #69). DOM-free.
//
// Each device keeps its own layers' part of a checkpoint on the GPU (?ckpt=N, engine saveSlot) and,
// with this, a copy on disk under the same slot number the frame header carries. A device that
// reloads (or a host that reloads and deals the layers again) reads its copies back into GPU slots
// before it says it is ready, so the host can load a checkpoint instead of prefilling the whole
// conversation. No new message: saves, drops and loads are still the sv / dp / ld of the frame
// header, and a device lists the slots it read back in its ai-ready, so the host forgets any it lacks.
//
// A copy is only valid for exactly the room, model and layer range (the engine's stateSignature(),
// which also covers the KV format) it was made on, so the file name carries the room code and a hash
// of { model, sig }, and the header repeats them in full; anything that does not match, or a file in
// another format version, reads as missing and is removed.
//
// File: [u32 magic][u32 header length][header JSON][part 0][part 1]... Written to a temp name and
// renamed, so a crash never leaves a half file under a real name. A temp file left by a tab that
// reloaded mid-write counts against the budget while it is fresh (another tab of this origin may
// still be writing it) and is removed once it is older than tmpStaleMs. The old copy of a slot is removed
// before a new one is written, so a failed write leaves the slot missing (a prefill), never stale.

export const CKPT_FORMAT = 1;
const MAGIC = 0x504f434b;   // "POCK"
const EXT = ".ckpt";

export class CkptFormatError extends Error {}

// the header bytes that go before the parts
export function encodeHeader({ room, model, sig, slot, pos, sizes, meta = {}, t = Date.now() }) {
  const json = new TextEncoder().encode(JSON.stringify({ f: CKPT_FORMAT, room, model, sig, slot, pos, sizes, meta, t }));
  const out = new Uint8Array(8 + json.length), dv = new DataView(out.buffer);
  dv.setUint32(0, MAGIC); dv.setUint32(4, json.length);
  out.set(json, 8);
  return out;
}
// the header length (bytes before the parts) from the first 8 bytes, or throws
export function headerLength(first8) {
  if (first8.byteLength < 8) throw new CkptFormatError("checkpoint file too short");
  const dv = new DataView(first8);
  if (dv.getUint32(0) !== MAGIC) throw new CkptFormatError("not a checkpoint file");
  return 8 + dv.getUint32(4);
}
// -> the header object, or throws (bad magic, truncated, bad JSON, another format version)
export function decodeHeader(buf) {
  const n = headerLength(buf.slice(0, 8));
  if (buf.byteLength < n) throw new CkptFormatError("checkpoint header truncated");
  let h;
  try { h = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n - 8))); } catch { throw new CkptFormatError("checkpoint header is not JSON"); }
  if (h?.f !== CKPT_FORMAT) throw new CkptFormatError(`checkpoint format ${h?.f}, this build reads ${CKPT_FORMAT}`);
  if (!Array.isArray(h.sizes) || !Number.isInteger(h.slot) || !Number.isInteger(h.pos)) throw new CkptFormatError("checkpoint header incomplete");
  return h;
}
// a whole file -> { header, parts: [ArrayBuffer] }, or throws
export function decodeCkpt(buf) {
  const header = decodeHeader(buf);
  let off = headerLength(buf.slice(0, 8));
  const total = header.sizes.reduce((s, n) => s + n, 0);
  if (buf.byteLength !== off + total) throw new CkptFormatError(`checkpoint body is ${buf.byteLength - off} bytes, header says ${total}`);
  const parts = header.sizes.map((n) => { const p = buf.slice(off, off + n); off += n; return p; });
  return { header, parts };
}
// why a header cannot be loaded here (a short reason), or null when it can
export function mismatch(h, { room, model, sig }) {
  if (h.room !== room) return "room";
  if (h.model !== model) return "model";
  if (JSON.stringify(h.sig) !== JSON.stringify(sig)) return "layers";
  return null;
}

// room codes go into file names: keep them to safe characters
export const safeRoom = (room) => String(room ?? "").replace(/[^A-Za-z0-9_-]/g, "_").slice(0, 40) || "_";
export async function sigHash(model, sig) {
  const h = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(JSON.stringify({ model, sig }))));
  return [...h.slice(0, 8)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
// <room>.<sig hash>.<slot>.ckpt -> { room, hash, slot } or null
export function parseName(name) {
  const m = /^([A-Za-z0-9_-]+)\.([0-9a-f]{16})\.(\d+)\.ckpt$/.exec(name);
  return m ? { room: m[1], hash: m[2], slot: +m[3] } : null;
}
const isQuota = (e) => e?.name === "QuotaExceededError" || /quota/i.test(e?.message || "");

export class CkptStore {
  // root: () => the OPFS root directory handle (tests pass an in-memory one)
  // tmpStaleMs: a <name>.ckpt.tmp this old is a write that never finished (the tab reloaded); now: the clock
  constructor({ dirName = "pooled-ckpt", budgetBytes = 4 * 2 ** 30, root = () => navigator.storage.getDirectory(), tmpStaleMs = 10 * 60e3, now = () => Date.now() } = {}) {
    this.dirName = dirName; this.budget = budgetBytes; this.root = root; this.dir = null;
    this.tmpStaleMs = tmpStaleMs; this.now = now;
    this.q = Promise.resolve();
    this.failures = 0;   // writes given up on (quota, or the state was gone before it was read)
  }
  async _d() {
    if (!this.dir) this.dir = await (await this.root()).getDirectoryHandle(this.dirName, { create: true });
    return this.dir;
  }
  // one operation at a time, in call order: a drop never overtakes the write it follows
  _run(fn) { const p = this.q.then(fn); this.q = p.catch(() => {}); return p; }

  // Save one slot. state: { sig, pos, parts } or an async function returning one (read when the
  // write's turn comes, so a slot dropped meanwhile is simply skipped). -> true if written.
  put(where, state, meta = {}) { return this._run(() => this._put(where, state, meta)); }
  async _put({ room, model, sig, slot }, state, meta) {
    const d = await this._d();
    const name = `${safeRoom(room)}.${await sigHash(model, sig)}.${slot}${EXT}`;
    await this._dropSlot(d, room, slot);   // the old copy goes first: a failed write leaves nothing stale
    let st;
    try { st = typeof state === "function" ? await state() : state; } catch { this.failures++; return false; }
    const header = encodeHeader({ room, model, sig, slot, pos: st.pos, sizes: st.parts.map((p) => p.byteLength), meta });
    const bytes = header.length + st.parts.reduce((s, p) => s + p.byteLength, 0);
    if (bytes > this.budget) { this.failures++; return false; }
    await this._evict(d, this.budget - bytes, { keep: room });
    for (let attempt = 0; ; attempt++) {
      try { await this._write(d, name, header, st.parts); break; }
      catch (e) {
        await d.removeEntry(name + ".tmp").catch(() => {});
        // out of quota: make room by dropping the oldest other copies (other rooms first), try once more
        if (attempt === 0 && isQuota(e) && await this._evict(d, 0, { need: bytes, keep: room })) continue;
        this.failures++;
        return false;
      }
    }
    return true;
  }
  async _write(d, name, header, parts) {
    const tmp = await d.getFileHandle(name + ".tmp", { create: true });
    const w = await tmp.createWritable();
    try {
      await w.write(header);
      for (const p of parts) await w.write(p);
      await w.close();
    } catch (e) { await w.abort?.().catch(() => {}); throw e; }
    if (tmp.move) await tmp.move(name);   // atomic rename where supported
    else {
      const f = await (await tmp.getFile()).arrayBuffer();
      const fin = await (await d.getFileHandle(name, { create: true })).createWritable();
      await fin.write(f); await fin.close();
      await d.removeEntry(name + ".tmp");
    }
  }

  // The copies this device can load for { room, model, sig }, newest first: [{ slot, pos, meta, t }].
  // Reads headers only. Files in another format, or whose header does not match, are removed.
  list(where) { return this._run(() => this._list(where)); }
  async _list({ room, model, sig }) {
    const d = await this._d(), hash = await sigHash(model, sig), out = [];
    for (const { name, file, p, tmp } of await this._files(d)) {
      if (tmp || p.room !== safeRoom(room) || p.hash !== hash) continue;
      let h;
      try {
        const n = headerLength(await readSlice(file, 0, 8));
        h = decodeHeader(await readSlice(file, 0, n));
      } catch { await d.removeEntry(name).catch(() => {}); continue; }
      if (mismatch(h, { room, model, sig }) || h.slot !== p.slot) { await d.removeEntry(name).catch(() => {}); continue; }
      out.push({ slot: h.slot, pos: h.pos, meta: h.meta || {}, t: h.t || 0 });
    }
    return out.sort((a, b) => b.t - a.t || b.slot - a.slot);   // (same millisecond: the higher slot is newer)
  }
  // -> { sig, pos, parts, meta } or null (missing, damaged, another format or another room/model/layers)
  get(where) { return this._run(() => this._get(where)); }
  async _get({ room, model, sig, slot }) {
    const d = await this._d();
    const name = `${safeRoom(room)}.${await sigHash(model, sig)}.${slot}${EXT}`;
    let buf;
    try { buf = await (await (await d.getFileHandle(name)).getFile()).arrayBuffer(); } catch { return null; }
    try {
      const { header: h, parts } = decodeCkpt(buf);
      if (mismatch(h, { room, model, sig }) || h.slot !== slot) throw new CkptFormatError("mismatch");
      return { sig: h.sig, pos: h.pos, parts, meta: h.meta || {} };
    } catch { await d.removeEntry(name).catch(() => {}); return null; }
  }
  // drop slots of a room (any layers); slots: a number, a list, or "all"
  drop(room, slots) { return this._run(async () => {
    const d = await this._d();
    if (slots === "all") { for (const f of await this._files(d)) if (!f.tmp && f.p.room === safeRoom(room)) await d.removeEntry(f.name).catch(() => {}); return; }
    for (const s of [].concat(slots)) await this._dropSlot(d, room, s);
  }); }
  async _dropSlot(d, room, slot) {
    for (const f of await this._files(d)) if (!f.tmp && f.p.room === safeRoom(room) && f.p.slot === slot) await d.removeEntry(f.name).catch(() => {});
  }
  // every copy in the directory, with the temp files of writes in progress (tmp: true; the budget
  // counts them and _evict may drop them). A temp file older than tmpStaleMs is a write that never
  // finished (the tab reloaded or closed mid-write): removed here.
  async _files(d) {
    const out = [];
    for await (const [name, h] of d.entries()) {
      const tmp = name.endsWith(EXT + ".tmp");
      const p = parseName(tmp ? name.slice(0, -4) : name);
      if (!p || h.kind === "directory") continue;
      let file;
      try { file = await h.getFile(); } catch { continue; }
      if (tmp && this.now() - file.lastModified > this.tmpStaleMs) { await d.removeEntry(name).catch(() => {}); continue; }
      out.push({ name, file, p, tmp, bytes: file.size, t: file.lastModified });
    }
    return out;
  }
  // Drop the oldest copies until at most `limit` bytes are kept (or, with need, until `need` bytes
  // were freed), other rooms before `keep`. -> true if anything was dropped.
  async _evict(d, limit, { need = 0, keep = null } = {}) {
    const fs = await this._files(d);
    let total = fs.reduce((s, f) => s + f.bytes, 0), freed = 0, any = false;
    fs.sort((a, b) => ((a.p.room === safeRoom(keep)) - (b.p.room === safeRoom(keep))) || a.t - b.t);
    for (const f of fs) {
      if (need ? freed >= need : total <= limit) break;
      await d.removeEntry(f.name).catch(() => {});
      total -= f.bytes; freed += f.bytes; any = true;
    }
    return any;
  }
}

async function readSlice(file, a, b) {
  if (file.slice) return file.slice(a, b).arrayBuffer();
  return (await file.arrayBuffer()).slice(a, b);
}
