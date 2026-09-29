// Prefix / session cache on disk (the browser's origin-private file system, OPFS): engine states
// from Qwen35Engine.exportState(), keyed by what they are the state of, so a coding session that
// comes back (same system prompt + tools, same conversation so far) skips the re-prefill.
//
// A state is only valid for exactly the tokens it was computed from, on exactly this device's
// layers, so the key is a hash of: the model, this device's layer range and KV format (the
// engine's stateSignature()) and the token ids. The plan for rooms (not wired yet; only the e2e
// tests use this today) is that every device stores its own part under the same token hash, and
// the host asks everyone to load it and falls back to a prefill if anyone is missing theirs.
//
// Files: <dir>/<key>.bin = [u32 header length][header JSON][part 0][part 1]... Written to a temp
// name and renamed, so a crash never leaves a half file under a real key. Least recently used
// entries are evicted past `budgetBytes`.

export async function tokenKey(sig, ids, model = "") {
  const head = new TextEncoder().encode(JSON.stringify({ model, sig }));
  const body = new Uint8Array(new Uint32Array(ids).buffer);
  const all = new Uint8Array(head.length + body.length);
  all.set(head); all.set(body, head.length);
  const h = new Uint8Array(await crypto.subtle.digest("SHA-256", all));
  return [...h.slice(0, 16)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export class StateCache {
  constructor({ dirName = "tabby-states", budgetBytes = 8 * 2 ** 30 } = {}) {
    this.dirName = dirName; this.budget = budgetBytes; this.dir = null;
  }
  async _d() {
    if (!this.dir) this.dir = await (await navigator.storage.getDirectory()).getDirectoryHandle(this.dirName, { create: true });
    return this.dir;
  }
  async has(key) {
    try { await (await this._d()).getFileHandle(key + ".bin"); return true; } catch { return false; }
  }
  // state: { sig, pos, parts: [ArrayBuffer] }; meta: anything small to keep alongside (e.g. ids length)
  async put(key, state, meta = {}) {
    const d = await this._d();
    const header = new TextEncoder().encode(JSON.stringify({ sig: state.sig, pos: state.pos, sizes: state.parts.map((p) => p.byteLength), meta, t: Date.now() }));
    const tmp = await d.getFileHandle(key + ".tmp", { create: true });
    const w = await tmp.createWritable();
    await w.write(new Uint32Array([header.length]));
    await w.write(header);
    for (const p of state.parts) await w.write(p);
    await w.close();
    if (tmp.move) await tmp.move(key + ".bin");   // atomic rename where supported
    else {
      const f = await (await tmp.getFile()).arrayBuffer();
      const fin = await (await d.getFileHandle(key + ".bin", { create: true })).createWritable();
      await fin.write(f); await fin.close();
      await d.removeEntry(key + ".tmp");
    }
    await this.evict();
  }
  // -> { sig, pos, parts, meta } or null
  async get(key) {
    let file;
    try { file = await (await (await this._d()).getFileHandle(key + ".bin")).getFile(); } catch { return null; }
    const buf = await file.arrayBuffer();
    const hl = new Uint32Array(buf, 0, 1)[0];
    const h = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
    let off = 4 + hl;
    const parts = h.sizes.map((n) => { const p = buf.slice(off, off + n); off += n; return p; });
    this._touch(key).catch(() => {});
    return { sig: h.sig, pos: h.pos, parts, meta: h.meta };
  }
  // -> whether a file was removed
  async remove(key) {
    const d = await this._d();
    return d.removeEntry(key + ".bin").then(() => true, () => false);
  }
  async _touch(key) {   // LRU order: rewrite nothing, just remember the use
    const m = JSON.parse(localStorageGet(this.dirName) || "{}"); m[key] = Date.now(); localStorageSet(this.dirName, JSON.stringify(m));
  }
  async entries() {
    const d = await this._d(), out = [];
    const used = JSON.parse(localStorageGet(this.dirName) || "{}");
    for await (const [name, h] of d.entries()) {
      if (!name.endsWith(".bin")) continue;
      const f = await h.getFile();
      const key = name.slice(0, -4);
      out.push({ key, bytes: f.size, t: used[key] || f.lastModified });
    }
    return out;
  }
  async evict() {
    const es = (await this.entries()).sort((a, b) => b.t - a.t);
    let total = 0;
    const d = await this._d();
    for (const e of es) { total += e.bytes; if (total > this.budget) await d.removeEntry(e.key + ".bin").catch(() => {}); }
  }
  async clear() { const d = await this._d(); for (const e of await this.entries()) await d.removeEntry(e.key + ".bin").catch(() => {}); }
}

function localStorageGet(k) { try { return globalThis.localStorage?.getItem("statecache:" + k); } catch { return null; } }
function localStorageSet(k, v) { try { globalThis.localStorage?.setItem("statecache:" + k, v); } catch { /* private mode: no LRU memory */ } }
