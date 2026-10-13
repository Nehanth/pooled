// ---- a small in-memory OPFS: directories, files, writables, move ----
const notFound = (n) => Object.assign(new Error(`${n} not found`), { name: "NotFoundError" });
export class FakeFile {
  constructor(parent, name) { this.kind = "file"; this.parent = parent; this.name = name; this.data = new Uint8Array(0); }
  getFile() { const d = this.data; return Promise.resolve({ size: d.byteLength, arrayBuffer: async () => d.slice().buffer, text: async () => new TextDecoder().decode(d) }); }
  createWritable() {
    const chunks = [], root = this.parent.root;
    return Promise.resolve({
      // bytes (appended), or { type: "write", position, data } (placed at position)
      write: async (p) => {
        if (root.failWrites) throw Object.assign(new Error("quota"), { name: "QuotaExceededError" });
        const at = p?.type === "write" ? p.position : null, d = p?.type === "write" ? p.data : p;
        chunks.push({ at, b: new Uint8Array(d.buffer ? d.buffer.slice(d.byteOffset, d.byteOffset + d.byteLength) : d) });
      },
      close: async () => {   // like Chrome: the file changes only on close
        if (root.beforeClose) await root.beforeClose(this.name);
        let end = 0, o = 0;
        for (const c of chunks) { const a = c.at ?? o; end = Math.max(end, a + c.b.byteLength); o = a + c.b.byteLength; }
        const out = new Uint8Array(end);
        o = 0;
        for (const c of chunks) { const a = c.at ?? o; out.set(c.b, a); o = a + c.b.byteLength; }
        this.data = out;
      },
      abort: async () => {},
    });
  }
  async move(name) { this.parent.children.delete(this.name); this.name = name; this.parent.children.set(name, this); }
}
export class FakeDir {
  constructor(name = "", root = null) { this.kind = "directory"; this.name = name; this.children = new Map(); this.root = root || this; }
  async getDirectoryHandle(n, { create = false } = {}) {
    let h = this.children.get(n);
    if (!h) { if (!create) throw notFound(n); h = new FakeDir(n, this.root); this.children.set(n, h); }
    return h;
  }
  async getFileHandle(n, { create = false } = {}) {
    let h = this.children.get(n);
    if (!h) { if (!create) throw notFound(n); h = new FakeFile(this, n); this.children.set(n, h); }
    return h;
  }
  async removeEntry(n) { if (!this.children.delete(n)) throw notFound(n); }
  async *entries() { for (const e of [...this.children.entries()]) yield e; }
  async *keys() { for (const k of [...this.children.keys()]) yield k; }
}
