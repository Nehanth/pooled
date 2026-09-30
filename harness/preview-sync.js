// Peers see the host's preview (docs/design/harness-app.md B.8). Transport-agnostic: the room
// wires send/broadcast (room/code.js), tests wire an in-memory pair.
//
//   host:  new PreviewPublisher(server, { send(id, msg), broadcast(msg), channel?(id) })
//          follows server.onUpdate by itself; onWant(from, d) answers ai-pv-want; helloTo(id) for late joiners
//   peer:  new PreviewSubscriber({ send(msg) (to the host), hostId }) -> a PreviewSource for mountPreview
//          onManifest / onBlob / onStop take ai-pv / ai-pv-blob / ai-pv-stop
//
// Files are content addressed (sha-256 prefix, harness/preview.js hashBytes): after a live reload
// a peer asks only for the blobs it does not hold. A peer trusts nothing it receives: limits,
// paths and hashes are checked again here, and mime types come from the file name, not the host.
import { hashBytes } from "./preview.js";
import { mimeFor } from "./preview-build.js";

// A blob goes out in chunks of at most CHUNK bytes (the subscriber refuses bigger ones) and at least
// MIN_CHUNK (so a file's chunk count stays bounded by its manifest size).
const CHUNK = 64 << 10, MIN_CHUNK = 4 << 10, HIGH_WATER = 1 << 20;
export const chunkCount = (size, chunk = MIN_CHUNK) => Math.max(1, Math.ceil(size / chunk));
const HASH = /^[0-9a-f]{20}$/;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// a relative project path: no "..", no leading "/", no empty or dot segments
const okPath = (p) => typeof p === "string" && p.length > 0 && p.length <= 300 && !p.startsWith("/") && p.split("/").every((s) => s && s !== "." && s !== "..");

export class PreviewPublisher {
  constructor(source, { send, broadcast, channel = () => null, chunk = CHUNK } = {}) {
    this.source = source; this.send = send; this.broadcast = broadcast; this.channel = channel;
    this.chunk = Math.min(CHUNK, Math.max(MIN_CHUNK, Math.floor(chunk) || CHUNK));
    this.queues = new Map();   // peer id -> promise chain, so one peer's blobs go out one at a time
    this.off = source.onUpdate((u) => (u.stopped ? this.stopped(u.port) : this.announce(u.port)));
  }
  manifest(port) {
    const s = this.source.snapshot(port);
    if (!s) return null;
    return { t: "ai-pv", port: s.port, dir: s.dir, entry: s.entry, rev: s.rev, bytes: s.bytes,
      manifest: [...s.files].map(([p, f]) => [p, f.type, f.hash, f.bytes.length]) };
  }
  announce(port) { const m = this.manifest(port); if (m) this.broadcast(m); }
  stopped(port) { this.broadcast({ t: "ai-pv-stop", port }); }
  helloTo(id) { for (const { port } of this.source.ports()) { const m = this.manifest(port); if (m) this.send(id, m); } }
  // only hashes of that port's current files; an older rev's want is answered with what still matches
  onWant(from, d) {
    const s = this.source.snapshot(d?.port);
    if (!s || !Array.isArray(d.hs)) return;
    const byHash = new Map();
    for (const f of s.files.values()) byHash.set(f.hash, f.bytes);
    const hs = [...new Set(d.hs)].filter((h) => byHash.has(h)).slice(0, s.files.size);
    const q = (this.queues.get(from) || Promise.resolve()).then(async () => {
      for (const h of hs) {
        const u8 = byHash.get(h), n = chunkCount(u8.length, this.chunk);
        for (let i = 0; i < n; i++) {
          // back-pressure: a big snapshot must not sit in front of chat and agent messages
          for (let k = 0; (this.channel(from)?.bufferedAmount || 0) > HIGH_WATER && k < 500; k++) await sleep(20);
          this.send(from, { t: "ai-pv-blob", h, i, n, b: u8.slice(i * this.chunk, (i + 1) * this.chunk).buffer });
        }
      }
    }).catch((e) => console.error("preview blobs", e));
    this.queues.set(from, q);
  }
  close() { this.off?.(); }
}

export class PreviewSubscriber {
  // maxPorts / maxHeld bound what a host can make a peer hold in total (every port's snapshot and
  // pending rev, blobs and half-received parts): the per-port limits alone allow ~64k ports.
  constructor({ send, hostId = null, maxBytes = 8 << 20, maxFile = 2 << 20, maxFiles = 400, maxPorts = 4, maxHeld = 2 * maxBytes } = {}) {
    this.sendHost = send; this.hostId = typeof hostId === "function" ? hostId : () => hostId;
    this.limits = { maxBytes, maxFile, maxFiles, maxPorts, maxHeld };
    this.retries = new Map();  // hash -> times re-asked after a bad copy
    this.snaps = new Map();    // port -> Snapshot (last complete rev)
    this.wants = new Map();    // port -> the manifest being completed
    this.blobs = new Map();    // hash -> Uint8Array
    this.parts = new Map();    // hash -> { n, got, recv, chunks: [], size }
    this.fns = new Set();
  }
  _from(from) { const h = this.hostId(); return !h || from === h; }
  snapshot(port) { return this.snaps.get(+port) || null; }
  ports() { return [...this.snaps.values()].map((s) => ({ port: s.port, dir: s.dir, entry: s.entry, rev: s.rev })); }
  onUpdate(fn) { this.fns.add(fn); return () => this.fns.delete(fn); }
  _emit(e) { for (const f of [...this.fns]) { try { f(e); } catch (err) { console.error(err); } } }

  onManifest(from, d) {
    if (!this._from(from) || !d) return;
    const port = Number(d.port), L = this.limits;
    if (!Number.isInteger(port) || port < 1024 || port > 65535 || !Array.isArray(d.manifest) || d.manifest.length > L.maxFiles) return;
    if (!Number.isInteger(d.rev) || d.rev < 1) return;
    if (!this.snaps.has(port) && !this.wants.has(port) && new Set([...this.snaps.keys(), ...this.wants.keys()]).size >= L.maxPorts) return;
    let total = 0;
    const files = [];
    for (const row of d.manifest) {
      if (!Array.isArray(row)) return;
      const [p, , h, size] = row;
      if (!okPath(p) || !HASH.test(h) || !Number.isInteger(size) || size < 0 || size > L.maxFile) return;
      total += size;
      files.push({ p, h, size });
    }
    if (total > L.maxBytes || !okPath(d.entry || "index.html")) return;
    if ((this.snaps.get(port)?.rev || 0) >= d.rev) return;   // stale or repeated
    const want = { port, dir: typeof d.dir === "string" ? d.dir.slice(0, 300) : "", entry: d.entry || "index.html", rev: d.rev, files };
    this.wants.set(port, want);
    this._gc();   // parts of a rev this one replaces are not coming any more
    const missing = [...new Set(files.filter((f) => !this.blobs.has(f.h)).map((f) => f.h))];
    for (const h of missing) if (!this.parts.has(h)) this.parts.set(h, { n: 0, got: 0, recv: 0, chunks: [], size: files.find((f) => f.h === h).size });
    if (missing.length) this.sendHost({ t: "ai-pv-want", port, rev: want.rev, hs: missing });
    else this._complete(port);
  }
  async onBlob(from, d) {
    if (!this._from(from) || !d || !this.parts.has(d.h)) return;
    const P = this.parts.get(d.h), n = d.n >>> 0, i = d.i >>> 0;
    const b = d.b instanceof ArrayBuffer ? new Uint8Array(d.b) : ArrayBuffer.isView(d.b) ? new Uint8Array(d.b.buffer, d.b.byteOffset, d.b.byteLength) : null;
    if (!b || !n || i >= n || n > chunkCount(P.size) || (P.n && P.n !== n) || P.chunks[i]) return;
    // sizes are checked per chunk, before anything is kept: a chunk over CHUNK, a file growing past
    // its manifest size, or everything held going past maxHeld is refused
    if (b.length > CHUNK || P.recv + b.length > P.size) { this._bad(d.h, "a file came bigger than its manifest said"); return; }
    if (this._held() + b.length > this.limits.maxHeld) { this.parts.delete(d.h); console.warn("preview: over the memory limit; a file was dropped"); return; }
    P.n = n; P.chunks[i] = b.slice(); P.got++; P.recv += b.length;
    if (P.got < n) return;
    this.parts.delete(d.h);
    if (P.recv !== P.size) { this._bad(d.h, "a file came smaller than its manifest said"); return; }
    const u8 = new Uint8Array(P.recv);
    let o = 0;
    for (const c of P.chunks) { u8.set(c, o); o += c.length; }
    if ((await hashBytes(u8)) !== d.h) { this._bad(d.h, "a file did not match its hash"); return; }
    this.retries.delete(d.h);
    this.blobs.set(d.h, u8);
    for (const port of [...this.wants.keys()]) this._complete(port);
  }
  onStop(from, d) {
    if (!this._from(from)) return;
    const port = Number(d?.port), s = this.snaps.get(port);
    this.wants.delete(port);
    if (!s) return;
    this.snaps.delete(port);
    this._gc();
    this._emit({ port, rev: s.rev, changed: [], stopped: true });
  }
  _complete(port) {
    const w = this.wants.get(port);
    if (!w || !w.files.every((f) => this.blobs.has(f.h))) return;
    this.wants.delete(port);
    const old = this.snaps.get(port), files = new Map();
    let bytes = 0;
    for (const f of w.files) { const u8 = this.blobs.get(f.h); files.set(f.p, { type: mimeFor(f.p), bytes: u8, hash: f.h }); bytes += u8.length; }
    const changed = [...files.keys()].filter((p) => old?.files.get(p)?.hash !== files.get(p).hash);
    if (old) for (const p of old.files.keys()) if (!files.has(p)) changed.push(p);
    this.snaps.set(port, Object.freeze({ port, dir: w.dir, entry: w.entry, rev: w.rev, bytes, files }));
    this._gc();
    this._emit({ port, rev: w.rev, changed: changed.sort() });
  }
  _held() {
    let n = 0;
    for (const u8 of this.blobs.values()) n += u8.length;
    for (const P of this.parts.values()) n += P.recv;
    return n;
  }
  // a bad copy (lost or corrupted on the way): ask for it again, twice at most, else the pending
  // rev would never complete and the peer would stay on the old one
  _bad(h, why) {
    const P = this.parts.get(h);
    this.parts.delete(h);
    const k = (this.retries.get(h) || 0) + 1;
    const w = [...this.wants.values()].find((x) => x.files.some((f) => f.h === h));
    if (k > 2 || !w) { console.warn(`preview: ${why}; dropped`); return; }
    this.retries.set(h, k);
    this.parts.set(h, { n: 0, got: 0, recv: 0, chunks: [], size: P?.size ?? w.files.find((f) => f.h === h).size });
    this.sendHost({ t: "ai-pv-want", port: w.port, rev: w.rev, hs: [h] });
  }
  // keep only blobs and parts some snapshot or pending manifest still uses
  _gc() {
    const live = new Set(), wanted = new Set();
    for (const s of this.snaps.values()) for (const f of s.files.values()) live.add(f.hash);
    for (const w of this.wants.values()) for (const f of w.files) { live.add(f.h); wanted.add(f.h); }
    for (const h of this.blobs.keys()) if (!live.has(h)) this.blobs.delete(h);
    for (const h of this.parts.keys()) if (!wanted.has(h)) this.parts.delete(h);
    for (const h of this.retries.keys()) if (!wanted.has(h)) this.retries.delete(h);
  }
}
