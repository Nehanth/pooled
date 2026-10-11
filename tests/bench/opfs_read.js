// Browser disk read probe for the SSD tier in a tab (roadmap/32 step 8; Qwen3.8-Flash-Next's experts and
// n-gram table). Answers: how fast does a tab read random expert-sized and n-gram-row-sized pieces of a file
// in the origin-private file system (OPFS), and what does getting them onto the GPU add? No model needed.
//   Chrome: node tests/bench/opfs_read.mjs  (tests/bench/opfs_read.html; the driver drops the page cache
//           between runs so reads are cold, see there)
//
// Setup writes GB GiB of test data to OPFS as FILES files (one readwrite sync access handle per file at a
// time, so each worker reads files of its own; run({ shared: true }) tries read-only handles instead, which
// several workers can hold on one file), plus CACHE_GB GiB of expert-sized entries in the Cache API, keyed
// the way room.js keys weight ranges (room/weightcache.js).
//
// Read sizes (roadmap/33-qwen38-flash.md, the Qwen3.8-Flash-Next design note, "Memory and bandwidth"):
//   expert  2,764,800 B   one routed expert, gate + up + down, 3 x 2560 x 640 at Q4_0 (one read, as from a
//                         per-expert file or a repacked layer file)
//   part      921,600 B   one of those three: in the GGUF each lives in its own stacked tensor, so a miss
//                         straight from the GGUF is three reads of this size
//   row            90 B   one n-gram row of per_layer_token_embd (160 values at Q4_0); a token needs 16
// Reads are at random offsets, without repeats within a run (each run reads parts of the file it has not
// read since the last page-cache drop). Each worker reads one piece at a time (queue depth 1 per worker,
// the shape of the engine's per-layer misses); 1-8 workers run in parallel (1-16 for rows).
//
// Modes (a dedicated worker per reader, FileSystemSyncAccessHandle.read):
//   read    the read alone, into a reused buffer
//   write   read, then queue.writeBuffer into a GPU buffer (the worker's own GPUDevice); at most RING pieces'
//           work in flight (it waits on the onSubmittedWorkDone of the piece RING back, as map waits on its slot)
//   map     read straight into a mapped MAP_WRITE staging buffer (a ring of 4 per worker), unmap,
//           copyBufferToBuffer to a STORAGE buffer, map the slot again for its next use (worker's device)
//   xfer    read into a fresh ArrayBuffer, transfer it to the page, which writeBuffers it on the page's
//           device (an engine that lives on the main thread; at most 4 pieces in flight per worker)
//   cache   Cache API: cache.match(key) + arrayBuffer() of a whole expert-sized entry, the way rangeFetch
//           reads cached weights; C requests in flight from the page's main thread (not from workers)
// Per run: GB/s (bytes / wall time from the first read's start to the last one's end, all workers), reads/s,
// and p10 / p50 / p99 of one piece's latency (read start to the piece being in the GPU queue, for the GPU
// modes). For pieces under 4 KiB also `hits`: reads that took under 20 us, i.e. came from the page cache
// (another read's readahead window) rather than the disk.
// A run's GPU modes await onSubmittedWorkDone before the clock stops.

export const SIZES = { expert: 2764800, part: 921600, row: 90 };
const DIR = "pooled-opfs-probe", RING = 4, CHUNK = 64 << 20;
const IS_WORKER = typeof WorkerGlobalScope !== "undefined" && self instanceof WorkerGlobalScope;
const now = () => performance.timeOrigin + performance.now();   // comparable across the page and workers

// 64 MiB of incompressible bytes (xorshift), shared by every chunk of every file; each chunk gets its index
// stamped in the first word so no two chunks are identical
function pattern(bytes) {
  const u = new Uint32Array(bytes >> 2); let x = 2463534242;
  for (let i = 0; i < u.length; i++) { x ^= x << 13; x ^= x >>> 17; x ^= x << 5; u[i] = x >>> 0; }
  return new Uint8Array(u.buffer);
}

// ================= worker side =================
if (IS_WORKER) {
  let handles = new Map(), dev = null, dst = null, ring = [], acks = null;
  const dir = async () => (await navigator.storage.getDirectory()).getDirectoryHandle(DIR, { create: true });
  const reply = (m, t) => self.postMessage(m, t || []);
  self.onmessage = async ({ data: m }) => {
    try {
      if (m.op === "write") {   // write one test file, m.bytes long
        const fh = await (await dir()).getFileHandle(m.name, { create: true }), h = await fh.createSyncAccessHandle();
        const buf = pattern(CHUNK), u32 = new Uint32Array(buf.buffer, 0, 1);
        h.truncate(0);
        for (let o = 0, k = m.seed; o < m.bytes; o += CHUNK, k++) { u32[0] = k; h.write(buf.subarray(0, Math.min(CHUNK, m.bytes - o)), { at: o }); }
        h.flush(); h.close(); reply({ ok: true });
      } else if (m.op === "open") {   // handles for m.files; m.shared: read-only mode, so other workers can open the same files
        for (const h of handles.values()) h.close(); handles = new Map();
        const d = await dir();
        for (const name of m.files) {
          const fh = await d.getFileHandle(name);
          handles.set(name, m.shared ? await fh.createSyncAccessHandle({ mode: "read-only" }) : await fh.createSyncAccessHandle());
        }
        if (m.gpu && !dev) {
          const ad = await navigator.gpu.requestAdapter(); dev = await ad.requestDevice();
        }
        if (m.gpu) {
          dst?.destroy(); for (const r of ring) r.buf.destroy(); ring = [];
          const S = (m.size + 3) & ~3;
          dst = dev.createBuffer({ size: S * RING, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
          if (m.gpu === "map") for (let i = 0; i < RING; i++) {
            const buf = dev.createBuffer({ size: S, usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
            ring.push({ buf, ready: Promise.resolve() });
          }
        }
        reply({ ok: true });
      } else if (m.op === "go") reply(await run(m));
      else if (m.op === "ack") acks.get(m.i)?.();
      else if (m.op === "close") { for (const h of handles.values()) h.close(); handles = new Map(); dst?.destroy(); for (const r of ring) r.buf.destroy(); ring = []; reply({ ok: true }); }
    } catch (e) { reply({ error: `${e.name}: ${e.message}` }); }
  };

  // m.jobs: [fileIndex, offset, ...] into m.files; m.size bytes each; m.mode as above
  async function run({ files, jobs, size, mode, at }) {
    while (now() < at) await new Promise((r) => setTimeout(r, 0));   // start together with the other workers
    const n = jobs.length / 2, lat = new Float64Array(n), S = (size + 3) & ~3;
    const scratch = new Uint8Array(S);
    const inFlight = [];   // xfer: ack promises of the pieces the page has not written yet
    const done = new Array(RING).fill(null);   // write: onSubmittedWorkDone of the piece RING back, per slot
    acks = new Map();
    const t0 = now();
    for (let i = 0; i < n; i++) {
      const h = handles.get(files[jobs[2 * i]]), off = jobs[2 * i + 1], s = now();
      if (mode === "map") {
        const slot = ring[i % RING];
        await slot.ready;   // the previous copy out of this slot is done and the slot is mapped again
        const view = new Uint8Array(slot.buf.getMappedRange(), 0, size);
        if (h.read(view, { at: off }) !== size) throw new Error("short read");
        slot.buf.unmap();
        const e = dev.createCommandEncoder(); e.copyBufferToBuffer(slot.buf, 0, dst, (i % RING) * S, S); dev.queue.submit([e.finish()]);
        slot.ready = slot.buf.mapAsync(GPUMapMode.WRITE);
        lat[i] = now() - s;
      } else if (mode === "xfer") {
        if (inFlight.length >= RING) await inFlight.shift();
        const buf = new Uint8Array(S);
        if (h.read(buf.subarray(0, size), { at: off }) !== size) throw new Error("short read");
        inFlight.push(new Promise((r) => acks.set(i, () => { lat[i] = now() - s; r(); })));
        self.postMessage({ op: "piece", i, slot: i % RING, buf: buf.buffer }, [buf.buffer]);
      } else {
        if (h.read(scratch.subarray(0, size), { at: off }) !== size) throw new Error("short read");
        if (mode === "write") {   // writeBuffer copies scratch before it returns; only bound the work in flight
          await done[i % RING];
          dev.queue.writeBuffer(dst, (i % RING) * S, scratch, 0, S);
          done[i % RING] = dev.queue.onSubmittedWorkDone();
        }
        lat[i] = now() - s;
      }
    }
    await Promise.all(inFlight);
    if (dev && (mode === "write" || mode === "map")) { await dev.queue.onSubmittedWorkDone(); await Promise.all(ring.map((r) => r.ready)); }
    return { lat, t0, t1: now() };
  }
}

// ================= page side =================
// One request / reply at a time per worker (the "piece" messages of xfer go to onPiece)
function ask(w, msg, onPiece) {
  return new Promise((res, rej) => {
    w.onmessage = ({ data }) => {
      if (data.op === "piece") { onPiece(data); return; }
      if (data.error) rej(new Error(data.error)); else res(data);
    };
    w.postMessage(msg);
  });
}
const pct = (sorted, p) => sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))] : NaN;
function shuffle(a, rnd) { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }
function rng(seed) { let x = seed >>> 0 || 1; return () => { x ^= x << 13; x ^= x >>> 17; x ^= x << 5; return (x >>> 0) / 2 ** 32; }; }
export const HIT_MS = 0.02;   // a read under 20 us came from the page cache (a cold 90 B read off the NVMe is ~0.2 ms)
export function summary(lat, bytes, t0, t1, S) {
  const s = Float64Array.from(lat).sort(), sec = (t1 - t0) / 1e3;
  return { reads: s.length, MB: +(bytes / 1e6).toFixed(1), ms: +(t1 - t0).toFixed(1), GBs: +(bytes / sec / 1e9).toFixed(2),
    readsPerS: Math.round(s.length / sec), p10: +pct(s, 0.1).toFixed(3), p50: +pct(s, 0.5).toFixed(3), p99: +pct(s, 0.99).toFixed(3),
    ...(S < 4096 ? { hits: s.filter((x) => x < HIT_MS).length } : {}) };
}

export class OpfsProbe {
  // device: the page's GPUDevice (xfer mode); log: a line printer; seed: of the random offsets (each run
  // bumps it; the driver's SEED, so reruns on a kept profile read other offsets)
  constructor({ device = null, log = console.log, seed = 1 } = {}) { this.device = device; this.log = log; this.workers = []; this.files = []; this.fileBytes = 0; this.shared = null; this.seed = seed; }

  worker() { return new Worker(new URL(import.meta.url), { type: "module" }); }
  async pool(n) { while (this.workers.length < n) this.workers.push(this.worker()); return this.workers.slice(0, n); }

  // navigator.storage.persist() and estimate(), before and after setup
  async storage() {
    const persisted = await navigator.storage.persisted?.().catch(() => null);
    const persist = await navigator.storage.persist?.().catch((e) => "error " + e.message);
    const est = await navigator.storage.estimate();
    return { persistedBefore: persisted, persist, usageGB: +(est.usage / 1e9).toFixed(2), quotaGB: +(est.quota / 1e9).toFixed(1), usageDetails: est.usageDetails ?? null };
  }

  // GB GiB as FILES equal files, written by FILES workers at once; files of the right size from an earlier run are kept
  async setup({ GB = 4, FILES = 8 } = {}) {
    const per = Math.floor(GB * 2 ** 30 / FILES / 4096) * 4096;
    this.files = Array.from({ length: FILES }, (_, i) => `data-${i}.bin`); this.fileBytes = per;
    const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle(DIR, { create: true });
    const have = await Promise.all(this.files.map(async (n) => { try { return (await (await dir.getFileHandle(n)).getFile()).size; } catch { return -1; } }));
    const todo = this.files.filter((_, i) => have[i] !== per);
    const t = now();
    if (todo.length) {
      const ws = await this.pool(Math.min(FILES, 8));
      for (let k = 0; k < todo.length; k += ws.length)
        await Promise.all(todo.slice(k, k + ws.length).map((name, j) => ask(ws[j], { op: "write", name, bytes: per, seed: this.files.indexOf(name) * 1e6 })));
    }
    const sec = (now() - t) / 1e3;
    this.log(`setup: ${FILES} files x ${(per / 2 ** 30).toFixed(3)} GiB in OPFS/${DIR}, ${todo.length ? `wrote ${todo.length} in ${sec.toFixed(1)} s (${(todo.length * per / sec / 1e9).toFixed(2)} GB/s)` : "kept from an earlier run"}`);
    return { files: FILES, fileBytes: per, wrote: todo.length, writeGBs: todo.length ? +(todo.length * per / sec / 1e9).toFixed(2) : null };
  }

  // Can several workers hold sync handles on one file (mode "read-only", Chrome 121+)? Otherwise each
  // worker gets its own files and reads only from them.
  async probeShared() {
    if (this.shared !== null) return this.shared;
    const [a, b] = await this.pool(2);
    try {
      await ask(a, { op: "open", files: [this.files[0]], shared: true });
      await ask(b, { op: "open", files: [this.files[0]], shared: true });
      this.shared = true;
    } catch (e) { this.shared = false; this.sharedError = e.message; }
    await ask(a, { op: "close" }); await ask(b, { op: "close" });
    return this.shared;
  }

  // One run: W workers, `reads` pieces of `size` bytes (or fewer: the file set's distinct slots), mode as above
  // shared: every worker opens every file with a read-only handle (where Chrome has them); by default each
  // worker has files of its own with ordinary handles. In Chromium 131 on Linux read-only handles on shared
  // files scale with workers like per-worker files (docs/bench-log.md 2026-10-11).
  async run({ size = "expert", W = 1, reads = 384, mode = "read", shared = false } = {}) {
    const S = SIZES[size] ?? +size;
    shared = shared && (await this.probeShared());
    if (W > this.files.length && !shared) throw new Error(`${W} workers need ${W} files without shared handles`);
    const ws = await this.pool(W), rnd = rng(++this.seed * 7919);
    // distinct S-sized slots across all files; each worker takes its share (from its own files when not shared)
    const slotsPer = Math.floor(this.fileBytes / S), owner = (f) => f % W;
    const want = Math.min(reads, slotsPer * this.files.length), perW = Math.floor(want / W);
    const pick = (fs, count) => {   // count random distinct (file, slot) from files fs
      const total = slotsPer * fs.length, seen = new Set(), out = [];
      while (out.length < 2 * count && seen.size < total) {
        const k = Math.floor(rnd() * total); if (seen.has(k)) continue; seen.add(k);
        out.push(fs[Math.floor(k / slotsPer)], (k % slotsPer) * S);
      }
      return out;
    };
    const fileIdx = this.files.map((_, i) => i);
    let jobs;
    if (shared) { const all = pick(fileIdx, perW * W), w2 = []; for (let w = 0; w < W; w++) w2.push(all.slice(w * 2 * perW, (w + 1) * 2 * perW)); jobs = w2; }
    else jobs = ws.map((_, w) => pick(fileIdx.filter((f) => owner(f) === w), perW));
    const gpu = mode === "write" || mode === "map" ? mode : null;
    await Promise.all(ws.map((w, i) => ask(w, { op: "open", files: shared ? this.files : this.files.filter((_, f) => owner(f) === i), shared, gpu, size: S })));
    // xfer: the page's device takes the pieces
    let dst = null; const SS = (S + 3) & ~3;
    if (mode === "xfer") dst = this.device.createBuffer({ size: SS * RING * W, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const at = now() + 50;
    const res = await Promise.all(ws.map((w, k) => ask(w, { op: "go", files: this.files, jobs: Float64Array.from(jobs[k]), size: S, mode, at },
      (p) => { this.device.queue.writeBuffer(dst, (k * RING + p.slot) * SS, p.buf); w.postMessage({ op: "ack", i: p.i }); })));
    let t1 = Math.max(...res.map((r) => r.t1));
    if (dst) { await this.device.queue.onSubmittedWorkDone(); t1 = Math.max(t1, now()); dst.destroy(); }
    await Promise.all(ws.map((w) => ask(w, { op: "close" })));
    const lat = []; for (const r of res) lat.push(...r.lat);
    const perWorker = res.map((r) => r.lat.length);
    if (perWorker.some((n) => n !== perW)) throw new Error(`uneven split: ${perWorker} reads, want ${perW} each`);
    return { size, S, mode, W, shared, ...summary(lat, lat.length * S, Math.min(...res.map((r) => r.t0)), t1, S) };
  }

  // ---- Cache API (room/weightcache.js key shape and x-swarm-len header, its own cache name) ----
  async cacheSetup({ CACHE_GB = 1 } = {}) {
    const { cacheKey } = await import("../../room/weightcache.js");
    const S = SIZES.expert, n = Math.floor(CACHE_GB * 2 ** 30 / S), c = await caches.open("pooled-opfs-probe");
    this.cacheKeys = Array.from({ length: n }, (_, i) => cacheKey("https://probe.invalid/expert.bin", i * S, (i + 1) * S - 1));
    const have = new Set((await c.keys()).map((r) => r.url));
    const todo = this.cacheKeys.filter((k) => !have.has(k)), buf = pattern(CHUNK);
    const t = now();
    for (const k of todo) {
      const i = this.cacheKeys.indexOf(k), o = (i * S) % (CHUNK - S) & ~3, body = buf.slice(o, o + S);
      new Uint32Array(body.buffer)[0] = i;
      await c.put(k, new Response(body, { status: 200, headers: { "content-type": "application/octet-stream", "x-swarm-len": String(S) } }));
    }
    const sec = (now() - t) / 1e3;
    this.log(`cache setup: ${n} entries of ${S} B (${(n * S / 2 ** 30).toFixed(2)} GiB), ${todo.length ? `put ${todo.length} in ${sec.toFixed(1)} s (${(todo.length * S / sec / 1e9).toFixed(2)} GB/s)` : "kept from an earlier run"}`);
    return { entries: n, wrote: todo.length, putGBs: todo.length ? +(todo.length * S / sec / 1e9).toFixed(2) : null };
  }

  // C requests in flight from the page's main thread, `reads` distinct entries in random order; read as rangeFetch
  // does (match, check the length, arrayBuffer). Not measured from workers.
  async cacheRun({ C = 1, reads = 256 } = {}) {
    const S = SIZES.expert, c = await caches.open("pooled-opfs-probe"), rnd = rng(++this.seed * 104729);
    const order = shuffle(this.cacheKeys.slice(), rnd).slice(0, reads), lat = [];
    let next = 0;
    const t0 = now();
    await Promise.all(Array.from({ length: C }, async () => {
      while (next < order.length) {
        const k = order[next++], s = now(), hit = await c.match(k);
        if (!hit || hit.headers.get("x-swarm-len") !== String(S)) throw new Error("cache miss " + k);
        if ((await hit.arrayBuffer()).byteLength !== S) throw new Error("short entry");
        lat.push(now() - s);
      }
    }));
    return { size: "expert", S, mode: "cache", W: C, ...summary(lat, lat.length * S, t0, now(), S) };
  }

  // Remove the test files and the cache (KEEP=0 in the driver)
  async cleanup() {
    for (const w of this.workers) w.terminate(); this.workers = [];
    await (await navigator.storage.getDirectory()).removeEntry(DIR, { recursive: true }).catch(() => {});
    await caches.delete("pooled-opfs-probe").catch(() => {});
  }
}
