// The room node's reader for expert offload's disk tier (roadmap/32-ssd-expert-offload.md, step 2): routed experts
// read from the model's local GGUF files when a layer's experts are not in memory, off the event loop.
//
// One expert of a stacked [nExp][rows][cols] tensor is one contiguous range of the file, and decode needs K of them
// per layer at once. A single synchronous handle reads them one after another (an NVMe drive gives ~1.8 GB/s at
// queue depth 1 and 5-6 GB/s at 4-8), so here a pool of worker threads, each with its own fd per shard, takes the
// range in ~1 MiB pieces of whole rows: K experts x 3 parts keep many reads in flight. Each worker converts its rows
// with engine/gguf.js convertEntry (the loader's own conversion: Q4_0 / Q8_0 blocks and the Q8 requant never span rows,
// so the bytes are exactly what the whole tensor's conversion has at that expert).
//
// The main thread touches no bytes. A part (or readAt range) of up to chunkBytes is one job: its worker reads it,
// converts it, and transfers the buffers it made to the main thread (postMessage's transfer list, no copy). A larger
// part, or a requantized one split over several workers for its CPU time, is read in pieces into a SharedArrayBuffer
// staging ring and gathered into buffers of its own by a worker. On Windows a fresh ~1 MiB buffer costs 0.3-0.45 ms
// to fault in (V8's allocator hands back new pages each time); when the main thread allocated and copied every part
// out of the ring, that alone held the pool to ~2.6 GB/s however many workers read (event loop 100% busy, 4 GB/s
// cached reads at most), the RTX 5070 PC's disk-bench ceiling. In the workers it runs in parallel.
//
//   const disk = expertDiskReader({ paths: model.localPaths });
//   const { qs, sc } = await disk.readPart(G.tensors["blk.30.ffn_up_exps.weight"], 17);
//
// The contract is engine/expert_disk.js's (PR 1): readPart(info, e) -> { qs, sc } for a header entry with shape
// [nExp, rows, cols], ggmlType, byteOffset and shard; and readAt(shard, byteOffset, byteLength) -> Uint8Array, so
// either the engine's in-process reader (expertDisk({ readAt: disk.readAt })) or this one (readPart converts in the
// workers) can serve the ExpertStore. shard.js wires it in for a deal's disk layers (expertReader, roadmap 32 step 5).
//
// Reads of the same range that overlap in time share one read (in-flight dedup; nothing is kept once it is done:
// a RAM cache of experts is step 3, T1). POOLED_DISK_THREADS sets the default pool size.
//
// A worker that dies (an uncaught error, OOM) fails the chunks it was running and is replaced, up to maxRespawns
// over the reader's life, once the pool has finished a job (a worker that cannot start at all, say a bundle without
// expert_disk_worker.js, is not restarted). With no worker left, every request fails at once with that error.
// There is no in-process fallback here, unlike convert.js: the engine's own reader (engine/expert_disk.js over
// readAt) is that fallback, and shard.js expertReader switches to it when this one has no worker left.
//
// Two worker pools in one room node: this one (up to 16 threads, mostly blocked in the kernel on reads) and
// convert.js's (up to 8, CPU-bound requant of whole tensors, only while a model loads). Both can run at once when a
// load converts while experts are read, so a node may have up to 24 extra threads; the requant work is the same
// (convertEntry's requantQ8Streaming on row slices). In a room node's load they don't overlap: shard.js makes this
// reader during the load, but its workers start on the first read, and convert.js's pool closes when the load ends.
// Sharing one pool is still a follow-up (roadmap 32, step 2).
import os from "node:os";
import { ggmlTypeBytes, GGML_F32, GGML_F16, GGML_BF16, GGML_Q4_0, GGML_Q8_0 } from "../../engine/gguf.js";

const MiB = 2 ** 20;

export function defaultDiskThreads(env = process.env) {
  const n = parseInt(env.POOLED_DISK_THREADS, 10);
  if (n > 0) return Math.min(n, 64);
  // 16: the GB10's NVMe drive reaches its random-read rate for decode-shaped reads at 16 reads in flight (4.2-4.9 GB/s
  // for Flash's experts, 2.0 at 4; docs/bench-log.md, tests/bench/disk_expert_bench.mjs). The workers spend that
  // time blocked in the kernel, not on a core; a smaller machine gets one fewer than its cores
  return Math.max(1, Math.min(16, (os.availableParallelism?.() ?? os.cpus().length) - 1));
}

// one expert of a stacked expert tensor: where it is and what it converts to
export function expertPart(info, e) {
  if (info.shape?.length !== 3) throw new Error(`expert disk: ${info.name || "tensor"} is not a stacked [nExp][rows][cols] expert tensor`);
  const [nExp, rows, cols] = info.shape, T = info.ggmlType;
  if (T === GGML_F32 || T === GGML_F16 || T === GGML_BF16) throw new Error(`expert disk: experts of ggml type ${T} stay f32, not Q4_0 / Q8_0`);
  const rowBytes = ggmlTypeBytes(T, cols);
  if (!(rowBytes > 0) || rowBytes % 1 || cols % 32) throw new Error(`expert disk: unsupported ggml type ${T} with ${cols} columns`);
  if (!(Number.isInteger(e) && e >= 0 && e < nExp)) throw new Error(`expert disk: expert ${e} of ${nExp}`);
  const kind = T === GGML_Q4_0 ? "q4" : "q8";
  const nb = (rows * cols) / 32;
  return {
    shard: info.shard || 0, off: info.byteOffset + e * rows * rowBytes, len: rows * rowBytes,
    rows, cols, rowBytes, ggmlType: T, kind,
    requant: T !== GGML_Q4_0 && T !== GGML_Q8_0,   // the slow conversion (convertEntry's requantQ8Streaming)
    qsRow: kind === "q4" ? cols / 2 : cols,   // converted bytes per row
    qsBytes: kind === "q4" ? rows * cols / 2 : rows * cols,
    scBytes: Math.ceil(nb / 2) * 4,   // f16 per block, padded to whole u32 words like convertEntry's Uint32Array
  };
}

// Requests for the same key that overlap in time share one read: shared(key, make, copy) runs make() for the first
// caller and hands it the result; every other one gets a copy made (copy(v)) before the first caller's code runs, so
// no caller sees another's changes. A copy that throws (out of memory) fails only the caller it was for.
// onJoin(): a request joined one in flight
export function sharedReads(onJoin = () => {}) {
  const shares = new Map();   // key -> the waiters of the read in flight
  return function shared(key, make, copy) {
    const ws = shares.get(key);
    if (ws) { onJoin(); return new Promise((resolve, reject) => ws.push({ resolve, reject })); }
    const waiters = [];
    shares.set(key, waiters);
    return make().then(
      (v) => {
        shares.delete(key);
        for (const w of waiters) { try { w.resolve(copy(v)); } catch (err) { w.reject(err); } }
        return v;
      },
      (err) => { shares.delete(key); for (const w of waiters) w.reject(err); throw err; });
  };
}

export function expertDiskReader({ paths, threads = defaultDiskThreads(), chunkBytes = MiB, ringSlots = 32, slotBytes = 4 * MiB,
  perWorker = 2, maxRespawns = threads, workerURL = new URL("./expert_disk_worker.js", import.meta.url) } = {}) {
  if (!paths?.length) throw new Error("expertDiskReader: paths (one local GGUF file per shard) are required");
  let workers = null, closed = false, seq = 0;
  let dead = null, worked = false;   // dead: the error every request gets once no worker is left; worked: a job finished
  const jobs = [], waiting = new Map();   // queued chunk jobs; id -> { resolve, reject, w }
  const inflight = new Map();   // range key -> the read every overlapping request shares
  // chunks: read (and convert) jobs; gathers: jobs copying bytes out of the ring; gatherMs: their time, plus joining
  // a long readAt's pieces
  const stats = { parts: 0, raw: 0, bytes: 0, chunks: 0, gathers: 0, readMs: 0, convMs: 0, gatherMs: 0, dedup: 0, slotWaits: 0, oversize: 0, lost: 0, respawns: 0 };

  // no worker left: fail what is queued and waiting for a slot, and every request from now on
  function die(err) {
    dead = new Error(`expert disk reader: no worker left (${String(err?.message || err).split("\n")[0]})`);
    while (jobs.length) jobs.shift().reject(dead);
    for (const s of slotWaiters.splice(0)) s.reject(dead);
  }
  function spawn(Worker) {
    const w = new Worker(workerURL, { workerData: { paths: [...paths] } });
    w.busy = 0;
    w.unref();   // only a worker with jobs holds the process open (refd in pump)
    w.on("message", (m) => {
      const p = waiting.get(m.id); if (!p) return;
      waiting.delete(m.id);
      if (--w.busy === 0) w.unref();
      worked = true;
      m.error ? p.reject(new Error(m.error)) : p.resolve(m);
      pump();
    });
    const lost = (err) => {   // a worker that died fails the chunks it had; a replacement takes its place
      if (!workers?.includes(w)) return;
      workers.splice(workers.indexOf(w), 1);
      stats.lost++;
      for (const [id, p] of waiting) if (p.w === w) { waiting.delete(id); p.reject(err); }
      if (!closed && worked && stats.respawns < maxRespawns) {
        stats.respawns++;
        try { workers.push(spawn(Worker)); } catch {}
      }
      if (!workers.length) die(err);
      pump();
    };
    w.on("error", (e) => lost(e));
    w.on("exit", (code) => lost(new Error(`expert disk worker exited (${code})`)));
    return w;
  }
  function start() {
    // dynamic import: a bundle or test that never reads from disk never loads worker_threads
    return import("node:worker_threads").then(({ Worker }) => {
      if (workers || closed) return;
      workers = [];
      let err = null;
      for (let i = 0; i < threads; i++) try { workers.push(spawn(Worker)); } catch (e) { err = e; }
      if (!workers.length) die(err);
    });
  }
  let starting = null;
  function pump() {
    while (jobs.length && workers?.length) {
      let w = null;
      for (const x of workers) if (x.busy < perWorker && (!w || x.busy < w.busy)) w = x;
      if (!w) return;
      const j = jobs.shift();
      waiting.set(j.msg.id, { resolve: j.resolve, reject: j.reject, w });
      if (w.busy++ === 0) w.ref();
      w.postMessage(j.msg);
    }
  }
  function run(msg) {
    if (closed) return Promise.reject(new Error("expert disk reader is closed"));
    if (dead) return Promise.reject(dead);
    return new Promise((resolve, reject) => {
      jobs.push({ msg: { id: ++seq, ...msg }, resolve, reject });
      if (workers) pump();
      else (starting ||= start()).then(pump, (e) => { while (jobs.length) jobs.shift().reject(e); });
    });
  }
  // every chunk job of one request, settled; the first error after all of them are done (a chunk still running
  // would otherwise write into a slot that was handed out again)
  async function settled(ps) {
    const rs = await Promise.allSettled(ps);
    const bad = rs.find((r) => r.status === "rejected");
    if (bad) throw bad.reason;
    return rs.map((r) => r.value);
  }
  const tally = (m) => {
    if (m.bufs) { stats.gatherMs += m.gatherMs || 0; stats.gathers++; }
    else { stats.readMs += m.readMs || 0; stats.convMs += m.convMs || 0; stats.chunks++; }
    return m;
  };
  async function all(ps) { (await settled(ps)).forEach(tally); }
  const job = (msg) => run(msg).then(tally);

  const shared = sharedReads(() => stats.dedup++);
  // rows per job for a part: chunkBytes of them, and for a part requantized to Q8 (Q4_1 / Q5_0 / K-quants: ~19 ms
  // of CPU per Flash part against ~0.5 for a Q4_0 repack) at most a quarter of the part, so its conversion runs on
  // four workers at once. A part that fits in one job skips the ring
  function rowsPerJob(p) {
    let per = Math.max(1, Math.floor(chunkBytes / p.rowBytes));
    if (p.requant) per = Math.min(per, Math.ceil(p.rows / Math.min(4, threads)));
    return per;
  }
  // a worker copies these [offset, length] ranges of a ring slot into buffers of its own and hands them over
  const gather = (sab, ranges) => job({ gather: ranges, src: sab }).then((m) => m.bufs.map((b) => new Uint8Array(b)));

  // the staging ring: ringSlots slots of slotBytes (4 MiB: a converted part of the 122B is 2.6 MB, Flash's 0.9),
  // one SharedArrayBuffer made on first use and reused. Not a new SharedArrayBuffer per read: allocating them
  // took most of the main thread and got slower the longer a process ran (a third reader in one process read at
  // a third of the first one's rate; docs/bench-log.md). A part larger than a slot gets a buffer of its own. A
  // request waits for a free slot; readPart and readAt give it back once a worker has gathered the bytes out.
  let ring = null, free = [], slotWaiters = [];
  function slotFor(need) {
    if (closed) return Promise.reject(new Error("expert disk reader is closed"));
    if (dead) return Promise.reject(dead);
    if (!ring) {
      const per = Math.ceil(slotBytes / 4096) * 4096;
      ring = { per, sab: new SharedArrayBuffer(per * ringSlots) };
      free = Array.from({ length: ringSlots }, (_, i) => i);
    }
    if (need > ring.per) { stats.oversize++; return Promise.resolve({ sab: new SharedArrayBuffer(need), base: 0, own: true }); }
    if (free.length) return Promise.resolve({ sab: ring.sab, base: free.pop() * ring.per });
    stats.slotWaits++;
    return new Promise((resolve, reject) => slotWaiters.push({ resolve, reject }));
  }
  function freeSlot(s) {
    if (s.own || !ring || s.sab !== ring.sab) return;
    const w = slotWaiters.shift();
    if (w) w.resolve(s); else free.push(s.base / ring.per);
  }

  // one expert part into a slot, its rows split over the workers -> { qs, sc, release() } (views of the ring:
  // valid until release(); a caller that holds more than ringSlots leases at once waits forever)
  async function leasePart(info, e) {
    const p = expertPart(info, e);
    const key = `${p.shard}:${p.off}`;
    let r = inflight.get(key);
    if (r) stats.dedup++;
    else {
      r = { refs: 0, slot: null };
      r.ready = (async () => {
        const slot = r.slot = await slotFor(p.qsBytes + p.scBytes);
        const scBase = slot.base + p.qsBytes;
        new Uint8Array(slot.sab, scBase, p.scBytes).fill(0, ((p.rows * p.cols) / 32) * 2);   // the padding no chunk writes
        const per = rowsPerJob(p);
        const ps = [];
        for (let r0 = 0; r0 < p.rows; r0 += per) {
          const rc = Math.min(per, p.rows - r0);
          ps.push(run({ shard: p.shard, off: p.off + r0 * p.rowBytes, len: rc * p.rowBytes, dst: slot.sab, dstOff: slot.base + r0 * p.qsRow,
            conv: { ggmlType: p.ggmlType, rows: rc, cols: p.cols, scOff: scBase + ((r0 * p.cols) / 32) * 2 } }));
        }
        try { await all(ps); } catch (err) { freeSlot(slot); throw err; }
        stats.parts++; stats.bytes += p.len;
      })();
      inflight.set(key, r);
      r.ready.then(() => inflight.delete(key), () => inflight.delete(key));
    }
    r.refs++;
    try { await r.ready; } catch (err) { r.refs--; throw err; }
    let done = false;
    const { sab, base } = r.slot;
    return {
      qs: new Uint8Array(sab, base, p.qsBytes), sc: new Uint8Array(sab, base + p.qsBytes, p.scBytes), kind: p.kind,
      release() { if (done) return; done = true; if (--r.refs === 0) freeSlot(r.slot); },
    };
  }

  // expert e of one part -> { qs, sc }: plain Uint8Arrays of their own, the engine's layout (what a parked
  // layer's slices hold; engine/expert_disk.js readPart's contract). One job when the part fits in one: the worker's
  // own buffers, transferred. Else through a ring slot (leasePart) and gathered out of it by a worker
  async function readPart(info, e) {
    const p = expertPart(info, e);
    return shared(`part:${p.shard}:${p.off}`, async () => {
      if (rowsPerJob(p) >= p.rows) {
        const m = await job({ shard: p.shard, off: p.off, len: p.len, own: true, conv: { ggmlType: p.ggmlType, rows: p.rows, cols: p.cols } });
        stats.parts++; stats.bytes += p.len;
        return { qs: new Uint8Array(m.qs), sc: new Uint8Array(m.sc) };
      }
      const l = await leasePart(info, e);
      try {
        const [qs, sc] = await gather(l.qs.buffer, [[l.qs.byteOffset, l.qs.byteLength], [l.sc.byteOffset, l.sc.byteLength]]);
        return { qs, sc };
      } finally { l.release(); }
    }, (v) => ({ qs: v.qs.slice(), sc: v.sc.slice() }));
  }

  return {
    threads,
    stats,
    // how many leases (leasePart) a caller may hold at once: the ring's slots
    leaseSlots: ringSlots,
    // the bytes [byteOffset, byteOffset + byteLength) of one shard -> a Uint8Array of its own (plain memory, not
    // shared). Up to chunkBytes: one job, the worker's buffer transferred. Longer: through ring slots a slot at a
    // time, each slot's bytes in chunkBytes pieces over the workers, gathered out by a worker (and, for a read longer
    // than a slot, put together here). A request that joins one already in flight gets a copy of its bytes, so
    // callers never share a buffer
    readAt(shard, byteOffset, byteLength) {
      return shared(`raw:${shard}:${byteOffset}:${byteLength}`, async () => {
        let out;
        if (byteLength <= chunkBytes) out = new Uint8Array((await job({ shard, off: byteOffset, len: byteLength, own: true })).buf);
        else {
          const per = ring?.per || Math.ceil(slotBytes / 4096) * 4096, pieces = [];
          for (let p0 = 0; p0 < byteLength; p0 += per) pieces.push((async () => {
            const n = Math.min(per, byteLength - p0), slot = await slotFor(n), ps = [];
            try {
              for (let o = 0; o < n; o += chunkBytes)
                ps.push(run({ shard, off: byteOffset + p0 + o, len: Math.min(chunkBytes, n - o), dst: slot.sab, dstOff: slot.base + o }));
              await all(ps);
              return (await gather(slot.sab, [[slot.base, n]]))[0];
            } finally { freeSlot(slot); }
          })());
          const got = await settled(pieces);
          if (got.length === 1) out = got[0];
          else {
            const t0 = performance.now();
            out = new Uint8Array(byteLength);
            got.forEach((b, i) => out.set(b, i * per));
            stats.gatherMs += performance.now() - t0;
          }
        }
        stats.raw++; stats.bytes += byteLength;
        return out;
      }, (b) => b.slice());
    },
    leasePart,
    readPart,
    // expert e of a layer's [gate, up, down] infos, all three at once (like readAt, readPart and leasePart: works
    // unbound, const { readExpert } = disk)
    readExpert(parts, e) { return Promise.all(parts.map((info) => readPart(info, e))); },
    // stops the workers (each closes its fds first); what is still queued, running or waiting for a slot fails
    // now. A lease still held stays readable (its views keep the ring alive); release() after close is a no-op
    async close() {
      if (closed) return;
      closed = true;
      const err = new Error("expert disk reader is closed");
      while (jobs.length) jobs.shift().reject(err);
      for (const w of slotWaiters.splice(0)) w.reject(err);
      // a chunk still running may write into its slot after this; nothing hands that slot out again
      for (const [id, p] of waiting) { waiting.delete(id); p.reject(err); }
      await starting?.catch(() => {});
      const ws = workers || [];
      workers = null;
      await Promise.all(ws.map(async (w) => {
        w.ref();
        await new Promise((resolve) => {
          const t = setTimeout(resolve, 2000);
          w.on("message", (m) => { if (m.closed) { clearTimeout(t); resolve(); } });
          w.postMessage({ id: 0, close: true });
        });
        await w.terminate().catch(() => {});
      }));
      ring = null; free = [];
    },
  };
}
