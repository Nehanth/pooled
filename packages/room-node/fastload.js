// A faster shard load from local model files (a room node's re-deal; shard.js loadFlash). Loading the Flash demo's PC
// share (layers 14-47 of Qwen3.8-Flash-Next, the experts of 31 layers parked in RAM: 46 GB read) took ~100 s on the
// RTX 5070 PC, every step one after another on the main thread: read 38.8 s (one 450 MB read at a time, ~1.2 GB/s),
// repack 34.9 s (q4Repack, inline), park 21.9 s, upload 2.3 s. Here:
//   - reads run ahead of the loader, in loader order, up to `aheadBytes` past what it has taken, each tensor in
//     `chunk`-byte pieces on the source's file handles at once (source.js localFileReader's lanes): 4 lanes of
//     64 MiB read 6.6 GB/s cold on that PC's NVMe, against 2.9 for one 1 GiB read;
//   - the big Q4_0 / Q8_0 tensors are read into shared memory and repacked by convert.js's worker threads straight into
//     shared output buffers (the inline loops, q4RepackInto / q8RepackInto: the same bytes), so the main thread only
//     parks (a memcpy into the mapped buffers) or uploads;
//   - those buffers are reused from tensor to tensor (a pool by size), so the load does not churn ~90 GB of garbage
//     past the RAM it parks.
// POOLED_FASTLOAD=0 turns it off (the loader reads and converts each tensor as before).
//
//   const fl = fastLoad({ src, G, plan, pool })   // plan: the tensors the loader will read, in its order (G.tensors infos)
//   bytesOf: fl.bytesOf, G.convert: fl.convert; after a parked entry is copied or an entry uploaded: fl.release(entry)
import { convertEntry, GGML_Q4_0, GGML_Q8_0 } from "../../engine/gguf.js";

const MiB = 2 ** 20;
const keyOf = (info) => `${info.shard || 0}:${info.byteOffset}`;

export const fastLoadOn = (env = process.env) => env.POOLED_FASTLOAD !== "0";

// src: source.js openModel (readInto: null when the model is streamed: then null here). G: the header (G.convert: the
// fallback conversion). plan: [info] in loader order. pool: convert.js convertPool (repackShared).
// minShared: tensors at least this big (Q4_0 / Q8_0, 2-D once stacked experts are a matrix) go through shared memory
export function fastLoad({ src, G, plan, pool, aheadBytes = 3 * 2 ** 30, chunk = 64 * MiB, minShared = 64 * MiB, log = () => {} }) {
  if (!src?.readInto || !pool?.repackShared || !(pool.threads >= 2) || typeof SharedArrayBuffer !== "function") return null;
  const fallback = G.convert || convertEntry;
  // stacked experts [nExp][rows][cols] are one matrix of nExp * rows rows (as engine/gguf.js ggufEntry makes them)
  const as2d = (info) => (info.shape.length === 3 ? { ...info, shape: [info.shape[0] * info.shape[1], info.shape[2]] } : info);
  const shared = (info) => info.byteLength >= minShared && (info.ggmlType === GGML_Q4_0 || info.ggmlType === GGML_Q8_0) && as2d(info).shape.length === 2;
  // the other quants (Q5_0, Q6_K, ...: requantized to Q8) convert on the threads too, as soon as they are read
  const requant = (info) => !!pool.requantizes?.(as2d(info));
  const order = [], byKey = new Map();
  for (const info of plan) {
    const k = keyOf(info);
    if (byKey.has(k) || !(info.byteLength > 0)) continue;
    const t = { info, k, p: null, conv: null, taken: false, tr: { name: info.name, len: info.byteLength } };
    byKey.set(k, t); order.push(t);
  }
  // shared buffers, reused: free lists by size (rounded up to a MiB)
  const free = [];
  const stats = { reads: 0, readBytes: 0, waitMs: 0, repacked: 0, inline: 0, newBytes: 0 };
  const acquire = (bytes) => {
    const want = Math.ceil(bytes / MiB) * MiB;
    let best = -1;
    for (let i = 0; i < free.length; i++) if (free[i].byteLength >= want && (best < 0 || free[i].byteLength < free[best].byteLength)) best = i;
    if (best >= 0) return free.splice(best, 1)[0];
    stats.newBytes += want;
    return new SharedArrayBuffer(want);
  };
  const give = (sab) => { if (!closed && sab instanceof SharedArrayBuffer && !free.includes(sab)) free.push(sab); };
  // ahead: bytes read (or being read) and not taken by the loader yet, so the read-ahead stops at aheadBytes
  let next = 0, ahead = 0, closed = false;
  // a shared tensor is repacked on the threads as soon as its bytes are in, ahead of the loader: -> the entry, or null
  // (the pool can't: the loader's convert does it inline). Its input buffer goes back to the pool once repacked.
  const repack = (t, bytes) => {
    const info = as2d(t.info), n = info.nElems ?? info.shape.reduce((a, b) => a * b, 1);
    const out = { qs: acquire(info.ggmlType === GGML_Q4_0 ? n / 2 : n), sc: acquire(Math.ceil(n / 64) * 4) };
    t.tr.c0 = performance.now();
    return pool.repackShared(info, bytes, out).then((e) => {
      t.tr.c1 = performance.now();
      if (!e) { give(out.qs); give(out.sc); return null; }
      give(bytes.buffer);
      stats.repacked++;
      e.release = () => { give(out.qs); give(out.sc); e.release = null; };
      return e;
    }, () => { give(out.qs); give(out.sc); return null; });
  };
  const read = (t) => {
    if (t.p) return t.p;
    const { info } = t, len = info.byteLength, sab = shared(info) ? acquire(len) : null;
    const into = sab ? new Uint8Array(sab, 0, len) : new Uint8Array(len);
    ahead += len;
    const parts = [];
    for (let o = 0; o < len; o += chunk) {
      const n = Math.min(chunk, len - o);
      parts.push(src.readInto(info.shard || 0, info.byteOffset + o, n, into.subarray(o, o + n)));
    }
    stats.reads++; stats.readBytes += len;
    t.tr.r0 = performance.now();
    t.p = Promise.all(parts).then(() => { t.tr.r1 = performance.now(); return into; });
    if (sab) { t.conv = t.p.then((b) => (closed ? null : repack(t, b))); t.conv.catch(() => {}); }
    else if (requant(info)) {
      t.conv = t.p.then((b) => (closed ? null : pool.convert(as2d(info), b, { any: true })), () => null).catch(() => null);
      t.plainConv = true;
    }
    t.p.catch(() => {});
    return t.p;
  };
  const pump = () => {
    while (!closed && next < order.length && (ahead < aheadBytes || ahead === 0)) {
      const t = order[next++];
      if (!t.taken && !t.p) read(t);
    }
  };
  pump();
  return {
    stats,
    trace: () => order.map((t) => t.tr),
    // the loader's bytesOf: a planned tensor from the read-ahead (once), anything else from the source
    async bytesOf(info) {
      const t = byKey.get(keyOf(info));
      if (!t || t.taken || closed) return src.bytesOf(info);
      t.taken = true;
      const w = t.tr.take = performance.now();
      const p = read(t);
      try { return await p; } finally { t.tr.got = performance.now(); t.p = null; ahead -= info.byteLength; stats.waitMs += performance.now() - w; pump(); }
    },
    // the loader's G.convert: a tensor read into shared memory comes back repacked by the threads (in shared buffers:
    // release() once used); anything else, or when the threads couldn't: the fallback (convert.js's convert)
    async convert(info, bytes) {
      const tt = byKey.get(keyOf(info)), w0 = performance.now();
      try { return await convert1(info, bytes); } finally { if (tt) tt.tr.conv = performance.now() - w0; }
    },
    upTrace(info, ms) { const tt = info && byKey.get(keyOf(info)); if (tt) tt.tr.up = ms; },
    // an entry convert made, once its bytes are no longer needed (parked: copied into the store's buffers; uploaded:
    // written to the GPU with no CPU copy kept): its buffers go back to the pool
    release(e) { if (e?.release && !e.qs && !e.scales) e.release(); },
    close() { closed = true; free.length = 0; for (const t of order) { t.p = null; t.conv = null; } },
  };
  async function convert1(info, bytes) {
    {
      const tp = byKey.get(keyOf(info));
      if (tp?.plainConv && tp.conv && !(bytes?.buffer instanceof SharedArrayBuffer)) {   // requantized ahead (a fresh entry)
        const c = tp.conv; tp.conv = null;
        const e = await c;
        if (e) { stats.requant = (stats.requant || 0) + 1; return e; }
        stats.inline++;
        return fallback(info, bytes);
      }
      const t = bytes?.buffer instanceof SharedArrayBuffer ? tp : null;
      const conv = t?.conv;
      if (t) t.conv = null;
      const e = conv ? await conv : null;
      if (e) { e.shape = info.shape; return e; }
      stats.inline++;
      const r = await fallback(info, bytes);   // a fresh entry: the input is free once it returns
      if (t) give(bytes.buffer);
      return r;
    }
  }
}
