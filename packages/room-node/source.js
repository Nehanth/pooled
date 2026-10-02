// Where a room node reads a model's bytes: a local copy of the room's GGUF when it has one (in
// modelDir: <model key>/<file name> as `pooled pull` keeps it, or the LOCAL layout below), else HTTP range requests to that URL,
// the same requests the room page makes. The room page's weight caches, peer-to-peer weight
// transfer and download pacing are browser features left out of this prototype.
import fs from "node:fs";
import path from "node:path";
import { MODELS } from "../../room/models.js";
import { parseGGUFHeader } from "../../engine/gguf.js";
import { mergeSplitHeaders } from "../../room/models.js";

// file name in the room's URL -> path under modelDir (the layout of tests/e2e/xroom.mjs LOCAL)
export const LOCAL = {
  "Qwen3.8-27B-Q4_0.gguf": "q38/model.gguf",
  "Qwen3-0.6B-Q8_0.gguf": "qwen/model.gguf",
  "Qwen3-1.7B-Q8_0.gguf": "qwen17/model.gguf",
  "Qwen_Qwen3.6-35B-A3B-Q4_0.gguf": "q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf",
};

// A split GGUF (room/models.js MODELS[key].shards: llama.cpp's gguf-split, e.g. the 122B in two files) is read as one
// model: the metadata is the first file's, every file lists its own tensors (offsets within that file), so each
// tensor remembers its file (info.shard) and bytesOf reads it from there. Local copies sit side by side
// (<modelDir>/<key>/<file>, as `pooled pull` keeps them); a model is local only when every file is.
export function openModel(modelKey, { modelDir = process.env.POOLED_MODELS || null, fetch: fetchImpl = globalThis.fetch, concurrency = 6, capBytes = 512 * MiB } = {}) {
  const M = MODELS[modelKey];
  if (!M || (M.kind !== "gguf" && M.kind !== "qwen35")) throw new Error(`unsupported model ${modelKey}`);
  const urls = M.shards?.length ? M.shards : [M.gguf];
  // `pooled pull`'s layout (<dir>/<model key>/<file>, cli/lib/cache.js) first, then the test layout
  const localOf = (url) => { const base = url.split("/").pop(); return [path.join(modelDir, modelKey, base), LOCAL[base] && path.join(modelDir, LOCAL[base])].find((f) => f && fs.existsSync(f)) || null; };
  const locals = modelDir ? urls.map(localOf) : urls.map(() => null);
  const isLocal = locals.every(Boolean);
  const local = isLocal ? locals[0] : null;
  // what this load has read so far (loadstat): bytes from the network or the disk since plan(), of total
  const stat = { from: isLocal ? "disk" : "Hugging Face", fetched: 0, total: 0, planned: false };
  const count = (n) => { if (stat.planned) stat.fetched += n; };
  const files = urls.map((url, i) => {
    const f = { url, fh: null, pre: null, local: isLocal ? locals[i] : null };
    f.readAt = f.local
      ? async (off, len) => {
        f.fh ||= await fs.promises.open(f.local, "r");
        const out = new Uint8Array(len);
        let o = 0;
        while (o < len) { const { bytesRead } = await f.fh.read(out, o, Math.min(len - o, 1 << 30), off + o); if (bytesRead <= 0) break; o += bytesRead; }
        if (o !== len) throw new Error(`short read in ${f.local} at ${off}`);
        count(len);
        return out;
      }
      : async (off, len) => (f.pre && (await f.pre.read(off, len))) || fetchRange(fetchImpl, url, off, off + len, count);
    return f;
  });
  const fileOf = (info) => files[info?.shard || 0] || files[0];
  const sideFile = (url) => {   // config.json / tokenizer.json for the dense models: next to the local GGUF, else the URL
    const f = local && path.join(path.dirname(local), url.split("/").pop());
    return f && fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, "utf8")) : fetchImpl(url).then((r) => r.json());
  };
  // one file's index (and the tokenizer when asked): 12 MB first, doubling, as room.js fetchGGUFHeader
  const headerOf = async (f, needTokenizer) => {
    for (let size = 12 * 2 ** 20; ; size *= 2) {
      const buf = await f.readAt(0, size);
      try { return parseGGUFHeader(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), { skipTokenizer: !needTokenizer }); }
      catch (e) { if (size > 256 * 2 ** 20) throw e; }
    }
  };
  return {
    M, local, url: M.gguf, stat, shards: urls.length,
    readAt: files[0].readAt,   // the first file (ckptdisk.js modelFileId)
    bytesOf: (info) => fileOf(info).readAt(info.byteOffset, info.byteLength),
    // the tensors this load will read, in the loader's order ([{ byteOffset, byteLength, shard? }]): streaming,
    // they are fetched ahead of the loader in large ranges, several at once; from disk, only counted
    plan(infos) {
      stat.planned = true; stat.fetched = 0;
      if (isLocal) { stat.total = infos.reduce((a, t) => a + t.byteLength, 0); return; }
      stat.total = 0;
      files.forEach((f, i) => {
        f.pre?.close();
        f.pre = rangePrefetcher({ fetchRange: (a, b, onBytes) => fetchRange(fetchImpl, f.url, a, b, (n) => { onBytes(n); count(n); }), concurrency, capBytes: capBytes / files.length });
        stat.total += f.pre.plan(infos.filter((t) => (t.shard || 0) === i).map((t) => ({ off: t.byteOffset, len: t.byteLength })));
      });
    },
    // the GGUF index (and the tokenizer when asked); a split model's every file merged into one (mergeSplitHeaders)
    async header(needTokenizer = true) {
      const G = await headerOf(files[0], needTokenizer);
      if (files.length === 1) return G;
      return mergeSplitHeaders([G, ...(await Promise.all(files.slice(1).map((f) => headerOf(f, false))))]);
    },
    cfg: () => (M.cfg ? sideFile(M.cfg) : null),
    tokJson: () => (M.tok ? sideFile(M.tok) : null),
    async close() { for (const f of files) { f.pre?.close(); f.pre = null; try { await f.fh?.close(); } catch {} f.fh = null; } },
  };
}

const MiB = 2 ** 20;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// One HTTP range [start, end) -> Uint8Array, counting bytes as they arrive (onBytes(n); a failed try
// takes back what it counted). A few tries with backoff on network errors, 5xx, 408 and 429.
export async function fetchRange(fetchImpl, url, start, end, onBytes = () => {}, { tries = 4, backoffMs = 500 } = {}) {
  const len = end - start;
  for (let attempt = 0; ; attempt++) {
    let got = 0;
    try {
      const r = await fetchImpl(url, { headers: { range: `bytes=${start}-${end - 1}` } });
      if (r.status !== 206 && !(r.status === 200 && start === 0)) {
        const e = new Error(`range fetch ${r.status} for ${url}`);
        e.retry = r.status >= 500 || r.status === 408 || r.status === 429;
        try { await r.body?.cancel(); } catch {}
        throw e;
      }
      const out = new Uint8Array(len);
      if (r.body?.getReader) {
        const rd = r.body.getReader();
        for (;;) {
          const { done, value } = await rd.read();
          if (done) break;
          const n = Math.min(value.length, len - got);
          out.set(n === value.length ? value : value.subarray(0, n), got);
          got += n; onBytes(n);
          if (got >= len) { try { await rd.cancel(); } catch {} break; }   // a 200 for the whole file: only the head
        }
      } else {
        const b = new Uint8Array(await r.arrayBuffer());
        got = Math.min(b.length, len); out.set(b.subarray(0, got)); onBytes(got);
      }
      if (got < len) { const e = new Error(`short range read for ${url}: ${got} of ${len} bytes at ${start}`); e.retry = true; throw e; }
      return out;
    } catch (e) {
      if (got) onBytes(-got);
      const retry = e.retry !== false && (e.retry || e.name === "TypeError" || /fetch failed|ECONNRESET|ETIMEDOUT|socket|terminated/i.test(String(e.message)));
      if (!retry || attempt + 1 >= tries) throw e;
      await sleep(backoffMs * 2 ** attempt);
    }
  }
}

// The ranges a load reads ([{ off, len }], in the loader's order) -> the ranges to fetch: sorted by
// offset, neighbours (gaps up to `gap`) merged up to maxChunk, a longer range cut into maxChunk
// pieces. Each chunk: { start, end, order } (order: the first read that needs it).
export function planChunks(ranges, { maxChunk = 64 * MiB, gap = MiB } = {}) {
  const rs = ranges.map((r, order) => ({ ...r, order })).filter((r) => r.len > 0).sort((a, b) => a.off - b.off);
  const chunks = [];
  let cur = null;
  for (const r of rs) {
    const end = r.off + r.len;
    if (cur && r.off - cur.end <= gap && Math.max(end, cur.end) - cur.start <= maxChunk) {
      cur.end = Math.max(cur.end, end); cur.order = Math.min(cur.order, r.order);
      continue;
    }
    if (cur && end <= cur.end) { cur.order = Math.min(cur.order, r.order); continue; }   // inside the last piece
    let s = cur && r.off < cur.end ? cur.end : r.off;   // overlaps the last chunk: the rest of it
    while (s < end) {
      cur = { start: s, end: Math.min(end, s + maxChunk), order: r.order };
      chunks.push(cur);
      s = cur.end;
    }
  }
  return chunks;
}

// Fetches a load's chunks ahead of the loader: up to `concurrency` at once, in the order the loader
// needs them, holding at most ~capBytes fetched but not yet read (the chunk a read waits for always
// goes). read(off, len) -> Uint8Array, or null when the plan does not cover it (the caller fetches it
// itself). A chunk is dropped once every planned read over it is done.
export function rangePrefetcher({ fetchRange: get, concurrency = 6, capBytes = 512 * MiB, maxChunk = 64 * MiB, gap = MiB }) {
  let chunks = [], order = [], next = 0, inflight = 0, held = 0, closed = false;
  const reads = new Map();   // "off:len" -> [chunk index] of each planned read not done yet
  const touching = (off, end) => {
    // chunks are sorted by start and do not overlap: binary search for the first that ends after off
    let lo = 0, hi = chunks.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (chunks[m].end <= off) lo = m + 1; else hi = m; }
    const out = [];
    for (let i = lo; i < chunks.length && chunks[i].start < end; i++) out.push(i);
    return out;
  };
  const covered = (off, end, idx) => {
    let at = off;
    for (const i of idx) { if (chunks[i].start > at) return false; at = Math.max(at, chunks[i].end); }
    return at >= end;
  };
  const start = (i) => {
    const c = chunks[i];
    if (c.p) return c.p;
    inflight++; held += c.end - c.start;
    c.p = get(c.start, c.end, () => {}).then(
      (data) => { inflight--; if (closed || c.dropped) { held -= c.end - c.start; c.data = null; } else c.data = data; pump(); return data; },
      (e) => { inflight--; held -= c.end - c.start; c.failed = true; pump(); throw e; });
    c.p.catch(() => {});
    return c.p;
  };
  const pump = () => {
    while (!closed && inflight < concurrency && next < order.length) {
      const c = chunks[order[next]];
      if (c.p) { next++; continue; }
      if (held + (c.end - c.start) > capBytes && held > 0) return;
      start(order[next++]);
    }
  };
  const drop = (i) => {
    const c = chunks[i];
    if (c.dropped) return;
    c.dropped = true;
    if (c.data) { held -= c.end - c.start; c.data = null; }
    pump();
  };
  return {
    // -> the bytes it will fetch
    plan(ranges) {
      chunks = planChunks(ranges, { maxChunk, gap });
      order = chunks.map((_, i) => i).sort((a, b) => chunks[a].order - chunks[b].order || chunks[a].start - chunks[b].start);
      for (const c of chunks) c.refs = 0;
      for (const r of ranges) {
        if (!(r.len > 0)) continue;
        const k = `${r.off}:${r.len}`;
        if (reads.has(k)) continue;
        const idx = touching(r.off, r.off + r.len);
        reads.set(k, idx);
        for (const i of idx) chunks[i].refs++;
      }
      pump();
      return chunks.reduce((a, c) => a + c.end - c.start, 0);
    },
    async read(off, len) {
      const end = off + len;
      const k = `${off}:${len}`;
      const idx = reads.get(k) || touching(off, end);
      if (!idx.length || !covered(off, end, idx) || idx.some((i) => chunks[i].dropped || chunks[i].failed)) return null;
      const parts = await Promise.all(idx.map((i) => start(i)));
      let out;
      if (idx.length === 1) { const c = chunks[idx[0]]; out = parts[0].subarray(off - c.start, end - c.start); }
      else {
        out = new Uint8Array(len);
        idx.forEach((i, j) => {
          const c = chunks[i], a = Math.max(off, c.start), b = Math.min(end, c.end);
          out.set(parts[j].subarray(a - c.start, b - c.start), a - off);
        });
      }
      if (reads.has(k)) { reads.delete(k); for (const i of idx) if (--chunks[i].refs <= 0) drop(i); }
      return out;
    },
    get held() { return held; },
    get inflight() { return inflight; },
    close() { closed = true; for (const c of chunks) c.data = null; },
  };
}
