// The experts' first save in a worker pool (engine/expert_ingest_pool.js), on the CPU: fake workers that run the
// worker's job (copyLayer) in this thread, a fake OPFS (sync access handles over byte arrays) and a fake HTTP server.
// The files must come out byte for byte what ingestExperts writes; the manifest never says more than the files hold;
// a failure or a stop closes every file; a pool that cannot start says it wrote nothing.
//   deno test --no-check tests/unit/expert_ingest_pool_test.js
import { expertLayout, ingestExperts } from "../../engine/expert_opfs.js";
import { copyLayer, ingestExpertsPool } from "../../engine/expert_ingest_pool.js";
import { expertDisk } from "../../engine/expert_disk.js";
import { ggmlTypeBytes, GGML_Q4_0, GGML_Q4_1 } from "../../engine/gguf.js";

const assert = (c, m) => { if (!c) throw new Error(m); };
const same = (a, b) => a.byteLength === b.byteLength && a.every((x, i) => x === b[i]);

// a fake GGUF: gate / up Q4_0, down Q4_1 (Flash's types), deterministic bytes, small finite f16 scales
function synthModel({ nL = 3, nExp = 8, rows = 32, cols = 64 } = {}) {
  const tensors = {}, parts = [];
  let off = 4096 + 18;
  for (let l = 0; l < nL; l++) for (const [p, type, shape] of [["gate", GGML_Q4_0, [nExp, rows, cols]], ["up", GGML_Q4_0, [nExp, rows, cols]], ["down", GGML_Q4_1, [nExp, cols, rows]]]) {
    const len = ggmlTypeBytes(type, shape[0] * shape[1] * shape[2]);
    tensors[`blk.${l}.ffn_${p}_exps.weight`] = { shape, ggmlType: type, byteOffset: off, byteLength: len, shard: 0 };
    parts.push([off, len, type]);
    off += len + 32;
  }
  const file = new Uint8Array(off);
  let s = 4242;
  for (let i = 0; i < file.length; i++) file[i] = (s = (s * 1103515245 + 12345) >>> 0) >>> 24;
  for (const [o0, len, type] of parts) {
    const blk = type === GGML_Q4_0 ? 18 : 20;
    for (let o = o0; o < o0 + len; o += blk) { file[o] = 0x00; file[o + 1] = 0x24; if (type === GGML_Q4_1) { file[o + 2] = 0x00; file[o + 3] = 0xa4; } }
  }
  return { G: { meta: { "general.name": "synth" }, tensors }, file, readAt: async (_s, o, n) => file.slice(o, o + n) };
}
async function wantFile(m, L) {
  const R = expertDisk({ readAt: m.readAt }), out = new Uint8Array(L.nExp * L.recBytes);
  for (let e = 0; e < L.nExp; e++) for (const [i, p] of ["gate", "up", "down"].entries()) {
    const { qs, sc } = await R.readPart(L.parts[p], e);
    out.set(qs, e * L.recBytes + L.offs[2 * i]); out.set(sc, e * L.recBytes + L.offs[2 * i + 1]);
  }
  return out;
}

// a fake OPFS: directories of files, each a growable byte array; sync handles exclusive per file
function fakeOpfs() {
  const files = new Map(), open = new Set();
  const dir = (path) => ({
    async getDirectoryHandle(n) { return dir([...path, n]); },
    async getFileHandle(n, { create = false } = {}) {
      const k = [...path, n].join("/");
      if (!files.has(k)) { if (!create) throw Object.assign(new Error("no file"), { name: "NotFoundError" }); files.set(k, new Uint8Array(0)); }
      return {
        async createSyncAccessHandle() {
          if (open.has(k)) throw Object.assign(new Error(`${k} is open`), { name: "NoModificationAllowedError" });
          open.add(k);
          let closed = false;
          const live = () => { if (closed) throw new Error("closed handle"); };
          return {
            getSize: () => { live(); return files.get(k).length; },
            truncate: (n) => { live(); const f = files.get(k), g = new Uint8Array(n); g.set(f.subarray(0, Math.min(n, f.length))); files.set(k, g); },
            write: (b, { at }) => {
              live();
              let f = files.get(k);
              if (f.length < at + b.byteLength) { const g = new Uint8Array(at + b.byteLength); g.set(f); f = g; files.set(k, f); }
              f.set(b, at); return b.byteLength;
            },
            flush: () => { live(); },
            close: () => { closed = true; open.delete(k); },
          };
        },
      };
    },
  });
  return { files, open, root: dir([]) };
}
// a fake HTTP server for the model: a 302 from "model" to "cdn" (once), 206s with Content-Range
function fakeFetch(file, { failAt = null } = {}) {
  const st = { reqs: 0, redirects: 0, direct: 0 };
  const f = async (url, init = {}) => {
    if (init.signal?.aborted) throw init.signal.reason;
    st.reqs++;
    const m = /bytes=(\d+)-(\d+)/.exec(init.headers?.Range || "");
    const lo = +m[1], hi = +m[2];
    if (failAt != null && lo <= failAt && failAt <= hi) return { status: 500, headers: { get: () => null }, body: null };
    const redirected = url === "model";
    if (redirected) st.redirects++; else st.direct++;
    const b = file.slice(lo, hi + 1);
    await new Promise((r) => setTimeout(r, 0));
    return { status: 206, redirected, url: redirected ? "cdn?Expires=99999999999" : url, headers: { get: (k) => (k === "content-range" ? `bytes ${lo}-${hi}/${file.length}` : null) }, arrayBuffer: async () => b.buffer };
  };
  return { f, st };
}
// fake workers: each runs copyLayer in this thread, with the worker's message protocol
function fakeWorkers(opfs, fetch) {
  const made = [];
  const make = () => {
    let ac = null;
    const direct = new Map();
    const w = {
      onmessage: null, onerror: null, terminated: false,
      postMessage(m) {
        if (m.op === "abort") { ac?.abort(Object.assign(new Error("the copy was stopped"), { name: "AbortError" })); return; }
        ac = new AbortController();
        const post = (x) => queueMicrotask(() => w.onmessage?.({ data: structuredClone(x) }));
        copyLayer(structuredClone(m), { getDirectory: async () => opfs.root, fetch, post, signal: ac.signal, direct })
          .then(() => post({ op: "done", layer: m.layer }), (e) => post({ op: "error", layer: m.layer, name: e?.name, message: String(e?.message || e), wrote: !!e?.wrote }));
      },
      terminate() { this.terminated = true; },
    };
    made.push(w);
    return w;
  };
  return { make, made };
}
const memManifest = () => {
  let m = null; const log = [];
  return { log, manifest: async () => (m ? JSON.parse(m) : null), saveManifest: async (x) => { m = JSON.stringify(x); log.push(JSON.parse(m)); } };
};
const DIR = ["pooled-experts", "k"];

Deno.test("pool: every layer file is byte for byte what ingestExperts writes; the redirect is followed once per worker", async () => {
  const m = synthModel(), lay = expertLayout(m.G, [0, 1, 2]);
  const opfs = fakeOpfs(), { f, st } = fakeFetch(m.file), W = fakeWorkers(opfs, f), man = memManifest();
  const prog = [];
  const r = await ingestExpertsPool({ layout: lay, urls: ["model"], dir: DIR, store: man, workers: 2, chunkExperts: 3, readAhead: 2, make: W.make,
    onProgress: (p) => prog.push(p.doneBytes) });
  assert(r.wroteBytes === lay.totalBytes && r.skippedBytes === 0, `wrote ${r.wroteBytes} of ${lay.totalBytes}`);
  for (const l of [0, 1, 2]) {
    const L = lay.layers[l];
    assert(same(opfs.files.get(`${DIR.join("/")}/${L.file}`), await wantFile(m, L)), `layer ${l}'s file`);
  }
  const last = man.log.at(-1);
  assert([0, 1, 2].every((l) => last.layers[l].done === 8), "the manifest has every expert");
  assert(prog.at(-1) === lay.totalBytes && prog.every((x, i) => i === 0 || x >= prog[i - 1]), "progress climbs to the total");
  assert(W.made.length === 2 && W.made.every((w) => w.terminated), "two workers, both terminated");
  assert(opfs.open.size === 0, "every file closed");
  // each worker asks the model URL once (its other reads wait for that answer), then the CDN, from layer to layer
  assert(st.redirects === 2 && st.direct === st.reqs - 2, `${st.redirects} redirects, ${st.direct} direct requests`);

  // the same through ingestExperts on a store over the fake OPFS: the same bytes
  const opfs2 = fakeOpfs();
  for (const l of [0, 1, 2]) {
    const L = lay.layers[l];
    const h = await (await (await (await opfs2.root.getDirectoryHandle("a")).getFileHandle(L.file, { create: true }))).createSyncAccessHandle();
    const store = { manifest: async () => null, saveManifest: async () => {}, open: async () => {}, write: async (_l, at, b) => { h.write(b, { at }); }, flush: async () => {}, close: async () => h.close() };
    await ingestExperts({ layout: expertLayout(m.G, [l]), readAt: m.readAt, store, chunkExperts: 5 });
    assert(same(opfs2.files.get(`a/${L.file}`), opfs.files.get(`${DIR.join("/")}/${L.file}`)), `layer ${l}: pool == ingestExperts`);
  }
});

Deno.test("pool: resumes each layer from its checkpoint (junk past it is truncated); a finished copy runs no worker", async () => {
  const m = synthModel(), lay = expertLayout(m.G, [0, 1]);
  const opfs = fakeOpfs(), { f } = fakeFetch(m.file), man = memManifest();
  await man.saveManifest({ version: lay.version, key: lay.key, layers: { 0: { nExp: 8, recBytes: lay.layers[0].recBytes, done: 4 }, 1: { nExp: 8, recBytes: lay.layers[1].recBytes, done: 8 } } });
  const want0 = await wantFile(m, lay.layers[0]), want1 = await wantFile(m, lay.layers[1]);
  const junk = want0.slice(); junk.fill(7, 4 * lay.layers[0].recBytes);
  opfs.files.set(`${DIR.join("/")}/${lay.layers[0].file}`, junk);
  opfs.files.set(`${DIR.join("/")}/${lay.layers[1].file}`, want1.slice());
  const W = fakeWorkers(opfs, f);
  const r = await ingestExpertsPool({ layout: lay, urls: ["model"], dir: DIR, store: man, workers: 4, chunkExperts: 3, make: W.make });
  assert(r.skippedBytes === 12 * lay.layers[0].recBytes && r.wroteBytes === 4 * lay.layers[0].recBytes, `skipped ${r.skippedBytes}, wrote ${r.wroteBytes}`);
  assert(W.made.length === 1, `one layer left: one worker (${W.made.length})`);
  assert(same(opfs.files.get(`${DIR.join("/")}/${lay.layers[0].file}`), want0), "layer 0 resumed to the right bytes");
  const W2 = fakeWorkers(opfs, f);
  const r2 = await ingestExpertsPool({ layout: lay, urls: ["model"], dir: DIR, store: man, workers: 4, make: W2.make });
  assert(r2.wroteBytes === 0 && W2.made.length === 0, "nothing left: no worker");
});

Deno.test("pool: a failed read stops every worker, closes every file, and the manifest never says more than is on disk", async () => {
  const m = synthModel({ nL: 3, nExp: 16 }), lay = expertLayout(m.G, [0, 1, 2]);
  const bad = lay.layers[1].parts.down.byteOffset + 10 * 640;   // inside layer 1's down experts, past its first checkpoint
  const opfs = fakeOpfs(), { f } = fakeFetch(m.file, { failAt: bad }), W = fakeWorkers(opfs, f), man = memManifest();
  let err = null;
  try {
    await ingestExpertsPool({ layout: lay, urls: ["model"], dir: DIR, store: man, workers: 3, chunkExperts: 2, readAhead: 1, checkpointBytes: 2 * lay.layers[0].recBytes, make: W.make });
  } catch (e) { err = e; }
  assert(err && /500/.test(err.message) && err.wrote === true, `the read's error, wrote: ${err?.message} ${err?.wrote}`);
  assert(opfs.open.size === 0, `every file closed (${[...opfs.open]})`);
  const last = man.log.at(-1);
  for (const l of [0, 1, 2]) {
    const L = lay.layers[l], done = last?.layers[l]?.done || 0, f2 = opfs.files.get(`${DIR.join("/")}/${L.file}`) || new Uint8Array(0);
    assert(f2.length >= done * L.recBytes, `layer ${l}: manifest ${done} experts, file ${f2.length} B`);
    assert(same(f2.subarray(0, done * L.recBytes), (await wantFile(m, L)).subarray(0, done * L.recBytes)), `layer ${l}: the checkpointed experts are right`);
  }
  assert(last.layers[1].done < 16, "layer 1 is not marked done");
});

Deno.test("pool: a stop aborts the reads and closes the files; a pool that cannot start wrote nothing", async () => {
  const m = synthModel({ nExp: 32 }), lay = expertLayout(m.G, [0, 1]);
  const opfs = fakeOpfs(), { f } = fakeFetch(m.file), W = fakeWorkers(opfs, f), man = memManifest();
  const ac = new AbortController();
  let n = 0, err = null;
  try {
    await ingestExpertsPool({ layout: lay, urls: ["model"], dir: DIR, store: man, workers: 2, chunkExperts: 2, make: W.make, signal: ac.signal,
      onProgress: () => { if (++n === 3) ac.abort(Object.assign(new Error("Stop"), { name: "AbortError" })); } });
  } catch (e) { err = e; }
  assert(err?.name === "AbortError", `stopped: ${err?.name} ${err?.message}`);
  assert(opfs.open.size === 0, "every file closed");
  let e2 = null;
  try { await ingestExpertsPool({ layout: lay, urls: ["model"], dir: DIR, store: memManifest(), workers: 2, make: () => { throw new Error("no module workers"); } }); }
  catch (e) { e2 = e; }
  assert(e2 && e2.wrote === false, `could not start: wrote false (${e2?.wrote})`);
});

Deno.test("pool: out of room is ingestQuotaError's error, counting what was saved", async () => {
  const m = synthModel(), lay = expertLayout(m.G, [0, 1]);
  const opfs = fakeOpfs(), { f } = fakeFetch(m.file), man = memManifest();
  let writes = 0;
  const root0 = opfs.root;
  const quotaRoot = {
    async getDirectoryHandle(n) {
      const d = await root0.getDirectoryHandle(n);
      const wrap = (dd) => ({
        getDirectoryHandle: async (x) => wrap(await dd.getDirectoryHandle(x)),
        getFileHandle: async (x, o) => {
          const fh = await dd.getFileHandle(x, o);
          return { createSyncAccessHandle: async () => { const h = await fh.createSyncAccessHandle(); const w = h.write; h.write = (b, a) => { if (++writes > 5) throw Object.assign(new Error("quota"), { name: "QuotaExceededError" }); return w(b, a); }; return h; } };
        },
      });
      return wrap(d);
    },
  };
  const W = fakeWorkers({ root: quotaRoot }, f);
  let err = null;
  try { await ingestExpertsPool({ layout: lay, urls: ["model"], dir: DIR, store: man, workers: 1, chunkExperts: 1, checkpointBytes: lay.layers[0].recBytes * 2, make: W.make, estimate: async () => ({ quota: 1, usage: 1 }) }); }
  catch (e) { err = e; }
  assert(err?.name === "QuotaExceededError" && err.totalBytes === lay.totalBytes && err.savedBytes === 4 * lay.layers[0].recBytes, `quota: ${err?.name} saved ${err?.savedBytes}: ${err?.message}`);
  assert(opfs.open.size === 0, "every file closed");
});
