// engine/ngram_copy_worker.js (the n-gram table's copy off the main thread) and room/flashdisk.js's use of it:
// ingestNgram's worker path, its fallback to the main thread, ngramCheck.
//   deno test --no-check --allow-read tests/unit/ngram_copy_worker_test.js
import { copyHeads, copyHeadsInWorker } from "../../engine/ngram_copy_worker.js";
import { ingestNgram, ngramCheck, ngramDir } from "../../room/flashdisk.js";
import { ngramFileName } from "../../engine/ngram.js";
import { FakeDir } from "../helpers/fake-opfs.js";

const eq = (a, b, m = "") => { if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(`${m} ${JSON.stringify(a)} != ${JSON.stringify(b)}`); };
const ok = (c, m = "assertion failed") => { if (!c) throw new Error(m); };
const tick = (ms = 0) => new Promise((r) => setTimeout(r, ms));

// a 4-head table, 8 rows of 8 bytes per head, row r of head h filled with h * 16 + r (as flashdisk_test.js)
const cfg = { nHeads: 4, vocab: [8, 8, 8, 8], rowBytes: 8, offsets: [0, 8, 16, 24], nRows: 32, shard: 0, byteOffset: 64, ggmlType: 2 };
const table = new Uint8Array(64 + 32 * 8);
for (let r = 0; r < 32; r++) table.fill((r >> 3) * 16 + (r & 7), 64 + r * 8, 64 + r * 8 + 8);
const URLS = ["https://example.test/q38flash-00001.gguf"];

// a fetch answering range requests from `table` (206), recording each init; gate(): a promise each read waits on
function server({ gate = null, failAt = -1 } = {}) {
  const st = { inits: [], reads: 0 };
  const fetch = async (url, init) => {
    st.inits.push(init); st.reads++;
    if (gate) await gate();
    if (init?.signal?.aborted) throw Object.assign(new Error("aborted"), { name: "AbortError" });
    const [a, b] = /bytes=(\d+)-(\d+)/.exec(init.headers.Range).slice(1).map(Number);
    if (failAt >= 0 && a >= 64 + failAt * 64 && a < 64 + (failAt + 1) * 64) throw new TypeError("network down");
    return new Response(table.slice(a, b + 1), { status: 206, headers: { "content-range": `bytes ${a}-${b}/${table.length}` } });
  };
  return { fetch, st };
}
// a small OPFS with sync access handles (the worker's API): write(bytes, { at }), truncate, flush, close
class SyncFile {
  constructor(name) { this.kind = "file"; this.name = name; this.data = new Uint8Array(0); this.locked = false; }
  async getFile() { return { size: this.data.byteLength }; }
  async createSyncAccessHandle() {
    if (this.locked) throw Object.assign(new Error("locked"), { name: "NoModificationAllowedError" });
    this.locked = true;
    const f = this;
    return {
      truncate(n) { f.data = f.data.slice(0, n); },
      write(b, { at = 0 } = {}) { const end = at + b.byteLength; if (end > f.data.length) { const g = new Uint8Array(end); g.set(f.data); f.data = g; } f.data.set(b, at); return b.byteLength; },
      flush() {}, close() { f.locked = false; },
    };
  }
}
class SyncDir {
  constructor() { this.kind = "directory"; this.children = new Map(); }
  async getDirectoryHandle(n, { create = false } = {}) {
    let h = this.children.get(n);
    if (!h) { if (!create) throw Object.assign(new Error(n), { name: "NotFoundError" }); h = new SyncDir(); this.children.set(n, h); }
    return h;
  }
  async getFileHandle(n, { create = false } = {}) {
    let h = this.children.get(n);
    if (!h) { if (!create) throw Object.assign(new Error(n), { name: "NotFoundError" }); h = new SyncFile(n); this.children.set(n, h); }
    return h;
  }
  async removeEntry(n) { this.children.delete(n); }
  async *keys() { for (const k of [...this.children.keys()]) yield k; }
  async *entries() { for (const e of [...this.children.entries()]) yield e; }
}
const at = async (root, dir) => { let d = root; for (const p of dir) d = await d.getDirectoryHandle(p); return d; };
const headData = async (root, dir, h) => (await (await at(root, dir)).getFileHandle(ngramFileName(h))).data;
const headOk = (d, h) => d.length === 64 && d[0] === h * 16 && d[63] === h * 16 + 7;
const plain = { shard: 0, byteOffset: 64, offsets: cfg.offsets, vocab: cfg.vocab, rowBytes: 8 };

Deno.test("copyHeads: each head written in order through a sync handle, fetched at low priority past the HTTP cache", async () => {
  const root = new SyncDir(), { fetch, st } = server(), posts = [], dir = ["models", "q38flash-ngram", "s1"];
  const job = copyHeads({ cfg: plain, urls: URLS, dir, heads: [0, 1, 2, 3], parallel: 2 }, { getDirectory: async () => root, fetch, post: (m) => posts.push(m) });
  await job.done;
  for (let h = 0; h < 4; h++) ok(headOk(await headData(root, dir, h), h), `head ${h}`);
  ok(st.inits.every((i) => i.cache === "no-store" && i.priority === "low" && i.signal), JSON.stringify(st.inits[0]));
  eq(posts.filter((m) => m.op === "head").map((m) => m.h).sort(), [0, 1, 2, 3]);
  eq(posts.filter((m) => m.op === "progress").at(-1).done, 256);
  ok([...(await at(root, dir)).children.values()].every((f) => !f.locked), "every handle closed");
  ok(job.wrote, "wrote");
});

Deno.test("copyHeads: one head at a time until parallel(2); a stop closes every file, leaves the head short, says AbortError", async () => {
  const root = new SyncDir(), dir = ["d"];
  let open = 0, peak = 0;
  const waits = [];
  const { fetch } = server({ gate: () => new Promise((r) => waits.push(r)) });
  const env = { getDirectory: async () => root, fetch: (u, i) => { open++; peak = Math.max(peak, open); return fetch(u, i).finally(() => open--); }, post: () => {} };
  const job = copyHeads({ cfg: { ...plain, vocab: [8, 8, 8, 8] }, urls: URLS, dir, heads: [0, 1, 2, 3], parallel: 1 }, env);
  await tick(5);
  eq(waits.length, 1, "one head's read");
  waits.splice(0).forEach((r) => r());   // head 0 done
  await tick(150);
  ok(peak === 1 && waits.length === 1, `one read at a time (${peak}, ${waits.length})`);
  job.parallel(2);
  await tick(150);
  ok(waits.length === 2, `two heads at once after parallel(2): ${waits.length}`);
  job.abort();
  waits.splice(0).forEach((r) => r());
  let err = null;
  try { await job.done; } catch (e) { err = e; }
  eq(err?.name, "AbortError");
  const d = await at(root, dir);
  ok([...d.children.values()].every((f) => !f.locked), "every handle closed");
  const sizes = [...d.children.values()].map((f) => f.data.length);
  eq(sizes, [64, 0, 0], "head 0 whole; heads 1 and 2 stopped short (copied again next time); head 3 never started");
});

// a stand-in Worker running copyHeads in this thread, as the real one does in its own
function inProcessWorker(root, fetch, log = []) {
  return () => {
    let job = null;
    const w = {
      terminated: false, onmessage: null, onerror: null,
      postMessage(m) {
        log.push(m.op);
        if (m.op === "copy") {
          job = copyHeads(m, { getDirectory: async () => root, fetch, post: (x) => queueMicrotask(() => w.onmessage?.({ data: x })) });
          job.done.then(() => w.onmessage?.({ data: { op: "done" } }), (e) => w.onmessage?.({ data: { op: "error", name: e.name, message: e.message, wrote: !!e.wrote } }));
        } else if (m.op === "parallel") job?.parallel(m.n);
        else if (m.op === "abort") job?.abort();
      },
      terminate() { w.terminated = true; },
    };
    return w;
  };
}

Deno.test("copyHeadsInWorker: resolves with the bytes; progress climbs; the worker is terminated after", async () => {
  const root = new SyncDir(), { fetch } = server(), log = [], seen = [];
  let made = null;
  const make = inProcessWorker(root, fetch, log);
  const n = await copyHeadsInWorker(cfg, { urls: URLS, dir: ["d"], heads: [1, 3], make: (u) => (made = make(u)), onProgress: (d, t) => seen.push([d, t]) });
  eq(n, 128);
  ok(headOk(await headData(root, ["d"], 1), 1) && headOk(await headData(root, ["d"], 3), 3), "heads 1 and 3");
  ok(seen.length && seen.at(-1)[0] === 128 && seen.every(([, t]) => t === 128), JSON.stringify(seen));
  ok(made.terminated, "terminated");
});

Deno.test("copyHeadsInWorker: a worker that fails before writing (no sync handles) rejects with wrote: false; one that wrote, true", async () => {
  const noSync = { onmessage: null, postMessage(m) { if (m.op === "copy") queueMicrotask(() => this.onmessage({ data: { op: "error", name: "TypeError", message: "createSyncAccessHandle is not a function", wrote: false } })); }, terminate() {} };
  let err = null;
  try { await copyHeadsInWorker(cfg, { urls: URLS, dir: ["d"], heads: [0], make: () => noSync }); } catch (e) { err = e; }
  ok(err && err.wrote === false && /createSyncAccessHandle/.test(err.message), String(err));
  err = null;
  try { await copyHeadsInWorker(cfg, { urls: URLS, dir: ["d"], heads: [0], make: () => { throw new Error("no module workers"); } }); } catch (e) { err = e; }
  ok(err && err.wrote === false, String(err));
  const root = new SyncDir(), { fetch } = server({ failAt: 2 });
  err = null;
  try { await copyHeadsInWorker(cfg, { urls: URLS, dir: ["d"], heads: [0, 1, 2, 3], parallel: 1, make: inProcessWorker(root, fetch) }); } catch (e) { err = e; }
  ok(err && err.wrote === true && /network down/.test(err.message), `${err} ${err?.wrote}`);
});

Deno.test("copyHeadsInWorker: a stop asks the worker to stop, then terminates it (at once when it answers, else after the wait)", async () => {
  const root = new SyncDir(), waits = [], { fetch } = server({ gate: () => new Promise((r) => waits.push(r)) }), log = [];
  const ac = new AbortController();
  let w = null;
  const make = inProcessWorker(root, fetch, log);
  const p = copyHeadsInWorker(cfg, { urls: URLS, dir: ["d"], heads: [0, 1], signal: ac.signal, make: (u) => (w = make(u)) });
  await tick(5);
  ac.abort(new Error("the start was stopped"));
  waits.splice(0).forEach((r) => r());
  let err = null;
  try { await p; } catch (e) { err = e; }
  ok(/stopped/.test(err?.message) && log.includes("abort") && w.terminated, `${err} ${log}`);
  ok([...(await at(root, ["d"])).children.values()].every((f) => !f.locked), "its files closed before it was let go");
  // a worker that never answers: terminated after closeWaitMs
  const mute = { onmessage: null, terminated: false, postMessage() {}, terminate() { this.terminated = true; } };
  const ac2 = new AbortController();
  const p2 = copyHeadsInWorker(cfg, { urls: URLS, dir: ["d"], heads: [0], signal: ac2.signal, make: () => mute, closeWaitMs: 30 });
  ac2.abort(new Error("stop"));
  err = null;
  try { await p2; } catch (e) { err = e; }
  ok(err?.message === "stop" && mute.terminated, String(err));
});

const storageOf = (root) => ({ getDirectory: async () => root, estimate: async () => ({ quota: 2 ** 40, usage: 0 }) });
const readAtOf = (fetch) => async (_f, off, len, opt = {}) => {
  const r = await fetch(URLS[0], { headers: { Range: `bytes=${off}-${off + len - 1}` }, ...opt });
  return new Uint8Array(await r.arrayBuffer());
};

Deno.test("ingestNgram worker: copies in the worker; one that can't start falls back to the main thread after mainAfter", async () => {
  const root = new SyncDir(), { fetch } = server(), dir = ngramDir("bbbb0001"), logs = [];
  eq(await ingestNgram(cfg, readAtOf(fetch), { dir, storage: storageOf(root), worker: { urls: URLS, make: inProcessWorker(root, fetch) }, log: (s) => logs.push(s) }), 256);
  for (let h = 0; h < 4; h++) ok(headOk(await headData(root, dir, h), h), `head ${h}`);
  ok(logs.some((s) => /in a worker/.test(s)), logs.join("\n"));
  // the fallback: the main thread's writables (FakeDir), only once mainAfter settles
  const root2 = new FakeDir(), states = [], noSync = () => ({ onmessage: null, postMessage(m) { if (m.op === "copy") queueMicrotask(() => this.onmessage({ data: { op: "error", name: "TypeError", message: "no sync handles", wrote: false } })); }, terminate() {} });
  let release, reads = 0;
  const after = new Promise((r) => { release = r; });
  const counted = async (...a) => { reads++; return readAtOf(server().fetch)(...a); };
  const p = ingestNgram(cfg, counted, { dir, storage: storageOf(root2), worker: { urls: URLS, make: noSync }, mainAfter: after, onState: (s) => states.push(s), log: (s) => logs.push(s) });
  await tick(20);
  eq([reads, states], [0, ["wait"]], "waits for the experts' copy");
  release();
  eq(await p, 256);
  eq(states, ["wait", "run"]);
  for (let h = 0; h < 4; h++) ok(headOk((await (await at(root2, dir)).getFileHandle(ngramFileName(h))).data, h), `fallback head ${h}`);
});

Deno.test("ingestNgram: a stop while waiting or copying throws its reason and copies nothing more", async () => {
  const root = new FakeDir(), dir = ngramDir("bbbb0002"), ac = new AbortController();
  const noSync = () => ({ onmessage: null, postMessage(m) { if (m.op === "copy") queueMicrotask(() => this.onmessage({ data: { op: "error", name: "TypeError", message: "x", wrote: false } })); }, terminate() {} });
  let reads = 0;
  const p = ingestNgram(cfg, async () => { reads++; return new Uint8Array(64); }, { dir, storage: storageOf(root), worker: { urls: URLS, make: noSync }, mainAfter: new Promise(() => {}), signal: ac.signal });
  await tick(10);
  ac.abort(new Error("stopped"));
  let err = null;
  try { await p; } catch (e) { err = e; }
  eq([err?.message, reads], ["stopped", 0]);
  // a main-thread copy stopped part way: the error is the stop's, no head half there
  const ac2 = new AbortController();
  const slow = async (_f, off, len, { signal } = {}) => { await tick(5); if (signal?.aborted) throw new Error("aborted read"); return table.slice(off, off + len); };
  const p2 = ingestNgram(cfg, slow, { dir, storage: storageOf(root), parallel: 1, signal: ac2.signal });
  await tick(7);
  ac2.abort(new Error("stopped 2"));
  err = null;
  try { await p2; } catch (e) { err = e; }
  eq(err?.message, "stopped 2");
  const d = await at(root, dir), sizes = [];
  for (let h = 0; h < 4; h++) { try { sizes.push((await d.getFileHandle(ngramFileName(h))).data.length); } catch { sizes.push(0); } }
  ok(sizes.every((n) => n === 0 || n === 64) && sizes.some((n) => n === 0), `whole heads only: ${sizes}`);
});

Deno.test("ngramCheck: what is there, without making a directory", async () => {
  const root = new FakeDir(), { fetch } = server(), readAt = readAtOf(fetch);
  const c0 = await ngramCheck(cfg, readAt, { storage: storageOf(root), url: URLS[0] });
  eq([c0.missing, c0.presentBytes, c0.totalBytes], [[0, 1, 2, 3], 0, 256]);
  eq(root.children.size, 0, "nothing made");
  await ingestNgram(cfg, readAt, { dir: c0.dir, storage: storageOf(root) });
  const d = await at(root, c0.dir);
  await d.removeEntry(ngramFileName(2));
  const c1 = await ngramCheck(cfg, readAt, { storage: storageOf(root), url: URLS[0] });
  eq([c1.dir, c1.missing, c1.presentBytes], [c0.dir, [2], 192]);
});
