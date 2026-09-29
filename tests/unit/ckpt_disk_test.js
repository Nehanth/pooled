// Room checkpoints on disk (room/ckpt-store.js, issue #69): the file format and its checks, the
// store over an in-memory stand-in for OPFS (quota errors included), and the room.js functions that
// use it (cut out of room.js by room_src.js), host and worker: a worker that reloads reads its part
// back and the host loads it; a host that reloads gets its checkpoint index back; a device missing
// its copy says so in its ai-ready and the host forgets that checkpoint (ckptPrune).
import { CKPT_FORMAT, CkptStore, CkptFormatError, encodeHeader, decodeHeader, decodeCkpt, headerLength, mismatch, parseName, safeRoom, sigHash }
  from "../../room/ckpt-store.js";
import { roomFns } from "./room_src.js";
import { PrefixIndex } from "../../harness/prefix.js";
import { DROP_ALL } from "../../room/transport.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const throwsIn = (f, re) => { try { f(); } catch (e) { if (re && !re.test(e.message)) throw new Error("wrong error: " + e.message); return e; } throw new Error("expected a throw"); };
const flush = (disk) => disk._run(() => {});   // wait for the background writes queued so far

// a whole file the way the store writes it
function file(h, parts) {
  const head = encodeHeader({ ...h, sizes: parts.map((p) => p.byteLength) });
  const out = new Uint8Array(head.length + parts.reduce((s, p) => s + p.byteLength, 0));
  out.set(head); let o = head.length;
  for (const p of parts) { out.set(new Uint8Array(p), o); o += p.byteLength; }
  return out.buffer;
}
const H = { room: "ABC", model: "m", sig: { lo: 0, hi: 8 }, slot: 3, pos: 5, meta: { ids: [1, 2, 3, 4, 5] }, t: 1 };

// ---------------------------------------------------------------------------------------------
// format

Deno.test("ckpt format: header and parts round trip", () => {
  const parts = [Uint8Array.of(1, 2, 3).buffer, new ArrayBuffer(0), Uint8Array.of(9).buffer];
  const { header, parts: back } = decodeCkpt(file(H, parts));
  eq([header.f, header.room, header.model, header.sig, header.slot, header.pos, header.meta, header.sizes], [CKPT_FORMAT, "ABC", "m", { lo: 0, hi: 8 }, 3, 5, { ids: [1, 2, 3, 4, 5] }, [3, 0, 1]]);
  eq(back.map((p) => [...new Uint8Array(p)]), [[1, 2, 3], [], [9]]);
});

Deno.test("ckpt format: damaged or foreign files are refused with CkptFormatError", () => {
  const good = file(H, [Uint8Array.of(1, 2).buffer]);
  const bad = (buf, re) => ok(throwsIn(() => decodeCkpt(buf), re) instanceof CkptFormatError);
  bad(new ArrayBuffer(4), /too short/);
  const u = new Uint8Array(good.slice(0)); u[0] ^= 1; bad(u.buffer, /not a checkpoint/);
  bad(good.slice(0, 20), /truncated/);
  bad(good.slice(0, good.byteLength - 1), /body is/);
  const long = new Uint8Array(good.byteLength + 1); long.set(new Uint8Array(good)); bad(long.buffer, /body is/);
  // a file from another format version (a future or older build) reads as foreign, not as garbage
  const other = new TextEncoder().encode(JSON.stringify({ f: CKPT_FORMAT + 1, slot: 1, pos: 0, sizes: [] }));
  const o = new Uint8Array(8 + other.length); new DataView(o.buffer).setUint32(0, 0x504f434b); new DataView(o.buffer).setUint32(4, other.length); o.set(other, 8);
  bad(o.buffer, /format/);
  // JSON that is not a header
  const js = new TextEncoder().encode("{nope"), j = new Uint8Array(8 + js.length);
  new DataView(j.buffer).setUint32(0, 0x504f434b); new DataView(j.buffer).setUint32(4, js.length); j.set(js, 8);
  bad(j.buffer, /not JSON/);
  eq(headerLength(good.slice(0, 8)), decodeHeader(good) && encodeHeader({ ...H, sizes: [2] }).length);
});

Deno.test("ckpt format: mismatch names what differs (room, model, layers or KV format)", () => {
  const w = { room: "ABC", model: "m", sig: { lo: 0, hi: 8 } };
  eq(mismatch(H, w), null);
  eq(mismatch(H, { ...w, room: "XYZ" }), "room");
  eq(mismatch(H, { ...w, model: "n" }), "model");
  eq(mismatch(H, { ...w, sig: { lo: 0, hi: 9 } }), "layers");
  eq(mismatch({ ...H, sig: { lo: 0, hi: 8, kvQ8: true } }, w), "layers");
});

Deno.test("ckpt names: room codes are made file-safe; names parse back", async () => {
  eq(safeRoom("AB-12_x"), "AB-12_x");
  eq(safeRoom("a/../b"), "a____b");
  eq(safeRoom(""), "_"); eq(safeRoom(null), "_");
  const h = await sigHash("m", { lo: 0 });
  ok(/^[0-9a-f]{16}$/.test(h));
  ok(h !== await sigHash("m", { lo: 1 }) && h !== await sigHash("n", { lo: 0 }));
  eq(parseName(`ABC.${h}.12.ckpt`), { room: "ABC", hash: h, slot: 12 });
  eq(parseName(`ABC.${h}.12.ckpt.tmp`), null);
  eq(parseName("other.bin"), null);
});

// ---------------------------------------------------------------------------------------------
// the store over an in-memory OPFS

class MemFile {
  constructor(bytes, t) { this.bytes = bytes; this.size = bytes.length; this.lastModified = t; }
  async arrayBuffer() { return this.bytes.slice().buffer; }
  slice(a, b) { const s = this.bytes.slice(a, b); return { arrayBuffer: async () => s.buffer }; }
}
// quota: total bytes the directory may hold; a write past it fails like OPFS (QuotaExceededError)
class MemDir {
  constructor({ quota = Infinity } = {}) { this.files = new Map(); this.clock = 1000; this.quota = quota; }
  used() { let n = 0; for (const f of this.files.values()) n += f.bytes.length; return n; }
  async getDirectoryHandle() { return this; }
  async getFileHandle(name, { create = false } = {}) {
    if (!this.files.has(name)) {
      if (!create) { const e = new Error("not found: " + name); e.name = "NotFoundError"; throw e; }
      this.files.set(name, { bytes: new Uint8Array(0), t: ++this.clock });
    }
    const dir = this;
    const h = {
      kind: "file", name,
      async getFile() { const f = dir.files.get(h.name); return new MemFile(f.bytes, f.t); },
      async createWritable() {
        const chunks = [];
        return {
          async write(c) { chunks.push(new Uint8Array(c.buffer ? c.buffer.slice(c.byteOffset, c.byteOffset + c.byteLength) : c)); },
          async close() {
            const n = chunks.reduce((s, c) => s + c.length, 0);
            if (dir.used() - dir.files.get(h.name).bytes.length + n > dir.quota) {
              const e = new Error("The operation failed because it would cause the application to exceed its storage quota."); e.name = "QuotaExceededError"; throw e;
            }
            const all = new Uint8Array(n); let o = 0; for (const c of chunks) { all.set(c, o); o += c.length; }
            dir.files.set(h.name, { bytes: all, t: ++dir.clock });
          },
          async abort() {},
        };
      },
      async move(to) { const f = dir.files.get(h.name); dir.files.delete(h.name); dir.files.set(to, f); h.name = to; },
    };
    return h;
  }
  async removeEntry(name) { if (!this.files.delete(name)) throw new Error("not found"); }
  async *entries() { for (const name of [...this.files.keys()]) yield [name, await this.getFileHandle(name)]; }
}
const store = (dirOpts, opts) => { const dir = new MemDir(dirOpts); const s = new CkptStore({ root: async () => dir, ...opts }); s.mem = dir; return s; };
const st = (sig, pos, ...bytes) => ({ sig, pos, parts: bytes.map((n, i) => Uint8Array.from({ length: n }, (_, j) => (i * 7 + j) & 255).buffer) });
const W = (slot, over = {}) => ({ room: "ABC", model: "m", sig: { lo: 0, hi: 8 }, slot, ...over });

Deno.test("CkptStore: put / get / list round trip; no temp file left", async () => {
  const s = store();
  ok(await s.put(W(1), st({ lo: 0, hi: 8 }, 4, 16, 0, 5), { ids: [1, 2, 3, 4] }));
  const g = await s.get(W(1));
  eq([g.pos, g.meta, g.parts.map((p) => p.byteLength)], [4, { ids: [1, 2, 3, 4] }, [16, 0, 5]]);
  eq((await s.list(W(0))).map((c) => [c.slot, c.pos, c.meta.ids]), [[1, 4, [1, 2, 3, 4]]]);
  ok([...s.mem.files.keys()].every((n) => n.endsWith(".ckpt")), [...s.mem.files.keys()].join());
  eq(await s.get(W(2)), null, "a slot never saved");
});

Deno.test("CkptStore: a copy for other layers, another model or another room is not ours", async () => {
  const s = store();
  await s.put(W(1), st({ lo: 0, hi: 8 }, 4, 8));
  eq(await s.get(W(1, { sig: { lo: 8, hi: 16 } })), null, "other layers");
  eq(await s.get(W(1, { model: "n" })), null, "other model");
  eq(await s.get(W(1, { room: "XYZ" })), null, "other room");
  eq(await s.list(W(0, { sig: { lo: 0, hi: 8, kvQ8: true } })), [], "another KV format");
  ok(await s.get(W(1)), "still there for its own layers");
});

Deno.test("CkptStore: a new copy of a slot replaces the old one, whatever layers it was for", async () => {
  const s = store();
  await s.put(W(1, { sig: { lo: 0, hi: 8 } }), st({ lo: 0, hi: 8 }, 4, 8));
  await s.put(W(1, { sig: { lo: 0, hi: 6 } }), st({ lo: 0, hi: 6 }, 9, 8));
  eq(await s.get(W(1)), null, "the copy for the old split is gone");
  eq((await s.get(W(1, { sig: { lo: 0, hi: 6 } }))).pos, 9);
  eq(s.mem.files.size, 1);
});

Deno.test("CkptStore: a state that cannot be read (slot dropped meanwhile) leaves the slot missing, not stale", async () => {
  const s = store();
  await s.put(W(1), st({ lo: 0, hi: 8 }, 4, 8));
  ok(!(await s.put(W(1), async () => { throw new Error("no saved slot 1"); })));
  eq(await s.get(W(1)), null);
  eq(s.failures, 1);
});

Deno.test("CkptStore: list is newest first, and damaged files are removed", async () => {
  const s = store();
  await s.put(W(1), st({ lo: 0, hi: 8 }, 1, 4));
  await new Promise((r) => setTimeout(r, 3));
  await s.put(W(2), st({ lo: 0, hi: 8 }, 2, 4));
  eq((await s.list(W(0))).map((c) => c.slot), [2, 1]);
  const name = [...s.mem.files.keys()].find((n) => n.endsWith(".1.ckpt"));
  s.mem.files.get(name).bytes = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  eq((await s.list(W(0))).map((c) => c.slot), [2]);
  ok(!s.mem.files.has(name), "the damaged file is gone");
  // a truncated body: the header reads, the get refuses it and removes it
  const n2 = [...s.mem.files.keys()][0];
  s.mem.files.get(n2).bytes = s.mem.files.get(n2).bytes.slice(0, -1);
  eq(await s.get(W(2)), null);
  eq(s.mem.files.size, 0);
});

Deno.test("CkptStore: drop one slot, a list, or all of a room", async () => {
  const s = store();
  for (const k of [1, 2, 3]) await s.put(W(k), st({ lo: 0, hi: 8 }, k, 4));
  await s.put(W(1, { room: "XYZ" }), st({ lo: 0, hi: 8 }, 1, 4));
  await s.drop("ABC", 2);
  eq((await s.list(W(0))).map((c) => c.slot).sort(), [1, 3]);
  await s.drop("ABC", [1, 7]);
  eq((await s.list(W(0))).map((c) => c.slot), [3]);
  await s.drop("ABC", "all");
  eq(await s.list(W(0)), []);
  eq((await s.list(W(0, { room: "XYZ" }))).length, 1, "other rooms are untouched");
});

Deno.test("CkptStore: operations run in call order (a drop never overtakes the write before it)", async () => {
  const s = store();
  let release;
  const slow = new Promise((r) => { release = r; });
  const p = s.put(W(1), async () => { await slow; return st({ lo: 0, hi: 8 }, 1, 4); });
  const d = s.drop("ABC", 1);
  release();
  await p; await d;
  eq(await s.list(W(0)), [], "the drop came after the write");
});

Deno.test("CkptStore: out of quota, the oldest copies of other rooms go first and the write is tried again", async () => {
  // each file is ~150 bytes of header + 400; the quota holds two
  const s = store({ quota: 1200 });
  ok(await s.put(W(1, { room: "OLD" }), st({ lo: 0, hi: 8 }, 1, 400)));
  ok(await s.put(W(1), st({ lo: 0, hi: 8 }, 1, 400)));
  ok(await s.put(W(2), st({ lo: 0, hi: 8 }, 2, 400)), "written after dropping the other room's copy");
  eq(await s.list(W(0, { room: "OLD" })), []);
  eq((await s.list(W(0))).map((c) => c.slot).sort(), [1, 2]);
  ok(![...s.mem.files.keys()].some((n) => n.endsWith(".tmp")));
});

Deno.test("CkptStore: out of quota with nothing to drop gives up cleanly", async () => {
  const s = store({ quota: 100 });
  ok(!(await s.put(W(1), st({ lo: 0, hi: 8 }, 1, 400))));
  eq(s.mem.files.size, 0, "no partial or temp file");
  eq(s.failures, 1);
  ok(!(await s.put(W(1), st({ lo: 0, hi: 8 }, 1, 400))), "and again next time, without throwing");
});

Deno.test("CkptStore: a state bigger than the budget is not written; the budget evicts the oldest", async () => {
  const s = store({}, { budgetBytes: 1200 });
  ok(!(await s.put(W(9), st({ lo: 0, hi: 8 }, 1, 2000))));
  for (const k of [1, 2, 3]) { ok(await s.put(W(k), st({ lo: 0, hi: 8 }, k, 400))); await new Promise((r) => setTimeout(r, 2)); }
  eq((await s.list(W(0))).map((c) => c.slot), [3, 2]);
});

Deno.test("CkptStore: a temp file left by a reload mid-write is removed once stale, and counts against the budget while fresh", async () => {
  // the directory's clock is the store's clock here: a file written at t is (now - t) old
  const dir = new MemDir();
  let now = 0;
  const s = new CkptStore({ root: async () => dir, budgetBytes: 1200, tmpStaleMs: 50, now: () => now }); s.mem = dir;
  ok(await s.put(W(1), st({ lo: 0, hi: 8 }, 1, 400)));
  // another tab reloaded while writing slot 2: its <name>.ckpt.tmp stays behind
  const real = [...dir.files.keys()][0], stray = real.replace(/\.1\.ckpt$/, ".2.ckpt.tmp");
  dir.files.set(stray, { bytes: new Uint8Array(600), t: ++dir.clock });
  now = dir.clock;
  eq((await s.list(W(0))).map((c) => c.slot), [1], "a temp file is never listed as a copy");
  ok(dir.files.has(stray), "fresh: another tab may still be writing it");
  // fresh, it counts against the budget: the next write makes room by dropping the oldest (the temp too)
  ok(await s.put(W(3), st({ lo: 0, hi: 8 }, 3, 400)));
  ok(dir.used() <= 1200, `the directory holds ${dir.used()} bytes, over the 1200 budget`);
  // a stale one is removed by the next listing
  dir.files.set(stray, { bytes: new Uint8Array(600), t: ++dir.clock });
  now = dir.clock + 51;
  eq((await s.list(W(0))).map((c) => c.slot), [3]);
  ok(!dir.files.has(stray), "the stale temp file is gone");
  ok([...dir.files.keys()].every((n) => n.endsWith(".ckpt")), [...dir.files.keys()].join());
});

// ---------------------------------------------------------------------------------------------
// room.js: host and worker with disk copies

// an engine whose whole state is the list of tokens it has run (see room_ckpt_test.js), with the
// state export / import of Qwen35Engine: states only load on the same layers
class FakeEngine {
  constructor(lo = 0, hi = 8) { this.st = []; this.slots = new Map(); this.log = []; this.lo = lo; this.hi = hi; this.dims = { dim: 1 }; this.NC = 64; }
  stateSignature() { return { v: 1, lo: this.lo, hi: this.hi }; }
  reset() { this.log.push("reset"); this.st = []; }
  saveSlot(k) { this.log.push("sv" + k); this.slots.set(k, this.st.slice()); }
  loadSlot(k) { this.log.push("ld" + k); const s = this.slots.get(k); if (!s) throw new Error("no saved slot " + k); this.st = s.slice(); }
  dropSlot(k) { this.log.push("dp" + k); this.slots.delete(k); }
  dropAllSlots() { this.log.push("dpAll"); this.slots.clear(); }
  restoreDN() {}
  async exportSlot(k) { const s = this.slots.get(k); if (!s) throw new Error("no saved slot " + k); return { sig: this.stateSignature(), pos: s.length, parts: [Uint32Array.from(s).buffer] }; }
  importState(x) {
    if (JSON.stringify(x.sig) !== JSON.stringify(this.stateSignature())) throw new Error("saved state is for a different model, layer range or KV format");
    this.log.push("import"); this.st = [...new Uint32Array(x.parts[0])];
  }
  run(toks, base) { if (this.st.length !== base) throw new Error(`desync: ${this.st.length} vs ${base}`); this.st.push(...toks); }
  async runHidden(x, pos) { this.run([x[0]], pos); return Float32Array.of(x[0]); }
  async runHiddenBatch(xs, base) { this.run(Array.from(xs), base); return Float32Array.from(xs); }
}
const HOST_FNS = ["sendChain", "resetState", "ckptClear", "ckptSave", "ckptResume", "ckptWhere", "ckptPersist", "ckptForget", "ckptRestore", "ckptRejoin", "ckptPrune"];
function host({ disk = store(), engine = new FakeEngine(0, 8), chain = ["w0"], ckptMax = 2, out = [], ckptN = 0 } = {}) {
  const ai = { role: "host", model: "m", chain, pendingCtl: {}, fed: [], pos: 0, engine, ckpt: null, ckptN };
  const fns = roomFns(HOST_FNS, { ai, CKPT_MAX: ckptMax, PrefixIndex, DROP_ALL, wireStats: { lastMax: 0 }, ckptDisk: disk, roomCode: "ABC", sendHidden: (to, msg) => out.push(msg) });
  return { ai, out, disk, engine, ...fns };
}
function worker({ disk = store(), engine = new FakeEngine(8, 16), ckptMax = 2 } = {}) {
  const ai = { role: "worker", model: "m", engine, next: "host", hostId: "host", range: [8, 16] };
  const fns = roomFns(["workerFrame", "ckptWhere", "ckptPersist", "ckptForget", "ckptRestore"], {
    ai, DROP_ALL, performance, ckptDisk: disk, roomCode: "ABC", CKPT_MAX: ckptMax, ckptClear: () => {},
    unpackWire: (d) => Float32Array.from(d.x), packWire: (h) => ({ x: Array.from(h) }), badF32: () => false,
    aiStatus: () => {}, sendTo: () => {}, teleNote: () => {}, compute: { pass() {} }, sendHidden: () => {},
  });
  return { ai, disk, engine, ...fns };
}
// one prefill frame of `toks` from the host through the worker; both engines run the tokens
async function lap(h, w, toks) {
  const base = h.ai.fed.length;
  h.engine.run(toks, base);
  h.ai.fed.push(...toks); h.ai.pos = h.ai.fed.length;
  h.sendChain({ t: "ai-hidden-b", basePos: base, n: toks.length, x: toks });
  await w.workerFrame(h.out[h.out.length - 1]);
}
// an answer ends: the host saves; the save goes out with the next frame
async function answer(h, w, toks) { await lap(h, w, toks); h.ckptSave(); }

Deno.test("room disk: a worker that reloads reads its part back, and the host resumes from it", async () => {
  const h = host(), w = worker();
  await answer(h, w, [1, 2, 3]);             // answer 1 saved as slot 1 (its save still pending)
  await answer(h, w, [4, 5]);                // the frame carries sv 1: every device saves slot 1; slot 2 pending
  await flush(h.disk); await flush(w.disk);
  eq((await w.disk.list({ room: "ABC", model: "m", sig: w.engine.stateSignature() })).map((c) => [c.slot, c.pos]), [[1, 3]]);
  eq((await h.disk.list({ room: "ABC", model: "m", sig: h.engine.stateSignature() })).map((c) => [c.slot, c.meta.ids]), [[1, [1, 2, 3]]]);

  // the worker's tab reloads: a fresh engine, same layers, same disk
  const w2 = worker({ disk: w.disk });
  eq(await w2.ckptRestore(), [1]);
  eq([...w2.engine.slots.keys()], [1]);
  eq(w2.engine.st, [], "restored into a slot; the live state starts empty");
  h.ai.fed = null; h.ckptRejoin();          // what aiRejoin does
  eq(h.ai.ckpt.items.map((x) => x.key), [1], "slot 2 never reached the chain: forgotten");
  eq(h.ai.pendingCtl.sv, undefined);
  // the next question starts with answers 1 and 2: resume from slot 1, prefill the rest
  const ids = [1, 2, 3, 4, 5, 6];
  eq(h.ckptResume(ids, 0), 3, "(3 reused)");
  eq(h.ai.pendingCtl.ld, 1);
  h.engine.st = h.engine.slots.get(1).slice();
  await lap(h, w2, [4, 5, 6]);
  eq(w2.engine.st, [1, 2, 3, 4, 5, 6], "the worker continues from its restored state, bit for bit");
});

Deno.test("room disk: a host that reloads gets its checkpoint index back, and slot numbers go on", async () => {
  const h = host(), w = worker();
  await answer(h, w, [1, 2, 3]);
  await answer(h, w, [4, 5]);
  await lap(h, w, [6]);                      // slot 2 goes out too
  await flush(h.disk);
  const h2 = host({ disk: h.disk, ckptN: 0 });
  h2.ckptClear();
  eq(await h2.ckptRestore(), [1, 2]);
  eq(h2.ai.ckpt.items.map((x) => [x.key, x.ids]).sort(), [[1, [1, 2, 3]], [2, [1, 2, 3, 4, 5]]]);
  eq(h2.ai.ckptN, 2, "the next save is slot 3, never a number a device may still hold");
  eq(h2.engine.st, [], "the host's live state is reset after the restore");
  eq(h2.ckptResume([1, 2, 3, 4, 5, 9], 0), 5);
  eq(h2.engine.st, [1, 2, 3, 4, 5]);
});

Deno.test("room disk: solo, the save goes to disk at once", async () => {
  const h = host({ chain: [] });
  h.engine.run([1, 2], 0); h.ai.fed = [1, 2];
  h.ckptSave();
  await flush(h.disk);
  eq((await h.disk.list({ room: "ABC", model: "m", sig: h.engine.stateSignature() })).map((c) => c.meta.ids), [[1, 2]]);
});

Deno.test("room disk: a device dealt other layers finds nothing, and a load would fail", async () => {
  const h = host(), w = worker();
  await answer(h, w, [1, 2, 3]);
  await lap(h, w, [4]);
  await flush(w.disk);
  const w2 = worker({ disk: w.disk, engine: new FakeEngine(8, 12) });
  eq(await w2.ckptRestore(), []);
  let err = null;
  try { await w2.workerFrame({ t: "ai-hidden-b", basePos: 3, n: 1, x: [9], ld: 1 }); } catch (e) { err = e; }
  ok(err && /no saved slot/.test(err.message), err?.message);
});

Deno.test("room disk: drops reach the disk on host and worker; DROP_ALL clears the room", async () => {
  const h = host({ ckptMax: 1 }), w = worker();
  await answer(h, w, [1]);
  await answer(h, w, [2]);                   // evicts slot 1 on the host now, on the worker with the next frame
  await flush(h.disk);
  eq((await h.disk.list({ room: "ABC", model: "m", sig: h.engine.stateSignature() })).length, 0, "slot 1 dropped; slot 2 not out yet");
  await lap(h, w, [3]);
  await flush(h.disk); await flush(w.disk);
  eq((await w.disk.list({ room: "ABC", model: "m", sig: w.engine.stateSignature() })).map((c) => c.slot), [2]);
  eq((await h.disk.list({ room: "ABC", model: "m", sig: h.engine.stateSignature() })).map((c) => c.slot), [2]);
  h.ckptClear(true);                         // a failure: nothing saved is trusted any more
  await lap(h, w, [4]);
  await flush(h.disk); await flush(w.disk);
  eq(h.disk.mem.files.size + w.disk.mem.files.size, 0);
});

Deno.test("room disk: without disk copies a rejoin still drops everything (the GPU-only behaviour)", () => {
  const h = host({ disk: null });
  h.engine.run([1], 0); h.ai.fed = [1];
  h.ckptSave();
  h.ckptRejoin();
  eq(h.ai.ckpt.items, []);
  eq(h.ai.pendingCtl.dp, [DROP_ALL]);
});

Deno.test("room disk: the host forgets the checkpoints a reloaded device did not read back", async () => {
  const h = host({ chain: ["w0", "w1"] }), w = worker();
  await answer(h, w, [1, 2, 3]);
  await answer(h, w, [4, 5]);
  await lap(h, w, [6]);                      // slots 1 and 2 are out on the chain
  eq(h.ai.ckpt.items.map((x) => x.key), [1, 2]);
  h.ai.pendingCtl = {};
  // w0 read back only slot 2 (its copy of slot 1 never reached the disk); w1 reports both
  h.ai.ckptHeld = new Map([["w0", [2]], ["w1", [1, 2]]]);
  h.ckptPrune();
  eq(h.ai.ckpt.items.map((x) => x.key), [2]);
  ok(!h.engine.slots.has(1), "the host's own GPU slot goes too");
  eq(h.ai.pendingCtl.dp, [1], "the chain drops it with the next frame");
  eq(h.ai.ckptHeld.size, 0, "each report is used once");
  eq(h.ckptResume([1, 2, 3, 4, 5, 9], 0), 5, "the resume only asks for a slot every device holds");
  // a device from an older build lists nothing: nothing it could be asked to load is kept
  h.ai.ckptHeld = new Map([["w1", undefined]]);
  h.ckptPrune();
  eq(h.ai.ckpt.items, []);
  // a report from a device that is no longer in the chain is ignored
  h.ai.ckpt.add([7], 9); h.ai.ckptHeld = new Map([["gone", []]]);
  h.ckptPrune();
  eq(h.ai.ckpt.items.map((x) => x.key), [9]);
});

Deno.test("room disk: slots a reloaded worker read back that the host does not index are dropped from its GPU and disk", async () => {
  const h = host(), w = worker();
  await answer(h, w, [1]);
  await answer(h, w, [2]);
  await answer(h, w, [3]);                   // slot 1 evicted everywhere; slot 2 out
  await lap(h, w, [4]);                      // slot 3 out
  // a copy the host no longer has (lost with its own reload): the worker still has it on disk
  w.engine.st = [1, 2, 3, 4]; w.engine.saveSlot(7); w.ckptPersist(7);
  await flush(w.disk);
  const w2 = worker({ disk: w.disk });
  const slots = await w2.ckptRestore();
  eq(slots.slice().sort(), [2, 3, 7], "a worker reads back up to CKPT_MAX + 1");
  eq([...w2.engine.slots.keys()].sort(), [2, 3, 7]);
  h.ai.pendingCtl = {}; h.ai.fed = []; h.engine.st = [];
  h.ai.ckptHeld = new Map([["w0", slots]]);
  h.ckptPrune();
  eq(h.ai.ckpt.items.map((x) => x.key).sort(), [2, 3], "the host's index is untouched");
  eq(h.ai.pendingCtl.dp, [7], "the orphan goes with the next frame");
  ok(h.ai.ckptN >= 7, "new slot numbers go past it");
  await lap(h, w2, [9]);
  await flush(w2.disk);
  eq([...w2.engine.slots.keys()].sort(), [2, 3], "no orphaned GPU slot on the worker");
  eq((await w2.disk.list({ room: "ABC", model: "m", sig: w2.engine.stateSignature() })).map((c) => c.slot).sort(), [2, 3]);
  h.ai.fed = [9]; h.engine.st = [9];
  h.ckptSave();
  ok(h.ai.pendingCtl.sv > 7, `the next save is slot ${h.ai.pendingCtl.sv}, not one being dropped`);
});

Deno.test("room disk: a host that reloads keeps the age order, so the next save evicts the oldest", async () => {
  const h = host(), w = worker();
  await answer(h, w, [1]);
  await answer(h, w, [2]);
  await answer(h, w, [3]);                   // slot 1 evicted; slot 2 out
  await lap(h, w, [4]);                      // slot 3 out
  await flush(h.disk);
  const h2 = host({ disk: h.disk });
  h2.ckptClear();
  eq(await h2.ckptRestore(), [2, 3], "oldest first");
  h2.engine.st = [1, 2, 3, 4, 5]; h2.ai.fed = [1, 2, 3, 4, 5];
  h2.ckptSave();
  eq(h2.ai.ckpt.items.map((x) => x.key).sort(), [3, 4], "slot 2, the oldest, made room");
});
