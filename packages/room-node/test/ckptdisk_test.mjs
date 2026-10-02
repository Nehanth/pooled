// No GPU: the pinned checkpoints' disk copies (ckptdisk.js) and the room node's restore after a
// restart, with a fake engine whose next token depends on everything in its caches (so a resume from
// a wrong state changes the answer).
//   node --test packages/room-node/test/ckptdisk_test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { CkptDisk, roomKey, localKey, prefixHash, canon, encodeHeader, decodeHeader, modelFileId, CkptDiskError } from "../ckptdisk.js";
import { RoomNode } from "../roomnode.js";

const tmpDir = () => fs.mkdtempSync(path.join(os.tmpdir(), "pkv-test-"));
const range = (a, n) => Array.from({ length: n }, (_, i) => a + i);
const SIG = { v: 1, lo: 0, hi: 20, mtp: true, flash: true, kvQ8: false, dims: [2048, 512, 32, 8192], ub: 256 };
const LOCAL = { model: "qwen3.6-35b-moe", file: "f".repeat(32), sig: SIG, ctx: 131072, kv: "f16" };
const ROOM = roomKey({ model: LOCAL.model, ctx: LOCAL.ctx, kv: "f16", devices: [{ file: LOCAL.file, sig: SIG, ctx: LOCAL.ctx, kv: "f16" }] });
const state = (n, fill = 7) => ({ pos: n, parts: [new Uint8Array(n * 6).fill(fill).buffer, new Float32Array([1, 2, 3]).buffer, new ArrayBuffer(0)] });

test("keys: the room key covers the model, context, KV mode and every device's file and signature; the prefix its exact ids", () => {
  const base = { model: "m", ctx: 16384, kv: "f16", devices: [{ file: "a", sig: { lo: 0, hi: 20 } }, { file: "b", sig: { lo: 20, hi: 40 } }] };
  const k = roomKey(base);
  assert.equal(k, roomKey(JSON.parse(JSON.stringify(base))), "stable");
  assert.equal(canon({ b: 1, a: [1, { d: 2, c: 3 }] }), canon({ a: [1, { c: 3, d: 2 }], b: 1 }), "key order does not matter");
  const variants = {
    model: { ...base, model: "m2" },
    ctx: { ...base, ctx: 8192 },
    kv: { ...base, kv: "q8" },
    file: { ...base, devices: [{ ...base.devices[0], file: "a2" }, base.devices[1]] },
    split: { ...base, devices: [{ file: "a", sig: { lo: 0, hi: 22 } }, { file: "b", sig: { lo: 22, hi: 40 } }] },
    signature: { ...base, devices: [base.devices[0], { file: "b", sig: { lo: 20, hi: 40, ub: 256 } }] },
    order: { ...base, devices: [base.devices[1], base.devices[0]] },
    solo: { ...base, devices: [base.devices[0]] },
  };
  for (const [what, v] of Object.entries(variants)) assert.notEqual(roomKey(v), k, what);
  assert.notEqual(localKey(LOCAL), localKey({ ...LOCAL, sig: { ...SIG, ub: undefined } }), "signature");
  assert.notEqual(localKey(LOCAL), localKey({ ...LOCAL, sig: { ...SIG, dp4a: 1 } }), "signature (dp4a)");
  assert.notEqual(localKey(LOCAL), localKey({ ...LOCAL, ctx: 65536 }), "context");
  const ids = range(100, 50);
  assert.equal(prefixHash("m", ids), prefixHash("m", Uint32Array.from(ids)));
  assert.notEqual(prefixHash("m", ids), prefixHash("m", [...ids.slice(0, -1), 999]), "one token differs");
  assert.notEqual(prefixHash("m", ids), prefixHash("m", ids.slice(0, -1)), "a shorter prefix");
  assert.notEqual(prefixHash("m", ids), prefixHash("m2", ids), "another model");
});

test("header: magic, format version, 4096-byte aligned parts", () => {
  const b = encodeHeader({ room: "r", local: LOCAL, h: "x", pos: 3, sizes: [4] });
  assert.equal(b.length % 4096, 0);
  const { header, off } = decodeHeader(b);
  assert.equal(off, b.length); assert.equal(header.pos, 3);
  const bad = Buffer.from(b); bad.writeUInt32BE(0, 0);
  assert.throws(() => decodeHeader(bad), CkptDiskError);
  const other = encodeHeader({ room: "r", local: LOCAL, h: "x", pos: 3, sizes: [4] });
  other.write(JSON.stringify({ f: 99 }).padEnd(other.readUInt32BE(4)), 8);
  assert.throws(() => decodeHeader(other), /format/);
});

test("put / get: round trip, ids kept, 0600 files in a 0700 dir, aligned views; has/list see it", async () => {
  const dir = path.join(tmpDir(), "ckpt"), d = new CkptDisk({ dir });
  const ids = range(5, 40), h = prefixHash(LOCAL.model, ids), where = { room: ROOM, local: LOCAL, h };
  assert.equal(await d.get(where), null);
  assert.equal(await d.put(where, state(40), { ids }), true);
  assert.equal(fs.statSync(dir).mode & 0o777, 0o700);
  const f = fs.readdirSync(dir);
  assert.equal(f.length, 1); assert.match(f[0], /^[0-9a-f]{16}\.[0-9a-f]{16}\.[0-9a-f]{16}\.pkv$/);
  assert.equal(fs.statSync(path.join(dir, f[0])).mode & 0o777, 0o600);
  const got = await d.get(where);
  assert.equal(got.pos, 40); assert.deepEqual(got.ids, ids);
  assert.deepEqual(got.parts.map((p) => p.byteLength), [240, 12, 0]);
  assert.ok(got.parts.every((p) => p.byteOffset % 4 === 0), "4-byte aligned for writeBuffer");
  assert.ok(got.parts[0].every((x) => x === 7));
  assert.deepEqual([...new Float32Array(got.parts[1].buffer, got.parts[1].byteOffset, 3)], [1, 2, 3]);
  assert.equal(await d.has(where), true);
  assert.deepEqual((await d.list({ room: ROOM, local: LOCAL })).map((c) => [c.h, c.n]), [[h, 40]]);
  // already there: not read or written again
  let read = 0;
  assert.equal(await d.put(where, async () => { read++; return state(40); }, { ids }), true);
  assert.equal(read, 0);
});

test("invalidation: another model, layer split, signature, context, room or prefix reads as missing", async () => {
  const dir = tmpDir(), d = new CkptDisk({ dir });
  const ids = range(5, 40), h = prefixHash(LOCAL.model, ids);
  await d.put({ room: ROOM, local: LOCAL, h }, state(40), { ids });
  const miss = {
    model: { room: ROOM, local: { ...LOCAL, model: "qwen3.8-27b" }, h },
    "model file": { room: ROOM, local: { ...LOCAL, file: "e".repeat(32) }, h },
    "layer split": { room: ROOM, local: { ...LOCAL, sig: { ...SIG, hi: 22 } }, h },
    signature: { room: ROOM, local: { ...LOCAL, sig: { ...SIG, ub: undefined } }, h },
    context: { room: ROOM, local: { ...LOCAL, ctx: 65536 }, h },
    "KV mode": { room: ROOM, local: { ...LOCAL, kv: "q8" }, h },
    room: { room: "0".repeat(64), local: LOCAL, h },
    prefix: { room: ROOM, local: LOCAL, h: prefixHash(LOCAL.model, [...ids.slice(0, -1), 1]) },
  };
  for (const [what, where] of Object.entries(miss)) {
    assert.equal(await d.get(where), null, what);
    if (what !== "prefix") assert.deepEqual(await d.list(where), [], what);   // (list is every prefix of this device in this room)
  }
  assert.ok(await d.get({ room: ROOM, local: LOCAL, h }), "the matching lookup still finds it");
  // a file whose name matches but whose header says another device (renamed, or a hash collision): removed
  const name = fs.readdirSync(dir)[0];
  const other = { ...LOCAL, sig: { ...SIG, lo: 1 } };
  const forged = new CkptDisk({ dir: tmpDir() });
  await forged.put({ room: ROOM, local: other, h }, state(40), { ids });
  fs.copyFileSync(path.join(forged.dir, fs.readdirSync(forged.dir)[0]), path.join(dir, name));
  assert.equal(await d.get({ room: ROOM, local: LOCAL, h }), null);
  assert.deepEqual(fs.readdirSync(dir), [], "removed");
  // ids that do not hash to h (a damaged host copy): never indexed
  const bad = new CkptDisk({ dir: tmpDir() });
  await bad.put({ room: ROOM, local: LOCAL, h }, state(40), { ids: range(6, 40) });
  assert.deepEqual(await bad.list({ room: ROOM, local: LOCAL }), []);
});

test("damaged files: truncated, appended to, or not a checkpoint: missing and removed", async () => {
  const dir = tmpDir(), d = new CkptDisk({ dir });
  const ids = range(5, 40), where = { room: ROOM, local: LOCAL, h: prefixHash(LOCAL.model, ids) };
  for (const damage of [(f) => fs.truncateSync(f, fs.statSync(f).size - 3), (f) => fs.appendFileSync(f, "x"), (f) => fs.truncateSync(f, 6), (f) => fs.writeFileSync(f, "hello")]) {
    await d.put(where, state(40), { ids });
    const f = path.join(dir, fs.readdirSync(dir)[0]);
    damage(f);
    assert.equal(await d.get(where), null);
    assert.equal(fs.existsSync(f), false);
  }
});

test("atomic writes: a failed or interrupted write leaves no file under a real name; old temp files are swept", async () => {
  const dir = tmpDir(), d = new CkptDisk({ dir, tmpStaleMs: 1000 });
  const ids = range(5, 40), where = { room: ROOM, local: LOCAL, h: prefixHash(LOCAL.model, ids) };
  // the state was gone by the time the write's turn came
  assert.equal(await d.put(where, async () => { throw new Error("the slot is gone"); }, { ids }), false);
  assert.deepEqual(fs.readdirSync(dir), []);
  // a part that fails mid-write: the temp file goes, nothing under the real name
  assert.equal(await d.put(where, { pos: 40, parts: [new ArrayBuffer(8), { byteLength: 8, get buffer() { throw new Error("boom"); } }] }, { ids }), false);
  assert.deepEqual(fs.readdirSync(dir), []);
  // a write that never finished (the process died): its temp file is not a checkpoint, and is swept when old
  const name = d.name(where);
  fs.writeFileSync(path.join(dir, `${name}.123.abcd.tmp`), "half");
  assert.equal(await d.get(where), null);
  assert.deepEqual(await d.list({ room: ROOM, local: LOCAL }), []);
  const old = new Date(Date.now() - 5000);
  fs.utimesSync(path.join(dir, `${name}.123.abcd.tmp`), old, old);
  assert.equal(await d.put(where, state(40), { ids }), true);
  assert.deepEqual(fs.readdirSync(dir), [name]);
  // concurrent puts of the same prefix are serialized: one file, read once
  let reads = 0;
  const d2 = new CkptDisk({ dir: tmpDir() });
  await Promise.all([1, 2, 3].map(() => d2.put(where, async () => { reads++; return state(40); }, { ids })));
  assert.equal(reads, 1);
});

test("cap: least recently used copies go first; a copy bigger than the cap is not written", async () => {
  const dir = tmpDir();
  const one = 4096 + 40 * 6 + 12;   // header page + parts
  const d = new CkptDisk({ dir, capBytes: 3 * one + 100 });
  const w = (i) => ({ room: ROOM, local: LOCAL, h: prefixHash(LOCAL.model, range(i, 40)) });
  for (const i of [1, 2, 3]) { await d.put(w(i), state(40), { ids: range(i, 40) }); await new Promise((r) => setTimeout(r, 15)); }
  await d.get(w(1));   // used: now the most recent
  await new Promise((r) => setTimeout(r, 15));
  await d.put(w(4), state(40), { ids: range(4, 40) });
  assert.equal(await d.has(w(2)), false, "the least recently used went");
  for (const i of [1, 3, 4]) assert.equal(await d.has(w(i)), true, `copy ${i} stays`);
  assert.ok(await d.size() <= 3 * one + 100);
  assert.equal(await new CkptDisk({ dir: tmpDir(), capBytes: 1000 }).put(w(5), state(40), { ids: range(5, 40) }), false);
});

test("fromEnv: POOLED_CKPT_DISK=0 turns it off; POOLED_CKPT_DIR / POOLED_CKPT_GB", () => {
  assert.equal(CkptDisk.fromEnv({ POOLED_CKPT_DISK: "0" }), null);
  const d = CkptDisk.fromEnv({ POOLED_CKPT_DIR: "/x/y", POOLED_CKPT_GB: "2" });
  assert.equal(d.dir, "/x/y"); assert.equal(d.cap, 2 * 2 ** 30);
  assert.equal(CkptDisk.fromEnv({}).dir, path.join(os.homedir(), ".pooled", "cache", "ckpt"));
});

test("modelFileId: size and the file's start and end; null when it cannot be read", async () => {
  const dir = tmpDir(), f = path.join(dir, "m.gguf");
  fs.writeFileSync(f, Buffer.alloc(6 * 2 ** 20, 1));
  const src = (file) => ({ local: file, url: "u", readAt: async (off, len) => { const fd = fs.openSync(file, "r"); const b = Buffer.alloc(Math.min(len, fs.statSync(file).size - off)); fs.readSync(fd, b, 0, b.length, off); fs.closeSync(fd); return b; } });
  const a = await modelFileId(src(f));
  assert.match(a, /^[0-9a-f]{32}$/);
  assert.equal(await modelFileId(src(f)), a);
  const fd = fs.openSync(f, "r+"); fs.writeSync(fd, Buffer.from([9]), 0, 1, 6 * 2 ** 20 - 10); fs.closeSync(fd);
  assert.notEqual(await modelFileId(src(f)), a, "the end changed");
  assert.equal(await modelFileId({ readAt: async () => { throw new Error("no"); } }), null);
});

// ---- the room node: persist on pin, restore after a restart ----
class FakeEngine {
  constructor(sig = SIG) { this.hist = []; this.pos = 0; this.maxSeq = 4096; this.slots = new Map(); this.dims = { dim: 4 }; this.sig = sig; this.imports = 0; }
  reset() { this.hist = []; this.pos = 0; }
  _put(id, pos) { this.hist.length = pos; this.hist.push(id); this.pos = pos + 1; }
  async prefillTokens(ids) { for (const id of ids) this._put(id, this.pos); }
  async prefillToken(id) { this._put(id, this.pos); }
  async embedRun(id, pos) { this._put(id, pos); return new Float32Array(4); }
  async headFromHidden() {
    let h = 7;
    for (const t of this.hist.slice(0, this.pos)) h = (Math.imul(h, 31) + t + 1) | 0;
    const lg = new Float32Array(50); lg[1 + ((h >>> 0) % 49)] = 1; return lg;
  }
  saveSlot(k) { this.slots.set(k, { hist: this.hist.slice(0, this.pos), pos: this.pos }); }
  loadSlot(k) { const s = this.slots.get(k); if (!s) throw new Error("no saved slot " + k); this.hist = s.hist.slice(); this.pos = s.pos; }
  dropSlot(k) { this.slots.delete(k); }
  dropAllSlots() { this.slots.clear(); }
  stateSignature() { return this.sig; }
  async exportSlot(k) { const s = this.slots.get(k); if (!s) throw new Error("no saved slot " + k); return { sig: this.sig, pos: s.pos, parts: [Uint32Array.from(s.hist).buffer] }; }
  importState(st) {
    if (JSON.stringify(st.sig) !== JSON.stringify(this.sig)) throw new Error("another signature");
    const p = st.parts[0];
    this.hist = [...new Uint32Array(p.buffer, p.byteOffset, p.byteLength / 4)]; this.pos = st.pos; this.imports++;
  }
}
const argmax = (lg) => { let b = 0; for (let i = 1; i < lg.length; i++) if (lg[i] > lg[b]) b = i; return b; };
function host(dir, { sig = SIG, file = "F", ckptDisk } = {}) {
  const n = new RoomNode({ name: "host", pledgeGB: 8, log: () => {}, ckptDisk: ckptDisk ?? (dir ? new CkptDisk({ dir }) : false) });
  n.isHost = true; n.code = "TEST"; n.peer = { id: "pooled-room-TEST" };
  Object.assign(n.ai, { engine: new FakeEngine(sig), online: true, model: "qwen3.6-35b-moe", role: "host", device: { queue: { onSubmittedWorkDone: async () => {} } },
    held: { model: "qwen3.6-35b-moe", range: [sig.lo, sig.hi], ctx: 4096, kv: "f16", file } });
  return n;
}
function worker(dir, { sig = { ...SIG, lo: 20, hi: 40, mtp: false }, file = "F" } = {}) {
  const w = new RoomNode({ name: "w", pledgeGB: 8, log: () => {}, ckptDisk: new CkptDisk({ dir }) });
  w.code = "TEST"; w.peer = { id: "w" };
  Object.assign(w.ai, { engine: new FakeEngine(sig), role: "worker", hostId: "pooled-room-TEST", model: "qwen3.6-35b-moe",
    held: { model: "qwen3.6-35b-moe", range: [sig.lo, sig.hi], ctx: 4096, kv: "f16", file } });
  return w;
}
// the host and one worker, their messages delivered in process (frames: the save rides them as on the wire)
function pair(h, w) {
  h.ai.chain = ["w"]; h.ai.ckptCap.set("w", true); h.conns.set("w", { conn: {} });
  h.ai.diskOf.set("w", w.diskLocal());
  h.sendTo = (id, m) => { if (id === "w") queueMicrotask(() => w.aiOnData("pooled-room-TEST", m)); };
  w.sendTo = (id, m) => queueMicrotask(() => h.aiOnData("w", m));
  w.conns.set("pooled-room-TEST", { conn: {} });
  return { frame(ctl) { if (ctl.sv != null) w.ai.engine.saveSlot(ctl.sv); w.diskAfterFrame(ctl); } };
}
const gen = (n, ids, pins = []) => n.generateOnce(ids, { stop: new Set([0]), maxNew: 6, sample: argmax, spec: false, pins, turn: 0 });
const S = range(100, 40), Q = [1, 2, 3];

test("solo host: a pinned prefix goes to disk; after a restart it is read back and the first answer resumes from it, same tokens", async () => {
  const dir = tmpDir();
  const cold = host(dir);
  cold.ai.diskRoom = cold.diskRoomKey();
  const a = await gen(cold, [...S, ...Q], [40]);
  assert.equal(a.from, null); assert.equal(a.pinned, 1);
  await cold.disk.q;
  assert.equal(fs.readdirSync(dir).length, 1);
  const warm = host(dir);
  const r = await warm.diskRestoreAll();
  assert.equal(r.restored, 1); assert.deepEqual(r.tokens, [40]);
  assert.deepEqual(warm.status().ckpt.pinned, [40]);
  const b = await gen(warm, [...S, ...Q], [40]);
  assert.equal(b.from, "pin"); assert.equal(b.prefilled, 3);
  assert.deepEqual(b.tokens, a.tokens, "the same answer");
  // another layer range or engine signature (wide prefill off): nothing restored, a normal prefill, same answer
  for (const sig of [{ ...SIG, hi: 30 }, { ...SIG, ub: undefined }]) {
    const other = host(dir, { sig });
    assert.equal((await other.diskRestoreAll()).restored, 0);
    const c = await gen(other, [...S, ...Q], [40]);
    assert.equal(c.from, null); assert.deepEqual(c.tokens, a.tokens);
  }
  // another model file
  assert.equal((await host(dir, { file: "G" }).diskRestoreAll()).restored, 0);
  // no disk cache at all
  assert.equal(host(null).diskRoomKey(), null);
});

test("host + worker: each device writes its own part when the save reaches it; restore needs every device's copy", async () => {
  const hdir = tmpDir(), wdir = tmpDir();
  const h = host(hdir), w = worker(wdir), net = pair(h, w);
  h.ai.diskRoom = h.diskRoomKey();
  assert.ok(h.ai.diskRoom);
  // the host pins its prefix: the save rides the next frame; the message may get there first
  h.ai.engine.hist = S.slice(); h.ai.engine.pos = 40; h.ai.fed = S.slice(); h.ai.pos = 40;
  w.ai.engine.hist = S.slice(); w.ai.engine.pos = 40;
  const slot = h.ckptSave(true);
  h.diskPersist(slot, h.ai.fed);
  await new Promise((r) => setTimeout(r, 5));
  assert.equal(w.ai.diskPend.size, 1, "waiting for the frame");
  net.frame({ sv: h.ai.pendingCtl.sv });
  await h.disk.q; await w.disk.q;
  assert.equal(fs.readdirSync(hdir).length, 1); assert.equal(fs.readdirSync(wdir).length, 1);
  // restart: both read their parts back into the same slot
  const h2 = host(hdir), w2 = worker(wdir); pair(h2, w2);
  const r = await h2.diskRestoreAll();
  assert.equal(r.restored, 1);
  const k = h2.ai.ckpt.items[0].key;
  assert.deepEqual(w2.ai.engine.slots.get(k).hist, S, "the worker's slot holds its part of the prefix");
  assert.deepEqual(h2.ai.engine.slots.get(k).hist, S);
  assert.deepEqual(h2.ai.dropQ, []);
  // the worker lost its copy: the host indexes nothing (a normal prefill), and drops its own slot
  fs.rmSync(path.join(wdir, fs.readdirSync(wdir)[0]));
  const h3 = host(hdir), w3 = worker(wdir); pair(h3, w3);
  assert.equal((await h3.diskRestoreAll()).restored, 0);
  assert.equal(h3.ai.ckpt.size, 0); assert.equal(h3.ai.engine.slots.size, 0);
  // a worker on another layer split: another room key, nothing to restore
  const h4 = host(hdir), w4 = worker(wdir, { sig: { ...SIG, lo: 22, hi: 40, mtp: false } }); pair(h4, w4);
  assert.notEqual(h4.diskRoomKey(), h2.diskRoomKey());
  assert.equal((await h4.diskRestoreAll()).restored, 0);
  // a worker that keeps no disk copies (an older tab): no room key, no restore, no persist
  const h5 = host(hdir), w5 = worker(wdir); pair(h5, w5); h5.ai.diskOf.set("w", null);
  assert.equal(h5.diskRoomKey(), null);
});

test("host + worker: a worker that answers no slots (or never answers) means a prefill, and slots only it read are dropped", async () => {
  const hdir = tmpDir(), wdir = tmpDir();
  const h = host(hdir), w = worker(wdir); pair(h, w);
  h.ai.diskRoom = h.diskRoomKey();
  h.ai.engine.hist = S.slice(); h.ai.engine.pos = 40; h.ai.fed = S.slice(); h.ai.pos = 40;
  h.diskPersist(h.ckptSave(true), h.ai.fed);
  await h.disk.q;
  const h2 = host(hdir), w2 = worker(wdir); pair(h2, w2);
  h2.sendTo = () => {};   // the worker never answers
  assert.equal((await h2.diskRestoreAll({ timeoutMs: 50 })).restored, 0);
  assert.equal(h2.ai.ckpt.size, 0);
});
