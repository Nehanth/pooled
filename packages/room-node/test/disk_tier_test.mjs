// Expert offload's disk tier on the room node (roadmap/32 step 5): what --disk says in the hello (meta.disk,
// meta.local), what the host deals (diskLo, diskRamBytes) and what a device checks before it loads, the reader
// shard.js gives the engine (the worker threads, else the engine's own reader in this thread), and freeing it.
// No GPU (the GPU path: test/e2e.mjs nodepair with WORKER_DISK=1, tests/test_q38_bits.js with OFFLOAD_DISK).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { RoomNode } from "../roomnode.js";
import { localModels } from "../source.js";
import { expertReader, diskLayers, offloadLayers } from "../shard.js";
import { expertDiskReader } from "../expert_disk.js";
import { expertDisk } from "../../../engine/expert_disk.js";
import { roomBytes } from "../../../room/models.js";
import { DISK_RAM_MIN } from "../../../room/plan.js";
import { PROTOCOL } from "../../../room/transport.js";
import { GGML_Q8_0 } from "../../../engine/gguf.js";

const GB = 2 ** 30, M35 = "qwen3.6-35b-moe";
function fakeNode({ host = true, name = "host", ramGB = 0, expertDisk = false } = {}) {
  const n = new RoomNode({ name, pledgeGB: 8, log: () => {}, ramGB, expertDisk });
  n.isHost = host; n.code = "TEST"; n.meta = { webgpu: true, contribGB: 8, gpu: "fake" };
  n.peer = { id: host ? "pooled-room-TEST" : "me", connect: () => null, disconnected: false };
  n.sent = {};
  n.sendTo = (id, m) => { (n.sent[id] ||= []).push(m); };
  n.sendHidden = (id, m) => { (n.sent[id] ||= []).push(m); };
  return n;
}
const msgs = (n, id, t) => (n.sent[id] || []).filter((m) => m.t === t);

test("disk tier: localModels lists the models whose every file is in the model dir", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "rn-disk-"));
  assert.deepEqual(localModels(dir), []);
  fs.mkdirSync(path.join(dir, M35));
  fs.writeFileSync(path.join(dir, M35, "Qwen_Qwen3.6-35B-A3B-Q4_0.gguf"), "x");
  fs.mkdirSync(path.join(dir, "qwen3.5-122b-moe"));
  fs.writeFileSync(path.join(dir, "qwen3.5-122b-moe", "Qwen_Qwen3.5-122B-A10B-Q4_0-00001-of-00002.gguf"), "x");   // one of two files
  assert.deepEqual(localModels(dir), [M35]);
  assert.deepEqual(localModels(null), []);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("disk tier: a device checks a deal's disk layers against what it offered", () => {
  const off = { lo: 20, hi: 40, vramBytes: 3 * GB, ramBytes: 6 * GB, ramSpare: 2 * GB, diskLo: 34, diskRamBytes: 3 * GB };
  assert.throws(() => fakeNode({ ramGB: 12 }).offloadOk(off), /no disk tier \(pooled join --disk\)/);
  const n = fakeNode({ ramGB: 12, expertDisk: true });
  assert.equal(n.offloadOk(off).ramCap, 12 * GB - 2 * GB - 3 * GB, "parked experts get what the spare and the disk cache leave");
  assert.throws(() => n.offloadOk({ ...off, diskRamBytes: 5 * GB }), /3\.0 GB for their disk cache|5\.0 GB for their disk cache/);
  assert.equal(fakeNode({ ramGB: 12 }).offloadOk({ ...off, diskLo: undefined, diskRamBytes: undefined }).ramCap, 10 * GB, "no disk layers: as before");
  assert.deepEqual(diskLayers(off, offloadLayers(off, [10, 40])), [34, 35, 36, 37, 38, 39]);
  assert.deepEqual(diskLayers({ lo: 20, hi: 40, vramBytes: 1 }, offloadLayers({ lo: 20, hi: 40, vramBytes: 1 }, [10, 40])), []);
});

test("disk tier: the host deals disk layers only to a --disk device with the model, and only when parking falls short", () => {
  const rb = roomBytes(M35, 32768, "f16");
  const pc = (disk, local = [M35]) => ({ webgpu: true, contribGB: 10, offload: true, ramGB: 8, ...(disk ? { disk: true, local } : {}) });
  const plan = (meta, model = M35) => RoomNode.dealPlan({ L: rb.L, layerBytes: rb.layerBytes, embedBytes: rb.hostBytes, self: { name: "h", meta: { webgpu: true, contribGB: 4 } },
    peers: [{ id: "p", name: "pc", meta }], fitBytes: rb, mode: "memory", model });
  const d = plan(pc(true));
  assert.ok(d.fit.fits && d.fit.disk, "fits with the PC's disk tier");
  const o = d.offload[1];
  assert.ok(Number.isInteger(o.diskLo) && o.diskLo > o.lo && o.hi === rb.L && o.diskRamBytes >= DISK_RAM_MIN, JSON.stringify(o));
  assert.ok(!plan(pc(false)).fit.fits, "no --disk: short");
  assert.ok(!plan(pc(true, ["qwen3-1.7b"])).fit.fits, "the model's files aren't there: short");
  assert.ok(!plan(pc(true), null).fit.fits, "no model named: no disk tier");
});

test("disk tier: a worker loads its disk layers with the deal's numbers and frees the reader with its layers", async () => {
  const n = fakeNode({ host: false, name: "pc", ramGB: 12, expertDisk: true });
  n.modelDir = fs.mkdtempSync(path.join(os.tmpdir(), "rn-disk-"));
  let seen = null, closed = 0;
  n.loadShardFn = async ({ offload }) => { seen = offload; return { engine: { maxSeq: 4096, reset() {}, dropAllSlots() {} }, device: { destroy() {} }, cfg: {}, experts: { close: async () => { closed++; } } }; };
  n.ensureLink = async () => true;
  await n.workerLoad("h", { t: "ai-load", v: PROTOCOL, model: M35, range: [20, 40], ctx: 4096, next: "host", host: "h",
    offload: { lo: 26, hi: 40, vramBytes: 3 * GB, ramBytes: 5 * GB, ramSpare: GB, diskLo: 33, diskRamBytes: 2 * GB } });
  assert.equal(seen.diskLo, 33); assert.equal(seen.ramCap, 9 * GB);
  assert.equal(msgs(n, "h", "ai-ready").length, 1, "online");
  n.freeLayers(null);
  await new Promise((r) => setImmediate(r));
  assert.equal(closed, 1, "freeing the layers closes the disk reader and its cache");
  assert.equal(n.ai.experts, null);
  fs.rmSync(n.modelDir, { recursive: true, force: true });
});

// a stacked Q8_0 expert tensor [nExp][rows][cols] of random bytes at byteOffset in a file
function q8File(dir, nExp = 4, rows = 8, cols = 64, byteOffset = 96) {
  const rowBytes = cols / 32 * 34, bytes = new Uint8Array(byteOffset + nExp * rows * rowBytes);
  let x = 7; for (let i = 0; i < bytes.length; i++) bytes[i] = (x = (x * 1103515245 + 12345) >>> 0) >>> 24;
  for (let o = byteOffset; o < bytes.length; o += 34) { bytes[o] = 0x00; bytes[o + 1] = 0x3c; }   // f16 scales: 1.0
  const f = path.join(dir, "m.gguf");
  fs.writeFileSync(f, bytes);
  return { f, info: { name: "blk.0.ffn_up_exps.weight", shape: [nExp, rows, cols], ggmlType: GGML_Q8_0, byteOffset, shard: 0 } };
}

test("disk tier: the reader falls back to this thread when the worker threads can't run, with the same bytes", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "rn-disk-")), { f, info } = q8File(dir);
  const real = expertDiskReader({ paths: [f], threads: 2 });
  const want = await Promise.all([0, 1, 2, 3].map((e) => real.readPart(info, e)));
  await real.close();
  const logs = [];
  const dead = expertReader({ paths: [f], log: (s) => logs.push(s), make: () => ({ readPart: async () => { throw new Error("expert disk reader: no worker left (cannot find module)"); }, close: async () => {} }) });
  for (let e = 0; e < 4; e++) {
    const got = await dead.readPart(info, e);
    assert.deepEqual([...got.qs], [...want[e].qs], `expert ${e} qs`);
    assert.deepEqual([...got.sc.subarray(0, want[e].sc.length)], [...want[e].sc], `expert ${e} scales`);
  }
  assert.equal(logs.length, 1, "said once");
  await dead.close();
  // any other error is the read's own: it is not hidden behind the fallback
  const bad = expertReader({ paths: [f], make: () => ({ readPart: async () => { throw new Error("short read"); }, close: async () => {} }) });
  await assert.rejects(bad.readPart(info, 0), /short read/);
  // the engine's in-process reader agrees with the workers too (the store takes either)
  const inproc = expertDisk({ readAt: async (s, off, len) => new Uint8Array(fs.readFileSync(f).subarray(off, off + len)) });
  assert.deepEqual([...(await inproc.readPart(info, 2)).qs], [...want[2].qs]);
  const live = expertReader({ paths: [f] });
  assert.deepEqual([...(await live.readPart(info, 3)).qs], [...want[3].qs], "the default: the worker threads");
  await live.close();
  fs.rmSync(dir, { recursive: true, force: true });
});
