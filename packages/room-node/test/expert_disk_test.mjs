// The room node's disk reader for expert offload (expert_disk.js): one expert of a stacked GGUF tensor, read in
// chunks by a pool of worker threads and converted there, is byte for byte the slice the whole tensor's conversion
// (engine/gguf.js convertEntry, what the loader parks) has at that expert. A synthetic two-file GGUF in a temp dir,
// one tensor per ggml type the experts come in, so no model file is needed.
import { test, after } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { expertDiskReader, expertPart, defaultDiskThreads, sharedReads } from "../expert_disk.js";
import { parseGGUFHeader, convertEntry, ggmlTypeBytes, GGML_F32, GGML_Q4_0, GGML_Q8_0, GGML_Q4_1, GGML_Q5_K, GGML_Q6_K } from "../../../engine/gguf.js";

// a GGUF v3 file with no metadata and these tensors ({ name, shape: [nExp, rows, cols], ggmlType, bytes })
function writeGGUF(file, tensors) {
  const parts = [];
  const u32 = (v) => { const b = Buffer.alloc(4); b.writeUInt32LE(v); parts.push(b); };
  const u64 = (v) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(v)); parts.push(b); };
  const str = (s) => { u64(Buffer.byteLength(s)); parts.push(Buffer.from(s)); };
  u32(0x46554747); u32(3); u64(tensors.length); u64(0);
  let off = 0;
  for (const t of tensors) {
    str(t.name); u32(t.shape.length);
    for (const d of [...t.shape].reverse()) u64(d);   // ggml order: innermost first
    u32(t.ggmlType); u64(off);
    off += Math.ceil(t.bytes.byteLength / 32) * 32;
  }
  let head = Buffer.concat(parts);
  head = Buffer.concat([head, Buffer.alloc(Math.ceil(head.length / 32) * 32 - head.length)]);
  const body = Buffer.alloc(off);
  off = 0;
  for (const t of tensors) { body.set(t.bytes, off); off += Math.ceil(t.bytes.byteLength / 32) * 32; }
  fs.writeFileSync(file, Buffer.concat([head, body]));
}

// random bytes with every byte's 0x88 bits clear: any two bytes read as an f16 are finite, whatever the block layout
function bytesFor(ggmlType, shape, seed) {
  const n = shape.reduce((a, b) => a * b, 1), b = new Uint8Array(ggmlTypeBytes(ggmlType, n));
  for (let i = 0; i < b.length; i++) { seed = (seed * 1103515245 + 12345) & 0x7fffffff; b[i] = (seed >> 8) & 0x77; }
  return b;
}

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "expert-disk-"));
after(() => fs.rmSync(dir, { recursive: true, force: true }));
// shard 0: Q4_0 gate, Q4_1 down (37 rows of one block: an odd block count per expert and per chunk), an f32 norm;
// shard 1: Q8_0, Q5_K and Q6_K experts
const SPEC = [
  [{ name: "blk.0.ffn_gate_exps.weight", shape: [6, 40, 64], ggmlType: GGML_Q4_0 },
    { name: "blk.0.ffn_down_exps.weight", shape: [5, 37, 32], ggmlType: GGML_Q4_1 },
    { name: "blk.0.attn_norm.weight", shape: [4, 2, 32], ggmlType: GGML_F32 }],
  [{ name: "blk.1.ffn_up_exps.weight", shape: [4, 24, 96], ggmlType: GGML_Q8_0 },
    { name: "blk.1.ffn_gate_exps.weight", shape: [3, 9, 256], ggmlType: GGML_Q5_K },
    { name: "blk.1.ffn_down_exps.weight", shape: [3, 11, 512], ggmlType: GGML_Q6_K }],
];
const paths = SPEC.map((_, s) => path.join(dir, `m-0000${s + 1}-of-00002.gguf`));
const tensors = {}, raw = {};
SPEC.forEach((ts, s) => {
  ts.forEach((t, i) => { raw[t.name] = bytesFor(t.ggmlType, t.shape, 17 + 31 * s + i); });
  writeGGUF(paths[s], ts.map((t) => ({ ...t, bytes: raw[t.name] })));
  const buf = fs.readFileSync(paths[s]);
  const G = parseGGUFHeader(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length));
  for (const [k, v] of Object.entries(G.tensors)) tensors[k] = { ...v, shard: s };
});
const experts = Object.values(tensors).filter((t) => t.name.includes("_exps"));

// what the loader parks for expert e: the whole tensor's conversion (ggufEntry flattens [nExp][rows][cols]), sliced
const whole = new Map();
function want(info, e) {
  if (!whole.has(info.name)) {
    const [nExp, rows, cols] = info.shape;
    whole.set(info.name, convertEntry({ ggmlType: info.ggmlType, shape: [nExp * rows, cols], nElems: nExp * rows * cols }, raw[info.name]));
  }
  const c = whole.get(info.name), p = expertPart(info, e);
  const sc = new Uint8Array(c.scales.buffer), scB = (p.rows * p.cols / 32) * 2;
  const out = new Uint8Array(p.scBytes);   // this expert's f16 scales, then zeros up to a whole u32 (convertEntry's own padding)
  out.set(sc.subarray(e * scB, (e + 1) * scB));
  return { qs: c.qs.subarray(e * p.qsBytes, (e + 1) * p.qsBytes), sc: out };
}
const same = (got, w, msg) => {
  assert.deepEqual(Buffer.from(got.qs), Buffer.from(w.qs), msg + " qs");
  assert.deepEqual(Buffer.from(got.sc), Buffer.from(w.sc), msg + " sc");
};

test("expertPart: where an expert is and what it converts to", () => {
  const g = tensors["blk.0.ffn_gate_exps.weight"], p = expertPart(g, 2);
  assert.equal(p.off, g.byteOffset + 2 * 40 * (64 / 32) * 18);
  assert.equal(p.len, 40 * 36); assert.equal(p.kind, "q4"); assert.equal(p.qsBytes, 40 * 32); assert.equal(p.scBytes, 80 * 2);
  const d = expertPart(tensors["blk.0.ffn_down_exps.weight"], 4);
  assert.equal(d.kind, "q8"); assert.equal(d.scBytes, 76);   // 37 blocks -> 19 u32 words
  assert.equal(expertPart(tensors["blk.1.ffn_up_exps.weight"], 0).shard, 1);
  assert.throws(() => expertPart(tensors["blk.0.attn_norm.weight"], 0), /stay f32/);
  assert.throws(() => expertPart(g, 6), /expert 6 of 6/);
  assert.throws(() => expertPart({ ...g, shape: [240, 64] }, 0), /not a stacked/);
});

test("expertDiskReader: every expert of every type equals the whole-tensor conversion, in chunks over the workers", async () => {
  // chunks of a few rows: one expert spans several workers, so the pieces have to land in the right places
  const disk = expertDiskReader({ paths, threads: 3, chunkBytes: 200, ringSlots: 4 });
  try {
    for (const info of experts) {
      const all = await Promise.all(Array.from({ length: info.shape[0] }, (_, e) => disk.readPart(info, e)));
      all.forEach((got, e) => same(got, want(info, e), `${info.name} expert ${e}`));
    }
    const nParts = experts.reduce((a, t) => a + t.shape[0], 0);
    assert.equal(disk.stats.parts, nParts);
    assert.ok(disk.stats.chunks > nParts, "experts were split into chunks");
    assert.ok(disk.stats.slotWaits > 0, "more experts in flight than ring slots waited for one");
    // the three parts of one expert at once (the engine's gate / up / down)
    const trio = ["blk.1.ffn_up_exps.weight", "blk.1.ffn_gate_exps.weight", "blk.1.ffn_down_exps.weight"].map((n) => tensors[n]);
    (await disk.readExpert(trio, 2)).forEach((got, j) => same(got, want(trio[j], 2), trio[j].name));
  } finally { await disk.close(); }
});

test("expertDiskReader: a part that fits in one job is read, converted and handed over by one worker, no ring", async () => {
  const disk = expertDiskReader({ paths, threads: 3 });   // chunkBytes 1 MiB: every test part fits
  try {
    for (const info of experts) {
      const all = await Promise.all(Array.from({ length: info.shape[0] }, (_, e) => disk.readPart(info, e)));
      all.forEach((got, e) => {
        same(got, want(info, e), `${info.name} expert ${e}`);
        assert.ok(!(got.qs.buffer instanceof SharedArrayBuffer) && !(got.sc.buffer instanceof SharedArrayBuffer));
        assert.equal(got.qs.byteOffset, 0); assert.equal(got.qs.buffer.byteLength, got.qs.byteLength, "a buffer of its own");
      });
    }
    // Q4_0 / Q8_0 parts: one job each; requantized parts: split over the workers, then one gather job each
    const direct = experts.filter((t) => t.ggmlType === GGML_Q4_0 || t.ggmlType === GGML_Q8_0).reduce((a, t) => a + t.shape[0], 0);
    assert.equal(disk.stats.parts, experts.reduce((a, t) => a + t.shape[0], 0));
    assert.ok(disk.stats.chunks > disk.stats.parts && direct > 0);
    const file = fs.readFileSync(paths[0]);
    const got = await disk.readAt(0, 3, 5000);
    assert.equal(got.buffer.byteLength, 5000, "readAt: the worker's buffer, not a copy of a slice");
    assert.deepEqual(Buffer.from(got), file.subarray(3, 5003));
  } finally { await disk.close(); }
});

test("expertDiskReader: readAt returns the file's bytes; chunks from a few bytes to whole experts; slots smaller than a read", async () => {
  const file = fs.readFileSync(paths[1]);
  // slotBytes 4096 (one page): a 10000-byte readAt goes through three slots, a Q6_K expert (6 KB converted) gets a buffer of its own
  for (const [chunkBytes, slotBytes] of [[7, undefined], [4096, 4096], [2 ** 20, undefined], [1000, 4096]]) {
    const disk = expertDiskReader({ paths, threads: 2, chunkBytes, slotBytes, ringSlots: 2 });
    try {
      const info = tensors["blk.1.ffn_down_exps.weight"];
      const got = await disk.readAt(1, info.byteOffset + 5, 10000);
      assert.ok(!(got.buffer instanceof SharedArrayBuffer), "readAt hands out plain memory");
      assert.deepEqual(Buffer.from(got), file.subarray(info.byteOffset + 5, info.byteOffset + 10005));
      for (let e = 0; e < 3; e++) same(await disk.readPart(info, e), want(info, e), `chunkBytes ${chunkBytes}, slotBytes ${slotBytes}`);
      if (slotBytes) assert.equal(disk.stats.oversize, 3);
    } finally { await disk.close(); }
  }
});

test("sharedReads: a copy that throws fails only its own waiter, the others still settle", async () => {
  let joins = 0, copies = 0;
  const shared = sharedReads(() => joins++);
  let go;
  const make = () => new Promise((r) => { go = r; });
  const copy = (v) => { if (++copies === 2) throw new RangeError("Array buffer allocation failed"); return v.slice(); };
  const ps = Array.from({ length: 4 }, () => shared("k", make, copy));
  assert.equal(joins, 3);
  go(new Uint8Array([1, 2, 3]));
  const rs = await Promise.allSettled(ps);
  assert.deepEqual(rs.map((r) => r.status), ["fulfilled", "fulfilled", "rejected", "fulfilled"]);
  assert.ok(rs[2].reason instanceof RangeError);
  for (const i of [0, 1, 3]) assert.deepEqual([...rs[i].value], [1, 2, 3]);
  assert.notEqual(rs[1].value, rs[0].value, "a waiter gets a copy, not the first caller's buffer");
  // the key is free again: the next request starts a read of its own
  const again = shared("k", () => Promise.resolve(new Uint8Array([9])), copy);
  assert.deepEqual([...(await again)], [9]);
  assert.equal(joins, 3);
});

test("expertDiskReader: overlapping reads of one expert share one read; leases hold their slot", async () => {
  const disk = expertDiskReader({ paths, threads: 2, chunkBytes: 256, ringSlots: 2 });
  try {
    const info = tensors["blk.0.ffn_gate_exps.weight"];
    const [a, b, c] = await Promise.all([disk.readPart(info, 3), disk.readPart(info, 3), disk.readPart(info, 3)]);
    assert.equal(disk.stats.dedup, 2); assert.equal(disk.stats.parts, 1);
    for (const x of [a, b, c]) same(x, want(info, 3), "shared read");
    assert.ok(!(a.qs.buffer instanceof SharedArrayBuffer), "readPart hands out copies, not views of the ring");
    // a lease is a view of a ring slot until release(): with two slots, a third read waits for one
    const l1 = await disk.leasePart(info, 0), l2 = await disk.leasePart(info, 1);
    let third = null;
    const p3 = disk.leasePart(info, 2).then((l) => { third = l; });
    await new Promise((r) => setTimeout(r, 50));
    assert.equal(third, null, "no free slot yet");
    same(l1, want(info, 0), "lease 0"); same(l2, want(info, 1), "lease 1");
    l1.release(); l1.release();   // twice is once
    await p3;
    same(third, want(info, 2), "lease 2");
    l2.release(); third.release();
    const r1 = await disk.readAt(0, 100, 64), r2 = await disk.readAt(0, 100, 64);
    assert.deepEqual(Buffer.from(r1), Buffer.from(r2));
  } finally { await disk.close(); }
});

test("expertDiskReader: a short read fails that request only; close() stops the workers and fails what is left", async () => {
  const trunc = path.join(dir, "trunc.gguf");
  fs.writeFileSync(trunc, fs.readFileSync(paths[0]).subarray(0, tensors["blk.0.ffn_gate_exps.weight"].byteOffset + 3 * 1440 + 100));
  const disk = expertDiskReader({ paths: [trunc], threads: 2, chunkBytes: 512 });
  const info = tensors["blk.0.ffn_gate_exps.weight"];
  await assert.rejects(disk.readPart(info, 3), /short read/);
  same(await disk.readPart(info, 1), want(info, 1), "after a failed read");
  await assert.rejects(disk.readAt(5, 0, 10), /no file for shard 5/);
  await disk.close();
  await disk.close();   // twice is fine
  await assert.rejects(disk.readPart(info, 0), /closed/);
  assert.throws(() => expertDiskReader({ paths: [] }), /paths/);
});

// a worker that runs expert_disk_worker.js but first does what a job's offset says: OFF_EXIT exits the thread
// (a crash mid-job), OFF_SLOW blocks it for 300 ms (a read still in flight)
const OFF_EXIT = 7777, OFF_SLOW = 8888;
const testWorker = path.join(dir, "test_worker.mjs");
fs.writeFileSync(testWorker, `import { parentPort } from "node:worker_threads";
parentPort.on("message", (m) => {
  if (m.off === ${OFF_EXIT}) process.exit(3);
  if (m.off === ${OFF_SLOW}) Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 300);
});
await import(${JSON.stringify(new URL("../expert_disk_worker.js", import.meta.url).href)});
`);
const testWorkerURL = pathToFileURL(testWorker);
// p settles within ms, or the test fails as hung
const within = (p, ms, what) => Promise.race([p, new Promise((_, reject) => setTimeout(() => reject(new Error(`${what}: still pending after ${ms} ms`)), ms).unref())]);

test("expertDiskReader: a worker that dies mid-job fails that read only and is replaced, up to maxRespawns", async () => {
  const disk = expertDiskReader({ paths, threads: 2, chunkBytes: 4096, ringSlots: 2, maxRespawns: 2, workerURL: testWorkerURL });
  const info = tensors["blk.0.ffn_gate_exps.weight"];
  try {
    same(await disk.readPart(info, 0), want(info, 0), "before");
    await assert.rejects(within(disk.readAt(0, OFF_EXIT, 16), 2000, "read on a dying worker"), /exited \(3\)/);
    assert.equal(disk.stats.lost, 1); assert.equal(disk.stats.respawns, 1);
    for (let e = 0; e < 6; e++) same(await within(disk.readPart(info, e), 2000, "after a respawn"), want(info, e), `after a respawn, expert ${e}`);
    // two more deaths: the second respawn, then no more; one worker is left and still reads
    await assert.rejects(within(disk.readAt(0, OFF_EXIT, 16), 2000, "2nd death"), /exited/);
    await assert.rejects(within(disk.readAt(0, OFF_EXIT, 16), 2000, "3rd death"), /exited/);
    assert.equal(disk.stats.lost, 3); assert.equal(disk.stats.respawns, 2);
    same(await within(disk.readPart(info, 5), 2000, "one worker left"), want(info, 5), "one worker left");
    // the last one dies: every request after it fails at once, and none of them keeps a ring slot
    await assert.rejects(within(disk.readAt(0, OFF_EXIT, 16), 2000, "last death"), /exited/);
    for (let i = 0; i < 5; i++) {   // more than ringSlots: a leaked slot would hang the third
      await assert.rejects(within(disk.readPart(info, i % 6), 2000, `request ${i} after the pool died`), /no worker left/);
      await assert.rejects(within(disk.readAt(0, 64 * i, 64), 2000, `readAt ${i} after the pool died`), /no worker left/);
    }
  } finally { await disk.close(); }
});

test("expertDiskReader: a pool that cannot start (no worker file) fails every request at once, not only the first", async () => {
  const disk = expertDiskReader({ paths, threads: 1, ringSlots: 2, workerURL: pathToFileURL(path.join(dir, "missing_worker.mjs")) });
  const info = tensors["blk.0.ffn_gate_exps.weight"];
  try {
    const rs = await within(Promise.allSettled([disk.readPart(info, 0), disk.readPart(info, 1), disk.readAt(0, 0, 64)]), 3000, "the first requests");
    for (const r of rs) assert.equal(r.status, "rejected");
    for (let i = 0; i < 4; i++) await assert.rejects(within(disk.readPart(info, i), 1000, `request ${i} later`), /no worker left/);
    assert.equal(disk.stats.respawns, 0, "a worker that never ran a job is not restarted");
  } finally { await disk.close(); }
});

test("expertDiskReader: close() fails a read in flight and one waiting for a slot; leases held stay readable", async () => {
  const disk = expertDiskReader({ paths, threads: 2, chunkBytes: 4096, ringSlots: 2, workerURL: testWorkerURL });
  const info = tensors["blk.0.ffn_gate_exps.weight"];
  const held = await disk.leasePart(info, 0), held2 = await disk.leasePart(info, 3);   // both slots
  const slow = disk.readAt(0, OFF_SLOW, 16).then(() => null, (e) => e);   // in a worker for 300 ms (one job, no slot)
  await new Promise((r) => setTimeout(r, 50));
  const waiting = disk.leasePart(info, 1).then(() => null, (e) => e);   // no slot left
  await new Promise((r) => setTimeout(r, 20));
  assert.equal(disk.stats.slotWaits, 1);
  await within(disk.close(), 3000, "close with a lease held");
  assert.match((await waiting)?.message, /closed/, "the request waiting for a slot");
  assert.match((await slow)?.message, /closed/, "the read in flight");
  same(held, want(info, 0), "a lease held across close()");
  held.release(); held.release(); held2.release();
  await assert.rejects(disk.leasePart(info, 2), /closed/);
  await assert.rejects(disk.readAt(0, 0, 16), /closed/);
});

test("expertDiskReader: callers that share one read get buffers of their own; methods work unbound", async () => {
  const disk = expertDiskReader({ paths, threads: 2 });
  try {
    const { readAt, readPart, readExpert } = disk;
    const [a, b] = await Promise.all([readAt(0, 100, 64), readAt(0, 100, 64)]);
    assert.equal(disk.stats.dedup, 1);
    assert.notEqual(a.buffer, b.buffer);
    assert.deepEqual(Buffer.from(a), Buffer.from(b));
    a.fill(0);
    assert.notDeepEqual(Buffer.from(b), Buffer.alloc(64), "a change to one caller's bytes is not the other's");
    const trio = ["blk.1.ffn_up_exps.weight", "blk.1.ffn_gate_exps.weight", "blk.1.ffn_down_exps.weight"].map((n) => tensors[n]);
    (await readExpert(trio, 1)).forEach((got, j) => same(got, want(trio[j], 1), `unbound readExpert ${trio[j].name}`));
    same(await readPart(trio[0], 0), want(trio[0], 0), "unbound readPart");
  } finally { await disk.close(); }
});

test("defaultDiskThreads: POOLED_DISK_THREADS, else up to 16", () => {
  assert.equal(defaultDiskThreads({ POOLED_DISK_THREADS: "12" }), 12);
  assert.equal(defaultDiskThreads({ POOLED_DISK_THREADS: "999" }), 64);
  const d = defaultDiskThreads({});
  assert.ok(d >= 1 && d <= 16);
  assert.equal(defaultDiskThreads({ POOLED_DISK_THREADS: "nope" }), d);
});
