// The room node's disk reader for expert offload (expert_disk.js): one expert of a stacked GGUF tensor, read in
// chunks by a pool of worker threads and converted there, is byte for byte the slice the whole tensor's conversion
// (engine/gguf.js convertEntry, what the loader parks) has at that expert. A synthetic two-file GGUF in a temp dir,
// one tensor per ggml type the experts come in, so no model file is needed.
import { test, after } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { expertDiskReader, expertPart, defaultDiskThreads } from "../expert_disk.js";
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
  // a read in flight at close() either finished first or fails as closed
  const pending = disk.readPart(info, 2).then(() => null, (e) => e);
  await disk.close();
  await disk.close();   // twice is fine
  const err = await pending;
  if (err) assert.match(err.message, /closed/);
  await assert.rejects(disk.readPart(info, 0), /closed/);
  assert.throws(() => expertDiskReader({ paths: [] }), /paths/);
});

test("defaultDiskThreads: POOLED_DISK_THREADS, else up to 16", () => {
  assert.equal(defaultDiskThreads({ POOLED_DISK_THREADS: "12" }), 12);
  assert.equal(defaultDiskThreads({ POOLED_DISK_THREADS: "999" }), 64);
  const d = defaultDiskThreads({});
  assert.ok(d >= 1 && d <= 16);
  assert.equal(defaultDiskThreads({ POOLED_DISK_THREADS: "nope" }), d);
});
