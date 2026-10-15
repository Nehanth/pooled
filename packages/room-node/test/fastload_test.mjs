// fastload.js: a load that reads ahead and repacks on threads hands the loader the same entries the plain path makes
// (engine/gguf.js convertEntry), byte for byte, and gives its shared buffers back for the next tensors.
import { test } from "node:test";
import assert from "node:assert/strict";
import { convertPool } from "../convert.js";
import { fastLoad } from "../fastload.js";
import { convertEntry, f32ToF16, GGML_Q4_0, GGML_Q8_0, GGML_Q4_1, GGML_F32 } from "../../../engine/gguf.js";

// a toy model file: tensors of Q4_0 (stacked experts, 3-D), Q8_0, Q4_1 (requantized) and F32, back to back
function toyFile() {
  let seed = 11;
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff);
  const block = { [GGML_Q4_0]: [18, 2], [GGML_Q8_0]: [34, 2], [GGML_Q4_1]: [20, 4] };
  const specs = [
    ["blk.0.ffn_gate_exps.weight", GGML_Q4_0, [8, 24, 64]], ["blk.0.attn_qkv.weight", GGML_Q8_0, [40, 96]],
    ["blk.0.ffn_down_shexp.weight", GGML_Q4_1, [33, 64]], ["blk.0.norm.weight", GGML_F32, [64]],
    ["blk.1.ffn_up_exps.weight", GGML_Q4_0, [8, 24, 64]], ["blk.1.ffn_down_exps.weight", GGML_Q8_0, [8, 9, 64]],
  ];
  const tensors = [], parts = [];
  let off = 0;
  for (const [name, T, shape] of specs) {
    const n = shape.reduce((a, b) => a * b, 1);
    let b;
    if (T === GGML_F32) { b = new Uint8Array(n * 4); new Float32Array(b.buffer).forEach((_, i, a) => { a[i] = (rnd() % 1000) / 999; }); }
    else {
      const [bb, hdr] = block[T], nb = n / 32;
      b = new Uint8Array(nb * bb);
      const dv = new DataView(b.buffer);
      for (let i = 0; i < nb; i++) {
        dv.setUint16(i * bb, f32ToF16(((rnd() % 1000) + 1) / 40000), true);
        if (hdr === 4) dv.setUint16(i * bb + 2, f32ToF16(-(rnd() % 1000) / 30000), true);
        for (let j = hdr; j < bb; j++) b[i * bb + j] = rnd() & 0xff;
      }
    }
    const pad = (32 - (off % 32)) % 32;   // GGUF's alignment
    off += pad; parts.push([off, b]);
    tensors.push({ name, ggmlType: T, shape, nElems: n, byteOffset: off, byteLength: b.byteLength });
    off += b.byteLength;
  }
  const file = new Uint8Array(off);
  for (const [o, b] of parts) file.set(b, o);
  let reads = 0;
  const src = {
    readInto: async (shard, o, len, into) => { reads++; await new Promise((r) => setTimeout(r, 1)); into.set(file.subarray(o, o + len)); return into; },
    bytesOf: async (info) => file.slice(info.byteOffset, info.byteOffset + info.byteLength),
    get reads() { return reads; },
  };
  return { tensors, src };
}
// what engine/gguf.js ggufEntry hands the converter: stacked experts as one matrix
const as2d = (t) => (t.shape.length === 3 ? { ...t, shape: [t.shape[0] * t.shape[1], t.shape[2]] } : t);
const same = (got, want) => {
  assert.equal(got.kind, want.kind);
  assert.deepEqual(got.shape, want.shape);
  if (want.kind === "f32") { assert.deepEqual(Buffer.from(got.data.buffer, got.data.byteOffset, got.data.byteLength), Buffer.from(want.data.buffer, want.data.byteOffset, want.data.byteLength)); return; }
  assert.deepEqual(Buffer.from(got.qs.buffer, got.qs.byteOffset, got.qs.byteLength), Buffer.from(want.qs));
  assert.deepEqual(Buffer.from(got.scales.buffer, got.scales.byteOffset, got.scales.byteLength), Buffer.from(want.scales.buffer));
};

test("fastLoad: every entry is convertEntry's, read ahead in chunks and repacked on threads", async () => {
  const { tensors, src } = toyFile(), pool = convertPool({ threads: 3 });
  const G = { tensors: Object.fromEntries(tensors.map((t) => [t.name, t])), convert: pool.convert };
  try {
    // tiny thresholds so the toy tensors take every path: shared memory (Q4_0 / Q8_0), threads (Q4_1), inline (F32)
    const fl = fastLoad({ src, G, plan: tensors, pool, chunk: 700, minShared: 1000, aheadBytes: 4096 });
    for (let pass = 0; pass < 2; pass++) for (const t of tensors) {
      const info = as2d(t), bytes = await fl.bytesOf(info);
      const got = await fl.convert(info, bytes);
      same(got, convertEntry(info, await src.bytesOf(info)));
      got.qs = got.scales = null;   // parked or uploaded: the caller is done with them
      fl.release(got);
    }
    assert.equal(fl.stats.repacked, 4, "the Q4_0 / Q8_0 tensors went through the threads");
    assert.equal(fl.stats.requant, 1, "the Q4_1 tensor was requantized ahead");
    assert.ok(src.reads > tensors.length, `big tensors are read in pieces (${src.reads} reads)`);
    fl.close();
  } finally { await pool.close(); }
});

test("fastLoad: shared buffers come back to the pool and are reused", async () => {
  const { tensors, src } = toyFile(), pool = convertPool({ threads: 2 });
  const big = tensors.filter((t) => t.ggmlType === GGML_Q4_0);
  const G = { tensors: Object.fromEntries(tensors.map((t) => [t.name, t])), convert: pool.convert };
  try {
    const fl = fastLoad({ src, G, plan: [...big, ...big.map((t) => ({ ...t, byteOffset: t.byteOffset }))], pool, minShared: 1000, aheadBytes: 1 });
    const e0 = await fl.convert(as2d(big[0]), await fl.bytesOf(as2d(big[0])));
    const before = fl.stats.newBytes;
    e0.qs = e0.scales = null; fl.release(e0);
    const e1 = await fl.convert(as2d(big[1]), await fl.bytesOf(as2d(big[1])));
    same(e1, convertEntry(as2d(big[1]), await src.bytesOf(big[1])));
    assert.ok(fl.stats.newBytes - before <= 3 * 2 ** 20, `the second tensor reused the first one's buffers (${fl.stats.newBytes - before} new bytes)`);
    // an entry still in use is not taken back
    fl.release(e1);
    assert.ok(e1.qs, "not released while its bytes are held");
    fl.close();
  } finally { await pool.close(); }
});

test("fastLoad: a streamed model (no readInto) or a pool without threads loads as before", () => {
  const pool = convertPool({ threads: 1 });
  assert.equal(fastLoad({ src: { readInto: null }, G: {}, plan: [], pool: convertPool({ threads: 4 }) }), null);
  assert.equal(fastLoad({ src: { readInto: () => {} }, G: {}, plan: [], pool }), null);
});
