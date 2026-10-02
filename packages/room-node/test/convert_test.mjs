// The room node's weight conversion pool (convert.js): a tensor requantized on worker threads, in row
// slices, is byte for byte what the inline conversion (engine/gguf.js convertEntry) makes.
import { test } from "node:test";
import assert from "node:assert/strict";
import { convertPool } from "../convert.js";
import { convertEntry, f32ToF16, GGML_Q4_1, GGML_Q4_0 } from "../../../engine/gguf.js";

function q41(rows, cols, seed = 7) {   // a Q4_1 tensor: per 32 values f16 d, f16 m, 16 bytes of nibbles
  const nb = (rows * cols) / 32, b = new Uint8Array(nb * 20), dv = new DataView(b.buffer);
  for (let i = 0; i < nb; i++) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    dv.setUint16(i * 20, f32ToF16(((seed % 1000) + 1) / 50000), true);
    dv.setUint16(i * 20 + 2, f32ToF16(-((seed >> 10) % 1000) / 20000), true);
    for (let j = 0; j < 16; j++) { seed = (seed * 1103515245 + 12345) & 0x7fffffff; b[i * 20 + 4 + j] = seed & 0xff; }
  }
  return { info: { ggmlType: GGML_Q4_1, shape: [rows, cols], nElems: rows * cols, byteLength: b.byteLength }, bytes: b };
}

test("convertPool: worker-thread requant equals the inline conversion", async () => {
  const pool = convertPool({ threads: 3, minBytes: 0 });
  try {
    for (const [rows, cols] of [[7, 64], [301, 256]]) {   // rows that don't divide evenly over the workers
      const { info, bytes } = q41(rows, cols);
      const want = convertEntry(info, bytes), got = await pool.convert(info, bytes);
      assert.equal(got.kind, "q8");
      assert.deepEqual(got.shape, want.shape);
      assert.deepEqual(Buffer.from(got.qs), Buffer.from(want.qs));
      assert.deepEqual(Buffer.from(got.scales.buffer), Buffer.from(want.scales.buffer));
    }
    // Q4_0 is repacked inline (fast), not sent to the pool
    const info = { ggmlType: GGML_Q4_0, shape: [2, 32], nElems: 64, byteLength: 36 };
    const bytes = new Uint8Array(36).fill(3);
    assert.deepEqual(Buffer.from((await pool.convert(info, bytes)).qs), Buffer.from(convertEntry(info, bytes).qs));
  } finally { await pool.close(); }
});

test("convertPool: converts inline when the workers can't start", async () => {
  const logs = [];
  const pool = convertPool({ threads: 2, minBytes: 0, workerURL: new URL("./no_such_worker.js", import.meta.url), log: (m) => logs.push(m) });
  const { info, bytes } = q41(4, 64);
  const got = await pool.convert(info, bytes);
  assert.deepEqual(Buffer.from(got.qs), Buffer.from(convertEntry(info, bytes).qs));
  assert.match(logs.join("\n"), /converting on the main thread/);
  await pool.close();
});
