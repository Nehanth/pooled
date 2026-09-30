// streamRequantToGPU (engine/gguf.js): a K-quant / Q4_1 / Q5_0 tensor requantized to Q8 a few rows
// at a time on its way to the GPU must upload exactly the bytes the whole-tensor CPU path
// (convertEntry -> requantQ8Streaming) produces, for any network chunking, and must fail with a
// "short tensor" error (the retry trigger in room.js) when the body ends early (#207).
import { convertEntry, streamEntryToGPU, streamable, ggmlTypeBytes, f32ToF16,
  GGML_Q4_0, GGML_Q4_1, GGML_Q5_0, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0, GGML_F32 } from "../../engine/gguf.js";

const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const eq = (a, b) => { if (a.length !== b.length) return false; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false; return true; };

function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32); }
// random bytes, with every f16 that the formats read as a scale kept to a sane magnitude
function synth(type, rows, cols, seed) {
  const r = rng(seed);
  const bytes = new Uint8Array(ggmlTypeBytes(type, rows * cols));
  for (let i = 0; i < bytes.length; i++) bytes[i] = (r() * 256) | 0;
  const f16 = (off) => { const h = f32ToF16((r() - 0.5) * 0.02); bytes[off] = h & 255; bytes[off + 1] = h >> 8; };
  const blk = { [GGML_Q4_1]: [20, [0, 2]], [GGML_Q5_0]: [22, [0]], [GGML_Q5_K]: [176, [0, 2]], [GGML_Q6_K]: [210, [208]] }[type];
  for (let b = 0; b < bytes.length / blk[0]; b++) for (const o of blk[1]) f16(b * blk[0] + o);
  if (type === GGML_Q6_K) for (let b = 0; b < bytes.length / 210; b++) for (let j = 0; j < 16; j++) bytes[b * 210 + 192 + j] = ((r() * 64) | 0) - 32 & 255;
  return { info: { name: "t", ggmlType: type, shape: [rows, cols], nElems: rows * cols, byteLength: bytes.length, byteOffset: 0 }, bytes };
}
function mockDevice() {
  globalThis.GPUBufferUsage ??= { STORAGE: 0x80, COPY_DST: 0x8, COPY_SRC: 0x4 };
  const writes = [];
  return {
    writes,
    pushErrorScope() {}, popErrorScope: async () => null,
    createBuffer: ({ size }) => ({ mem: new Uint8Array(size), size }),
    queue: { writeBuffer(buf, off, src, srcOff = 0, size) {
      const s = ArrayBuffer.isView(src) ? new Uint8Array(src.buffer, src.byteOffset, src.byteLength) : new Uint8Array(src);
      const n = size ?? s.length - srcOff;
      if (off % 4 || n % 4) throw new Error(`writeBuffer alignment: offset ${off}, size ${n}`);
      if (off + n > buf.size) throw new Error(`writeBuffer range ${off}+${n}/${buf.size}`);
      buf.mem.set(s.subarray(srcOff, srcOff + n), off);
      writes.push(n);
    } },
  };
}
const body = (bytes, sizes, cut = 0) => async () => {
  let off = 0, i = 0;
  const end = bytes.length - cut;
  return { ok: true, status: 200, body: { getReader: () => ({ read: async () => {
    if (off >= end) return { done: true };
    const n = Math.min(end - off, sizes[i++ % sizes.length]);
    const v = bytes.slice(off, off + n); off += n;
    return { done: false, value: v };
  } }) } };
};

const TYPES = { Q4_1: GGML_Q4_1, Q5_0: GGML_Q5_0, Q5_K: GGML_Q5_K, Q6_K: GGML_Q6_K };

Deno.test("streamable: the quant types stream, float tensors stay on the CPU path", () => {
  for (const t of [GGML_Q4_0, GGML_Q8_0, ...Object.values(TYPES)]) ok(streamable(t), "type " + t);
  ok(!streamable(GGML_F32), "f32");
});

for (const [nm, type] of Object.entries(TYPES)) {
  Deno.test(`streamRequantToGPU ${nm}: identical to convertEntry for any chunking and staging size`, async () => {
    // 37 rows: an odd row count, so the last chunk is short; 7x96 and 3x32: odd block counts (padded scale writes)
    for (const [rows, cols] of [[37, 256], [18, 512], [5, 256], ...(type === GGML_Q4_1 || type === GGML_Q5_0 ? [[7, 96], [3, 32]] : [])]) {
      const { info, bytes } = synth(type, rows, cols, 11 + type + rows);
      const ref = convertEntry(info, bytes);
      ok(ref.kind === "q8", "reference is q8");
      for (const staging of [1, 3000, 16384, 1 << 22]) for (const sizes of [[1 << 20], [1, 17, 4099, 999, 333], [bytes.length]]) {
        const dev = mockDevice();
        const e = await streamEntryToGPU(dev, info, body(bytes, sizes), { staging });
        ok(e.kind === "q8" && e.gpu.kind === "q8", "q8 entry");
        ok(eq(e.gpu.qs.mem.subarray(0, ref.qs.length), ref.qs), `qs ${nm} ${rows}x${cols} staging ${staging}`);
        const rs = new Uint8Array(ref.scales.buffer, 0, rows * cols / 32 * 2);
        ok(eq(e.gpu.sc.mem.subarray(0, rs.length), rs), `scales ${nm} ${rows}x${cols} staging ${staging}`);
        // the peak JS it holds is one chunk, never the tensor: no single write is bigger than a chunk's output
        if (staging < bytes.length) ok(Math.max(...dev.writes) <= Math.max(2, staging / (cols * 4) | 0) * cols + 64, "chunked writes");
      }
    }
  });
}

Deno.test("streamRequantToGPU: a body that ends early is a 'short tensor' error (room.js retries on it)", async () => {
  const { info, bytes } = synth(GGML_Q5_K, 8, 256, 3);
  let err = null;
  try { await streamEntryToGPU(mockDevice(), info, body(bytes, [1000], 100)); } catch (e) { err = e; }
  ok(err && /short tensor/.test(String(err)), String(err));
});
