// Weight conversion off the event loop: the tensors the GPU has no kernel for (Q4_1, Q5_0, Q5_K,
// Q6_K) are requantized to Q8 on the CPU at load (engine/gguf.js convertEntry). In JS that is slow:
// one ffn_down_exps of the 35B MoE (Q4_1, 160 MB) took 3-6.6 s on the RTX 5070 PC, every layer,
// and it ran on the event loop, so a gateway or a room node stopped answering its peers meanwhile
// (a Mac peer dropped out of an OpenClaw room during a 13 s stall). Here a pool of worker threads
// takes row slices of the tensor; each runs requantQ8Streaming, the inline path's own function, and
// Q8 blocks never span rows, so the bytes are the same. Everything else (Q4_0 / Q8_0 repacks, f32)
// stays inline: it is fast. If the pool can't start (a bundle without the worker file, say), the
// tensor converts inline as before.
//   G.convert = convertPool().convert   (engine/gguf.js ggufEntry calls it when set)
import os from "node:os";
import { convertEntry, ggmlTypeBytes, GGML_F32, GGML_F16, GGML_Q8_0, GGML_Q4_0, GGML_BF16 } from "../../engine/gguf.js";

const inline = (t) => t === GGML_Q8_0 || t === GGML_Q4_0 || t === GGML_F32 || t === GGML_F16 || t === GGML_BF16;

export function convertPool({ threads = Math.max(1, Math.min(8, (os.availableParallelism?.() ?? os.cpus().length) - 2)),
  minBytes = 8 * 2 ** 20, workerURL = new URL("./convert_worker.js", import.meta.url), log = () => {} } = {}) {
  let workers = null, broken = false, seq = 0;
  const waiting = new Map();
  async function start() {
    const { Worker } = await import("node:worker_threads");
    workers = [];
    for (let i = 0; i < threads; i++) {
      const w = new Worker(workerURL);
      w.unref();   // an idle pool never holds the process open
      w.on("message", (m) => { const p = waiting.get(m.id); if (!p) return; waiting.delete(m.id); m.error ? p.reject(new Error(m.error)) : p.resolve(m); });
      w.on("error", (e) => { broken = true; for (const p of waiting.values()) p.reject(e); waiting.clear(); });
      workers.push(w);
    }
  }
  const run = (w, msg, transfer) => new Promise((resolve, reject) => { const id = ++seq; waiting.set(id, { resolve, reject }); w.postMessage({ id, ...msg }, transfer); });
  async function requant(info, bytes) {
    const [rows, cols] = info.shape, rowBytes = ggmlTypeBytes(info.ggmlType, cols);
    if (!workers) await start();
    const per = Math.ceil(rows / workers.length);
    const parts = [];
    for (let r0 = 0, k = 0; r0 < rows; r0 += per, k++) {
      const rc = Math.min(per, rows - r0);
      const slice = bytes.slice(r0 * rowBytes, (r0 + rc) * rowBytes);   // a copy the worker owns
      parts.push(run(workers[k], { ggmlType: info.ggmlType, rows: rc, cols, bytes: slice.buffer }, [slice.buffer]).then((m) => ({ r0, rc, ...m })));
    }
    const n = rows * cols, qs = new Uint8Array(n), scales = new Uint32Array(Math.ceil(n / 32 / 2)), sc16 = new Uint16Array(scales.buffer);
    for (const p of await Promise.all(parts)) {
      qs.set(p.qs, p.r0 * cols);
      sc16.set(new Uint16Array(p.scales.buffer, 0, (p.rc * cols) / 32), (p.r0 * cols) / 32);
    }
    return { kind: "q8", qs, scales, shape: info.shape };
  }
  return {
    // (info, bytes) -> the entry convertEntry(info, bytes) returns
    async convert(info, bytes) {
      const T = info.ggmlType;
      if (broken || info.shape.length !== 2 || inline(T) || info.byteLength < minBytes || threads < 2 || ggmlTypeBytes(T, info.shape[1]) < 0) return convertEntry(info, bytes);
      try { return await requant(info, bytes); }
      catch (e) { broken = true; log(`weight conversion pool unavailable (${String(e?.message || e).split("\n")[0]}): converting on the main thread`); return convertEntry(info, bytes); }
    },
    async close() { const ws = workers || []; workers = null; await Promise.all(ws.map((w) => w.terminate().catch(() => {}))); },
  };
}
