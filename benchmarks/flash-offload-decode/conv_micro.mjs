// How long a disk worker's Q4_0 conversion of one Flash expert part takes (engine/gguf.js q4Repack, as
// expert_disk_worker.js runs it), and how much of that is the fresh output buffers: THREADS workers at once, each
// converting N parts of PART bytes (default 0.88 MiB, a Flash gate / up part), fresh output per part (as now) or into
// one reused output. Run with --no-maglev on Windows, as a room node does.
import { Worker, isMainThread, workerData, parentPort } from "node:worker_threads";
import { q4Repack } from "../../engine/gguf.js";
const PART = +(process.env.PART || 18 * 51200), N = +(process.env.N || 400);
if (isMainThread) {
  for (const threads of (process.env.THREADS || "1,16").split(",").map(Number)) for (const mode of ["fresh", "reuse"]) {
    const ws = Array.from({ length: threads }, () => new Worker(new URL(import.meta.url), { workerData: { mode } }));
    const rs = await Promise.all(ws.map((w) => new Promise((r) => w.once("message", r))));
    ws.forEach((w) => w.terminate());
    const ms = rs.map((r) => r.ms / N).sort((a, b) => a - b);
    console.log(JSON.stringify({ threads, mode, msPerPartMedian: +ms[Math.floor(ms.length / 2)].toFixed(3), msPerPartMax: +ms.at(-1).toFixed(3) }));
  }
} else {
  const raw = new Uint8Array(PART); for (let i = 0; i < PART; i++) raw[i] = (i * 2654435761) >>> 24;
  const info = { nElems: (PART / 18) * 32 };
  const qs = new Uint8Array(info.nElems / 2), sc = new Uint32Array(Math.ceil(info.nElems / 64));
  // q4Repack's loop into outputs made once
  const into = (bytes) => {
    const nb = info.nElems / 32, src = new Uint16Array(bytes.buffer, bytes.byteOffset, nb * 9), dst = new Uint16Array(qs.buffer), sc16 = new Uint16Array(sc.buffer);
    for (let b = 0, s = 0, d = 0; b < nb; b++, s += 9, d += 8) {
      sc16[b] = src[s];
      dst[d] = src[s + 1]; dst[d + 1] = src[s + 2]; dst[d + 2] = src[s + 3]; dst[d + 3] = src[s + 4];
      dst[d + 4] = src[s + 5]; dst[d + 5] = src[s + 6]; dst[d + 6] = src[s + 7]; dst[d + 7] = src[s + 8];
    }
  };
  const one = workerData.mode === "reuse" ? into : (b) => q4Repack(info, b);
  for (let i = 0; i < 20; i++) one(raw);   // warm
  const t0 = performance.now();
  for (let i = 0; i < N; i++) one(raw);
  parentPort.postMessage({ ms: performance.now() - t0 });
}
