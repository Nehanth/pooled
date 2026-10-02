// A worker of convert.js's pool: requantizes rows of one tensor to Q8 (engine/gguf.js requantQ8Streaming,
// the same function the inline path runs) and hands the result back without a copy.
import { parentPort } from "node:worker_threads";
import { requantQ8Streaming } from "../../engine/gguf.js";

parentPort.on("message", ({ id, ggmlType, rows, cols, bytes }) => {
  try {
    const { qs, scales } = requantQ8Streaming({ ggmlType, shape: [rows, cols] }, new Uint8Array(bytes));
    parentPort.postMessage({ id, qs, scales }, [qs.buffer, scales.buffer]);
  } catch (e) { parentPort.postMessage({ id, error: String(e?.message || e) }); }
});
