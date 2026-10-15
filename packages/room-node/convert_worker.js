// A worker of convert.js's pool: requantizes rows of one tensor to Q8 (engine/gguf.js requantQ8Streaming,
// the same function the inline path runs) and hands the result back without a copy; or repacks blocks of a Q4_0 / Q8_0
// tensor in shared memory (q4RepackInto / q8RepackInto, the inline q4Repack / q8Repack's own loops).
import { parentPort } from "node:worker_threads";
import { requantQ8Streaming, q4RepackInto, q8RepackInto, GGML_Q4_0 } from "../../engine/gguf.js";

parentPort.on("message", ({ id, ggmlType, rows, cols, bytes, repack }) => {
  try {
    // repack (fastload.js): blocks [b0, b1) of a Q4_0 / Q8_0 tensor, its bytes and the outputs all in shared memory
    if (repack) {
      const { src, qs, sc, b0, b1 } = repack;
      (ggmlType === GGML_Q4_0 ? q4RepackInto : q8RepackInto)(new Uint8Array(src), new Uint8Array(qs), new Uint16Array(sc), b0, b1);
      parentPort.postMessage({ id });
      return;
    }
    const { qs, scales } = requantQ8Streaming({ ggmlType, shape: [rows, cols] }, new Uint8Array(bytes));
    parentPort.postMessage({ id, qs, scales }, [qs.buffer, scales.buffer]);
  } catch (e) { parentPort.postMessage({ id, error: String(e?.message || e) }); }
});
