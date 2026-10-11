// A worker of expert_disk.js's pool: reads byte ranges of the model's GGUF files with its own file descriptors
// (positional readSync, one fd per shard, opened on first use) and, for an expert's rows, converts them with
// engine/gguf.js convertEntry, the function the engine's loader runs. The result goes straight into the
// pool's SharedArrayBuffer at the offsets the job names; only a short reply goes back.
import fs from "node:fs";
import { workerData, parentPort } from "node:worker_threads";
import { convertEntry } from "../../engine/gguf.js";

const { paths } = workerData;
const fds = new Map();   // shard -> fd
let scratch = new Uint8Array(0);   // raw rows before conversion, reused

function fdOf(shard) {
  let fd = fds.get(shard);
  if (fd === undefined) {
    if (!paths[shard]) throw new Error(`no file for shard ${shard} (${paths.length} files)`);
    fd = fs.openSync(paths[shard], "r");
    fds.set(shard, fd);
  }
  return fd;
}

// len bytes at off of the shard into out (a view of len bytes), or an error naming where it fell short
function readFully(shard, off, out) {
  const fd = fdOf(shard);
  let got = 0;
  while (got < out.byteLength) {
    const n = fs.readSync(fd, out, got, out.byteLength - got, off + got);
    if (n <= 0) throw new Error(`short read in ${paths[shard]}: ${got} of ${out.byteLength} B at ${off}`);
    got += n;
  }
}

parentPort.on("message", (m) => {
  if (m.close) {
    for (const fd of fds.values()) try { fs.closeSync(fd); } catch {}
    fds.clear();
    parentPort.postMessage({ id: m.id, closed: true });
    return;
  }
  const { id, shard, off, len, dst, dstOff, conv } = m;
  try {
    const t0 = performance.now();
    if (!conv) {   // raw bytes: read straight into the shared buffer
      readFully(shard, off, new Uint8Array(dst, dstOff, len));
      parentPort.postMessage({ id, readMs: performance.now() - t0, convMs: 0 });
      return;
    }
    if (scratch.byteLength < len) scratch = new Uint8Array(len);
    const raw = scratch.subarray(0, len);
    readFully(shard, off, raw);
    const t1 = performance.now();
    // conv.rows whole rows of one expert: Q4_0 / Q8_0 blocks and the Q8 requant never span rows, so these
    // bytes are exactly the whole tensor's conversion at this row range
    const { ggmlType, rows, cols, scOff } = conv;
    const e = convertEntry({ ggmlType, shape: [rows, cols], nElems: rows * cols }, raw);
    if (e.kind !== "q4" && e.kind !== "q8") throw new Error(`ggml type ${ggmlType} converts to ${e.kind}, not q4 / q8`);
    new Uint8Array(dst, dstOff, e.qs.byteLength).set(e.qs);
    // the f16 scales of these rows' blocks only: the u32 padding of an odd count belongs to the next chunk
    const scB = (rows * cols / 32) * 2;
    new Uint8Array(dst, scOff, scB).set(new Uint8Array(e.scales.buffer, e.scales.byteOffset, scB));
    parentPort.postMessage({ id, readMs: t1 - t0, convMs: performance.now() - t1 });
  } catch (e) {
    parentPort.postMessage({ id, error: String(e?.message || e) });
  }
});
