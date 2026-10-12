// A worker of expert_disk.js's pool: reads byte ranges of the model's GGUF files with its own file descriptors
// (positional readSync, one fd per shard, opened on first use: on Windows reads through one shared handle run one at
// a time) and, for an expert's rows, converts them with engine/gguf.js convertEntry, the function the engine's loader
// runs. Three kinds of job:
//   own     the range (converted when conv is set) into buffers this worker makes, transferred with the reply
//   dst     into the pool's SharedArrayBuffer at the offsets the job names; only a short reply goes back
//   gather  [offset, length] ranges of a SharedArrayBuffer copied into buffers of this worker's own, transferred
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

// a typed array's bytes as an ArrayBuffer that is only them (the array's own when it spans all of it)
const whole = (v) => (v.byteOffset === 0 && v.byteLength === v.buffer.byteLength ? v.buffer : v.buffer.slice(v.byteOffset, v.byteOffset + v.byteLength));

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
  const { id, shard, off, len, dst, dstOff, conv, own } = m;
  try {
    const t0 = performance.now();
    if (m.gather) {
      const bufs = m.gather.map(([o, n]) => { const b = new ArrayBuffer(n); new Uint8Array(b).set(new Uint8Array(m.src, o, n)); return b; });
      parentPort.postMessage({ id, bufs, copyMs: performance.now() - t0 }, bufs);
      return;
    }
    if (!conv) {   // raw bytes: straight into a buffer of this worker's own, or into the shared one
      const buf = own ? new ArrayBuffer(len) : null;
      readFully(shard, off, own ? new Uint8Array(buf) : new Uint8Array(dst, dstOff, len));
      parentPort.postMessage({ id, buf, readMs: performance.now() - t0, convMs: 0 }, own ? [buf] : []);
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
    if (own) {   // convertEntry's own arrays (exactly qsBytes and scBytes, the scales padded to whole u32 words)
      const qs = whole(e.qs), sc = whole(e.scales);
      parentPort.postMessage({ id, qs, sc, readMs: t1 - t0, convMs: performance.now() - t1 }, [qs, sc]);
      return;
    }
    new Uint8Array(dst, dstOff, e.qs.byteLength).set(e.qs);
    // the f16 scales of these rows' blocks only: the u32 padding of an odd count belongs to the next chunk
    const scB = (rows * cols / 32) * 2;
    new Uint8Array(dst, scOff, scB).set(new Uint8Array(e.scales.buffer, e.scales.byteOffset, scB));
    parentPort.postMessage({ id, readMs: t1 - t0, convMs: performance.now() - t1 });
  } catch (e) {
    parentPort.postMessage({ id, error: String(e?.message || e) });
  }
});
