// The drive under expert offload's disk tier, buffered against direct I/O (benchmarks/disk-tier-direct-io.md, step 1a):
// Flash-shaped expert parts (the real ffn_{gate,up,down}_exps part ranges of the model's GGUF files, random experts),
// one positional read per part, from N worker threads with a file descriptor per shard each (expert_disk_worker.js's
// shape), N = 1 / 4 / 16 / 32 in flight.
//   buffered  fs.openSync(file, "r"), fs.readSync of the part's exact range (what expert_disk_worker.js does today)
//   direct    opened with O_DIRECT (Linux: fs.constants.O_DIRECT) or FILE_FLAG_NO_BUFFERING (Windows: libuv's
//             UV_FS_O_DIRECT, 0x02000000, which uv_fs_open maps to it), reading the part's 4 KiB-aligned superset into
//             a sector-aligned view of a scratch buffer
// Every arm reads parts no earlier arm of this run (or of a run with another RUN index) read: a disjoint slice of one
// seeded shuffle of every part, so buffered arms are first-touch whatever ran before (the files are larger than RAM
// on the PC; Linux also drops each file's pages first). The direct arm's first parts are byte-checked against a buffered read.
//
//   node tests/bench/direct_io_probe.mjs <model dir or .gguf files>
//   ARMS=buffered:1,direct:1,buffered:16,direct:16   GB=2 (per arm)   RUN=0 (which disjoint slice set)   SEED=1
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { Worker, isMainThread, workerData, parentPort } from "node:worker_threads";

const MiB = 2 ** 20, GiB = 2 ** 30, SECTOR = 4096;
export const DIRECT_FLAG = process.platform === "win32" ? 0x02000000 : (fs.constants.O_DIRECT || 0);

// a view of len bytes of a fresh buffer whose address is sector-aligned: the address can't be read from JS, so try
// each 512-byte step until a direct read of one aligned sector of the file is accepted
function alignedScratch(fd, len) {
  const raw = Buffer.allocUnsafeSlow(len + SECTOR);
  for (let k = 0; k < SECTOR; k += 512) {
    const v = raw.subarray(k, k + len);
    try { fs.readSync(fd, v, 0, SECTOR, 0); return v; } catch (e) { if (e.code !== "EINVAL" && e.errno !== -22 && e.code !== "UNKNOWN") throw e; }
  }
  throw new Error("no sector-aligned view accepted a direct read");
}

if (!isMainThread) {
  const { paths, mode, parts, next, maxLen } = workerData;   // parts: Float64Array [shard, off, len]*; next: Int32Array counter
  const direct = mode === "direct";
  const fds = paths.map((p) => fs.openSync(p, direct ? fs.constants.O_RDONLY | DIRECT_FLAG : "r"));
  const buf = direct ? alignedScratch(fds[0], maxLen + 2 * SECTOR) : Buffer.allocUnsafeSlow(maxLen);
  const lat = [];
  let bytes = 0;
  for (let i; (i = Atomics.add(next, 0, 1)) < parts.length / 3; ) {
    const shard = parts[3 * i], off = parts[3 * i + 1], len = parts[3 * i + 2];
    const t0 = performance.now();
    if (direct) {
      const a = Math.floor(off / SECTOR) * SECTOR, alen = Math.ceil((off + len) / SECTOR) * SECTOR - a;
      let got = 0;
      while (got < alen) { const n = fs.readSync(fds[shard], buf, got, alen - got, a + got); if (n <= 0) break; got += n; }
      if (got < off + len - a) throw new Error(`short direct read at ${off}`);
    } else {
      let got = 0;
      while (got < len) { const n = fs.readSync(fds[shard], buf, got, len - got, off + got); if (n <= 0) throw new Error("short read"); got += n; }
    }
    lat.push(performance.now() - t0);
    bytes += len;
  }
  for (const fd of fds) fs.closeSync(fd);
  parentPort.postMessage({ lat, bytes });
} else {
  const { parseGGUFHeader } = await import("../../engine/gguf.js");
  const { expertPart } = await import("../../packages/room-node/expert_disk.js");
  const env = (k, d) => process.env[k] ?? d;
  const args = process.argv.slice(2);
  const one = (p) => (fs.statSync(p).isDirectory() ? fs.readdirSync(p).filter((f) => f.endsWith(".gguf") && !/^mtp-/.test(f)).sort().map((f) => path.join(p, f)) : [p]);
  const files = (args.length ? args : [path.join(os.homedir(), "llm/models/q38flash")]).flatMap(one);
  const header = (file) => {
    const fd = fs.openSync(file, "r");
    try {
      for (let size = 16 * MiB; ; size *= 2) {
        const b = Buffer.alloc(size), n = fs.readSync(fd, b, 0, size, 0);
        try { return parseGGUFHeader(b.buffer.slice(b.byteOffset, b.byteOffset + n), { skipTokenizer: true }); }
        catch (e) { if (size >= 512 * MiB || n < size) throw e; }
      }
    } finally { fs.closeSync(fd); }
  };
  const tensors = {};
  files.map(header).forEach((h, s) => { for (const [k, v] of Object.entries(h.tensors)) tensors[k] = { ...v, shard: s }; });
  const all = [];
  for (const [k, t] of Object.entries(tensors)) {
    if (!/^blk\.\d+\.ffn_(gate|up|gate_up|down)_exps\.weight$/.test(k) || t.shape?.length !== 3) continue;
    for (let e = 0; e < t.shape[0]; e++) { const p = expertPart(t, e); all.push([p.shard, p.off, p.len]); }
  }
  // one seeded shuffle; arm j of run r takes slice r * nArms + j
  let s = +env("SEED", 1) >>> 0 || 1;
  const rnd = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 2 ** 32; };
  for (let i = all.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [all[i], all[j]] = [all[j], all[i]]; }
  const ARMS = env("ARMS", "buffered:1,direct:1,buffered:16,direct:16").split(",").map((a) => { const [mode, n] = a.split(":"); return { mode, n: +n }; });
  const meanLen = all.reduce((a, p) => a + p[2], 0) / all.length;
  const per = Math.floor((+env("GB", 2) * 1e9) / meanLen), RUN = +env("RUN", 0);
  if ((RUN + 1) * ARMS.length * per > all.length) throw new Error(`not enough distinct parts: ${all.length} for ${(RUN + 1) * ARMS.length} slices of ${per}`);
  const maxLen = Math.max(...all.map((p) => p[2]));
  const pct = (a, p) => { const x = [...a].sort((u, v) => u - v); return x[Math.min(x.length - 1, Math.floor((p / 100) * x.length))]; };
  console.error(`${files.length} files, ${all.length} parts (mean ${(meanLen / MiB).toFixed(2)} MiB), ${per} parts per arm (${(per * meanLen / 1e9).toFixed(2)} GB), run ${RUN}, ${process.platform} node ${process.version}, direct flag 0x${DIRECT_FLAG.toString(16)}`);
  if (process.platform === "linux") for (const f of files) execFileSync("dd", [`if=${f}`, "iflag=nocache", "count=0", "status=none"]);

  // byte check: the direct path returns the same bytes as a buffered read
  {
    const fdB = fs.openSync(files[all[0][0]], "r"), fdD = fs.openSync(files[all[0][0]], fs.constants.O_RDONLY | DIRECT_FLAG);
    const [, off, len] = all[0], a = Math.floor(off / SECTOR) * SECTOR, alen = Math.ceil((off + len) / SECTOR) * SECTOR - a;
    const b = Buffer.alloc(len); fs.readSync(fdB, b, 0, len, off);
    const d = alignedScratch(fdD, alen); fs.readSync(fdD, d, 0, alen, a);
    console.error(`byte check (part at ${off}, ${len} B): ${Buffer.compare(b, d.subarray(off - a, off - a + len)) === 0 ? "identical" : "DIFFERENT"}`);
    fs.closeSync(fdB); fs.closeSync(fdD);
  }

  const rows = [];
  for (const [j, arm] of ARMS.entries()) {
    const slice = all.slice((RUN * ARMS.length + j) * per, (RUN * ARMS.length + j + 1) * per);
    const parts = new Float64Array(new SharedArrayBuffer(slice.length * 3 * 8));
    slice.forEach((p, i) => parts.set(p, 3 * i));
    const next = new Int32Array(new SharedArrayBuffer(4));
    const t0 = performance.now();
    const res = await Promise.all(Array.from({ length: arm.n }, () => new Promise((resolve, reject) => {
      const w = new Worker(new URL(import.meta.url), { workerData: { paths: files, mode: arm.mode, parts, next, maxLen } });
      w.on("message", resolve); w.on("error", reject);
    })));
    const sec = (performance.now() - t0) / 1000, bytes = res.reduce((a, r) => a + r.bytes, 0), lat = res.flatMap((r) => r.lat);
    const row = { slot: j + 1, mode: arm.mode, inflight: arm.n, GBps: +(bytes / 1e9 / sec).toFixed(2), p50: +pct(lat, 50).toFixed(2), p99: +pct(lat, 99).toFixed(2), GB: +(bytes / 1e9).toFixed(2), s: +sec.toFixed(1) };
    rows.push(row);
    console.log(JSON.stringify(row));
  }
  console.log(JSON.stringify({ run: RUN, platform: process.platform, node: process.version, rows }));
}
