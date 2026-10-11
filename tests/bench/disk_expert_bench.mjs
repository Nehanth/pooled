// Expert offload's disk tier (roadmap/32-ssd-expert-offload.md, step 2): how fast a room node reads routed experts
// from the model's GGUF files the way decode asks for them. Per token, per layer, K distinct random experts at once
// (each its gate, up and down parts), the layers one after another; GB/s over the pass and per-expert latency
// (issue to all three parts in memory) at several worker counts, cold (the files' pages dropped from the page cache
// first) and warm (the same sequence again, now cached).
//
//   node tests/bench/disk_expert_bench.mjs [model dir or .gguf files]   (default ~/.pooled/models/qwen3.6-35b-moe)
//   THREADS=1,2,4,8,16  MODES=uv,raw,conv  GB=6 (bytes read per pass: sets the token count; TOKENS=n instead)
//   K=n (default the header's expert_used_count)  CHUNK_KB=1024  COLD=0|1  WARM=0|1  SEED=1  GPU=1 (the upload probes below)
//
// Modes:
//   uv    fs.promises reads, one per part (what source.js does today): libuv's thread pool, UV_THREADPOOL_SIZE
//         (default 4) reads at a time; THREADS does not apply
//   raw   packages/room-node/expert_disk.js readAt: the three parts' bytes in CHUNK_KB pieces over the worker pool,
//         no conversion (IO alone)
//   conv  expert_disk.js readPart: the same reads, converted in the workers to the engine's layout (engine/gguf.js
//         convertEntry), copied out of the staging ring: what the engine's disk tier gets
//
// Cold needs Linux: `dd iflag=nocache count=0` (posix_fadvise DONTNEED, no root) drops a file's clean pages before
// each cold pass. Pages another process has mapped stay; run it with nothing else reading the model. Elsewhere only
// warm passes run.
//
// GPU=1 also answers what the T1 cache (step 3) needs from Dawn: whether queue.writeBuffer takes a view of a
// SharedArrayBuffer (the staging ring), the memcpy rate from the ring into a mapped MAP_WRITE buffer, mapAsync(WRITE)
// latency for one expert-sized buffer, and how many expert-sized MAP_WRITE buffers can be made and mapped at once.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { pathToFileURL } from "node:url";
import { parseGGUFHeader } from "../../engine/gguf.js";
import { expertDiskReader, expertPart } from "../../packages/room-node/expert_disk.js";

const env = (k, d) => process.env[k] ?? d;
const list = (k, d) => env(k, d).split(",").map((s) => s.trim()).filter(Boolean);
const MiB = 2 ** 20, GiB = 2 ** 30;

// the model's files: a directory's .gguf files in name order (gguf-split shards sort by their -0000N-of-), not MTP heads
function modelFiles(args) {
  const one = (p) => (fs.statSync(p).isDirectory()
    ? fs.readdirSync(p).filter((f) => f.endsWith(".gguf") && !/^mtp-/.test(f)).sort().map((f) => path.join(p, f)) : [p]);
  const files = (args.length ? args : [path.join(os.homedir(), ".pooled/models/qwen3.6-35b-moe")]).flatMap(one);
  if (!files.length) throw new Error("no .gguf files");
  return files;
}
function header(file) {
  const fd = fs.openSync(file, "r");
  try {
    for (let size = 16 * MiB; ; size *= 2) {
      const b = Buffer.alloc(size), n = fs.readSync(fd, b, 0, size, 0);
      try { return parseGGUFHeader(b.buffer.slice(b.byteOffset, b.byteOffset + n), { skipTokenizer: true }); }
      catch (e) { if (size >= 512 * MiB || n < size) throw e; }
    }
  } finally { fs.closeSync(fd); }
}

const files = modelFiles(process.argv.slice(2));
const heads = files.map(header);
const meta = heads[0].meta, arch = meta["general.architecture"];
const tensors = {};
heads.forEach((h, s) => { for (const [k, v] of Object.entries(h.tensors)) tensors[k] = { ...v, shard: s }; });
// layer -> its routed experts' parts ([gate, up, down], or [gate_up, down] where a file fuses them). An MTP head's
// layer (blk.N.nextn.*, the 35B's layer 40) is left out: decode's trunk does not read it
const layers = [];
for (let l = 0; tensors[`blk.${l}.attn_norm.weight`] || tensors[`blk.${l}.ffn_down_exps.weight`]; l++) {
  if (Object.keys(tensors).some((k) => k.startsWith(`blk.${l}.nextn.`))) continue;
  const parts = ["gate", "up", "gate_up", "down"].map((p) => tensors[`blk.${l}.ffn_${p}_exps.weight`]).filter(Boolean);
  if (parts.length >= 2) layers.push({ l, parts });
}
if (!layers.length) throw new Error("no routed expert tensors (blk.N.ffn_{gate,up,down}_exps.weight) in " + files.join(", "));
const nExp = layers[0].parts[0].shape[0];
const K = +env("K", meta[`${arch}.expert_used_count`] || 8);
const expBytes = (L) => L.parts.reduce((a, t) => a + expertPart(t, 0).len, 0);
const perToken = layers.reduce((a, L) => a + K * expBytes(L), 0);
const TOKENS = +env("TOKENS", Math.max(2, Math.ceil((+env("GB", 6) * GiB) / perToken)));
const THREADS = list("THREADS", "1,2,4,8,16").map(Number), MODES = list("MODES", "uv,raw,conv");
const CHUNK = +env("CHUNK_KB", 1024) * 1024;
const COLD = env("COLD", process.platform === "linux" ? "1" : "0") === "1" && process.platform === "linux", WARM = env("WARM", "1") === "1";

// the routing a pass reads: per token, per layer, K distinct experts (a fixed seed: every configuration reads the same)
function routing(seed) {
  let s = seed >>> 0 || 1;
  const rnd = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 2 ** 32; };
  const out = [];
  for (let t = 0; t < TOKENS; t++) for (const L of layers) {
    const pick = new Set();
    while (pick.size < Math.min(K, nExp)) pick.add(Math.floor(rnd() * nExp));
    out.push({ L, ids: [...pick] });
  }
  return out;
}
const ROUTE = routing(+env("SEED", 1));

function dropCache() {
  for (const f of files) execFileSync("dd", [`if=${f}`, "iflag=nocache", "count=0", "status=none"]);
}

// one pass over ROUTE -> { s, bytes, lat: per-expert ms, layer: per-layer ms }
async function pass(read) {
  const lat = [], layer = [];
  let bytes = 0;
  const t0 = performance.now();
  for (const { L, ids } of ROUTE) {
    const l0 = performance.now();
    await Promise.all(ids.map(async (e) => {
      const a = performance.now();
      const n = await read(L, e);   // not bytes += await ...: that reads bytes before the await and loses the others' adds
      bytes += n;
      lat.push(performance.now() - a);
    }));
    layer.push(performance.now() - l0);
  }
  return { s: (performance.now() - t0) / 1000, bytes, lat, layer };
}
const pct = (a, p) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))]; };

// mode, worker count -> a reader (L, e) -> bytes read, and its close()
function reader(mode, threads) {
  if (mode === "uv") {
    const fhs = files.map((f) => fs.promises.open(f, "r"));
    const readRange = async (shard, off, len) => {
      const fh = await fhs[shard], b = Buffer.allocUnsafe(len);
      let o = 0;
      while (o < len) { const { bytesRead } = await fh.read(b, o, len - o, off + o); if (bytesRead <= 0) throw new Error("short read"); o += bytesRead; }
      return len;
    };
    return {
      read: async (L, e) => (await Promise.all(L.parts.map((t) => { const p = expertPart(t, e); return readRange(p.shard, p.off, p.len); }))).reduce((a, b) => a + b, 0),
      close: async () => { for (const fh of fhs) await (await fh).close(); },
      extra: () => "",
    };
  }
  const disk = expertDiskReader({ paths: files, threads, chunkBytes: CHUNK });
  // the workers started and their read path compiled before the clock starts (4 KiB of the header: no expert's pages)
  const ready = Promise.all(Array.from({ length: threads * 4 }, () => disk.readAt(0, 0, 4096)));
  const read = mode === "raw"
    ? async (L, e) => (await Promise.all(L.parts.map((t) => { const p = expertPart(t, e); return disk.readAt(p.shard, p.off, p.len).then((b) => b.byteLength); }))).reduce((a, b) => a + b, 0)
    : async (L, e) => { await disk.readExpert(L.parts, e); return expBytes(L); };
  return {
    read, close: () => disk.close(), ready,
    // per expert: worker read and convert time (summed over its chunks), the copy out of the ring
    extra: () => {
      const n = disk.stats.parts / 3 || disk.stats.raw / 3 || 1, st = disk.stats;
      return mode === "conv" ? ` | ${(st.convMs / n).toFixed(2)} | ${(st.copyMs / n).toFixed(2)}` : ` | | `;
    },
    reset: () => { for (const k of Object.keys(disk.stats)) disk.stats[k] = 0; },
  };
}

const GBs = (r) => (r.bytes / 1e9 / r.s).toFixed(2);
const row = (mode, threads, kind, r, x) =>
  console.log(`| ${mode} | ${mode === "uv" ? `libuv ${process.env.UV_THREADPOOL_SIZE || 4}` : threads} | ${kind} | ${GBs(r)} | ${pct(r.lat, 50).toFixed(2)} | ${pct(r.lat, 99).toFixed(2)} | ${pct(r.layer, 50).toFixed(2)} | ${(r.s * 1000 / TOKENS).toFixed(0)}${x} |`);

const sizeGB = files.reduce((a, f) => a + fs.statSync(f).size, 0) / 1e9;
console.log(`${files.map((f) => path.basename(f)).join(", ")} (${sizeGB.toFixed(1)} GB): ${layers.length} MoE layers, ${nExp} experts, K = ${K}, ` +
  `${(expBytes(layers[0]) / MiB).toFixed(2)} MiB per expert (gate + up + down), ${(perToken / 1e9).toFixed(2)} GB per token if nothing is cached`);
console.log(`${TOKENS} tokens per pass (${(perToken * TOKENS / 1e9).toFixed(1)} GB, ${ROUTE.length * K} experts), chunks ${CHUNK / 1024} KiB, ` +
  `${os.cpus().length} CPUs, node ${process.version}, ${os.platform()} ${os.arch()}${COLD ? "" : " (no cold passes)"}\n`);
console.log("| mode | workers | cache | GB/s | p50 ms / expert | p99 ms / expert | p50 ms / layer | ms / token | convert ms / expert | copy ms / expert |");
console.log("|---|---|---|---|---|---|---|---|---|---|");
for (const mode of MODES) {
  for (const threads of mode === "uv" ? [0] : THREADS) {
    const R = reader(mode, threads);
    try {
      await R.ready;
      if (COLD) { dropCache(); R.reset?.(); row(mode, threads, "cold", await pass(R.read), R.extra()); }
      if (WARM) {
        if (!COLD) await pass(R.read);   // fill the cache first
        R.reset?.(); row(mode, threads, "warm", await pass(R.read), R.extra());
      }
    } finally { await R.close(); }
  }
}
if (COLD) dropCache();   // leave the page cache as it was before (empty of these files), not full of experts

if (env("GPU", "0") === "1") await gpuProbe();

// The upload side of a disk read (step 3's questions), on Dawn: npm "webgpu" from packages/room-node (WEBGPU=path)
async function gpuProbe() {
  const where = process.env.WEBGPU || path.resolve(new URL("../../packages/room-node/node_modules/webgpu/index.js", import.meta.url).pathname);
  const mod = await import(pathToFileURL(where).href);
  const { create, globals } = mod.default?.create ? mod.default : mod;
  Object.assign(globalThis, globals);
  const gpu = create([]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize } });
  const info = adapter.info || {};
  console.log(`\nGPU probe: ${info.vendor || "?"} ${info.architecture || ""} ${info.description || ""}`.trim());
  const size = Math.ceil(expBytes(layers[0]) / 4096) * 4096;
  const dst = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  // 1. writeBuffer from a view of a SharedArrayBuffer (the ring) vs a plain copy
  const sab = new SharedArrayBuffer(size), view = new Uint8Array(sab).fill(7);
  let sabOk = "yes";
  device.pushErrorScope("validation");
  try { device.queue.writeBuffer(dst, 0, view); } catch (e) { sabOk = `no (${String(e.message || e).split("\n")[0]})`; }
  const verr = await device.popErrorScope();
  if (verr) sabOk = `no (${verr.message.split("\n")[0]})`;
  await device.queue.onSubmittedWorkDone();
  console.log(`writeBuffer from a SharedArrayBuffer view: ${sabOk}`);
  // 2. writeBuffer rate for expert-sized writes, from the SAB view and from a plain ArrayBuffer
  for (const [name, src] of [["SAB view", view], ["ArrayBuffer", new Uint8Array(size).fill(7)]]) {
    if (name === "SAB view" && sabOk !== "yes") continue;
    const n = 200, t0 = performance.now();
    for (let i = 0; i < n; i++) device.queue.writeBuffer(dst, 0, src);
    await device.queue.onSubmittedWorkDone();
    console.log(`writeBuffer ${name}, ${(size / MiB).toFixed(2)} MiB x ${n}: ${((size * n) / 1e9 / ((performance.now() - t0) / 1000)).toFixed(2)} GB/s`);
  }
  // 3. mapAsync(WRITE) latency, then the memcpy from the ring into the mapped range
  const mb = device.createBuffer({ size, usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC, mappedAtCreation: false });
  const maps = [], cps = [];
  for (let i = 0; i < 50; i++) {
    const t0 = performance.now();
    await mb.mapAsync(GPUMapMode.WRITE);
    const t1 = performance.now();
    new Uint8Array(mb.getMappedRange()).set(view);
    const t2 = performance.now();
    mb.unmap();
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(mb, 0, dst, 0, size);
    device.queue.submit([enc.finish()]);
    maps.push(t1 - t0); cps.push(t2 - t1);
  }
  await device.queue.onSubmittedWorkDone();
  console.log(`mapAsync(WRITE) after a copy from it was submitted: p50 ${pct(maps, 50).toFixed(3)} ms, p99 ${pct(maps, 99).toFixed(3)} ms; ` +
    `ring -> mapped memcpy ${(size / 1e9 / (pct(cps, 50) / 1000)).toFixed(1)} GB/s (p50 ${pct(cps, 50).toFixed(3)} ms)`);
  // 4. how many expert-sized MAP_WRITE buffers, all mapped at once (T1's slots), up to GPU_SLOTS_GB
  const capGB = +env("GPU_SLOTS_GB", 8), bufs = [];
  device.pushErrorScope("out-of-memory");
  let failed = null;
  const t0 = performance.now();
  try {
    while (bufs.length * size < capGB * GiB) {
      const b = device.createBuffer({ size, usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
      bufs.push(b);
    }
  } catch (e) { failed = String(e.message || e).split("\n")[0]; }
  const oom = await device.popErrorScope();
  console.log(`MAP_WRITE expert slots mapped at creation: ${bufs.length} (${(bufs.length * size / GiB).toFixed(1)} GiB) in ${(performance.now() - t0).toFixed(0)} ms` +
    `${oom || failed ? `, stopped: ${oom?.message || failed}` : `, cap GPU_SLOTS_GB=${capGB} reached`}`);
  for (const b of bufs) b.destroy();
  mb.destroy(); dst.destroy(); device.destroy();
}
