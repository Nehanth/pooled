// Run tests/bench/opfs_read.html in Chrome: random OPFS / Cache API reads at Qwen3.8-Flash-Next expert and
// n-gram-row sizes, 1-8 workers, cold and warm, plus the read -> GPU paths. Prints one line per run and
// the whole result as JSON at the end (OUT=<file> also writes it there).
//   node tests/bench/opfs_read.mjs
// env: GB (test data in OPFS, 4), FILES (16), WORKERS (1,2,4,8), ROW_WORKERS (1,4,8,16: n-gram rows, a token
//      needs 16), CACHE_GB (1; 0 skips the Cache API),
//      READS (expert-sized reads per run, 384; parts 3x, rows 40x), COLD (1: drop the page cache before each
//      run), WARM (1: also run everything with the data in the page cache), KEEP (0: delete the test data
//      at the end; 1 keeps it for the next run), PROFILE (Chrome profile dir, kept between runs),
//      NATIVE (1: the same random reads from node worker threads on OPFS's own files, the no-browser baseline),
//      CHROME_BIN (a Chrome other than Playwright's, e.g. "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
//      SEED (of the random offsets; printed, by default from the clock, so a rerun on a KEEP=1 profile reads
//      other offsets), PLAN (comma-separated subset of expert,ro,part,row,native,cache; all by default)
//
// The profile must be on disk: an incognito / in-memory profile keeps OPFS in RAM (~1 GB/s, not a disk
// number), so this uses a persistent context, never newContext().
// Cold reads: on Linux the driver drops the profile's files from the page cache (sync, then
// posix_fadvise DONTNEED on every file under PROFILE; no root needed). On macOS it runs `sudo -n purge`
// (whole-system; needs a sudoers entry or a recent `sudo` in that terminal), and without it reports warm
// numbers only, marked so. Uses the GPU lightly (copies only).
import { chromium } from "playwright"; import { spawnSync } from "node:child_process"; import { Worker } from "node:worker_threads";
import { serveRepo } from "../e2e/engine_synth.mjs"; import { summary } from "./opfs_read.js";
import fs from "node:fs"; import os from "node:os"; import path from "node:path";
const env = (k, d) => process.env[k] ?? d;
const GB = +env("GB", 4), FILES = +env("FILES", 16), CACHE_GB = +env("CACHE_GB", 1), READS = +env("READS", 384);
const WORKERS = env("WORKERS", "1,2,4,8").split(",").map(Number), ROW_WORKERS = env("ROW_WORKERS", "1,4,8,16").split(",").map(Number), KEEP = env("KEEP", "0") === "1";
const PROFILE = env("PROFILE", path.join(os.tmpdir(), "pooled-opfs-probe-profile"));
const SEED = +env("SEED", Date.now() % 1e6), PLAN = new Set(env("PLAN", "expert,ro,part,row,native,cache").split(","));
const root = new URL("../..", import.meta.url).pathname;

// ---- page cache control ----
function evict() {
  if (process.platform === "linux") {
    const py = `import os,sys
os.sync()
n=0
for d,_,fs in os.walk(sys.argv[1]):
  for f in fs:
    p=os.path.join(d,f)
    try:
      fd=os.open(p,os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd); n+=1
    except OSError: pass
print(n)`;
    return spawnSync("python3", ["-c", py, PROFILE]).status === 0;
  }
  if (process.platform === "darwin") return spawnSync("sudo", ["-n", "purge"]).status === 0;
  return false;
}
function warm() {   // read every file of the profile once, so they are in the page cache
  const buf = Buffer.alloc(8 << 20);
  for (const d of fs.readdirSync(PROFILE, { recursive: true })) {
    const p = path.join(PROFILE, d); let fd;
    try { if (!fs.statSync(p).isFile()) continue; fd = fs.openSync(p, "r"); while (fs.readSync(fd, buf, 0, buf.length, null) > 0); } catch {} finally { if (fd !== undefined) fs.closeSync(fd); }
  }
}

// cross-origin isolated, so performance.now() in the tab and its workers is not rounded to 0.1 ms
// ---- native baseline: readSync from node worker threads, on the files Chrome stores the OPFS data in ----
const NATIVE_WORKER = `const { parentPort, workerData: { files, jobs, size, at } } = require("node:worker_threads"); const fs = require("node:fs");
const fds = files.map((f) => fs.openSync(f, "r")), buf = Buffer.alloc(size), now = () => performance.timeOrigin + performance.now();
while (now() < at);
const lat = new Float64Array(jobs.length / 2), t0 = now();
for (let i = 0; i < lat.length; i++) { const s = now(); if (fs.readSync(fds[jobs[2 * i]], buf, 0, size, jobs[2 * i + 1]) !== size) throw new Error("short read"); lat[i] = now() - s; }
parentPort.postMessage({ lat, t0, t1: now() }); for (const fd of fds) fs.closeSync(fd);`;
function opfsFiles(bytes) {   // the probe's files, found by size in the profile (Chrome names them by number)
  return fs.readdirSync(PROFILE, { recursive: true }).map((d) => path.join(PROFILE, d)).filter((p) => { try { const st = fs.statSync(p); return st.isFile() && st.size === bytes; } catch { return false; } });
}
async function nativeRun({ size, W, reads }, files, fileBytes, seed, summary) {
  const S = { expert: 2764800, part: 921600, row: 90 }[size], slots = Math.floor(fileBytes / S), seen = new Set(), all = [];
  let x = seed * 2654435761 >>> 0 || 1; const rnd = () => { x ^= x << 13; x ^= x >>> 17; x ^= x << 5; return (x >>> 0) / 2 ** 32; };
  const want = Math.min(reads, slots * files.length) - Math.min(reads, slots * files.length) % W;
  while (all.length < 2 * want) { const k = Math.floor(rnd() * slots * files.length); if (seen.has(k)) continue; seen.add(k); all.push(Math.floor(k / slots), (k % slots) * S); }
  const at = performance.timeOrigin + performance.now() + 100, per = all.length / W;
  const res = await Promise.all(Array.from({ length: W }, (_, w) => new Promise((ok, bad) => {
    const t = new Worker(NATIVE_WORKER, { eval: true, workerData: { files, jobs: all.slice(w * per, (w + 1) * per), size: S, at } });
    t.once("message", ok); t.once("error", bad);
  })));
  const lat = res.flatMap((r) => [...r.lat]);
  return { size, S, mode: "native", W, ...summary(lat, lat.length * S, Math.min(...res.map((r) => r.t0)), Math.max(...res.map((r) => r.t1)), S) };
}

const srv = serveRepo(8794, {}, { headers: () => ({ "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" }) });
const args = ["--no-sandbox", "--headless=new", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist",
  ...(process.platform === "linux" ? ["--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan"] : [])];
fs.mkdirSync(PROFILE, { recursive: true });
const ctx = await chromium.launchPersistentContext(PROFILE, { headless: false, args, ...(process.env.CHROME_BIN ? { executablePath: process.env.CHROME_BIN } : {}) });
const p = ctx.pages()[0] || await ctx.newPage();
p.on("console", (m) => console.log("  tab:", m.text())); p.on("crash", () => console.log("TAB CRASHED"));
const out = { host: os.hostname(), platform: `${process.platform} ${os.arch()}`, ramGB: +(os.totalmem() / 1e9).toFixed(0), runs: [] };
try {
  await p.goto("http://127.0.0.1:8794/tests/bench/opfs_read.html");
  await p.waitForFunction(() => window.INFO, null, { timeout: 60e3 });
  out.info = await p.evaluate(() => window.INFO);
  if (out.info.error) throw new Error(out.info.error);
  out.seed = SEED; await p.evaluate((s) => { probe.seed = s; }, SEED);
  console.log(`SEED=${SEED}`);
  out.setup = await p.evaluate((o) => probe.setup(o), { GB, FILES });
  if (CACHE_GB > 0) out.cacheSetup = await p.evaluate((o) => probe.cacheSetup(o), { CACHE_GB });
  out.shared = await p.evaluate(() => probe.probeShared().then((s) => ({ shared: s, error: probe.sharedError ?? null })));
  out.storage = await p.evaluate(() => probe.storage());
  console.log(`sync handles shared between workers (mode "read-only"): ${out.shared.shared}${out.shared.error ? " (" + out.shared.error + ")" : ""}`);
  console.log(`storage: ${JSON.stringify(out.storage)}`);

  const plan = [], ro = out.shared.shared && PLAN.has("ro");
  if (PLAN.has("expert")) for (const W of WORKERS) for (const mode of ["read", "write", "map", "xfer"]) plan.push({ size: "expert", W, mode, reads: READS });
  // the same reads with every worker on every file through read-only handles ("read/ro"), where Chrome has them
  if (ro) for (const W of WORKERS) if (W > 1) plan.push({ size: "expert", W, mode: "read", reads: READS, shared: true });
  if (PLAN.has("part")) for (const W of WORKERS) plan.push({ size: "part", W, mode: "read", reads: 3 * READS });
  if (PLAN.has("row")) for (const W of ROW_WORKERS) plan.push({ size: "row", W, mode: "read", reads: 40 * READS });
  if (ro) for (const W of ROW_WORKERS) if (W > 1) plan.push({ size: "row", W, mode: "read", reads: 40 * READS, shared: true });
  const native = env("NATIVE", "1") === "1" && PLAN.has("native") ? opfsFiles(out.setup.fileBytes) : [];
  if (native.length) for (const size of ["expert", "row"]) for (const W of size === "row" ? ROW_WORKERS : WORKERS) plan.push({ native: true, size, W, reads: size === "row" ? 40 * READS : READS });
  else if (env("NATIVE", "1") === "1" && PLAN.has("native")) console.log("native baseline: the OPFS files were not found under PROFILE, skipped");
  if (CACHE_GB > 0 && PLAN.has("cache")) for (const C of WORKERS) plan.push({ cache: true, C, reads: Math.min(READS, out.cacheSetup.entries) });
  const passes = [];
  if (env("WARM", "1") === "1") passes.push("warm");
  if (env("COLD", "1") === "1") passes.push("cold");
  for (const pass of passes) {
    if (pass === "warm") warm();
    console.log(`\n${pass} (${pass === "cold" ? "page cache dropped before each run" : "test data in the page cache"}):`);
    console.log("  size    mode     W   GB/s   reads/s   p10 ms   p50 ms   p99 ms  hits (<20 us, rows)");
    for (const cfg of plan) {
      let cold = pass === "cold" ? evict() : false;
      if (pass === "cold" && !cold) { console.log("  (could not drop the page cache: no cold numbers on this machine)"); break; }
      let r;
      try { r = cfg.native ? await nativeRun(cfg, native, out.setup.fileBytes, SEED + out.runs.length + 1, summary) : await p.evaluate((c) => c.cache ? probe.cacheRun({ C: c.C, reads: c.reads }) : probe.run(c), cfg); }
      catch (e) { r = { ...cfg, error: e.message.split("\n")[0] }; }
      r.pass = pass; out.runs.push(r);
      if (r.error) { console.log(`  ${cfg.size || "expert"} ${cfg.mode || "cache"} W${cfg.W || cfg.C}: ERROR ${r.error}`); continue; }
      console.log(`  ${r.size.padEnd(7)} ${(r.mode + (r.shared ? "/ro" : "")).padEnd(7)} ${String(r.W).padStart(2)} ${r.GBs.toFixed(r.GBs < 0.1 ? 3 : 2).padStart(6)} ${String(r.readsPerS).padStart(9)} ${r.p10.toFixed(3).padStart(8)} ${r.p50.toFixed(3).padStart(8)} ${r.p99.toFixed(3).padStart(8)}${r.hits !== undefined ? `  ${r.hits} (${(100 * r.hits / r.reads).toFixed(1)}%)` : ""}`);
    }
  }
  if (!KEEP) await p.evaluate(() => probe.cleanup());
} catch (e) { console.log("ERROR", e.message); out.error = e.message; }
console.log(JSON.stringify(out));
if (process.env.OUT) fs.writeFileSync(process.env.OUT, JSON.stringify(out, null, 1));
await ctx.close(); srv.close();
