// Run tests/bench/tab_disk.html in Chrome with the real GPU: a MoE with its experts in the tab's OPFS (the disk tier,
// roadmap/32 step 8) against the same model resident, each run in a fresh browser on one persistent profile (the
// OPFS copy is made once and kept). Prints each run's fingerprint and decode tok/s, then whether the fingerprints
// match (bit-exact) and the result as JSON (OUT=<file> also writes it there).
//   node tests/bench/tab_disk.mjs
// env: MODEL (under the repo; default models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf), RUNS (comma list of
//      mode[?query] runs, default "disk?slots=16,resident"; a run's query is added to QUERY's), QUERY (for every run,
//      e.g. "tokens=64&workers=4&ring=64"), PROFILE (Chrome profile dir; default ~/.cache/pooled-tabdisk-profile:
//      the OPFS copy lives there, ~18 GiB for the 35B), COLD (1: drop the profile's files from the page cache
//      before each disk run, Linux only; the default), CHROME_BIN, PORT (8793)
//
// Mac runbook (Chrome, not headless; the GPU is Metal, so no ANGLE / Vulkan flags):
//   1. npm ci; node tests/bench/serve.mjs . 8793      (the model at models/..., or any path under the repo)
//   2. open http://127.0.0.1:8793/tests/bench/tab_disk.html?model=/models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf&mode=ingest
//      in a normal (not incognito: incognito OPFS lives in RAM) Chrome window, once: copies the experts into OPFS
//   3. the same URL with mode=disk&mem=36 (the default cache for a 36 GB Mac, ~19.8 GiB), then offgb=4, 8, 12, 18
//      (a sweep: Apple's wired GPU memory starves the page cache that serves the reads), each with &ring=0 too
//   4. mode=resident for the reference fingerprint (BITS line) when the model fits; else compare with this script's
//      run on the Spark: the fingerprint is the same on any device whose kernels give the same bits
// Or the whole thing from a terminal: CHROME_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" COLD=0 node tests/bench/tab_disk.mjs
import { chromium } from "playwright"; import { spawn, spawnSync } from "node:child_process"; import fs from "node:fs"; import os from "node:os"; import path from "node:path";
const env = (k, d) => process.env[k] ?? d;
const root = new URL("../..", import.meta.url).pathname, PORT = env("PORT", "8793");
const MODEL = env("MODEL", "models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf"), QUERY = env("QUERY", "");
const RUNS = env("RUNS", "disk?slots=16,resident").split(",").map((r) => { const [mode, qs = ""] = r.split("?"); return { mode, qs }; });
const PROFILE = env("PROFILE", path.join(os.homedir(), ".cache", "pooled-tabdisk-profile")), COLD = env("COLD", "1") === "1";

function evict() {   // the profile's files out of the page cache (posix_fadvise DONTNEED; no root needed)
  if (process.platform !== "linux") return false;
  const py = "import os,sys\nos.sync()\nfor d,_,fs in os.walk(sys.argv[1]):\n  for f in fs:\n    try:\n      fd=os.open(os.path.join(d,f),os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)\n    except OSError: pass";
  return spawnSync("python3", ["-c", py, PROFILE]).status === 0;
}
const srv = spawn("node", [root + "tests/bench/serve.mjs", root, PORT], { stdio: "inherit" });
await new Promise((r) => setTimeout(r, 600));
const args = ["--no-sandbox", "--headless=new", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--js-flags=--max-old-space-size=16384",
  ...(process.platform === "linux" ? ["--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan"] : [])];
const exe = process.env.CHROME_BIN ? { executablePath: process.env.CHROME_BIN } : {};
const results = [];
try {
  for (const run of RUNS) {
    const cold = COLD && run.mode === "disk" && evict();
    const ctx = await chromium.launchPersistentContext(PROFILE, { headless: false, args, ...exe });
    const p = await ctx.newPage();
    p.on("console", (m) => console.log(`  ${run.mode}:`, m.text())); p.on("crash", () => console.log("TAB CRASHED"));
    const qs = [`model=/${MODEL}`, `mode=${run.mode}`, QUERY, run.qs].filter(Boolean).join("&");
    console.log(`run ${run.mode}${run.qs ? "?" + run.qs : ""}${cold ? " (page cache dropped)" : ""}: tests/bench/tab_disk.html?${qs}`);
    await p.goto(`http://127.0.0.1:${PORT}/tests/bench/tab_disk.html?${qs}`);
    const r = await p.waitForFunction(() => window.RESULT, null, { timeout: 60 * 60e3, polling: 2000 }).then((h) => h.jsonValue()).catch((e) => ({ error: e.message }));
    results.push({ run: `${run.mode}${run.qs ? "?" + run.qs : ""}`, cold, ...r });
    await ctx.close();
  }
} finally { srv.kill(); }
const bits = results.filter((r) => r.bits);
const exact = bits.length > 1 && bits.every((r) => r.bits.plain === bits[0].bits.plain && r.bits.hidden === bits[0].bits.hidden && r.bits.tokens === bits[0].bits.tokens);
for (const r of results) console.log(`${r.run}: ${r.error ? "ERROR " + r.error : `bits ${r.bits?.plain}/${r.bits?.hidden}, decode ${r.decode?.map((d) => d.tps.toFixed(2)).join(" / ")} tok/s${r.decode?.[0]?.hit != null ? `, hits ${r.decode.map((d) => (100 * d.hit).toFixed(1)).join(" / ")}%, ${r.decode.map((d) => d.waitMsPerTok.toFixed(1)).join(" / ")} ms/token on OPFS` : ""}`}`);
if (bits.length > 1) console.log(exact ? `BIT-EXACT: ${bits.map((r) => r.run).join(" == ")}` : "FINGERPRINTS DIFFER");
if (process.env.OUT) fs.writeFileSync(process.env.OUT, JSON.stringify(results, null, 1));
console.log(JSON.stringify(results));
process.exit(results.some((r) => r.error) || (bits.length > 1 && !exact) ? 1 : 0);
