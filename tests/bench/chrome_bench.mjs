// Run tests/bench/bench.html in Chrome with the real GPU. node tests/bench/chrome_bench.mjs <model path under repo> [tokens] [extra query]
// extra query: e.g. "draftvocab=65536&predraft=0&specfuse=0" (see the knobs at the top of bench.html)
//   wide prefill A/B: "ubatch=256&prefilllen=2048" (prefill tok/s with prefillUbatch off and on, logits relDiff)
// env: MOE_FUSE=0, MOE_DN_ROWS=1|2|4, MOE_KERNEL=legacy|default|JSON (unfused expert GEMV layout)
//      PREFILL_MATH=f32|f16|sgmatrix (prefill GEMM operand precision), PREFILL=N (prefill tok/s on N tokens, A/B vs f32),
//      SGM='{"TM":128,"KB":1,"PAD":8}' (tensor-core GEMM tuning). The prefill GEMM needs 16 columns: add "batchcols=16".
// Loading (not part of the tok/s numbers) is made cheap for repeated runs:
//   WCACHE=1 (default): the page takes pre-converted tensors from serve.mjs's weight cache
//     (tests/weight_cache.js, ~/.cache/swarmllm-weights) instead of repacking in JS; WCACHE=0 = old path.
//   CHROME_PROFILE=<dir> (default ~/.cache/swarmllm-chrome-bench): a persistent Chrome profile, so
//     Chrome's on-disk GPU/shader caches survive between runs; CHROME_PROFILE=0 = a fresh profile each run.
// CHROME_BIN=<path>: run that Chrome/Chromium build instead of Playwright's bundled one (e.g. a newer
//   build that exposes chromium-experimental-subgroup-matrix for PREFILL_MATH=sgmatrix).
// Decode is untouched either way: same page, same engine, same kernels.
import { chromium } from "playwright"; import { spawn } from "node:child_process"; import os from "node:os"; import path from "node:path";
// PORT=<n>: the static server's port (default 8791; another worktree's server on it would serve its own files)
const PORT = process.env.PORT || "8791";
const root = new URL("../..", import.meta.url).pathname, model = process.argv[2], N = process.argv[3] || 40, EXTRA = process.argv[4] ? "&" + process.argv[4] : "", MOEK = process.env.MOE_KERNEL || "";
const GOLD = { "q36moe": ["```python\ndef two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:", "A hash map is a data structure that stores key-value pairs, allowing for efficient retrieval, insertion, and deletion operations. It uses a hash function to compute an index into an array of buckets or slots"] };
const wcache = process.env.WCACHE !== "0", prof = process.env.CHROME_PROFILE ?? path.join(os.homedir(), ".cache", "swarmllm-chrome-bench");
const srv = spawn("node", [root + "tests/bench/serve.mjs", root, PORT], { stdio: "inherit" }); await new Promise((r) => setTimeout(r, 600));
// the ANGLE / Vulkan flags are Linux-only: on macOS they leave the tab without a WebGPU adapter (Chrome runs WebGPU on Metal by default)
// WEBGPU_DEV=1: full-precision timestamps (?gap=1; Chrome quantizes them to 100 µs otherwise)
const args = ["--no-sandbox", "--headless=new", "--enable-unsafe-webgpu", ...(process.env.WEBGPU_DEV === "1" ? ["--enable-webgpu-developer-features"] : []), "--ignore-gpu-blocklist", "--js-flags=--max-old-space-size=65536",
  ...(process.platform === "linux" ? ["--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan"] : [])];
let b, ctx;
const exe = process.env.CHROME_BIN ? { executablePath: process.env.CHROME_BIN } : {};
if (prof && prof !== "0") { ctx = await chromium.launchPersistentContext(prof, { headless: false, args, ...exe }); console.log("chrome profile:", prof); }
else { b = await chromium.launch({ headless: false, args, ...exe }); ctx = await b.newContext(); }
const p = await ctx.newPage(); p.on("console", (m) => console.log("  tab:", m.text())); p.on("crash", () => console.log("TAB CRASHED"));
const gold = GOLD[Object.keys(GOLD).find((k) => model.includes(k))] || [];
await p.goto(`http://127.0.0.1:${PORT}/tests/bench/bench.html?model=/${model}&tokens=${N}&wcache=${wcache ? 1 : 0}&gold=${encodeURIComponent(JSON.stringify(gold))}${EXTRA}${process.env.MOE_FUSE === "0" ? "&moefuse=0" : ""}${process.env.MOE_DN_ROWS ? "&moednrows=" + process.env.MOE_DN_ROWS : ""}${MOEK ? "&moe=" + encodeURIComponent(MOEK) : ""}${process.env.ATTN_PREFILL_TILE ? "&attnptile=" + (process.env.ATTN_PREFILL_TILE === "0" ? 0 : 1) : ""}${process.env.PREFILL_MATH ? "&prefillmath=" + process.env.PREFILL_MATH : ""}${process.env.PREFILL ? "&pmab=" + process.env.PREFILL : ""}${process.env.SGM ? "&sgm=" + encodeURIComponent(process.env.SGM) : ""}`);
await p.waitForFunction(() => window.RESULT, null, { timeout: 30 * 60e3, polling: 2000 }).catch((e) => console.log("timeout", e.message));
await ctx.close(); if (b) await b.close(); srv.kill();
