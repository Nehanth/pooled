// Run tests/bench/prof.html in Chrome with the real GPU (as tests/bench/chrome_bench.mjs does).
//   node tests/prof/prof_chrome.mjs models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf [out.json] [extra query]
// --enable-webgpu-developer-features: full-precision timestamps (Chrome quantizes them to 100 µs otherwise)
import { chromium } from "playwright"; import { spawn } from "node:child_process"; import fs from "node:fs";
// PORT=<n>: the static server's port (default 8792; another worktree's server on it would serve its own files)
const PORT = process.env.PORT || "8792";
const root = new URL("../..", import.meta.url).pathname, model = process.argv[2], OUT = process.argv[3], EXTRA = process.argv[4] ? "&" + process.argv[4] : "";
const srv = spawn("node", [root + "tests/bench/serve.mjs", root, PORT], { stdio: "inherit" }); await new Promise((r) => setTimeout(r, 600));
// CHROME_BIN=<path>: a Chrome build other than Playwright's bundled one (on a Mac: /Applications/Google Chrome.app/Contents/MacOS/Google Chrome);
// the ANGLE / Vulkan flags are Linux-only (on macOS Chrome runs WebGPU on Metal by default)
const args = ["--no-sandbox", "--headless=new", "--enable-unsafe-webgpu", "--enable-webgpu-developer-features", "--ignore-gpu-blocklist", "--js-flags=--max-old-space-size=65536",
  ...(process.platform === "linux" ? ["--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan"] : [])];
const b = await chromium.launch({ headless: false, args, ...(process.env.CHROME_BIN ? { executablePath: process.env.CHROME_BIN } : {}) });
const p = await b.newPage(); p.on("console", (m) => console.log("  tab:", m.text())); p.on("crash", () => console.log("TAB CRASHED"));
await p.goto(`http://127.0.0.1:${PORT}/tests/bench/prof.html?model=/${model}${EXTRA}`);
await p.waitForFunction(() => window.RESULT, null, { timeout: 30 * 60e3, polling: 2000 }).catch((e) => console.log("timeout", e.message));
const r = await p.evaluate(() => window.RESULT).catch(() => null);
if (OUT && r) fs.writeFileSync(OUT, JSON.stringify(r, null, 1));
await b.close(); srv.kill();
