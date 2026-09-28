// Run tests/bench/bw_probe.html in Chrome with the real GPU: achievable bandwidth and dispatch floors, no model.
//   node tests/bench/bw_probe.mjs ['MB=1024&REPS=8']
// CHROME_BIN=<path>: a Chrome build other than Playwright's bundled one (e.g. /Applications/Google Chrome.app/... on a Mac).
// Uses the GPU: never run it while another job owns the GPU.
import { chromium } from "playwright"; import { spawn } from "node:child_process";
const root = new URL("../..", import.meta.url).pathname, EXTRA = process.argv[2] ? "?" + process.argv[2] : "";
const srv = spawn("node", [root + "tests/bench/serve.mjs", root, "8793"], { stdio: "inherit" }); await new Promise((r) => setTimeout(r, 600));
const args = ["--no-sandbox", "--headless=new", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist",
  ...(process.platform === "linux" ? ["--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan"] : [])];
const b = await chromium.launch({ headless: false, args, ...(process.env.CHROME_BIN ? { executablePath: process.env.CHROME_BIN } : {}) }), p = await (await b.newContext()).newPage();
p.on("console", (m) => console.log("  tab:", m.text())); p.on("crash", () => console.log("TAB CRASHED"));
await p.goto(`http://127.0.0.1:8793/tests/bench/bw_probe.html${EXTRA}`);
await p.waitForFunction(() => window.RESULT, null, { timeout: 10 * 60e3, polling: 1000 }).catch((e) => console.log("timeout", e.message));
await b.close(); srv.kill();
