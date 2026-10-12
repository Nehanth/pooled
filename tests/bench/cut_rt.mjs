// Run tests/bench/cut_rt.html in Chrome with the real GPU: the floor of one GPU -> CPU round trip (expert offload's cut).
//   node tests/bench/cut_rt.mjs ['n=300&work=3000']      CHROME_BIN=<path>: another Chrome (e.g. on a Mac)
// Node / Dawn: node --no-maglev tests/dawn_run.mjs tests/bench/cut_rt.js
import { chromium } from "playwright"; import { spawn } from "node:child_process";
const root = new URL("../..", import.meta.url).pathname, EXTRA = process.argv[2] ? "?" + process.argv[2] : "", PORT = process.env.PORT || "8794";
const srv = spawn("node", [root + "tests/bench/serve.mjs", root, PORT], { stdio: "inherit" }); await new Promise((r) => setTimeout(r, 600));
const args = ["--no-sandbox", "--headless=new", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist",
  ...(process.platform === "linux" ? ["--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan"] : [])];
const b = await chromium.launch({ headless: false, args, ...(process.env.CHROME_BIN ? { executablePath: process.env.CHROME_BIN } : {}) }), p = await (await b.newContext()).newPage();
p.on("console", (m) => console.log("  tab:", m.text())); p.on("crash", () => console.log("TAB CRASHED"));
console.log(`chrome ${b.version()}`);
await p.goto(`http://127.0.0.1:${PORT}/tests/bench/cut_rt.html${EXTRA}`);
await p.waitForFunction(() => window.RESULT, null, { timeout: 10 * 60e3, polling: 500 }).catch((e) => console.log("timeout", e.message));
await b.close(); srv.kill();
