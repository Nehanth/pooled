// Run tests/bench/wmma_gemm.html in Chrome with the real GPU: the wide prefill GEMM (f32 tiled) vs the
// subgroup-matrix GEMM (engine/wgsl/gemm_wmma.js), per 27B / MoE shape: ms, TOPS, errors.
//   CHROME_BIN=<chrome 145+> node tests/bench/wmma_gemm.mjs ['reps=10&w=256&tile={"BM":64,"BN":64,"SM":32,"SN":32}']
// The extension needs --enable-unsafe-webgpu (set here). Uses the GPU: never run it while another job owns it.
import { chromium } from "playwright"; import { spawn } from "node:child_process";
const root = new URL("../..", import.meta.url).pathname, EXTRA = process.argv[2] ? "?" + process.argv[2] : "", port = +(process.env.PORT || 8793);
const srv = spawn("node", [root + "tests/bench/serve.mjs", root, String(port)], { stdio: "inherit" }); await new Promise((r) => setTimeout(r, 600));
const args = ["--no-sandbox", "--headless=new", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist",
  ...(process.platform === "linux" ? ["--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan"] : [])];
const b = await chromium.launch({ headless: false, args, ...(process.env.CHROME_BIN ? { executablePath: process.env.CHROME_BIN } : {}) }), p = await (await b.newContext()).newPage();
p.on("console", (m) => console.log("  tab:", m.text())); p.on("crash", () => console.log("TAB CRASHED"));
await p.goto(`http://127.0.0.1:${port}/tests/bench/wmma_gemm.html${EXTRA}`);
const r = await p.waitForFunction(() => window.RESULT, null, { timeout: 20 * 60e3, polling: 1000 }).then((h) => h.jsonValue()).catch((e) => ({ timeout: e.message }));
if (process.env.OUT) (await import("node:fs")).writeFileSync(process.env.OUT, JSON.stringify(r, null, 1));
await b.close(); srv.kill();
