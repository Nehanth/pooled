// Run tests/bench/tab_cut.html in Chrome with the real GPU, one fresh browser per run, and compare the fingerprints.
//   RUNS="parked?slots=16,parked?slots=16&kick=0,resident" node tests/bench/tab_cut.mjs
// env: RUNS (comma list of mode[?query]), QUERY (added to every run), MODEL (under the repo), CHROME_BIN, PORT (8795)
import { chromium } from "playwright"; import { spawn } from "node:child_process";
const env = (k, d) => process.env[k] ?? d;
const root = new URL("../..", import.meta.url).pathname, PORT = env("PORT", "8795"), QUERY = env("QUERY", "");
const MODEL = env("MODEL", "models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf");
const RUNS = env("RUNS", "parked?slots=16,parked?slots=16&kick=0,resident").split(",").map((r) => { const i = r.indexOf("?"); return i < 0 ? { mode: r, qs: "" } : { mode: r.slice(0, i), qs: r.slice(i + 1) }; });
const srv = spawn("node", [root + "tests/bench/serve.mjs", root, PORT], { stdio: "inherit" });
await new Promise((r) => setTimeout(r, 600));
const args = ["--no-sandbox", "--headless=new", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--js-flags=--max-old-space-size=16384",
  ...(process.platform === "linux" ? ["--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan"] : [])];
const results = [];
try {
  for (const run of RUNS) {
    const b = await chromium.launch({ headless: false, args, ...(process.env.CHROME_BIN ? { executablePath: process.env.CHROME_BIN } : {}) });
    const p = await (await b.newContext()).newPage();
    const name = `${run.mode}${run.qs ? "?" + run.qs : ""}`;
    p.on("console", (m) => console.log(`  ${name}:`, m.text())); p.on("crash", () => console.log("TAB CRASHED"));
    const qs = [`model=/${MODEL}`, `mode=${run.mode}`, QUERY, run.qs].filter(Boolean).join("&");
    await p.goto(`http://127.0.0.1:${PORT}/tests/bench/tab_cut.html?${qs}`);
    const r = await p.waitForFunction(() => window.RESULT, null, { timeout: 30 * 60e3, polling: 1000 }).then((x) => x.jsonValue()).catch((e) => ({ error: e.message }));
    results.push({ run: name, ...r });
    await b.close();
  }
} finally { srv.kill(); }
const flat = results.flatMap((r) => (r.runs || [r]).map((x) => ({ ...x, run: r.run + (x.v && Object.keys(x.v).length ? " " + JSON.stringify(x.v) : ""), error: r.error })));
const bits = flat.filter((r) => r.bits);
const exact = bits.length > 1 && bits.every((r) => r.bits.plain === bits[0].bits.plain && r.bits.hidden === bits[0].bits.hidden && r.bits.tokens === bits[0].bits.tokens);
for (const r of flat) console.log(`${r.run}: ${r.error ? "ERROR " + r.error : `bits ${r.bits?.plain}/${r.bits?.hidden}, decode ${r.decode?.map((d) => d.tps.toFixed(2)).join(" / ")} tok/s`}`);
if (bits.length > 1) console.log(exact ? `BIT-EXACT: ${bits.map((r) => r.run).join(" == ")}` : "FINGERPRINTS DIFFER");
if (process.env.OUT) (await import("node:fs")).writeFileSync(process.env.OUT, JSON.stringify(results, null, 1));
process.exit(results.some((r) => r.error) || (bits.length > 1 && !exact) ? 1 : 0);
