// The eval suite's driver (docs/design/harness-light.md C.3): serves the repo, opens
// tests/eval/eval.html in Chromium, runs the tasks and writes the results.
//
//   node tests/eval/run.mjs --model mock                        scripted golden runs, no GPU (CI-safe)
//   E2E_GPU=real node tests/eval/run.mjs --model engine --weights models/q38/model.gguf --headed
//        [--tasks tetris,calculator] [--repeat 3] [--ctx 16384]
//   --no-selftest   skip checking that each task's `bad` files fail its check (on by default for mock)
//   --verbose       print page errors (the apps' own, expected ones included)
//   --port 18995
//   --engine-opts '<JSON>'  extra engine options (A/B), e.g. '{"attnPrefillTile":false,"moeGroupPrefill":0,"prefillUbatch":0}'
// Code mode's settings (the Code mode eval, docs/design/harness-core.md):
//   --codemode      Code mode's model settings: the template's call style, maxNew 8192, greedy JSON / focused XML
//   --dense <dir>   a Qwen3 dense model dir (config.json, tokenizer.json, model.gguf) instead of --weights
//   --hcore 0|1     the model path: 1 = the serve v2 core in process (harness/core-model.js), 0 = legacy
//   --sampling <preset>  override the sampler (room/sampling.js)
//   --label <name>  names the results file and directory (default: the model kind)
// Writes tests/eval/results/<stamp>-<model>.jsonl (one record per task run) and a directory of
// the same name with one trajectory JSON per task run (the full conversation and final files).
// Exit code: non-zero when a mock run or a self-test fails. Needs playwright on NODE_PATH.
import fs from "fs";
import path from "path";
import { loadPlaywright, chromiumPath, GPU_ARGS, serveRepo } from "../e2e/engine_synth.mjs";

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const flag = (k) => argv.includes("--" + k);
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");
const MODEL = arg("model", "mock"), PORT = +arg("port", 18995), REPEAT = +arg("repeat", 1);
if (!["mock", "engine"].includes(MODEL)) { console.error("--model mock|engine (a room runs from Code mode: p2p.html?eval=all)"); process.exit(2); }
const weights = arg("weights", null);
const dense = arg("dense", null);   // a Qwen3 dense model dir (config.json, tokenizer.json, model.gguf)
if (MODEL === "engine" && !weights && !dense) { console.error("--model engine needs --weights <file.gguf>"); process.exit(2); }

const srv = serveRepo(PORT, dense ? { "/__m.gguf": path.resolve(dense, "model.gguf"), "/__cfg.json": path.resolve(dense, "config.json"), "/__tok.json": path.resolve(dense, "tokenizer.json") }
  : weights ? { "/__m.gguf": path.resolve(weights) } : {});
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromiumPath(), headless: !flag("headed"),
  // --site-per-process as in desktop Chrome: the preview relay and the probe run in their own process
  args: [...(MODEL === "engine" ? GPU_ARGS : ["--no-sandbox"]), "--site-per-process"] });
const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
const outDir = path.join(ROOT, "tests/eval/results", `${stamp}-${arg("label", MODEL)}`);
fs.mkdirSync(outDir, { recursive: true });
const jsonl = outDir + ".jsonl";
let code = 0;
try {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  await ctx.route("**/*", (route) => {
    const url = route.request().url();
    if (url.startsWith(`http://127.0.0.1:${PORT}/`) || url === `http://localhost:${PORT}/harness/preview-relay.html`) return route.continue();
    return route.abort();   // no network: the CDNs the preview allows are not needed by the golden runs
  });
  const page = await ctx.newPage();
  if (flag("verbose")) page.on("pageerror", (e) => console.error("page error:", String(e).slice(0, 300)));   // the apps' own errors show here too
  await page.goto(`http://127.0.0.1:${PORT}/tests/eval/eval.html?model=${MODEL}${arg("ctx", "") ? "&ctx=" + arg("ctx") : ""}${dense ? "&dense=1" : ""}${flag("codemode") ? "&codemode=1" : ""}${arg("hcore", "") ? "&hcore=" + arg("hcore") : ""}${arg("sampling", "") ? "&sampling=" + arg("sampling") : ""}${arg("engine-opts", "") ? "&opts=" + encodeURIComponent(arg("engine-opts")) : ""}`);
  await page.waitForFunction(() => window.__eval, null, { timeout: 30000 });
  await page.evaluate(() => window.__eval.init());
  if (MODEL === "engine") console.log("engine:", JSON.stringify(await page.evaluate(() => window.__engineInfo)));
  const all = await page.evaluate(() => window.__eval.tasks);
  const ids = arg("tasks", "") ? arg("tasks").split(",") : all;
  const unknown = ids.filter((i) => !all.includes(i));
  if (unknown.length) throw new Error(`unknown tasks: ${unknown.join(", ")} (have ${all.join(", ")})`);

  if (MODEL === "mock" && !flag("no-selftest")) {
    let bad = 0;
    for (const id of ids) {
      const r = await page.evaluate((i) => window.__eval.selfTest(i), id);
      if (!r.ok) { bad++; console.log(`SELFTEST FAIL ${id}: its bad files pass the check`); }
    }
    console.log(`self-test: ${ids.length - bad}/${ids.length} checks reject a wrong solution`);
    if (bad) code = 1;
  }

  const recs = [];
  const pad = (s, n) => String(s).padEnd(n);
  console.log(`${pad("task", 12)} ${pad("ok", 4)} ${pad("reason", 8)} ${pad("steps", 5)} ${pad("calls", 5)} ${pad("prefill", 8)} ${pad("reused", 8)} ${pad("gen", 7)} ${pad("ctx", 6)} ${pad("s", 6)} cards`);
  for (let k = 0; k < REPEAT; k++) {
    for (const id of ids) {
      const { rec, trajectory } = await page.evaluate((i) => window.__eval.task(i), id);
      rec.run = k;
      recs.push(rec);
      fs.appendFileSync(jsonl, JSON.stringify(rec) + "\n");
      fs.writeFileSync(path.join(outDir, `${id}${REPEAT > 1 ? "-" + k : ""}.json`), JSON.stringify(trajectory, null, 1));
      const cards = Object.entries(rec.cards).map(([c, n]) => `${c}×${n}`).join(" ");
      console.log(`${pad(id, 12)} ${pad(rec.ok ? "PASS" : "FAIL", 4)} ${pad(rec.reason, 8)} ${pad(rec.steps, 5)} ${pad(rec.calls, 5)} ${pad(rec.prompt, 8)} ${pad(rec.reused, 8)} ${pad(rec.generated, 7)} ${pad(rec.ctx, 6)} ${pad((rec.ms / 1000).toFixed(1), 6)} ${cards}`);
      if (!rec.ok) console.log("    " + rec.check.split("\n").slice(0, 6).join("\n    "));
    }
  }
  const line = await page.evaluate(async (r) => (await import("/tests/eval/suite.js")).summary(r), recs);
  console.log(line);
  if (MODEL === "engine") console.log("GPU errors:", await page.evaluate(() => window.__engineInfo?.gpuErrors));
  console.log(`results: ${path.relative(ROOT, jsonl)} (+ trajectories in ${path.relative(ROOT, outDir)}/)`);
  if (MODEL === "mock" && recs.some((r) => !r.ok)) code = 1;
} catch (e) {
  console.error(e);
  code = 1;
} finally {
  await browser.close();
  srv.close();
}
process.exit(code);
