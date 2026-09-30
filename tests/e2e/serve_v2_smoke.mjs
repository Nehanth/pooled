// `pooled serve` v2 smoke, on the GPU: a real host room (headless Chromium, the model on this
// machine's GPU) answers v2 asks (docs/protocol.md "API clients") sent straight through the CLI's
// Bridge (cli/lib/room.js): tools, parallel calls, required / named choices, a JSON schema, reasoning
// then a call, a tool-result follow-up that reuses the room's caches, and a v1 ask as before.
// Manual trigger only (a GPU and the model files; through gpurun.sh on the Spark):
//
//   (cd cli && npm install) && npm install && ln -s ~/bello/models models
//   node tests/e2e/serve_v2_smoke.mjs [--model qwen3-1.7b | qwen3.6-35b-moe] [--gb 8] [--port 8343] [--query "ctx=65536"] [--oldcli DIR]
//
// Prints one JSON line with every check; exit code 0 only when all passed.
import { chromium } from "playwright";
import http from "http";
import https from "https";
import fs from "fs";
import os from "os";
import path from "path";
import { spawn, execSync } from "child_process";

const arg = (k, d) => { const i = process.argv.indexOf("--" + k); return i > 0 ? process.argv[i + 1] : d; };
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");
const CLI = path.join(ROOT, "cli");
const MODEL = arg("model", "qwen3-1.7b");
const MOE = /moe|27b/.test(MODEL);
const GBV = arg("gb", MOE ? "40" : "8");
const PORT = +arg("port", 8343), TLS_PORT = PORT + 1, SIG_PORT = PORT + 2;
const LOCAL = { "Qwen3-1.7B-Q8_0.gguf": "models/qwen17/model.gguf", "Qwen3.8-27B-Q4_0.gguf": "models/q38/model.gguf", "Qwen_Qwen3.6-35B-A3B-Q4_0.gguf": "models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf" };
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png" };
const t0 = Date.now(); const log = (...a) => console.error(((Date.now() - t0) / 1000).toFixed(1) + "s", ...a);

const srv = http.createServer((q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { r.statusCode = 404; r.end(); return; }
  r.setHeader("content-type", MIME[path.extname(p)] || "application/octet-stream"); fs.createReadStream(p).pipe(r);
}).listen(PORT, "127.0.0.1");
const tlsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-v2-smoke-"));
execSync(`openssl req -x509 -newkey rsa:2048 -nodes -keyout ${tlsDir}/k.pem -out ${tlsDir}/c.pem -days 2 -subj /CN=127.0.0.1 2>/dev/null`);
const wsrv = https.createServer({ key: fs.readFileSync(`${tlsDir}/k.pem`), cert: fs.readFileSync(`${tlsDir}/c.pem`) }, (q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p)) { r.statusCode = 404; r.end(); return; }
  const size = fs.statSync(p).size, m = /bytes=(\d+)-(\d*)/.exec(q.headers.range || "");
  const lo = m ? +m[1] : 0, hi = m && m[2] ? Math.min(+m[2], size - 1) : size - 1;
  r.writeHead(m ? 206 : 200, { "content-type": "application/octet-stream", "content-range": `bytes ${lo}-${hi}/${size}`, "accept-ranges": "bytes", "content-length": String(hi - lo + 1), "access-control-allow-origin": "*", "access-control-expose-headers": "content-range, content-length, accept-ranges" });
  fs.createReadStream(p, { start: lo, end: hi }).pipe(r);
}).listen(TLS_PORT, "127.0.0.1");
const peerServer = spawn(path.join(ROOT, "node_modules/.bin/peerjs"), ["--port", String(SIG_PORT), "--path", "/"], { stdio: "ignore" });
await new Promise((r) => setTimeout(r, 1500));

// a profile on disk (the MoE's weights do not fit an in-memory Cache API store)
const prof = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-v2-prof-"));
const ctx = await chromium.launchPersistentContext(prof, { headless: false, ignoreHTTPSErrors: true,
  userAgent: "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
  args: ["--no-sandbox", "--headless=new", "--enable-unsafe-webgpu", "--enable-webgpu-developer-features", "--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--allow-loopback-in-peer-connection", "--js-flags=--max-old-space-size=65536"] });
await ctx.route("**/*.gguf", (route) => {
  const file = LOCAL[route.request().url().split("/").pop().split("?")[0]];
  if (!file || !fs.existsSync(path.join(ROOT, file))) return route.continue();
  return route.continue({ url: `https://127.0.0.1:${TLS_PORT}/${file}` });
});
await ctx.route(/cdn\.jsdelivr\.net\/npm\/peerjs@/, (r) => r.fulfill({ path: path.join(CLI, "node_modules/peerjs/dist/peerjs.min.js"), contentType: "text/javascript" }));
const page = await ctx.newPage();
const pageErrs = [];
page.on("pageerror", (e) => pageErrs.push(String(e).slice(0, 200)));

const checks = [];
const check = (name, cond, detail = "") => { checks.push({ name, ok: !!cond, ...(cond ? {} : { detail: String(detail).slice(0, 400) }) }); log(cond ? "ok  " : "FAIL", name, cond ? "" : String(detail).slice(0, 400)); };
const soft = async (name, fn) => { try { await fn(); } catch (e) { check(name, false, e.stack || e); } };
let bridge = null;
try {
  await page.goto(`http://127.0.0.1:${PORT}/p2p.html?signal=127.0.0.1:${SIG_PORT}&dev=1` + (arg("query") ? "&" + arg("query") : ""));
  await page.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
  await page.waitForFunction(() => document.getElementById("join-gb").value !== "1", null, { timeout: 45000 }).catch(() => {});
  await page.fill("#name-input", "spark-host"); await page.fill("#join-gb", GBV);
  await page.click("#create-btn");
  await page.waitForFunction(() => /[A-Z0-9]{4}/.test(document.getElementById("side-code").textContent), null, { timeout: 30000 });
  const code = (await page.textContent("#side-code")).trim().match(/[A-Z0-9]{4}/)[0];
  log("room", code);
  await page.selectOption("#ai-model", MODEL);
  await page.click("#ai-start");
  await page.waitForFunction(() => document.getElementById("ai-panel").classList.contains("online") || /^failed:/.test(document.getElementById("ai-status").textContent), null, { timeout: 1800000, polling: 1000 });
  if (/^failed:/.test(await page.textContent("#ai-status"))) throw new Error("load " + await page.textContent("#ai-status"));
  log("online:", await page.textContent("#ai-status"));

  const { Bridge } = await import(path.join(CLI, "lib/room.js"));
  bridge = new Bridge({ code, signal: `127.0.0.1:${SIG_PORT}`, name: "v2-smoke", client: "smoke", log: (m) => log("[bridge]", m) });
  await bridge.connect();
  for (let i = 0; i < 100 && !bridge.ready; i++) await new Promise((r) => setTimeout(r, 200));
  check("hello: the host answers v2 and says its context", bridge.hostApi === 2 && bridge.hostMeta.ctx > 0, JSON.stringify(bridge.hostMeta));

  let n = 0;
  // one ask -> { msgs, start, done, busy, text, think, calls: [{name, args}] (from the fragments) }
  const ask = (body, ms = 600000) => new Promise((resolve) => {
    const rid = "s" + (n++), msgs = [];
    const timer = setTimeout(() => resolve({ msgs, timeout: true }), ms);
    bridge.ask(rid, body, (d) => {
      msgs.push(d);
      if (d.t !== "ai-gendone" && d.t !== "ai-busy") return;
      clearTimeout(timer);
      const text = msgs.filter((m) => m.t === "ai-token" && !m.th).map((m) => m.text).join("");
      const think = msgs.filter((m) => m.t === "ai-token" && m.th).map((m) => m.text).join("");
      const calls = [];
      for (const m of msgs) if (m.t === "ai-call") { if (m.name != null) calls[m.i] = { name: m.name, args: "" }; else if (m.a != null) calls[m.i].args += m.a; }
      resolve({ msgs, start: msgs.find((m) => m.t === "ai-genstart"), done: d.t === "ai-gendone" ? d : null, busy: d.t === "ai-busy" ? d : null, text, think, calls });
    });
  });
  const TOOLS = [
    { name: "get_weather", description: "Get the current weather in a city.", parameters: { type: "object", properties: { city: { type: "string" }, unit: { type: "string", enum: ["celsius", "fahrenheit"] } }, required: ["city"] } },
    { name: "search", description: "Search the web.", parameters: { type: "object", properties: { query: { type: "string" }, limit: { type: "integer" } }, required: ["query"] } },
  ];
  const v2 = (messages, params = {}, tools = TOOLS) => ({ api: 2, system: "", messages, ...(tools ? { tools } : {}), params: { maxTokens: 200, temperature: 0, client: "smoke", toolChoice: "auto", parallel: true, ...params } });
  const same = (r) => r.done && r.calls.length === (r.done.calls || []).length && r.calls.every((c, i) => c.args === r.done.calls[i].args && c.name === r.done.calls[i].name);
  const parsed = (r) => r.calls.map((c) => ({ name: c.name, args: JSON.parse(c.args) }));

  await soft("v1", async () => {
    const r = await ask({ system: "", messages: [{ role: "user", text: "In one short sentence: what is the capital of France?" }], params: { maxTokens: 40, temperature: 0, client: "smoke" } });
    check("v1 ask: answered as before (genstart api 1, text)", r.start?.api === 1 && r.done?.api === 1 && /paris/i.test(r.text), JSON.stringify({ start: r.start, text: r.text, busy: r.busy }));
  });
  await soft("v2 plain", async () => {
    const r = await ask(v2([{ role: "user", text: "In one short sentence: what is the capital of Italy?" }], {}, null));
    check("v2 ask without tools: text, genstart api 2", r.start?.api === 2 && r.done?.api === 2 && /rome/i.test(r.text) && !r.calls.length, JSON.stringify({ start: r.start, text: r.text, busy: r.busy }));
  });
  let first = null;
  await soft("auto", async () => {
    const r = first = await ask(v2([{ role: "user", text: "What's the weather in Paris right now?" }]));
    check("auto: one get_weather call for Paris, streamed = final", same(r) && r.calls.length >= 1 && parsed(r)[0].name === "get_weather" && /paris/i.test(parsed(r)[0].args.city) && r.done.reason === "stop",
      JSON.stringify({ calls: r.calls, done: r.done, text: r.text }));
    check("auto: no call markup in the content", !/<tool_call>|<function=|"name":/.test(r.text), r.text);
  });
  await soft("follow-up", async () => {
    if (!first?.done?.calls?.length) throw new Error("no first call");
    const calls = first.done.calls.map((c) => ({ name: c.name, args: JSON.parse(c.args) }));
    const r = await ask(v2([{ role: "user", text: "What's the weather in Paris right now?" }, { role: "assistant", text: first.text, calls }, ...calls.map(() => ({ role: "tool", text: '{"temp_c": 18, "sky": "cloudy"}' }))]));
    check("tool result follow-up: an answer using it", r.done && /18/.test(r.text) && !r.calls.length, JSON.stringify({ text: r.text, calls: r.calls }));
    check("tool result follow-up: the room's caches held the first prompt and answer", r.done?.reused >= first.start.promptTokens, JSON.stringify({ reused: r.done?.reused, first: first.start?.promptTokens, prompt: r.start?.promptTokens }));
  });
  await soft("parallel", async () => {
    const r = await ask(v2([{ role: "user", text: "What's the weather in Paris and in Tokyo? Call the tool once per city, both at once." }]));
    const cities = parsed(r).map((c) => c.args.city).join(",");
    check("parallel: two calls", same(r) && r.calls.length === 2 && /paris/i.test(cities) && /tokyo/i.test(cities), JSON.stringify(r.calls));
    const one = await ask(v2([{ role: "user", text: "What's the weather in Paris and in Tokyo? Call the tool once per city, both at once." }], { parallel: false }));
    check("parallel false: one call", same(one) && one.calls.length === 1, JSON.stringify(one.calls));
  });
  await soft("required", async () => {
    const r = await ask(v2([{ role: "user", text: "Hi! How are you?" }], { toolChoice: "required" }));
    check("required: a call even for small talk", same(r) && r.calls.length >= 1 && !r.text.trim(), JSON.stringify({ calls: r.calls, text: r.text }));
    const s = await ask(v2([{ role: "user", text: "What's the weather in Paris?" }], { toolChoice: { name: "search" } }));
    check("named: the named tool", same(s) && s.calls.length === 1 && s.calls[0].name === "search" && typeof parsed(s)[0].args.query === "string", JSON.stringify(s.calls));
    const none = await ask(v2([{ role: "user", text: "What's the weather in Paris right now?" }], { toolChoice: "none", maxTokens: 60 }));
    check("none: no call, no markup", none.done && !none.calls.length && !/<tool|<function/.test(none.text), JSON.stringify({ text: none.text, calls: none.calls }));
  });
  await soft("format", async () => {
    const schema = { type: "object", properties: { city: { type: "string" }, population_millions: { type: "number" } }, required: ["city", "population_millions"], additionalProperties: false };
    const r = await ask(v2([{ role: "user", text: "Largest city in Japan, and its population in millions?" }], { format: { type: "schema", schema }, maxTokens: 80 }, null));
    let o = null; try { o = JSON.parse(r.text); } catch {}
    check("json schema: the answer parses and fits", o && typeof o.city === "string" && typeof o.population_millions === "number" && Object.keys(o).length === 2, r.text);
  });
  await soft("thinking", async () => {
    const r = await ask(v2([{ role: "user", text: "What's the weather in Rome?" }], { thinking: true, maxTokens: 700 }));
    const firstCall = r.msgs.findIndex((m) => m.t === "ai-call"), lastThink = r.msgs.map((m) => m.t === "ai-token" && m.th).lastIndexOf(true);
    check("thinking: reasoning, then the call", same(r) && r.calls.length >= 1 && r.think.length > 20 && lastThink < firstCall && r.done.usage.think > 0, JSON.stringify({ think: r.think.slice(0, 80), calls: r.calls, usage: r.done?.usage, reason: r.done?.reason }));
  });
  // an older CLI (its cli/lib, e.g. from `git show 5af2f66:cli/lib/…`): v1 asks, as it always sent
  if (arg("oldcli")) await soft("old cli", async () => {
    const { Bridge: OldBridge } = await import(path.join(path.resolve(arg("oldcli")), "lib/room.js"));
    const old = new OldBridge({ code: bridge.code, signal: `127.0.0.1:${SIG_PORT}`, name: "old-cli", client: "old", log: (m) => log("[old]", m) });
    await old.connect();
    for (let i = 0; i < 100 && !old.ready; i++) await new Promise((r) => setTimeout(r, 200));
    const got = await new Promise((resolve) => {
      const msgs = [];
      old.ask("old1", { system: "", messages: [{ role: "user", text: "In one short sentence: what is the capital of Spain?" }], params: { maxTokens: 40, temperature: 0, client: "old" } }, (d) => { msgs.push(d); if (d.t === "ai-gendone" || d.t === "ai-busy") resolve(msgs); });
    });
    const text = got.filter((m) => m.t === "ai-token").map((m) => m.text).join("");
    check("an older CLI against this host: served as before", got.some((m) => m.t === "ai-gendone") && /madrid/i.test(text) && !got.some((m) => m.t === "ai-call"), text);
    await old.leave();
  });
  await soft("bad", async () => {
    const r = await ask(v2([{ role: "user", text: "x" }], { toolChoice: { name: "nope" } }));
    check("an undeclared named tool: ai-busy bad", r.busy?.code === "bad", JSON.stringify(r.busy));
  });
  check("no page errors", pageErrs.length === 0, pageErrs.join(" | "));
} catch (e) {
  check("run", false, e.stack || e);
} finally {
  try { await bridge?.leave(); } catch {}
  await ctx.close().catch(() => {});
  peerServer.kill(); srv.close(); wsrv.close();
  fs.rmSync(prof, { recursive: true, force: true });
}
const failed = checks.filter((c) => !c.ok);
console.log(JSON.stringify({ model: MODEL, ok: !failed.length, passed: checks.length - failed.length, failed: failed.length, checks }));
process.exit(failed.length ? 1 : 0);
