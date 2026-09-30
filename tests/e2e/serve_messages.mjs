// `pooled serve`'s Anthropic Messages endpoint on the GPU: a real host room (headless Chromium, the
// model on this machine's GPU), the CLI's Bridge and HTTP server (cli/lib/http.js), and the official
// @anthropic-ai/sdk client: tool_use, tool_result follow-ups that reuse the room's caches, tool_choice
// any / tool / none, disable_parallel_tool_use, thinking with the reasoning carried in the signature,
// stop sequences. With --claude, also a real Claude Code run (`claude -p`, ANTHROPIC_BASE_URL) that
// reads a file with its Read tool; Claude Code's prompt is ~20 k tokens, so give the room a context
// (--query "ctx=65536").
// Manual trigger only (a GPU and the model files; through gpurun.sh on the Spark):
//
//   (cd cli && npm install) && npm install && ln -s ~/bello/models models
//   node tests/e2e/serve_messages.mjs [--model qwen3-1.7b | qwen3.6-35b-moe] [--gb 8] [--port 8353] [--query "ctx=65536"] [--claude]
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
const PORT = +arg("port", 8353), TLS_PORT = PORT + 1, SIG_PORT = PORT + 2;
const LOCAL = { "Qwen3-1.7B-Q8_0.gguf": "models/qwen17/model.gguf", "Qwen3.8-27B-Q4_0.gguf": "models/q38/model.gguf", "Qwen_Qwen3.6-35B-A3B-Q4_0.gguf": "models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf" };
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png" };
const t0 = Date.now(); const log = (...a) => console.error(((Date.now() - t0) / 1000).toFixed(1) + "s", ...a);

const srv = http.createServer((q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { r.statusCode = 404; r.end(); return; }
  r.setHeader("content-type", MIME[path.extname(p)] || "application/octet-stream"); fs.createReadStream(p).pipe(r);
}).listen(PORT, "127.0.0.1");
const tlsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-msg-e2e-"));
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
const prof = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-msg-prof-"));
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
let bridge = null, api = null;
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
  bridge = new Bridge({ code, signal: `127.0.0.1:${SIG_PORT}`, name: "messages-e2e", client: "e2e", log: (m) => log("[bridge]", m) });
  await bridge.connect();
  for (let i = 0; i < 100 && !bridge.ready; i++) await new Promise((r) => setTimeout(r, 200));
  check("hello: the host answers v2 and says its context", bridge.hostApi === 2 && bridge.hostMeta.ctx > 0, JSON.stringify(bridge.hostMeta));


  // pooled serve's HTTP side on this Bridge, and the official client
  const { createServer } = await import(path.join(CLI, "lib/http.js"));
  const { blob } = await import(path.join(CLI, "lib/common.js"));
  api = createServer({ bridge, port: 0, log: (m) => log("[serve]", m) });
  const port = await api.listen();
  const { createRequire } = await import("module");
  const Anthropic = (await import(createRequire(path.join(CLI, "package.json")).resolve("@anthropic-ai/sdk"))).default;
  const an = new Anthropic({ baseURL: `http://127.0.0.1:${port}`, apiKey: "pooled", maxRetries: 0, timeout: 900000 });

  const TOOLS = [
    { name: "get_weather", description: "Get the current weather in a city.", input_schema: { type: "object", properties: { city: { type: "string" }, unit: { type: "string", enum: ["celsius", "fahrenheit"] } }, required: ["city"] } },
    { name: "search", description: "Search the web.", input_schema: { type: "object", properties: { query: { type: "string" }, limit: { type: "integer" } }, required: ["query"] } },
  ];
  const base = { model: "claude-x", max_tokens: 300, temperature: 0, tools: TOOLS };
  const uses = (m) => m.content.filter((b) => b.type === "tool_use");
  const textOf = (m) => m.content.filter((b) => b.type === "text").map((b) => b.text).join("");
  const Q = "What's the weather in Paris right now?";

  let first = null;
  await soft("tool_use", async () => {
    const m = first = await an.messages.create({ ...base, messages: [{ role: "user", content: Q }] });
    const u = uses(m);
    check("non-stream: a get_weather tool_use for Paris, stop_reason tool_use", m.stop_reason === "tool_use" && u.length >= 1 && u[0].name === "get_weather" && /paris/i.test(u[0].input.city) && /^toolu_/.test(u[0].id), JSON.stringify(m));
    const s = await an.messages.stream({ ...base, messages: [{ role: "user", content: Q }] }).finalMessage();
    check("stream: the same call through content_block_start / input_json_delta", s.stop_reason === "tool_use" && JSON.stringify(uses(s).map((b) => [b.name, b.input])) === JSON.stringify(u.map((b) => [b.name, b.input])), JSON.stringify(s));
  });
  await soft("tool_result", async () => {
    if (!uses(first || { content: [] }).length) throw new Error("no first call");
    const msgs = [{ role: "user", content: Q }, { role: "assistant", content: first.content },
      { role: "user", content: uses(first).map((b) => ({ type: "tool_result", tool_use_id: b.id, content: '{"temp_c": 18, "sky": "cloudy"}' })) }];
    const m = await an.messages.stream({ ...base, messages: msgs }).finalMessage();
    check("tool_result follow-up: an answer using it, end_turn", m.stop_reason === "end_turn" && /18/.test(textOf(m)) && !uses(m).length, JSON.stringify(m));
    check("tool_result follow-up: cache_read_input_tokens covers the first turn", m.usage.cache_read_input_tokens >= first.usage.input_tokens + first.usage.cache_read_input_tokens, JSON.stringify({ first: first.usage, next: m.usage }));
  });
  await soft("tool_choice", async () => {
    const any = await an.messages.create({ ...base, tool_choice: { type: "any" }, messages: [{ role: "user", content: "Hi! How are you?" }] });
    check("tool_choice any: a call even for small talk", any.stop_reason === "tool_use" && uses(any).length >= 1, JSON.stringify(any));
    const named = await an.messages.create({ ...base, tool_choice: { type: "tool", name: "search" }, messages: [{ role: "user", content: Q }] });
    check("tool_choice tool: the named tool", named.stop_reason === "tool_use" && uses(named).length === 1 && uses(named)[0].name === "search" && typeof uses(named)[0].input.query === "string", JSON.stringify(named));
    const none = await an.messages.create({ ...base, max_tokens: 80, tool_choice: { type: "none" }, messages: [{ role: "user", content: Q }] });
    check("tool_choice none: no call, no markup", !uses(none).length && !/<tool|<function/.test(textOf(none)), JSON.stringify(none));
    const one = await an.messages.create({ ...base, tool_choice: { type: "auto", disable_parallel_tool_use: true }, messages: [{ role: "user", content: "What's the weather in Paris and in Tokyo? Call the tool once per city, both at once." }] });
    check("disable_parallel_tool_use: one call", uses(one).length === 1, JSON.stringify(one));
  });
  await soft("thinking", async () => {
    const req = { ...base, max_tokens: 800, thinking: { type: "adaptive", display: "omitted" }, messages: [{ role: "user", content: "What's the weather in Rome?" }] };
    const m = await an.messages.stream(req).finalMessage();
    const th = m.content[0];
    const r = blob.decode(th?.signature);
    check("thinking omitted: an empty thinking block whose signature carries the reasoning, then the call", th?.type === "thinking" && th.thinking === "" && r && r.length > 20 && uses(m).length >= 1, JSON.stringify(m).slice(0, 600));
    const next = await an.messages.create({ ...req, messages: [...req.messages, { role: "assistant", content: m.content }, { role: "user", content: uses(m).map((b) => ({ type: "tool_result", tool_use_id: b.id, content: "sunny, 24 C" })) }] });
    check("thinking round trip: the follow-up answers and reuses the reasoning-bearing prefix", /24/.test(textOf(next)) && next.usage.cache_read_input_tokens >= m.usage.input_tokens + m.usage.cache_read_input_tokens, JSON.stringify({ text: textOf(next), first: m.usage, next: next.usage }));
  });
  await soft("stop", async () => {
    const m = await an.messages.create({ model: "x", max_tokens: 80, temperature: 0, stop_sequences: ["\n"], messages: [{ role: "user", content: "List three fruits, one per line, nothing else." }] });
    check("stop_sequences: stop_reason stop_sequence", m.stop_reason === "stop_sequence" && m.stop_sequence === "\n" && !textOf(m).includes("\n"), JSON.stringify(m));
  });
  if (arg("claude") != null || process.argv.includes("--claude")) await soft("claude code", async () => {
    const work = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-cc-work-")), cfg = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-cc-cfg-"));
    const word = "lighthouse-" + Math.random().toString(36).slice(2, 7);
    fs.writeFileSync(path.join(work, "notes.txt"), `The secret word is ${word}.\n`);
    const t = Date.now();
    const out = await new Promise((resolve) => {
      const p = spawn("claude", ["-p", "Read notes.txt and tell me the secret word in it.", "--model", "pooled", "--allowedTools", "Read", "--output-format", "json"], { cwd: work,
        env: { ...Object.fromEntries(Object.entries(process.env).filter(([k]) => !/^(CLAUDE|ANTHROPIC_)/.test(k))), CLAUDE_CONFIG_DIR: cfg, ANTHROPIC_BASE_URL: `http://127.0.0.1:${port}`, ANTHROPIC_API_KEY: "pooled", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: "1" } });
      let s = ""; p.stdout.on("data", (c) => (s += c)); p.stderr.on("data", (c) => (s += c));
      const kill = setTimeout(() => p.kill(), 1800000);
      p.on("close", () => { clearTimeout(kill); resolve(s); });
    });
    const line = out.split("\n").find((l) => l.startsWith("{\""));
    let j = null; try { j = JSON.parse(line); } catch {}
    check("Claude Code: reads the file with its Read tool and answers with the word", j && !j.is_error && j.result.includes(word) && j.num_turns >= 2, JSON.stringify({ out: out.slice(-800), s: (Date.now() - t) / 1000 }));
    log("claude code:", JSON.stringify({ result: j?.result, turns: j?.num_turns, usage: j?.usage && { in: j.usage.input_tokens, cached: j.usage.cache_read_input_tokens, out: j.usage.output_tokens }, s: (Date.now() - t) / 1000 }));
    fs.rmSync(work, { recursive: true, force: true }); fs.rmSync(cfg, { recursive: true, force: true });
  });
  check("no page errors", pageErrs.length === 0, pageErrs.join(" | "));
} catch (e) {
  check("run", false, e.stack || e);
} finally {
  try { api?.closeAll("e2e over"); api?.server.close(); } catch {}
  try { await bridge?.leave(); } catch {}
  await ctx.close().catch(() => {});
  peerServer.kill(); srv.close(); wsrv.close();
  fs.rmSync(prof, { recursive: true, force: true });
}
const failed = checks.filter((c) => !c.ok);
console.log(JSON.stringify({ model: MODEL, ok: !failed.length, passed: checks.length - failed.length, failed: failed.length, checks }));
process.exit(failed.length ? 1 : 0);
