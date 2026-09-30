// `pooled serve` Responses API end to end, on the GPU: a real host room (headless Chromium, the model on
// this machine's GPU), `pooled serve` joined to it (cli/bin/pooled.js), and the openai SDK plus the
// Codex CLI against its /v1/responses. Manual trigger only (a GPU and the model files; through
// gpurun.sh on the Spark):
//
//   (cd cli && npm install) && npm install && ln -s ~/bello/models models
//   node tests/e2e/serve_responses.mjs [--model qwen3-1.7b | qwen3.6-35b-moe] [--gb 8] [--port 8443] [--query "ctx=32768"] [--no-codex]
//
// Checks: a plain response (non-stream); a streamed function call (event order, sequence numbers,
// argument deltas = final arguments, ids equal between stream and final); previous_response_id through
// the call's output, reusing the room's caches (cached_tokens); reasoning items with
// encrypted_content restoring the reasoning; text.format json_schema; tool_choice required;
// GET / DELETE of a stored response; Codex CLI reading a file with exec_command (agents/codex.mjs).
// Prints one JSON line with every check; exit code 0 only when all passed.
//
// 2026-09-29 on the Spark (--query ctx=32768): Qwen3.6 35B MoE 21 of 21. Qwen3 1.7B 19 of 21: every
// API check passes, but under Codex's long prompt the 1.7B says it will run the command instead of
// calling exec_command (a model limit; use the MoE, or --no-codex, for the 1.7B).
import { chromium } from "playwright";
import http from "http";
import https from "https";
import fs from "fs";
import os from "os";
import path from "path";
import net from "net";
import { spawn, execSync } from "child_process";
import { createRequire } from "module";

const arg = (k, d) => { const i = process.argv.indexOf("--" + k); return i > 0 ? process.argv[i + 1] : d; };
const flag = (k) => process.argv.includes("--" + k);
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");
const CLI = path.join(ROOT, "cli");
const MODEL = arg("model", "qwen3-1.7b");
const MOE = /moe|27b/.test(MODEL);
const GBV = arg("gb", MOE ? "40" : "8");
const PORT = +arg("port", 8443), TLS_PORT = PORT + 1, SIG_PORT = PORT + 2;
const LOCAL = { "Qwen3-1.7B-Q8_0.gguf": "models/qwen17/model.gguf", "Qwen3.8-27B-Q4_0.gguf": "models/q38/model.gguf", "Qwen_Qwen3.6-35B-A3B-Q4_0.gguf": "models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf" };
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png" };
const t0 = Date.now(); const log = (...a) => console.error(((Date.now() - t0) / 1000).toFixed(1) + "s", ...a);

const srv = http.createServer((q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { r.statusCode = 404; r.end(); return; }
  r.setHeader("content-type", MIME[path.extname(p)] || "application/octet-stream"); fs.createReadStream(p).pipe(r);
}).listen(PORT, "127.0.0.1");
const tlsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-resp-e2e-"));
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
const prof = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-resp-prof-"));
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
const freePort = () => new Promise((res) => { const s = net.createServer().listen(0, "127.0.0.1", () => { const p = s.address().port; s.close(() => res(p)); }); });
let serve = null;
try {
  await page.goto(`http://127.0.0.1:${PORT}/p2p.html?ask=0&signal=127.0.0.1:${SIG_PORT}&dev=1` + (arg("query") ? "&" + arg("query") : ""));
  await page.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
  await page.waitForFunction(() => document.getElementById("join-gb").value !== "1", null, { timeout: 45000 }).catch(() => {});
  await page.fill("#name-input", "spark-host"); await page.fill("#join-gb", GBV);
  await page.click("#create-btn");
  await page.waitForFunction(() => /[A-Z0-9]{3}-?[A-Z0-9]{3}|[A-Z0-9]{4}/.test(document.getElementById("side-code").textContent), null, { timeout: 30000 });
  const code = (await page.textContent("#side-code")).trim().replace("-", "").match(/[A-Z0-9]{6}|[A-Z0-9]{4}/)[0];
  log("room", code);
  await page.selectOption("#ai-model", MODEL);
  await page.click("#ai-start");
  await page.waitForFunction(() => document.getElementById("ai-panel").classList.contains("online") || /^failed:/.test(document.getElementById("ai-status").textContent), null, { timeout: 1800000, polling: 1000 });
  if (/^failed:/.test(await page.textContent("#ai-status"))) throw new Error("load " + await page.textContent("#ai-status"));
  log("online:", await page.textContent("#ai-status"));

  const P = await freePort(), BASE = `http://127.0.0.1:${P}`;
  serve = spawn(process.execPath, [path.join(CLI, "bin/pooled.js"), "serve", code, "--port", String(P), "--signal", `127.0.0.1:${SIG_PORT}`, "--name", "responses-e2e"], { stdio: ["ignore", "pipe", "pipe"] });
  serve.out = ""; serve.stdout.on("data", (c) => { serve.out += c; }); serve.stderr.on("data", (c) => { serve.out += c; });
  for (let i = 0; ; i++) {
    try { const h = await (await fetch(BASE + "/health")).json(); if (h.ready) break; } catch {}
    if (i > 300) throw new Error("pooled serve not ready: " + serve.out.slice(-400));
    await new Promise((r) => setTimeout(r, 300));
  }
  const require = createRequire(path.join(CLI, "package.json"));
  const { default: OpenAI } = await import(require.resolve("openai"));
  const client = new OpenAI({ baseURL: BASE + "/v1", apiKey: "x", maxRetries: 0, timeout: 600000 });
  const WEATHER = { type: "function", name: "get_weather", description: "Get the current weather in a city.", strict: false,
    parameters: { type: "object", properties: { city: { type: "string" }, unit: { type: "string", enum: ["celsius", "fahrenheit"] } }, required: ["city"] } };
  const Q = "What's the weather in Paris right now?";

  await soft("plain", async () => {
    const r = await client.responses.create({ model: "x", input: "In one short sentence: what is the capital of France?", max_output_tokens: 60, temperature: 0 });
    check("plain: completed, one message, Paris", r.status === "completed" && r.output.length === 1 && r.output[0].type === "message" && /paris/i.test(r.output_text), JSON.stringify(r.output));
    check("plain: usage", r.usage.input_tokens > 0 && r.usage.output_tokens > 0 && r.usage.total_tokens === r.usage.input_tokens + r.usage.output_tokens, JSON.stringify(r.usage));
  });
  let call = null, first = null;
  await soft("stream call", async () => {
    const stream = client.responses.stream({ model: "x", input: Q, tools: [WEATHER, { type: "web_search" }], temperature: 0, max_output_tokens: 200 });
    const evs = [];
    stream.on("event", (e) => evs.push(e));
    first = await stream.finalResponse();
    const seq = evs.map((e) => e.sequence_number);
    check("stream: sequence numbers 0, 1, 2, …", seq.every((n, i) => n === i), JSON.stringify(seq.slice(0, 20)));
    check("stream: created, in_progress first; completed last", evs[0]?.type === "response.created" && evs[1]?.type === "response.in_progress" && evs.at(-1)?.type === "response.completed", evs.map((e) => e.type).join(","));
    call = first.output.find((o) => o.type === "function_call");
    let args = null; try { args = JSON.parse(call?.arguments); } catch {}
    check("stream: a get_weather call for Paris", call?.name === "get_weather" && /paris/i.test(args?.city || ""), JSON.stringify(first.output));
    const deltas = evs.filter((e) => e.type === "response.function_call_arguments.delta" && e.item_id === call?.id).map((e) => e.delta).join("");
    check("stream: argument deltas join into the final arguments", call && deltas === call.arguments, JSON.stringify({ deltas, args: call?.arguments }));
    const added = evs.find((e) => e.type === "response.output_item.added" && e.item.type === "function_call");
    check("stream: the streamed item ids are the final ones", added?.item.id === call?.id && added?.item.call_id === call?.call_id && evs[0].response.id === first.id, JSON.stringify({ added: added?.item, call }));
    check("stream: no call markup in any message", !first.output.some((o) => o.type === "message" && /<tool_call>|<function=/.test(o.content[0].text)), JSON.stringify(first.output));
  });
  await soft("chain", async () => {
    if (!call) throw new Error("no call to answer");
    const r = await client.responses.create({ model: "x", previous_response_id: first.id, tools: [WEATHER], temperature: 0, max_output_tokens: 120,
      input: [{ type: "function_call_output", call_id: call.call_id, output: '{"temp_c": 18, "sky": "cloudy"}' }] });
    check("previous_response_id: an answer using the tool's output", r.status === "completed" && /18/.test(r.output_text), JSON.stringify(r.output));
    check("previous_response_id: the room's caches held the first turn (cached_tokens)", r.usage.input_tokens_details.cached_tokens >= first.usage.input_tokens, JSON.stringify({ now: r.usage, first: first.usage }));
    const got = await client.responses.retrieve(r.id);
    check("GET /v1/responses/{id}: the stored response", got.id === r.id && got.output_text === r.output_text, JSON.stringify(got).slice(0, 200));
    await client.responses.delete(r.id);
    let gone = false; try { await client.responses.retrieve(r.id); } catch (e) { gone = e.status === 404; }
    check("DELETE: then 404", gone);
  });
  await soft("reasoning", async () => {
    const r = await client.responses.create({ model: "x", input: "What's the weather in Rome?", tools: [WEATHER], reasoning: { effort: "low" }, include: ["reasoning.encrypted_content"], store: false, max_output_tokens: 800, temperature: 0 });
    const rs = r.output.find((o) => o.type === "reasoning"), c = r.output.find((o) => o.type === "function_call");
    check("reasoning: a reasoning item, then the call", rs && rs.content[0].text.length > 20 && c && r.output.indexOf(rs) < r.output.indexOf(c) && r.usage.output_tokens_details.reasoning_tokens > 0, JSON.stringify(r.output).slice(0, 400));
    check("reasoning: encrypted_content carries it (pooled1.)", rs?.encrypted_content?.startsWith("pooled1.") && Buffer.from(rs.encrypted_content.slice(8), "base64url").toString() === rs.content[0].text);
    if (!c) return;
    // stateless round trip (store: false): the items back as input, reasoning restored from the blob
    const r2 = await client.responses.create({ model: "x", store: false, tools: [WEATHER], reasoning: { effort: "low" }, max_output_tokens: 600, temperature: 0,
      input: [{ role: "user", content: "What's the weather in Rome?" }, { ...rs, content: [] }, c, { type: "function_call_output", call_id: c.call_id, output: '{"temp_c": 24}' }] });
    check("reasoning: stateless follow-up answers from the tool output, caches reused", /24/.test(r2.output_text) && r2.usage.input_tokens_details.cached_tokens >= r.usage.input_tokens, JSON.stringify({ text: r2.output_text, u: r2.usage, first: r.usage }));
  });
  await soft("format", async () => {
    const schema = { type: "object", properties: { city: { type: "string" }, population_millions: { type: "number" } }, required: ["city", "population_millions"], additionalProperties: false };
    const r = await client.responses.create({ model: "x", input: "Largest city in Japan, and its population in millions?", text: { format: { type: "json_schema", name: "city", schema, strict: true } }, max_output_tokens: 80, temperature: 0 });
    let o = null; try { o = JSON.parse(r.output_text); } catch {}
    check("text.format json_schema: the answer parses and fits", o && typeof o.city === "string" && typeof o.population_millions === "number" && Object.keys(o).length === 2, r.output_text);
  });
  await soft("required", async () => {
    const r = await client.responses.create({ model: "x", input: "Hi! How are you?", tools: [WEATHER], tool_choice: "required", max_output_tokens: 100, temperature: 0 });
    check("tool_choice required: a call even for small talk", r.output.some((o) => o.type === "function_call") && !r.output.some((o) => o.type === "message"), JSON.stringify(r.output));
  });
  if (!flag("no-codex")) await soft("codex", async () => {
    const { runCodex, summarize } = await import("./agents/codex.mjs");
    const run = await runCodex({ base: BASE, prompt: "Run `cat a.txt` and tell me what it says.", files: { "a.txt": "hello from a.txt\n" }, codex: arg("codex", "codex"), contextWindow: +arg("codex-ctx", 32768) });
    const s = summarize(run);
    check("codex: exited 0", run.code === 0, run.stderr.slice(-400) + JSON.stringify(s.failed));
    check("codex: ran cat a.txt through exec_command", s.commands.some((c) => /a\.txt/.test(c.command) && /hello from a\.txt/.test(c.output || "")), JSON.stringify(s.commands));
    check("codex: its answer quotes the file", /hello from a\.txt/i.test(s.answer || ""), JSON.stringify(s));
  });
  check("no page errors", pageErrs.length === 0, pageErrs.join(" | "));
} catch (e) {
  check("run", false, e.stack || e);
} finally {
  if (serve) { serve.kill("SIGINT"); await new Promise((r) => setTimeout(r, 500)); }
  await ctx.close().catch(() => {});
  peerServer.kill(); srv.close(); wsrv.close();
  fs.rmSync(prof, { recursive: true, force: true });
}
const failed = checks.filter((c) => !c.ok);
console.log(JSON.stringify({ model: MODEL, ok: !failed.length, passed: checks.length - failed.length, failed: failed.length, checks }));
process.exit(failed.length ? 1 : 0);
