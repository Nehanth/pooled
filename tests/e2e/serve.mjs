// `pooled serve` end to end: a local PeerJS server, a real host room in headless Chromium running
// Qwen3 1.7B (models/qwen17) on this machine's GPU, and the CLI bridge (cli/) joined to it. Then plain
// fetch (and the official openai / @anthropic-ai/sdk clients from cli's devDependencies) against the
// bridge's HTTP endpoint. Manual trigger only (it needs a GPU and the model file):
//
//   (cd cli && npm install) && npm install
//   node tests/e2e/serve.mjs [--model qwen3-1.7b | qwen3.6-35b-moe] [--gb 8] [--port 8243] [--query "ctx=65536"]
//                            [--codex] [--claude] [--keep]
//
// (models/ is a symlink to the GGUF folder, e.g. ln -s ~/bello/models models; on the Spark run it
// through gpurun.sh.) --codex and --claude also run one real agent turn (OpenAI Codex CLI over the
// Responses API, Claude Code over Messages) that reads a file with its own tool; their prompts are
// 15-20 k tokens, so give the room a context with --query "ctx=65536".
//
// Checks, in order: /v1/models in both shapes; OpenAI non-stream and stream; Anthropic non-stream and
// stream (event order); temperature 0 twice gives the same text; turn 2 resending turn 1 reuses the
// room's caches (cached_tokens > 0); a second concurrent request queues (keep-alives) and completes
// after the first; stop sequences on both APIs; Chat Completions tool calls (whole, streamed, results, named, parallel off,
// JSON schema); the same weather round trip on the Responses API (streamed call through the SDK,
// previous_response_id with the output, json_schema, tool_choice required / named) and on Messages
// (tool_use, streamed input_json_delta, tool_result, tool_choice any / tool, output_format); a Responses
// custom tool; Chat custom tools -> 400; an image -> a note; tool schemas too large together -> 400 from
// the host, and the room still answers; the context in /health and /v1/models; Origin -> 403, --token -> 401; the SDKs
// parse both APIs; a client that goes away mid-stream frees the room; the host's card says API
// client; the host's Disconnect ends the bridge's link and it answers 503 without reconnecting.
// Prints one JSON line with every check; exit code 0 only when all passed.
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
const BIG = /moe|27b/.test(MODEL);
const GBV = arg("gb", BIG ? "40" : "8");
const QUERY = arg("query", "");
const CTX = +(/ctx=(\d+)/.exec(QUERY)?.[1] || 0);
const PORT = +arg("port", 8243), TLS_PORT = PORT + 1, SIG_PORT = PORT + 2;
const LOCAL = { "Qwen3-0.6B-Q8_0.gguf": "models/qwen/model.gguf", "Qwen3-1.7B-Q8_0.gguf": "models/qwen17/model.gguf",
  "Qwen3.8-27B-Q4_0.gguf": "models/q38/model.gguf", "Qwen_Qwen3.6-35B-A3B-Q4_0.gguf": "models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf" };
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png" };
const t0 = Date.now(); const log = (...a) => console.error(((Date.now() - t0) / 1000).toFixed(1) + "s", ...a);

if (!fs.existsSync(path.join(CLI, "node_modules/peerjs"))) { console.error("run (cd cli && npm install) first"); process.exit(2); }
const freePort = () => new Promise((res) => { const s = net.createServer().listen(0, "127.0.0.1", () => { const p = s.address().port; s.close(() => res(p)); }); });

// the site, and the weights over https with Range support from local disk (as tests/e2e/room.mjs)
const srv = http.createServer((q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { r.statusCode = 404; r.end(); return; }
  r.setHeader("content-type", MIME[path.extname(p)] || "application/octet-stream"); fs.createReadStream(p).pipe(r);
}).listen(PORT, "127.0.0.1");
const tlsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-serve-e2e-"));
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

// a profile on disk: the big models' weights do not fit an in-memory Cache API store
const prof = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-serve-prof-"));
const args = ["--no-sandbox", "--headless=new", "--enable-unsafe-webgpu", "--enable-webgpu-developer-features", "--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--allow-loopback-in-peer-connection", "--js-flags=--max-old-space-size=65536"];
const ctx = await chromium.launchPersistentContext(prof, { headless: false, args, ignoreHTTPSErrors: true, userAgent: "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36" });
await ctx.route("**/*.gguf", (route) => {
  const file = LOCAL[route.request().url().split("/").pop().split("?")[0]];
  if (!file || !fs.existsSync(path.join(ROOT, file))) return route.continue();
  return route.continue({ url: `https://127.0.0.1:${TLS_PORT}/${file}` });
});
// the room page's PeerJS from cli/node_modules: the same 1.5.4 build, no CDN needed
await ctx.route(/cdn\.jsdelivr\.net\/npm\/peerjs@/, (r) => r.fulfill({ path: path.join(CLI, "node_modules/peerjs/dist/peerjs.min.js"), contentType: "text/javascript" }));
const page = await ctx.newPage();
const pageErrs = [];
page.on("pageerror", (e) => pageErrs.push(String(e).slice(0, 200)));
page.on("console", (m) => { if (m.type() === "error" && !/favicon|404/.test(m.text())) pageErrs.push(m.text().slice(0, 200)); });

const bridges = [];
function startBridge(code, port, extra = []) {
  const b = spawn(process.execPath, [path.join(CLI, "bin/pooled.js"), "serve", code, "--port", String(port), "--signal", `127.0.0.1:${SIG_PORT}`, ...extra],
    { env: { ...process.env, POOLED_KEEPALIVE_MS: "1000" }, stdio: ["ignore", "pipe", "pipe"] });
  b.out = "";
  b.stdout.on("data", (c) => { b.out += c; });
  b.stderr.on("data", (c) => { b.out += c; if (flag("verbose")) process.stderr.write("[bridge] " + c); });
  bridges.push(b);
  return b;
}
async function waitHealth(base, pred, ms = 60000, headers = {}) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    try { const h = await (await fetch(base + "/health", { headers })).json(); if (pred(h)) return h; } catch {}
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error("bridge not ready in time");
}
// a streamed response -> events [{ event, data }] (data parsed unless [DONE]), plus comments seen
async function readSSE(res, onEvent = () => {}) {
  const dec = new TextDecoder(); let buf = ""; const events = []; let comments = 0;
  for await (const chunk of res.body) {
    buf += dec.decode(chunk, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, i); buf = buf.slice(i + 2);
      if (block.startsWith(":")) { comments++; continue; }
      const ev = {}; for (const line of block.split("\n")) { if (line.startsWith("event: ")) ev.event = line.slice(7); else if (line.startsWith("data: ")) ev.data = line.slice(6); }
      if (ev.data && ev.data !== "[DONE]") ev.data = JSON.parse(ev.data);
      events.push(ev); onEvent(ev);
    }
  }
  return { events, comments, raw: buf };
}

const checks = [];
const check = (name, cond, detail = "") => { checks.push({ name, ok: !!cond, ...(cond ? {} : { detail: String(detail).slice(0, 300) }) }); log(cond ? "ok  " : "FAIL", name, cond ? "" : String(detail).slice(0, 300)); };
const soft = async (name, fn) => { try { await fn(); } catch (e) { check(name, false, e.stack || e); } };
let code = null;
try {
  await page.goto(`http://127.0.0.1:${PORT}/p2p.html?signal=127.0.0.1:${SIG_PORT}&dev=1` + (QUERY ? "&" + QUERY : ""));
  await page.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
  await page.waitForFunction(() => document.getElementById("join-gb").value !== "1", null, { timeout: 45000 }).catch(() => {});
  await page.fill("#name-input", "spark-host"); await page.fill("#join-gb", GBV);
  await page.click("#create-btn");
  await page.waitForFunction(() => /[A-Z0-9]{4}/.test(document.getElementById("side-code").textContent), null, { timeout: 30000 });
  code = (await page.textContent("#side-code")).trim().match(/[A-Z0-9]{4}/)[0];
  log("room", code);

  // the bridge can join before the model runs: it serves 503 + Retry-After until the room is ready
  const P = await freePort(), BASE = `http://127.0.0.1:${P}`;
  const bridge = startBridge(code, P);
  await waitHealth(BASE, (h) => h.connected);
  const early = await fetch(BASE + "/v1/chat/completions", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ model: "x", messages: [{ role: "user", content: "hi" }] }) });
  check("before the model is ready: 503 with Retry-After", early.status === 503 && early.headers.get("retry-after") === "5", early.status);
  const emptyModels = await (await fetch(BASE + "/v1/models")).json();
  check("before the model is ready: /v1/models is empty", Array.isArray(emptyModels.data) && emptyModels.data.length === 0, JSON.stringify(emptyModels));
  await page.waitForFunction(() => [...document.querySelectorAll(".peer-card")].some((c) => /API client/.test(c.textContent)), null, { timeout: 15000 });
  check("host: the bridge's card says API client and has Disconnect", await page.evaluate(() => { const c = [...document.querySelectorAll(".peer-card")].find((c) => /API client/.test(c.textContent)); return !!c?.querySelector(".api-kick"); }));
  check("host: the log says API client … joined", await page.evaluate(() => [...document.querySelectorAll("#chat-log div")].some((d) => /API client .* joined/.test(d.textContent))));
  check("host: the header does not count the API client as a device", await page.evaluate(() => /1 device · 1 API client/.test(document.getElementById("cluster-summary").textContent)), await page.textContent("#cluster-summary"));

  await page.selectOption("#ai-model", MODEL);
  await page.click("#ai-start");
  log("model start pressed");
  await page.waitForFunction(() => document.getElementById("ai-panel").classList.contains("online") || /^failed:/.test(document.getElementById("ai-status").textContent), null, { timeout: 1800000, polling: 1000 });
  if (/^failed:/.test(await page.textContent("#ai-status"))) throw new Error("load " + await page.textContent("#ai-status"));
  log("online:", await page.textContent("#ai-status"));
  const H = await waitHealth(BASE, (h) => h.ready, 30000);
  check("health: ready with the room's model", H.ready && H.model === `pooled/${MODEL}`, JSON.stringify(H));
  const MID = `pooled/${MODEL}`;
  const J = { "content-type": "application/json" };
  const post = (p, body, headers = {}) => fetch(BASE + p, { method: "POST", headers: { ...J, ...headers }, body: JSON.stringify(body) });
  const Q = "In one short sentence: what is the capital of France?";

  // a guest that joins after the model runs (an ask-only device, no layers): the host's hello says it
  // serves API clients, which must not make the guest show the host as one
  await soft("guest view", async () => {
    const g = await ctx.newPage();
    g.on("pageerror", (e) => pageErrs.push("guest: " + String(e).slice(0, 200)));
    await g.goto(`http://127.0.0.1:${PORT}/p2p.html?signal=127.0.0.1:${SIG_PORT}&dev=1`);
    await g.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
    await g.fill("#name-input", "guest"); await g.fill("#code-input", code); await g.click("#join-btn");
    await g.waitForFunction(() => [...document.querySelectorAll(".peer-card")].some((c) => /spark-host/.test(c.textContent)), null, { timeout: 30000 });
    await g.waitForTimeout(1500);
    const hostCard = await g.evaluate(() => [...document.querySelectorAll(".peer-card")].find((c) => /spark-host/.test(c.textContent))?.textContent || "");
    check("guest: the host's card is a computer, not an API client", hostCard && !/API client/.test(hostCard), hostCard.slice(0, 200));
    await g.close();
  });
  // changing an answer style must not turn API clients off (styleChanged kept only the style fields)
  await soft("style change", async () => {
    for (const v of ["concise", "default"]) await page.evaluate((v) => { const s = document.getElementById("ai-persona"); s.value = v; s.dispatchEvent(new Event("change")); }, v);
    const r = await post("/v1/chat/completions", { model: "x", temperature: 0, max_tokens: 8, messages: [{ role: "user", content: "Say hi." }] });
    check("after an answer-style change: API requests still answered", r.status === 200, r.status + " " + (await r.text()).slice(0, 200));
  });

  await soft("models", async () => {
    const m = await (await fetch(BASE + "/v1/models")).json();
    check("GET /v1/models (OpenAI shape)", m.object === "list" && m.data[0]?.id === MID && m.data[0]?.owned_by === "pooled", JSON.stringify(m));
    const a = await (await fetch(BASE + "/v1/models", { headers: { "anthropic-version": "2023-06-01" } })).json();
    check("GET /v1/models (Anthropic shape)", a.data?.[0]?.type === "model" && a.data[0].id === MID && /Pooled room/.test(a.data[0].display_name) && a.first_id === MID, JSON.stringify(a));
    const one = await fetch(BASE + "/v1/models/" + encodeURIComponent(MID));
    check("GET /v1/models/{id}", one.status === 200 && (await one.json()).id === MID);
  });

  let turn1 = null;
  await soft("openai non-stream", async () => {
    const r = await post("/v1/chat/completions", { model: "gpt-4o", messages: [{ role: "user", content: Q }], temperature: 0, max_tokens: 60 });
    const j = await r.json();
    turn1 = j.choices?.[0]?.message?.content;
    check("OpenAI non-stream: 200, text, usage", r.status === 200 && j.object === "chat.completion" && j.model === MID && turn1?.length > 0 && j.usage.prompt_tokens > 0 && j.usage.completion_tokens > 0
      && ["stop", "length"].includes(j.choices[0].finish_reason), JSON.stringify(j));
    check("OpenAI non-stream: answers the question", /paris/i.test(turn1 || ""), turn1);
    const again = await (await post("/v1/chat/completions", { model: "x", messages: [{ role: "user", content: Q }], temperature: 0, max_tokens: 60 })).json();
    check("temperature 0 twice: the same text", again.choices?.[0]?.message?.content === turn1, JSON.stringify([turn1, again.choices?.[0]?.message?.content]));
  });

  await soft("openai stream", async () => {
    const r = await post("/v1/chat/completions", { model: "x", messages: [{ role: "user", content: "Count from 1 to 5, separated by spaces." }], temperature: 0, max_tokens: 40, stream: true, stream_options: { include_usage: true } });
    check("OpenAI stream: text/event-stream", r.status === 200 && /text\/event-stream/.test(r.headers.get("content-type")), r.status + " " + r.headers.get("content-type"));
    const { events } = await readSSE(r);
    const data = events.map((e) => e.data);
    const text = data.filter((d) => d !== "[DONE]" && d.choices?.[0]?.delta?.content).map((d) => d.choices[0].delta.content).join("");
    const fin = data.find((d) => d !== "[DONE]" && d.choices?.[0]?.finish_reason);
    const usage = data.find((d) => d !== "[DONE]" && d.usage);
    check("OpenAI stream: role chunk, content chunks, finish, usage, [DONE]",
      data[0]?.choices?.[0]?.delta?.role === "assistant" && data.filter((d) => d?.choices?.[0]?.delta?.content).length > 3 && !!fin && usage?.usage?.completion_tokens > 0 && data[data.length - 1] === "[DONE]"
      && data.every((d) => d === "[DONE]" || (d.object === "chat.completion.chunk" && d.id === data[0].id)), JSON.stringify(data.slice(0, 3)) + " … " + JSON.stringify(data.slice(-3)));
    check("OpenAI stream: counts", /1\D+2\D+3\D+4\D+5/.test(text), text);
  });

  await soft("anthropic non-stream", async () => {
    const r = await post("/v1/messages", { model: "claude-sonnet-4-5", max_tokens: 60, temperature: 0, system: "Answer briefly.", messages: [{ role: "user", content: Q }] }, { "anthropic-version": "2023-06-01", "x-api-key": "anything" });
    const j = await r.json();
    check("Anthropic non-stream: 200, message, usage", r.status === 200 && j.type === "message" && j.role === "assistant" && j.content?.[0]?.type === "text" && /paris/i.test(j.content[0].text)
      && ["end_turn", "max_tokens"].includes(j.stop_reason) && j.usage.input_tokens > 0 && j.usage.output_tokens > 0 && j.model === MID, JSON.stringify(j));
  });

  await soft("anthropic stream", async () => {
    const r = await post("/v1/messages", { model: "x", max_tokens: 40, temperature: 0, stream: true, messages: [{ role: "user", content: "Say hello in French." }] }, { "anthropic-version": "2023-06-01" });
    const { events } = await readSSE(r);
    const kinds = events.map((e) => e.event);
    // without thinking the ping comes before the first block: the endpoint cannot know yet whether text
    // or a tool call comes first (serve.md 4, Messages); Anthropic sends it after content_block_start
    const order = ["message_start", "ping", "content_block_start"];
    const text = events.filter((e) => e.event === "content_block_delta").map((e) => e.data.delta.text).join("");
    check("Anthropic stream: event order", order.every((k, i) => kinds[i] === k) && kinds.slice(-3).join() === "content_block_stop,message_delta,message_stop"
      && kinds.filter((k) => k === "content_block_delta").length > 1 && events.every((e) => e.data?.type === e.event)
      && events[0].data.message.usage.input_tokens > 0 && events[events.length - 2].data.usage.output_tokens > 0, kinds.join(","));
    check("Anthropic stream: text", text.length > 0, text);
  });

  await soft("reuse", async () => {
    if (!turn1) throw new Error("no turn 1");
    const r = await (await post("/v1/chat/completions", { model: "x", temperature: 0, max_tokens: 30, messages: [{ role: "user", content: Q }, { role: "assistant", content: turn1 }, { role: "user", content: "And of Italy?" }] })).json();
    // turn 1 ran again just before (the temperature 0 check), then the stream and Anthropic requests
    // replaced the caches: run turn 1 once more so the caches hold it, then turn 2
    await post("/v1/chat/completions", { model: "x", messages: [{ role: "user", content: Q }], temperature: 0, max_tokens: 60 }).then((x) => x.json());
    const r2 = await (await post("/v1/chat/completions", { model: "x", temperature: 0, max_tokens: 30, messages: [{ role: "user", content: Q }, { role: "assistant", content: turn1 }, { role: "user", content: "And of Italy?" }] })).json();
    const cached = r2.usage?.prompt_tokens_details?.cached_tokens || 0;
    check("turn 2 resends turn 1: the room reuses its caches (exact ids)", cached > 0 && cached < r2.usage.prompt_tokens, JSON.stringify(r2.usage));
    check("turn 2 answers", /rome/i.test(r2.choices?.[0]?.message?.content || r.choices?.[0]?.message?.content || ""), r2.choices?.[0]?.message?.content);
  });

  await soft("queue", async () => {
    const t = Date.now();
    const a = post("/v1/chat/completions", { model: "x", temperature: 0, max_tokens: 200, stream: true, messages: [{ role: "user", content: "Write a short paragraph about the sea." }] });
    await new Promise((r) => setTimeout(r, 150));
    const b = post("/v1/chat/completions", { model: "x", temperature: 0, max_tokens: 20, stream: true, messages: [{ role: "user", content: "Say OK." }] });
    let aDone = 0, bFirst = 0;
    const [ra, rb] = await Promise.all([a, b]);
    const [sa, sb] = await Promise.all([
      readSSE(ra).then((x) => { aDone = Date.now(); return x; }),
      readSSE(rb, (ev) => { if (!bFirst && ev.data?.choices?.[0]?.delta?.content) bFirst = Date.now(); }),
    ]);
    check("queue: the second request's first token comes after the first request finished", bFirst >= aDone && aDone > t, JSON.stringify({ aMs: aDone - t, bFirstMs: bFirst - t }));
    check("queue: the waiting stream got keep-alive comments", sb.comments > 0 || aDone - t < 1200, `${sb.comments} comments, first took ${aDone - t} ms`);
    check("queue: both complete", sa.events.at(-1)?.data === "[DONE]" && sb.events.at(-1)?.data === "[DONE]");
  });

  await soft("stop", async () => {
    const r = await (await post("/v1/chat/completions", { model: "x", temperature: 0, max_tokens: 80, stop: ["\n"], messages: [{ role: "user", content: "List three fruits, one per line, nothing else." }] })).json();
    const c = r.choices?.[0]?.message?.content ?? "";
    check("OpenAI stop: ends before the first newline", r.choices?.[0]?.finish_reason === "stop" && !c.includes("\n") && c.length > 0, JSON.stringify(r.choices?.[0]));
    const a = await (await post("/v1/messages", { model: "x", max_tokens: 80, temperature: 0, stop_sequences: ["\n"], messages: [{ role: "user", content: "List three fruits, one per line, nothing else." }] })).json();
    check("Anthropic stop_sequences: stop_reason stop_sequence", a.stop_reason === "stop_sequence" && a.stop_sequence === "\n" && !a.content[0].text.includes("\n"), JSON.stringify(a));
  });

  await soft("errors", async () => {
    const t = await post("/v1/chat/completions", { model: "x", messages: [{ role: "user", content: "hi" }], tools: [{ type: "custom", custom: { name: "f" } }] });
    const tj = await t.json();
    check("OpenAI custom tools: 400 with the message", t.status === 400 && /custom tools are not supported/.test(tj.error?.message), JSON.stringify(tj));
    // an image in a user turn becomes a note (a pasted screenshot must not break every later request)
    const at = await post("/v1/messages", { model: "x", max_tokens: 5, messages: [{ role: "user", content: [{ type: "text", text: "What is this?" }, { type: "image", source: { type: "base64", media_type: "image/png", data: "" } }] }] });
    check("Anthropic image: answered, with a note in its place", at.status === 200 && (await at.json()).type === "message");
    // max_tokens past 65536 is capped on every API; no max_tokens on Chat means 16384
    const big = await post("/v1/chat/completions", { model: "x", max_tokens: 100000, messages: [{ role: "user", content: "Say hi." }] });
    check("OpenAI max_tokens 100000: capped, answered", big.status === 200, big.status);
    // tool schemas that fit one by one but not together: the host refuses them up front, and the room
    // is not left busy
    // (small in characters, so the bridge's early context check lets it through on a 16 k room too:
    // eight arrays deep per property, eight grammar nodes for ~90 characters)
    const deep = () => { let v = 1; for (let d = 0; d < 8; d++) v = { items: v }; return v; };
    const props = {}; for (let i = 0; i < 330; i++) props[i.toString(36)] = deep();
    const huge = await post("/v1/chat/completions", { model: "x", max_tokens: 5, messages: [{ role: "user", content: "hi" }],
      tools: Array.from({ length: 4 }, (_, i) => ({ type: "function", function: { name: "t" + i, parameters: { type: "object", properties: props } } })) });
    const hj = await huge.json();
    check("tool schemas too large together: 400 from the host", huge.status === 400 && /too large together/.test(hj.error?.message), JSON.stringify(hj).slice(0, 200));
    const after = await post("/v1/chat/completions", { model: "x", max_tokens: 5, messages: [{ role: "user", content: "hi" }] });
    check("… and the room answers the next request", after.status === 200, after.status);
    const hl = await (await fetch(BASE + "/health")).json();
    const ml = await (await fetch(BASE + "/v1/models")).json();
    check("the room's context: /health ctx and /v1/models max_model_len", hl.ctx > 0 && ml.data?.[0]?.max_model_len === hl.ctx, JSON.stringify({ ctx: hl.ctx, m: ml.data?.[0] }));
    const o = await fetch(BASE + "/v1/models", { headers: { origin: "https://evil.example" } });
    check("a request with an Origin: 403", o.status === 403);
    const h = await new Promise((res) => http.get({ host: "127.0.0.1", port: P, path: "/v1/models", headers: { host: "evil.example" } }, (r) => { r.resume(); res(r.statusCode); }));
    check("a foreign Host: 403", h === 403, h);
    const nf = await post("/v1/embeddings", { input: "x" });
    check("/v1/embeddings: 404", nf.status === 404);
    const ctxLong = await post("/v1/chat/completions", { model: "x", messages: [{ role: "user", content: "word ".repeat(Math.max(20000, CTX + 4000)) }] });
    const cj = await ctxLong.json();
    check("a prompt over the context: 400 context_length_exceeded", ctxLong.status === 400 && cj.error?.code === "context_length_exceeded", JSON.stringify(cj).slice(0, 200));
  });

  await soft("token", async () => {
    const P2 = await freePort();
    startBridge(code, P2, ["--token", "s3cret", "--name", "token-bridge"]);
    await waitHealth(`http://127.0.0.1:${P2}`, (h) => h.ready, 30000, { authorization: "Bearer s3cret" });
    const bare = await (await fetch(`http://127.0.0.1:${P2}/health`)).json();
    check("--token: /health without it says only ok (no room code)", JSON.stringify(bare) === '{"ok":true}', JSON.stringify(bare));
    const no = await fetch(`http://127.0.0.1:${P2}/v1/models`);
    check("--token: 401 without it", no.status === 401 && (await no.json()).error?.code === "invalid_api_key");
    const noA = await fetch(`http://127.0.0.1:${P2}/v1/messages`, { method: "POST", headers: { ...J, "x-api-key": "wrong" }, body: "{}" });
    check("--token: Anthropic 401 authentication_error", noA.status === 401 && (await noA.json()).error?.type === "authentication_error");
    const yes = await fetch(`http://127.0.0.1:${P2}/v1/models`, { headers: { authorization: "Bearer s3cret" } });
    check("--token: 200 with it", yes.status === 200);
    const b2 = bridges[bridges.length - 1];
    b2.kill("SIGINT");
    await new Promise((r) => b2.on("exit", r));
    check("Ctrl-C: the bridge exits 0", b2.exitCode === 0, b2.exitCode);
    await page.waitForFunction(() => ![...document.querySelectorAll(".peer-card")].some((c) => /token-bridge/.test(c.textContent)), null, { timeout: 10000 })
      .then(() => check("Ctrl-C: the host drops its card at once (leaving)", true), () => check("Ctrl-C: the host drops its card at once (leaving)", false));
  });

  await soft("sdks", async () => {
    const req = createRequire(path.join(CLI, "package.json"));
    let OpenAI, Anthropic;
    try { OpenAI = (await import(req.resolve("openai"))).default; Anthropic = (await import(req.resolve("@anthropic-ai/sdk"))).default; }
    catch (e) { check("SDKs installed (cli devDependencies)", false, e.message); return; }
    const oa = new OpenAI({ baseURL: BASE + "/v1", apiKey: "anything" });
    const c = await oa.chat.completions.create({ model: "pooled", messages: [{ role: "user", content: Q }], temperature: 0, max_tokens: 40 });
    check("openai SDK: non-stream", /paris/i.test(c.choices[0].message.content) && c.usage.total_tokens > 0, JSON.stringify(c));
    let s = "";
    for await (const ch of await oa.chat.completions.create({ model: "pooled", messages: [{ role: "user", content: Q }], temperature: 0, max_tokens: 40, stream: true })) s += ch.choices[0]?.delta?.content || "";
    check("openai SDK: stream", /paris/i.test(s), s);
    const models = await oa.models.list();
    check("openai SDK: models.list", models.data?.[0]?.id === MID, JSON.stringify(models.data));
    const an = new Anthropic({ baseURL: BASE, apiKey: "anything" });
    const m = await an.messages.create({ model: "claude-x", max_tokens: 40, temperature: 0, messages: [{ role: "user", content: Q }] });
    check("anthropic SDK: non-stream", /paris/i.test(m.content[0].text), JSON.stringify(m));
    const st = an.messages.stream({ model: "claude-x", max_tokens: 40, temperature: 0, messages: [{ role: "user", content: Q }] });
    const fm = await st.finalMessage();
    check("anthropic SDK: stream finalMessage", /paris/i.test(fm.content.map((b) => b.text || "").join("")) && fm.usage.output_tokens > 0, JSON.stringify(fm));
  });

  // Chat Completions with tools (docs/design/serve.md 6): a call, whole and streamed through the SDK,
  // the tool result back, a named choice, parallel off, JSON schema
  await soft("chat tools", async () => {
    const req = createRequire(path.join(CLI, "package.json"));
    const OpenAI = (await import(req.resolve("openai"))).default;
    const oa = new OpenAI({ baseURL: BASE + "/v1", apiKey: "anything" });
    const W = { type: "function", function: { name: "get_weather", description: "Current weather for a city", parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } } };
    const T = { type: "function", function: { name: "get_time", description: "Local time in a city", parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } } };
    const q = [{ role: "user", content: "What is the weather in Paris? Use the tool." }];
    const c = await oa.chat.completions.create({ model: "x", messages: q, tools: [W], temperature: 0, max_tokens: 200 });
    const tc = c.choices[0].message.tool_calls || [];
    check("chat tools: a call, finish_reason tool_calls", c.choices[0].finish_reason === "tool_calls" && tc[0]?.function.name === "get_weather" && /paris/i.test(JSON.parse(tc[0].function.arguments).city) && /^call_[0-9A-Za-z]{24}$/.test(tc[0].id), JSON.stringify(c.choices[0]));
    const st = await oa.chat.completions.stream({ model: "x", messages: q, tools: [W], temperature: 0, max_tokens: 200, stream_options: { include_usage: true } }).finalChatCompletion();
    check("chat tools: streamed calls accumulate to the same arguments", st.choices[0].message.tool_calls?.[0]?.function.arguments === tc[0]?.function.arguments && st.usage?.total_tokens > 0, JSON.stringify(st.choices[0]));
    const f = await oa.chat.completions.create({ model: "x", temperature: 0, max_tokens: 200, tools: [W],
      messages: [...q, c.choices[0].message, { role: "tool", tool_call_id: tc[0]?.id, content: "18 C and sunny" }] });
    check("chat tools: the tool result is used, prompt reused", f.choices[0].finish_reason === "stop" && /18|sunny/i.test(f.choices[0].message.content) && (f.usage.prompt_tokens_details?.cached_tokens || 0) > 0, JSON.stringify(f));
    const n = await oa.chat.completions.create({ model: "x", messages: [{ role: "user", content: "Hi there" }], tools: [W, T], tool_choice: { type: "function", function: { name: "get_time" } }, temperature: 0, max_tokens: 200 });
    check("chat tools: a named tool_choice forces that tool (finish_reason stop)", n.choices[0].message.tool_calls?.[0]?.function.name === "get_time" && n.choices[0].finish_reason === "stop", JSON.stringify(n.choices[0]));
    const p = await oa.chat.completions.create({ model: "x", messages: [{ role: "user", content: "Weather and time in Paris and in Tokyo? Use the tools." }], tools: [W, T], tool_choice: "required", parallel_tool_calls: false, temperature: 0, max_tokens: 300 });
    check("chat tools: required + parallel_tool_calls false gives exactly one call", p.choices[0].message.tool_calls?.length === 1, JSON.stringify(p.choices[0]));
    const j = await oa.chat.completions.create({ model: "x", messages: [{ role: "user", content: "Give the capital of France." }], temperature: 0, max_tokens: 200,
      response_format: { type: "json_schema", json_schema: { name: "capital", schema: { type: "object", properties: { capital: { type: "string" } }, required: ["capital"], additionalProperties: false } } } });
    let jv = null; try { jv = JSON.parse(j.choices[0].message.content); } catch { /* checked below */ }
    check("chat tools: json_schema answer parses and has the key", typeof jv?.capital === "string", JSON.stringify(j.choices[0]));
  });

  // the same weather round trip on the other two APIs, through the official SDKs; every JSON answer is
  // checked against its schema by the small validator below (independent of the host's grammar)
  const SCHEMA = { type: "object", additionalProperties: false, required: ["city", "country", "population_millions", "landmarks", "size"],
    properties: { city: { type: "string" }, country: { type: "string" }, population_millions: { type: "number" },
      landmarks: { type: "array", items: { type: "string" }, minItems: 1, maxItems: 3 }, size: { type: "string", enum: ["small", "medium", "large"] } } };
  const JQ = "Describe the largest city in Japan.";
  const validates = (sc, v) => {
    if (sc.enum && !sc.enum.includes(v)) return false;
    switch (sc.type) {
      case "object":
        if (!v || typeof v !== "object" || Array.isArray(v)) return false;
        if ((sc.required || []).some((k) => !(k in v))) return false;
        if (sc.additionalProperties === false && Object.keys(v).some((k) => !(k in (sc.properties || {})))) return false;
        return Object.entries(sc.properties || {}).every(([k, s2]) => !(k in v) || validates(s2, v[k]));
      case "array": return Array.isArray(v) && v.length >= (sc.minItems ?? 0) && v.length <= (sc.maxItems ?? Infinity) && v.every((x) => validates(sc.items || {}, x));
      case "string": return typeof v === "string";
      case "number": return typeof v === "number" && Number.isFinite(v);
      case "integer": return Number.isInteger(v);
      case "boolean": return typeof v === "boolean";
      default: return true;
    }
  };
  const parsesTo = (text) => { try { return JSON.parse(text); } catch { return undefined; } };
  const sdk = async () => {
    const req = createRequire(path.join(CLI, "package.json"));
    return { OpenAI: (await import(req.resolve("openai"))).default, Anthropic: (await import(req.resolve("@anthropic-ai/sdk"))).default };
  };

  await soft("chat json_schema validates", async () => {
    const { OpenAI } = await sdk();
    const oa = new OpenAI({ baseURL: BASE + "/v1", apiKey: "anything", maxRetries: 0, timeout: 900000 });
    const j = await oa.chat.completions.create({ model: "x", messages: [{ role: "user", content: JQ }], temperature: 0, max_tokens: 300,
      response_format: { type: "json_schema", json_schema: { name: "city", strict: true, schema: SCHEMA } } });
    check("chat tools: json_schema (nested, enum, array) validates", validates(SCHEMA, parsesTo(j.choices[0].message.content)), j.choices[0].message.content);
  });

  // Responses (/v1/responses): docs/design/serve.md, Responses section
  await soft("responses tools", async () => {
    const { OpenAI } = await sdk();
    const oa = new OpenAI({ baseURL: BASE + "/v1", apiKey: "anything", maxRetries: 0, timeout: 900000 });
    const W = { type: "function", name: "get_weather", description: "Current weather for a city", strict: false, parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } };
    const T = { type: "function", name: "get_time", description: "Local time in a city", strict: false, parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } };
    const Qw = "What is the weather in Paris? Use the tool.";
    const r = await oa.responses.create({ model: "x", input: Qw, tools: [W], temperature: 0, max_output_tokens: 300 });
    const c = r.output.find((o) => o.type === "function_call");
    check("responses tools: a get_weather function_call for Paris", r.status === "completed" && c?.name === "get_weather" && /paris/i.test(parsesTo(c.arguments)?.city || "") && /^fc_/.test(c.id) && /^call_/.test(c.call_id), JSON.stringify(r.output));
    const evs = [];
    const stream = oa.responses.stream({ model: "x", input: Qw, tools: [W], temperature: 0, max_output_tokens: 300 });
    stream.on("event", (e) => evs.push(e));
    const sr = await stream.finalResponse();
    const sc = sr.output.find((o) => o.type === "function_call");
    const deltas = evs.filter((e) => e.type === "response.function_call_arguments.delta" && e.item_id === sc?.id).map((e) => e.delta).join("");
    check("responses tools: streamed through the SDK, the same call; deltas join into the arguments",
      sc && sc.arguments === c?.arguments && deltas === sc.arguments && evs.every((e, i) => e.sequence_number === i) && evs.at(-1)?.type === "response.completed", JSON.stringify({ call: sc, deltas, types: evs.map((e) => e.type).slice(0, 30) }));
    if (!c) throw new Error("no call to answer");
    const f = await oa.responses.create({ model: "x", previous_response_id: r.id, tools: [W], temperature: 0, max_output_tokens: 200,
      input: [{ type: "function_call_output", call_id: c.call_id, output: "18 C and sunny" }] });
    check("responses tools: previous_response_id + function_call_output, the answer uses it", f.status === "completed" && /18|sunny/i.test(f.output_text) && !f.output.some((o) => o.type === "function_call"), JSON.stringify(f.output));
    check("responses tools: previous_response_id reuses the room's caches", (f.usage.input_tokens_details?.cached_tokens || 0) > 0, JSON.stringify({ first: r.usage, next: f.usage }));
    const req = await oa.responses.create({ model: "x", input: "Hi! How are you?", tools: [W, T], tool_choice: "required", temperature: 0, max_output_tokens: 300 });
    check("responses tools: tool_choice required gives a call for small talk", req.output.some((o) => o.type === "function_call") && !req.output.some((o) => o.type === "message"), JSON.stringify(req.output));
    const named = await oa.responses.create({ model: "x", input: "Hi there", tools: [W, T], tool_choice: { type: "function", name: "get_time" }, temperature: 0, max_output_tokens: 300 });
    const nc = named.output.filter((o) => o.type === "function_call");
    check("responses tools: a named tool_choice calls that tool", nc.length >= 1 && nc.every((o) => o.name === "get_time"), JSON.stringify(named.output));
    const js = await oa.responses.create({ model: "x", input: JQ, temperature: 0, max_output_tokens: 300, text: { format: { type: "json_schema", name: "city", strict: true, schema: SCHEMA } } });
    check("responses tools: text.format json_schema validates", validates(SCHEMA, parsesTo(js.output_text)), js.output_text);
    // a custom (free-form) tool, as Codex declares apply_patch with a GPT-5 model name: one raw string in,
    // a custom_tool_call item out
    const SH = { type: "custom", name: "run_shell", description: "Run one shell command line (the raw command, nothing else)" };
    const cu = await oa.responses.create({ model: "x", input: "List the files in the current directory.", tools: [SH], tool_choice: { type: "custom", name: "run_shell" }, temperature: 0, max_output_tokens: 200 });
    const cc = cu.output.find((o) => o.type === "custom_tool_call");
    check("responses tools: a custom tool gives a custom_tool_call with a raw string input", cu.status === "completed" && cc?.name === "run_shell" && typeof cc.input === "string" && cc.input.trim().length > 0 && /^ctc_/.test(cc.id), JSON.stringify(cu.output));
  });

  // Anthropic Messages (/v1/messages)
  await soft("messages tools", async () => {
    const { Anthropic } = await sdk();
    const an = new Anthropic({ baseURL: BASE, apiKey: "anything", maxRetries: 0, timeout: 900000 });
    const W = { name: "get_weather", description: "Current weather for a city", input_schema: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } };
    const T = { name: "get_time", description: "Local time in a city", input_schema: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } };
    const base = { model: "claude-x", max_tokens: 300, temperature: 0, tools: [W, T] };
    const uses = (m) => m.content.filter((b) => b.type === "tool_use");
    const textOf = (m) => m.content.filter((b) => b.type === "text").map((b) => b.text).join("");
    const q = [{ role: "user", content: "What is the weather in Paris? Use the tool." }];
    const m = await an.messages.create({ ...base, messages: q });
    const u = uses(m);
    check("messages tools: a get_weather tool_use for Paris, stop_reason tool_use", m.stop_reason === "tool_use" && u[0]?.name === "get_weather" && /paris/i.test(u[0].input.city || "") && /^toolu_/.test(u[0].id), JSON.stringify(m));
    const deltas = [];
    const st = an.messages.stream({ ...base, messages: q });
    st.on("streamEvent", (e) => { if (e.type === "content_block_delta" && e.delta.type === "input_json_delta") deltas.push(e.delta.partial_json); });
    const sm = await st.finalMessage();
    check("messages tools: streamed through the SDK (input_json_delta), the same call",
      sm.stop_reason === "tool_use" && JSON.stringify(uses(sm).map((b) => [b.name, b.input])) === JSON.stringify(u.map((b) => [b.name, b.input])) && deltas.length > 0, JSON.stringify({ sm, deltas }));
    if (!u.length) throw new Error("no call to answer");
    const f = await an.messages.create({ ...base, messages: [...q, { role: "assistant", content: m.content }, { role: "user", content: u.map((b) => ({ type: "tool_result", tool_use_id: b.id, content: "18 C and sunny" })) }] });
    check("messages tools: the tool_result is used, end_turn", f.stop_reason === "end_turn" && /18|sunny/i.test(textOf(f)) && !uses(f).length, JSON.stringify(f));
    check("messages tools: the follow-up reuses the room's caches", (f.usage.cache_read_input_tokens || 0) > 0, JSON.stringify({ first: m.usage, next: f.usage }));
    const any = await an.messages.create({ ...base, tool_choice: { type: "any" }, messages: [{ role: "user", content: "Hi! How are you?" }] });
    check("messages tools: tool_choice any gives a call for small talk", any.stop_reason === "tool_use" && uses(any).length >= 1, JSON.stringify(any));
    const named = await an.messages.create({ ...base, tool_choice: { type: "tool", name: "get_time" }, messages: [{ role: "user", content: "Hi there" }] });
    check("messages tools: tool_choice tool calls that tool", uses(named).length >= 1 && uses(named).every((b) => b.name === "get_time"), JSON.stringify(named));
    const js = await an.messages.create({ model: "claude-x", max_tokens: 300, temperature: 0, output_format: { type: "json_schema", schema: SCHEMA }, messages: [{ role: "user", content: JQ }] });
    check("messages tools: output_format json_schema validates", validates(SCHEMA, parsesTo(textOf(js))), textOf(js));
  });

  // one real agent turn each (opt-in: they need the agent CLIs and a long context)
  if (flag("codex")) await soft("codex", async () => {
    const { runCodex, summarize } = await import("./agents/codex.mjs");
    const t = Date.now();
    const run = await runCodex({ base: BASE, prompt: "Run `cat a.txt` and tell me what it says.", files: { "a.txt": "hello from a.txt\n" }, codex: arg("codex-bin", "codex"), contextWindow: (await (await fetch(BASE + "/health")).json()).ctx || CTX || 32768, timeoutMs: 1800000 });
    const s = summarize(run);
    log("codex:", JSON.stringify({ code: run.code, commands: s.commands, answer: s.answer, failed: s.failed, s: (Date.now() - t) / 1000 }));
    check("Codex CLI: ran cat a.txt through exec_command", s.commands.some((c) => /a\.txt/.test(c.command) && /hello from a\.txt/.test(c.output || "")), JSON.stringify({ code: run.code, s, err: run.stderr.slice(-300) }));
    check("Codex CLI: exits 0 and its answer quotes the file", run.code === 0 && /hello from a\.txt/i.test(s.answer || ""), JSON.stringify({ code: run.code, answer: s.answer, failed: s.failed }));
  });
  if (flag("claude")) await soft("claude code", async () => {
    const work = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-cc-work-")), cfg = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-cc-cfg-"));
    const word = "lighthouse-" + Math.random().toString(36).slice(2, 7);
    fs.writeFileSync(path.join(work, "notes.txt"), `The secret word is ${word}.\n`);
    const roomCtx = (await (await fetch(BASE + "/health")).json()).ctx || CTX || 32768;
    const t = Date.now();
    const out = await new Promise((resolve) => {
      const p = spawn(arg("claude-bin", "claude"), ["-p", "Read notes.txt and tell me the secret word in it.", "--model", "pooled", "--allowedTools", "Read", "--output-format", "json"], { cwd: work,
        env: { ...Object.fromEntries(Object.entries(process.env).filter(([k]) => !/^(CLAUDE|ANTHROPIC_)/.test(k))), CLAUDE_CONFIG_DIR: cfg, ANTHROPIC_BASE_URL: BASE, ANTHROPIC_API_KEY: "pooled", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: "1",
          // the room's real window (Claude Code does not know a model called pooled; the banner prints this)
          CLAUDE_CODE_MAX_CONTEXT_TOKENS: String(roomCtx), CLAUDE_CODE_MAX_OUTPUT_TOKENS: String(Math.max(1024, Math.min(32000, Math.floor(roomCtx / 4)))) } });
      let s = ""; p.stdout.on("data", (c) => (s += c)); p.stderr.on("data", (c) => (s += c));
      const kill = setTimeout(() => p.kill(), 1800000);
      p.on("close", () => { clearTimeout(kill); resolve(s); });
    });
    const j = parsesTo(out.split("\n").find((l) => l.startsWith("{\"")) || "");
    log("claude code:", JSON.stringify({ result: j?.result, turns: j?.num_turns, usage: j?.usage && { in: j.usage.input_tokens, cached: j.usage.cache_read_input_tokens, out: j.usage.output_tokens }, s: (Date.now() - t) / 1000 }));
    check("Claude Code: reads the file with its Read tool and answers with the word", j && !j.is_error && j.result?.includes(word) && j.num_turns >= 2, JSON.stringify({ out: out.slice(-600), s: (Date.now() - t) / 1000 }));
    fs.rmSync(work, { recursive: true, force: true }); fs.rmSync(cfg, { recursive: true, force: true });
  });

  await soft("client gone", async () => {
    const ac = new AbortController();
    const r = await fetch(BASE + "/v1/chat/completions", { method: "POST", headers: J, signal: ac.signal, body: JSON.stringify({ model: "x", max_tokens: 400, stream: true, messages: [{ role: "user", content: "Write a long story about a lighthouse." }] }) });
    // read until the answer is streaming, then go away
    const dec = new TextDecoder(); let seen = "";
    for await (const chunk of r.body) { seen += dec.decode(chunk, { stream: true }); if ((seen.match(/"content":"[^"]/g) || []).length >= 3) break; }
    ac.abort();
    const t = Date.now();
    const q = await (await post("/v1/chat/completions", { model: "x", max_tokens: 5, temperature: 0, messages: [{ role: "user", content: "Say OK." }] })).json();
    check("a client that goes away mid-stream frees the room within a lap", q.choices?.[0]?.message && Date.now() - t < 5000, `${Date.now() - t} ms ${JSON.stringify(q).slice(0, 120)}`);
    check("host: the stopped answer is marked stopped", await page.evaluate(() => /stopped/.test([...document.querySelectorAll(".m.bot")].map((m) => m.textContent).join(" "))));
  });

  await soft("unicode", async () => {
    const LINE = "🧑‍💻 👍🏽 🦀 🫠 नमस्ते दुनिया ∃y 𝔘𝔫𝔦";
    const body = { model: "x", temperature: 0, max_tokens: 80, messages: [{ role: "user", content: `Repeat exactly, nothing else: ${LINE}` }] };
    const j = await (await post("/v1/chat/completions", body)).json();
    const text = j.choices?.[0]?.message?.content || "";
    const s = await readSSE(await post("/v1/chat/completions", { ...body, stream: true }));
    const streamed = s.events.map((e) => e.data?.choices?.[0]?.delta?.content || "").join("");
    check("characters split across tokens arrive whole (no U+FFFD), non-stream and stream", text.length > 5 && !text.includes("\uFFFD") && !streamed.includes("\uFFFD") && streamed === text, JSON.stringify({ text, streamed }));
  });

  await soft("anthropic usage", async () => {
    const msgs = [{ role: "user", content: Q }, { role: "assistant", content: turn1 || "Paris." }, { role: "user", content: "And of Spain?" }];
    const o = await (await post("/v1/chat/completions", { model: "x", temperature: 0, max_tokens: 20, messages: msgs })).json();
    const a = await (await post("/v1/messages", { model: "x", temperature: 0, max_tokens: 20, messages: msgs }, { "anthropic-version": "2023-06-01" })).json();
    const u = a.usage || {};
    // a cache read only happens when the room still holds this prefix; either way input + cache reads must equal the prompt
    check("Anthropic input_tokens leaves out cache reads (input + cache reads = OpenAI prompt_tokens)", u.input_tokens + (u.cache_read_input_tokens || 0) === o.usage?.prompt_tokens, JSON.stringify({ openai: o.usage, anthropic: u }));
  });

  await soft("thinking budget", async () => {
    const m = await (await post("/v1/messages", { model: "x", max_tokens: 300, temperature: 0, thinking: { type: "enabled", budget_tokens: 48 },
      messages: [{ role: "user", content: "Think it through, then answer in one sentence: why is the sky blue?" }] }, { "anthropic-version": "2023-06-01" })).json();
    const think = m.content?.find((b) => b.type === "thinking")?.thinking || "", text = m.content?.find((b) => b.type === "text")?.text || "";
    check("thinking.budget_tokens: the reasoning stops at the budget and the answer gets the rest", think.length > 0 && text.trim().length > 0 && m.usage?.output_tokens <= 300, JSON.stringify(m).slice(0, 400));
  });

  await soft("host stop", async () => {
    const r = await post("/v1/chat/completions", { model: "x", max_tokens: 400, stream: true, messages: [{ role: "user", content: "Write a long story about a lighthouse." }] });
    let clicked = false;
    const s = await readSSE(r, async (ev) => {
      if (!clicked && ev.data?.choices?.[0]?.delta?.content) { clicked = true; await page.waitForSelector("#ai-send.stop", { timeout: 5000 }); await page.click("#ai-send"); }
    });
    const last = s.events[s.events.length - 1]?.data;
    check("host Stop: the client gets an error, not a finished answer", !!last?.error && /host stopped/.test(last.error.message) && !s.events.some((e) => e.data?.choices?.[0]?.finish_reason), JSON.stringify(last).slice(0, 200));
  });

  await soft("killed bridge", async () => {
    const P3 = await freePort();
    const b3 = startBridge(code, P3, ["--name", "kill-bridge"]);
    await waitHealth(`http://127.0.0.1:${P3}`, (h) => h.ready, 30000);
    const r = await fetch(`http://127.0.0.1:${P3}/v1/chat/completions`, { method: "POST", headers: J, body: JSON.stringify({ model: "x", max_tokens: 600, stream: true, messages: [{ role: "user", content: "Write a long story about a lighthouse." }] }) });
    const dec = new TextDecoder(); let seen = "";
    for await (const chunk of r.body) { seen += dec.decode(chunk, { stream: true }); if ((seen.match(/"content":"[^"]/g) || []).length >= 3) break; }
    b3.kill("SIGKILL");
    const t = Date.now();
    const q = await (await post("/v1/chat/completions", { model: "x", max_tokens: 5, temperature: 0, messages: [{ role: "user", content: "Say OK." }] })).json();
    const ms = Date.now() - t;
    check("a bridge killed mid-answer frees the room in under 30 s (was 73 s)", q.choices?.[0]?.message && ms < 30000, `${ms} ms ${JSON.stringify(q).slice(0, 120)}`);
    const gone = await page.waitForFunction(() => ![...document.querySelectorAll(".peer-card")].some((c) => /kill-bridge/.test(c.textContent)), null, { timeout: 30000 }).then(() => true, () => false);
    check("host: the killed bridge's card goes", gone);
    log("killed bridge: next answer after", ms, "ms");
  });

  await soft("chat", async () => {
    const shown = await page.evaluate(() => [...document.querySelectorAll(".m")].map((m) => m.textContent).join("\n"));
    check("host chat: API exchanges show with the client and the note", /\(API\)/.test(shown) && /via API · not part of this chat's memory/.test(shown), shown.slice(-400));
  });

  await soft("disconnect", async () => {
    await page.evaluate(() => { const c = [...document.querySelectorAll(".peer-card")].find((c) => /API client/.test(c.textContent)); c.querySelector(".api-kick").click(); });
    const h = await waitHealth(BASE, (x) => !!x.closed, 10000);
    check("Disconnect: the bridge learns why", /disconnected this API client/.test(h.closed), JSON.stringify(h));
    const r = await post("/v1/chat/completions", { model: "x", messages: [{ role: "user", content: "hi" }] });
    check("Disconnect: requests now get 503", r.status === 503, r.status);
    await new Promise((r) => setTimeout(r, 4000));
    const h2 = await (await fetch(BASE + "/health")).json();
    check("Disconnect: the bridge does not reconnect", !h2.connected && await page.evaluate(() => ![...document.querySelectorAll(".peer-card")].some((c) => /API client/.test(c.textContent))), JSON.stringify(h2));
  });
  check("no page errors on the host", pageErrs.length === 0, pageErrs.join(" | "));
} catch (e) {
  check("run", false, e.stack || e);
  try { log("host status:", await page.textContent("#ai-status")); } catch {}
} finally {
  const failed = checks.filter((c) => !c.ok);
  console.log(JSON.stringify({ ok: failed.length === 0, model: MODEL, code, passed: checks.length - failed.length, failed, checks: checks.map((c) => (c.ok ? "ok " : "FAIL ") + c.name) }, null, 1));
  if (failed.length) for (const b of bridges) console.error("--- bridge log ---\n" + b.out.slice(-3000));
  for (const b of bridges) { try { b.kill("SIGINT"); } catch {} }
  if (!flag("keep")) { await ctx.close().catch(() => {}); fs.rmSync(prof, { recursive: true, force: true }); }
  srv.close(); wsrv.close(); peerServer.kill(); fs.rmSync(tlsDir, { recursive: true, force: true });
  process.exit(failed.length ? 1 : 0);
}
