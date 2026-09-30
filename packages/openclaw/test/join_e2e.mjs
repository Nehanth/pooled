// "Join a room" end to end: a browser tab creates the room (the room page from the Pooled checkout,
// local signaling) and is the model host; the plugin's room joins it with this process's GPU and holds
// layers; then the plugin's StreamFn (what OpenClaw calls) asks through the room's host over WebRTC,
// once plain and once with a tool (v2 ask to a browser host). No OpenClaw gateway here: the StreamFn
// is driven directly, as OpenClaw's agent loop would.
//   OC_ROOT=<dir with node_modules/openclaw> node --import ./test/oc-resolve.mjs test/join_e2e.mjs   (GPU)
// env: MODELS (model dir), CHROME_BIN (a Chromium with WebGPU; default playwright's), WORK, OUT
import http from "node:http";
import https from "node:https";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn, execSync } from "node:child_process";
import { createPooledStream } from "../src/stream.js";
import { createAssistantMessageEventStream } from "openclaw/plugin-sdk/llm";
import { createEmptyTransportUsage, failTransportStream } from "openclaw/plugin-sdk/provider-transport-runtime";
import { current } from "../src/pool.js";

const N = process.env.WORK || fs.mkdtempSync(path.join(os.tmpdir(), "pooled-oc-"));   // logs, browser profiles
const ROOT = path.resolve(process.env.POOLED_ROOT || new URL("../../..", import.meta.url).pathname);   // serves the room page
const MODELS = path.resolve(process.env.MODELS || path.join(ROOT, "models"));
const NN = process.env.NN || path.resolve(new URL("../../room-node/node_modules", import.meta.url).pathname);
const CHROME = process.env.CHROME_BIN || undefined;
const PORT = 8341, SIG = PORT + 2;
const T0 = Date.now();
const log = (...a) => console.error(((Date.now() - T0) / 1000).toFixed(1) + "s", ...a);
const out = {};
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png" };
const srv = http.createServer((q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { r.statusCode = 404; r.end(); return; }
  r.setHeader("content-type", MIME[path.extname(p)] || "application/octet-stream"); fs.createReadStream(p).pipe(r);
}).listen(PORT, "127.0.0.1");
const tlsDir = fs.mkdtempSync(path.join(N, "ocp-tls-"));
execSync(`openssl req -x509 -newkey rsa:2048 -nodes -keyout ${tlsDir}/k.pem -out ${tlsDir}/c.pem -days 2 -subj /CN=127.0.0.1 2>/dev/null`);
const wsrv = https.createServer({ key: fs.readFileSync(`${tlsDir}/k.pem`), cert: fs.readFileSync(`${tlsDir}/c.pem`) }, (q, r) => {
  const p = path.join(MODELS, decodeURIComponent(q.url.split("?")[0]).replace(/^\/models\//, ""));
  if (!p.startsWith(MODELS) || !fs.existsSync(p)) { r.statusCode = 404; r.end(); return; }
  const size = fs.statSync(p).size, m = /bytes=(\d+)-(\d*)/.exec(q.headers.range || "");
  const lo = m ? +m[1] : 0, hi = m && m[2] ? Math.min(+m[2], size - 1) : size - 1;
  r.writeHead(m ? 206 : 200, { "content-type": "application/octet-stream", "content-range": `bytes ${lo}-${hi}/${size}`, "accept-ranges": "bytes", "content-length": String(hi - lo + 1), "access-control-allow-origin": "*", "access-control-expose-headers": "content-range, content-length, accept-ranges" });
  fs.createReadStream(p, { start: lo, end: hi }).pipe(r);
}).listen(PORT + 1, "127.0.0.1");
const peerServer = spawn(path.join(NN, ".bin/peerjs"), ["--port", String(SIG), "--path", "/", "--host", "127.0.0.1"], { stdio: "ignore" });
await new Promise((r) => setTimeout(r, 1500));
let ctx = null;
const finish = async (c) => {
  console.log(JSON.stringify(out));
  if (process.env.OUT) fs.writeFileSync(process.env.OUT, JSON.stringify(out, null, 1));
  try { const h = current(); const r = await h?.ready; await r?.close(); } catch {}
  try { await Promise.race([ctx?.close(), new Promise((r) => setTimeout(r, 10000))]); } catch {}
  peerServer.kill(); srv.close(); wsrv.close(); fs.rmSync(tlsDir, { recursive: true, force: true });
  setTimeout(() => process.exit(c), 300);
};
try {
  const { chromium } = await import(path.join(NN, "playwright-core/index.mjs"));
  ctx = await chromium.launchPersistentContext(fs.mkdtempSync(path.join(N, "ocp-prof-")), { headless: true, executablePath: CHROME, ignoreHTTPSErrors: true,
    userAgent: "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36",
    args: ["--no-sandbox", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--allow-loopback-in-peer-connection", "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
      "--disable-features=WebRtcHideLocalIpsWithMdns,LocalNetworkAccessChecks,BlockInsecurePrivateNetworkRequests", "--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan"] });
  await ctx.route("**/*.gguf", (route) => route.request().url().endsWith("Qwen3-1.7B-Q8_0.gguf") ? route.continue({ url: `https://127.0.0.1:${PORT + 1}/models/qwen17/model.gguf` }) : route.continue());
  await ctx.route("https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/*.json", (route) =>
    route.fulfill({ path: path.join(MODELS, "qwen17", route.request().url().split("/").pop()), contentType: "application/json", headers: { "access-control-allow-origin": "*" } }));
  await ctx.route("https://cdn.jsdelivr.net/npm/peerjs@1.5.4/dist/peerjs.min.js", (route) => route.fulfill({ path: path.join(NN, "peerjs/dist/peerjs.min.js"), contentType: "text/javascript" }));
  const page = ctx.pages()[0] || await ctx.newPage();
  page.on("crash", () => log("tab CRASHED"));
  await page.goto(`http://127.0.0.1:${PORT}/p2p.html?signal=127.0.0.1:${SIG}&peerweights=0&dev=1&ckpt=0&split=memory`);
  await page.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
  await page.waitForTimeout(1500);
  await page.fill("#name-input", "mac-tab"); await page.fill("#join-gb", "6");
  await page.click("#create-btn");
  await page.waitForFunction(() => /[A-Z0-9]{4}/.test(document.getElementById("side-code").textContent), null, { timeout: 30000 });
  const code = (await page.textContent("#side-code")).trim().match(/[A-Z0-9]{4}/)[0];
  out.code = code; log("tab room", code);
  // the plugin, as onboarding "Join a room" left it
  const cfg = { mode: "join", code, model: "qwen3-1.7b", pledgeGB: 3, signal: `127.0.0.1:${SIG}`, modelDir: MODELS, name: "spark-openclaw", waitSeconds: 5 };
  const stream = createPooledStream({ getPluginConfig: () => cfg, log: (m) => log("[pooled]", m), sdk: { createAssistantMessageEventStream, createEmptyTransportUsage, failTransportStream } });
  const model = { id: "room", provider: "pooled", api: "openai-completions", maxTokens: 512, reasoning: false };
  const run = async (context) => {
    const st = stream(model, context, { maxTokens: 200 });
    for await (const _ of st) {}
    const m = await st.result();
    return { stop: m.stopReason, err: m.errorMessage, text: m.content.filter((c) => c.type === "text").map((c) => c.text).join(""), calls: m.content.filter((c) => c.type === "toolCall").map((c) => ({ name: c.name, args: c.arguments })), usage: m.usage.input + "/" + m.usage.output };
  };
  // before the model is up: the plugin's clear error
  const early = await run({ systemPrompt: "", messages: [{ role: "user", content: "hi", timestamp: 1 }] });
  out.beforeStart = early; log("before start:", JSON.stringify(early));
  await page.waitForFunction(() => document.querySelectorAll(".peer-card").length >= 2, null, { timeout: 60000 });
  await page.waitForTimeout(1500);
  await page.selectOption("#ai-model", "qwen3-1.7b");
  await page.evaluate(() => { const s = document.getElementById("ai-sampling"); s.value = "exact"; s.dispatchEvent(new Event("change")); });
  await page.click("#ai-start");
  await page.waitForFunction(() => document.getElementById("ai-panel").classList.contains("online") || /^failed:/.test(document.getElementById("ai-status").textContent), null, { timeout: 600000, polling: 1000 });
  const r = await current().ready;
  out.nodeRange = r.node.ai.range; out.nodeRole = r.node.ai.role; out.nodeIsModelHost = r.node.hosting();
  log("online; node holds", JSON.stringify(r.node.ai.range), "model host:", r.node.hosting());
  out.plain = await run({ systemPrompt: "You are concise.", messages: [{ role: "user", content: "In one sentence: why is the sky blue?", timestamp: 1 }] });
  log("plain:", JSON.stringify(out.plain));
  const tools = [{ name: "read", description: "Read a file from the workspace.", parameters: { type: "object", properties: { path: { type: "string", description: "file path" } }, required: ["path"] } }];
  const c1 = { systemPrompt: "You are a helpful assistant with file tools.", messages: [{ role: "user", content: "Read notes.txt and tell me the secret word.", timestamp: 1 }], tools };
  out.tool1 = await run(c1); log("tool 1:", JSON.stringify(out.tool1));
  if (out.tool1.calls.length) {
    const c2 = { ...c1, messages: [...c1.messages, { role: "assistant", content: out.tool1.calls.map((c, i) => ({ type: "toolCall", id: "c" + i, name: c.name, arguments: c.args })), stopReason: "toolUse" },
      ...out.tool1.calls.map((c, i) => ({ role: "toolResult", toolCallId: "c" + i, toolName: c.name, content: [{ type: "text", text: "The secret word is PELICAN-42." }], isError: false, timestamp: 2 }))] };
    out.tool2 = await run(c2); log("tool 2:", JSON.stringify(out.tool2));
  }
  out.tabShows = await page.evaluate(() => [...document.querySelectorAll(".m.bot .bubble")].slice(-2).map((b) => b.textContent.slice(0, 300)));
  await finish(0);
} catch (e) { out.error = String(e?.stack || e).slice(0, 800); log("FAILED", out.error); await finish(1); }
