// End to end on one machine: OpenClaw's gateway with this plugin hosts Pooled room <code> (its config,
// written by the onboarding), a headless browser tab opens the room page and joins as the second
// device, the room deals the layers over both, and `openclaw agent` runs real chat turns through it.
//   node test/e2e_oc.mjs   (GPU; OpenClaw on PATH with OPENCLAW_CONFIG_PATH / OPENCLAW_STATE_DIR set, the plugin
//                            installed with `openclaw plugins install --link packages/openclaw` and onboarded as host)
// env: POOLED_ROOT (the Pooled checkout serving the room page; default this one), MODELS, TAB_GB (4),
//      OUT (result JSON), TURNS ("chat,tool"), CHROME_BIN, NN (node_modules with playwright-core + peer), WORK
import http from "node:http";
import https from "node:https";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn, execSync, execFileSync } from "node:child_process";

const N = process.env.WORK || fs.mkdtempSync(path.join(os.tmpdir(), "pooled-oc-"));   // logs, browser profiles
const ROOT = path.resolve(process.env.POOLED_ROOT || new URL("../../..", import.meta.url).pathname);   // serves the room page
const MODELS = path.resolve(process.env.MODELS || path.join(ROOT, "models"));
const NN = process.env.NN || path.resolve(new URL("../../room-node/node_modules", import.meta.url).pathname);   // peer, playwright-core, peerjs
const CHROME = process.env.CHROME_BIN || undefined;
const CFG = JSON.parse(fs.readFileSync(process.env.OPENCLAW_CONFIG_PATH, "utf8"));
const PC = CFG.plugins.entries.pooled.config;
const SIG = +PC.signal.split(":")[1], PORT = SIG - 2;
const TAB_GB = +(process.env.TAB_GB || 4);
const TURNS = (process.env.TURNS || "chat,tool").split(",");
const STATE = path.join(process.env.OPENCLAW_STATE_DIR, "pooled", "status.json");
const T0 = Date.now();
const log = (...a) => console.error(((Date.now() - T0) / 1000).toFixed(1) + "s", ...a);
const out = { code: PC.code, model: PC.model, turns: [] };
// FAKERUN: an XDG_RUNTIME_DIR without the user's systemd bus, so a test gateway never touches the user's own
const ocEnv = { ...process.env, ...(process.env.FAKERUN ? { XDG_RUNTIME_DIR: process.env.FAKERUN } : {}), POOLED_DEBUG: "1" };
if (process.env.FAKERUN) delete ocEnv.DBUS_SESSION_BUS_ADDRESS;

// --- servers: the room page (http), weights (https + Range), PeerJS signaling
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png" };
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

// --- OpenClaw's gateway (the plugin's service opens the room)
try { fs.rmSync(STATE); } catch {}
const gwLog = fs.openSync(process.env.GW_LOG || `${N}/oc_gateway.log`, "w");
const gw = spawn("openclaw", ["gateway", "run", "--port", String(CFG.gateway.port), "--allow-unconfigured", "--verbose"], { env: ocEnv, stdio: ["ignore", gwLog, gwLog] });
gw.on("exit", (c, s) => log(`gateway exited ${c ?? s}`));
const readState = () => { try { return JSON.parse(fs.readFileSync(STATE, "utf8")); } catch { return null; } };
async function until(fn, ms, what) {
  const t = Date.now();
  for (;;) { const v = await fn(); if (v) return v; if (Date.now() - t > ms) throw new Error("timeout: " + what); await new Promise((r) => setTimeout(r, 500)); }
}

let ctx = null, page = null, code = 1;
const finish = async (c) => {
  code = c;
  out.state = readState();
  out.elapsedS = (Date.now() - T0) / 1000;
  console.log(JSON.stringify(out));
  if (process.env.OUT) fs.writeFileSync(process.env.OUT, JSON.stringify(out, null, 1));
  try { await Promise.race([ctx?.close(), new Promise((r) => setTimeout(r, 10000))]); } catch {}
  gw.kill("SIGTERM"); await new Promise((r) => setTimeout(r, 4000)); try { gw.kill("SIGKILL"); } catch {}
  peerServer.kill(); srv.close(); wsrv.close(); fs.rmSync(tlsDir, { recursive: true, force: true });
  setTimeout(() => process.exit(c), 300);
};

function agentTurn(message, session) {
  const t = Date.now();
  let stdout = "", stderr = "";
  try {
    stdout = execFileSync("openclaw", ["agent", "--agent", "main", "--session-id", session, "--message", message, "--json", "--timeout", "900"], { env: ocEnv, encoding: "utf8", maxBuffer: 64 << 20, timeout: 960000 });
  } catch (e) { stdout = e.stdout || ""; stderr = String(e.stderr || e.message).slice(-2000); }
  let json = null; try { json = JSON.parse(stdout.slice(stdout.indexOf("{"))); } catch {}
  return { message, seconds: (Date.now() - t) / 1000, json, raw: json ? undefined : stdout.slice(-3000), stderr: stderr || undefined };
}

try {
  const st0 = await until(() => { const s = readState(); return s?.code ? s : null; }, 120000, "the plugin opening the room");
  log(`room ${st0.code} open (${st0.link}); devices ${st0.devices?.length}`);
  out.roomOpenS = (Date.now() - T0) / 1000;
  // the second device: the real room page in a headless tab
  const { chromium } = await import(path.join(NN, "playwright-core/index.mjs"));
  const prof = fs.mkdtempSync(path.join(N, "ocp-prof-"));
  ctx = await chromium.launchPersistentContext(prof, { headless: true, executablePath: CHROME, ignoreHTTPSErrors: true,
    userAgent: "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36",
    args: ["--no-sandbox", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--allow-loopback-in-peer-connection",
      "--disable-background-timer-throttling", "--disable-renderer-backgrounding", "--disable-features=WebRtcHideLocalIpsWithMdns,LocalNetworkAccessChecks,BlockInsecurePrivateNetworkRequests",
      "--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan"] });
  await ctx.route("**/*.gguf", (route) => {
    const f = { "Qwen3-1.7B-Q8_0.gguf": "models/qwen17/model.gguf", "Qwen_Qwen3.6-35B-A3B-Q4_0.gguf": "models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf" }[route.request().url().split("/").pop()];
    return f ? route.continue({ url: `https://127.0.0.1:${PORT + 1}/${f}` }) : route.continue();
  });
  await ctx.route("https://huggingface.co/Qwen/Qwen3-1.7B/resolve/*/*.json", (route) =>
    route.fulfill({ path: path.join(MODELS, "qwen17", route.request().url().split("/").pop()), contentType: "application/json", headers: { "access-control-allow-origin": "*" } }));
  await ctx.route("https://cdn.jsdelivr.net/npm/peerjs@1.5.4/dist/peerjs.min.js", (route) => route.fulfill({ path: path.join(NN, "peerjs/dist/peerjs.min.js"), contentType: "text/javascript" }));
  page = ctx.pages()[0] || await ctx.newPage();
  page.on("crash", () => log("tab CRASHED"));
  page.on("pageerror", (e) => log("tab pageerror:", String(e).slice(0, 200)));
  await page.goto(`http://127.0.0.1:${PORT}/p2p.html?signal=127.0.0.1:${SIG}&peerweights=0&dev=1&ckpt=0&split=memory`);
  await page.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
  await page.waitForTimeout(1500);
  await page.fill("#name-input", "phone-tab"); await page.fill("#join-gb", String(TAB_GB));
  await page.fill("#code-input", st0.link); await page.click("#join-btn");   // the invite link: in without asking
  const tJoin = Date.now();
  // the plugin's service sees 2 devices with enough memory, deals the layers, loads its own share
  const st1 = await until(() => { const s = readState(); return s?.online ? s : null; }, 600000, "the room going online");
  out.onlineS = (Date.now() - tJoin) / 1000;
  out.split = st1.split; out.devices = st1.devices;
  log(`room online: ${st1.split?.join(" · ")}`);
  out.tabStatus = await page.evaluate(() => document.getElementById("ai-status")?.textContent);
  // turn 1: plain chat
  if (TURNS.includes("chat")) {
    const r = agentTurn("In one sentence: why is the sky blue?", "pooled-e2e-chat");
    out.turns.push(r); log("chat turn:", JSON.stringify(r.json?.result?.payloads || r.json?.payloads || r.raw || r.stderr).slice(0, 600));
  }
  // turn 2: a tool-using turn (OpenClaw's read tool on the workspace file)
  if (TURNS.includes("tool")) {
    const r = agentTurn("Use your read tool to open notes.txt in the workspace, then tell me the secret word it contains.", "pooled-e2e-tool");
    out.turns.push(r); log("tool turn:", JSON.stringify(r.json?.result?.payloads || r.json?.payloads || r.raw || r.stderr).slice(0, 600));
  }
  out.tabShows = await page.evaluate(() => [...document.querySelectorAll(".m.bot .bubble")].slice(-2).map((b) => b.textContent.slice(0, 400)));
  await finish(0);
} catch (e) {
  out.error = String(e?.stack || e).slice(0, 800); log("FAILED", out.error);
  try { if (page) out.tabStatus = await page.evaluate(() => document.getElementById("ai-status")?.textContent); } catch {}
  await finish(1);
}
