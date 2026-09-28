// Room emulator: real host + worker (+ optional phone-shaped tab) in headless Chromium on one
// machine, real PeerJS signaling and WebRTC, the machine's own GPU, a small model split across
// the tabs. Manual trigger only; nothing runs this automatically.
//
//   npm run e2e -- --phone                 host + worker + phone, wire on (default stripe4)
//   npm run e2e -- --wire off              old PeerJS message path
//   npm run e2e -- --model qwen3-1.7b --prompt "..." --rounds 3
//   npm run e2e -- --phone --model qwen3.8-27b   27B from models/q38/model.gguf, host pledge sized to the model's need
//   npm run e2e -- --devices 16 --phones 8 --model qwen3.8-27b   16 tabs, half phone-shaped, 1 GB / 0.5 GB pledges
//   Signaling runs on a local PeerServer (node_modules/.bin/peerjs) unless --signal cloud.
//
// The phone tab gets a mobile user agent (the room then treats it as a phone: 0.5 GB pledge,
// 256 MB buffer cap, no weight cache) and a 4x CPU throttle. Needs `npm install` (playwright)
// and a Chromium that exposes WebGPU on this machine; on Linux/NVIDIA the flags below do.
import { chromium } from "playwright";
import http from "http";
import fs from "fs";
import path from "path";
import { NEED_GB } from "../../room/models.js";

const arg = (k, d) => { const i = process.argv.indexOf("--" + k); return i > 0 ? process.argv[i + 1] : d; };
const flag = (k) => process.argv.includes("--" + k);
const WIRE = arg("wire", "stripe4"), MODEL = arg("model", "qwen3-0.6b"), ROUNDS = +arg("rounds", 2);
const PROMPT = arg("prompt", "Write three sentences about the ocean.");
const PORT = +arg("port", 8123);
// --devices N: total tabs including the host; --phones K: how many of the joiners are phone-shaped
const DEVICES = Math.max(2, +arg("devices", flag("phone") ? 3 : 2));
const PHONES = Math.min(DEVICES - 1, +arg("phones", flag("phone") ? 1 : 0));
const PHONE = PHONES > 0;
// signaling: our own PeerServer on this machine (default), or the public PeerJS cloud with --signal cloud
const SIGNAL_PORT = +arg("signal-port", 9000);
const CLOUD = arg("signal", "local") === "cloud";
// pledges in GB: host,worker,phone. The model's need comes from room/models.js (the 27B needs 17 GB).
// host pledge: whatever the joiners (1 GB desktop, 0.5 GB phone) leave of the model's need, at least 2 GB
const NEED = NEED_GB[MODEL] || 2;
const HOST_GB = arg("host-gb", String(Math.max(2, Math.ceil(NEED + 0.5 - (DEVICES - 1 - PHONES) * 1 - PHONES * 0.5))));
// GGUF files already on this machine stand in for Hugging Face (Range requests served from disk)
const LOCAL = { "Qwen3.8-27B-Q4_0.gguf": "models/q38/model.gguf", "Qwen3-0.6B-Q8_0.gguf": "models/qwen/model.gguf", "Qwen3-1.7B-Q8_0.gguf": "models/qwen17/model.gguf" };
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png" };

// static server for the repo root (the room is /p2p.html)
const srv = http.createServer((q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { r.statusCode = 404; r.end(); return; }
  r.setHeader("content-type", MIME[path.extname(p)] || "application/octet-stream"); fs.createReadStream(p).pipe(r);
}).listen(PORT, "127.0.0.1");
const BASE = `http://127.0.0.1:${PORT}/p2p.html?wire=${WIRE}` + (CLOUD ? "" : `&signal=127.0.0.1:${SIGNAL_PORT}`);
import { spawn } from "child_process";
let peerServer = null;
if (!CLOUD) {
  peerServer = spawn(path.join(ROOT, "node_modules/.bin/peerjs"), ["--port", String(SIGNAL_PORT), "--path", "/"], { stdio: "ignore" });
  await new Promise((r) => setTimeout(r, 1500));
}

const args = ["--no-sandbox", "--headless=new", "--enable-unsafe-webgpu", "--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--allow-loopback-in-peer-connection"];
const browser = await chromium.launch({ headless: false, args });
// Hugging Face answers the HeadlessChrome user agent with an HTML page: use a normal one
const UA_DESKTOP = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";
const UA_PHONE = "Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1";
// Local weight server: https (Playwright can only rewrite a request to the same protocol as the
// original, and Hugging Face is https), self-signed cert, Range requests streamed from disk.
import https from "https";
import { execSync } from "child_process";
import os from "os";
const TLS_PORT = PORT + 1;
const tlsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-e2e-"));
execSync(`openssl req -x509 -newkey rsa:2048 -nodes -keyout ${tlsDir}/k.pem -out ${tlsDir}/c.pem -days 2 -subj /CN=127.0.0.1 2>/dev/null`);
const wsrv = https.createServer({ key: fs.readFileSync(`${tlsDir}/k.pem`), cert: fs.readFileSync(`${tlsDir}/c.pem`) }, (q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p)) { r.statusCode = 404; r.end(); return; }
  const size = fs.statSync(p).size, m = /bytes=(\d+)-(\d*)/.exec(q.headers.range || "");
  const lo = m ? +m[1] : 0, hi = m && m[2] ? Math.min(+m[2], size - 1) : size - 1;
  r.writeHead(m ? 206 : 200, { "content-type": "application/octet-stream", "content-range": `bytes ${lo}-${hi}/${size}`, "accept-ranges": "bytes", "content-length": String(hi - lo + 1), "access-control-allow-origin": "*", "access-control-expose-headers": "content-range, content-length, accept-ranges" });
  fs.createReadStream(p, { start: lo, end: hi }).pipe(r);
}).listen(TLS_PORT, "127.0.0.1");
async function localWeights(context) {
  await context.route("**/*.gguf", (route) => {
    const name = route.request().url().split("/").pop().split("?")[0];
    const file = LOCAL[name];
    if (!file || !fs.existsSync(path.join(ROOT, file))) return route.continue();
    return route.continue({ url: `https://127.0.0.1:${TLS_PORT}/${file}` });
  });
}
const ctx = await browser.newContext({ userAgent: UA_DESKTOP, ignoreHTTPSErrors: true });
if (!flag("no-local-weights")) await localWeights(ctx);
const tabs = { host: await ctx.newPage() };
const nDesk = DEVICES - 1 - PHONES;
for (let i = 0; i < nDesk; i++) tabs[nDesk === 1 ? "worker" : "worker" + (i + 1)] = await ctx.newPage();
if (PHONE) {
  const pctx = await browser.newContext({ userAgent: UA_PHONE, viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, ignoreHTTPSErrors: true });
  if (!flag("no-local-weights")) await localWeights(pctx);
  for (let i = 0; i < PHONES; i++) {
    const pg = await pctx.newPage(); tabs[PHONES === 1 ? "phone" : "phone" + (i + 1)] = pg;
    const cdp = await pctx.newCDPSession(pg);
    await cdp.send("Emulation.setCPUThrottlingRate", { rate: 4 });
  }
}
const errs = Object.fromEntries(Object.keys(tabs).map((k) => [k, []]));
for (const [name, p] of Object.entries(tabs)) {
  p.on("console", (m) => { if (m.type() === "error") errs[name].push(m.text().slice(0, 200)); });
  p.on("pageerror", (e) => errs[name].push("pageerror: " + String(e).slice(0, 200)));
}
const t0 = Date.now(); const T = () => ((Date.now() - t0) / 1000).toFixed(1) + "s";
const log = (...a) => console.error(T(), ...a);
const status = (p) => p.evaluate(() => [document.getElementById("ai-status").textContent, document.getElementById("ldg-sub").textContent, document.getElementById("ldg-fill").style.width].join(" | "));

try {
  for (const p of Object.values(tabs)) await p.goto(BASE);
  for (const p of Object.values(tabs)) await p.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
  for (const [name, p] of Object.entries(tabs)) { await p.fill("#name-input", name + "-e2e"); await p.fill("#join-gb", name === "host" ? HOST_GB : name.startsWith("phone") ? "0.5" : "1"); }
  if (PHONE) log("phone tabs:", PHONES, "iPhone UA:", await Object.values(tabs).filter((_, i) => i === Object.keys(tabs).findIndex((k) => k.startsWith("phone")))[0].evaluate(() => navigator.userAgent.includes("iPhone")));
  await tabs.host.click("#create-btn");
  await tabs.host.waitForFunction(() => /[A-Z0-9]{4}/.test(document.getElementById("side-code").textContent), null, { timeout: 30000 });
  const code = (await tabs.host.textContent("#side-code")).trim().match(/[A-Z0-9]{4}/)[0];
  log("room", code);
  for (const name of Object.keys(tabs).filter((k) => k !== "host")) { await tabs[name].fill("#code-input", code); await tabs[name].click("#join-btn"); await tabs[name].waitForTimeout(150); }
  const N = Object.keys(tabs).length;
  for (const p of Object.values(tabs)) await p.waitForFunction((n) => document.querySelectorAll(".peer-card").length >= n, N, { timeout: 60000 + 2000 * N });
  log(N, "devices in room");
  await tabs.host.waitForTimeout(3000);   // stripe connections
  // from the load card to the chat, no tab may show the model picker in between (it did for the
  // seconds between the host's own layers and the room's: the "Pick a model" flash)
  for (const p of Object.values(tabs)) await p.evaluate(() => {
    const panel = document.getElementById("ai-panel"), w = window.__pickFlash = { armed: false, since: 0, shows: [] };
    const check = () => {
      const online = panel.classList.contains("online"), loading = panel.classList.contains("loading");
      if (loading) w.armed = true;
      const picker = w.armed && !online && !loading;
      if (picker && !w.since) w.since = performance.now();
      if (!picker && w.since) { w.shows.push(Math.round(performance.now() - w.since)); w.since = 0; }
    };
    new MutationObserver(check).observe(panel, { attributes: true, attributeFilter: ["class"] });
  });
  await tabs.host.selectOption("#ai-model", MODEL);
  await tabs.host.click("#ai-start");
  log("model start pressed");
  const poll = setInterval(async () => { try { for (const [n, p] of Object.entries(tabs).slice(0, 4)) log(n + ":", (await status(p)).slice(0, 120)); } catch {} }, 15000);
  // fail fast: a tab that reports a load failure means the room can never come online
  const failed = setInterval(async () => { try { for (const [n, p] of Object.entries(tabs)) { const st = await p.evaluate(() => document.getElementById("ai-status").textContent); if (/^failed:/.test(st)) { clearInterval(failed); throw new Error(`${n} ${st}`); } } } catch (e) { if (String(e).includes("failed:")) { console.error("FAILED:", String(e).slice(0, 300)); process.exit(2); } } }, 5000);
  for (const [name, p] of Object.entries(tabs)) await p.waitForFunction(() => document.getElementById("ai-panel").classList.contains("online"), null, { timeout: name === "host" ? 900000 : 300000 });
  clearInterval(failed);
  clearInterval(poll);
  log("online:", await tabs.host.textContent("#ai-status"));
  const split = await tabs.host.evaluate(() => [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent).filter((t) => /layer split/.test(t)).slice(-1)[0] || "");
  log(split);
  const pickFlash = {};
  for (const [n, p] of Object.entries(tabs)) { const ms = await p.evaluate(() => window.__pickFlash.shows); if (ms.length) pickFlash[n] = ms; }
  log("model picker shown between the load card and the chat (ms):", JSON.stringify(pickFlash));
  const results = [];
  for (let r = 0; r < ROUNDS; r++) {
    await tabs.host.fill("#ai-prompt", PROMPT); await tabs.host.click("#ai-send");
    await tabs.host.waitForFunction(() => /^ready — prefill|^generation failed/.test(document.getElementById("ai-status").textContent), null, { timeout: 300000 });
    const st = await tabs.host.textContent("#ai-status"); log("round", r, st);
    const reply = await tabs.host.evaluate(() => { const b = document.querySelectorAll(".m.bot .bubble"); return (b[b.length - 1]?.textContent || "").slice(0, 240); });
    results.push({ status: st, reply });
    await tabs.host.waitForTimeout(1000);
  }
  const wire = {}; for (const [n, p] of Object.entries(tabs)) wire[n] = await p.evaluate(() => window.pooledDebug?.());
  // the room's own log carries GPU validation errors that never reach the console
  const roomErrs = {}; for (const [n, p] of Object.entries(tabs)) roomErrs[n] = await p.evaluate(() => [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent).filter((t) => t.includes("\u26a0")).map((t) => t.slice(0, 160)));
  const nRoomErrs = Object.values(roomErrs).reduce((a, e) => a + e.length, 0);
  const ok = results.every((s) => s.status.startsWith("ready")) && Object.values(errs).every((e) => e.length === 0) && nRoomErrs === 0 && !Object.keys(pickFlash).length;
  const linkSummary = Object.fromEntries(Object.entries(wire).map(([n, l]) => [n, (l || []).map((x) => `${x.name}:${x.chans}ch ${x.sent}/${x.recv}`).join(", ")]));
  console.log(JSON.stringify({ ok, wire: WIRE, model: MODEL, devices: DEVICES, phones: PHONES, code, split, pickFlash, results, links: DEVICES > 6 ? Object.fromEntries(Object.entries(linkSummary).slice(0, 4)) : linkSummary, errors: Object.fromEntries(Object.entries(errs).filter(([, v]) => v.length)), roomErrors: { count: nRoomErrs, first: Object.fromEntries(Object.entries(roomErrs).filter(([, v]) => v.length).map(([k, v]) => [k, v.slice(0, 2)]).slice(0, 3)) } }, null, 1));
  process.exitCode = ok ? 0 : 1;
} catch (e) {
  console.error("FAILED:", String(e).slice(0, 400));
  for (const [n, p] of Object.entries(tabs)) { try { console.error(n + ":", await status(p)); } catch {} }
  console.error("errors:", JSON.stringify(errs));
  process.exitCode = 2;
} finally {
  await browser.close(); srv.close(); wsrv.close(); fs.rmSync(tlsDir, { recursive: true, force: true }); if (peerServer) peerServer.kill();
}
