// Room hop profiler: one room of N devices on this machine (one headless Chromium each) (real PeerJS signaling,
// real WebRTC on loopback, the machine's one GPU), greedy answers in plain and speculative mode, and
// a timestamp trace of every activation frame through every tab, so each lap splits into
// host compute, pack, send, wire, deliver, queue, unpack, GPU, readback, pack, send ... head.
// Manual trigger only (research branch exp/base).
//
//   node tests/e2e/room_prof.mjs --model qwen3.6-35b-moe --devices 2 [--maxnew 48] [--modes plain,spec] [--out f.json] [--query gpusample=1]
//
// room.js and room/transport.js on disk are not changed: this harness serves them with trace marks
// added (patchRoom / patchTransport in room_trace.mjs; they fail loudly if the anchors move), plus the dev-only
// plain-decode switch room_latency uses (window.__nospec). An init script in every tab adds a GPU
// timestamp pair around every command buffer while tracing (empty timestamped passes at the start
// and end of each encoder, so the engine's passes are untouched) and times every mapAsync.
// Clock: marks are performance.timeOrigin + performance.now() in each browser, shifted onto the
// harness's clock by a measured offset per browser (clockOffset).
import { chromium } from "playwright";
import http from "http";
import https from "https";
import fs from "fs";
import os from "os";
import path from "path";
import { spawn, execSync } from "child_process";
import { patchRoom, patchTransport, INIT, clockOffset } from "./room_trace.mjs";

const arg = (k, d) => { const i = process.argv.indexOf("--" + k); return i > 0 ? process.argv[i + 1] : d; };
const MODEL = arg("model", "qwen3.6-35b-moe");
const DEVICES = Math.max(2, +arg("devices", 2));
const MODES = arg("modes", "plain,spec").split(",");
const MAXNEW = +arg("maxnew", 48);
const OUT = arg("out", "");
const WIRE = arg("wire", "stripe4");
const PORT = +arg("port", 8131), SIGNAL_PORT = +arg("signal-port", 9011), TLS_PORT = PORT + 1;
const PROMPT = arg("prompt", "Write the Python code for two sum. Code only.");
const NEED = { "qwen3.8-27b": 16.5, "qwen3.6-35b-moe": 22.5, "qwen3-1.7b": 2.0, "qwen3-0.6b": 0.8 }[MODEL] || 2;
const GB = (name) => String(Math.ceil(NEED / DEVICES + (name === "host" ? 2 : 0.5)));
const LOCAL = { "Qwen3.8-27B-Q4_0.gguf": "models/q38/model.gguf", "Qwen3-0.6B-Q8_0.gguf": "models/qwen/model.gguf", "Qwen3-1.7B-Q8_0.gguf": "models/qwen17/model.gguf", "Qwen_Qwen3.6-35B-A3B-Q4_0.gguf": "models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf" };
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png" };

const PATCHED = { [path.join(ROOT, "room.js")]: patchRoom, [path.join(ROOT, "room/transport.js")]: patchTransport };

const srv = http.createServer((q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { r.statusCode = 404; r.end(); return; }
  r.setHeader("content-type", MIME[path.extname(p)] || "application/octet-stream");
  if (PATCHED[p]) { r.end(PATCHED[p](fs.readFileSync(p, "utf8"))); return; }
  fs.createReadStream(p).pipe(r);
}).listen(PORT, "127.0.0.1");
// check the patches apply before starting a browser
for (const [p, fn] of Object.entries(PATCHED)) fn(fs.readFileSync(p, "utf8"));
if (process.argv.includes("--check")) { for (const [p, fn] of Object.entries(PATCHED)) fs.writeFileSync(path.join(os.tmpdir(), "prof_" + path.basename(p)), fn(fs.readFileSync(p, "utf8"))); console.log("patches apply"); process.exit(0); }
const tlsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-prof-"));
execSync(`openssl req -x509 -newkey rsa:2048 -nodes -keyout ${tlsDir}/k.pem -out ${tlsDir}/c.pem -days 2 -subj /CN=127.0.0.1 2>/dev/null`);
const wsrv = https.createServer({ key: fs.readFileSync(`${tlsDir}/k.pem`), cert: fs.readFileSync(`${tlsDir}/c.pem`) }, (q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p)) { r.statusCode = 404; r.end(); return; }
  const size = fs.statSync(p).size, m = /bytes=(\d+)-(\d*)/.exec(q.headers.range || "");
  const lo = m ? +m[1] : 0, hi = m && m[2] ? Math.min(+m[2], size - 1) : size - 1;
  r.writeHead(m ? 206 : 200, { "content-type": "application/octet-stream", "content-range": `bytes ${lo}-${hi}/${size}`, "accept-ranges": "bytes", "content-length": String(hi - lo + 1), "access-control-allow-origin": "*", "access-control-expose-headers": "content-range, content-length, accept-ranges" });
  fs.createReadStream(p, { start: lo, end: hi }).pipe(r);
}).listen(TLS_PORT, "127.0.0.1");
const peerServer = spawn(path.join(ROOT, "node_modules/.bin/peerjs"), ["--port", String(SIGNAL_PORT), "--path", "/"], { stdio: "ignore" });
await new Promise((r) => setTimeout(r, 1500));
// --query "a=1&b=2": extra room URL parameters on every tab (e.g. gpusample=1)
const BASE = `http://127.0.0.1:${PORT}/p2p.html?split=memory&signal=127.0.0.1:${SIGNAL_PORT}&maxnew=${MAXNEW}&peerweights=0&wire=${WIRE}` + (arg("query") ? "&" + arg("query") : "");

// One Chromium per device, each with its own on-disk profile (as tests/e2e/room_latency.mjs on
// bench/latency does: with every tab in one off-the-record context the Cache API weight store lives
// in RAM and the 35B MoE crashes the browser). Separate browser and GPU processes, like separate machines.
const ARGS = ["--no-sandbox", "--headless=new", "--enable-unsafe-webgpu", "--enable-webgpu-developer-features", "--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--allow-loopback-in-peer-connection", "--js-flags=--max-old-space-size=65536"];
const UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";
const profiles = [], contexts = [], tabs = {};
for (let i = 0; i < DEVICES; i++) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-prof-profile-")); profiles.push(dir);
  const c = await chromium.launchPersistentContext(dir, { headless: false, args: ARGS, userAgent: UA, ignoreHTTPSErrors: true });
  await c.addInitScript(INIT);
  await c.route("**/*.gguf", (route) => {
    const file = LOCAL[route.request().url().split("/").pop().split("?")[0]];
    if (!file || !fs.existsSync(path.join(ROOT, file))) return route.continue();
    return route.continue({ url: `https://127.0.0.1:${TLS_PORT}/${file}` });
  });
  contexts.push(c);
  tabs[i === 0 ? "host" : "worker" + i] = c.pages()[0] || await c.newPage();
}
const errs = [];
for (const [n, p] of Object.entries(tabs)) {
  p.on("pageerror", (e) => errs.push(n + ": " + String(e).slice(0, 200)));
  p.on("crash", () => { errs.push(n + ": crashed"); console.error(n, "CRASHED"); });
  p.on("console", (m) => { if (m.type() === "error") console.error(n, "console:", m.text().slice(0, 200)); });
}
const t0 = Date.now(); const log = (...a) => console.error(((Date.now() - t0) / 1000).toFixed(0) + "s", ...a);
const host = tabs.host;
const out = { model: MODEL, devices: DEVICES, wire: WIRE, maxnew: MAXNEW, prompt: PROMPT, split: "", runs: [] };
try {
  for (const p of Object.values(tabs)) await p.goto(BASE);
  for (const p of Object.values(tabs)) await p.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
  for (const [n, p] of Object.entries(tabs)) { await p.fill("#name-input", n); await p.fill("#join-gb", GB(n)); }
  await host.click("#create-btn");
  await host.waitForFunction(() => /[A-Z0-9]{4}/.test(document.getElementById("side-code").textContent), null, { timeout: 30000 });
  const code = (await host.textContent("#side-code")).trim().match(/[A-Z0-9]{4}/)[0];
  for (const n of Object.keys(tabs).filter((k) => k !== "host")) { await tabs[n].fill("#code-input", code); await tabs[n].click("#join-btn"); await tabs[n].waitForTimeout(150); }
  for (const p of Object.values(tabs)) await p.waitForFunction((n) => document.querySelectorAll(".peer-card").length >= n, DEVICES, { timeout: 120000 });
  await host.waitForTimeout(3000);
  await host.selectOption("#ai-model", MODEL);
  const tLoad = Date.now();
  await host.click("#ai-start");
  const poll = setInterval(async () => { for (const [n, p] of Object.entries(tabs)) { try { log(n, (await p.textContent("#ai-status")).slice(0, 100)); } catch {} } }, 30000);
  for (const [n, p] of Object.entries(tabs)) await p.waitForFunction(() => document.getElementById("ai-panel").classList.contains("online") || /^failed:/.test(document.getElementById("ai-status").textContent), null, { timeout: 1800000, polling: 2000 });
  clearInterval(poll);
  const st0 = await host.textContent("#ai-status");
  if (/^failed:/.test(st0)) throw new Error("load " + st0);
  out.loadS = Math.round((Date.now() - tLoad) / 1000);
  out.split = await host.evaluate(() => [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent).filter((t) => /layer split/.test(t)).slice(-1)[0] || "");
  log("online in", out.loadS, "s;", out.split);
  await host.evaluate(() => { const s = document.getElementById("ai-sampling"); s.value = "exact"; s.dispatchEvent(new Event("change")); });
  const ask = async () => {
    await host.evaluate(() => document.getElementById("new-chat").click());
    await host.waitForTimeout(500);
    await host.evaluate((text) => { const box = document.getElementById("ai-prompt"); box.value = text; box.dispatchEvent(new Event("input")); document.getElementById("ai-send").click(); }, PROMPT);
    await host.waitForFunction(() => /^ready — prefill|^generation failed/.test(document.getElementById("ai-status").textContent), null, { timeout: 600000, polling: 250 });
    return host.textContent("#ai-status");
  };
  for (const mode of MODES) {
    for (const p of Object.values(tabs)) await p.evaluate((ns) => { window.__nospec = ns; }, mode === "plain");
    log(mode, "warm-up:", (await ask()).slice(0, 120));
    for (const p of Object.values(tabs)) await p.evaluate(() => { window.__hp = []; window.__gp = { subs: [], maps: [] }; window.__hpOn = true; });
    const st = await ask();
    for (const p of Object.values(tabs)) await p.evaluate(() => { window.__hpOn = false; });
    for (const p of Object.values(tabs)) await p.evaluate(() => window.__gpResolve?.());
    const traces = {};
    for (const [n, p] of Object.entries(tabs)) {
      const c = await clockOffset(p);
      const tr = await p.evaluate(() => ({ hp: window.__hp, gp: window.__gp }));
      // shift every mark onto this process's clock
      for (const e of tr.hp) e[3] -= c.off;
      for (const x of tr.gp.subs) { x.t -= c.off; x.tEnc -= c.off; }
      for (const m of tr.gp.maps) { m.t0 -= c.off; if (m.t1) m.t1 -= c.off; }
      traces[n] = { ...tr, clock: c };
    }
    const reply = await host.evaluate(() => { const b = document.querySelectorAll(".m.bot .bubble"); return (b[b.length - 1]?.textContent || "").slice(0, 160); });
    const crumb = await host.evaluate(() => { try { return JSON.parse(localStorage.getItem("pooled-crumb") || "{}").s || ""; } catch { return ""; } });
    log(mode, "measured:", st.slice(0, 160));
    out.runs.push({ mode, status: st, reply, crumb, traces });
  }
} catch (e) {
  out.error = String(e).slice(0, 400);
  log("FAILED", out.error);
} finally {
  out.errors = errs;
  if (OUT) fs.writeFileSync(OUT, JSON.stringify(out));
  else console.log(JSON.stringify(out));
  peerServer.kill(); fs.rmSync(tlsDir, { recursive: true, force: true });
  await Promise.race([Promise.all(contexts.map((c) => c.close().catch(() => {}))), new Promise((r) => setTimeout(r, 15000))]);   // a crashed browser can hang close()
  for (const d of profiles) fs.rmSync(d, { recursive: true, force: true });
  srv.close(); wsrv.close();
  process.exit(out.error ? 1 : 0);
}
