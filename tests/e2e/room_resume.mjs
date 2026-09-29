// Room resume emulator (#207): a host, a desktop worker and a phone-shaped tab in headless Chromium,
// real PeerJS signaling and WebRTC, a small model split three ways. Each scenario interrupts an
// answer in flight and checks that it finishes with the same text as an uninterrupted one
// ("exact" sampling is greedy, so the answer does not depend on the room's shape).
//
//   node tests/e2e/room_resume.mjs                       all scenarios: lock, reload, kill
//   node tests/e2e/room_resume.mjs --scenarios lock,kill
//   node tests/e2e/room_resume.mjs --victim worker       which tab is interrupted (default phone)
//
//   lock    the tab "locks": it goes hidden, its signaling websocket closes (as iOS does) and the
//           page is frozen (no JS, no timers) for --away seconds, then comes back. The host notices
//           the silence, the answer waits; the tab reconnects by itself, keeps its layers, and the
//           answer carries on.
//   reload  the tab reloads mid-answer (what iOS does after killing a tab for memory): it rejoins
//           under the same name, reloads its layers, and the answer carries on.
//   kill    the tab closes for good: after the grace period (60 s) the host re-deals the layers
//           over the two devices left (experimental auto re-deal) and the answer carries on.
//
// With a real phone (tests/e2e/resume_phone.mjs drives it over WebDriver from the Mac it is attached to):
//   node tests/e2e/room_resume.mjs --url https://<preview>/room --external iphone --rounds 2 --code-out code.txt
//   the room is served from --url with the public signaling server; the host and the worker are tabs here,
//   the third device is the phone joining with the code written to --code-out. The phone script does the
//   interrupting; this side asks --rounds questions after the baseline and checks each answer finishes with
//   the baseline's text.
//
// Manual trigger only. Needs `npm install` (playwright, peer) and a WebGPU Chromium (see room.mjs).
import { chromium } from "playwright";
import http from "http";
import https from "https";
import fs from "fs";
import os from "os";
import path from "path";
import { spawn, execSync } from "child_process";

const arg = (k, d) => { const i = process.argv.indexOf("--" + k); return i > 0 ? process.argv[i + 1] : d; };
const MODEL = arg("model", "qwen3-0.6b");
const PROMPT = arg("prompt", "Tell a long story about a lighthouse keeper and the sea, with many details.");
const SCEN = arg("scenarios", "lock,reload,kill").split(",");
const VICTIM = arg("victim", "phone");
const AWAY = +arg("away", 20);
const AT_TOK = +arg("at", 24);          // interrupt once the answer has this many tokens
const MAXNEW = +arg("maxnew", 220);
const PORT = +arg("port", 8133), SIGNAL_PORT = +arg("signal-port", 9013);
const URL0 = arg("url", null), EXTERNAL = arg("external", null), ROUNDS = +arg("rounds", 2), CODE_OUT = arg("code-out", null);
const LOCAL = { "Qwen3-0.6B-Q8_0.gguf": "models/qwen/model.gguf", "Qwen3-1.7B-Q8_0.gguf": "models/qwen17/model.gguf" };
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png" };

const srv = http.createServer((q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { r.statusCode = 404; r.end(); return; }
  r.setHeader("content-type", MIME[path.extname(p)] || "application/octet-stream"); fs.createReadStream(p).pipe(r);
}).listen(PORT, "127.0.0.1");
const BASE = URL0 ? URL0 + (URL0.includes("?") ? "&" : "?") + "dev=1" : `http://127.0.0.1:${PORT}/p2p.html?signal=127.0.0.1:${SIGNAL_PORT}`;
const peerServer = URL0 ? null : spawn(path.join(ROOT, "node_modules/.bin/peerjs"), ["--port", String(SIGNAL_PORT), "--path", "/"], { stdio: "ignore" });
if (peerServer) await new Promise((r) => setTimeout(r, 1500));
const tlsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-resume-"));
execSync(`openssl req -x509 -newkey rsa:2048 -nodes -keyout ${tlsDir}/k.pem -out ${tlsDir}/c.pem -days 2 -subj /CN=127.0.0.1 2>/dev/null`);
const TLS_PORT = PORT + 1;
const wsrv = https.createServer({ key: fs.readFileSync(`${tlsDir}/k.pem`), cert: fs.readFileSync(`${tlsDir}/c.pem`) }, (q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p)) { r.statusCode = 404; r.end(); return; }
  const size = fs.statSync(p).size, m = /bytes=(\d+)-(\d*)/.exec(q.headers.range || "");
  const lo = m ? +m[1] : 0, hi = m && m[2] ? Math.min(+m[2], size - 1) : size - 1;
  r.writeHead(m ? 206 : 200, { "content-type": "application/octet-stream", "content-range": `bytes ${lo}-${hi}/${size}`, "accept-ranges": "bytes", "content-length": String(hi - lo + 1), "access-control-allow-origin": "*", "access-control-expose-headers": "content-range, content-length, accept-ranges" });
  fs.createReadStream(p, { start: lo, end: hi }).pipe(r);
}).listen(TLS_PORT, "127.0.0.1");
const localWeights = (context) => context.route("**/*.gguf", (route) => {
  const file = LOCAL[route.request().url().split("/").pop().split("?")[0]];
  if (!file || !fs.existsSync(path.join(ROOT, file))) return route.continue();
  return route.continue({ url: `https://127.0.0.1:${TLS_PORT}/${file}` });
});

const args = ["--no-sandbox", "--headless=new", "--enable-unsafe-webgpu", "--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--allow-loopback-in-peer-connection"];
const browser = await chromium.launch({ headless: false, args });
const UA_DESKTOP = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";
const UA_PHONE = "Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1";
// every tab records its websockets, so a "lock" can close the signaling link the way iOS does
const WS_SPY = () => { const W = window.WebSocket; window.__ws = []; window.WebSocket = class extends W { constructor(...a) { super(...a); window.__ws.push(this); } }; };
const dctx = await browser.newContext({ userAgent: UA_DESKTOP, ignoreHTTPSErrors: true });
const pctx = await browser.newContext({ userAgent: UA_PHONE, viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, ignoreHTTPSErrors: true });
for (const c of [dctx, pctx]) { await localWeights(c); await c.addInitScript(WS_SPY); }
const tabs = { host: await dctx.newPage(), worker: await dctx.newPage(), ...(EXTERNAL ? {} : { phone: await pctx.newPage() }) };
const errs = { host: [], worker: [], phone: [] };
const watch = (name, p) => {
  p.on("console", (m) => { if (m.type() === "error" && !/WebSocket|ERR_|Failed to load resource|peer/i.test(m.text())) errs[name].push(m.text().slice(0, 200)); });
  p.on("pageerror", (e) => errs[name].push("pageerror: " + String(e).slice(0, 200)));
};
for (const [n, p] of Object.entries(tabs)) watch(n, p);
const t0 = Date.now(); const T = () => ((Date.now() - t0) / 1000).toFixed(1) + "s";
const log = (...a) => console.error(T(), ...a);
const st = (p) => p.evaluate(() => document.getElementById("ai-status").textContent);
const roomLog = (p) => p.evaluate(() => [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent));
const lastReply = () => tabs.host.evaluate(() => { const b = document.querySelectorAll(".m.bot .bubble"); return b[b.length - 1]?.textContent || ""; });
const lastStats = () => tabs.host.evaluate(() => { const b = document.querySelectorAll(".m.bot"); return b[b.length - 1]?.querySelector(".stats, .meta, small")?.textContent || ""; });

async function ask() {
  await tabs.host.click("#new-chat").catch(() => {});
  await tabs.host.waitForTimeout(500);
  await tabs.host.fill("#ai-prompt", PROMPT);
  await tabs.host.click("#ai-send");
}
const doneRe = /^ready — prefill|^generation failed/;
async function waitTokens(n) {
  await tabs.host.waitForFunction((n) => { const m = /generating… (\d+) tok/.exec(document.getElementById("ai-status").textContent); return m && +m[1] >= n; }, n, { timeout: 120000, polling: 100 });
}
async function waitDone(ms) {
  await tabs.host.waitForFunction((re) => new RegExp(re).test(document.getElementById("ai-status").textContent), doneRe.source, { timeout: ms, polling: 250 });
  return st(tabs.host);
}
async function waitOnline(p, ms = 300000) { await p.waitForFunction(() => document.getElementById("ai-panel").classList.contains("online"), null, { timeout: ms }); }

// lock: hidden + signaling closed + frozen for `secs`, then visible again
async function lock(p, secs) {
  const cdp = await p.context().newCDPSession(p);
  await p.evaluate(() => {
    Object.defineProperty(document, "visibilityState", { configurable: true, get: () => "hidden" });
    Object.defineProperty(document, "hidden", { configurable: true, get: () => true });
    document.dispatchEvent(new Event("visibilitychange"));
    for (const w of window.__ws || []) { try { w.close(); } catch {} }
  });
  await p.waitForTimeout(300);
  let frozen = true;
  try { await cdp.send("Page.setWebLifecycleState", { state: "frozen" }); } catch (e) { frozen = false; log("could not freeze the page:", e.message); }
  log(`locked (${frozen ? "frozen" : "hidden only"}) for ${secs} s`);
  await new Promise((r) => setTimeout(r, secs * 1000));
  if (frozen) await cdp.send("Page.setWebLifecycleState", { state: "active" });
  await p.evaluate(() => {
    delete document.visibilityState; delete document.hidden;
    document.dispatchEvent(new Event("visibilitychange"));
  });
  log("unlocked");
  return frozen;
}

const out = { model: MODEL, victim: VICTIM, scenarios: {} };
try {
  for (const p of Object.values(tabs)) await p.goto(BASE + (p === tabs.host ? `&maxnew=${MAXNEW}` : ""));
  for (const p of Object.values(tabs)) await p.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
  for (const [n, p] of Object.entries(tabs)) { await p.fill("#name-input", n + "-rs"); await p.fill("#join-gb", n === "host" ? "2" : n === "phone" ? "0.5" : "1"); }
  await tabs.host.click("#create-btn");
  await tabs.host.waitForFunction(() => /[A-Z0-9]{4}/.test(document.getElementById("side-code").textContent), null, { timeout: 30000 });
  const code = (await tabs.host.textContent("#side-code")).trim().match(/[A-Z0-9]{4}/)[0];
  out.code = code; log("room", code);
  if (CODE_OUT) fs.writeFileSync(CODE_OUT, code);
  for (const n of ["worker", "phone"].filter((k) => tabs[k])) { await tabs[n].fill("#code-input", code); await tabs[n].click("#join-btn"); await tabs[n].waitForTimeout(200); }
  if (EXTERNAL) log(`waiting for ${EXTERNAL} to join room ${code}…`);
  for (const p of Object.values(tabs)) await p.waitForFunction(() => document.querySelectorAll(".peer-card").length >= 3, null, { timeout: EXTERNAL ? 900000 : 60000 });
  await tabs.host.waitForTimeout(2500);
  await tabs.host.evaluate(() => { const s = document.getElementById("ai-sampling"); s.value = "exact"; s.dispatchEvent(new Event("change")); });
  await tabs.host.evaluate((m) => { const s = document.getElementById("ai-model"); s.value = m; s.dispatchEvent(new Event("change")); }, MODEL);
  await tabs.host.click("#ai-start");
  for (const p of Object.values(tabs)) await waitOnline(p, 600000);
  log("online:", await st(tabs.host));
  out.split = (await roomLog(tabs.host)).filter((t) => /layer split/.test(t)).slice(-1)[0] || "";
  log(out.split);
  out.awake = await Promise.all(Object.values(tabs).map((p) => p.evaluate(() => { const e = document.getElementById("awake-ind"); return e ? { shown: !e.hidden, cls: e.className } : null; })));

  // the reference: an answer nobody interrupts
  await ask();
  const baseSt = await waitDone(300000);
  const base = await lastReply();
  out.baseline = { status: baseSt.slice(0, 160), chars: base.length };
  log("baseline:", baseSt.slice(0, 120), base.length, "chars");

  // a real phone does its own interrupting: ask the rounds, check every answer
  if (EXTERNAL) {
    for (let i = 0; i < ROUNDS; i++) {
      const r = out.scenarios["round" + (i + 1)] = { ok: false };
      try {
        await tabs.host.waitForTimeout(8000);   // the phone script is ready for the next answer
        await waitOnline(tabs.host, 400000);
        await tabs.host.waitForFunction(() => !document.getElementById("ai-send").disabled && document.getElementById("ai-row").style.display !== "none", null, { timeout: 400000 });
        const t1 = Date.now();
        await ask();
        const done = await waitDone(600000);
        const reply = await lastReply();
        r.secs = Math.round((Date.now() - t1) / 1000);
        r.status = done.slice(0, 200); r.stats = (await lastStats()).slice(0, 200);
        r.sameText = reply === base; r.chars = reply.length;
        if (!r.sameText) { let k = 0; while (k < reply.length && reply[k] === base[k]) k++; r.diffAt = k; r.got = reply.slice(Math.max(0, k - 40), k + 60); r.want = base.slice(Math.max(0, k - 40), k + 60); }
        r.hostLog = (await roomLog(tabs.host)).slice(-10).map((t) => t.slice(0, 180));
        r.ok = done.startsWith("ready") && r.sameText;
        log("round", i + 1, r.ok ? "OK" : "FAILED", r.secs, "s:", r.status.slice(0, 140), "|", r.stats.slice(0, 100));
      } catch (e) { r.error = String(e).slice(0, 300); r.hostStatus = await st(tabs.host).catch(() => ""); r.hostLog = (await roomLog(tabs.host).catch(() => [])).slice(-10).map((t) => t.slice(0, 180)); log("round", i + 1, "ERROR", r.error); }
    }
  }
  for (const sc of EXTERNAL ? [] : SCEN) {
    const r = out.scenarios[sc] = { ok: false };
    try {
      await ask();
      await waitTokens(AT_TOK);
      const at = await st(tabs.host);
      r.interruptedAt = at.slice(0, 60);
      log(sc, "interrupting", VICTIM, "at", at.slice(0, 60));
      const t1 = Date.now();
      if (sc === "lock") {
        r.frozen = await lock(tabs[VICTIM], AWAY);
      } else if (sc === "reload") {
        await tabs[VICTIM].reload();
        log("reloaded", VICTIM);
      } else if (sc === "kill") {
        await tabs[VICTIM].close();
        log("closed", VICTIM, "for good");
      }
      const done = await waitDone(sc === "kill" ? 400000 : 300000);
      const reply = await lastReply();
      r.secs = Math.round((Date.now() - t1) / 1000);
      r.status = done.slice(0, 200);
      r.stats = (await lastStats()).slice(0, 200);
      r.sameText = reply === base;
      r.chars = reply.length;
      if (!r.sameText) { let i = 0; while (i < reply.length && reply[i] === base[i]) i++; r.diffAt = i; r.got = reply.slice(Math.max(0, i - 40), i + 60); r.want = base.slice(Math.max(0, i - 40), i + 60); }
      r.hostLog = (await roomLog(tabs.host)).slice(-8).map((t) => t.slice(0, 180));
      if (sc !== "kill") {
        await waitOnline(tabs[VICTIM], 120000).catch(() => {});
        r.victimStatus = (await st(tabs[VICTIM])).slice(0, 160);
        r.victimLog = (await roomLog(tabs[VICTIM])).slice(-5).map((t) => t.slice(0, 180));
      }
      r.ok = done.startsWith("ready") && r.sameText;
      log(sc, r.ok ? "OK" : "FAILED", r.secs, "s:", r.status.slice(0, 140));
    } catch (e) {
      r.error = String(e).slice(0, 300);
      r.hostStatus = await st(tabs.host).catch(() => "");
      r.hostLog = (await roomLog(tabs.host).catch(() => [])).slice(-10).map((t) => t.slice(0, 180));
      log(sc, "ERROR", r.error);
    }
    if (sc === "kill") break;   // the victim is gone: nothing after this
  }
  out.errors = Object.fromEntries(Object.entries(errs).filter(([, v]) => v.length).map(([k, v]) => [k, v.slice(0, 5)]));
  out.ok = Object.values(out.scenarios).length > 0 && Object.values(out.scenarios).every((s) => s.ok);
  console.log(JSON.stringify(out, null, 1));
  process.exitCode = out.ok ? 0 : 1;
} catch (e) {
  console.error("FAILED:", String(e).slice(0, 400));
  for (const [n, p] of Object.entries(tabs)) { try { console.error(n + ":", await st(p)); } catch {} }
  console.log(JSON.stringify({ ...out, fatal: String(e).slice(0, 400), errors: errs }, null, 1));
  process.exitCode = 2;
} finally {
  await browser.close(); srv.close(); wsrv.close(); fs.rmSync(tlsDir, { recursive: true, force: true }); peerServer?.kill();
}
