// Drop detection in a room (room/liveness.js): a device whose network dies silently mid-answer
// must be noticed within ~3-5 s (the answer fails fast, the host offers a re-deal), and a bad but
// working network (300 ms latency, 5% loss) must never be mistaken for a dead device.
//
//   node tests/e2e/room_drop.mjs --devices 3                  every scenario below, Qwen3 1.7B
//   node tests/e2e/room_drop.mjs --devices 3 --only noise300  one scenario
//   node tests/e2e/room_drop.mjs --devices 3 --query hb=0     the same without drop detection (A/B)
//
// Scenarios: baseline; noise300 / noise600 (RTT 300 / 600 ms with 5% loss both ways: the answer
// completes and matches the baseline, no device is dropped; reports the longest silence the host
// saw per device against its limit, which must stay 0.5 s under it); die-mid (the middle device of the chain goes silent
// mid-answer); die-last (the last device, which sends the hidden states back to the host); after
// each death the host re-deals over the devices left and answers again.
//
// Real WebRTC and PeerJS signaling on this machine, one Chromium per device, every WebRTC packet
// through a userland UDP relay here that adds latency, drops packets or cuts a device off
// (no root needed). The relay is the one in tests/e2e/room_chaos.mjs.
// Output: one JSON line on stdout; progress on stderr. Exit code 1 when a check fails.
import { chromium } from "playwright";
import http from "http";
import https from "https";
import net from "net";
import dgram from "dgram";
import fs from "fs";
import os from "os";
import path from "path";
import { spawn, execSync } from "child_process";

const arg = (k, d) => { const i = process.argv.indexOf("--" + k); return i > 0 ? process.argv[i + 1] : d; };
const flag = (k) => process.argv.includes("--" + k);
const MODEL = arg("model", "qwen3-1.7b"), DEVICES = Math.max(3, +arg("devices", 4)), QUERY = arg("query", "");
const MAXNEW = +arg("maxnew", 256), PORT = +arg("port", 8470), TLS_PORT = PORT + 1, SIG_PORT = +arg("signal-port", 9470);
const ONLY = arg("only", "");   // comma list of scenario names to run from the plan
const PROMPT = arg("prompt", "Explain in about a hundred words how a bicycle gear system works, then list three tips for riding up a steep hill.");
const NEED = { "qwen3-1.7b": 2.0, "qwen3-0.6b": 0.8 }[MODEL] || 2;
// the host lends the most, so it is the model host (room/plan.js pickModelHost), and the layers
// are dealt by memory so every device is in the chain (the speed split leaves spare devices out)
const GB = (i) => String(Math.ceil(NEED / DEVICES + 0.5) + (i === 0 ? 1 : 0));
const LOCAL = { "Qwen3-0.6B-Q8_0.gguf": "models/qwen/model.gguf", "Qwen3-1.7B-Q8_0.gguf": "models/qwen17/model.gguf" };
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png" };
const t0 = Date.now();
const log = (...a) => console.error(((Date.now() - t0) / 1000).toFixed(1) + "s", ...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------- static files ----------
const srv = http.createServer((q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { r.statusCode = 404; r.end(); return; }
  r.setHeader("content-type", MIME[path.extname(p)] || "application/octet-stream");
  fs.createReadStream(p).pipe(r);
}).listen(PORT, "127.0.0.1");
const tlsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-drop-"));
execSync(`openssl req -x509 -newkey rsa:2048 -nodes -keyout ${tlsDir}/k.pem -out ${tlsDir}/c.pem -days 2 -subj /CN=127.0.0.1 2>/dev/null`);
const wsrv = https.createServer({ key: fs.readFileSync(`${tlsDir}/k.pem`), cert: fs.readFileSync(`${tlsDir}/c.pem`) }, (q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p)) { r.statusCode = 404; r.end(); return; }
  const size = fs.statSync(p).size, m = /bytes=(\d+)-(\d*)/.exec(q.headers.range || "");
  const lo = m ? +m[1] : 0, hi = m && m[2] ? Math.min(+m[2], size - 1) : size - 1;
  r.writeHead(m ? 206 : 200, { "content-type": "application/octet-stream", "content-range": `bytes ${lo}-${hi}/${size}`, "accept-ranges": "bytes", "content-length": String(hi - lo + 1), "access-control-allow-origin": "*", "access-control-expose-headers": "content-range, content-length, accept-ranges" });
  fs.createReadStream(p, { start: lo, end: hi }).pipe(r);
}).listen(TLS_PORT, "127.0.0.1");
const peerServer = spawn(path.join(ROOT, "node_modules/.bin/peerjs"), ["--port", String(SIG_PORT), "--path", "/", "--host", "127.0.0.1"], { stdio: "ignore" });
await sleep(1500);

// ---------- UDP shaper for WebRTC ----------
const C = { rtt: 0, loss: 0, frozen: new Map(), dead: new Set(), freezeAll: 0 };   // frozen: tab -> until (ms)
const stats = { pkts: 0, lost: 0, frozen: 0, dead: 0 };
const localCand = new Map();   // "ip:port" -> tab that gathered it
const relays = new Map();      // "tab|ip:port" -> relay port
function shape(from, to, buf, send) {
  stats.pkts++;
  if (C.dead.has(from) || C.dead.has(to)) { stats.dead++; return; }
  const now = Date.now();
  if (now < C.freezeAll || now < (C.frozen.get(from) || 0) || now < (C.frozen.get(to) || 0)) { stats.frozen++; return; }
  if (C.loss && Math.random() < C.loss) { stats.lost++; return; }
  const d = C.rtt / 2;
  if (d) setTimeout(send, d); else send();
}
async function relayFor(tab, ip, port) {
  const key = `${tab}|${ip}:${port}`;
  if (relays.has(key)) return relays.get(key);
  const lis = dgram.createSocket("udp4");
  await new Promise((r) => lis.bind(0, "127.0.0.1", r));
  const clients = new Map();
  const owner = () => localCand.get(`${ip}:${port}`) || "?";
  lis.on("message", (buf, ri) => {
    const k = ri.address + ":" + ri.port;
    let out = clients.get(k);
    if (!out) {
      out = dgram.createSocket("udp4");
      out.on("message", (b2) => shape(owner(), tab, b2, () => lis.send(b2, ri.port, ri.address)));
      out.on("error", () => {});
      clients.set(k, out);
    }
    shape(tab, owner(), buf, () => out.send(buf, port, ip));
  });
  lis.on("error", () => {});
  const p = lis.address().port;
  relays.set(key, p);
  return p;
}
const INIT = `(() => {
  const O = window.RTCPeerConnection; if (!O) return;
  window.__ice = []; let n = 0;
  class P extends O {
    constructor(...a) {
      super(...a); const id = ++n;
      this.addEventListener("icecandidate", (e) => { const c = e.candidate?.candidate; if (c) window.__chaosLocal(c).catch(() => {}); });
      this.addEventListener("iceconnectionstatechange", () => window.__ice.push([Date.now(), id, this.iceConnectionState]));
    }
    async addIceCandidate(c, ...rest) {
      if (!c || !c.candidate) return super.addIceCandidate(c, ...rest);
      const f = c.candidate.split(" ");
      if ((f[2] || "").toLowerCase() !== "udp" || f[7] !== "host" || f[4].includes(":")) return;
      const port = await window.__chaosRelay(f[4], +f[5]);
      f[4] = "127.0.0.1"; f[5] = String(port);
      return super.addIceCandidate({ candidate: f.join(" "), sdpMid: c.sdpMid, sdpMLineIndex: c.sdpMLineIndex, usernameFragment: c.usernameFragment }, ...rest);
    }
    setRemoteDescription(d, ...rest) {
      if (d && d.sdp) d = { type: d.type, sdp: d.sdp.split("\\r\\n").filter((l) => !l.startsWith("a=candidate")).join("\\r\\n") };
      return super.setRemoteDescription(d, ...rest);
    }
  }
  window.RTCPeerConnection = P; window.webkitRTCPeerConnection = P;
  // answer text growth, sampled every 50 ms: [Date.now(), chars] on change
  window.__tok = []; let last = -1;
  setInterval(() => { const b = [...document.querySelectorAll(".m.bot .bubble")].pop(); const L = b ? b.textContent.length : 0; if (L !== last) { last = L; window.__tok.push([Date.now(), L]); } }, 50);
  // every toast, with its time
  window.__toasts = [];
  addEventListener("DOMContentLoaded", () => { const box = document.getElementById("toasts"); if (box) new MutationObserver((ms) => { for (const m of ms) for (const nd of m.addedNodes) window.__toasts.push([Date.now(), nd.textContent]); }).observe(box, { childList: true }); });
})();`;

// ---------- signaling proxies ----------
function sigProxy(tab, port) {
  const st = { tab, port, mode: "pass", pairs: new Set(), server: null, held: [] };
  const listen = () => new Promise((res) => {
    st.server = net.createServer((c) => {
      c.on("error", () => {});
      const pair = { c, u: null, q: [] };
      st.pairs.add(pair);
      const u = pair.u = net.connect(SIG_PORT, "127.0.0.1");
      u.on("error", () => {});
      const fwd = (dst, d) => { if (st.mode === "pass") dst.write(d); else pair.q.push([dst, d]); };
      c.on("data", (d) => fwd(u, d)); u.on("data", (d) => fwd(c, d));
      const end = () => { st.pairs.delete(pair); c.destroy(); u.destroy(); };
      c.on("close", end); u.on("close", end);
    });
    st.server.listen(port, "127.0.0.1", res);
  });
  st.set = async (mode) => {
    const was = st.mode; st.mode = mode;
    if (mode === "down") {
      for (const p of st.pairs) { p.c.resetAndDestroy?.() ?? p.c.destroy(); p.u.destroy(); }
      st.pairs.clear();
      if (st.server) { await new Promise((r) => st.server.close(r)); st.server = null; }
    } else if (!st.server) await listen();
    if (mode === "pass" && was === "blackhole") for (const p of st.pairs) { for (const [dst, d] of p.q) dst.write(d); p.q = []; }
  };
  st.ready = listen();
  return st;
}

// ---------- browsers ----------
const ARGS = ["--no-sandbox", "--headless=new", "--enable-unsafe-webgpu", "--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan", "--ignore-gpu-blocklist",
  "--allow-loopback-in-peer-connection", "--disable-features=WebRtcHideLocalIpsWithMdns", "--js-flags=--max-old-space-size=16384"];
const UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";
const profiles = [], contexts = [], tabs = {}, sig = {}, errs = [];
async function launch(name, i) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-drop-profile-")); profiles.push(dir);
  const c = await chromium.launchPersistentContext(dir, { headless: false, args: ARGS, userAgent: UA, ignoreHTTPSErrors: true });
  await c.route("**/*.gguf", (route) => {
    const file = LOCAL[route.request().url().split("/").pop().split("?")[0]];
    if (!file || !fs.existsSync(path.join(ROOT, file))) return route.continue();
    return route.continue({ url: `https://127.0.0.1:${TLS_PORT}/${file}` });
  });
  await c.exposeBinding("__chaosLocal", (_s, cand) => { const f = cand.split(" "); if (f[4] && f[5]) localCand.set(`${f[4]}:${f[5]}`, name); });
  await c.exposeBinding("__chaosRelay", (_s, ip, port) => relayFor(name, ip, port));
  await c.addInitScript(INIT);
  contexts.push(c);
  const p = c.pages()[0] || await c.newPage();
  p.on("pageerror", (e) => errs.push(name + ": " + String(e).slice(0, 200)));
  p.on("crash", () => errs.push(name + ": crashed"));
  p.on("console", (m) => { const t = m.text(); if (m.type() === "error" || /PeerJS|ice|lost|GPU/i.test(t)) { if (!/favicon|404/.test(t)) console.error("  [" + name + "]", t.slice(0, 220)); } });
  sig[name] = sigProxy(name, SIG_PORT + 1 + i); await sig[name].ready;
  tabs[name] = p;
  return p;
}
const url = (name) => `http://127.0.0.1:${PORT}/p2p.html?ask=0&signal=127.0.0.1:${sig[name].port}&maxnew=${MAXNEW}&peerweights=0${QUERY ? "&" + QUERY : ""}`;

// what a screen shows, in a few fields
const snap = (p) => Promise.race([sleep(2000).then(() => ({ err: "snapshot timed out" })), snap1(p)]);
const snap1 = (p) => p.evaluate(() => {
  const vis = (id) => { const e = document.getElementById(id); return !!e && !e.hidden && e.offsetParent !== null; };
  const tx = (id) => (document.getElementById(id)?.textContent || "").replace(/\s+/g, " ").trim();
  return { status: tx("ai-status").slice(0, 160), join: tx("join-status").slice(0, 200), over: vis("room-over") ? tx("room-over-h") + ": " + tx("room-over-why") : "",
    cards: document.querySelectorAll(".peer-card").length, redeal: vis("ai-redeal") ? tx("redeal-why").slice(0, 200) : "", inRoom: document.getElementById("room-screen")?.style.display === "flex",
    bot: ([...document.querySelectorAll(".m.bot .bubble")].pop()?.textContent || "").length };
}).catch((e) => ({ err: String(e).slice(0, 80) }));
// watch every screen every 250 ms; record each change of what it shows
let watching = null;
function watch() {
  const tl = []; const last = {}; const w0 = Date.now(); let on = true;
  (async () => { while (on) { for (const [n, p] of Object.entries(tabs)) { const s = await snap(p); const { bot, ...rest } = s; const k = JSON.stringify(rest); if (k !== last[n]) { last[n] = k; tl.push({ t: Date.now() - w0, tab: n, ...rest }); } } await sleep(250); } })();
  watching = { tl, w0, stop: () => { on = false; return tl; } };
  return watching;
}

async function setupRoom(names) {
  for (const n of names) await tabs[n].goto(url(n));
  for (const n of names) await tabs[n].waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
  for (const [i, n] of names.entries()) { await tabs[n].fill("#name-input", n); await tabs[n].fill("#join-gb", GB(i)); }
  const host = tabs[names[0]];
  await host.click("#create-btn");
  await host.waitForFunction(() => /[A-Z0-9]{3}-?[A-Z0-9]{3}|[A-Z0-9]{4}/.test(document.getElementById("side-code").textContent), null, { timeout: 30000 });
  const code = (await host.textContent("#side-code")).trim().replace("-", "").match(/[A-Z0-9]{6}|[A-Z0-9]{4}/)[0];
  for (const n of names.slice(1)) { await tabs[n].fill("#code-input", code); await tabs[n].click("#join-btn"); await tabs[n].waitForTimeout(200); }
  for (const n of names) await tabs[n].waitForFunction((k) => document.querySelectorAll(".peer-card").length >= k, names.length, { timeout: 60000 });
  return code;
}
async function loadModel(names) {
  const host = tabs[names[0]];
  await host.waitForTimeout(2000);
  await host.selectOption("#ai-model", MODEL);
  await host.evaluate(() => { const s = document.getElementById("ai-split"); s.value = "memory"; s.dispatchEvent(new Event("change", { bubbles: true })); });
  const tl = Date.now();
  await host.click("#ai-start");
  // progress every 20 s, so a stuck load shows where it stopped
  let loading = true;
  (async () => { while (loading) { await sleep(20000); if (!loading) break; for (const n of names) log("  load", n, JSON.stringify(await snap(tabs[n])).slice(0, 300), (await tabs[n].textContent("#ai-status").catch(() => "")).slice(0, 160)); } })();
  try {
  for (const n of names) await tabs[n].waitForFunction(() => document.getElementById("ai-panel").classList.contains("online") || /^failed:/.test(document.getElementById("ai-status").textContent), null, { timeout: 900000, polling: 1000 });
  } finally { loading = false; }
  const st = await host.textContent("#ai-status");
  if (/^failed:/.test(st)) throw new Error("load " + st);
  await host.evaluate(() => { const s = document.getElementById("ai-sampling"); s.value = "exact"; s.dispatchEvent(new Event("change")); });
  const split = await host.evaluate(() => [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent).filter((t) => /layer split/.test(t)).slice(-1)[0] || "");
  log("online in", ((Date.now() - tl) / 1000).toFixed(0), "s", split.slice(0, 200));
  return split;
}
// ask on the host; resolves when the answer ends (or fails), with timings. during(tStart) runs mid-answer.
async function ask(hostName, { during, timeoutMs = 240000, newChat = true } = {}) {
  const host = tabs[hostName];
  if (newChat) { await host.evaluate(() => document.getElementById("new-chat").click()); await host.waitForTimeout(400); }
  for (const p of Object.values(tabs)) await p.evaluate(() => { window.__tok = []; window.__ice = []; window.__toasts = []; }).catch(() => {});
  const ts = Date.now();
  await host.evaluate((text) => { const box = document.getElementById("ai-prompt"); box.value = text; box.dispatchEvent(new Event("input")); document.getElementById("ai-send").click(); }, PROMPT);
  let dur = null;
  if (during) dur = during(ts).catch((e) => ({ err: String(e) }));
  let timedOut = false;
  try { await host.waitForFunction(() => /^ready — |^generation failed/.test(document.getElementById("ai-status").textContent), null, { timeout: timeoutMs, polling: 100 }); }
  catch { timedOut = true; }
  const tEnd = Date.now();
  const st = await host.textContent("#ai-status");
  const toks = await host.evaluate(() => window.__tok);
  const text = await host.evaluate(() => [...document.querySelectorAll(".m.bot .bubble")].pop()?.textContent || "");
  const first = toks.find(([, L]) => L > 0);
  let maxGap = 0, gapAt = null;
  for (let i = 1; i < toks.length; i++) if (toks[i][1] > 0 && toks[i - 1][1] > 0) { const g = toks[i][0] - toks[i - 1][0]; if (g > maxGap) { maxGap = g; gapAt = toks[i - 1][0] - ts; } }
  const dec = /(\d+) tok · ([\d.]+) tok\/s/.exec(st);
  const guestSeen = {};
  for (const [n, p] of Object.entries(tabs)) if (n !== hostName) guestSeen[n] = await p.evaluate(() => ({ chars: ([...document.querySelectorAll(".m.bot .bubble")].pop()?.textContent || "").length, status: document.getElementById("ai-status")?.textContent.slice(0, 120) })).catch(() => null);
  return { ok: !timedOut && /^ready — /.test(st), timedOut, totalS: +((tEnd - ts) / 1000).toFixed(1), ttftS: first ? +((first[0] - ts) / 1000).toFixed(2) : null,
    tokens: dec && +dec[1], tps: dec && +dec[2], maxStallS: +(maxGap / 1000).toFixed(2), stallAtS: gapAt && +(gapAt / 1000).toFixed(1), chars: text.length, text,
    status: st.slice(0, 200), guestSeen, during: dur ? await dur : null };
}
const iceLog = async (n) => (await tabs[n].evaluate(() => window.__ice).catch(() => [])).map(([t, id, s]) => [t, id, s]);
const toastLog = async (n) => (await tabs[n].evaluate(() => window.__toasts).catch(() => []));
// wait until fn(snapshot) holds on a tab; returns ms or null
async function until(n, pred, ms, t0 = Date.now()) {
  while (Date.now() - t0 < ms) { const s = await snap(tabs[n]); if (pred(s)) return Date.now() - t0; await sleep(200); }
  return null;
}

// ---------- scenarios ----------
const out = { model: MODEL, devices: DEVICES, maxnew: MAXNEW, query: QUERY, rows: [], checks: [] };
const names = Array.from({ length: DEVICES }, (_, i) => (i === 0 ? "host" : "guest" + i));
const want = (s) => !ONLY || ONLY.split(",").includes(s);
function row(r) { const { text, ...rest } = r; log(JSON.stringify(rest).slice(0, 900)); out.rows.push(r); }
function check(name, pass, detail) { out.checks.push({ name, pass: !!pass, detail }); log(pass ? "PASS" : "FAIL", name, detail ?? ""); }
let baselineText = null;
const same = (t) => baselineText == null ? null : t === baselineText ? "identical" : `differs (${t.length} vs ${baselineText.length} chars; first diff at ${[...t].findIndex((c, i) => c !== baselineText[i])})`;
const liveness = () => tabs[names[0]].evaluate(() => window.pooledLiveness?.() || null).catch(() => null);
// chain order from the host's "layer split" line: device names in the order they appear
function chainOrder(split) {
  const at = names.slice(1).map((n) => [n, split.search(new RegExp("\\b" + n + "\\b"))]).filter(([, i]) => i >= 0);
  return at.sort((a, b) => a[1] - b[1]).map(([n]) => n);
}
async function redealAndWait(H) {
  const tr = Date.now();
  const clicked = await tabs[H].evaluate(() => { const b = document.getElementById("ai-redeal"); if (b && !b.hidden) { b.click(); return true; } return false; });
  if (!clicked) return { clicked };
  await tabs[H].waitForFunction(() => document.getElementById("ai-panel").classList.contains("online") && /^cluster online/.test(document.getElementById("ai-status").textContent), null, { timeout: 300000, polling: 500 }).catch(() => {});
  const split = await tabs[H].evaluate(() => [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent).filter((t) => /layer split/.test(t)).slice(-1)[0] || "");
  return { clicked, redealS: +((Date.now() - tr) / 1000).toFixed(1), split: split.slice(0, 200) };
}
// a device dies silently mid-answer: all its packets and its signaling vanish, its tab stays open
async function dieMid(H, V, label) {
  const w = watch();
  let deathAt = 0;
  const r = await ask(H, { timeoutMs: 180000, during: async (ts) => {
    await tabs[H].waitForFunction(() => ([...document.querySelectorAll(".m.bot .bubble")].pop()?.textContent || "").length > 40, null, { timeout: 90000, polling: 50 });
    deathAt = Date.now(); C.dead.add(V); await sig[V].set("blackhole");
    // how long until the host's screen says so, polled on the host alone
    const seenMs = await until(H, (x) => /stopped responding|left|failed/.test(x.status || "") || !!x.redeal, 60000, deathAt);
    return { deathAtS: +((deathAt - ts) / 1000).toFixed(1), seenMs };
  } });
  // when the host said so: the first timeline entry after the death that names it or shows the re-deal
  w.stop();
  const hit = w.tl.find((e) => e.tab === H && e.t + w.w0 >= deathAt && (/stopped responding|left|failed/.test(e.status) || e.redeal));
  const detectMs = r.during?.seenMs ?? (hit ? hit.t + w.w0 - deathAt : null);
  const scr = await snap(tabs[H]);
  row({ scenario: label, victim: V, ...r, detectMs, hostScreen: scr, timeline: w.tl.filter((e) => e.tab === H) });
  return { r, detectMs, scr };
}

let failed = false;
try {
  for (const [i, n] of names.entries()) await launch(n, i);
  const code = await setupRoom(names);
  log("room", code, "with", names.join(", "));
  let split = await loadModel(names);
  out.split = split;
  const H = names[0];
  C.rtt = 20;   // a LAN: packets still take the relay
  const base = await ask(H); baselineText = base.text;
  row({ scenario: "baseline", ...base, liveness: await liveness() });
  check("baseline answers", base.ok, base.status);

  for (const [nm, rtt] of [["noise300", 300], ["noise600", 600]]) {
    if (!want(nm)) continue;
    C.rtt = rtt; C.loss = 0.05;
    const lost0 = stats.lost;
    const r = await ask(H, { timeoutMs: 600000 });
    const lv = await liveness();
    C.loss = 0; C.rtt = 20;
    row({ scenario: nm, rtt, loss: 0.05, ...r, sameAsBaseline: same(r.text), liveness: lv, packetsLost: stats.lost - lost0 });
    check(`${nm}: answer completes, nobody dropped`, r.ok && !/stopped responding|left/.test(r.status), r.status);
    check(`${nm}: same text as the baseline`, same(r.text) === "identical", same(r.text));
    // every device's longest silence stays half a second under the limit it was held to (the
    // limit follows its measured RTT; the 3.5 s floor when none was measured)
    const tight = lv ? Object.entries(lv.maxSilence).filter(([n, ms]) => ms > (lv.limit?.[n] ?? 3500) - 500) : null;
    check(`${nm}: longest silence well under the limit`, tight != null && Object.keys(lv.maxSilence).length > 0 && !tight.length, JSON.stringify(lv));
    const sn = await snap(tabs[H]);
    check(`${nm}: every device still in the room`, sn.cards === names.length && !sn.redeal, JSON.stringify(sn));
  }

  const hb = !/(^|&)hb=0/.test(QUERY);
  for (const [nm, pick] of [["die-mid", (o) => o[Math.floor((o.length - 1) / 2)]], ["die-last", (o) => o[o.length - 1]]]) {
    if (!want(nm)) continue;
    const order = chainOrder(split);
    if (order.length < 2 && nm === "die-mid") { log("skip", nm, "chain too short", order); continue; }
    const V = pick(order);
    const { r, detectMs, scr } = await dieMid(H, V, nm);
    if (hb) {
      check(`${nm}: host notices ${V} within 6 s`, detectMs != null && detectMs <= 6000, `${detectMs} ms`);
      check(`${nm}: the answer ends (the link fails or the device is dropped)`, !r.timedOut && r.totalS <= 45 && /generation failed/.test(r.status), `${r.status} after ${r.totalS} s`);
      // the device is held (a freeze keeps its place) until the ping loop drops it as silent: re-deal offered
      const redealMs = await until(H, (x) => !!x.redeal, 45000);
      row({ scenario: nm + "_evict", victim: V, redealOfferedMsAfterAnswerEnd: redealMs });
      check(`${nm}: re-deal offered within 35 s of the answer failing`, redealMs != null && redealMs <= 35000, `${redealMs} ms`);
    }
    names.splice(names.indexOf(V), 1); delete tabs[V];   // out of the room (its browser stays open, cut off)
    const rd = await redealAndWait(H);
    split = rd.split || split;
    const next = await ask(H, { timeoutMs: 300000 }).catch((e) => ({ err: String(e) }));
    row({ scenario: nm + "_recover", ...rd, ...next });
    if (hb) check(`${nm}: after a re-deal the room answers again`, rd.clicked && next.ok, next.status || next.err);
  }
} catch (e) {
  out.error = String(e.stack || e).slice(0, 600);
  log("FAILED", out.error);
  failed = true;
} finally {
  out.errors = errs; out.relay = stats;
  failed = failed || out.checks.some((c) => !c.pass);
  console.log(JSON.stringify(out));
  peerServer.kill(); fs.rmSync(tlsDir, { recursive: true, force: true });
  await Promise.race([Promise.all(contexts.map((c) => c.close().catch(() => {}))), sleep(15000)]);
  for (const d of profiles) fs.rmSync(d, { recursive: true, force: true });
  process.exit(failed ? 1 : 0);
}
