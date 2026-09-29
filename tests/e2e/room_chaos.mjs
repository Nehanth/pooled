// Room under bad networks: N devices on this machine (one Chromium each, real PeerJS signaling,
// real WebRTC), with every WebRTC packet and every signaling byte going through a userland
// shaper in this process, so a scenario can add latency, drop packets, freeze a device's
// network, kill a device's network silently, or take the signaling server away. No root needed.
//
//   node tests/e2e/room_chaos.mjs --model qwen3-1.7b --devices 3 --plan net     latency / loss / freezes
//   node tests/e2e/room_chaos.mjs --devices 3 --plan death                         a guest's network dies (idle, then mid-answer)
//   node tests/e2e/room_chaos.mjs --devices 3 --plan signal                        signaling down mid-session, joins while down
//   node tests/e2e/room_chaos.mjs --devices 2 --plan join --nomodel                signaling unreachable at join time
//   node tests/e2e/room_chaos.mjs --devices 3 --plan turn [--nomodel]              no direct path: join fails without a relay,
//                                                                                   works through a TURN relay (tests/e2e/turn_server.mjs)
//
// How the shaping works:
// - WebRTC: an init script subclasses RTCPeerConnection. Remote ICE candidates are rewritten to
//   127.0.0.1:<relay port> (UDP host candidates only; TCP, srflx and IPv6 ones are dropped), and
//   the relay forwards to the real address. Every connectivity check and SCTP packet therefore
//   crosses the relay, which knows which device sent it and to whom, and can delay it (half the
//   RTT each way), drop it at random (loss), or drop everything to or from a device (freeze, dead).
//   Real SCTP retransmission and real ICE consent/timeout behaviour follow from that.
// - Signaling: each device gets its own TCP proxy in front of the local PeerServer (?signal=
//   points at it). Modes: pass; down (listener closed: connection refused, open sockets reset);
//   blackhole (connections accepted, bytes held until the mode goes back to pass, like a path that
//   drops packets while TCP keeps retrying).
// Output: one JSON line on stdout with a row per scenario (time to detect, completion, recovery,
// what each screen showed); progress on stderr.
import { chromium } from "playwright";
import http from "http";
import https from "https";
import net from "net";
import dgram from "dgram";
import fs from "fs";
import os from "os";
import path from "path";
import { spawn, execSync } from "child_process";
import { startTurn } from "./turn_server.mjs";

const arg = (k, d) => { const i = process.argv.indexOf("--" + k); return i > 0 ? process.argv[i + 1] : d; };
const flag = (k) => process.argv.includes("--" + k);
const MODEL = arg("model", "qwen3-1.7b"), DEVICES = Math.max(2, +arg("devices", 3)), PLAN = arg("plan", "net");
const MAXNEW = +arg("maxnew", 96), PORT = +arg("port", 8460), TLS_PORT = PORT + 1, SIG_PORT = +arg("signal-port", 9460);
const NOMODEL = flag("nomodel");
const ONLY = arg("only", "");   // comma list of scenario names to run from the plan
const PROMPT = arg("prompt", "Explain in about a hundred words how a bicycle gear system works, then list three tips for riding up a steep hill.");
const NEED = { "qwen3-1.7b": 2.0, "qwen3-0.6b": 0.8 }[MODEL] || 2;
// the first tab pledges the most, so it is the model host (the strongest device deals the layers
// and runs the sampler); the others pledge little, and the split by memory still gives each a share
const GB = (i) => arg("gb", String(i === 0 ? Math.ceil(NEED) + 1 : 1));
const LOCAL = { "Qwen3-0.6B-Q8_0.gguf": "models/qwen/model.gguf", "Qwen3-1.7B-Q8_0.gguf": "models/qwen17/model.gguf" };
// --root serves another checkout (e.g. origin/main, to compare) with this harness
const ROOT = path.resolve(arg("root", path.resolve(new URL(".", import.meta.url).pathname, "../..")));
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
const tlsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-chaos-"));
execSync(`openssl req -x509 -newkey rsa:2048 -nodes -keyout ${tlsDir}/k.pem -out ${tlsDir}/c.pem -days 2 -subj /CN=127.0.0.1 2>/dev/null`);
const wsrv = https.createServer({ key: fs.readFileSync(`${tlsDir}/k.pem`), cert: fs.readFileSync(`${tlsDir}/c.pem`) }, (q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p)) { r.statusCode = 404; r.end(); return; }
  const size = fs.statSync(p).size, m = /bytes=(\d+)-(\d*)/.exec(q.headers.range || "");
  const lo = m ? +m[1] : 0, hi = m && m[2] ? Math.min(+m[2], size - 1) : size - 1;
  r.writeHead(m ? 206 : 200, { "content-type": "application/octet-stream", "content-range": `bytes ${lo}-${hi}/${size}`, "accept-ranges": "bytes", "content-length": String(hi - lo + 1), "access-control-allow-origin": "*", "access-control-expose-headers": "content-range, content-length, accept-ranges" });
  fs.createReadStream(p, { start: lo, end: hi }).pipe(r);
}).listen(TLS_PORT, "127.0.0.1");
const peerServer = spawn(path.resolve(new URL(".", import.meta.url).pathname, "../../node_modules/.bin/peerjs"), ["--port", String(SIG_PORT), "--path", "/", "--host", "127.0.0.1"], { stdio: "ignore" });
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
  window.__ice = []; window.__pcs = []; let n = 0;
  class P extends O {
    constructor(...a) {
      super(...a); const id = ++n; window.__pcs.push([id, this]);
      this.addEventListener("icecandidate", (e) => { const c = e.candidate?.candidate; if (c) window.__chaosLocal(c).catch(() => {}); });
      this.addEventListener("iceconnectionstatechange", () => window.__ice.push([Date.now(), id, this.iceConnectionState]));
    }
    async addIceCandidate(c, ...rest) {
      if (!c || !c.candidate) return super.addIceCandidate(c, ...rest);
      const f = c.candidate.split(" ");
      if ((f[2] || "").toLowerCase() !== "udp" || f[4].includes(":")) return;
      const port = await window.__chaosRelay(f[4], +f[5], f[7]);
      if (port === "pass") return super.addIceCandidate(c, ...rest);
      if (!port) return;
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
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-chaos-profile-")); profiles.push(dir);
  const c = await chromium.launchPersistentContext(dir, { headless: false, args: ARGS, userAgent: UA, ignoreHTTPSErrors: true });
  await c.route("**/*.gguf", (route) => {
    const file = LOCAL[route.request().url().split("/").pop().split("?")[0]];
    if (!file || !fs.existsSync(path.join(ROOT, file))) return route.continue();
    return route.continue({ url: `https://127.0.0.1:${TLS_PORT}/${file}` });
  });
  await c.exposeBinding("__chaosLocal", (_s, cand) => { const f = cand.split(" "); if (f[4] && f[5]) localCand.set(`${f[4]}:${f[5]}`, name); });
  // host candidates go through the shaper; with --plan turn they are dropped (no direct path) and
  // the TURN server's relay candidates pass unchanged; anything else is dropped
  await c.exposeBinding("__chaosRelay", (_s, ip, port, type) => PLAN === "turn" ? (type === "relay" ? "pass" : null) : type === "host" ? relayFor(name, ip, port) : null);
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
const EXTRA = {};   // per-tab extra query, e.g. a TURN relay
const url = (name) => `http://127.0.0.1:${PORT}/p2p.html?signal=127.0.0.1:${sig[name].port}&maxnew=${MAXNEW}&peerweights=0${EXTRA[name] || ""}`;

// what a screen shows, in a few fields
const snap = (p) => p.evaluate(() => {
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
  // a tab whose WebGPU adapter request came back empty (the GPU busy with other browsers) joins
  // as an ask-only guest and never gets layers: reload it until it has an adapter
  if (!NOMODEL) for (const n of names) for (let k = 0; k < 4; k++) {
    if (await tabs[n].evaluate(async () => !!(await navigator.gpu?.requestAdapter()))) break;
    log(n, "has no WebGPU adapter, reloading"); await sleep(2000);
    await tabs[n].goto(url(n)); await tabs[n].waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
  }
  for (const [i, n] of names.entries()) { await tabs[n].fill("#name-input", n); await tabs[n].fill("#join-gb", GB(i)); }
  const host = tabs[names[0]];
  await host.click("#create-btn");
  await host.waitForFunction(() => /[A-Z0-9]{4}/.test(document.getElementById("side-code").textContent), null, { timeout: 30000 });
  const code = (await host.textContent("#side-code")).trim().match(/[A-Z0-9]{4}/)[0];
  for (const n of names.slice(1)) { await tabs[n].fill("#code-input", code); await tabs[n].click("#join-btn"); await tabs[n].waitForTimeout(200); }
  for (const n of names) await tabs[n].waitForFunction((k) => document.querySelectorAll(".peer-card").length >= k, names.length, { timeout: 60000 });
  const noGpu = await host.evaluate(() => [...document.querySelectorAll(".peer-gpu")].map((e) => e.textContent).filter((t) => /no WebGPU/.test(t)));
  if (noGpu.length && !NOMODEL) log("WARNING: devices without WebGPU (they get no layers):", noGpu.join(" | "));
  return code;
}
async function loadModel(names) {
  const host = tabs[names[0]];
  await host.waitForTimeout(2000);
  await host.selectOption("#ai-model", MODEL);
  const tl = Date.now();
  await host.click("#ai-start");
  const prog = setInterval(async () => { const st = {}; for (const n of names) st[n] = (await snap(tabs[n])).status; log("loading", JSON.stringify(st).slice(0, 400)); }, 20000);
  // the host's panel goes online once every device in the deal has loaded; a device left out of
  // the deal (no WebGPU) never goes online, so only wait for it briefly
  const up = () => document.getElementById("ai-panel").classList.contains("online") || /^failed:/.test(document.getElementById("ai-status").textContent);
  try {
    await host.waitForFunction(up, null, { timeout: 900000, polling: 1000 });
    for (const n of names.slice(1)) await tabs[n].waitForFunction(up, null, { timeout: 30000, polling: 1000 }).catch(() => log(n, "is not in the deal:", ""));
  } finally { clearInterval(prog); }
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
  // clear every status line, so the previous answer's "ready — …" does not count as this one's end
  for (const p of Object.values(tabs)) await p.evaluate(() => { window.__tok = []; window.__ice = []; window.__toasts = []; const e = document.getElementById("ai-status"); if (e) e.textContent = ""; }).catch(() => {});
  const ts = Date.now();
  await host.evaluate((text) => { const box = document.getElementById("ai-prompt"); box.value = text; box.dispatchEvent(new Event("input")); document.getElementById("ai-send").click(); }, PROMPT);
  let dur = null;
  if (during) dur = during(ts).catch((e) => ({ err: String(e) }));
  let timedOut = false;
  // the answer's final status shows on the model host, which may not be the tab that asked
  const fin = async () => { for (const p of Object.values(tabs)) { const t = await p.textContent("#ai-status").catch(() => ""); if (/^ready — |^generation failed/.test(t)) return t; } return null; };
  let st = null;
  while (!(st = await fin())) { if (Date.now() - ts > timeoutMs) { timedOut = true; st = await host.textContent("#ai-status"); break; } await sleep(100); }
  const tEnd = Date.now();
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
// what each tab's links look like after a failure: wire state, every peer connection's states,
// the room log's last lines
const diag = async () => { const o = {}; for (const [n, p] of Object.entries(tabs)) o[n] = await p.evaluate(() => ({
  wire: window.pooledDebug?.(),
  pcs: (window.__pcs || []).map(([id, pc]) => [id, pc.iceConnectionState, pc.connectionState, pc.sctp?.state || null]),
  log: [...document.querySelectorAll("#chat-log div")].slice(-8).map((d) => d.textContent.slice(0, 160)),
})).catch((e) => String(e).slice(0, 80)); return o; };
const iceLog = async (n) => (await tabs[n].evaluate(() => window.__ice).catch(() => [])).map(([t, id, s]) => [t, id, s]);
const toastLog = async (n) => (await tabs[n].evaluate(() => window.__toasts).catch(() => []));
// wait until fn(snapshot) holds on a tab; returns ms or null
async function until(n, pred, ms, t0 = Date.now()) {
  while (Date.now() - t0 < ms) { const s = await snap(tabs[n]); if (pred(s)) return Date.now() - t0; await sleep(200); }
  return null;
}

// ---------- plans ----------
const out = { model: MODEL, devices: DEVICES, maxnew: MAXNEW, plan: PLAN, rows: [] };
const names = Array.from({ length: DEVICES }, (_, i) => (i === 0 ? "host" : "guest" + i));
const want = (s) => !ONLY || ONLY.split(",").includes(s);
function row(r) { const { text, ...rest } = r; log(JSON.stringify(rest).slice(0, 900)); out.rows.push(r); }
let baselineText = null;
const same = (t) => baselineText == null ? null : t === baselineText ? "identical" : `differs (${t.length} vs ${baselineText.length} chars; first diff at ${[...t].findIndex((c, i) => c !== baselineText[i])})`;

try {
  for (const [i, n] of names.entries()) await launch(n, i);
  if (PLAN === "join") {
    // signaling unreachable at join time: no model needed
    const [h, g] = names;
    for (const mode of ["down", "blackhole"]) {
      if (want("join-host-" + mode)) {
        await sig[h].set(mode);
        await tabs[h].goto(url(h)); await tabs[h].waitForFunction(() => document.getElementById("join-gb").value !== "");
        const w = watch(); const ts = Date.now();
        await tabs[h].click("#create-btn");
        const ms = await until(h, (s) => s.inRoom || /error|fail|no room|could not/i.test(s.join), 45000);
        await sleep(1000); const s = await snap(tabs[h]); w.stop();
        await sig[h].set("pass");
        const recover = await until(h, (s) => s.inRoom, 20000);
        row({ scenario: "join-host-" + mode, what: `signaling ${mode} when the host opens a room`, detectMs: ms, screen: s, recoveredAfterServerBackMs: recover, timeline: w.tl.filter((e) => e.tab === h) });
      }
      if (want("join-guest-" + mode)) {
        await sig[h].set("pass"); await sig[g].set("pass");
        await tabs[h].goto(url(h)); await tabs[g].goto(url(g));
        const code = await setupRoom([h]).catch((e) => { throw e; });
        await sig[g].set(mode);
        await tabs[g].waitForFunction(() => document.getElementById("join-gb").value !== "");
        await tabs[g].fill("#code-input", code);
        const w = watch();
        await tabs[g].click("#join-btn");
        const ms = await until(g, (s) => s.inRoom || /error|fail|no room|could not/i.test(s.join), 45000);
        await sleep(1000); const s = await snap(tabs[g]);
        await sig[g].set("pass");
        const rec = await until(g, (s) => s.inRoom, 30000);
        w.stop();
        row({ scenario: "join-guest-" + mode, what: `signaling ${mode} when a guest joins`, detectMs: ms, screen: s, recoveredAfterServerBackMs: rec, timeline: w.tl.filter((e) => e.tab === g) });
      }
    }
  } else if (PLAN === "turn") {
    // No direct path between the devices (every host candidate is dropped, as behind symmetric NAT
    // on both sides). Without a relay a join must fail and say a relay would help; with ?turn= it
    // must connect through the relay (fallback), and with ?relay=1 use only the relay.
    const turn = await startTurn({ user: "pooled", pass: "chaos-" + process.pid });
    out.turnPort = turn.port;
    const [h, g] = names;
    if (want("turn-none")) {
      for (const n of names) EXTRA[n] = "";
      await tabs[h].goto(url(h)); await tabs[g].goto(url(g));
      const code = await setupRoom([h]);
      await tabs[g].waitForFunction(() => document.getElementById("join-gb").value !== "");
      await tabs[g].fill("#code-input", code);
      const ts = Date.now();
      await tabs[g].click("#join-btn");
      const ms = await until(g, (s) => s.inRoom || /could not|failed|no room/i.test(s.join), 45000, ts);
      const s = await snap(tabs[g]);
      const netOpen = await tabs[g].evaluate(() => !!document.getElementById("join-net")?.open);
      row({ scenario: "turn-none", what: "no direct path, no relay configured: the join fails and points at a relay", ok: !s.inRoom && /relay/i.test(s.join), detectMs: ms, screen: s, networkBoxOpened: netOpen });
    }
    for (const [nm, force] of [["turn-fallback", false], ["turn-only", true]]) {
      if (!want(nm)) continue;
      const q = `&turn=${encodeURIComponent("turn:127.0.0.1:" + turn.port)}&turnuser=pooled&turncred=${encodeURIComponent("chaos-" + process.pid)}${force ? "&relay=1" : ""}`;
      for (const n of names) EXTRA[n] = q;
      const before = { ...turn.stats };
      const ts = Date.now();
      let code = null, err = null;
      try { code = await setupRoom(names); } catch (e) { err = String(e).slice(0, 200); }
      const joinS = +((Date.now() - ts) / 1000).toFixed(1);
      await sleep(4500);   // notePath reads getStats 3 s after a link opens
      const dbg = {}; for (const n of names) dbg[n] = await tabs[n].evaluate(() => window.pooledDebug?.()).catch(() => null);
      const paths = Object.values(dbg).flat().filter(Boolean).map((d) => d.path);
      let ans = null;
      if (!err && !NOMODEL && nm === "turn-only") { out.split = await loadModel(names); ans = await ask(h); }
      const { text, ...a } = ans || {};
      row({ scenario: nm, what: force ? "no direct path, relay only (?relay=1)" : "no direct path, relay configured: ICE falls back to it",
        ok: !err && paths.length > 0 && paths.every((p) => p === "relay") && (!ans || ans.ok), joinS, err, code, paths, debug: dbg,
        turn: Object.fromEntries(Object.entries(turn.stats).map(([k, v]) => [k, v - (before[k] || 0)])), answer: ans ? a : null });
      // back to the join screen for the next scenario
      for (const n of names) await tabs[n].goto("about:blank");
    }
    turn.close();
  } else if (PLAN === "blip") {
    // signaling drops for a few seconds and comes back (a wifi hiccup, a PeerJS cloud restart).
    // No model needed. Does the room stay joinable, and does each device get its registration back?
    const code = await setupRoom(names);
    const H = names[0];
    for (const [i, secs] of [5, 30].entries()) {
      const nm = `sig-blip${secs}s`;
      if (!want(nm)) continue;
      const w = watch(); const ts = Date.now();
      for (const n of names) await sig[n].set("down");
      await sleep(secs * 1000);
      for (const n of names) await sig[n].set("pass");
      await sleep(10000);
      const scr = {}; for (const n of names) scr[n] = await snap(tabs[n]);
      const N = "late" + i;
      await launch(N, names.length + i);
      await tabs[N].goto(url(N)); await tabs[N].waitForFunction(() => document.getElementById("join-gb").value !== "");
      await tabs[N].fill("#name-input", N); await tabs[N].fill("#code-input", code);
      const tj = Date.now();
      await tabs[N].click("#join-btn");
      const ms = await until(N, (s) => s.inRoom || /error|fail|no room|could not/i.test(s.join), 45000, tj);
      await sleep(1500); w.stop();
      row({ scenario: nm, what: `signaling down ${secs} s for every device, then back; a new device joins 10 s later`, screensAfter: scr,
        lateJoinMs: ms, lateScreen: await snap(tabs[N]), hostCards: (await snap(tabs[H])).cards, sinceBlipS: +((Date.now() - ts) / 1000).toFixed(1), timeline: w.tl });
      await tabs[N].close(); delete tabs[N];
    }
  } else {
    const code = await setupRoom(names);
    log("room", code, "with", names.join(", "));
    const split = NOMODEL ? "" : await loadModel(names);
    out.split = split;
    const H = names[0], G1 = names[1], GL = names[names.length - 1];
    const base = await ask(H); baselineText = base.text;
    row({ scenario: "baseline", ...base });
    if (PLAN === "net") {
      for (const rtt of [50, 150, 300]) if (want("rtt" + rtt)) { C.rtt = rtt; C.loss = 0; const r = await ask(H); row({ scenario: `rtt${rtt}`, rtt, ...r, sameAsBaseline: same(r.text) }); }
      for (const loss of (arg("losses", "0.01,0.05")).split(",").map(Number)) if (want("loss" + Math.round(loss * 100))) { C.rtt = 50; C.loss = loss; const r = await ask(H); row({ scenario: `loss${Math.round(loss * 100)}pct_rtt50`, rtt: 50, loss, ...r, sameAsBaseline: same(r.text), relay: { ...stats } }); }
      C.loss = 0;
      for (const [fz, who] of [[2000, G1], [2000, "all"], [5000, G1], [12000, G1], [8000, "all"], [20000, G1]]) {
        const nm = `freeze${fz / 1000}s_${who}`;
        if (!want(nm)) continue;
        C.rtt = 50;
        const r = await ask(H, { during: async (ts) => {
          await tabs[H].waitForFunction(() => ([...document.querySelectorAll(".m.bot .bubble")].pop()?.textContent || "").length > 40, null, { timeout: 60000, polling: 50 });
          const at = Date.now();
          if (who === "all") C.freezeAll = at + fz; else C.frozen.set(who, at + fz);
          return { freezeAtS: +((at - ts) / 1000).toFixed(1), ms: fz };
        } });
        await sleep(3000);
        const ice = {}; for (const n of names) ice[n] = (await iceLog(n)).filter(([, , s]) => s !== "checking" && s !== "new");
        const toasts = {}; for (const n of names) toasts[n] = await toastLog(n);
        row({ scenario: nm, ...r, sameAsBaseline: same(r.text), ice, toasts, ...(r.ok ? {} : { diag: await diag() }) });
        // the room still answers afterwards?
        if (!r.ok) { const again = await ask(H).catch((e) => ({ err: String(e) })); row({ scenario: nm + "_next", ...again, sameAsBaseline: same(again.text), ...(again.ok ? {} : { diag: await diag() }) }); }
      }
      C.rtt = 0;
    }
    if (PLAN === "death") {
      // 1) idle: the last guest's network dies silently, nobody is answering
      if (want("death-idle")) {
        const w = watch(); const ts = Date.now();
        C.dead.add(GL); await sig[GL].set("blackhole");
        const hostSees = await until(H, (s) => s.cards < names.length || !!s.redeal, +arg("deathwait", 120000), ts);
        const others = {}; for (const n of names.slice(1, -1)) others[n] = await until(n, (s) => s.cards < names.length, 5000);
        const deadSees = await until(GL, (s) => !!s.over || /left|lost/i.test(s.status), 60000, ts);
        await sleep(2000); w.stop();
        const scr = {}; for (const n of names) scr[n] = await snap(tabs[n]);
        const ice = {}; for (const n of names) ice[n] = (await iceLog(n)).map(([t, id, s]) => [+((t - ts) / 1000).toFixed(1), id, s]).filter(([, , s]) => s !== "checking" && s !== "new" && s !== "connected" || true);
        row({ scenario: "death-idle", what: `${GL}'s network dies silently while the room is idle`, hostDetectMs: hostSees, deadGuestNoticesMs: deadSees, screens: scr, ice, timeline: w.tl });
        // recovery: re-deal over the devices still here
        const redeal = await tabs[H].evaluate(() => { const b = document.getElementById("ai-redeal"); if (b && !b.hidden) { b.click(); return true; } return false; });
        let rec = null;
        if (redeal) { const tr = Date.now(); await tabs[H].waitForFunction(() => document.getElementById("ai-panel").classList.contains("online") && !/re-deal|left/.test(document.getElementById("ai-status").textContent), null, { timeout: 300000, polling: 500 }).catch(() => {}); rec = { redealS: +((Date.now() - tr) / 1000).toFixed(1) }; }
        names.pop(); delete tabs[GL];   // out of the room for the rest of the plan (its browser stays open)
        const next = await ask(H).catch((e) => ({ err: String(e) }));
        row({ scenario: "death-idle_recover", redealClicked: redeal, ...rec, ...next });
      }
      // 2) mid-answer: the (now) last guest's network dies with an answer in flight
      if (want("death-mid") && names.length >= 2) {
        const V = names[names.length - 1];
        const w = watch();
        const r = await ask(H, { timeoutMs: 180000, during: async (ts) => {
          await tabs[H].waitForFunction(() => ([...document.querySelectorAll(".m.bot .bubble")].pop()?.textContent || "").length > 40, null, { timeout: 60000, polling: 50 });
          const at = Date.now(); C.dead.add(V); await sig[V].set("blackhole");
          return { deathAtS: +((at - ts) / 1000).toFixed(1) };
        } });
        const t1 = Date.now();
        const hostDropsCard = await until(H, (s) => s.cards < names.length, 90000);
        w.stop();
        const scr = {}; for (const n of names) scr[n] = await snap(tabs[n]);
        const toasts = {}; for (const n of names) toasts[n] = await toastLog(n);
        row({ scenario: "death-mid", what: `${V}'s network dies silently mid-answer`, ...r, hostDropsCardMsAfterAnswerEnd: hostDropsCard, screens: scr, toasts, timeline: w.tl });
      }
    }
    if (PLAN === "signal") {
      // signaling goes away mid-answer (listener closed, sockets reset) for every device
      if (want("sig-down-mid")) {
        const w = watch();
        const r = await ask(H, { during: async (ts) => {
          await tabs[H].waitForFunction(() => ([...document.querySelectorAll(".m.bot .bubble")].pop()?.textContent || "").length > 40, null, { timeout: 60000, polling: 50 });
          const at = Date.now(); for (const n of names) await sig[n].set("down");
          return { downAtS: +((at - ts) / 1000).toFixed(1) };
        } });
        await sleep(3000); w.stop();
        const scr = {}; for (const n of names) scr[n] = await snap(tabs[n]);
        row({ scenario: "sig-down-mid", ...r, sameAsBaseline: same(r.text), screens: scr, timeline: w.tl });
        const r2 = await ask(H);
        row({ scenario: "sig-down_next-answer", ...r2, sameAsBaseline: same(r2.text) });
      }
      // a newcomer while signaling is down, then after it is back
      const N = "late";
      await launch(N, names.length);
      for (const [mode, label] of [["down", "while signaling is down"], ["pass", "after signaling is back"]]) {
        if (mode === "pass") for (const n of Object.keys(sig)) await sig[n].set("pass");
        await sig[N].set(mode === "down" ? "down" : "pass");
        await tabs[N].goto(url(N)); await tabs[N].waitForFunction(() => document.getElementById("join-gb").value !== "");
        await tabs[N].fill("#name-input", N); await tabs[N].fill("#code-input", code);
        const ts = Date.now();
        await tabs[N].click("#join-btn");
        const ms = await until(N, (s) => s.inRoom || /error|fail|no room|could not/i.test(s.join), 45000, ts);
        await sleep(1500);
        row({ scenario: "sig-late-join-" + mode, what: "a new device joins " + label, detectMs: ms, screen: await snap(tabs[N]), hostCards: (await snap(tabs[H])).cards });
      }
      // does the host's PeerJS registration come back on its own?
      const hostPeer = await tabs[H].evaluate(() => document.getElementById("join-status").textContent);
      row({ scenario: "sig-host-after", hostJoinStatus: hostPeer });
      // the chain needs new links after a re-deal: kill signaling, drop a guest, re-deal
      if (want("sig-down-redeal") && names.length >= 3) {
        for (const n of Object.keys(sig)) await sig[n].set("down");
        const V = names[names.length - 1];
        await tabs[V].close(); delete tabs[V]; names.pop();
        const gone = await until(H, (s) => !!s.redeal, 60000);
        await tabs[H].evaluate(() => document.getElementById("ai-redeal").click());
        const tr = Date.now();
        await tabs[H].waitForFunction(() => document.getElementById("ai-panel").classList.contains("online") && !/re-deal|left/.test(document.getElementById("ai-status").textContent) || /fail/i.test(document.getElementById("ai-status").textContent), null, { timeout: 180000, polling: 500 }).catch(() => {});
        const r = await ask(H, { timeoutMs: 150000 }).catch((e) => ({ err: String(e) }));
        row({ scenario: "sig-down-redeal", what: "signaling down, a device closes its tab, host re-deals", redealShownMs: gone, redealS: +((Date.now() - tr) / 1000).toFixed(1), ...r });
      }
    }
  }
} catch (e) {
  out.error = String(e.stack || e).slice(0, 600);
  log("FAILED", out.error);
} finally {
  out.errors = errs; out.relay = stats;
  console.log(JSON.stringify(out));
  peerServer.kill(); fs.rmSync(tlsDir, { recursive: true, force: true });
  await Promise.race([Promise.all(contexts.map((c) => c.close().catch(() => {}))), sleep(15000)]);
  for (const d of profiles) fs.rmSync(d, { recursive: true, force: true });
  process.exit(out.error ? 1 : 0);
}
