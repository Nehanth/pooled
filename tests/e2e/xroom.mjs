// Two-machine room: one headless browser per machine, each serving its own checkout on 127.0.0.1 and
// loading its layers from its own disk (?peerweights=0), PeerJS signaling on the host machine, a direct
// WebRTC link between the machines (host candidates; mDNS hiding off). Manual trigger only: it needs
// two machines with GPUs. tests/e2e/xroom_pair.sh drives both ends from one of them.
//
//   host : node tests/e2e/xroom.mjs --role host [--model qwen3.6-35b-moe|qwen3.8-27b] [--gb 13]
//          [--rounds 2] [--maxnew 128] [--modes plain,spec] [--prompts japan,twosum]
//          [--trace-rounds 2] [--out result.json] [--solo]
//   guest: node tests/e2e/xroom.mjs --role guest --signal <host ip>:9000 --code ABCD|--codefile f [--gb 12]
//          [--trace-out guest-trace.json]
//   both : [--port 8123] (http; https weights on port + 1) [--signal-port 9000] [--query "a=1&b=2"]
//          [--name gb10] [--chrome <path>] (macOS defaults to /Applications/Google Chrome.app)
//   host : [--signal <ip>:<port> --signal-server 0] when the signaling server runs elsewhere
//   both : [--signal cloud] the public PeerJS server (the page's default; rooms with a phone, see
//          tests/e2e/xroom_phone.mjs); host: [--peers N] wait for N devices, itself included (default 2)
//   diagnostics (host): [--fixk K] every draft-head step drafts K (the room otherwise picks 3/5/7 by
//          measured tok/s); [--tune WG,ROWS] forces the GEMV shape the load-time autotune would pick
//
// The host prints "CODE XXXX" (and writes it to --codefile) once the room exists, waits for the
// guest, loads the model, then for every mode, prompt and round: new chat, ask, record prefill,
// decode tok/s, acceptance, TTFT, the ping RTT on the peer card and the whole answer (and its
// sha-256), then prints one JSON line (also the autotune pick and the selected ICE candidate pair).
// Every round uses the "exact" (greedy) preset, so within a run every answer to one prompt must be
// identical, plain or speculative. (Across GPU vendors, and against a --solo run, the f16 wire and
// different kernels' rounding can move a close argmax: compare with care.)
// --solo: no guest; the host holds every layer (the one-device reference for the same page path).
//
// Both ends must serve the same commit. --root defaults to the checkout this file is in.
//
// Tracing (--trace-rounds 2,5: round indices, counted over every mode x prompt x round in order):
// both ends serve room.js and room/transport.js with the trace marks of tests/e2e/room_trace.mjs
// (host and guest alike, so the guest must be started with --trace-out), and the host switches
// tracing on in both pages for exactly those rounds (the guest learns it from a dev-only room
// message the patched room.js handles). The host's result then carries its traces per traced
// round; the guest rewrites its whole trace to --trace-out after every traced round (it marks each
// round's start, "x-trace" with the round index). tests/e2e/xroom_report.mjs
// aligns the two clocks from the frames themselves and splits every lap.
// Rounds that are not traced run the same served code with tracing off (no GPU timestamps, no marks),
// so their tok/s is the clean number; compare the traced round's tok/s to see what tracing costs.
import { chromium } from "playwright";
import http from "http";
import https from "https";
import fs from "fs";
import os from "os";
import path from "path";
import crypto from "crypto";
import { spawn, execSync } from "child_process";
import { patchRoom, patchTransport, INIT, clockOffset } from "./room_trace.mjs";

const arg = (k, d) => { const i = process.argv.indexOf("--" + k); return i > 0 ? process.argv[i + 1] : d; };
const flag = (k) => process.argv.includes("--" + k);
const ROLE = arg("role", "host"), SOLO = flag("solo");
const ROOT = path.resolve(arg("root", path.join(path.dirname(new URL(import.meta.url).pathname), "../..")));
const MODEL = arg("model", "qwen3.6-35b-moe");
const ROUNDS = +arg("rounds", 2), MAXNEW = +arg("maxnew", 128), GBV = arg("gb", SOLO ? "40" : ROLE === "host" ? "13" : "12");
const MODES = arg("modes", "plain,spec").split(",");
const PORT = +arg("port", 8123), TLS_PORT = PORT + 1, SIG_PORT = +arg("signal-port", 9000);
const SIGNAL = arg("signal", `127.0.0.1:${SIG_PORT}`), MAXMIN = +arg("maxmin", 60);
// --signal cloud: the page's default signaling (the public PeerJS server), what an https page on a
// phone uses (it cannot reach a plain ws:// server): every device in the room must use the same one
const CLOUD = SIGNAL === "cloud";
const PEERS = +arg("peers", 2);   // host: wait for this many devices in the room (itself included)
const TRACE_ROUNDS = new Set(String(arg("trace-rounds", "")).split(",").filter((x) => x !== "").map(Number));
const TRACE = ROLE === "host" ? TRACE_ROUNDS.size > 0 : !!arg("trace-out");
const QUERY = arg("query", "");   // extra room URL parameters, "a=1&b=2"
const FIXK = Math.max(0, Math.min(7, parseInt(arg("fixk", "0"), 10) || 0));
const TUNE = arg("tune") ? arg("tune").split(",").map((x) => parseInt(x, 10)) : null;   // e.g. 64,4
if (TUNE && !(TUNE.length === 2 && [64, 128, 256].includes(TUNE[0]) && [4, 8].includes(TUNE[1]))) throw new Error("--tune WG,ROWS with WG 64|128|256 and ROWS 4|8");
const PROMPTS = {
  // docs/bench-log.md "Standard prompts"
  japan: "I am planning a two week trip through Japan in late October with my partner. We land in Tokyo, want three days there, then a day trip to Nikko, then the bullet train to Kyoto for four days with a side trip to Nara, then two nights in Osaka, and we fly home from Osaka. We like food markets, old temples, hiking, and small neighborhood bars, and we want to avoid the most crowded tourist spots where we can. Our budget is moderate, around two hundred dollars a day for the two of us not counting hotels. Please give me a day by day itinerary with one main activity each morning and afternoon, a neighborhood to eat dinner in each night, and tell me which days I should buy a rail pass for and whether it is worth it at all.",
  // tests/e2e/room_prof.mjs's default: code, where drafts are accepted far more often
  twosum: "Write the Python code for two sum. Code only.",
};
const PROMPT_NAMES = arg("prompts", "japan").split(",");
for (const n of PROMPT_NAMES) if (!PROMPTS[n]) throw new Error("unknown prompt " + n + " (have " + Object.keys(PROMPTS).join(", ") + ")");
const LOCAL = { "Qwen3.8-27B-Q4_0.gguf": "models/q38/model.gguf", "Qwen3-0.6B-Q8_0.gguf": "models/qwen/model.gguf", "Qwen3-1.7B-Q8_0.gguf": "models/qwen17/model.gguf", "Qwen_Qwen3.6-35B-A3B-Q4_0.gguf": "models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf" };
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png" };

// Served room.js: window.__nospec (plain one-token-per-lap decode, as tests/e2e/room_latency.mjs),
// and with tracing the room_trace marks plus two dev-only hooks: window.__xBroadcast sends a message
// to every peer, and an "x-trace" message switches the receiving page's tracing on or off (off also
// resolves its GPU timestamps, so a long guest trace never runs out of query slots).
function rep(src, a, b) { if (!src.includes(a)) throw new Error("room.js changed: anchor not found: " + a.slice(0, 80)); return src.replace(a, b); }
// window.__xConns: the room's links, for the WebRTC path the result reports (xroom's linkStats).
function serveRoom(src) {
  src = rep(src, "function broadcastAll(obj) {", "window.__xBroadcast = (o) => broadcastAll(o); window.__xConns = () => conns;\n" +
    // the workers' own compute ms per frame kind (their ai-tele reports: an EMA of unpack + layers +
    // readback, "one" = a plain token, "spec" = a verify block, "pre" = prefill), by device name
    "window.__xTele = () => Object.fromEntries([...(ai.teleBy || new Map())].map(([id, v]) => [conns.get(id)?.name || id, v]));\n" +
    "function broadcastAll(obj) {");
  // the cooperative GEMV shape this device's autotune picked (timed at load, so it can differ
  // between loads, and the GEMV's summation order follows it)
  // (--tune WG,ROWS forces a shape instead: the diagnostic for whether the shape changes the output)
  src = rep(src, "  ai.tune = await autotuneCoop(ai.device).catch(() => ({ wg: 256, rows: 4 }));", TUNE
    ? `  ai.tune = { wg: ${TUNE[0]}, rows: ${TUNE[1]}, forced: 1 }; window.__xTune = ai.tune;`
    : "  ai.tune = await autotuneCoop(ai.device).catch(() => ({ wg: 256, rows: 4 })); window.__xTune = ai.tune;");
  // --fixk K: every draft-head step in a room drafts K (the room otherwise picks 3, 5 or 7 by
  // measured tok/s, which depends on timing); the diagnostic for acceptance against a solo run (K = 3)
  if (FIXK) src = rep(src, "      const pickK = () => {\n", `      const pickK = () => { if (ai.chain.length) return ${FIXK};\n`);
  if (TRACE) {
    src = patchRoom(src);   // includes the __nospec switch
    return rep(src, "function onData(from, d) {", "function onData(from, d) {\n  if (d && d.t === \"x-trace\") { window.__hpOn = !!d.on; if (d.on) window.__hpMark?.(\"x-trace\", \"on\", d.idx); else window.__gpResolve?.(); return; }");
  }
  return rep(src, "else if (ai.engine.mtp && ai.engine.specStep) {", "else if (ai.engine.mtp && ai.engine.specStep && !window.__nospec) {");
}
// the selected ICE candidate pair of every room link and its wire stripes, with Chrome's own STUN
// round trip on it (currentRoundTripTime: the network with no application in the way)
const linkStats = () => p.evaluate(async () => {
  const out = [];
  for (const [id, e] of window.__xConns?.() || []) {
    for (const [k, c] of [["main", e.conn], ...(e.stripes || []).map((c, i) => ["stripe" + (i + 1), c])]) {
      const pc = c?.peerConnection; if (!pc) continue;
      const st = await pc.getStats(); let pair = null; const cand = {}; const chans = [];
      st.forEach((r) => { if (r.type === "candidate-pair" && r.nominated && r.state === "succeeded") pair = r; if (/candidate$/.test(r.type)) cand[r.id] = r; if (r.type === "data-channel") chans.push({ label: r.label, id: r.dataChannelIdentifier, sent: r.messagesSent, recv: r.messagesReceived, bytesSent: r.bytesSent }); });
      const l = pair && cand[pair.localCandidateId], r = pair && cand[pair.remoteCandidateId];
      out.push({ peer: e.name || id, link: k, local: l && `${l.address} ${l.candidateType} ${l.protocol}`, remote: r && `${r.address} ${r.candidateType}`,
        stunRttMs: pair ? Math.round(pair.currentRoundTripTime * 10000) / 10 : null, stunRttAvgMs: pair && pair.responsesReceived ? Math.round(pair.totalRoundTripTime / pair.responsesReceived * 10000) / 10 : null, chans });
    }
  }
  return out;
});
const SERVED = { [path.join(ROOT, "room.js")]: serveRoom, ...(TRACE ? { [path.join(ROOT, "room/transport.js")]: patchTransport } : {}) };
for (const [p, fn] of Object.entries(SERVED)) fn(fs.readFileSync(p, "utf8"));   // fail before any browser starts

const srv = http.createServer((q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { r.statusCode = 404; r.end(); return; }
  r.setHeader("content-type", MIME[path.extname(p)] || "application/octet-stream");
  if (SERVED[p]) { r.end(SERVED[p](fs.readFileSync(p, "utf8"))); return; }
  fs.createReadStream(p).pipe(r);
}).listen(PORT, "127.0.0.1");
// the weights over https with Range support (the room refuses anything else), from local disk
const tlsDir = fs.mkdtempSync(path.join(os.tmpdir(), "xroom-"));
execSync(`openssl req -x509 -newkey rsa:2048 -nodes -keyout ${tlsDir}/k.pem -out ${tlsDir}/c.pem -days 2 -subj /CN=127.0.0.1 2>/dev/null`);
const wsrv = https.createServer({ key: fs.readFileSync(`${tlsDir}/k.pem`), cert: fs.readFileSync(`${tlsDir}/c.pem`) }, (q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p)) { r.statusCode = 404; r.end(); return; }
  const size = fs.statSync(p).size, m = /bytes=(\d+)-(\d*)/.exec(q.headers.range || "");
  const lo = m ? +m[1] : 0, hi = m && m[2] ? Math.min(+m[2], size - 1) : size - 1;
  r.writeHead(m ? 206 : 200, { "content-type": "application/octet-stream", "content-range": `bytes ${lo}-${hi}/${size}`, "accept-ranges": "bytes", "content-length": String(hi - lo + 1), "access-control-allow-origin": "*", "access-control-expose-headers": "content-range, content-length, accept-ranges" });
  fs.createReadStream(p, { start: lo, end: hi }).pipe(r);
}).listen(TLS_PORT, "127.0.0.1");
// the signaling server listens on every interface: the guest machine reaches it over the LAN / tailnet
// (--signal-server 0: someone else runs it, e.g. xroom_pair.sh when the host is the other machine)
let peerServer = null;
if (ROLE === "host" && !CLOUD && arg("signal-server", "1") !== "0") {   // a solo room still registers its code
  peerServer = spawn(arg("peerjs", path.join(ROOT, "node_modules/.bin/peerjs")), ["--port", String(SIG_PORT), "--path", "/", "--host", "0.0.0.0"], { stdio: "ignore" });
  await new Promise((r) => setTimeout(r, 1500));
}
const BASE = `http://127.0.0.1:${PORT}/p2p.html?${CLOUD ? "" : `signal=${SIGNAL}&`}maxnew=${MAXNEW}&peerweights=0&dev=1` + (QUERY ? "&" + QUERY : "");
const mac = process.platform === "darwin";
const ARGS = [...(mac ? [] : ["--no-sandbox", "--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan"]),
  "--headless=new", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--disable-features=WebRtcHideLocalIpsWithMdns", "--js-flags=--max-old-space-size=65536",
  ...(TRACE ? ["--enable-webgpu-developer-features"] : [])];   // unquantized GPU timestamps
const UA = mac ? "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36" : "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";
const CHROME = arg("chrome", mac ? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" : "");
const prof = fs.mkdtempSync(path.join(os.tmpdir(), "xroom-profile-"));
const ctx = await chromium.launchPersistentContext(prof, { headless: false, args: ARGS, userAgent: UA, ignoreHTTPSErrors: true, ...(CHROME ? { executablePath: CHROME } : {}) });
if (TRACE) await ctx.addInitScript(INIT);
await ctx.route("**/*.gguf", (route) => {
  const file = LOCAL[route.request().url().split("/").pop().split("?")[0]];
  if (!file || !fs.existsSync(path.join(ROOT, file))) return route.continue();
  return route.continue({ url: `https://127.0.0.1:${TLS_PORT}/${file}` });
});
const p = ctx.pages()[0] || await ctx.newPage();
const errs = [];
p.on("pageerror", (e) => errs.push(String(e).slice(0, 200)));
p.on("crash", () => { errs.push("crashed"); console.error("CRASHED"); });
p.on("console", (m) => { if (m.type() === "error" || /GPU|lost|memory|webrtc|ice/i.test(m.text())) console.error("console:", m.text().slice(0, 200)); });
const t0 = Date.now(); const log = (...a) => console.error(((Date.now() - t0) / 1000).toFixed(0) + "s", ...a);
const snap = () => p.evaluate(() => ({ status: document.getElementById("ai-status")?.textContent, sub: document.getElementById("ldg-sub")?.textContent, last: [...document.querySelectorAll("#chat-log div")].slice(-1)[0]?.textContent,
  peers: [...document.querySelectorAll(".peer-card")].map((c) => c.textContent.replace(/\s+/g, " ").trim().slice(0, 160)), rtt: [...document.querySelectorAll(".peer-card .rtt")].map((e) => e.textContent) }));
// the page's trace, shifted onto this process's clock (Date.now() based: the machines' system clocks;
// xroom_report refines the guest's offset from the frames)
async function takeTrace(reset) {
  await p.evaluate(() => window.__gpResolve?.());
  const c = await clockOffset(p);
  const tr = await p.evaluate((r) => { const t = { hp: window.__hp, gp: window.__gp }; if (r) { window.__hp = []; window.__gp = { subs: [], maps: [] }; } return t; }, reset);
  for (const e of tr.hp) e[3] -= c.off;
  for (const x of tr.gp.subs) { x.t -= c.off; x.tEnc -= c.off; }
  for (const m of tr.gp.maps) { m.t0 -= c.off; if (m.t1) m.t1 -= c.off; }
  return { ...tr, clock: c };
}
async function writeGuestTrace() {
  try { fs.writeFileSync(arg("trace-out"), JSON.stringify({ role: "guest", platform: process.platform, trace: await takeTrace(false) })); log("trace written to", arg("trace-out")); }
  catch (e) { log("trace not written:", String(e).slice(0, 200)); }
}
const out = { role: ROLE, model: MODEL, solo: SOLO, platform: process.platform, rows: [], traces: [] };
let finished = false;
const finish = async (code) => {
  if (finished) return; finished = true;
  if (ROLE === "guest" && TRACE) await writeGuestTrace();
  out.errors = errs;
  const line = JSON.stringify(out);
  if (arg("out")) fs.writeFileSync(arg("out"), line);
  console.log(ROLE === "host" && TRACE ? JSON.stringify({ ...out, traces: `${out.traces.length} traced rounds in ${arg("out") || "(no --out)"}` }) : line);
  peerServer?.kill();
  await Promise.race([ctx.close().catch(() => {}), new Promise((r) => setTimeout(r, 15000))]);
  fs.rmSync(prof, { recursive: true, force: true }); fs.rmSync(tlsDir, { recursive: true, force: true });
  srv.close(); wsrv.close();
  process.exit(code);
};
process.on("SIGTERM", () => finish(1));
process.on("SIGINT", () => finish(1));
try {
  await p.goto(BASE);
  await p.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
  // the page's GPU probe overwrites join-gb with its own suggestion when it finishes: let it, then pledge
  await p.waitForFunction(() => document.getElementById("join-gb").value !== "1", null, { timeout: 45000 }).catch(() => {});
  await p.waitForTimeout(1000);
  await p.fill("#name-input", arg("name", mac ? "m5max" : "gb10")); await p.fill("#join-gb", GBV);
  if (ROLE === "guest") {
    // --code, or (--codefile) wait for whoever starts the host to write the code there: a guest can
    // then be started, and hold its GPU, before the host exists
    let code = arg("code");
    if (!code) log("waiting for the room code in", arg("codefile"));
    for (const tEnd = Date.now() + 30 * 60e3; !code && Date.now() < tEnd; await p.waitForTimeout(1000)) {
      try { code = (fs.readFileSync(arg("codefile"), "utf8").match(/[A-Z0-9]{4}/) || [])[0]; } catch {}
      if (!code && !arg("codefile")) throw new Error("--code or --codefile");
    }
    if (!code) throw new Error("no room code in " + arg("codefile"));
    await p.fill("#code-input", code); await p.click("#join-btn");
    await p.waitForFunction(() => document.querySelectorAll(".peer-card").length >= 2, null, { timeout: 120000 });
    log("joined", JSON.stringify(await snap()));
    p.waitForFunction(() => window.__xTune, null, { timeout: 40 * 60e3, polling: 1000 }).then(async () => log("autotune", JSON.stringify(await p.evaluate(() => window.__xTune)))).catch(() => {});
    // serve until the host leaves; with tracing, write the trace after every traced round (the host
    // switches tracing off when the round ends), so the file is complete however this process ends
    const end = Date.now() + MAXMIN * 60e3; let seen = 0, lastLog = 0, wasOn = false, marks = 0;
    while (Date.now() < end) {
      await p.waitForTimeout(2000);
      const s = await snap();
      if (Date.now() - lastLog > 20000) { lastLog = Date.now(); log(JSON.stringify(s).slice(0, 400)); }
      if (TRACE) {
        const on = await p.evaluate(() => [!!window.__hpOn, window.__hp?.length || 0]);
        if (wasOn && !on[0] && on[1] !== marks) { marks = on[1]; await writeGuestTrace(); }
        wasOn = on[0];
      }
      // the host's tab closed: the room says so (it waits a minute for a host that reloads)
      if (/host left/.test(s.status || "")) break;
      if (s.peers.length >= 2) seen = Date.now(); else if (seen && Date.now() - seen > 6000) break;
    }
    await finish(0);
  }
  await p.click("#create-btn");
  await p.waitForFunction(() => /[A-Z0-9]{4}/.test(document.getElementById("side-code").textContent), null, { timeout: 30000 });
  const code = (await p.textContent("#side-code")).trim().match(/[A-Z0-9]{4}/)[0];
  if (!SOLO) {
    console.log("CODE " + code);
    if (arg("codefile")) fs.writeFileSync(arg("codefile"), code);
    await p.waitForFunction((n) => document.querySelectorAll(".peer-card").length >= n, PEERS, { timeout: 15 * 60e3, polling: 1000 });
    log("guest in", JSON.stringify(await snap()));
  }
  await p.waitForTimeout(3000);
  await p.selectOption("#ai-model", MODEL);
  const tLoad = Date.now();
  await p.click("#ai-start");
  const poll = setInterval(async () => { try { log(JSON.stringify(await snap()).slice(0, 400)); } catch {} }, 20000);
  await p.waitForFunction(() => document.getElementById("ai-panel").classList.contains("online") || /^failed:/.test(document.getElementById("ai-status").textContent), null, { timeout: 40 * 60e3, polling: 2000 });
  clearInterval(poll);
  const st0 = await p.textContent("#ai-status");
  if (/^failed:/.test(st0)) throw new Error("load " + st0);
  out.loadS = Math.round((Date.now() - tLoad) / 1000);
  out.tune = await p.evaluate(() => window.__xTune || null);
  out.split = await p.evaluate(() => [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent).filter((t) => /layer split|solo:/.test(t)).slice(-1)[0] || document.getElementById("ai-status").textContent);
  log("online in", out.loadS, "s;", out.split);
  await p.evaluate(() => { const s = document.getElementById("ai-sampling"); s.value = "exact"; s.dispatchEvent(new Event("change")); });
  let idx = 0;
  for (const mode of MODES) {
    await p.evaluate((ns) => { window.__nospec = ns; }, mode === "plain");
    for (const pn of PROMPT_NAMES) for (let r = 0; r < ROUNDS; r++, idx++) {
      const traced = TRACE_ROUNDS.has(idx);
      await p.evaluate(() => document.getElementById("new-chat").click());
      await p.waitForTimeout(500);
      if (traced) {
        await p.evaluate((idx) => { window.__hp = []; window.__gp = { subs: [], maps: [] }; window.__hpOn = true; window.__xBroadcast?.({ t: "x-trace", on: 1, idx }); }, idx);
        await p.waitForTimeout(300);   // the guest switches on before the first frame
      }
      const tRound = Date.now();
      await p.evaluate((text) => {
        const box = document.getElementById("ai-prompt"); box.value = text; box.dispatchEvent(new Event("input"));
        window.__ttft = null; const t = performance.now();
        const ob = new MutationObserver(() => { const b = [...document.querySelectorAll(".m.bot .bubble")].pop(); if (b && b.textContent.trim()) { window.__ttft = performance.now() - t; ob.disconnect(); } });
        ob.observe(document.body, { subtree: true, childList: true, characterData: true });
        document.getElementById("ai-send").click();
      }, PROMPTS[pn]);
      await p.waitForFunction(() => /^ready — prefill|^generation failed/.test(document.getElementById("ai-status").textContent), null, { timeout: 900000, polling: 250 });
      const st = await p.textContent("#ai-status"), s = await snap();
      const ttft = await p.evaluate(() => window.__ttft);
      const pre = /prefill (\d+) tok in ([\d.]+)s/.exec(st), dec = /(\d+) tok · ([\d.]+) tok\/s/.exec(st), acc = /(\d+)% drafts accepted/.exec(st), lk = /(\d+) tok by lookup/.exec(st);
      const answer = await p.evaluate(() => [...document.querySelectorAll(".m.bot .bubble")].pop()?.textContent || "");
      const crumb = await p.evaluate(() => { try { return JSON.parse(localStorage.getItem("pooled-crumb") || "{}").s || ""; } catch { return ""; } });
      const tele = await p.evaluate(() => window.__xTele?.() || null).catch(() => null);
      const row = { idx, mode, prompt: pn, round: r, traced, tele, t0: tRound, t1: Date.now(), ttftMs: ttft && Math.round(ttft), prefillTok: pre && +pre[1], prefillS: pre && +pre[2], tokens: dec && +dec[1], tps: dec && +dec[2],
        accepted: acc ? +acc[1] / 100 : null, lookupTok: lk ? +lk[1] : 0, rtt: s.rtt, status: st.slice(0, 240), crumb: crumb.slice(0, 300),
        answerSha: crypto.createHash("sha256").update(answer).digest("hex").slice(0, 16), answer };
      if (traced) {
        await p.evaluate(() => { window.__hpOn = false; window.__xBroadcast?.({ t: "x-trace", on: 0 }); });
        out.traces.push({ idx, mode, prompt: pn, round: r, status: st, reply: answer.slice(0, 160), crumb, t0: tRound, t1: Date.now(), host: await takeTrace(true) });
      }
      log(JSON.stringify({ ...row, answer: undefined })); out.rows.push(row);
    }
  }
  out.peers = (await snap()).peers;
  out.links = await linkStats().catch((e) => String(e).slice(0, 200));
} catch (e) { out.error = String(e).slice(0, 400); log("FAILED", out.error); try { out.snap = await snap(); } catch {} }
await finish(out.error ? 1 : 0);
