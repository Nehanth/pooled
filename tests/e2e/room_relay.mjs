// Rooms at work: a 2-tab room that can only connect through a TURN relay over TCP or TLS, with the
// relay's credentials minted by the real api/turn.mjs (coturn "use-auth-secret" mode), exactly as a
// deployment with TURN_SECRET/TURN_URLS set would hand them out. Manual trigger only.
//
//   node tests/e2e/room_relay.mjs --coturn /path/to/turnserver [--transport tcp|tls|all] [--nomodel]
//
// What makes it a "work network":
//   - Chromium runs with --force-webrtc-ip-handling-policy=disable_non_proxied_udp: WebRTC may not
//     send UDP at all (no UDP host candidates, no STUN), only TCP to a TURN server
//   - the page asks for relay-only (?relay=1 -> iceTransportPolicy "relay")
//   - the page gets its relay from POST /api/turn (?turnapi=1, since this is localhost), served here
//     by the same handler Vercel runs, against a local coturn (TCP 3478-style port, TLS port with a
//     self-signed cert for "localhost"). --transport picks which URLs the endpoint hands out.
// Checks: both tabs asked /api/turn and got 200; every link's selected pair is a relay candidate
// reached over TCP/TLS (pooledDebug().via); coturn logged authenticated TCP/TLS allocations and no UDP
// ones; the room log says the link goes through the relay; with a model (qwen3-1.7b by default,
// split across both tabs) two answers come back through the relay.
// coturn without root: `apt-get download coturn` plus its libraries, dpkg -x into a directory, and
// LD_LIBRARY_PATH to its usr/lib/<arch> (see docs/rooms-at-work.md, "Testing").
import { chromium } from "playwright";
import http from "http";
import https from "https";
import fs from "fs";
import os from "os";
import path from "path";
import { spawn, execSync } from "child_process";
import { handle } from "../../api/turn.mjs";

const arg = (k, d) => { const i = process.argv.indexOf("--" + k); return i > 0 ? process.argv[i + 1] : d; };
const flag = (k) => process.argv.includes("--" + k);
const COTURN = arg("coturn", process.env.COTURN || "turnserver");
const TRANSPORT = arg("transport", "all");
const MODEL = arg("model", "qwen3-1.7b"), ROUNDS = +arg("rounds", 2), NOMODEL = flag("nomodel");
const PORT = +arg("port", 8431), SIGNAL_PORT = PORT + 2, TLS_PORT = PORT + 1;
const TURN_PORT = +arg("turn-port", 34780), TURNS_PORT = TURN_PORT + 1;
const SECRET = "e2e-" + Math.random().toString(36).slice(2);
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");
const LOCAL = { "Qwen3-0.6B-Q8_0.gguf": "models/qwen/model.gguf", "Qwen3-1.7B-Q8_0.gguf": "models/qwen17/model.gguf" };
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png" };
const t0 = Date.now(); const log = (...a) => console.error(((Date.now() - t0) / 1000).toFixed(1) + "s", ...a);

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-relay-"));
execSync(`openssl req -x509 -newkey rsa:2048 -nodes -keyout ${dir}/k.pem -out ${dir}/c.pem -days 2 -subj /CN=localhost -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" 2>/dev/null`);

// the relay URLs the endpoint hands out
const URLS = {
  tcp: [`turn:127.0.0.1:${TURN_PORT}?transport=tcp`],
  tls: [`turns:localhost:${TURNS_PORT}?transport=tcp`],
  all: [`turn:127.0.0.1:${TURN_PORT}?transport=udp`, `turn:127.0.0.1:${TURN_PORT}?transport=tcp`, `turns:localhost:${TURNS_PORT}?transport=tcp`],
  none: [],   // the endpoint is not configured (204): the page must say why the join fails, not hang
}[TRANSPORT];
if (!URLS) throw new Error("--transport tcp|tls|all|none");
const NONE = TRANSPORT === "none";
const ENV = NONE ? {} : { TURN_SECRET: SECRET, TURN_URLS: URLS.join(","), TURN_TTL: "900" };

// coturn: REST-API secret auth, loopback only, verbose log to a file we read at the end
const turnLog = path.join(dir, "turn.log");
const turn = spawn(COTURN, ["-n", "--listening-ip=127.0.0.1", "--relay-ip=127.0.0.1", `--listening-port=${TURN_PORT}`, `--tls-listening-port=${TURNS_PORT}`,
  "--use-auth-secret", `--static-auth-secret=${SECRET}`, "--realm=pooled.test", `--cert=${dir}/c.pem`, `--pkey=${dir}/k.pem`,
  // tcp: no UDP listener; tls: only TLS (coturn detects TLS on either port); all: every listener,
  // so the page is offered UDP too and must not use it
  ...(TRANSPORT === "tcp" ? ["--no-udp"] : TRANSPORT === "tls" ? ["--no-udp", "--no-tcp"] : []),
  "--allow-loopback-peers", "--no-cli", "--no-dtls", "--min-port=51000", "--max-port=51200", "--verbose", `--log-file=${turnLog}`, "--simple-log", "--no-stdout-log"],
  { stdio: ["ignore", "ignore", "pipe"] });
let turnErr = ""; turn.stderr.on("data", (d) => { turnErr += d; });
turn.on("exit", (c) => { if (c) log("coturn exited", c, turnErr.slice(0, 300)); });

// static site + POST /api/turn through the real handler
const apiCalls = [];
const srv = http.createServer(async (q, r) => {
  const u = q.url.split("?")[0];
  if (u === "/api/turn") {
    const req = new Request(`http://${q.headers.host}${q.url}`, { method: q.method, headers: Object.fromEntries(Object.entries(q.headers).filter(([, v]) => typeof v === "string")) });
    const res = await handle(req, { env: ENV });
    apiCalls.push({ method: q.method, origin: q.headers.origin || null, status: res.status });
    r.writeHead(res.status, Object.fromEntries(res.headers)); r.end(Buffer.from(await res.arrayBuffer()));
    return;
  }
  const p = path.join(ROOT, decodeURIComponent(u));
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { r.statusCode = 404; r.end(); return; }
  r.setHeader("content-type", MIME[path.extname(p)] || "application/octet-stream"); fs.createReadStream(p).pipe(r);
}).listen(PORT, "127.0.0.1");
// model weights from disk in place of Hugging Face (https, so the request can be rewritten)
const wsrv = https.createServer({ key: fs.readFileSync(`${dir}/k.pem`), cert: fs.readFileSync(`${dir}/c.pem`) }, (q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p)) { r.statusCode = 404; r.end(); return; }
  const size = fs.statSync(p).size, m = /bytes=(\d+)-(\d*)/.exec(q.headers.range || "");
  const lo = m ? +m[1] : 0, hi = m && m[2] ? Math.min(+m[2], size - 1) : size - 1;
  r.writeHead(m ? 206 : 200, { "content-type": "application/octet-stream", "content-range": `bytes ${lo}-${hi}/${size}`, "accept-ranges": "bytes", "content-length": String(hi - lo + 1), "access-control-allow-origin": "*", "access-control-expose-headers": "content-range, content-length, accept-ranges" });
  fs.createReadStream(p, { start: lo, end: hi }).pipe(r);
}).listen(TLS_PORT, "127.0.0.1");
const peerServer = spawn(path.join(ROOT, "node_modules/.bin/peerjs"), ["--port", String(SIGNAL_PORT), "--path", "/"], { stdio: "ignore" });
await new Promise((r) => setTimeout(r, 1500));

const BASE = `http://127.0.0.1:${PORT}/p2p.html?split=memory&turnapi=1${NONE ? "" : "&relay=1"}&signal=127.0.0.1:${SIGNAL_PORT}`;
// --nomodel: no GPU at all (the tabs join as chat-only devices), so it needs no GPU queue
const GPU = NOMODEL ? ["--disable-gpu"] : ["--enable-unsafe-webgpu", "--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan", "--ignore-gpu-blocklist"];
const args = ["--no-sandbox", "--headless=new", ...GPU, "--allow-loopback-in-peer-connection", "--force-webrtc-ip-handling-policy=disable_non_proxied_udp", "--ignore-certificate-errors"];
// The command-line flag alone is not enough in full (non-shell) Chromium: the profile preference
// webrtc.ip_handling_policy overrides it, so the profile is written with the same policy.
const profile = path.join(dir, "profile");
fs.mkdirSync(path.join(profile, "Default"), { recursive: true });
fs.writeFileSync(path.join(profile, "Default", "Preferences"), JSON.stringify({ webrtc: { ip_handling_policy: "disable_non_proxied_udp" } }));
const ctx = await chromium.launchPersistentContext(profile, { headless: false, args, ignoreHTTPSErrors: true,
  userAgent: "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36" });
const browser = ctx;
await ctx.route("**/*.gguf", (route) => {
  const file = LOCAL[route.request().url().split("/").pop().split("?")[0]];
  if (!file || !fs.existsSync(path.join(ROOT, file))) return route.continue();
  return route.continue({ url: `https://127.0.0.1:${TLS_PORT}/${file}` });
});
const tabs = { host: ctx.pages()[0] || await ctx.newPage(), worker: await ctx.newPage() };
const errs = { host: [], worker: [] };
for (const [n, p] of Object.entries(tabs)) {
  p.on("pageerror", (e) => errs[n].push("pageerror: " + String(e).slice(0, 200)));
  p.on("console", (m) => { if (m.type() === "error") errs[n].push(m.text().slice(0, 200)); });
}
const roomLog = (p) => p.evaluate(() => [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent).filter((t) => /relay|UDP/.test(t)));
let ok = false;
try {
  for (const p of Object.values(tabs)) await p.goto(BASE);
  for (const p of Object.values(tabs)) await p.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
  await tabs.host.fill("#name-input", "host-relay"); await tabs.worker.fill("#name-input", "worker-relay");
  if (!NOMODEL) { await tabs.host.fill("#join-gb", "3"); await tabs.worker.fill("#join-gb", "1"); }   // chat-only tabs have no memory box
  await tabs.host.click("#create-btn");
  await tabs.host.waitForFunction(() => /[A-Z0-9]{4}/.test(document.getElementById("side-code").textContent), null, { timeout: 30000 });
  const code = (await tabs.host.textContent("#side-code")).trim().match(/[A-Z0-9]{4}/)[0];
  log("room", code);
  await tabs.worker.fill("#code-input", code); await tabs.worker.click("#join-btn");
  const tj = Date.now();
  await tabs.worker.waitForFunction(() => document.querySelectorAll(".peer-card").length >= 2 || /reach|connect/i.test(document.getElementById("join-status").textContent) && !document.getElementById("join-btn").disabled, null, { timeout: 90000 });
  const joinStatus = await tabs.worker.textContent("#join-status");
  const joined = (await tabs.worker.$$(".peer-card")).length >= 2;
  log(joined ? `joined in ${((Date.now() - tj) / 1000).toFixed(1)} s` : "join failed: " + joinStatus);
  if (NONE) {
    // no relay on a network without UDP: the join must end with the explanation and the docs link
    const net = await tabs.worker.evaluate(() => window.pooledNet?.());
    const docs = await tabs.worker.evaluate(() => [...document.querySelectorAll("#join-status a")].map((a) => a.href));
    const netOpen = await tabs.worker.evaluate(() => !!document.getElementById("join-net")?.open);
    const checks = {
      apiSaidNone: apiCalls.length >= 1 && apiCalls.every((c) => c.status === 204),
      joinFailedWithReason: !joined && /blocks direct \(UDP\) connections and there is no relay/.test(joinStatus),
      docsLinked: docs.some((h) => h.includes("rooms-at-work")),
      networkBoxOpened: netOpen,
      secondsToAnswer: +((Date.now() - tj) / 1000).toFixed(1),
    };
    ok = !joined && checks.apiSaidNone && checks.joinFailedWithReason && checks.docsLinked && checks.networkBoxOpened;
    console.log(JSON.stringify({ ok, transport: TRANSPORT, joined, joinStatus, net, checks, apiCalls, errors: errs }, null, 1));
    throw "done";
  }
  if (!joined) throw new Error("join failed: " + joinStatus);
  await tabs.host.waitForTimeout(4500);   // notePath reads the selected pair 3 s after a link opens
  const links = {}; for (const [n, p] of Object.entries(tabs)) links[n] = await p.evaluate(() => window.pooledDebug());
  log("links", JSON.stringify(links));
  const results = [];
  if (!NOMODEL) {
    await tabs.host.selectOption("#ai-model", MODEL);
    await tabs.host.click("#ai-start");
    log("model start pressed:", MODEL);
    for (const [n, p] of Object.entries(tabs)) await p.waitForFunction(() => document.getElementById("ai-panel").classList.contains("online") || /^failed:/.test(document.getElementById("ai-status").textContent), null, { timeout: n === "host" ? 900000 : 300000 });
    log("online:", await tabs.host.textContent("#ai-status"));
    for (let r = 0; r < ROUNDS; r++) {
      await tabs.host.fill("#ai-prompt", "Write three sentences about the ocean."); await tabs.host.click("#ai-send");
      await tabs.host.waitForFunction(() => /^ready — prefill|^generation failed/.test(document.getElementById("ai-status").textContent), null, { timeout: 300000 });
      const st = await tabs.host.textContent("#ai-status");
      const reply = await tabs.host.evaluate(() => { const b = document.querySelectorAll(".m.bot .bubble"); return (b[b.length - 1]?.textContent || "").slice(0, 200); });
      log("round", r, st); results.push({ status: st, reply });
      await tabs.host.waitForTimeout(800);
    }
  }
  const after = {}; for (const [n, p] of Object.entries(tabs)) after[n] = await p.evaluate(() => window.pooledDebug());
  const notes = {}; for (const [n, p] of Object.entries(tabs)) notes[n] = await roomLog(p);
  const net = {}; for (const [n, p] of Object.entries(tabs)) net[n] = await p.evaluate(() => window.pooledNet?.());
  // coturn's view: authenticated allocations, and over which transport the clients reached it
  await new Promise((r) => setTimeout(r, 500));
  const tl = fs.existsSync(turnLog) ? fs.readFileSync(turnLog, "utf8") : "";
  const lines = tl.split("\n");
  const turnStats = {
    sessions: lines.filter((l) => /: new, realm=<[^>]*>, username=<[^>]+>/.test(l)).length,
    allocations: lines.filter((l) => /user <[^>]+>: incoming packet ALLOCATE processed, success/.test(l)).length,
    channelBinds: lines.filter((l) => /CHANNEL_BIND processed, success/.test(l)).length,
    tcpOrTlsConnections: lines.filter((l) => /tcp or tls connected to/.test(l)).length,
    // the first request of every allocation is challenged (401 with no user): only a 401 for a named user is a bad credential
    badCredentials: lines.filter((l) => /user <[^>]+>.*error 401/.test(l)).length,
    sample: lines.filter((l) => /ALLOCATE processed, success/.test(l)).slice(0, 2).map((l) => l.slice(0, 200)),
  };
  const allLinks = Object.values(after).flat();
  const vias = [...new Set(allLinks.map((l) => l.via))];
  const want = TRANSPORT === "tls" ? ["tls"] : TRANSPORT === "tcp" ? ["tcp"] : ["tcp", "tls"];
  const checks = {
    apiCalled: apiCalls.filter((c) => c.method === "POST" && c.status === 200).length >= 2,
    everyLinkRelayed: allLinks.length >= 2 && allLinks.every((l) => l.path === "relay"),
    overTcpOrTls: vias.length > 0 && vias.every((v) => want.includes(v)),
    coturnAllocated: turnStats.allocations >= 2 && turnStats.tcpOrTlsConnections >= 2 && turnStats.badCredentials === 0,
    roomSaysRelay: Object.values(notes).every((n) => n.some((t) => /goes through the relay/.test(t))),
    // the page noticed UDP is blocked (no candidates from its STUN probe) and said the relay covers it
    udpBlockedSeen: Object.values(net).every((x) => x?.udp && x.udp.udpOut === false) && Object.values(notes).every((n) => n.some((t) => /blocks direct \(UDP\)/.test(t))),
    answers: NOMODEL || (results.length === ROUNDS && results.every((r) => r.status.startsWith("ready") && r.reply.length > 20)),
    noErrors: Object.values(errs).every((e) => e.length === 0),
  };
  ok = Object.values(checks).every(Boolean);
  console.log(JSON.stringify({ ok, transport: TRANSPORT, urls: URLS, model: NOMODEL ? null : MODEL, checks, net, apiCalls, vias, links: after, turn: turnStats, notes, results, errors: errs }, null, 1));
} catch (e) {
  if (e === "done") { /* the none plan reported above */ } else {
  console.error("FAILED:", String(e).slice(0, 400));
  try { console.error("api calls:", JSON.stringify(apiCalls)); for (const [n, p] of Object.entries(tabs)) console.error(n, JSON.stringify(await roomLog(p)), await p.textContent("#join-status")); } catch {}
  console.error("errors:", JSON.stringify(errs));
  if (fs.existsSync(turnLog)) console.error("coturn log tail:\n" + fs.readFileSync(turnLog, "utf8").split("\n").slice(-25).join("\n"));
  }
} finally {
  process.exitCode = ok ? 0 : 1;
  await browser.close().catch(() => {}); srv.close(); wsrv.close(); peerServer.kill(); turn.kill();
  if (!flag("keep")) fs.rmSync(dir, { recursive: true, force: true }); else console.error("kept", dir);
}
