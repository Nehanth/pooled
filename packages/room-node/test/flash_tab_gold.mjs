// Qwen3.8-Flash-Next through a browser tab's room path with expert offload's disk tier (room/flashdisk.js), against
// llama.cpp's greedy ids (tests/golden/q38flash_llama_greedy.json). The real room page (p2p.html, ?dev=1) in Chrome
// hosts the room; the golden prompt ids go through the host's generate (room.js window.__pooledGen: prefill and decode
// over every device's layers, greedy, no stop token), and every step's logits are fingerprinted (BITS: two runs with
// the same fingerprint computed the same logits).
//   node packages/room-node/test/flash_tab_gold.mjs tab       the tab alone: TAB_GB (12) is far under the model, so the
//        deal offloads its layers, every one read from the tab's OPFS (ingested there on the first run: ~70 GB of
//        experts and the 28.8 GB n-gram table, kept in PROFILE for the next run)
//   node packages/room-node/test/flash_tab_gold.mjs tabnode   the tab hosts, a room node in this process joins with
//        NODE_GB / NODE_RAM / --disk (NODE_DISK=1, default) and takes what the tab's pledge does not hold
//   node packages/room-node/test/flash_tab_gold.mjs resident  the tab alone with TAB_GB=90 (no offload): the reference BITS
// env: MODELS (~/llm/models: <MODELS>/q38flash/<the three files>), WEIGHTS_URL (fetch the files from there instead),
//      CASES (default two-sum,bash,eos-mid), TAB_GB, TAB_RAM
//      (the tab's ?tabram=, 3), NODE_GB (8), NODE_RAM (24), NODE_DISK (1), OFFLOAD_DISCRETE (1: the node counts as a
//      discrete GPU on the GB10's unified memory), PROFILE (Chrome profile with the OPFS copy; default
//      ~/.cache/pooled-flashtab-profile), CHROME_BIN (default Playwright's Chromium), HEADLESS (1), MAXNEW (cap per
//      case, default the golden's length), PORT (8261), OUT (result JSON), SPLIT (memory), QUERY (extra page params)
import http from "node:http";
import https from "node:https";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn, execSync } from "node:child_process";
import { joinRoom } from "../index.js";

const MODE = process.argv[2] || "tab";
const ROOT = path.resolve(new URL("../../..", import.meta.url).pathname);
const NM = path.resolve(new URL("../node_modules", import.meta.url).pathname);
const env = (k, d) => process.env[k] ?? d;
const PORT = +env("PORT", 8261), SIG = PORT + 2, SIGNAL = `127.0.0.1:${SIG}`;
const MODELS = path.resolve(env("MODELS", path.join(os.homedir(), "llm/models")));
const FLASH_DIR = fs.existsSync(path.join(MODELS, "q38flash")) ? path.join(MODELS, "q38flash") : MODELS;
const TAB_GB = +env("TAB_GB", MODE === "resident" ? 90 : 12), TAB_RAM = env("TAB_RAM", "3");
const PROFILE = env("PROFILE", path.join(os.homedir(), ".cache", "pooled-flashtab-profile"));
const T0 = Date.now();
const log = (...a) => console.error(((Date.now() - T0) / 1000).toFixed(1) + "s", ...a);
const gold = JSON.parse(fs.readFileSync(path.join(ROOT, "tests/golden/q38flash_llama_greedy.json"), "utf8"));
const CASES = env("CASES", "two-sum,bash,eos-mid").split(",");
const out = { mode: MODE, tabGB: TAB_GB, cases: {} };
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png", ".txt": "text/plain" };

// --- servers: the page (http), the weights (https + Range), PeerJS signaling
const srv = http.createServer((q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { r.statusCode = 404; r.end(); return; }
  r.setHeader("content-type", MIME[path.extname(p)] || "application/octet-stream"); fs.createReadStream(p).pipe(r);
}).listen(PORT, "127.0.0.1");
const tlsDir = fs.mkdtempSync(path.join(os.tmpdir(), "flash-tab-"));
execSync(`openssl req -x509 -newkey rsa:2048 -nodes -keyout ${tlsDir}/k.pem -out ${tlsDir}/c.pem -days 2 -subj /CN=127.0.0.1 2>/dev/null`);
let served = 0;
// WEIGHTS_URL: the model's files from another machine (an http Range server, e.g. the Spark's over the LAN), proxied here
// (a routed request must keep its https)
const REMOTE = env("WEIGHTS_URL", "").replace(/\/?$/, "/");
const wsrv = https.createServer({ key: fs.readFileSync(`${tlsDir}/k.pem`), cert: fs.readFileSync(`${tlsDir}/c.pem`) }, (q, r) => {
  if (REMOTE !== "/") {
    const up = http.get(REMOTE + path.basename(q.url.split("?")[0]), { headers: q.headers.range ? { range: q.headers.range } : {} }, (u) => {
      r.writeHead(u.statusCode, { ...u.headers, "access-control-allow-origin": "*", "access-control-expose-headers": "content-range, content-length, accept-ranges" });
      u.on("data", (b) => { served += b.length; });
      u.pipe(r);
    });
    up.on("error", () => { r.statusCode = 502; r.end(); });
    q.on("close", () => up.destroy());
    return;
  }
  const p = path.join(FLASH_DIR, path.basename(decodeURIComponent(q.url.split("?")[0])));
  if (!fs.existsSync(p)) { r.statusCode = 404; r.end(); return; }
  const size = fs.statSync(p).size, m = /bytes=(\d+)-(\d*)/.exec(q.headers.range || "");
  const lo = m ? +m[1] : 0, hi = m && m[2] ? Math.min(+m[2], size - 1) : size - 1;
  served += hi - lo + 1;
  r.writeHead(m ? 206 : 200, { "content-type": "application/octet-stream", "content-range": `bytes ${lo}-${hi}/${size}`, "accept-ranges": "bytes", "content-length": String(hi - lo + 1),
    "access-control-allow-origin": "*", "access-control-expose-headers": "content-range, content-length, accept-ranges" });
  fs.createReadStream(p, { start: lo, end: hi, highWaterMark: 8 << 20 }).pipe(r);
}).listen(PORT + 1, "127.0.0.1");
const WEIGHTS = `https://127.0.0.1:${PORT + 1}/`;
const peerServer = spawn(path.join(NM, ".bin/peerjs"), ["--port", String(SIG), "--path", "/", "--host", "127.0.0.1"], { stdio: "ignore" });
await new Promise((r) => setTimeout(r, 1500));

let ctx = null, page = null, worker = null, code = 1;
async function openTab() {
  const { chromium } = await import("playwright-core");
  fs.mkdirSync(PROFILE, { recursive: true });
  ctx = await chromium.launchPersistentContext(PROFILE, { headless: env("HEADLESS", "1") === "1", executablePath: process.env.CHROME_BIN || undefined, ignoreHTTPSErrors: true,
    args: ["--no-sandbox", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--allow-loopback-in-peer-connection", "--js-flags=--max-old-space-size=16384",
      "--disable-background-timer-throttling", "--disable-renderer-backgrounding", "--disable-features=WebRtcHideLocalIpsWithMdns,LocalNetworkAccessChecks,BlockInsecurePrivateNetworkRequests,PrivateNetworkAccessRespectPreflightResults,PrivateNetworkAccessSendPreflights",
      ...(process.platform === "linux" ? ["--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan"] : [])] });
  // the model's files from this machine (the room page asks Hugging Face for them)
  await ctx.route(/huggingface\.co\/bartowski\/Qwen3\.8-Flash-Next-GGUF\/.*\.gguf$/, (route) => route.continue({ url: WEIGHTS + route.request().url().split("/").pop() }));
  await ctx.route("https://cdn.jsdelivr.net/npm/peerjs@1.5.4/dist/peerjs.min.js", (route) => route.fulfill({ path: path.join(NM, "peerjs/dist/peerjs.min.js"), contentType: "text/javascript" }));
  page = ctx.pages()[0] || await ctx.newPage();
  page.on("crash", () => log("tab CRASHED"));
  page.on("pageerror", (e) => log("tab pageerror:", String(e).slice(0, 300)));
  page.on("console", (m) => { const t = m.text(); if (m.type() === "error" || /expert|OPFS|n-gram|offload|ingest|disk tier/i.test(t)) log("tab:", t.slice(0, 400)); });
  await page.goto(`http://127.0.0.1:${PORT}/p2p.html?signal=${SIGNAL}&peerweights=0&dev=1&ckpt=0&split=${env("SPLIT", "memory")}&ask=0&tabram=${TAB_RAM}${env("QUERY", "") ? "&" + env("QUERY", "") : ""}`);
  await page.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
  await page.waitForTimeout(1500);
  await page.fill("#name-input", "flash-tab"); await page.fill("#join-gb", String(TAB_GB));
}
const tabStatus = () => page.evaluate(() => ({ status: document.getElementById("ai-status")?.textContent, log: [...document.querySelectorAll("#chat-log div")].slice(-8).map((d) => d.textContent) }));
try {
  await openTab();
  out.ua = await page.evaluate(() => navigator.userAgent);
  await page.click("#create-btn");
  await page.waitForFunction(() => /[A-Z0-9]{3}-?[A-Z0-9]{3}/.test(document.getElementById("side-code").textContent), null, { timeout: 30000 });
  const roomCode = (await page.textContent("#side-code")).trim().replace("-", "").match(/[A-Z0-9]{4,6}/)[0];
  log("room", roomCode);
  if (MODE === "tabnode") {
    const MEM = env("OFFLOAD_DISCRETE", "1") === "1" ? { kind: "discrete", name: "test", totalGB: 0, freeGB: 0 } : null;
    worker = await joinRoom(roomCode, { pledgeGB: +env("NODE_GB", 8), ramGB: +env("NODE_RAM", 24), expertDisk: env("NODE_DISK", "1") === "1", mem: MEM,
      name: "flash-node", signal: SIGNAL, modelDir: MODELS, log: (s) => log("[node] " + s) });
    await page.waitForFunction(() => document.querySelectorAll(".peer-card").length >= 2, null, { timeout: 60000 });
    await page.waitForTimeout(1500);
  }
  await page.selectOption("#ai-model", "qwen3.8-flash");
  const tStart = Date.now();
  await page.click("#ai-start");
  let lastS = "";
  const poll = setInterval(async () => { try { const s = (await tabStatus()).status; if (s !== lastS) { lastS = s; log("status:", s.slice(0, 200)); } } catch {} }, 5000);
  try {
    await page.waitForFunction(() => document.getElementById("ai-panel").classList.contains("online") || /^failed:/.test(document.getElementById("ai-status").textContent), null, { timeout: +env("ONLINE_MS", 4 * 3600e3), polling: 2000 });
  } finally { clearInterval(poll); }
  out.onlineS = (Date.now() - tStart) / 1000;
  out.tab = await tabStatus();
  log("tab log:\n  " + out.tab.log.join("\n  "));
  if (/^failed:/.test(out.tab.status)) throw new Error("tab start " + out.tab.status);
  log(`online in ${out.onlineS.toFixed(0)} s; ${(served / 2 ** 30).toFixed(1)} GiB served from the model files`);
  let fail = 0;
  for (const name of CASES) {
    const g = gold[name];
    if (!g) throw new Error("no golden case " + name);
    const n = Math.min(g.ids.length, +env("MAXNEW", g.ids.length)), t = Date.now();
    const r = await page.evaluate(([ids, n]) => window.__pooledGen(ids, n), [g.prompt, n]);
    const ids = r.tokens;
    let diff = ids.findIndex((x, i) => i < n && x !== g.ids[i]);
    if (diff < 0 && ids.length < n) diff = ids.length;
    const ties = (g.nearTies || []).map((x) => x[0]), tieOk = diff >= 0 && ties.some((i) => i <= diff);
    const verdict = diff < 0 ? "MATCH" : tieOk ? "TIE" : "MISMATCH";
    if (verdict === "MISMATCH") fail++;
    const e = r.experts, hit = e && e.lookups ? e.hits / e.lookups : null;
    out.cases[name] = { verdict, n, diff, bits: r.bits, tps: +r.tps.toFixed(2), prefillS: +(r.tPre / 1000).toFixed(1), wallS: +((Date.now() - t) / 1000).toFixed(1), prompt: g.prompt.length,
      ...(e ? { hostExperts: { ...e, hitRate: +hit.toFixed(3), diskReadsPerTok: +(e.diskReads / (n + g.prompt.length)).toFixed(1), t1Hits: e.t1Hits } } : {}),
      ...(diff >= 0 ? { got: ids.slice(Math.max(0, diff - 2), diff + 4), want: g.ids.slice(Math.max(0, diff - 2), diff + 4) } : {}) };
    log(`${name}: ${verdict}${diff >= 0 ? ` at ${diff}` : ` (${n} ids)`} · bits ${r.bits} · prompt ${g.prompt.length} tok in ${(r.tPre / 1000).toFixed(1)} s · decode ${r.tps.toFixed(2)} tok/s${hit != null ? ` · host pool hits ${(100 * hit).toFixed(1)}%, ${e.diskReads} disk reads, ${(e.diskWaitMs / 1000).toFixed(1)} s waiting` : ""}`);
    if (r.summary) out.hostSummary = r.summary;
    if (worker?.ai?.experts) { const s = worker.ai.experts.summary(); out.cases[name].nodeExperts = s; log("[node] " + s); }
  }
  out.pass = !fail;
  code = fail ? 1 : 0;
} catch (e) {
  out.error = String(e?.stack || e);
  log("FAILED: " + (e?.message || e));
  try { out.tabAtFail = await tabStatus(); log(JSON.stringify(out.tabAtFail)); } catch {}
} finally {
  if (process.env.OUT) fs.writeFileSync(process.env.OUT, JSON.stringify(out, null, 1));
  console.log(JSON.stringify(out));
  console.log(out.pass ? "FLASH TAB GOLD PASS" : "FLASH TAB GOLD FAIL");
  try { await worker?.close?.(); } catch {}
  try { await Promise.race([ctx?.close(), new Promise((r) => setTimeout(r, 15000))]); } catch {}
  try { peerServer.kill(); } catch {}
  srv.close(); wsrv.close(); fs.rmSync(tlsDir, { recursive: true, force: true });
  setTimeout(() => process.exit(code), 500);
}
