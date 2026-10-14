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
//   node packages/room-node/test/flash_tab_gold.mjs tabwait   the tab hosts and waits for EXT (1) devices from other
//        machines (SIGNAL_HOST: this machine's LAN address; they run pooled join <code> --signal <host>:<PORT+2>)
//   node packages/room-node/test/flash_tab_gold.mjs resident  the tab alone with TAB_GB=90 (no offload): the reference BITS
//   node packages/room-node/test/flash_tab_gold.mjs tabjoin   the demo's "PC joins": the tab comes online alone, then a
//        room node joins (NODE_*), the tab's host re-deals (its Re-deal button) and the room runs the cases. Reports the
//        bytes the tab fetched from the model files for each of the two loads (netGiB.alone, netGiB.redeal): on a warm
//        tab the re-deal reads the layers that became resident from the OPFS copy, not the network.
//   node packages/room-node/test/flash_tab_gold.mjs demo      the demo's flow with a device from another machine: the tab
//        comes online alone and runs CASES (REPS times), then writes its invite link to INVITE_FILE and waits for EXT (1)
//        device(s) (on the PC: pooled join "<link>" --gb 10 --ram 43 --disk), re-deals (its Re-deal button: redealS is
//        the click to online) and runs CASES again (alone / pooled in the result). Each answer also reports its laps
//        (host ms, wire + the other devices) and the workers' reports (compute ms, expert hits). With ORIGIN (e.g.
//        https://pooled.run) the page is served from this checkout under that origin (its OPFS copy and caches are
//        that origin's: a profile that ran the demo there starts warm) and the room uses the page's own signaling.
//   node packages/room-node/test/flash_tab_gold.mjs ckpt      checkpoints with offload (?ckpt=CKPT, default 2): turn A
//        (two-sum, CKPT_N ids), turn B (bash, another prompt: the caches no longer hold A), turn C (A's prompt and ids,
//        CKPT_N more): with checkpoints C restores A's end state (loadSlot) and prefills one token; with CKPT=0 it
//        prefills everything. A + C must be the golden ids, C must reuse A's checkpoint (CKPT=2) or none (CKPT=0), and C's BITS be the same for both.
// Consent: the tab answers its disk question (p2p.html #disk-ask) the way TABDISK says: allow (default: it clicks
// Allow, as the demo's user does once), no, or seed (localStorage pooled-tabdisk=allow before the page loads, as an
// earlier yes leaves it: no question). The URL can't answer yes (room/flashdisk.js tabDiskChoice). The profile
// remembers the answer.
// Prerequisites: npm ci at the repo root and in packages/room-node (playwright-core, peerjs); a Chrome with WebGPU
// (CHROME_BIN; on the Spark ~/.cache/ms-playwright/chromium-1148/chrome-linux/chrome); the model files; openssl. A
// fresh PROFILE takes ~96 GB of disk for the copy (check df -h first; delete test profiles after).
// env: MODELS (~/llm/models: <MODELS>/q38flash/<the three files>), WEIGHTS_URL (fetch the files from there instead),
//      CASES (default two-sum,bash,eos-mid), TAB_GB, TAB_RAM
//      (the tab's ?tabram=, 3), NODE_GB (8), NODE_RAM (24), NODE_DISK (1), OFFLOAD_DISCRETE (1: the node counts as a
//      discrete GPU on the GB10's unified memory), PROFILE (Chrome profile with the OPFS copy; default
//      ~/.cache/pooled-flashtab-profile), CHROME_BIN (default Playwright's Chromium), HEADLESS (1), MAXNEW (cap per
//      case, default the golden's length), PORT (8261), OUT (result JSON), SPLIT (memory), QUERY (extra page params),
//      TABDISK (allow | no | seed), CKPT (the page's ?ckpt=: 0 here, 2 in mode ckpt), CKPT_N (16)
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
// SIGNAL_HOST: this machine's LAN address, for devices on other machines (the PeerServer then listens on every address)
const PORT = +env("PORT", 8261), SIG = PORT + 2, SIGNAL_HOST = env("SIGNAL_HOST", ""), SIGNAL = `${SIGNAL_HOST || "127.0.0.1"}:${SIG}`;
const MODELS = path.resolve(env("MODELS", path.join(os.homedir(), "llm/models")));
const FLASH_DIR = fs.existsSync(path.join(MODELS, "q38flash")) ? path.join(MODELS, "q38flash") : MODELS;
const TAB_GB = +env("TAB_GB", MODE === "resident" ? 90 : 12), TAB_RAM = env("TAB_RAM", "3");
const PROFILE = env("PROFILE", path.join(os.homedir(), ".cache", "pooled-flashtab-profile"));
const TABDISK = env("TABDISK", "allow"), CKPT = env("CKPT", MODE === "ckpt" ? "2" : "0");
const TRACE = env("TRACE", "") === "1";
const ORIGIN = env("ORIGIN", "").replace(/\/$/, "");   // serve the page from here under that origin (demo)
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
const peerServer = spawn(path.join(NM, ".bin/peerjs"), ["--port", String(SIG), "--path", "/", "--host", SIGNAL_HOST ? "0.0.0.0" : "127.0.0.1"], { stdio: "ignore" });
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
  // (demo with ORIGIN and no WEIGHTS_URL: the page fetches from Hugging Face itself, as on the site: a warm profile
  // fetches only the files' headers there)
  if (!ORIGIN || REMOTE !== "/") await ctx.route(/huggingface\.co\/bartowski\/Qwen3\.8-Flash-Next-GGUF\/.*\.gguf$/, (route) => route.continue({ url: WEIGHTS + route.request().url().split("/").pop() }));
  await ctx.route("https://cdn.jsdelivr.net/npm/peerjs@1.5.4/dist/peerjs.min.js", (route) => route.fulfill({ path: path.join(NM, "peerjs/dist/peerjs.min.js"), contentType: "text/javascript" }));
  if (ORIGIN) {
    // this checkout's files under ORIGIN (vercel.json's rewrites: /room and /r/<code> are p2p.html); its /api is not here
    await ctx.route((u) => u.origin === ORIGIN, (route) => {
      const u = new URL(route.request().url());
      const rel = u.pathname === "/room" || u.pathname.startsWith("/r/") ? "p2p.html" : decodeURIComponent(u.pathname).replace(/^\/+/, "") || "index.html";
      const p = path.join(ROOT, rel);
      if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) return u.pathname.startsWith("/api/") ? route.continue() : route.fulfill({ status: 404, body: "" });
      return route.fulfill({ path: p, contentType: MIME[path.extname(p)] || "application/octet-stream", headers: { "cache-control": "no-store" } });
    });
  }
  // demo: the host's Re-deal button is pressed (and timed) here, so devices are not added on their own (room/autoadd.js;
  // AUTOADD=on keeps the page's own behaviour)
  if (MODE === "demo" && env("AUTOADD", "off") !== "on") await ctx.addInitScript(() => { try { localStorage.setItem("pooled-autoadd", "off"); } catch { /* no storage */ } });
  if (TABDISK === "seed") await ctx.addInitScript(() => { try { localStorage.setItem("pooled-tabdisk", "allow"); } catch { /* no storage */ } });
  page = ctx.pages()[0] || await ctx.newPage();
  page.on("crash", () => log("tab CRASHED"));
  page.on("pageerror", (e) => log("tab pageerror:", String(e).slice(0, 300)));
  page.on("console", (m) => { const t = m.text(); if (m.type() === "error" || TRACE || /expert|OPFS|n-gram|offload|ingest|disk tier/i.test(t)) log("tab:", t.slice(0, 400)); });
  const q = `peerweights=0&dev=1&ckpt=${CKPT}&split=${env("SPLIT", "memory")}&ask=0&tabram=${TAB_RAM}${env("QUERY", "") ? "&" + env("QUERY", "") : ""}`;
  await page.goto(ORIGIN ? `${ORIGIN}/room?${q}` : `http://127.0.0.1:${PORT}/p2p.html?signal=${SIGNAL}&${q}`);
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
  if (MODE === "tabwait") {   // devices on other machines join (pooled join <code> --signal <SIGNAL>): wait for EXT of them
    const want = 1 + +env("EXT", 1);
    log(`waiting for ${want - 1} device(s): pooled join ${roomCode} --signal ${SIGNAL}`);
    if (env("CODE_FILE", "")) fs.writeFileSync(env("CODE_FILE", ""), roomCode + "\n");
    await page.waitForFunction((n) => document.querySelectorAll(".peer-card").length >= n, want, { timeout: +env("JOIN_MS", 1800e3), polling: 1000 });
    await page.waitForTimeout(3000);
  }
  const joinNode = async () => {
    const MEM = env("OFFLOAD_DISCRETE", "1") === "1" ? { kind: "discrete", name: "test", totalGB: 0, freeGB: 0 } : null;
    worker = await joinRoom(roomCode, { pledgeGB: +env("NODE_GB", 8), ramGB: +env("NODE_RAM", 24), expertDisk: env("NODE_DISK", "1") === "1", mem: MEM,
      name: "flash-node", signal: SIGNAL, modelDir: MODELS, log: (s) => log("[node] " + s) });
    await page.waitForFunction(() => document.querySelectorAll(".peer-card").length >= 2, null, { timeout: 60000 });
    await page.waitForTimeout(1500);
  };
  if (MODE === "tabnode") await joinNode();
  // the tab's disk question (p2p.html #disk-ask), answered as TABDISK says, while it waits for the room to come online
  out.diskAsked = 0;
  const answer = async () => {
    if (TABDISK === "seed") return;
    try {
      if (await page.isVisible("#disk-ask")) {
        out.diskAsked++;
        log("disk question:", (await page.textContent("#disk-ask-why")).slice(0, 200));
        await page.click(TABDISK === "no" ? "#disk-ask-no" : "#disk-ask-allow");
      }
    } catch { /* the page is busy */ }
  };
  const waitOnline = async (what) => {
    const t0 = Date.now();
    let lastS = "";
    const poll = setInterval(async () => { await answer(); try { const s = (await tabStatus()).status; if (s !== lastS) { lastS = s; log("status:", s.slice(0, 200)); } } catch {} }, 2000);
    try {
      await page.waitForFunction(() => document.getElementById("ai-panel").classList.contains("online") || /^failed:|didn't start/.test(document.getElementById("ai-status").textContent), null, { timeout: +env("ONLINE_MS", 4 * 3600e3), polling: 2000 });
    } finally { clearInterval(poll); }
    const st = await tabStatus();
    log(`${what}: tab log:\n  ` + st.log.join("\n  "));
    if (!/online/.test(st.status) || /^failed:|didn't start/.test(st.status)) throw new Error(`${what}: tab start ${st.status}`);
    return { s: (Date.now() - t0) / 1000, st };
  };
  await page.selectOption("#ai-model", "qwen3.8-flash");
  let served0 = served;
  await page.click("#ai-start");
  const on1 = await waitOnline("start");
  out.onlineS = on1.s;
  out.tab = on1.st;
  const gib = (b) => +(b / 2 ** 30).toFixed(2);
  out.netGiB = { alone: gib(served - served0) };
  log(`online in ${out.onlineS.toFixed(0)} s; ${(served / 2 ** 30).toFixed(1)} GiB served from the model files`);
  if (MODE === "tabjoin") {
    // the demo's second device: it joins the running room, the host re-deals (its Re-deal button), and every layer that
    // moves from the tab's disk to its GPU must come from the OPFS copy
    await joinNode();
    await page.waitForFunction(() => !document.getElementById("ai-redeal").hidden, null, { timeout: 120000, polling: 1000 });
    log("redeal offered:", await page.textContent("#redeal-why"));
    served0 = served;
    const tR = Date.now();
    await page.evaluate(() => document.getElementById("ai-redeal").click());
    await page.waitForFunction(() => !document.getElementById("ai-panel").classList.contains("online"), null, { timeout: 60000 });
    const on2 = await waitOnline("re-deal");
    out.redealS = (Date.now() - tR) / 1000;
    out.netGiB.redeal = gib(served - served0);
    out.tabRedeal = on2.st;
    log(`re-dealt in ${out.redealS.toFixed(0)} s; the tab fetched ${out.netGiB.redeal} GiB from the model files (alone: ${out.netGiB.alone} GiB)`);
  }
  const fx = (v, d = 2) => (Number.isFinite(v) ? +v.toFixed(d) : null);
  // TRACE=1: the tab's whole console, and its status line every 15 s while a prompt runs (where a stuck one stopped)
  const gen = async (ids, n) => {
    let lastS = "";
    const poll = TRACE ? setInterval(async () => { try { const s = (await tabStatus()).status; if (s !== lastS) { lastS = s; log("gen status:", s.slice(0, 200)); } } catch {} }, 15000) : null;
    let r;
    try { r = await page.evaluate(([ids, n]) => window.__pooledGen(ids, n), [ids, n]); } finally { clearInterval(poll); }
    log(`raw ${JSON.stringify({ ...r, tokens: r.tokens?.length, summary: undefined, laps: r.laps?.length, tele: r.tele?.length })}`);
    return r;
  };
  const med = (a) => { const b = a.filter(Number.isFinite).sort((x, y) => x - y); return b.length ? +b[b.length >> 1].toFixed(1) : null; };
  // the golden CASES (REPS times each), each answer's verdict, speed and where its time went, into bag
  const runCases = async (bag, tag = "", cases = CASES) => {
    for (let rep = 0; rep < +env("REPS", 1); rep++) for (const name of cases) {
      const key = bag[name] ? `${name}#${rep + 1}` : name;
      const g = gold[name];
      if (!g) throw new Error("no golden case " + name);
      const n = Math.min(g.ids.length, +env("MAXNEW", g.ids.length)), t = Date.now();
      const r = await gen(g.prompt, n);
      const ids = r.tokens;
      let diff = ids.findIndex((x, i) => i < n && x !== g.ids[i]);
      if (diff < 0 && ids.length < n) diff = ids.length;
      const ties = (g.nearTies || []).map((x) => x[0]), tieOk = diff >= 0 && ties.some((i) => i <= diff);
      const verdict = diff < 0 ? "MATCH" : tieOk ? "TIE" : "MISMATCH";
      if (verdict === "MISMATCH") fail++;
      const e = r.experts, hit = e && e.lookups ? e.hits / e.lookups : null;
      const c = bag[key] = { verdict, n, diff, bits: r.bits, tps: fx(r.tps), prefillS: fx(r.tPre / 1000, 1), wallS: +((Date.now() - t) / 1000).toFixed(1), prompt: g.prompt.length,
        ...(e ? { hostExperts: { ...e, hitRate: fx(hit, 3), diskReadsPerTok: +(e.diskReads / (n + g.prompt.length)).toFixed(1), t1Hits: e.t1Hits } } : {}),
        ...(diff >= 0 ? { got: ids.slice(Math.max(0, diff - 2), diff + 4), want: g.ids.slice(Math.max(0, diff - 2), diff + 4) } : {}) };
      log(`${tag ? tag + " " : ""}${key}: ${verdict}${diff >= 0 ? ` at ${diff}` : ` (${n} ids)`} · bits ${r.bits} · prompt ${g.prompt.length} tok in ${fx(r.tPre / 1000, 1)} s · decode ${fx(r.tps)} tok/s${hit != null ? ` · host pool hits ${(100 * hit).toFixed(1)}%, ${e.diskReads} disk reads, ${(e.diskWaitMs / 1000).toFixed(1)} s waiting` : ""}`);
      // the laps of its decode tokens (each: the host's layers, then the hidden state round the other devices and back)
      const dl = (r.laps || []).slice(-(Math.max(1, n - 1)));
      if (dl.length) {
        c.lapMs = med(dl.map((x) => x[0])); c.hostMs = med(dl.map((x) => x[1])); c.chainMs = med(dl.map((x) => x[0] - x[1])); c.headMs = med(dl.map((x) => x[2]));
      // between one lap's head and the next lap: sampling, the page's own work, and whatever kept its main thread busy
      const gaps = dl.slice(1).map((x, j) => x[3] - (dl[j][3] + dl[j][0] + (dl[j][2] || 0)));
      if (env("RAW_LAPS", "") === "1") c.laps = dl.map((x) => x.map((v) => Math.round(v * 10) / 10));
      c.gapMs = med(gaps); c.gapP90 = gaps.length ? +gaps.sort((a, b) => a - b)[Math.floor(0.9 * (gaps.length - 1))].toFixed(1) : null;
        c.tokMs = +(1000 / r.tps).toFixed(1); c.lapP90 = +dl.map((x) => x[0]).sort((a, b) => a - b)[Math.floor(0.9 * (dl.length - 1))].toFixed(1);
        const tl = (r.tele || []).filter((x) => x.k?.one != null);
        if (tl.length) c.workerMs = med(tl.map((x) => x.k.one));
        const ex = (r.tele || []).map((x) => x.exp).filter(Boolean), H = ex.reduce((a, x) => a + x.hits, 0), M = ex.reduce((a, x) => a + x.misses, 0), T = ex.reduce((a, x) => a + x.tok, 0);
        if (H + M) c.workerExperts = { hitRate: fx(H / (H + M), 3), missPerTok: T ? fx(M / T, 1) : null, readsPerTok: T && ex.some((x) => x.reads != null) ? fx(ex.reduce((a, x) => a + (x.reads || 0), 0) / T, 1) : null };
        log(`  ${key}: ${c.tokMs} ms a token: lap ${c.lapMs} (p90 ${c.lapP90}) = host ${c.hostMs} + chain ${c.chainMs}, head ${c.headMs}, between laps ${c.gapMs} (p90 ${c.gapP90})${c.workerMs != null ? ` (worker ${c.workerMs})` : ""}${c.workerExperts ? ` · worker hits ${(100 * c.workerExperts.hitRate).toFixed(1)}%, ${c.workerExperts.missPerTok} misses a token` : ""}`);
      }
      if (r.summary) out.hostSummary = r.summary;
      if (worker?.ai?.experts) { const s = worker.ai.experts.summary(); c.nodeExperts = s; log("[node] " + s); }
    }
  };
  let fail = 0;
  if (MODE === "ckpt") {
    // A, B, C (see the header): C restores A's end state with checkpoints on
    const N = +env("CKPT_N", 16), A = gold["two-sum"], B = gold.bash;
    const ra = await gen(A.prompt, N), rb = await gen(B.prompt, Math.min(8, B.ids.length));
    // (A's end checkpoint holds its prompt and all N ids: C's prompt is one golden id longer, as a follow-up is, so the
    // checkpoint is a strict prefix of it and only that id is prefilled)
    const rc = await gen([...A.prompt, ...ra.tokens, A.ids[N]], N);
    const both = [...ra.tokens, A.ids[N], ...rc.tokens], want = A.ids.slice(0, both.length);
    const diff = both.findIndex((x, i) => x !== want[i]);
    const restored = CKPT === "0" ? rc.reused === 0 : rc.reused === A.prompt.length + N;
    const verdict = diff < 0 && both.length === 2 * N + 1 && restored ? "MATCH" : "MISMATCH";
    if (verdict !== "MATCH") fail++;
    out.cases.ckpt = { verdict, ckpt: CKPT, n: both.length, diff, reusedC: rc.reused, bitsA: ra.bits, bitsB: rb.bits, bitsC: rc.bits, prefillC: fx(rc.tPre / 1000, 1), tps: fx(rc.tps) };
    log(`ckpt=${CKPT}: A+C ${verdict}${diff >= 0 ? ` at ${diff}` : ""} · C reused ${rc.reused} of ${A.prompt.length + N + 1} prompt ids · bits A ${ra.bits} B ${rb.bits} C ${rc.bits} · C's prefill ${fx(rc.tPre / 1000, 1)} s`);
  } else if (MODE !== "demo") await runCases(out.cases);
  else {
    // the demo: alone, then a device from another machine joins, the re-deal, then pooled
    out.alone = {};
    await runCases(out.alone, "alone", env("ALONE_CASES", "") ? env("ALONE_CASES", "").split(",") : CASES);
    const link = await page.evaluate(() => window.__pooledInvite?.());
    if (!link) throw new Error("no invite link (window.__pooledInvite)");
    if (env("INVITE_FILE", "")) fs.writeFileSync(env("INVITE_FILE", ""), link + "\n");
    const want = 1 + +env("EXT", 1);
    log(`waiting for ${want - 1} device(s) to join (the invite link is in ${env("INVITE_FILE", "INVITE_FILE")})`);
    await page.waitForFunction((n) => document.querySelectorAll(".peer-card").length >= n, want, { timeout: +env("JOIN_MS", 1800e3), polling: 1000 });
    await page.waitForFunction(() => !document.getElementById("ai-redeal").hidden, null, { timeout: 120000, polling: 500 });
    await page.waitForTimeout(+env("REDEAL_AFTER_MS", 2000));
    log("redeal offered:", await page.textContent("#redeal-why"));
    const tR = Date.now();
    await page.evaluate(() => document.getElementById("ai-redeal").click());
    await page.waitForFunction(() => !document.getElementById("ai-panel").classList.contains("online"), null, { timeout: 60000 });
    const on2 = await waitOnline("re-deal");
    out.redealS = (Date.now() - tR) / 1000;
    out.tabRedeal = on2.st;
    log(`re-dealt in ${out.redealS.toFixed(1)} s`);
    out.pooled = {};
    await runCases(out.pooled, "pooled");
    const med = (a) => { const b = a.filter(Number.isFinite).sort((x, y) => x - y); return b.length ? b[b.length >> 1] : null; };
    out.summary = Object.fromEntries(["alone", "pooled"].map((k) => [k, { tps: Object.values(out[k]).map((c) => c.tps), median: med(Object.values(out[k]).map((c) => c.tps)) }]));
    log("summary " + JSON.stringify(out.summary) + ` · re-deal ${out.redealS.toFixed(1)} s`);
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
