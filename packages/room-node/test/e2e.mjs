// Room node end to end on one machine (GPU), with a local PeerJS server and the room page served from
// this checkout (weights from local disk over https, as tests/e2e/xroom.mjs):
//   node packages/room-node/test/e2e.mjs nodehost   the node hosts; a headless browser tab joins and holds layers
//   node packages/room-node/test/e2e.mjs tabhost    a browser tab hosts; the node joins and holds layers
//   node packages/room-node/test/e2e.mjs nodepair   two room nodes in this process (host + worker), no browser
//   node packages/room-node/test/e2e.mjs api        `pooled serve`'s bridge (cli/lib/room.js) asks a node host
//   node packages/room-node/test/e2e.mjs auto       createRoom + ask, no start(): the ask deals the layers (solo)
//   node packages/room-node/test/e2e.mjs wedge      a joiner (its own process) is stopped while the host still loads its
//        layers (slowed down by HOST_DELAY_MS, 40 s): the room must not go online without it (it used to, and
//        every ask said "the model is still loading"). BACK=after (default): it starts again once the host's
//        load is in, is dealt its layers again and the room answers. BACK=none: it stays away past GRACE_MS
//        (default 15 s here), the room re-deals, finds HOST_GB (2.5) short and stops saying so; then it
//        comes back, a start deals it in again and the room answers. KILL=9: SIGKILL instead of SIGTERM.
//   node packages/room-node/test/e2e.mjs cache      OpenClaw's recorded request through the host's checkpoints:
//        a cold first turn, a side request, the follow-up, a new session, the next day (see cacheRun);
//        SETUP=solo|pair|proc|tab (the node alone, + a second node, the second node in its own process, + a browser tab), REQ (requests.jsonl
//        recorded from an OpenClaw gateway), CKPT=0 (no checkpoints: the baseline), CTX, STEPS=turn1 (the cold turn only)
// env: MODELS (model dir, layout of source.js LOCAL; default <checkout>/models), MODEL (qwen3-1.7b), PROMPT,
//      MAXNEW (48), REF (solo JSON from test/solo.mjs, to compare), NODE_GB, TAB_GB, WORKER_GB / WORKER_RAM / HOST_RAM (nodepair), NODE_RAM (tabhost), OFFLOAD_DISCRETE=1 (offload on unified memory, as a discrete GPU), CHROME_BIN (a Chromium or
//      headless_shell with WebGPU; default playwright's), PORT (8231), OUT (result JSON path),
//      SIGNAL=cloud / PAGE=live (the public PeerJS server / the room page on https://pooled.run)
import http from "node:http";
import https from "node:https";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn, execSync } from "node:child_process";
import { createRoom, joinRoom } from "../index.js";
import { loadShard } from "../shard.js";

const MODE = process.argv[2] || "nodehost";
const ROOT = path.resolve(new URL("../../..", import.meta.url).pathname);
const NM = path.resolve(new URL("../node_modules", import.meta.url).pathname);
const PORT = +(process.env.PORT || 8231), SIG = PORT + 2;
// SIGNAL=cloud: the public PeerJS server (what pooled.run uses); PAGE=live: the room page from https://pooled.run
const CLOUD = process.env.SIGNAL === "cloud" || process.env.PAGE === "live";
const SIGNAL = CLOUD ? null : `127.0.0.1:${SIG}`;
const PAGE = process.env.PAGE === "live" ? "https://pooled.run/room" : `http://127.0.0.1:${PORT}/p2p.html`;
const PROMPT = process.env.PROMPT || "Why is the sky blue? Answer in two sentences.";
const MAXNEW = +(process.env.MAXNEW || 48);
const MODEL = process.env.MODEL || "qwen3-1.7b";
const NODE_GB = +(process.env.NODE_GB || 4), TAB_GB = +(process.env.TAB_GB || 4);
const MODELS = path.resolve(process.env.MODELS || path.join(ROOT, "models"));
const T0 = Date.now();
const log = (...a) => console.error(((Date.now() - T0) / 1000).toFixed(1) + "s", ...a);
const out = { mode: MODE, model: MODEL, prompt: PROMPT, maxNew: MAXNEW };
const CKPT = process.env.CKPT === "0" ? false : {};
const CTX = +(process.env.CTX || 0);
// OFFLOAD_DISCRETE=1: the nodes count as discrete GPUs for expert offload (env.js probeMeta refuses it on unified
// memory: a GB10 testing what a PC with a discrete GPU does)
const MEM = process.env.OFFLOAD_DISCRETE === "1" ? { kind: "discrete", name: "test", totalGB: 0, freeGB: 0 } : null;

// --- servers: the page (http), the weights (https + Range), PeerJS signaling
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png" };
const srv = http.createServer((q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { r.statusCode = 404; r.end(); return; }
  r.setHeader("content-type", MIME[path.extname(p)] || "application/octet-stream"); fs.createReadStream(p).pipe(r);
}).listen(PORT, "127.0.0.1");
const tlsDir = fs.mkdtempSync(path.join(os.tmpdir(), "rn-e2e-"));
execSync(`openssl req -x509 -newkey rsa:2048 -nodes -keyout ${tlsDir}/k.pem -out ${tlsDir}/c.pem -days 2 -subj /CN=127.0.0.1 2>/dev/null`);
const wsrv = https.createServer({ key: fs.readFileSync(`${tlsDir}/k.pem`), cert: fs.readFileSync(`${tlsDir}/c.pem`) }, (q, r) => {
  const rel = decodeURIComponent(q.url.split("?")[0]).replace(/^\/models\//, "");
  const p = path.join(MODELS, rel);
  if (!p.startsWith(MODELS) || !fs.existsSync(p)) { r.statusCode = 404; r.end(); return; }
  const size = fs.statSync(p).size, m = /bytes=(\d+)-(\d*)/.exec(q.headers.range || "");
  const lo = m ? +m[1] : 0, hi = m && m[2] ? Math.min(+m[2], size - 1) : size - 1;
  r.writeHead(m ? 206 : 200, { "content-type": "application/octet-stream", "content-range": `bytes ${lo}-${hi}/${size}`, "accept-ranges": "bytes", "content-length": String(hi - lo + 1), "access-control-allow-origin": "*", "access-control-expose-headers": "content-range, content-length, accept-ranges" });
  fs.createReadStream(p, { start: lo, end: hi }).pipe(r);
}).listen(PORT + 1, "127.0.0.1");
const peerServer = CLOUD ? null : spawn(path.join(NM, ".bin/peerjs"), ["--port", String(SIG), "--path", "/", "--host", "127.0.0.1"], { stdio: "ignore" });
if (peerServer) await new Promise((r) => setTimeout(r, 1500));
out.page = PAGE; out.signal = SIGNAL || "cloud";

// --- the browser tab (the real room page)
let ctx = null, page = null;
async function openTab(name, gb) {
  const { chromium } = await import("playwright-core");
  const exe = process.env.CHROME_BIN || undefined;
  const prof = fs.mkdtempSync(path.join(os.tmpdir(), "rn-prof-"));
  ctx = await chromium.launchPersistentContext(prof, { headless: true, executablePath: exe, ignoreHTTPSErrors: true,
    userAgent: "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36",
    args: ["--no-sandbox", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--allow-loopback-in-peer-connection",
      "--disable-background-timer-throttling", "--disable-renderer-backgrounding", "--disable-features=WebRtcHideLocalIpsWithMdns,LocalNetworkAccessChecks,BlockInsecurePrivateNetworkRequests,PrivateNetworkAccessRespectPreflightResults,PrivateNetworkAccessSendPreflights",
      "--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan"] });
  await ctx.route("**/*.gguf", (route) => {
    const base = route.request().url().split("/").pop();
    const f = { "Qwen3-1.7B-Q8_0.gguf": "models/qwen17/model.gguf", "Qwen_Qwen3.6-35B-A3B-Q4_0.gguf": "models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf" }[base]
      || (/^Qwen_Qwen3\.5-122B-A10B-Q4_0-0000[12]-of-00002\.gguf$/.test(base) ? `models/q35-122b/${base}` : null);   // the 122B's two files
    if (!f) return route.continue();
    if (process.env.PAGE !== "live" || process.env.FULFILL !== "1") return route.continue({ url: `https://127.0.0.1:${PORT + 1}/${f}` });
    // a public page may not fetch from 127.0.0.1 (Private Network Access): answer the range here
    const file = path.join(MODELS, f.replace(/^models\//, "")), size = fs.statSync(file).size;
    const m = /bytes=(\d+)-(\d*)/.exec(route.request().headers().range || "");
    const lo = m ? +m[1] : 0, hi = m && m[2] ? Math.min(+m[2], size - 1) : size - 1;
    const body = Buffer.alloc(hi - lo + 1), fd = fs.openSync(file, "r");
    try { fs.readSync(fd, body, 0, body.length, lo); } finally { fs.closeSync(fd); }
    return route.fulfill({ status: m ? 206 : 200, body, headers: { "content-type": "application/octet-stream", "content-range": `bytes ${lo}-${hi}/${size}`, "accept-ranges": "bytes",
      "access-control-allow-origin": "*", "access-control-expose-headers": "content-range, content-length, accept-ranges" } });
  });
  await ctx.route("https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/*.json", (route) =>
    route.fulfill({ path: path.join(MODELS, "qwen17", route.request().url().split("/").pop()), contentType: "application/json", headers: { "access-control-allow-origin": "*" } }));
  await ctx.route("https://cdn.jsdelivr.net/npm/peerjs@1.5.4/dist/peerjs.min.js", (route) =>
    route.fulfill({ path: path.join(NM, "peerjs/dist/peerjs.min.js"), contentType: "text/javascript" }));
  page = ctx.pages()[0] || await ctx.newPage();
  page.on("crash", () => log("tab CRASHED"));
  page.on("pageerror", (e) => log("tab pageerror:", String(e).slice(0, 200)));
  page.on("console", (m) => { if (m.type() === "error") log("tab console:", m.text().slice(0, 200)); });
  await page.goto(`${PAGE}?${SIGNAL ? `signal=${SIGNAL}&` : ""}peerweights=0&dev=1&maxnew=${MAXNEW}&ckpt=0&split=memory&ask=0`);   // ask=0: a tab host lets devices in without its Allow (the gate)
  await page.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
  await page.waitForFunction(() => document.getElementById("join-gb").value !== "1", null, { timeout: 30000 }).catch(() => {});
  await page.waitForTimeout(800);
  await page.fill("#name-input", name); await page.fill("#join-gb", String(gb));
}
const tabStatus = () => page.evaluate(() => ({ status: document.getElementById("ai-status")?.textContent, log: [...document.querySelectorAll("#chat-log div")].slice(-4).map((d) => d.textContent) }));

let host = null, worker = null, joiner = null, code = 1;
const finish = async (c) => {
  code = c;
  out.errors = out.errors || [];
  console.log(JSON.stringify(out));
  if (process.env.OUT) fs.writeFileSync(process.env.OUT, JSON.stringify(out, null, 1));
  try { await worker?.close(); } catch {}
  joiner?.kill("SIGTERM");
  try { await host?.close(); } catch {}
  try { await Promise.race([ctx?.close(), new Promise((r) => setTimeout(r, 10000))]); } catch {}
  peerServer?.kill(); srv.close(); wsrv.close(); fs.rmSync(tlsDir, { recursive: true, force: true });
  setTimeout(() => process.exit(c), 300);
};
const nodeLog = (tag) => (s) => log(`[${tag}] ${s}`);
const ask = async (room) => {
  const t = performance.now(); let first = null, text = "", done = null;
  for await (const ev of room.ask([{ role: "user", content: PROMPT }], { maxTokens: MAXNEW, temperature: 0 })) {
    if (ev.type === "token") { if (first == null) first = performance.now() - t; text += ev.text; process.stderr.write(ev.text); }
    else done = ev;
  }
  process.stderr.write("\n");
  return { text, ttftMs: Math.round(first), ...done };
};

const READ = [{ name: "read", description: "Read a file from the workspace.", parameters: { type: "object", properties: { path: { type: "string", description: "file path" } }, required: ["path"] } }];
async function toolTurns(room) {
  const run = async (messages) => {
    let text = "", done = null;
    for await (const ev of room.ask(messages, { tools: READ, maxTokens: 200, temperature: 0 })) { if (ev.type === "token" && !ev.think) text += ev.text; if (ev.type === "done") done = ev; }
    return { text, reason: done.reason, err: done.err, calls: done.calls, usage: done.usage, reused: done.reused, stats: done.stats };
  };
  const q = [{ role: "system", content: "You are a helpful assistant with file tools." }, { role: "user", content: "Read notes.txt and tell me the secret word." }];
  const first = await run(q);
  log("tool turn 1:", JSON.stringify(first));
  if (!first.calls?.length) return { first };
  const second = await run([...q, { role: "assistant", content: first.text, tool_calls: first.calls.map((c) => ({ id: c.id, type: "function", function: { name: c.name, arguments: c.args } })) },
    ...first.calls.map((c) => ({ role: "tool", tool_call_id: c.id, content: "The secret word is PELICAN-42." }))]);
  log("tool turn 2:", JSON.stringify(second));
  return { first, second, ok: /PELICAN-42/.test(second.text) };
}

// OpenClaw's traffic on one host (the recorded request: a ~28k-character system prompt with its
// STABLE / DYNAMIC cache boundary and 11 tools): what each step costs before the first token.
//   turn1     the first turn after the gateway starts (cold: nothing cached)
//   title     a side request with its own short system prompt and no tools (a session title)
//   turn2     the same session's next turn (the conversation so far + a new question)
//   session2  a new session's first turn (same system prompt and tools, another session id)
//   nextday   a new session on another day (the date after the cache boundary changed)
//   turn2b    the first session again, after all that (its answer checkpoint must still be there)
function recorded() {
  const file = process.env.REQ;
  const rec = fs.readFileSync(file, "utf8").trim().split("\n").map((l) => JSON.parse(l)).find((r) => r.body?.messages && String(r.path).includes("chat"));
  const text = (c) => (typeof c === "string" ? c : (c || []).map((p) => p.text || "").join(""));
  const b = rec.body;
  return { messages: b.messages.map((m) => ({ role: m.role, content: text(m.content) })),
    tools: (b.tools || []).map((t) => ({ name: t.function.name, description: t.function.description, parameters: t.function.parameters })) };
}
async function cacheRun(room) {
  const R = recorded();
  const maxTokens = MAXNEW;
  const run = async (tag, messages, tools = R.tools) => {
    const t = performance.now(); let first = null, text = "", done = null, pre = null;
    const onPre = (p) => { pre = p; };
    room.on("prefill", onPre);
    for await (const ev of room.ask(messages, { tools, maxTokens, temperature: 0 })) {
      if (ev.type === "token" || (ev.type === "call" && ev.name)) { if (first == null) first = performance.now() - t; if (ev.text && !ev.think) text += ev.text; }
      if (ev.type === "done") done = ev;
    }
    room.off("prefill", onPre);
    const r = { tag, ttftS: +((first ?? performance.now() - t) / 1000).toFixed(2), totalS: +((performance.now() - t) / 1000).toFixed(2), promptTokens: pre?.total, prefilled: pre?.prefilled, reused: pre?.reused, from: pre?.from, pinned: pre?.pinned,
      prefillS: pre ? +(pre.tPre / 1000).toFixed(2) : null, decodeTok: pre?.count, reason: done?.reason, err: done?.err, calls: done?.calls?.map((c) => c.name), text };
    log(`${tag}: first token ${r.ttftS} s (prompt ${r.promptTokens}: ${r.prefilled} read in ${r.prefillS} s, ${r.reused} reused from ${r.from}) ${JSON.stringify(text.slice(0, 80))}`);
    return r;
  };
  const steps = [];
  const turn1 = await run("turn1", R.messages); steps.push(turn1);
  if (process.env.STEPS === "turn1") return { steps, status: room.status() };   // the cold first turn only
  steps.push(await run("title", [{ role: "system", content: "Write a short title (at most 6 words) for this conversation. Reply with the title only." }, { role: "user", content: R.messages.filter((m) => m.role === "user").map((m) => m.content).join("\n").slice(0, 400) }], null));
  const follow = [...R.messages, { role: "assistant", content: turn1.text }, { role: "user", content: "And what is the capital of Germany? One word." }];
  const turn2 = await run("turn2", follow); steps.push(turn2);
  const other = R.messages.map((m) => (m.role === "user" ? { ...m, content: m.content.replace(/What is the capital of France\?/, "Name one prime number.").replace(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/g, "11111111-2222-3333-4444-555555555555") } : m));
  steps.push(await run("session2", other));
  const day = other.map((m) => (m.role === "system" ? { ...m, content: m.content.replace(/Current date: \d{4}-\d\d-\d\d/, "Current date: 2026-10-01") } : m));
  steps.push(await run("nextday", day));
  steps.push(await run("turn2b", [...follow, { role: "assistant", content: turn2.text }, { role: "user", content: "Thanks. Reply with OK." }]));
  return { steps, status: room.status() };
}

// the wedge scenario's second node, in its own process; -> { proc, said: [lines], exited }
function spawnJoiner(code, name, gb) {
  const j = { said: [], exited: null };
  j.proc = spawn(process.execPath, [path.join(ROOT, "packages/room-node/join.mjs"), code, "--gb", String(gb), "--name", name,
    ...(SIGNAL ? ["--signal", SIGNAL] : []), "--models", MODELS], { stdio: ["ignore", "pipe", "pipe"] });
  for (const st of [j.proc.stdout, j.proc.stderr]) st.on("data", (b) => String(b).split("\n").filter((l) => l && !/^(TU|MESA|Warning)/.test(l)).forEach((l) => { j.said.push(l); log(`[${name}] ` + l); }));
  j.proc.on("exit", (c, sig) => { j.exited = sig || c; });
  return j;
}
const waitFor = async (f, ms, what) => { const t0 = Date.now(); while (!f()) { if (Date.now() - t0 > ms) throw new Error("timed out: " + what); await new Promise((r) => setTimeout(r, 200)); } };
async function wedgeRun() {
  const DELAY = +(process.env.HOST_DELAY_MS || 40000), BACK = process.env.BACK || "after";
  const GRACE = +(process.env.GRACE_MS || (BACK === "none" ? 15000 : 60000));
  const HOST_GB = +(process.env.HOST_GB || (BACK === "none" ? 2.5 : NODE_GB));
  const checks = out.checks = [];
  const check = (name, ok, extra = "") => { checks.push({ name, ok: !!ok }); log(ok ? "PASS" : "FAIL", name, ok ? "" : String(extra).slice(0, 300)); };
  host = await createRoom({ model: MODEL, pledgeGB: HOST_GB, name: "node-host", signal: SIGNAL, modelDir: MODELS, log: nodeLog("host") });
  host.rejoinGraceMs = GRACE;
  // a slow host (the 35B MoE on a PC: ~5 min): its own load waits DELAY first
  let hostLoaded = false;
  host.loadShardFn = async (o) => { await new Promise((r) => setTimeout(r, DELAY)); const r = await loadShard(o); hostLoaded = true; return r; };
  const ask1 = async () => { const r = { text: "", err: null }; for await (const ev of host.ask([{ role: "user", content: PROMPT }], { maxTokens: 24, temperature: 0 })) { if (ev.type === "token") r.text += ev.text; if (ev.type === "done" && ev.reason === "error") r.err = ev.err; } return r; };
  let j = spawnJoiner(host.code, "node-b", NODE_GB);
  await waitFor(() => host.gpuPeers().length >= 1, 60000, "node-b joins");
  const p = host.start(MODEL, { minDevices: 2, waitMs: 60000 }); p.catch(() => {});
  await waitFor(() => j.said.some((l) => /holding layers/.test(l)), 120000, "node-b loads its layers");
  check("the joiner holds its layers while the host still loads", !hostLoaded);
  log(`stopping node-b (${process.env.KILL === "9" ? "SIGKILL" : "SIGTERM"}) while the host loads`);
  j.proc.kill(process.env.KILL === "9" ? "SIGKILL" : "SIGTERM");
  await waitFor(() => j.exited != null, 20000, "node-b exits");
  await waitFor(() => hostLoaded, DELAY + 120000, "the host's load");
  await new Promise((r) => setTimeout(r, 3000));
  const st1 = host.status();
  out.afterLoad = { online: st1.online, short: st1.short, engine: !!host.ai.engine, chain: host.ai.chain.length };
  check("after its load the host is not online without the device that left", !st1.online, JSON.stringify(out.afterLoad));
  if (BACK === "none") {
    await waitFor(() => host.status().short || host.status().online, GRACE + 60000, "the re-deal");
    const st2 = host.status();
    check("past the grace the re-deal finds the room short and stops (not online, nothing loaded)", st2.short && !st2.online && !host.ai.engine, JSON.stringify(st2));
    const a = await ask1();
    check("an ask meanwhile says the room is short (not 'still loading')", a.err && /GB short/.test(a.err) && !/still loading/.test(a.err), JSON.stringify(a));
  }
  j = spawnJoiner(host.code, "node-b", NODE_GB);
  if (BACK === "none") {   // pooled host / the plugin start again once a device joins a short room
    await waitFor(() => host.gpuPeers().length >= 1, 60000, "node-b joins again");
    host.start(MODEL).catch((e) => log("start again failed: " + e.message));
  }
  await waitFor(() => host.status().online, 180000, "the room online again").catch((e) => log(e.message));
  check("node-b back: it is dealt its layers again", j.said.some((l) => /holding layers/.test(l)), j.said.slice(-6).join(" | "));
  const a = await ask1();
  out.node = a;
  check("the room answers", a.text.length > 0 && !a.err, JSON.stringify(a));
  check("with both devices", host.status().online && host.ai.chain.length === 1, JSON.stringify(host.status().split));
  joiner = j.proc;
  return checks.every((c) => c.ok);
}

try {
  if (MODE === "wedge") { const ok = await wedgeRun(); await finish(ok ? 0 : 1); await new Promise(() => {}); }   // (finish exits)
  else if (MODE === "cache") {
    const SETUP = process.env.SETUP || "solo";
    host = await createRoom({ model: MODEL, pledgeGB: NODE_GB, name: "node-host", signal: SIGNAL, modelDir: MODELS, log: nodeLog("host"), ckpt: CKPT, ctx: CTX });
    if (SETUP === "pair") worker = await joinRoom(host.code, { pledgeGB: NODE_GB, name: "node-b", signal: SIGNAL, modelDir: MODELS, log: nodeLog("b") });
    if (SETUP === "proc") {   // the second node in its own process (its own JS thread and GPU device, as on two machines)
      joiner = spawn(process.execPath, [path.join(ROOT, "packages/room-node/join.mjs"), host.code, "--gb", String(NODE_GB), "--name", "node-b",
        ...(SIGNAL ? ["--signal", SIGNAL] : []), "--models", MODELS], { stdio: ["ignore", "pipe", "pipe"] });
      for (const s of [joiner.stdout, joiner.stderr]) s.on("data", (b) => String(b).split("\n").filter((l) => l && !/^(TU|MESA|Warning)/.test(l)).forEach((l) => log("[b] " + l)));
      const tJoin = Date.now();
      while (host.gpuPeers().length < 1) { if (Date.now() - tJoin > 60000) throw new Error("the second node never joined"); await new Promise((r) => setTimeout(r, 200)); }
    }
    if (SETUP === "tab") {
      await openTab("tab", TAB_GB);
      await page.fill("#code-input", host.code); await page.click("#join-btn");
      const tJoin = Date.now();
      while (host.gpuPeers().length < 1) { if (Date.now() - tJoin > 60000) throw new Error("the tab never joined"); await new Promise((r) => setTimeout(r, 200)); }
    }
    const tStart = Date.now();
    await host.start(MODEL, { minDevices: SETUP === "solo" ? 1 : 2, waitMs: 60000 });
    out.onlineS = (Date.now() - tStart) / 1000;
    out.split = host.split; out.ckpt = CKPT !== false; out.setup = SETUP; out.ctx = host.ctxMax();
    out.cache = await cacheRun(host);
    if (page) out.tab = await tabStatus();
  } else if (MODE === "nodehost") {
    host = await createRoom({ model: MODEL, pledgeGB: NODE_GB, name: "node-host", signal: SIGNAL, modelDir: MODELS, log: nodeLog("host"), chatMaxNew: MAXNEW });
    log("room", host.code);
    await openTab("tab", TAB_GB);
    await page.fill("#code-input", host.code); await page.click("#join-btn");
    const tJoin = Date.now();
    while (host.gpuPeers().length < 1) { if (Date.now() - tJoin > 60000) throw new Error("the tab never joined"); await new Promise((r) => setTimeout(r, 200)); }
    out.joinS = (Date.now() - tJoin) / 1000;
    const tStart = Date.now();
    host.on("progress", (p) => { if (p.pct % 25 === 0) log(`[host] ${p.name} loading ${p.pct}%`); });
    await host.start(MODEL);
    out.onlineS = (Date.now() - tStart) / 1000;
    out.split = host.split;
    out.node = await ask(host);
    out.sent = host.sent;
    // a tool call through the node host (the serve v2 path: grammar, call parsing), then its result
    if (process.env.TOOLS !== "0") out.tools = await toolTurns(host);
    // the tab asks too (the room's chat path): the answer streams to the tab's screen
    await page.fill("#ai-prompt", PROMPT); await page.click("#ai-send");
    const chat = await new Promise((res) => host.once("chatanswer", res));
    out.tabAsk = { reply: chat.reply, stats: chat.stats };
    await page.waitForTimeout(500);
    out.tabShows = await page.evaluate(() => [...document.querySelectorAll(".m.bot .bubble")].pop()?.textContent || "");
    out.tab = await tabStatus();
  } else if (MODE === "tabhost") {
    await openTab("tab-host", TAB_GB);
    await page.click("#create-btn");
    // (six-character codes show as "ABC-DEF"; older four-character ones as they are)
    await page.waitForFunction(() => /[A-Z0-9]{3}-?[A-Z0-9]{3}/.test(document.getElementById("side-code").textContent), null, { timeout: 30000 });
    const roomCode = (await page.textContent("#side-code")).trim().replace("-", "").match(/[A-Z0-9]{4,6}/)[0];
    log("tab room", roomCode);
    // NODE_RAM: the node offers RAM for experts (expert offload: the tab's deal gives it what the pledges can't hold)
    worker = await joinRoom(roomCode, { pledgeGB: NODE_GB, ramGB: +(process.env.NODE_RAM || 0), mem: MEM, name: "node-worker", signal: SIGNAL, modelDir: MODELS, log: nodeLog("worker") });
    await page.waitForFunction(() => document.querySelectorAll(".peer-card").length >= 2, null, { timeout: 60000 });
    await page.waitForTimeout(1500);
    await page.selectOption("#ai-model", MODEL);
    await page.evaluate(() => { const s = document.getElementById("ai-sampling"); s.value = "exact"; s.dispatchEvent(new Event("change")); });
    const tStart = Date.now();
    await page.click("#ai-start");
    await page.waitForFunction(() => document.getElementById("ai-panel").classList.contains("online") || /^failed:/.test(document.getElementById("ai-status").textContent), null, { timeout: +(process.env.ONLINE_MS || 600000), polling: 1000 }).catch(async (e) => { out.tabAtFail = await tabStatus(); throw e; });
    out.onlineS = (Date.now() - tStart) / 1000;
    out.tabSplit = (await tabStatus()).log;
    if (/^failed:/.test((await tabStatus()).status)) throw new Error("tab start " + (await tabStatus()).status);
    let text = "", stats = null;
    // the page picks the model host (room/plan.js pickModelHost): the tab when it pledged more, else
    // this node, which then deals the layers itself (ai-start-req) and answers the tab's question
    out.nodeWasModelHost = !!worker.modelHost;
    const got = new Promise((res) => {
      worker.on("chat", (d) => { if (d.t === "ai-token") text += d.text; if (d.t === "ai-gendone") { stats = d.stats; res(); } });
      worker.on("chatanswer", (c) => { text = c.reply; stats = c.stats; res(); });
    });
    await page.evaluate(() => { const s = document.getElementById("ai-sampling"); s.value = "exact"; s.dispatchEvent(new Event("change")); });
    await page.fill("#ai-prompt", PROMPT); await page.click("#ai-send");
    await got;
    out.node = { text, stats, sent: worker.sent, frames: worker.frames, frameMsAvg: worker.frames ? +(worker.frameMs / worker.frames).toFixed(2) : null, range: worker.ai.range, split: worker.split };
    out.tab = await tabStatus();
  } else if (MODE === "nodepair") {
    host = await createRoom({ model: MODEL, pledgeGB: NODE_GB, ramGB: +(process.env.HOST_RAM || 0), mem: MEM, name: "node-a", signal: SIGNAL, modelDir: MODELS, log: nodeLog("a") });
    // WORKER_GB / WORKER_RAM: the worker's own pledge, and RAM for experts (expert offload: it holds what the pledges can't)
    worker = await joinRoom(host.code, { pledgeGB: +(process.env.WORKER_GB || NODE_GB), ramGB: +(process.env.WORKER_RAM || 0), mem: MEM, name: "node-b", signal: SIGNAL, modelDir: MODELS, log: nodeLog("b") });
    const tStart = Date.now();
    await host.start(MODEL, { minDevices: 2, waitMs: 30000 });
    out.onlineS = (Date.now() - tStart) / 1000;
    out.split = host.split;
    out.node = await ask(host);
    out.second = await ask(host);   // again: the caches' prefix is reused
    out.workerFrames = worker.frames; out.sent = { host: host.sent, worker: worker.sent };
  }
  if (MODE === "auto") {   // createRoom + ask: no start() call, the ask deals the layers (solo here)
    host = await createRoom({ model: MODEL, pledgeGB: NODE_GB, name: "node-host", signal: SIGNAL, modelDir: MODELS, log: nodeLog("host") });
    out.node = await ask(host);
  }
  if (MODE === "api") {
    // `pooled serve`'s bridge (cli/lib/room.js) in the node-hosted room: the API-client path of docs/protocol.md
    host = await createRoom({ model: MODEL, pledgeGB: NODE_GB, name: "node-host", signal: SIGNAL, modelDir: MODELS, log: nodeLog("host") });
    await host.start(MODEL);
    const { Bridge } = await import("../../../cli/lib/room.js");
    const { setupNode } = await import("../index.js");
    const br = new Bridge({ code: host.code, signal: SIGNAL, name: "serve", client: "e2e", log: (s) => log("[bridge] " + s), Peer: (await setupNode()).Peer });
    await br.connect();
    while (!br.ready) await new Promise((r) => setTimeout(r, 100));
    let text = "", t0 = performance.now(), first = null;
    const done = await new Promise((res) => br.ask("r1", { system: "", messages: [{ role: "user", text: PROMPT }], params: { maxTokens: MAXNEW, temperature: 0, client: "e2e" } }, (m) => {
      if (m.t === "ai-token") { if (first == null) first = performance.now() - t0; text += m.text; }
      if (m.t === "ai-gendone" || m.t === "ai-busy" || m.t === "x-fail") res(m);
    }));
    out.node = { text, ttftMs: Math.round(first), reason: done.reason, usage: done.usage, stats: done.stats, err: done.err || done.why };
    out.bridge = { model: br.model, label: br.modelLabel };
    await br.leave();
  }
  if (process.env.REF) {
    const ref = JSON.parse(fs.readFileSync(process.env.REF, "utf8"));
    const a = out.node?.text ?? "";
    let k = 0; while (k < a.length && a[k] === ref.text[k]) k++;
    out.matchSolo = a === ref.text;
    out.commonPrefixChars = k; out.refChars = ref.text.length;
    if (out.tabAsk) out.tabMatchSolo = out.tabAsk.reply === ref.text;
    log(out.matchSolo ? "MATCH solo" : `DIFFERS from solo after ${k} chars`);
  }
  await finish(0);
} catch (e) {
  out.error = String(e?.stack || e).slice(0, 800); log("FAILED", out.error);
  try { if (page) out.tab = await tabStatus(); } catch {}
  await finish(1);
}
