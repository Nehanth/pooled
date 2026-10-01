// What a device's screen shows when it joins, reloads or comes back on a new link: the real room
// (p2p.html + room.js) in headless Chromium, a local PeerServer, real WebRTC between the tabs, and the
// synthetic model (tests/e2e/synth.mjs) standing in for "qwen3.8-27b", so no GPU and no network are
// needed (SwiftShader; E2E_GPU=real uses the machine's GPU, as on the GB10). Every tab is its own
// browser context (its own sessionStorage, as separate devices have).
//
//   node tests/e2e/room_state.mjs                         every scenario
//   node tests/e2e/room_state.mjs --only leftout,rejoin   some of them
//
//   leftout        a laptop and a phone-size window join a room whose creator holds the whole model
//                  ("For speed", the default split): neither is dealt layers. Each shows the chat (not
//                  a Loading card stuck at 0%) with the "Not needed" note and gets an answer; the
//                  phone-size one reloads and comes back the same way (not to the model picker).
//                  (starter-kit run, 2026-09-30)
//   leftout-guest  the same with the laptop lending the most, so the laptop runs the model and the
//                  room's creator is left out too: the model host links every device before it
//                  deals, and welcomes a device that reloads.
//   rejoin         a worker's link drops mid-answer and its replacement link reaches the host before
//                  the host sees the old one close (the host's close handling is held back 8 s): its
//                  input comes back, and it can ask (#265).
//   duplicate      a worker reloads without saying "leaving" (its tab killed) while the host's close
//                  of the old link is held back: the host lists it once, not beside itself, so a
//                  re-deal can't deal it twice (#22).
//
// In CI's browser job. Locally: npm install at the repo root (playwright, peer); in a worktree,
// symlink node_modules (ES modules ignore NODE_PATH).
import fs from "fs";
import os from "os";
import path from "path";
import { spawn } from "child_process";
import { writeSynth } from "./synth.mjs";
import { loadPlaywright, chromiumPath, GPU_ARGS, serveRepo } from "./engine_synth.mjs";

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const ONLY = new Set((arg("only", "leftout,leftout-guest,rejoin,duplicate")).split(","));
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");
const PORT = +arg("port", 8191), SIG = PORT + 1;
const MODEL_KEY = "qwen3.8-27b";
const MODEL_URL = "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_0.gguf";
const PEERJS_URL = "https://cdn.jsdelivr.net/npm/peerjs@1.5.4/dist/peerjs.min.js";
const UA_PHONE = "Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1";

const t0 = Date.now();
const log = (...a) => console.error(((Date.now() - t0) / 1000).toFixed(1).padStart(6) + "s", ...a);
const results = [];
const check = (name, ok, extra = "") => { results.push({ name, ok: !!ok }); log(ok ? "PASS" : "FAIL", name, ok ? "" : String(extra).slice(0, 400)); };

function findPkg(name) {
  for (const d of [path.join(ROOT, "node_modules"), path.join(ROOT, "cli/node_modules"), ...(process.env.NODE_PATH || "").split(path.delimiter).filter(Boolean)]) {
    if (fs.existsSync(path.join(d, name, "package.json"))) return path.join(d, name);
  }
  return null;
}

// ---- the page, served from the checkout, with two test hooks
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-room-state-"));
const modelBytes = fs.readFileSync(writeSynth(path.join(tmp, "synth.gguf"), { seed: 1, eosAt: 40 }).file);
const extra = {};
// a fixed cooperative-GEMV pick (the real autotune takes minutes on SwiftShader), as room_synth.mjs
extra["/engine/autotune.js"] = path.join(tmp, "autotune.js");
fs.writeFileSync(extra["/engine/autotune.js"], "export async function autotuneCoop() { return { wg: 64, rows: 4, results: [], stub: true }; }\n");
// window.__delayClose = ms: PeerJS's "close" for a link reaches the room that much later (what a phone's
// dead link does), so a replacement link can come in first. The anchor fails loudly if room.js changes.
const room = fs.readFileSync(path.join(ROOT, "room.js"), "utf8");
const ANCHOR = '  conn.on("close", onClose);\n';
if (!room.includes(ANCHOR)) { console.error(`room.js has no ${JSON.stringify(ANCHOR)}: update tests/e2e/room_state.mjs`); process.exit(2); }
extra["/room.js"] = path.join(tmp, "room.js");
// window.__dropLinks(): this device drops every link, the path a screen back from a lock takes
fs.writeFileSync(extra["/room.js"], room.replace(ANCHOR, '  conn.on("close", () => { if (window.__delayClose) setTimeout(onClose, window.__delayClose); else onClose(); });\n')
  + '\nwindow.__dropLinks = () => { for (const id of [...conns.keys()]) dropLink(id, "e2e: the link dropped"); };\n');
const srv = serveRepo(PORT, extra);
const peerDir = findPkg("peer");
if (!peerDir) { console.error("the peer package is not installed: npm install at the repo root"); process.exit(2); }
const sig = spawn(process.execPath, [path.join(peerDir, "dist/bin/peerjs.js"), "--port", String(SIG), "--host", "127.0.0.1", "--path", "/"], { stdio: "ignore" });
const peerjsDir = findPkg("peerjs");
const peerjsJs = peerjsDir ? fs.readFileSync(path.join(peerjsDir, "dist/peerjs.min.js")) : null;
for (let i = 0; ; i++) {
  if (await fetch(`http://127.0.0.1:${SIG}/peerjs/id`).then((r) => r.ok, () => false)) break;
  if (i > 50) { console.error("PeerServer did not start"); process.exit(2); }
  await new Promise((r) => setTimeout(r, 200));
}
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromiumPath(), args: [...GPU_ARGS, "--allow-loopback-in-peer-connection", "--disable-features=WebRtcHideLocalIpsWithMdns"] });

const ctxs = [];
async function device(name, { phone = false, size = null } = {}) {
  const ctx = await browser.newContext({ viewport: size || (phone ? { width: 390, height: 844 } : { width: 1280, height: 900 }), ...(phone ? { userAgent: UA_PHONE, isMobile: true, hasTouch: true } : {}), ignoreHTTPSErrors: true });
  ctxs.push(ctx);
  await ctx.route("**/*", async (route) => {
    const req = route.request(), url = req.url().split("?")[0];
    if (url.startsWith(`http://127.0.0.1:${PORT}/`) || url.startsWith(`http://127.0.0.1:${SIG}/`)) return route.continue();
    if (url === PEERJS_URL) return peerjsJs ? route.fulfill({ status: 200, contentType: "text/javascript", body: peerjsJs }) : route.continue();
    if (url === MODEL_URL) {
      const cors = { "access-control-allow-origin": "*", "access-control-expose-headers": "content-range, content-length, accept-ranges" };
      const m = /bytes=(\d+)-(\d*)/.exec((await req.allHeaders()).range || "");
      if (!m) return route.fulfill({ status: 200, headers: { ...cors, "accept-ranges": "bytes" }, body: modelBytes });
      const lo = +m[1], hi = m[2] ? Math.min(+m[2], modelBytes.length - 1) : modelBytes.length - 1;
      return route.fulfill({ status: 206, headers: { ...cors, "content-type": "application/octet-stream", "accept-ranges": "bytes", "content-range": `bytes ${lo}-${hi}/${modelBytes.length}` }, body: modelBytes.subarray(lo, hi + 1) });
    }
    if (url.startsWith("https://fonts.googleapis.com/")) return route.fulfill({ status: 200, contentType: "text/css", body: "" });
    return route.abort();
  });
  await ctx.addInitScript((n) => { try { if (!sessionStorage.getItem("pooled-name")) sessionStorage.setItem("pooled-name", n); } catch {} }, name);
  const p = await ctx.newPage();
  p.label = name; p.errs = [];
  p.on("pageerror", (e) => p.errs.push(String(e).slice(0, 200)));
  return p;
}
const wired = (p) => p.waitForFunction(() => typeof Peer === "function" && window.pooledWired, null, { timeout: 60000 });
// a room: devices [[page, name, GB]], the first creates it; query: for the creator's URL
async function openRoom(devs, query = "") {
  const base = `http://127.0.0.1:${PORT}/p2p.html?signal=127.0.0.1:${SIG}&maxnew=24`;
  for (const [p, , , q = ""] of devs) { await p.goto(base + q + (p === devs[0][0] ? "&ask=0" + query : "")); await wired(p); }
  for (const [p, name, gb] of devs) {
    await p.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
    await p.fill("#name-input", name); await p.fill("#join-gb", String(gb));
  }
  const host = devs[0][0];
  await host.click("#create-btn");
  await host.waitForFunction(() => /^[A-Z0-9]{3}-[A-Z0-9]{3}$/.test(document.getElementById("side-code").textContent), null, { timeout: 30000 });
  const code = (await host.textContent("#side-code")).trim().replace("-", "");
  for (const [p] of devs.slice(1)) { await p.fill("#code-input", code); await p.click("#join-btn"); }
  for (const [p] of devs) await p.waitForFunction((k) => document.querySelectorAll(".peer-card").length >= k, devs.length, { timeout: 60000 });
  await host.waitForTimeout(1500);
  log("room", code, devs.map(([, n, g]) => `${n} ${g} GB`).join(", "));
  return code;
}
async function startModel(host, split) {
  await host.evaluate(([m, sp]) => {
    for (const [id, v] of [["ai-model", m], ["ai-sampling", "exact"], ...(sp ? [["ai-split", sp]] : [])]) { const s = document.getElementById(id); s.value = v; s.dispatchEvent(new Event("change", { bubbles: true })); }
  }, [MODEL_KEY, split]);
  await host.waitForFunction(() => !document.getElementById("ai-start").disabled, null, { timeout: 20000 });
  await host.click("#ai-start");
}
const online = (p, ms = 300000) => p.waitForFunction(() => document.getElementById("ai-panel").classList.contains("online"), null, { timeout: ms }).then(() => true, () => false);
const chatMode = (p) => p.evaluate(() => { const c = document.getElementById("mode-chat"); if (c && c.getAttribute("aria-selected") !== "true") c.click(); });
// what a tab shows
const view = (p) => p.evaluate(() => {
  const $ = (id) => document.getElementById(id);
  return {
    loading: $("load-card").classList.contains("on"),
    online: $("ai-panel").classList.contains("online"),
    input: $("ai-row").style.display !== "none" && $("ai-prompt").getClientRects().length > 0,
    picker: !$("ai-start").disabled && $("ai-start").getClientRects().length > 0,
    inRoom: document.body.classList.contains("in-room"),
    status: $("ai-status").textContent.slice(0, 120),
    note: [...document.querySelectorAll("#ai-output .sys, #ai-empty-t")].map((e) => e.textContent).find((t) => /Not needed/.test(t)) || "",
    cards: [...document.querySelectorAll("#peers .peer-card")].map((c) => c.dataset.name),
  };
});
const isModelHost = (p) => p.evaluate(() => /^cluster online · \d+ device/.test(document.getElementById("ai-status").textContent));
// ask from a tab (its own Send) and wait for the answer on it
async function ask(p, text, ms = 180000) {
  await chatMode(p);
  const n = await p.evaluate(() => document.querySelectorAll("#ai-output .m.bot .stats").length);
  await p.evaluate((t) => { const b = document.getElementById("ai-prompt"); b.value = t; b.dispatchEvent(new Event("input")); document.getElementById("ai-send").click(); }, text);
  return p.waitForFunction((k) => document.querySelectorAll("#ai-output .m.bot .stats").length > k, n, { timeout: ms, polling: 250 })
    .then(() => p.evaluate(() => [...document.querySelectorAll("#ai-output .m.bot")].pop()?.querySelector(".bubble")?.textContent || "x"), () => "");
}
const closeAll = async () => { for (const c of ctxs.splice(0)) await c.close().catch(() => {}); };

// ---------------------------------------------------------------- scenarios
async function leftout(label, gbs) {
  const desk = await device("desk"), laptop = await device("laptop"), phone = await device("phone", { size: { width: 390, height: 844 } });
  const tabs = { desk, laptop, phone };
  await openRoom([[desk, "desk", gbs[0]], [laptop, "laptop", gbs[1]], [phone, "phone", gbs[2]]]);
  await startModel(desk, null);   // the room's default split: For speed
  check(`${label}: the room's default split is For speed`, (await desk.evaluate(() => document.getElementById("ai-split").value)) === "speed");
  // whichever device runs the model (room/plan.js pickModelHost: the biggest pledge) holds it all
  const boss = await Promise.any(Object.entries(tabs).map(([n, p]) => p.waitForFunction(() => /^cluster online · \d+ device/.test(document.getElementById("ai-status").textContent), null, { timeout: 300000 }).then(() => n))).catch(() => null);
  log(label, "model host:", boss);
  check(`${label}: the model starts`, !!boss);
  if (!boss) return;
  const out = Object.entries(tabs).filter(([n]) => n !== boss);
  for (const [n, p] of out) {
    const up = await online(p, 30000);
    const v = await view(p);
    check(`${label}: ${n} (left out) is not stuck on the Loading card`, up && !v.loading, JSON.stringify(v));
    check(`${label}: ${n} shows the chat and its input`, v.online && v.input, JSON.stringify(v));
    check(`${label}: ${n} says why it holds no layers`, /Not needed for this model/.test(v.note) && /can still ask/.test(v.note), JSON.stringify(v));
  }
  const a1 = await ask(phone, "Hello from the phone.");
  check(`${label}: the phone-size tab's question is answered`, a1.length > 0);
  // a reload: back in the room in the same state (the chat and the note), not the model picker
  await phone.reload(); await wired(phone);
  const up = await online(phone, 60000);
  await phone.waitForTimeout(1500);
  const v = await view(phone);
  check(`${label}: after a reload the phone-size tab is back with the chat, not the picker`, up && v.inRoom && v.input && !v.picker && !v.loading, JSON.stringify(v));
  check(`${label}: after a reload it still says why it holds no layers`, /Not needed for this model/.test(v.note), JSON.stringify(v));
  const a2 = await ask(phone, "And again after a reload.");
  check(`${label}: after a reload its question is answered`, a2.length > 0);
  const hv = await view(tabs[boss]);
  check(`${label}: the room lists each device once`, hv.cards.length === 3 && new Set(hv.cards).size === 3, JSON.stringify(hv.cards));
}

async function rejoin() {
  const host = await device("host"), worker = await device("worker", { phone: true });
  await openRoom([[host, "host", 12], [worker, "worker", 6]], "&split=memory&phonelayers=1");
  await startModel(host, "memory");
  const up = (await online(host)) && (await online(worker, 60000));
  check("rejoin: the room is online with the worker holding layers", up && (await worker.evaluate(() => !!document.getElementById("ai-status").textContent.match(/serving layers/))), (await view(worker)).status);
  if (!up) return;
  await chatMode(worker);
  check("rejoin: the worker's input shows", (await view(worker)).input);
  // an answer in flight, then the worker's link drops; the host sees the close only 8 s later
  await host.evaluate(() => { const b = document.getElementById("ai-prompt"); b.value = "Write a story about a ship."; b.dispatchEvent(new Event("input")); document.getElementById("ai-send").click(); });
  await host.waitForFunction(() => /generating|prefill/.test(document.getElementById("ai-status").textContent), null, { timeout: 60000 }).catch(() => {});
  await host.evaluate(() => { window.__delayClose = 8000; });
  await worker.evaluate(() => window.__dropLinks());
  await worker.waitForFunction(() => !document.getElementById("room-over").hidden, null, { timeout: 15000 }).catch(() => {});
  await worker.waitForFunction(() => document.getElementById("room-over").hidden, null, { timeout: 60000 }).catch(() => {});
  await host.waitForTimeout(9500);   // past the held-back close
  await host.waitForFunction(() => !/generating|prefill/.test(document.getElementById("ai-status").textContent), null, { timeout: 120000 }).catch(() => {});
  await host.evaluate(() => { window.__delayClose = 0; });
  await host.waitForTimeout(1000);
  const v = await view(worker);
  check("rejoin: after a replacement link the worker's input is back", v.online && v.input, JSON.stringify(v));
  const a = await ask(worker, "Are you back?");
  check("rejoin: and it can ask", a.length > 0, JSON.stringify(await view(worker)));
}

async function duplicate() {
  const host = await device("host"), worker = await device("worker");
  await openRoom([[host, "host", 12], [worker, "worker", 6]], "&split=memory");
  await startModel(host, "memory");
  const up = (await online(host)) && (await online(worker, 60000));
  check("duplicate: the room is online", up);
  if (!up) return;
  await host.evaluate(() => { window.__delayClose = 20000; });
  // its tab is killed: no "leaving" (the pagehide listeners do not run), and the host hears the close late
  await worker.evaluate(() => { addEventListener("pagehide", (e) => e.stopImmediatePropagation(), true); });
  await worker.reload(); await wired(worker);
  await online(worker, 60000);
  await host.waitForTimeout(4000);   // the name probe (1.5 s) and the re-seat
  const hv = await view(host);
  check("duplicate: the host lists the worker once (not beside its old link)", hv.cards.length === 2 && !hv.cards.some((n) => / 2$/.test(n)), JSON.stringify(hv.cards));
  const wv = await view(worker);
  check("duplicate: the worker keeps its name", (await worker.evaluate(() => document.getElementById("name-input").value)) === "worker", JSON.stringify(wv.cards));
  await host.evaluate(() => { window.__delayClose = 0; });
}

let code = 1;
try {
  for (const [name, fn] of [["leftout", () => leftout("leftout", [12, 8, 8])], ["leftout-guest", () => leftout("leftout-guest", [8, 12, 8])], ["rejoin", rejoin], ["duplicate", duplicate]]) {
    if (!ONLY.has(name)) continue;
    log("---", name);
    try { await fn(); } catch (e) { check(`${name}: ran to the end`, false, String(e).slice(0, 300)); }
    await closeAll();
  }
  code = results.length && results.every((r) => r.ok) ? 0 : 1;
} finally {
  const fail = results.filter((r) => !r.ok).map((r) => r.name);
  console.log(JSON.stringify({ ok: code === 0, pass: results.length - fail.length, fail }));
  await browser.close().catch(() => {}); srv.close(); sig.kill(); fs.rmSync(tmp, { recursive: true, force: true });
  process.exit(code);
}
