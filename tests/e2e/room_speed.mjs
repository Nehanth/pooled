// The room's speed panel (room/speedo.js, renderSpeed in room.js), the demo's "the model gets faster as each
// device joins": the real room (p2p.html + room.js) in headless Chromium, a local PeerServer, real WebRTC
// between the tabs and the synthetic model (tests/e2e/synth.mjs) standing in for "qwen3.8-27b", so no GPU
// and no network are needed (SwiftShader; E2E_GPU=real uses the machine's GPU). As room_state.mjs.
//
//   node tests/e2e/room_speed.mjs                    the checks
//   node tests/e2e/room_speed.mjs --shots DIR        and screenshots of each step in DIR (1920x1080, 390x844)
//
// Two rooms. One: a "mac" runs the model alone and answers twice: the readout shows the speed live ("Writing about
// N words a second") and then the last answer's ("Last answer: …"), the history two bars marked "1 device", one
// device card that runs the whole model, the event line "Running on 1 device"; a "pc" joins after the start: "pc
// joined", its card waits for the model to be shared out again. Two: the mac and the pc from the start, split by
// pledge: "Running on 2 devices", two cards with how much of the model they hold and their GB (their layers and part
// of each token's time in the card's tooltip), bars marked "2 devices" on both screens. The exact numbers (tok/s,
// layers, ms) are in tooltips for a visitor (dev=0); the bar of one token's time (#sp-share) is the dev view's
// (dev=1), checked on a second tab of the host's room. A phone-size tab joins: its card says it asks only, and its own screen
// shows the readout and the room's history. (The re-deal itself is not run: see the note in the body.)
// Three: the mac and a room node (`pooled join`, packages/room-node, in this process on Dawn: the machine's GPU,
// the synthetic model from a local model dir) as the pc: the node's card has its part of each token's time from
// its ai-tele (room/tele.js), not "no timing from this device". NODE=0 skips it (no Dawn / GPU here).
import fs from "fs";
import os from "os";
import path from "path";
import { spawn } from "child_process";
import { writeSynth } from "./synth.mjs";
import { loadPlaywright, chromiumPath, GPU_ARGS, serveRepo } from "./engine_synth.mjs";

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const SHOTS = arg("shots", "");
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");
const PORT = +arg("port", 8195), SIG = PORT + 1;
const MODEL_KEY = "qwen3.8-27b";
const MODEL_URL = "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_0.gguf";
const PEERJS_URL = "https://cdn.jsdelivr.net/npm/peerjs@1.5.4/dist/peerjs.min.js";
const UA_PHONE = "Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1";

const NODE = process.env.NODE !== "0";
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

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-room-speed-"));
const modelBytes = fs.readFileSync(writeSynth(path.join(tmp, "synth.gguf"), { seed: 1, eosAt: 400 }).file);
// the room node reads the model from disk: <modelDir>/<model key>/<file name> (packages/room-node/source.js)
const modelDir = path.join(tmp, "models");
fs.mkdirSync(path.join(modelDir, MODEL_KEY), { recursive: true });
fs.writeFileSync(path.join(modelDir, MODEL_KEY, MODEL_URL.split("/").pop()), modelBytes);
const extra = { "/engine/autotune.js": path.join(tmp, "autotune.js") };
fs.writeFileSync(extra["/engine/autotune.js"], "export async function autotuneCoop() { return { wg: 64, rows: 4, results: [], stub: true }; }\n");
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
const SOFT = process.env.E2E_GPU !== "real";
const browser = await chromium.launch({ executablePath: chromiumPath(), ...(SOFT && !chromiumPath() ? { channel: "chromium" } : {}),
  args: [...GPU_ARGS, ...(SOFT ? ["--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--use-vulkan=swiftshader", "--disable-vulkan-surface", "--ignore-gpu-blocklist"] : []), "--allow-loopback-in-peer-connection", "--disable-features=WebRtcHideLocalIpsWithMdns"] });

const ctxs = [];
async function device(name, { phone = false, size = null } = {}) {
  const ctx = await browser.newContext({ viewport: size || (phone ? { width: 390, height: 844 } : { width: 1920, height: 1080 }), deviceScaleFactor: phone ? 2 : 1,
    ...(phone ? { userAgent: UA_PHONE, isMobile: true, hasTouch: true } : {}), ignoreHTTPSErrors: true });
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
// dev=0: the page as a visitor sees it (signal= turns the dev extras on)
const base = `http://127.0.0.1:${PORT}/p2p.html?signal=127.0.0.1:${SIG}&maxnew=1200&dev=0`;
const wired = (p) => p.waitForFunction(() => typeof Peer === "function" && window.pooledWired, null, { timeout: 60000 });
async function enter(p, name, gb, q = "") {
  // (q with dev=1: the developer view, in place of the visitor's dev=0)
  await p.goto(/[?&]dev=1\b/.test(q) ? (base + q).replace("&dev=0", "") : base + q); await wired(p);
  await p.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
  await p.fill("#name-input", name); await p.fill("#join-gb", String(gb));
}
const online = (p, ms = 300000) => p.waitForFunction(() => document.getElementById("ai-panel").classList.contains("online"), null, { timeout: ms }).then(() => true, () => false);
// a fresh chat first, so every answer is about as long (the synthetic model's end token comes with the
// context's length, eosAt; maxnew caps it, long enough for a screenshot mid-answer)
const fresh = (p) => p.evaluate(() => { const b = document.getElementById("new-chat"); if (b && !b.hidden && !b.disabled) b.click(); });
async function ask(p, text, ms = 240000) {
  await fresh(p);
  const n = await p.evaluate(() => document.querySelectorAll("#ai-output .m.bot .stats").length);
  await p.evaluate((t) => { const b = document.getElementById("ai-prompt"); b.value = t; b.dispatchEvent(new Event("input")); document.getElementById("ai-send").click(); }, text);
  return p.waitForFunction((k) => document.querySelectorAll("#ai-output .m.bot .stats").length > k, n, { timeout: ms, polling: 250 }).then(() => true, () => false);
}
// what the speed panel shows
const panel = (p) => p.evaluate(() => {
  const q = (s) => document.querySelector(s);
  return {
    shown: !!q("#speedo") && q("#speedo").getClientRects().length > 0,
    // the readout is a sentence; its tooltip has the exact tok/s ("18.3 tok/s · …"), read as the number
    line: q("#sp-tps")?.textContent || "",
    tps: +(/^(\d+(?:\.\d+)?) tok\/s/.exec(q("#sp-tps")?.title || "")?.[1] || 0),
    unit: q("#sp-unit")?.textContent || "",
    model: q("#sp-model")?.textContent || "",
    params: q("#sp-params")?.textContent || "",
    bars: [...document.querySelectorAll("#sp-hist .bar")].map((b) => ({ tps: +b.dataset.tps, n: +b.dataset.n })),
    gain: q("#sp-gain")?.textContent || "",
    events: [...document.querySelectorAll("#sp-events li")].map((e) => e.textContent),
    devs: [...document.querySelectorAll("#sp-devs .dv")].map((d) => ({ name: d.dataset.name, text: d.innerText.replace(/\s+/g, " ").trim(), tip: d.title || "" })),
    share: q("#sp-share")?.innerText.replace(/\s+/g, " ").trim() || "",
  };
});
async function shot(p, file, opts = {}) {
  if (!SHOTS) return;
  fs.mkdirSync(SHOTS, { recursive: true });
  await p.waitForTimeout(400);
  await p.screenshot({ path: path.join(SHOTS, file), ...opts });
  log("screenshot", path.join(SHOTS, file));
}
// a screenshot mid-answer: the readout live
async function shotLive(p, file, text) {
  await fresh(p);
  const n = await p.evaluate(() => document.querySelectorAll("#ai-output .m.bot .stats").length);
  await p.evaluate((t) => { const b = document.getElementById("ai-prompt"); b.value = t; b.dispatchEvent(new Event("input")); document.getElementById("ai-send").click(); }, text);
  const live = await p.waitForFunction(() => document.getElementById("speedo")?.classList.contains("live") && /^Writing /.test(document.getElementById("sp-tps").textContent) && parseFloat(document.getElementById("sp-tps").title) > 0, null, { timeout: 120000, polling: 100 }).then(() => true, () => false);
  if (live) await shot(p, file);
  await p.waitForFunction((k) => document.querySelectorAll("#ai-output .m.bot .stats").length > k, n, { timeout: 240000, polling: 250 }).catch(() => {});
  return live;
}

// a room: the first device creates it, the others join with its code; then the model starts (split by pledge).
// others(code): devices that are not tabs (a room node) join here; it says how many
async function room(devs, others = null) {
  for (const [p, name, gb, q = ""] of devs) await enter(p, name, gb, q + (p === devs[0][0] ? "&ask=0&split=memory" : ""));
  const host = devs[0][0];
  await host.click("#create-btn");
  await host.waitForFunction(() => /^[A-Z0-9]{3}-[A-Z0-9]{3}$/.test(document.getElementById("side-code").textContent), null, { timeout: 30000 });
  const roomCode = (await host.textContent("#side-code")).trim().replace("-", "");
  for (const [p] of devs.slice(1)) { await p.fill("#code-input", roomCode); await p.click("#join-btn"); }
  const nOther = others ? await others(roomCode) : 0;
  for (const [p] of devs) await p.waitForFunction((k) => document.querySelectorAll(".peer-card").length >= k, devs.length + nOther, { timeout: 60000 });
  log("room", roomCode, devs.map(([, n, g]) => `${n} ${g} GB`).join(", "));
  await host.evaluate((m) => {
    for (const [id, v] of [["ai-model", m], ["ai-sampling", "exact"], ["ai-split", "memory"]]) { const s = document.getElementById(id); s.value = v; s.dispatchEvent(new Event("change", { bubbles: true })); }
  }, MODEL_KEY);
  await host.waitForFunction(() => !document.getElementById("ai-start").disabled, null, { timeout: 120000 });
  await host.click("#ai-start");
  return roomCode;
}
const closeAll = async () => { for (const c of ctxs.splice(0)) await c.close().catch(() => {}); };

let code = 1, node = null;
try {
  // ---- one device: the Mac alone
  const mac = await device("mac");
  const code1 = await room([[mac, "mac", 24]]);
  check("1 device: the room is online", await online(mac));
  let v = await panel(mac);
  check("the panel shows the model's name and size", v.shown && /Qwen3\.8/.test(v.model) && /^27B model$/.test(v.params), JSON.stringify(v));
  check("1 device: one device card that holds every layer (its layers in the tooltip)", v.devs.length === 1 && /runs the whole model/.test(v.devs[0].text) && /layers 1–\d+/.test(v.devs[0].tip) && /GB/.test(v.devs[0].text), JSON.stringify(v.devs));
  check("1 device: the event line says Running on 1 device", /Running on 1 device/.test(v.events.at(-1) || ""), JSON.stringify(v.events));
  check("1 device: the readout live while it answers", await shotLive(mac, "1-mac-solo-live-1080p.png", "Tell me about the sea."));
  check("1 device: a second answer", await ask(mac, "And about the mountains?"));
  v = await panel(mac);
  check("1 device: two bars in the history, marked 1 device", v.bars.length === 2 && v.bars.every((b) => b.n === 1 && b.tps > 0), JSON.stringify(v.bars));
  check("1 device: the readout keeps the last answer's speed", v.tps > 0 && /^Last answer: /.test(v.line) && v.unit === "", JSON.stringify(v));
  await shot(mac, "2-mac-solo-1080p.png");
  // the PC joins after the start: the event line says so, its card waits for a re-deal
  const pc = await device("pc");
  await enter(pc, "pc", 12);
  await pc.fill("#code-input", code1); await pc.click("#join-btn");
  await mac.waitForFunction(() => /pc joined/.test(document.getElementById("sp-events").textContent) && document.querySelectorAll("#sp-devs .dv").length === 2, null, { timeout: 60000 }).catch(() => {});
  v = await panel(mac);
  check("the event line says the pc joined", /pc joined/.test(v.events.at(-1) || ""), JSON.stringify(v.events));
  check("its card says it waits for the model to be shared out again", /waits for the model to be shared out again/.test(v.devs.find((d) => d.name === "pc")?.text || ""), JSON.stringify(v.devs));
  await mac.waitForFunction(() => !document.getElementById("ai-redeal").hidden, null, { timeout: 60000 }).catch(() => {});
  await shot(mac, "3-pc-joined-1080p.png");
  // (the re-deal itself is not run here: a tab's second WebGPU device right after its first fails in headless
  // Chromium on this kind of host, vkCreateDevice VK_ERROR_INITIALIZATION_FAILED, or stalls on SwiftShader)
  await closeAll();

  // ---- two devices: a room the PC is in from the start (split by pledge: both hold layers)
  const mac2 = await device("mac"), pc2 = await device("pc");
  const code2 = await room([[mac2, "mac", 24], [pc2, "pc", 12]]);
  const up = (await online(mac2)) && (await online(pc2, 60000));
  check("2 devices: the room is online", up);
  await mac2.waitForFunction(() => document.querySelectorAll("#sp-devs .dv.holds").length === 2, null, { timeout: 30000 }).catch(() => {});
  v = await panel(mac2);
  check("2 devices: the event line says Running on 2 devices", /Running on 2 devices/.test(v.events.at(-1) || ""), JSON.stringify(v.events));
  check("2 devices: two device cards with their share and GB (layers in the tooltip)", v.devs.length === 2 && v.devs.every((d) => /of the model/.test(d.text) && /GB/.test(d.text) && /layers \d+–\d+/.test(d.tip)), JSON.stringify(v.devs));
  check("2 devices: the readout live while it answers", await shotLive(mac2, "4-mac-pc-live-1080p.png", "Now tell me about rivers."));
  check("2 devices: a second answer", await ask(mac2, "And lakes?"));
  v = await panel(mac2);
  check("2 devices: the bars are marked 2 devices", v.bars.length === 2 && v.bars.every((b) => b.n === 2), JSON.stringify(v.bars));
  // (a visitor sees no bar of one token's time: the dev view's; each card's tooltip has its ms of each token)
  check("2 devices: no token-time bar for a visitor", !v.share, v.share);
  check("2 devices: the cards' tooltips say how much of each token each device takes", v.devs.every((d) => /ms of each \d+ ms token/.test(d.tip)), JSON.stringify(v.devs));
  const pv2 = await panel(pc2);
  check("2 devices: the worker's screen shows the same history", pv2.bars.length === 2 && pv2.bars.every((b, i) => b.tps === v.bars[i].tps), JSON.stringify(pv2.bars));
  log("panel with 2 devices:", JSON.stringify(v));
  await shot(mac2, "5-mac-pc-1080p.png");

  // a phone joins: ask-only
  const phone = await device("phone", { phone: true });
  await enter(phone, "phone", 2);
  await phone.fill("#code-input", code2); await phone.click("#join-btn");
  check("phone: online as an ask-only guest", await online(phone, 120000));
  await mac2.waitForFunction(() => document.querySelectorAll("#sp-devs .dv").length === 3, null, { timeout: 30000 }).catch(() => {});
  v = await panel(mac2);
  const ph = v.devs.find((d) => d.name === "phone");
  check("phone: its card on the host says it asks (joined after the start: until a re-deal)", !!ph && /asks/.test(ph.text), JSON.stringify(v.devs));
  check("phone: the event line says it joined", v.events.some((e) => /phone joined/.test(e)), JSON.stringify(v.events));
  const ok = await ask(phone, "Hi from the phone.");
  check("phone: its question is answered", ok);
  const pv = await panel(phone);
  check("phone: its readout and history show the room's speed", pv.shown && pv.tps > 0 && /^Last answer: /.test(pv.line) && pv.bars.length >= 1 && pv.bars.at(-1).n === 2, JSON.stringify(pv));
  await shot(phone, "6-phone-390.png");
  await phone.evaluate(() => document.getElementById("band-toggle").click());   // unfolded: the device cards
  await shot(phone, "7-phone-390-unfolded.png");
  await shot(mac2, "8-mac-pc-phone-1080p.png");
  for (const p of [mac2, pc2, phone]) check(`${p.label}: no page errors`, !p.errs.length, p.errs.join(" | "));
  await closeAll();

  // ---- a room node as the pc: `pooled join` on this machine's GPU (Dawn), the tab hosts
  if (NODE) {
    const { joinRoom } = await import("../../packages/room-node/index.js");
    const mac3 = await device("mac");
    const nodeLog = [];
    await room([[mac3, "mac", 24, "&dev=1"]], async (c) => {
      // 8 GB, under half the mac's 24: the tab stays the model host (room/plan.js pickModelHost would pick the node's
      // real GPU over SwiftShader otherwise) and the node is its worker
      node = await joinRoom(c, { pledgeGB: 8, name: "pc", signal: `127.0.0.1:${SIG}`, modelDir, log: (m) => nodeLog.push(String(m)) });
      return 1;
    });
    check("room node: the room is online with the node holding layers", await online(mac3), nodeLog.slice(-8).join(" | "));
    check("room node: the tab is the model host, the node its worker", !node.modelHost && node.ai.role === "worker", JSON.stringify({ modelHost: node.modelHost, role: node.ai.role }));
    check("room node: an answer", await ask(mac3, "Tell me about forests."));
    check("room node: a second answer", await ask(mac3, "And deserts?"));
    await mac3.waitForFunction(() => /ms of each \d+ ms token/.test(document.querySelector('#sp-devs .dv[data-name="pc"]')?.title || ""), null, { timeout: 15000 }).catch(() => {});
    v = await panel(mac3);
    const pcCard = v.devs.find((d) => d.name === "pc")?.text || "";
    const pcTip = v.devs.find((d) => d.name === "pc")?.tip || "";
    check("room node: its card has its part of each token's time (its ai-tele)", /layers \d+–\d+/.test(pcTip) && /ms of each \d+ ms token/.test(pcTip) && !/no timing/.test(pcCard + pcTip), JSON.stringify(v.devs));
    // (mac3 is the dev view, dev=1: one token's bar of each device's part)
    check("room node: one token's bar has a part for it, not 'wire + pc' (dev view)", /pc \d+%/.test(v.share) && !/wire \+ pc/.test(v.share), v.share);
    log("panel with a room node:", JSON.stringify(v));
    await shot(mac3, "9-mac-roomnode-1080p.png");
    check("mac (room node room): no page errors", !mac3.errs.length, mac3.errs.join(" | "));
  }
  code = results.length && results.every((r) => r.ok) ? 0 : 1;
} catch (e) {
  check("ran to the end", false, String(e?.stack || e).slice(0, 500));
} finally {
  const fail = results.filter((r) => !r.ok).map((r) => r.name);
  console.log(JSON.stringify({ ok: code === 0, pass: results.length - fail.length, fail }));
  await closeAll();
  await node?.close().catch(() => {});
  await browser.close().catch(() => {}); srv.close(); sig.kill(); fs.rmSync(tmp, { recursive: true, force: true });
  process.exit(code);
}
