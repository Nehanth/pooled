// The screen before and at ready is calm (room.js speedReady, speedMeta; room/speedo.js bandLine): when the model
// goes ready the band is one line (the model, "Ready", the split, "Show the layers"), with nothing about speed: no
// panel opens by itself, no tok/s, no "Waiting for a question", nothing bigger than the model's name. A one-token
// answer leaves it so. After a real answer the head gets one 12px line, "Last answer: about N words a second", and
// the band keeps its height. The first question's wait says "Reading your message…", then once, after 4 s with no
// word, "The first answer takes a little longer…" (never on the second question). A viewer who opened the band
// ("Show the layers", kept in localStorage) sees the model and the device cards, no readout or chart, until a real
// answer. The real room (p2p.html + room.js) in headless Chromium, a local PeerServer, WebRTC between the tabs and
// the synthetic model (tests/e2e/synth.mjs), as tests/e2e/room_speed.mjs.
//
//   node tests/e2e/ready_calm.mjs                 the checks (E2E_GPU=real: the machine's GPU, as on the GB10)
//   node tests/e2e/ready_calm.mjs --shots DIR     and screenshots
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
const PORT = +arg("port", 8508), SIG = PORT + 1;
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

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-ready-calm-"));
const modelBytes = fs.readFileSync(writeSynth(path.join(tmp, "synth.gguf"), { seed: 1, eosAt: 400 }).file);
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
async function device(name, { phone = false, size = null, bandOpen = false } = {}) {
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
  // a viewer who opened the band before ("Show the layers", kept in this browser)
  if (bandOpen) await ctx.addInitScript(() => { try { localStorage.setItem("pooled-band-chat", "open"); } catch {} });
  // the slow first question: window.__holdUntil (performance.now() ms) holds every GPU readback until then, so the
  // first token comes late without the page's own timers being blocked
  await ctx.addInitScript(() => {
    if (!self.GPUBuffer) return;
    const map = GPUBuffer.prototype.mapAsync;
    GPUBuffer.prototype.mapAsync = async function (...a) {
      const d = (self.__holdUntil || 0) - performance.now();
      if (d > 0) await new Promise((r) => setTimeout(r, d));
      return map.apply(this, a);
    };
  });
  const p = await ctx.newPage();
  p.label = name; p.errs = [];
  p.on("pageerror", (e) => p.errs.push(String(e).slice(0, 200)));
  return p;
}
// dev=0: the page as a visitor sees it (signal= turns the dev extras on)
let base = `http://127.0.0.1:${PORT}/p2p.html?signal=127.0.0.1:${SIG}&maxnew=1&dev=0`;
const wired = (p) => p.waitForFunction(() => typeof Peer === "function" && window.pooledWired, null, { timeout: 60000 });
async function enter(p, name, gb, q = "") {
  // (q with dev=1: the developer view, in place of the visitor's dev=0)
  await p.goto(/[?&]dev=1\b/.test(q) ? (base + q).replace("&dev=0", "") : base + q); await wired(p);
  await p.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
  await p.fill("#name-input", name); await p.fill("#join-gb", String(gb));
}
const online = (p, ms = 300000) => p.waitForFunction(() => document.getElementById("ai-panel").classList.contains("online"), null, { timeout: ms }).then(() => true, () => false);

// the band as a viewer sees it
const band = (p) => p.evaluate(() => {
  const q = (s) => document.querySelector(s), vis = (e) => !!e && e.getClientRects().length > 0 && getComputedStyle(e).visibility !== "hidden";
  const sm = q("#swarm-map");
  let big = 0, bigWho = "";
  for (const e of sm.querySelectorAll("*")) {
    if (!vis(e) || !(e.textContent || "").trim()) continue;
    const f = parseFloat(getComputedStyle(e).fontSize);
    if (f > big) { big = f; bigWho = (e.id || e.className || e.tagName) + ""; }
  }
  return {
    h: Math.round(sm.getBoundingClientRect().height * 10) / 10,
    folded: sm.classList.contains("folded"),
    speedo: vis(q("#speedo")), devs: vis(q("#sp-devs")), quiet: q("#speedo").classList.contains("quiet"),
    read: vis(q(".sp-read")), hist: vis(q(".sp-hist")),
    meta: q(".sm-meta").textContent, metaTip: q(".sm-meta").title, metaFont: parseFloat(getComputedStyle(q(".sm-meta")).fontSize),
    model: vis(q(".sm-model")) ? q(".sm-model").textContent : "", pill: q(".sm-lt").textContent,
    big, bigWho,
    speedText: (/tok\/s|words a second|Waiting for a question/.exec(q("#room-screen").innerText) || [""])[0],
  };
});
// at ready / after a one-token answer: one line, nothing about speed
function calm(label, v, { open = false } = {}) {
  if (!open) {
    check(`${label}: the band is one line (≤ 56px)`, v.folded && v.h <= 56, JSON.stringify(v));
    check(`${label}: no speed panel and no device card`, !v.speedo && !v.devs, JSON.stringify(v));
    check(`${label}: the line says the model and Ready`, /Qwen3\.8/.test(v.model) && /^(Ready|Reading…|Writing…)$/.test(v.pill), JSON.stringify(v));
  } else {
    check(`${label}: the band opened (this viewer's choice) shows the model and the cards only`, !v.folded && v.speedo && v.quiet && !v.read && !v.hist, JSON.stringify(v));
  }
  check(`${label}: the head's speed line is empty`, v.meta === "", JSON.stringify(v));
  check(`${label}: no text about speed anywhere on the room screen`, v.speedText === "", v.speedText);
  // (folded: nothing big at all; opened by this viewer, the panel's model name is its one large word)
  if (!open) check(`${label}: nothing in the band bigger than 18px`, v.big <= 18, `${v.big}px ${v.bigWho}`);
  else check(`${label}: nothing bigger than 18px but the model's name`, v.big <= 18 || v.bigWho === "sp-model", `${v.big}px ${v.bigWho}`);
}
async function shot(p, file) {
  if (!SHOTS) return;
  fs.mkdirSync(SHOTS, { recursive: true });
  await p.waitForTimeout(400);
  await p.screenshot({ path: path.join(SHOTS, file) });
  log("screenshot", path.join(SHOTS, file));
}
// a question; resolves with the answer's stats line. watch: every text the bubble's wait line showed
async function ask(p, text, { hold = 0 } = {}) {
  const n = await p.evaluate(() => document.querySelectorAll("#ai-output .m.bot .stats").length);
  await p.evaluate(({ t, hold }) => {
    window.__waitSeen = [];
    const seen = () => { const w = document.querySelector("#ai-output .m.bot.live .working .wk-v"); if (w && w.textContent !== window.__waitSeen.at(-1)) window.__waitSeen.push(w.textContent); };
    clearInterval(window.__waitTimer); window.__waitTimer = setInterval(seen, 100);
    if (hold) window.__holdUntil = performance.now() + hold;
    const b = document.getElementById("ai-prompt"); b.value = t; b.dispatchEvent(new Event("input")); document.getElementById("ai-send").click();
  }, { t: text, hold });
  const ok = await p.waitForFunction((k) => document.querySelectorAll("#ai-output .m.bot .stats").length > k, n, { timeout: 240000, polling: 250 }).then(() => true, () => false);
  return p.evaluate((ok) => { clearInterval(window.__waitTimer); const s = [...document.querySelectorAll("#ai-output .m.bot .stats")].at(-1); return { ok, stats: s?.title || s?.textContent || "", seen: window.__waitSeen }; }, ok);
}
async function room(host, guests = []) {
  await enter(host, host.label, 24, "&ask=0");
  await host.click("#create-btn");
  await host.waitForFunction(() => /^[A-Z0-9]{3}-[A-Z0-9]{3}$/.test(document.getElementById("side-code").textContent), null, { timeout: 30000 });
  const roomCode = (await host.textContent("#side-code")).trim().replace("-", "");
  await host.evaluate((m) => {
    for (const [id, v] of [["ai-model", m], ["ai-sampling", "exact"]]) { const s = document.getElementById(id); s.value = v; s.dispatchEvent(new Event("change", { bubbles: true })); }
  }, MODEL_KEY);
  await host.waitForFunction(() => !document.getElementById("ai-start").disabled, null, { timeout: 120000 });
  await host.click("#ai-start");
  const up = await online(host);
  // the others join once the model runs: they only ask
  for (const g of guests) { await enter(g, g.label, 2); await g.fill("#code-input", roomCode); await g.click("#join-btn"); }
  for (const g of guests) await online(g, 120000);
  return up;
}
const closeAll = async () => { for (const c of ctxs.splice(0)) await c.close().catch(() => {}); };

let code = 1;
try {
  // ---- one: the model answers one token at a time (maxnew=1, "Paris")
  const mac = await device("mac"), phone = await device("phone", { phone: true }), opener = await device("opener", { bandOpen: true });
  check("room 1: the model is ready", await room(mac, [phone, opener]));
  await mac.waitForTimeout(1500);   // (the first maps, the deal's events)
  calm("ready, 1920x1080", await band(mac));
  calm("ready, 390x844 (a phone)", await band(phone));
  calm("ready, band opened before (localStorage)", await band(opener), { open: true });
  await shot(mac, "1-ready-1080p.png"); await shot(phone, "2-ready-390.png"); await shot(opener, "3-ready-open-1080p.png");
  const one = await ask(mac, "The capital of France, in one word?");
  log("one-token answer:", JSON.stringify(one));
  check("a one-token answer", one.ok && /^1 tok\b/.test(one.stats), JSON.stringify(one));
  await mac.waitForTimeout(800);
  calm("after a one-token answer, 1920x1080", await band(mac));
  calm("after a one-token answer, a phone", await band(phone));
  calm("after a one-token answer, band opened", await band(opener), { open: true });
  for (const p of [mac, phone, opener]) check(`${p.label}: no page errors (room 1)`, !p.errs.length, p.errs.join(" | "));
  await closeAll();

  // ---- two: real answers (maxnew=20); the first question's wait is slow (its first token held 5.5 s)
  base = base.replace("&maxnew=1&", "&maxnew=20&");
  const mac2 = await device("mac"), phone2 = await device("phone", { phone: true });
  check("room 2: the model is ready", await room(mac2, [phone2]));
  await mac2.waitForTimeout(1500);
  const r0 = await band(mac2), p0 = await band(phone2);
  calm("room 2 ready, 1920x1080", r0);
  const a1 = await ask(mac2, "Tell me about the sea.", { hold: 5500 });
  log("first answer:", JSON.stringify(a1));
  check("the first question's wait: Reading your message…, then The first answer takes a little longer…",
    a1.seen[0] === "Reading your message…" && a1.seen.includes("The first answer takes a little longer…") && a1.seen.every((s) => /^(Reading your message|The first answer takes a little longer)…$/.test(s)), JSON.stringify(a1.seen));
  check("a 20-token answer", a1.ok && /^\d+ tok · /.test(a1.stats) && +/^(\d+)/.exec(a1.stats)[1] >= 2, a1.stats);
  await mac2.waitForTimeout(600);
  const r1 = await band(mac2);
  check("after a real answer: the head's line says the last answer's speed in words", /^Last answer: (about \d+ words? |less than a word )a second$/.test(r1.meta), JSON.stringify(r1));
  check("its tooltip has the numbers", /^\d+(\.\d+)? tok\/s · \d+ tokens · 1 device$/.test(r1.metaTip), r1.metaTip);
  check("the line is 12px", r1.metaFont === 12, String(r1.metaFont));
  check("the band keeps its height (within 2px of ready)", Math.abs(r1.h - r0.h) <= 2 && r1.folded && !r1.speedo, `${r0.h} -> ${r1.h} ${JSON.stringify(r1)}`);
  await phone2.waitForFunction(() => document.querySelector(".sm-meta").textContent !== "", null, { timeout: 10000 }).catch(() => {});
  const p1 = await band(phone2);
  check("a phone: the same line, the band keeps its height", /^Last answer: /.test(p1.meta) && Math.abs(p1.h - p0.h) <= 2 && !p1.speedo, `${p0.h} -> ${p1.h} ${JSON.stringify(p1)}`);
  await shot(mac2, "4-answered-1080p.png"); await shot(phone2, "5-answered-390.png");
  const a2 = await ask(mac2, "And about the mountains?", { hold: 5500 });
  log("second answer:", JSON.stringify(a2));
  check("the second question never says the first answer takes longer", a2.ok && !a2.seen.some((s) => /first answer/.test(s)) && a2.seen.length >= 1, JSON.stringify(a2.seen));
  await mac2.click("#band-toggle");
  await mac2.waitForTimeout(400);
  const r2 = await band(mac2);
  check("Show the layers: the panel opens with the readout and the chart (no quiet)", !r2.folded && r2.speedo && !r2.quiet && r2.read && r2.hist, JSON.stringify(r2));
  check("Show the layers: no Waiting for a question", !/Waiting for a question/.test(await mac2.textContent("#speedo")));
  await shot(mac2, "6-opened-1080p.png");
  for (const p of [mac2, phone2]) check(`${p.label}: no page errors (room 2)`, !p.errs.length, p.errs.join(" | "));
  code = results.length && results.every((r) => r.ok) ? 0 : 1;
} catch (e) {
  check("ran to the end", false, String(e?.stack || e).slice(0, 500));
} finally {
  const fail = results.filter((r) => !r.ok).map((r) => r.name);
  console.log(JSON.stringify({ ok: code === 0, pass: results.length - fail.length, fail }));
  await closeAll();
  await browser.close().catch(() => {}); srv.close(); sig.kill(); fs.rmSync(tmp, { recursive: true, force: true });
  process.exit(code);
}
