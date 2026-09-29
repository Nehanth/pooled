// How much memory an iPhone's Safari tab can use before iOS kills it, and what a layer load costs at
// its peak (issue #207). Runs on the Mac the phone is USB-attached to (Node 18+, no dependencies),
// drives the phone over WebDriver (safaridriver) and watches for the kill: iOS reloads the tab
// without an exception, so a run is over when the page's boot id changes, the page stops answering,
// or it reports that the previous run died (memprobe.html keeps a breadcrumb in localStorage before
// every step).
//
//   probe mode (memprobe.html):
//     node tests/e2e/memprobe_phone.mjs --base https://<preview>/memprobe.html \
//          --run "mode=js&step=64" --run "mode=gpu&step=256" --run "mode=load&model=moe&lo=37&hi=40&ballast=1024" ...
//   room mode (the real room page, as a guest): allocate --ballast MB of JS first, then join --code
//     node tests/e2e/memprobe_phone.mjs --room https://<preview>/room --code ABCD|--codefile f --gb 0.5 [--ballast 0]
//
// Every run is one line of JSON on stdout: the query, how it ended (done | died | error | timeout),
// the step it died at (the breadcrumb), and a 1 s timeline of the page's own accounting (JS MB it
// allocated, GPU MB of buffers alive, phase, last event). Room mode samples the room's status line,
// the GPU bytes allocated (createBuffer is wrapped before the join), the bytes the page has read from
// fetch bodies, and the page's crumb (localStorage "pooled-crumb": what it was doing when it died).
// Only one WebDriver session can drive the phone: take it with devq.sh (scratchpad) first.
import { spawn } from "child_process";
import fs from "fs";

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const args = (k) => argv.flatMap((a, i) => (a === "--" + k ? [argv[i + 1]] : []));
const WD = `http://127.0.0.1:${+arg("wd-port", 4444)}`;
const t0 = Date.now();
const log = (...a) => console.error(((Date.now() - t0) / 1000).toFixed(0) + "s [memprobe]", ...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let driver = null, sid = null;

async function wd(method, path, body, timeoutMs = 60000) {
  const r = await fetch(WD + path, { method, headers: { "content-type": "application/json" }, body: body ? JSON.stringify(body) : undefined, signal: AbortSignal.timeout(timeoutMs) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(`webdriver ${method} ${path}: ${r.status} ${JSON.stringify(j.value || j).slice(0, 300)}`);
  return j.value;
}
const S = (p) => `/session/${sid}${p}`;
const exec = (fn, ...a) => wd("POST", S("/execute/sync"), { script: `return (${fn}).apply(null, arguments)`, args: a }, 20000);
async function start() {
  driver = spawn("safaridriver", ["-p", String(+arg("wd-port", 4444))], { stdio: "ignore" });
  for (let i = 0; i < 40; i++) { try { await wd("GET", "/status", null, 2000); break; } catch { await sleep(250); } }
  const s = await wd("POST", "/session", { capabilities: { alwaysMatch: { browserName: "safari", platformName: "iOS" } } }, 90000);
  sid = s.sessionId;
  log("session", sid, s.capabilities?.["safari:deviceName"] || "", s.capabilities?.browserVersion || "");
}
async function stop() {
  if (sid) { try { await wd("DELETE", `/session/${sid}`, null, 30000); } catch {} sid = null; }
  if (driver) { driver.kill(); driver = null; }
}
process.on("SIGTERM", async () => { await stop(); process.exit(1); });
process.on("SIGINT", async () => { await stop(); process.exit(1); });

// ---------- probe mode ----------
const MP = () => { const m = window.__mp; if (!m) return null; const e = m.events[m.events.length - 1] || {};
  return { boot: m.boot, run: m.run, phase: m.phase, jsMB: m.jsMB, gpuMB: e.gpuMB, steps: m.steps, last: e.ev + (e.info ? " " + e.info : ""), t: m.t,
    died: m.died, err: m.err, result: m.result, adapter: m.adapter, lost: m.lost || null, load: m.load ? { ...m.load, tensors: m.load.tensors.length } : null, cacheHits: m.cacheHits || 0 }; };
async function probeRun(base, q, maxMin) {
  const run = `r${Date.now().toString(36)}`;
  const url = base + (base.includes("?") ? "&" : "?") + q + "&run=" + run;
  const out = { q, run, url, timeline: [] };
  await wd("POST", S("/url"), { url }, 90000);
  let boot = null, misses = 0; const tEnd = Date.now() + maxMin * 60e3;
  while (Date.now() < tEnd) {
    await sleep(1000);
    let s;
    try { s = await exec(MP); misses = 0; } catch (e) { if (++misses > 20) { out.end = "no answer"; out.error = String(e).slice(0, 200); break; } continue; }
    if (!s) continue;
    if (!boot) boot = s.boot;
    out.timeline.push([Math.round((Date.now() - t0) / 1000), s.phase, Math.round(s.jsMB), Math.round(s.gpuMB || 0), s.steps, s.last.slice(0, 80)]);
    if (s.boot !== boot || s.phase === "died") {
      // the tab was reloaded: the fresh page read the breadcrumb of the run that died
      out.end = "died"; out.died = s.died; out.reloadedAfterS = s.died?.noticedAfterS; break;
    }
    if (s.phase === "done" || s.phase === "error") { out.end = s.phase; out.result = s.result; out.err = s.err; out.adapter = s.adapter; out.load = s.load; out.lost = s.lost; break; }
    if (out.timeline.length % 10 === 1) log(q, JSON.stringify(out.timeline[out.timeline.length - 1]));
  }
  out.end = out.end || "timeout";
  if (out.timeline.length > 400) out.timeline = [...out.timeline.slice(0, 50), ...out.timeline.slice(-350)];
  // a clean page for the next run (and let the phone settle)
  try { await wd("POST", S("/url"), { url: base + (base.includes("?") ? "&" : "?") + "mode=clear" }, 60000); } catch {}
  await sleep(+arg("rest", 8) * 1000);
  return out;
}

// ---------- room mode ----------
const HOOK = (ballastMB) => {
  if (window.__mh) return "already";
  const H = window.__mh = { gpu: 0, gpuPeak: 0, fetchRead: 0, fetchOpen: 0, maxChunk: 0, ballast: [], boot: Math.random().toString(36).slice(2), marks: [] };
  for (let i = 0; i < ballastMB / 64; i++) { const ab = new ArrayBuffer(64 * 2 ** 20); new Uint32Array(ab).fill(i + 1); H.ballast.push(ab); }
  H.ballastMB = ballastMB;
  const oReq = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function (desc) {
    const d = await oReq.call(this, desc);
    const oc = d.createBuffer.bind(d);
    d.createBuffer = (x) => { const b = oc(x), sz = x.size; H.gpu += sz; H.gpuPeak = Math.max(H.gpuPeak, H.gpu); const od = b.destroy.bind(b); let g = false; b.destroy = () => { if (!g) { g = true; H.gpu -= sz; } return od(); }; return b; };
    const odd = d.destroy.bind(d); d.destroy = () => { H.marks.push(["device-destroy", Date.now()]); return odd(); };
    return d;
  };
  const oRead = ReadableStreamDefaultReader.prototype.read;
  ReadableStreamDefaultReader.prototype.read = function () { return oRead.call(this).then((r) => { if (r.value?.byteLength) { H.fetchRead += r.value.byteLength; H.maxChunk = Math.max(H.maxChunk, r.value.byteLength); } return r; }); };
  const oFetch = window.fetch;
  window.fetch = function (u, o) { const rg = o?.headers?.Range; if (rg) { H.fetchOpen++; H.marks.push(["fetch", Date.now(), rg]); if (H.marks.length > 3000) H.marks.splice(0, 1000); } return oFetch.apply(this, arguments); };
  return "ok";
};
const RSNAP = () => { const $ = (id) => document.getElementById(id), H = window.__mh || null;
  let crumb = null; try { crumb = JSON.parse(localStorage.getItem("pooled-crumb") || "null"); } catch {}
  return { boot: H?.boot || null, status: $("ai-status")?.textContent || "", mine: $("compute-open")?.dataset.tip || "", inRoom: document.body.classList.contains("in-room"),
    peers: document.querySelectorAll(".peer-card").length, gpuMB: H ? Math.round(H.gpu / 2 ** 20) : null, gpuPeakMB: H ? Math.round(H.gpuPeak / 2 ** 20) : null,
    readMB: H ? Math.round(H.fetchRead / 2 ** 20) : null, fetches: H?.fetchOpen, maxChunkKB: H ? Math.round(H.maxChunk / 1024) : null, crumb, hostLeft: !!($("room-over") && !$("room-over").hidden),
    lastFetches: H ? H.marks.filter((m) => m[0] === "fetch").slice(-3).map((m) => m[2]) : null, destroys: H ? H.marks.filter((m) => m[0] === "device-destroy").length : null }; };
async function roomRun() {
  const out = { room: arg("room"), gb: arg("gb", "0.5"), ballastMB: +arg("ballast", 0), timeline: [] };
  await wd("POST", S("/url"), { url: arg("room") + (arg("query") ? "?" + arg("query") : "") }, 90000);
  for (let i = 0; i < 120; i++) { if (await exec(() => { const g = document.getElementById("join-gb"); return !!g && g.value !== "" && g.value !== "1"; })) break; await sleep(500); }
  out.hook = await exec(HOOK, out.ballastMB);
  await exec((name, gb) => { for (const [id, v] of [["name-input", name], ["join-gb", gb]]) { const el = document.getElementById(id); el.value = v; el.dispatchEvent(new Event("input", { bubbles: true })); el.dispatchEvent(new Event("change", { bubbles: true })); } }, arg("name", "iphone"), out.gb);
  let code = arg("code");
  for (const tEnd = Date.now() + 30 * 60e3; !code && Date.now() < tEnd; await sleep(1000)) { try { code = (fs.readFileSync(arg("codefile"), "utf8").match(/[A-Z0-9]{4}/) || [])[0]; } catch {} }
  if (!code) throw new Error("no room code");
  out.code = code;
  await exec((c) => { const el = document.getElementById("code-input"); el.value = c; el.dispatchEvent(new Event("input", { bubbles: true })); }, code);
  try { const el = await wd("POST", S("/element"), { using: "css selector", value: "#join-btn" }); await wd("POST", S(`/element/${Object.values(el)[0]}/click`), {}); } catch {}
  await sleep(3000);
  if (!(await exec(() => document.getElementById("join-status")?.textContent || document.body.classList.contains("in-room")))) await exec(() => { getSelection()?.removeAllRanges(); document.getElementById("join-btn").click(); });
  let boot = null, misses = 0, seen = Date.now(); const tEnd = Date.now() + +arg("maxmin", 30) * 60e3;
  while (Date.now() < tEnd) {
    await sleep(1000);
    let s;
    try { s = await exec(RSNAP); misses = 0; } catch (e) { if (++misses > 30) { out.end = "no answer"; break; } continue; }
    if (!boot) boot = s.boot;
    if (s.boot !== boot) { out.end = "died"; out.afterReload = s; break; }
    const tl = [Math.round((Date.now() - t0) / 1000), s.gpuMB, s.readMB, s.fetches, s.status.slice(0, 90), s.crumb?.s?.slice(0, 90) || ""];
    out.timeline.push(tl);
    if (out.timeline.length % 10 === 1) log(JSON.stringify(tl), s.mine, "peers", s.peers);
    if (/holds layers/.test(s.mine) && !out.loadedAt) { out.loadedAt = tl; out.mine = s.mine; out.snap = s; log("loaded", s.mine); if (arg("leave-after-load")) { out.end = "loaded"; break; } }
    if (/failed/.test(s.status)) { out.end = "failed"; out.snap = s; break; }
    if (s.hostLeft) { out.end = "host left"; out.snap = s; break; }
    if (s.peers >= 2) seen = Date.now(); else if (s.inRoom && Date.now() - seen > 20000) { out.end = "room empty"; out.snap = s; break; }
  }
  out.end = out.end || "timeout";
  if (out.timeline.length > 600) out.timeline = [...out.timeline.slice(0, 100), ...out.timeline.slice(-500)];
  return out;
}

try {
  await start();
  if (arg("room")) { const r = await roomRun(); console.log(JSON.stringify(r)); }
  else {
    const base = arg("base");
    for (const q of args("run")) {
      const r = await probeRun(base, q, +arg("maxmin", 15));
      log(q, "->", r.end, r.died ? `died: phase ${r.died.phase} js ${Math.round(r.died.jsMB)} gpu ${Math.round(r.died.gpuMB)} ballast ${r.died.ballastMB} last ${JSON.stringify(r.died.last)}` : JSON.stringify(r.result || r.err || "").slice(0, 300));
      console.log(JSON.stringify(r));
    }
  }
} catch (e) { log("FAILED", String(e).slice(0, 400)); console.log(JSON.stringify({ error: String(e).slice(0, 400) })); }
await stop();
