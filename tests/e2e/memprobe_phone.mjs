// Drive memprobe.html on the iPhone over WebDriver (issue #207): how much memory a Pooled tab gets
// before iOS kills it, and what a real layer load costs. Runs on the Mac the phone is USB-attached to
// (Node 18+, no dependencies), like tests/e2e/xroom_phone.mjs; needs "Remote Automation" on the phone.
//
//   node tests/e2e/memprobe_phone.mjs --base https://<preview-or-pooled.run> [--wd-port 4444] [--no-driver]
//        [--timeout-s 900] [--out runs.jsonl] "mode=js&step=64" "mode=gpu&step=64" "mode=load&model=moe&lo=39&then=js" ...
//
// Every argument that is not a flag is one run: the query of memprobe.html. Each run gets a fresh
// WebDriver session (a fresh tab), a run id, and is polled every 0.5 s (window.__mp). It ends when the
// page says done / error, or when the tab dies: iOS reloads a page that used too much memory, which
// shows here as a new boot id (or a lost session); the page's own breadcrumb (localStorage, written
// before every step) then says the last step it started. One JSON line per run on stdout (and --out):
// the outcome (died | done | error | timeout), the last state the poller saw, the page's breadcrumb,
// and the page's event list (timeline of allocations / tensors, from the last poll).
import fs from "fs";
import { spawn } from "child_process";

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const flag = (k) => argv.includes("--" + k);
const takesValue = new Set(["base", "wd-port", "timeout-s", "out", "tag"]);
const runs = argv.filter((a, i) => !a.startsWith("--") && !(i > 0 && takesValue.has(argv[i - 1].slice(2))));
const BASE = arg("base", "https://pooled.run").replace(/\/$/, "");
const WD = `http://127.0.0.1:${+arg("wd-port", 4444)}`;
const TIMEOUT = +arg("timeout-s", 900) * 1000;
const TAG = arg("tag", "mp" + Date.now().toString(36));
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
const exec = (fn) => wd("POST", S("/execute/sync"), { script: `return (${fn})()`, args: [] }, 20000);
async function newSession() {
  const v = await wd("POST", "/session", { capabilities: { alwaysMatch: { browserName: "safari", platformName: "iOS" } } }, 120000);
  sid = v.sessionId;
}
async function endSession() { if (sid) { try { await wd("DELETE", `/session/${sid}`, null, 30000); } catch (e) { log("delete:", String(e).slice(0, 120)); } sid = null; } }
async function cleanup() { await endSession(); if (driver) { driver.kill(); driver = null; } }
process.on("SIGTERM", async () => { await cleanup(); process.exit(1); });
process.on("SIGINT", async () => { await cleanup(); process.exit(1); });

const POLL = () => {
  const m = window.__mp; if (!m) return null;
  return JSON.stringify({ boot: m.boot, run: m.run, phase: m.phase, jsMB: m.jsMB, gpuMB: m.events.at(-1)?.gpuMB, steps: m.steps, last: m.events.at(-1), err: m.err, died: m.died, result: m.result || null, lost: m.lost || null, adapter: m.adapter || null, n: m.events.length });
};
const EVENTS = () => JSON.stringify({ events: window.__mp?.events || [], load: window.__mp?.load || null });

async function oneRun(query, i) {
  const run = `${TAG}-${i}`;
  const url = `${BASE}/memprobe.html?${query}&run=${run}`;
  const rec = { run, query, url, outcome: null, seen: null, breadcrumb: null, events: null, startedAt: new Date().toISOString() };
  await newSession();
  await wd("POST", S("/url"), { url: `${BASE}/memprobe.html?mode=clear` }, 120000);
  await wd("POST", S("/url"), { url }, 120000);
  let boot = null, lastEvents = null, lastEvAt = 0, fails = 0;
  const tStart = Date.now();
  while (!rec.outcome) {
    if (Date.now() - tStart > TIMEOUT) { rec.outcome = "timeout"; break; }
    let s = null;
    try { const r = await exec(POLL); s = r ? JSON.parse(r) : null; fails = 0; }
    catch (e) { if (++fails > 40) { rec.outcome = "lost-session"; rec.why = String(e).slice(0, 200); break; } await sleep(500); continue; }
    if (s) {
      if (!boot) boot = s.boot;
      if (s.boot !== boot) { rec.outcome = "died"; rec.reloadedPage = s; break; }   // iOS reloaded the tab
      rec.seen = s;
      if (s.phase === "done") rec.outcome = "done";
      else if (s.phase === "error") rec.outcome = "error";
      else if (s.phase === "died") rec.outcome = "died";
      if (Date.now() - lastEvAt > 5000 || rec.outcome) { try { lastEvents = JSON.parse(await exec(EVENTS)); lastEvAt = Date.now(); } catch {} }
    }
    if (!rec.outcome) await sleep(500);
  }
  rec.events = lastEvents;
  // the breadcrumb, from a report page (it survives the reload / a lost tab)
  try {
    if (rec.outcome === "lost-session") { await endSession(); await newSession(); }
    await wd("POST", S("/url"), { url: `${BASE}/memprobe.html?mode=report` }, 120000);
    await sleep(1500);
    const r = await exec(() => JSON.stringify({ prev: window.__mp?.prev || null }));
    rec.breadcrumb = JSON.parse(r).prev;
  } catch (e) { rec.breadcrumbErr = String(e).slice(0, 200); }
  rec.endedAt = new Date().toISOString();
  rec.secs = Math.round((Date.now() - tStart) / 1000);
  await endSession();
  return rec;
}

try {
  if (!flag("no-driver")) {
    driver = spawn("safaridriver", ["-p", String(+arg("wd-port", 4444))], { stdio: "ignore" });
    await sleep(3000);
  }
  for (let i = 0; i < runs.length; i++) {
    log(`run ${i + 1}/${runs.length}: ${runs[i]}`);
    let rec;
    try { rec = await oneRun(runs[i], i); }
    catch (e) { rec = { query: runs[i], outcome: "harness-error", why: String(e).slice(0, 300) }; await endSession(); }
    const b = rec.breadcrumb, s = rec.seen;
    log(`  -> ${rec.outcome} after ${rec.secs}s: seen JS ${s?.jsMB} MB GPU ${s?.gpuMB} MB (${s?.phase}); breadcrumb JS ${b?.jsMB} GPU ${b?.gpuMB?.toFixed?.(0)} step ${b?.steps} phase ${b?.phase}`);
    const line = JSON.stringify(rec);
    console.log(line);
    if (arg("out")) fs.appendFileSync(arg("out"), line + "\n");
    await sleep(3000);
  }
} finally { await cleanup(); }
