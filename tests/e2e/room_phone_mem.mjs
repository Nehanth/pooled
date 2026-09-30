// Phone memory in a room (#207), on one machine: a host and a desktop worker, then a phone-shaped tab
// (iPhone user agent: capped pledge, 256 MB buffers, no weight cache) that joins after the start and
// is dealt layers by a re-deal, so part of its ranges come from the worker's cache over WebRTC (the
// streamed, flow-controlled path). Then the phone's tab is "killed" while it loads (reloaded while its
// breadcrumb says loading, which is what a jetsam kill leaves behind) and rejoins: the host must
// re-deal with a smaller share for it, and after a second kill without it, and the room must come
// back online each time. Manual trigger only (needs a GPU; the 27B from models/q38/model.gguf).
//
//   node tests/e2e/room_phone_mem.mjs [--model qwen3.8-27b] [--host-gb 16] [--root <checkout>] [--port 8141]
//        [--no-kill] [--out result.json]
//
// Reports per phase: the split, whether every tab came online, the greedy answer (the "exact"
// preset: the same prompt must give the same answer on every split), and for the phone tab the peak
// resident size of its renderer and network service while it loaded (from /proc, every 200 ms), plus
// how many MB came from the room vs the network (the room's own log line).
// --root: serve another checkout (e.g. origin/main) with this script, for a before/after peak.
import { chromium } from "playwright";
import http from "http";
import https from "https";
import fs from "fs";
import os from "os";
import path from "path";
import { spawn, execSync } from "child_process";

const arg = (k, d) => { const i = process.argv.indexOf("--" + k); return i > 0 ? process.argv[i + 1] : d; };
const flag = (k) => process.argv.includes("--" + k);
const MODEL = arg("model", "qwen3.8-27b"), HOST_GB = arg("host-gb", "16"), PORT = +arg("port", 8141), SIGNAL_PORT = PORT + 2;
const ROOT = path.resolve(arg("root", path.join(path.dirname(new URL(import.meta.url).pathname), "../..")));
const HERE = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../..");
const PROMPT = "Name three primary colors, one per line.";
const LOCAL = { "Qwen3.8-27B-Q4_0.gguf": "models/q38/model.gguf", "Qwen_Qwen3.6-35B-A3B-Q4_0.gguf": "models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf" };
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png" };
const t0 = Date.now(), log = (...a) => console.error(((Date.now() - t0) / 1000).toFixed(1) + "s", ...a);

const srv = http.createServer((q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { r.statusCode = 404; r.end(); return; }
  r.setHeader("content-type", MIME[path.extname(p)] || "application/octet-stream"); fs.createReadStream(p).pipe(r);
}).listen(PORT, "127.0.0.1");
const tlsDir = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-pm-"));
execSync(`openssl req -x509 -newkey rsa:2048 -nodes -keyout ${tlsDir}/k.pem -out ${tlsDir}/c.pem -days 2 -subj /CN=127.0.0.1 2>/dev/null`);
const wsrv = https.createServer({ key: fs.readFileSync(`${tlsDir}/k.pem`), cert: fs.readFileSync(`${tlsDir}/c.pem`) }, (q, r) => {
  const p = path.join(HERE, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(HERE) || !fs.existsSync(p)) { r.statusCode = 404; r.end(); return; }
  const size = fs.statSync(p).size, m = /bytes=(\d+)-(\d*)/.exec(q.headers.range || "");
  const lo = m ? +m[1] : 0, hi = m && m[2] ? Math.min(+m[2], size - 1) : size - 1;
  r.writeHead(m ? 206 : 200, { "content-type": "application/octet-stream", "content-range": `bytes ${lo}-${hi}/${size}`, "accept-ranges": "bytes", "content-length": String(hi - lo + 1), "access-control-allow-origin": "*", "access-control-expose-headers": "content-range, content-length, accept-ranges" });
  fs.createReadStream(p, { start: lo, end: hi }).pipe(r);
}).listen(PORT + 1, "127.0.0.1");
const peerServer = spawn(path.join(HERE, "node_modules/.bin/peerjs"), ["--port", String(SIGNAL_PORT), "--path", "/"], { stdio: "ignore" });
await new Promise((r) => setTimeout(r, 1500));
const BASE = `http://127.0.0.1:${PORT}/p2p.html?signal=127.0.0.1:${SIGNAL_PORT}&maxnew=24&dev=1&split=memory&phonelayers=1`;   // phonelayers: the phone tab holds layers even when the computers could hold the model (#233)

const ARGS = ["--no-sandbox", "--headless=new", "--enable-unsafe-webgpu", "--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--allow-loopback-in-peer-connection", "--js-flags=--max-old-space-size=65536"];
const UA_DESKTOP = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";
const UA_PHONE = "Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1";
const local = (c) => c.route("**/*.gguf", (route) => {
  const file = LOCAL[route.request().url().split("/").pop().split("?")[0]];
  if (!file || !fs.existsSync(path.join(HERE, file))) return route.continue();
  return route.continue({ url: `https://127.0.0.1:${PORT + 1}/${file}` });
});
// host and worker: one browser each with a profile on disk (their weight caches are separate, like two
// machines, and live on disk: an in-memory context would hold every cached range in RAM); the phone
// tab in a third browser (phones cache nothing)
const profs = [];
const persistent = async (opts) => { const d = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-pm-prof-")); profs.push(d); return chromium.launchPersistentContext(d, { headless: false, args: ARGS, ignoreHTTPSErrors: true, ...opts }); };
const hctx = await persistent({ userAgent: UA_DESKTOP }); await local(hctx);
const wctx = await persistent({ userAgent: UA_DESKTOP }); await local(wctx);
const pctx = await persistent({ userAgent: UA_PHONE, viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true }); await local(pctx);
const phoneProfile = profs.at(-1);
const browsers = [hctx, wctx, pctx];
const host = hctx.pages()[0] || await hctx.newPage(), worker = wctx.pages()[0] || await wctx.newPage(), phone = pctx.pages()[0] || await pctx.newPage();
const tabs = { host, worker, phone };
const errs = { host: [], worker: [], phone: [] };
for (const [n, p] of Object.entries(tabs)) {
  p.on("pageerror", (e) => errs[n].push(String(e).slice(0, 200)));
  p.on("crash", () => { errs[n].push("tab crashed"); log(n, "TAB CRASHED"); });
  p.on("close", () => log(n, "tab closed"));
}
for (const [n, c] of [["host", hctx], ["worker", wctx], ["phone", pctx]]) c.on("close", () => log(n, "BROWSER CLOSED"));
const out = { model: MODEL, root: ROOT, phases: [], errors: errs };
const status = (p) => p.evaluate(() => document.getElementById("ai-status").textContent);
const roomLog = (p) => p.evaluate(() => [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent));
let code = null;
async function join(p, name, gb, create = false) {
  await p.goto(BASE);
  await p.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
  await p.fill("#name-input", name);
  if (gb) await p.fill("#join-gb", gb);
  if (create) {
    await p.click("#create-btn");
    await p.waitForFunction(() => /[A-Z0-9]{4}/.test(document.getElementById("side-code").textContent), null, { timeout: 30000 });
    code = (await p.textContent("#side-code")).trim().match(/[A-Z0-9]{4}/)[0];
  } else { await p.fill("#code-input", code); await p.click("#join-btn"); }
}
const online = async (p, ms = 20 * 60e3) => {
  const t = setInterval(async () => { try { log("waiting:", ...(await Promise.all(Object.entries(tabs).filter(([, q]) => !q.isClosed()).map(async ([n, q]) => n + "=" + (await status(q)).slice(0, 70))))); } catch {} }, 30000);
  try { await onlineWait(p, ms); } finally { clearInterval(t); }
};
const onlineWait = (p, ms) => p.waitForFunction(() => document.getElementById("ai-panel").classList.contains("online") || /^failed:/.test(document.getElementById("ai-status").textContent), null, { timeout: ms, polling: 1000 });
async function ask() {
  await host.evaluate(() => document.getElementById("new-chat")?.click());
  await host.waitForTimeout(500);
  await host.fill("#ai-prompt", PROMPT); await host.click("#ai-send");
  await host.waitForFunction(() => /^ready — prefill|^generation failed/.test(document.getElementById("ai-status").textContent), null, { timeout: 300000 });
  return host.evaluate(() => { const b = document.querySelectorAll(".m.bot .bubble"); return (b[b.length - 1]?.textContent || "").trim(); });
}
// the phone tab's memory while it loads: the resident size of its browser's renderer (JS, ArrayBuffers,
// fetch bodies handed to the page) and of its network service (bodies nobody read yet), from /proc,
// sampled every 200 ms (the phone runs in its own browser, so these processes are only its own)
function descendants(root) {
  const rows = execSync("ps -eo pid=,ppid=,rss=,args=", { maxBuffer: 64 << 20 }).toString().trim().split("\n").map((l) => { const m = l.trim().match(/^(\d+)\s+(\d+)\s+(\d+)\s+(.*)$/); return m && { pid: +m[1], ppid: +m[2], rss: +m[3], args: m[4] }; }).filter(Boolean);
  const set = new Set([root]); let grew = true;
  while (grew) { grew = false; for (const r of rows) if (!set.has(r.pid) && set.has(r.ppid)) { set.add(r.pid); grew = true; } }
  return rows.filter((r) => set.has(r.pid));
}
async function sampler() {
  // the phone's browser: the process started with its profile directory (not a child: no --type)
  const root = descendants(1).find((r) => r.args.includes(phoneProfile) && !/--type=/.test(r.args))?.pid;
  if (!root) log("no phone browser process found: memory not sampled");
  let peakR = 0, peakN = 0, stop = false, n = 0;
  (async () => { while (!stop) { try { const ps = descendants(root); const kb = (re) => ps.filter((r) => re.test(r.args)).reduce((a, r) => a + r.rss, 0); peakR = Math.max(peakR, kb(/--type=renderer/)); peakN = Math.max(peakN, kb(/network\.mojom|--utility-sub-type=network/)); n++; } catch {} await new Promise((r) => setTimeout(r, 200)); } })();
  return () => { stop = true; return { rendererPeakMB: Math.round(peakR / 1024), networkPeakMB: Math.round(peakN / 1024), samples: n }; };
}
const phoneShare = async () => (await roomLog(phone)).filter((t) => /weights came from devices in the room/.test(t)).slice(-1)[0] || null;
const split = async () => (await roomLog(host)).filter((t) => /layer split/.test(t)).slice(-1)[0] || "";
const clickRedeal = async () => { await host.waitForFunction(() => !document.getElementById("ai-redeal").hidden, null, { timeout: 60000, polling: 500 }); await host.click("#ai-redeal"); };

try {
  // A: host + worker
  await join(host, "host-pm", HOST_GB, true);
  await join(worker, "worker-pm", "1");
  await host.waitForFunction(() => document.querySelectorAll(".peer-card").length >= 2, null, { timeout: 60000 });
  await host.waitForTimeout(2000);
  await host.selectOption("#ai-model", MODEL);
  await host.evaluate(() => { const s = document.getElementById("ai-sampling"); if (s) { s.value = "exact"; s.dispatchEvent(new Event("change")); } });
  await host.click("#ai-start");
  await online(host); await online(worker);
  log("A online:", await status(host));
  const aA = await ask();
  out.phases.push({ phase: "A host+worker", split: await split(), status: await status(host), answer: aA });
  log("A", out.phases.at(-1).split, JSON.stringify(aA).slice(0, 80));

  // B: the phone joins the online room; the re-deal gives it layers, partly from the worker's cache
  const stopB = await sampler();
  await join(phone, "phone-pm", "");
  out.phoneJoinGB = await phone.evaluate(() => document.getElementById("join-gb").value);
  await host.waitForFunction(() => document.querySelectorAll(".peer-card").length >= 3, null, { timeout: 60000 });
  await clickRedeal();
  await host.waitForTimeout(3000);
  await online(host); await online(phone);
  log("B online:", await status(host), "| phone:", await status(phone));
  const memB = stopB();
  const aB = await ask();
  out.phases.push({ phase: "B +phone (re-deal)", split: await split(), status: await status(host), answer: aB, same: aB === aA, phoneMem: memB, phoneWeights: await phoneShare() });
  log("B", JSON.stringify(out.phases.at(-1)).slice(0, 400));

  if (!flag("no-kill")) {
    // C: the worker leaves; re-deal; the phone's tab is killed while it loads, twice
    await worker.close();
    // the phone lends one step more, so the re-deal gives it a new range: a device dealt the slot it
    // already holds keeps its layers and loads nothing (fix/i207-resume), and there would be no load to kill
    await phone.evaluate(() => document.getElementById("ap-plus").click());
    await clickRedeal();
    for (const k of [1, 2]) {
      await phone.waitForFunction(() => { try { return !!JSON.parse(localStorage.getItem("pooled-crumb") || "null")?.loading; } catch { return false; } }, null, { timeout: 5 * 60e3, polling: 100 });
      await phone.waitForTimeout(1500);
      log(`C${k}: killing the phone tab while it loads:`, await status(phone));
      await join(phone, "phone-pm", "");   // a reload (what a jetsam kill looks like to the page), then join again
      out.phoneRejoinGB = await phone.evaluate(() => document.getElementById("join-gb").value);
      await host.waitForFunction((k) => [...document.querySelectorAll("#chat-log div")].filter((d) => /closed its tab while it loaded/.test(d.textContent)).length >= k, k, { timeout: 60000, polling: 500 });
      const why = (await roomLog(host)).filter((t) => /closed its tab while it loaded/.test(t)).slice(-1)[0];
      log(`C${k}:`, why);
      if (k === 1) out.phases.push({ phase: "C1 phone killed while loading", host: why });
      else {
        await online(host);
        const aC = await ask();
        out.phases.push({ phase: "C2 killed again: re-dealt without it", host: why, split: await split(), status: await status(host), answer: aC, same: aC === aA, phoneStatus: await status(phone) });
      }
    }
  }
  out.ok = out.phases.every((p) => !/failed/.test(p.status || "")) && out.phases.filter((p) => "same" in p).every((p) => p.same) && Object.values(errs).every((e) => !e.length);
} catch (e) {
  out.ok = false; out.error = String(e).slice(0, 400);
  for (const [n, p] of Object.entries(tabs)) { try { out[n + "Status"] = await status(p); out[n + "Log"] = (await roomLog(p)).slice(-8); } catch {} }
} finally {
  const line = JSON.stringify(out);
  if (arg("out")) fs.writeFileSync(arg("out"), line);
  console.log(line);
  for (const b of browsers) await b.close().catch(() => {}); for (const d of profs) fs.rmSync(d, { recursive: true, force: true }); srv.close(); wsrv.close(); peerServer.kill(); fs.rmSync(tlsDir, { recursive: true, force: true });
  process.exit(out.ok ? 0 : 1);
}
