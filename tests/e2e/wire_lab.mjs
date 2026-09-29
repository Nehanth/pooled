// Wire lab driver: where one activation frame's time goes between two browsers, per send path,
// frame size, SCTP setting, idle gap and an answer-shaped lap pattern. CPU only (no model, no WebGPU),
// so it never needs a GPU lock. The page is tests/bench/wire_lab.html (see its header for the paths).
//
// Loopback (both browsers on this machine, local PeerJS server):
//   node tests/e2e/wire_lab.mjs --loopback [--suite paths,gap,answer] [--out f.json]
// Two machines (B first; A runs the PeerJS server on every interface):
//   B: node tests/e2e/wire_lab.mjs --role b --signal <A's ip>:9047
//   A: node tests/e2e/wire_lab.mjs --role a [--suite ...] [--out f.json]
//
// Suites (A decides; B follows each run's config from the connection metadata):
//   paths   every path x --sizes (default 1,2,4,8,16,32,64 KB), --reps frames 20 ms apart (a lap's idle)
//   gap     wire4 at 4 and 16 KB, idle between frames 0..1000 ms, keep-alive off and every 10 ms
//   tail    wire4 vs wire4dup (each slice also sent on the next stripe), 4 KB, 3000 frames each, 25 ms apart,
//           interleaved in blocks of 100: p99.9 and the count of stalled frames
//   answer  a plain answer's lap: A sends, B "computes" 7 ms, echoes, A "computes" 12 ms, 128 laps an
//           answer, 3 s between answers, 4 answers; 4 KB (MoE), 10 KB (27B), 16 KB (4-column verify);
//           keep-alive off / on. The first lap after the pause is reported apart.
// Per frame: rtt, A's send call, B's turnaround, and for each end the time from the first / last raw
// message of the frame to its delivery (spread / decode). one-way = (rtt - B's turnaround) / 2.
import { chromium } from "playwright";
import http from "http";
import fs from "fs";
import os from "os";
import path from "path";
import { spawn } from "child_process";

const arg = (k, d) => { const i = process.argv.indexOf("--" + k); return i > 0 ? process.argv[i + 1] : d; };
const flag = (k) => process.argv.includes("--" + k);
const LOOP = flag("loopback"), ROLE = LOOP ? "a" : arg("role", "a");
const REPS = +arg("reps", 200), GAP = +arg("gap", 20);
const SIZES = arg("sizes", "1,2,4,8,16,32,64").split(",").map((x) => Math.round(+x * 1024));
const SUITES = arg("suite", "paths,gap,answer,tail").split(",");
const PATHS = arg("paths", "pj,pjraw,raw,rawslice,rawunord,rawrtx0,wire1,wire4,wire4unord,wire4rtx0,wire4dup").split(",");
const PORT = +arg("port", 8187), SIG_PORT = +arg("signal-port", 9047);
const SIGNAL = arg("signal", `127.0.0.1:${SIG_PORT}`), BID = arg("id", "pooled-wirelab-b-" + (LOOP ? process.pid : "x"));
const ROOT = path.resolve(arg("root", path.join(path.dirname(new URL(import.meta.url).pathname), "../..")));
const OUT = arg("out", "");

// transport.js as served: attachWire takes a channel id and options, so the lab can open several
// wire links (ordered, unordered, unreliable, one stripe) on the same peer connections. Nothing
// else changes; the anchors fail loudly if transport.js moves.
function transportSrc() {
  let s = fs.readFileSync(path.join(ROOT, "room/transport.js"), "utf8");
  const a1 = "export function attachWire(link, conn, onFrame, { ordered = true } = {}) {";
  const a2 = "{ negotiated: true, id: WIRE_ID, ordered, ...(ordered ? {} : { maxRetransmits: 0 }) }";
  if (!s.includes(a1) || !s.includes(a2)) throw new Error("room/transport.js changed: attachWire anchors not found");
  s = s.replace(a1, "export function attachWire(link, conn, onFrame, { ordered = true, __id = WIRE_ID, __opt = null } = {}) {");
  s = s.replace(a2, "{ negotiated: true, id: __id, ordered, ...(__opt ? (__opt.maxRetransmits != null ? { maxRetransmits: __opt.maxRetransmits } : {}) : (ordered ? {} : { maxRetransmits: 0 })) }");
  // wire4dup: every slice also goes out on the next stripe (a hedge against one association's
  // loss recovery); the receiver keeps the first copy (duplicate slices and frames are ignored)
  const a3 = "    const ch = open[(link.rr++) % open.length];\n    ch.send(buf);";
  if (!s.includes(a3)) throw new Error("room/transport.js changed: sendFrame anchor not found");
  s = s.replace(a3, a3 + "\n    if (link.dup && open.length > 1) open[link.rr % open.length].send(buf);");
  return s;
}
const srv = http.createServer((q, r) => {
  const u = new URL(q.url, "http://x");
  r.setHeader("cross-origin-opener-policy", "same-origin");
  r.setHeader("cross-origin-embedder-policy", "require-corp");
  try {
    if (u.pathname === "/lab.html") { r.setHeader("content-type", "text/html; charset=utf-8"); r.end(fs.readFileSync(path.join(ROOT, "tests/bench/wire_lab.html"))); return; }
    if (u.pathname === "/room/transport.js") { r.setHeader("content-type", "text/javascript; charset=utf-8"); r.end(transportSrc()); return; }
  } catch (e) { r.statusCode = 500; r.end(String(e)); return; }
  r.statusCode = 404; r.end();
}).listen(PORT, "127.0.0.1");

let peerServer = null;
if (ROLE === "a" && arg("signal-server", "1") !== "0") {
  peerServer = spawn(arg("peerjs", path.join(ROOT, "node_modules/.bin/peerjs")), ["--port", String(SIG_PORT), "--path", "/", ...(LOOP ? [] : ["--host", "0.0.0.0"])], { stdio: "ignore" });
  await new Promise((r) => setTimeout(r, 1500));
}
const mac = process.platform === "darwin";
const ARGS = [...(mac ? [] : ["--no-sandbox"]), "--headless=new", "--disable-gpu", "--disable-features=WebRtcHideLocalIpsWithMdns", "--disable-background-timer-throttling", ...(LOOP ? ["--allow-loopback-in-peer-connection"] : [])];
const CHROME = arg("chrome", mac ? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" : "");
const profiles = [], ctxs = [];
async function browser(role) {
  const prof = fs.mkdtempSync(path.join(os.tmpdir(), "wirelab-")); profiles.push(prof);
  const ctx = await chromium.launchPersistentContext(prof, { headless: false, args: ARGS, ...(CHROME ? { executablePath: CHROME } : {}) }); ctxs.push(ctx);
  const p = ctx.pages()[0];
  p.on("pageerror", (e) => console.error(role, "pageerror", String(e).slice(0, 300)));
  p.on("console", (m) => { if (m.type() === "error") console.error(role, "console", m.text().slice(0, 300)); });
  const stun = LOOP ? "&stun=0" : "";
  await p.goto(`http://127.0.0.1:${PORT}/lab.html?role=${role}&signal=${SIGNAL}&id=${BID}&to=${BID}${stun}`);
  await p.waitForFunction(() => window.lab?.loaded, null, { timeout: 30000 });
  return p;
}
const quant = (a, f) => { const s = a.filter(Number.isFinite).sort((x, y) => x - y); return s.length ? s[Math.min(s.length - 1, Math.floor(f * s.length))] : null; };
const pct = (a) => ({ p50: quant(a, 0.5), p90: quant(a, 0.9), p95: quant(a, 0.95), p99: quant(a, 0.99), max: a.length ? Math.max(...a) : null, n: a.length });
const f2 = (x) => (x == null ? "-" : x.toFixed(2));
const out = { role: ROLE, loopback: LOOP, reps: REPS, gap: GAP, host: os.hostname(), started: new Date().toISOString(), results: [] };

async function finish(code) {
  if (OUT) fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
  peerServer?.kill();
  await Promise.race([Promise.all(ctxs.map((c) => c.close().catch(() => {}))), new Promise((r) => setTimeout(r, 10000))]);
  for (const d of profiles) fs.rmSync(d, { recursive: true, force: true });
  srv.close();
  process.exit(code);
}

if (ROLE === "b") {
  const p = await browser("b");
  console.error("B ready as", BID);
  await p.waitForFunction(() => window.lab.done, null, { timeout: +arg("maxmin", 120) * 60e3, polling: 500 });
  await finish(0);
}

let pb = null;
if (LOOP) pb = await browser("b");
const pa = await browser("a");

function summarize(rows) {
  const ok = rows.filter(Boolean);
  const one = ok.map((r) => (r.rtt - r.turnB) / 2);
  const med = (k) => quant(ok.map((r) => r[k]), 0.5);
  // JS-visible cost on the two ends vs everything under the page (Chrome's IPC, usrsctp/dcSCTP, the
  // OS and the network): residual = rtt - A's send call - B's decode - B's turnaround - A's decode
  const resid = ok.map((r) => r.rtt - r.send - Math.max(0, r.decodeB) - r.turnB - Math.max(0, r.decodeA));
  return { n: ok.length, lost: rows.length - ok.length, oneway: pct(one), rtt: pct(ok.map((r) => r.rtt)),
    send: med("send"), turnB: med("turnB"), spreadB: med("spreadB"), decodeB: med("decodeB"), spreadA: med("spreadA"), decodeA: med("decodeA"),
    resid: quant(resid, 0.5), bufMax: Math.max(0, ...ok.map((r) => r.buf)) };
}
const line = (tag, s) => console.log(`${tag}  one-way p50 ${f2(s.oneway.p50)} p90 ${f2(s.oneway.p90)} p99 ${f2(s.oneway.p99)} max ${f2(s.oneway.max)}  rtt p50 ${f2(s.rtt.p50)}  send ${f2(s.send)} decB ${f2(s.decodeB)} sprB ${f2(s.spreadB)} turnB ${f2(s.turnB)} decA ${f2(s.decodeA)} sprA ${f2(s.spreadA)} resid ${f2(s.resid)}  buf ${s.bufMax}  lost ${s.lost}/${s.n + s.lost}`);

let runN = 0;
async function withRun(cfg, fn) {
  await pa.evaluate(([run, c]) => window.lab.open(run, c), [`r${process.pid}-${runN++}-`, cfg]);
  const ice = await pa.evaluate(() => window.lab.path());
  if (cfg.ka) await pa.evaluate((ms) => window.lab.keepalive(ms), cfg.ka);
  await pa.evaluate(() => window.lab.run("wire4", 4096, 40, 5));   // warm-up (congestion windows open)
  try { await fn(ice); } finally { await pa.evaluate(() => window.lab.close(false)); await pa.waitForTimeout(300); }
  return ice;
}
const run = (p, bytes, reps, gap) => pa.evaluate(([a, b, c, d]) => window.lab.run(a, b, c, d), [p, bytes, reps, gap]);

try {
  if (SUITES.includes("paths")) {
    await withRun({}, async (ice) => {
      out.ice = ice; console.log("ICE", JSON.stringify(ice));
      // interleave: every path once per size per round, rounds of reps/2 frames, 2 rounds
      for (let round = 0; round < 2; round++) for (const bytes of SIZES) for (const p of PATHS) {
        const rows = await run(p, bytes, Math.ceil(REPS / 2), GAP);
        const prev = out.results.find((r) => r.suite === "paths" && r.path === p && r.bytes === bytes);
        if (prev) prev.rows.push(...rows); else out.results.push({ suite: "paths", path: p, bytes, gap: GAP, rows });
      }
      for (const r of out.results.filter((x) => x.suite === "paths")) { Object.assign(r, summarize(r.rows)); line(`paths ${r.path.padEnd(10)} ${String(r.bytes / 1024).padStart(4)} KB`, r); }
    });
  }
  if (SUITES.includes("gap")) {
    const gaps = arg("gaps", "0,2,5,10,20,50,100,200,500,1000").split(",").map(Number);
    for (const ka of arg("ka", "0,10").split(",").map(Number)) {
      await withRun({ ka }, async () => {
        for (const bytes of [4096, 16384]) for (const gap of gaps) {
          const reps = Math.max(30, Math.min(REPS, Math.round(30000 / Math.max(gap, 1))));
          const rows = await run("wire4", bytes, reps, gap);
          const r = { suite: "gap", path: "wire4", bytes, gap, ka, rows, ...summarize(rows) }; out.results.push(r);
          line(`gap ka ${String(ka).padStart(2)} ${String(bytes / 1024).padStart(2)} KB idle ${String(gap).padStart(4)} ms`, r);
        }
      });
    }
  }
  if (SUITES.includes("tail")) {
    // long interleaved runs for the tail (p99.9, stalls): blocks of 100 frames per path in turn
    const tp = arg("tail-paths", "wire4,wire4dup").split(","), n = +arg("tail-n", 3000), tg = +arg("tail-gap", 25);
    for (const ka of arg("tail-ka", "10").split(",").map(Number)) {
      await withRun({ ka }, async () => {
        for (const bytes of arg("tail-sizes", "4").split(",").map((x) => +x * 1024)) {
          const acc = Object.fromEntries(tp.map((p) => [p, []]));
          for (let done = 0; done < n; done += 100) for (const p of tp) acc[p].push(...await run(p, bytes, Math.min(100, n - done), tg));
          for (const p of tp) {
            const one = acc[p].filter(Boolean).map((x) => (x.rtt - x.turnB) / 2);
            const r = { suite: "tail", path: p, bytes, gap: tg, ka, rows: acc[p], ...summarize(acc[p]), p999: quant(one, 0.999), over20: one.filter((x) => x > 20).length, over100: one.filter((x) => x > 100).length };
            out.results.push(r);
            line(`tail ka ${String(ka).padStart(2)} ${String(bytes / 1024).padStart(2)} KB ${p.padEnd(8)}`, r);
            console.log(`   p95 ${f2(r.oneway.p95)} p99.9 ${f2(r.p999)}  frames > 20 ms: ${r.over20}, > 100 ms: ${r.over100}`);
          }
        }
      });
    }
  }
  if (SUITES.includes("answer")) {
    const holdB = +arg("holdb", 7), holdA = +arg("holda", 12), laps = +arg("laps", 128), answers = +arg("answers", 4), think = +arg("think", 3000);
    for (const ka of arg("ka", "0,10").split(",").map(Number)) {
      await withRun({ ka, holdB }, async () => {
        for (const bytes of arg("answer-sizes", "4,10,16").split(",").map((x) => +x * 1024)) {
          const all = [], first = [];
          for (let a = 0; a < answers; a++) {
            await pa.waitForTimeout(think);
            const rows = await run(arg("answer-path", "wire4"), bytes, laps, holdA);
            first.push(rows[0]); all.push(...rows.slice(1));
          }
          const r = { suite: "answer", path: arg("answer-path", "wire4"), bytes, ka, holdA, holdB, laps, answers, think, rows: all, first: first.map((x) => x && (x.rtt - x.turnB) / 2), ...summarize(all) };
          out.results.push(r);
          line(`answer ka ${String(ka).padStart(2)} ${String(bytes / 1024).padStart(2)} KB`, r);
          console.log(`   first lap after ${think} ms idle, one-way: ${r.first.map(f2).join(" ")}`);
        }
      });
    }
  }
  if (!LOOP) await pa.evaluate(() => window.lab.bye()).catch(() => {});
} catch (e) { out.error = String(e).slice(0, 400); console.error("FAILED", out.error); }
// rows are kept for the tail plots; drop them with --slim
if (flag("slim")) for (const r of out.results) delete r.rows;
await finish(out.error ? 1 : 0);
