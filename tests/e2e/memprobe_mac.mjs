// Peak vs steady memory of a layer load in desktop Safari (issue #207): the same WebKit fetch,
// Response and writeBuffer paths the iPhone runs, but on a Mac, where `footprint` reads the tab's
// real process footprint (what iOS jetsam counts) instead of guessing it from outside.
//
//   node tests/e2e/memprobe_mac.mjs --root <repo checkout> [--port 8799] --run "mode=load&model=moe&lo=39" ...
//
// Serves the checkout on 127.0.0.1, opens each run in a new Safari tab (`open -a Safari`), finds the
// tab's WebContent process (the one that was not there before), and samples the phys_footprint of
// it and of Safari's GPU process every 200 ms. The page POSTs each of its events to /mp-log
// (?beacon=1), so every sample is tagged with what the page was doing: the peak comes with the
// tensor and step that caused it. One JSON line per run on stdout. Take the Mac GPU with devq.sh.
import http from "http";
import fs from "fs";
import path from "path";
import { execFileSync, execFile, spawn } from "child_process";

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const runs = argv.flatMap((a, i) => (a === "--run" ? [argv[i + 1]] : []));
const ROOT = path.resolve(arg("root", "."));
const PORT = +arg("port", 8799);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const t0 = Date.now();
const log = (...a) => console.error(((Date.now() - t0) / 1000).toFixed(0) + "s [memprobe-mac]", ...a);
const MIME = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".css": "text/css", ".wgsl": "text/plain", ".svg": "image/svg+xml" };

let cur = null;   // the run being measured
const srv = http.createServer((req, res) => {
  if (req.method === "POST" && req.url === "/mp-log") {
    let b = ""; req.on("data", (c) => (b += c)); req.on("end", () => {
      let stop = false;
      try { const e = JSON.parse(b); if (cur && e.run === cur.run && !cur.over) { if (e.ev !== "hb") { cur.ev = e; cur.events.push({ ...e, at: Date.now() - cur.t0 }); } } else stop = true; } catch {}
      res.end(stop ? "stop" : "ok");
    });
    return;
  }
  const u = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const f = path.join(ROOT, u === "/" ? "index.html" : u);
  if (!f.startsWith(ROOT) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { "content-type": MIME[path.extname(f)] || "application/octet-stream", "cache-control": "no-store" });
  fs.createReadStream(f).pipe(res);
});
await new Promise((r) => srv.listen(PORT, "127.0.0.1", r));
// Safari suspends pages while the display sleeps: keep it awake for the whole batch
const awake = spawn("caffeinate", ["-dimu", "-t", "7200"], { stdio: "ignore" });

const pids = (pat) => { try { return execFileSync("pgrep", ["-f", pat]).toString().trim().split("\n").filter(Boolean).map(Number); } catch { return []; } };
const fp = (pid) => new Promise((res) => execFile("footprint", ["-p", String(pid)], { timeout: 3000 }, (err, out) => {
  const m = /phys_footprint:\s+([\d.]+)\s*(KB|MB|GB)/.exec(out || ""), pk = /phys_footprint_peak:\s+([\d.]+)\s*(KB|MB|GB)/.exec(out || "");
  const mb = (x) => x ? +x[1] * { KB: 1 / 1024, MB: 1, GB: 1024 }[x[2]] : null;
  res({ mb: mb(m), peak: mb(pk) });
}));

for (const q of runs) {
  const run = "m" + Date.now().toString(36);
  const before = new Set(pids("com.apple.WebKit.WebContent"));
  const gpuPid = pids("com.apple.WebKit.GPU")[0];
  const gpu0 = gpuPid ? (await fp(gpuPid)).mb : null;
  cur = { run, q, t0: Date.now(), ev: null, events: [], samples: [] };
  execFileSync("open", ["-a", "Safari", `http://127.0.0.1:${PORT}/memprobe.html?${q}&beacon=1&run=${run}&hold=${arg("hold", 4000)}`]);
  let wc = null, peak = { wc: 0 }, gpuPeak = { gpu: 0 }, end = "timeout";
  const tEnd = Date.now() + +arg("maxmin", 15) * 60e3;
  while (Date.now() < tEnd) {
    await sleep(200);
    // the tab's process: Safari may start more than one (a prewarmed spare), so take the biggest new one
    const fresh = pids("com.apple.WebKit.WebContent").filter((p) => !before.has(p));
    const fps = await Promise.all(fresh.map(async (p) => [p, await fp(p)]));
    let a = { mb: null };
    for (const [p, f] of fps) if (f.mb != null && (a.mb == null || f.mb > a.mb)) { a = f; if (wc !== p) { wc = p; log(q, "tab process", wc, f.mb, "MB"); } }
    const g = gpuPid ? await fp(gpuPid) : { mb: null };
    const ev = cur.ev || {};
    const s = { t: Date.now() - cur.t0, wc: a.mb, gpu: g.mb != null && gpu0 != null ? +(g.mb - gpu0).toFixed(1) : null, phase: ev.phase, ev: ev.ev, info: (ev.info || "").slice(0, 60), gpuAcct: ev.gpuMB, jsAcct: ev.jsAcctMB };
    cur.samples.push(s);
    if (a.mb > peak.wc) peak = { ...s, wcPeakCounter: a.peak };
    if (s.gpu > gpuPeak.gpu) gpuPeak = s;
    if (ev.phase === "done" || ev.phase === "error") { end = ev.phase; break; }
  }
  cur.over = true;   // the page's next heartbeat gets "stop" and empties the tab
  // steady: the tab after the load settled (the page waits ?settle ms before "done")
  const steady = cur.samples.slice(-3);
  const last = wc ? await fp(wc) : {};
  if (last.peak == null) last.peak = Math.max(0, ...cur.samples.map((x) => x.wc || 0));
  const out = { q, run, end, wcPid: wc, wcPeakMB: last.peak, wcPeakSample: peak, gpuPeakSample: gpuPeak,
    steadyWcMB: steady.length ? Math.min(...steady.map((x) => x.wc ?? Infinity)) : null, steadyGpuMB: steady.length ? steady[steady.length - 1].gpu : null,
    result: cur.events.find((e) => e.ev === "done")?.info || cur.events.find((e) => e.ev === "error")?.info || null,
    // the 12 highest samples, with what the page was doing
    top: [...cur.samples].sort((x, y) => (y.wc || 0) - (x.wc || 0)).slice(0, 12),
    timeline: cur.samples.filter((_, i) => i % 5 === 0).map((s) => [s.t, s.wc, s.gpu, s.ev, s.info]) };
  console.log(JSON.stringify(out));
  log(q, "->", end, "tab peak", out.wcPeakMB, "MB (sampled", peak.wc, "during", peak.ev, peak.info, ") steady", out.steadyWcMB, "gpu +", out.steadyGpuMB);
  await sleep(+arg("hold", 4000) + 4000);
}
srv.close();
awake.kill();
process.exit(0);
