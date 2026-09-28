// Drive tests/bench/layer_prof.html (a room worker's layer slice, profiled per kernel) on an iPhone's
// Safari over WebDriver, or in Chrome. Manual trigger only.
//
//   node tests/bench/layer_prof.mjs --model models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf --lo 39 --hi 40
//        [--browser ios|chrome] [--ip 10.0.0.210] [--port 8443] [--wd-port 4460] [--n 40] [--cols 4]
//        [--phone 1] [--gaps 0,20,40] [--out result.json]
//
// ios: serves this checkout over https on --ip:--port with a throwaway self-signed certificate (the
//   phone needs a secure context for WebGPU; the WebDriver session accepts the certificate with
//   acceptInsecureCerts), weights from local disk with Range support, and opens the page on the phone
//   attached to this machine (safaridriver -p --wd-port; one WebDriver session per phone). The session
//   is always deleted. --phone defaults to 1 (the room's 256 MB buffer cap on phones).
// chrome: the same page on http://127.0.0.1 in headless Chrome (Playwright), --phone 0 by default.
import https from "https";
import http from "http";
import fs from "fs";
import os from "os";
import path from "path";
import { spawn, execSync } from "child_process";

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../..");
const BROWSER = arg("browser", "ios"), PORT = +arg("port", 8443), IP = arg("ip", "127.0.0.1");
const MODEL = arg("model"); if (!MODEL) throw new Error("--model <path under the checkout>");
const PHONE = arg("phone", BROWSER === "ios" ? "1" : "0");
const QS = `model=/${MODEL}&lo=${arg("lo", 0)}&hi=${arg("hi", +arg("lo", 0) + 1)}&n=${arg("n", 40)}&cols=${arg("cols", 4)}&phone=${PHONE}&gaps=${arg("gaps", "")}`;
const t0 = Date.now(); const log = (...a) => console.error(((Date.now() - t0) / 1000).toFixed(0) + "s", ...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".json": "application/json" };
function handler(q, r) {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { r.statusCode = 404; r.end(); return; }
  const size = fs.statSync(p).size, m = /bytes=(\d+)-(\d*)/.exec(q.headers.range || "");
  const lo = m ? +m[1] : 0, hi = m && m[2] ? Math.min(+m[2], size - 1) : size - 1;
  r.writeHead(m ? 206 : 200, { "content-type": MIME[path.extname(p)] || "application/octet-stream", "content-length": String(hi - lo + 1), "accept-ranges": "bytes", ...(m ? { "content-range": `bytes ${lo}-${hi}/${size}` } : {}), "cache-control": "no-store" });
  fs.createReadStream(p, { start: lo, end: hi }).pipe(r);
}
let result = null;
if (BROWSER === "chrome") {
  const { chromium } = await import("playwright");
  const srv = http.createServer(handler).listen(PORT, "127.0.0.1");
  const mac = process.platform === "darwin";
  const b = await chromium.launch({ headless: false, executablePath: mac ? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" : undefined,
    args: [...(mac ? [] : ["--no-sandbox", "--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan"]), "--headless=new", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-webgpu-developer-features"] });
  const p = await b.newPage();
  p.on("console", (m) => console.error("page:", m.text().slice(0, 300)));
  await p.goto(`http://127.0.0.1:${PORT}/tests/bench/layer_prof.html?${QS}`);
  await p.waitForFunction(() => window.RESULT, null, { timeout: 30 * 60e3, polling: 1000 });
  result = await p.evaluate(() => ({ ...window.RESULT, log: window.LOG() }));
  await b.close(); srv.close();
} else {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "layerprof-"));
  execSync(`openssl req -x509 -newkey rsa:2048 -nodes -keyout ${dir}/k.pem -out ${dir}/c.pem -days 1 -subj /CN=${IP} 2>/dev/null`);
  const srv = https.createServer({ key: fs.readFileSync(`${dir}/k.pem`), cert: fs.readFileSync(`${dir}/c.pem`) }, handler).listen(PORT, "0.0.0.0");
  const WD = `http://127.0.0.1:${+arg("wd-port", 4460)}`;
  const wd = async (method, p, body, ms = 120000) => {
    const r = await fetch(WD + p, { method, headers: { "content-type": "application/json" }, body: body ? JSON.stringify(body) : undefined, signal: AbortSignal.timeout(ms) });
    const j = await r.json().catch(() => ({})); if (!r.ok) throw new Error(`webdriver ${method} ${p}: ${r.status} ${JSON.stringify(j.value || j).slice(0, 300)}`); return j.value;
  };
  const driver = spawn("safaridriver", ["-p", String(+arg("wd-port", 4460))], { stdio: "ignore" });
  let sid = null;
  const done = async () => { if (sid) await wd("DELETE", `/session/${sid}`, null, 30000).catch(() => {}); sid = null; driver.kill(); srv.close(); fs.rmSync(dir, { recursive: true, force: true }); };
  process.on("SIGTERM", async () => { await done(); process.exit(1); });
  process.on("SIGINT", async () => { await done(); process.exit(1); });
  try {
    for (let i = 0; i < 40; i++) { try { await wd("GET", "/status", null, 2000); break; } catch { await sleep(250); } }
    sid = (await wd("POST", "/session", { capabilities: { alwaysMatch: { browserName: "safari", platformName: "iOS", acceptInsecureCerts: true } } }, 90000)).sessionId;
    await wd("POST", `/session/${sid}/url`, { url: `https://${IP}:${PORT}/tests/bench/layer_prof.html?${QS}` }, 120000);
    const ex = (s) => wd("POST", `/session/${sid}/execute/sync`, { script: s, args: [] });
    let last = "";
    for (const tEnd = Date.now() + 30 * 60e3; Date.now() < tEnd; await sleep(3000)) {
      const st = await ex("return [!!window.RESULT, window.LOG ? window.LOG() : document.body.innerText.slice(0, 400), isSecureContext]");
      if (st[1] !== last) { console.error(st[1].slice(last.length).trimEnd()); last = st[1]; }
      if (st[0]) { result = await ex("return window.RESULT"); break; }
    }
  } catch (e) { result = { error: String(e).slice(0, 400) }; }
  await done();
}
const line = JSON.stringify(result);
if (arg("out")) fs.writeFileSync(arg("out"), line);
console.log(line);
process.exit(result && !result.error ? 0 : 1);
