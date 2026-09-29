// Drive tests/bench/readback_probe.html on an iPhone's Safari over WebDriver (run on the machine the
// phone is USB-attached to) or in Chrome (Playwright). Manual trigger only.
//
//   node tests/bench/readback_probe.mjs [--browser ios|chrome] [--ip 10.0.0.210] [--port 8443] [--wd-port 4460]
//        [--q 'ms=4&dim=2048&n=60&rounds=5&gap=0'] [--out result.json]
// ios: serves this checkout over https (self-signed, accepted by the WebDriver session: WebGPU needs a
//   secure context) and opens the page on the phone; the session is always deleted.
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
const PAGE = `/tests/bench/readback_probe.html?${arg("q", "")}`;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8" };
function handler(q, r) {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { r.statusCode = 404; r.end(); return; }
  r.writeHead(200, { "content-type": MIME[path.extname(p)] || "application/octet-stream", "cache-control": "no-store" });
  fs.createReadStream(p).pipe(r);
}
let result = null;
if (BROWSER === "chrome") {
  const { chromium } = await import("playwright");
  const srv = http.createServer(handler).listen(PORT, "127.0.0.1");
  const mac = process.platform === "darwin";
  const b = await chromium.launch({ headless: false, executablePath: mac ? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" : undefined,
    args: [...(mac ? [] : ["--no-sandbox", "--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan"]), "--headless=new", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist"] });
  const p = await b.newPage();
  p.on("console", (m) => console.error("page:", m.text().slice(0, 400)));
  await p.goto(`http://127.0.0.1:${PORT}${PAGE}`);
  await p.waitForFunction(() => window.RESULT, null, { timeout: 20 * 60e3, polling: 500 });
  result = await p.evaluate(() => window.RESULT);
  await b.close(); srv.close();
} else {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "rbprobe-"));
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
    await wd("POST", `/session/${sid}/url`, { url: `https://${IP}:${PORT}${PAGE}` }, 120000);
    const ex = (s) => wd("POST", `/session/${sid}/execute/sync`, { script: s, args: [] });
    let last = "";
    for (const tEnd = Date.now() + 20 * 60e3; Date.now() < tEnd; await sleep(2000)) {
      const st = await ex("return [!!window.RESULT, window.LOG ? window.LOG() : document.body.innerText.slice(0, 400)]");
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
