// Code mode's Download (room/code-export.js, harness/export.js) in a real room: a desktop host
// driving a scripted model (tests/scripted-model.js) and a phone guest, local PeerServer, real
// WebRTC, no WebGPU.
//
//   the agent writes a page, its style, a script and an image, and serves them
//   -> host: Download -> Project (.zip): every file, the .env left out, bytes intact
//   -> host: One HTML file: opened from disk it runs the app (script, style, image inlined) in a
//      sandboxed frame (opaque origin, the preview's CSP), and the page around it runs nothing
//   -> guest (phone): the share sheet gets the preview's files as a .zip; with no share sheet, a
//      download; a share refused for want of a fresh tap offers Save, which shares on the next tap
//
//   NODE_PATH=<dir with peer + peerjs + playwright> node tests/e2e/code_export.mjs [--headed] [--shots <dir>]
//   --port 18996 --signal-port 9017
import fs from "fs";
import os from "os";
import path from "path";
import zlib from "zlib";
import { spawn } from "child_process";
import { loadPlaywright, chromiumPath, serveRepo } from "./engine_synth.mjs";

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const flag = (k) => argv.includes("--" + k);
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");
const SHOTS = arg("shots", "");   // a folder: screenshots of the menu, desktop and phone
const PORT = +arg("port", 18996), SIGNAL_PORT = +arg("signal-port", 9017);
const PEERJS_URL = "https://cdn.jsdelivr.net/npm/peerjs@1.5.4/dist/peerjs.min.js";

function resolvePkg(name) {
  const dirs = [...(process.env.NODE_PATH || "").split(path.delimiter).filter(Boolean), path.join(ROOT, "node_modules")];
  for (const d of dirs) for (const base of [d, path.join(d, "node_modules")]) if (fs.existsSync(path.join(base, name, "package.json"))) return path.join(base, name);
  return null;
}
const peerjsDir = resolvePkg("peerjs"), peerDir = resolvePkg("peer");
if (!peerjsDir || !fs.existsSync(path.join(peerjsDir, "dist/peerjs.min.js"))) { console.log("SKIP: peerjs client missing (npm i --no-save peerjs@1.5.4, or put it on NODE_PATH)"); process.exit(0); }
if (!peerDir) { console.log("SKIP: the `peer` server package is missing"); process.exit(0); }
const peerjsJs = fs.readFileSync(path.join(peerjsDir, "dist/peerjs.min.js"));

// a zip read back in Node: name -> Buffer (stored or deflated)
function unzip(buf) {
  const e = buf.length - 22, out = new Map();
  if (buf.readUInt32LE(e) !== 0x06054b50) throw new Error("no end of central directory");
  let o = buf.readUInt32LE(e + 16);
  for (let k = buf.readUInt16LE(e + 10); k > 0; k--) {
    const method = buf.readUInt16LE(o + 10), csize = buf.readUInt32LE(o + 20), nl = buf.readUInt16LE(o + 28), lo = buf.readUInt32LE(o + 42);
    const name = buf.subarray(o + 46, o + 46 + nl).toString("utf8");
    const at = lo + 30 + buf.readUInt16LE(lo + 26) + buf.readUInt16LE(lo + 28), data = buf.subarray(at, at + csize);
    out.set(name, method === 8 ? zlib.inflateRawSync(data) : Buffer.from(data));
    o += 46 + nl + buf.readUInt16LE(o + 30) + buf.readUInt16LE(o + 32);
  }
  return out;
}

// the scripted model (runs in the host page): a page, its style, a script and a PNG, plus a .env; served
const PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
async function installModel(png) {
  const { scripted, xmlCall } = await import("/tests/scripted-model.js");
  const html = `<!doctype html><html><head><meta charset="utf-8"><title>Counter</title><link rel="stylesheet" href="style.css"></head><body><h1>Counter</h1><img id="dot" src="img/dot.png" width="8" height="8"><p id="n">0</p><button id="inc">+1</button><script type="module" src="js/app.js"></script></body></html>\n`;
  const css = "body { font: 16px system-ui; margin: 24px; } h1 { color: rgb(42, 69, 224); }\n";
  const js = `import { step } from "./step.js";\nconst n = document.getElementById("n");\ndocument.getElementById("inc").onclick = () => { n.textContent = String(+n.textContent + step); };\nn.textContent = "ready";\nsetTimeout(() => { n.textContent = "0"; }, 50);\n`;
  const step = "export const step = 1;\n";
  window.__pooledMock.model = scripted([
    "The page.\n" + xmlCall("write_file", { path: "index.html", content: html }),
    xmlCall("write_file", { path: "style.css", content: css }),
    xmlCall("write_file", { path: "js/app.js", content: js }),
    xmlCall("write_file", { path: "js/step.js", content: step }),
    xmlCall("write_file", { path: ".env", content: "API_KEY=do-not-ship\n" }),
    xmlCall("serve", { port: 5173 }),
    "The counter runs on :5173.",
  ], { piece: 16, delay: 2 });
  // the image: binary, written straight into the project (the model writes text)
  window.__pngB64 = png;
}

const srv = serveRepo(PORT, {});
const peerServer = spawn(process.execPath, [path.join(peerDir, "dist/bin/peerjs.js"), "--port", String(SIGNAL_PORT), "--host", "127.0.0.1", "--path", "/"], { stdio: "ignore" });
for (let i = 0; ; i++) {
  if (await fetch(`http://127.0.0.1:${SIGNAL_PORT}/peerjs/id`).then((r) => r.ok, () => false)) break;
  if (i > 50 || peerServer.exitCode !== null) { console.error("PeerServer did not start"); process.exit(2); }
  await new Promise((r) => setTimeout(r, 200));
}
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromiumPath(), headless: !flag("headed"),
  args: ["--no-sandbox", "--site-per-process", "--allow-loopback-in-peer-connection", "--disable-features=WebRtcHideLocalIpsWithMdns"] });
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-export-"));
const results = [];
const check = (name, ok, detail = "") => { results.push({ name, ok: !!ok }); console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail && !ok ? "  " + String(detail).slice(0, 400) : ""}`); };
let code = 1;
try {
  const errs = [];
  async function page(name, phone) {
    const ctx = await browser.newContext(phone ? { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 2, acceptDownloads: true }
      : { viewport: { width: 1400, height: 900 }, acceptDownloads: true });
    await ctx.route("**/*", (route) => {
      const url = route.request().url();
      if (url.startsWith(`http://127.0.0.1:${PORT}/`) || url.startsWith(`http://127.0.0.1:${SIGNAL_PORT}/`)) return route.continue();
      if (url === `http://localhost:${PORT}/harness/preview-relay.html`) return route.continue();
      if (url.split("?")[0] === PEERJS_URL) return route.fulfill({ status: 200, contentType: "text/javascript", body: peerjsJs });
      if (url.startsWith("https://fonts.googleapis.com/")) return route.fulfill({ status: 200, contentType: "text/css", body: "" });
      return route.abort();
    });
    const p = await ctx.newPage();
    p.on("console", (m) => { if (m.type() === "error" && !/net::ERR_FAILED|Failed to load resource|Could not connect to peer|img\/dot\.png/.test(m.text())) errs.push(`${name}: ${m.text().slice(0, 300)}`); });
    p.on("pageerror", (e) => errs.push(`${name} pageerror: ${String(e).slice(0, 300)}`));
    return p;
  }
  const host = await page("host", false), guest = await page("guest", true);
  // the guest's share sheet: records what it was given; "refuse" once answers NotAllowedError
  await guest.addInitScript(() => {
    window.__shares = [];
    window.__shareMode = "ok";
    navigator.canShare = (d) => !!d?.files?.length;
    navigator.share = async (d) => {
      if (window.__shareMode === "refuse") { window.__shareMode = "ok"; throw new DOMException("no user activation", "NotAllowedError"); }
      const f = d.files[0];
      window.__shares.push({ name: f.name, type: f.type, b64: btoa(String.fromCharCode(...new Uint8Array(await f.arrayBuffer()))) });
    };
  });
  const base = `http://127.0.0.1:${PORT}/p2p.html?signal=127.0.0.1:${SIGNAL_PORT}&dev=0`;
  await host.goto(base + "&mock=code" + (arg("hcore", "") ? "&hcore=" + arg("hcore") : "")); await guest.goto(base);
  for (const [p, n] of [[host, "host"], [guest, "guest"]]) {
    await p.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 30000 });
    await p.fill("#name-input", n);
  }
  await host.click("#create-btn");
  await host.waitForFunction(() => /[A-Z0-9]{4}/.test(document.getElementById("room-badge").textContent), null, { timeout: 30000 });
  const room = (await host.textContent("#room-badge")).trim();
  await guest.fill("#code-input", room); await guest.tap("#join-btn");
  for (const p of [host, guest]) await p.waitForFunction(() => document.querySelectorAll(".peer-card").length >= 2, null, { timeout: 30000 });

  await host.click("#mode-code");
  await host.waitForSelector("#code-project:not([hidden])", { state: "attached", timeout: 15000 });
  // before anything exists: the menu says so
  await host.click("#code-dl");
  check("an empty room: Download says there is nothing yet", await host.evaluate(() => /Nothing to download/.test(document.getElementById("code-dl-note").textContent)
    && [...document.querySelectorAll("#code-dl-menu button[data-dl]")].every((b) => b.disabled)));
  await host.keyboard.press("Escape");
  check("Escape closes the menu", await host.evaluate(() => document.getElementById("code-dl-menu").hidden && document.getElementById("code-dl").getAttribute("aria-expanded") === "false"));

  await host.evaluate(installModel, PNG_B64);
  await host.fill("#code-prompt", "make a counter");
  await host.click("#code-send");
  await host.waitForFunction(() => document.querySelectorAll(".cm-stats").length >= 1, null, { timeout: 60000 });
  await host.waitForSelector('#code-tree .f[data-path="js/step.js"]', { timeout: 10000 });
  // the image goes in through the project's workspace, as a picked file would (the preview served
  // before it shows a missing img/dot.png: the console filter above lets that one through)
  await host.evaluate(async () => {
    const { openProject, listProjects } = await import("/harness/projects.js");
    const cur = document.getElementById("code-proj-select").value, p = await openProject(cur || (await listProjects())[0].id);
    await p.ws.writeBytes("img/dot.png", Uint8Array.from(atob(window.__pngB64), (c) => c.charCodeAt(0)));
  });
  await host.click("#pv-reload").catch(() => {});
  await host.waitForTimeout(800);

  // ---- host: the zip
  await host.click("#code-dl");
  check("host: both choices on", await host.evaluate(() => [...document.querySelectorAll("#code-dl-menu button[data-dl]")].every((b) => !b.disabled)));
  let [dl] = await Promise.all([host.waitForEvent("download", { timeout: 15000 }), host.click('#code-dl-menu [data-dl="zip"]')]);
  const zipPath = path.join(tmp, dl.suggestedFilename());
  await dl.saveAs(zipPath);
  check("host: the zip is named after the project", /^[a-z0-9-]+\.zip$/.test(dl.suggestedFilename()), dl.suggestedFilename());
  const z = unzip(fs.readFileSync(zipPath)), names = [...z.keys()].sort();
  const top = names[0]?.split("/")[0];
  check("host: every file under one folder", names.length === 5 && names.every((n) => n.startsWith(top + "/"))
    && ["index.html", "style.css", "js/app.js", "js/step.js", "img/dot.png"].every((f) => z.has(`${top}/${f}`)), names.join(","));
  check("host: the .env stays out", !names.some((n) => n.endsWith(".env")));
  check("host: bytes intact (text and PNG)", z.get(`${top}/js/step.js`)?.toString() === "export const step = 1;\n" && z.get(`${top}/img/dot.png`)?.equals(Buffer.from(PNG_B64, "base64")));
  await host.waitForFunction(() => /Saved .*\.zip/.test(document.getElementById("code-dl-note").textContent), null, { timeout: 5000 }).catch(() => {});
  const zipNote = await host.textContent("#code-dl-note");
  if (SHOTS) await host.screenshot({ path: path.join(SHOTS, "export-desktop.png") });
  check("host: the note says what was saved and what was left out", /Saved .*\.zip · .* · 5 files · 1 hidden or secret file left out/.test(zipNote), zipNote);

  // ---- host: one HTML file
  [dl] = await Promise.all([host.waitForEvent("download", { timeout: 15000 }), host.click('#code-dl-menu [data-dl="html"]')])
    .catch(async (e) => { throw new Error(e.message.split("\n")[0] + " · note: " + await host.textContent("#code-dl-note")); });
  const htmlPath = path.join(tmp, dl.suggestedFilename());
  await dl.saveAs(htmlPath);
  check("host: the HTML file is named after the project", /^[a-z0-9-]+\.html$/.test(dl.suggestedFilename()), dl.suggestedFilename());
  const html = fs.readFileSync(htmlPath, "utf8");
  check("host: the page around the app runs no script", !/<script/i.test(html.replace(/srcdoc="[^"]*"/, "")));
  check("host: one sandbox flag, allow-scripts", /<iframe sandbox="allow-scripts"/.test(html) && !/allow-same-origin/.test(html));
  const ctx = await browser.newContext();
  const fp = await ctx.newPage();
  const net = [];
  fp.on("request", (r) => { if (!/^(file|data|blob|about):/.test(r.url())) net.push(r.url()); });
  await fp.goto("file://" + htmlPath);
  let app = null;
  for (let i = 0; i < 50 && !app; i++) {
    for (const f of fp.frames()) if (f !== fp.mainFrame() && await f.evaluate(() => !!document.getElementById("inc")).catch(() => false)) app = f;
    if (!app) await fp.waitForTimeout(100);
  }
  check("file: the app loads from disk", !!app);
  await app?.waitForFunction(() => document.getElementById("n").textContent === "0", null, { timeout: 5000 }).catch(() => {});
  await app?.click("#inc"); await app?.click("#inc");
  const st = await app?.evaluate(() => ({
    n: document.getElementById("n").textContent, color: getComputedStyle(document.querySelector("h1")).color,
    img: document.getElementById("dot").naturalWidth, origin: origin, csp: !!document.querySelector('meta[http-equiv="Content-Security-Policy"]'),
    storage: (() => { try { return typeof localStorage.getItem("x"); } catch { return "threw"; } })(),
  }));
  check("file: modules, style and image all inlined and working", st?.n === "2" && st.color === "rgb(42, 69, 224)" && st.img === 1, JSON.stringify(st));
  check("file: runs in an opaque origin under the preview's CSP", st?.origin === "null" && st.csp, JSON.stringify(st));
  check("file: nothing fetched from the network", !net.length, net.join(","));
  check("file: the tab is titled after the app", (await fp.title()) === "Counter");
  await ctx.close();

  // ---- guest (phone): the preview's files through the share sheet
  await guest.waitForSelector("#mode-code", { state: "visible", timeout: 15000 });
  await guest.tap("#mode-code");
  await guest.waitForFunction(() => document.querySelector("#pv-tabs .pv-tab"), null, { timeout: 15000 });
  await guest.tap("#ctab-files");
  await guest.tap("#code-dl");
  const tap = await guest.evaluate(() => Math.round(document.querySelector('#code-dl-menu [data-dl="zip"]').getBoundingClientRect().height));
  check("guest (phone): menu choices are 44px or taller", tap >= 44, tap);
  check("guest: the zip is the preview's files", /:5173/.test(await guest.textContent('#code-dl-menu [data-dl="zip"] span')));
  await guest.tap('#code-dl-menu [data-dl="zip"]');
  await guest.waitForFunction(() => window.__shares.length === 1, null, { timeout: 15000 }).catch(() => {});
  const sh = await guest.evaluate(() => window.__shares[0]);
  const gz = sh ? unzip(Buffer.from(sh.b64, "base64")) : new Map();
  check("guest: the share sheet gets a .zip", sh?.type === "application/zip" && /\.zip$/.test(sh.name), JSON.stringify(sh && { name: sh.name, type: sh.type }));
  check("guest: the served files, in one folder", [...gz.keys()].some((n) => n.endsWith("/js/step.js")) && [...gz.keys()].some((n) => n.endsWith("/img/dot.png")) && ![...gz.keys()].some((n) => n.endsWith(".env")), [...gz.keys()].join(","));
  if (SHOTS) await guest.screenshot({ path: path.join(SHOTS, "export-phone.png") });
  check("guest: the note says it was shared", /^Shared /.test(await guest.textContent("#code-dl-note")));
  // the share sheet refused (the tap too old): Save appears, and shares on the next tap
  await guest.evaluate(() => { window.__shareMode = "refuse"; });
  await guest.tap('#code-dl-menu [data-dl="html"]');
  await guest.waitForSelector("#code-dl-menu button.primary:not([hidden])", { timeout: 10000 });
  check("guest: a refused share sheet offers Save", /ready .*tap Save/.test(await guest.textContent("#code-dl-note")));
  await guest.tap("#code-dl-menu button.primary");
  await guest.waitForFunction(() => window.__shares.length === 2, null, { timeout: 10000 }).catch(() => {});
  const sh2 = await guest.evaluate(() => window.__shares[1]);
  check("guest: Save shares the HTML file", sh2?.type === "text/html" && /\.html$/.test(sh2.name) && /sandbox="allow-scripts"/.test(Buffer.from(sh2.b64, "base64").toString()));
  check("guest: Save goes away after", await guest.evaluate(() => document.querySelector("#code-dl-menu button.primary").hidden));
  // no share sheet at all: an ordinary download
  await guest.evaluate(() => { delete navigator.share; navigator.share = undefined; });
  [dl] = await Promise.all([guest.waitForEvent("download", { timeout: 15000 }), guest.tap('#code-dl-menu [data-dl="zip"]')]);
  check("guest: no share sheet, a download", /\.zip$/.test(dl.suggestedFilename()));
  check("guest: no horizontal scroll", await guest.evaluate(() => document.documentElement.scrollWidth - innerWidth <= 0));

  check("no console errors", !errs.length, errs.join("\n"));
  code = results.every((r) => r.ok) ? 0 : 1;
} catch (e) {
  console.error(e); check("ran to the end", false, e.message);
} finally {
  await browser.close(); srv.close(); peerServer.kill();
  fs.rmSync(tmp, { recursive: true, force: true });
}
console.log(`${results.filter((r) => r.ok).length}/${results.length} passed`);
process.exit(code);
