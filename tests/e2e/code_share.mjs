// Code mode, "Share with the room" (room/code.js, harness/app-export.js): two tabs in a real room
// (local PeerServer, loopback WebRTC, no WebGPU), the host driving a scripted model
// (tests/scripted-model.js):
//
//   build + serve -> the host shares :5173 rev 1 -> both timelines get the card -> the guest's
//   Download is the app as one .html file whose only content is a sandboxed frame (opened from
//   disk, the app runs in an opaque origin) -> Open full screen puts the guest's own preview frame
//   over the screen, Escape and the x bring it back -> a second request makes rev 2, the guest
//   shares it (the card names the guest) -> the first card still downloads rev 1.
//
//   NODE_PATH=<dir with peer + peerjs + playwright> node tests/e2e/code_share.mjs [--headed] [--shots <dir>] --port 18996 --signal-port 9017
import fs from "fs";
import path from "path";
import { spawn } from "child_process";
import { pathToFileURL } from "url";
import { loadPlaywright, chromiumPath, serveRepo } from "./engine_synth.mjs";

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const flag = (k) => argv.includes("--" + k);
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");
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

// the scripted model (runs in the host page): a counter app; the second request relabels it
async function installModel() {
  const { scripted, xmlCall } = await import("/tests/scripted-model.js");
  const html = `<!doctype html><html><head><meta charset="utf-8"><title>Counter</title></head><body><h1 id="t">Counter one</h1><p id="count">0</p><button id="add" onclick="document.getElementById('count').textContent=+document.getElementById('count').textContent+1">Add</button><script>console.log("counter ready")</script></body></html>\n`;
  window.__pooledMock.model = scripted([
    "The page.\n" + xmlCall("write_file", { path: "index.html", content: html }),
    xmlCall("serve", { port: 5173 }),
    "The counter is running on :5173.",
    "Relabeling.\n" + xmlCall("edit_file", { path: "index.html", old: "Counter one", new: "Counter two" }),
    "Done.",
  ], { piece: 12, delay: 4 });
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
const results = [];
const check = (name, ok, detail = "") => { results.push({ name, ok: !!ok }); console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail && !ok ? "  " + String(detail).slice(0, 400) : ""}`); };
const OUT = fs.mkdtempSync(path.join(process.env.TMPDIR || "/tmp", "code-share-"));
let code = 1;
try {
  const errs = [];
  async function roomPage(name) {
    const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 }, acceptDownloads: true });
    await ctx.route("**/*", (route) => {
      const url = route.request().url();
      if (url.startsWith(`http://127.0.0.1:${PORT}/`) || url.startsWith(`http://127.0.0.1:${SIGNAL_PORT}/`)) return route.continue();
      if (url === `http://localhost:${PORT}/harness/preview-relay.html`) return route.continue();
      if (url.split("?")[0] === PEERJS_URL) return route.fulfill({ status: 200, contentType: "text/javascript", body: peerjsJs });
      if (url.startsWith("https://fonts.googleapis.com/")) return route.fulfill({ status: 200, contentType: "text/css", body: "" });
      return route.abort();
    });
    const p = await ctx.newPage();
    p.on("console", (m) => { if (m.type() === "error" && !/net::ERR_FAILED|Failed to load resource|Could not connect to peer/.test(m.text())) errs.push(`${name}: ${m.text().slice(0, 300)}`); });
    p.on("pageerror", (e) => errs.push(`${name} pageerror: ${String(e).slice(0, 300)}`));
    return p;
  }
  const host = await roomPage("host"), guest = await roomPage("guest");
  const base = `http://127.0.0.1:${PORT}/p2p.html?ask=0&signal=127.0.0.1:${SIGNAL_PORT}&dev=0`;
  await host.goto(base + "&mock=code"); await guest.goto(base);
  for (const [p, n] of [[host, "host"], [guest, "guest"]]) {
    await p.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 30000 });
    await p.fill("#name-input", n + "-dev");
  }
  await host.click("#create-btn");
  await host.waitForFunction(() => /[A-Z0-9]{3}-?[A-Z0-9]{3}|[A-Z0-9]{4}/.test(document.getElementById("room-badge").textContent), null, { timeout: 30000 });
  const room = (await host.textContent("#room-badge")).trim();
  await guest.fill("#code-input", room); await guest.click("#join-btn");
  for (const p of [host, guest]) await p.waitForFunction(() => document.querySelectorAll(".peer-card").length >= 2, null, { timeout: 30000 });

  await host.click("#mode-code");
  await host.waitForSelector("#code-project:not([hidden])", { state: "attached", timeout: 15000 });
  await host.evaluate(installModel);
  await host.fill("#code-prompt", "make a counter");
  await host.click("#code-send");
  await host.waitForFunction(() => document.querySelectorAll(".cm-stats").length >= 1, null, { timeout: 30000 });
  await guest.waitForSelector("#mode-code", { state: "visible", timeout: 15000 });
  await guest.click("#mode-code");
  const revReady = (p, r) => p.waitForFunction((r) => document.getElementById("pv-state").textContent === `rev ${r}`, r, { timeout: 20000 });
  await revReady(host, 1); await revReady(guest, 1);

  check("host: Share shows beside Open", await host.isVisible("#pv-share"));
  check("guest (can drive): Share shows too", await guest.isVisible("#pv-share"));
  await host.click("#pv-share");
  for (const [p, n] of [[host, "host"], [guest, "guest"]]) {
    await p.waitForSelector(".cm-share", { timeout: 10000 }).catch(() => {});
    const t = await p.evaluate(() => [...document.querySelectorAll(".cm-share")].map((c) => c.textContent));
    check(`${n}: the share card names the host, the app and rev 1`, t.length === 1 && /host-dev shared/.test(t[0]) && /:5173 · rev 1/.test(t[0]), JSON.stringify(t));
  }
  if (arg("shots")) { await guest.waitForTimeout(300); await guest.screenshot({ path: path.join(arg("shots"), "share-guest.png") }); }
  await host.click("#pv-share");
  await host.waitForTimeout(500);
  check("sharing the same rev twice makes no second card", (await guest.$$(".cm-share")).length === 1);

  // guest: Download
  const download = async (p, nth, label) => {
    const [dl] = await Promise.all([p.waitForEvent("download", { timeout: 10000 }), p.locator(".cm-share").nth(nth).locator("button.ok").click()]);
    const file = path.join(OUT, label + "-" + dl.suggestedFilename());
    await dl.saveAs(file);
    return { name: dl.suggestedFilename(), file, text: fs.readFileSync(file, "utf8") };
  };
  const g1 = await download(guest, 0, "guest");
  check("guest: the download is named after the project and rev", /^[a-z0-9-]+-rev1\.html$/.test(g1.name), g1.name);
  const outer = g1.text.replace(/srcdoc="[^"]*"/, 'srcdoc=""');
  check("the file is only a sandboxed frame (no script of its own)", !/<script/i.test(outer) && /<iframe sandbox="allow-scripts" allow=""/.test(outer) && /Content-Security-Policy/.test(outer), outer.slice(0, 400));
  check("the file holds the app", g1.text.includes("Counter one"));
  const h1 = await download(host, 0, "host");
  check("host: the same file (both built from the same verified bytes)", h1.text === g1.text);

  // the file opened from disk: the app runs, in an opaque origin
  const fctx = await browser.newContext();
  const fp = await fctx.newPage();
  await fctx.route("**/*", (r) => (r.request().url().startsWith("file:") ? r.continue() : r.abort()));
  await fp.goto(pathToFileURL(g1.file).href);
  let appF = null;
  for (let i = 0; i < 50 && !appF; i++) {
    for (const f of fp.frames()) if (f !== fp.mainFrame() && await f.evaluate(() => !!document.getElementById("add")).catch(() => false)) appF = f;
    if (!appF) await fp.waitForTimeout(100);
  }
  check("opened from disk, the app runs", !!appF);
  if (appF) {
    await appF.click("#add"); await appF.click("#add");
    const st = await appF.evaluate(() => {
      let storage = "ok"; try { document.cookie = "x=1"; void document.cookie; } catch { storage = "blocked"; }   // (localStorage is the preview's in-memory shim)
      let top = "ok"; try { void parent.document.title; } catch { top = "blocked"; }
      return { count: document.getElementById("count").textContent, origin: self.origin, storage, top };
    });
    check("its buttons work, in an opaque origin with no storage and no reach into the page", st.count === "2" && st.origin === "null" && st.storage === "blocked" && st.top === "blocked", JSON.stringify(st));
  }
  await fctx.close();

  // guest: Open full screen, its own sandboxed preview frame
  await guest.locator(".cm-share").nth(0).locator("button:not(.ok)").click();
  await guest.waitForTimeout(400);
  const fs1 = await guest.evaluate(() => {
    const v = document.querySelector(".pv-view.pv-full"), f = v?.querySelector("iframe"), r = f?.getBoundingClientRect();
    return { full: !!v, api: document.fullscreenElement === v, sandbox: f?.getAttribute("sandbox"), w: Math.round(r?.width || 0), h: Math.round(r?.height || 0), iw: innerWidth, ih: innerHeight, x: !!v?.querySelector(".pv-full-x") };
  });
  if (arg("shots")) await guest.screenshot({ path: path.join(arg("shots"), "share-full.png") });
  check("guest: Open full screen puts its sandboxed preview over the screen", fs1.full && fs1.sandbox === "allow-scripts" && fs1.w >= fs1.iw - 2 && fs1.h >= fs1.ih - 2 && fs1.x, JSON.stringify(fs1));
  await guest.keyboard.press("Escape");
  await guest.waitForTimeout(400);
  check("Escape leaves full screen", await guest.evaluate(() => !document.querySelector(".pv-full") && !document.fullscreenElement));
  await guest.locator(".cm-share").nth(0).locator("button:not(.ok)").click();
  await guest.waitForTimeout(300);
  await guest.click(".pv-full-x");
  await guest.waitForTimeout(300);
  check("the x leaves full screen too", await guest.evaluate(() => !document.querySelector(".pv-full") && !document.fullscreenElement));
  check("the preview still runs after full screen", await guest.evaluate(() => document.getElementById("pv-state").textContent) === "rev 1");

  // rev 2; the guest shares it
  await host.fill("#code-prompt", "relabel it");
  await host.click("#code-send");
  await host.waitForFunction(() => document.querySelectorAll(".cm-stats").length >= 2, null, { timeout: 30000 });
  await revReady(guest, 2);
  await guest.click("#pv-share");
  for (const [p, n] of [[host, "host"], [guest, "guest"]]) {
    await p.waitForFunction(() => document.querySelectorAll(".cm-share").length >= 2, null, { timeout: 10000 }).catch(() => {});
    const t = await p.evaluate(() => [...document.querySelectorAll(".cm-share")].map((c) => c.textContent));
    check(`${n}: the guest's share names the guest, rev 2`, t.length === 2 && /guest-dev shared/.test(t[1]) && /rev 2/.test(t[1]), JSON.stringify(t));
  }
  const g2 = await download(guest, 1, "guest2");
  check("the new card downloads rev 2", /-rev2\.html$/.test(g2.name) && g2.text.includes("Counter two") && !g2.text.includes("Counter one"), g2.name);
  const g1b = await download(guest, 0, "guest1b");
  check("the first card still downloads rev 1 (kept on the device)", /-rev1\.html$/.test(g1b.name) && g1b.text === g1.text, g1b.name);

  check("no console errors", !errs.length, errs.join("\n"));
  code = results.every((r) => r.ok) ? 0 : 1;
} catch (e) {
  console.error(e); check("ran to the end", false, e.message);
} finally {
  await browser.close(); srv.close(); peerServer.kill();
  fs.rmSync(OUT, { recursive: true, force: true });
}
console.log(`${results.filter((r) => r.ok).length}/${results.length} passed`);
process.exit(code);
