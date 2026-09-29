// Code mode on a phone (640px and narrower): one view at a time, Agent / Preview / Files, from a tab
// bar at the bottom (room/code-ui.js phone tabs). Two phone-sized tabs in a real room (local
// PeerServer, real WebRTC, no WebGPU), the host driving a scripted model (tests/scripted-model.js):
//
//   build -> the approval waits above the prompt -> the first app served opens Preview, its
//   700px-wide layout scaled to fit the phone, a click on its button still landing
//   -> a second request asks again while the host is on Files: a dot on Agent
//   -> approved from Agent -> the new revision puts a dot on Preview -> Files -> a file opens
//   full screen, Back returns to the tree -> the guest gets the same layout, prompt included
//   -> the host's phone in landscape (568x320 to 932x430) keeps the phone layout, and upright again after.
//
//   NODE_PATH=<dir with peer + peerjs + playwright> node tests/e2e/phone_code.mjs [--headed] [--width 390 --height 844]
//   --port 18995 --signal-port 9016
import fs from "fs";
import path from "path";
import { spawn } from "child_process";
import { loadPlaywright, chromiumPath, serveRepo } from "./engine_synth.mjs";

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const flag = (k) => argv.includes("--" + k);
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");
const PORT = +arg("port", 18995), SIGNAL_PORT = +arg("signal-port", 9016);
const W = +arg("width", 390), H = +arg("height", 844);
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

// the scripted model (runs in the host page): a page and its style, served; the second request edits the style
async function installModel() {
  const { scripted, xmlCall } = await import("/tests/scripted-model.js");
  const html = `<!doctype html><html><head><meta charset="utf-8"><title>Todo</title><link rel="stylesheet" href="style.css"></head><body><h1>Todo</h1><div class="board" style="display:flex;gap:20px;width:700px"><ul style="width:300px;height:400px;margin:0;background:#eef"><li>milk</li><li>eggs</li></ul><div style="width:380px"><p id="count">2 items</p><button id="add" style="width:140px;height:44px" onclick="document.getElementById('count').textContent='added'">Add</button></div></div><script>console.log("todo ready")</script></body></html>\n`;
  const css = "body { font: 16px system-ui; margin: 24px; }\nh1 { font-size: 22px; }\n";
  window.__pooledMock.model = scripted([
    "Two files: the page and its style.\n" + xmlCall("write_file", { path: "index.html", content: html }),
    xmlCall("write_file", { path: "style.css", content: css }),
    xmlCall("serve", { port: 5173 }),
    "The todo app is running on :5173.",
    "Making the title blue.\n" + xmlCall("edit_file", { path: "style.css", old: "h1 { font-size: 22px; }", new: "h1 { font-size: 22px; color: #2a45e0; }" }),
    "The title is blue now.",
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
let code = 1;
try {
  const errs = [];
  async function phonePage(name) {
    const ctx = await browser.newContext({ viewport: { width: W, height: H }, isMobile: true, hasTouch: true, deviceScaleFactor: 2 });
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
  const host = await phonePage("host"), guest = await phonePage("guest");
  const base = `http://127.0.0.1:${PORT}/p2p.html?split=memory&signal=127.0.0.1:${SIGNAL_PORT}&dev=0`;
  await host.goto(base + "&mock=code"); await guest.goto(base);
  for (const [p, n] of [[host, "host"], [guest, "guest"]]) {
    await p.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 30000 });
    await p.fill("#name-input", n + "-phone");
  }
  await host.tap("#create-btn");
  await host.waitForFunction(() => /[A-Z0-9]{4}/.test(document.getElementById("room-badge").textContent), null, { timeout: 30000 });
  const room = (await host.textContent("#room-badge")).trim();
  await guest.fill("#code-input", room); await guest.tap("#join-btn");
  for (const p of [host, guest]) await p.waitForFunction(() => document.querySelectorAll(".peer-card").length >= 2, null, { timeout: 30000 });

  // what shows: the selected tab, the panes on screen, the dots on the tabs
  const look = (p) => p.evaluate(() => {
    const vis = (id) => { const el = document.getElementById(id); return !!el && el.checkVisibility() && el.getBoundingClientRect().height > 0; };
    return {
      tab: document.getElementById("code-pane").dataset.ptab,
      bar: vis("code-tabs"), agent: vis("code-agent"), out: vis("code-out"), files: vis("code-files"), prompt: vis("code-row"),
      preview: vis("pv-frame-wrap"), editor: vis("ed"), back: vis("ed-back"),
      dots: Object.fromEntries([...document.querySelectorAll("#code-tabs [role=tab]")].map((b) => [b.dataset.ptab, b.classList.contains("badge")])),
      selected: [...document.querySelectorAll("#code-tabs [role=tab]")].filter((b) => b.getAttribute("aria-selected") === "true").map((b) => b.dataset.ptab).join(),
      overflow: document.documentElement.scrollWidth - innerWidth,
      barBottom: Math.round(innerHeight - document.getElementById("code-tabs").getBoundingClientRect().bottom),
      tabH: Math.round(document.querySelector("#code-tabs [role=tab]").getBoundingClientRect().height),
    };
  });

  await host.tap("#mode-code");
  await host.waitForSelector("#code-project:not([hidden])", { state: "attached", timeout: 15000 });
  let s = await look(host);
  check("the tab bar sits at the bottom, tabs 44px or taller", s.bar && s.barBottom <= 1 && s.tabH >= 44, JSON.stringify(s));
  check("Agent shows alone, with its prompt", s.agent && s.prompt && !s.out && !s.files && s.selected === s.tab, JSON.stringify(s));
  await host.tap("#ctab-files");
  s = await look(host);
  check("Files shows alone", s.files && !s.agent && !s.out && s.selected === "files", JSON.stringify(s));
  await host.tap("#ctab-agent");

  await host.evaluate(installModel);
  await host.tap("#code-auto-l");
  check("auto-approve off", !(await host.isChecked("#code-auto")));
  await host.fill("#code-prompt", "make a todo app");
  await host.tap("#code-send");
  await host.waitForSelector("#code-dock .cm-approve button.ok", { timeout: 20000 });
  check("the approval waits above the prompt", await host.evaluate(() => {
    const a = document.querySelector("#code-dock .cm-approve").getBoundingClientRect(), r = document.getElementById("code-row").getBoundingClientRect();
    return a.bottom <= r.top + 1 && a.top >= 0 && a.bottom <= innerHeight;
  }));
  check("the Agent tab pulses while the agent works", await host.evaluate(() => document.getElementById("ctab-agent").classList.contains("busy")));
  await host.tap("text=Allow edits for this task");
  await host.waitForFunction(() => document.getElementById("code-pane").dataset.ptab === "preview", null, { timeout: 30000 })
    .then(() => check("the first app served opens Preview", true), () => check("the first app served opens Preview", false));
  s = await look(host);
  check("Preview shows alone, the app filling it", s.preview && !s.agent && !s.files && s.selected === "preview", JSON.stringify(s));
  check("the preview frame fills most of the screen", await host.evaluate(() => document.getElementById("pv-frame-wrap").getBoundingClientRect().height > innerHeight * 0.35));
  // the sandbox tag is hidden on phones, so the address itself must not read as a server on this machine
  check("the preview address reads sandbox:5173, not localhost", /^sandbox:5173\//.test((await host.textContent("#pv-addr")).trim()), await host.textContent("#pv-addr"));
  // the app is 700px wide (a fixed layout, as a model writes a game): scaled into the phone's box, and a click still lands
  let app = null;
  for (let i = 0; i < 50 && !app; i++) {
    for (const f of host.frames()) if (await f.evaluate(() => !!document.getElementById("add")).catch(() => false)) app = f;
    if (!app) await host.waitForTimeout(100);
  }
  await host.waitForTimeout(800);
  const fitted = await host.evaluate(() => {
    const f = document.querySelector("#pv-frame-wrap iframe"), r = f.getBoundingClientRect(), w = document.getElementById("pv-frame-wrap").getBoundingClientRect();
    return { transform: f.style.transform, inside: r.left >= w.left - 1 && r.right <= w.right + 1 && r.top >= w.top - 1 && r.bottom <= w.bottom + 1, overflow: document.documentElement.scrollWidth - innerWidth };
  });
  const inApp = await app?.evaluate(() => ({ sw: document.documentElement.scrollWidth, iw: innerWidth }));
  check("a 700px-wide app is scaled to fit the phone's preview", /scale\(0\.\d+\)/.test(fitted.transform) && fitted.inside && fitted.overflow <= 0 && inApp && inApp.sw <= inApp.iw, JSON.stringify({ fitted, inApp }));
  const btn = await app.evaluate(() => { const r = document.getElementById("add").getBoundingClientRect(); return { x: r.left + r.width / 2, y: r.top + r.height / 2 }; });
  const at = await host.evaluate(({ x, y }) => {
    const f = document.querySelector("#pv-frame-wrap iframe"), r = f.getBoundingClientRect(), s = r.width / f.offsetWidth;
    return { x: r.left + x * s, y: r.top + y * s };
  }, btn);
  // a click where the button shows on screen (headless Chromium does not route synthetic touches
  // into the relay's cross-site frame, scaled or not, so this is a mouse click)
  await host.mouse.click(at.x, at.y);
  await app.waitForFunction(() => document.getElementById("count").textContent === "added", null, { timeout: 3000 }).catch(() => {});
  check("a click on a button inside the scaled app registers", await app.evaluate(() => document.getElementById("count").textContent) === "added", JSON.stringify({ btn, at }));
  await host.waitForFunction(() => document.querySelectorAll(".cm-stats").length >= 1, null, { timeout: 30000 });
  await host.waitForFunction(() => !document.getElementById("ctab-agent").classList.contains("busy"), null, { timeout: 5000 }).catch(() => {});
  check("done: the Agent tab stops pulsing", await host.evaluate(() => !document.getElementById("ctab-agent").classList.contains("busy")));

  // a second request; the host looks at Files while it waits for approval
  await host.tap("#ctab-agent");
  await host.fill("#code-prompt", "make the title blue");
  await host.tap("#code-send");
  await host.tap("#ctab-files");
  await host.waitForSelector("#code-dock .cm-approve button.ok", { state: "attached", timeout: 20000 });
  s = await look(host);
  check("an approval waiting elsewhere puts a dot on Agent", s.dots.agent && s.tab === "files", JSON.stringify(s));
  await host.tap("#ctab-agent");
  s = await look(host);
  check("Agent clears its dot when opened", !s.dots.agent, JSON.stringify(s));
  await host.tap("#code-dock button.ok");
  await host.waitForFunction(() => document.querySelectorAll(".cm-stats").length >= 2, null, { timeout: 30000 });
  await host.waitForFunction(() => document.getElementById("ctab-preview").classList.contains("badge"), null, { timeout: 10000 }).catch(() => {});
  s = await look(host);
  check("a new revision puts a dot on Preview (no switch)", s.dots.preview && s.tab === "agent", JSON.stringify(s));
  await host.tap("#ctab-preview");
  check("Preview clears its dot when opened", !(await look(host)).dots.preview);

  // Files: a file opens full screen, Back returns
  await host.tap("#ctab-files");
  await host.waitForSelector('#code-tree .f[data-path="style.css"]', { timeout: 10000 });
  await host.tap('#code-tree .f[data-path="style.css"]');
  await host.waitForFunction(() => /color: #2a45e0/.test(document.getElementById("ed-text").value), null, { timeout: 10000 }).catch(() => {});
  s = await look(host);
  check("a file opens full screen, with a way back", s.editor && s.back && !s.files && s.tab === "files", JSON.stringify(s));
  await host.tap("#ed-back");
  s = await look(host);
  check("Back returns to the tree", s.files && !s.editor, JSON.stringify(s));
  check("the tab is kept for the session", await host.evaluate(() => sessionStorage.getItem("pooled-code-tab")) === "files");
  // keyboard arrows move between the tabs
  await host.focus("#ctab-files"); await host.keyboard.press("ArrowRight");
  check("arrow keys move between the tabs", (await look(host)).tab === "agent");

  // the guest: the same layout, and a prompt
  await guest.waitForSelector("#mode-code", { state: "visible", timeout: 15000 });
  await guest.tap("#mode-code");
  await guest.waitForTimeout(800);
  s = await look(guest);
  check("guest: the tab bar and one view", s.bar && [s.agent, s.out, s.files].filter(Boolean).length === 1, JSON.stringify(s));
  await guest.tap("#ctab-agent");
  s = await look(guest);
  check("guest: Agent has the prompt", s.agent && s.prompt, JSON.stringify(s));
  await guest.tap("#ctab-preview");
  check("guest: Preview shows the app", (await look(guest)).preview);

  for (const [p, n] of [[host, "host"], [guest, "guest"]]) check(`${n}: no horizontal scroll`, (await look(p)).overflow <= 0);

  // the host's phone turned sideways (#142, #143): still the phone layout, however wide, with the
  // prompt above the tab bar, on screen, and the log (two tasks long by now) scrolling above it
  await host.tap("#ctab-agent");
  for (const [w, hh] of [[568, 320], [640, 360], [740, 360], [844, 390], [932, 430]]) {
    await host.setViewportSize({ width: w, height: hh });
    await host.waitForTimeout(300);
    const l = await host.evaluate(() => {
      const r = (id) => document.getElementById(id).getBoundingClientRect();
      const row = r("code-row"), tabs = r("code-tabs"), log = r("code-log");
      const hdr = document.querySelector("header");
      return { row: [row.top, row.bottom], tabs: [tabs.top, tabs.bottom], log: [log.top, log.bottom], h: innerHeight, w: innerWidth, tab: document.getElementById("code-pane").dataset.ptab,
        overflow: document.documentElement.scrollWidth - innerWidth, hdrRight: Math.max(...[...hdr.children].filter((c) => c.getBoundingClientRect().width).map((c) => c.getBoundingClientRect().right)) };
    });
    // the header row fits the screen: if it did not, the page would zoom out (innerWidth grows past
    // the screen's width, so overflow alone would not show it)
    check(`${w}x${hh}: the phone tabs, the prompt above them and on screen, the log gets the rest`,
      l.tab === "agent" && l.w === w && l.h === hh && l.hdrRight <= w && l.tabs[1] >= l.h - 1 && l.tabs[1] <= l.h + 1 && l.row[1] <= l.tabs[0] + 1 && l.row[0] >= l.log[1] - 1 && l.log[1] - l.log[0] >= 100 && l.overflow <= 0, JSON.stringify(l));
  }
  // turned upright again: the page is not left zoomed out, the tab bar is on screen and answers a tap
  await host.setViewportSize({ width: 390, height: 844 });
  await host.waitForTimeout(300);
  const up = await host.evaluate(() => ({ w: innerWidth, vv: Math.round(visualViewport.height), bottom: Math.round(document.getElementById("code-tabs").getBoundingClientRect().bottom) }));
  check("390x844 again: not zoomed, the tab bar at the bottom", up.w === 390 && up.vv === 844 && up.bottom <= 845, JSON.stringify(up));
  await host.tap("#ctab-files", { timeout: 5000 });
  check("390x844 again: a tap on Files lands", (await look(host)).files);
  check("no console errors", !errs.length, errs.join("\n"));
  code = results.every((r) => r.ok) ? 0 : 1;
} catch (e) {
  console.error(e); check("ran to the end", false, e.message);
} finally {
  await browser.close(); srv.close(); peerServer.kill();
}
console.log(`${results.filter((r) => r.ok).length}/${results.length} passed`);
process.exit(code);
