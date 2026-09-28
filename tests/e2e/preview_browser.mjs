// The Code-mode preview sandbox alone (docs/design/harness-app.md G.2): a PreviewServer over an
// in-memory project, mounted with mountPreview, driven through the serve / preview_logs tools.
// The page is on 127.0.0.1, so the preview runs isolated in harness/preview-relay.html on localhost
// (another site: its own process with --site-per-process, the default in desktop Chrome); one mount
// runs in local mode (a blob: frame of the page). No WebGPU, no room, no PeerJS; ~20 seconds.
//   node tests/e2e/preview_browser.mjs
import { loadPlaywright, chromiumPath, serveRepo } from "./engine_synth.mjs";
const PORT = 18986;

async function setup() {
  const { MemoryWorkspace, watch } = await import("/harness/workspace.js");
  const { PreviewServer } = await import("/harness/preview.js");
  const { previewTools } = await import("/harness/preview-tools.js");
  const { codingTools } = await import("/harness/codetools.js");
  const { mountPreview } = await import("/harness/preview-frame.js");
  const RJ = await import("/harness/run-js.js"), { runJsTool } = RJ;
  const c = new OffscreenCanvas(4, 4), g = c.getContext("2d");
  g.fillStyle = "#0f0"; g.fillRect(0, 0, 4, 4);
  const png = new Uint8Array(await (await c.convertToBlob({ type: "image/png" })).arrayBuffer());
  localStorage.setItem("room-secret", "s3cret");   // the frame must not see this
  const ws = watch(new MemoryWorkspace({
    "index.html": `<!doctype html>
<html><head><title>t</title><link rel="stylesheet" href="style.css"></head>
<body><canvas id="board" width="120" height="60"></canvas><img id="spr" src="img/sprite.png">
<a id="next" href="page2.html">next</a>
<script type="module" src="game.js"></script></body></html>`,
    "style.css": "body { background: rgb(1, 2, 3); margin: 0 }",
    "game.js": `import { draw } from "./lib/draw.js";
const ctx = document.getElementById("board").getContext("2d");
draw(ctx);
window.__level = await (await fetch("data/level.json")).json();
setTimeout(() => {
  boom();
}, 50);
`,
    "lib/draw.js": `import { COLOR } from "./color.js";\nexport function draw(ctx) {\n  console.log("drawing", COLOR);\n  ctx.fillStyle = COLOR; ctx.fillRect(0, 0, 120, 60);\n}\n`,
    "lib/color.js": `export const COLOR = "rgb(255, 0, 0)";\n`,
    "data/level.json": `{"rows": 20}`,
    "page2.html": `<!doctype html><p id="p2">page two</p><script>console.log("on page 2")</script>`,
  }));
  await ws.writeBytes("img/sprite.png", png);
  const server = new PreviewServer(ws);
  const T = Object.fromEntries([...codingTools(ws, { server }), ...previewTools(server), runJsTool(server)].map((t) => [t.name, t]));
  const el = document.createElement("div");
  el.style.cssText = "width:400px;height:300px";
  document.body.append(el);
  const logs = [];
  const view = mountPreview(el, server, 5173, { onLog: (e) => logs.push(e) });
  Object.assign(window, { __ws: ws, __server: server, __T: T, __logs: logs, __view: view, __mountPreview: mountPreview, __RJ: RJ });
  return await T.serve.run({});
}

const srv = serveRepo(PORT, {});
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromiumPath(), args: ["--no-sandbox", "--site-per-process"] });
const results = [];
const check = (n, ok, d = "") => { results.push(ok); console.log(`${ok ? "PASS" : "FAIL"} ${n}${d && !ok ? "  " + String(d).slice(0, 400) : ""}`); };
let code = 1;
try {
  const page = await browser.newPage();
  const pageErrors = [];
  page.on("pageerror", (e) => { if (!/boom is not defined|Cannot access 'ctx'/.test(String(e))) pageErrors.push(String(e)); });   // the app's own bug is expected
  await page.goto(`http://127.0.0.1:${PORT}/__blank.html`);
  const serveOut = await page.evaluate(setup);
  console.log("--- serve result\n" + serveOut + "\n---");
  check("serve reports the port and files", /^serving \. on :5173 \(index\.html, 8 files, [\d.]+ (KB|B)\)/.test(serveOut), serveOut);
  check("serve reports the load and the error with its file:line", /loaded in \d+ ms · 1 error:/.test(serveOut) && /error game\.js:6:\d+ ReferenceError: boom is not defined/.test(serveOut), serveOut);

  // the app's document: in the relay a srcdoc child, in local mode a blob: frame
  const frame = () => page.frames().find((f) => /^(about:srcdoc|blob:)/.test(f.url()));
  check("relay mode: the preview runs in the relay on the other site", await page.evaluate(() => window.__view.mode) === "relay"
    && page.frames().some((f) => f.url().startsWith(`http://localhost:${PORT}/harness/preview-relay.html`)), page.frames().map((f) => f.url()).join(" "));
  const F = await frame().evaluate(async (port) => {
    const out = {};
    const px = document.getElementById("board").getContext("2d").getImageData(5, 5, 1, 1).data;
    out.pixel = [...px];
    out.bg = getComputedStyle(document.body).backgroundColor;
    out.img = document.getElementById("spr").naturalWidth;
    out.origin = window.origin;
    out.base = document.baseURI;
    out.level = window.__level;
    const threw = (f) => { try { f(); return false; } catch { return true; } };
    out.parentDoc = threw(() => parent.document.title);
    out.parentStorage = threw(() => parent.localStorage.getItem("room-secret"));
    out.cookie = threw(() => document.cookie);
    localStorage.setItem("hi", "5");
    out.shim = localStorage.getItem("hi") === "5" && localStorage.getItem("room-secret") === null;
    out.opfs = await navigator.storage.getDirectory().then(() => "opened", (e) => "blocked: " + e.name);
    out.idb = await new Promise((res) => { try { const q = indexedDB.open("pooled-projects"); q.onsuccess = () => res("opened"); q.onerror = () => res("blocked"); } catch (e) { res("blocked: " + e.name); } });
    out.fetchRoom = await fetch(`http://127.0.0.1:${port}/p2p.html`).then(() => "fetched", (e) => "blocked: " + e.name);
    out.fetchRel = await fetch("/p2p.html").then((r) => r.status, () => "threw");
    out.imgRoom = await new Promise((res) => { const i = new Image(); i.onload = () => res("loaded"); i.onerror = () => res("blocked"); i.src = `http://127.0.0.1:${port}/favicon.svg`; });
    out.top = threw(() => { top.location.href = "https://example.com/"; });
    out.alert = (alert("hello"), "returned");
    return out;
  }, PORT);
  check("module chain of 3 drew the canvas", F.pixel.join() === "255,0,0,255", F.pixel);
  check("CSS applied", F.bg === "rgb(1, 2, 3)", F.bg);
  check("image from the project loaded", F.img === 4, F.img);
  check("fetch of a project file through the shim", F.level?.rows === 20, JSON.stringify(F.level));
  check("opaque origin", F.origin === "null", F.origin);
  check("the document's base URL is not the room page's", !F.base.includes("__blank") && !F.base.includes("127.0.0.1"), F.base);
  check("parent.document blocked", F.parentDoc);
  check("parent localStorage blocked", F.parentStorage);
  check("cookies blocked", F.cookie);
  check("localStorage shim works and holds nothing of the room's", F.shim);
  check("OPFS blocked", F.opfs.startsWith("blocked"), F.opfs);
  check("IndexedDB blocked", F.idb.startsWith("blocked"), F.idb);
  check("fetch to the room's origin refused by the CSP", F.fetchRoom.startsWith("blocked"), F.fetchRoom);
  check("rooted fetch is a 404 from the shim, not a request", F.fetchRel === 404, F.fetchRel);
  check("image from the room's origin refused", F.imgRoom === "blocked", F.imgRoom);
  check("top navigation blocked", F.top);
  check("alert does not block", F.alert === "returned");

  // the frame's messages reach this page after the evaluate above returns
  await page.waitForFunction(() => window.__logs.some((e) => e.text === "alert: hello") && window.__logs.some((e) => /404 p2p\.html \(fetch\)/.test(e.text)), null, { timeout: 5000 }).catch(() => {});
  const L = await page.evaluate(() => window.__logs.map((e) => ({ level: e.level, text: e.text.split("\n")[0], src: e.src, line: e.line })));
  check("console.log carries its file and line", L.some((e) => e.level === "log" && e.text === 'drawing rgb(255, 0, 0)' && e.src === "lib/draw.js" && e.line === 3), JSON.stringify(L));
  check("the uncaught error carries game.js:6", L.some((e) => e.level === "error" && /boom is not defined/.test(e.text) && e.src === "game.js" && e.line === 6), JSON.stringify(L));
  check("sandbox escapes logged as errors, alert as info", L.some((e) => /404 p2p\.html \(fetch\)/.test(e.text)) && L.some((e) => e.level === "info" && e.text === "alert: hello"), JSON.stringify(L));

  // live reload: an edit through the tools bumps the rev and the frame redraws
  const [edit, rev2] = await page.evaluate(async () => {
    let u = null;
    const off = window.__server.onUpdate((e) => { u ||= e; });
    const r = await window.__T.edit_file.run({ path: "lib/color.js", old: "rgb(255, 0, 0)", new: "rgb(0, 0, 255)" });
    off();
    return [r, u];
  });
  check("edit result names the preview's new rev", edit === "edited lib/color.js line 1 (1 -> 1 lines) · preview :5173 reloaded (rev 2)", edit);
  check("the update landed before the tool answered, with the changed file", rev2?.rev === 2 && rev2.changed.join() === "lib/color.js", JSON.stringify(rev2));
  await page.evaluate(() => window.__server.whenIdle(5173, 3000));
  const px2 = await frame().evaluate(() => [...document.getElementById("board").getContext("2d").getImageData(5, 5, 1, 1).data].join());
  check("frame reloaded with the edit", px2 === "0,0,255,255", px2);
  const logsOut = await page.evaluate(() => window.__T.preview_logs.run({ since: 0 }));
  console.log("--- preview_logs\n" + logsOut + "\n---");
  check("preview_logs lists both revs and a cursor", /\(rev 1\)/.test(logsOut) && /\(rev 2, current\)/.test(logsOut) && /next: since=\d+$/.test(logsOut), logsOut);

  // a relative link loads the other page in the same sandbox
  await frame().evaluate(() => document.getElementById("next").click());
  await page.waitForFunction(() => window.__logs.some((e) => e.text === "on page 2"), null, { timeout: 5000 }).catch(() => {});
  const p2 = await frame().evaluate(() => document.getElementById("p2")?.textContent).catch(() => null);
  check("link navigation rebuilds the document for page2.html", p2 === "page two", p2);

  // a second port from a subfolder: a top-level TDZ bug in a module and a missing image
  const broken = await page.evaluate(async () => {
    await window.__ws.write("broken/index.html", `<canvas id=b></canvas><img src="gone.png"><script type=module src="main.js"></script>`);
    await window.__ws.write("broken/main.js", `// the classic first-draft bug\nctx.fillRect(0, 0, 1, 1);\nconst ctx = document.getElementById("b").getContext("2d");\n`);
    const el = document.createElement("div"); document.body.append(el);
    window.__view2 = window.__mountPreview(el, window.__server, 5174);
    return window.__T.serve.run({ dir: "broken", port: 5174 });
  });
  console.log("--- serve :5174\n" + broken + "\n---");
  check("second port: TDZ error at main.js:2 and the missing image", /^serving broken on :5174 \(index\.html, 2 files/.test(broken)
    && /· 2 errors:/.test(broken) && /error main\.js:2:\d+ ReferenceError: Cannot access 'ctx' before initialization/.test(broken) && /\nmissing: gone\.png$/.test(broken), broken);
  const ports = await page.evaluate(() => window.__server.ports().map((p) => `${p.port}:${p.dir}`).join(" "));
  check("both ports listed", ports === "5173: 5174:broken", ports);

  // a peer-style mount waits for a click before running anything
  const gated = await page.evaluate(async () => {
    const el = document.createElement("div"); document.body.append(el);
    const v = window.__mountPreview(el, window.__server, 5173, { autorun: false });
    const before = !v.loaded && el.querySelector(".pv-run")?.textContent;
    el.querySelector(".pv-run").click();
    const after = v.loaded;
    v.destroy();
    return { before, after };
  });
  check("click-to-run gate", gated.before === "Run preview :5173" && gated.after, JSON.stringify(gated));

  // fit: a fixed-size app wider than a phone's preview (a 300x600 board and a 400px side panel) is
  // scaled down into the box, and a click and a key still reach it; pages that fit are left alone
  const wide = `<!doctype html><html><head><meta charset="utf-8"><style>body{margin:0;background:#14161f;color:#eee;font:14px monospace}
.game{display:flex;gap:0;width:700px;height:640px}.board{width:300px;height:600px;background:#223;border:2px solid #556}.side{width:396px;padding:0}
button{width:120px;height:40px;margin:20px}</style></head><body><div class="game"><canvas class="board" width="300" height="600"></canvas>
<div class="side"><p id="score">score 0</p><button id="fitbtn" onclick="document.getElementById('score').textContent='clicked'">Start</button></div></div>
<script>addEventListener("keydown", (e) => { document.getElementById("score").textContent = "key " + e.key; });</script></body></html>`;
  const appFrame = async (id) => {
    for (let i = 0; i < 50; i++) {
      for (const f of page.frames()) if (await f.evaluate((id) => !!document.getElementById(id), id).catch(() => false)) return f;
      await page.waitForTimeout(100);
    }
    return null;
  };
  const settle = () => page.waitForTimeout(700);
  await page.evaluate(async (wide) => {
    await window.__ws.write("wide/index.html", wide);
    await window.__ws.write("fluid/index.html", `<!doctype html><body style="margin:0"><p id=fl style="width:100%">a paragraph that wraps to any width</p></body>`);
    await window.__ws.write("vh/index.html", `<!doctype html><body style="min-height:100vh;padding:30px;margin:0"><p id=vh>full height plus padding</p></body>`);
    await window.__ws.write("long/index.html", `<!doctype html><body style="margin:0"><div id=lg style="height:3000px">a long page</div></body>`);
    window.__fit = {};
    for (const [dir, port] of [["wide", 5180], ["fluid", 5181], ["vh", 5182], ["long", 5183]]) {
      const el = document.createElement("div");
      el.style.cssText = "width:390px;height:600px;position:relative";
      el.id = "fit-" + dir;
      document.body.append(el);
      window.__fit[dir] = window.__mountPreview(el, window.__server, port);
      await window.__T.serve.run({ dir, port });
    }
  }, wide);
  const wf = await appFrame("fitbtn");
  await settle();
  const fitOf = () => page.evaluate(() => Object.fromEntries(Object.entries(window.__fit).map(([k, v]) => {
    const el = document.getElementById("fit-" + k), r = v.frame.getBoundingClientRect(), b = el.getBoundingClientRect();
    return [k, { scale: +v.scale.toFixed(3), w: Math.round(r.width), h: Math.round(r.height), inside: r.left >= b.left - 1 && r.top >= b.top - 1 && r.right <= b.right + 1 && r.bottom <= b.bottom + 1 }];
  })));
  let fit = await fitOf();
  const inner = await wf.evaluate(() => ({ sw: document.documentElement.scrollWidth, iw: innerWidth, ih: innerHeight }));
  check("fit: a 700px-wide app is scaled into a 390px box", fit.wide.scale > 0.5 && fit.wide.scale < 0.6 && fit.wide.inside && fit.wide.w <= 391, JSON.stringify({ fit, inner }));
  check("fit: the scaled app has no horizontal overflow", inner.sw <= inner.iw, JSON.stringify(inner));
  check("fit: a responsive page keeps scale 1", fit.fluid.scale === 1 && fit.fluid.w === 390, JSON.stringify(fit.fluid));
  check("fit: min-height 100vh plus padding is not shrunk", fit.vh.scale === 1, JSON.stringify(fit.vh));
  check("fit: a long page scrolls instead of shrinking", fit.long.scale === 1, JSON.stringify(fit.long));
  // a click where the button shows on screen reaches it (the transform maps the pointer)
  const btn = await wf.evaluate(() => { const r = document.getElementById("fitbtn").getBoundingClientRect(); return { x: r.left + r.width / 2, y: r.top + r.height / 2 }; });
  const at = await page.evaluate(({ x, y }) => { const f = window.__fit.wide.frame.getBoundingClientRect(), s = window.__fit.wide.scale; return { x: f.left + x * s, y: f.top + y * s }; }, btn);
  await page.mouse.click(at.x, at.y);
  check("fit: a click on the scaled app's button registers", await wf.evaluate(() => document.getElementById("score").textContent) === "clicked", JSON.stringify({ btn, at }));
  await page.keyboard.press("ArrowLeft");
  check("fit: keys reach the scaled app", await wf.evaluate(() => document.getElementById("score").textContent) === "key ArrowLeft");
  // a wider box: no scale; narrower again: scaled again
  await page.evaluate(() => { document.getElementById("fit-wide").style.cssText = "width:1000px;height:800px;position:relative"; });
  await settle();
  const big = (await fitOf()).wide;
  await page.evaluate(() => { document.getElementById("fit-wide").style.cssText = "width:390px;height:600px;position:relative"; });
  await settle();
  const small = (await fitOf()).wide;
  check("fit: follows the box (never above 1, down again when it narrows)", big.scale === 1 && small.scale < 0.6 && small.inside, JSON.stringify({ big, small }));
  // a new revision that fits starts over at scale 1
  await page.evaluate(() => window.__ws.write("wide/index.html", `<!doctype html><body style="margin:0"><p id=fitnew>fits now</p></body>`));
  await appFrame("fitnew");
  await settle();
  fit = await fitOf();
  check("fit: a new revision that fits is back at scale 1", fit.wide.scale === 1 && fit.wide.w === 390, JSON.stringify(fit.wide));
  await page.evaluate(() => { for (const [k, v] of Object.entries(window.__fit)) { v.destroy(); document.getElementById("fit-" + k).remove(); } for (const p of [5180, 5181, 5182, 5183]) window.__server.stop(p); });

  // local mode (no other site): a blob: document, whose base URL is not the page's either, and a
  // page that navigates itself away is put back, then stopped. It goes to a data: URL, which stays
  // in this process: a page that goes to another site (or to an error page) moves the frame to
  // another process and back each time, and Playwright sometimes loses track of such a frame and
  // dies on an internal assert (FrameManager.frameAttached; 1.49 here, the same code in 1.63).
  const local = await page.evaluate(async () => {
    await window.__ws.write("away/index.html", `<p id=a>stay</p><script>console.log("base " + document.baseURI); setTimeout(() => { location.href = "data:text/html,<p>away</p>"; }, 200)</script>`);
    const el = document.createElement("div"); document.body.append(el);
    const logs = [];
    let v = null, src = "";
    v = window.__mountPreview(el, window.__server, 5175, { relay: null, onLog: (e) => logs.push(e.text), onStatus: (s) => { if (s.state === "loading" && !src) src = v?.frame.src.slice(0, 5); } });
    await window.__T.serve.run({ dir: "away", port: 5175 });
    for (let i = 0; i < 60 && !logs.some((t) => /preview stopped/.test(t)); i++) await new Promise((r) => setTimeout(r, 100));
    const out = { mode: v.mode, src, logs };
    v.destroy();
    window.__server.stop(5175);
    return out;
  });
  check("local mode: a blob: document", local.mode === "local" && local.src === "blob:", JSON.stringify(local));
  check("local mode: base URL is a blob URL, not the room page", local.logs.some((t) => /^base blob:/.test(t) || t === "base index.html") && !local.logs.some((t) => /__blank/.test(t)), JSON.stringify(local.logs));
  check("a page navigating away is put back, then stopped", local.logs.filter((t) => /navigation blocked/.test(t)).length === 3 && local.logs.some((t) => /preview stopped/.test(t)), JSON.stringify(local.logs));

  // an infinite loop in the app hangs the relay's process, not this page: the heartbeat stops, the
  // frame is removed and the agent reads why
  const hang = await page.evaluate(async () => {
    await window.__ws.write("loop/index.html", `<p>spin</p><script>setTimeout(() => { for (;;) {} }, 300)</script>`);
    const el = document.createElement("div"); document.body.append(el);
    const states = [];
    const v = window.__mountPreview(el, window.__server, 5176, { onStatus: (s) => states.push(s.state) });
    await window.__T.serve.run({ dir: "loop", port: 5176 });
    let ticks = 0;
    const t0 = performance.now(), iv = setInterval(() => ticks++, 100);
    for (let i = 0; i < 80 && !states.includes("hung"); i++) await new Promise((r) => setTimeout(r, 100));
    clearInterval(iv);
    const ms = Math.round(performance.now() - t0);
    // the relay is one process: the visible preview :5173 hung with it, and its own watchdog says so
    // up to a second later; wait for that here, not in the middle of the run_js checks below
    for (let i = 0; i < 40 && !window.__logs.some((e) => /preview hung/.test(e.text)); i++) await new Promise((r) => setTimeout(r, 100));
    const logs = await window.__T.preview_logs.run({ port: 5176 });
    const out = { states, ticks, ms, frame: !!v.frame, gate: el.querySelector(".pv-run")?.textContent, logs };
    v.destroy();
    return out;
  });
  check("an infinite loop is detected; the room page kept running", hang.states.includes("hung") && hang.ticks >= hang.ms / 250 && !hang.frame, JSON.stringify(hang));
  check("the hang is in preview_logs and the pane offers to run it again", /preview hung \(infinite loop\?\)/.test(hang.logs) && /run :5176 again/.test(hang.gate || ""), JSON.stringify(hang));

  // run_js: a hidden frame of its own, through the relay; the visible preview is untouched
  const rj = await page.evaluate(async () => {
    const T = window.__T;
    await window.__ws.write("lib/tetris.js", "export function clear(rows) {\n  return rows.filter((r) => !r.every(Boolean));\n}\nexport function bad() {\n  throw new Error('no board');\n}\n");
    await new Promise((r) => setTimeout(r, 1500));   // the visible preview reloads for the new file
    const logs = window.__logs.length;
    const out = {};
    out.value = await T.run_js.run({ code: "import { clear } from './lib/tetris.js';\nconsole.log(clear([[1, 1], [0, 1]]).length);" });
    out.dom = await T.run_js.run({ code: "await new Promise((r) => setTimeout(r, 200));\nconsole.log(document.getElementById('board').width, window.__level.rows);", page: "index.html" });
    out.thrown = await T.run_js.run({ code: "import { bad } from './lib/tetris.js';\nbad();" });
    out.late = await T.run_js.run({ code: "setTimeout(() => console.log('late tick'), 40);\nconsole.log('first');" });
    out.fake = await T.run_js.run({ code: "window.__pvDone(true, 1, 'guess');\nawait new Promise((r) => setTimeout(r, 300));\nconsole.log('after the fake done');" });
    const t0 = performance.now();
    let ticks = 0;
    const iv = setInterval(() => ticks++, 100);
    out.loop = await T.run_js.run({ code: "for (;;) {}" });
    clearInterval(iv);
    out.loopMs = Math.round(performance.now() - t0); out.ticks = ticks;
    out.frames = [...document.querySelectorAll("iframe")].filter((f) => f.parentElement?.style.opacity === "0").length;   // run frames left
    out.visibleLogs = window.__logs.length - logs; out.newLogs = window.__logs.slice(logs).map((e) => e.text);
    // the visible preview runs again after the loop (its hung relay process was replaced quietly)
    await new Promise((r) => setTimeout(r, 1500));
    out.again = await T.serve.run({});
    out.hungLogs = window.__logs.slice(logs).filter((e) => /preview hung/.test(e.text)).length;
    // without a relay the runner refuses instead of running the loop in this tab
    const t1 = performance.now();
    const noRelay = window.__RJ.browserRunner(document, { mount: (el, src, port, o) => window.__mountPreview(el, src, port, { ...o, relay: null }) });
    const nr = await noRelay(window.__RJ.runSnapshot({ files: new Map() }, { code: "for (;;) {}" }), { timeout: 3000 });
    out.nohost = window.__RJ.formatRun(nr); out.nohostMs = Math.round(performance.now() - t1);
    return out;
  });
  console.log("--- run_js\n" + [rj.value, rj.dom, rj.thrown, rj.late, rj.fake, rj.loop, rj.again, rj.nohost].join("\n") + "\n---");
  check("run_js: a project function's value", /^ok in \d+ ms\n1$/.test(rj.value), rj.value);
  // (the page's own game.js throws "boom is not defined" 50 ms after load: that error is the run's)
  check("run_js: page loads first, the snippet sees its DOM and state; the page's own error is reported",
    /^error in \d+ ms/.test(rj.dom) && /^120 20$/m.test(rj.dom) && /error game\.js:6:\d+ ReferenceError: boom is not defined/.test(rj.dom), rj.dom);
  check("run_js: output from a timer right after the module finished is kept", /^ok in \d+ ms\nfirst\nlate tick$/.test(rj.late), rj.late);
  check("run_js: a page calling the done hook itself does not end the run", /after the fake done/.test(rj.fake), rj.fake);
  check("run_js: a throw is reported with its file:line", /^error in \d+ ms/.test(rj.thrown) && /Error: no board/.test(rj.thrown) && /error lib\/tetris\.js:5:\d+ Error: no board/.test(rj.thrown), rj.thrown);
  check("run_js: an infinite loop times out, the room page keeps running", /^timed out after 3 s/.test(rj.loop) && rj.loopMs < 6000 && rj.ticks >= rj.loopMs / 250, JSON.stringify(rj));
  // (the relay is one site, so one process: the loop also stalls the visible preview; its watchdog
  // pauses while a run frame lives, and it gets a fresh frame when the run ends hung)
  check("run_js: its frames are gone, no hang in the visible preview's logs", rj.frames === 0 && rj.hungLogs === 0, `frames ${rj.frames} logs ${JSON.stringify(rj.newLogs)}`);
  check("run_js: the visible preview runs again after the loop", /loaded in \d+ ms/.test(rj.again), rj.again);
  check("run_js: without a relay it refuses at once (no loop in the room's tab)", /needs the isolated preview host/.test(rj.nohost) && rj.nohostMs < 1000, `${rj.nohost} ${rj.nohostMs}`);

  const stopped = await page.evaluate(async () => { window.__server.stop(5173); await new Promise((r) => setTimeout(r, 50)); return !window.__view.loaded; });
  check("stopping the port clears the preview", stopped);
  check("no errors in the room page", !pageErrors.length, pageErrors.join("\n"));
  code = results.every(Boolean) ? 0 : 1;
  console.log(`\n${results.filter(Boolean).length}/${results.length} checks passed`);
  console.log(code ? "PREVIEW FAIL" : "PREVIEW PASS");
} catch (e) {
  console.error("FAILED:", e.stack || e);
  code = 2;
} finally {
  await browser.close(); srv.close();
}
process.exit(code);
