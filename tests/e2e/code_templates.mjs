// Code mode's starter templates (harness/templates.js, room/code.js newProject): two tabs in a real
// room (local PeerServer, loopback WebRTC, no WebGPU, no model needed):
//
//   host: New -> picks Game -> the name fills in -> Enter -> the files are in the tree, :5173 is
//   served (rev 1) and the game runs in the preview, the prompt box holds a first change to ask for
//   -> the guest sees the preview too -> the guest makes a project from the Form app template:
//   the host's timeline says who and which template, the host's prompt box is left alone -> an
//   empty project still starts empty.
//
//   NODE_PATH=<dir with peer + peerjs + playwright> node tests/e2e/code_templates.mjs [--headed] [--shots <dir>] --port 18996 --signal-port 9017
import fs from "fs";
import path from "path";
import { spawn } from "child_process";
import { loadPlaywright, chromiumPath, serveRepo } from "./engine_synth.mjs";

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const flag = (k) => argv.includes("--" + k);
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");
const SHOTS = arg("shots", "");
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
// the preview frame that holds a selector (the app runs in a sandboxed frame inside the relay)
async function inPreview(p, sel, timeout = 15000) {
  for (const end = Date.now() + timeout; Date.now() < end; await p.waitForTimeout(250)) {
    for (const f of p.frames()) if (await f.$(sel).catch(() => null)) return f;
  }
  return null;
}
let code = 1;
try {
  const errs = [];
  async function roomPage(name) {
    const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } });
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
  const base = `http://127.0.0.1:${PORT}/p2p.html?signal=127.0.0.1:${SIGNAL_PORT}&dev=0`;
  await host.goto(base + "&mock=code"); await guest.goto(base);
  for (const [p, n] of [[host, "host"], [guest, "guest"]]) {
    await p.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 30000 });
    await p.fill("#name-input", n + "-dev");
  }
  await host.click("#create-btn");
  await host.waitForFunction(() => /[A-Z0-9]{4}/.test(document.getElementById("room-badge").textContent), null, { timeout: 30000 });
  const room = (await host.textContent("#room-badge")).trim();
  await guest.fill("#code-input", room); await guest.click("#join-btn");
  for (const p of [host, guest]) await p.waitForFunction(() => document.querySelectorAll(".peer-card").length >= 2, null, { timeout: 30000 });

  await host.click("#mode-code");
  await host.waitForSelector("#code-project:not([hidden])", { state: "attached", timeout: 15000 });

  // ---- host: New -> Game
  await host.click("#code-new");
  check("New shows the name and the template picker", await host.isVisible("#code-new-name") && await host.isVisible("#code-new-tpl"));
  const opts = await host.evaluate(() => [...document.getElementById("code-new-tpl").options].map((o) => o.value));
  check("the picker: an empty project first, then the templates", opts[0] === "" && ["game", "dashboard", "form", "landing"].every((id) => opts.includes(id)), opts.join(","));
  await host.selectOption("#code-new-tpl", "game");
  await host.selectOption("#code-new-tpl", "landing");
  check("picking another one renames a name a template filled", (await host.inputValue("#code-new-name")) === "Landing page");
  await host.selectOption("#code-new-tpl", "game");
  check("picking a template fills an empty name", (await host.inputValue("#code-new-name")) === "Game");
  if (SHOTS) await host.screenshot({ path: path.join(SHOTS, "templates-new.png") });
  await host.press("#code-new-name", "Enter");
  check("the form closes", !(await host.isVisible("#code-new-tpl")) && await host.isVisible("#code-proj-select"));
  await host.waitForSelector('#code-tree .f[data-path="app.js"]', { timeout: 10000 }).catch(() => {});
  const tree = await host.evaluate(() => [...document.querySelectorAll("#code-tree .f")].map((f) => f.dataset.path).sort());
  check("host: the template's files are in the tree", JSON.stringify(tree) === JSON.stringify(["app.js", "index.html", "style.css"]), tree.join(","));
  await host.waitForFunction(() => document.getElementById("pv-state").textContent === "rev 1", null, { timeout: 15000 }).catch(() => {});
  check("host: served straight away", (await host.textContent("#pv-state")) === "rev 1", await host.textContent("#pv-state"));
  const game = await inPreview(host, "#game");
  check("host: the game runs in the preview", !!game && (await game.textContent("#lives")) === "3");
  check("host: the prompt box holds a first change", /high score/.test(await host.inputValue("#code-prompt")));
  check("host: the timeline says which template", /starts from the Game template/.test(await host.textContent("#code-log")));
  if (SHOTS) await host.screenshot({ path: path.join(SHOTS, "templates-host.png") });

  // ---- guest: sees it, then starts a Form app
  await guest.waitForSelector("#mode-code", { state: "visible", timeout: 15000 });
  await guest.click("#mode-code");
  await guest.waitForFunction(() => document.getElementById("pv-state").textContent === "rev 1", null, { timeout: 20000 }).catch(() => {});
  check("guest: the game's preview", !!(await inPreview(guest, "#game")));
  await host.fill("#code-prompt", "host draft");
  await guest.click("#code-new");
  await guest.selectOption("#code-new-tpl", "form");
  await guest.fill("#code-new-name", "signups");
  await guest.press("#code-new-name", "Enter");
  await host.waitForFunction(() => /guest-dev started the project signups from the Form app template/.test(document.getElementById("code-log").textContent), null, { timeout: 15000 }).catch(() => {});
  check("host: the note names the guest and the template", /guest-dev started the project signups from the Form app template/.test(await host.textContent("#code-log")));
  check("host: its own prompt box is left alone", (await host.inputValue("#code-prompt")) === "host draft");
  const form = await inPreview(guest, "#form", 20000);
  check("guest: the form app runs in its preview", !!form);
  if (form) {
    await form.fill("#name", "Ada"); await form.fill("#email", "ada@example.com"); await form.click("#add");
    check("guest: the form app works", (await form.textContent("#count")) === "1");
  }

  // ---- an empty project stays empty
  await host.click("#code-new");
  await host.fill("#code-new-name", "blank one");
  await host.press("#code-new-name", "Enter");
  await host.waitForFunction(() => /blank one<\/b> is empty|blank one is empty/.test(document.getElementById("code-log").innerHTML + document.getElementById("code-log").textContent), null, { timeout: 10000 }).catch(() => {});
  check("an empty project has no files", await host.evaluate(() => !document.querySelector("#code-tree .f")));
  check("no console errors", !errs.length, errs.join("\n"));
  code = results.every((r) => r.ok) ? 0 : 1;
} catch (e) {
  console.error(e); check("ran to the end", false, e.message);
} finally {
  await browser.close(); srv.close(); peerServer.kill();
}
console.log(`${results.filter((r) => r.ok).length}/${results.length} passed`);
process.exit(code);
