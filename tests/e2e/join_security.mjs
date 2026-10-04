// Joining a room (room/joingate.js, docs/protocol.md "Joining a room") in headless Chromium on one
// machine: a local PeerServer, a host and three guests, no GPU and no model (the room is never
// started). Manual trigger (node tests/e2e/join_security.mjs [--shots DIR]); needs `npm install`
// (playwright, peer).
//
//   1 the host's room has a six-character code, shown as 4TK-G9P
//   2 a tab that opens the invite link (#k=) is in at once: no prompt on the host, no key left in its address bar
//   3 two tabs that type the code wait ("Waiting for the host to let you in"), get nothing from the room,
//     and the host sees one prompt with "1 more waiting", without it taking focus
//   4 Allow (keyboard) lets the first in; Deny sends the second back to the join screen with a message,
//     and it does not knock again
//   5 reloading the allowed tab brings it back without a prompt; so does reloading the host
//   6 pooled serve (cli/, needs `cd cli && npm ci`): with the code alone the host is asked (API client)
//     and Allow lets it in; with the invite link it is in at once
//   7 Ask off: a typed code gets in without a prompt
//   8 a room node device (pooled join's, packages/room-node, no GPU) with the invite link: in at once,
//     the host's proof checked and the room's mesh key in hand; it links to a browser tab of the room
//     and the two prove the mesh key to each other (stripes too)
// Proofs (room/chanauth.js): 2 and 6 come in by proving the key; 3 checks that the host's prompt and
// the waiting tab show the same six digits.
// --shots DIR: screenshots of the prompt and the waiting screen at 1280 and 390 wide.
import { chromium } from "playwright";
import http from "http";
import fs from "fs";
import path from "path";
import { spawn } from "child_process";

const arg = (k, d) => { const i = process.argv.indexOf("--" + k); return i > 0 ? process.argv[i + 1] : d; };
const PORT = +arg("port", 8351), SIG = PORT + 1;
const SHOTS = arg("shots", "");
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png", ".woff2": "font/woff2" };
const srv = http.createServer((q, r) => {
  let p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (/^\/(r\/[^/]+|room)\/?$/.test(q.url.split("?")[0])) p = path.join(ROOT, "p2p.html");   // vercel.json rewrites
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { r.statusCode = 404; r.end(); return; }
  r.setHeader("content-type", MIME[path.extname(p)] || "application/octet-stream"); fs.createReadStream(p).pipe(r);
}).listen(PORT, "127.0.0.1");
let sigProc = null;
async function peerServer(port) {
  sigProc = spawn(path.join(ROOT, "node_modules/.bin/peerjs"), ["--port", String(port), "--path", "/"], { stdio: "ignore" });
  for (let i = 0; i < 50; i++) {
    await new Promise((r) => setTimeout(r, 200));
    const ok = await new Promise((r) => http.get(`http://127.0.0.1:${port}/peerjs/id`, (res) => { res.resume(); r(res.statusCode === 200); }).on("error", () => r(false)));
    if (ok) return;
  }
  throw new Error("peer server did not start");
}

const t0 = Date.now(); const T = () => ((Date.now() - t0) / 1000).toFixed(1) + "s";
const log = (...a) => console.error(T(), ...a);
const results = [];
const check = (name, ok, extra = "") => { results.push({ name, ok }); log(ok ? "PASS" : "FAIL", name, extra); };
const UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";
const BASE = `http://127.0.0.1:${PORT}/p2p.html?signal=127.0.0.1:${SIG}&dev=0`;

let browser;
const pages = [], bridges = [];
async function tab(name, url = BASE) {
  const ctx = await browser.newContext({ userAgent: UA, viewport: { width: 1280, height: 800 } });
  await ctx.addInitScript((n) => { try { if (!sessionStorage.getItem("pooled-name")) sessionStorage.setItem("pooled-name", n); } catch {} }, name);
  const p = await ctx.newPage();
  p.errs = []; p.label = name;
  p.on("pageerror", (e) => p.errs.push(String(e).slice(0, 200)));
  await p.goto(url);
  await p.waitForFunction(() => typeof Peer === "function" && window.pooledWired, null, { timeout: 30000 });
  if (!url.includes("code=") && !url.includes("#k=")) { await p.fill("#name-input", name); if (!(await p.inputValue("#join-gb"))) await p.fill("#join-gb", "1"); }
  pages.push(p);
  return p;
}
const inRoom = (p) => p.evaluate(() => document.body.classList.contains("in-room"));
const waitIn = (p, ms = 30000) => p.waitForFunction(() => document.body.classList.contains("in-room"), null, { timeout: ms }).then(() => true, () => false);
const cardNames = (p) => p.evaluate(() => [...document.querySelectorAll(".peer-card")].map((c) => c.dataset.name));
const waitCards = (p, n, ms = 30000) => p.waitForFunction((k) => document.querySelectorAll(".peer-card").length >= k, n, { timeout: ms }).then(() => true, () => false);
const promptShown = (p) => p.evaluate(() => !document.getElementById("join-reqs").hidden);
const waitPrompt = (p, re, ms = 20000) => p.waitForFunction((s) => !document.getElementById("join-reqs").hidden && new RegExp(s).test(document.getElementById("jr-line").textContent), re.source, { timeout: ms }).then(() => true, () => false);
// did the host's prompt show at any time while fn ran?
async function watchPrompt(host, fn) {
  await host.evaluate(() => { window.__prompted = 0; window.__mo?.disconnect(); window.__mo = new MutationObserver(() => { if (!document.getElementById("join-reqs").hidden) window.__prompted++; }); window.__mo.observe(document.getElementById("join-reqs"), { attributes: true }); });
  await fn();
  return host.evaluate(() => window.__prompted);
}
async function shot(p, name, width) {
  if (!SHOTS) return;
  fs.mkdirSync(SHOTS, { recursive: true });
  const was = p.viewportSize();
  if (width) await p.setViewportSize({ width, height: 844 });
  await p.waitForTimeout(400);
  await p.screenshot({ path: path.join(SHOTS, name) });
  if (width) await p.setViewportSize(was);
}

let code = 0;
try {
  await peerServer(SIG);
  browser = await chromium.launch({ headless: true, args: ["--no-sandbox", "--disable-gpu", "--allow-loopback-in-peer-connection"] });

  // ---- 1 ----
  const host = await tab("host");
  await host.click("#create-btn");
  await host.waitForFunction(() => /^[A-Z0-9]{3}-[A-Z0-9]{3}$/.test(document.getElementById("side-code").textContent), null, { timeout: 30000 }).catch(() => {});
  const shown = (await host.textContent("#side-code")).trim();
  const room = shown.replace("-", "");
  check("1 a six-character code, shown in two groups", /^[A-HJKMNP-TV-Z2-9]{3}-[A-HJKMNP-TV-Z2-9]{3}$/.test(shown) && (await host.textContent("#room-badge")) === shown, shown);
  await host.click("#share-btn");
  const link = (await host.textContent("#share-url")).trim();
  const shareCode = (await host.textContent("#share-code")).trim();
  const orLine = await host.textContent("#share-or");
  await host.click("#share-close");
  check("1 the invite link carries the key in its fragment", /#k=[A-Za-z0-9_-]{22}$/.test(link) && link.includes(`code=${room}`), link);
  check("1 the Invite sheet shows the grouped code and says typed codes are asked about", shareCode === shown && /you let the device in/.test(orLine), `${shareCode} / ${orLine}`);
  check("1 Ask before new devices join is on", await host.isChecked("#ask-join"));
  if (SHOTS) {
    await host.click("#room-menu > summary"); await host.waitForTimeout(400);
    await host.locator("#room-menu .menu-pop").screenshot({ path: path.join(SHOTS, "settings-desktop.png") });
    await host.click("#menu-close");
    await host.click("#share-btn"); await shot(host, "invite-desktop.png"); await host.click("#share-close");
  }

  // ---- 2: the link ----
  const byLink = await tab("linky", link);
  const prompts2 = await watchPrompt(host, async () => { await waitIn(byLink); await waitCards(host, 2); await host.waitForTimeout(800); });
  check("2 the invite link joins without a prompt", (await inRoom(byLink)) && prompts2 === 0 && (await cardNames(host)).length === 2, `prompts ${prompts2}`);
  check("2 no key left in its address bar", !byLink.url().includes("#k="), byLink.url());

  // ---- 3: typed codes wait ----
  const typed = await tab("typed");
  const denied = await tab("denied");
  await host.focus("#share-btn");   // the host is "doing something": the prompt must not take focus
  await typed.fill("#code-input", shown.toLowerCase()); await typed.click("#join-btn");
  const promptOk = await waitPrompt(host, /^typed wants to join \(computer, chat only\)$/);
  await denied.fill("#code-input", room); await denied.click("#join-btn");
  const moreOk = await host.waitForFunction(() => document.getElementById("jr-more").textContent === "1 more waiting", null, { timeout: 20000 }).then(() => true, () => false);
  check("3 the host sees the request", promptOk, await host.textContent("#jr-line"));
  check("3 a second one queues behind it", moreOk && (await host.textContent("#jr-line")).startsWith("typed"));
  check("3 the prompt did not take focus", await host.evaluate(() => document.activeElement?.id === "share-btn"));
  const heard = await host.waitForFunction(() => /wants to join .*Allow or Deny/.test(document.getElementById("jr-live").textContent), null, { timeout: 5000 }).then(() => true, () => false);
  check("3 a screen reader hears it", heard, await host.textContent("#jr-live"));
  const waitText = await typed.waitForFunction(() => /Waiting for the host to let you in/.test(document.getElementById("jw-t").textContent), null, { timeout: 15000 }).then(() => true, () => false);
  check("3 the typed tab waits", waitText && !(await inRoom(typed)) && !(await typed.isHidden("#jw-cancel")));
  const hostSas = ((await host.textContent("#jr-sas")) || "").match(/\d{3} \d{3}/)?.[0];
  const typedSas = await typed.waitForFunction(() => document.getElementById("join-status").textContent.match(/\d{3} \d{3}/)?.[0], null, { timeout: 5000 }).then((h) => h.jsonValue(), () => null);
  check("3 both screens show the same six-digit code", !!hostSas && hostSas === typedSas, `${hostSas} / ${typedSas}`);
  await host.waitForTimeout(1500);
  const leaks = await typed.evaluate(() => ({ text: document.body.innerText.includes("linky"), cards: [...document.querySelectorAll(".peer-card")].map((c) => c.dataset.name) }));
  check("3 a waiting tab gets no roster", !leaks.text && !leaks.cards.includes("linky"), JSON.stringify(leaks));
  check("3 the host's room doesn't list it", !(await cardNames(host)).includes("typed") && !(await cardNames(byLink)).includes("typed"), JSON.stringify(await cardNames(host)));
  await shot(host, "prompt-desktop.png");
  await shot(typed, "waiting-desktop.png");
  await shot(host, "prompt-390.png", 390);
  await shot(typed, "waiting-390.png", 390);

  // ---- 4: Allow with the keyboard, then Deny ----
  await host.focus("#jr-allow");
  await host.keyboard.press("Enter");
  check("4 Allow lets it in", await waitIn(typed) && await waitCards(host, 3));
  check("4 it sees the room", await typed.waitForFunction(() => [...document.querySelectorAll(".peer-card")].some((c) => c.dataset.name === "linky"), null, { timeout: 15000 }).then(() => true, () => false));
  const next = await waitPrompt(host, /^denied wants to join/);
  check("4 the next request comes up, with focus on its Allow", next && await host.evaluate(() => document.activeElement?.id === "jr-allow"));
  await host.click("#jr-deny");
  const deniedMsg = await denied.waitForFunction(() => /didn't let this device in/.test(document.getElementById("join-status").textContent) && !document.getElementById("join-screen").classList.contains("waiting"), null, { timeout: 15000 }).then(() => true, () => false);
  check("4 Deny ends with a message", deniedMsg && !(await inRoom(denied)), await denied.textContent("#join-status"));
  const knocks = await watchPrompt(host, () => host.waitForTimeout(7000));
  check("4 the denied tab does not knock again", knocks === 0 && !(await promptShown(host)) && !(await cardNames(host)).includes("denied"));
  await shot(denied, "denied-desktop.png");

  // ---- 5: reloads ----
  const prompts5 = await watchPrompt(host, async () => {
    await typed.reload();
    await waitIn(typed, 30000);
    await host.waitForTimeout(3000);
  });
  check("5 reloading the allowed tab: back in, no prompt", (await inRoom(typed)) && prompts5 === 0 && (await cardNames(host)).filter((n) => n === "typed").length === 1, `prompts ${prompts5}; ${JSON.stringify(await cardNames(host))}`);
  await host.reload();
  await waitIn(host, 30000);
  await host.evaluate(() => { window.__prompted = 0; window.__mo = new MutationObserver(() => { if (!document.getElementById("join-reqs").hidden) window.__prompted++; }); window.__mo.observe(document.getElementById("join-reqs"), { attributes: true }); });
  const back = await waitCards(host, 3, 45000);
  await host.waitForTimeout(1500);
  const hp = await host.evaluate(() => window.__prompted);
  check("5 the host reloads: same code, its devices come back without a prompt", back && hp === 0 && (await host.textContent("#side-code")).trim() === shown, `${hp} prompts; ${JSON.stringify(await cardNames(host))}`);
  await host.click("#share-btn");
  const link2 = (await host.textContent("#share-url")).trim();
  await host.click("#share-close");
  check("5 the invite link survives the host's reload", link2 === link, link2);

  // ---- 6: API clients ----
  if (fs.existsSync(path.join(ROOT, "cli/node_modules/node-datachannel"))) {
    const bridge = (room, port, name) => {
      const b = spawn(process.execPath, [path.join(ROOT, "cli/bin/pooled.js"), "serve", room, "--port", String(port), "--signal", `127.0.0.1:${SIG}`, "--name", name], { stdio: ["ignore", "pipe", "pipe"] });
      b.out = ""; b.stdout.on("data", (c) => { b.out += c; }); b.stderr.on("data", (c) => { b.out += c; });
      bridges.push(b);
      return b;
    };
    const health = async (port) => { try { return await (await fetch(`http://127.0.0.1:${port}/health`)).json(); } catch { return {}; } };
    const waitHealth = async (port, pred, ms = 30000) => { for (const end = Date.now() + ms; Date.now() < end; await new Promise((r) => setTimeout(r, 300))) if (pred(await health(port))) return true; return false; };
    const b1 = bridge(shown, PORT + 10, "cli typed");
    const asked = await waitPrompt(host, /^cli typed wants to join \(API client\)$/, 30000);
    const notYet = await health(PORT + 10);
    check("6 pooled serve with the code: the host is asked, the client waits", asked && notYet.connected === false && /waiting for the host/.test(b1.out), JSON.stringify(notYet));
    await host.click("#jr-allow");
    check("6 Allow lets it in", await waitHealth(PORT + 10, (h) => h.connected === true), b1.out.slice(-300));
    const b2Prompts = await watchPrompt(host, async () => { bridge(link, PORT + 11, "cli link"); await waitHealth(PORT + 11, (h) => h.connected === true); });
    check("6 pooled serve with the invite link: in at once", (await health(PORT + 11)).connected === true && b2Prompts === 0, `prompts ${b2Prompts}`);
  } else log("SKIP 6: no cli/node_modules (cd cli && npm ci)");

  // ---- 8: a room node device with the invite link, and its proved link to a browser tab ----
  if (fs.existsSync(path.join(ROOT, "packages/room-node/node_modules/node-datachannel"))) {
    const { joinRoom } = await import(path.join(ROOT, "packages/room-node/roomnode.js"));
    const noGpu = async () => ({ create: () => ({ requestAdapter: async () => null }), globals: {} });
    const key = link.split("#k=")[1];
    const rlog = [];
    const prompts8 = await watchPrompt(host, async () => {
      const node = await joinRoom(room, { name: "node-dev", key, pledgeGB: 1, signal: `127.0.0.1:${SIG}`, setup: { webgpu: noGpu }, stripes: 2, selfTest: false, log: (s) => rlog.push(s) });
      bridges.push({ kill: () => node.close().catch(() => {}) });
      let verified = null;
      node.on("admitted", (x) => { verified = x.verified; });
      for (let i = 0; i < 200 && node.admission !== "in"; i++) await new Promise((r) => setTimeout(r, 100));
      check("8 a room node device with the invite link: in, the host's proof checked, the mesh key in hand", node.admission === "in" && verified === true && !!node.mk, rlog.slice(-3).join(" | "));
      for (let i = 0; i < 100 && !(node.members || []).some((m) => m.name === "linky"); i++) await new Promise((r) => setTimeout(r, 100));
      const linky = (node.members || []).find((m) => m.name === "linky");
      const linked = linky ? await node.ensureLink(linky.id, 20000) : false;
      for (let i = 0; i < 100 && !(node.conns.get(linky?.id)?.stripes.length >= 1); i++) await new Promise((r) => setTimeout(r, 100));
      check("8 it links to a browser tab and both prove the mesh key (a stripe too)", linked && node.conns.get(linky.id)?.stripes.length >= 1 && !rlog.some((l) => /didn't prove|no proof/.test(l)), rlog.slice(-3).join(" | "));
    });
    check("8 no prompt for it", prompts8 === 0, `prompts ${prompts8}`);
  } else log("SKIP 8: no packages/room-node/node_modules (cd packages/room-node && npm ci)");

  // ---- 7: Ask off ----
  await host.evaluate(() => { const s = document.getElementById("ask-join"); s.checked = false; s.dispatchEvent(new Event("change")); });
  const open = await tab("open");
  const prompts6 = await watchPrompt(host, async () => { await open.fill("#code-input", room); await open.click("#join-btn"); await waitIn(open); await host.waitForFunction(() => [...document.querySelectorAll(".peer-card")].some((c) => c.dataset.name === "open"), null, { timeout: 30000 }).catch(() => {}); });
  check("7 Ask off: a typed code gets in without a prompt", (await inRoom(open)) && prompts6 === 0);

  const errs = pages.flatMap((p) => p.errs.map((e) => `${p.label}: ${e}`));
  check("no page errors", errs.length === 0, errs.slice(0, 3).join(" | "));
} catch (e) {
  log("ERROR", e?.stack || e); code = 2;
} finally {
  const failed = results.filter((r) => !r.ok);
  console.log(JSON.stringify({ passed: results.length - failed.length, failed: failed.map((r) => r.name) }));
  if (failed.length && !code) code = 1;
  for (const b of bridges) try { b.kill("SIGINT"); } catch {}
  await browser?.close().catch(() => {});
  try { sigProc?.kill("SIGKILL"); } catch {}
  srv.close();
  process.exit(code);
}
