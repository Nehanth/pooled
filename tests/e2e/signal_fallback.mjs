// Signaling fallback (room/signal.js) with the signaling server unreachable, in headless Chromium on one
// machine: local PeerServers stand in for the fallbacks, the public cloud is made unreachable with a
// resolver rule, and a TCP port that accepts and never answers plays a black-holed server. No GPU, no
// model: only finding each other. Manual trigger (node tests/e2e/signal_fallback.mjs); needs
// `npm install` (playwright, peer).
//
//   1 cloud down: POOLED_SIGNAL_SERVERS = [cloud, black hole, local A]; the host lands on A, the invite
//     link names A, a joiner by link and a joiner by typed code both get in.
//   2 split view (B on a custom path): A down for the host only (resolver), up for the joiner: the host lands on B, the
//     joiner finds no room on A and moves on to B.
//   3 all down: the join screen says the signaling server can't be reached (and links the self-host doc).
//   4 drop mid-room: A stops; every tab shows the signaling notice and keeps its links; A comes back;
//     the notice goes and a new device can join again.
import { chromium } from "playwright";
import http from "http";
import net from "net";
import fs from "fs";
import path from "path";
import { spawn } from "child_process";

const arg = (k, d) => { const i = process.argv.indexOf("--" + k); return i > 0 ? process.argv[i + 1] : d; };
const PORT = +arg("port", 8331), SA = PORT + 1, SB = PORT + 2, HOLE = PORT + 3, SLOW = PORT + 4;
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png" };
const srv = http.createServer((q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { r.statusCode = 404; r.end(); return; }
  r.setHeader("content-type", MIME[path.extname(p)] || "application/octet-stream"); fs.createReadStream(p).pipe(r);
}).listen(PORT, "127.0.0.1");
// a black hole: accepts the connection, never says anything
const holes = new Set();
const hole = net.createServer((s) => { holes.add(s); s.on("error", () => {}); }).listen(HOLE, "127.0.0.1");
// a slow link to A: every chunk each way waits SLOW_MS (a congested phone network, not an outage)
const SLOW_MS = 1500;
const slow = net.createServer((c) => {
  const u = net.connect(SA, "127.0.0.1");
  const later = (to) => (d) => setTimeout(() => { try { to.write(d); } catch {} }, SLOW_MS);
  c.on("data", later(u)); u.on("data", later(c));
  const end = () => setTimeout(() => { c.destroy(); u.destroy(); }, SLOW_MS);
  c.on("close", end); u.on("close", end); c.on("error", () => {}); u.on("error", () => {});
  holes.add(c); holes.add(u);
}).listen(SLOW, "127.0.0.1");
const servers = {};
async function peerServer(port, at = "/") {
  const p = spawn(path.join(ROOT, "node_modules/.bin/peerjs"), ["--port", String(port), "--path", at], { stdio: "ignore" });
  for (let i = 0; i < 50; i++) {   // up when it answers /peerjs/id
    await new Promise((r) => setTimeout(r, 200));
    const ok = await new Promise((r) => http.get(`http://127.0.0.1:${port}${at.replace(/\/$/, "")}/peerjs/id`, (res) => { res.resume(); r(res.statusCode === 200); }).on("error", () => r(false)));
    if (ok) return (servers[port] = p);
  }
  throw new Error("peer server on " + port + " did not start");
}
const stopServer = (port) => { try { servers[port]?.kill("SIGKILL"); } catch {} delete servers[port]; };

const t0 = Date.now(); const T = () => ((Date.now() - t0) / 1000).toFixed(1) + "s";
const log = (...a) => console.error(T(), ...a);
const results = [];
const check = (name, ok, extra = "") => { results.push({ name, ok }); log(ok ? "PASS" : "FAIL", name, extra); };

const UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";
const browsers = [];
// one browser per resolver view (--host-resolver-rules is per browser); no GPU: --disable-gpu
async function browserWith(rules) {
  const b = await chromium.launch({ headless: true, args: ["--no-sandbox", "--disable-gpu", "--allow-loopback-in-peer-connection", `--host-resolver-rules=${rules}`] });
  browsers.push(b);
  return b;
}
async function tab(browser, list, url = `http://127.0.0.1:${PORT}/p2p.html?ask=0`) {
  const ctx = await browser.newContext({ userAgent: UA });
  if (list) await ctx.addInitScript((l) => { window.POOLED_SIGNAL_SERVERS = l; }, list);
  const p = await ctx.newPage();
  p.errs = [];
  p.on("pageerror", (e) => p.errs.push(String(e).slice(0, 200)));
  await p.goto(url);
  await p.waitForFunction(() => typeof Peer === "function" && document.getElementById("create-btn"), null, { timeout: 30000 });
  await p.waitForTimeout(500);
  if (!(await p.inputValue("#join-gb"))) await p.fill("#join-gb", "1");
  return p;
}
const inRoom = (p) => p.evaluate(() => document.body.classList.contains("in-room"));
const cards = (p) => p.evaluate(() => document.querySelectorAll(".peer-card").length);
const status = (p) => p.evaluate(() => document.getElementById("join-status").textContent);
async function create(p) {
  await p.click("#create-btn");
  await p.waitForFunction(() => /[A-Z0-9]{3}-?[A-Z0-9]{3}|[A-Z0-9]{4}/.test(document.getElementById("side-code")?.textContent || ""), null, { timeout: 60000 });
  return (await p.textContent("#side-code")).trim().replace("-", "").match(/[A-Z0-9]{6}|[A-Z0-9]{4}/)[0];
}
async function join(p, code) { await p.fill("#code-input", code); await p.click("#join-btn"); }
async function inviteLink(p) {
  await p.click("#share-btn"); const u = (await p.textContent("#share-url")).trim(); await p.click("#share-close"); return u;
}
const waitCards = (p, n, ms = 60000) => p.waitForFunction((k) => document.querySelectorAll(".peer-card").length >= k, n, { timeout: ms }).then(() => true, () => false);

let code = 0;
try {
  await peerServer(SA); await peerServer(SB, "/pooled");   // B on a custom path
  const A = `ws://127.0.0.1:${SA}`, HOLEURL = `ws://127.0.0.1:${HOLE}`;

  // ---- 1: the cloud is down, fallbacks in order ----
  const noCloud = await browserWith("MAP 0.peerjs.com ~NOTFOUND");
  const list1 = ["cloud", HOLEURL, A];
  const host = await tab(noCloud, list1);
  let t = Date.now();
  const c1 = await create(host);
  const hostSecs = (Date.now() - t) / 1000;
  const link = await inviteLink(host);
  log("room", c1, "host in after", hostSecs.toFixed(1), "s; link", link);
  check("1 host falls back past the cloud and a black hole", await inRoom(host), `${hostSecs.toFixed(1)} s`);
  check("1 invite link names the fallback server", link.includes("signal=" + encodeURIComponent(`ws://127.0.0.1:${SA}`)));
  const byLink = await tab(noCloud, list1, link);   // the link joins by itself (?code=), straight to A
  t = Date.now();
  check("1 joiner by link gets in", await waitCards(byLink, 2, 30000), `${((Date.now() - t) / 1000).toFixed(1)} s`);
  check("1 the link keeps the joiner out of dev mode", link.includes("dev=0") && !(await byLink.evaluate(() => document.documentElement.classList.contains("dev"))));
  const byCode = await tab(noCloud, list1);
  t = Date.now(); await join(byCode, c1);
  check("1 joiner by typed code walks the list and gets in", await waitCards(byCode, 3, 60000), `${((Date.now() - t) / 1000).toFixed(1)} s`);
  check("1 host sees all three", await waitCards(host, 3, 10000));

  // ---- 0: the old ?signal=host:port form, and a slow (not dead) first server is kept ----
  const legacy = await tab(noCloud, null, `http://127.0.0.1:${PORT}/p2p.html?ask=0&signal=127.0.0.1:${SA}`);
  await create(legacy);
  check("0 ?signal=host:port still works", await inRoom(legacy));
  const slowTab = await tab(noCloud, [`ws://127.0.0.1:${SLOW}`, `ws://127.0.0.1:${SB}/pooled`]);
  t = Date.now(); await create(slowTab);
  const ls = await inviteLink(slowTab);
  check(`0 a slow server (${SLOW_MS} ms each way) is kept, not skipped`, !ls.includes("signal="), `${((Date.now() - t) / 1000).toFixed(1)} s; ${ls}`);

  // ---- 2: A is down for the host only; the joiner sees A but the room is on B ----
  const hView = await browserWith("MAP sig-a.test ~NOTFOUND, MAP sig-b.test 127.0.0.1");
  const jView = await browserWith("MAP sig-a.test 127.0.0.1, MAP sig-b.test 127.0.0.1");
  const list2 = [`ws://sig-a.test:${SA}`, `ws://sig-b.test:${SB}/pooled`];
  const h2 = await tab(hView, list2), j2 = await tab(jView, list2);
  const c2 = await create(h2);
  const l2 = await inviteLink(h2);
  check("2 host lands on B (custom path)", l2.includes(encodeURIComponent(`sig-b.test:${SB}/pooled`)), l2);
  t = Date.now(); await join(j2, c2);
  check("2 joiner finds no room on A, moves on to B, gets in", await waitCards(j2, 2, 60000), `${((Date.now() - t) / 1000).toFixed(1)} s`);

  // ---- 3: nothing answers ----
  const d1 = await tab(noCloud, null);   // the default list: the cloud only
  t = Date.now(); await d1.click("#create-btn");
  const ok3 = await d1.waitForFunction(() => document.getElementById("join-status").classList.contains("signal-down"), null, { timeout: 30000 }).then(() => true, () => false);
  const s3 = await status(d1);
  check("3 cloud unreachable: join screen says so", ok3 && /signaling server/.test(s3) && !(await inRoom(d1)), `${((Date.now() - t) / 1000).toFixed(1)} s: ${s3}`);
  check("3 buttons usable again", !(await d1.isDisabled("#create-btn")) && !(await d1.isDisabled("#join-btn")));
  check("3 links the self-host doc", await d1.evaluate(() => !!document.querySelector("#join-status a[href*='self-host-signaling']")));
  const d2 = await tab(noCloud, null, `http://127.0.0.1:${PORT}/p2p.html?ask=0&signal=${encodeURIComponent(HOLEURL)}`);
  t = Date.now(); await join(d2, "ZZZZ");
  const ok3b = await d2.waitForFunction(() => document.getElementById("join-status").classList.contains("signal-down"), null, { timeout: 40000 }).then(() => true, () => false);
  check("3 ?signal= black hole (joiner): times out (20 s, the last server) and says so, no fallback to the cloud", ok3b && (await status(d2)).includes(`127.0.0.1:${HOLE}`), `${((Date.now() - t) / 1000).toFixed(1)} s`);

  // ---- 4: A drops mid-room (room 1 lives on A) and comes back ----
  stopServer(SA);
  const tabs4 = [host, byLink, byCode];
  const noted = await Promise.all(tabs4.map((p) => p.waitForFunction(() => !document.getElementById("signal-note").hidden, null, { timeout: 20000 }).then(() => true, () => false)));
  check("4 every tab shows the signaling notice", noted.every(Boolean), JSON.stringify(noted));
  log("notice:", await host.textContent("#signal-note"));
  await host.waitForTimeout(4000);
  check("4 links stay up while signaling is down", (await Promise.all(tabs4.map(cards))).every((n) => n >= 3) && (await Promise.all(tabs4.map(inRoom))).every(Boolean));
  await peerServer(SA);
  t = Date.now();
  const back = await Promise.all(tabs4.map((p) => p.waitForFunction(() => document.getElementById("signal-note").hidden, null, { timeout: 60000 }).then(() => true, () => false)));
  check("4 notice goes when the server is back", back.every(Boolean), `${((Date.now() - t) / 1000).toFixed(1)} s`);
  const late = await tab(noCloud, list1, link);
  check("4 a new device can join again", await waitCards(late, 4, 30000));

  const errs = [host, byLink, byCode, legacy, slowTab, h2, j2, d1, d2, late].flatMap((p) => p.errs);
  check("no page errors", errs.length === 0, errs.slice(0, 3).join(" | "));
} catch (e) {
  log("ERROR", e?.stack || e); code = 2;
} finally {
  const failed = results.filter((r) => !r.ok);
  console.log(JSON.stringify({ passed: results.length - failed.length, failed: failed.map((r) => r.name) }));
  if (failed.length && !code) code = 1;
  for (const b of browsers) await b.close().catch(() => {});
  for (const p of Object.keys(servers)) stopServer(+p);
  for (const s of holes) s.destroy();
  hole.close(); slow.close(); srv.close();
  process.exit(code);
}
