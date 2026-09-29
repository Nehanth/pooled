// An iPhone (Safari) in a room, interrupted mid-answer over WebDriver, then watched until it is back
// in its slot (#207). Runs on the Mac the phone is USB-attached to ("Remote Automation" on in
// Settings > Apps > Safari > Advanced). The computer side is tests/e2e/room_resume.mjs --external.
// No dependencies (Node 18+: fetch).
//
//   node tests/e2e/resume_phone.mjs --code ABCD --url https://<preview>/room --actions background,reload
//        [--name iphone] [--gb 0.5] [--away 30] [--skip 1] [--wd-port 4447] [--maxmin 20] [--out r.json]
//
// For every action, the phone waits for the next answer the room streams (after --skip answers), lets
// it run a few seconds, then:
//   background  opens a new Safari tab and switches to it, so the room's tab is hidden (Safari
//               suspends a background tab's timers and, on iOS, its WebRTC) for --away seconds, then
//               switches back: the closest WebDriver gets to locking the screen
//   reload      reloads the room's tab (what iOS does after killing a tab for memory)
//   visibility  in-page only: the page is told it is hidden and its signaling socket closes (weakest)
// and polls the page until the answer ends and the phone serves its layers again. WebDriver cannot
// press the lock button, so a real screen lock is not covered here.
import fs from "fs";
import { spawn } from "child_process";

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const URL0 = arg("url", "https://pooled.run/room"), CODE = arg("code");
const WD_PORT = +arg("wd-port", 4447), WD = `http://127.0.0.1:${WD_PORT}`;
const NAME = arg("name", "iphone"), GB = arg("gb", "0.5"), AWAY = +arg("away", 30), SKIP = +arg("skip", 1), MAXMIN = +arg("maxmin", 20);
const ACTIONS = arg("actions", "background,reload").split(",");
const t0 = Date.now();
const log = (...a) => console.error(((Date.now() - t0) / 1000).toFixed(0) + "s [phone]", ...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const out = { name: NAME, url: URL0, code: CODE, actions: [], errors: [] };
let driver = null, sid = null, done = false;

async function wd(method, path, body, timeoutMs = 120000) {
  const r = await fetch(WD + path, { method, headers: { "content-type": "application/json" }, body: body ? JSON.stringify(body) : undefined, signal: AbortSignal.timeout(timeoutMs) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(`webdriver ${method} ${path}: ${r.status} ${JSON.stringify(j.value || j).slice(0, 300)}`);
  return j.value;
}
const S = (p) => `/session/${sid}${p}`;
const exec = (fn, ...args) => wd("POST", S("/execute/sync"), { script: `return (${fn}).apply(null, arguments)`, args });
async function click(css) { const el = await wd("POST", S("/element"), { using: "css selector", value: css }); await wd("POST", S(`/element/${Object.values(el)[0]}/click`), {}); }
async function finish(code, why) {
  if (done) return; done = true;
  out.end = why; out.totalS = Math.round((Date.now() - t0) / 1000);
  if (sid) { try { await wd("DELETE", `/session/${sid}`, null, 30000); } catch (e) { log("session delete failed:", String(e).slice(0, 120)); } }
  if (driver) driver.kill();   // only the safaridriver this script started
  const line = JSON.stringify(out);
  if (arg("out")) fs.writeFileSync(arg("out"), line);
  console.log(line);
  process.exit(code);
}
process.on("SIGTERM", () => finish(1, "killed"));
process.on("SIGINT", () => finish(1, "killed"));
setTimeout(() => finish(1, `time limit (${MAXMIN} min)`), MAXMIN * 60000);

// the page's state, in one round trip
const PAGE_STATE = () => {
  const $ = (id) => document.getElementById(id);
  const bots = document.querySelectorAll(".m.bot");
  const last = bots[bots.length - 1];
  const ind = $("awake-ind");
  return {
    joined: document.body.classList.contains("in-room"), online: !!$("ai-panel")?.classList.contains("online"),
    status: $("ai-status")?.textContent || "", joinStatus: $("join-status")?.textContent || "", over: $("room-over") && !$("room-over").hidden ? $("room-over-h")?.textContent : "",
    answers: bots.length, lastChars: last?.querySelector(".bubble")?.textContent.length || 0, lastEnded: !!last?.querySelector(".stats"),
    lastStats: last?.querySelector(".stats")?.textContent || "", awake: ind ? (ind.hidden ? "hidden" : ind.className || "shown") : "none",
    visible: document.visibilityState, log: [...document.querySelectorAll("#chat-log div")].slice(-4).map((d) => d.textContent.slice(0, 160)),
    links: (window.pooledDebug?.() || []).map((x) => `${x.name}:${x.chans}`).join(" "),
  };
};
const state = () => exec(PAGE_STATE);
async function until(pred, ms, what, every = 1500) {
  const tEnd = Date.now() + ms;
  let s = null, tLog = Date.now();
  while (Date.now() < tEnd) {
    try { s = await state(); if (pred(s)) return s; } catch (e) { s = { err: String(e).slice(0, 160) }; }
    if (Date.now() - tLog > 30000) { tLog = Date.now(); log("still waiting for", what + ":", JSON.stringify(s).slice(0, 300)); }
    await sleep(every);
  }
  throw new Error(`timeout waiting for ${what}: ${JSON.stringify(s).slice(0, 400)}`);
}

try {
  if (!CODE) throw new Error("--code is required");
  driver = spawn("safaridriver", ["-p", String(WD_PORT)], { stdio: "ignore" });
  await sleep(2500);
  sid = (await wd("POST", "/session", { capabilities: { alwaysMatch: { browserName: "safari", platformName: "iOS" } } })).sessionId;
  log("session", sid);
  await wd("POST", S("/url"), { url: URL0 + (URL0.includes("?") ? "&" : "?") + "dev=1" });
  await until(() => true, 20000, "page");
  // the GPU probe fills in the pledge: set ours after it, or it overwrites it
  for (let i = 0; i < 120; i++) { if (await exec(() => { const g = document.getElementById("join-gb"); return !!g && g.value !== "" && g.value !== "1"; }).catch(() => false)) break; await sleep(500); }
  await exec((name, gb, code) => {
    const set = (id, v) => { const el = document.getElementById(id); el.value = v; el.dispatchEvent(new Event("input", { bubbles: true })); el.dispatchEvent(new Event("change", { bubbles: true })); };
    set("name-input", name); set("join-gb", gb); set("code-input", code);
  }, NAME, GB, CODE);
  await click("#join-btn").catch((e) => log("tap failed:", String(e).slice(0, 120)));   // a real tap: the wake lock wants one
  await sleep(1500);
  if (!(await exec(() => document.getElementById("join-status")?.textContent || document.body.classList.contains("in-room")))) {
    log("the tap did not join; clicking from the page");
    await exec(() => { document.getElementById("join-btn").click(); return true; });
  }
  let s = await until((x) => x.joined, 120000, "join");
  log("joined");
  s = await until((x) => x.online, 900000, "the room to come online", 3000);
  out.awakeOnline = s.awake;
  log("online:", s.status.slice(0, 100), "| awake:", s.awake);
  let seen = s.answers;
  let skip = SKIP;
  for (const action of ACTIONS) {
    const a = { action };
    out.actions.push(a);
    // the next answer to interrupt (the host asks one after the other)
    for (;;) {
      s = await until((x) => x.answers > seen, 900000, "an answer", 1000);
      seen = s.answers;
      if (skip-- > 0) { log("letting answer", seen, "run (baseline)"); await until((x) => x.lastEnded, 600000, "the baseline to end", 2000); continue; }
      break;
    }
    s = await until((x) => x.lastChars > 60 || x.lastEnded, 300000, "the answer to stream", 500);
    a.before = { chars: s.lastChars, status: s.status.slice(0, 100), links: s.links };
    if (s.lastEnded) { a.error = "the answer ended before the interrupt"; log(a.error); continue; }
    const ta = Date.now();
    log(action, "at", s.lastChars, "chars");
    if (action === "background") {
      const room = await wd("GET", S("/window"));
      const w = await wd("POST", S("/window/new"), { type: "tab" });
      await wd("POST", S("/window"), { handle: w.handle });
      await wd("POST", S("/url"), { url: "about:blank" }).catch(() => {});
      await sleep(AWAY * 1000);
      await wd("DELETE", S("/window")).catch(() => {});
      await wd("POST", S("/window"), { handle: room });
    } else if (action === "reload") {
      await wd("POST", S("/refresh"), {});
    } else if (action === "visibility") {
      await exec(() => {
        Object.defineProperty(document, "visibilityState", { configurable: true, get: () => "hidden" });
        document.dispatchEvent(new Event("visibilitychange"));
        return true;
      });
      await sleep(AWAY * 1000);
      await exec(() => { delete document.visibilityState; document.dispatchEvent(new Event("visibilitychange")); return true; });
    }
    log(action, "done; watching the room");
    const back = await state().catch((e) => ({ err: String(e) }));
    a.right_after = { status: back.status?.slice(0, 120), over: back.over, links: back.links, awake: back.awake, visible: back.visible };
    // after a reload the page rejoins on its own; it may have no chat history (answers restart at 0)
    // (a reloaded tab has no chat history and never sees this answer's end: there, back online is the end)
    s = await until((x) => x.online && !x.over && (action === "reload" ? x.answers === 0 || x.lastEnded : x.lastEnded), 600000, "the phone back online and the answer ended", 2000);
    a.secs = Math.round((Date.now() - ta) / 1000);
    a.after = { status: s.status.slice(0, 140), stats: s.lastStats.slice(0, 160), links: s.links, awake: s.awake, log: s.log };
    log(action, "ended after", a.secs, "s:", s.lastStats.slice(0, 120));
    seen = s.answers;
  }
  await finish(0, "done");
} catch (e) {
  out.errors.push(String(e).slice(0, 500));
  log("FAILED", String(e).slice(0, 300));
  await finish(1, "error");
}
