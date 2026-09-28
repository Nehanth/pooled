// An iPhone (Safari) as a room member, driven over WebDriver: open the room page, set the device's
// name and memory pledge, join with the room code, then report the page's status (its layers, the
// room's peer cards, errors) until the host leaves. Manual trigger only: it needs a real iPhone with
// "Remote Automation" on (Settings > Apps > Safari > Advanced), USB-attached to the Mac this runs on
// (safaridriver talks to the phone through that Mac). No dependencies (Node 18+: fetch).
//
//   node tests/e2e/xroom_phone.mjs --code ABCD|--codefile f [--url https://pooled.run/room]
//        [--query "dev=1"] [--name iphone] [--gb 0.5] [--wd-port 4444] [--no-driver] [--maxmin 60]
//        [--out result.json] [--shot last.png] [--insecure]
//   node tests/e2e/xroom_phone.mjs --probe [--url ...]   # WebGPU adapter of the phone on that page, then exit
//
// --url: the page the phone opens. iOS Safari only gives WebGPU to a secure context, so this is an
//   https origin: https://pooled.run/room (production = main) or a Vercel preview of a branch
//   (https://pooled-git-<branch>-<team>.vercel.app/room), or an https server of your own with
//   --insecure (the WebDriver capability acceptInsecureCerts). The page uses its default signaling
//   (the public PeerJS server), so every other device in the room must too (xroom.mjs --signal cloud).
// --gb: the phone's pledge. A phone's minimum is 0.5 GB; by-memory dealing gives it about
//   L * gb / (sum of pledges) layers, and at least one.
// --wd-port: safaridriver's port (started here unless --no-driver: then one must be listening).
//   Only one WebDriver session can drive the phone at a time.
//
// Output: progress lines on stderr (every 20 s: status, this device's layers, the peer cards), and one
// JSON line on stdout at the end (also written to --out): joinS, the layers the phone was dealt,
// loadS (join to "online"), the status samples, the page's errors, and why it ended.
// The WebDriver session is always deleted and a safaridriver started here is stopped, on any exit.
import fs from "fs";
import { spawn } from "child_process";

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const flag = (k) => argv.includes("--" + k);
const URL0 = arg("url", "https://pooled.run/room"), QUERY = arg("query", "dev=1");
const PAGE = URL0 + (QUERY ? (URL0.includes("?") ? "&" : "?") + QUERY : "");
const WD = `http://127.0.0.1:${+arg("wd-port", 4444)}`;
const MAXMIN = +arg("maxmin", 60), NAME = arg("name", "iphone"), GB = arg("gb", "0.5");
const t0 = Date.now();
const log = (...a) => console.error(((Date.now() - t0) / 1000).toFixed(0) + "s [phone]", ...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let driver = null, sid = null, finished = false;
const out = { role: "phone", page: PAGE, name: NAME, gb: +GB, samples: [], errors: [] };

async function wd(method, path, body, timeoutMs = 120000) {
  const r = await fetch(WD + path, { method, headers: { "content-type": "application/json" }, body: body ? JSON.stringify(body) : undefined, signal: AbortSignal.timeout(timeoutMs) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(`webdriver ${method} ${path}: ${r.status} ${JSON.stringify(j.value || j).slice(0, 300)}`);
  return j.value;
}
const S = (p) => `/session/${sid}${p}`;
const exec = (fn, ...args) => wd("POST", S("/execute/sync"), { script: `return (${fn}).apply(null, arguments)`, args });
const execAsync = (fn, ...args) => wd("POST", S("/execute/async"), { script: `const done = arguments[arguments.length - 1]; Promise.resolve((${fn}).apply(null, [].slice.call(arguments, 0, -1))).then(done, (e) => done("ERR " + e));`, args });
async function click(css) {   // a real tap (trusted: the page's wake lock wants one)
  const el = await wd("POST", S("/element"), { using: "css selector", value: css });
  await wd("POST", S(`/element/${Object.values(el)[0]}/click`), {});
}

async function finish(code, why) {
  if (finished) return; finished = true;
  out.end = why; out.totalS = Math.round((Date.now() - t0) / 1000);
  if (sid && arg("shot")) {
    try { fs.writeFileSync(arg("shot"), Buffer.from(await wd("GET", S("/screenshot"), null, 30000), "base64")); } catch (e) { log("no screenshot:", String(e).slice(0, 120)); }
  }
  if (sid) { try { await wd("DELETE", `/session/${sid}`, null, 30000); } catch (e) { log("session delete failed:", String(e).slice(0, 160)); } }
  if (driver) driver.kill();
  const line = JSON.stringify(out);
  if (arg("out")) fs.writeFileSync(arg("out"), line);
  console.log(line);
  process.exit(code);
}
process.on("SIGTERM", () => finish(1, "killed"));
process.on("SIGINT", () => finish(1, "killed"));

// the page's state: status line, this device's layers (the header's device mark), the peer cards,
// and errors the page threw (collected by a hook installed after load)
const SNAP = () => {
  const $ = (id) => document.getElementById(id);
  return {
    status: $("ai-status")?.textContent || "", joinStatus: $("join-status")?.textContent || "",
    inRoom: document.body.classList.contains("in-room"), code: ($("side-code")?.textContent || "").trim(),
    mine: $("compute-open")?.dataset.tip || "", sub: $("ldg-sub")?.textContent || "",
    peers: [...document.querySelectorAll(".peer-card")].map((c) => c.textContent.replace(/\s+/g, " ").trim().slice(0, 160)),
    hostLeft: /host left/.test($("ai-status")?.textContent || "") || !!($("room-over") && !$("room-over").hidden),
    errs: (window.__xErrs || []).splice(0), visible: document.visibilityState,
  };
};

try {
  if (!flag("no-driver")) {
    driver = spawn("safaridriver", ["-p", String(+arg("wd-port", 4444))], { stdio: "ignore" });
    driver.on("exit", (c) => { if (!finished) log("safaridriver exited", c); });
    for (let i = 0; i < 40; i++) { try { await wd("GET", "/status", null, 2000); break; } catch { await sleep(250); } }
  }
  const caps = { browserName: "safari", platformName: "iOS", ...(flag("insecure") ? { acceptInsecureCerts: true } : {}) };
  const s = await wd("POST", "/session", { capabilities: { alwaysMatch: caps } }, 90000);
  sid = s.sessionId; out.device = s.capabilities?.["safari:deviceName"] || s.capabilities?.platformName; out.safari = s.capabilities?.browserVersion;
  log("session", sid, out.device || "", out.safari || "");
  await wd("POST", S("/url"), { url: PAGE }, 90000);
  if (flag("probe")) {
    out.gpu = await execAsync(async () => {
      if (!navigator.gpu) return "no navigator.gpu (isSecureContext " + isSecureContext + ")";
      const a = await navigator.gpu.requestAdapter(); if (!a) return "no adapter";
      const L = a.limits;
      return { features: [...a.features].sort(), maxBuffer: L.maxBufferSize, maxStorageBinding: L.maxStorageBufferBindingSize, wgStorage: L.maxComputeWorkgroupStorageSize, invocations: L.maxComputeInvocationsPerWorkgroup, ua: navigator.userAgent, secure: isSecureContext, href: location.href };
    });
    out.title = await exec(() => document.title);
    await finish(typeof out.gpu === "object" ? 0 : 1, "probe");
  }
  await exec(() => { window.__xErrs = []; addEventListener("error", (e) => window.__xErrs.push(String(e.message).slice(0, 200))); addEventListener("unhandledrejection", (e) => window.__xErrs.push("rejection: " + String(e.reason).slice(0, 200))); });
  // the GPU probe fills the pledge box when it finishes (a phone gets 0.5): wait for it, then set ours
  for (let i = 0; i < 120; i++) { if (await exec(() => { const g = document.getElementById("join-gb"); return !!g && g.value !== "" && g.value !== "1"; })) break; await sleep(500); }
  await sleep(1000);
  await exec((name, gb) => {
    for (const [id, v] of [["name-input", name], ["join-gb", gb]]) { const el = document.getElementById(id); el.value = v; el.dispatchEvent(new Event("input", { bubbles: true })); el.dispatchEvent(new Event("change", { bubbles: true })); }
  }, NAME, GB);
  let code = arg("code");
  if (!code) log("waiting for the room code in", arg("codefile"));
  for (const tEnd = Date.now() + 30 * 60e3; !code && Date.now() < tEnd; await sleep(1000)) {
    try { code = (fs.readFileSync(arg("codefile"), "utf8").match(/[A-Z0-9]{4}/) || [])[0]; } catch {}
    if (!code && !arg("codefile")) throw new Error("--code or --codefile");
  }
  if (!code) throw new Error("no room code in " + arg("codefile"));
  out.code = code;
  await exec((c) => { const el = document.getElementById("code-input"); el.value = c; el.dispatchEvent(new Event("input", { bubbles: true })); }, code);
  const tJoin = Date.now();
  // a WebDriver tap on iOS can land as a long press (it selects the button's text, no click event):
  // then press it from the page (an untrusted click still joins; only the wake lock may be refused)
  await click("#join-btn").catch((e) => log("tap failed:", String(e).slice(0, 120)));
  await sleep(3000);
  if (!(await exec(() => document.getElementById("join-status")?.textContent || document.body.classList.contains("in-room")))) {
    out.jsClick = true; log("the tap did not join; clicking from the page");
    await exec(() => { getSelection()?.removeAllRanges(); document.getElementById("join-btn").click(); });
  }
  for (;;) {
    const s = await exec(SNAP);
    if (s.inRoom && s.peers.length >= 2) { out.joinS = +((Date.now() - tJoin) / 1000).toFixed(1); log("joined", JSON.stringify(s).slice(0, 400)); break; }
    if (/fail|no room|error/i.test(s.joinStatus) && !/Trying a relay/.test(s.joinStatus)) throw new Error("join: " + s.joinStatus);
    if (Date.now() - tJoin > 120000) throw new Error("join timed out: " + s.joinStatus);
    await sleep(1000);
  }
  // serve until the host leaves (or the room empties); sample the page every 2 s
  const end = Date.now() + MAXMIN * 60e3; let lastLog = 0, seen = Date.now(), tLoad = null, lastStatus = "";
  while (Date.now() < end) {
    await sleep(2000);
    let s;
    try { s = await exec(SNAP); }
    catch (e) { out.errors.push("snapshot: " + String(e).slice(0, 200)); if (out.errors.length > 5) throw new Error("the page stopped answering (tab reloaded or crashed?)"); continue; }
    out.errors.push(...s.errs);
    if (!tLoad && /loading/.test(s.mine)) { tLoad = Date.now(); out.layers = s.mine; log("dealt:", s.mine); }
    if (tLoad && !out.loadS && /holds layers/.test(s.mine)) { out.loadS = +((Date.now() - tLoad) / 1000).toFixed(1); out.layers = s.mine; log("online:", s.mine, "after", out.loadS, "s"); }
    if (s.status !== lastStatus) { lastStatus = s.status; out.samples.push({ t: Math.round((Date.now() - t0) / 1000), status: s.status.slice(0, 200), mine: s.mine }); if (out.samples.length > 200) out.samples.splice(0, 50); }
    if (Date.now() - lastLog > 20000) { lastLog = Date.now(); log(JSON.stringify({ ...s, errs: undefined }).slice(0, 500)); }
    if (s.hostLeft) { out.last = s; await finish(0, "host left"); }
    if (s.peers.length >= 2) seen = Date.now(); else if (Date.now() - seen > 6000) { out.last = s; await finish(0, "room empty"); }
  }
  await finish(0, "maxmin");
} catch (e) {
  out.error = String(e).slice(0, 400); log("FAILED", out.error);
  try { if (sid) out.last = await exec(SNAP); } catch {}
  await finish(1, "error");
}
