// An iPhone (Safari) as a room member, driven over WebDriver: open the room page, set the device's
// name and memory pledge, join with the room code, then report the page's status (its layers, the
// room's peer cards, errors) until the host leaves. Manual trigger only: it needs a real iPhone with
// "Remote Automation" on (Settings > Apps > Safari > Advanced), USB-attached to the Mac this runs on
// (safaridriver talks to the phone through that Mac). No dependencies (Node 18+: fetch).
//
//   node tests/e2e/xroom_phone.mjs --code ABCD|--codefile f [--url https://pooled.run/room]
//        [--query "dev=1"] [--name iphone] [--gb 0.5] [--wd-port 4444] [--no-driver] [--maxmin 60]
//        [--out result.json] [--shot last.png] [--insecure] [--trace-out phone-trace.json]
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
// --trace-out: trace the phone's side of every lap from the moment it joins, without changing the page
//   (production pages cannot be patched at serve time): a script injected over WebDriver before the join
//   timestamps every wire frame slice the page sends or receives (send0/send1/rx0/rx1, parsed from the
//   room/transport.js header), puts a GPU timestamp pair around every command buffer (timestamp-query,
//   which iOS Safari has), times every mapAsync and writeBuffer, samples the event loop's lateness and,
//   every 10 s, the selected ICE pair's STUN round trip of every link. The marks use the same shape as
//   tests/e2e/room_trace.mjs ([event, kind, pos, ms since epoch on the phone's clock, ...]), so the
//   chain report (tests/e2e/xroom_chain_report.mjs) reads a phone trace like a computer guest's. The
//   timestamps are resolved every few seconds from the poll loop (a ring of 4096 queries per device).
//   Tracing is on for the whole session: compare with an untraced run for its cost.
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
  if (sid && TRACE_OUT) { try { await pullTrace(true); } catch (e) { log("last trace pull failed:", String(e).slice(0, 160)); } writeTrace(); }
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

// ---- phone-side trace (--trace-out), injected before the join ----
const TRACE_OUT = arg("trace-out");
const PT_INIT = () => {
  if (window.__pt) return "already";
  const T = () => performance.timeOrigin + performance.now();
  const P = window.__pt = { hp: [], subs: [], maps: [], writes: [], lag: [], links: [], devs: [], pcs: [], nsub: 0, notes: [] };
  const KINDS = ["ai-hidden", "ai-hidden-b", "ai-hiddenret", "ai-hiddenret-b"];
  const hdr = (b) => {
    if (!(b instanceof ArrayBuffer) || b.byteLength < 32) return null;
    const dv = new DataView(b); if (dv.getUint16(0) !== 0x5357) return null;
    return { kind: KINDS[dv.getUint8(2)], id: dv.getUint32(4), pos: dv.getUint32(8), n: dv.getUint16(12), k: dv.getUint16(14), ns: dv.getUint16(16), bytes: dv.getUint32(20) };
  };
  const DC = RTCDataChannel.prototype, oSend = DC.send;
  DC.send = function (b) {
    const h = hdr(b); if (h && h.k === 0) P.hp.push(["send0", h.kind, h.pos, T(), h.bytes, h.ns]);
    const r = oSend.call(this, b);
    if (h && h.k === h.ns - 1) P.hp.push(["send1", h.kind, h.pos, T()]);
    return r;
  };
  const rxc = new Map(), wrap = (fn) => function (ev) {
    const h = hdr(ev.data);
    if (h) { const t = T(), c = (rxc.get(h.id) || 0) + 1; if (c === 1) P.hp.push(["rx0", h.kind, h.pos, t, h.ns]); if (c >= h.ns) { rxc.delete(h.id); P.hp.push(["rx1", h.kind, h.pos, t]); } else rxc.set(h.id, c); }
    return fn.call(this, ev);
  };
  const dm = Object.getOwnPropertyDescriptor(DC, "onmessage");
  if (dm && dm.set) Object.defineProperty(DC, "onmessage", { configurable: true, enumerable: dm.enumerable, get() { return dm.get.call(this); }, set(fn) { dm.set.call(this, typeof fn === "function" ? wrap(fn) : fn); } });
  else P.notes.push("no onmessage accessor: rx marks missing");
  const oCdc = RTCPeerConnection.prototype.createDataChannel;
  RTCPeerConnection.prototype.createDataChannel = function (...a) { if (!P.pcs.includes(this)) P.pcs.push(this); return oCdc.apply(this, a); };
  // event-loop lateness: a 50 ms timer, how late it fires
  let due = performance.now() + 50;
  setInterval(() => { const n = performance.now(), late = n - due; if (late > 4) P.lag.push([T(), Math.round(late * 10) / 10]); due = n + 50; }, 50);
  if (!self.GPUAdapter) { P.notes.push("no WebGPU"); return "no gpu"; }
  const oReq = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function (desc = {}) {
    const ts = this.features.has("timestamp-query");
    const d = await oReq.call(this, ts ? { ...desc, requiredFeatures: [...new Set([...(desc.requiredFeatures || []), "timestamp-query"])] } : desc);
    const did = P.devs.length, rec0 = { did, ts, nq: 0 }; P.devs.push(rec0);
    d.lost?.then((i) => { rec0.dead = "lost: " + (i?.reason || "") + " " + (i?.message || "").slice(0, 80); for (const s of P.subs) if (s.d === did && s.q >= 0) { s.q = -3; s.gpu = -1; } });
    const q = d.queue, oSub = q.submit.bind(q), oWr = q.writeBuffer.bind(q);
    q.writeBuffer = (buf, off, data, ...rest) => { P.writes.push([T(), did, rest.length > 1 ? rest[1] * (data.BYTES_PER_ELEMENT || 1) : data.byteLength]); return oWr(buf, off, data, ...rest); };
    const cbRec = new WeakMap();
    if (ts) {
      const MAXQ = 4096, qs = d.createQuerySet({ type: "timestamp", count: MAXQ });
      const res = d.createBuffer({ size: MAXQ * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      const rd = d.createBuffer({ size: MAXQ * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const oEnc = d.createCommandEncoder.bind(d);
      d.createCommandEncoder = (dd) => {
        const e = oEnc(dd);
        // the timestamps go on the encoder's own compute passes (the first pass's beginning, every
        // pass's end: the last one written wins): on iOS Safari an empty timestamped pass reads 0
        const rec = { q: rec0.nq % MAXQ, d: did, tEnc: T(), t: 0, gpu: -1, passes: 0 }; rec0.nq += 2;
        const obp = e.beginComputePass.bind(e);
        e.beginComputePass = (pd = {}) => {
          if (pd.timestampWrites) return obp(pd);
          const tw = { querySet: qs, endOfPassWriteIndex: rec.q + 1 };
          if (!rec.passes++) tw.beginningOfPassWriteIndex = rec.q;
          return obp({ ...pd, timestampWrites: tw });
        };
        const fin = e.finish.bind(e);
        e.finish = (x) => { const cb = fin(x); if (rec.passes) cbRec.set(cb, rec); return cb; };
        return e;
      };
      let busy = false;
      rec0.resolve = async () => {
        if (busy) return; busy = true;
        try {
          const tRes = T(), pend = P.subs.filter((s) => s.d === did && s.q >= 0 && s.t && s.t <= tRes);
          if (!pend.length) return;
          const e = oEnc(); e.resolveQuerySet(qs, 0, MAXQ, res, 0); e.copyBufferToBuffer(res, 0, rd, 0, MAXQ * 8); oSub([e.finish()]);
          await oMap.call(rd, GPUMapMode.READ); const t = new BigUint64Array(rd.getMappedRange().slice(0)); rd.unmap();
          for (const s of pend) { if (t[s.q + 1] > t[s.q]) { s.gpu = Number(t[s.q + 1] - t[s.q]) / 1e6; s.g0 = Number(t[s.q]) / 1e6; } else s.gpu = 0; s.q = -2; }
        } finally { busy = false; }
      };
    }
    q.submit = (cbs) => { const t = T(); for (const cb of cbs) { const r = cbRec.get(cb) || { q: -1, d: did, tEnc: t, gpu: -1 }; r.t = t; P.subs.push(r); } P.nsub++; return oSub(cbs); };
    return d;
  };
  const oMap = GPUBuffer.prototype.mapAsync;
  GPUBuffer.prototype.mapAsync = function (...a) {
    const r = { t0: T(), t1: 0, bytes: a[2] ?? this.size }; P.maps.push(r);
    return oMap.apply(this, a).then((v) => { r.t1 = T(); return v; });
  };
  P.take = async () => {
    // a device can be destroyed under us (the room's throwaway self-test device, a reload of the
    // shard): its submits that were never resolved are dropped, and it is not asked again
    for (const dv of P.devs) if (dv.resolve && !dv.dead) {
      try { await dv.resolve(); } catch (e) { dv.dead = String(e).slice(0, 120); P.notes.push("device " + dv.did + " not resolved: " + dv.dead); for (const s of P.subs) if (s.d === dv.did && s.q >= 0) { s.q = -3; s.gpu = -1; } }
    }
    const done = (s) => s.q < 0;
    const subs = P.subs.filter(done); P.subs = P.subs.filter((s) => !done(s));
    const maps = P.maps.filter((m) => m.t1); P.maps = P.maps.filter((m) => !m.t1);
    return { hp: P.hp.splice(0), subs: subs.map((s) => ({ t: s.t, tEnc: s.tEnc, gpu: s.gpu, g0: s.g0, d: s.d })), maps, writes: P.writes.splice(0), lag: P.lag.splice(0), notes: P.notes.splice(0), devs: P.devs.map((x) => ({ did: x.did, ts: x.ts, nq: x.nq, dead: x.dead || null })) };
  };
  P.linkStats = async () => {
    const out = [];
    for (const pc of P.pcs) {
      if (pc.connectionState === "closed") continue;
      const st = await pc.getStats(); let pair = null; const cand = {};
      st.forEach((r) => { if (r.type === "candidate-pair" && r.nominated && r.state === "succeeded") pair = r; if (/candidate$/.test(r.type)) cand[r.id] = r; });
      if (!pair) continue;
      const l = cand[pair.localCandidateId], r = cand[pair.remoteCandidateId];
      out.push({ local: l && `${l.address || l.ip} ${l.candidateType} ${l.protocol}`, remote: r && `${r.address || r.ip} ${r.candidateType}`, rtt: pair.currentRoundTripTime != null ? Math.round(pair.currentRoundTripTime * 10000) / 10 : null,
        avg: pair.responsesReceived ? Math.round(pair.totalRoundTripTime / pair.responsesReceived * 10000) / 10 : null, n: pair.responsesReceived });
    }
    return out;
  };
  return "ok";
};
const TR = { hp: [], gp: { subs: [], maps: [] }, writes: [], lag: [], links: [], notes: [], clock: [] };
let lastLinks = 0;
async function pullTrace(final) {
  const c = await execAsync(async () => (window.__pt ? await window.__pt.take() : null));
  if (!c || typeof c !== "object") { if (c) out.errors.push("trace take: " + String(c).slice(0, 200)); return; }
  TR.hp.push(...c.hp); TR.gp.subs.push(...c.subs); TR.gp.maps.push(...c.maps); TR.writes.push(...c.writes); TR.lag.push(...c.lag); TR.notes.push(...c.notes); TR.devs = c.devs;
  if (final || Date.now() - lastLinks > 10000) {
    lastLinks = Date.now();
    const l = await execAsync(async () => (window.__pt ? await window.__pt.linkStats() : [])).catch(() => []);
    if (Array.isArray(l) && l.length) TR.links.push({ t: Date.now(), links: l });
    // the phone's clock against this machine's (lowest round trip of 3; WebDriver's round trip is long,
    // so this is only a coarse check: the chain report aligns clocks from the frames)
    let best = null;
    for (let i = 0; i < 3; i++) { const a = Date.now(); const t = await exec(() => performance.timeOrigin + performance.now()); const b = Date.now(); if (!best || b - a < best.rtt) best = { t: a, rtt: b - a, off: t - (a + b) / 2 }; }
    TR.clock.push(best);
  }
}
function writeTrace() {
  if (!TRACE_OUT) return;
  try { fs.writeFileSync(TRACE_OUT, JSON.stringify({ role: "phone", name: NAME, trace: TR })); log("trace:", TR.hp.length, "marks,", TR.gp.subs.length, "submits,", TR.gp.maps.length, "maps ->", TRACE_OUT); } catch (e) { log("trace not written:", String(e).slice(0, 200)); }
}

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
    wake: window.pooledWake?.() || null,   // GPU wake counters (room/gpuwake.js), on pages that have it
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
  if (TRACE_OUT) { out.trace = await exec(PT_INIT); log("trace hooks:", out.trace); }
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
    if (s.wake) out.wake = s.wake;
    if (TRACE_OUT) { try { await pullTrace(false); } catch (e) { out.errors.push("trace pull: " + String(e).slice(0, 160)); } }
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
