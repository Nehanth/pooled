// The Pooled room this OpenClaw process belongs to: one per process (a global, so a plugin registry
// reload does not open a second room), started by the plugin's service at gateway startup or by the
// first request, whichever comes first.
//
//   mode "host": this machine creates room <code>, holds the embedding, the head and its share of the
//                layers, and answers OpenClaw's requests itself (RoomNode.request, in process: no HTTP).
//                The room has the join gate on (as pooled.run and `pooled host`): the invite link
//                (…/r/<CODE>#k=<key>) gets a device in at once, a code alone waits until the owner
//                says /pooled allow. The key and the passes it gave out are kept (state.js), so a
//                gateway restart keeps the link and lets devices back in.
//   mode "join": this machine joins room <code> as a device that holds layers when the host deals
//                it some; OpenClaw's requests go to the room's host as API asks (the `pooled serve`
//                bridge, cli/lib/room.js) over WebRTC. It presents the invite key from the pasted
//                link, or the pass the host gave it before (onboarding or an earlier run), so the
//                host is asked at most once. If the room picks this machine to run the model, the
//                requests are answered here, as in host mode.
// Either way a request is the same ai-ask body with the same answer messages (transport() below).
import os from "node:os";
import { parseCode, formatCode, keyFragment, validKey, saveGate } from "../../../room/joingate.js";
import { roomCodeFrom, roomKeyFrom } from "../../../cli/lib/room.js";
import { roomFitNow, gpuLabel } from "../../../cli/lib/hostui.js";
import { fmtBytes } from "../../../cli/lib/cache.js";
import { MODELS, MODEL_CHOICES, modelInfo, modelsDir, isPulled, pluginAsk, lib, shortCtxNote, ownShortCtxNote } from "./models.js";
import { PooledError, pooledModules } from "./runtime.js";
import { savedGate, saveHostGate, joinState, saveJoinState } from "./state.js";
import { download, pullState, pullLine, downloadingMessage } from "./download.js";

export { PooledError, MODEL_CHOICES, modelInfo };
const KEY = Symbol.for("pooled.openclaw.room");
export const ROOM_ORIGIN = "https://pooled.run";
export const LOBBY_MS = 15000;   // an ask waits this long for the host's Allow, then the chat says so
export const fmtCode = formatCode;

// the room's invite link (the room page opens /r/<CODE>; #k= lets a device in without asking)
export const roomLink = (code, { key = null, signal = null } = {}) =>
  `${ROOM_ORIGIN}/r/${code}${signal ? `?signal=${encodeURIComponent(signal)}` : ""}${keyFragment(key)}`;

// a code or a link as typed or pasted -> { code, key } (code "" when there is none)
export function parseRoom(input) {
  const s = String(input ?? "").trim();
  const code = parseCode(roomCodeFrom(s) || s);
  return { code, key: roomKeyFrom(s) };
}

export const defaultName = (host = os.hostname()) => `${String(host).split(".")[0].replace(/[^\w-]/g, "").slice(0, 28) || "this machine"} (OpenClaw)`;
const off = (v) => /^(0|false|off|no)$/i.test(String(v ?? ""));

// the settings, from plugins.entries.pooled.config (onboarding writes them) and POOLED_* env
export function roomSettings(pluginConfig = {}, env = process.env) {
  const c = { ...(pluginConfig || {}) };
  const link = env.POOLED_LINK ? parseRoom(env.POOLED_LINK) : null;
  const raw = String(env.POOLED_CODE || link?.code || c.code || "").trim();
  return {
    mode: env.POOLED_MODE || c.mode || null,
    code: parseCode(raw) || raw.toUpperCase() || null,
    // the invite key: from a link given in env (tests, scripts); onboarding keeps it in state.js
    key: env.POOLED_KEY || link?.key || null,
    model: env.POOLED_MODEL || c.model || "qwen3-1.7b",
    pledgeGB: +(env.POOLED_PLEDGE_GB || c.pledgeGB || 0) || null,
    minDevices: +(env.POOLED_MIN_DEVICES || c.minDevices || 1),
    waitSeconds: +(env.POOLED_WAIT_S || c.waitSeconds || 120),
    ctx: +(env.POOLED_CTX || c.ctx || 0) || null,
    signal: env.POOLED_SIGNAL || c.signal || null,
    modelDir: env.POOLED_MODELS || c.modelDir || modelsDir({ env }),
    name: env.POOLED_NAME || c.name || defaultName(),
    page: env.POOLED_PAGE || c.page || null,   // a room page other than pooled.run (local tests)
    // host: hold a device that has the code but not the invite link until /pooled allow (default on)
    ask: !off(env.POOLED_ASK ?? c.ask),
    // host: download a model that is not in the cache yet, in the background (default on); off streams
    // this machine's layers from Hugging Face at each start
    pull: !off(env.POOLED_PULL ?? c.pull),
    // host: when the room comes online, ask once with the last system prompt and tools, so the first
    // real question starts from their checkpoint (prewarm.js; default on)
    prewarm: !off(env.POOLED_PREWARM ?? c.prewarm),
    // other devices' API clients may ask this gateway's room: another OpenClaw in "join" mode, `pooled
    // serve`. On by default (the room page's default, and what a joined OpenClaw needs); false keeps this
    // machine's GPU queue and checkpoint cache to this OpenClaw alone
    allowApiClients: !off(env.POOLED_ALLOW_API ?? c.allowApiClients),
  };
}

export function current() { return globalThis[KEY] || null; }

// -> the room handle: { s, node, code, link, status(), close() }
export function ensureRoom(settings, log = () => {}) {
  const cur = globalThis[KEY];
  if (cur && cur.key === keyOf(settings)) return cur.ready;
  if (cur) { cur.ready.then((r) => r.close()).catch(() => {}); globalThis[KEY] = null; }
  const h = { key: keyOf(settings), s: settings };
  h.ready = openRoom(settings, log).catch((err) => { if (globalThis[KEY] === h) globalThis[KEY] = null; throw err; });
  globalThis[KEY] = h;
  return h.ready;
}
export const keyOf = (s) => JSON.stringify([s.mode, s.code, s.key, s.model, s.pledgeGB, s.signal, s.ctx, s.allowApiClients, s.ask, s.modelDir, s.name]);

async function openRoom(s, log) {
  if (!s.mode) throw new PooledError("setup", "Pooled is not set up on this machine: run `openclaw onboard` (or `openclaw models auth login --provider pooled`) and pick Pooled");
  if (s.mode === "host" && !MODELS[s.model]) throw new PooledError("setup", `unknown model ${String(s.model).slice(0, 40)}: one of ${MODEL_CHOICES.join(", ")}`);
  const P = await pooledModules();
  const r = { s, P, code: s.code, key: null, bridge: null, bridgeTry: null, events: [], pull: null, pullP: null, refused: null };
  const note = (m) => { r.events.push({ t: Date.now(), m }); if (r.events.length > 50) r.events.shift(); log(m); };
  r.note = note;
  const common = { pledgeGB: s.pledgeGB || undefined, signal: s.signal, modelDir: s.modelDir, name: s.name, log: note,
    ctx: pluginAsk(s.model, s.ctx || 0), setup: { webgpu: P.dawn } };
  const fail = (code, msg) => (err) => { throw err instanceof PooledError ? err : new PooledError(code, `${msg}: ${err.message}`); };
  if (s.mode === "host") {
    if (s.code && !parseCode(s.code)) throw new PooledError("setup", `room code ${s.code} is not one the room page opens: 6 (or 4) of ABCDEFGHJKMNPQRSTVWXYZ23456789`);
    // visibility "asker": the other devices' screens never show OpenClaw's prompts, answers or tool
    // calls (they still compute them: every device holding layers sees the hidden states)
    r.node = await P.createRoom({ model: s.model, code: s.code || undefined, visibility: "asker", allowApi: s.allowApiClients,
      gate: true, ask: s.ask, gateState: s.code ? savedGate(s.code) : null, ...common }).catch(fail("start", `could not open Pooled room ${s.code || ""}`.trim()));
    if (s.pledgeGB) r.node.setPledge(s.pledgeGB);
    r.code = r.node.code;
    r.key = r.node.gate?.key || null;
    // keep the gate (invite key, passes) so a restart keeps the link and lets devices back in
    let saveErr = false;
    r.persist = () => {
      if (!r.node.gate) return;
      try { saveHostGate(r.code, saveGate(r.node.gate)); saveErr = false; }
      catch (e) { if (!saveErr) note(`could not save the room's invite key: ${e.message}`); saveErr = true; }
    };
    r.persist();
    r.node.on("members", r.persist);
    r.node.on("joinrequests", r.persist);
    r.node.on("joinrequest", (q) => note(`${q.line}: /pooled allow lets it in, /pooled deny turns it away`));
    if (s.pull && !isPulled(s.modelDir, s.model)) startPull(r);
  } else if (s.mode === "join") {
    const code = parseCode(s.code || "");
    if (!code) throw new PooledError("setup", `no room code${s.code ? ` (${s.code} is not one)` : ""}: run \`openclaw onboard\`, pick Pooled → Join a room and paste the room's link`);
    const js = joinState(code);
    r.code = code;
    r.key = validKey(s.key) ? s.key : validKey(js.key) ? js.key : null;
    r.node = await P.joinRoom(code, { ...common, key: r.key, pass: validKey(js.pass) ? js.pass : null }).catch((err) => {
      if (err instanceof PooledError) throw err;
      throw new PooledError("noroom", `could not join Pooled room ${fmtCode(code)}: ${err.message}. Is the room still open on the other device?`);
    });
    const remember = () => {
      if (r.node.admission !== "in") return;
      const hm = r.node.conns.get(r.P.PREFIX + code)?.meta;
      try { saveJoinState(code, { pass: validKey(r.node.pass) ? r.node.pass : null, host: r.node.hostName || null, model: hm?.model || null }); } catch {}
    };
    r.node.on("lobby", () => note(`waiting for the host of room ${fmtCode(code)} to let this device in (it sees "${r.node.name} wants to join")`));
    r.node.on("admitted", remember);
    r.node.on("members", remember);
    r.node.on("bye", (why) => { if (r.node.admission !== "in") r.refused = why || "the host turned this device away"; });
  } else throw new PooledError("setup", `unknown Pooled mode ${s.mode}`);
  r.link = roomLink(r.code, { key: r.key, signal: s.page ? null : s.signal });
  // the link without the invite key: for chat text and logs, which other people may read (a group
  // channel, a shared log). /pooled link (operator.admin) and status.json (0600) show the real one
  r.shareLink = roomLink(r.code, { signal: s.page ? null : s.signal });
  r.node.on("degraded", (why) => note(`room degraded: ${why}`));
  r.node.on("hostgone", () => note("lost the link to the room's host"));
  r.node.on("loaded", (x) => note(`this device holds layers ${x.range[0]}-${x.range[1] - 1} of ${x.model}`));
  r.node.on("online", () => note("room online"));
  r.close = async () => { r.closed = true; r.pullAbort?.abort(); try { await r.bridge?.leave?.(); } catch {} try { r.bridgeTry?.b?.destroy?.(); } catch {} try { await r.node?.close(); } catch {} };
  r.status = () => status(r);
  note(`${s.mode === "host" ? "hosting" : "joined"} Pooled room ${fmtCode(r.code)}${s.mode === "host" ? " (/pooled link shows its invite link)" : ""}`);
  return r;
}

// host: download the model into the shared cache while devices join
function startPull(r) {
  const { s } = r, key = s.model;
  const ac = new AbortController();
  r.pullAbort = ac;
  r.pull = pullState(key);
  r.note(`${key} is not in ${s.modelDir.replace(os.homedir(), "~")} yet: downloading it (${fmtBytes(r.pull.total)}); devices can join the room meanwhile`);
  let step = -1;
  r.pullP = download(s.modelDir, key, { signal: ac.signal, st: r.pull, onChange: (st) => {
    const pct = st.total ? Math.floor((st.done / st.total) * 10) : 0;
    if (st.state === "running" && pct > step) { step = pct; r.note(`downloading ${key}: ${pullLine(st)}`); }
  } }).then((st) => {
    if (st.state === "done") r.note(`${key} downloaded`);
    else if (st.error !== "stopped") r.note(`${key}: ${pullLine(st)}; this machine streams its layers from Hugging Face instead`);
    return st;
  });
}
export const _startPull = startPull;

export function status(r) {
  const st = r.node.status();
  const out = { mode: r.s.mode, link: r.link, ...st, model: st.model || r.s.model, modelHost: st.hosting, needGB: modelInfo(r.s.model, r.s.ctx || 0).needGB };
  if (r.s.mode === "host") out.waiting = (r.node.waitingJoins?.() || []).map((q) => ({ id: q.id, name: q.name, line: q.line }));
  else { out.admission = r.node.admission || "wait"; out.hostName = r.node.hostName || null; if (r.refused) out.refused = r.refused; }
  if (r.pull) out.download = { ...r.pull, line: pullLine(r.pull) };
  out.rows = deviceRows(r, st);
  return out;
}

// the room's devices for /pooled: { name, gpu, gb, range, self }, this device first (the host's view:
// its links; a joined device's: the room's member list)
export function deviceRows(r, st = r.node.status()) {
  const n = r.node;
  const ranges = new Map((n.split?.names || []).map((nm, i) => [nm, n.split.ranges[i]]));
  const row = (name, meta, gb, self = false) => ({ name, gpu: gpuLabel(meta?.gpu) || "", gb, range: ranges.get(name) || null, self });
  if (st.hosting) {
    const out = [row(n.name, n.meta, +n.meta?.contribGB || st.devices?.[0]?.gb || 0, true)];
    for (const id of n.gpuPeers?.() || []) { const e = n.conns.get(id); out.push(row(e?.name || id, e?.meta, +e?.meta?.contribGB || 0)); }
    return out;
  }
  return (n.members || []).filter((m) => m.meta?.webgpu && !m.meta?.api).map((m) => row(m.name, m.meta, +m.meta?.contribGB || 0, m.name === n.name));
}

// whether the pledges in the room hold the model, as the room page decides it -> { fits, note }
export function fitNow(r, st = status(r)) {
  const devices = st.devices.map((d) => ({ name: d.name, meta: { contribGB: d.gb, webgpu: true } }));
  return roomFitNow(lib, { model: r.s.model, devices, ctxAsk: pluginAsk(r.s.model, r.s.ctx || 0) });
}

// The host side before an ask: wait until enough devices (and memory) are in the room, then deal the
// layers; a room waiting for a device to come back waits too. Throws PooledError with a message
// meant for the chat. waitPull: wait for a download in progress (the service) instead of saying so.
// waitMs: how long to wait for devices (default the waitSeconds setting; the service waits for good)
export async function ensureOnline(r, { signal, onWait = () => {}, waitPull = false, waitMs = r.s.waitSeconds * 1000 } = {}) {
  const n = r.node, s = r.s, info = modelInfo(s.model, s.ctx || 0), code = fmtCode(r.code);
  if (n.ai.online && !n.ai.degraded) return;
  if (r.pull && (r.pull.state === "waiting" || r.pull.state === "running" || r.pull.state === "checking")) {
    if (!waitPull) throw new PooledError("downloading", downloadingMessage(r.pull, code));
    await r.pullP;
  }
  const t0 = Date.now();
  let said = 0;
  if (n.ai.engine && n.ai.degraded) {   // a device dropped: it may come back into its slot, or the room re-deals
    while (!n.whole()) {
      if (signal?.aborted || r.closed) throw new PooledError("abort", "aborted");
      // the re-deal found the devices still here short of the model and stopped the room (no device
      // is dealt past its pledge): wait for devices to join, as a Start the pledges don't cover does
      if (!n.ai.engine && !n.ai.starting && !n.ai.loadingShard) break;
      if (Date.now() - t0 > waitMs) throw new PooledError("degraded", `a device left Pooled room ${code} while it held layers of the model (${n.missingNames().join(", ") || "reloading"}). ` +
        `Re-open ${r.shareLink} on that device (or its invite link: /pooled link); the room re-deals over the devices still there after a minute`);
      await new Promise((res) => setTimeout(res, 500));
    }
    if (n.whole()) return;
  }
  for (;;) {
    if (signal?.aborted || r.closed) throw new PooledError("abort", "aborted");
    const st = status(r);
    const fit = fitNow(r, st);
    if (st.devices.length >= s.minDevices && fit.fits) break;
    if (Date.now() - t0 > waitMs) {
      const waiting = st.waiting?.length ? ` ${st.waiting.map((q) => q.line).join("; ")}: send \`/pooled allow\` to let ${st.waiting.length > 1 ? "them" : "it"} in.` : "";
      if (st.devices.length < s.minDevices)
        throw new PooledError("waiting", `${st.devices.length} of ${s.minDevices} devices are in the room.${waiting} ` +
          "Open the room's invite link on your other device (`/pooled link` shows it), or paste it in OpenClaw there (Pooled → Join a room). Then ask again.");
      throw new PooledError("memory", `${info.name} needs about ${fit.needGB ?? info.needGB} GB; the devices in the room lend ${st.pledgedGB} GB ` +
        `(${st.devices.map((d) => `${d.name} ${d.gb} GB`).join(", ")}).${waiting} ` +
        "Add a device with the room's invite link (`/pooled link`), lend more with `/pooled pledge <GB>`, or pick a smaller model.");
    }
    if (Date.now() - said > 10000) { said = Date.now(); onWait(st); }
    await new Promise((res) => setTimeout(res, 500));
  }
  // one start at a time (the service and a question can both get here)
  r.startP ||= n.start(s.model).finally(() => { r.startP = null; });
  await r.startP.catch((err) => {
    throw new PooledError(err.short || /memory|allocate|OOM|out of memory|maxBufferSize/i.test(err.message) ? "memory" : "start",
      `Pooled room ${code} could not load ${info.name}: ${err.message}`);
  });
  // the room was short of memory for 16k and opened the model at 8k (room/models.js pickCtx): too short
  // for OpenClaw's prompt; say so once per start (/pooled shows it)
  const st = n.status(), short = ownShortCtxNote(s.model, st.ctx, st.ctxNote, info.needGB);
  if (short) r.note(short);
}

const lobbyMessage = (r) => `waiting for the host of room ${fmtCode(r.code)} to let this device in. On the host: press Allow (pooled.run), ` +
  "a (`pooled host`) or send `/pooled allow` (OpenClaw). The room's invite link gets in without asking: " +
  "run `openclaw onboard` here, pick Pooled → Join a room and paste it.";
const deniedMessage = (r, why) => `the host of room ${fmtCode(r.code)} turned this device away (${why}). Ask them for the room's invite link, then run \`openclaw onboard\` → Pooled → Join a room with it.`;

// a joined device: in the room (the host let it in), or a PooledError after lobbyMs in the lobby
export async function admitted(r, { signal, lobbyMs = LOBBY_MS } = {}) {
  const n = r.node, t0 = Date.now();
  for (;;) {
    if (r.refused) throw new PooledError("denied", deniedMessage(r, r.refused));
    if (n.admission === "in") return;
    if (signal?.aborted) throw new PooledError("abort", "aborted");
    if (Date.now() - t0 > lobbyMs) throw new PooledError("lobby", lobbyMessage(r));
    await new Promise((res) => setTimeout(res, 100));
  }
}

// The joined side: an API bridge to the room's host (created on first use). It shows the host this
// device's pass, so the host sees one device, not a second join request.
export async function bridgeFor(r, { signal, lobbyMs = LOBBY_MS } = {}) {
  if (r.bridge && !r.bridge.kicked) return r.bridge;
  if (!r.bridgeTry) {
    const { Peer } = await r.P.setupNode({ webgpu: r.P.dawn });   // the room node's WebRTC stack, not a second one
    const b = new r.P.Bridge({ code: r.code, key: r.key, signal: r.s.signal, name: `${r.node.name} asks`, client: "OpenClaw", log: (m) => r.node.log(`[bridge] ${m}`), Peer });
    b.pass = validKey(r.node.pass) ? r.node.pass : null;
    const t = { b, done: false, err: null };
    t.p = b.connect().then(() => {
      t.done = true; r.bridge = b;
      if (validKey(b.pass) && !validKey(r.node.pass)) try { saveJoinState(r.code, { pass: b.pass }); } catch {}
      if (Number.isInteger(b.hostMeta?.ctx)) try { saveJoinState(r.code, { ctx: b.hostMeta.ctx }); } catch {}
      const short = shortCtxNote(b.model || b.hostMeta?.model, b.hostMeta?.ctx, b.hostName || "the host");
      if (short) r.note?.(short);   // (/pooled shows it; OpenClaw itself only says "Context overflow")
    }, (err) => { t.err = err; }).finally(() => { if (r.bridgeTry === t) r.bridgeTry = null; });
    r.bridgeTry = t;
  }
  const t = r.bridgeTry;
  let lobbyAt = null;
  for (;;) {
    if (t.done) return t.b;
    if (t.err) {
      if (t.b.kicked) throw new PooledError("denied", deniedMessage(r, t.b.kicked));
      throw new PooledError("noroom", `could not reach Pooled room ${fmtCode(r.code)}'s host: ${t.err.message}`);
    }
    if (signal?.aborted) throw new PooledError("abort", "aborted");
    if (t.b.waiting) { lobbyAt ??= Date.now(); if (Date.now() - lobbyAt > lobbyMs) throw new PooledError("lobby", lobbyMessage(r)); }
    await new Promise((res) => setTimeout(res, 100));
  }
}

// Where an ask goes: { hostMeta, ask(rid, body, handler) -> { stop() } }. The in-process host
// (RoomNode.request) and the bridge to another host (Bridge.ask) take the same ai-ask body and call
// handler with the same room messages (ai-genstart, ai-token, ai-call, ai-gendone, ai-busy).
export async function transport(r, { signal, lobbyMs = LOBBY_MS } = {}) {
  if (r.node.hosting()) {
    await ensureOnline(r, { signal, onWait: (st) => r.node.log(`waiting for devices in room ${fmtCode(r.code)}: ${st.devices.map((d) => `${d.name} ${d.gb} GB`).join(", ")} (need ${st.needGB} GB)`) });
    return { hostMeta: r.node.hostMeta, ask: (rid, body, handler) => r.node.request(body, handler, { rid }) };
  }
  await admitted(r, { signal, lobbyMs });
  const b = await bridgeFor(r, { signal, lobbyMs });
  // a browser host ignores asks until its model is up (room.js: ai-ask only when ai.role is "host"):
  // wait for its ai-ready-all, then say so plainly
  const t0 = Date.now();
  while (!b.ready) {
    if (signal?.aborted) throw new PooledError("abort", "aborted");
    if (b.kicked) throw new PooledError("off", `the host of Pooled room ${fmtCode(r.code)} closed the link: ${b.kicked}`);
    if (Date.now() - t0 > r.s.waitSeconds * 1000)
      throw new PooledError("waiting", `room ${fmtCode(r.code)} has no model running yet. Press Start on the room's host, or wait for it to finish loading, then ask again.`);
    await new Promise((res) => setTimeout(res, 500));
  }
  return {
    hostMeta: b.hostMeta,
    ask: (rid, body, handler) => {
      if (!b.ask(rid, body, handler)) throw new PooledError("gone", `lost the link to Pooled room ${fmtCode(r.code)}'s host`);
      return { stop: () => b.stop(rid) };
    },
  };
}
