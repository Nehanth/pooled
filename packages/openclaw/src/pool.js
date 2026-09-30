// The Pooled room this OpenClaw process belongs to: one per process (a global, so a plugin registry
// reload does not open a second room), started by the plugin's service at gateway startup or by the
// first request, whichever comes first.
//
//   mode "host": this machine creates room <code>, holds the embedding, the head and its share of the
//                layers, and answers OpenClaw's requests itself (RoomNode.request, in process: no HTTP).
//   mode "join": this machine joins room <code> as a device that holds layers when the host deals
//                it some; OpenClaw's requests go to the room's host as API asks (the `pooled serve`
//                bridge, cli/lib/room.js) over WebRTC. If the room picks this machine to run the model
//                (a browser host's pickModelHost: it pledged the most, or its GPU is clearly faster),
//                the requests are answered here, as in host mode.
// Either way a request is the same ai-ask body with the same answer messages (transport() below).
import { MODELS, NEED_GB, CTX, maxSeqFor } from "../../../room/models.js";

const KEY = Symbol.for("pooled.openclaw.room");
export const ROOM_ORIGIN = "https://pooled.run";
// what onboarding offers (room/models.js PICKER), with the context the plugin asks for: the model's
// largest room context (room/models.js CTX; the MoE's 128k, OpenClaw's prompts alone are 8-12k)
export const MODEL_CHOICES = ["qwen3-1.7b", "qwen3.8-27b", "qwen3.6-35b-moe"];
export const modelInfo = (key) => ({
  name: String(MODELS[key]?.label || key).split("·")[0].trim(),
  needGB: NEED_GB[key] ?? null,
  ctx: CTX[key]?.max ?? maxSeqFor(key),
});

export class PooledError extends Error {
  constructor(code, message) { super(message); this.code = code; }
}

export const roomLink = (code, signal) => `${ROOM_ORIGIN}/room/${code}${signal ? `?signal=${encodeURIComponent(signal)}` : ""}`;

// the settings, from plugins.entries.pooled.config (onboarding writes them) and POOLED_* env
export function roomSettings(pluginConfig = {}, env = process.env) {
  const c = { ...(pluginConfig || {}) };
  return {
    mode: env.POOLED_MODE || c.mode || null,
    code: String(env.POOLED_CODE || c.code || "").toUpperCase() || null,
    model: env.POOLED_MODEL || c.model || "qwen3-1.7b",
    pledgeGB: +(env.POOLED_PLEDGE_GB || c.pledgeGB || 0) || null,
    minDevices: +(env.POOLED_MIN_DEVICES || c.minDevices || 1),
    waitSeconds: +(env.POOLED_WAIT_S || c.waitSeconds || 120),
    ctx: +(env.POOLED_CTX || c.ctx || 0) || null,
    signal: env.POOLED_SIGNAL || c.signal || null,
    modelDir: env.POOLED_MODELS || c.modelDir || null,
    name: env.POOLED_NAME || c.name || null,
    page: env.POOLED_PAGE || c.page || null,   // a room page other than pooled.run (local tests)
  };
}

// the runtime: @pooled/room-node (Dawn + node-datachannel) and the `pooled serve` bridge, loaded on
// first use so the plugin registers without touching the GPU
let modP = null;
async function pooledModules() {
  return modP ||= (async () => {
    try {
      const node = await import("../../room-node/index.js");
      const { Bridge } = await import("../../../cli/lib/room.js");
      return { ...node, Bridge };
    } catch (err) {
      modP = null;
      throw new PooledError("install", `the Pooled runtime did not load (${err.message}); run npm install in packages/room-node`);
    }
  })();
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
export const keyOf = (s) => JSON.stringify([s.mode, s.code, s.model, s.pledgeGB, s.signal, s.ctx]);

async function openRoom(s, log) {
  if (!s.mode) throw new PooledError("setup", "Pooled is not set up on this machine: run `openclaw onboard` (or `openclaw models auth login --provider pooled`) and pick Pooled");
  const P = await pooledModules();
  const r = { s, P, code: s.code, bridge: null, events: [] };
  const note = (m) => { r.events.push({ t: Date.now(), m }); if (r.events.length > 50) r.events.shift(); log(m); };
  const common = { pledgeGB: s.pledgeGB || undefined, signal: s.signal, modelDir: s.modelDir, name: s.name || undefined, log: note, ctx: s.ctx || modelInfo(s.model).ctx };
  if (s.mode === "host") {
    r.node = await P.createRoom({ model: s.model, code: s.code || undefined, ...common });
    r.code = r.node.code;
  } else if (s.mode === "join") {
    if (!s.code) throw new PooledError("setup", "no room code: run the Pooled setup again and enter the room code");
    r.node = await P.joinRoom(s.code, common).catch((err) => {
      throw new PooledError("noroom", `could not join Pooled room ${s.code}: ${err.message}. Is the room still open on the other device?`);
    });
  } else throw new PooledError("setup", `unknown Pooled mode ${s.mode}`);
  r.link = roomLink(r.code, s.page ? null : s.signal);
  r.node.on("degraded", (why) => note(`room degraded: ${why}`));
  r.node.on("hostgone", () => note("lost the link to the room's host"));
  r.node.on("loaded", (x) => note(`this device holds layers ${x.range[0]}-${x.range[1] - 1} of ${x.model}`));
  r.node.on("online", () => note("room online"));
  r.close = async () => { try { await r.bridge?.leave?.(); } catch {} try { await r.node?.close(); } catch {} };
  r.status = () => status(r);
  note(`${s.mode === "host" ? "hosting" : "joined"} Pooled room ${r.code} (${r.link})`);
  return r;
}

export function status(r) {
  const st = r.node.status();
  return { mode: r.s.mode, link: r.link, ...st, model: st.model || r.s.model, modelHost: st.hosting, needGB: modelInfo(r.s.model).needGB };
}

// The host side before an ask: wait until enough devices (and memory) are in the room, then deal the
// layers; a room waiting for a device to come back waits too. Throws PooledError with a message
// meant for the chat.
export async function ensureOnline(r, { signal, onWait = () => {} } = {}) {
  const n = r.node, s = r.s, info = modelInfo(s.model);
  if (n.ai.online && !n.ai.degraded) return;
  const t0 = Date.now(), waitMs = s.waitSeconds * 1000;
  let said = 0;
  if (n.ai.engine && n.ai.degraded) {   // a device dropped: it may come back into its slot, or the room re-deals
    while (!n.whole()) {
      if (signal?.aborted) throw new PooledError("abort", "aborted");
      if (Date.now() - t0 > waitMs) throw new PooledError("degraded", `a device left Pooled room ${r.code} while it held layers of the model (${n.missingNames().join(", ") || "reloading"}). ` +
        `Re-open ${r.link} on that device; the room re-deals over the devices still there after a minute`);
      await new Promise((res) => setTimeout(res, 500));
    }
    return;
  }
  for (;;) {
    if (signal?.aborted) throw new PooledError("abort", "aborted");
    const st = status(r);
    if (st.devices.length >= s.minDevices && st.pledgedGB >= (info.needGB || 0)) break;
    if (Date.now() - t0 > waitMs) {
      if (st.devices.length < s.minDevices)
        throw new PooledError("waiting", `Pooled room ${r.code} is waiting for devices: ${st.devices.length} of ${s.minDevices} joined. ` +
          `Open ${r.link} on your other device (or pick "Join a room" in OpenClaw there, code ${r.code}), then ask again`);
      throw new PooledError("memory", `not enough memory in Pooled room ${r.code} for ${info.name}: ` +
        `the devices pledged ${st.pledgedGB} GB (${st.devices.map((d) => `${d.name} ${d.gb} GB`).join(", ")}), it needs about ${info.needGB} GB. ` +
        `Add a device at ${r.link}, raise a pledge, or pick a smaller model`);
    }
    if (Date.now() - said > 10000) { said = Date.now(); onWait(st); }
    await new Promise((res) => setTimeout(res, 500));
  }
  await n.start(s.model).catch((err) => {
    throw new PooledError(/memory|allocate|OOM|out of memory|maxBufferSize/i.test(err.message) ? "memory" : "start",
      `Pooled room ${r.code} could not load ${info.name}: ${err.message}`);
  });
}

// The joined side: an API bridge to the room's host (created on first use)
export async function bridgeFor(r) {
  if (r.bridge && !r.bridge.kicked) return r.bridge;
  const b = new r.P.Bridge({ code: r.code, signal: r.s.signal, name: `${r.node.name}-openclaw`, client: "OpenClaw", log: (m) => r.node.log(`[bridge] ${m}`),
    Peer: (await r.P.setupNode()).Peer });   // the room node's WebRTC stack, not a second one
  await b.connect().catch((err) => { throw new PooledError("noroom", `could not reach Pooled room ${r.code}'s host: ${err.message}`); });
  r.bridge = b;
  return b;
}

// Where an ask goes: { hostMeta, ask(rid, body, handler) -> { stop() } }. The in-process host
// (RoomNode.request) and the bridge to another host (Bridge.ask) take the same ai-ask body and call
// handler with the same room messages (ai-genstart, ai-token, ai-call, ai-gendone, ai-busy).
export async function transport(r, { signal } = {}) {
  if (r.node.hosting()) {
    await ensureOnline(r, { signal, onWait: (st) => r.node.log(`waiting for devices in room ${r.code}: ${st.devices.map((d) => `${d.name} ${d.gb} GB`).join(", ")} (need ${st.needGB} GB)`) });
    return { hostMeta: r.node.hostMeta, ask: (rid, body, handler) => r.node.request(body, handler, { rid }) };
  }
  const b = await bridgeFor(r);
  // a browser host ignores asks until its model is up (room.js: ai-ask only when ai.role is "host"):
  // wait for its ai-ready-all, then say so plainly
  const t0 = Date.now();
  while (!b.ready) {
    if (signal?.aborted) throw new PooledError("abort", "aborted");
    if (b.kicked) throw new PooledError("off", `the host of Pooled room ${r.code} closed the link: ${b.kicked}`);
    if (Date.now() - t0 > r.s.waitSeconds * 1000)
      throw new PooledError("waiting", `Pooled room ${r.code} has no model running yet: press Start on the room's host (${r.link}), or wait for it to finish loading, then ask again`);
    await new Promise((res) => setTimeout(res, 500));
  }
  return {
    hostMeta: b.hostMeta,
    ask: (rid, body, handler) => {
      if (!b.ask(rid, body, handler)) throw new PooledError("gone", `lost the link to Pooled room ${r.code}'s host`);
      return { stop: () => b.stop(rid) };
    },
  };
}
