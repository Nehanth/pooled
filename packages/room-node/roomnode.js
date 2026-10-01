// A headless room node: a Pooled room member (or host) in a Node process, on Dawn (WebGPU) and
// node-datachannel (WebRTC) under PeerJS, speaking the room protocol (docs/protocol.md) so it can
// sit in the same room as browser tabs and phones on pooled.run.
//
//   const room = await createRoom({ model: "qwen3-1.7b", pledgeGB: 8 });   // host
//   room.code; await room.start();                                          // deal layers
//   for await (const ev of room.ask([{ role: "user", content: "Hi" }], { temperature: 0 })) ...
//   const node = await joinRoom("ABCD", { pledgeGB: 8 });                   // worker
//
// What comes from where (room.js is one DOM module; nothing of it is imported):
//   shared as-is (DOM-free modules, the same code the browser room and `pooled serve` run):
//     room/transport.js (wire frames, stripes, ordered delivery, keep-alive), room/wire.js,
//     room/plan.js, room/models.js, room/pledge.js (phone caps, a smaller share after a killed load),
//     room/conversation.js, room/sampling.js, room/liveness.js (silence rules, lap timeouts),
//     room/lookup.js, room/resume.js (an answer carries on after a device drops, sameShard),
//     room/gpuspeed.js (the copy speed that picks the model host), engine/preset.js;
//     API asks: room/api.js (validateApiAsk, apiPrompt / apiPrompt2, apiRun / apiRun2: the host side
//     of tool calling), and cli/lib (common.js finishRequest + askBody, answer.js Ask: the client side
//     `pooled serve` uses). ask() and request() below go through exactly that path, so there is one
//     tool-call implementation for the browser host, `pooled serve` and this node.
//   extracted from room.js with the DOM taken out (same logic, same messages):
//     the link layer (wire, ensureLink, sendHidden, roster, hello, ping, leaving, bye); aiLoadShard ->
//     shard.js; aiStart -> start() (memory split); aiMaybeReady (with ai-linked relinks); aiPeerLeft,
//     aiRejoin, aiLoadDeath (a device comes back into its slot, or the room is re-dealt); lapWait /
//     failWaiters / lapDone, sendChain (ai-wake for phones), resetState; aiPipeToken (host fuse:
//     headAhead), aiPrefill, roomGenerate (plain and speculative decode, lookup drafts, preTrunk);
//     workerFrame and the worker's ai-load (keeps layers it already holds), ai-next relink ->
//     ai-linked, ai-share, knocking on the host id after the host link drops (hello back: 1);
//     aiAsk + aiGenerate (a chat question from a browser tab); apiAsk + apiGenerate.
//   reimplemented: the queue (a promise chain instead of ai.queue + ai-queued), the ping loop.
//   checkpoints (ckpt.js): the browser room's pinned prefix + answer checkpoints (ckptSave /
//     ckptResume, sv / ld / dp on the frame header), with more slots: pinned system prompts and an
//     agent's cache boundary, answer states kept by last use, one index for every session.
//   left out: disk copies of checkpoints, host resume after a reload, the speed split, dead-link redial
//     (ICE state watch), changing the visibility at run time (the constructor sets it), Code mode, reactions/typing, the room
//     map, weight caches and peer weights (ai-wget answered "miss"), the bandwidth test.
import fs from "node:fs";
import { EventEmitter } from "node:events";
import { setupNode, probeMeta } from "./env.js";
import * as env from "./env.js";
import { openModel } from "./source.js";
import { loadShard } from "./shard.js";
import { makeLink, attachWire, wireReady, sendFrame, PROTOCOL, DROP_ALL } from "../../room/transport.js";
import { packWire, unpackWire, badF32 } from "../../room/wire.js";
import { planSplit, planForSpeed, phonesToLeaveOut, isPhoneMeta, roomFit } from "../../room/plan.js";
import { MODELS, CTX, roomBytes, maxSeqFor, ctxForBinding, kvModeFor, kvForLoad, kvBytesPerLayerPos, MAX_NEW, MIN_ROOM, pickCtx, ctxShortNote } from "../../room/models.js";
import { CkptIndex, CKPT_DEFAULTS, boundaryPin, pinPoints, cutPoints, turnPoint } from "./ckpt.js";
import { isPrefix } from "../../harness/prefix.js";
import { PERSONAS, specials, fitContext, reusablePrefix, templateProfile } from "../../room/conversation.js";
import { pickSampler } from "../../room/sampling.js";
import { validateApiAsk, apiPrompt, apiRun, AnswerCache, helloMeta, pieceDecoder, API_LIMITS, apiPrompt2, apiRun2, TurnCache, EncodeCache } from "../../room/api.js";
import { tokenTexts } from "../../harness/model-common.js";
import { uniqueName, PING_MS, lastHeard, isSilentGone, lapTimeout, staleNamesakes, NAME_PROBE_MS } from "../../room/liveness.js";
import { lookupDrafts, denseLookupDrafts } from "../../room/lookup.js";
import { resumableGenerate, waitForRoom, sameShard, linkSilent, REJOIN_GRACE_MS, LINK_SILENT_MS } from "../../room/resume.js";
import { pledgeGB, afterLoadDeath } from "../../room/pledge.js";
import { GGML_EMBED, GGML_OUTPUT, ggmlLayerNames, qwen35ShardBytes, qwen35MtpBytes } from "../../engine/gguf.js";
import { guardChunks } from "../../cli/lib/room.js";
import { parseServer, openPeer, reconnectDelay, FALLBACK_ERRORS } from "../../room/signal.js";
import { withDefaults, finishRequest, askBody, needsV2, ApiError } from "../../cli/lib/common.js";
import { chatRecipients } from "../../room/visibility.js";
import { Ask, Collector } from "../../cli/lib/answer.js";
import { hostGate, gateHelloFields, holdConn, allowJoin, denyJoin, waitingJoins, joinHelloFields, deviceGateMessage, onHostHello,
  keyFragment, randomCode, CODE_LEN } from "./gate.js";

export const PREFIX = "pooled-room-";
const ICE = { iceServers: [{ urls: ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"] }] };
const CODE_ABC = "ABCDEFGHJKMNPQRSTVWXYZ23456789";   // room/plan.js codeFromLocation's alphabet
export const randCode = (n = 4) => Array.from(crypto.getRandomValues(new Uint32Array(n)), (x) => CODE_ABC[x % CODE_ABC.length]).join("");
const PREFILL_WINDOW = 6;
export const RELINK_MS = 15000;      // how long the host waits for an ai-linked (an older device never sends one)
export const HOST_WAIT_MS = 60000;   // a worker knocks on the host id this long after the host link dropped
// messages only the room's host (or the device it made model host) may send
const FROM_HOST = new Set(["ai-layers", "ai-ready-all", "ai-reset", "ai-redeal", "ai-degraded", "ai-map", "ai-genstart",
  "ai-token", "ai-gendone", "ai-history", "ai-reacts", "ai-queue", "ai-queued", "ai-regen", "ai-hostprog", "ai-next",
  "ai-visibility", "ai-style", "ai-busy", "ai-wait", "ai-start-failed", "ai-load", "ai-share", "ai-wake"]);
// The context a node opens a room with: what was asked (clamped by room/models.js), else the largest
// room context the model allows (room/models.js CTX: 16k on the 1.7B, 64k on the 27B, 128k on the MoE),
// as the OpenClaw plugin opens it (packages/openclaw pluginCtx): an agent's prompt alone is 8-12k
// tokens, and the 1.7B at 8k ended every OpenClaw turn in "Context overflow". The 1.7B's f32 cache
// costs 1.8 GB more at 16k (5.6 GB in all); a room whose pledges hold it only at 8k opens it there
// (room/models.js pickCtx, as the room page does), and --ctx 8192 asks for 8k. A model without a CTX
// entry keeps the room's default. (qwen35: the deal lowers it to what every device can bind.)
export const nodeCtxFor = (model, ask = 0) => (ask > 0 || !CTX[model] ? maxSeqFor(model, ask) : CTX[model].max);
// what a host that closes its room says to the devices in it (bye {closed: 1})
export const HOST_CLOSED = "The host closed the room.";
export const cleanName = (s, id) => String(s ?? id).replace(/[\u0000-\u001f\u007f<>"'`&]/g, "").trim().slice(0, 40) || String(id).slice(0, 8);

export class RoomNode extends EventEmitter {
  // name, pledgeGB, signal ("host:port" PeerServer, null = the PeerJS cloud), modelDir, flags (engine
  // switches, engine/preset.js), stripes, log, selfTest, chatMaxNew, ctx (context to ask for; clamped
  // by room/models.js), gbps (pin the copy speed; default measured), autoRedeal (re-deal without a
  // device that does not come back in REJOIN_GRACE_MS; default on), ckpt (checkpoints on the host:
  // { answers, pins, minPin } over ckpt.js CKPT_DEFAULTS, or false for none), visibility (who sees the
  // answers on the room's screens, room/visibility.js: "all" as the room page's default, "asker": only
  // whoever asked; an agent host uses "asker" so its prompts and answers stay off other devices'
  // screens), allowApi (answer API asks from other devices; default on, as the room page)
  constructor({ name, pledgeGB, signal = null, modelDir, flags = "", stripes = 4, log = null, selfTest = true, chatMaxNew = MAX_NEW, ctx = 0,
    gbps = null, autoRedeal = true, ckpt = {}, visibility = "all", allowApi = true, setup = {}, expectHost = null, key = null, pass = null, beforeLoad = null, split = "memory" } = {}) {
    super();
    // split: how a deal spreads the layers, as the room page's "Layer split" (room.js ai-split):
    // "memory" = over every device in proportion to what it lends (this node's default so far);
    // "speed" = the fastest devices first, each up to what it lends, the rest not needed (room/plan.js
    // planForSpeed; until answers are measured: the host first, then the biggest). setSplit() changes it.
    this.splitMode = split === "speed" ? "speed" : "memory";
    // beforeLoad(modelKey): awaited before a dealt shard opens the model (pooled join finishes pulling
    // it to disk there); a throw fails that load like any other load error
    this.beforeLoad = typeof beforeLoad === "function" ? beforeLoad : null;
    this.loadStat = null;   // the last "loadstat" of the shard loading here (watchLoad)
    // joining (gate.js, docs/protocol.md "Joining a room"). A device: the invite key from its link and the
    // pass the host gave it; admission: null | "wait" | "lobby" | "in". A host: its gate (createRoom), and
    // the links waiting in its lobby
    this.key = key; this.pass = pass; this.admission = null;
    this.gate = null; this.lobbyConns = new Map(); this.protocol = PROTOCOL;
    this.setup = setup;   // setupNode options (webgpu: a loader for Dawn, dawnFlags)
    // a worker: the host's name it will serve under this code (a rejoin after the room was over).
    // Room codes are short and reusable; a host of another name is another room, which this device
    // did not choose to join: it leaves (the "otherhost" event) before it hears anything else
    this.expectHost = expectHost;
    this.visibility = visibility === "asker" || visibility === "host" ? visibility : "all";
    this.allowApi = allowApi !== false;
    this.ctxAsk = ctx;
    this.ckptOpts = ckpt === false ? null : { ...CKPT_DEFAULTS, ...(ckpt || {}) };
    this.chatMaxNew = chatMaxNew;
    this.name = name || "node-" + randCode(3).toLowerCase();
    this.pledgeGB = pledgeGB; this.signal = signal; this.modelDir = modelDir; this.flags = flags;
    this.stripes = stripes; this.selfTest = selfTest; this.gbpsPin = gbps; this.autoRedeal = autoRedeal;
    this.log = log || ((s) => this.emit("log", s));
    this.peer = null; this.isHost = false; this.code = null; this.meta = null;
    this.conns = new Map();    // peer id -> { conn, name, meta, link, stripes, seen, missed, rtt }
    this.roster = new Map();   // host: id -> { name, meta }
    this.probedHellos = new WeakMap();   // host: a hello held back while its namesake is pinged -> when
    this.pending = new Map();  // ensureLink in flight
    this.ai = { role: null, engine: null, tok: null, cfg: null, device: null, chain: [], next: null, hostId: null,
      readyPeers: new Set(), pos: 0, fed: [], pendingCtl: {}, waiters: new Map(), q: Promise.resolve(), lock: Promise.resolve(),
      conv: { turns: [] }, settings: { persona: "default", sampling: "exact", thinking: false },
      apiCache: new AnswerCache(8), apiTurns: new TurnCache(), apiEnc: new EncodeCache(), apiProf: null, apiTT: null,
      apis: new Map(), runs: new Map(), degraded: false, model: null, range: null, online: false,
      plan: new Map(), gone: new Set(), chainNames: [], relinks: new Map(), lapStat: null, held: null,
      shareCap: new Map(), dropped: new Set(), loadDeaths: new Map(),
      ckpt: null, dropQ: [], ckptCap: new Map(), bounds: new Map() };
  }

  // ---------------- link layer (room.js wire / onData, without the DOM) ----------------
  async open(id) {
    await setupNode(this.setup);
    this.meta = await probeMeta(this.pledgeGB, { gbps: this.gbpsPin });
    // the first signaling server that answers (room/signal.js openPeer: a server that is down or
    // unreachable hands over to the next; a taken code or a bad id is an answer, not an outage)
    const servers = nodeServers(this.signal);
    // take connections from the moment the Peer exists: the signaling server can hand one over in the
    // same read as "open" (a device knocking or rejoining while this host starts), before the await
    // below returns, and PeerJS drops a connection event nobody listens to
    const Base = env.Peer, node = this;
    const PeerWithAccept = function (pid, o) { const p = new Base(pid, o); p.on("connection", (conn) => { if (!p.destroyed) node.accept(conn); }); return p; };
    let got;
    try {
      got = await openPeer(PeerWithAccept, id, { debug: 0, config: ICE }, servers, {
        onTry: (s, i, err) => { if (err) this.log(`signaling: ${servers[i - 1].label} failed (${err.type || err.message}); trying ${s.label}`); },
      });
    } catch (err) {
      const e = new Error(err.type === "signaling-down" ? err.message : `peer error: ${err.type || err.message}`);
      e.type = err.type; e.tried = err.tried;
      throw e;
    }
    this.peer = got.peer; this.server = got.server;
    // (a lost signaling server is watchSignaling's to report)
    this.peer.on("error", (err) => { if (err.type !== "peer-unavailable" && !FALLBACK_ERRORS.has(err.type)) this.log(`peer error: ${err.type || err.message}`); });
    this.watchSignaling();
    this.pingTimer = setInterval(() => this.pingTick(), PING_MS);
  }
  // the signaling server dropped (the cloud restarts, the network blips): links already open keep
  // working, only new devices can't find the room. Reconnect with room/signal.js's backoff (2, 4, 8,
  // 16, 30 s ...) until it is back, as the room page does (room.js watchSignaling).
  watchSignaling() {
    const p = this.peer;
    let tries = 0, timer = null;
    this.signalDown = false;
    const again = () => {
      timer = null;
      if (this.closing || p !== this.peer || p.destroyed || p.open) return;
      // still connecting from the last try: under Node a refused WebSocket never closes, so PeerJS
      // would wait on it forever ("still trying to make the initial connection"); drop it first
      if (!p.disconnected) { try { p.disconnect(); } catch {} }
      try { p.reconnect(); } catch {}
      if (!timer) { timer = setTimeout(again, reconnectDelay(tries++)); timer.unref?.(); }
    };
    p.on("disconnected", () => {
      if (this.closing || p !== this.peer || p.destroyed) return;
      if (!this.signalDown) { this.signalDown = true; this.log("lost the signaling server: links already open keep working; reconnecting"); this.emit("signaling", false); }
      if (!timer) { timer = setTimeout(again, reconnectDelay(tries++)); timer.unref?.(); }
    });
    p.on("open", () => {
      clearTimeout(timer); timer = null; tries = 0;
      if (this.signalDown) { this.signalDown = false; this.log("signaling is back"); this.emit("signaling", true); }
    });
  }
  accept(conn) {
    guardChunks(conn);
    conn.on("open", () => {
      // a host with a gate: a new link waits in the lobby until its hello lets it in (gate.js)
      if (this.isHost && this.gate && holdConn(this, conn)) return;
      if (conn.label === "stripe") {   // extra association for the wire, not a new peer
        const e = this.conns.get(conn.peer);
        if (e) this.attachStripe(e, conn);
        return;
      }
      this.wire(conn);
      conn.send(this.helloMsg());
    });
  }
  // (a host's meta names the model it will run, so a device joining before Start can say which)
  helloMsg(extra = {}) { return { t: "hello", name: this.name, meta: this.isHost ? { ...this.meta, api: 2, ctx: this.ctxMax(), ...(this.ai.model ? { model: this.ai.model } : {}) } : this.meta, v: PROTOCOL, ...(this.isHost ? gateHelloFields(this.gate) : {}), ...extra }; }
  attachStripe(e, conn) { attachWire(e.link, conn, (m) => this.onData(conn.peer, m)); e.stripes.push(conn); }
  // host: the room's invite link fragment (#k=...), and Allow / Deny for a device in the lobby
  get inviteFragment() { return keyFragment(this.gate?.key); }
  allowJoin(id) { return allowJoin(this, id); }
  denyJoin(id) { return denyJoin(this, id); }
  waitingJoins() { return waitingJoins(this); }
  // a new link replaces any older one to the same peer id (a device that came back): the old one's
  // close handler sees it is not current and does nothing
  wire(conn, name, initiator = false) {
    const old = this.conns.get(conn.peer);
    const e = { conn, name: name || old?.name || conn.peer, meta: old?.meta || {}, link: makeLink(), stripes: [], seen: performance.now(), missed: 0, rtt: old?.rtt ?? null };
    this.conns.set(conn.peer, e);
    if (old && old.conn !== conn) {
      for (const s of old.stripes) try { s.close(); } catch {}
      try { old.conn.close(); } catch {}
      // a chain device on a fresh link (a phone back from a lock under the same id): what was in
      // flight on the old one is gone, and its hello {back} re-seats it (rejoin)
      if (this.hosting()) this.chainLeft(conn.peer, old.name, "reconnected");
    }
    if (this.stripes > 0) {
      attachWire(e.link, conn, (m) => this.onData(conn.peer, m));
      if (initiator) for (let i = 1; i < this.stripes; i++) {
        const sc = this.peer.connect(conn.peer, { reliable: true, label: "stripe" });
        if (!sc) continue;
        sc.on("open", () => attachWire(e.link, sc, (m) => this.onData(conn.peer, m)));
        sc.on("error", () => {});
        e.stripes.push(sc);
      }
    }
    conn.on("data", (d) => this.onData(conn.peer, d));
    conn.on("close", () => { const x = this.conns.get(conn.peer); if (!x || x.conn !== conn) return; this.peerGone(conn.peer, x); });
    conn.on("error", () => {});
    return e;
  }
  sendTo(id, obj) { try { this.conns.get(id)?.conn.send(obj); } catch {} }
  broadcast(obj, filter = () => true) { for (const [id, e] of this.conns) if (filter(id, e)) this.sendTo(id, obj); }
  sendHidden(id, msg) {
    const e = this.conns.get(id);
    this.sent ||= { wire: 0, msg: 0 };
    if (e?.link && wireReady(e.link) && sendFrame(e.link, msg)) { this.sent.wire++; return; }
    this.sent.msg++;
    this.sendTo(id, msg);   // PeerJS message fallback, as room.js
  }
  ensureLink(id, timeoutMs = 60000) {
    if (!id || id === "host" || this.conns.has(id)) return Promise.resolve(true);
    if (!this.pending.has(id)) {
      this.pending.set(id, true);
      const conn = this.peer.connect(id, { reliable: true });
      if (conn) {
        guardChunks(conn);
        conn.on("open", () => { this.wire(conn, undefined, true); conn.send(this.helloMsg()); });
        conn.on("error", () => {});
      }
    }
    return new Promise((res) => {
      const t0 = performance.now();
      const t = setInterval(() => {
        if (this.conns.has(id)) { clearInterval(t); this.pending.delete(id); res(true); }
        else if (performance.now() - t0 > timeoutMs) { clearInterval(t); this.pending.delete(id); res(false); }
      }, 100);
    });
  }
  // close a link on purpose (a relink): quietly, it is not a departure
  dropLink(id) {
    const e = this.conns.get(id);
    if (!e) return;
    this.conns.delete(id);
    for (const s of e.stripes) try { s.close(); } catch {}
    try { e.conn.close(); } catch {}
  }
  peerGone(id, e) {
    this.conns.delete(id);
    if (this.closing) return;   // close() tore the links down: not a departure (no degraded room, no re-deal)
    if (this.isHost) {
      this.roster.delete(id); this.broadcastRoster();
      this.log(`${e?.name || id} left`);
      const api = this.ai.apis.get(id);
      if (api) { this.ai.apis.delete(id); for (const [k, ac] of this.ai.runs) if (k.startsWith(id + ":")) ac.abort(); }
    }
    if (this.hosting()) this.chainLeft(id, e?.name || id);
    if (!this.isHost && (id === this.ai.hostId || id === PREFIX + this.code)) this.hostGone();
    this.emit("members");
  }
  broadcastRoster() {
    const members = [{ id: this.peer.id, name: this.name, meta: this.meta }, ...[...this.roster].map(([id, m]) => ({ id, ...m }))];
    this.broadcast({ t: "roster", members });
  }
  pingTick() {
    const now = performance.now(), late = now - (this.pingAt || now) > 2 * PING_MS;
    this.pingAt = now;
    for (const [id, e] of [...this.conns]) {
      if (e.meta?.api) { if (this.isHost && ++e.missed > 6) { this.log(`API client ${e.name} stopped answering`); try { e.conn.close(); } catch {} } continue; }
      if (late) { e.seen = now; continue; }
      const loading = this.ai.loadingShard || (this.hosting() && this.ai.starting && this.ai.chain.includes(id) && !this.ai.readyPeers.has(id));
      // a chain device silent for 12 s (45 s while loading) is gone for a while (room/resume.js: a
      // locked phone says nothing); any other link by the ping loop's rules (room/liveness.js)
      const chainSilent = this.hosting() && this.ai.chain.includes(id) && linkSilent(lastHeard(e), now, this.ai.starting ? 45000 : LINK_SILENT_MS);
      if (chainSilent || isSilentGone({ now, heard: lastHeard(e), toHost: id === this.ai.hostId, phone: isPhoneMeta(e.meta), loading })) {
        this.log(`${e.name || id} stopped answering: dropping it`);
        this.conns.delete(id);
        try { e.conn.close(); } catch {}
        this.peerGone(id, e);
      }
    }
    this.broadcast({ t: "ping", ts: now });
  }
  onData(from, d) {
    if (d instanceof ArrayBuffer || ArrayBuffer.isView(d)) return;   // bandwidth-test payloads
    const e = this.conns.get(from);
    if (e) e.seen = performance.now();
    if (!d || typeof d.t !== "string") return;
    if (process.env.RN_DEBUG && d.t !== "ping" && d.t !== "pong") this.log(`<- ${d.t} from ${e?.name || from}`);
    if (this.otherHost && from === PREFIX + this.code) return;   // a stranger's room under this code: nothing from it
    if (d.t.startsWith("ai-")) { this.aiOnData(from, d).catch((err) => this.log("error: " + err.message)); return; }
    switch (d.t) {
      case "hello": {
        if (d.v !== PROTOCOL) {
          this.sendTo(from, { t: "bye", reason: `${this.name} speaks room protocol ${PROTOCOL}, this device ${d.v}: reload the older one` });
          this.emit("version", { theirs: d.v, theyHost: !this.isHost && from === PREFIX + this.code, name: cleanName(d.name, from) });
          return;
        }
        // the name is held by another link (a `pooled join` killed and started again, a reloaded
        // tab, before the old link times out): ping it and decide in a moment, as room.js does for a
        // namesake quiet for a second; a namesake that stays silent is dropped, and this device takes
        // its name and its slot. Any namesake is pinged, not only a quiet one: a process started
        // again at once comes back while its old link was still heard less than a second ago
        if (this.isHost && !d.back) {
          const heardOf = (id) => lastHeard(this.conns.get(id));
          const name = cleanName(d.name, from);
          if (!this.probedHellos.has(d)) {
            const quiet = [...this.roster].find(([id, m]) => id !== from && m.name === name)?.[0];
            if (quiet) {
              const since = performance.now();
              this.sendTo(quiet, { t: "ping", ts: since });
              this.probedHellos.set(d, since);
              setTimeout(() => { if (this.conns.get(from) === e && !this.closing) this.onData(from, d); }, NAME_PROBE_MS).unref?.();
              return;
            }
          } else {
            for (const id of staleNamesakes(name, from, this.roster, heardOf, this.probedHellos.get(d))) {
              const old = this.conns.get(id);
              this.log(`${name} is back under a new link: dropping its old one, silent for ${old ? Math.round((performance.now() - lastHeard(old)) / 1000) : "?"} s`);
              this.roster.delete(id);
              if (old) { this.conns.delete(id); try { old.conn.close(); } catch {} this.peerGone(id, old); }
            }
          }
        }
        d.name = cleanName(d.name, from);
        // a worker: the host's hello comes first on its link. Under this code before (or asked for by
        // expectHost) there was a host of another name: this is another room, so leave it
        if (!this.isHost && from === PREFIX + this.code) {
          const want = this.expectHost || this.hostName;
          if (want && d.name !== want) {
            this.otherHost = { was: want, now: d.name };
            this.log(`room ${this.code} now has another host (${d.name}, was ${want}): leaving it`);
            this.dropLink(from);
            clearInterval(this.knock); this.knock = null; this.freeLayers(null); this.ai.online = false;
            this.emit("otherhost", this.otherHost);
            return;
          }
        }
        if (!this.isHost && from === PREFIX + this.code) onHostHello(this, d);
        d.meta = helloMeta(d.meta, this.isHost);
        // a device coming back under its own name while its old link is still open but silent (a
        // phone back from a lock): the old link is dead, drop it now (room.js dropStaleNamesake)
        if (this.isHost && d.back) for (const [id, m] of [...this.roster]) {
          const old = id !== from && m.name === d.name && this.conns.get(id);
          if (old && performance.now() - lastHeard(old) > 1500) { this.conns.delete(id); try { old.conn.close(); } catch {} this.peerGone(id, old); }
        }
        // a device coming back under its own name keeps it (it is re-seated in its slot below)
        const back = this.isHost && this.ai.plan.has(d.name) && ![...this.roster].some(([id, m]) => id !== from && m.name === d.name && this.conns.has(id));
        if (this.isHost && !back) d.name = uniqueName(d.name, from, this.name, this.roster);
        if (!e) return;
        e.name = d.name; e.meta = d.meta || {};
        if (this.isHost) {
          if (d.meta?.api) this.ai.apis.set(from, { name: d.name, client: d.meta.client });
          this.roster.set(from, { name: d.name, meta: d.meta });
          this.broadcastRoster();
          if (this.visibility !== "all") this.sendTo(from, { t: "ai-visibility", mode: this.visibility });
          if (this.hosting() && !this.loadDeath(from, d)) this.rejoin(from, d.name);
          // a newcomer while the room is online is an ask-only guest (aiWelcome). The layer map first:
          // a device back after the room re-dealt without it (away past the grace, it missed that
          // deal's ai-layers) still holds its old layers, and frees them when it is not in the map
          // (also while a deal loads: it is the map of the deal in progress)
          if (this.ai.layersByName && !this.ai.chain.includes(from)) this.sendTo(from, { t: "ai-layers", by: this.ai.layersByName });
          if (this.ai.online && !this.ai.chain.includes(from)) this.sendTo(from, { t: "ai-ready-all", model: this.ai.model, label: MODELS[this.ai.model]?.label, ctx: this.ctxMax(), ...(this.ai.ctxWant ? { ctxWant: this.ai.ctxWant } : {}) });
          this.log(`${d.meta?.api ? "API client " : ""}${d.name} ${d.back ? "came back" : "joined"}${d.meta?.webgpu ? ` (${d.meta.gpu}, ${d.meta.contribGB} GB)` : ""}`);
        } else if (from === PREFIX + this.code) this.hostName = d.name;
        this.emit("members");
        return;
      }
      case "leaving": try { e?.conn.close(); } catch {} return;
      case "lobby": case "admit": if (!this.isHost && from === PREFIX + this.code) deviceGateMessage(this, d); return;
      case "bye":
        // the host closed the room for good (pooled host q): no knocking, the room is over
        if (!this.isHost && from === PREFIX + this.code && d.closed) {
          this.roomClosed = true; clearInterval(this.knock); this.knock = null;
          // (no log line: whoever listens to "closed" says it, once)
          this.emit("closed", d.reason || HOST_CLOSED);
          return;
        }
        this.log(`bye from ${e?.name || from}: ${d.reason}`); this.emit("bye", d.reason); return;
      case "roster":
        if (from !== PREFIX + this.code) return;
        this.members = d.members;
        for (const m of d.members || []) {
          const c = this.conns.get(m.id); if (c) { c.meta = m.meta || {}; c.name = m.name; }
          if (m.id === this.peer.id && m.name && m.name !== this.name) this.name = m.name;   // the host made it unique
        }
        this.emit("members");
        return;
      case "ping": this.sendTo(from, { t: "pong", ts: d.ts }); return;
      case "pong": if (e) { e.missed = 0; if (Number.isFinite(d.ts)) e.rtt = performance.now() - d.ts; } return;
      case "pledge":
        if (e) e.meta = { ...e.meta, contribGB: d.gb };
        if (this.isHost && this.roster.has(from)) { this.roster.get(from).meta = { ...this.roster.get(from).meta, contribGB: d.gb }; this.broadcastRoster(); }
        return;
    }
  }

  // ---------------- messages: worker and host ----------------
  async aiOnData(from, d) {
    const ai = this.ai, e = this.conns.get(from);
    if (FROM_HOST.has(d.t)) { if (this.hosting()) return; if (from !== (ai.hostId || PREFIX + this.code)) return; }
    if ((d.t === "ai-hiddenret" || d.t === "ai-hiddenret-b") && from !== ai.chain[ai.chain.length - 1]) return;
    switch (d.t) {
      // --- any device: the room picked a model host (room/plan.js pickModelHost: the strongest device,
      // whoever pressed Start); the room's creator stays the PeerJS hub either way
      case "ai-start-req":
        if (d.boss === this.peer.id) { if (!this.hosting()) { this.modelHost = true; ai.role = "host"; } this.start(d.model).catch((err) => this.log("start failed: " + err.message)); }
        else if (this.conns.has(d.boss)) ai.hostId = d.boss;
        return;
      // --- worker
      case "ai-load": return this.workerLoad(from, d);
      case "ai-next":
        // relink: the device after this one came back under the same id; the old link to it is dead
        if (d.relink && this.conns.has(d.next)) this.dropLink(d.next);
        ai.next = d.next;
        this.ensureLink(d.next).then((ok) => { if (d.relink) this.sendTo(ai.hostId || from, { t: "ai-linked", next: d.next, ok }); });
        return;
      case "ai-layers":
        if (ai.role === "worker" && !d.by?.[this.name]) this.freeLayers("guest");
        return;
      case "ai-start-failed": ai.startFailed = d.why || "stopped"; if (!ai.loadingShard) this.freeLayers(null); this.emit("startfailed", d.why); return;
      case "ai-ready-all": ai.online = true; ai.model = d.model; if (!ai.role) ai.role = "guest";
        // the host opened the model at its fallback context (room/models.js pickCtx): say so
        ai.ctxNote = d.ctxWant > d.ctx && MODELS[d.model] ? ctxShortNote(String(MODELS[d.model].label).split("·")[0].trim(), d.ctx, d.ctxWant) : "";
        this.emit("online", d); return;
      case "ai-degraded": case "ai-redeal": ai.online = false; this.emit(d.t.slice(3), d); return;
      case "ai-share":   // the host lowered this device's share (or left it out) after its load was killed
        if (!d.drop && d.gb > 0) this.meta.contribGB = Math.max(0.1, Math.min(this.meta.contribGB || d.gb, d.gb));
        this.log(String(d.why || "the host changed this device's share"));
        return;
      case "ai-wake": return;   // a phone's GPU wake hint; a computer's GPU does not clock down between laps
      case "ai-hidden": case "ai-hidden-b":
        if (ai.role !== "worker") return;
        ai.q = ai.q.then(() => this.workerFrame(d)).catch((err) => { this.log("frame failed: " + err.message); this.sendTo(ai.hostId, { t: "ai-error", message: err.message }); });
        return;
      case "ai-inv-req": this.sendTo(from, { t: "ai-inv", url: d.url, have: [] }); return;   // no weight cache here
      case "ai-wget": this.sendTo(from, { t: "ai-wpart", id: d.id, miss: 1 }); return;
      case "ai-genstart": case "ai-token": case "ai-gendone": case "ai-busy": case "ai-queued": this.emit("chat", d); return;
      // --- host
      case "ai-ready":
        if (!this.hosting() || !ai.chain.includes(from)) return;
        ai.readyPeers.add(from); ai.ckptCap.set(from, !!d.ckpt); this.emit("progress", { name: e?.name, pct: 100 }); this.maybeReady();
        return;
      case "ai-linked":   // worker -> host: its fresh link to a device that came back is up
        if (!this.hosting() || !ai.chain.includes(from)) return;
        ai.relinks.delete(d.next);
        this.maybeReady();
        return;
      case "ai-linklost":   // a worker's link to another chain device dropped: frames on it are gone
        if (!this.hosting() || !ai.chain.includes(from) || d.up) return;
        if (ai.waiters.size || ai.fed != null) { this.failWaiters(new Error(`the link to ${cleanName(d.name, "a device")} dropped; ask again`)); ai.fed = null; this.ckptClear(true); }
        return;
      case "ai-progress": this.emit("progress", { name: e?.name || from, pct: d.pct }); return;
      case "ai-error":
        this.log(`${e?.name || from} failed: ${d.message}`);
        if (this.hosting() && ai.chain.includes(from)) {
          this.failWaiters(new Error(`${e?.name || from}: ${d.message}`));
          if (d.load && ai.starting) ai.startErr?.(new Error(`${e?.name || from} couldn't load its layers (${d.message})`));
        }
        return;
      case "ai-hiddenret": this.lapDone(d.pos, unpackWire(d)); return;
      case "ai-hiddenret-b": this.lapDone("b" + d.basePos, unpackWire(d)); return;
      case "ai-tele": return;
      case "ai-ask":
        if (!this.hosting()) return;
        if (d.api) return this.apiAsk(from, d);
        return this.chatAsk(String(d.text || "").slice(0, 8000), e?.name || "guest", from);
      case "ai-stop":
        if (!this.hosting()) return;
        if (d.rid != null) { ai.runs.get(from + ":" + d.rid)?.abort(); return; }
        if (ai.askerId === from) ai.chatAbort?.abort();
        return;
    }
  }

  // ---------------- worker (room.js ai-load + workerFrame) ----------------
  freeLayers(role) {
    const ai = this.ai;
    ai.role = role; ai.range = null; ai.engine = null; ai.held = null;
    try { ai.device?.destroy(); } catch {}
    ai.device = null;
  }
  async workerLoad(from, d) {
    const ai = this.ai;
    if (d.v != null && d.v !== PROTOCOL) { this.sendTo(from, { t: "ai-error", message: `protocol ${PROTOCOL} here, ${d.v} on the host` }); this.emit("version", { theirs: d.v, theyHost: true }); return; }
    if (!MODELS[d.model]) { this.sendTo(from, { t: "ai-error", message: `unknown model ${d.model}`, load: 1 }); return; }
    const kv = kvForLoad(d.model, d.kv, null);
    // re-seated in its slot (the host link dropped and came back) with the same layers still on the
    // GPU: no reload, only a reset (room.js keeps them the same way)
    const keep = ai.engine && sameShard(ai.held, d) && ai.held.kv === kv;
    if (ai.loadingShard && ai.loadKey === `${d.model}:${d.range}`) { ai.next = d.next; ai.hostId = d.host || from; this.ensureLink(d.next); return; }
    if (ai.device && !keep) this.freeLayers(null);
    ai.role = "worker"; ai.next = d.next; ai.hostId = d.host || from; ai.q = Promise.resolve(); ai.startFailed = null;
    this.ensureLink(d.next);   // open the link to the chain neighbour while the weights load
    const t0 = performance.now();
    try {
      if (keep) {
        try { ai.engine.reset?.(); ai.engine.dropAllSlots?.(); } catch {}
        this.log(`back in the room: layers ${d.range[0]}-${d.range[1] - 1} are still loaded, no reload`);
      } else {
        this.log(`dealt layers ${d.range[0]}-${d.range[1] - 1} of ${d.model}; next: ${d.next}`);
        let lastPct = -1;
        ai.loadingShard = true; ai.loadKey = `${d.model}:${d.range}`;
        if (this.beforeLoad) await this.beforeLoad(d.model);
        if (ai.startFailed) throw new Error(ai.startFailed);
        const src = openModel(d.model, { modelDir: this.modelDir });
        const unwatch = this.watchLoad(src);
        try {
          const r = await loadShard({ modelKey: d.model, range: d.range, hasEmbed: false, hasHead: false, ctx: d.ctx || maxSeqFor(d.model),
            kv, src, flags: this.flags, selfTest: this.selfTest, log: this.log,
            onGpuError: (m) => { this.log("GPU error: " + m); this.sendTo(ai.hostId, { t: "ai-error", message: "GPU error: " + m.slice(0, 300) }); },
            onProgress: (done, total) => {
              if (ai.startFailed) throw new Error(ai.startFailed);
              const pct = Math.round(total ? (done / total) * 100 : 0);
              if (pct !== lastPct) { lastPct = pct; this.sendTo(ai.hostId, { t: "ai-progress", pct }); this.emit("loadprogress", pct); }
            } });
          Object.assign(ai, { engine: r.engine, device: r.device, cfg: r.cfg, range: d.range, model: d.model });
          ai.held = { model: d.model, range: [d.range[0], d.range[1]], ctx: d.ctx, kv };
        } finally { unwatch(); await src.close(); }
      }
      if (!(await this.ensureLink(d.next))) throw new Error("could not connect to the next device in the chain");
      this.log(`layers ${d.range[0]}-${d.range[1] - 1} ready in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
      // ckpt: this device applies checkpoint control (sv / ld / dp) on its frames
      this.sendTo(ai.hostId, { t: "ai-ready", slots: [], ckpt: ai.engine?.saveSlot ? 1 : 0 });
      this.emit("loaded", { range: d.range, model: d.model, s: (performance.now() - t0) / 1000 });
    } catch (err) {
      if (ai.startFailed) { this.freeLayers(null); return; }
      this.log("load failed: " + err.message);
      this.freeLayers(null);
      this.sendTo(ai.hostId, { t: "ai-error", message: err.message, load: 1 });
    } finally { ai.loadingShard = false; ai.loadKey = null; }
  }
  // A shard load's progress for a status line, ~4 times a second: "loadstat" { from: "Hugging Face" |
  // "disk", fetched, total, bps } (fetched: bytes received from the network, or read from the disk, of
  // this shard's total; bps: the rate over the last few seconds), also kept as node.loadStat.
  // src: source.js openModel(). -> stop() (emits the last one)
  watchLoad(src, { everyMs = 250, windowMs = 3000 } = {}) {
    const hist = [];
    const tick = () => {
      const st = src.stat;
      if (!st?.planned) return;
      const now = performance.now();
      hist.push([now, st.fetched]);
      while (hist.length > 2 && now - hist[0][0] > windowMs) hist.shift();
      const [t0, f0] = hist[0];
      const bps = now - t0 > 0 ? Math.max(0, ((st.fetched - f0) / (now - t0)) * 1000) : 0;
      this.loadStat = { from: st.from, fetched: Math.min(st.fetched, st.total || st.fetched), total: st.total, bps: Math.round(bps) };
      this.emit("loadstat", this.loadStat);
    };
    const t = setInterval(tick, everyMs);
    t.unref?.();
    return () => { clearInterval(t); tick(); };
  }
  // frames run one at a time in arrival order; control rides on the frame and goes on with it
  async workerFrame(d) {
    const ai = this.ai, E = ai.engine;
    if (!E) return;
    const ctl = {};
    if (d.rb != null) { E.restoreDN?.(d.rb); ctl.rb = d.rb; }
    if (d.sv != null) { E.saveSlot?.(d.sv); ctl.sv = d.sv; }
    if (d.dp != null) { for (const k of [].concat(d.dp)) k === DROP_ALL ? E.dropAllSlots?.() : E.dropSlot?.(k); ctl.dp = d.dp; }
    if (d.reset) { E.reset?.(); ctl.reset = 1; }
    if (d.ld != null) { E.loadSlot?.(d.ld); ctl.ld = d.ld; }
    const t0 = performance.now();
    if (d.t === "ai-hidden-b") {
      const xs = unpackWire(d), nTok = d.n || 4, wdim = E.dims.dim, NC = E.NC || 4;
      const hb = new Float32Array(nTok * wdim);
      for (let c = 0; c < nTok; c += NC) {
        const m = Math.min(NC, nTok - c);
        hb.set(await E.runHiddenBatch(xs.subarray(c * wdim, (c + m) * wdim), d.basePos + c, d.spec ? { base: c, total: nTok } : false), c * wdim);
      }
      if (badF32(hb)) this.sendTo(ai.hostId, { t: "ai-error", message: "NaN in batched prefill" });
      const bmsg = { basePos: d.basePos, n: nTok, ...(d.spec ? { spec: 1 } : {}), ...packWire(hb) };
      if (ai.next === "host") this.sendHidden(ai.hostId, { t: "ai-hiddenret-b", ...bmsg });
      else this.sendHidden(ai.next, { t: "ai-hidden-b", ...bmsg, ...ctl });
    } else {
      const h = await E.runHidden(unpackWire(d), d.pos);
      if (badF32(h)) this.sendTo(ai.hostId, { t: "ai-error", message: `NaN produced on node layers ${ai.range[0]}-${ai.range[1] - 1}` });
      const msg = { pos: d.pos, ...packWire(h) };
      if (ai.next === "host") this.sendHidden(ai.hostId, { t: "ai-hiddenret", ...msg });
      else this.sendHidden(ai.next, { t: "ai-hidden", ...msg, ...ctl });
    }
    this.frames = (this.frames || 0) + 1; this.frameMs = (this.frameMs || 0) + performance.now() - t0;
  }
  // the host link dropped: knock on the host id every 3 s for a minute (a host back from a lost
  // network, or its link redialed) and say hello {back: 1}; the host re-seats this device
  hostGone() {
    const ai = this.ai;
    this.failWaiters(new Error("the host left"));
    this.log("lost the link to the host");
    this.admission = null;   // back in through the gate (with the pass it was given) when it knocks
    if (this.roomClosed) return;   // the host said it closed the room: nothing to wait for
    this.emit("hostgone");
    if (this.closing || this.knock || this.otherHost) return;
    const hostId = PREFIX + this.code, t0 = Date.now();
    this.knock = setInterval(() => {
      if (this.closing || this.conns.has(hostId)) { clearInterval(this.knock); this.knock = null; return; }
      if (Date.now() - t0 > HOST_WAIT_MS) {
        clearInterval(this.knock); this.knock = null;
        ai.online = false; this.freeLayers(null);
        this.log("the host did not come back; the room is over");
        this.emit("roomover");
        return;
      }
      if (this.peer.disconnected) { try { this.peer.reconnect(); } catch {} return; }
      const conn = this.peer.connect(hostId, { reliable: true });
      if (!conn) return;
      guardChunks(conn);
      conn.on("open", () => {
        if (this.conns.has(hostId)) { try { conn.close(); } catch {} return; }
        clearInterval(this.knock); this.knock = null;
        this.wire(conn, "host", true);
        conn.send(this.helloMsg({ back: 1, ...joinHelloFields(this) }));
        ai.hostId = hostId;
        this.log("back in the room");
        this.emit("back");
      });
      conn.on("error", () => {});
    }, 3000);
  }

  // ---------------- host: dealing (room.js aiStart / aiMaybeReady / aiRejoin) ----------------
  start(modelKey = this.ai.model, opts = {}) {
    if (typeof modelKey === "object") { opts = modelKey; modelKey = this.ai.model; }
    if (this.ai.engine && this.startP) return this.startP;
    const p = this.startP = this._start(modelKey, opts);
    p.catch(() => { if (this.startP === p) this.startP = null; });
    return p;
  }
  // the devices the host deals layers to, and the plan (layer ranges by memory) for them:
  // pure, so it can be unit tested. peers: [{ id, name, meta }] (GPU devices, not API clients).
  // -> { chain: [id], ranges: [[lo, hi)], assigned, leftOut: [id] }; ranges[0] is this device's.
  // The room's context after every device's binding limit (meta.maxBindMB; a device that reports none,
  // e.g. an older tab, counts as WebGPU's 128 MiB): room/models.js ctxForBinding.
  static ctxForDevices(ggufMeta, ctx, kv, metas) {
    const bind = Math.min(...metas.map((m) => (m?.maxBindMB || 128) * 2 ** 20));
    return ctxForBinding(ggufMeta, ctx, kv, bind);
  }
  // The room's context for these pledges (room/models.js pickCtx, as the room page's aiStart): `want`
  // when the pledges hold the model there (roomBytes: weights + KV at that context, and what the host
  // holds besides), else the model's fallback when they hold it there (the 1.7B: 8k for 16k); an asked
  // --ctx stays. -> { ctx, want, fits, fellBack, note } (note: what the room says when it fell back)
  static ctxPick(modelKey, { want, ask = 0, kv = "f16", self, peers = [], shareCap = new Map() }) {
    const pl = [self, ...peers].map((d) => pledgeGB(d.meta, shareCap.get(d.name)) * 2 ** 30);
    const pick = pickCtx(modelKey, { want, ask, fitsAt: (c) => {
      const rb = roomBytes(modelKey, c, kv === "q8" ? "q8" : "f16");
      return !rb || roomFit(rb.L, pl, rb.layerBytes, rb.hostBytes).fits;
    } });
    return { ...pick, note: pick.fellBack ? ctxShortNote(String(MODELS[modelKey]?.label || modelKey).split("·")[0].trim(), pick.ctx, pick.want) : "" };
  }
  setSplit(mode) { this.splitMode = mode === "speed" ? "speed" : "memory"; }
  // fitBytes: room/models.js roomBytes() for the model at this context (weights + KV per layer, what the
  // host holds besides): the speed split fills each device by it, as the room page's roomFit does
  static dealPlan({ L, layerBytes, embedBytes, self, peers, shareCap = new Map(), mode = "memory", fitBytes = null }) {
    const pledgeOf = (m, name) => pledgeGB(m, shareCap.get(name)) * 2 ** 30;
    let chain = peers.map((p) => p.id);
    let caps = [Math.max(pledgeOf(self.meta, self.name) - embedBytes, layerBytes / 2), ...peers.map((p) => Math.max(pledgeOf(p.meta, p.name), layerBytes / 2))];
    // phones hold layers only when the computers cannot hold the model (room/plan.js)
    const out = phonesToLeaveOut(L, caps.map((c) => c / layerBytes), [false, ...peers.map((p) => isPhoneMeta(p.meta))]);
    const leftOut = out.map((i) => chain[i - 1]);
    if (out.length) { chain = chain.filter((id) => !leftOut.includes(id)); caps = caps.filter((_, i) => !out.includes(i)); }
    let { assigned, ranges } = planSplit(L, caps);
    if (mode === "speed") {
      // fill the host first, then the biggest devices, each up to its pledge (in whole layers); a device
      // not needed joins without layers. Short (the pledges hold less than L in whole layers): by memory
      const per = fitBytes?.layerBytes || layerBytes, hostB = fitBytes ? fitBytes.hostBytes : embedBytes;
      const allIds = [self, ...peers.filter((p) => chain.includes(p.id))];
      const layerCaps = allIds.map((d, i) => Math.max(0, (pledgeOf(d.meta, d.name) - (i === 0 ? hostB : 0)) / per));
      const sp = planForSpeed(L, layerCaps, [], layerCaps.map(() => false));
      if (!sp.short) {
        const used = sp.used.filter((i) => i > 0);
        leftOut.push(...chain.filter((_, k) => !used.includes(k + 1)));
        chain = used.map((i) => chain[i - 1]);
        assigned = sp.used.map((i) => sp.assigned[i]); ranges = sp.used.map((i) => sp.ranges[i]);
        caps = sp.used.map((i) => caps[i]);
      }
    }
    return { chain, ranges, assigned, leftOut, needGB: (L * layerBytes + embedBytes) / 2 ** 30, haveGB: caps.reduce((s, c) => s + c, embedBytes) / 2 ** 30 };
  }
  async _start(modelKey, { minDevices = 1, waitMs = 0, redeal = false } = {}) {
    const ai = this.ai;
    if (!this.hosting()) throw new Error("only the host deals layers");
    if (ai.engine && !redeal) return;
    const M = MODELS[modelKey];
    if (!M) throw new Error("unknown model " + modelKey);
    if (minDevices > 1) {   // wait for devices that hold layers
      const t0 = Date.now();
      while (this.gpuPeers().length + 1 < minDevices) {
        if (waitMs && Date.now() - t0 > waitMs) throw new Error(`only ${this.gpuPeers().length + 1} of ${minDevices} devices joined`);
        await new Promise((r) => setTimeout(r, 250));
      }
    }
    ai.starting = true; ai.degraded = false; ai.readyPeers = new Set(); ai.model = modelKey; ai.online = false;
    // a fresh deal: every device starts without checkpoints (a worker that keeps its layers drops its
    // slots); what each worker applies comes with its ai-ready, which may arrive before this device's load ends
    ai.ckpt?.clear(); ai.dropQ = []; ai.ckptCap = new Map();
    ai.relinks = new Map(); ai.gone = new Set(); ai.plan = new Map(); ai.lapStat = null;
    let ctx = nodeCtxFor(modelKey, this.ctxAsk);
    const kv = kvModeFor(modelKey, null);
    ai.apiCache = new AnswerCache(8); ai.apiTurns.clear(); ai.apiEnc.clear(); ai.apiProf = null; ai.apiTT = null; ai.bounds.clear();
    const src = openModel(modelKey, { modelDir: this.modelDir });
    let L, layerBytes, embedBytes;
    try {
      if (M.kind === "qwen35") {
        const G = await src.header(false);
        // one attention layer's K (or V) cache is a single GPU buffer: hold the context to what the
        // smallest binding limit among this room's devices fits (room.js aiStart does the same)
        const fit = RoomNode.ctxForDevices(G.meta, ctx, kv, [this.meta, ...this.gpuPeers().map((id) => this.conns.get(id)?.meta)]);
        if (fit < ctx) { this.log(`context ${ctx} needs bigger GPU buffers than a device here allows: using ${fit}`); ctx = fit; }
        L = G.meta["qwen35.block_count"] - (G.meta["qwen35.nextn_predict_layers"] || 0);
        layerBytes = qwen35ShardBytes(G, { lo: 0, hi: 4, hasEmbed: false, hasHead: false }) / 4 + ctx * kvBytesPerLayerPos(G.meta, kv);
        embedBytes = (G.tensors[GGML_EMBED]?.byteLength || 0) + (G.tensors[GGML_OUTPUT]?.byteLength || 0) + qwen35MtpBytes(G);
      } else {
        L = (await src.cfg()).num_hidden_layers;
        const G = await src.header(false);
        layerBytes = Object.values(ggmlLayerNames(0)).reduce((s, nm) => s + (G.tensors[nm]?.byteLength || 0), 0);
        embedBytes = (G.tensors[GGML_EMBED]?.byteLength || 0) + (G.tensors[GGML_OUTPUT]?.byteLength || 0);
      }
    } catch (err) { ai.starting = false; await src.close(); throw err; }
    const nameOf = (id) => this.conns.get(id)?.name || id;
    ai.dealtPeers = new Set(this.gpuPeers());   // every device this deal saw, left out or not (for a --devices host's re-deal)
    const peers = this.gpuPeers().sort().filter((id) => !ai.dropped.has(nameOf(id))).map((id) => ({ id, name: nameOf(id), meta: this.conns.get(id)?.meta }));
    // the model's default context, or its fallback (the 1.7B: 8k for 16k) when only that fits the
    // room's pledges (room/models.js pickCtx, the room page's rule); an asked --ctx stays as asked
    const pick = RoomNode.ctxPick(modelKey, { want: ctx, ask: this.ctxAsk, kv, self: { name: this.name, meta: this.meta }, peers, shareCap: ai.shareCap });
    ctx = pick.ctx;
    ai.ctxWant = pick.fellBack ? pick.want : 0;
    ai.ctxNote = pick.note;
    if (ai.ctxNote) this.log(ai.ctxNote);
    const plan = RoomNode.dealPlan({ L, layerBytes, embedBytes, self: { name: this.name, meta: this.meta }, peers, shareCap: ai.shareCap, mode: this.splitMode,
      fitBytes: roomBytes(modelKey, ctx, kv === "q8" ? "q8" : "f16") });
    const { ranges, assigned } = plan;
    ai.chain = plan.chain; ai.chainNames = ai.chain.map(nameOf); ai.layerGB = layerBytes / 2 ** 30;
    ai.layersN = Object.fromEntries([[this.name, assigned[0]], ...ai.chain.map((id, i) => [nameOf(id), assigned[i + 1]])]);
    if (plan.leftOut.length) this.log(`${plan.leftOut.map(nameOf).join(", ")} ask without holding layers: the other devices hold the whole model${this.splitMode === "speed" ? " (split: fastest first)" : ""}`);
    if (plan.needGB > plan.haveGB * 1.15) this.log(`this model needs ~${plan.needGB.toFixed(1)} GB but the room pledged ~${plan.haveGB.toFixed(1)} GB: it may not fit`);
    ai.layersByName = Object.fromEntries([[this.name, `${ranges[0][0]}–${ranges[0][1] - 1}`], ...ai.chain.map((id, i) => [nameOf(id), `${ranges[i + 1][0]}–${ranges[i + 1][1] - 1}`])]);
    this.log(`${M.label}: layer split ${[`${this.name} ${assigned[0]}+embed`, ...ai.chain.map((id, i) => `${nameOf(id)} ${assigned[i + 1]}`)].join(" · ")}`);
    this.split = { L, ranges, names: [this.name, ...ai.chain.map(nameOf)] };
    const readyAll = new Promise((res, rej) => { ai.startOk = res; ai.startErr = rej; });
    readyAll.catch(() => {});
    ai.chain.forEach((id, i) => {
      const msg = { t: "ai-load", v: PROTOCOL, model: modelKey, range: ranges[i + 1], ctx, kv, next: i + 1 < ai.chain.length ? ai.chain[i + 1] : "host", host: this.peer.id };
      ai.plan.set(nameOf(id), { msg });
      this.sendTo(id, msg);
    });
    this.broadcast({ t: "ai-layers", by: ai.layersByName });
    const t0 = performance.now();
    const keep = ai.engine && sameShard(ai.held, { model: modelKey, range: ranges[0], ctx }) && ai.held.kv === kv;
    let unwatch = () => {};
    try {
      if (!keep) {
        this.freeLayers("host");
        ai.loadingShard = true;
        let lastPct = -1;
        unwatch = this.watchLoad(src);
        const r = await loadShard({ modelKey, range: ranges[0], hasEmbed: true, hasHead: true, ctx, kv, src, flags: this.flags, selfTest: this.selfTest, log: this.log,
          onGpuError: (m) => this.log("GPU error: " + m),
          onProgress: (done, total) => { const pct = Math.round(total ? (done / total) * 100 : 0); if (pct !== lastPct) { lastPct = pct; this.emit("loadprogress", pct); } } });
        Object.assign(ai, { engine: r.engine, device: r.device, tok: r.tok, cfg: r.cfg, range: ranges[0], role: "host" });
        ai.held = { model: modelKey, range: [ranges[0][0], ranges[0][1]], ctx, kv };
      }
    } catch (err) { ai.starting = false; this.broadcast({ t: "ai-start-failed", why: err.message }); throw err; }
    finally { unwatch(); ai.loadingShard = false; await src.close(); }
    this.log(`host layers ${ranges[0][0]}-${ranges[0][1] - 1} + embedding/head ready in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    ai.fed = []; ai.pendingCtl = {}; ai.pos = 0;
    try { ai.engine?.dropAllSlots?.(); } catch {}   // this device's own slots (it may have kept its layers)
    this.maybeReady();
    try { await readyAll; }
    catch (err) { ai.starting = false; this.freeLayers(null); ai.chain = []; ai.plan = new Map(); this.broadcast({ t: "ai-start-failed", why: err.message }); throw err; }
    ai.starting = false;
  }
  // deal the layers again over the devices in the room now (after a departure, or to include late joiners)
  async redeal(why = "re-dealing the layers") {
    const ai = this.ai;
    if (!this.hosting()) throw new Error("only the host deals layers");
    this.log(why);
    clearTimeout(ai.idleRedeal);
    this.broadcast({ t: "ai-redeal", by: this.name, model: ai.model });
    ai.online = false; ai.fed = null;
    // this device keeps its own layers when its range is unchanged (_start's keep)
    const p = this.startP = this._start(ai.model, { redeal: true });
    p.catch(() => { if (this.startP === p) this.startP = null; });
    return p;
  }
  hosting() { return this.isHost || !!this.modelHost; }
  // lend another amount (GB), as the room page's stepper does: a device tells its host ("pledge"), a
  // host tells the room; the next deal uses it
  setPledge(gb) {
    const v = Math.min(64, Math.max(0.5, +gb || 0));
    this.pledgeGB = v;
    if (this.meta) this.meta.contribGB = v;
    if (this.isHost) { this.broadcast({ t: "pledge", gb: v }); this.emit("members"); }
    else if (this.ai.hostId && this.conns.has(this.ai.hostId)) this.sendTo(this.ai.hostId, { t: "pledge", gb: v });
    return v;
  }
  ctxMax() { return this.ai.engine?.maxSeq || nodeCtxFor(this.ai.model || "qwen3-1.7b", this.ctxAsk); }
  // v2 asks (tools): the loaded model's template profile and token texts (room.js apiProfile / apiTokenTexts)
  apiProfile() { const ai = this.ai; return ai.apiProf ||= templateProfile(ai.tok?.chatTemplate || "", ai.tok); }
  apiTokenTexts() { const ai = this.ai; return ai.apiTT ||= tokenTexts(ai.tok); }
  gpuPeers() { return [...this.conns.keys()].filter((id) => this.conns.get(id)?.meta?.webgpu && !this.conns.get(id)?.meta?.api); }
  relinking() {
    for (const [id, until] of this.ai.relinks) if (Date.now() > until || !this.ai.chain.includes(id)) this.ai.relinks.delete(id);
    return this.ai.relinks.size > 0;
  }
  whole() { const ai = this.ai; return !!ai.engine && !ai.degraded && !ai.loadingShard && ai.readyPeers.size >= ai.chain.length && ai.chain.every((id) => this.conns.has(id)) && !this.relinking(); }
  maybeReady() {
    const ai = this.ai;
    if (!this.hosting() || !ai.engine || ai.readyPeers.size < ai.chain.length || this.relinking()) return;
    ai.degraded = false; ai.online = true; ai.role = "host";
    clearTimeout(ai.idleRedeal);
    this.broadcast({ t: "ai-ready-all", model: ai.model, label: MODELS[ai.model]?.label, ctx: this.ctxMax(), ...(ai.ctxWant ? { ctxWant: ai.ctxWant } : {}) });
    this.log(`room online · ${ai.chain.length + 1} device(s), ${ai.cfg.num_hidden_layers} layers`);
    this.emit("online");
    ai.startOk?.();
  }
  // a device in the chain left (aiPeerLeft): laps in flight fail now, and the room waits for it to
  // come back into its slot, re-dealing without it after REJOIN_GRACE_MS when autoRedeal is on
  chainLeft(id, name, verb = "left") {
    const ai = this.ai;
    if (this.closing || !ai.chain.includes(id) || ai.gone.has(id)) return;
    const layers = ai.layersByName?.[name];
    const why = `${name} ${verb}${layers ? ` (layers ${layers})` : ""}`;
    if (ai.starting) { ai.startErr?.(new Error(why)); return; }
    ai.degraded = true; ai.online = false; ai.fed = null; ai.readyPeers.delete(id); ai.gone.add(id);
    this.ckptClear(true);   // frames in flight (and the saves on them) are gone: no checkpoint is known good
    this.failWaiters(new Error(why));
    this.broadcast({ t: "ai-degraded", why: `${why}: waiting for it to come back` });
    this.emit("degraded", why);
    clearTimeout(ai.idleRedeal);
    if (this.autoRedeal && !ai.starting) ai.idleRedeal = setTimeout(() => {
      if (!ai.degraded || ai.busy || ai.loadingShard || !this.missingNames().length) return;
      this.redeal(`${this.missingNames().join(", ")} did not come back in ${Math.round(REJOIN_GRACE_MS / 1000)} s: re-dealing the layers`).catch((err) => this.log("re-deal failed: " + err.message));
    }, REJOIN_GRACE_MS);
    ai.idleRedeal?.unref?.();
  }
  missingNames() { const ai = this.ai; return ai.chain.map((id, i) => (this.conns.has(id) ? null : ai.chainNames[i] || id)).filter(Boolean); }
  // a device that left comes back to its slot (a reloaded tab with a new peer id, or a phone back
  // from a lock under the same one): a fresh ai-load for its old slot, and ai-next {relink} to the
  // device before it (whose answer, ai-linked, the room waits for)
  rejoin(newId, name) {
    const ai = this.ai;
    if (!ai.plan.has(name)) return;
    const i = ai.chainNames.indexOf(name);
    if (i < 0) return;
    if (ai.chain[i] === newId ? !ai.gone.has(newId) : ai.chain.includes(newId)) return;
    const oldId = ai.chain[i];
    ai.chain[i] = newId;
    ai.readyPeers.delete(oldId);
    ai.gone.delete(oldId); ai.gone.delete(newId);
    clearTimeout(ai.idleRedeal);
    const { msg } = ai.plan.get(name);
    const fresh = { ...msg, next: i + 1 < ai.chain.length ? ai.chain[i + 1] : "host", host: this.peer.id };
    if (i > 0) { this.sendTo(ai.chain[i - 1], { t: "ai-next", next: newId, relink: 1 }); ai.relinks.set(newId, Date.now() + RELINK_MS); setTimeout(() => this.maybeReady(), RELINK_MS + 100).unref?.(); }
    this.sendTo(newId, fresh);
    ai.fed = null;
    this.ckptClear(true);
    this.log(`${name} came back into its slot`);
  }
  // a chain device's tab was killed while it loaded its layers (hello died.loading): re-deal with a
  // smaller share for it, or without it (room/pledge.js afterLoadDeath). -> true when handled
  loadDeath(newId, d) {
    const ai = this.ai, name = d.name;
    if (!d.died?.loading || !ai.plan.has(name) || !ai.chainNames.includes(name)) return false;
    const deaths = (ai.loadDeaths.get(name) || 0) + 1;
    ai.loadDeaths.set(name, deaths);
    const r = afterLoadDeath({ layers: ai.layersN?.[name] || 1, layerGB: ai.layerGB || 0.5, gb: pledgeGB(this.conns.get(newId)?.meta, ai.shareCap.get(name)), deaths });
    if (r.drop) { ai.dropped.add(name); this.sendTo(newId, { t: "ai-share", drop: true, why: "This device's browser closed the tab while it loaded its layers, so the room runs without it. You can still ask questions." }); }
    else { ai.shareCap.set(name, r.gb); this.sendTo(newId, { t: "ai-share", gb: r.gb, why: `This device's browser closed the tab while it loaded its layers, so it now holds less of the model (${r.gb} GB).` }); }
    setTimeout(() => this.redeal(`${name}'s tab was killed while it loaded its layers: re-dealing ${r.drop ? "without it" : `with ${r.gb} GB for it`}`).catch((err) => this.log("re-deal failed: " + err.message)), 500);
    return true;
  }

  // ---------------- host: laps (room.js lapWait / sendChain / aiPipeToken / aiPrefill) ----------------
  lapWait(key, ms, what) {
    return new Promise((res, rej) => {
      const timer = setTimeout(() => { this.ai.waiters.delete(key); rej(new Error(`pipeline timeout (${what}): a device in the room stopped answering`)); }, ms);
      this.ai.waiters.set(key, { res: (h) => { clearTimeout(timer); res(h); }, rej: (e) => { clearTimeout(timer); rej(e); } });
    });
  }
  failWaiters(err) { for (const [k, w] of this.ai.waiters) { this.ai.waiters.delete(k); w.rej(err); } }
  lapDone(key, h) { const w = this.ai.waiters.get(key); if (w) { this.ai.waiters.delete(key); w.res(h); } }
  chainRtt() { return Math.max(0, ...this.ai.chain.map((id) => this.conns.get(id)?.rtt || 0)); }
  noteLap(lapMs, hostMs) {
    const L = this.ai.lapStat ||= { lap: 0, host: 0, n: 0, max: 0 };
    L.lap = L.n ? 0.7 * L.lap + 0.3 * lapMs : lapMs;
    L.max = Math.max(lapMs, 0.98 * (L.max || 0));
    L.host = L.n ? 0.7 * L.host + 0.3 * hostMs : hostMs;
    L.n++;
  }
  // a decode lap starts: workers that asked for it (phones: hello meta.wake) wake their GPU now
  wakeChain(pos) { for (const id of this.ai.chain) if (this.conns.get(id)?.meta?.wake) this.sendTo(id, { t: "ai-wake", pos }); }
  // a pending reset, rollback or checkpoint control rides with the frame, so it reaches every device
  // strictly before the frame it applies to. A frame header carries at most two drops (room/transport.js
  // packCkpt): evictions wait in dropQ and go out two per frame (a slot waiting to be dropped only
  // costs a worker memory for a few frames longer; slot numbers are never reused before it is gone).
  sendChain(msg) {
    const ai = this.ai, ctl = ai.pendingCtl; ai.pendingCtl = {};
    if (ai.dropQ.length) {
      const dp = [].concat(ctl.dp ?? []);
      if (dp.includes(DROP_ALL)) ai.dropQ = [];
      else { while (dp.length < 2 && ai.dropQ.length) dp.push(ai.dropQ.shift()); ctl.dp = dp; }
    }
    ai.frames = (ai.frames || 0) + 1;
    this.sendHidden(ai.chain[0], { ...msg, ...ctl });
  }
  // forget the conversation state: here now, on the chain with the next frame. A pending rollback
  // and checkpoint save / drops still go out first (the save records the last answer's end state)
  resetState() {
    const ai = this.ai;
    try { ai.engine.reset?.(); } catch {}
    ai.pos = 0; ai.fed = [];
    const { rb, sv, dp } = ai.pendingCtl || {};
    ai.pendingCtl = ai.chain.length ? { ...(rb != null ? { rb } : {}), ...(sv != null ? { sv } : {}), ...(dp != null ? { dp } : {}), reset: 1 } : {};
  }

  // ---------------- host: checkpoints (room.js ckptSave / ckptResume, ckpt.js) ----------------
  // on for this room: the host's engine keeps GPU slots and every chain device applies the frames'
  // checkpoint control (every qwen35 engine does; a dense-model tab says so in its ai-ready)
  ckptOn() {
    const ai = this.ai;
    if (!this.ckptOpts || !ai.engine?.saveSlot || !this.hosting()) return false;
    return MODELS[ai.model]?.kind === "qwen35" || ai.chain.every((id) => ai.ckptCap.get(id));
  }
  // forget every checkpoint (a device dropped, frames were lost, the engines were rebuilt).
  // tellChain: the chain drops its copies with the next frame
  ckptClear(tellChain = false) {
    const ai = this.ai;
    const keys = ai.ckpt ? ai.ckpt.clear() : [];
    for (const k of keys) { try { ai.engine?.dropSlot?.(k); } catch {} }
    ai.dropQ = [];
    if (tellChain && keys.length && ai.chain.length) {
      const { sv, ...rest } = ai.pendingCtl || {};
      ai.pendingCtl = { ...rest, dp: [DROP_ALL] };
    }
  }
  // Save the state the caches hold (ai.fed) as a checkpoint on every device: pinned (a prompt's
  // fixed start) or an answer's end. The host saves now; the chain saves with the next frame (sv).
  // -> the slot number, or null when none was saved
  ckptSave(pin = false, turn = false) {
    const ai = this.ai, E = ai.engine;
    if (!this.ckptOn() || !ai.fed?.length) return null;
    ai.ckpt ||= new CkptIndex(this.ckptOpts);
    // a worker applies sv before dp: a save riding with DROP_ALL would be gone at once on every worker
    if (ai.chain.length && [].concat(ai.pendingCtl?.dp ?? []).includes(DROP_ALL)) return null;
    // a save still waiting for its frame never reached the chain: the header carries one save, so
    // this one supersedes it (a pinned one at the same tokens stays pinned)
    const prev = ai.chain.length ? ai.pendingCtl?.sv : null;
    if (prev != null) {
      const p = ai.ckpt.find(prev);
      if (p?.pin && p.ids.length === ai.fed.length && isPrefix(p.ids, ai.fed)) pin = true;
      ai.ckpt.remove(prev); try { E.dropSlot(prev); } catch {}
      const { sv, ...rest } = ai.pendingCtl; ai.pendingCtl = rest;
    }
    const plan = ai.ckpt.plan(ai.fed, { pin });
    if (plan.skip) return plan.key;
    for (const k of plan.drop) { ai.ckpt.remove(k); try { E.dropSlot(k); } catch {} }
    if (ai.chain.length) ai.dropQ.push(...plan.drop);
    E.pos = ai.pos;
    E.saveSlot(plan.key);
    ai.ckpt.commit(plan.key, ai.fed.slice(), pin, { xAt: ai.xAt === ai.pos, turn: !pin && turn });
    if (ai.chain.length) ai.pendingCtl = { ...ai.pendingCtl, sv: plan.key };
    return plan.key;
  }
  // resume from the longest checkpoint that is a prefix of ids, if it beats what the caches hold
  // (`reused`). -> { reused, from: "pin" | "answer" | null }
  ckptResume(ids, reused) {
    const ai = this.ai;
    if (!this.ckptOn() || !ai.ckpt?.size) return { reused, from: null };
    const x = ai.ckpt.best(ids, reused);
    if (!x) return { reused, from: null };
    ai.engine.loadSlot(x.key);
    ai.pos = x.ids.length; ai.fed = ids.slice(0, ai.pos);
    ai.xAt = x.xAt ? ai.pos : null;   // the draft head's hidden is in the slot only after a speculative answer
    if (ai.chain.length) { const { reset, ...rest } = ai.pendingCtl || {}; ai.pendingCtl = { ...rest, ld: x.key }; }
    return { reused: ai.pos, from: x.pin ? "pin" : x.turn ? "turn" : "answer" };
  }
  // the id of "user" right after <|im_start|> (one token in the Qwen vocabularies), or null
  userTok() {
    const ai = this.ai;
    if (ai.userTokFor !== ai.tok) { const e = ai.tok.encode("user"); ai.userTok = e.length === 1 ? e[0] : null; ai.userTokFor = ai.tok; }
    return ai.userTok;
  }
  // POOLED_CKPT_DEBUG: where a prompt leaves each checkpoint (the text around the first differing token)
  ckptDebug(req, prompt) {
    const ai = this.ai, ids = prompt.ids;
    const asst = req.messages.filter((m) => m.role === "assistant").length;
    const lines = [`ckpt debug: prompt ${ids.length} tok, ${asst} assistant turns, exact ${prompt.exact ?? "?"}`];
    for (const x of ai.ckpt.items) {
      let n = 0; const m = Math.min(x.ids.length, ids.length);
      while (n < m && x.ids[n] === ids[n]) n++;
      const dec = (a) => JSON.stringify(ai.tok.decode(Array.from(a.slice(Math.max(0, n - 12), n + 12))));
      lines.push(`  ${x.pin ? "pin" : "answer"} ${x.ids.length} tok: shares ${n}${n === x.ids.length ? " (prefix)" : `; ckpt ${dec(x.ids)} vs prompt ${dec(ids)}`}`);
    }
    this.log(lines.join("\n"));
    if (process.env.POOLED_CKPT_DEBUG.endsWith(".jsonl")) {
      try { fs.appendFileSync(process.env.POOLED_CKPT_DEBUG, JSON.stringify({ at: Date.now(), n: ids.length, exact: prompt.exact, messages: req.messages.map((m) => ({ role: m.role, text: String(m.text || "").slice(0, 400), textLen: String(m.text || "").length, calls: m.calls, reasoning: m.reasoning ? String(m.reasoning).slice(0, 200) : undefined, reasoningLen: m.reasoning?.length })) }) + "\n"); } catch {}
    }
  }
  // the pinned points of a v2 prompt (ckpt.js pinPoints): its system prompt + tools, and an agent's
  // cache boundary (memoized per system text and tool set: it renders the prompt's start once more)
  pinsFor(req, prompt) {
    const ai = this.ai, o = this.ckptOpts;
    if (!o || !prompt?.ids) return [];
    const key = JSON.stringify([!!prompt.thinking, req.params?.effort || "", req.params?.toolChoice === "none", req.system || "", req.tools || null]);
    let b = ai.bounds.get(key);
    if (b == null) {
      try { b = boundaryPin(ai.tok, req, prompt, { encode: (s) => ai.apiEnc.encode(ai.tok, s), minPin: o.minPin }); } catch { b = 0; }
      ai.bounds.set(key, b);
      if (ai.bounds.size > 16) ai.bounds.delete(ai.bounds.keys().next().value);
    }
    return pinPoints(prompt, { boundary: b, minPin: o.minPin });
  }
  fillDrafts(h, ids, i0, basePos, n) {
    const E = this.ai.engine;
    if (!E?.mtp) return;
    const dim = E.dims.dim;
    if (E.mtpBatchFill !== false && E._mtpFillBatch && E.B && n > 1 && n <= (E.NC || 4)) {
      for (let c = 0; c < n; c++) E.device.queue.writeBuffer(E.B.x.buf, c * E.B.x.stride, h.subarray(c * dim, (c + 1) * dim));
      E._mtpFillBatch(ids, i0, basePos, n);
      return;
    }
    for (let c = 0; c < n; c++) {
      const next = ids[i0 + c + 1];
      if (next === undefined) break;
      E.setHidden(h.subarray(c * dim, (c + 1) * dim));
      E.mtpRun(null, next, basePos + c + 1, false);
    }
  }
  // ahead (plain greedy decode in a chain, engine headAhead): { h, t0, defer, onSent }, as room.js
  async pipeToken(id, needLogits = true, fillNext, desc = null, ahead = null) {
    const ai = this.ai, E = ai.engine, pos = ai.pos;
    if (!ai.chain.length && !needLogits) {
      E.pos = pos; await E.prefillToken(id);
      if (pos % 8 === 7) await ai.device.queue.onSubmittedWorkDone();
      ai.pos++; ai.fed?.push(id);
      return null;
    }
    const tHost = ahead?.t0 ?? performance.now();
    if (needLogits) this.wakeChain(pos);
    let h = ahead?.h || await E.embedRun(id, pos);
    if (badF32(h)) throw new Error(`NaN after host layers (pos ${pos})`);
    if (ai.chain.length) {
      const hostMs = performance.now() - tHost;
      const returned = this.lapWait(pos, lapTimeout(ai.lapStat, 30000, this.chainRtt()), "token");
      this.sendChain({ t: "ai-hidden", pos, ...packWire(h) });
      ahead?.onSent?.();
      h = await returned;
      if (badF32(h)) throw new Error(`NaN in hidden returned by peers (pos ${pos})`);
      this.noteLap(performance.now() - tHost, hostMs);
      ai.lastHidden = h;
      if (!needLogits && fillNext !== undefined) this.fillDrafts(h, [id, fillNext], 0, pos, 1);
    }
    ai.pos++; ai.fed?.push(id);
    if (!needLogits || ahead?.defer) return null;
    if (desc) { const c = await E.headFromHiddenIds(h, desc); if (c.bad) throw new Error(`NaN in logits (pos ${ai.pos})`); return c; }
    const logits = await E.headFromHidden(h);
    if (badF32(logits)) throw new Error(`NaN in logits (pos ${ai.pos})`);
    return logits;
  }
  async prefill(ids, { aborted, desc }) {
    const ai = this.ai, E = ai.engine;
    if (!ai.chain.length && E.prefillTokens && ids.length > 1) {
      E.pos = ai.pos; await E.prefillTokens(ids.slice(0, -1));
      ai.pos = E.pos; ai.fed.push(...ids.slice(0, -1));
      return this.pipeToken(ids[ids.length - 1], true, undefined, desc);
    }
    let i = 0, tailLogits = null;
    const flex = !!(ai.chain.length && E.specStep && E.embedRunBatch);
    if (E.embedRunBatch && (ids.length > 5 || flex)) {
      const hdim = E.dims.dim, NC = E.NC || 4;
      const widths = [NC, ...[8, 4].filter((w) => w < NC)];
      const inflight = [];
      try {
        outer: for (const W of widths) while (ids.length - 1 - i >= W) {
          if (aborted()) break outer;
          const nChunks = Math.max(1, Math.min(Math.floor(16 / W), Math.floor((ids.length - 1 - i) / W)));
          const n = nChunks * W, basePos = ai.pos, i0 = i;
          const hb = new Float32Array(n * hdim);
          for (let c = 0; c < nChunks; c++) hb.set(await E.embedRunBatch(ids.slice(i + c * W, i + (c + 1) * W), basePos + c * W), c * W * hdim);
          if (badF32(hb)) throw new Error(`NaN in batched prefill (pos ${basePos})`);
          if (ai.chain.length) {
            while (inflight.length >= PREFILL_WINDOW) await inflight.shift();
            const p = this.lapWait("b" + basePos, 90000, "batch prefill").then((h) => this.fillDrafts(h, ids, i0, basePos, n));
            p.catch(() => {});
            inflight.push(p);
            this.sendChain({ t: "ai-hidden-b", basePos, n, ...packWire(hb) });
          }
          ai.pos = basePos + n; for (let k = i0; k < i0 + n; k++) ai.fed.push(ids[k]); i += n;
        }
        if (flex && !aborted() && i < ids.length) {
          const n = ids.length - i, basePos = ai.pos, i0 = i;
          const hb = await E.embedRunBatch(ids.slice(i), basePos);
          if (badF32(hb)) throw new Error(`NaN in batched prefill (pos ${basePos})`);
          const p = this.lapWait("b" + basePos, 90000, "prefill tail");
          p.catch(() => {});
          this.sendChain({ t: "ai-hidden-b", basePos, n, ...packWire(hb) });
          ai.pos = basePos + n; for (let k = i0; k < ids.length; k++) ai.fed.push(ids[k]); i = ids.length;
          for (const q of inflight) await q;
          const h = await p;
          if (badF32(h)) throw new Error(`NaN in hidden returned by peers (pos ${basePos})`);
          this.fillDrafts(h, ids, i0, basePos, n);
          const dim = E.dims.dim;
          ai.lastHidden = h.slice((n - 1) * dim, n * dim);
          tailLogits = desc ? await E.headFromHiddenIds(ai.lastHidden, desc) : await E.headFromHidden(ai.lastHidden);
        }
        for (const p of inflight) await p;
      } catch (err) { this.failWaiters(err); throw err; }
    }
    if (aborted()) return null;
    if (tailLogits) return tailLogits;
    let logits = null;
    for (; i < ids.length; i++) {
      if (aborted()) return null;
      logits = await this.pipeToken(ids[i], i === ids.length - 1, ids[i + 1], desc);
    }
    return logits;
  }

  // room.js roomGenerate: a device in the chain that drops mid-answer does not fail it; the answer
  // waits for the room to be whole again (room/resume.js) and carries on from the last token
  async generate(ids, opts = {}) {
    const aborted = () => !!opts.signal?.aborted;
    const r = await resumableGenerate((x, o) => this.generateOnce(x, o), ids, { maxNew: MAX_NEW, ...opts }, {
      aborted,
      recover: async ({ err }) => {
        if (!this.hosting()) throw err;
        this.log(`the answer is waiting: ${err.message}`);
        await waitForRoom({ ready: () => this.whole(), gone: () => this.missingNames(), aborted,
          redeal: () => this.redeal(`${this.missingNames().join(", ")} did not come back: re-dealing the layers`), autoRedeal: () => this.autoRedeal,
          status: (s) => this.log(s) });
      },
      onResume: ({ emitted }) => this.log(emitted ? `the room is whole again: carrying on after ${emitted} tokens` : "the room is whole again: starting over"),
    });
    return r.resumed ? { ...r, stats: r.stats + ` · carried on after ${r.resumed > 1 ? r.resumed + " drops" : "a device dropped"}` } : r;
  }
  // pins: where the prompt's fixed start ends (ckpt.js pinPoints); the prefill pauses there to save a
  // pinned checkpoint when the caches do not hold it yet
  async generateOnce(ids, { onToken = () => {}, stop, maxNew = MAX_NEW, sample = pickSampler(this.ai.settings.sampling), signal, spec: useSpec = true, pins = [], turn = 0 } = {}) {
    const ai = this.ai, E = ai.engine;
    if (!E) throw new Error("the model is not loaded");
    if (ai.degraded) throw new Error("a device left: re-deal the layers first");
    const ctxMax = E.maxSeq;
    const aborted = () => !!signal?.aborted;
    const eos = (t) => stop.has(t);
    const tokens = [];
    let count = 0, capped = false, acc = null, copied = 0, tPre = 0, tDecode = 0, reused = 0, prefilled = 0, from = null, pinned = 0, turned = 0;
    const desc = E.gpuDescFor?.(sample) || null;
    try {
      reused = reusablePrefix(ai.fed, ids);
      if (reused) from = "live";
      const r0 = this.ckptResume(ids, reused);
      if (r0.from) { reused = r0.reused; from = r0.from; }
      if (!reused) this.resetState();
      const rest = ids.slice(reused);
      if (reused && rest.length && E.mtp && ai.xAt === ai.pos) E.mtpRun(null, rest[0], ai.pos, false);
      ai.xAt = null;
      prefilled = rest.length;
      maxNew = Math.min(maxNew, ctxMax - ids.length);
      const t0Pre = performance.now(); ai.frames = 0;
      // the fixed start first, a pinned checkpoint at each of its pins, then the rest (the same tokens
      // at the same positions, so the answer is the same; the head's logits after each part are unused)
      // and a turn checkpoint (not pinned) where the last user turn starts
      const cuts = this.ckptOn() ? cutPoints(reused, turn ? [...pins, turn] : pins, ids.length) : [];
      let at = reused, logits = null;
      for (const c of cuts) {
        await this.prefill(ids.slice(at, c), { aborted, desc });
        if (aborted()) break;
        const pin = pins.includes(c);
        if (ai.fed?.length === c && this.ckptSave(pin, !pin) != null) { if (pin) pinned++; else turned++; }
        at = c;
      }
      if (!aborted() && at < ids.length) logits = await this.prefill(ids.slice(at), { aborted, desc });
      tPre = performance.now() - t0Pre;
      const t0 = performance.now();
      const emit = (tok, drafted) => { tokens.push(tok); count++; onToken(tok, drafted); };
      // a verify lap round the chain: every column's hidden through every device (ai-hidden-b {spec})
      const chainSpec = () => (ai.chain.length ? {
        // pre: { hs, t0 } when the engine already ran the host's layers with the drafts (hostFuse)
        runTrunk: async (toks, pos, pre = null) => {
          const tLap = pre?.t0 ?? performance.now();
          this.wakeChain(pos);
          const n = toks.length, hdim = E.dims.dim, NC = E.NC || 4;
          const hb = pre?.hs || new Float32Array(n * hdim);
          if (!pre) for (let c = 0; c < n; c += NC) { const m = Math.min(NC, n - c); hb.set(await E.embedRunBatch(toks.slice(c, c + m), pos + c, { base: c, total: n }), c * hdim); }
          if (badF32(hb)) throw new Error(`NaN after host layers (pos ${pos})`);
          const hostMs = performance.now() - tLap;
          const returned = this.lapWait("b" + pos, lapTimeout(ai.lapStat, 90000, this.chainRtt()), "verify");
          this.sendChain({ t: "ai-hidden-b", basePos: pos, n, spec: 1, ...packWire(hb) });
          const h = await returned;
          if (badF32(h)) throw new Error(`NaN in hidden returned by peers (pos ${pos})`);
          this.noteLap(performance.now() - tLap, hostMs);
          return h;
        },
        onReject: async (k) => { ai.pendingCtl = { rb: k }; },
        preTrunk: true,
      } : {});
      if (!logits) { /* stopped during prefill */ }
      else if (useSpec && E.mtp && E.specStep) {
        const spec = chainSpec();
        if (ai.chain.length && ai.lastHidden) E.setHidden(ai.lastHidden);
        E.pos = ai.pos;
        const kc = { cand: [3, 5, 7], ema: {}, n: {}, step: 0 };
        const pickK = () => {
          if (!ai.chain.length) return 3;
          kc.step++;
          if (kc.step <= 3) return 3;
          const untried = kc.cand.find((k) => !kc.n[k]);
          if (untried) return untried;
          let best = 3;
          for (const k of kc.cand) if (kc.ema[k] > kc.ema[best]) best = k;
          if (kc.step % 16 === 0) { const alt = kc.cand.filter((k) => k !== best); return alt[(kc.step / 16) % alt.length | 0]; }
          return best;
        };
        const st0 = { ...E.mtp.stats };
        let next = sample(logits), done = false, lkFull = false;
        if (eos(next)) done = true; else emit(next, 0);
        while (!done && count < maxNew && !aborted()) {
          let K = pickK();
          const roomLeft = ctxMax - E.pos - 2;
          if (roomLeft < 1) { capped = true; break; }
          K = Math.min(K, roomLeft, maxNew - count);
          const tStep = performance.now();
          const lkMax = lkFull ? (E.maxDrafts || 7) : 7;
          const lk = E.specStepDrafts && ai.fed ? lookupDrafts([...ai.fed, next], Math.min(lkMax, roomLeft, maxNew - count)) : [];
          const viaLookup = lk.length >= 2;
          const toks = viaLookup ? await E.specStepDrafts(next, sample, lk, spec) : await E.specStep(next, sample, K, spec);
          if (viaLookup) copied += toks.length - 1;
          lkFull = viaLookup && toks.length === lk.length + 1;
          ai.fed.push(next, ...toks.slice(0, -1));
          const tps = toks.length / ((performance.now() - tStep) / 1000);
          if (!viaLookup) { kc.ema[K] = kc.n[K] ? 0.6 * kc.ema[K] + 0.4 * tps : tps; kc.n[K] = (kc.n[K] || 0) + 1; }
          for (let j = 0; j < toks.length; j++) {
            const tk = toks[j];
            if (eos(tk)) { done = true; break; }
            if (count >= maxNew) { done = true; capped = true; break; }
            emit(tk, j < toks.length - 1 ? (viaLookup ? 2 : 1) : 0);
          }
          next = toks[toks.length - 1];
          const dd = E.mtp.stats.drafts - st0.drafts;
          acc = dd ? (E.mtp.stats.accepted - st0.accepted) / dd : null;
        }
        if (!done && count >= maxNew) capped = true;
        ai.pos = E.pos; ai.xAt = ai.pos;
      }
      else if (useSpec && !E.mtp && E.specStepDrafts && !E.specStep && ai.chain.length) {
        // a model without a draft head (the dense Qwen3s) in a split room, as the room page (#278):
        // when prompt lookup finds the text repeating the context, the tokens that followed it go
        // round as drafts in one lap (specStepDrafts: the same output as plain decoding), only while
        // every device in the chain handles dense verify frames (its hello's dspec). No drafts: a plain lap.
        const spec = chainSpec();
        const st0 = { ...(E.specStats || { drafts: 0, accepted: 0 }) };
        let next = sample(logits), done = false, lkFull = false;
        if (eos(next)) done = true; else emit(next, 0);
        while (!done && count < maxNew && !aborted()) {
          const roomLeft = ctxMax - ai.pos - 2;
          if (roomLeft < 0) { capped = true; break; }
          const kMax = Math.min(E.maxDrafts || 7, roomLeft, maxNew - count);
          const lk = ai.fed ? denseLookupDrafts(ai.chain.map((id) => this.conns.get(id)?.meta), [...ai.fed, next], kMax, { full: lkFull }) : [];
          let toks;
          if (lk.length) {
            E.pos = ai.pos;
            toks = await E.specStepDrafts(next, sample, lk, spec);
            ai.fed.push(next, ...toks.slice(0, -1));
            ai.pos = E.pos; ai.lastHidden = E.lastHidden;
            copied += toks.length - 1;
            lkFull = toks.length === lk.length + 1;
          } else {
            lkFull = false;
            const lg = await this.pipeToken(next, true, undefined, desc);
            toks = [sample(lg)];
          }
          for (let j = 0; j < toks.length; j++) {
            const tk = toks[j];
            if (eos(tk)) { done = true; break; }
            if (count >= maxNew) { done = true; capped = true; break; }
            emit(tk, j < toks.length - 1 ? 2 : 0);
          }
          next = toks[toks.length - 1];
          const st = E.specStats, d = st ? st.drafts - st0.drafts : 0;
          acc = d ? (st.accepted - st0.accepted) / d : null;
        }
        if (!done && count >= maxNew) capped = true;
        ai.xAt = ai.pos;
      }
      else {
        // plain decoding. ahead (greedy GPU sampling in a chain, host fuse): from the second lap on,
        // the head of the returned hidden and the host's layers on its pick are one submit
        // (engine headAhead); a pick that is not piped has its layers undone (dropAhead)
        const ahead = ai.chain.length > 0 && desc?.kind === "greedy" && !!E.canHeadAhead?.();
        let deferred = false;
        try {
          for (let i = 0; i < maxNew && !aborted(); i++) {
            let next, pre = null;
            const tLap = performance.now();
            if (!deferred) next = sample(logits);
            else {
              const r = await E.headAhead(ai.lastHidden, ai.pos, desc);
              if (r.cands.bad) throw new Error(`NaN in logits (pos ${ai.pos})`);
              logits = r.cands; deferred = false;
              next = sample(r.cands);
              if (r.h && next === r.cands.ids[0]) pre = r.h;
            }
            if (eos(next)) break;
            if (ai.pos >= ctxMax - 1) { emit(next, 0); capped = true; break; }
            if (ahead) {
              if (pre) E.keepAhead(); else E.dropAhead();
              logits = null; deferred = true;
              await this.pipeToken(next, true, undefined, desc, { h: pre, t0: tLap, defer: true, onSent: () => emit(next, 0) });
            } else {
              emit(next, 0);
              logits = await this.pipeToken(next, true, undefined, desc);
            }
          }
        } finally { if (ahead) E.dropAhead(); }
        if (count >= maxNew) capped = true;
      }
      tDecode = performance.now() - t0;
    } catch (err) {
      // the chain's state is unknown (a lap failed, a frame may be lost with a save on it): start
      // over, and forget every checkpoint here and on the chain
      ai.fed = null; ai.pendingCtl = {};
      this.ckptClear(true);
      throw err;
    }
    this.ckptSave();   // this answer's end state, on every device, for the next turn or a retry
    const tps = count / Math.max(tDecode / 1000, 1e-3);
    const full = capped && ai.pos >= ctxMax - 2;
    const stats = `${count} tok · ${tps.toFixed(1)} tok/s · ${ai.chain.length + 1} device${ai.chain.length ? "s" : ""}`
      + (acc != null ? ` · ${Math.round(acc * 100)}% drafts accepted` : "") + (copied ? ` · ${copied} tok by lookup` : "")
      + ` · prompt ${ids.length} tok: ${prefilled} read in ${(tPre / 1000).toFixed(1)} s` + (reused ? `, ${reused} from ${from === "pin" ? "a pinned checkpoint" : from === "answer" ? "an earlier answer" : from === "turn" ? "an earlier turn" : "the caches"}` : "");
    const reason = aborted() ? "abort" : capped ? (full ? "ctx" : "max") : "stop";
    this.emit("prefill", { total: ids.length, reused, from, prefilled, pinned, tPre, tDecode, count });
    return { tokens, reason, reused, prefilled, from, pinned, count, tps, acc, copied, tPre, tDecode, stats, capped };
  }
  // one generation at a time (room.js ai.busy + ai.queue, as a promise chain)
  locked(fn) {
    const p = this.ai.lock.then(async () => { this.ai.busy = true; try { return await fn(); } finally { this.ai.busy = false; } });
    this.ai.lock = p.catch(() => {});
    return p;
  }

  // ---------------- host: asks ----------------
  // What the host tells API clients in its hello: { api: 2, ctx } (docs/protocol.md "API clients").
  get hostMeta() { return { api: 2, ctx: this.ctxMax() }; }
  // One API ask, in process: exactly what a `pooled serve` bridge sends over WebRTC (cli/lib/common.js
  // askBody: { api?, system, messages, tools?, params }), answered through the same host path
  // (validateApiAsk, apiRun / apiRun2). handler(msg) gets ai-genstart / ai-token / ai-call /
  // ai-gendone / ai-busy for this rid, as Bridge.ask's handler does. -> { rid, stop() }
  request(body, handler, { rid = "n" + randCode(10) } = {}) {
    const ac = new AbortController();
    const d = { t: "ai-ask", api: 1, rid, ...body };
    const self = this.peer?.id || "self";
    const go = async () => {
      if (!this.hosting()) throw new Error("this device does not host the room: ask through the host");
      if (!this.ai.online) await (this.startP ||= this.start());   // not started yet: deal over whoever is here now
      const v = validateApiAsk(d, { profile: d.api === 2 && this.ai.tok ? this.apiProfile() : null });
      if (v.err) { handler({ t: "ai-busy", rid, code: v.code, why: v.err }); return; }
      this.ai.runs.set(self + ":" + rid, ac);
      try { await this.locked(() => this.runApi(v.req, self, this.name, handler, ac.signal)); }
      finally { this.ai.runs.delete(self + ":" + rid); }
    };
    go().catch((err) => handler({ t: "ai-busy", rid, code: "start", why: err.message }));
    return { rid, stop: () => ac.abort() };
  }
  // An OpenAI-style conversation, through the same client path `pooled serve` uses: normalized and
  // checked (finishRequest), sent as an API ask (request above), and the answer checked (Ask).
  //   messages: [{ role: "system" | "user" | "assistant" | "tool", content | text, tool_calls? / calls?, tool_call_id? }]
  //   opts: maxTokens, temperature, topK, stop, thinking, tools ([{ name, description, parameters }]),
  //         toolChoice, parallel, format, signal, client
  // Yields { type: "start", promptTokens }, { type: "token", text, think? },
  //   { type: "call", i, id, name } / { type: "call", i, a } / { type: "call", i, end: 1, args },
  //   then { type: "done", reason, usage, reused, calls: [{ id, name, args }], stats } or
  //   { type: "done", reason: "error", code, err }.
  ask(messages, opts = {}) {
    const q = [], wake = [];
    const push = (x) => { q.push(x); wake.splice(0).forEach((f) => f()); };
    try {
      const req = toApiRequest(messages, opts);
      const fin = finishRequest(req, { hostMeta: this.hostMeta });
      const v2 = needsV2(fin);
      let stats = "";
      const encoder = eventEncoder(push);
      const a = new Ask({ req: fin, v2, encoders: [new Collector(), encoder], idFor: (i) => `call_${i}`, log: this.log, label: "ask" });
      const h = this.request(askBody(fin, v2), (d) => {
        if (d.t === "ai-busy") { push({ type: "done", reason: "error", code: d.code, err: d.why, n: d.n, max: d.max }); return; }
        if (d.t === "ai-gendone") stats = d.stats || "";
        const r = a.feed(d);
        if (r?.error) push({ type: "done", reason: "error", code: r.error.kind || "server", err: r.error.message });
        else if (r?.answer) push({ type: "done", reason: r.answer.reason, usage: r.answer.usage, reused: r.answer.reused, calls: r.answer.calls, open: r.answer.open, stopSeq: r.answer.stopSeq, stats });
      });
      opts.signal?.addEventListener?.("abort", () => h.stop());
    } catch (err) {
      push({ type: "done", reason: "error", code: err instanceof ApiError ? err.kind : "bad", err: err.message });
    }
    return (async function* () {
      for (;;) {
        while (q.length) { const x = q.shift(); yield x; if (x.type === "done") return; }
        await new Promise((r) => wake.push(r));
      }
    })();
  }
  apiAsk(from, d) {
    const rid = typeof d.rid === "string" ? d.rid.slice(0, API_LIMITS.rid) : "";
    // as room.js apiAsk: only a device that joined as an API client, and only while the host allows them
    if (!this.ai.apis.has(from)) { this.sendTo(from, { t: "ai-busy", rid, code: "bad", why: "this device did not join as an API client" }); return; }
    if (!this.allowApi) { this.sendTo(from, { t: "ai-busy", rid, code: "off", why: "the host does not allow API clients in this room" }); return; }
    const v = validateApiAsk(d, { profile: d.api === 2 && this.ai.tok ? this.apiProfile() : null });
    if (v.err) { this.sendTo(from, { t: "ai-busy", rid, code: v.code, why: v.err }); return; }
    const ac = new AbortController();
    this.ai.runs.set(from + ":" + rid, ac);
    return this.locked(() => this.runApi(v.req, from, this.conns.get(from)?.name || "API", (m) => this.sendTo(from, m), ac.signal))
      .finally(() => this.ai.runs.delete(from + ":" + rid));
  }
  // room.js apiGenerate: the answer goes to the asker (send) and to the room's screens
  async runApi(req, from, name, send, signal) {
    const ai = this.ai, rid = req.rid, v2 = req.api === 2;
    if (!ai.engine || !ai.online) { send({ t: "ai-busy", rid, code: ai.degraded ? "degraded" : "loading", why: ai.degraded ? "a device left the room; the host has to re-deal the layers first" : "the model is still loading" }); return; }
    let prompt;
    try {
      prompt = v2 ? apiPrompt2(ai.tok, req, ai.engine.maxSeq, { profile: this.apiProfile(), cache: ai.apiTurns, encoder: ai.apiEnc, model: ai.model || "" })
        : apiPrompt(ai.tok, req, ai.engine.maxSeq, ai.apiCache);
    } catch (err) { prompt = { err: err.message, code: "bad" }; }
    if (prompt.err) { send({ t: "ai-busy", rid, code: prompt.code, why: prompt.err, n: prompt.n, max: prompt.max }); return; }
    // the room's screens (not API clients, not the asker, which gets its own stream): the full message
    // where the visibility allows the text, else the hidden stand-in (room.js apiGenerate)
    const toScreens = (msg) => {
      const ids = [...this.conns].filter(([id, e]) => id !== from && !e.meta?.api).map(([id]) => id);
      const { full, hidden } = chatRecipients(this.visibility, from, ids);
      for (const id of full) this.sendTo(id, msg);
      if (msg.t !== "ai-token") for (const id of hidden) this.sendTo(id, { t: msg.t, name: msg.name, stats: msg.stats, asker: msg.asker, ctx: msg.ctx, api: 1, hidden: true });
    };
    try {
      const label = `${name} · ${req.params.client} (API)`;
      const q = v2 ? [...req.messages].reverse().find((m) => m.role === "user" && !m.aside) : req.messages[req.messages.length - 1];
      const last = q ? q.text : "(tool results)";
      const mid = ai.msgSeq = (ai.msgSeq || 0) + 1;
      toScreens({ t: "ai-genstart", name: label, text: last.slice(0, API_LIMITS.shown), asker: from, cont: 0, mid, api: 1 });
      send({ t: "ai-genstart", rid, api: v2 ? 2 : 1, client: req.params.client, promptTokens: prompt.ids.length, model: ai.model, name: label, asker: from, mid, ...(v2 ? { style: prompt.profile.style } : {}) });
      // the system prompt + tools (and an agent's cache boundary in it), when long, are kept as pinned checkpoints
      const pins = v2 && this.ckptOn() ? this.pinsFor(req, prompt) : [];
      // and where its last user turn starts, for an agent's next call (ckpt.js turnPoint)
      const turn = v2 && this.ckptOn() && this.ckptOpts.turns !== false
        ? turnPoint(prompt.ids, { imStart: prompt.S?.imStart, user: this.userTok(), systemLen: prompt.systemLen || 0 }) : 0;
      if (process.env.POOLED_CKPT_DEBUG && ai.ckpt?.size) this.ckptDebug(req, prompt);
      const common = { tok: ai.tok, req, prompt, ctxMax: ai.engine.maxSeq, fallback: pickSampler(ai.settings.sampling), signal, generate: (ids, o) => this.generate(ids, { ...o, pins, turn }) };
      // the room's screens: v2 content and a compact line per tool call (room.js apiScreenMsg, simplified)
      const screen = (piece, d) => toScreens({ t: "ai-token", text: piece, d: d || 0 });
      const res = v2
        ? await apiRun2({ ...common, cache: ai.apiTurns, tt: this.apiTokenTexts(), log: this.log,
          send: (msg) => { send(msg); if (msg.t === "ai-token" && !msg.th) screen(msg.text, msg.d); else if (msg.t === "ai-call" && msg.name) screen(`\n→ ${msg.name}(…)\n`, 0); } })
        : await apiRun({ ...common, cache: ai.apiCache, send, onPiece: screen });
      const stats = (res.err ? "failed: " + res.err : res.stats) + " · via API";
      const ctx = { used: ai.fed ? ai.pos : 0, max: ai.engine.maxSeq };
      toScreens({ t: "ai-gendone", stats, ctx, failed: res.err ? 1 : 0, capped: 0, api: 1 });
      send({ t: "ai-gendone", rid, api: v2 ? 2 : 1, reason: res.reason, stopSeq: res.stopSeq || undefined, usage: res.usage, reused: res.reused, stats, ctx, failed: res.err ? 1 : 0, err: res.err || undefined,
        ...(v2 ? { calls: res.calls, ...(res.open ? { open: res.open } : {}) } : {}) });
      this.emit("answer", { from, rid, ...res });
    } catch (e) {
      // whatever throws in here, the asker gets a failed gendone and the room is not left busy
      const err = String(e?.message || e).slice(0, 300);
      this.log("API answer failed: " + err);
      toScreens({ t: "ai-gendone", stats: "failed: " + err, failed: 1, capped: 0, api: 1 });
      send({ t: "ai-gendone", rid, api: v2 ? 2 : 1, reason: "error", usage: { in: prompt.ids.length, out: 0 }, reused: 0, stats: "failed: " + err, failed: 1, err });
    }
  }
  // a question typed on a room screen: the room's own conversation (room.js aiGenerate)
  chatAsk(text, who, askerId) {
    if (!text) return;
    return this.locked(async () => {
      const ai = this.ai;
      if (!ai.engine || !ai.online) { this.sendTo(askerId, { t: "ai-busy", why: "the model is still loading" }); return; }
      const S = specials(ai.tok), persona = PERSONAS[ai.settings.persona] || PERSONAS.default;
      const sample = pickSampler(ai.settings.sampling), stopIds = new Set([S.imEnd, S.eot]);
      const mid = ai.msgSeq = (ai.msgSeq || 0) + 1;
      ai.askerId = askerId; ai.chatAbort = new AbortController();
      const toAll = (m) => {
        const { full, hidden } = chatRecipients(this.visibility, askerId, [...this.conns].filter(([, e]) => !e.meta?.api).map(([id]) => id));
        for (const id of full) this.sendTo(id, m);
        if (m.t !== "ai-token") for (const id of hidden) this.sendTo(id, { t: m.t, name: m.name, stats: m.stats, asker: m.asker, ctx: m.ctx, hidden: true });
      };
      toAll({ t: "ai-genstart", name: who, text, asker: askerId, cont: 0, mid });
      const answer = [], pieces = pieceDecoder(ai.tok);
      let failed = null, r = null, reply = "";
      const show = (p, d) => { if (p) { reply += p; toAll({ t: "ai-token", text: p, d: d || 0 }); } };
      try {
        const fit = fitContext(ai.tok, { system: persona.system, turns: [...ai.conv.turns, { role: "user", text, name: who }], thinking: false }, ai.engine.maxSeq, MIN_ROOM);
        ai.conv.turns = fit.turns;
        r = await this.generate(fit.ids, { stop: stopIds, sample, maxNew: this.chatMaxNew, signal: ai.chatAbort.signal, onToken: (t, d) => { answer.push(t); show(pieces.push(t), d); } });
        show(pieces.flush(), 0);
      } catch (err) { failed = err; this.log("chat answer failed: " + err.message); }
      ai.conv.turns.push({ role: "assistant", ids: answer });
      toAll({ t: "ai-gendone", stats: failed ? "failed: " + failed.message : r.stats, ctx: { used: ai.fed ? ai.pos : 0, max: ai.engine?.maxSeq || 0 }, failed: failed ? 1 : 0, capped: 0 });
      this.emit("chatanswer", { who, text, reply, stats: r?.stats });
    });
  }

  // a snapshot for a status line or a plugin's state file
  status() {
    const ai = this.ai;
    const devs = this.hosting()
      ? [{ name: this.name, gb: this.meta?.contribGB || 0, self: true }, ...this.gpuPeers().map((id) => { const e = this.conns.get(id); return { name: e?.name || id, gb: +pledgeGB(e?.meta) || 0 }; })]
      : (this.members || []).filter((m) => m.meta?.webgpu && !m.meta?.api).map((m) => ({ name: m.name, gb: +m.meta?.contribGB || 0 }));
    return { code: this.code, name: this.name, hosting: this.hosting(), role: ai.role, model: ai.model, online: !!ai.online, degraded: !!ai.degraded,
      range: ai.range, devices: devs, pledgedGB: +devs.reduce((a, d) => a + d.gb, 0).toFixed(1),
      split: this.split?.names?.map((nm, i) => `${nm} ${this.split.ranges[i][0]}-${this.split.ranges[i][1] - 1}`) || null,
      ctx: ai.engine?.maxSeq || null, ctxNote: ai.ctxNote || null, loading: !!ai.loadingShard, signaling: !this.signalDown,
      passes: this.hosting() ? ai.frames || 0 : this.frames || 0,
      ckpt: ai.ckpt ? { pinned: ai.ckpt.items.filter((x) => x.pin).map((x) => x.ids.length), answers: ai.ckpt.items.filter((x) => !x.pin).map((x) => x.ids.length), hits: { ...ai.ckpt.hits } } : null };
  }

  async close() {
    this.closing = true;
    clearInterval(this.pingTimer); clearInterval(this.knock); clearTimeout(this.ai.idleRedeal);
    // a host closing for good tells the room first (the room page shows it as "Room over"), so no
    // device knocks for a minute waiting for it to come back
    if (this.isHost) try { this.broadcast({ t: "bye", reason: HOST_CLOSED, closed: 1 }); } catch {}
    try { this.broadcast({ t: "leaving" }); } catch {}
    for (const L of this.lobbyConns.values()) try { L.conn.send({ t: "bye", reason: "the room closed" }); } catch {}
    await new Promise((r) => setTimeout(r, 200));
    try { this.peer?.destroy(); } catch {}
    this.lobbyConns.clear();
    clearTimeout(this.ai.idleRedeal);
    this.failWaiters(new Error("the room closed"));
    this.ai.online = false; this.ai.chain = []; this.ai.ckpt = null;
    this.freeLayers(null);   // the engine, its checkpoint slots and the device (device.destroy frees every buffer)
  }
}

// ---------------- the client side of an ask (shared with `pooled serve`) ----------------
const partText = (c) => typeof c === "string" ? c : Array.isArray(c) ? c.map((p) => typeof p === "string" ? p : p?.text || "").join("") : c == null ? "" : String(c);
// OpenAI-style messages (or the internal shape: text / calls) -> the internal request cli/lib/common.js
// finishRequest takes (withDefaults' fields)
export function toApiRequest(messages, { maxTokens = 1024, temperature = null, topK = null, stop = [], thinking = false, client = "node",
  tools = null, toolChoice = "auto", parallel = true, format = null, allowed = null, maxCalls = null, effort = null } = {}) {
  const out = [];
  for (const m of messages || []) {
    const role = m.role === "developer" ? "system" : m.role;
    const text = m.text ?? partText(m.content);
    if (role === "system" || role === "user") out.push({ role, text });
    else if (role === "assistant") {
      const calls = (m.calls || m.tool_calls || []).map((c, i) => {
        const f = c.function || c;
        let args = f.args ?? f.arguments ?? {};
        if (typeof args === "string") { try { args = JSON.parse(args || "{}"); } catch { args = {}; } }
        return { id: c.id || `call_${i}`, name: f.name, args: args && typeof args === "object" && !Array.isArray(args) ? args : {} };
      });
      out.push({ role, text, ...(calls.length ? { calls } : {}), ...(m.reasoning ? { reasoning: m.reasoning } : {}) });
    } else if (role === "tool") out.push({ role, text, ...(m.tool_call_id || m.id ? { id: m.tool_call_id || m.id } : {}) });
  }
  return withDefaults({ client, messages: out, maxTokens, temperature, topK, stop, thinking, tools: tools?.length ? tools.map((t) => ({ name: t.name, description: t.description || "", parameters: t.parameters || { type: "object", properties: {} } })) : null,
    toolChoice, parallel, format, allowed, maxCalls, effort });
}
// an Ask encoder (cli/lib/answer.js) that turns the checked answer into ask()'s events
export function eventEncoder(push) {
  return {
    start: (n) => push({ type: "start", promptTokens: n }),
    think: (t) => push({ type: "token", text: t, think: true }),
    text: (t) => push({ type: "token", text: t }),
    callStart: (i, id, name) => push({ type: "call", i, id, name }),
    callArgs: (i, a) => push({ type: "call", i, a }),
    callEnd: (i, args) => push({ type: "call", i, end: 1, args }),
    done() {}, error() {}, keepAlive() {},
  };
}

// Host a room on this machine. -> the RoomNode (room.code, room.start(), room.ask(), room.close())
// gate: hold links at the room page's gate (gate.js; pooled host turns it on). Off by default, so a
// caller without a way to answer join requests (the OpenClaw plugin) keeps a room anyone with the code
// joins, as before. ask (with the gate): hold new devices until allowJoin() (default), false to let
// anyone with the code in; a device with the room's invite key (node.inviteFragment) is let in either way
export async function createRoom({ model = "qwen3-1.7b", pledgeGB, code = randomCode(CODE_LEN), ask = true, gate = false, gateState = null, ...opts } = {}) {
  const node = new RoomNode({ pledgeGB, ...opts });
  node.isHost = true; node.code = code; node.ai.model = model; node.ai.role = "host";
  if (gate) node.gate = hostGate({ ask, saved: gateState });   // gateState: gate.js saveGate() from an earlier run
  await node.open(PREFIX + code);
  return node;
}

// Join a room as a device that holds layers when the host deals it some.
export async function joinRoom(code, { pledgeGB, joinMs = 20000, ...opts } = {}) {
  const node = new RoomNode({ pledgeGB, ...opts });
  node.code = String(code).toUpperCase();
  await node.open(undefined);
  const hostId = PREFIX + node.code;
  node.ai.hostId = hostId;
  try {
    await new Promise((resolve, reject) => {
      const fail = (e) => { clearTimeout(t); node.peer.off("error", onPeerErr); reject(e); };
      const notFound = () => Object.assign(new Error(`no room ${node.code}`), { code: "room-not-found" });
      // the signaling server knows no such id: the room does not exist (or its host left)
      const onPeerErr = (err) => { if (err?.type === "peer-unavailable" && String(err.message || "").includes(hostId)) fail(notFound()); };
      node.peer.on("error", onPeerErr);
      const t = setTimeout(() => fail(notFound()), joinMs);
      const conn = node.peer.connect(hostId, { reliable: true });
      guardChunks(conn);
      conn.on("open", () => {
        node.wire(conn, "host", true);
        conn.send(node.helloMsg(joinHelloFields(node)));
        clearTimeout(t); node.peer.off("error", onPeerErr); resolve();
      });
      conn.on("error", (e) => fail(e));
    });
  } catch (err) { await node.close().catch(() => {}); throw err; }
  return node;
}

// --signal: "host:port" (the old form: TLS only on port 443, as cli/lib/room.js signalOpts), or
// room/signal.js specs as a comma list (cloud, wss://host:port/path, ...); none -> the PeerJS cloud
export function nodeServers(signal) {
  const out = [];
  for (const s of String(signal || "cloud").split(",").map((x) => x.trim()).filter(Boolean)) {
    const bare = /^[^/:\[\]]+:(\d+)$/.exec(s);
    const p = parseServer(s, bare ? +bare[1] === 443 : true);
    if (p && !out.some((x) => x.spec === p.spec)) out.push(p);
  }
  if (!out.length) throw new Error(`--signal: no usable server in "${signal}"`);
  return out;
}
