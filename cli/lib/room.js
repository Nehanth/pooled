// The bridge: joins a Pooled room as one more ask-only guest with no layers, over the same PeerJS
// client the room page loads (peerjs 1.5.4) running on node-datachannel's WebRTC. It speaks the
// room protocol (docs/protocol.md, "API clients") and hands each API request's messages to the
// right HTTP response by its rid.
import { EventEmitter } from "node:events";
import { cleanText } from "./common.js";

export const PREFIX = "pooled-room-";
export const PROTOCOL = 4;          // room/transport.js PROTOCOL: the host says bye to any other
const JOIN_MS = 15000, KNOCK_MS = 3000, HOST_WAIT_MS = 60000;
// PeerJS rebuilds a message split into chunks with the chunk count the sender states, with no
// limit: one message from the host could grow as large as it likes before our code sees it. The
// biggest the room sends (a welcome with the chat's recent transcript) is a few hundred KB.
const MAX_CHUNKS = 256, MAX_PARTIAL = 8;   // ~4 MB per message (16 KB chunks), 8 being rebuilt at once
export function guardChunks(conn) {
  const orig = typeof conn._handleChunk === "function" ? conn._handleChunk.bind(conn) : null;
  if (!orig) return;
  conn._handleChunk = (data) => {
    const { total, n } = data || {};
    if (!Number.isInteger(total) || total < 1 || total > MAX_CHUNKS || !Number.isInteger(n) || n < 0 || n >= total) return;
    const partial = conn._chunkedData || {};
    if (!partial[data.__peerData] && Object.keys(partial).length >= MAX_PARTIAL) return;
    orig(data);
  };
}
const ICE = { iceServers: [{ urls: ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"] }] };

// room/chanauth.js (proving the invite key and pass without sending them): the copy built into dist/
// (npm run build; what the npm package ships), else the checkout's
let authP = null;
export function loadAuth() {
  return authP ||= (async () => {
    let last;
    for (const u of [new URL("../dist/chanauth.js", import.meta.url), new URL("../../room/chanauth.js", import.meta.url)]) {
      try { return await import(u.href); } catch (e) { last = e; }
    }
    authP = null;
    throw last;
  })();
}
const HOST_HELLO_WAIT_MS = 2000;

let PeerClass = null;
// PeerJS is a browser library: give it RTCPeerConnection & co. (node-datachannel/polyfill) and the
// few globals it reads, then load its CommonJS bundle (its named exports sit on default).
async function loadPeer() {
  if (PeerClass) return PeerClass;
  const rtc = await import("node-datachannel/polyfill");
  for (const [k, v] of Object.entries(rtc)) if (k !== "default" && globalThis[k] === undefined) globalThis[k] = v;
  globalThis.window ??= globalThis;
  globalThis.navigator ??= { userAgent: "pooled-cli" };
  globalThis.location ??= { protocol: "https:" };   // peerjs util.isSecure() reads it
  const m = await import("peerjs");
  PeerClass = (m.default?.Peer ? m.default : m).Peer;
  return PeerClass;
}

// "4TK-G9P", "4tkg9p", "ABCD" (rooms from before six-character codes), "https://pooled.run/r/4TKG9P#k=…",
// "…/room?code=4TKG9P", "…#ABCD" -> "4TKG9P" (or null)
export function roomCodeFrom(s) {
  s = String(s || "").trim();
  const ok = (c) => { c = String(c || "").replace(/-/g, "").toUpperCase(); return /^[A-Z0-9]{4}$|^[A-Z0-9]{6}$/.test(c) ? c : null; };
  if (ok(s)) return ok(s);
  try {
    const u = new URL(s);
    return ok(u.searchParams.get("code")) || ok(u.pathname.split("/").filter(Boolean).pop()) || ok(u.hash.slice(1));
  } catch { return null; }
}
// the invite key in a room link's fragment ("…/r/4TKG9P#k=…"): the host lets a client that has it in
// without asking (room/joingate.js). null when there is none
export function roomKeyFrom(s) {
  try {
    const h = new URL(String(s || "").trim()).hash.replace(/^#/, "");
    for (const part of h.split("&")) { const [k, v] = part.split("="); if (k === "k" && /^[A-Za-z0-9_-]{22,64}$/.test(v || "")) return v; }
  } catch {}
  return null;
}

// "host:port" -> PeerJS server options; none -> the PeerJS cloud (what the room page uses by default)
export function signalOpts(signal) {
  if (!signal) return {};
  const [host, p] = String(signal).split(":");
  const port = +p || 443;
  return { host, port, path: "/", secure: port === 443 };
}

export class Bridge extends EventEmitter {
  // Peer: a PeerJS class already set up in this process (@pooled/room-node's), so one process never
  // loads two WebRTC stacks; default: load node-datachannel + peerjs here
  // legacySend: send the raw invite key or pass to a gated host from before channel-bound proofs, as
  // before. Off by default (this client waits for that host's Allow); POOLED_LEGACY_AUTH=1 turns it on,
  // which is unsafe: anyone in the middle of signaling can pose as such a host and read the key
  constructor({ code, key = null, signal = null, name, client, log = () => {}, Peer = null, legacySend = process.env.POOLED_LEGACY_AUTH === "1" }) {
    super();
    this.legacySend = legacySend === true;
    this.code = code; this.signal = signal; this.name = name; this.client = client; this.log = log; this.PeerClass = Peer;
    this.key = key;             // the room's invite key, from its link: in without the host's Allow
    this.pass = null;           // what the host gave us once it let us in: back in after a reconnect
    this.waiting = false;       // in the host's lobby: it was asked to let us in
    this.peer = null; this.conn = null;
    this.hostMeta = null; this.hostName = null;
    this.connected = false;     // a link to the host is open and it said hello
    this.ready = false;         // the host's model is up (ai-ready-all)
    this.model = null; this.modelLabel = null; this.readySince = null;
    this.kicked = null;         // the host's bye reason: no more requests, no reconnect
    this.gone = null;           // why the host is unreachable, while knocking
    this.reqs = new Map();      // rid -> handlers
    this.closing = false;
  }
  // join: 1 = this client can wait in the host's lobby (a host that asks before devices join holds it
  // until the host allows it); auth fields: what it can prove (room/chanauth.js joinerStart), never the
  // invite key or pass themselves. legacy: the raw key and pass, for a gated host from before the proofs
  helloMsg(back, fields = {}) {
    return { t: "hello", name: this.name, v: PROTOCOL, join: 1, ...(back ? { back: 1 } : {}), ...fields,
      meta: { api: 1, client: this.client, webgpu: false, ua: "API" } };
  }
  // -> [hello fields, the link's auth state (null: nothing to check)]. Holding a key or a pass, it waits
  // briefly for the host's hello first: it says whether the host speaks the proofs
  async joinFields(link) {
    const A = await loadAuth();
    if (this.key || this.pass) {
      const t0 = Date.now();
      while (!link.greeted && Date.now() - t0 < HOST_HELLO_WAIT_MS && link.conn.open !== false) await new Promise((r) => setTimeout(r, 25));
    }
    const hh = link.greeted;
    if (hh && hh.gate && !(+hh.auth >= A.AUTH_V) && (this.key || this.pass)) {
      if (this.legacySend) {
        this.log("this room's host runs an older Pooled: the invite key goes to it the old way, unprotected (POOLED_LEGACY_AUTH=1)");
        return [{ ...(this.pass ? { pass: this.pass } : {}), ...(this.key ? { key: this.key } : {}) }, null];
      }
      this.log("this room's host runs an older Pooled: ask them to let this client in, or to update (it doesn't get the invite key: it can't check it safely)");
      this.oldHost = true;
      const st = await A.joinerStart({});
      return [st.helloFields, st];
    }
    const st = await A.joinerStart({ key: this.key, pass: this.pass });
    return [st.helloFields, st];
  }
  // -> resolves once the host said hello with meta.api; rejects with a message for the user
  async connect() {
    const Peer = this.PeerClass || await loadPeer();
    return new Promise((resolve, reject) => {
      let settled = false;
      const fail = (msg) => { if (settled) return; settled = true; clearTimeout(timer); this.destroy(); reject(new Error(msg)); };
      let timer = setTimeout(() => fail(`no room ${this.code} (is the host page open?)`), JOIN_MS);
      // held in the host's lobby: no join timeout while the host decides (Ctrl-C gives up)
      this.once("lobby", () => { clearTimeout(timer); timer = null; });
      // turned away before we got in (Deny, API clients off, a full lobby, an older client)
      this.once("refused", (why) => fail(`the host of room ${this.code} said: ${why}`));
      this.peer = new Peer(undefined, { debug: 0, config: ICE, ...signalOpts(this.signal) });
      this.peer.on("error", (err) => {
        if (!settled) {
          if (err.type === "peer-unavailable") fail(`no room ${this.code} (is the host page open?)`);
          else if (err.type === "network" || err.type === "server-error" || err.type === "socket-error") fail(`could not reach the signaling server${this.signal ? " " + this.signal : ""} (${err.type})`);
          else fail(`could not join room ${this.code}: ${err.type || err.message}`);
          return;
        }
        if (err.type !== "peer-unavailable") this.log(`peer error: ${err.type || err.message}`);
      });
      // nobody has a reason to dial the bridge: refuse links it did not open (each costs a native connection)
      this.peer.on("connection", (c) => { try { c.close(); } catch {} });
      this.peer.on("disconnected", () => { if (!this.closing && !this.peer.destroyed) setTimeout(() => { try { this.peer.reconnect(); } catch {} }, 1000); });
      this.peer.on("open", () => {
        this.dial(false, (hello) => {
          if (settled) return;
          if (!hello.meta?.api) { fail(`the host of room ${this.code} runs an older Pooled; reload the host page`); return; }
          settled = true; clearTimeout(timer);
          resolve();
        });
      });
    });
  }
  // open a link to the host; onHello(hostHello) once it greets us
  dial(back, onHello) {
    const conn = this.peer.connect(PREFIX + this.code, { reliable: true });
    guardChunks(conn);
    let greeted = null, admitted = false;
    const link = { conn, greeted: null, auth: null };
    // in: the host said hello and (a host with the gate) let us in
    const inRoom = () => {
      if (!greeted || !admitted || this.connected) return;
      this.connected = true; this.gone = null; this.waiting = false;
      onHello?.(greeted);
      this.emit("state");
    };
    conn.on("open", () => {
      if (this.conn && this.conn !== conn && this.conn.open) { try { conn.close(); } catch {} return; }
      this.conn = conn;
      this.joinFields(link).then(([fields, st]) => { link.auth = st; conn.send(this.helloMsg(back, fields)); })
        .catch((err) => { this.log(`could not prepare the join: ${err.message}`); try { conn.close(); } catch {} });
    });
    // the host let us in: when this client proved its invite key or pass, the host must prove it back
    // (room/chanauth.js), or this is not the room's host (or someone sits in the middle of the link)
    const onAdmit = async (d) => {
      const A = await loadAuth();
      const v = link.auth ? await A.joinerCheckAdmit(link.auth, d) : "ok";
      // nothing proved back (a host restarted without its passes, a link from an earlier room, an older
      // host): in on trust, as with a typed code, and said plainly
      if (v === "tofu") this.log(`warning: couldn't verify the room's host with the invite link or pass${link.auth.sas ? ` (code ${link.auth.sas})` : ""}: fine if it restarted or runs an older Pooled; otherwise someone may be in the middle of the connection`);
      if (v === "bad") {
        this.kicked = "couldn't verify the room's host: its proof of the invite key is wrong (someone may be in the middle of the connection)";
        this.waiting = false;
        this.log(this.kicked);
        this.emit("refused", this.kicked);
        this.emit("state");
        try { conn.close(); } catch {}
        return;
      }
      if (typeof d.pass === "string" && /^[A-Za-z0-9_-]{22,64}$/.test(d.pass)) this.pass = d.pass;
      if (this.waiting) this.log("the host let this client in");
      admitted = true;
      inRoom();
    };
    conn.on("data", (d) => {
      if (!d || typeof d.t !== "string") return;
      if (d.t === "hello" && !greeted) {
        greeted = d; link.greeted = d;
        this.hostMeta = d.meta || {}; this.hostName = cleanText(d.name, 40);
        // a host from before the gate lets everyone in (and never got a key or pass from this client):
        // in on trust, said plainly
        if (!d.gate && (this.key || this.pass)) this.log("warning: couldn't verify the room's host: it runs a Pooled from before the gate");
        if (!d.gate) admitted = true;
        inRoom();
        return;
      }
      if (d.t === "auth" && !admitted && link.auth && !link.auth.hn) {   // the host's challenge
        loadAuth().then((A) => A.joinerProof(link.auth, d, { fps: A.linkFingerprints(conn), me: this.peer?.id, host: conn.peer }))
          .then((m) => { if (m) try { conn.send(m); } catch {} }).catch((err) => this.log(`auth: ${err.message}`));
        return;
      }
      if (d.t === "admit" && !admitted) {
        onAdmit(d).catch((err) => this.log(`admit: ${err.message}`));
        return;
      }
      if (d.t === "lobby" && !admitted) {
        if (link.auth) link.auth.lobbied = true;
        const sas = link.auth?.sas;
        // the hint fits what this client came with: an invite link that didn't get it in (an older host,
        // a link from an earlier room) is not fixed by the invite link
        const hint = this.oldHost ? "it runs an older Pooled that can't check the invite link: ask them to let this client in, or to update"
          : this.key ? "the invite link didn't check out with this host: ask for a fresh one, or for the host to let this client in"
          : "start pooled serve with the room's invite link to skip this";
        if (!this.waiting) this.log(`waiting for the host of room ${this.code} to let this client in (${hint})${sas ? `; the host sees code ${sas} beside the request` : ""}`);
        this.waiting = true;
        this.sas = sas || null;
        this.emit("lobby", { sas: sas || null });
        this.emit("state");
        return;
      }
      if (d.t === "bye" && !admitted) {
        this.kicked = cleanText(d.reason, 300) || "the host closed the link";
        this.waiting = false;
        this.emit("refused", this.kicked);
        this.emit("state");
        return;
      }
      if (!admitted) return;   // nothing from the room until we are in (the host sends nothing anyway)
      this.onData(d);
    });
    conn.on("close", () => { if (this.conn === conn) this.lost("lost the link to the host"); });
    conn.on("error", () => {});
  }
  onData(d) {
    switch (d.t) {
      case "ping": this.send({ t: "pong", ts: d.ts }); return;
      case "bye":
        this.kicked = cleanText(d.reason, 300) || "the host closed the link";
        this.connected = false; this.ready = false;
        this.failAll("unavailable", this.kicked);
        this.log(`the host said: ${this.kicked}`);
        this.emit("state");
        return;
      case "ai-ready-all":
        // the model's id and label end up in the terminal and in /v1/models: plain, short text only
        this.ready = true;
        this.model = cleanText(d.model, 80).replace(/[^\w.:+\-\/]/g, "") || this.model;
        this.modelLabel = cleanText(d.label, 80) || this.modelLabel;
        this.readySince ??= Math.floor(Date.now() / 1000);
        // a v2 host says its context size (it changes with the model): the early context check uses it
        if (this.hostMeta && Number.isInteger(d.ctx) && d.ctx > 0) this.hostMeta.ctx = d.ctx;
        this.emit("state");
        return;
      case "ai-degraded": case "ai-redeal":
        this.ready = false;
        this.emit("state");
        return;
      case "ai-queued": case "ai-genstart": case "ai-token": case "ai-call": case "ai-gendone": case "ai-busy": {
        if (d.rid == null) return;   // the room's chat, not ours
        const h = this.reqs.get(String(d.rid));
        if (!h) return;
        if (d.t === "ai-gendone" || d.t === "ai-busy") this.reqs.delete(String(d.rid));
        h(d);
        return;
      }
    }
  }
  send(msg) { try { if (this.conn?.open) { this.conn.send(msg); return true; } } catch {} return false; }
  // what the host answers: 2 = v2 asks too (tools, formats; hello meta.api), 1 = plain ones only
  get hostApi() { return Math.max(1, Math.floor(+this.hostMeta?.api) || 1); }
  // an ask: handler(msg) gets ai-queued / ai-genstart / ai-token / ai-call / ai-gendone / ai-busy
  // for this rid. body: common.askBody(req, v2) (a v2 body carries api: 2; send one only when hostApi >= 2)
  ask(rid, body, handler) {
    this.reqs.set(rid, handler);
    if (!this.send({ t: "ai-ask", api: 1, rid, ...body })) { this.reqs.delete(rid); return false; }
    return true;
  }
  stop(rid) { this.reqs.delete(rid); this.send({ t: "ai-stop", rid }); }
  failAll(kind, why) {
    for (const [rid, h] of this.reqs) { this.reqs.delete(rid); h({ t: "x-fail", rid, kind, why }); }
  }
  // the host's link closed: every open request fails now; knock every 3 s for a minute, like a browser guest
  lost(why) {
    if (this.closing || this.kicked) return;
    this.connected = false; this.ready = false; this.conn = null;
    this.gone = why;
    this.failAll("unavailable", `${why}; waiting for the host to come back`);
    this.log(`${why}; knocking for a minute in case the host page reloads`);
    this.emit("state");
    const t0 = Date.now();
    clearInterval(this.knock);
    this.knock = setInterval(() => {
      if (this.closing || this.kicked || this.connected) { clearInterval(this.knock); return; }
      if (Date.now() - t0 > HOST_WAIT_MS) {
        clearInterval(this.knock);
        this.gone = `the host of room ${this.code} did not come back`;
        this.log(this.gone);
        this.emit("state");
        return;
      }
      if (this.peer?.destroyed) return;
      if (this.peer?.disconnected) { try { this.peer.reconnect(); } catch {} return; }
      this.dial(true, () => { clearInterval(this.knock); this.log("the host is back"); });
    }, KNOCK_MS);
  }
  destroy() {
    this.closing = true;
    clearInterval(this.knock);
    try { this.peer?.destroy(); } catch {}
  }
  // Ctrl-C: say so, so the host drops the card now instead of when ICE times out
  async leave() {
    this.closing = true;
    this.send({ t: "leaving" });
    await new Promise((r) => setTimeout(r, 150));
    this.destroy();
  }
}
