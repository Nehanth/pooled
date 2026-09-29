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

// "ABCD", "abcd", "https://pooled.run/r/ABCD", "…/room?code=ABCD", "…#ABCD" -> "ABCD" (or null)
export function roomCodeFrom(s) {
  s = String(s || "").trim();
  const ok = (c) => (/^[A-Z0-9]{4,6}$/i.test(c || "") ? c.toUpperCase() : null);
  if (ok(s)) return ok(s);
  try {
    const u = new URL(s);
    return ok(u.searchParams.get("code")) || ok(u.pathname.split("/").filter(Boolean).pop()) || ok(u.hash.slice(1));
  } catch { return null; }
}

// "host:port" -> PeerJS server options; none -> the PeerJS cloud (what the room page uses by default)
export function signalOpts(signal) {
  if (!signal) return {};
  const [host, p] = String(signal).split(":");
  const port = +p || 443;
  return { host, port, path: "/", secure: port === 443 };
}

export class Bridge extends EventEmitter {
  constructor({ code, signal = null, name, client, log = () => {} }) {
    super();
    this.code = code; this.signal = signal; this.name = name; this.client = client; this.log = log;
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
  helloMsg(back) {
    return { t: "hello", name: this.name, v: PROTOCOL, ...(back ? { back: 1 } : {}),
      meta: { api: 1, client: this.client, webgpu: false, ua: "API" } };
  }
  // -> resolves once the host said hello with meta.api; rejects with a message for the user
  async connect() {
    const Peer = await loadPeer();
    return new Promise((resolve, reject) => {
      let settled = false;
      const fail = (msg) => { if (settled) return; settled = true; clearTimeout(timer); this.destroy(); reject(new Error(msg)); };
      const timer = setTimeout(() => fail(`no room ${this.code} (is the host page open?)`), JOIN_MS);
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
    let greeted = false;
    conn.on("open", () => {
      if (this.conn && this.conn !== conn && this.conn.open) { try { conn.close(); } catch {} return; }
      this.conn = conn;
      conn.send(this.helloMsg(back));
    });
    conn.on("data", (d) => {
      if (!d || typeof d.t !== "string") return;
      if (d.t === "hello" && !greeted) {
        greeted = true;
        this.hostMeta = d.meta || {}; this.hostName = cleanText(d.name, 40);
        this.connected = true; this.gone = null;
        onHello?.(d);
        this.emit("state");
        return;
      }
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
        this.emit("state");
        return;
      case "ai-degraded": case "ai-redeal":
        this.ready = false;
        this.emit("state");
        return;
      case "ai-queued": case "ai-genstart": case "ai-token": case "ai-gendone": case "ai-busy": {
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
  // an ask: handler(msg) gets ai-queued / ai-genstart / ai-token / ai-gendone / ai-busy for this rid
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
