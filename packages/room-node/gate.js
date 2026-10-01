// Who gets into a room a node hosts, and how a node gets into someone else's: the room page's gate
// (room/joingate.js, docs/protocol.md "Joining a room") on the headless room node. Kept apart from
// roomnode.js so the two sides stay small and the gate can change shape without touching the rest.
//
// Host (a node made by createRoom): a link the host did not open waits in the lobby until its hello;
// the gate lets it in with the room's invite key (the link's #k=), a pass the host gave it before,
// the host's Allow (allowJoin), or at once when "ask" is off (pooled host --allow-all). Until then the
// link is not in node.conns: no roster, chat, ai-* or pings reach it, and nothing it sends reaches the
// room (ping is answered; up to 64 other messages are kept and handled once it is in).
//
// Device (a node made by joinRoom): its hello to the host carries join: 1 (it can wait in the lobby),
// the invite key from its link and the pass from an earlier admit; key and pass go only to the host.
// A host from before the gate (no gate: 1 in its hello) lets it in at once, as before.
import { makeGate, decide, enqueue, allow, deny, withdraw, requestLine, validKey, keyFragment,
  randomCode, formatCode, parseCode, CODE_LEN, OLD_TAB_TEXT } from "../../room/joingate.js";

export { keyFragment, formatCode, parseCode, randomCode, CODE_LEN, validKey };
const LOBBY_BUF = 64;

// ---------------- host ----------------
// the host's gate: ask = hold new devices until allowed (pooled host's default); key: the room's
// invite key (a new random one by default)
export function hostGate({ ask = true, key = null } = {}) { return makeGate({ ask, key }); }

// what a host's hello adds (docs/protocol.md): this host holds new devices at the gate
export const gateHelloFields = (g) => (g ? { gate: 1, ask: g.ask ? 1 : 0 } : {});

// accept(): a link opened by someone else, on a host with a gate. It is held (node.lobbyConns) until
// its hello lets it in. Stripes (extra associations for the wire) of a held device wait with it.
export function holdConn(node, conn) {
  const id = conn.peer;
  if (conn.label === "stripe") {
    const L = node.lobbyConns.get(id);
    if (L) { L.stripes.push(conn); return true; }
    return false;   // its device is in already: roomnode attaches it
  }
  const prev = node.lobbyConns.get(id);
  if (prev) { try { prev.conn.close(); } catch {} }
  const L = { conn, hello: null, buf: [], stripes: [], in: false };
  node.lobbyConns.set(id, L);
  conn.on("data", (d) => {
    if (L.in || node.lobbyConns.get(id) !== L) return;
    if (!d || typeof d.t !== "string") return;   // binary (a bandwidth test): nothing a waiting device sends
    if (d.t === "hello" && !L.hello) {
      L.hello = d;
      gateHello(node, L, d).catch((err) => { node.log(`gate: ${err.message}`); refuse(node, L, "The host couldn't check this device. Try again."); });
      return;
    }
    if (d.t === "ping") { try { conn.send({ t: "pong", ts: d.ts }); } catch {} return; }
    if (d.t === "leaving") { try { conn.close(); } catch {} return; }
    if (L.buf.length < LOBBY_BUF) L.buf.push(d);
  });
  conn.on("close", () => {
    if (L.in || node.lobbyConns.get(id) !== L) return;
    node.lobbyConns.delete(id);
    const r = withdraw(node.gate, id);
    if (r) { node.log(`${r.name || id} stopped waiting to join`); node.emit("joinrequests", waitingJoins(node)); }
  });
  conn.on("error", () => {});
  try { conn.send(node.helloMsg()); } catch {}
  return true;
}

async function gateHello(node, L, d) {
  const id = L.conn.peer;
  const name = String(d.name ?? "").replace(/[\u0000-\u001f\u007f<>"'`&]/g, "").trim().slice(0, 40) || id.slice(0, 8);
  // another protocol: the room node's own hello handling says so (bye) and reports it
  if (d.v !== node.protocol) { admit(node, L, null, "version"); return; }
  if (d.meta?.api && !node.allowApi) { refuse(node, L, "the host does not allow API clients in this room"); return; }
  const r = await decide(node.gate, id, d);
  if (node.lobbyConns.get(id) !== L || node.closing) return;   // it left (or a newer link took over) while the hash ran
  if (r.kind === "admit") { admit(node, L, r.pass, r.via); return; }
  if (r.kind === "refuse") {
    // an older device: say how to fix it in its own terms (a pooled join / serve from before the gate)
    const why = r.reason === OLD_TAB_TEXT && (d.meta?.api || d.meta?.native === "node-dawn")
      ? "This room's host asks before new devices join, and this pooled is too old to wait for that. Update it (npx @pooled/cli@latest) or ask the host for the invite link."
      : r.reason;
    refuse(node, L, why);
    return;
  }
  const req = enqueue(node.gate, id, name, d.meta);
  try { L.conn.send({ t: "lobby" }); } catch {}
  if (!node.listenerCount("joinrequest")) node.log(`${requestLine(name, d.meta)}: allowJoin() lets it in, denyJoin() turns it away`);
  node.emit("joinrequest", { id, name, meta: d.meta || {}, line: requestLine(name, d.meta), at: req.at });
  node.emit("joinrequests", waitingJoins(node));
}

function admit(node, L, pass, via) {
  const conn = L.conn, id = conn.peer;
  node.lobbyConns.delete(id);
  L.in = true;
  if (via !== "version") try { conn.send({ t: "admit", ...(pass ? { pass } : {}) }); } catch {}
  if (via === "key" || via === "allowed") node.log(`${String(L.hello?.name || id).slice(0, 40)} ${via === "key" ? "came in with the invite link" : "was let in"}`);
  const e = node.wire(conn);
  for (const s of L.stripes) node.attachStripe(e, s);
  node.onData(id, L.hello);
  for (const m of L.buf) node.onData(id, m);
  L.buf = [];
}

function refuse(node, L, reason) {
  const id = L.conn.peer;
  if (node.lobbyConns.get(id) === L) node.lobbyConns.delete(id);
  L.in = true;
  node.log(`turned away ${String(L.hello?.name || id).slice(0, 40)}: ${reason}`);
  try { L.conn.send({ t: "bye", reason }); } catch {}
  setTimeout(() => { try { L.conn.close(); } catch {} for (const s of L.stripes) try { s.close(); } catch {} }, 300).unref?.();
}

// the host said yes / no to a device waiting in the lobby (the oldest one when id is left out)
export async function allowJoin(node, id = node.gate?.lobby[0]?.id) {
  const L = node.lobbyConns.get(id);
  const r = id ? await allow(node.gate, id) : null;
  if (!r) return null;
  if (L && node.lobbyConns.get(id) === L) admit(node, L, r.pass, "allowed");
  node.emit("joinrequests", waitingJoins(node));
  return r.req;
}
export function denyJoin(node, id = node.gate?.lobby[0]?.id) {
  const L = node.lobbyConns.get(id);
  const r = id ? deny(node.gate, id) : null;
  if (!r) return null;
  if (L) refuse(node, L, "The host didn't let this device in.");
  node.emit("joinrequests", waitingJoins(node));
  return r;
}
export const waitingJoins = (node) => (node.gate ? node.gate.lobby.map((r) => ({ ...r, line: requestLine(r.name, r.meta) })) : []);

// ---------------- device ----------------
// the hello a device sends the host (never to another device): can wait, and what lets it in
export function joinHelloFields(node) {
  return { join: 1, ...(validKey(node.pass) ? { pass: node.pass } : {}), ...(validKey(node.key) ? { key: node.key } : {}) };
}
// the host's gate messages on a device -> true when handled. admission: "wait" (linked, the host has
// not answered) -> "lobby" (the host was asked) -> "in"
export function deviceGateMessage(node, d) {
  switch (d.t) {
    case "lobby":
      if (node.admission !== "in") {
        if (node.admission !== "lobby") node.log(`waiting for the host to let this device in${node.key ? "" : " (a room's invite link gets in without asking)"}`);
        node.admission = "lobby";
        node.emit("lobby");
      }
      return true;
    case "admit":
      if (typeof d.pass === "string" && validKey(d.pass)) node.pass = d.pass;
      if (node.admission !== "in") {
        const waited = node.admission === "lobby";
        node.admission = "in";
        if (waited) node.log("the host let this device in");
        node.emit("admitted", { waited });
      }
      return true;
  }
  return false;
}
// the host's hello on a device: a host from before the gate lets everyone in
export function onHostHello(node, d) {
  if (node.admission === "in") return;
  node.admission = d.gate ? "wait" : "in";
  if (!d.gate) node.emit("admitted", { waited: false, old: true });
}
