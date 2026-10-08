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
// The key and the pass never cross the link (room/chanauth.js): the device's hello says which it
// holds (auth: 1, a commitment to its nonce), the host answers with its own nonce ("auth"), the device
// sends HMAC proofs bound to the link's DTLS fingerprints ("auth-proof"), and the host's admit carries
// its proof back (hp), which the device checks before it takes anything from the room. A device let
// in by Allow has nothing to prove: both sides can show six digits (sas) that match only when nobody
// sits in the middle of the link.
//
// Device (a node made by joinRoom): its hello to the host carries join: 1 (it can wait in the lobby)
// and what it can prove. It never sends the raw key or pass: a host with the gate but from before the
// proofs (no auth: 1) can't check them, so the device waits for that host's Allow (legacySend, off by
// default and unsafe, restores the old way). A host from before the gate lets it in at once, as before.
import { makeGate, decide, decideAuth, startAuth, hostWantsAuth, enqueue, allow, deny, withdraw, requestLine, validKey, keyFragment,
  randomCode, formatCode, parseCode, CODE_LEN, OLD_TAB_TEXT, saveGate } from "../../room/joingate.js";
import { AUTH_V, linkFingerprints, joinerStart, joinerProof, joinerCheckAdmit, validMeshKey, meshProof, meshCheck } from "../../room/chanauth.js";

export { keyFragment, formatCode, parseCode, randomCode, CODE_LEN, validKey, saveGate };
const LOBBY_BUF = 64;

// ---------------- host ----------------
// the host's gate: ask = hold new devices until allowed (pooled host's default); key: the room's
// invite key (a new random one by default); saved: a gate kept from before (saveGate: its invite key
// and the passes it gave out), so a host that restarts keeps its links and lets its devices back in.
// ask is the host's current setting, not the saved one
// legacy: let in a device from before the proofs that sends the raw key or pass (with a warning)
export function hostGate({ ask = true, key = null, saved = null, legacy = true } = {}) {
  if (saved && typeof saved === "object") return makeGate({ ask, legacy, key: key || saved.key, passes: saved.passes, mk: saved.mk });
  return makeGate({ ask, key, legacy });
}

// what a host's hello adds (docs/protocol.md): this host holds new devices at the gate, and proves
// the key or pass back instead of reading it (auth)
export const gateHelloFields = (g) => (g ? { gate: 1, ask: g.ask ? 1 : 0, auth: AUTH_V } : {});

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
  if (prev) {
    try { prev.conn.close(); } catch {}
    // its request goes too: an allowJoin must answer the link it showed (the new link asks again)
    if (withdraw(node.gate, id)) node.emit("joinrequests", waitingJoins(node));
  }
  const L = { conn, hello: null, buf: [], stripes: [], in: false, auth: null };
  node.lobbyConns.set(id, L);
  const broke = (err) => { node.log(`gate: ${err.message}`); refuse(node, L, "The host couldn't check this device. Try again."); };
  conn.on("data", (d) => {
    if (L.in || node.lobbyConns.get(id) !== L) return;
    if (!d || typeof d.t !== "string") return;   // binary (a bandwidth test): nothing a waiting device sends
    if (d.t === "hello" && !L.hello) {
      L.hello = d;
      gateHello(node, L, d).catch(broke);
      return;
    }
    // its proofs, for the challenge this host sent (once)
    if (d.t === "auth-proof" && L.auth) {
      const p = L.auth; L.auth = null;
      gateProof(node, L, p, d).catch(broke);
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
  // it speaks the proofs: challenge it, and decide on its answer (gateProof)
  if (hostWantsAuth(d)) {
    const { pending, msg } = startAuth(node.gate, d);
    L.auth = pending;
    try { L.conn.send(msg); } catch {}
    return;
  }
  const r = await decide(node.gate, id, d);
  if (node.lobbyConns.get(id) !== L || node.closing) return;   // it left (or a newer link took over) while the hash ran
  if (r.legacy) {
    node.log(`${name} came in with its ${r.via === "key" ? "invite key" : "pass"} sent the old way (an older Pooled): ask it to update`);
    node.emit("legacyjoin", { id, name, via: r.via });
  }
  gateAnswer(node, L, d, name, r);
}
// the device's proofs -> in (with the host's proof back), held for Allow (with the six digits), or a bye
async function gateProof(node, L, pending, proof) {
  const id = L.conn.peer, d = L.hello;
  const name = String(d.name ?? "").replace(/[\u0000-\u001f\u007f<>"'`&]/g, "").trim().slice(0, 40) || id.slice(0, 8);
  const r = await decideAuth(node.gate, id, d, pending, proof, { fps: linkFingerprints(L.conn), me: node.peer?.id, peer: id });
  if (node.lobbyConns.get(id) !== L || node.closing) return;
  if (r.failed) node.log(`${name}: its invite link or pass didn't check out on this link (an old link, or a link someone sits in the middle of)`);
  gateAnswer(node, L, d, name, r);
}
function gateAnswer(node, L, d, name, r) {
  const id = L.conn.peer;
  if (r.kind === "admit") { admit(node, L, r.pass, r.via, r.hp); return; }
  if (r.kind === "refuse") {
    // an older device: say how to fix it in its own terms (a pooled join / serve from before the gate)
    const why = r.reason === OLD_TAB_TEXT && (d.meta?.api || d.meta?.native === "node-dawn")
      ? "This room's host asks before new devices join, and this pooled is too old to wait for that. Update it (npx @pooled/cli@latest) or ask the host for the invite link."
      : r.reason;
    refuse(node, L, why);
    return;
  }
  const req = enqueue(node.gate, id, name, d.meta, Date.now(), r.sas, !!r.oldRaw);
  try { L.conn.send({ t: "lobby" }); } catch {}
  // a device from before the proofs that came with the raw invite key, which this host doesn't take
  if (r.oldRaw) node.log(`${name} runs an older Pooled and can't prove the invite key: it waits for Allow (ask it to update)`);
  if (!node.listenerCount("joinrequest")) node.log(`${requestLine(name, d.meta)}${r.sas ? ` (check code ${r.sas})` : ""}: allowJoin() lets it in, denyJoin() turns it away`);
  node.emit("joinrequest", { id, name, meta: d.meta || {}, line: requestLine(name, d.meta), at: req.at, sas: req.sas, ...(r.oldRaw ? { old: true } : {}) });
  node.emit("joinrequests", waitingJoins(node));
}

// in: the admit carries the pass to come back with, the host's proof for the secret the device proved
// (via + hp), and the room's mesh key for its links to the other devices
function admit(node, L, pass, via, hp = null) {
  const conn = L.conn, id = conn.peer;
  node.lobbyConns.delete(id);
  L.in = true;
  // (an API client never links to other devices: no mesh key for it, so one the host disconnects can't dial them)
  const mk = node.gate?.mk && !L.hello?.meta?.api ? { mk: node.gate.mk } : {};
  if (via !== "version") try { conn.send({ t: "admit", ...(pass ? { pass } : {}), ...(hp ? { via, hp } : {}), ...mk }); } catch {}
  if (via === "key" || via === "allowed") node.log(`${String(L.hello?.name || id).slice(0, 40)} ${via === "key" ? "came in with the invite link" : "was let in"}`);
  const e = node.wire(conn);
  node.onData(id, L.hello);   // first: the roster then says whether it proves its links (its stripes do too)
  for (const s of L.stripes) node.attachStripe(e, s);
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
const HOST_HELLO_WAIT_MS = 2000;
// the hello a device sends the host (never to another device): it can wait in the lobby, and what it
// can prove. Holding a key or a pass, it first waits (briefly) for the host's hello, which says
// whether the host speaks the proofs. A gated host from before them gets nothing it can't check: the
// device waits for its Allow. Only node.legacySend (POOLED_LEGACY_AUTH=1: unsafe, anyone in the middle
// of signaling can pose as such a host) sends it the raw key or pass. -> the fields to add to the hello
export async function joinHello(node, conn) {
  const key = validKey(node.key) ? node.key : null, pass = validKey(node.pass) ? node.pass : null;
  const e = node.conns.get(conn.peer);
  if (key || pass) {
    const t0 = Date.now();
    while (e && !e.hostHello && Date.now() - t0 < HOST_HELLO_WAIT_MS && conn.open !== false) await new Promise((r) => setTimeout(r, 25));
  }
  const hh = e?.hostHello;
  if (hh && hh.gate && !(+hh.auth >= AUTH_V) && (key || pass)) {
    node.emit("legacyhost");
    if (node.legacySend === true) {
      node.log("this room's host runs an older Pooled: the invite key or pass goes to it the old way, unprotected (POOLED_LEGACY_AUTH=1)");
      return { join: 1, ...(pass ? { pass } : {}), ...(key ? { key } : {}) };
    }
    node.log("this room's host runs an older Pooled: ask them to let you in, or to update. (This device doesn't send it the invite key: it can't check it safely)");
    if (e) e.jauth = await joinerStart({});
    return { join: 1, ...(e?.jauth?.helloFields || {}) };
  }
  const st = await joinerStart({ key, pass });
  if (e) e.jauth = st;
  return { join: 1, ...st.helloFields };
}
// the host's gate messages on a device -> true when handled. admission: "wait" (linked, the host has
// not answered) -> "lobby" (the host was asked) -> "in"
export function deviceGateMessage(node, d, from = null) {
  const e = from ? node.conns.get(from) : null;
  switch (d.t) {
    case "auth": {   // the host's challenge: prove what this device holds, bound to this link
      const st = e?.jauth;
      if (!st || st.hn) return true;
      joinerProof(st, d, { fps: linkFingerprints(e.conn), me: node.peer?.id, host: from }).then((m) => {
        if (m && node.conns.get(from) === e) try { e.conn.send(m); } catch {}
      }).catch((err) => node.log(`auth: ${err.message}`));
      return true;
    }
    case "lobby":
      if (e?.jauth) e.jauth.lobbied = true;
      if (node.admission !== "in") {
        const sas = e?.jauth?.sas;
        if (node.admission !== "lobby") node.log(`waiting for the host to let this device in${node.key ? "" : " (a room's invite link gets in without asking)"}${sas ? `; the host sees code ${sas} beside this request` : ""}`);
        node.admission = "lobby";
        node.emit("lobby", { sas: sas || null });
      }
      return true;
    case "admit":
      // what the host sends next (ai-ready-all comes right behind admit) waits behind it while its proof
      // is checked (roomnode.js onData), and is handled in order once this device is in (release)
      if (e?.hold) e.hold.admitting = true;
      admitted(node, e, from, d).catch((err) => {
        node.log(`admit: ${err.message}`);
        if (e?.hold) { e.hold.admitting = false; try { e.conn.close(); } catch {} }
      });
      return true;
  }
  return false;
}
// the host let this device in: when this device proved a secret, the host must prove it back before
// anything from the room counts (a host that can't is not the room's host, or the link has someone
// in the middle: the link closes)
async function admitted(node, e, from, d) {
  const st = e?.jauth;
  const v = st ? await joinerCheckAdmit(st, d) : "ok";
  if (v === "tofu") {
    node.log(`warning: couldn't verify this room's host: the invite link or pass didn't check out with it, and it let this device in anyway${st.sas ? ` (code ${st.sas})` : ""}. Fine after a host restart; otherwise someone may be in the middle of the link`);
    node.emit("tofu", { host: from, sas: st.sas || null });
  }
  if (v === "bad") {
    node.log("couldn't verify this room's host on this link: its proof of the invite key or pass is wrong (someone may be in the middle of the link). Leaving it");
    node.authFailed = true;
    node.emit("unverified", { host: from });
    if (e) { e.unverified = true; try { e.conn.close(); } catch {} }
    return;
  }
  if (e && node.conns.get(from) !== e) return;
  if (typeof d.pass === "string" && validKey(d.pass)) node.pass = d.pass;
  if (validMeshKey(d.mk)) node.mk = d.mk;
  if (e) { e.jauth = null; node.release?.(e); }
  if (node.admission !== "in") {
    const waited = node.admission === "lobby";
    node.admission = "in";
    if (waited) node.log("the host let this device in");
    node.emit("admitted", { waited, verified: v === "ok" && !!st?.proved });
  }
}
// the host's hello on a device: a host from before the gate lets everyone in
export function onHostHello(node, d, from = null) {
  const e = from ? node.conns.get(from) : null;
  if (e) e.hostHello = d;
  if (node.admission === "in") { if (!d.gate && e) node.release?.(e); return; }
  // a host from before the gate never got this device's key or pass, and can't prove it holds them:
  // trust on first use, said out loud
  if (!d.gate && (validKey(node.key) || validKey(node.pass))) {
    node.log("warning: couldn't verify this room's host: it runs a Pooled from before the gate and can't prove it holds the invite key");
    node.emit("tofu", { host: from, sas: null });
  }
  node.admission = d.gate ? "wait" : "in";
  if (!d.gate) { if (e) node.release?.(e); node.emit("admitted", { waited: false, old: true }); }
}

// ---------------- mesh links ----------------
// A link between two devices of the room (or a stripe of any link) is held until both ends prove the
// room's mesh key on it (chanauth.js meshProof). -> this end's { t: "mesh", ... } message
export async function meshHello(node, conn, role) {
  const dialer = role === "dial" ? node.peer?.id : conn.peer, acceptor = role === "dial" ? conn.peer : node.peer?.id;
  const p = node.mk ? await meshProof(node.mk, { fps: linkFingerprints(conn), dialer, acceptor, role }) : null;
  return p ? { t: "mesh", v: AUTH_V, p } : { t: "mesh", v: AUTH_V, none: 1 };
}
// the other end's mesh message (or its hello, for an end from before them) -> "ok" | "bad" | "legacy" | "wait"
//   legacy: let it in under the legacy rule (no mesh key here, or the host's roster lists it as a
//   device from before the proofs); wait: the roster doesn't list it yet
export async function meshVerdict(node, conn, role, d) {
  if (!node.mk) return "ok";   // a room whose host gave no mesh key: nothing to check (an older host)
  if (d?.t === "mesh" && d.p) {
    const theirs = role === "dial" ? "accept" : "dial";
    const dialer = role === "dial" ? node.peer?.id : conn.peer, acceptor = role === "dial" ? conn.peer : node.peer?.id;
    return (await meshCheck(node.mk, { fps: linkFingerprints(conn), dialer, acceptor, role: theirs }, d.p)) ? "ok" : "bad";
  }
  // no proof: only a device the host listed as one from before the proofs, while that is allowed
  if (node.legacyAuth === false) return "bad";
  const m = node.isHost ? (node.roster.has(conn.peer) ? { a: node.roster.get(conn.peer).a } : null) : (node.members || []).find((x) => x.id === conn.peer);
  if (!m) return "wait";
  return m.a ? "bad" : "legacy";
}
