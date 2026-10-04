// room.js dead links: a network frozen for longer than ICE's write timeout leaves a link whose
// RTCPeerConnection says "failed" while its data channels still say "open" (PeerJS never closes
// it). The room replaces such a link instead of waiting on it forever. room.js is DOM-bound, so
// room_src.js cuts the real functions out of its source and runs them over stubs here.
import { roomFns } from "./room_src.js";
import { joinerStart, AUTH_V } from "../../room/chanauth.js";
import { validKey } from "../../room/joingate.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// a PeerJS DataConnection with an RTCPeerConnection whose states the test sets
class FakePC extends EventTarget {
  constructor() { super(); this.connectionState = "connected"; this.iceConnectionState = "connected"; }
  set(conn, ice = this.iceConnectionState) {
    this.connectionState = conn; this.iceConnectionState = ice;
    this.dispatchEvent(new Event("iceconnectionstatechange"));
    this.dispatchEvent(new Event("connectionstatechange"));
  }
}
class FakeConn {
  constructor(peer, label) { this.peer = peer; this.label = label; this.open = true; this.peerConnection = new FakePC(); this.h = {}; this.sent = []; this.closed = 0; }
  on(ev, f) { (this.h[ev] ||= []).push(f); }
  emit(ev) { for (const f of this.h[ev] || []) f(); }
  send(d) { this.sent.push(d); }
  close() { this.closed++; if (this.open) { this.open = false; this.emit("close"); } }
}

const NAMES = ["watchLink", "linkDied", "relink", "retire", "chainLinkLost", "linkState", "noteLink", "linksUp", "helloFor", "linkOpened", "joinHello"];
function setup({ role = "host", chain = [], waiters = 0, relinkWait = 40, pass = "", key = "" } = {}) {
  const logs = [], dialed = [], wired = [], sentTo = [], failed = [];
  const conns = new Map();
  const ai = { role, engine: role === "host" ? {} : null, chain, chainNames: chain.map((id) => "n-" + id), waiters: new Map(), fed: [1, 2, 3], hostId: role === "worker" ? "pooled-room-ABCD" : null };
  for (let i = 0; i < waiters; i++) ai.waiters.set(i, {});
  const peer = { id: role === "host" ? "pooled-room-ABCD" : "me", destroyed: false, connect(id) { const c = new FakeConn(id); c.open = false; dialed.push(c); return c; } };
  const g = {
    ai, conns, peer, PROTOCOL: 4, PREFIX: "pooled-room-", roomCode: "ABCD", myName: "me", myMeta: {}, RELINK_WAIT_MS: relinkWait,
    isHost: role === "host", myPass: pass, joinKey: key,
    linksDown: new Map(), performance,
    log: (_f, t) => logs.push(t),
    wire: (c, name, meta, initiator, opts) => { const e = { conn: c, name, meta, initiator, stripes: [], hold: opts?.hold || null }; conns.set(c.peer, e); wired.push(e); return e; },
    meshKey: null, sendMesh: (c, role) => c.send({ t: "mesh", v: 1, none: 1, role }),
    AUTH_V, joinerStart, validKey, HOST_HELLO_WAIT_MS: 0, LEGACY_AUTH: true, toast: () => {},
    sendTo: (id, obj) => sentTo.push([id, obj]),
    failWaiters: (err) => { failed.push(err.message); ai.waiters.clear(); },
    ckptClear: () => {},
    aiStatus: () => {},
    setTimeout: (f, ms) => setTimeout(f, Math.min(ms, 60)),   // relink backoffs and the 20 s dial timeout, shortened
  };
  const fns = roomFns(NAMES, g);
  const entry = (id, initiator) => { const e = { conn: new FakeConn(id), name: "n-" + id, meta: {}, stripes: [new FakeConn(id, "stripe")], initiator }; conns.set(id, e); fns.watchLink(e.conn, () => fns.linkDied(e)); return e; };
  return { g, fns, ai, conns, logs, dialed, wired, sentTo, failed, entry };
}

Deno.test("relink: a link whose peer connection fails is redialed by the side that dialed it", async () => {
  const t = setup({ role: "worker" });
  const e = t.entry("peerB", true);
  e.conn.peerConnection.set("disconnected", "disconnected");
  eq(t.dialed.length, 0, "a link that is only disconnected is left alone (it may come back)");
  e.conn.peerConnection.set("failed", "disconnected");   // Chrome: ICE stays "disconnected", the connection says "failed"
  e.conn.peerConnection.set("failed", "disconnected");   // a repeated event does not dial twice
  eq(t.dialed.length, 1, "one new connection dialed");
  eq(t.dialed[0].peer, "peerB");
  const c = t.dialed[0]; c.open = true; c.emit("open");
  eq(t.wired.length, 1, "the new connection is wired in");
  ok(t.wired[0].initiator, "as the dialing side again (it opens the stripes)");
  const hello = c.sent.find((m) => m.t === "hello");
  ok(hello && hello.back === 1 && hello.v === 4, "and says hello as a device coming back");
  ok(!("pass" in hello) && !("join" in hello), "to another device: no pass");
});

Deno.test("relink: the redial to the host proves the pass the host gave this device (never sends it), and shows nothing to anyone else", async () => {
  const P = "AbCdEfGhIjKlMnOpQrStUv", K = "ZyXwVuTsRqPoNmLkJiHgFe";
  const t = setup({ role: "worker", pass: P, key: K });
  const h = t.entry("pooled-room-ABCD", true);
  h.conn.peerConnection.set("failed");
  const c = t.dialed[0]; c.open = true; c.emit("open");
  for (let i = 0; i < 50 && !c.sent.some((m) => m.t === "hello"); i++) await sleep(10);
  const hello = c.sent.find((m) => m.t === "hello");
  eq([hello.join, hello.pass, hello.key, hello.back, hello.kc, hello.auth], [1, undefined, undefined, 1, 1, 1], "what it can prove, so the host lets it back in without asking");
  ok(/^[0-9a-f]{32}$/.test(hello.pid) && /^[0-9a-f]{64}$/.test(hello.jc), JSON.stringify(hello));
  eq(t.wired[0].hold, "host", "held until the host proves itself back");
  const o = t.entry("peerC", true);
  o.conn.peerConnection.set("failed");
  const c2 = t.dialed[1]; c2.open = true; c2.emit("open");
  const h2 = c2.sent.find((m) => m.t === "hello");
  ok(!("pass" in h2) && !("key" in h2) && !("join" in h2), JSON.stringify(h2));
});

Deno.test("relink: a dial that does not open is retried, and given up after a few tries (the link then closes)", async () => {
  const t = setup({ role: "worker" });
  const e = t.entry("peerB", true);
  e.conn.peerConnection.set("failed");
  for (let i = 0; i < 20 && !e.conn.closed; i++) await sleep(80);
  ok(t.dialed.length >= 2, `retried (${t.dialed.length} dials)`);
  ok(t.dialed.length <= 8, "a bounded number of times");
  eq(e.conn.closed, 1, "then the dead link is closed, which is the ordinary 'device left' path");
});

Deno.test("relink: the side that was dialed waits for the redial, then closes a link nobody replaced", async () => {
  const t = setup({ role: "worker", relinkWait: 30 });
  const a = t.entry("peerA", false);
  a.conn.peerConnection.set("failed");
  eq(t.dialed.length, 0, "it does not dial itself");
  const b = t.entry("peerB", false);
  b.conn.peerConnection.set("failed");
  // peerB's side redials: its new link replaces the entry before the wait runs out
  t.conns.set("peerB", { conn: new FakeConn("peerB"), name: "n-peerB", stripes: [] });
  await sleep(120);
  eq(a.conn.closed, 1, "peerA never came back: its link is closed");
  eq(b.conn.closed, 0, "peerB's old link is not closed by the wait (wire() retires it)");
});

Deno.test("relink: a closed link is not a failed one", () => {
  const t = setup({ role: "worker" });
  const e = t.entry("peerB", true);
  e.conn.open = false;
  e.conn.peerConnection.set("failed");
  eq(t.dialed.length, 0);
});

Deno.test("relink: retire closes the old link and its stripes", () => {
  const t = setup({ role: "worker" });
  const e = t.entry("peerB", true);
  t.conns.set("peerB", { conn: new FakeConn("peerB") });   // already replaced
  t.fns.retire(e);
  eq(e.conn.closed, 1); eq(e.stripes[0].closed, 1);
});

Deno.test("relink: the host fails the laps in flight on a chain link, waits for it before the next question", async () => {
  const t = setup({ role: "host", chain: ["g1", "g2"], waiters: 2 });
  const e = t.entry("g1", false);
  e.conn.peerConnection.set("failed");
  eq(t.failed.length, 1, "the laps in flight fail now, not after the lap timeout");
  ok(/g1.*dropped/.test(t.failed[0]), t.failed[0]);
  eq(t.ai.fed, null, "the next question prefills from scratch");
  eq(t.g.linksDown.size, 1, "the link is marked down");
  let waited = false;
  const w = t.fns.linksUp().then(() => { waited = true; });
  await sleep(30);
  ok(!waited, "a question waits while a chain link is down");
  // g1 redials: wire() swaps in the new link and retires the old one
  t.conns.set("g1", { conn: new FakeConn("g1"), name: "n-g1", stripes: [] });
  t.fns.retire(e);
  eq(t.g.linksDown.size, 0, "back up");
  await w; ok(waited);
});

Deno.test("relink: the host ignores its links to devices outside the chain", () => {
  const t = setup({ role: "host", chain: ["g1"], waiters: 1 });
  const e = t.entry("guestAskOnly", false);
  e.conn.peerConnection.set("failed");
  eq(t.failed.length, 0); eq(t.g.linksDown.size, 0);
});

Deno.test("relink: a worker tells the host about its links to other workers, down and back up, never about the host link", () => {
  const t = setup({ role: "worker" });
  const e = t.entry("peerB", true);
  const h = t.entry("pooled-room-ABCD", true);
  e.conn.peerConnection.set("failed");
  h.conn.peerConnection.set("failed");
  eq(t.sentTo, [["pooled-room-ABCD", { t: "ai-linklost", name: "n-peerB", up: 0 }]]);
  t.conns.set("peerB", { conn: new FakeConn("peerB") });
  t.fns.retire(e);
  eq(t.sentTo[1], ["pooled-room-ABCD", { t: "ai-linklost", name: "n-peerB", up: 1 }]);
});

Deno.test("relink: linksUp gives up after RELINK_WAIT_MS so a question is never stuck behind a link that never comes back", async () => {
  const t = setup({ role: "host", chain: ["g1"], relinkWait: 50 });
  t.g.linksDown.set("x|g1", { name: "g1", at: performance.now() });
  const t0 = performance.now();
  await t.fns.linksUp();
  ok(performance.now() - t0 < 1000);
});

Deno.test("relink: while cut off from signaling PeerJS dials nothing; relink keeps trying instead of throwing", async () => {
  const t = setup({ role: "worker" });
  const e = t.entry("peerB", true);
  let calls = 0;
  t.g.peer.connect = () => { calls++; return undefined; };
  e.conn.peerConnection.set("failed");
  for (let i = 0; i < 20 && !e.conn.closed; i++) await sleep(80);
  ok(calls >= 2, `tried again (${calls})`);
  eq(e.conn.closed, 1, "and gave up in the end");
});
