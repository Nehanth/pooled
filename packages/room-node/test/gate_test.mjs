// The gate on the room node (gate.js): a host holds new links in its lobby until the invite key, a
// pass or the host's Allow lets them in; a device waits, keeps its pass, and walks straight into a
// host from before the gate. No GPU, no network: fake links.
import { test } from "node:test";
import assert from "node:assert/strict";
import { RoomNode } from "../roomnode.js";
import { hostGate, saveGate, joinHello, deviceGateMessage } from "../gate.js";
import { joinerStart, joinerProof, meshProof, linkFingerprints } from "../../../room/chanauth.js";
import { PROTOCOL } from "../../../room/transport.js";

const tick = () => new Promise((r) => setTimeout(r, 20));

function hostNode({ ask = true, legacy = true } = {}) {
  const n = new RoomNode({ name: "host", pledgeGB: 8, log: () => {}, stripes: 0 });
  n.isHost = true; n.code = "4TKG9P"; n.meta = { webgpu: true, contribGB: 8, gpu: "fake" };
  n.peer = { id: "pooled-room-4TKG9P", connect: () => null, disconnected: false };
  n.gate = hostGate({ ask, legacy });
  n.mk = n.gate.mk;   // as createRoom
  return n;
}
// a link someone else opened to the host: what the host sends it, and messages delivered by hand
// a fake certificate fingerprint pair, as an RTCPeerConnection's descriptions show them
const sdp = (c) => `v=0\r\na=fingerprint:sha-256 ${Array(32).fill(c).join(":")}\r\n`;
const pcOf = (mine, theirs) => ({ localDescription: { sdp: sdp(mine) }, remoteDescription: { sdp: sdp(theirs) } });
function incoming(n, peer, label = undefined, pc = pcOf("BB", "AA")) {
  const h = {}, sent = [];
  const conn = { peer, label, open: true, peerConnection: pc, on: (ev, f) => { (h[ev] ||= []).push(f); }, send: (m) => sent.push(m), close() { this.closed = true; for (const f of h.close || []) f(); } };
  n.accept(conn);
  for (const f of h.open || []) f();
  return { conn, sent, say: (m) => { for (const f of h.data || []) f(m); } };
}
const hello = (name, extra = {}) => ({ t: "hello", name, v: PROTOCOL, meta: { webgpu: true, contribGB: 4, ua: "Mac" }, ...extra });
const types = (L) => L.sent.map((m) => m.t);

test("host: its hello says it gates; a device with the invite key is let in at once and given a pass", async () => {
  const n = hostNode();
  const L = incoming(n, "dev-a");
  assert.equal(L.sent[0].t, "hello"); assert.equal(L.sent[0].gate, 1); assert.equal(L.sent[0].ask, 1);
  assert.ok(n.inviteFragment.startsWith("#k="));
  L.say(hello("mac", { join: 1, key: n.gate.key }));
  await tick();
  const admit = L.sent.find((m) => m.t === "admit");
  assert.ok(admit?.pass && admit.pass.length >= 22, "a pass to come back with");
  assert.ok(n.conns.has("dev-a") && n.roster.get("dev-a")?.name === "mac", "in the room");
  // back later with that pass (a reload, a dropped link): in without asking
  const L2 = incoming(n, "dev-a2");
  L2.say(hello("mac", { join: 1, pass: admit.pass, back: 1 }));
  await tick();
  assert.ok(types(L2).includes("admit")); assert.ok(n.conns.has("dev-a2"));
});

test("host: without a key a device waits in the lobby, sees nothing of the room, then Allow lets it in", async () => {
  const n = hostNode();
  const reqs = [];
  n.on("joinrequest", (r) => reqs.push(r));
  const L = incoming(n, "dev-b");
  L.say(hello("otter", { join: 1, key: "not-the-key-AAAAAAAAAAAAAA" }));
  await tick();
  assert.deepEqual(types(L), ["hello", "lobby"]);
  assert.equal(reqs[0]?.line, "otter wants to join (Mac, 4 GB)");
  assert.ok(!n.conns.has("dev-b") && !n.roster.has("dev-b"), "not in the room");
  L.say({ t: "ping", ts: 5 });
  L.say({ t: "pledge", gb: 6 });
  assert.equal(L.sent.at(-1).t, "pong", "pings answered");
  n.broadcast({ t: "roster", members: [] });
  assert.ok(!types(L).includes("roster"), "the room's messages don't reach it");
  assert.equal(n.waitingJoins().length, 1);
  const who = await n.allowJoin();
  assert.equal(who?.name, "otter");
  assert.ok(types(L).includes("admit"));
  assert.ok(n.conns.has("dev-b"));
  assert.equal(n.roster.get("dev-b").meta.contribGB, 6, "what it sent while waiting is handled once it is in");
  assert.equal(n.waitingJoins().length, 0);
});

test("host: Deny turns it away and its knocks after that too; an older device is told to update; API clients off", async () => {
  const n = hostNode();
  const L = incoming(n, "dev-c");
  L.say(hello("stranger", { join: 1 }));
  await tick();
  assert.equal(n.denyJoin()?.name, "stranger");
  assert.equal(L.sent.at(-1).t, "bye"); assert.match(L.sent.at(-1).reason, /didn't let this device in/);
  const again = incoming(n, "dev-c");
  again.say(hello("stranger", { join: 1 }));
  await tick();
  assert.equal(again.sent.at(-1).t, "bye");
  // a pooled join from before the gate: no join: 1
  const old = incoming(n, "dev-d");
  old.say({ ...hello("old"), meta: { webgpu: true, native: "node-dawn" } });
  await tick();
  assert.match(old.sent.at(-1).reason, /too old to wait.*npx @pooled\/cli@latest/);
  // a device leaving the lobby withdraws its request
  const gone = incoming(n, "dev-e");
  gone.say(hello("quitter", { join: 1 }));
  await tick();
  assert.equal(n.waitingJoins().length, 1);
  gone.conn.close();
  assert.equal(n.waitingJoins().length, 0);
  // API clients while the host doesn't allow them: refused before the host is asked
  n.allowApi = false;
  const api = incoming(n, "api-1");
  api.say({ t: "hello", name: "pooled chat", v: PROTOCOL, join: 1, meta: { api: 1, webgpu: false, ua: "API" } });
  await tick();
  assert.match(api.sent.at(-1).reason, /does not allow API clients/);
});

test("host --allow-all: anyone with the code comes in (a device that can keep one gets a pass); stripes of a held device wait with it", async () => {
  const n = hostNode({ ask: false });
  const L = incoming(n, "dev-f");
  const s = incoming(n, "dev-f", "stripe", null);
  assert.equal(n.lobbyConns.get("dev-f").stripes.length, 1, "the stripe is held with its device");
  void s;
  L.sent.length = 0;
  L.say(hello("laptop", { join: 1 }));
  await tick();
  assert.equal(L.sent.find((m) => m.t === "admit")?.pass?.length, 22);
  // (a device from before the proofs: its stripe proves nothing, and is taken under the legacy rule)
  await new Promise((r) => setTimeout(r, 50));
  assert.equal(n.conns.get("dev-f").stripes.length, 1, "attached once it is in");
  const old = incoming(n, "tab-old");
  old.say(hello("old tab"));
  await tick();
  assert.ok(n.conns.has("tab-old"), "an older tab is let in when the host doesn't ask");
});

test("device: its hello carries join, the key and the pass (only to the host); it waits in the lobby; an older host lets it in", () => {
  const w = new RoomNode({ name: "mac", pledgeGB: 8, log: () => {}, key: "AbCdEfGhIjKlMnOpQrStUv" });
  w.code = "4TKG9P"; w.meta = { webgpu: true };
  w.peer = { id: "me" };
  w.conns.set("pooled-room-4TKG9P", { conn: { close() {}, send() {} }, name: "host", meta: {}, stripes: [], seen: performance.now(), missed: 0 });
  const ev = [];
  for (const k of ["lobby", "admitted"]) w.on(k, (x) => ev.push(k));
  w.onData("pooled-room-4TKG9P", { t: "hello", name: "host", v: PROTOCOL, gate: 1, ask: 1, meta: { api: 2 } });
  assert.equal(w.admission, "wait");
  w.onData("pooled-room-4TKG9P", { t: "lobby" });
  assert.equal(w.admission, "lobby");
  w.onData("someone-else", { t: "admit", pass: "ZZZZZZZZZZZZZZZZZZZZZZ" });
  assert.equal(w.admission, "lobby", "only the host lets it in");
  w.onData("pooled-room-4TKG9P", { t: "admit", pass: "PpPpPpPpPpPpPpPpPpPpPp" });
  assert.equal(w.admission, "in"); assert.equal(w.pass, "PpPpPpPpPpPpPpPpPpPpPp");
  assert.deepEqual(ev, ["lobby", "admitted"]);
  const h = w.helloMsg({ back: 1, join: 1, key: w.key, pass: w.pass });
  assert.equal(h.key, w.key); assert.equal(h.pass, w.pass);
  assert.equal(w.helloMsg().key, undefined, "a plain hello (to other devices) carries neither");
  // a host from before the gate: in at once
  const o = new RoomNode({ name: "mac", pledgeGB: 8, log: () => {} });
  o.code = "ABCD"; o.peer = { id: "me" };
  o.conns.set("pooled-room-ABCD", { conn: { close() {}, send() {} }, name: "host", meta: {}, stripes: [], seen: performance.now(), missed: 0 });
  o.onData("pooled-room-ABCD", { t: "hello", name: "host", v: PROTOCOL, meta: { api: 2 } });
  assert.equal(o.admission, "in");
});

test("host restarted with its saved gate: the old invite key and the passes it gave out still let devices in", async () => {
  const a = hostNode();
  const L = incoming(a, "dev-a");
  L.say(hello("mac", { join: 1, key: a.gate.key }));
  await tick();
  const pass = L.sent.find((m) => m.t === "admit").pass;
  const saved = JSON.parse(JSON.stringify(saveGate(a.gate)));
  assert.ok(!JSON.stringify(saved).includes(pass), "only the pass's hash is kept");
  // a new process: same key, the pass comes back in without asking; ask follows today's setting
  const b = hostNode();
  b.gate = hostGate({ ask: true, saved });
  assert.equal(b.gate.key, a.gate.key);
  const back = incoming(b, "dev-a2");
  back.say(hello("mac", { join: 1, pass }));
  await tick();
  assert.ok(types(back).includes("admit") && !types(back).includes("lobby"));
  const c = hostGate({ ask: false, saved });
  assert.equal(c.ask, false);
  assert.equal(hostGate({ saved: null }).key.length >= 22, true, "no saved gate: a new key");
});

// ---- proofs instead of raw secrets (room/chanauth.js) ----
const until = async (f, ms = 2000) => { const t0 = Date.now(); while (!f()) { if (Date.now() - t0 > ms) throw new Error("timed out"); await tick(); } };

test("host: a device that speaks the proofs is challenged, proves the key bound to its link, and gets the host's proof and the mesh key", async () => {
  const n = hostNode();
  const L = incoming(n, "dev-p");
  const st = await joinerStart({ key: n.gate.key });
  L.say(hello("mac", { join: 1, ...st.helloFields }));
  await until(() => types(L).includes("auth"));
  assert.ok(!types(L).includes("admit"), "nothing before the proof");
  const chal = L.sent.find((m) => m.t === "auth");
  // the device's end of the same link: the fingerprints the other way round
  L.say(await joinerProof(st, chal, { fps: linkFingerprints(pcOf("AA", "BB")), me: "dev-p", host: n.peer.id }));
  await until(() => types(L).includes("admit"));
  const admit = L.sent.find((m) => m.t === "admit");
  assert.equal(admit.via, "key"); assert.match(admit.hp, /^[0-9a-f]{64}$/); assert.equal(admit.mk, n.mk);
  assert.equal(n.roster.get("dev-p")?.a, 1, "the roster marks it as one that proves its links");
});

test("host: the same proof from a link with someone in the middle waits for Allow; a host that refuses raw keys ignores an old hello's key", async () => {
  const n = hostNode();
  const reqs = [];
  n.on("joinrequest", (r) => reqs.push(r));
  const L = incoming(n, "dev-m", undefined, pcOf("BB", "C0"));   // the host's link ends at the attacker (C0)
  const st = await joinerStart({ key: n.gate.key });
  L.say(hello("mac", { join: 1, ...st.helloFields }));
  await until(() => types(L).includes("auth"));
  L.say(await joinerProof(st, L.sent.find((m) => m.t === "auth"), { fps: linkFingerprints(pcOf("AA", "D1")), me: "dev-m", host: n.peer.id }));
  await until(() => types(L).includes("lobby"));
  assert.ok(!types(L).includes("admit"));
  assert.ok(reqs[0].sas && reqs[0].sas !== st.sas, "different codes on the two screens");
  const strict = hostNode({ legacy: false });
  const sreqs = [];
  strict.on("joinrequest", (r) => sreqs.push(r));
  const O = incoming(strict, "old-dev");
  O.say(hello("old", { join: 1, key: strict.gate.key }));
  await until(() => types(O).includes("lobby"));
  assert.ok(!types(O).includes("admit"), "the raw key is not taken");
  assert.equal(sreqs[0]?.old, true, "the host is told it is an older Pooled that can't prove the link");
});

function deviceNode(opts = {}) {
  const w = new RoomNode({ name: "mac", pledgeGB: 8, log: () => {}, stripes: 0, ...opts });
  w.code = "4TKG9P"; w.meta = { webgpu: true };
  w.peer = { id: "me" };
  const sent = [];
  const conn = { peer: "pooled-room-4TKG9P", open: true, peerConnection: pcOf("AA", "BB"), on() {}, send: (m) => sent.push(m), close() { this.closed = true; } };
  w.wire(conn, "host", true, { hold: "host" });
  return { w, conn, sent };
}

test("device: no secret in its hello to a host that speaks the proofs; it checks the host's proof before anything from the room counts", async () => {
  const { w, conn, sent } = deviceNode({ key: "AbCdEfGhIjKlMnOpQrStUv", pass: "PpPpPpPpPpPpPpPpPpPpPp" });
  const evs = [];
  w.on("unverified", () => evs.push("unverified")); w.on("admitted", (x) => evs.push(x));
  w.onData("pooled-room-4TKG9P", { t: "hello", name: "host", v: PROTOCOL, gate: 1, ask: 1, auth: 1, meta: { api: 2 } });
  const f = await joinHello(w, conn);
  assert.equal(f.kc, 1); assert.match(f.pid, /^[0-9a-f]{32}$/); assert.match(f.jc, /^[0-9a-f]{64}$/);
  assert.equal(f.key, undefined); assert.equal(f.pass, undefined);
  w.onData("pooled-room-4TKG9P", { t: "auth", v: 1, hn: "HnHnHnHnHnHnHnHnHnHnHn" });
  await until(() => sent.some((m) => m.t === "auth-proof"));
  const p = sent.find((m) => m.t === "auth-proof");
  assert.ok(p.kp && p.pp && !JSON.stringify(p).includes(w.key));
  // anything else from the "host" waits; then an admit with a wrong proof: it leaves
  w.onData("pooled-room-4TKG9P", { t: "ai-ready-all", model: "x" });
  assert.equal(w.ai.online, false);
  w.onData("pooled-room-4TKG9P", { t: "admit", via: "key", hp: "0".repeat(64), mk: "M".repeat(43) });
  await until(() => evs.length);
  assert.deepEqual(evs, ["unverified"]); assert.ok(conn.closed); assert.equal(w.mk, null); assert.notEqual(w.admission, "in");
});

test("device: an older host with the gate never gets the raw key (it waits for Allow); only legacySend sends it", async () => {
  const a = deviceNode({ key: "AbCdEfGhIjKlMnOpQrStUv" });
  a.w.onData("pooled-room-4TKG9P", { t: "hello", name: "host", v: PROTOCOL, gate: 1, ask: 1, meta: {} });
  const fa = await joinHello(a.w, a.conn);
  assert.equal(fa.key, undefined); assert.equal(fa.kc, undefined, "nothing to prove to it either: it waits for Allow");
  const b = deviceNode({ key: "AbCdEfGhIjKlMnOpQrStUv", legacySend: true });
  b.w.onData("pooled-room-4TKG9P", { t: "hello", name: "host", v: PROTOCOL, gate: 1, ask: 1, meta: {} });
  const fb = await joinHello(b.w, b.conn);
  assert.equal(fb.key, "AbCdEfGhIjKlMnOpQrStUv", "the unsafe opt-in");
});

test("device: a key that didn't check out, then the host's Allow after the lobby: in, trust on first use", async () => {
  const { w, conn, sent } = deviceNode({ key: "AbCdEfGhIjKlMnOpQrStUv" });
  w.onData("pooled-room-4TKG9P", { t: "hello", name: "host", v: PROTOCOL, gate: 1, ask: 1, auth: 1, meta: {} });
  await joinHello(w, conn);
  w.onData("pooled-room-4TKG9P", { t: "auth", v: 1, hn: "HnHnHnHnHnHnHnHnHnHnHn" });
  await until(() => sent.some((m) => m.t === "auth-proof"));
  w.onData("pooled-room-4TKG9P", { t: "lobby" });
  assert.equal(w.admission, "lobby");
  w.onData("pooled-room-4TKG9P", { t: "admit", pass: "QqQqQqQqQqQqQqQqQqQqQq", mk: "M".repeat(43) });
  await until(() => w.admission === "in");
  assert.equal(w.mk, "M".repeat(43)); assert.ok(!w.conns.get("pooled-room-4TKG9P").hold, "the link is released");
});

test("mesh: a link from another device is held until it proves the room's key; a wrong proof closes it", async () => {
  const w = new RoomNode({ name: "w", pledgeGB: 8, log: () => {}, stripes: 0 });
  w.code = "4TKG9P"; w.meta = {}; w.peer = { id: "me", connect: () => null };
  w.mk = "K".repeat(43);
  const mk = (peer, pc) => { const h = {}, sent = []; const conn = { peer, open: true, peerConnection: pc, on: (ev, f) => { (h[ev] ||= []).push(f); }, send: (m) => sent.push(m), close() { this.closed = true; for (const f of h.close || []) f(); } }; w.accept(conn); for (const f of h.open || []) f(); return { conn, sent, say: (m) => { for (const f of h.data || []) f(m); } }; };
  const good = mk("dev-g", pcOf("AA", "BB"));
  good.say(hello("g"));
  good.say({ t: "ai-next", next: "elsewhere" });
  assert.ok(w.conns.get("dev-g").hold, "held");
  assert.notEqual(w.ai.next, "elsewhere", "nothing counts before the proof");
  const p = await meshProof(w.mk, { fps: linkFingerprints(pcOf("BB", "AA")), dialer: "dev-g", acceptor: "me", role: "dial" });
  good.say({ t: "mesh", v: 1, p });
  await until(() => !w.conns.get("dev-g")?.hold);
  assert.ok(good.sent.some((m) => m.t === "mesh" && m.p), "it proved its own end too");
  const bad = mk("dev-x", pcOf("AA", "C0"));
  bad.say({ t: "mesh", v: 1, p });   // a proof from another link
  await until(() => bad.conn.closed);
  assert.ok(!w.conns.has("dev-x"));
});

test("host: a newer link under the same id withdraws the request on screen (Allow answers the link it showed)", async () => {
  const n = hostNode();
  const L = incoming(n, "dev-r");
  L.say(hello("first", { join: 1 }));
  await until(() => n.waitingJoins().length === 1);
  incoming(n, "dev-r");   // same id, a new link, no hello yet
  assert.equal(n.waitingJoins().length, 0, "the old request is gone");
});

test("device: holding a key, a \"host\" without the gate lets it in on trust, with a warning", () => {
  const { w } = deviceNode({ key: "AbCdEfGhIjKlMnOpQrStUv" });
  const ev = [];
  w.on("tofu", () => ev.push("tofu")); w.on("unverified", () => ev.push("unverified"));
  w.onData("pooled-room-4TKG9P", { t: "hello", name: "host", v: PROTOCOL, meta: {} });
  assert.deepEqual(ev, ["tofu"]); assert.equal(w.admission, "in");
});

test("device: a host that proves nothing back lets it in on trust (warned), with or without the lobby", async () => {
  for (const lobby of [false, true]) {
    const { w, conn, sent } = deviceNode({ key: "AbCdEfGhIjKlMnOpQrStUv" });
    const ev = [];
    w.on("tofu", () => ev.push("tofu")); w.on("unverified", () => ev.push("unverified"));
    w.onData("pooled-room-4TKG9P", { t: "hello", name: "host", v: PROTOCOL, gate: 1, ask: 1, auth: 1, meta: {} });
    await joinHello(w, conn);
    w.onData("pooled-room-4TKG9P", { t: "auth", v: 1, hn: "HnHnHnHnHnHnHnHnHnHnHn" });
    await until(() => sent.some((m) => m.t === "auth-proof"));
    if (lobby) w.onData("pooled-room-4TKG9P", { t: "lobby" });
    w.onData("pooled-room-4TKG9P", { t: "admit", pass: "QqQqQqQqQqQqQqQqQqQqQq" });
    await until(() => w.admission === "in");
    assert.deepEqual(ev, ["tofu"]); assert.ok(!w.authFailed);
  }
});
