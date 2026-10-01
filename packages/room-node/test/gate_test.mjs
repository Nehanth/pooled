// The gate on the room node (gate.js): a host holds new links in its lobby until the invite key, a
// pass or the host's Allow lets them in; a device waits, keeps its pass, and walks straight into a
// host from before the gate. No GPU, no network: fake links.
import { test } from "node:test";
import assert from "node:assert/strict";
import { RoomNode } from "../roomnode.js";
import { hostGate } from "../gate.js";
import { PROTOCOL } from "../../../room/transport.js";

const tick = () => new Promise((r) => setTimeout(r, 20));

function hostNode({ ask = true } = {}) {
  const n = new RoomNode({ name: "host", pledgeGB: 8, log: () => {}, stripes: 0 });
  n.isHost = true; n.code = "4TKG9P"; n.meta = { webgpu: true, contribGB: 8, gpu: "fake" };
  n.peer = { id: "pooled-room-4TKG9P", connect: () => null, disconnected: false };
  n.gate = hostGate({ ask });
  return n;
}
// a link someone else opened to the host: what the host sends it, and messages delivered by hand
function incoming(n, peer, label = undefined) {
  const h = {}, sent = [];
  const conn = { peer, label, open: true, on: (ev, f) => { (h[ev] ||= []).push(f); }, send: (m) => sent.push(m), close() { this.closed = true; for (const f of h.close || []) f(); } };
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
  const s = incoming(n, "dev-f", "stripe");
  assert.equal(n.lobbyConns.get("dev-f").stripes.length, 1, "the stripe is held with its device");
  void s;
  L.sent.length = 0;
  L.say(hello("laptop", { join: 1 }));
  await tick();
  assert.equal(L.sent.find((m) => m.t === "admit")?.pass?.length, 22);
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
