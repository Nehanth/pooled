// Channel-bound proofs over real WebRTC on this machine, no GPU (a Dawn stand-in with no adapter): a
// gated room node host, room node devices and the `pooled serve` / `pooled chat` bridge (cli/lib/room.js)
// on a local signaling server (mitm_signal.mjs), first honest, then lying: it hands links to an
// attacker that relays every message between the two real ends.
//   honest: the invite key gets a device and the bridge in without asking, both checked the host's proof
//   back, the room's mesh key reached the devices, two devices prove it to each other, a stranger who
//   dials a device is closed, and a typed code waits with the same six digits on both sides;
//   in the middle: the relayed proofs don't let the bridge or a device in, the raw key never crosses,
//   the two sides show different digits; a device's link to another device is refused; and the old way
//   (the raw key in the hello) is what the attacker could use, unless the host refuses it.
// Skipped without packages/room-node's dev dependencies (ws, from the "peer" server): npm install
import { test, after } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

const HERE = path.resolve(new URL(".", import.meta.url).pathname);
const skip = !fs.existsSync(path.join(HERE, "../node_modules/ws")) && "no ws (cd packages/room-node && npm install)";
// node-datachannel can keep native handles alive after every link closed: don't let them hold the run
after(() => { setTimeout(() => process.exit(), 1500).unref(); });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(f, ms = 15000, what = "condition") {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) { const v = await f(); if (v) return v; await sleep(50); }
  throw new Error(`timed out waiting for ${what}`);
}

test("channel-bound join: honest links get in and prove; links with someone in the middle don't", { skip, timeout: 240000 }, async () => {
  const { startSignal, relay } = await import("./mitm_signal.mjs");
  const { createRoom, joinRoom, PREFIX } = await import("../roomnode.js");
  const env = await import("../env.js");
  const { Bridge } = await import("../../../cli/lib/room.js");
  const { PROTOCOL } = await import("../../../room/transport.js");
  const sig = await startSignal();
  const signal = `127.0.0.1:${sig.port}`;
  const noGpu = async () => ({ create: () => ({ requestAdapter: async () => null }), globals: {} });
  const common = { signal, setup: { webgpu: noGpu }, stripes: 2, selfTest: false };
  const logs = [];
  if (process.env.SHOW_LOGS) logs.push = (...a) => { for (const x of a) console.error(x); return Array.prototype.push.apply(logs, a); };
  const nodes = [], bridges = [];
  const code = "Q7KX4M", hostId = PREFIX + code;
  let mitm = null;
  try {
    const host = await createRoom({ code, gate: true, ask: true, name: "host", log: (s) => logs.push("host: " + s), ...common });
    nodes.push(host);
    const reqs = [];
    host.on("joinrequest", (q) => reqs.push(q));
    const key = host.gate.key;

    // ---- honest signaling ----
    // 1. the bridge with the invite key: in at once, no request
    const b1 = new Bridge({ code, key, signal, name: "serve-1", client: "test", Peer: env.Peer, log: (s) => logs.push("b1: " + s) });
    bridges.push(b1);
    await b1.connect();
    assert.equal(b1.connected, true); assert.equal(reqs.length, 0, "no join request for the invite key");
    assert.ok(b1.pass && b1.pass.length >= 22, "a pass back");
    // 2. a device with the key: in, the host's proof checked, the room's mesh key in hand
    const devA = await joinRoom(code, { name: "dev-a", key, pledgeGB: 2, log: (s) => logs.push("a: " + s), ...common });
    nodes.push(devA);
    await until(() => devA.admission === "in", 15000, "dev-a in");
    assert.equal(devA.mk, host.mk, "the mesh key came with the admit");
    // 3. a device with the code alone: waits; both sides show the same six digits; Allow lets it in
    let lobbySas = null;
    const devB = await joinRoom(code, { name: "dev-b", pledgeGB: 2, log: (s) => logs.push("b: " + s), ...common });
    nodes.push(devB);
    devB.on("lobby", (x) => { lobbySas = x?.sas; });
    await until(() => reqs.length === 1 && lobbySas, 15000, "dev-b's request");
    assert.match(reqs[0].sas, /^\d{3} \d{3}$/);
    assert.equal(reqs[0].sas, lobbySas, "the same code on both screens");
    await host.allowJoin(reqs[0].id);
    await until(() => devB.admission === "in" && devB.mk === host.mk, 15000, "dev-b in");
    // the roster marks both as devices that prove their links
    await until(() => (devA.members || []).filter((m) => m.a).length >= 3, 10000, "roster with a: 1");
    // 4. dev-a and dev-b link and prove the mesh key to each other (stripes too)
    assert.equal(await devA.ensureLink(devB.peer.id, 20000), true, "a proved link between two devices");
    await until(() => devB.conns.get(devA.peer.id) && !devB.conns.get(devA.peer.id).hold, 10000, "dev-b's end proved");
    await until(() => devA.conns.get(devB.peer.id)?.stripes.length >= 1, 15000, "a stripe proved and attached");
    // 5. a stranger who learned dev-a's id dials it: a made-up proof closes the link, and nothing it says counts
    const stranger = new env.Peer(undefined, { host: "127.0.0.1", port: sig.port, path: "/", secure: false, debug: 0 });
    await new Promise((r) => stranger.on("open", r));
    const sc = stranger.connect(devA.peer.id, { reliable: true });
    let strangerClosed = false;
    sc.on("close", () => { strangerClosed = true; });
    await new Promise((r) => sc.on("open", r));
    sc.send({ t: "hello", name: "stranger", v: PROTOCOL, auth: 1, meta: {} });
    sc.send({ t: "mesh", v: 1, p: "0".repeat(64) });
    sc.send({ t: "ai-next", next: stranger.id });
    await until(() => strangerClosed || !devA.conns.has(stranger.id), 15000, "the stranger's link closed");
    assert.notEqual(devA.ai.next, stranger.id, "its ai-next never counted");
    try { stranger.destroy(); } catch {}

    // ---- the signaling server lies: links to the host go to an attacker that relays both ways ----
    mitm = await relay(env.Peer, { port: sig.port, attackerId: "attacker-1", target: hostId, signal: sig });
    // 6. the bridge with the invite key: the relayed proof doesn't get it in
    const before = reqs.length;
    let b2sas = null;
    const b2 = new Bridge({ code, key, signal, name: "serve-2", client: "test", Peer: env.Peer, log: (s) => logs.push("b2: " + s) });
    bridges.push(b2);
    b2.on("lobby", (x) => { b2sas = x?.sas; });
    b2.connect().catch(() => {});
    await until(() => reqs.length === before + 1 && b2sas, 20000, "the relayed bridge held in the lobby");
    assert.equal(b2.connected, false, "not in");
    assert.ok(reqs[before].sas && reqs[before].sas !== b2sas, "the host and the bridge show different codes");
    assert.ok(logs.some((l) => /didn't check out on this link/.test(l)), "the host says the link's key didn't check out");
    assert.ok(!JSON.stringify(mitm.seen).includes(key), "the invite key never crossed the relayed link");
    host.denyJoin(reqs[before].id);
    await until(() => b2.kicked, 10000, "the bridge told no");
    // 7. a device with the key through the relay: not let in, no mesh key (a fresh attacker: the host
    // remembers the one it just turned away, by its peer id)
    mitm.stop();
    mitm = await relay(env.Peer, { port: sig.port, attackerId: "attacker-1b", target: hostId, signal: sig });
    const devC = await joinRoom(code, { name: "dev-c", key, pledgeGB: 2, log: (s) => logs.push("c: " + s), ...common });
    nodes.push(devC);
    await until(() => reqs.length === before + 2, 20000, "dev-c's relayed request");
    assert.notEqual(devC.admission, "in"); assert.equal(devC.mk, null, "no mesh key for a link with someone in the middle");
    host.denyJoin(reqs[before + 1].id);
    // 8. the old way: a hello with the raw key, through the relay. The attacker reads it (it is what it
    // would replay), and a host that still accepts raw keys lets the relayed link in; a host that
    // refuses them asks instead
    const oldHello = async (name) => {
      const p = new env.Peer(undefined, { host: "127.0.0.1", port: sig.port, path: "/", secure: false, debug: 0 });
      await new Promise((r) => p.on("open", r));
      const c = p.connect(hostId, { reliable: true });
      const got = [];
      c.on("data", (d) => got.push(d?.t));
      await new Promise((r) => c.on("open", r));
      c.send({ t: "hello", name, v: PROTOCOL, join: 1, key, meta: {} });
      await until(() => got.includes("admit") || got.includes("lobby"), 15000, "the host's answer to an old hello");
      try { p.destroy(); } catch {}
      return got;
    };
    mitm.stop();
    mitm = await relay(env.Peer, { port: sig.port, attackerId: "attacker-1c", target: hostId, signal: sig });
    const legacyGot = await oldHello("old-1");
    assert.ok(legacyGot.includes("admit"), "legacy on: the raw key is still taken (with a warning)");
    assert.ok(JSON.stringify(mitm.seen).includes(key), "and the attacker in the middle saw it");
    assert.ok(logs.some((l) => /sent the old way/.test(l)), "the host warned about it");
    host.gate.legacy = false;
    const strictGot = await oldHello("old-2");
    assert.ok(strictGot.includes("lobby") && !strictGot.includes("admit"), "legacy off: the raw key is ignored, the host is asked");
    host.gate.legacy = true;
    mitm.stop(); mitm = null;

    // 9. the attacker in the middle of a link between two devices: refused
    mitm = await relay(env.Peer, { port: sig.port, attackerId: "attacker-2", target: devB.peer.id, signal: sig });
    const devD = await joinRoom(code, { name: "dev-d", key, pledgeGB: 2, log: (s) => logs.push("d: " + s), ...common });
    nodes.push(devD);
    // (dev-d's link to the host is not hijacked: only links to dev-b are)
    await until(() => devD.admission === "in", 20000, "dev-d in");
    assert.equal(await devD.ensureLink(devB.peer.id, 15000), false, "no proved link through the attacker");
    assert.ok(logs.some((l) => /^(b|d): .*(didn't prove|no proof)/.test(l)), "a side closed the unproved link");
  } finally {
    if (mitm) mitm.stop();
    for (const b of bridges) try { b.destroy(); } catch {}
    for (const n of nodes.reverse()) try { await n.close(); } catch {}
    sig.close();

  }
});

test("downgrade: a forged 'older host' hello gets no raw key or pass out of a new device or bridge", { skip, timeout: 120000 }, async () => {
  const { startSignal, relay } = await import("./mitm_signal.mjs");
  const { createRoom, joinRoom, PREFIX } = await import("../roomnode.js");
  const env = await import("../env.js");
  const { Bridge } = await import("../../../cli/lib/room.js");
  const sig = await startSignal();
  const signal = `127.0.0.1:${sig.port}`;
  const noGpu = async () => ({ create: () => ({ requestAdapter: async () => null }), globals: {} });
  const common = { signal, setup: { webgpu: noGpu }, stripes: 1, selfTest: false };
  const code = "Q7KX4N", hostId = PREFIX + code;
  const nodes = []; const logs = []; let mitm = null;
  try {
    const host = await createRoom({ code, gate: true, ask: true, name: "host", log: (s) => logs.push("host: " + s), ...common });
    nodes.push(host);
    const reqs = []; host.on("joinrequest", (q) => reqs.push(q));
    const key = host.gate.key;
    // the attacker strips auth from the host's hello: the device takes it for a host from before the proofs
    const strip = (m) => (m?.t === "hello" ? (({ auth, ...rest }) => rest)(m) : m);
    mitm = await relay(env.Peer, { port: sig.port, attackerId: "attacker-d", target: hostId, signal: sig, down: strip });
    const dev = await joinRoom(code, { name: "victim", key, pledgeGB: 2, log: (s) => logs.push("v: " + s), ...common });
    nodes.push(dev);
    await until(() => reqs.length === 1, 20000, "the device asked about instead");
    mitm.stop();
    mitm = await relay(env.Peer, { port: sig.port, attackerId: "attacker-e", target: hostId, signal: sig, down: strip });
    const b = new Bridge({ code, key, signal, name: "victim-serve", client: "test", Peer: env.Peer, log: (s) => logs.push("b: " + s) });
    nodes.push({ close: async () => b.destroy() });
    b.connect().catch(() => {});
    await until(() => reqs.length === 2, 20000, "the bridge asked about instead");
    const ups = JSON.stringify(mitm.seen.filter((x) => x.dir === "up"));
    assert.ok(!ups.includes(key), "no raw key crossed");
    assert.ok(logs.some((l) => /^v: .*older Pooled: ask them to let you in/.test(l)), "the device says why it waits");
    assert.ok(logs.some((l) => /^b: .*older Pooled/.test(l)), "so does the bridge");
    assert.notEqual(dev.admission, "in"); assert.equal(b.connected, false);
  } finally {
    if (mitm) mitm.stop();
    for (const n of nodes.reverse()) try { await n.close(); } catch {}
    sig.close();
  }
});

for (const ask of [false, true]) test(`a host restarted without its saved gate (ask ${ask}): its device comes back, warned, not refused`, { skip, timeout: 120000 }, async () => {
  const { startSignal } = await import("./mitm_signal.mjs");
  const { createRoom, joinRoom } = await import("../roomnode.js");
  const sig = await startSignal();
  const signal = `127.0.0.1:${sig.port}`;
  const noGpu = async () => ({ create: () => ({ requestAdapter: async () => null }), globals: {} });
  const common = { signal, setup: { webgpu: noGpu }, stripes: 1, selfTest: false };
  const code = ask ? "Q7KX4R" : "Q7KX4P"; const logs = []; const nodes = [];
  try {
    const h1 = await createRoom({ code, gate: true, ask, name: "host", log: (s) => logs.push("h1: " + s), ...common });
    nodes.push(h1);
    const dev = await joinRoom(code, { name: "dev", key: h1.gate.key, pledgeGB: 2, log: (s) => logs.push("d: " + s), ...common });
    nodes.push(dev);
    await until(() => dev.admission === "in" && dev.pass, 15000, "dev in");
    let unverified = false, tofu = false;
    dev.on("unverified", () => { unverified = true; }); dev.on("tofu", () => { tofu = true; });
    // a crash: no bye; a new run of the host under the same code, with a new key and no passes
    h1.closing = true; clearInterval(h1.pingTimer); try { h1.peer.destroy(); } catch {} nodes.splice(nodes.indexOf(h1), 1);
    await sleep(1500);
    const h2 = await createRoom({ code, gate: true, ask, name: "host", log: (s) => logs.push("h2: " + s), ...common });
    nodes.push(h2);
    const reqs = []; h2.on("joinrequest", (q) => reqs.push(q));
    await until(() => unverified || (h2.roster.size > 0 && dev.admission === "in") || reqs.length, 40000, "dev back or asked about");
    if (reqs.length) { await h2.allowJoin(reqs[0].id); await until(() => unverified || (dev.admission === "in" && h2.roster.size > 0), 15000, "in after Allow"); }
    assert.equal(unverified, false, "the device did not leave a host that let it in");
    assert.equal(dev.admission, "in"); assert.ok(tofu, "it warned that it couldn't verify the host");
    assert.equal(dev.mk, h2.mk, "the new run's mesh key");
  } finally { for (const n of nodes.reverse()) try { await n.close(); } catch {} sig.close(); }
});
