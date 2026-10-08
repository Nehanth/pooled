// Phone memory in rooms (#207): pledges, peer weights streamed with flow control (a window,
// not a whole range), and the host re-dealing when a phone's tab is killed while loading.
// Prefetch memory invariants now exercise the imported loader in model_loader_test.js.
// The room.js pieces are the real functions, cut out of its source by room_src.js.
import { pledgeRule, pledgeGB, afterLoadDeath, IOS_MAX_GB } from "../../room/pledge.js";
import { roomFns } from "./room_src.js";
import { weightsOverLink } from "../../room/ice.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const tick = (ms = 0) => new Promise((r) => setTimeout(r, ms));

// ---- pledges ----
Deno.test("pledgeRule: iPhones and iPads lend 0.5 GB by default and at most 1 GB, with a reason", () => {
  for (const k of ["iPhone", "iPad"]) {
    const r = pledgeRule(k);
    eq([r.min, r.def, r.max, r.step, r.capped], [0.5, 0.5, 1, 0.5, true], k);
    ok(/1\.5 GB/.test(r.why) && /at most 1 GB/.test(r.why), r.why);
  }
});
Deno.test("pledgeRule: Android by navigator.deviceMemory; desktops uncapped", () => {
  eq([0.25, 1, 2, 4, 8, undefined].map((m) => pledgeRule("Android", m).max), [0.5, 0.5, 0.5, 1, 2, 1]);
  eq(pledgeRule("Android", 2).def, 0.5);
  eq(pledgeRule("Android tablet", 8).capped, true);
  for (const k of ["Mac", "Device", undefined]) { const r = pledgeRule(k); eq([r.capped, r.max, r.min], [false, 64, 1], String(k)); }
});
Deno.test("pledgeGB: the host holds an old or tampered phone pledge to the cap, and to its own lowered share", () => {
  eq(pledgeGB({ ua: "iPhone", contribGB: 4 }), IOS_MAX_GB, "iPhone from before the cap");
  eq(pledgeGB({ ua: "iPhone", contribGB: 0.5 }), 0.5);
  eq(pledgeGB({ ua: "Android", contribGB: 3, pledgeMax: 1 }), 1, "the device's own cap");
  eq(pledgeGB({ ua: "Mac", contribGB: 24 }), 24);
  eq(pledgeGB({ ua: "Mac", contribGB: 24 }, 6), 6, "host share cap");
  eq(pledgeGB({ ua: "Mac", maxBufGB: 4 }), 2, "no pledge: half the buffer limit, as before");
  eq(pledgeGB({ ua: "iPhone", contribGB: 1 }, 0.25), 0.25);
});
Deno.test("afterLoadDeath: halve the share (in whole layers, at least one), drop at one layer or on a second death", () => {
  eq(afterLoadDeath({ layers: 3, layerGB: 0.46, gb: 1.5, deaths: 1 }), { drop: false, gb: 0.46 });
  eq(afterLoadDeath({ layers: 4, layerGB: 0.21, gb: 1, deaths: 1 }), { drop: false, gb: 0.42 });
  eq(afterLoadDeath({ layers: 8, layerGB: 0.1, gb: 1, deaths: 1 }), { drop: false, gb: 0.4 });
  eq(afterLoadDeath({ layers: 1, layerGB: 0.46, gb: 0.5, deaths: 1 }), { drop: true });
  eq(afterLoadDeath({ layers: 4, layerGB: 0.21, gb: 1, deaths: 2 }), { drop: true });
});

// ---- weights from a device in the room ----
// Two ends wired back to back: the requester's peerGet/onWeightPart and the server's
// serveWeight/onWeightAck, with the server's cache holding one range.
function wire({ data, win, part = 64 * 1024, dropAfter = Infinity, path = "direct", relayOn = false }) {
  const toServer = [], toClient = [];
  let clientF, serverF;
  const server = { ai: {}, conns: new Map(), wServes: new Map() };
  const client = { ai: {}, wGets: new Map() };
  const conn = { send: (m) => { if (m.off >= dropAfter) return; toClient.push(m); }, dataChannel: { bufferedAmount: 0 } };
  server.conns.set("C", { conn });
  const cache = { match: async () => new Response(data, { headers: { "x-swarm-len": String(data.length) } }) };
  serverF = roomFns(["serveWeight", "onWeightAck"], {
    ai: server.ai, conns: server.conns, wServes: server.wServes, W_PART: part,
    getWeightCache: async () => cache, cacheKey: () => "k", sendTo: (to, m) => toClient.push(m),
    concat: (a, b) => { const m = new Uint8Array(a.length + b.length); m.set(a); m.set(b, a.length); return m; },
  });
  clientF = roomFns(["peerGet", "onWeightPart"], {
    ai: client.ai, wGets: client.wGets, wSeq: 0, W_WIN: win, peer: { id: "C" },
    ensureLink: async () => true, sendTo: (to, m) => toServer.push(m),
    pathOf: async () => path, weightsOverLink, relayOn,
  });
  // deliver messages in order, one hop per tick, like a data channel
  let stop = false;
  const pump = (async () => {
    while (!stop) {
      while (toServer.length) {
        const m = toServer.shift();
        if (m.t === "ai-wget") serverF.serveWeight("C", m); else if (m.t === "ai-wack") serverF.onWeightAck(m);
      }
      while (toClient.length) clientF.onWeightPart(toClient.shift());
      await tick();
    }
  })();
  return { clientF, client, server, toServer, stop: () => { stop = true; return pump; } };
}
const range = (n) => { const d = new Uint8Array(n); for (let i = 0; i < n; i++) d[i] = (i * 31 + (i >> 9)) & 255; return d; };

Deno.test("peerGet: a range streams from a room device in order, byte for byte", async () => {
  const data = range(3 * 2 ** 20 + 12345);
  const w = wire({ data, win: 2 ** 20 });
  const body = await w.clientF.peerGet("S", "u", 0, data.length - 1);
  const got = new Uint8Array(await new Response(body).arrayBuffer());
  await w.stop();
  eq(got.length, data.length);
  ok(got.every((v, i) => v === data[i]), "bytes differ");
  eq(w.client.ai.peerBytes, data.length);
});
Deno.test("peerGet: flow control holds an unread range to about one window on the requester", async () => {
  const data = range(6 * 2 ** 20);
  const win = 2 ** 20;
  const w = wire({ data, win });
  const body = await w.clientF.peerGet("S", "u", 0, data.length - 1);
  await tick(300);   // a prefetched range that nobody reads yet
  const held = w.client.ai.peerBytes;
  ok(held <= win + 64 * 1024, `the requester holds ${held} bytes of an unread range (window ${win})`);
  // a slow reader: every byte arrives, and the requester never holds much more than a window
  const rd = body.getReader();
  let n = 0, maxAhead = 0;
  for (;;) {
    const { value, done } = await rd.read();
    if (done) break;
    n += value.length;
    maxAhead = Math.max(maxAhead, w.client.ai.peerBytes - n);
    if (n % (2 ** 20) < 65536) await tick(5);
  }
  await w.stop();
  eq(n, data.length);
  ok(maxAhead <= win + 2 * 64 * 1024, `read ahead ${maxAhead}`);
});
Deno.test("peerGet: a source that stops mid-range errors the reader with retryNet (the loaders then use the network)", async () => {
  const data = range(2 ** 20);
  const w = wire({ data, win: 2 ** 20, dropAfter: 256 * 1024 });
  const body = await w.clientF.peerGet("S", "u", 0, data.length - 1);
  let err = null;
  try { await new Response(body).arrayBuffer(); } catch (e) { err = e; }
  await w.stop();
  ok(err, "expected an error");
  // the server finished (sent done) after the dropped parts: a short range
  ok(err.retryNet || /short|order/.test(String(err)), String(err));
});

// ---- a phone's tab killed while loading ----
function hostWith(layersN) {
  const sent = [], logs = [], redeals = [], closed = [];
  const conns = new Map([["new", { name: "phone", meta: { ua: "iPhone", contribGB: 1, webgpu: true } }],
    ["old", { name: "phone", conn: { close: () => closed.push("old") } }]]);
  const ai = { role: "host", plan: new Map([["phone", {}]]), chainNames: ["phone"], layersN, layerGB: 0.46 };
  const f = roomFns(["aiLoadDeath", "aiAutoRedeal"], {
    ai, conns, afterLoadDeath, pledgeGB, sendTo: (to, m) => sent.push([to, m]), log: (a, b) => logs.push(b), sysNote: () => {},
    aiStatus: () => {}, aiRedeal: () => redeals.push(1), setTimeout: (fn) => fn(),
  });
  return { f, ai, sent, logs, redeals, closed };
}
Deno.test("aiLoadDeath: first kill while loading shrinks the phone's share and re-deals; the second leaves it out", () => {
  const h = hostWith({ host: 30, phone: 2 });
  const died = { during: "streaming blk.38.ffn_down_exps.weight (160 MB)", ago: 20, at: 1000, loading: true };
  ok(h.f.aiLoadDeath("new", { name: "phone", died }), "handled");
  ok(h.f.aiLoadDeath("new", { name: "phone", died: { ...died, ago: 25 } }), "the same kill reported again (another link)");
  eq(h.redeals.length, 1, "counted once");
  eq(h.ai.shareCap.get("phone"), 0.46);
  eq(h.sent[0][1].t, "ai-share"); eq(h.sent[0][1].gb, 0.46);
  eq(h.redeals.length, 1, "re-dealt");
  eq(h.closed, ["old"], "the stale link is closed");
  ok(h.f.aiLoadDeath("new", { name: "phone", died: { ...died, at: 2000 } }), "a second kill");
  ok(h.ai.dropped.has("phone"), "left out after a second kill");
  eq(h.sent[1][1].drop, true);
  eq(h.redeals.length, 2);
});
Deno.test("aiLoadDeath: not a load death (no loading flag, not in the plan, not the host) is left to aiRejoin", () => {
  const h = hostWith({ phone: 2 });
  ok(!h.f.aiLoadDeath("new", { name: "phone", died: { during: "idle", ago: 20 } }), "killed while idle");
  ok(!h.f.aiLoadDeath("new", { name: "phone" }), "no breadcrumb");
  ok(!h.f.aiLoadDeath("new", { name: "laptop", died: { loading: true } }), "not in the chain");
  h.ai.role = "worker";
  ok(!h.f.aiLoadDeath("new", { name: "phone", died: { loading: true } }), "not the host");
  eq(h.redeals.length, 0);
});
Deno.test("aiAutoRedeal: waits for the host's own layers, then re-deals", () => {
  const h = hostWith({ phone: 2 });
  h.ai.loadingShard = true;
  h.f.aiAutoRedeal("why");
  eq(h.redeals.length, 0); ok(h.ai.redealPending, "pending");
  h.ai.loadingShard = false;
  h.f.aiAutoRedeal("why");
  eq(h.redeals.length, 1); ok(!h.ai.redealPending, "done");
});

Deno.test("peerGet: never over a relayed link (the relay's owner pays per GB): it asks nothing and the loader uses the network", async () => {
  for (const [path, relayOn] of [["relay", true], [null, true]]) {
    const w = wire({ data: range(1000), win: 8 * 2 ** 20, path, relayOn });
    let err = null;
    try { await w.clientF.peerGet("S", "u", 0, 999); } catch (e) { err = e; }
    await w.stop();
    if (!err || !err.relayed) throw new Error(`${path}: expected a relayed refusal, got ${err}`);
    if (w.toServer.some((m) => m.t === "ai-wget")) throw new Error("asked the source anyway");
  }
  // direct, or no relay at all (the path can only be direct): weights flow as before
  for (const [path, relayOn] of [["direct", true], [null, false]]) {
    const w = wire({ data: range(1000), win: 8 * 2 ** 20, path, relayOn });
    const body = await w.clientF.peerGet("S", "u", 0, 999);
    const got = new Uint8Array(await new Response(body).arrayBuffer());
    await w.stop();
    if (got.length !== 1000) throw new Error(`${path}/${relayOn}: got ${got.length} bytes`);
  }
});

Deno.test("answerWget: the serving side says miss over a relayed link, serves over a direct one", async () => {
  const run = async (path, relayOn, PEER_WEIGHTS = true) => {
    const sent = [], served = [];
    const f = roomFns(["answerWget"], { PEER_WEIGHTS, relayOn, weightsOverLink, pathOf: async () => path,
      sendTo: (to, m) => sent.push(m), serveWeight: (from, d) => served.push(d.id) });
    await f.answerWget("C", { id: "r1", url: "u", lo: 0, hi: 9 });
    return { sent, served };
  };
  const eqj = (a, b, m) => { if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(`${m}: ${JSON.stringify(a)}`); };
  eqj(await run("relay", true), { sent: [{ t: "ai-wpart", id: "r1", miss: 1 }], served: [] }, "relay");
  eqj(await run(null, true), { sent: [{ t: "ai-wpart", id: "r1", miss: 1 }], served: [] }, "unknown path with a relay");
  eqj(await run("direct", true), { sent: [], served: ["r1"] }, "direct");
  eqj(await run(null, false), { sent: [], served: ["r1"] }, "no relay configured");
  eqj(await run("direct", false, false), { sent: [{ t: "ai-wpart", id: "r1", miss: 1 }], served: [] }, "peerweights=0");
});
