// No GPU, no network: the room node's host logic with its links and engine stubbed out.
//   node --test packages/room-node/test/*_test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { RoomNode, toApiRequest, eventEncoder } from "../roomnode.js";
import { openModel, LOCAL } from "../source.js";
import { pledgeFor } from "../env.js";
import { PROTOCOL } from "../../../room/transport.js";
import { finishRequest, askBody, needsV2 } from "../../../cli/lib/common.js";

const FX = new URL("../../../tests/fixtures/api/", import.meta.url);
const readFx = (name) => JSON.parse(fs.readFileSync(new URL(name, FX), "utf8"));
const GB = 2 ** 30;

// a node with fake links: sent[id] collects what it sends; no PeerJS, no GPU
function fakeNode({ host = true, name = "host" } = {}) {
  const n = new RoomNode({ name, pledgeGB: 8, log: () => {} });
  n.isHost = host; n.code = "TEST"; n.meta = { webgpu: true, contribGB: 8, gpu: "fake" };
  n.peer = { id: host ? "pooled-room-TEST" : "me", connect: () => null, disconnected: false };
  n.sent = {};
  n.sendTo = (id, m) => { (n.sent[id] ||= []).push(m); };
  n.sendHidden = (id, m) => { (n.sent[id] ||= []).push(m); };
  n.addPeer = (id, pname, meta = { webgpu: true, contribGB: 4 }) => n.conns.set(id, { conn: { close() {}, send() {} }, name: pname, meta, link: null, stripes: [], seen: performance.now(), missed: 0 });
  return n;
}
const msgs = (n, id, t) => (n.sent[id] || []).filter((m) => m.t === t);

test("source: a model maps to its local GGUF, else to its URL; unknown models throw", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "rn-src-"));
  fs.mkdirSync(path.join(dir, "qwen17"));
  fs.writeFileSync(path.join(dir, "qwen17/model.gguf"), "x");
  const a = openModel("qwen3-1.7b", { modelDir: dir });
  assert.equal(a.local, path.join(dir, "qwen17/model.gguf"));
  const b = openModel("qwen3.6-35b-moe", { modelDir: dir });
  assert.equal(b.local, null);
  assert.match(b.url, /^https:\/\/.*Qwen3\.6-35B-A3B-Q4_0\.gguf$/);
  assert.ok(Object.values(LOCAL).includes("q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf"));
  assert.throws(() => openModel("no-such-model"), /unsupported model/);
  fs.rmSync(dir, { recursive: true });
});

test("env: the pledge is what was asked, else half the largest buffer, 1..64 GB", () => {
  assert.equal(pledgeFor(12, 100), 12);
  assert.equal(pledgeFor(null, 30), 15);
  assert.equal(pledgeFor(0, 0), 1);
  assert.equal(pledgeFor(500, 0), 64);
});

test("dealPlan: layers by pledge, an iPhone held to 1 GB, phones left out when computers hold the model", () => {
  const self = { name: "spark", meta: { contribGB: 8 } };
  const L = 28, layerBytes = 0.06 * GB, embedBytes = 0.6 * GB;
  const two = RoomNode.dealPlan({ L, layerBytes, embedBytes, self, peers: [{ id: "b", name: "mac", meta: { contribGB: 8 } }] });
  assert.deepEqual(two.chain, ["b"]);
  assert.equal(two.ranges[0][0], 0); assert.equal(two.ranges.at(-1)[1], L);
  assert.equal(two.ranges[0][1], two.ranges[1][0]);
  // a phone that says 6 GB counts as 1 GB (room/pledge.js), and is left out: the computers hold it all
  const withPhone = RoomNode.dealPlan({ L, layerBytes, embedBytes, self, peers: [{ id: "b", name: "mac", meta: { contribGB: 8 } }, { id: "p", name: "iphone", meta: { contribGB: 6, ua: "iPhone", phone: true } }] });
  assert.deepEqual(withPhone.chain, ["b"]);
  assert.deepEqual(withPhone.leftOut, ["p"]);
  // a model the computers cannot hold alone: the phone gets layers, at most 1 GB of them (14 + 24 + 2 layers)
  const big = RoomNode.dealPlan({ L: 40, layerBytes: 0.5 * GB, embedBytes: 1 * GB, self: { name: "a", meta: { contribGB: 8 } },
    peers: [{ id: "b", name: "mac", meta: { contribGB: 12 } }, { id: "p", name: "iphone", meta: { contribGB: 6, ua: "iPhone", phone: true } }] });
  assert.deepEqual(big.chain, ["b", "p"]);
  const phoneLayers = big.ranges[2][1] - big.ranges[2][0];
  assert.ok(phoneLayers >= 1 && phoneLayers * 0.5 <= 1.0001, `phone holds ${phoneLayers} layers`);
  // a smaller share the host set after a killed load
  const capped = RoomNode.dealPlan({ L, layerBytes, embedBytes, self, peers: [{ id: "b", name: "mac", meta: { contribGB: 8 } }], shareCap: new Map([["mac", 0.3]]) });
  assert.ok(capped.ranges[1][1] - capped.ranges[1][0] <= 5, JSON.stringify(capped.ranges));
});

test("hello: another protocol version is told to reload; a host's hello says api 2 and its context", () => {
  const n = fakeNode();
  n.addPeer("x", "x");
  n.onData("x", { t: "hello", name: "old tab", v: PROTOCOL - 1, meta: {} });
  assert.match(msgs(n, "x", "bye")[0].reason, /protocol/);
  n.ai.model = "qwen3-1.7b";
  const h = n.helloMsg();
  assert.equal(h.meta.api, 2); assert.ok(h.meta.ctx > 0); assert.equal(h.v, PROTOCOL);
});

test("worker: ai-next {relink} drops the dead link, opens a fresh one and answers ai-linked", async () => {
  const n = fakeNode({ host: false, name: "w" });
  n.ai.hostId = "pooled-room-TEST"; n.addPeer("pooled-room-TEST", "host"); n.addPeer("next", "phone");
  let dialed = null;
  n.ensureLink = async (id) => { dialed = id; return true; };
  await n.aiOnData("pooled-room-TEST", { t: "ai-next", next: "next", relink: 1 });
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(dialed, "next");
  assert.ok(!n.conns.has("next"), "the old link was dropped");
  assert.deepEqual(msgs(n, "pooled-room-TEST", "ai-linked")[0], { t: "ai-linked", next: "next", ok: true });
  // from anyone but the host: ignored
  await n.aiOnData("next", { t: "ai-next", next: "evil", relink: 1 });
  assert.equal(n.ai.next, "next");
});

test("host: a chain device that leaves degrades the room; back under its name it is re-seated and the room waits for ai-linked", () => {
  const n = fakeNode();
  n.autoRedeal = false;
  n.addPeer("a", "mac"); n.addPeer("b", "phone");
  const load = (i) => ({ t: "ai-load", v: PROTOCOL, model: "qwen3-1.7b", range: [[10, 20], [20, 28]][i], ctx: 4096, kv: "f16", next: i ? "host" : "b", host: n.peer.id });
  n.ai.chain = ["a", "b"]; n.ai.chainNames = ["mac", "phone"];
  n.ai.plan = new Map([["mac", { msg: load(0) }], ["phone", { msg: load(1) }]]);
  n.ai.engine = { maxSeq: 4096 }; n.ai.cfg = { num_hidden_layers: 28 }; n.ai.model = "qwen3-1.7b";
  n.ai.readyPeers = new Set(["a", "b"]); n.ai.online = true;
  let degraded = null; n.on("degraded", (w) => { degraded = w; });
  let failed = null; n.ai.waiters.set(7, { res: () => {}, rej: (e) => { failed = e.message; } });
  n.peerGone("b", n.conns.get("b"));
  assert.ok(n.ai.degraded && !n.ai.online);
  assert.match(degraded, /phone left/); assert.match(failed, /phone left/);
  assert.equal(n.ai.fed, null);
  assert.deepEqual(n.missingNames(), ["phone"]);
  // the phone comes back under a new peer id
  n.addPeer("b2", "phone");
  n.onData("b2", { t: "hello", name: "phone", v: PROTOCOL, back: 1, meta: { webgpu: true, contribGB: 1, ua: "iPhone" } });
  assert.deepEqual(n.ai.chain, ["a", "b2"]);
  assert.deepEqual(msgs(n, "b2", "ai-load")[0].range, [20, 28]);
  assert.deepEqual(msgs(n, "a", "ai-next")[0], { t: "ai-next", next: "b2", relink: 1 });
  n.aiOnData("b2", { t: "ai-ready", slots: [] });
  assert.ok(!n.ai.online, "not whole until the device before it relinked");
  n.aiOnData("a", { t: "ai-linked", next: "b2", ok: true });
  assert.ok(n.ai.online && !n.ai.degraded, "whole again");
  assert.ok(msgs(n, "b2", "ai-ready-all").length === 1);
});

test("host: a device started again under its name (no back) takes its old slot once the old link stays silent after a ping", async () => {
  const n = fakeNode();
  n.autoRedeal = false;
  n.addPeer("a", "mac"); n.roster.set("a", { name: "mac", meta: { webgpu: true, contribGB: 4 } });
  n.ai.chain = ["a"]; n.ai.chainNames = ["mac"];
  n.ai.plan = new Map([["mac", { msg: { t: "ai-load", v: PROTOCOL, model: "qwen3-1.7b", range: [10, 28], ctx: 4096, kv: "f16", next: "host", host: n.peer.id } }]]);
  n.ai.engine = { maxSeq: 4096 }; n.ai.cfg = { num_hidden_layers: 28 }; n.ai.model = "qwen3-1.7b";
  n.ai.readyPeers = new Set(["a"]); n.ai.online = true;
  n.conns.get("a").seen = performance.now();   // killed just now and started again at once: heard <1 s ago
  n.addPeer("a2", "mac");
  n.onData("a2", { t: "hello", name: "mac", v: PROTOCOL, meta: { webgpu: true, contribGB: 4 } });
  assert.equal(msgs(n, "a", "ping").length, 1, "the quiet namesake is pinged first");
  assert.ok(!n.roster.has("a2"), "the hello waits for the probe");
  await new Promise((r) => setTimeout(r, 1700));
  assert.ok(!n.conns.has("a") && !n.roster.has("a"), "the silent old link is dropped");
  assert.equal(n.roster.get("a2")?.name, "mac", "the new link keeps the name");
  assert.deepEqual(n.ai.chain, ["a2"]);
  assert.deepEqual(msgs(n, "a2", "ai-load")[0].range, [10, 28], "and is re-seated in its slot");
  // a namesake that answers the ping is alive: the newcomer gets another name
  const m = fakeNode();
  m.addPeer("x", "laptop"); m.roster.set("x", { name: "laptop", meta: { webgpu: true, contribGB: 4 } });
  m.conns.get("x").seen = performance.now() - 5000;
  m.addPeer("x2", "laptop");
  m.onData("x2", { t: "hello", name: "laptop", v: PROTOCOL, meta: { webgpu: true, contribGB: 4 } });
  m.onData("x", { t: "pong", ts: 0 });
  await new Promise((r) => setTimeout(r, 1700));
  assert.ok(m.conns.has("x"));
  assert.equal(m.roster.get("x2")?.name, "laptop 2");
});

test("host: a device that joins an online room gets the layer map, so an old worker left out of the deal frees its layers", () => {
  const n = fakeNode();
  n.ai.chain = []; n.ai.chainNames = []; n.ai.plan = new Map();
  n.ai.engine = { maxSeq: 4096 }; n.ai.cfg = { num_hidden_layers: 40 }; n.ai.model = "qwen3.6-35b-moe";
  n.ai.readyPeers = new Set(); n.ai.online = true; n.ai.layersByName = { host: "0–39" };
  n.addPeer("m2", "mac");
  n.onData("m2", { t: "hello", name: "mac", v: PROTOCOL, back: 1, meta: { webgpu: true, contribGB: 23 } });
  const sent = (n.sent.m2 || []).map((m) => m.t);
  assert.ok(sent.indexOf("ai-layers") >= 0 && sent.indexOf("ai-layers") < sent.indexOf("ai-ready-all"), sent.join(","));
  // the worker side: not in the map -> its layers are freed and it is a guest
  const w = fakeNode({ host: false, name: "mac" });
  w.ai.hostId = "pooled-room-TEST"; w.ai.role = "worker"; w.ai.range = [20, 40]; let destroyed = false; w.ai.device = { destroy() { destroyed = true; } };
  w.aiOnData("pooled-room-TEST", { t: "ai-layers", by: { host: "0–39" } });
  assert.equal(w.ai.role, "guest"); assert.equal(w.ai.range, null); assert.ok(destroyed);
});

test("worker: a host of another name under the same code is another room: it leaves before hearing anything else", () => {
  // a rejoin that expects its old host
  const w = fakeNode({ host: false, name: "mac" });
  w.expectHost = "host-ab";
  let other = null; w.on("otherhost", (x) => { other = x; });
  w.addPeer("pooled-room-TEST", "pooled-room-TEST");
  w.onData("pooled-room-TEST", { t: "hello", name: "stranger", v: PROTOCOL, meta: { api: 2 } });
  assert.deepEqual(other, { was: "host-ab", now: "stranger" });
  assert.ok(!w.conns.has("pooled-room-TEST"), "the link to it is closed");
  w.onData("pooled-room-TEST", { t: "ai-load", v: PROTOCOL, model: "qwen3-1.7b", range: [0, 28], next: "host" });
  assert.equal(w.ai.role, null, "and an ai-load from it is ignored");
  // knocking after the host link dropped: the name heard before is the one expected
  const k = fakeNode({ host: false, name: "mac" });
  k.addPeer("pooled-room-TEST", "pooled-room-TEST");
  k.onData("pooled-room-TEST", { t: "hello", name: "host-ab", v: PROTOCOL, meta: { api: 2 } });
  assert.equal(k.hostName, "host-ab"); assert.ok(!k.otherHost);
  k.onData("pooled-room-TEST", { t: "hello", name: "host-ab", v: PROTOCOL, meta: { api: 2 }, back: 1 });
  assert.ok(!k.otherHost, "the same host back is fine");
  k.onData("pooled-room-TEST", { t: "hello", name: "host-zz", v: PROTOCOL, meta: { api: 2 } });
  assert.equal(k.otherHost?.now, "host-zz");
});

test("host: ai-linklost from a chain worker fails the laps in flight", () => {
  const n = fakeNode();
  n.addPeer("a", "mac"); n.ai.chain = ["a"]; n.ai.fed = [1, 2];
  let failed = null; n.ai.waiters.set(3, { res: () => {}, rej: (e) => { failed = e.message; } });
  n.aiOnData("a", { t: "ai-linklost", name: "phone", up: 0 });
  assert.match(failed, /link to phone dropped/); assert.equal(n.ai.fed, null);
});

test("ask(): OpenAI-style messages become the same ask `pooled serve` sends (v1 plain, v2 with tools)", () => {
  const plain = finishRequest(toApiRequest([{ role: "system", content: "Be brief." }, { role: "user", content: [{ type: "text", text: "Hi" }] }], { maxTokens: 64, temperature: 0 }), { hostMeta: { api: 2, ctx: 4096 } });
  assert.equal(needsV2(plain), false);
  assert.deepEqual(askBody(plain, false), { system: "Be brief.", messages: [{ role: "user", text: "Hi" }], params: { maxTokens: 64, temperature: 0, topK: undefined, stop: [], thinking: false, thinkBudget: undefined, client: "node" } });
  const tools = [{ name: "read", description: "Read a file", parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] } }];
  const r = finishRequest(toApiRequest([
    { role: "user", content: "What is in notes.txt?" },
    { role: "assistant", content: "", tool_calls: [{ id: "c1", type: "function", function: { name: "read", arguments: "{\"path\":\"notes.txt\"}" } }] },
    { role: "tool", tool_call_id: "c1", content: "PELICAN-42" },
  ], { tools }), { hostMeta: { api: 2, ctx: 4096 } });
  assert.equal(needsV2(r), true);
  const b = askBody(r, true);
  assert.equal(b.api, 2);
  assert.deepEqual(b.messages[1], { role: "assistant", text: "", calls: [{ name: "read", args: { path: "notes.txt" } }] });
  assert.deepEqual(b.messages[2], { role: "tool", text: "PELICAN-42" });
  assert.equal(b.tools[0].name, "read");
  // an older host (api 1) cannot take tools: refused before anything is sent
  assert.throws(() => finishRequest(toApiRequest([{ role: "user", content: "x" }], { tools }), { hostMeta: { api: 1 } }), /older Pooled/);
});

// the host path end to end with a recorded tool call (tests/fixtures/api): ask() -> request() ->
// validateApiAsk -> apiRun2 (grammar, call parsing) -> Ask on the client side -> events
function replayTok(texts) {
  const vocab = {}, byId = [];
  const add = (t) => { if (vocab[t] === undefined) { vocab[t] = byId.length; byId.push(t); } return vocab[t]; };
  for (const t of ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "\n", "\n\n", " "]) add(t);
  for (const t of texts) add(t);
  const CH = 1 << 20;
  return { vocab, byId, encode: (s) => [...s].map((c) => vocab[c] ?? CH + c.codePointAt(0)), decode: (ids) => ids.map((i) => (i >= CH ? String.fromCodePoint(i - CH) : byId[i] ?? "")).join("") };
}
function replayHost(fx) {
  const n = fakeNode();
  const tok = replayTok(fx.texts), ids = fx.texts.map((t) => tok.vocab[t]);
  Object.assign(n.ai, { tok, engine: { maxSeq: 1 << 16 }, online: true, model: fx.model, apiProf: fx.profile });
  n.generate = async (pids, { stop, maxNew, sample, signal, onToken }) => {
    let k = 0;
    for (; k < maxNew; k++) {
      if (signal?.aborted) return { reason: "abort", reused: 0, stats: "" };
      const want = ids[k];
      if (want === undefined) break;
      const lg = new Float32Array(tok.byId.length).fill(0); lg[want] = 10;
      const t = sample(lg);
      if (stop.has(t)) break;
      onToken(t, 0);
    }
    return { reason: "stop", reused: 0, count: k, stats: `${k} tok · replay` };
  };
  return n;
}
const collect = async (it) => { const out = []; for await (const ev of it) out.push(ev); return out; };

test("ask(): a tool call through the host path, streamed as call events and checked like `pooled serve` checks it", async () => {
  const fx = readFx("qwen3-1.7b-call.json");
  const n = replayHost(fx);
  const evs = await collect(n.ask(fx.req.messages.map((m) => ({ role: m.role, content: m.text })), { tools: fx.req.tools, maxTokens: 200, temperature: 0 }));
  const done = evs.at(-1);
  assert.equal(done.type, "done", JSON.stringify(done));
  assert.equal(done.reason, "stop");
  assert.equal(done.calls.length, 1);
  assert.equal(done.calls[0].name, "get_weather");
  assert.deepEqual(JSON.parse(done.calls[0].args), { city: "Paris" });
  const start = evs.find((e) => e.type === "call" && e.name);
  assert.deepEqual([start.i, start.id, start.name], [0, "call_0", "get_weather"]);
  assert.ok(evs.some((e) => e.type === "call" && e.end === 1));
  assert.ok(evs[0].type === "start" && evs[0].promptTokens > 0);
});

test("ask(): refusals come back as error events (context, a device gone, not the host)", async () => {
  const fx = readFx("qwen3-1.7b-call.json");
  const n = replayHost(fx);
  n.ai.engine.maxSeq = 40;
  const [ctx] = await collect(n.ask([{ role: "user", content: "x".repeat(400) }], { maxTokens: 8 }));
  assert.equal(ctx.reason, "error"); assert.equal(ctx.code, "ctx");
  const m = replayHost(fx); m.ai.degraded = true; m.ai.online = false; m.startP = Promise.resolve();
  const [dg] = await collect(m.ask([{ role: "user", content: "hi" }]));
  assert.equal(dg.code, "degraded");
  const worker = fakeNode({ host: false });
  const [nh] = await collect(worker.ask([{ role: "user", content: "hi" }]));
  assert.equal(nh.reason, "error"); assert.match(nh.err, /does not host/);
});

test("eventEncoder maps the checked answer to ask() events", () => {
  const out = []; const e = eventEncoder((x) => out.push(x));
  e.start(5); e.think("hm"); e.text("ok"); e.callStart(0, "call_0", "f"); e.callArgs(0, "{}"); e.callEnd(0, "{}");
  assert.deepEqual(out.map((x) => x.type), ["start", "token", "token", "call", "call", "call"]);
  assert.equal(out[1].think, true);
});

test("close(): the links it tears down are not departures (no degraded room, no re-deal armed), and the GPU is freed", async () => {
  const n = fakeNode();
  let destroyed = 0, redeals = 0;
  const conn = { handlers: {}, on(ev, f) { this.handlers[ev] = f; }, close() {}, send() {}, peer: "b" };
  n.stripes = 0; n.wire(conn, "mac");
  // PeerJS destroy() closes every connection, which fires their close handlers
  n.peer.destroy = () => conn.handlers.close?.();
  Object.assign(n.ai, { engine: { maxSeq: 1024 }, device: { destroy: () => destroyed++ }, online: true, chain: ["b"], chainNames: ["mac"], model: "qwen3-1.7b" });
  n.redeal = async () => { redeals++; };
  await n.close();
  assert.equal(n.ai.degraded, false);
  assert.equal(n.ai.idleRedeal, undefined);
  assert.equal(redeals, 0);
  assert.equal(n.ai.engine, null);
  assert.equal(destroyed, 1);
});

test("visibility asker: an API answer reaches the asker only; the other screens get hidden stand-ins, and newcomers are told", async () => {
  const fx = readFx("qwen3-1.7b-call.json");
  const n = replayHost(fx);
  n.visibility = "asker";
  n.addPeer("tab", "phone", { webgpu: true, contribGB: 1 });
  const evs = await collect(n.ask(fx.req.messages.map((m) => ({ role: m.role, content: m.text })), { tools: fx.req.tools, maxTokens: 200, temperature: 0 }));
  assert.equal(evs.at(-1).reason, "stop");
  const seen = n.sent.tab || [];
  assert.equal(seen.filter((m) => m.t === "ai-token").length, 0);
  assert.ok(seen.some((m) => m.t === "ai-genstart" && m.hidden && m.text === undefined));
  assert.ok(seen.some((m) => m.t === "ai-gendone" && m.hidden));
  // a device saying hello hears the mode
  n.addPeer("new", "new");
  n.onData("new", { t: "hello", name: "laptop", meta: { webgpu: true, contribGB: 4 }, v: PROTOCOL });
  assert.deepEqual(msgs(n, "new", "ai-visibility"), [{ t: "ai-visibility", mode: "asker" }]);
});

test("API asks: only from a device that joined as an API client, and only while the host allows them", async () => {
  const fx = readFx("qwen3-1.7b-call.json");
  const n = replayHost(fx);
  n.addPeer("tab", "tab");
  n.apiAsk("tab", { t: "ai-ask", api: 1, rid: "r1", system: "", messages: [{ role: "user", text: "hi" }], params: {} });
  assert.equal(msgs(n, "tab", "ai-busy")[0].code, "bad");
  n.addPeer("cli", "cli", { api: 2 }); n.ai.apis.set("cli", { name: "cli" });
  n.allowApi = false;
  n.apiAsk("cli", { t: "ai-ask", api: 1, rid: "r2", system: "", messages: [{ role: "user", text: "hi" }], params: {} });
  assert.equal(msgs(n, "cli", "ai-busy")[0].code, "off");
});

test("nodeServers: --signal's old host:port form keeps its meaning, room/signal.js specs and lists work too", async () => {
  const { nodeServers } = await import("../roomnode.js");
  assert.deepEqual(nodeServers(null).map((s) => s.spec), ["cloud"]);
  const local = nodeServers("127.0.0.1:9000")[0].opts;
  assert.deepEqual(local, { host: "127.0.0.1", port: 9000, path: "/", secure: false });   // as cli/lib/room.js signalOpts
  assert.equal(nodeServers("sig.example.com:443")[0].opts.secure, true);
  const list = nodeServers("wss://sig.example.com/pooled, cloud, cloud");
  assert.deepEqual(list.map((s) => s.label), ["sig.example.com", "0.peerjs.com"]);
  assert.equal(list[0].opts.path, "/pooled/");
  assert.throws(() => nodeServers("bad host!"), /no usable server/);
});

test("setPledge: a host tells the room, a device tells its host; the host's hello names its model", () => {
  const h = fakeNode();
  h.addPeer("p1", "mac");
  h.ai.model = "qwen3-1.7b";
  assert.equal(h.helloMsg().meta.model, "qwen3-1.7b");
  assert.equal(h.setPledge(12), 12);
  assert.equal(h.meta.contribGB, 12);
  assert.deepEqual(msgs(h, "p1", "pledge"), [{ t: "pledge", gb: 12 }]);
  assert.equal(h.setPledge(500), 64, "at most 64 GB per device");
  const d = fakeNode({ host: false, name: "node-abc" });
  d.ai.hostId = "pooled-room-TEST";
  d.addPeer("pooled-room-TEST", "host");
  d.setPledge(6);
  assert.deepEqual(msgs(d, "pooled-room-TEST", "pledge"), [{ t: "pledge", gb: 6 }]);
  assert.equal(d.helloMsg().meta.contribGB, 6);
  assert.equal(d.helloMsg().meta.model, undefined, "a device's hello does not name a model");
  // the host takes a device's new pledge for the next deal
  h.roster.set("p1", { name: "mac", meta: { webgpu: true, contribGB: 4 } });
  h.onData("p1", { t: "pledge", gb: 9 });
  assert.equal(h.conns.get("p1").meta.contribGB, 9);
});

test("dealPlan split: speed puts it all on the host when its pledge holds it; memory spreads by pledge; speed spills over when it doesn't", () => {
  const self = { name: "spark", meta: { contribGB: 8 } };
  const L = 28, layerBytes = 0.06 * GB, embedBytes = 0.6 * GB;
  const peers = [{ id: "b", name: "mac", meta: { contribGB: 8 } }];
  const speed = RoomNode.dealPlan({ L, layerBytes, embedBytes, self, peers, mode: "speed" });
  assert.deepEqual(speed.chain, []); assert.deepEqual(speed.leftOut, ["b"]);
  assert.deepEqual(speed.ranges, [[0, L]]);
  const spread = RoomNode.dealPlan({ L, layerBytes, embedBytes, self, peers, mode: "memory" });
  assert.deepEqual(spread.chain, ["b"]); assert.ok(spread.assigned[1] > 0);
  // the host lends 1 GB: 6 layers there, the rest on the mac
  const spill = RoomNode.dealPlan({ L, layerBytes, embedBytes, self: { name: "spark", meta: { contribGB: 1 } }, peers, mode: "speed" });
  assert.deepEqual(spill.chain, ["b"]); assert.equal(spill.ranges[0][0], 0); assert.equal(spill.ranges.at(-1)[1], L);
  assert.equal(spill.assigned[0], Math.floor((1 - 0.6) / 0.06));
  const n = new RoomNode({ pledgeGB: 4, split: "speed" }); assert.equal(n.splitMode, "speed");
  n.setSplit("spread"); assert.equal(n.splitMode, "memory");
});

test("dealPlan speed counts the KV cache like the room page: 1.7B, 2 GB + 2 GB splits (it does not all go to the host)", async () => {
  const { roomBytes } = await import("../../../room/models.js");
  const fitBytes = roomBytes("qwen3-1.7b", 8192, "f16");   // (2 + 2 GB is short at 16k: ctxPick opens it at 8k)
  const p = RoomNode.dealPlan({ L: 28, layerBytes: 53494784, embedBytes: 330612736, self: { name: "spark", meta: { contribGB: 2 } },
    peers: [{ id: "b", name: "mac", meta: { contribGB: 2 } }], mode: "speed", fitBytes });
  assert.deepEqual(p.chain, ["b"]);
  assert.ok(p.assigned[0] > 0 && p.assigned[0] < 28, `host holds ${p.assigned[0]}`);
});

test("ctxPick: the 1.7B at 16k when the pledges hold it, 8k when they are short for 16k, an asked --ctx stays", async () => {
  const { nodeCtxFor } = await import("../roomnode.js");
  const want = nodeCtxFor("qwen3-1.7b");
  assert.equal(want, 16384);
  const self = (gb) => ({ name: "laptop", meta: { contribGB: gb } });
  const p4 = RoomNode.ctxPick("qwen3-1.7b", { want, self: self(4) });
  assert.deepEqual([p4.ctx, p4.fits, p4.fellBack], [8192, true, true], "one laptop lending 4 GB: 8k");
  assert.equal(p4.note, "Qwen3 1.7B · 8K context: the room's memory is short for 16K");
  const p6 = RoomNode.ctxPick("qwen3-1.7b", { want, self: self(6) });
  assert.deepEqual([p6.ctx, p6.fits, p6.fellBack, p6.note], [16384, true, false, ""]);
  // two devices together hold 16k
  const two = RoomNode.ctxPick("qwen3-1.7b", { want, self: self(4), peers: [{ id: "b", name: "mac", meta: { contribGB: 3 } }] });
  assert.equal(two.ctx, 16384);
  // a share the host lowered (shareCap) counts as the device's pledge
  const capped = RoomNode.ctxPick("qwen3-1.7b", { want, self: self(4), peers: [{ id: "b", name: "mac", meta: { contribGB: 3 } }], shareCap: new Map([["mac", 0.5]]) });
  assert.equal(capped.ctx, 8192);
  // --ctx 16384: no fallback (and short on 4 GB); short for both: 8k, not fitting
  const asked = RoomNode.ctxPick("qwen3-1.7b", { want: nodeCtxFor("qwen3-1.7b", 16384), ask: 16384, self: self(4) });
  assert.deepEqual([asked.ctx, asked.fits, asked.fellBack], [16384, false, false]);
  assert.deepEqual([RoomNode.ctxPick("qwen3-1.7b", { want, self: self(3) }).ctx, RoomNode.ctxPick("qwen3-1.7b", { want, self: self(3) }).fits], [8192, false]);
  // the MoE has no fallback: its context stays
  assert.equal(RoomNode.ctxPick("qwen3.6-35b-moe", { want: 131072, self: self(4) }).ctx, 131072);
});

test("close() on a host tells the room it is over (bye closed, then leaving); a device told so stops waiting for it", async () => {
  const n = fakeNode();
  n.addPeer("a", "mac"); n.addPeer("b", "phone");
  n.peer.destroy = () => {};
  await n.close();
  for (const id of ["a", "b"]) {
    assert.deepEqual(n.sent[id].map((m) => m.t), ["bye", "leaving"]);
    assert.equal(n.sent[id][0].closed, 1); assert.equal(n.sent[id][0].reason, "The host closed the room.");
  }
  // a device closing sends no bye: only its host can end the room
  const d = fakeNode({ host: false, name: "mac" });
  d.addPeer("pooled-room-TEST", "host"); d.peer.destroy = () => {};
  await d.close();
  assert.deepEqual(d.sent["pooled-room-TEST"].map((m) => m.t), ["leaving"]);

  // the device's side: "closed" (not "bye", which reads as being turned away), and when the host's
  // link then drops it neither knocks nor reports the host gone
  const w = fakeNode({ host: false, name: "mac" });
  w.addPeer("pooled-room-TEST", "host"); w.ai.hostId = "pooled-room-TEST"; w.admission = "in";
  const seen = [];
  for (const ev of ["closed", "bye", "hostgone", "roomover"]) w.on(ev, (x) => seen.push([ev, x]));
  w.onData("pooled-room-TEST", { t: "bye", reason: "The host closed the room.", closed: 1 });
  assert.deepEqual(seen, [["closed", "The host closed the room."]]);
  w.hostGone();
  assert.equal(w.knock, null, "no knocking on a closed room");
  assert.deepEqual(seen.map((x) => x[0]), ["closed"]);
  // a bye without closed from the host is still a bye (turned away)
  const k = fakeNode({ host: false, name: "mac" });
  k.addPeer("pooled-room-TEST", "host");
  let bye = null; k.on("bye", (r) => { bye = r; });
  k.onData("pooled-room-TEST", { t: "bye", reason: "no" });
  assert.equal(bye, "no"); assert.ok(!k.roomClosed);
});

// #271: a pledge is a promise. The OpenClaw host (the room node) once re-dealt all 40 layers of the
// MoE plus the embedding onto a Mac lending 14 GB after the other device left ("it may not fit").
// Every layer a deal gives a device, with its KV cache (and the host's embedding/head), fits its pledge.
const heldOk = (p, pledgesGB, fit) => p.assigned.every((a, k) => a * fit.layerBytes + (k === 0 ? fit.hostBytes : 0) <= pledgesGB[k] * GB + 1);
test("dealPlan: a room short of the model deals nothing (the MoE on a Mac lending 14 GB); one that fits stays within every pledge", async () => {
  const { roomBytes } = await import("../../../room/models.js");
  const fit = roomBytes("qwen3.6-35b-moe", 16384, "f16");
  const mac = { name: "mac", meta: { contribGB: 14 } }, stoat = { id: "s", name: "stoat", meta: { contribGB: 12 } };
  for (const mode of ["speed", "memory"]) {
    const alone = RoomNode.dealPlan({ L: fit.L, layerBytes: fit.layerBytes, embedBytes: fit.hostBytes, self: mac, peers: [], mode, fitBytes: fit });
    assert.equal(alone.fit.fits, false, mode);
    assert.deepEqual([alone.chain, alone.ranges, alone.assigned], [[], [], []], `${mode}: nothing dealt`);
    const both = RoomNode.dealPlan({ L: fit.L, layerBytes: fit.layerBytes, embedBytes: fit.hostBytes, self: mac, peers: [stoat], mode, fitBytes: fit });
    assert.equal(both.fit.fits, true, mode);
    assert.deepEqual(both.chain, ["s"], mode);
    assert.equal(both.assigned.reduce((a, b) => a + b, 0), fit.L, mode);
    assert.ok(heldOk(both, [14, 12], fit), `${mode}: ${both.assigned}`);
  }
  // a share the host lowered after a killed load counts: stoat at 4 GB leaves the room short again
  const capped = RoomNode.dealPlan({ L: fit.L, layerBytes: fit.layerBytes, embedBytes: fit.hostBytes, self: mac, peers: [stoat], fitBytes: fit, shareCap: new Map([["stoat", 4]]) });
  assert.equal(capped.fit.fits, false);
  assert.deepEqual(capped.chain, []);
});

test("dealPlan: never more layers than a pledge holds, whatever the pledges (random rooms)", async () => {
  const { roomBytes } = await import("../../../room/models.js");
  let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (const key of ["qwen3-1.7b", "qwen3.6-35b-moe", "qwen3.8-27b"]) {
    const fit = roomBytes(key, 16384, "f16");
    for (let t = 0; t < 150; t++) {
      const n = 1 + Math.floor(rnd() * 4), gbs = Array.from({ length: n }, () => Math.round(rnd() * 160) / 10 + 0.5);
      const peers = gbs.slice(1).map((gb, i) => ({ id: "p" + i, name: "d" + i, meta: { contribGB: gb, ...(rnd() < 0.2 ? { phone: true, ua: "iPhone" } : {}) } }));
      const p = RoomNode.dealPlan({ L: fit.L, layerBytes: fit.layerBytes, embedBytes: fit.hostBytes, self: { name: "h", meta: { contribGB: gbs[0] } }, peers, mode: rnd() < 0.5 ? "speed" : "memory", fitBytes: fit });
      if (!p.fit.fits) { assert.deepEqual(p.assigned, []); continue; }
      const pl = [gbs[0], ...p.chain.map((id) => Math.min(peers.find((x) => x.id === id).meta.phone ? 1 : Infinity, peers.find((x) => x.id === id).meta.contribGB))];
      assert.equal(p.assigned.reduce((a, b) => a + b, 0), fit.L);
      assert.ok(heldOk(p, pl, fit), `${key} ${gbs} -> ${p.assigned}`);
    }
  }
});

// the host's re-deal after a device in the chain left, with the model's source stubbed (no GPU, no
// download): the 1.7B at 16k (28 layers of ~0.18 GB with their KV, ~0.62 GB on the host besides)
function redealNode({ hostGB, peers }) {
  const n = fakeNode({ name: "mac" });
  n.meta.contribGB = hostGB; n.pledgeGB = hostGB;
  for (const [id, name, gb] of peers) n.addPeer(id, name, { webgpu: true, contribGB: gb });
  n.openSource = () => ({ M: { kind: "gguf" }, stat: {}, cfg: async () => ({ num_hidden_layers: 28 }),
    header: async () => ({ tensors: new Proxy({}, { get: () => ({ byteLength: 1 << 20 }) }), meta: {} }), close: async () => {} });
  n.broadcast = (m) => { for (const id of n.conns.keys()) n.sendTo(id, m); };
  n.logs = []; n.log = (m) => n.logs.push(m);
  return n;
}
test("re-deal after a device left: short of the pledges, the room stops and says so instead of over-dealing; when it fits it deals within the pledges", async () => {
  const { roomBytes } = await import("../../../room/models.js");
  const fit = roomBytes("qwen3-1.7b", 16384, "f16");
  // mac lends 2 GB (7 layers + the embedding at 16k, or 12 at 8k), stoat 4 GB: the room held the model
  // with stoat; stoat leaves and does not come back
  const n = redealNode({ hostGB: 2, peers: [["s", "stoat", 4], ["k", "kiwi", 0.5]] });
  n.ai.model = "qwen3-1.7b";
  n.ai.engine = { maxSeq: 16384 }; n.ai.held = { model: "qwen3-1.7b", range: [0, 7], ctx: 16384, kv: "f16" }; n.ai.device = { destroy() {} };
  n.ai.chain = ["s"]; n.ai.chainNames = ["stoat"]; n.ai.layersByName = { mac: "0–6", stoat: "7–27" }; n.ai.degraded = true; n.ai.gone = new Set(["s"]);
  n.conns.delete("s");
  let shortSaid = null; n.on("short", (x) => { shortSaid = x; });
  await assert.rejects(n.redeal("stoat did not come back: re-dealing the layers"), (e) => e.short === true && /GB short for Qwen3 1\.7B/.test(e.message));
  assert.equal(msgs(n, "k", "ai-load").length, 0, "no device is dealt layers");
  assert.match(msgs(n, "k", "ai-start-failed")[0].why, /short/);
  assert.equal(n.ai.engine, null, "the host's layers are freed: the room is stopped, not half up");
  assert.deepEqual([n.ai.degraded, n.ai.online, n.ai.starting, n.ai.chain.length], [false, false, false, 0]);
  assert.match(shortSaid, /^This room is [\d.]+ GB short for Qwen3 1\.7B/);
  assert.equal(n.status().short, shortSaid);
  assert.ok(!n.logs.some((l) => /may not fit|layer split/.test(l)), n.logs.join("\n"));
  // a device joins and the room holds the model again: the next start deals, each within its pledge
  n.addPeer("t", "tern", { webgpu: true, contribGB: 4 });
  const p = n.start("qwen3-1.7b");
  await assert.rejects(p);   // (no GPU here: the host's own load fails after the deal went out)
  const load = msgs(n, "t", "ai-load")[0];
  assert.ok(load, "tern is dealt layers");
  const at = roomBytes("qwen3-1.7b", load.ctx, load.kv === "q8" ? "q8" : "f16");
  const gbOf = { mac: 2, kiwi: 0.5, tern: 4 };
  assert.equal(n.split.ranges.at(-1)[1], fit.L);
  n.split.names.forEach((nm, k) => {
    const [lo, hi] = n.split.ranges[k];
    assert.ok((hi - lo) * at.layerBytes + (k === 0 ? at.hostBytes : 0) <= gbOf[nm] * GB, `${nm} holds ${hi - lo} layers at ${load.ctx}/${load.kv}`);
  });
  assert.equal(n.status().short, null);
});

// A joiner that drops while the host still loads its own layers (the 35B MoE on a PC: a ~5 min load,
// the Mac killed meanwhile) must not wedge the room: the host used to go "online · 2 devices" without
// it once its load ended, with no engine behind ask() ("the model is still loading") and an empty plan,
// so the Mac coming back was told it holds no layers until the gateway restarted.
// slowHost(): the host's own load waits until release(); the workers' loads are theirs to report.
function slowHost(n) {
  let release;
  const gate = new Promise((r) => { release = r; });
  n.loadShardFn = async () => { await gate; return { engine: { maxSeq: 16384, dropAllSlots() {}, reset() {} }, device: { destroy() {} }, tok: {}, cfg: { num_hidden_layers: 28 } }; };
  return () => release();
}
const until = async (f, ms = 2000) => { const t0 = Date.now(); while (!f()) { if (Date.now() - t0 > ms) throw new Error("timed out: " + f); await new Promise((r) => setTimeout(r, 5)); } };

test("a device that leaves while the host loads: the room doesn't go online without it; back within the grace it is dealt its layers again", async () => {
  const n = redealNode({ hostGB: 2, peers: [["s", "stoat", 4]] });
  n.rejoinGraceMs = 60000;
  const release = slowHost(n);
  const p = n.start("qwen3-1.7b"); p.catch(() => {});
  await until(() => msgs(n, "s", "ai-load").length);
  const range = msgs(n, "s", "ai-load")[0].range;
  n.aiOnData("s", { t: "ai-ready", slots: [] });   // stoat loads fast...
  n.peerGone("s", n.conns.get("s"));               // ...and is killed while the host still loads
  assert.ok(n.ai.starting, "the deal waits for it");
  // a second start (the plugin's service, a question) joins this one instead of dealing again
  assert.equal(n.start("qwen3-1.7b"), p);
  release();
  await until(() => !n.ai.loadingShard);
  await new Promise((r) => setTimeout(r, 20));
  assert.equal(n.ai.online, false, "not online with a device of the chain gone");
  assert.ok(n.ai.engine && n.ai.starting, "the host keeps its layers and waits");
  assert.equal(msgs(n, "s", "ai-ready-all").length, 0);
  assert.deepEqual(n.missingNames(), ["stoat"]);
  // stoat started again (a new process: a new peer id, no back): back into its slot with the same layers
  n.addPeer("s2", "stoat", { webgpu: true, contribGB: 4 });
  n.onData("s2", { t: "hello", name: "stoat", v: PROTOCOL, meta: { webgpu: true, contribGB: 4 } });
  assert.deepEqual(msgs(n, "s2", "ai-load")[0]?.range, range, "dealt its layers again");
  assert.equal(msgs(n, "s2", "ai-ready-all").length, 0, "not told the room is up without it");
  n.aiOnData("s2", { t: "ai-ready", slots: [] });
  await p;
  assert.ok(n.ai.online && !n.ai.degraded && n.whole());
  assert.equal(msgs(n, "s2", "ai-ready-all").length, 1);
});

test("a device that leaves while the host loads and stays away: after the grace the room re-deals (short: it stops and says so, not online)", async () => {
  // mac (2 GB) can't hold the 1.7B alone: the re-deal without stoat is short
  const n = redealNode({ hostGB: 2, peers: [["s", "stoat", 4]] });
  n.rejoinGraceMs = 50;
  const release = slowHost(n);
  let shortSaid = null; n.on("short", (x) => { shortSaid = x; });
  const p = n.start("qwen3-1.7b"); p.catch(() => {});
  await until(() => msgs(n, "s", "ai-load").length);
  n.aiOnData("s", { t: "ai-ready", slots: [] });
  n.peerGone("s", n.conns.get("s"));
  await new Promise((r) => setTimeout(r, 120));   // the grace runs out while the host still loads
  assert.ok(n.ai.loadingShard && !n.ai.online);
  release();
  await assert.rejects(p, (e) => e.short === true && /GB short/.test(e.message));
  assert.match(shortSaid, /^This room is [\d.]+ GB short/);
  assert.deepEqual([n.ai.online, n.ai.starting, !!n.ai.engine, n.ai.chain.length], [false, false, false, 0]);
  assert.equal(n.status().short, shortSaid);
  assert.ok(n.logs.some((l) => /stoat .*did not come back/.test(l)), n.logs.join("\n"));
  // stoat comes back: a new start deals it its layers again (the plugin's ensureOnline, pooled host's start again)
  n.addPeer("s2", "stoat", { webgpu: true, contribGB: 4 });
  n.onData("s2", { t: "hello", name: "stoat", v: PROTOCOL, meta: { webgpu: true, contribGB: 4 } });
  const q = n.start("qwen3-1.7b"); q.catch(() => {});
  await until(() => msgs(n, "s2", "ai-load").length);
  n.aiOnData("s2", { t: "ai-ready", slots: [] });
  await q;
  assert.ok(n.ai.online && n.whole());
});

test("a device that leaves while the host loads and stays away: a room that fits without it re-deals and comes online without it", async () => {
  const n = redealNode({ hostGB: 4, peers: [["s", "stoat", 4], ["k", "kiwi", 4]] });
  n.rejoinGraceMs = 50;
  const release = slowHost(n);
  const p = n.start("qwen3-1.7b"); p.catch(() => {});
  await until(() => msgs(n, "s", "ai-load").length && msgs(n, "k", "ai-load").length);
  n.aiOnData("s", { t: "ai-ready", slots: [] }); n.aiOnData("k", { t: "ai-ready", slots: [] });
  n.peerGone("s", n.conns.get("s"));
  await new Promise((r) => setTimeout(r, 120));
  release();
  await until(() => msgs(n, "k", "ai-load").length === 2);   // the re-deal: kiwi gets its new share
  assert.deepEqual(n.ai.chain, ["k"]);
  assert.ok(!n.ai.online);
  n.aiOnData("k", { t: "ai-ready", slots: [] });
  await p;
  assert.ok(n.ai.online && n.whole());
  assert.ok(n.split.names.every((nm) => nm !== "stoat"));
});
