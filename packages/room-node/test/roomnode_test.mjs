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
  // a model the computers cannot hold alone: the phone gets layers, at most 1 GB of them
  const big = RoomNode.dealPlan({ L: 40, layerBytes: 0.5 * GB, embedBytes: 1 * GB, self: { name: "a", meta: { contribGB: 8 } },
    peers: [{ id: "b", name: "mac", meta: { contribGB: 11 } }, { id: "p", name: "iphone", meta: { contribGB: 6, ua: "iPhone", phone: true } }] });
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
