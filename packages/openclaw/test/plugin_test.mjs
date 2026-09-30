// No GPU, no network, no OpenClaw install: the plugin's request conversion, its StreamFn against a
// scripted room (in process and over a bridge), and onboarding's config.
//   node --test packages/openclaw/test/*_test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { toRequest, toAsk, toolSchema, MAX_TOKENS } from "../src/convert.js";
import { createPooledStream, busyMessage } from "../src/stream.js";
import { roomSettings, keyOf, modelInfo, MODEL_CHOICES, roomLink } from "../src/pool.js";
import { providerConfig, applyToConfig, setupFromEnv, modelRef, PROVIDER, newCode } from "../src/setup.js";
import { validateApiAsk, apiPrompt2, TurnCache, EncodeCache } from "../../../room/api.js";
import { templateProfile } from "../../../room/conversation.js";
import { makeTokenizer } from "../../../engine/tokenizer.js";

const FX = JSON.parse(fs.readFileSync(new URL("fixtures/openclaw-tools.json", import.meta.url), "utf8"));
const HOST2 = { api: 2, ctx: 65536 };
const TOOLS = [{ name: "read", description: "Read a file", parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] } }];

// ---------------- convert ----------------
test("convert: a plain chat is a v1 ask; the system prompt, text parts and images are carried", () => {
  const { v2, body, req } = toAsk({ systemPrompt: "Be brief.", messages: [{ role: "user", content: [{ type: "text", text: "Hi " }, { type: "image", data: "x" }] }] },
    { maxTokens: 100000, temperature: 0 }, { maxTokens: 4096 }, { hostMeta: HOST2 });
  assert.equal(v2, false);
  assert.equal(body.system, "Be brief.");
  assert.match(body.messages[0].text, /^Hi \[image omitted/);
  assert.equal(body.params.maxTokens, MAX_TOKENS, "capped");
  assert.equal(body.params.client, "OpenClaw");
  assert.equal(req.temperature, 0);
});

test("convert: a tool turn is a v2 ask; calls, results (in call order), reasoning and errors map over", () => {
  const ctx = {
    systemPrompt: "sys",
    tools: TOOLS,
    messages: [
      { role: "user", content: "Read a.txt and b.txt" },
      { role: "assistant", stopReason: "toolUse", content: [{ type: "thinking", thinking: "two reads" }, { type: "toolCall", id: "t1", name: "read", arguments: { path: "a.txt" } }, { type: "toolCall", id: "t2", name: "read", arguments: { path: "b.txt" } }] },
      { role: "toolResult", toolCallId: "t2", content: [{ type: "text", text: "B" }], isError: true },
      { role: "toolResult", toolCallId: "t1", content: [{ type: "text", text: "A" }] },
    ],
  };
  const { v2, body } = toAsk(ctx, { reasoning: "high" }, { reasoning: true }, { hostMeta: HOST2 });
  assert.equal(v2, true);
  assert.equal(body.api, 2);
  assert.deepEqual(body.messages[1], { role: "assistant", text: "", calls: [{ name: "read", args: { path: "a.txt" } }, { name: "read", args: { path: "b.txt" } }], reasoning: "two reads" });
  assert.deepEqual(body.messages.slice(2), [{ role: "tool", text: "A" }, { role: "tool", text: "Error: B" }], "results sorted into call order");
  assert.equal(body.params.thinking, true);
  assert.equal(body.params.toolChoice, "auto");
  // a model without reasoning never thinks
  assert.equal(toAsk(ctx, { reasoning: "high" }, { reasoning: false }, { hostMeta: HOST2 }).body.params.thinking, false);
});

test("convert: a failed turn is dropped, a conversation ending on the assistant gets a Continue, bad schemas go free", () => {
  const r = toRequest({ messages: [{ role: "user", content: "q" }, { role: "assistant", stopReason: "error", content: [] }, { role: "assistant", content: [{ type: "text", text: "partial" }] }] });
  assert.deepEqual(r.messages.map((m) => m.role), ["user", "assistant", "user"]);
  assert.equal(r.messages.at(-1).text, "Continue.");
  const logs = [];
  assert.deepEqual(toolSchema({ name: "x", parameters: { type: "object", properties: { a: { $ref: "#/nowhere" } } } }, (m) => logs.push(m)), { type: "object" });
  assert.match(logs[0], /arguments unconstrained/);
  assert.deepEqual(toolSchema({ name: "y" }), { type: "object", properties: {} });
});

test("convert: an older host (v1 only) refuses tools before anything is sent", () => {
  assert.throws(() => toAsk({ tools: TOOLS, messages: [{ role: "user", content: "q" }] }, {}, {}, { hostMeta: { api: 1 } }), /older Pooled/);
  // a plain chat still works with it
  assert.equal(toAsk({ messages: [{ role: "user", content: "q" }] }, {}, {}, { hostMeta: { api: 1 } }).v2, false);
});

// the host's side of the same request: OpenClaw's real tools compile, render and fit (offline)
function qwenByteTok() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  const vocab = {};
  cs.forEach((c, i) => { vocab[String.fromCharCode(c)] = i; });
  const added = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<tool_call>", "</tool_call>", "<think>", "</think>", "<tool_response>", "</tool_response>"].map((content, k) => ({ id: 256 + k, content, special: true }));
  return makeTokenizer({ model: { vocab, merges: [] }, added_tokens: added });
}
test("offline: OpenClaw's 11 real tools pass the host's checks and render for Qwen3 and the MoE's template", () => {
  const ctx = { systemPrompt: FX.systemPrompt, tools: FX.tools, messages: [{ role: "user", content: "What is the secret word in notes.txt?" }] };
  const { body } = toAsk(ctx, { maxTokens: 4096 }, { maxTokens: 4096 }, { hostMeta: HOST2 });
  assert.equal(body.tools.length, 11);
  const tok = qwenByteTok();
  for (const tpl of ["qwen3-1.7b", "qwen3.6-35b-moe"]) {
    const profile = templateProfile(fs.readFileSync(new URL(`../../../tests/fixtures/api/templates/${tpl}.jinja`, import.meta.url), "utf8"), tok);
    const v = validateApiAsk({ rid: "oc1", ...body }, { profile });
    assert.ok(!v.err, `${tpl}: ${v.err}`);
    const p = apiPrompt2(tok, v.req, 1 << 20, { profile, cache: new TurnCache(), encoder: new EncodeCache(), model: tpl });
    assert.ok(!p.err, p.err);
    const text = tok.decode(p.ids);
    for (const t of FX.tools) assert.ok(text.includes(t.name), `${tpl}: tool ${t.name} rendered`);
    assert.ok(text.includes("notes.txt"));
    // too small a context: refused as ctx, never trimmed
    const small = apiPrompt2(tok, v.req, 2048, { profile, cache: new TurnCache(), encoder: new EncodeCache(), model: tpl });
    assert.equal(small.code, "ctx");
  }
});

// ---------------- the StreamFn ----------------
// OpenClaw's event stream, as far as the plugin uses it
function fakeSdk() {
  return {
    createEmptyTransportUsage: () => ({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0 }),
    createAssistantMessageEventStream() {
      const evs = [], wake = []; let final = null, done = false;
      return {
        push(e) { evs.push(e); wake.splice(0).forEach((f) => f()); },
        end(m) { final = m; done = true; wake.splice(0).forEach((f) => f()); },
        async result() { while (!done) await new Promise((r) => wake.push(r)); return final; },
        async *[Symbol.asyncIterator]() { let i = 0; for (;;) { while (i < evs.length) yield evs[i++]; if (done) return; await new Promise((r) => wake.push(r)); } },
      };
    },
    failTransportStream({ stream, output, error }) { output.stopReason = "error"; output.errorMessage = error.message; stream.push({ type: "error", reason: "error", error: output }); stream.end(output); },
  };
}
// a room whose host answers from a script: [ai-* messages] per ask
function scriptedRoom({ hosting = true, hostMeta = HOST2, script }) {
  const asks = [];
  const handler = (rid, body, h) => {
    asks.push(body);
    const msgs = script[asks.length - 1] || [];
    setTimeout(() => { for (const m of msgs) h({ rid, ...m }); }, 0);
    return { stop() {} };
  };
  const node = { hosting: () => hosting, hostMeta, ai: { online: true, degraded: false }, log() {}, request: (body, h, { rid }) => handler(rid, body, h), status: () => ({ devices: [] }) };
  const bridge = { ready: true, kicked: null, hostMeta, ask: (rid, body, h) => { handler(rid, body, h); return true; }, stop() {} };
  return { node, bridge, asks };
}
function install(room, cfg) {
  const s = roomSettings(cfg, {});
  globalThis[Symbol.for("pooled.openclaw.room")] = { key: keyOf(s), s, ready: Promise.resolve({ s, node: room.node, bridge: room.bridge, code: "TEST", link: "https://pooled.run/room/TEST", P: {} }) };
  return createPooledStream({ getPluginConfig: () => cfg, sdk: fakeSdk() });
}
const MODEL = { id: "qwen3-1.7b", provider: "pooled", api: "openai-completions", maxTokens: 4096, reasoning: false };
async function run(fn, context) {
  const st = fn(MODEL, context, { maxTokens: 512 });
  const types = []; for await (const ev of st) types.push(ev.type);
  const msg = await st.result();
  return { types, msg };
}
const CALL_TURN = [{ t: "ai-genstart", api: 2, promptTokens: 123 }, { t: "ai-call", i: 0, name: "read" }, { t: "ai-call", i: 0, a: "{\"path\":" }, { t: "ai-call", i: 0, a: "\"notes.txt\"}" }, { t: "ai-call", i: 0, end: 1 },
  { t: "ai-gendone", api: 2, reason: "stop", usage: { in: 123, out: 20 }, reused: 100, calls: [{ name: "read", args: "{\"path\":\"notes.txt\"}" }], stats: "20 tok" }];
const TEXT_TURN = [{ t: "ai-genstart", api: 2, promptTokens: 160 }, { t: "ai-token", text: "The secret " }, { t: "ai-token", text: "word is PELICAN-42." }, { t: "ai-gendone", api: 2, reason: "stop", usage: { in: 160, out: 9 }, reused: 150, calls: [] }];

for (const hosting of [true, false]) {
  test(`stream (${hosting ? "this device hosts" : "through the bridge"}): a tool call, its result, then text; refusals in the chat`, async () => {
    const room = scriptedRoom({ hosting, script: [CALL_TURN, TEXT_TURN, [{ t: "ai-busy", code: "ctx", why: "too long", n: 20000, max: 16352 }]] });
    const fn = install(room, { mode: hosting ? "host" : "join", code: "TEST", model: "qwen3-1.7b" });
    const c1 = { systemPrompt: "You are helpful.", messages: [{ role: "user", content: "What is the secret word in notes.txt?", timestamp: 1 }], tools: TOOLS };
    const r1 = await run(fn, c1);
    assert.equal(r1.msg.stopReason, "toolUse", r1.msg.errorMessage);
    const call = r1.msg.content.find((c) => c.type === "toolCall");
    assert.deepEqual(call.arguments, { path: "notes.txt" });
    assert.match(call.id, /^call_[A-Za-z0-9]{24}$/);
    assert.deepEqual(r1.types, ["start", "toolcall_start", "toolcall_delta", "toolcall_delta", "toolcall_end", "done"]);
    assert.deepEqual([r1.msg.usage.input, r1.msg.usage.cacheRead, r1.msg.usage.output], [23, 100, 20]);
    const c2 = { ...c1, messages: [...c1.messages, { role: "assistant", content: r1.msg.content, stopReason: "toolUse" },
      { role: "toolResult", toolCallId: call.id, toolName: "read", content: [{ type: "text", text: "The secret word is PELICAN-42." }], isError: false, timestamp: 2 }] };
    const r2 = await run(fn, c2);
    assert.equal(r2.msg.stopReason, "stop");
    assert.equal(r2.msg.content[0].text, "The secret word is PELICAN-42.");
    assert.deepEqual(room.asks[1].messages.slice(1), [{ role: "assistant", text: "", calls: [{ name: "read", args: { path: "notes.txt" } }] }, { role: "tool", text: "The secret word is PELICAN-42." }]);
    const r3 = await run(fn, c1);
    assert.equal(r3.msg.stopReason, "error");
    assert.match(r3.msg.errorMessage, /^Pooled: the conversation is 20000 tokens; .*16352/);
  });
}

test("stream: a call the request did not allow, or broken arguments, end the answer as an error (the host is untrusted)", async () => {
  const bad = [{ t: "ai-genstart", api: 2, promptTokens: 5 }, { t: "ai-call", i: 0, name: "rm_rf" }];
  const broken = [{ t: "ai-genstart", api: 2, promptTokens: 5 }, { t: "ai-call", i: 0, name: "read" }, { t: "ai-call", i: 0, a: "{\"path\":" }, { t: "ai-call", i: 0, end: 1 }];
  const room = scriptedRoom({ script: [bad, broken] });
  const fn = install(room, { mode: "host", code: "TEST", model: "qwen3-1.7b" });
  const c = { messages: [{ role: "user", content: "hi" }], tools: TOOLS };
  const a = await run(fn, c);
  assert.equal(a.msg.stopReason, "error"); assert.match(a.msg.errorMessage, /not a tool this request allows/);
  const b = await run(fn, c);
  assert.equal(b.msg.stopReason, "error"); assert.match(b.msg.errorMessage, /not a JSON object/);
});

test("stream: an older host gets no tools; the chat says why", async () => {
  const room = scriptedRoom({ hostMeta: { api: 1 }, script: [] });
  const fn = install(room, { mode: "join", code: "TEST", model: "qwen3-1.7b" });
  const r = await run(fn, { messages: [{ role: "user", content: "hi" }], tools: TOOLS });
  assert.match(r.msg.errorMessage, /^Pooled: .*older Pooled/);
  assert.equal(room.asks.length, 0);
});

test("busyMessage: every refusal reads as a sentence with the room code", () => {
  const r = { code: "K7QX", link: "https://pooled.run/room/K7QX" };
  for (const code of ["ctx", "loading", "degraded", "off", "queue", "gone", "other"]) assert.match(busyMessage({ code, n: 1, max: 2, err: "x" }, r), /K7QX/);
});

// ---------------- settings and onboarding ----------------
test("settings: plugin config, overridden by POOLED_* env", () => {
  const s = roomSettings({ mode: "host", code: "abcd", model: "qwen3.6-35b-moe", pledgeGB: 12 }, { POOLED_PLEDGE_GB: "20" });
  assert.deepEqual([s.mode, s.code, s.model, s.pledgeGB, s.minDevices, s.waitSeconds], ["host", "ABCD", "qwen3.6-35b-moe", 20, 1, 120]);
  assert.equal(roomSettings({}, {}).mode, null);
  assert.equal(roomLink("ABCD"), "https://pooled.run/room/ABCD");
  assert.equal(roomLink("ABCD", "127.0.0.1:9000"), "https://pooled.run/room/ABCD?signal=127.0.0.1%3A9000");
});

test("models: onboarding offers the room's models with their largest context (the MoE: 128k)", () => {
  assert.deepEqual(MODEL_CHOICES, ["qwen3-1.7b", "qwen3.8-27b", "qwen3.6-35b-moe"]);
  assert.deepEqual(modelInfo("qwen3.6-35b-moe"), { name: "Qwen3.6 35B MoE", needGB: 22.5, ctx: 131072 });
  assert.equal(modelInfo("qwen3-1.7b").ctx, 16384);
});

test("onboarding: a host config puts one model in the catalog, turns the plugin on and makes it the default", () => {
  const s = { mode: "host", code: "K7QX", model: "qwen3.6-35b-moe", pledgeGB: 12, minDevices: 2 };
  const pc = providerConfig(s);
  assert.equal(pc.models.length, 1);
  assert.deepEqual([pc.models[0].id, pc.models[0].contextWindow, pc.models[0].reasoning, pc.authHeader], ["qwen3.6-35b-moe", 131072, true, false]);
  assert.equal(providerConfig({ ...s, ctx: 20000 }).models[0].contextWindow, 19968);
  const cfg = applyToConfig({ agents: { defaults: { model: { primary: "openai/gpt" } } } }, s);
  assert.equal(cfg.agents.defaults.model.primary, "pooled/qwen3.6-35b-moe");
  assert.deepEqual(cfg.plugins.entries[PROVIDER], { enabled: true, config: { mode: "host", code: "K7QX", model: "qwen3.6-35b-moe", pledgeGB: 12, minDevices: 2 } });
  assert.equal(cfg.agents.defaults.models["pooled/qwen3.6-35b-moe"].agentRuntime.id, "openclaw");
  const join = providerConfig({ mode: "join", code: "K7QX" });
  assert.deepEqual([join.models[0].id, join.models[0].name], ["room", "Pooled room K7QX"]);
  assert.equal(modelRef({ mode: "join" }), "pooled/room");
});

test("onboarding (non-interactive): POOLED_* env; a join needs a real code", () => {
  const h = setupFromEnv({});
  assert.equal(h.mode, "host"); assert.match(h.code, /^[A-HJKMNP-TV-Z2-9]{4}$/);
  assert.equal(setupFromEnv({ POOLED_MODE: "join", POOLED_CODE: "k7qx" }).code, "K7QX");
  assert.throws(() => setupFromEnv({ POOLED_MODE: "join", POOLED_CODE: "I0O1" }), /POOLED_CODE/);
  assert.match(newCode(), /^[A-HJKMNP-TV-Z2-9]{4}$/);
});

// With a local OpenClaw (OC_ROOT=<dir with node_modules/openclaw>, run with --import ./test/oc-resolve.mjs):
// the plugin entry loads, and the StreamFn drives OpenClaw's real event stream.
test("with OpenClaw's SDK: the entry registers and the StreamFn feeds its real event stream", { skip: !process.env.OC_ROOT && "OC_ROOT not set" }, async () => {
  const entry = (await import("../index.js")).default;
  const got = {};
  entry.register({ registrationMode: "setup", logger: {}, pluginConfig: {}, registerProvider: (p) => { got.provider = p; }, registerService: () => {} });
  assert.equal(got.provider.id, "pooled");
  assert.equal(got.provider.auth[0].id, "room");
  const llm = await import("openclaw/plugin-sdk/llm");
  const tr = await import("openclaw/plugin-sdk/provider-transport-runtime");
  const room = scriptedRoom({ script: [CALL_TURN] });
  install(room, { mode: "host", code: "TEST", model: "qwen3-1.7b" });
  const fn = createPooledStream({ getPluginConfig: () => ({ mode: "host", code: "TEST", model: "qwen3-1.7b" }),
    sdk: { createAssistantMessageEventStream: llm.createAssistantMessageEventStream, createEmptyTransportUsage: tr.createEmptyTransportUsage, failTransportStream: tr.failTransportStream } });
  const r = await run(fn, { messages: [{ role: "user", content: "read notes.txt", timestamp: 1 }], tools: TOOLS });
  assert.equal(r.msg.stopReason, "toolUse", r.msg.errorMessage);
  assert.deepEqual(r.msg.content.find((c) => c.type === "toolCall").arguments, { path: "notes.txt" });
});
