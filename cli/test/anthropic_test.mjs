// The Anthropic Messages endpoint (cli/lib/anthropic.js) end to end through cli/lib/http.js against
// a stand-in room: request mapping (Claude Code's captured requests included), and the exact bytes
// of responses and streams for fixed room messages (docs/design/serve.md section 4, "Anthropic").
import { test } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import fs from "node:fs";
import { EventEmitter } from "node:events";
import { createServer } from "../lib/http.js";
import { parseAnthropic } from "../lib/anthropic.js";
import { finishRequest, askBody, blob, ApiError, IMAGE_PLACEHOLDER } from "../lib/common.js";

class FakeBridge extends EventEmitter {
  constructor() { super(); this.code = "ABCD"; this.connected = true; this.ready = true; this.model = "qwen36"; this.hostMeta = { api: 2, ctx: 65536 }; this.onAsk = null; this.stopped = []; this.asks = []; }
  ask(rid, body, h) { this.asks.push(body); this.onAsk?.(rid, h, body); return true; }
  stop(rid) { this.stopped.push(rid); }
}
async function start() {
  const bridge = new FakeBridge();
  const logs = [];
  const api = createServer({ bridge, port: 0, log: (m) => logs.push(m) });
  const port = await api.listen();
  const req = (method, path, { headers = {}, body = null } = {}) => new Promise((res) => {
    const r = http.request({ host: "127.0.0.1", port, method, path, headers: { "content-type": "application/json", "anthropic-version": "2023-06-01", ...headers } }, (s) => {
      let d = ""; s.on("data", (c) => (d += c)); s.on("end", () => res({ status: s.statusCode, headers: s.headers, body: d }));
    });
    r.on("error", (e) => res({ status: "error " + e.message }));
    if (body != null) r.write(typeof body === "string" ? body : JSON.stringify(body));
    r.end();
  });
  const close = () => new Promise((r) => { api.closeAll("test over"); api.server.closeAllConnections?.(); api.server.close(r); });
  return { bridge, port, req, logs, close };
}
// the room plays these messages (rid filled in), in order
const play = (msgs) => (rid, h) => setImmediate(() => { for (const d of msgs) h({ rid, ...d }); });
const GS = { t: "ai-genstart", api: 2, promptTokens: 900 };
const done = (over = {}) => ({ t: "ai-gendone", api: 2, reason: "stop", usage: { in: 900, out: 12, think: 3 }, reused: 850, calls: [], ...over });
const WEATHER = { name: "get_weather", description: "Weather for a city", input_schema: { type: "object", properties: { city: { type: "string" }, days: { type: "integer" } }, required: ["city"] } };
const TIME = { name: "get_time", input_schema: { type: "object", properties: { tz: { type: "string" } } } };
const body = (over = {}) => ({ model: "claude-x", max_tokens: 200, messages: [{ role: "user", content: "Weather in Paris?" }], tools: [WEATHER, TIME], ...over });
// random ids -> stable names, in order of appearance
function stable(s) {
  const seen = new Map();
  return s.replace(/toolu_[0-9A-Za-z]{24}/g, (m) => { if (!seen.has(m)) seen.set(m, `toolu_${seen.size}`); return seen.get(m); }).replace(/msg_[0-9a-z]+/g, "msg_R");
}
const sse = (events) => events.map(([type, obj]) => `event: ${type}\ndata: ${JSON.stringify({ type, ...obj })}\n\n`).join("");
const START = (input = 900) => ["message_start", { message: { id: "msg_R", type: "message", role: "assistant", model: "pooled/qwen36", content: [], stop_reason: null, stop_sequence: null,
  usage: { input_tokens: input, cache_creation_input_tokens: 0, cache_read_input_tokens: 0, output_tokens: 0 } } }];
const PING = ["ping", {}];
const USAGE = { input_tokens: 50, cache_creation_input_tokens: 0, cache_read_input_tokens: 850, output_tokens: 12 };
const blockStart = (index, content_block) => ["content_block_start", { index, content_block }];
const delta = (index, d) => ["content_block_delta", { index, delta: d }];
const stop = (index) => ["content_block_stop", { index }];
const END = (stop_reason, usage = USAGE, stop_sequence = null) => [["message_delta", { delta: { stop_reason, stop_sequence }, usage }], ["message_stop", {}]];

// ---- mapping ----

const CC = JSON.parse(fs.readFileSync(new URL("./fixtures/claude_code_messages.json", import.meta.url), "utf8"));

test("Claude Code's captured requests map without a 400: billing header out, tools, adaptive thinking omitted, effort, env system message folded back", () => {
  const logs = [];
  const reqs = CC.map((c) => finishRequest(parseAnthropic(c.body, { log: (m) => logs.push(m) }), { hostMeta: { api: 2, ctx: 65536 } }));
  for (const r of reqs) {
    assert.ok(r.system.startsWith("You are a Claude agent"), "the per-request billing block is gone");
    assert.ok(!r.system.includes("x-anthropic-billing-header"));
    assert.equal(r.tools.length, 23);
    assert.deepEqual([r.thinking, r.showThinking, r.thinkBudget, r.effort, r.maxTokens, r.toolChoice, r.parallel], [true, false, null, "high", 32000, "auto", true]);
  }
  // [user, system(env)] and [user, system(env), assistant, tool_result] render the same first user turn
  assert.equal(reqs[0].messages.length, 1);
  assert.equal(reqs[1].messages[0].text, reqs[0].messages[0].text);
  assert.ok(reqs[0].messages[0].text.includes("read a.txt\n\n# Environment"), "the env system message folds into the user turn before it");
  assert.deepEqual(reqs[1].messages.slice(1), [
    { role: "assistant", text: "Reading.", calls: [{ id: "toolu_01A", name: "Read", args: { file_path: "/work/project" } }] },
    { role: "tool", id: "toolu_01A", text: "1\thello from a.txt\n2\t" },
  ]);
  const ask = askBody(reqs[1], true);
  assert.equal(ask.api, 2);
  assert.deepEqual(ask.messages.slice(1), [{ role: "assistant", text: "Reading.", calls: [{ name: "Read", args: { file_path: "/work/project" } }] }, { role: "tool", text: "1\thello from a.txt\n2\t" }]);
  assert.deepEqual({ ...ask.params, client: undefined }, { maxTokens: 32000, temperature: undefined, topK: undefined, stop: [], thinking: true, thinkBudget: undefined, client: undefined, toolChoice: "auto", parallel: true, effort: "high" });
  assert.deepEqual(logs, []);
});

const P = (b) => parseAnthropic({ model: "m", max_tokens: 100, ...b });
const throws = (f, re, m) => assert.throws(f, (e) => e instanceof ApiError && e.kind === "bad" && re.test(e.message), m);

test("tool_choice, disable_parallel_tool_use, output_config and thinking map onto the internal request", () => {
  const msgs = [{ role: "user", content: "q" }];
  assert.deepEqual([P({ messages: msgs, tools: [WEATHER], tool_choice: { type: "any" } }).toolChoice, P({ messages: msgs, tools: [WEATHER], tool_choice: { type: "tool", name: "get_weather" } }).toolChoice,
    P({ messages: msgs, tool_choice: { type: "none" } }).toolChoice, P({ messages: msgs, tool_choice: { type: "auto", disable_parallel_tool_use: true } }).parallel],
  ["required", { name: "get_weather" }, "none", false]);
  const r = P({ messages: msgs, tools: [WEATHER], output_config: { effort: "low", format: { type: "json_schema", schema: { type: "object" } } }, thinking: { type: "enabled", budget_tokens: 40 } });
  assert.deepEqual([r.effort, r.format, r.thinking, r.thinkBudget, r.showThinking], ["low", { type: "schema", schema: { type: "object" } }, true, 40, true]);
  assert.deepEqual(P({ messages: msgs, output_format: { type: "json_schema", schema: { type: "array" } } }).format, { type: "schema", schema: { type: "array" } });
  assert.deepEqual(r.tools, [{ name: "get_weather", description: "Weather for a city", parameters: WEATHER.input_schema }]);
  assert.equal(P({ messages: msgs, max_tokens: 128000 }).maxTokens, 65536, "an agent's output limit is capped, not refused");
  assert.equal(P({ messages: msgs, thinking: { type: "disabled" } }).thinking, false);
  throws(() => P({ messages: msgs, output_config: { effort: "extreme" } }), /effort/, "effort");
  throws(() => P({ messages: msgs, tool_choice: { type: "sometimes" } }), /tool_choice.type/, "choice");
  throws(() => P({ messages: msgs, mcp_servers: [{ type: "url", url: "https://x", name: "x" }] }), /mcp_servers/, "mcp");
  throws(() => P({ messages: msgs, tools: [{ name: "f" }] }), /input_schema: Field required/, "schema");
  throws(() => P({ messages: msgs, tools: [{ name: "bad name", input_schema: {} }] }), /tool names/, "name");
  throws(() => P({ messages: msgs, tools: [{ type: "mystery", name: "f", input_schema: {} }] }), /unsupported tool type/, "type");
  throws(() => P({ messages: msgs, thinking: { type: "sometimes" } }), /thinking.type/, "thinking");
  throws(() => finishRequest(P({ messages: msgs, tool_choice: { type: "any" } }), { hostMeta: { api: 2 } }), /needs tools/, "any without tools");
  throws(() => finishRequest(P({ messages: msgs, tools: [WEATHER], tool_choice: { type: "tool", name: "nope" } }), { hostMeta: { api: 2 } }), /nope is not in tools/, "unknown name");
});

test("Anthropic's own tools and their blocks are skipped with one warning; images refused in user turns, a placeholder in tool results", () => {
  const logs = [];
  const log = (m) => logs.push(m);
  const b = { messages: [
    { role: "user", content: "search" },
    { role: "assistant", content: [{ type: "server_tool_use", id: "srvtoolu_1", name: "web_search", input: { query: "x" } }, { type: "web_search_tool_result", tool_use_id: "srvtoolu_1", content: [] }, { type: "text", text: "found" }] },
    { role: "user", content: "thanks" }],
  tools: [{ type: "web_search_20250305", name: "web_search", max_uses: 3 }, { type: "bash_20250124", name: "bash" }, WEATHER] };
  const r = parseAnthropic({ model: "m", max_tokens: 10, ...b }, { log });
  parseAnthropic({ model: "m", max_tokens: 10, ...b }, { log });
  assert.deepEqual(r.tools.map((t) => t.name), ["get_weather"]);
  assert.deepEqual(r.messages[1], { role: "assistant", text: "found" });
  assert.equal(logs.length, 4, "one warning per kind, once per process: " + logs.join(" | "));
  assert.equal(parseAnthropic({ model: "m", max_tokens: 10, tools: [{ type: "web_search_20250305", name: "web_search" }], messages: [{ role: "user", content: "q" }] }).tools, null);
  throws(() => P({ messages: [{ role: "user", content: [{ type: "image", source: {} }] }] }), /only text content/, "image");
  const t = P({ messages: [{ role: "user", content: "q" }, { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "Read", input: {} }] },
    { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", is_error: true, content: [{ type: "text", text: "see:" }, { type: "image", source: {} }] }] }] });
  assert.deepEqual(t.messages[2], { role: "tool", id: "t1", text: "see:" + IMAGE_PLACEHOLDER });
});

test("history: our thinking signature restores the reasoning, results follow their calls' order, text next to results is an aside", () => {
  const r = finishRequest(P({ tools: [WEATHER, TIME], messages: [
    { role: "user", content: "Paris weather and time?" },
    { role: "assistant", content: [{ type: "thinking", thinking: "", signature: blob.encode("two calls") }, { type: "tool_use", id: "a", name: "get_weather", input: { city: "Paris" } }, { type: "tool_use", id: "b", name: "get_time", input: { tz: "CET" } }] },
    { role: "user", content: [{ type: "tool_result", tool_use_id: "b", content: "12:00" }, { type: "tool_result", tool_use_id: "a", content: [{ type: "text", text: "sunny" }] }, { type: "text", text: "<system-reminder>be brief</system-reminder>" }] },
    { role: "assistant", content: [{ type: "thinking", thinking: "seen elsewhere", signature: "EqQBCkYIBxgC" }, { type: "redacted_thinking", data: "xx" }, { type: "text", text: "Sunny, noon." }] },
    { role: "user", content: "thanks" }] }), { hostMeta: { api: 2 } });
  assert.deepEqual(r.messages, [
    { role: "user", text: "Paris weather and time?" },
    { role: "assistant", text: "", calls: [{ id: "a", name: "get_weather", args: { city: "Paris" } }, { id: "b", name: "get_time", args: { tz: "CET" } }], reasoning: "two calls" },
    { role: "tool", id: "a", text: "sunny" }, { role: "tool", id: "b", text: "12:00" },
    { role: "user", text: "<system-reminder>be brief</system-reminder>", aside: true },
    { role: "assistant", text: "Sunny, noon.", reasoning: "seen elsewhere" },
    { role: "user", text: "thanks" }]);
  throws(() => finishRequest(P({ messages: [{ role: "user", content: "q" }, { role: "assistant", content: "pre" }] }), {}), /last message must be from the user or a tool result/, "prefill");
  throws(() => finishRequest(P({ tools: [WEATHER], messages: [{ role: "user", content: "q" }] }), { hostMeta: { api: 1 } }), /older Pooled without tool calling/, "old host");
  throws(() => P({ messages: [{ role: "user", content: [{ type: "mystery" }] }] }), /unsupported content block "mystery"/, "unknown block");
});

// ---- the wire, byte for byte ----

const CALL1 = [GS, { t: "ai-token", text: "Let me check.", th: 1 }, { t: "ai-call", i: 0, name: "get_weather" }, { t: "ai-call", i: 0, a: '{"city": "' }, { t: "ai-call", i: 0, a: 'Paris"}' }, { t: "ai-call", i: 0, end: 1 },
  done({ calls: [{ name: "get_weather", args: '{"city": "Paris"}' }] })];

test("stream: thinking then one tool call (tool_use block, input_json_delta, stop_reason tool_use)", async () => {
  const t = await start();
  t.bridge.onAsk = play(CALL1);
  const r = await t.req("POST", "/v1/messages", { body: body({ stream: true, thinking: { type: "adaptive" } }) });
  assert.equal(r.status, 200);
  assert.equal(r.headers["content-type"], "text/event-stream; charset=utf-8");
  assert.equal(stable(r.body), sse([START(), blockStart(0, { type: "thinking", thinking: "", signature: "" }), PING,
    delta(0, { type: "thinking_delta", thinking: "Let me check." }), delta(0, { type: "signature_delta", signature: blob.encode("Let me check.") }), stop(0),
    blockStart(1, { type: "tool_use", id: "toolu_0", name: "get_weather", input: {} }),
    delta(1, { type: "input_json_delta", partial_json: '{"city": "' }), delta(1, { type: "input_json_delta", partial_json: 'Paris"}' }), stop(1),
    ...END("tool_use")]));
  assert.deepEqual(t.bridge.asks[0].tools.map((x) => x.name), ["get_weather", "get_time"]);
  assert.equal(t.bridge.asks[0].params.thinking, true);
  await t.close();
});

test("non-stream: the same answer as content blocks; the signature carries the reasoning even when omitted", async () => {
  const t = await start();
  t.bridge.onAsk = play(CALL1);
  const r = await t.req("POST", "/v1/messages", { body: body({ thinking: { type: "adaptive", display: "omitted" } }) });
  assert.equal(r.status, 200);
  const m = JSON.parse(r.body);
  assert.match(m.content[1].id, /^toolu_[0-9A-Za-z]{24}$/);
  assert.equal(stable(r.body), JSON.stringify({ id: "msg_R", type: "message", role: "assistant", model: "pooled/qwen36", content: [
    { type: "thinking", thinking: "", signature: blob.encode("Let me check.") },
    { type: "tool_use", id: "toolu_0", name: "get_weather", input: { city: "Paris" } }],
  stop_reason: "tool_use", stop_sequence: null, usage: USAGE }));
  // round trip: the client sends the answer back with the result; the reasoning and the call reach the room
  t.bridge.onAsk = play([GS, { t: "ai-token", text: "Sunny." }, done()]);
  const next = await t.req("POST", "/v1/messages", { body: body({ thinking: { type: "adaptive", display: "omitted" }, messages: [{ role: "user", content: "Weather in Paris?" }, { role: "assistant", content: m.content },
    { role: "user", content: [{ type: "tool_result", tool_use_id: m.content[1].id, content: "sunny, 21 C" }] }] }) });
  assert.equal(next.status, 200);
  assert.deepEqual(t.bridge.asks[1].messages, [{ role: "user", text: "Weather in Paris?" }, { role: "assistant", text: "", calls: [{ name: "get_weather", args: { city: "Paris" } }], reasoning: "Let me check." }, { role: "tool", text: "sunny, 21 C" }]);
  assert.deepEqual(JSON.parse(next.body).content, [{ type: "thinking", thinking: "", signature: blob.encode("") }, { type: "text", text: "Sunny." }]);
  await t.close();
});

test("stream: omitted thinking sends no thinking_delta; text before two parallel calls", async () => {
  const t = await start();
  t.bridge.onAsk = play([GS, { t: "ai-token", text: "hidden", th: 1 }, { t: "ai-token", text: "Checking" }, { t: "ai-token", text: " both." },
    { t: "ai-call", i: 0, name: "get_weather" }, { t: "ai-call", i: 0, a: '{"city": "Oslo"}' }, { t: "ai-call", i: 0, end: 1 },
    { t: "ai-call", i: 1, name: "get_time" }, { t: "ai-call", i: 1, a: "{}" }, { t: "ai-call", i: 1, end: 1 },
    done({ calls: [{ name: "get_weather", args: '{"city": "Oslo"}' }, { name: "get_time", args: "{}" }] })]);
  const r = await t.req("POST", "/v1/messages", { body: body({ stream: true, thinking: { type: "enabled", budget_tokens: 50, display: "omitted" } }) });
  assert.equal(stable(r.body), sse([START(), blockStart(0, { type: "thinking", thinking: "", signature: "" }), PING,
    delta(0, { type: "signature_delta", signature: blob.encode("hidden") }), stop(0),
    blockStart(1, { type: "text", text: "" }), delta(1, { type: "text_delta", text: "Checking" }), delta(1, { type: "text_delta", text: " both." }), stop(1),
    blockStart(2, { type: "tool_use", id: "toolu_0", name: "get_weather", input: {} }), delta(2, { type: "input_json_delta", partial_json: '{"city": "Oslo"}' }), stop(2),
    blockStart(3, { type: "tool_use", id: "toolu_1", name: "get_time", input: {} }), delta(3, { type: "input_json_delta", partial_json: "{}" }), stop(3),
    ...END("tool_use")]));
  assert.equal(t.bridge.asks[0].params.thinkBudget, 50);
  await t.close();
});

test("plain answers: no thinking block without thinking; stop_sequence, max_tokens and a full context", async () => {
  const t = await start();
  t.bridge.onAsk = play([GS, { t: "ai-token", text: "Hi" }, done({ reason: "stop_seq", stopSeq: "END", reused: 0 })]);
  const U0 = { ...USAGE, input_tokens: 900, cache_read_input_tokens: 0 };
  const r = await t.req("POST", "/v1/messages", { body: body({ stream: true, tools: undefined, stop_sequences: ["END"] }) });
  assert.equal(stable(r.body), sse([START(), PING, blockStart(0, { type: "text", text: "" }), delta(0, { type: "text_delta", text: "Hi" }), stop(0), ...END("stop_sequence", U0, "END")]));
  t.bridge.onAsk = play([GS, done({ reason: "ctx", reused: 0 })]);
  const c = await t.req("POST", "/v1/messages", { body: body({ stream: true }) });
  assert.equal(stable(c.body), sse([START(), PING, blockStart(0, { type: "text", text: "" }), stop(0), ...END("model_context_window_exceeded", U0)]), "an empty answer still has its text block");
  t.bridge.onAsk = play([GS, { t: "ai-token", text: "Hi" }, done({ reason: "max", reused: 0 })]);
  assert.equal(stable((await t.req("POST", "/v1/messages", { body: body() })).body), JSON.stringify({ id: "msg_R",
    type: "message", role: "assistant", model: "pooled/qwen36", content: [{ type: "text", text: "Hi" }], stop_reason: "max_tokens", stop_sequence: null, usage: U0 }));
  await t.close();
});

test("a call cut off by max_tokens: streamed block closed, left out of the non-stream answer; a named choice's call is tool_use", async () => {
  const t = await start();
  const cut = [GS, { t: "ai-call", i: 0, name: "get_weather" }, { t: "ai-call", i: 0, a: '{"city": "Ro' }, done({ reason: "max", calls: [], open: { i: 0, name: "get_weather" } })];
  t.bridge.onAsk = play(cut);
  const s = await t.req("POST", "/v1/messages", { body: body({ stream: true }) });
  assert.equal(stable(s.body), sse([START(), PING, blockStart(0, { type: "tool_use", id: "toolu_0", name: "get_weather", input: {} }),
    delta(0, { type: "input_json_delta", partial_json: '{"city": "Ro' }), stop(0), ...END("max_tokens")]));
  const n = JSON.parse((await t.req("POST", "/v1/messages", { body: body() })).body);
  assert.deepEqual([n.content, n.stop_reason], [[{ type: "text", text: "" }], "max_tokens"]);
  t.bridge.onAsk = play(CALL1);
  const named = JSON.parse((await t.req("POST", "/v1/messages", { body: body({ tool_choice: { type: "tool", name: "get_weather" } }) })).body);
  assert.equal(named.stop_reason, "tool_use");
  assert.deepEqual(t.bridge.asks.at(-1).params.toolChoice, { name: "get_weather" });
  t.bridge.onAsk = play(CALL1);
  await t.req("POST", "/v1/messages", { body: body({ tool_choice: { type: "any", disable_parallel_tool_use: true } }) });
  assert.deepEqual([t.bridge.asks.at(-1).params.toolChoice, t.bridge.asks.at(-1).params.parallel], ["required", false]);
  await t.close();
});

test("errors: before the stream in Anthropic's shape; mid-stream an error event and no message_stop; a context overflow keeps Claude Code's message", async () => {
  const t = await start();
  const b = await t.req("POST", "/v1/messages", { body: body({ tool_choice: { type: "tool", name: "nope" } }) });
  assert.equal(b.status, 400);
  assert.deepEqual(JSON.parse(b.body), { type: "error", error: { type: "invalid_request_error", message: "tool_choice: tool nope is not in tools" } });
  t.bridge.onAsk = play([GS, { t: "ai-token", text: "par" }, { t: "ai-gendone", reason: "error", err: "engine died" }]);
  const s = await t.req("POST", "/v1/messages", { body: body({ stream: true }) });
  assert.equal(stable(s.body), sse([START(), PING, blockStart(0, { type: "text", text: "" }), delta(0, { type: "text_delta", text: "par" }),
    ["error", { error: { type: "api_error", message: "generation failed in the room: engine died" } }]]));
  t.bridge.onAsk = (rid, h) => setImmediate(() => h({ t: "ai-busy", rid, code: "ctx", n: 70000, max: 65536 }));
  const c = await t.req("POST", "/v1/messages", { body: body() });
  assert.deepEqual([c.status, JSON.parse(c.body).error.message], [400, "prompt is too long: 70000 tokens > 65536 maximum"]);
  t.bridge.onAsk = play([GS, { t: "ai-call", i: 0, name: "rm_rf" }]);
  const u = await t.req("POST", "/v1/messages", { body: body() });
  assert.equal(u.status, 500, "a call to an undeclared tool ends the request");
  await t.close();
});

test("the Anthropic SDK reads the stream and the whole answer: tool_use input, thinking, usage", async () => {
  const { default: Anthropic } = await import("@anthropic-ai/sdk");
  const t = await start();
  const an = new Anthropic({ baseURL: `http://127.0.0.1:${t.port}`, apiKey: "pooled", maxRetries: 0 });
  t.bridge.onAsk = play(CALL1);
  const fm = await an.messages.stream({ ...body(), thinking: { type: "enabled", budget_tokens: 100 } }).finalMessage();
  assert.equal(fm.stop_reason, "tool_use");
  assert.deepEqual(fm.content.map((c) => c.type), ["thinking", "tool_use"]);
  assert.equal(fm.content[0].thinking, "Let me check.");
  assert.equal(blob.decode(fm.content[0].signature), "Let me check.");
  assert.deepEqual(fm.content[1].input, { city: "Paris" });
  assert.deepEqual(fm.usage, USAGE, "message_delta carries the final split");
  t.bridge.onAsk = play(CALL1);
  const m = await an.messages.create(body());
  assert.deepEqual([m.content[0].type, m.content[1].input, m.stop_reason], ["thinking", { city: "Paris" }, "tool_use"]);
  await t.close();
});
