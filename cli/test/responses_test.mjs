// The Responses API (cli/lib/responses.js) against a stand-in v2 room: request mapping (the ask the
// room gets), the response object, the exact event stream, the store behind previous_response_id and
// GET / DELETE, and the openai SDK reading both. Run with `node --test cli/test/`.
import { test } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { EventEmitter } from "node:events";
import OpenAI from "openai";
import { createServer } from "../lib/http.js";
import { adapter } from "../lib/responses.js";

class FakeBridge extends EventEmitter {
  constructor() { super(); this.code = "ABCD"; this.connected = true; this.ready = true; this.model = "qwen17"; this.hostMeta = { api: 2, ctx: 32768 }; this.onAsk = null; this.stopped = []; this.asks = []; }
  ask(rid, body, h) { this.asks.push(body); this.onAsk?.(rid, h, body); return true; }
  stop(rid) { this.stopped.push(rid); }
}

async function start(opts = {}) {
  const bridge = new FakeBridge();
  const logs = [];
  const api = createServer({ bridge, port: 0, log: (m) => logs.push(m), ...opts });
  const port = await api.listen();
  const req = (method, path, { headers = {}, body = null } = {}) => new Promise((res) => {
    const r = http.request({ host: "127.0.0.1", port, method, path, headers: { "content-type": "application/json", ...headers } }, (s) => {
      let d = ""; s.on("data", (c) => (d += c)); s.on("end", () => res({ status: s.statusCode, headers: s.headers, body: d }));
    });
    r.setTimeout(3000, () => { r.destroy(); res({ status: "timeout" }); });
    if (body != null) r.write(typeof body === "string" ? body : JSON.stringify(body));
    r.end();
  });
  const post = (body, headers) => req("POST", "/v1/responses", { body, headers });
  const close = () => new Promise((r) => { api.closeAll("test over"); api.server.closeAllConnections?.(); api.server.close(r); });
  return { bridge, api, port, req, post, logs, close };
}

// a v2 room that plays a script: [["think", s] | ["text", s] | ["call", name, [frags…]] | ["open", name, [frags…]]], then gendone
const room = (script, { reason = "stop", usage = { in: 20, out: 9, think: 3 }, reused = 0, calls: callsOver } = {}) => (rid, h) => setImmediate(() => {
  h({ t: "ai-genstart", rid, promptTokens: usage.in, api: 2 });
  const calls = [];
  let open = null;
  for (const s of script) {
    if (s[0] === "think") h({ t: "ai-token", rid, text: s[1], th: 1 });
    else if (s[0] === "text") h({ t: "ai-token", rid, text: s[1] });
    else {
      const i = calls.length + (open ? 1 : 0);
      h({ t: "ai-call", rid, i, name: s[1] });
      for (const a of s[2]) h({ t: "ai-call", rid, i, a });
      if (s[0] === "call") { h({ t: "ai-call", rid, i, end: 1 }); calls.push({ name: s[1], args: s[2].join("") }); }
      else open = { i, name: s[1] };
    }
  }
  h({ t: "ai-gendone", rid, api: 2, reason, usage, reused, calls: callsOver ?? calls, ...(open ? { open } : {}) });
});

// ids and times vary: replace each with a stable name by first appearance
function norm(s) {
  const seen = new Map();
  return s.replace(/\b(resp|msg|rs|fc|ctc|call)_[0-9A-Za-z]{24}\b/g, (id, p) => {
    if (!seen.has(id)) seen.set(id, `${p}_${[...seen.keys()].filter((k) => k.startsWith(p + "_")).length + 1}`);
    return seen.get(id);
  }).replace(/"created_at":\d+/g, '"created_at":0');
}
const events = (body) => body.split("\n\n").filter(Boolean).map((b) => {
  const [ev, data] = b.split("\n");
  return { event: ev.replace(/^event: /, ""), data: JSON.parse(data.replace(/^data: /, "")) };
});

const WEATHER = { type: "function", name: "get_weather", description: "Weather in a city", parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] }, strict: false };

// the response object's fields every response echoes (request defaults)
const echo = (over = {}) => ({ background: false, error: null, incomplete_details: null, instructions: null, max_output_tokens: null, max_tool_calls: null, model: "pooled/qwen17",
  parallel_tool_calls: true, previous_response_id: null, reasoning: { effort: null, summary: null }, service_tier: "default", store: true, temperature: 1,
  text: { format: { type: "text" } }, tool_choice: "auto", tools: [], top_p: 1, truncation: "disabled", user: null, metadata: {}, ...over });

test("a plain request: instructions and string input, the ask the room gets, the whole response", async () => {
  const t = await start();
  t.bridge.onAsk = room([["text", "Hello"], ["text", " there."]], { usage: { in: 12, out: 3, think: 0 }, reused: 4 });
  const r = await t.post({ model: "anything", instructions: "Be brief.", input: "Hi", max_output_tokens: 50, temperature: 0.2, metadata: { k: "v" } });
  assert.equal(r.status, 200, r.body);
  assert.deepEqual(JSON.parse(JSON.stringify(t.bridge.asks[0])), { api: 2, system: "Be brief.", messages: [{ role: "user", text: "Hi" }],
    params: { maxTokens: 50, temperature: 0.2, stop: [], thinking: false, client: "API", toolChoice: "auto", parallel: true } });
  const got = JSON.parse(norm(r.body));
  const { id, ...rest } = got;
  assert.equal(id, "resp_1");
  const { created_at, object, status, output, usage, ...fields } = rest;
  assert.equal(object, "response"); assert.equal(status, "completed"); assert.equal(created_at, 0);
  assert.deepEqual(fields, echo({ instructions: "Be brief.", max_output_tokens: 50, temperature: 0.2, metadata: { k: "v" } }));
  assert.deepEqual(output, [{ id: "msg_1", type: "message", status: "completed", role: "assistant", content: [{ type: "output_text", text: "Hello there.", annotations: [], logprobs: [] }] }]);
  assert.deepEqual(usage, { input_tokens: 12, input_tokens_details: { cached_tokens: 4 }, output_tokens: 3, output_tokens_details: { reasoning_tokens: 0 }, total_tokens: 15 });
  await t.close();
});

test("the stream: exact events for reasoning then text, sequence numbers from 0", async () => {
  const t = await start();
  t.bridge.onAsk = room([["think", "Plan"], ["think", "."], ["text", "\n\n"], ["text", "Hi"], ["text", "!"]], { usage: { in: 8, out: 5, think: 2 } });
  const r = await t.post({ input: "Hi", stream: true, reasoning: { effort: "low" } });
  assert.equal(r.status, 200);
  assert.match(r.headers["content-type"], /text\/event-stream/);
  assert.equal(t.bridge.asks[0].params.thinking, true);
  assert.equal(t.bridge.asks[0].params.effort, "low");
  const resp = (status, output, extra = {}) => JSON.stringify({ id: "resp_1", object: "response", created_at: 0, status, ...echo({ reasoning: { effort: "low", summary: null } }), output, usage: null, ...extra });
  const inProgress = JSON.parse(resp("in_progress", []));
  const want = [
    ["response.created", { response: inProgress }],
    ["response.in_progress", { response: inProgress }],
    ["response.output_item.added", { output_index: 0, item: { id: "rs_1", type: "reasoning", summary: [], content: [] } }],
    ["response.content_part.added", { item_id: "rs_1", output_index: 0, content_index: 0, part: { type: "reasoning_text", text: "" } }],
    ["response.reasoning_text.delta", { item_id: "rs_1", output_index: 0, content_index: 0, delta: "Plan" }],
    ["response.reasoning_text.delta", { item_id: "rs_1", output_index: 0, content_index: 0, delta: "." }],
    ["response.reasoning_text.done", { item_id: "rs_1", output_index: 0, content_index: 0, text: "Plan." }],
    ["response.content_part.done", { item_id: "rs_1", output_index: 0, content_index: 0, part: { type: "reasoning_text", text: "Plan." } }],
    ["response.output_item.done", { output_index: 0, item: { id: "rs_1", type: "reasoning", summary: [], content: [{ type: "reasoning_text", text: "Plan." }] } }],
    ["response.output_item.added", { output_index: 1, item: { id: "msg_1", type: "message", status: "in_progress", role: "assistant", content: [] } }],
    ["response.content_part.added", { item_id: "msg_1", output_index: 1, content_index: 0, part: { type: "output_text", text: "", annotations: [], logprobs: [] } }],
    ["response.output_text.delta", { item_id: "msg_1", output_index: 1, content_index: 0, delta: "\n\nHi", logprobs: [] }],
    ["response.output_text.delta", { item_id: "msg_1", output_index: 1, content_index: 0, delta: "!", logprobs: [] }],
    ["response.output_text.done", { item_id: "msg_1", output_index: 1, content_index: 0, text: "\n\nHi!", logprobs: [] }],
    ["response.content_part.done", { item_id: "msg_1", output_index: 1, content_index: 0, part: { type: "output_text", text: "\n\nHi!", annotations: [], logprobs: [] } }],
    ["response.output_item.done", { output_index: 1, item: { id: "msg_1", type: "message", status: "completed", role: "assistant", content: [{ type: "output_text", text: "\n\nHi!", annotations: [], logprobs: [] }] } }],
    ["response.completed", { response: JSON.parse(resp("completed", [
      { id: "rs_1", type: "reasoning", summary: [], content: [{ type: "reasoning_text", text: "Plan." }] },
      { id: "msg_1", type: "message", status: "completed", role: "assistant", content: [{ type: "output_text", text: "\n\nHi!", annotations: [], logprobs: [] }] },
    ], { usage: { input_tokens: 8, input_tokens_details: { cached_tokens: 0 }, output_tokens: 5, output_tokens_details: { reasoning_tokens: 2 }, total_tokens: 13 } })) }],
  ];
  const wire = want.map(([type, obj], n) => `event: ${type}\ndata: ${JSON.stringify({ type, sequence_number: n, ...obj })}\n\n`).join("");
  // key order of the response object aside (usage sits in its own place), the stream is exactly this
  const canon = (s) => events(s).map((e) => JSON.stringify(sortKeys(e)));
  assert.deepEqual(canon(norm(r.body)), canon(wire));
  assert.equal(r.body.split("\n\n").filter(Boolean).every((b) => /^event: (response\.[a-z_.]+)\ndata: \{"type":"\1","sequence_number":\d+,/.test(b)), true, "event line, then data starting with type and sequence_number");
  await t.close();
});
function sortKeys(v) {
  if (Array.isArray(v)) return v.map(sortKeys);
  if (v && typeof v === "object") return Object.fromEntries(Object.keys(v).sort().map((k) => [k, sortKeys(v[k])]));
  return v;
}

test("function calls stream as function_call items; streamed ids equal the final ones; the request maps tools and choice", async () => {
  const t = await start();
  t.bridge.onAsk = room([["call", "get_weather", ['{"city": ', '"Paris"}']]], { usage: { in: 30, out: 6, think: 0 } });
  const body = { input: [{ role: "user", content: "Weather in Paris?" }], tools: [WEATHER, { type: "web_search" }], tool_choice: "auto", parallel_tool_calls: false, stream: true, store: false, reasoning: null, include: [] };
  const r = await t.post(body, { "user-agent": "codex_exec/0.104.0 (Ubuntu 24.4.0; aarch64)" });
  assert.equal(r.status, 200);
  const ask = t.bridge.asks[0];
  assert.deepEqual(ask.tools, [{ name: "get_weather", description: "Weather in a city", parameters: WEATHER.parameters }], "web_search skipped");
  assert.equal(ask.params.parallel, false);
  assert.equal(ask.params.toolChoice, "auto");
  assert.ok(t.logs.some((l) => /skipped the hosted tool web_search/.test(l)), "logged");
  const ev = events(norm(r.body));
  assert.deepEqual(ev.slice(2).map((e) => e.event), ["response.output_item.added", "response.function_call_arguments.delta", "response.function_call_arguments.delta",
    "response.function_call_arguments.done", "response.output_item.done", "response.completed"]);
  assert.deepEqual(ev[2].data.item, { id: "fc_1", type: "function_call", status: "in_progress", arguments: "", call_id: "call_1", name: "get_weather" });
  assert.deepEqual(ev[3].data, { type: "response.function_call_arguments.delta", sequence_number: 3, item_id: "fc_1", output_index: 0, delta: '{"city": ' });
  assert.deepEqual(ev[5].data, { type: "response.function_call_arguments.done", sequence_number: 5, item_id: "fc_1", output_index: 0, name: "get_weather", arguments: '{"city": "Paris"}' });
  const item = { id: "fc_1", type: "function_call", status: "completed", arguments: '{"city": "Paris"}', call_id: "call_1", name: "get_weather" };
  assert.deepEqual(ev[6].data.item, item);
  const done = ev[7].data.response;
  assert.equal(done.status, "completed");
  assert.deepEqual(done.output, [item]);
  assert.equal(done.store, false);
  assert.equal(done.parallel_tool_calls, false);
  assert.deepEqual(done.tools, [WEATHER]);
  assert.deepEqual(ev.map((e) => e.data.sequence_number), ev.map((_, n) => n));
  // store: false keeps nothing
  assert.equal((await t.req("GET", `/v1/responses/${JSON.parse(r.body.split("\n\n").filter(Boolean).pop().split("\ndata: ")[1]).response.id}`)).status, 404);
  await t.close();
});

test("a Codex-shaped conversation: developer message, function_call and its output, reasoning items, hosted items dropped", async () => {
  const t = await start();
  t.bridge.onAsk = room([["text", "It says hello."]]);
  const enc = "pooled1." + Buffer.from("I should read it.").toString("base64url");
  const r = await t.post({
    model: "qwen", instructions: "You are a coding agent.",
    input: [
      { type: "message", role: "developer", content: [{ type: "input_text", text: "<permissions>read-only</permissions>" }] },
      { type: "message", role: "user", content: [{ type: "input_text", text: "<environment_context>cwd</environment_context>" }] },
      { type: "message", role: "user", content: [{ type: "input_text", text: "show a.txt" }] },
      { type: "reasoning", id: "rs_x", summary: [], content: [{ type: "reasoning_text", text: "ignored: the blob wins" }], encrypted_content: enc },
      { type: "message", role: "assistant", id: "msg_x", content: [{ type: "output_text", text: "Reading it." }] },
      { type: "function_call", name: "exec_command", arguments: '{"cmd":"cat a.txt"}', call_id: "call_1" },
      { type: "web_search_call", id: "ws_1", status: "completed", action: { type: "search", query: "x" } },
      { type: "function_call_output", call_id: "call_1", output: "hello from a.txt\n" },
    ],
    tools: [{ type: "function", name: "exec_command", strict: false, parameters: { type: "object", properties: { cmd: { type: "string" } }, required: ["cmd"] } }, { type: "web_search" }],
    tool_choice: "auto", parallel_tool_calls: false, reasoning: null, store: false, stream: false, include: [], prompt_cache_key: "abc",
  });
  assert.equal(r.status, 200, r.body);
  const ask = t.bridge.asks[0];
  assert.equal(ask.system, "You are a coding agent.\n\n<permissions>read-only</permissions>");
  assert.deepEqual(ask.messages, [
    { role: "user", text: "<environment_context>cwd</environment_context>\n\nshow a.txt" },
    { role: "assistant", text: "Reading it.", calls: [{ name: "exec_command", args: { cmd: "cat a.txt" } }], reasoning: "I should read it." },
    { role: "tool", text: "hello from a.txt\n" },
  ]);
  assert.ok(t.logs.some((l) => /dropped web_search_call items/.test(l)));
  assert.equal(JSON.parse(r.body).output[0].content[0].text, "It says hello.");
  await t.close();
});

test("previous_response_id continues the stored conversation; GET, input_items and DELETE read the store", async () => {
  const t = await start();
  t.bridge.onAsk = room([["think", "Need weather."], ["call", "get_weather", ['{"city": "Paris"}']]]);
  const first = await t.post({ input: "Weather in Paris?", tools: [WEATHER], reasoning: { effort: "high" }, include: ["reasoning.encrypted_content"] });
  assert.equal(first.status, 200, first.body);
  const a = JSON.parse(first.body);
  assert.deepEqual(a.output.map((o) => o.type), ["reasoning", "function_call"]);
  assert.equal(a.output[0].encrypted_content, "pooled1." + Buffer.from("Need weather.").toString("base64url"));
  const call = a.output[1];
  assert.match(call.id, /^fc_[0-9A-Za-z]{24}$/); assert.match(call.call_id, /^call_[0-9A-Za-z]{24}$/);

  t.bridge.onAsk = room([["text", "Sunny, 21 C."]]);
  const second = await t.post({ previous_response_id: a.id, input: [{ type: "function_call_output", call_id: call.call_id, output: '{"temp": 21}' }], tools: [WEATHER], instructions: "New instructions." });
  assert.equal(second.status, 200, second.body);
  assert.deepEqual(t.bridge.asks[1].messages, [
    { role: "user", text: "Weather in Paris?" },
    { role: "assistant", text: "", calls: [{ name: "get_weather", args: { city: "Paris" } }], reasoning: "Need weather." },
    { role: "tool", text: '{"temp": 21}' },
  ]);
  assert.equal(t.bridge.asks[1].system, "New instructions.", "instructions are not inherited");
  const b = JSON.parse(second.body);
  assert.equal(b.previous_response_id, a.id);

  // a third turn chains through both
  t.bridge.onAsk = room([["text", "You're welcome."]]);
  const third = await t.post({ previous_response_id: b.id, input: "Thanks" });
  assert.deepEqual(t.bridge.asks[2].messages.map((m) => [m.role, m.text]), [["user", "Weather in Paris?"], ["assistant", ""], ["tool", '{"temp": 21}'], ["assistant", "Sunny, 21 C."], ["user", "Thanks"]]);

  const got = await t.req("GET", `/v1/responses/${a.id}`);
  assert.equal(got.status, 200);
  assert.deepEqual(JSON.parse(got.body), a);
  const items = JSON.parse((await t.req("GET", `/v1/responses/${b.id}/input_items?order=asc`)).body);
  assert.equal(items.object, "list"); assert.equal(items.has_more, false);
  assert.equal(items.data.length, 1);
  assert.equal(items.data[0].type, "function_call_output");
  assert.equal(items.first_id, items.data[0].id);
  // an item_reference to a stored item works like the item itself
  t.bridge.onAsk = room([["text", "ok"]]);
  const ref = await t.post({ input: [{ type: "item_reference", id: JSON.parse(third.body).output[0].id }, { role: "user", content: "and?" }] });
  assert.equal(ref.status, 200, ref.body);
  assert.deepEqual(t.bridge.asks[3].messages, [{ role: "assistant", text: "You're welcome." }, { role: "user", text: "and?" }]);

  const del = await t.req("DELETE", `/v1/responses/${a.id}`);
  assert.deepEqual(JSON.parse(del.body), { id: a.id, object: "response.deleted", deleted: true });
  const gone = await t.req("GET", `/v1/responses/${a.id}`);
  assert.equal(gone.status, 404);
  assert.deepEqual(JSON.parse(gone.body), { error: { message: `Response with id '${a.id}' not found.`, type: "invalid_request_error", param: null, code: "not_found" } });
  assert.equal((await t.req("DELETE", `/v1/responses/${a.id}`)).status, 404);
  const lost = await t.post({ previous_response_id: "resp_nope", input: "x" });
  assert.equal(lost.status, 400);
  assert.equal(JSON.parse(lost.body).error.code, "previous_response_not_found");
  assert.equal(JSON.parse(lost.body).error.param, "previous_response_id");
  await t.close();
});

test("cut at max_output_tokens: incomplete, with an open call as an incomplete item (whole and streamed)", async () => {
  const t = await start();
  const script = [["text", "Let me check."], ["open", "get_weather", ['{"city": "Pa']]];
  t.bridge.onAsk = room(script, { reason: "max" });
  const r = JSON.parse((await t.post({ input: "w?", tools: [WEATHER], max_output_tokens: 8 })).body);
  assert.equal(r.status, "incomplete");
  assert.deepEqual(r.incomplete_details, { reason: "max_output_tokens" });
  assert.deepEqual(r.output.map((o) => [o.type, o.status]), [["message", "completed"], ["function_call", "incomplete"]]);
  assert.equal(r.output[1].arguments, '{"city": "Pa');
  t.bridge.onAsk = room(script, { reason: "max" });
  const s = events(norm((await t.post({ input: "w?", tools: [WEATHER], max_output_tokens: 8, stream: true })).body));
  assert.deepEqual(s.map((e) => e.event).slice(2), ["response.output_item.added", "response.content_part.added", "response.output_text.delta", "response.output_text.done",
    "response.content_part.done", "response.output_item.done", "response.output_item.added", "response.function_call_arguments.delta", "response.output_item.done", "response.incomplete"]);
  // every added item gets its done: the cut call's is incomplete (no arguments.done for it)
  assert.deepEqual(s.at(-2).data.item, { id: "fc_1", type: "function_call", status: "incomplete", arguments: '{"city": "Pa', call_id: "call_1", name: "get_weather" });
  const end = s.at(-1).data.response;
  assert.equal(end.status, "incomplete");
  assert.deepEqual(end.output[1], { id: "fc_1", type: "function_call", status: "incomplete", arguments: '{"city": "Pa', call_id: "call_1", name: "get_weather" });
  // text only, cut: the message itself is incomplete
  t.bridge.onAsk = room([["text", "Once upon"]], { reason: "max" });
  const m = JSON.parse((await t.post({ input: "story" })).body);
  assert.equal(m.output[0].status, "incomplete");
  await t.close();
});

test("the host stopping mid-stream is response.failed, never completed", async () => {
  const t = await start();
  t.bridge.onAsk = (rid, h) => setImmediate(() => {
    h({ t: "ai-genstart", rid, promptTokens: 5, api: 2 });
    h({ t: "ai-token", rid, text: "par" });
    h({ t: "ai-gendone", rid, api: 2, reason: "abort", usage: { in: 5, out: 1 } });
  });
  const s = await t.post({ input: "x", stream: true });
  const ev = events(s.body);
  assert.deepEqual(ev.map((e) => e.event), ["response.created", "response.in_progress", "response.output_item.added", "response.content_part.added", "response.output_text.delta", "response.failed"]);
  const f = ev.at(-1).data.response;
  assert.equal(f.status, "failed");
  assert.equal(f.error.code, "service_unavailable");
  assert.match(f.error.message, /host stopped this answer/);
  assert.equal((await t.post({ input: "x" })).status, 503, "non-stream: an error status");
  await t.close();
});

test("request mapping: tool_choice, text.format, effort, max_tool_calls", async () => {
  const t = await start();
  t.bridge.onAsk = room([["text", "{}"]]);
  const params = async (over) => { const r = await t.post({ input: "x", tools: [WEATHER], ...over }); assert.equal(r.status, 200, r.body); return t.bridge.asks.at(-1).params; };
  assert.equal((await params({ tool_choice: "required" })).toolChoice, "required");
  assert.deepEqual((await params({ tool_choice: { type: "function", name: "get_weather" } })).toolChoice, { name: "get_weather" });
  const al = await params({ tool_choice: { type: "allowed_tools", mode: "required", tools: [{ type: "function", name: "get_weather" }] } });
  assert.equal(al.toolChoice, "required"); assert.deepEqual(al.allowed, ["get_weather"]);
  assert.equal((await params({ tool_choice: { type: "allowed_tools", mode: "auto", tools: [{ type: "web_search" }] } })).toolChoice, "none");
  assert.deepEqual((await params({ text: { format: { type: "json_object" } } })).format, { type: "json" });
  const schema = { type: "object", properties: { a: { type: "integer" } }, required: ["a"] };
  assert.deepEqual((await params({ text: { format: { type: "json_schema", name: "out", schema, strict: true } } })).format, { type: "schema", schema, name: "out" });
  assert.equal((await params({ max_tool_calls: 2 })).maxCalls, 2);
  const off = await params({ reasoning: { effort: "minimal" } });
  assert.equal(off.thinking, false); assert.equal(off.effort, undefined);
  assert.equal((await params({ reasoning: { effort: "xhigh" } })).effort, "xhigh");
  assert.equal((await params({})).maxTokens, 16384, "no max_output_tokens: the Responses default");
  // past what the room ever writes: capped, not refused (the echo keeps the client's value)
  const big = await t.post({ input: "x", max_output_tokens: 128000 });
  assert.equal(big.status, 200, big.body);
  assert.equal(t.bridge.asks.at(-1).params.maxTokens, 65536);
  assert.equal(JSON.parse(big.body).max_output_tokens, 128000);
  await t.close();
});

test("refused with 400: hosted choices, malformed allowed_tools, unknown items, background, logprobs, bad effort", async () => {
  const t = await start();
  t.bridge.onAsk = room([["text", "x"]]);
  const cases = [
    [{ input: "x", tools: [WEATHER], tool_choice: { type: "web_search_preview" } }, "tool_choice", /not supported/],
    // the Chat shape of an allowed tool ({function: {name}}) must not silently turn tools off
    [{ input: "x", tools: [WEATHER], tool_choice: { type: "allowed_tools", mode: "auto", tools: [{ type: "function", function: { name: "get_weather" } }] } }, "tool_choice", /name is required/],
    [{ input: "x", tools: [WEATHER], tool_choice: { type: "allowed_tools", mode: "auto", tools: [{ type: "gizmo" }] } }, "tool_choice", /unknown tool type/],
    [{ input: "x", tools: [WEATHER], tool_choice: { type: "custom", name: "get_weather" } }, "tool_choice", /not a custom tool/],
    [{ input: [{ role: "user", content: [{ type: "input_audio" }] }] }, "input[0].content", /only text content/],
    [{ input: [{ type: "frobnicate" }] }, "input[0].type", /not supported/],
    [{ input: "x", background: true }, "background", /background/],
    [{ input: "x", include: ["message.output_text.logprobs"] }, "include", /logprobs/],
    [{ input: "x", reasoning: { effort: "huge" } }, "reasoning.effort", /reasoning.effort/],
    [{ input: "x", tool_choice: "required" }, "tool_choice", /needs tools/],
    [{ input: "x", text: { format: { type: "json_schema", name: "x" } } }, "text.format", /JSON schema object/],
    [{ input: [{ type: "item_reference", id: "msg_nope" }] }, "input[0].id", /not stored/],
    [{ input: [{ role: "assistant", content: "prefill" }] }, "input", /last input item/],
    [{ input: [] }, "input", /at least one item/],
    [{}, "input", /input is required/],
  ];
  for (const [body, param, re] of cases) {
    const r = await t.post(body);
    assert.equal(r.status, 400, JSON.stringify(body));
    const e = JSON.parse(r.body).error;
    assert.equal(e.type, "invalid_request_error");
    assert.equal(e.param, param, JSON.stringify(body));
    assert.match(e.message, re);
  }
  // images in user input (Codex's view_image, a pasted screenshot) and in function outputs: a note
  const im = await t.post({ input: [{ role: "user", content: [{ type: "input_text", text: "this: " }, { type: "input_image", image_url: "data:x" }, { type: "input_file", file_id: "f" }] }] });
  assert.equal(im.status, 200, im.body);
  assert.equal(t.bridge.asks.at(-1).messages[0].text, "this: [image omitted: this model reads text only][file omitted: this model reads text only]");
  const ok = await t.post({ input: [{ role: "user", content: "look" }, { type: "function_call", call_id: "c1", name: "get_weather", arguments: "{}" },
    { type: "function_call_output", call_id: "c1", output: [{ type: "input_text", text: "see: " }, { type: "input_image", image_url: "data:x" }] }], tools: [WEATHER] });
  assert.equal(ok.status, 200, ok.body);
  assert.equal(t.bridge.asks.at(-1).messages.at(-1).text, "see: [image omitted: this model reads text only]");
  // the routes that are not served
  assert.equal((await t.req("POST", "/v1/responses/input_tokens", { body: {} })).status, 404);
  assert.equal((await t.req("GET", "/v1/responses/resp_x/input_items")).status, 404);
  await t.close();
});

test("an older host: plain Responses requests go as v1, tools get the reload-the-host 400", async () => {
  const t = await start();
  t.bridge.hostMeta = { api: 1 };
  t.bridge.onAsk = (rid, h) => setImmediate(() => { h({ t: "ai-genstart", rid, promptTokens: 3 }); h({ t: "ai-token", rid, text: "ok" }); h({ t: "ai-gendone", rid, reason: "stop", usage: { in: 3, out: 1 } }); });
  const r = await t.post({ input: "hi" });
  assert.equal(r.status, 200, r.body);
  assert.equal(t.bridge.asks[0].api, undefined);
  assert.equal(JSON.parse(r.body).output[0].content[0].text, "ok");
  const tools = await t.post({ input: "hi", tools: [WEATHER] });
  assert.equal(tools.status, 400);
  assert.match(JSON.parse(tools.body).error.message, /older Pooled without tool calling/);
  await t.close();
});

test("the openai SDK: responses.create, responses.stream().finalResponse() and a chain", async () => {
  const t = await start();
  const client = new OpenAI({ baseURL: `http://127.0.0.1:${t.port}/v1`, apiKey: "x", maxRetries: 0 });
  t.bridge.onAsk = room([["think", "Hmm."], ["text", "Hello!"]]);
  const r = await client.responses.create({ model: "m", input: "hi", reasoning: { effort: "low" } });
  assert.equal(r.output_text, "Hello!");
  t.bridge.onAsk = room([["think", "Weather"], ["think", "?"], ["text", "Checking."], ["call", "get_weather", ['{"city"', ': "Oslo"}']]]);
  const stream = client.responses.stream({ model: "m", input: "Oslo weather", tools: [WEATHER], reasoning: { effort: "low" } });
  const deltas = [];
  stream.on("response.function_call_arguments.delta", (e) => deltas.push(e.delta));
  const fin = await stream.finalResponse();
  assert.equal(fin.status, "completed");
  assert.deepEqual(fin.output.map((o) => o.type), ["reasoning", "message", "function_call"]);
  assert.equal(fin.output[0].content[0].text, "Weather?");
  assert.equal(fin.output[1].content[0].text, "Checking.");
  assert.equal(deltas.join(""), fin.output[2].arguments);
  assert.deepEqual(JSON.parse(fin.output[2].arguments), { city: "Oslo" });
  t.bridge.onAsk = room([["text", "Cold."]]);
  const next = await client.responses.create({ model: "m", previous_response_id: fin.id, tools: [WEATHER],
    input: [{ type: "function_call_output", call_id: fin.output[2].call_id, output: "-3 C" }] });
  assert.equal(next.output_text, "Cold.");
  assert.deepEqual(t.bridge.asks.at(-1).messages.map((m) => m.role), ["user", "assistant", "tool"]);
  assert.equal(t.bridge.asks.at(-1).messages[1].reasoning, "Weather?");
  const got = await client.responses.retrieve(next.id);
  assert.equal(got.id, next.id);
  const del = await client.responses.delete(next.id);
  void del;
  await assert.rejects(client.responses.retrieve(next.id), (e) => e.status === 404);
  await t.close();
});

test("the adapter exposes its store (bounded, per process)", () => {
  assert.equal(typeof adapter.store.get, "function");
  assert.equal(adapter.store.max, 256);
  assert.equal(adapter.store.maxBytes, 64 << 20);
});

test("custom (free-form) tools: a function with one string argument to the room, custom_tool_call items out (whole and streamed), and back in", async () => {
  const t = await start();
  const PATCH = { type: "custom", name: "apply_patch", description: "Apply a patch", format: { type: "grammar", syntax: "lark", definition: "start: /.+/" } };
  t.bridge.onAsk = room([["text", "Patching."], ["call", "apply_patch", ['{"input": "*** Begin', ' Patch\\n*** End Patch"}']]]);
  const r = await t.post({ input: "fix it", tools: [PATCH, WEATHER] });
  assert.equal(r.status, 200, r.body);
  const tool = t.bridge.asks[0].tools[0];
  assert.equal(tool.name, "apply_patch");
  assert.deepEqual(tool.parameters.required, ["input"]);
  assert.match(tool.description, /Apply a patch[\s\S]*lark grammar:\nstart: \/\.\+\//);
  const body = JSON.parse(norm(r.body));
  assert.deepEqual(body.output[1], { id: "ctc_1", type: "custom_tool_call", status: "completed", call_id: "call_1", name: "apply_patch", input: "*** Begin Patch\n*** End Patch" });
  assert.deepEqual(body.tools[0], { type: "custom", name: "apply_patch", description: "Apply a patch", format: PATCH.format });
  // streamed: the input goes out whole when the call ends
  t.bridge.onAsk = room([["call", "apply_patch", ['{"input": "a', 'b"}']]]);
  const s = events(norm((await t.post({ input: "fix it", tools: [PATCH], stream: true })).body));
  assert.deepEqual(s.map((e) => e.event).slice(2), ["response.output_item.added", "response.custom_tool_call_input.delta", "response.custom_tool_call_input.done", "response.output_item.done", "response.completed"]);
  assert.deepEqual(s[2].data.item, { id: "ctc_1", type: "custom_tool_call", status: "in_progress", call_id: "call_1", name: "apply_patch", input: "" });
  assert.equal(s[4].data.input, "ab");
  assert.deepEqual(s.at(-1).data.response.output, [{ id: "ctc_1", type: "custom_tool_call", status: "completed", call_id: "call_1", name: "apply_patch", input: "ab" }]);
  // the call and its output in the next request's input (Codex resends the history)
  t.bridge.onAsk = room([["text", "Done."]]);
  const back = await t.post({ tools: [PATCH], tool_choice: { type: "custom", name: "apply_patch" }, input: [{ role: "user", content: "fix it" }, { type: "custom_tool_call", call_id: "c9", name: "apply_patch", input: "ab" },
    { type: "custom_tool_call_output", call_id: "c9", output: "Success" }] });
  assert.equal(back.status, 200, back.body);
  const ask = t.bridge.asks.at(-1);
  assert.deepEqual(ask.messages.slice(1), [{ role: "assistant", text: "", calls: [{ name: "apply_patch", args: { input: "ab" } }] }, { role: "tool", text: "Success" }]);
  assert.deepEqual(ask.params.toolChoice, { name: "apply_patch" });
  await t.close();
});

test("keep-alives are real events (Codex's idle timer ignores SSE comments): response.created while queued, then response.in_progress", async () => {
  const t = await start({ keepAliveMs: 40 });
  t.bridge.onAsk = (rid, h) => setTimeout(() => room([["text", "late"]])(rid, h), 150);
  const s = events((await t.post({ input: "x", stream: true })).body);
  const names = s.map((e) => e.event);
  assert.equal(names[0], "response.created");
  assert.ok(names.filter((n) => n === "response.in_progress").length >= 2, names.join(" "));
  assert.equal(names.at(-1), "response.completed");
  assert.equal(names.filter((n) => n === "response.created").length, 1);
  assert.deepEqual(s.map((e) => e.data.sequence_number), s.map((_, i) => i));
  await t.close();
});
