// POST /v1/chat/completions with tools, tool results, JSON mode and reasoning (docs/design/serve.md
// section 6), through cli/lib/http.js against a stand-in room: the ask the room gets, and the exact
// bytes the client gets back, whole and streamed. Run with `node --test cli/test/`.
import { test } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { EventEmitter } from "node:events";
import OpenAI from "openai";
import { createServer } from "../lib/http.js";

class FakeBridge extends EventEmitter {
  constructor() { super(); this.code = "ABCD"; this.connected = true; this.ready = true; this.model = "qwen36"; this.hostMeta = { api: 2, ctx: 32768 }; this.onAsk = null; this.asks = []; this.stopped = []; }
  ask(rid, body, h) { this.asks.push(body); this.onAsk?.(rid, h, body); return true; }
  stop(rid) { this.stopped.push(rid); }
}
async function start() {
  const bridge = new FakeBridge();
  const logs = [];
  const api = createServer({ bridge, port: 0, log: (m) => logs.push(m) });
  const port = await api.listen();
  const req = (body, { headers = {} } = {}) => new Promise((res) => {
    const r = http.request({ host: "127.0.0.1", port, method: "POST", path: "/v1/chat/completions", headers: { "content-type": "application/json", ...headers } }, (s) => {
      let d = ""; s.on("data", (c) => (d += c)); s.on("end", () => res({ status: s.statusCode, headers: s.headers, body: d }));
    });
    r.write(JSON.stringify(body));
    r.end();
  });
  const close = () => new Promise((r) => { api.closeAll("test over"); api.server.closeAllConnections?.(); api.server.close(r); });
  return { bridge, port, req, logs, close };
}
// an ask as it goes over the wire (JSON: undefined fields left out)
const wire = (x) => JSON.parse(JSON.stringify(x));
// a v2 room that sends these messages (rid filled in; genstart echoes api 2)
const room = (msgs) => (rid, h) => setImmediate(() => { for (const m of msgs) h({ rid, ...(m.t === "ai-genstart" ? { api: 2 } : {}), ...m }); });
// stable text for exact comparisons: ids in order of appearance, created 0
function stable(s) {
  const seen = new Map();
  return s.replace(/call_[0-9A-Za-z]{24}/g, (m) => { if (!seen.has(m)) seen.set(m, `call_${seen.size + 1}`); return seen.get(m); })
    .replace(/"chatcmpl-[^"]+"/g, '"chatcmpl-R"').replace(/"created":\d+/g, '"created":0');
}

const WEATHER = { type: "function", function: { name: "get_weather", description: "Weather for a city", parameters: { type: "object", properties: { city: { type: "string" }, days: { type: "integer" } }, required: ["city"] }, strict: true } };
const TIME = { type: "function", function: { name: "get_time", parameters: { type: "object", properties: { city: { type: "string" } } } } };
const user = (content) => ({ role: "user", content });

// the room's answer: reasoning, a line of text, then two calls
const TWO_CALLS = [
  { t: "ai-genstart", promptTokens: 40 },
  { t: "ai-token", text: "Two cities.", th: 1 },
  { t: "ai-token", text: "Checking." },
  { t: "ai-call", i: 0, name: "get_weather" },
  { t: "ai-call", i: 0, a: '{"city": "' }, { t: "ai-call", i: 0, a: 'Paris"' }, { t: "ai-call", i: 0, a: "}" },
  { t: "ai-call", i: 0, end: 1 },
  { t: "ai-call", i: 1, name: "get_time" },
  { t: "ai-call", i: 1, a: '{"city": "Tokyo"}' },
  { t: "ai-call", i: 1, end: 1 },
  { t: "ai-gendone", reason: "stop", usage: { in: 40, out: 30, think: 4 }, reused: 32, calls: [{ name: "get_weather", args: '{"city": "Paris"}' }, { name: "get_time", args: '{"city": "Tokyo"}' }] },
];

test("chat: the request maps onto a v2 ask (tools, choice, history with calls and results, format, effort)", async () => {
  const t = await start();
  t.bridge.onAsk = room(TWO_CALLS);
  const body = {
    model: "anything",
    messages: [
      { role: "system", content: "Be brief." },
      user("Weather in Paris, time in Tokyo?"),
      { role: "assistant", content: null, reasoning_content: "need tools", tool_calls: [
        { id: "call_a", type: "function", function: { name: "get_weather", arguments: '{"city":"Paris"}' } },
        { id: "call_b", type: "function", function: { name: "get_time", arguments: '{"city":"Tokyo"}' } }] },
      // results out of order: they go back in the calls' order
      { role: "tool", tool_call_id: "call_b", content: "09:00" },
      { role: "tool", tool_call_id: "call_a", content: [{ type: "text", text: "sunny" }] },
      { role: "developer", content: "Answer in French." },
    ],
    tools: [WEATHER, TIME], tool_choice: "required", parallel_tool_calls: false,
    reasoning_effort: "low", max_completion_tokens: 300, temperature: 0.2, stop: ["END"],
    seed: 7, store: false, metadata: { a: "b" }, user: "u", service_tier: "auto", verbosity: "low", top_p: 0.9,
  };
  const r = await t.req(body);
  assert.equal(r.status, 200, r.body);
  assert.deepEqual(wire(t.bridge.asks[0]), {
    api: 2, system: "Be brief.",
    messages: [
      { role: "user", text: "Weather in Paris, time in Tokyo?" },
      { role: "assistant", text: "", calls: [{ name: "get_weather", args: { city: "Paris" } }, { name: "get_time", args: { city: "Tokyo" } }], reasoning: "need tools" },
      { role: "tool", text: "sunny" },
      { role: "tool", text: "09:00" },
      { role: "user", text: "Answer in French.", aside: true },
    ],
    tools: [
      { name: "get_weather", description: "Weather for a city", parameters: WEATHER.function.parameters },
      { name: "get_time", description: "", parameters: TIME.function.parameters },
    ],
    params: { maxTokens: 300, temperature: 0.2, stop: ["END"], thinking: true, client: "API", toolChoice: "required", parallel: false, effort: "low" },
  });
  // a named tool, allowed_tools, json_object, json_schema, reasoning_effort none
  const cases = [
    [{ tools: [WEATHER, TIME], tool_choice: { type: "function", function: { name: "get_time" } } }, { toolChoice: { name: "get_time" } }],
    [{ tools: [WEATHER, TIME], tool_choice: { type: "allowed_tools", allowed_tools: { mode: "required", tools: [{ type: "function", function: { name: "get_weather" } }] } } }, { toolChoice: "required", allowed: ["get_weather"] }],
    [{ response_format: { type: "json_object" } }, { format: { type: "json" } }],
    [{ response_format: { type: "json_schema", json_schema: { name: "city", strict: true, schema: { type: "object", properties: { name: { type: "string" } }, required: ["name"] } } } },
      { format: { type: "schema", schema: { type: "object", properties: { name: { type: "string" } }, required: ["name"] }, name: "city" } }],
    [{ reasoning_effort: "none" }, { thinking: false }],
    [{ reasoning_effort: "xhigh" }, { thinking: true, effort: "xhigh" }],
    [{ chat_template_kwargs: { enable_thinking: true } }, { thinking: true }],
  ];
  for (const [over, want] of cases) {
    t.bridge.onAsk = room(TWO_CALLS.slice(0, 3).concat([{ t: "ai-gendone", reason: "stop", usage: { in: 1, out: 1 } }]));
    const x = await t.req({ messages: [user("q")], ...over });
    assert.equal(x.status, 200, x.body);
    const p = t.bridge.asks.at(-1).params;
    for (const [k, v] of Object.entries(want)) assert.deepEqual(p[k], v, `${JSON.stringify(over)}: ${k}`);
  }
  await t.close();
});

test("chat: tool calls, whole: content, reasoning, tool_calls, finish_reason and usage exactly", async () => {
  const t = await start();
  t.bridge.onAsk = room(TWO_CALLS);
  const r = await t.req({ messages: [user("Weather in Paris, time in Tokyo?")], tools: [WEATHER, TIME], reasoning_effort: "high" });
  assert.equal(stable(r.body), JSON.stringify({
    id: "chatcmpl-R", object: "chat.completion", created: 0, model: "pooled/qwen36", system_fingerprint: null,
    choices: [{ index: 0, message: { role: "assistant", content: "Checking.", refusal: null, reasoning_content: "Two cities.",
      tool_calls: [
        { id: "call_1", type: "function", function: { name: "get_weather", arguments: '{"city": "Paris"}' } },
        { id: "call_2", type: "function", function: { name: "get_time", arguments: '{"city": "Tokyo"}' } }] },
    logprobs: null, finish_reason: "tool_calls" }],
    usage: { prompt_tokens: 40, completion_tokens: 30, total_tokens: 70, prompt_tokens_details: { cached_tokens: 32 }, completion_tokens_details: { reasoning_tokens: 4 } },
  }));
  // only calls: content is null. A named choice ends with "stop" (as OpenAI / vLLM)
  t.bridge.onAsk = room([TWO_CALLS[0], ...TWO_CALLS.slice(3, 8), { t: "ai-gendone", reason: "stop", usage: { in: 40, out: 9 }, calls: [{ name: "get_weather", args: '{"city": "Paris"}' }] }]);
  const n = JSON.parse(await t.req({ messages: [user("q")], tools: [WEATHER], tool_choice: { type: "function", function: { name: "get_weather" } } }).then((x) => x.body));
  assert.equal(n.choices[0].message.content, null);
  assert.equal(n.choices[0].message.tool_calls.length, 1);
  assert.equal(n.choices[0].finish_reason, "stop");
  assert.deepEqual(n.usage, { prompt_tokens: 40, completion_tokens: 9, total_tokens: 49 });
  // cut by max_tokens inside the second call: the complete one stays, the open one is left out
  t.bridge.onAsk = room([...TWO_CALLS.slice(0, 9), { t: "ai-call", i: 1, a: '{"ci' },
    { t: "ai-gendone", reason: "max", usage: { in: 40, out: 20 }, calls: [{ name: "get_weather", args: '{"city": "Paris"}' }], open: { i: 1, name: "get_time" } }]);
  const c = JSON.parse((await t.req({ messages: [user("q")], tools: [WEATHER, TIME] })).body);
  assert.equal(c.choices[0].finish_reason, "length");
  assert.deepEqual(c.choices[0].message.tool_calls.map((x) => x.function), [{ name: "get_weather", arguments: '{"city": "Paris"}' }]);
  await t.close();
});

test("chat: tool calls, streamed: every chunk byte for byte, with include_usage", async () => {
  const t = await start();
  t.bridge.onAsk = room(TWO_CALLS);
  const r = await t.req({ messages: [user("Weather in Paris, time in Tokyo?")], tools: [WEATHER, TIME], reasoning_effort: "high", stream: true, stream_options: { include_usage: true } });
  assert.equal(r.status, 200);
  assert.equal(r.headers["content-type"], "text/event-stream; charset=utf-8");
  const B = '{"id":"chatcmpl-R","object":"chat.completion.chunk","created":0,"model":"pooled/qwen36","system_fingerprint":null,';
  const ch = (delta, finish = null) => `data: ${B}"choices":[{"index":0,"delta":${delta},"logprobs":null,"finish_reason":${finish}}]}\n\n`;
  assert.equal(stable(r.body),
    ch('{"role":"assistant","content":""}') +
    ch('{"reasoning_content":"Two cities."}') +
    ch('{"content":"Checking."}') +
    ch('{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]}') +
    ch('{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\": \\""}}]}') +
    ch('{"tool_calls":[{"index":0,"function":{"arguments":"Paris\\""}}]}') +
    ch('{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]}') +
    ch('{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"get_time","arguments":""}}]}') +
    ch('{"tool_calls":[{"index":1,"function":{"arguments":"{\\"city\\": \\"Tokyo\\"}"}}]}') +
    ch("{}", '"tool_calls"') +
    `data: ${B}"choices":[],"usage":{"prompt_tokens":40,"completion_tokens":30,"total_tokens":70,"prompt_tokens_details":{"cached_tokens":32},"completion_tokens_details":{"reasoning_tokens":4}}}\n\n` +
    "data: [DONE]\n\n");
  // no include_usage (or include_usage without stream): no usage chunk
  t.bridge.onAsk = room(TWO_CALLS);
  const s = await t.req({ messages: [user("q")], tools: [WEATHER, TIME], stream: true });
  assert.doesNotMatch(s.body, /"usage"/);
  assert.match(s.body, /"finish_reason":"tool_calls"\}\]\}\n\ndata: \[DONE\]\n\n$/);
  await t.close();
});

test("chat: JSON mode answers are plain content; a plain request to a v2 host is unchanged", async () => {
  const t = await start();
  t.bridge.onAsk = room([{ t: "ai-genstart", promptTokens: 8 }, { t: "ai-token", text: '{"name": ' }, { t: "ai-token", text: '"Paris"}' }, { t: "ai-gendone", reason: "stop", usage: { in: 8, out: 4 } }]);
  const r = JSON.parse((await t.req({ messages: [user("a city")], response_format: { type: "json_schema", json_schema: { name: "c", schema: { type: "object" } } } })).body);
  assert.deepEqual(r.choices[0], { index: 0, message: { role: "assistant", content: '{"name": "Paris"}', refusal: null }, logprobs: null, finish_reason: "stop" });
  assert.deepEqual(r.usage, { prompt_tokens: 8, completion_tokens: 4, total_tokens: 12 });
  await t.close();
});

test("chat: the OpenAI SDK accumulates streamed calls into the same message as the whole answer", async () => {
  const t = await start();
  const client = new OpenAI({ apiKey: "x", baseURL: `http://127.0.0.1:${t.port}/v1`, maxRetries: 0 });
  t.bridge.onAsk = room(TWO_CALLS);
  const whole = await client.chat.completions.create({ model: "m", messages: [user("q")], tools: [WEATHER, TIME], reasoning_effort: "high" });
  t.bridge.onAsk = room(TWO_CALLS);
  const stream = client.chat.completions.stream({ model: "m", messages: [user("q")], tools: [WEATHER, TIME], reasoning_effort: "high", stream_options: { include_usage: true } });
  const fin = await stream.finalChatCompletion();
  const strip = (m) => ({ content: m.content, tool_calls: m.tool_calls.map((c) => ({ type: c.type, name: c.function.name, arguments: c.function.arguments })) });
  assert.deepEqual(strip(fin.choices[0].message), strip(whole.choices[0].message));
  assert.equal(fin.choices[0].finish_reason, "tool_calls");
  assert.deepEqual(fin.choices[0].message.tool_calls.map((c) => c.function.parsed_arguments ?? JSON.parse(c.function.arguments)), [{ city: "Paris" }, { city: "Tokyo" }]);
  assert.equal(fin.usage.total_tokens, 70);
  await t.close();
});

test("chat: unsupported or malformed requests are a clear 400 in OpenAI's shape", async () => {
  const t = await start();
  t.bridge.onAsk = room(TWO_CALLS);
  const cases = [
    [{ tool_choice: "required" }, "tool_choice", /`tools` must be set/],
    [{ tool_choice: { type: "function", function: { name: "f" } } }, "tool_choice", /`tools` must be set/],
    [{ tools: [WEATHER], tool_choice: { type: "function", function: { name: "nope" } } }, "tool_choice", /does not match any of the specified `tools`/],
    [{ tools: [WEATHER], tool_choice: { type: "function", function: {} } }, "tool_choice.function.name", /Expected field `name`/],
    [{ tools: [WEATHER], tool_choice: "sometimes" }, "tool_choice", /Invalid value for `tool_choice`: sometimes/],
    [{ tools: [WEATHER], tool_choice: { type: "allowed_tools", allowed_tools: { mode: "auto", tools: [{ type: "function", function: { name: "x" } }] } } }, "tool_choice.allowed_tools.tools", /not in tools/],
    [{ tools: [{ type: "custom", custom: { name: "c" } }] }, "tools[0].type", /custom tools are not supported/],
    [{ tools: [{ type: "function", function: { name: "bad name" } }] }, "tools[0].function", /tool names must be/],
    [{ tools: [WEATHER, WEATHER] }, "tools", /declared twice/],
    [{ functions: [{ name: "f" }] }, "functions", /deprecated/],
    [{ function_call: { name: "f" } }, "function_call", /deprecated/],
    [{ messages: [{ role: "function", name: "f", content: "x" }] }, "messages[0].role", /deprecated/],
    [{ messages: [{ role: "assistant", content: "x", function_call: { name: "f" } }, user("q")] }, "messages[0].function_call", /deprecated/],
    [{ messages: [{ role: "assistant", content: null, tool_calls: [{ id: "c", type: "function", function: {} }] }, user("q")] }, "messages[0].tool_calls[0].function.name", /name is required/],
    [{ messages: [user("q"), { role: "assistant", content: "prefill" }] }, "messages", /last message must be from the user or a tool result/],
    [{ messages: [user([{ type: "input_audio", input_audio: { data: "", format: "wav" } }])] }, "messages[0].content", /only text content/],
    [{ response_format: { type: "json_schema", json_schema: { name: "x" } } }, "response_format.json_schema.schema", /schema is required/],
    [{ response_format: { type: "yaml" } }, "response_format.type", /json_object or json_schema/],
    [{ reasoning_effort: "extreme" }, "reasoning_effort", /must be one of none, minimal, low, medium, high, xhigh, max/],
    [{ parallel_tool_calls: "yes" }, "parallel_tool_calls", /boolean/],
    [{ n: 2 }, "n", /n must be 1/],
    [{ logprobs: true }, "logprobs", /logprobs/],
    [{ prediction: { type: "content", content: "x" } }, "prediction", /not supported/],
    [{ web_search_options: {} }, "web_search_options", /not supported/],
    [{ audio: { voice: "x" } }, "modalities", /only text/],
  ];
  for (const [over, param, re] of cases) {
    const r = await t.req({ messages: [user("q")], ...over });
    assert.equal(r.status, 400, `${JSON.stringify(over)}: ${r.body}`);
    const e = JSON.parse(r.body).error;
    assert.equal(e.type, "invalid_request_error");
    assert.equal(e.param, param, JSON.stringify(over));
    assert.match(e.message, re, JSON.stringify(over));
  }
  assert.equal(t.bridge.asks.length, 0, "none reached the room");
  // accepted: tool_choice auto / none with no tools or an empty list, a tool result last
  for (const over of [{ tool_choice: "auto" }, { tool_choice: "none", tools: [] }, { tools: [] }, { messages: [user("q"), { role: "assistant", content: null, tool_calls: [{ id: "c1", type: "function", function: { name: "get_weather", arguments: "{}" } }] }, { role: "tool", tool_call_id: "c1", content: "x" }] }]) {
    t.bridge.onAsk = room(TWO_CALLS.slice(0, 3).concat([{ t: "ai-gendone", reason: "stop", usage: { in: 1, out: 1 } }]));
    assert.equal((await t.req({ messages: [user("q")], ...over })).status, 200, JSON.stringify(over));
  }
  await t.close();
});

test("chat: arguments that are not a JSON object go to the model as {} and are logged", async () => {
  const t = await start();
  t.bridge.onAsk = room([{ t: "ai-genstart", promptTokens: 1 }, { t: "ai-token", text: "ok" }, { t: "ai-gendone", reason: "stop", usage: { in: 1, out: 1 } }]);
  const r = await t.req({ messages: [user("q"), { role: "assistant", content: "", tool_calls: [{ id: "c1", type: "function", function: { name: "f", arguments: "{oops" } }] }, { role: "tool", tool_call_id: "c1", content: "r" }] });
  assert.equal(r.status, 200, r.body);
  assert.deepEqual(t.bridge.asks[0].messages[1].calls, [{ name: "f", args: {} }]);
  assert.ok(t.logs.some((l) => /arguments are not a JSON object.*\{oops/.test(l)), t.logs.join("\n"));
  await t.close();
});

test("chat: an older host: plain chat still works as v1, tools and JSON mode are refused with a 400", async () => {
  const t = await start();
  t.bridge.hostMeta = { api: 1 };
  t.bridge.onAsk = (rid, h) => setImmediate(() => { h({ t: "ai-genstart", rid, promptTokens: 3 }); h({ t: "ai-token", rid, text: "hi" }); h({ t: "ai-gendone", rid, reason: "stop", usage: { in: 3, out: 1 } }); });
  const p = await t.req({ messages: [{ role: "system", content: "s" }, user("q")], reasoning_effort: "medium" });
  assert.equal(p.status, 200);
  assert.deepEqual(wire(t.bridge.asks[0]), { system: "s", messages: [{ role: "user", text: "q" }], params: { maxTokens: 16384, stop: [], thinking: true, client: "API" } });
  for (const over of [{ tools: [WEATHER] }, { response_format: { type: "json_object" } }]) {
    const r = await t.req({ messages: [user("q")], ...over });
    assert.equal(r.status, 400);
    assert.match(JSON.parse(r.body).error.message, /older Pooled without tool calling/);
  }
  await t.close();
});

test("chat: the OpenAI SDK's runTools loop (streamed): calls, its tool results back, then the answer", async () => {
  const t = await start();
  const client = new OpenAI({ apiKey: "x", baseURL: `http://127.0.0.1:${t.port}/v1`, maxRetries: 0 });
  const answers = [TWO_CALLS, [{ t: "ai-genstart", promptTokens: 60 }, { t: "ai-token", text: "Sunny in Paris; 09:00 in Tokyo." }, { t: "ai-gendone", reason: "stop", usage: { in: 60, out: 9 }, reused: 50 }]];
  t.bridge.onAsk = (rid, h) => room(answers.shift())(rid, h);
  const seen = [];
  const fn = (name, out) => ({ type: "function", function: { name, parameters: name === "get_weather" ? WEATHER.function.parameters : TIME.function.parameters, parse: JSON.parse,
    function: (args) => { seen.push([name, args]); return out; } } });
  const runner = client.chat.completions.runTools({ model: "m", stream: true, messages: [user("Weather in Paris, time in Tokyo?")], tools: [fn("get_weather", "sunny"), fn("get_time", "09:00")] });
  assert.equal(await runner.finalContent(), "Sunny in Paris; 09:00 in Tokyo.");
  assert.deepEqual(seen, [["get_weather", { city: "Paris" }], ["get_time", { city: "Tokyo" }]]);
  // the second ask carries the calls and their results, in order, as v2 messages
  assert.deepEqual(wire(t.bridge.asks[1].messages), [
    { role: "user", text: "Weather in Paris, time in Tokyo?" },
    { role: "assistant", text: "Checking.", calls: [{ name: "get_weather", args: { city: "Paris" } }, { name: "get_time", args: { city: "Tokyo" } }], reasoning: "Two cities." },
    { role: "tool", text: "sunny" }, { role: "tool", text: "09:00" },
  ]);
  await t.close();
});

test("chat: no max_tokens means 16384 (a long call is not cut at 1024); a larger one is capped, not refused; images become a note", async () => {
  const t = await start();
  const one = [{ t: "ai-genstart", promptTokens: 5 }, { t: "ai-token", text: "ok" }, { t: "ai-gendone", reason: "stop", usage: { in: 5, out: 1 } }];
  t.bridge.onAsk = room(one);
  assert.equal((await t.req({ messages: [user("q")], tools: [WEATHER] })).status, 200);
  assert.equal(t.bridge.asks.at(-1).params.maxTokens, 16384);
  for (const over of [{ max_tokens: 100000 }, { max_completion_tokens: 128000 }]) {
    t.bridge.onAsk = room(one);
    const r = await t.req({ messages: [user("q")], ...over });
    assert.equal(r.status, 200, r.body);
    assert.equal(t.bridge.asks.at(-1).params.maxTokens, 65536);
  }
  t.bridge.onAsk = room(one);
  const im = await t.req({ messages: [user([{ type: "text", text: "what is this? " }, { type: "image_url", image_url: { url: "data:" } }])] });
  assert.equal(im.status, 200, im.body);
  assert.equal(t.bridge.asks.at(-1).messages[0].text, "what is this? [image omitted: this model reads text only]");
  await t.close();
});

test("chat: the host is not trusted with calls: arguments that are not a JSON object, or a call the tool_choice rules out, end the request", async () => {
  const t = await start();
  const answer = (name, args) => [{ t: "ai-genstart", promptTokens: 5 }, { t: "ai-call", i: 0, name }, { t: "ai-call", i: 0, a: args }, { t: "ai-call", i: 0, end: 1 },
    { t: "ai-gendone", reason: "stop", usage: { in: 5, out: 3 }, calls: [{ name, args }] }];
  const cases = [
    [{ tools: [WEATHER] }, "get_weather", "not json", /not a JSON object/],
    [{ tools: [WEATHER], tool_choice: "none" }, "get_weather", "{}", /not a tool this request allows/],
    [{ tools: [WEATHER, TIME], tool_choice: { type: "function", function: { name: "get_time" } } }, "get_weather", "{}", /not a tool this request allows/],
    [{ tools: [WEATHER, TIME], tool_choice: { type: "allowed_tools", allowed_tools: { mode: "auto", tools: [{ type: "function", function: { name: "get_time" } }] } } }, "get_weather", "{}", /not a tool this request allows/],
  ];
  for (const [over, name, args, re] of cases) {
    t.bridge.onAsk = room(answer(name, args));
    const r = await t.req({ messages: [user("q")], ...over });
    assert.equal(r.status, 500, JSON.stringify(over) + r.body);
    assert.match(JSON.parse(r.body).error.message, re);
  }
  t.bridge.onAsk = room(answer("get_time", '{"city": "Oslo"}'));
  assert.equal((await t.req({ messages: [user("q")], tools: [WEATHER, TIME], tool_choice: { type: "function", function: { name: "get_time" } } })).status, 200);
  await t.close();
});

test("chat: the bridge log gives a tool-call turn's real finish reason", async () => {
  const t = await start();
  try {
    t.bridge.onAsk = room(TWO_CALLS);
    const r = await t.req({ messages: [user("q")], tools: [WEATHER, TIME] });
    assert.equal(r.status, 200, r.body);
    assert.match(t.logs.find((l) => l.includes("40 prompt + 30 tokens")), /40 prompt \+ 30 tokens, tool_calls, 2 calls, 32 reused$/);
    // a named tool_choice ends with "stop", as its finish_reason does
    const one = TWO_CALLS.filter((m) => m.t !== "ai-call" || m.i === 0).map((m) => (m.t === "ai-gendone" ? { ...m, usage: { in: 41, out: 20 }, calls: m.calls.slice(0, 1) } : m));
    t.bridge.onAsk = room(one);
    const n = await t.req({ messages: [user("q")], tools: [WEATHER, TIME], tool_choice: { type: "function", function: { name: "get_weather" } } });
    assert.equal(n.status, 200, n.body);
    assert.equal(JSON.parse(n.body).choices[0].finish_reason, "stop", n.body);
    assert.match(t.logs.find((l) => l.includes("41 prompt + 20 tokens")), /tokens, stop, 1 call, 32 reused$/);
  } finally { await t.close(); }
});
