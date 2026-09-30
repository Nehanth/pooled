// cli/lib/openai.js: `pooled serve`'s OpenAI chat completions mapping. Request validation and the
// byte-exact stream for a fixed room message sequence (ai-genstart, ai-token x N, ai-gendone).
import { parseOpenAI, openaiResponse, OpenAIStream, openaiError, openaiModels, finishReason } from "../../cli/lib/openai.js";
import { ApiError, clientFromUA, finishRequest, withDefaults } from "../../cli/lib/common.js";
import { sseData, sseEvent, sseComment } from "../../cli/lib/sse.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ":\n" + ja + "\n!=\n" + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const throws400 = (b, re, m) => {
  try { parseOpenAI(b); } catch (e) { ok(e instanceof ApiError && e.kind === "bad", m + ": kind " + e.kind); ok(re.test(e.message), m + ": " + e.message); eq(openaiError(e).status, 400); return; }
  throw new Error(m + ": no error");
};
const msg = (content, role = "user") => ({ role, content });

const fin = (b) => finishRequest(parseOpenAI(b), { hostMeta: { api: 2 } });
Deno.test("openai: a plain request maps to the internal request with defaults", () => {
  const r = fin({ model: "anything", messages: [msg("be brief", "system"), msg("hi")] });
  eq(r, withDefaults({ api: "openai", stream: false, includeUsage: false, system: "be brief", messages: [{ role: "user", text: "hi" }],
    tools: null, toolChoice: "auto", allowed: null, parallel: true, format: null, thinking: false, effort: null,
    maxTokens: 16384, temperature: null, topK: null, stop: [], extra: {} }));
});
Deno.test("openai: leading system / developer messages are the system prompt; later ones fold into the user turn before; same roles merge", () => {
  const r = fin({ messages: [msg("a", "system"), msg("q1"), msg([{ type: "text", text: "q" }, { type: "text", text: "2" }]), msg("x", "developer"), msg("ans", "assistant"), msg("q3")],
    max_completion_tokens: 7, temperature: 0, top_k: 500, stop: "\n", reasoning_effort: "high", user: "alice", stream: true, stream_options: { include_usage: true } });
  eq(r.system, "a");
  eq(r.messages, [{ role: "user", text: "q1\n\nq2\n\nx" }, { role: "assistant", text: "ans" }, { role: "user", text: "q3" }]);
  eq([r.maxTokens, r.temperature, r.topK, r.stop, r.thinking, r.effort, r.client, r.stream, r.includeUsage], [7, 0, 64, ["\n"], true, "high", "", true, true], "user is not the label");
  eq(parseOpenAI({ messages: [msg("q")], reasoning_effort: "minimal" }).thinking, false);
  eq(parseOpenAI({ messages: [msg("q")], stream_options: { include_usage: true } }).includeUsage, false, "include_usage only with stream");
});
Deno.test("openai: tools, tool_choice, tool history, response_format", () => {
  const f = { type: "function", function: { name: "f", parameters: { type: "object", properties: { a: { type: "string" } } } } };
  const r = fin({ messages: [msg("q"), { role: "assistant", content: null, reasoning: "hm", tool_calls: [{ id: "c1", type: "function", function: { name: "f", arguments: '{"a":"x"}' } }] },
    { role: "tool", tool_call_id: "c1", content: "res" }], tools: [f, { type: "function", function: { name: "g" } }], tool_choice: { type: "function", function: { name: "g" } }, parallel_tool_calls: false,
    response_format: { type: "json_object" } });
  eq(r.tools, [{ name: "f", description: "", parameters: f.function.parameters }, { name: "g", description: "", parameters: { type: "object", properties: {} } }]);
  eq([r.toolChoice, r.parallel, r.format], [{ name: "g" }, false, { type: "json" }]);
  eq(r.messages, [{ role: "user", text: "q" }, { role: "assistant", text: "", reasoning: "hm", calls: [{ id: "c1", name: "f", args: { a: "x" } }] }, { role: "tool", text: "res", id: "c1" }]);
  eq(parseOpenAI({ messages: [msg("q")], tools: [f], tool_choice: { type: "allowed_tools", allowed_tools: { mode: "auto", tools: [{ type: "function", function: { name: "f" } }] } } }).allowed, ["f"]);
});
Deno.test("openai: unsupported features are a clear 400", () => {
  throws400({ messages: [msg("q")], tools: [{ type: "custom", custom: { name: "f" } }] }, /custom tools are not supported/, "custom tool");
  throws400({ messages: [msg("q")], tool_choice: "required" }, /`tools` must be set/, "tool_choice");
  throws400({ messages: [msg("q")], tool_choice: { type: "function", function: { name: "f" } } }, /`tools` must be set/, "tool_choice object");
  for (const tc of ["auto", "none"]) eq(parseOpenAI({ messages: [msg("q")], tool_choice: tc, tools: [] }).messages.length, 1, `tool_choice ${tc} with no tools is fine`);
  throws400({ messages: [msg("q"), { role: "function", content: "r", name: "x" }] }, /deprecated/, "function role");
  throws400({ messages: [msg("q")], functions: [{ name: "f" }] }, /deprecated/, "functions");
  eq(parseOpenAI({ messages: [msg([{ type: "image_url", image_url: { url: "data:" } }])] }).messages[0].text, "[image omitted: this model reads text only]", "image: a note");
  throws400({ messages: [msg([{ type: "input_audio", input_audio: {} }])] }, /only text content is supported/, "audio");
  throws400({ messages: [msg("q")], n: 2 }, /n must be 1/, "n");
  throws400({ messages: [msg("q")], logprobs: true }, /logprobs/, "logprobs");
  throws400({ messages: [msg("q")], response_format: { type: "json_schema", json_schema: { name: "x" } } }, /schema is required/, "json_schema");
  throws400({ messages: [msg("q")], reasoning_effort: "huge" }, /reasoning_effort/, "effort");
  throws400({ messages: [msg("q")], presence_penalty: 0.5 }, /presence_penalty/, "penalty");
  throws400({ messages: [msg("q")], logit_bias: { 1: 2 } }, /logit_bias/, "bias");
  throws400({ messages: [msg("q")], temperature: 2.5 }, /temperature/, "temperature");
  throws400({ messages: [msg("q")], max_tokens: 0 }, /max_tokens/, "max_tokens");
  throws400({ messages: [msg("q")], stop: ["a", "b", "c", "d", "e"] }, /at most 4/, "stops");
  throws400({ messages: [{ role: "robot", content: "x" }] }, /role/, "role");
  throws400({}, /messages is required/, "no messages");
  const finThrows = (b, re, m) => { try { fin(b); } catch (e) { ok(e.kind === "bad" && re.test(e.message), m + ": " + e.message); return; } throw new Error(m + ": no error"); };
  finThrows({ messages: [msg("q"), msg("prefill", "assistant")] }, /last message must be from the user or a tool result/, "prefill");
  finThrows({ messages: [msg("s", "system")] }, /at least one user message/, "no user");
  finThrows({ messages: [msg("x".repeat(1500001))] }, /at most 1500000/, "size");
  // accepted and ignored
  parseOpenAI({ messages: [msg("q")], top_p: 0.9, seed: 1, presence_penalty: 0, frequency_penalty: 0, logit_bias: {}, response_format: { type: "text" }, tool_choice: "none", store: true, metadata: {}, service_tier: "auto" });
});

// the room's messages for one answer, as the bridge receives them
const ROOM = [
  { t: "ai-genstart", rid: "r7", promptTokens: 12 },
  { t: "ai-token", rid: "r7", text: "Hel" }, { t: "ai-token", rid: "r7", text: "lo" },
  { t: "ai-gendone", rid: "r7", reason: "stop", usage: { in: 12, out: 2 } },
];
const BASE = '{"id":"chatcmpl-r7","object":"chat.completion.chunk","created":1790000000,"model":"pooled/qwen3-1.7b","system_fingerprint":null,';
const play = (s, room) => room.map((d) => d.t === "ai-genstart" ? s.start() : d.t === "ai-token" ? s.token(d.text, !!d.th) : s.done({ reason: d.reason, calls: [], usage: { think: 0, ...d.usage }, reused: d.reused || 0 })).join("");

Deno.test("openai: the stream, byte for byte", () => {
  const s = new OpenAIStream({ id: "r7", created: 1790000000, model: "pooled/qwen3-1.7b", includeUsage: false });
  eq(play(s, ROOM),
    `data: ${BASE}"choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}\n\n` +
    `data: ${BASE}"choices":[{"index":0,"delta":{"content":"Hel"},"logprobs":null,"finish_reason":null}]}\n\n` +
    `data: ${BASE}"choices":[{"index":0,"delta":{"content":"lo"},"logprobs":null,"finish_reason":null}]}\n\n` +
    `data: ${BASE}"choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}]}\n\n` +
    "data: [DONE]\n\n");
});
Deno.test("openai: stream with usage, thinking and a length ending", () => {
  const s = new OpenAIStream({ id: "r7", created: 1790000000, model: "pooled/qwen3-1.7b", includeUsage: true });
  const out = play(s, [ROOM[0], { t: "ai-token", text: "hmm", th: 1 }, { t: "ai-token", text: "Hi" }, { t: "ai-gendone", reason: "max", usage: { in: 12, out: 5 }, reused: 10 }]);
  const lines = out.split("\n\n").filter(Boolean);
  eq(lines.length, 6);
  eq(JSON.parse(lines[1].slice(6)).choices[0].delta, { reasoning_content: "hmm" });
  eq(JSON.parse(lines[2].slice(6)).choices[0].delta, { content: "Hi" });
  eq(JSON.parse(lines[3].slice(6)).choices[0].finish_reason, "length");
  eq(JSON.parse(lines[4].slice(6)), { id: "chatcmpl-r7", object: "chat.completion.chunk", created: 1790000000, model: "pooled/qwen3-1.7b", system_fingerprint: null, choices: [],
    usage: { prompt_tokens: 12, completion_tokens: 5, total_tokens: 17, prompt_tokens_details: { cached_tokens: 10 } } });
  eq(lines[5], "data: [DONE]");
});
Deno.test("openai: an error after the headers is a data line without [DONE]; keep-alives are comments", () => {
  const s = new OpenAIStream({ id: "r7", created: 1, model: "m", includeUsage: false });
  eq(s.error(new ApiError("server", "boom")), 'data: {"error":{"message":"boom","type":"server_error","param":null,"code":"server_error"}}\n\n');
  eq(s.keepAlive(2), ": queued, 2 ahead\n\n");
});
Deno.test("openai: non-stream response and finish reasons", () => {
  const r = openaiResponse({ id: "r7", created: 1790000000, model: "pooled/qwen3-1.7b", text: "Hello", think: "", calls: [], reason: "stop_seq", usage: { in: 12, out: 2, think: 0 } });
  eq(r, { id: "chatcmpl-r7", object: "chat.completion", created: 1790000000, model: "pooled/qwen3-1.7b", system_fingerprint: null,
    choices: [{ index: 0, message: { role: "assistant", content: "Hello", refusal: null }, logprobs: null, finish_reason: "stop" }],
    usage: { prompt_tokens: 12, completion_tokens: 2, total_tokens: 14 } });
  const t = openaiResponse({ id: "a", created: 0, model: "m", text: "x", think: "t", calls: [], reason: "stop", usage: { in: 1, out: 1, think: 1 } }, { thinking: true });
  eq(t.choices[0].message.reasoning_content, "t");
  eq(t.usage.completion_tokens_details, { reasoning_tokens: 1 });
  const c = openaiResponse({ id: "a", created: 0, model: "m", text: "", think: "", calls: [{ id: "call_1", name: "f", args: "{}" }], reason: "stop", usage: { in: 1, out: 1, think: 0 } }, { toolChoice: "auto" });
  eq(c.choices[0].message, { role: "assistant", content: null, refusal: null, tool_calls: [{ id: "call_1", type: "function", function: { name: "f", arguments: "{}" } }] });
  eq(c.choices[0].finish_reason, "tool_calls");
  eq(["tool", "stop", "stop_seq", "length", "ctx"].map(finishReason), ["tool_calls", "stop", "stop", "length", "length"]);
});
Deno.test("openai: error shapes and statuses", () => {
  const e = (kind, m, o) => openaiError(new ApiError(kind, m, o));
  eq(e("auth", "invalid API key"), { status: 401, body: { error: { message: "invalid API key", type: "invalid_request_error", param: null, code: "invalid_api_key" } } });
  eq(e("ctx", "too long", { param: "messages" }).body.error, { message: "too long", type: "invalid_request_error", param: "messages", code: "context_length_exceeded" });
  eq([e("forbidden").status, e("notfound").status, e("toolarge").status, e("busy").status, e("unavailable").status, e("server").status], [403, 404, 413, 429, 503, 500]);
  eq(openaiError(new Error("x")).status, 500);
});
Deno.test("openai: models list", () => {
  eq(openaiModels("pooled/qwen3-1.7b", 1790000000), { object: "list", data: [{ id: "pooled/qwen3-1.7b", object: "model", created: 1790000000, owned_by: "pooled" }] });
  eq(openaiModels(null, 0), { object: "list", data: [] });
});
Deno.test("sse: framing helpers and client labels", () => {
  eq(sseData({ a: 1 }), 'data: {"a":1}\n\n'); eq(sseData("[DONE]"), "data: [DONE]\n\n");
  eq(sseEvent("ping", { type: "ping" }), 'event: ping\ndata: {"type":"ping"}\n\n');
  eq(sseComment("hi"), ": hi\n\n");
  eq(["OpenAI/Python 1.40.0", "OpenAI/JS 5.23.2", "Anthropic/JS 0.60.0", "curl/8.5.0", "Mozilla/5.0 Continue", "", "weird-tool/1 (x)"].map(clientFromUA),
    ["OpenAI/Python", "OpenAI/JS", "Anthropic/JS", "curl", "Continue", "", "weird-tool"]);
});
