// cli/lib/openai.js: `pooled serve`'s OpenAI chat completions mapping. Request validation and the
// byte-exact stream for a fixed room message sequence (ai-genstart, ai-token x N, ai-gendone).
import { parseOpenAI, openaiResponse, OpenAIStream, openaiError, openaiModels, finishReason } from "../../cli/lib/openai.js";
import { ApiError, clientFromUA } from "../../cli/lib/common.js";
import { sseData, sseEvent, sseComment } from "../../cli/lib/sse.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ":\n" + ja + "\n!=\n" + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const throws400 = (b, re, m) => {
  try { parseOpenAI(b); } catch (e) { ok(e instanceof ApiError && e.kind === "bad", m + ": kind " + e.kind); ok(re.test(e.message), m + ": " + e.message); eq(openaiError(e).status, 400); return; }
  throw new Error(m + ": no error");
};
const msg = (content, role = "user") => ({ role, content });

Deno.test("openai: a plain request maps to the internal request with defaults", () => {
  const r = parseOpenAI({ model: "anything", messages: [msg("be brief", "system"), msg("hi")] });
  eq(r, { api: "openai", client: "", stream: false, includeUsage: false, system: "be brief", messages: [{ role: "user", text: "hi" }],
    maxTokens: 1024, temperature: null, topK: null, stop: [], thinking: false });
});
Deno.test("openai: system and developer messages anywhere join the system prompt; same roles merge", () => {
  const r = parseOpenAI({ messages: [msg("a", "system"), msg("q1"), msg([{ type: "text", text: "q" }, { type: "text", text: "2" }]), msg("x", "developer"), msg("ans", "assistant"), msg("q3")],
    max_completion_tokens: 7, temperature: 0, top_k: 500, stop: "\n", reasoning_effort: "high", user: "alice", stream: true, stream_options: { include_usage: true } });
  eq(r.system, "a\n\nx");
  eq(r.messages, [{ role: "user", text: "q1\n\nq2" }, { role: "assistant", text: "ans" }, { role: "user", text: "q3" }]);
  eq([r.maxTokens, r.temperature, r.topK, r.stop, r.thinking, r.client, r.stream, r.includeUsage], [7, 0, 64, ["\n"], true, "alice", true, true]);
  eq(parseOpenAI({ messages: [msg("q")], reasoning_effort: "minimal" }).thinking, false);
});
Deno.test("openai: unsupported features are a clear 400", () => {
  throws400({ messages: [msg("q")], tools: [{ type: "function", function: { name: "f" } }] }, /tool calls are not supported by pooled serve yet \(v1 is chat only\)/, "tools");
  throws400({ messages: [msg("q")], tool_choice: "auto" }, /tool calls/, "tool_choice");
  throws400({ messages: [msg("q"), { role: "tool", content: "r", tool_call_id: "x" }] }, /tool calls/, "tool role");
  throws400({ messages: [{ role: "assistant", content: null, tool_calls: [{ id: "x" }] }, msg("q")] }, /tool calls/, "tool_calls");
  throws400({ messages: [msg([{ type: "image_url", image_url: { url: "data:" } }])] }, /only text content is supported/, "image");
  throws400({ messages: [msg("q")], n: 2 }, /n must be 1/, "n");
  throws400({ messages: [msg("q")], logprobs: true }, /logprobs/, "logprobs");
  throws400({ messages: [msg("q")], response_format: { type: "json_object" } }, /JSON mode is not supported yet/, "json");
  throws400({ messages: [msg("q")], presence_penalty: 0.5 }, /presence_penalty/, "penalty");
  throws400({ messages: [msg("q")], logit_bias: { 1: 2 } }, /logit_bias/, "bias");
  throws400({ messages: [msg("q")], temperature: 2.5 }, /temperature/, "temperature");
  throws400({ messages: [msg("q")], max_tokens: 0 }, /max_tokens/, "max_tokens");
  throws400({ messages: [msg("q")], stop: ["a", "b", "c", "d", "e"] }, /at most 4/, "stops");
  throws400({ messages: [msg("q"), msg("prefill", "assistant")] }, /last message must be from the user/, "prefill");
  throws400({ messages: [msg("s", "system")] }, /at least one user message/, "no user");
  throws400({ messages: [{ role: "robot", content: "x" }] }, /role/, "role");
  throws400({}, /messages is required/, "no messages");
  throws400({ messages: [msg("x".repeat(400001))] }, /at most 400000/, "size");
  // accepted and ignored
  parseOpenAI({ messages: [msg("q")], top_p: 0.9, seed: 1, presence_penalty: 0, frequency_penalty: 0, logit_bias: {}, response_format: { type: "text" }, tool_choice: "none" });
});

// the room's messages for one answer, as the bridge receives them
const ROOM = [
  { t: "ai-genstart", rid: "r7", promptTokens: 12 },
  { t: "ai-token", rid: "r7", text: "Hel" }, { t: "ai-token", rid: "r7", text: "lo" },
  { t: "ai-gendone", rid: "r7", reason: "stop", usage: { in: 12, out: 2 } },
];
const BASE = '{"id":"chatcmpl-r7","object":"chat.completion.chunk","created":1790000000,"model":"pooled/qwen3-1.7b","system_fingerprint":null,';
const play = (s, room) => room.map((d) => d.t === "ai-genstart" ? s.start() : d.t === "ai-token" ? s.token(d.text, !!d.th) : s.done({ reason: d.reason, usage: d.usage, reused: d.reused || 0 })).join("");

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
  const r = openaiResponse({ id: "r7", created: 1790000000, model: "pooled/qwen3-1.7b", text: "Hello", think: "", reason: "stop_seq", usage: { in: 12, out: 2 } });
  eq(r, { id: "chatcmpl-r7", object: "chat.completion", created: 1790000000, model: "pooled/qwen3-1.7b", system_fingerprint: null,
    choices: [{ index: 0, message: { role: "assistant", content: "Hello", refusal: null }, logprobs: null, finish_reason: "stop" }],
    usage: { prompt_tokens: 12, completion_tokens: 2, total_tokens: 14 } });
  eq(openaiResponse({ id: "a", created: 0, model: "m", text: "x", think: "t", reason: "stop", usage: { in: 1, out: 1 } }).choices[0].message.reasoning_content, "t");
  eq(["stop", "stop_seq", "abort", "max", "ctx"].map(finishReason), ["stop", "stop", "stop", "length", "length"]);
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
