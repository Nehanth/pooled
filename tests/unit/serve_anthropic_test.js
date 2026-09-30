// cli/lib/anthropic.js: `pooled serve`'s Anthropic Messages mapping. Request validation and the
// byte-exact event stream for fixed answers (plain, thinking, tool calls, stop sequence, error).
// The same endpoint through the HTTP server and the Anthropic SDK: cli/test/anthropic_test.mjs.
import { parseAnthropic, anthropicResponse, AnthropicStream, anthropicError, anthropicModels, stopReason, usageOf } from "../../cli/lib/anthropic.js";
import { ApiError, blob, IMAGE_PLACEHOLDER, FILE_PLACEHOLDER } from "../../cli/lib/common.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ":\n" + ja + "\n!=\n" + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const throws400 = (b, re, m) => {
  try { parseAnthropic(b); } catch (e) { ok(e instanceof ApiError && e.kind === "bad", m + ": kind " + e.kind); ok(re.test(e.message), m + ": " + e.message); eq(anthropicError(e).status, 400); return; }
  throw new Error(m + ": no error");
};
const req = (over = {}) => ({ model: "claude-x", max_tokens: 100, messages: [{ role: "user", content: "hi" }], ...over });

Deno.test("anthropic: a request maps to the internal request", () => {
  const r = parseAnthropic(req({ system: [{ type: "text", text: "x-anthropic-billing-header: cc_version=1;" }, { type: "text", text: "be" }, { type: "text", text: " brief", cache_control: { type: "ephemeral" } }],
    temperature: 1, top_k: 5, stop_sequences: ["END"], metadata: { user_id: "bob" }, thinking: { type: "enabled", budget_tokens: 1024 }, stream: true,
    messages: [{ role: "user", content: [{ type: "text", text: "a" }] }, { role: "user", content: "b" }, { role: "assistant", content: [{ type: "thinking", thinking: "t", signature: "" }, { type: "text", text: "ans" }] }, { role: "user", content: "c" }] }));
  eq([r.api, r.stream, r.system, r.messages, r.maxTokens, r.temperature, r.topK, r.stop, r.thinking, r.thinkBudget, r.showThinking, r.tools, r.toolChoice, r.parallel],
    ["anthropic", true, "be brief", [{ role: "user", text: "a" }, { role: "user", text: "b" }, { role: "assistant", text: "ans", reasoning: "t" }, { role: "user", text: "c" }], 100, 1, 5, ["END"], true, null, true, null, "auto", true],
    "a budget over max_tokens is no budget; the billing header is dropped");
  eq(parseAnthropic(req({ max_tokens: 2000, thinking: { type: "enabled", budget_tokens: 1024 } })).thinkBudget, 1024);
  eq(parseAnthropic(req({ thinking: { type: "disabled" } })).thinking, false);
  const a = parseAnthropic(req({ thinking: { type: "adaptive", display: "omitted" } }));
  eq([a.thinking, a.thinkBudget, a.showThinking], [true, null, false]);
});
Deno.test("anthropic: required fields, ranges and unsupported features", () => {
  throws400({ messages: [], max_tokens: 1 }, /model/, "model");
  throws400({ model: "m", messages: [{ role: "user", content: "q" }] }, /max_tokens: Field required/, "max_tokens");
  throws400(req({ max_tokens: 0 }), /max_tokens/, "max_tokens 0");
  throws400(req({ temperature: 1.5 }), /temperature/, "temperature is 0..1 here");
  // images and documents become a note (a pasted screenshot must not break every later request)
  eq(parseAnthropic(req({ messages: [{ role: "user", content: [{ type: "image", source: {} }, { type: "document", source: {} }] }] })).messages[0].text, IMAGE_PLACEHOLDER + FILE_PLACEHOLDER, "image, document");
  throws400(req({ messages: [{ role: "user", content: [{ type: "search_result", source: "s", title: "t", content: [] }] }] }), /only text content/, "search_result");
  throws400(req({ stop_sequences: ["x".repeat(65)] }), /1 to 64/, "stop length");
  throws400(req({ thinking: { type: "sometimes" } }), /thinking.type/, "thinking");
  throws400(req({ tool_choice: { type: "tool" } }), /tool_choice.name/, "named choice without a name");
  throws400(req({ output_config: { format: { type: "regex" } } }), /json_schema/, "format");
  throws400(req({ messages: [{ role: "developer", content: "x" }] }), /role/, "role");
  for (const type of ["auto", "none"]) eq(parseAnthropic(req({ tool_choice: { type }, tools: [] })).messages.length, 1, `tool_choice ${type} with no tools is fine`);
});

// a fixed answer played into a stream: [kind, ...args]
const play = (s, steps) => steps.map(([k, ...a]) => s[k](...a)).join("");
const events = (out) => out.split("\n\n").filter(Boolean).map((e) => { const [a, b] = e.split("\n"); return [a.slice(7), JSON.parse(b.slice(6))]; });
const A = (over = {}) => ({ id: "r7", model: "m", think: "", text: "", calls: [], open: null, reason: "stop", stopSeq: null, usage: { in: 812, out: 2, think: 0 }, reused: 0, ...over });
const R = (over = {}) => ({ toolChoice: "auto", thinking: false, showThinking: true, ...over });

Deno.test("anthropic: the stream, byte for byte", () => {
  const s = new AnthropicStream({ id: "r7", model: "pooled/qwen3-1.7b" });
  eq(play(s, [["start", 812], ["text", "Hel"], ["text", "lo"], ["done", A({ text: "Hello" }), R()]]),
    'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_r7","type":"message","role":"assistant","model":"pooled/qwen3-1.7b","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":812,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":0}}}\n\n' +
    'event: ping\ndata: {"type":"ping"}\n\n' +
    'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n' +
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}\n\n' +
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}\n\n' +
    'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n' +
    'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":812,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":2}}\n\n' +
    'event: message_stop\ndata: {"type":"message_stop"}\n\n');
});
Deno.test("anthropic: thinking is block 0 with the reasoning in its signature, the text block is 1", () => {
  const s = new AnthropicStream({ id: "r7", model: "m", thinking: true });
  const evs = events(play(s, [["start", 812], ["thinkText", "hmm"], ["text", "Hi"], ["done", A({ think: "hmm", text: "Hi", reason: "stop_seq", stopSeq: "\n" }), R({ thinking: true })]]));
  eq(evs.map((e) => e[0]), ["message_start", "content_block_start", "ping", "content_block_delta", "content_block_delta", "content_block_stop", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]);
  eq(evs[1][1].content_block, { type: "thinking", thinking: "", signature: "" });
  eq(evs[3][1].delta, { type: "thinking_delta", thinking: "hmm" });
  eq(evs[4][1].delta, { type: "signature_delta", signature: blob.encode("hmm") });
  eq(blob.decode(evs[4][1].delta.signature), "hmm");
  eq([evs[6][1].index, evs[7][1].delta], [1, { type: "text_delta", text: "Hi" }]);
  eq(evs[9][1].delta, { stop_reason: "stop_sequence", stop_sequence: "\n" });
});
Deno.test("anthropic: thinking asked but none came still ends with a text block; errors and pings", () => {
  const s = new AnthropicStream({ id: "r7", model: "m", thinking: true });
  const out = play(s, [["start", 1], ["done", A({ reason: "max" }), R({ thinking: true })]]);
  ok(out.includes('"index":1,"content_block":{"type":"text","text":""}') && out.includes('"stop_reason":"max_tokens"'), out);
  eq(s.error(new ApiError("server", "boom")), 'event: error\ndata: {"type":"error","error":{"type":"api_error","message":"boom"}}\n\n');
  eq(s.keepAlive(), 'event: ping\ndata: {"type":"ping"}\n\n');
});
Deno.test("anthropic: tool calls stream as tool_use blocks with input_json_delta; stop_reason tool_use", () => {
  const s = new AnthropicStream({ id: "r7", model: "m" });
  const calls = [{ id: "toolu_a", name: "f", args: '{"x": 1}' }];
  const evs = events(play(s, [["start", 5], ["text", "ok"], ["callStart", 0, "toolu_a", "f"], ["callArgs", 0, '{"x": '], ["callArgs", 0, "1}"], ["callEnd", 0], ["done", A({ text: "ok", calls }), R()]]));
  eq(evs.map((e) => [e[0], e[1].index ?? null]), [["message_start", null], ["ping", null], ["content_block_start", 0], ["content_block_delta", 0], ["content_block_stop", 0],
    ["content_block_start", 1], ["content_block_delta", 1], ["content_block_delta", 1], ["content_block_stop", 1], ["message_delta", null], ["message_stop", null]]);
  eq(evs[5][1].content_block, { type: "tool_use", id: "toolu_a", name: "f", input: {} });
  eq(evs[6][1].delta.partial_json + evs[7][1].delta.partial_json, '{"x": 1}');
  eq(evs[9][1].delta.stop_reason, "tool_use");
  eq(stopReason(A({ calls }), R({ toolChoice: { name: "f" } })), "tool_use", "a named choice's call is tool_use too");
  eq(["stop", "stop_seq", "max", "ctx"].map((reason) => stopReason(A({ reason }), R())), ["end_turn", "stop_sequence", "max_tokens", "model_context_window_exceeded"]);
});
Deno.test("anthropic: non-stream response", () => {
  eq(anthropicResponse(A({ text: "Hello" }), R()),
    { id: "msg_r7", type: "message", role: "assistant", model: "m", content: [{ type: "text", text: "Hello" }], stop_reason: "end_turn", stop_sequence: null,
      usage: { input_tokens: 812, cache_creation_input_tokens: 0, cache_read_input_tokens: 0, output_tokens: 2 } });
  const t = anthropicResponse(A({ text: "x", think: "t", reason: "stop_seq", stopSeq: "END", usage: { in: 39, out: 1 }, reused: 22 }), R({ thinking: true }));
  eq(t.content, [{ type: "thinking", thinking: "t", signature: blob.encode("t") }, { type: "text", text: "x" }]);
  eq([t.stop_reason, t.stop_sequence], ["stop_sequence", "END"]);
  eq(t.usage, { input_tokens: 17, cache_creation_input_tokens: 0, cache_read_input_tokens: 22, output_tokens: 1 }, "input_tokens leaves out the cache reads");
  const c = anthropicResponse(A({ think: "why", calls: [{ id: "toolu_a", name: "f", args: '{"x": [1]}' }], open: { id: "toolu_b", name: "f", args: '{"x' } }), R({ thinking: true, showThinking: false }));
  eq(c.content, [{ type: "thinking", thinking: "", signature: blob.encode("why") }, { type: "tool_use", id: "toolu_a", name: "f", input: { x: [1] } }], "no empty text next to calls; the open call is left out");
  eq(c.stop_reason, "tool_use");
  eq(usageOf({ in: 5, out: 1 }, 9), { input_tokens: 0, cache_creation_input_tokens: 0, cache_read_input_tokens: 9, output_tokens: 1 });
});
Deno.test("anthropic: error shapes and statuses", () => {
  const e = (kind, m) => anthropicError(new ApiError(kind, m));
  eq(e("auth", "invalid API key"), { status: 401, body: { type: "error", error: { type: "authentication_error", message: "invalid API key" } } });
  eq([e("busy").status, e("busy").body.error.type, e("unavailable").status, e("notfound").body.error.type, e("toolarge").body.error.type, e("server").body.error.type],
    [529, "overloaded_error", 529, "not_found_error", "request_too_large", "api_error"]);
});
Deno.test("anthropic: models list", () => {
  eq(anthropicModels("pooled/q", "Qwen3 1.7B · Q8 (Pooled room ABCD)", Date.UTC(2026, 8, 28)),
    { data: [{ type: "model", id: "pooled/q", display_name: "Qwen3 1.7B · Q8 (Pooled room ABCD)", created_at: "2026-09-28T00:00:00.000Z" }], has_more: false, first_id: "pooled/q", last_id: "pooled/q" });
  eq(anthropicModels(null, "", 0).data, []);
});
