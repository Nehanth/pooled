// cli/lib/anthropic.js: `pooled serve`'s Anthropic Messages mapping. Request validation and the
// byte-exact event stream for fixed room message sequences (plain, thinking, stop sequence, error).
import { parseAnthropic, anthropicResponse, AnthropicStream, anthropicError, anthropicModels, stopReason } from "../../cli/lib/anthropic.js";
import { ApiError } from "../../cli/lib/common.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ":\n" + ja + "\n!=\n" + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const throws400 = (b, re, m) => {
  try { parseAnthropic(b); } catch (e) { ok(e instanceof ApiError && e.kind === "bad", m + ": kind " + e.kind); ok(re.test(e.message), m + ": " + e.message); eq(anthropicError(e).status, 400); return; }
  throw new Error(m + ": no error");
};
const req = (over = {}) => ({ model: "claude-x", max_tokens: 100, messages: [{ role: "user", content: "hi" }], ...over });

Deno.test("anthropic: a request maps to the internal request", () => {
  const r = parseAnthropic(req({ system: [{ type: "text", text: "be" }, { type: "text", text: " brief" }], temperature: 1, top_k: 5, stop_sequences: ["END"], metadata: { user_id: "bob" },
    thinking: { type: "enabled", budget_tokens: 1024 }, stream: true,
    messages: [{ role: "user", content: [{ type: "text", text: "a" }] }, { role: "user", content: "b" }, { role: "assistant", content: [{ type: "thinking", thinking: "t", signature: "" }, { type: "text", text: "ans" }] }, { role: "user", content: "c" }] }));
  eq(r, { api: "anthropic", client: "bob", stream: true, system: "be brief",
    messages: [{ role: "user", text: "a\n\nb" }, { role: "assistant", text: "ans" }, { role: "user", text: "c" }],
    maxTokens: 100, temperature: 1, topK: 5, stop: ["END"], thinking: true });
});
Deno.test("anthropic: required fields, ranges and unsupported features", () => {
  throws400({ messages: [], max_tokens: 1 }, /model/, "model");
  throws400({ model: "m", messages: [{ role: "user", content: "q" }] }, /max_tokens: Field required/, "max_tokens");
  throws400(req({ temperature: 1.5 }), /temperature/, "temperature is 0..1 here");
  throws400(req({ tools: [{ name: "f", input_schema: {} }] }), /tool calls are not supported by pooled serve yet/, "tools");
  throws400(req({ messages: [{ role: "user", content: [{ type: "tool_result", tool_use_id: "x", content: "r" }] }] }), /tool calls/, "tool_result");
  throws400(req({ messages: [{ role: "user", content: [{ type: "image", source: {} }] }] }), /only text content is supported/, "image");
  throws400(req({ messages: [{ role: "user", content: [{ type: "document", source: {} }] }] }), /only text content/, "document");
  throws400(req({ messages: [{ role: "user", content: "q" }, { role: "assistant", content: "pre" }] }), /last message must be from the user/, "prefill");
  throws400(req({ stop_sequences: ["x".repeat(65)] }), /1 to 64/, "stop length");
  throws400(req({ thinking: { type: "sometimes" } }), /thinking.type/, "thinking");
});

const ROOM = [
  { t: "ai-genstart", promptTokens: 812 },
  { t: "ai-token", text: "Hel" }, { t: "ai-token", text: "lo" },
  { t: "ai-gendone", reason: "stop", usage: { in: 812, out: 2 } },
];
const play = (s, room) => room.map((d) => d.t === "ai-genstart" ? s.start(d.promptTokens) : d.t === "ai-token" ? s.token(d.text, !!d.th) : s.done({ reason: d.reason, stopSeq: d.stopSeq, usage: d.usage })).join("");

Deno.test("anthropic: the stream, byte for byte", () => {
  const s = new AnthropicStream({ id: "r7", model: "pooled/qwen3-1.7b", thinking: false });
  eq(play(s, ROOM),
    'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_r7","type":"message","role":"assistant","model":"pooled/qwen3-1.7b","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":812,"output_tokens":0}}}\n\n' +
    'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n' +
    'event: ping\ndata: {"type":"ping"}\n\n' +
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}\n\n' +
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}\n\n' +
    'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n' +
    'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}\n\n' +
    'event: message_stop\ndata: {"type":"message_stop"}\n\n');
});
Deno.test("anthropic: thinking is block 0 with a signature delta, the text block is 1", () => {
  const s = new AnthropicStream({ id: "r7", model: "m", thinking: true });
  const out = play(s, [ROOM[0], { t: "ai-token", text: "hmm", th: 1 }, { t: "ai-token", text: "Hi" }, { t: "ai-gendone", reason: "stop_seq", stopSeq: "\n", usage: { in: 812, out: 9 } }]);
  const evs = out.split("\n\n").filter(Boolean).map((e) => { const [a, b] = e.split("\n"); return [a.slice(7), JSON.parse(b.slice(6))]; });
  eq(evs.map((e) => e[0]), ["message_start", "content_block_start", "ping", "content_block_delta", "content_block_delta", "content_block_stop", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]);
  eq(evs[1][1].content_block, { type: "thinking", thinking: "" });
  eq(evs[3][1].delta, { type: "thinking_delta", thinking: "hmm" });
  eq(evs[4][1].delta, { type: "signature_delta", signature: "" });
  eq([evs[6][1].index, evs[7][1].delta], [1, { type: "text_delta", text: "Hi" }]);
  eq(evs[9][1].delta, { stop_reason: "stop_sequence", stop_sequence: "\n" });
});
Deno.test("anthropic: thinking asked but none came still ends with a text block; errors and pings", () => {
  const s = new AnthropicStream({ id: "r7", model: "m", thinking: true });
  const out = play(s, [ROOM[0], { t: "ai-gendone", reason: "max", usage: { in: 1, out: 0 } }]);
  ok(out.includes('"index":1,"content_block":{"type":"text","text":""}') && out.includes('"stop_reason":"max_tokens"'), out);
  eq(s.error(new ApiError("server", "boom")), 'event: error\ndata: {"type":"error","error":{"type":"api_error","message":"boom"}}\n\n');
  eq(s.keepAlive(), 'event: ping\ndata: {"type":"ping"}\n\n');
});
Deno.test("anthropic: non-stream response", () => {
  eq(anthropicResponse({ id: "r7", model: "pooled/q", text: "Hello", think: "", thinking: false, reason: "stop", usage: { in: 812, out: 2 } }),
    { id: "msg_r7", type: "message", role: "assistant", model: "pooled/q", content: [{ type: "text", text: "Hello" }], stop_reason: "end_turn", stop_sequence: null, usage: { input_tokens: 812, output_tokens: 2 } });
  const t = anthropicResponse({ id: "a", model: "m", text: "x", think: "t", thinking: true, reason: "stop_seq", stopSeq: "END", usage: { in: 1, out: 1 }, reused: 3 });
  eq(t.content, [{ type: "thinking", thinking: "t", signature: "" }, { type: "text", text: "x" }]);
  eq([t.stop_reason, t.stop_sequence, t.usage.cache_read_input_tokens], ["stop_sequence", "END", 3]);
  eq(["stop", "abort", "stop_seq", "max", "ctx"].map(stopReason), ["end_turn", "end_turn", "stop_sequence", "max_tokens", "max_tokens"]);
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
