// The Anthropic Messages API: request validation and mapping onto the internal request, response
// bodies, stream events and errors (docs/design/serve.md section 4).
import { ApiError, bad, TOOLS_MSG, TEXT_MSG, LIMITS, parseStop, checkInt, checkNum, finishMessages } from "./common.js";

const STATUS = { bad: 400, ctx: 400, auth: 401, forbidden: 403, notfound: 404, toolarge: 413, busy: 529, unavailable: 529, server: 500 };
const TYPE = { bad: "invalid_request_error", ctx: "invalid_request_error", auth: "authentication_error", forbidden: "permission_error", notfound: "not_found_error", toolarge: "request_too_large", busy: "overloaded_error", unavailable: "overloaded_error", server: "api_error" };

export function anthropicError(e) {
  const kind = e instanceof ApiError ? e.kind : "server";
  return { status: STATUS[kind] || 500, body: { type: "error", error: { type: TYPE[kind] || "api_error", message: e.message } } };
}

function textOf(content, where) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) throw bad(`${where} must be a string or a list of content blocks`);
  return content.map((p, j) => {
    if (p?.type === "text" && typeof p.text === "string") return p.text;
    if (p?.type === "tool_use" || p?.type === "tool_result" || p?.type === "server_tool_use") throw bad(TOOLS_MSG);
    if (p?.type === "thinking" || p?.type === "redacted_thinking") return "";   // earlier reasoning is not resent to the model
    throw bad(`${TEXT_MSG} (${where}[${j}] is ${JSON.stringify(p?.type ?? typeof p)})`);
  }).join("");
}

export function parseAnthropic(b) {
  if (!b || typeof b !== "object" || Array.isArray(b)) throw bad("the body must be a JSON object");
  if (typeof b.model !== "string" || !b.model) throw bad("model: Field required");
  if (b.max_tokens == null) throw bad("max_tokens: Field required");
  if (!Array.isArray(b.messages)) throw bad("messages: Field required");
  if (b.tools?.length || (b.tool_choice != null && b.tool_choice?.type !== "none")) throw bad(TOOLS_MSG);
  const system = b.system == null ? "" : textOf(b.system, "system");
  const msgs = b.messages.map((m, i) => {
    if (!m || (m.role !== "user" && m.role !== "assistant")) throw bad(`messages.${i}.role: must be user or assistant`);
    return { role: m.role, text: textOf(m.content, `messages.${i}.content`) };
  });
  const messages = finishMessages(msgs, system);
  const maxTokens = checkInt(b.max_tokens, 1, LIMITS.maxTokens, "max_tokens");
  const temperature = checkNum(b.temperature, 0, 1, "temperature");
  checkNum(b.top_p, 0, 1, "top_p");   // accepted, ignored
  const tk = b.top_k == null ? null : checkInt(b.top_k, 1, 1e9, "top_k");
  const t = b.thinking;
  if (t != null && !(t?.type === "enabled" || t?.type === "disabled")) throw bad("thinking.type must be enabled or disabled");
  return {
    api: "anthropic",
    stream: !!b.stream,
    system, messages, maxTokens,
    temperature, topK: tk == null ? null : Math.min(tk, LIMITS.topK),
    stop: parseStop(b.stop_sequences, "stop_sequences"),
    thinking: t?.type === "enabled",
  };
}

const STOP = { stop: "end_turn", abort: "end_turn", stop_seq: "stop_sequence", max: "max_tokens", ctx: "max_tokens" };
export const stopReason = (r) => STOP[r] || "end_turn";
// input_tokens leaves out the tokens read from the cache, as in the Messages API (the OpenAI side counts
// them in prompt_tokens, with cached_tokens as a detail)
const usageOf = (u, reused) => ({ input_tokens: Math.max(0, u.in - (reused || 0)), output_tokens: u.out, ...(reused ? { cache_read_input_tokens: reused } : {}) });

export function anthropicResponse({ id, model, text, think, thinking, reason, stopSeq, usage, reused }) {
  const content = [];
  if (thinking || think) content.push({ type: "thinking", thinking: think || "", signature: "" });
  content.push({ type: "text", text });
  return { id: "msg_" + id, type: "message", role: "assistant", model, content,
    stop_reason: stopReason(reason), stop_sequence: reason === "stop_seq" ? stopSeq ?? null : null, usage: usageOf(usage, reused) };
}

const ev = (type, obj) => `event: ${type}\ndata: ${JSON.stringify(obj)}\n\n`;

// Stream events in exactly the order the Messages API sends them. With thinking on, block 0 is the
// thinking block and the text block is 1.
export class AnthropicStream {
  constructor({ id, model, thinking }) {
    this.id = "msg_" + id; this.model = model; this.thinking = thinking;
    this.block = -1; this.kind = null;
  }
  open(kind) {
    this.block++; this.kind = kind;
    return ev("content_block_start", { type: "content_block_start", index: this.block, content_block: kind === "thinking" ? { type: "thinking", thinking: "" } : { type: "text", text: "" } });
  }
  close() {
    let s = "";
    if (this.kind === "thinking") s += ev("content_block_delta", { type: "content_block_delta", index: this.block, delta: { type: "signature_delta", signature: "" } });
    s += ev("content_block_stop", { type: "content_block_stop", index: this.block });
    this.kind = null;
    return s;
  }
  start(inputTokens, reused = 0) {
    return ev("message_start", { type: "message_start", message: { id: this.id, type: "message", role: "assistant", model: this.model, content: [], stop_reason: null, stop_sequence: null,
      usage: { input_tokens: inputTokens, output_tokens: 0, ...(reused ? { cache_read_input_tokens: reused } : {}) } } })
      + this.open(this.thinking ? "thinking" : "text") + ev("ping", { type: "ping" });
  }
  token(text, th) {
    if (!text) return "";
    let s = "";
    if (th && this.kind !== "thinking") return "";   // reasoning after the answer started: nothing sane to do with it
    if (!th && this.kind !== "text") s += this.close() + this.open("text");
    return s + ev("content_block_delta", { type: "content_block_delta", index: this.block, delta: th ? { type: "thinking_delta", thinking: text } : { type: "text_delta", text } });
  }
  done({ reason, stopSeq, usage, reused }) {
    let s = "";
    if (this.kind !== "text") s += this.close() + this.open("text");
    s += this.close();
    s += ev("message_delta", { type: "message_delta", delta: { stop_reason: stopReason(reason), stop_sequence: reason === "stop_seq" ? stopSeq ?? null : null }, usage: usageOf(usage, reused) });
    return s + ev("message_stop", { type: "message_stop" });
  }
  error(e) { return ev("error", anthropicError(e).body); }
  keepAlive() { return ev("ping", { type: "ping" }); }
}

// GET /v1/models in Anthropic's shape (picked by the anthropic-version or x-api-key header)
export function anthropicModels(model, label, createdMs) {
  const data = model ? [{ type: "model", id: model, display_name: label, created_at: new Date(createdMs).toISOString() }] : [];
  return { data, has_more: false, first_id: data[0]?.id ?? null, last_id: data[0]?.id ?? null };
}
