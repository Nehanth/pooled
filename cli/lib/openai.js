// The OpenAI chat completions API: request validation and mapping onto the internal request,
// response bodies, stream chunks and errors (docs/design/serve.md section 4).
import { ApiError, bad, TOOLS_MSG, TEXT_MSG, LIMITS, parseStop, checkInt, checkNum, finishMessages } from "./common.js";

const STATUS = { bad: 400, ctx: 400, auth: 401, forbidden: 403, notfound: 404, method: 405, toolarge: 413, busy: 429, unavailable: 503, timeout: 504, server: 500 };
const TYPE = { bad: "invalid_request_error", ctx: "invalid_request_error", auth: "invalid_request_error", forbidden: "permission_error", notfound: "invalid_request_error", method: "invalid_request_error", toolarge: "invalid_request_error", busy: "rate_limit_exceeded", unavailable: "server_error", timeout: "server_error", server: "server_error" };
const CODE = { ctx: "context_length_exceeded", auth: "invalid_api_key", notfound: "not_found", method: "method_not_allowed", toolarge: "request_too_large", busy: "rate_limit_exceeded", unavailable: "service_unavailable", timeout: "timeout", server: "server_error" };

export function openaiError(e) {
  const kind = e instanceof ApiError ? e.kind : "server";
  return { status: STATUS[kind] || 500, body: { error: { message: e.message, type: TYPE[kind] || "server_error", param: e.param ?? null, code: CODE[kind] ?? null } } };
}

// content: a string, or a list of parts of which only text is accepted
function textOf(content, i) {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) throw bad(`messages[${i}].content must be a string or a list of parts`, `messages[${i}].content`);
  return content.map((p, j) => {
    if (p?.type === "text" && typeof p.text === "string") return p.text;
    if (p?.type === "refusal") return "";
    throw bad(`${TEXT_MSG} (messages[${i}].content[${j}] is ${JSON.stringify(p?.type ?? typeof p)})`, `messages[${i}].content`);
  }).join("");
}

// body (parsed JSON) -> internal request { api, stream, includeUsage, system, messages, maxTokens, temperature, topK, stop, thinking }
export function parseOpenAI(b) {
  if (!b || typeof b !== "object" || Array.isArray(b)) throw bad("the body must be a JSON object");
  if (!Array.isArray(b.messages)) throw bad("messages is required", "messages");
  if (b.tools?.length || b.functions?.length || (b.tool_choice != null && b.tool_choice !== "none") || b.function_call != null) throw bad(TOOLS_MSG, "tools");
  if (b.n != null && b.n !== 1) throw bad("n must be 1", "n");
  if (b.logprobs || b.top_logprobs != null) throw bad("logprobs are not supported", "logprobs");
  if (b.response_format != null && b.response_format?.type !== "text") throw bad("JSON mode is not supported yet", "response_format");
  if (b.audio != null || (Array.isArray(b.modalities) && b.modalities.some((m) => m !== "text"))) throw bad(TEXT_MSG, "modalities");
  for (const k of ["presence_penalty", "frequency_penalty"]) if (b[k] != null && b[k] !== 0) throw bad(`${k} is not supported (only 0)`, k);
  if (b.logit_bias != null && (typeof b.logit_bias !== "object" || Object.keys(b.logit_bias).length)) throw bad("logit_bias is not supported", "logit_bias");
  checkNum(b.top_p, 0, 1, "top_p");   // accepted, ignored: the sampler has no nucleus cut
  const system = [], msgs = [];
  b.messages.forEach((m, i) => {
    if (!m || typeof m !== "object") throw bad(`messages[${i}] must be an object`, `messages[${i}]`);
    if (m.role === "tool" || m.role === "function" || m.tool_calls?.length || m.function_call) throw bad(TOOLS_MSG, `messages[${i}]`);
    if (m.role === "system" || m.role === "developer") { system.push(textOf(m.content, i)); return; }
    if (m.role !== "user" && m.role !== "assistant") throw bad(`messages[${i}].role must be system, developer, user or assistant`, `messages[${i}].role`);
    msgs.push({ role: m.role, text: textOf(m.content, i) });
  });
  const sys = system.join("\n\n");
  const messages = finishMessages(msgs, sys);
  const mt = checkInt(b.max_completion_tokens ?? b.max_tokens, 1, LIMITS.maxTokens, b.max_completion_tokens != null ? "max_completion_tokens" : "max_tokens");
  const temperature = checkNum(b.temperature, 0, 2, "temperature");
  const tk = b.top_k == null ? null : checkInt(b.top_k, 1, 1e9, "top_k");
  const effort = b.reasoning_effort;
  return {
    api: "openai",
    stream: !!b.stream,
    includeUsage: !!b.stream_options?.include_usage,
    system: sys, messages,
    maxTokens: mt ?? LIMITS.defaultMaxTokens,
    temperature, topK: tk == null ? null : Math.min(tk, LIMITS.topK),
    stop: parseStop(b.stop, "stop"),
    thinking: typeof effort === "string" && effort !== "none" && effort !== "minimal",
  };
}

const FINISH = { stop: "stop", stop_seq: "stop", max: "length", ctx: "length" };   // abort never gets here (http.js: an error)
export const finishReason = (r) => FINISH[r] || "stop";
const usageOf = (u, reused) => ({ prompt_tokens: u.in, completion_tokens: u.out, total_tokens: u.in + u.out, ...(reused ? { prompt_tokens_details: { cached_tokens: reused } } : {}) });

// the whole answer, non-stream
export function openaiResponse({ id, created, model, text, think, reason, usage, reused }) {
  const message = { role: "assistant", content: text, refusal: null };
  if (think) message.reasoning_content = think;
  return { id: "chatcmpl-" + id, object: "chat.completion", created, model, system_fingerprint: null,
    choices: [{ index: 0, message, logprobs: null, finish_reason: finishReason(reason) }], usage: usageOf(usage, reused) };
}

// Stream chunks: one per room token (no batching), in the order OpenAI sends them.
export class OpenAIStream {
  constructor({ id, created, model, includeUsage }) {
    this.base = { id: "chatcmpl-" + id, object: "chat.completion.chunk", created, model, system_fingerprint: null };
    this.includeUsage = includeUsage;
  }
  chunk(delta, finish = null) { return `data: ${JSON.stringify({ ...this.base, choices: [{ index: 0, delta, logprobs: null, finish_reason: finish }] })}\n\n`; }
  start() { return this.chunk({ role: "assistant", content: "" }); }
  token(text, th) { return text ? this.chunk(th ? { reasoning_content: text } : { content: text }) : ""; }
  done({ reason, usage, reused }) {
    let s = this.chunk({}, finishReason(reason));
    if (this.includeUsage) s += `data: ${JSON.stringify({ ...this.base, choices: [], usage: usageOf(usage, reused) })}\n\n`;
    return s + "data: [DONE]\n\n";
  }
  // after the headers went out: an error object, then the connection closes without [DONE]
  error(e) { return `data: ${JSON.stringify(openaiError(e).body)}\n\n`; }
  keepAlive(ahead) { return `: queued, ${ahead} ahead\n\n`; }
}

export function openaiModels(model, created) {
  return { object: "list", data: model ? [{ id: model, object: "model", created, owned_by: "pooled" }] : [] };
}
