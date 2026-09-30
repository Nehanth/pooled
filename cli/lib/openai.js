// The OpenAI Chat Completions API (docs/design/serve.md 4 and 12): request validation and mapping
// onto the internal request (tools, tool_choice, tool results, reasoning in history, response_format,
// reasoning_effort), the response body, stream chunks (tool_calls deltas included) and errors.
import { ApiError, bad, TEXT_MSG, LIMITS, EFFORTS, parseStop, checkInt, checkNum, withDefaults, ids, normTool, parseArgs, toolText, outcome } from "./common.js";

const STATUS = { bad: 400, ctx: 400, auth: 401, forbidden: 403, notfound: 404, method: 405, toolarge: 413, busy: 429, unavailable: 503, timeout: 504, server: 500 };
const TYPE = { bad: "invalid_request_error", ctx: "invalid_request_error", auth: "invalid_request_error", forbidden: "permission_error", notfound: "invalid_request_error", method: "invalid_request_error", toolarge: "invalid_request_error", busy: "rate_limit_exceeded", unavailable: "server_error", timeout: "server_error", server: "server_error" };
const CODE = { ctx: "context_length_exceeded", auth: "invalid_api_key", notfound: "not_found", method: "method_not_allowed", toolarge: "request_too_large", busy: "rate_limit_exceeded", unavailable: "service_unavailable", timeout: "timeout", server: "server_error" };

export function openaiError(e) {
  const kind = e instanceof ApiError ? e.kind : "server";
  return { status: STATUS[kind] || 500, body: { error: { message: e.message, type: TYPE[kind] || "server_error", param: e.param ?? null, code: CODE[kind] ?? null } } };
}

const isObj = (v) => !!v && typeof v === "object" && !Array.isArray(v);
const NAMED = 'Correct usage: `{"type": "function", "function": {"name": "my_function"}}`';

// user / system text: a string, or a list of parts of which only text is accepted (an assistant's
// refusal parts are dropped)
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

// tools: [{ type: "function", function: { name, description?, parameters?, strict? } }] -> Tool[] | null.
// strict is accepted and needs nothing: every call is grammar-constrained to its schema anyway.
function parseTools(tools) {
  if (tools == null) return null;
  if (!Array.isArray(tools)) throw bad("tools must be a list", "tools");
  return tools.map((t, i) => {
    const p = `tools[${i}]`;
    if (!isObj(t)) throw bad(`${p} must be an object`, p);
    if (t.type === "custom") throw bad(`${p}: custom tools are not supported; use function tools`, `${p}.type`);
    if (t.type != null && t.type !== "function") throw bad(`${p}: only function tools are supported (got type ${JSON.stringify(t.type)})`, `${p}.type`);
    if (!isObj(t.function)) throw bad(`${p}.function is required`, `${p}.function`);
    return normTool(t.function, `${p}.function`);
  });
}

// tool_choice -> { toolChoice, allowed }. Names are checked against tools here (param-precise
// messages as vLLM's); common.finishRequest checks the normalized result again.
function parseToolChoice(tc, tools) {
  const names = new Set((tools || []).map((t) => t.name));
  if (tc == null || tc === "auto" || tc === "none") return { toolChoice: tc ?? "auto", allowed: null };
  if (!tools?.length) throw bad("When using `tool_choice`, `tools` must be set.", "tool_choice");
  if (tc === "required") return { toolChoice: "required", allowed: null };
  if (typeof tc === "string") throw bad(`Invalid value for \`tool_choice\`: ${tc}! Only named tools, "none", "auto" or "required" are supported.`, "tool_choice");
  if (!isObj(tc)) throw bad("tool_choice must be a string or an object", "tool_choice");
  if (tc.type === "function") {
    const name = isObj(tc.function) ? tc.function.name : tc.name;   // the flat Responses shape too
    if (typeof name !== "string" || !name) throw bad(`Expected field \`name\` in \`function\` in \`tool_choice\`! ${NAMED}`, "tool_choice.function.name");
    if (!names.has(name)) throw bad("The tool specified in `tool_choice` does not match any of the specified `tools`", "tool_choice");
    return { toolChoice: { name }, allowed: null };
  }
  if (tc.type === "allowed_tools") {
    const at = isObj(tc.allowed_tools) ? tc.allowed_tools : tc;
    const mode = at.mode ?? "auto";
    if (mode !== "auto" && mode !== "required") throw bad('tool_choice.allowed_tools.mode must be "auto" or "required"', "tool_choice.allowed_tools.mode");
    if (!Array.isArray(at.tools) || !at.tools.length) throw bad("tool_choice.allowed_tools.tools must be a non-empty list", "tool_choice.allowed_tools.tools");
    const allowed = at.tools.map((t, i) => {
      const name = isObj(t?.function) ? t.function.name : t?.name;
      if (t?.type != null && t.type !== "function") throw bad(`tool_choice.allowed_tools.tools[${i}]: only function tools are supported`, "tool_choice.allowed_tools.tools");
      if (typeof name !== "string" || !names.has(name)) throw bad(`tool_choice.allowed_tools.tools[${i}]: ${JSON.stringify(String(name ?? "")).slice(0, 60)} is not in tools`, "tool_choice.allowed_tools.tools");
      return name;
    });
    return { toolChoice: mode, allowed: [...new Set(allowed)] };
  }
  if (tc.type === "custom") throw bad("tool_choice: custom tools are not supported", "tool_choice");
  throw bad(`Invalid value for \`tool_choice\`! Only named tools, "none", "auto", "required" or allowed_tools are supported.`, "tool_choice");
}

// response_format -> null | { type: "json" } | { type: "schema", schema, name? }
function parseFormat(rf) {
  if (rf == null) return null;
  if (!isObj(rf)) throw bad("response_format must be an object", "response_format");
  if (rf.type === "text") return null;
  if (rf.type === "json_object") return { type: "json" };
  if (rf.type === "json_schema") {
    const js = rf.json_schema;
    if (!isObj(js)) throw bad("response_format.json_schema is required", "response_format.json_schema");
    if (!isObj(js.schema)) throw bad("response_format.json_schema.schema is required (a JSON schema object)", "response_format.json_schema.schema");
    return { type: "schema", schema: js.schema, ...(typeof js.name === "string" && js.name ? { name: js.name } : {}) };
  }
  throw bad(`response_format.type must be text, json_object or json_schema (got ${JSON.stringify(rf.type)})`, "response_format.type");
}

// reasoning_effort: none / minimal -> thinking off; low … max -> on at that effort; absent -> off,
// unless chat_template_kwargs.enable_thinking (the vLLM / SGLang extension) says otherwise
function parseEffort(b) {
  const e = b.reasoning_effort;
  if (e != null) {
    if (e === "none" || e === "minimal") return { thinking: false, effort: null };
    if (typeof e !== "string" || !EFFORTS.includes(e)) throw bad(`reasoning_effort must be one of none, minimal, ${EFFORTS.join(", ")}`, "reasoning_effort");
    return { thinking: true, effort: e };
  }
  const k = b.chat_template_kwargs?.enable_thinking ?? b.chat_template_kwargs?.thinking;
  return { thinking: k === true, effort: null };
}

// one message -> Msg (or null for an empty system message)
function parseMessage(m, i, warn) {
  const p = `messages[${i}]`;
  if (!isObj(m)) throw bad(`${p} must be an object`, p);
  switch (m.role) {
    case "system": case "developer":
      return { role: "system", text: textOf(m.content, i) };
    case "user":
      return { role: "user", text: textOf(m.content, i) };
    case "assistant": {
      if (m.function_call != null) throw bad(`${p}.function_call is deprecated and not supported; use tool_calls`, `${p}.function_call`);
      const out = { role: "assistant", text: textOf(m.content, i) };
      const r = typeof m.reasoning_content === "string" ? m.reasoning_content : typeof m.reasoning === "string" ? m.reasoning : "";
      if (r) out.reasoning = r;
      if (m.tool_calls != null) {
        if (!Array.isArray(m.tool_calls)) throw bad(`${p}.tool_calls must be a list`, `${p}.tool_calls`);
        if (m.tool_calls.length > LIMITS.calls) throw bad(`${p}: at most ${LIMITS.calls} tool calls in one message`, `${p}.tool_calls`);
        const calls = m.tool_calls.map((c, j) => {
          const q = `${p}.tool_calls[${j}]`;
          if (!isObj(c)) throw bad(`${q} must be an object`, q);
          if (c.type === "custom") throw bad(`${q}: custom tool calls are not supported`, `${q}.type`);
          if (c.type != null && c.type !== "function") throw bad(`${q}.type must be "function"`, `${q}.type`);
          const f = c.function;
          if (!isObj(f) || typeof f.name !== "string" || !f.name || f.name.length > 128) throw bad(`${q}.function.name is required (1 to 128 characters)`, `${q}.function.name`);
          if (c.id != null && typeof c.id !== "string") throw bad(`${q}.id must be a string`, `${q}.id`);
          return { id: c.id ?? "", name: f.name, args: parseArgs(f.arguments, warn) };
        });
        if (calls.length) out.calls = calls;
      }
      return out;
    }
    case "tool": {
      if (m.tool_call_id != null && typeof m.tool_call_id !== "string") throw bad(`${p}.tool_call_id must be a string`, `${p}.tool_call_id`);
      return { role: "tool", text: toolText(m.content, `${p}.content`), ...(m.tool_call_id != null ? { id: m.tool_call_id } : {}) };
    }
    case "function":
      throw bad(`${p}: the function role is deprecated and not supported; use role "tool" with tool_call_id`, `${p}.role`);
    default:
      throw bad(`${p}.role must be system, developer, user, assistant or tool`, `${p}.role`);
  }
}

// body (parsed JSON) -> the internal request (docs/design/serve.md 4.1; normalized later by
// common.finishRequest). Warnings (arguments that are not a JSON object) go to req.extra.warnings.
export function parseOpenAI(b) {
  if (!isObj(b)) throw bad("the body must be a JSON object");
  if (!Array.isArray(b.messages)) throw bad("messages is required", "messages");
  // still unsupported (docs/design/serve.md D14)
  if (b.n != null && b.n !== 1) throw bad("n must be 1", "n");
  if (b.logprobs || b.top_logprobs != null) throw bad("logprobs are not supported", "logprobs");
  if (b.audio != null || (Array.isArray(b.modalities) && b.modalities.some((m) => m !== "text"))) throw bad(TEXT_MSG, "modalities");
  if (b.prediction != null) throw bad("predicted outputs (prediction) are not supported", "prediction");
  if (b.web_search_options != null) throw bad("web search (web_search_options) is not supported", "web_search_options");
  if (b.functions?.length || (b.function_call != null && b.function_call !== "none" && b.function_call !== "auto")) throw bad("functions and function_call are deprecated and not supported; use tools and tool_choice", b.functions?.length ? "functions" : "function_call");
  for (const k of ["presence_penalty", "frequency_penalty"]) if (b[k] != null && b[k] !== 0) throw bad(`${k} is not supported (only 0)`, k);
  if (b.logit_bias != null && (typeof b.logit_bias !== "object" || Object.keys(b.logit_bias).length)) throw bad("logit_bias is not supported", "logit_bias");
  checkNum(b.top_p, 0, 1, "top_p");   // accepted, ignored: the sampler has no nucleus cut
  // tools (an empty list, as some clients send, means none)
  const tools = parseTools(b.tools);
  const { toolChoice, allowed } = parseToolChoice(b.tool_choice, tools);
  if (b.parallel_tool_calls != null && typeof b.parallel_tool_calls !== "boolean") throw bad("parallel_tool_calls must be a boolean", "parallel_tool_calls");
  // messages
  const warnings = [];
  const warn = (w) => { if (!warnings.includes(w)) warnings.push(w); };
  // an empty system message adds nothing (and folded mid-conversation it would add a blank line)
  const messages = b.messages.map((m, i) => parseMessage(m, i, warn)).filter((m) => m.role !== "system" || m.text);
  const mt = checkInt(b.max_completion_tokens ?? b.max_tokens, 1, LIMITS.maxTokens, b.max_completion_tokens != null ? "max_completion_tokens" : "max_tokens");
  const temperature = checkNum(b.temperature, 0, 2, "temperature");
  const tk = b.top_k == null ? null : checkInt(b.top_k, 1, 1e9, "top_k");
  const { thinking, effort } = parseEffort(b);
  return withDefaults({
    api: "openai",
    stream: !!b.stream,
    includeUsage: !!(b.stream && b.stream_options?.include_usage),
    system: "", messages,
    tools: tools?.length ? tools : null, toolChoice, allowed,
    parallel: b.parallel_tool_calls !== false,
    format: parseFormat(b.response_format),
    thinking, effort,
    maxTokens: mt ?? LIMITS.defaultMaxTokens,
    temperature, topK: tk == null ? null : Math.min(tk, LIMITS.topK),
    stop: parseStop(b.stop, "stop"),
    extra: warnings.length ? { warnings } : {},
  });
}

// common.outcome -> finish_reason. A named tool_choice ends with "stop" (as OpenAI and vLLM).
const FINISH = { tool: "tool_calls", stop: "stop", stop_seq: "stop", length: "length", ctx: "length" };
export const finishReason = (o) => FINISH[o] || "stop";

export function usageOf(a, req = {}) {
  const u = a.usage, reused = a.reused || 0;
  return { prompt_tokens: u.in, completion_tokens: u.out, total_tokens: u.in + u.out,
    ...(reused ? { prompt_tokens_details: { cached_tokens: reused } } : {}),
    ...(u.think || req.thinking ? { completion_tokens_details: { reasoning_tokens: u.think || 0 } } : {}) };
}

// the complete calls as tool_calls (an open call, cut by max_tokens, is left out)
const toolCalls = (calls) => calls.map((c) => ({ id: c.id, type: "function", function: { name: c.name, arguments: c.args } }));

// the whole answer, non-stream. a: answer.js Answer
export function openaiResponse(a, req = {}) {
  const calls = a.calls || [];
  const message = { role: "assistant", content: calls.length && !a.text ? null : a.text, refusal: null };
  if (a.think) message.reasoning_content = a.think;
  if (calls.length) message.tool_calls = toolCalls(calls);
  return { id: "chatcmpl-" + a.id, object: "chat.completion", created: a.created, model: a.model, system_fingerprint: null,
    choices: [{ index: 0, message, logprobs: null, finish_reason: finishReason(outcome(a, req)) }], usage: usageOf(a, req) };
}

// Stream chunks: one per room message (no batching), in the order OpenAI sends them. A call opens
// with its index, id, type and name (arguments ""), then argument fragments.
export class OpenAIStream {
  constructor({ id, created, model, includeUsage }) {
    this.base = { id: "chatcmpl-" + id, object: "chat.completion.chunk", created, model, system_fingerprint: null };
    this.includeUsage = includeUsage;
  }
  chunk(delta, finish = null) { return `data: ${JSON.stringify({ ...this.base, choices: [{ index: 0, delta, logprobs: null, finish_reason: finish }] })}\n\n`; }
  start() { return this.chunk({ role: "assistant", content: "" }); }
  token(text, th) { return text ? this.chunk(th ? { reasoning_content: text } : { content: text }) : ""; }
  callStart(i, id, name) { return this.chunk({ tool_calls: [{ index: i, id, type: "function", function: { name, arguments: "" } }] }); }
  callArgs(i, frag) { return frag ? this.chunk({ tool_calls: [{ index: i, function: { arguments: frag } }] }) : ""; }
  done(a, req = {}) {
    let s = this.chunk({}, finishReason(outcome(a, req)));
    if (this.includeUsage) s += `data: ${JSON.stringify({ ...this.base, choices: [], usage: usageOf(a, req) })}\n\n`;
    return s + "data: [DONE]\n\n";
  }
  // after the headers went out: an error object, then the connection closes without [DONE]
  error(e) { return `data: ${JSON.stringify(openaiError(e).body)}\n\n`; }
  keepAlive(ahead) { return ahead == null ? ": keep-alive\n\n" : `: queued, ${ahead} ahead\n\n`; }
}

export function openaiModels(model, created) {
  return { object: "list", data: model ? [{ id: model, object: "model", created, owned_by: "pooled" }] : [] };
}

// ---- the adapter (docs/design/serve.md 11): an Encoder writing the stream above ----
class OpenAIEncoder {
  constructor(req, sse, meta) { this.req = req; this.sse = sse; this.s = new OpenAIStream({ id: meta.id, created: meta.created, model: meta.model, includeUsage: req.includeUsage }); }
  start() { this.sse.write(this.s.start()); }
  think(t) { this.sse.write(this.s.token(t, true)); }
  text(t) { this.sse.write(this.s.token(t, false)); }
  callStart(i, id, name) { this.sse.write(this.s.callStart(i, id, name)); }
  callArgs(i, frag) { this.sse.write(this.s.callArgs(i, frag)); }
  callEnd() {}
  done(a) { this.sse.write(this.s.done(a, this.req)); }
  error(e) { this.sse.write(this.s.error(e)); }
  keepAlive(ahead) { this.sse.write(this.s.keepAlive(ahead)); }
}
export const adapter = {
  api: "openai",
  label: "chat",
  routes: [{ method: "POST", path: "/v1/chat/completions" }],
  parse: (body) => parseOpenAI(body),
  idFor: () => () => ids.call(),
  encoder: (req, sse, meta) => new OpenAIEncoder(req, sse, meta),
  final: (a, req) => openaiResponse(a, req),
  error: openaiError,
  streamError: (e) => `data: ${JSON.stringify(openaiError(e).body)}\n\n`,
};
