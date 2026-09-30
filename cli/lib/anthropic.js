// The Anthropic Messages API: request validation and mapping onto the internal request, response
// bodies, stream events and errors (docs/design/serve.md section 4, "Anthropic"): tools, tool_use /
// tool_result blocks, tool_choice, thinking (with the reasoning carried in the signature), system
// as a string or blocks, and the usage split Claude Code counts its context with.
import { ApiError, bad, TEXT_MSG, LIMITS, EFFORTS, parseStop, checkInt, checkNum, capTokens, withDefaults, ids, blob, normTool, parseArgs, toolText, nonTextPart, outcome, warnOnce, cleanText } from "./common.js";

const STATUS = { bad: 400, ctx: 400, auth: 401, forbidden: 403, notfound: 404, method: 405, toolarge: 413, busy: 529, unavailable: 529, timeout: 504, server: 500 };
const TYPE = { bad: "invalid_request_error", ctx: "invalid_request_error", auth: "authentication_error", forbidden: "permission_error", notfound: "not_found_error", method: "invalid_request_error", toolarge: "request_too_large", busy: "overloaded_error", unavailable: "overloaded_error", timeout: "timeout_error", server: "api_error" };

export function anthropicError(e) {
  const kind = e instanceof ApiError ? e.kind : "server";
  return { status: STATUS[kind] || 500, body: { type: "error", error: { type: TYPE[kind] || "api_error", message: e.message } } };
}

// Anthropic's own tools (run on Anthropic's servers, or client tools with a fixed schema the model
// was trained on): not ours to run or render. Skipped with a warning, never a 400 (a client that
// always sends one must still work). Their calls and results in the history are dropped the same way.
const HOSTED_TOOL = /^(web_search|web_fetch|code_execution|computer|bash|text_editor|memory|advisor|tool_search_tool)(_|$)/;
const hostedBlock = (t) => t === "server_tool_use" || t === "mcp_tool_use" || t === "mcp_tool_result" || t === "container_upload" || (t !== "tool_result" && /_tool_result$/.test(t));
// the per-request header Claude Code puts first in the system blocks: it changes every request, so
// keeping it would make every prompt differ from its first token and defeat the room's caches
const BILLING = "x-anthropic-billing-header:";
// warnings once per process and kind: common.warnOnce (bounded, and the client's text cleaned)

const typeOf = (p) => JSON.stringify(p?.type ?? typeof p);

function systemText(s) {
  if (s == null) return "";
  if (typeof s === "string") return s;
  if (!Array.isArray(s)) throw bad("system: must be a string or a list of text blocks");
  return s.map((p, j) => {
    if (p?.type !== "text" || typeof p.text !== "string") throw bad(`system.${j}: only text blocks are supported (got ${typeOf(p)})`);
    return p.text.startsWith(BILLING) ? "" : p.text;
  }).join("");
}

// a system message's text (images and documents become a note: this model reads text only)
function plainText(content, where, log) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) throw bad(`${where}: must be a string or a list of content blocks`);
  return content.map((p, j) => {
    if (p?.type === "text" && typeof p.text === "string") return p.text;
    const note = nonTextPart(p?.type, log, "messages");
    if (note != null) return note;
    throw bad(`${TEXT_MSG} (${where}.${j} is ${typeOf(p)})`);
  }).join("");
}

// one assistant message -> { role: "assistant", text, calls?, reasoning? }
function assistantMsg(content, where, log) {
  if (typeof content === "string") return { role: "assistant", text: content };
  if (!Array.isArray(content)) throw bad(`${where}: must be a string or a list of content blocks`);
  let text = "";
  const calls = [], reasoning = [];
  content.forEach((p, j) => {
    const t = p?.type;
    if (t === "text" && typeof p.text === "string") text += p.text;
    else if (t === "thinking") {
      // our own signature carries the reasoning exactly (and is all there is when the display was
      // omitted); a foreign one (a conversation begun elsewhere) leaves the visible text
      const r = blob.decode(p.signature) ?? (typeof p.thinking === "string" ? p.thinking : "");
      if (r) reasoning.push(r);
    } else if (t === "redacted_thinking") { /* nothing readable */ }
    else if (t === "tool_use") {
      if (typeof p.name !== "string" || !p.name) throw bad(`${where}.${j}.name: Field required`);
      if (typeof p.id !== "string" || !p.id) throw bad(`${where}.${j}.id: Field required`);
      calls.push({ id: p.id, name: p.name, args: parseArgs(p.input, log) });
    } else if (hostedBlock(t)) warnOnce(log, "block " + t, `messages: ${cleanText(t, 60)} blocks (Anthropic's server tools) are left out of the conversation`);
    else throw bad(`${where}.${j}: unsupported content block ${typeOf(p)} in an assistant message`);
  });
  const m = { role: "assistant", text };
  if (calls.length) m.calls = calls;
  if (reasoning.length) m.reasoning = reasoning.join("\n\n");
  return m;
}

// one user message -> its tool results ({ role: "tool" } each, in order), then its text: a user
// turn, or, sent along with tool results, an aside (its own turn, not a new question)
function userMsgs(content, where, log) {
  if (typeof content === "string") return [{ role: "user", text: content }];
  if (!Array.isArray(content)) throw bad(`${where}: must be a string or a list of content blocks`);
  const tools = [];
  let text = "", hasText = false;
  content.forEach((p, j) => {
    const t = p?.type;
    if (t === "text" && typeof p.text === "string") { text += p.text; hasText = true; }
    else if (t === "tool_result") {
      if (typeof p.tool_use_id !== "string" || !p.tool_use_id) throw bad(`${where}.${j}.tool_use_id: Field required`);
      // is_error: the result is the error's text, as the model reads it either way
      tools.push({ role: "tool", id: p.tool_use_id, text: toolText(p.content, `${where}.${j}.content`) });
    } else if (t === "image" || t === "document") { text += nonTextPart(t, log, "messages"); hasText = true; }   // a note: a pasted screenshot must not break every later request
    else if (t === "search_result" || t === "container_upload") throw bad(`${TEXT_MSG} (${where}.${j} is ${typeOf(p)})`);
    else if (hostedBlock(t) || t === "server_tool_use") warnOnce(log, "block " + t, `messages: ${cleanText(t, 60)} blocks (Anthropic's server tools) are left out of the conversation`);
    else throw bad(`${where}.${j}: unsupported content block ${typeOf(p)} in a user message`);
  });
  if (!tools.length) return [{ role: "user", text }];
  return hasText && text ? [...tools, { role: "user", text, aside: true }] : tools;
}

function parseTools(list, log) {
  if (list == null) return null;
  if (!Array.isArray(list)) throw bad("tools: must be a list");
  const out = [];
  list.forEach((t, j) => {
    if (!t || typeof t !== "object") throw bad(`tools.${j}: must be an object`);
    if (t.type != null && t.type !== "custom") {
      if (typeof t.type === "string" && (HOSTED_TOOL.test(t.type) || /_\d{8}$/.test(t.type))) {
        warnOnce(log, "tool " + t.type, `tools: ${cleanText(t.type, 60)} is one of Anthropic's own tools; the room's model does not get it`);
        return;
      }
      throw bad(`tools.${j}.type: unsupported tool type ${JSON.stringify(t.type)}`);
    }
    if (t.input_schema == null) throw bad(`tools.${j}.input_schema: Field required`);
    out.push(normTool({ name: t.name, description: t.description, parameters: t.input_schema }, `tools.${j}`));
  });
  return out.length ? out : null;
}

// -> { toolChoice, parallel }
function parseChoice(c) {
  if (c == null) return { toolChoice: "auto", parallel: true };
  if (typeof c !== "object") throw bad("tool_choice: must be an object");
  const parallel = !c.disable_parallel_tool_use;
  switch (c.type) {
    case "auto": return { toolChoice: "auto", parallel };
    case "any": return { toolChoice: "required", parallel };
    case "none": return { toolChoice: "none", parallel };
    case "tool":
      if (typeof c.name !== "string" || !c.name) throw bad("tool_choice.name: Field required");
      return { toolChoice: { name: c.name }, parallel };
  }
  throw bad("tool_choice.type: must be auto, any, tool or none");
}

function parseFormat(f, param) {
  if (f == null) return null;
  if (f?.type !== "json_schema") throw bad(`${param}.type: only json_schema is supported`);
  if (!f.schema || typeof f.schema !== "object" || Array.isArray(f.schema)) throw bad(`${param}.schema: a JSON schema object is required`);
  return { type: "schema", schema: f.schema };
}

// body (parsed JSON) -> the internal request (docs/design/serve.md 4.1)
export function parseAnthropic(b, { log = () => {} } = {}) {
  if (!b || typeof b !== "object" || Array.isArray(b)) throw bad("the body must be a JSON object");
  if (typeof b.model !== "string" || !b.model) throw bad("model: Field required");
  if (b.max_tokens == null) throw bad("max_tokens: Field required");
  if (!Array.isArray(b.messages)) throw bad("messages: Field required");
  if (Array.isArray(b.mcp_servers) && b.mcp_servers.length) throw bad("mcp_servers: MCP connectors are not supported by pooled serve (connect MCP servers in the client instead)");
  const system = systemText(b.system);
  const messages = [];
  b.messages.forEach((m, i) => {
    if (!m || typeof m !== "object") throw bad(`messages.${i}: must be an object`);
    const where = `messages.${i}.content`;
    if (m.role === "user") messages.push(...userMsgs(m.content, where, log));
    else if (m.role === "assistant") messages.push(assistantMsg(m.content, where, log));
    else if (m.role === "system") messages.push({ role: "system", text: plainText(m.content, where, log) });   // mid-conversation (Claude Code): folded by normalizeMessages
    else throw bad(`messages.${i}.role: must be user or assistant`);
  });
  // max_tokens past what the room ever writes is capped, not refused: agents send their model's
  // output limit (32000, 64000, 128000) whatever model sits behind the base URL
  const maxTokens = capTokens(b.max_tokens, "max_tokens");
  const temperature = checkNum(b.temperature, 0, 1, "temperature");
  checkNum(b.top_p, 0, 1, "top_p");   // accepted, ignored
  const tk = b.top_k == null ? null : checkInt(b.top_k, 1, 1e9, "top_k");
  const t = b.thinking;
  if (t != null && !(t?.type === "enabled" || t?.type === "disabled" || t?.type === "adaptive")) throw bad("thinking.type: must be enabled, adaptive or disabled");
  if (t?.display != null && t.display !== "summarized" && t.display !== "omitted") throw bad("thinking.display: must be summarized or omitted");
  // budget_tokens: the reasoning stops there and the rest of max_tokens goes to the answer
  const budget = t?.type === "enabled" ? checkInt(t.budget_tokens, 1, Number.MAX_SAFE_INTEGER, "thinking.budget_tokens") : null;
  const oc = b.output_config;
  if (oc != null && (typeof oc !== "object" || Array.isArray(oc))) throw bad("output_config: must be an object");
  const effort = oc?.effort ?? null;
  if (effort != null && !EFFORTS.includes(effort)) throw bad(`output_config.effort: must be one of ${EFFORTS.join(", ")}`);
  const format = parseFormat(oc?.format, "output_config.format") || parseFormat(b.output_format, "output_format");
  const tools = parseTools(b.tools, log);
  const { toolChoice, parallel } = parseChoice(b.tool_choice);
  return withDefaults({
    api: "anthropic",
    stream: !!b.stream,
    system, messages, tools, toolChoice, parallel, format,
    maxTokens,
    temperature, topK: tk == null ? null : Math.min(tk, LIMITS.topK),
    stop: parseStop(b.stop_sequences, "stop_sequences"),
    thinking: t?.type === "enabled" || t?.type === "adaptive",
    thinkBudget: budget != null && budget < maxTokens ? budget : null,
    showThinking: t?.display !== "omitted",
    effort,
  });
}

// abort never gets here (http.js turns the host's Stop into an error); a full context window is the
// Messages API's model_context_window_exceeded, not max_tokens. Calls that ended the answer are
// tool_use, whatever the tool_choice (the client runs them either way).
const STOP = { stop: "end_turn", stop_seq: "stop_sequence", length: "max_tokens", ctx: "model_context_window_exceeded" };
export function stopReason(a, req) {
  const o = outcome(a, req);
  if (o === "tool" || (o === "stop" && a.calls?.length)) return "tool_use";
  return STOP[o] || "end_turn";
}
// input_tokens leaves out the tokens read from the cache, as in the Messages API (the OpenAI side counts
// them in prompt_tokens, with cached_tokens as a detail); Claude Code adds the three up for its context
export const usageOf = (u, reused = 0) => ({ input_tokens: Math.max(0, u.in - reused), cache_creation_input_tokens: 0, cache_read_input_tokens: reused, output_tokens: u.out });

const inputOf = (args) => { try { const v = JSON.parse(args); return v && typeof v === "object" && !Array.isArray(v) ? v : {}; } catch { return {}; } };

// the whole answer, non-stream: [thinking] [text] [tool_use…]. The thinking block's signature
// carries the reasoning ("pooled1." + base64url), so a client that sends the block back (Claude Code
// does) gives the room its reasoning back even when the display was omitted.
export function anthropicResponse(a, req) {
  const content = [];
  if (req.thinking || a.think) content.push({ type: "thinking", thinking: req.showThinking === false ? "" : a.think, signature: blob.encode(a.think) });
  if (a.text || !a.calls.length) content.push({ type: "text", text: a.text });
  for (const c of a.calls) content.push({ type: "tool_use", id: c.id, name: c.name, input: inputOf(c.args) });
  return { id: "msg_" + a.id, type: "message", role: "assistant", model: a.model, content,
    stop_reason: stopReason(a, req), stop_sequence: a.reason === "stop_seq" ? a.stopSeq ?? null : null, usage: usageOf(a.usage, a.reused) };
}

const ev = (type, obj) => `event: ${type}\ndata: ${JSON.stringify({ type, ...obj })}\n\n`;

// Stream events in the order the Messages API sends them: message_start, then each block in the
// order it is written (content_block_start, deltas, content_block_stop, one block at a time, index
// counting up), message_delta with the stop reason and final usage, message_stop.
//   thinking: opened at the start when thinking is on; thinking_delta (unless omitted), then a
//             signature_delta with the reasoning blob before its stop
//   text:     opened on the first text
//   tool_use: { id, name, input: {} }, then input_json_delta fragments that join into the arguments
export class AnthropicStream {
  constructor({ id, model, thinking = false, showThinking = true }) {
    this.id = "msg_" + id; this.model = model; this.thinking = thinking; this.show = showThinking;
    this.block = -1; this.kind = null; this.call = -1;
    this.think = ""; this.thought = false; this.answered = false;
  }
  open(kind, content_block) {
    this.block++; this.kind = kind;
    if (kind === "thinking") this.thought = true; else this.answered = true;
    return ev("content_block_start", { index: this.block, content_block });
  }
  close() {
    if (this.kind == null) return "";
    let s = "";
    if (this.kind === "thinking") s += ev("content_block_delta", { index: this.block, delta: { type: "signature_delta", signature: blob.encode(this.think) } });
    s += ev("content_block_stop", { index: this.block });
    this.kind = null; this.call = -1;
    return s;
  }
  start(inputTokens) {
    // the cache split is known only at the end (message_delta): here everything counts as input
    return ev("message_start", { message: { id: this.id, type: "message", role: "assistant", model: this.model, content: [], stop_reason: null, stop_sequence: null,
      usage: { input_tokens: inputTokens, cache_creation_input_tokens: 0, cache_read_input_tokens: 0, output_tokens: 0 } } })
      + (this.thinking ? this.open("thinking", { type: "thinking", thinking: "", signature: "" }) : "") + ev("ping", {});
  }
  thinkText(text) {
    if (!text) return "";
    let s = "";
    if (this.kind !== "thinking") {
      if (this.thought || this.answered) return "";   // reasoning after the answer started: nothing sane to do with it
      s += this.open("thinking", { type: "thinking", thinking: "", signature: "" });
    }
    this.think += text;
    return this.show ? s + ev("content_block_delta", { index: this.block, delta: { type: "thinking_delta", thinking: text } }) : s;
  }
  text(text) {
    if (!text) return "";
    let s = "";
    if (this.kind !== "text") s += this.close() + this.open("text", { type: "text", text: "" });
    return s + ev("content_block_delta", { index: this.block, delta: { type: "text_delta", text } });
  }
  token(text, th) { return th ? this.thinkText(text) : this.text(text); }
  callStart(i, id, name) {
    const s = this.close() + this.open("tool_use", { type: "tool_use", id, name, input: {} });
    this.call = i;
    return s;
  }
  callArgs(i, frag) {
    if (!frag || this.kind !== "tool_use" || this.call !== i) return "";
    return ev("content_block_delta", { index: this.block, delta: { type: "input_json_delta", partial_json: frag } });
  }
  callEnd(i) { return this.kind === "tool_use" && this.call === i ? this.close() : ""; }
  // an answer with no text and no calls still has its (empty) text block, as the Messages API's
  done(a, req) {
    let s = this.close();   // a call cut off here is closed as it is (stop_reason max_tokens)
    if (!this.answered) s += this.open("text", { type: "text", text: "" }) + this.close();
    s += ev("message_delta", { delta: { stop_reason: stopReason(a, req), stop_sequence: a.reason === "stop_seq" ? a.stopSeq ?? null : null }, usage: usageOf(a.usage, a.reused) });
    return s + ev("message_stop", {});
  }
  error(e) { return ev("error", { error: anthropicError(e).body.error }); }
  keepAlive() { return ev("ping", {}); }
}

// GET /v1/models in Anthropic's shape (picked by the anthropic-version or x-api-key header)
export function anthropicModels(model, label, createdMs) {
  const data = model ? [{ type: "model", id: model, display_name: label, created_at: new Date(createdMs).toISOString() }] : [];
  return { data, has_more: false, first_id: data[0]?.id ?? null, last_id: data[0]?.id ?? null };
}

// ---- the adapter (docs/design/serve.md 11)
class AnthropicEncoder {
  constructor(req, sse, meta) {
    this.req = req; this.sse = sse;
    this.s = new AnthropicStream({ id: meta.id, model: meta.model, thinking: req.thinking, showThinking: req.showThinking });
  }
  start(promptTokens) { this.sse.write(this.s.start(promptTokens)); }
  think(t) { this.sse.write(this.s.thinkText(t)); }
  text(t) { this.sse.write(this.s.text(t)); }
  callStart(i, id, name) { this.sse.write(this.s.callStart(i, id, name)); }
  callArgs(i, frag) { this.sse.write(this.s.callArgs(i, frag)); }
  callEnd(i) { this.sse.write(this.s.callEnd(i)); }
  done(a) { this.sse.write(this.s.done(a, this.req)); }
  error(e) { this.sse.write(this.s.error(e)); }
  keepAlive() { this.sse.write(this.s.keepAlive()); }
}
export const adapter = {
  api: "anthropic",
  label: "messages",
  routes: [{ method: "POST", path: "/v1/messages" }],
  parse: (body, headers, ctx) => parseAnthropic(body, ctx),
  idFor: () => () => ids.toolu(),
  encoder: (req, sse, meta) => new AnthropicEncoder(req, sse, meta),
  final: anthropicResponse,
  error: anthropicError,
  streamError: (e) => ev("error", { error: anthropicError(e).body.error }),
};
