// Shared by the API adapters (openai.js, anthropic.js, responses.js): the internal request they all
// turn into, its normalization and checks, the error kinds and their statuses, limits, ids, and the
// client label the room shows (docs/design/serve.md sections 4 and 9).
import { randomBytes } from "node:crypto";

// Limits (docs/design/serve.md section 9). The host checks the same ones again.
//   chars / messages: what the v1-shaped parsers (finishMessages) still enforce; totalChars /
//   totalMessages: finishRequest's, for everything (the endpoints move to these)
export const LIMITS = {
  body: 4 << 20,          // bytes of the HTTP body
  askBytes: 3670016,      // bytes of the serialized ask (3.5 MB: PeerJS drops a message it cannot rebuild, ~4 MB)
  chars: 400000,          // total text (v1 parsers)
  totalChars: 1500000,    // total text
  messages: 200,          // (v1 parsers)
  totalMessages: 1000,
  tools: 128, schemaChars: 32000, calls: 64, maxCalls: 128,
  toolName: /^[A-Za-z0-9_.:-]{1,128}$/,
  stops: 4, stopLen: 64,
  defaultMaxTokens: 1024,
  maxTokens: 65536,
  topK: 64,               // engine/topk.js TOPK_MAX
  client: 40,
};

// An error with the kind that picks its status and body in each API's shape.
//   bad 400 · ctx 400 (context_length_exceeded) · auth 401 · forbidden 403 · notfound 404
//   method 405 · toolarge 413 · busy 429 / 529 · unavailable 503 / 529 · timeout 504 · server 500
export class ApiError extends Error {
  constructor(kind, message, { param = null, retryAfter = null } = {}) {
    super(message);
    this.kind = kind; this.param = param; this.retryAfter = retryAfter;
  }
}
export const bad = (message, param = null) => new ApiError("bad", message, { param });

export const TOOLS_MSG = "tool calls are not supported by pooled serve yet (v1 is chat only)";
export const TEXT_MSG = "only text content is supported";

// "Continue", "OpenAI/Python", "Anthropic/JS", "curl", … from a User-Agent header
export function clientFromUA(ua) {
  ua = String(ua || "").trim();
  if (!ua) return "";
  const known = [
    [/^OpenAI\/Python/i, "OpenAI/Python"], [/^OpenAI\/JS/i, "OpenAI/JS"], [/^AsyncOpenAI\/Python/i, "OpenAI/Python"],
    [/^Anthropic\/Python/i, "Anthropic/Python"], [/^Anthropic\/JS/i, "Anthropic/JS"], [/^AsyncAnthropic\/Python/i, "Anthropic/Python"],
    [/continue/i, "Continue"], [/open-?webui/i, "Open WebUI"], [/litellm/i, "LiteLLM"], [/^curl\//i, "curl"],
    [/^python-requests/i, "python-requests"], [/^node-fetch|^undici|^node$/i, "Node"], [/^Mozilla\//, "a browser"],
  ];
  for (const [re, name] of known) if (re.test(ua)) return name;
  return ua.split(/[\s/(]/)[0].slice(0, LIMITS.client);
}
// text from the room for the terminal: no control characters (a hostile host could otherwise send
// escape sequences: clear the screen, write the clipboard through OSC 52, fake log lines), capped
export const cleanText = (s, n = 500) => String(s ?? "").replace(/[\u0000-\u001f\u007f-\u009f]/g, "").slice(0, n);
export const cleanLabel = (s) => String(s ?? "").replace(/[\u0000-\u001f\u007f<>"'`&]/g, "").trim().slice(0, LIMITS.client);

// stop: a string or a list of up to 4 non-empty strings of at most 64 characters
export function parseStop(v, param) {
  if (v == null) return [];
  const list = typeof v === "string" ? [v] : v;
  if (!Array.isArray(list)) throw bad(`${param} must be a string or a list of strings`, param);
  if (list.length > LIMITS.stops) throw bad(`${param}: at most ${LIMITS.stops} stop sequences`, param);
  for (const s of list) if (typeof s !== "string" || !s || s.length > LIMITS.stopLen) throw bad(`${param}: each stop sequence must be 1 to ${LIMITS.stopLen} characters`, param);
  return list;
}

export function checkInt(v, lo, hi, param) {
  if (v == null) return null;
  if (typeof v !== "number" || !Number.isInteger(v) || v < lo || v > hi) throw bad(`${param} must be an integer from ${lo} to ${hi}`, param);
  return v;
}
export function checkNum(v, lo, hi, param) {
  if (v == null) return null;
  if (typeof v !== "number" || !Number.isFinite(v) || v < lo || v > hi) throw bad(`${param} must be a number from ${lo} to ${hi}`, param);
  return v;
}

// consecutive same-role messages merged (blank line between); the last must be the user's
export function finishMessages(msgs, system) {
  const out = [];
  for (const m of msgs) {
    const last = out[out.length - 1];
    if (last && last.role === m.role) last.text += "\n\n" + m.text;
    else out.push({ role: m.role, text: m.text });
  }
  if (!out.length) throw bad("messages must contain at least one user message", "messages");
  if (out.length > LIMITS.messages) throw bad(`at most ${LIMITS.messages} messages`, "messages");
  if (out[out.length - 1].role !== "user") throw bad("the last message must be from the user (assistant prefill is not supported yet)", "messages");
  const chars = out.reduce((n, m) => n + m.text.length, system.length);
  if (chars > LIMITS.chars) throw bad(`the conversation is ${chars} characters; at most ${LIMITS.chars}`, "messages");
  return out;
}

let seq = 0;
export const newRid = () => (Date.now() % 1e8).toString(36) + (seq++).toString(36) + Math.random().toString(36).slice(2, 6);

// ============================================================================================
// The internal request (docs/design/serve.md 4.1). Each adapter's parse() returns one:
//   { api, stream, client, includeUsage, system, messages: Msg[], tools: Tool[] | null,
//     toolChoice: "auto" | "none" | "required" | { name }, allowed: string[] | null, parallel,
//     maxCalls, format: null | { type: "json" } | { type: "schema", schema, name? }, thinking,
//     effort, thinkBudget, showThinking, maxTokens, temperature, topK, stop, extra }
//   Msg = { role: "user", text, aside? } | { role: "assistant", text, calls?: [{ id, name, args }], reasoning? }
//       | { role: "tool", text, id? } | { role: "system", text }   (mid-conversation only)
// ============================================================================================
export const EFFORTS = ["low", "medium", "high", "xhigh", "max"];
export const OLD_HOST_MSG = "the room's host runs an older Pooled without tool calling (reload the host page)";
export const IMAGE_PLACEHOLDER = "[image omitted: this model reads text only]";

// every field an adapter may leave out, at its default
export function withDefaults(r) {
  return { client: "", includeUsage: false, system: "", messages: [], tools: null, toolChoice: "auto", allowed: null, parallel: true, maxCalls: null,
    format: null, thinking: false, effort: null, thinkBudget: null, showThinking: true, temperature: null, topK: null, stop: [], extra: {}, ...r };
}

// ids: prefix + 24 base62 characters
const B62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
export function id24(prefix) {
  const b = randomBytes(24);
  let s = prefix;
  for (let i = 0; i < 24; i++) s += B62[b[i] % 62];
  return s;
}
export const ids = { call: () => id24("call_"), toolu: () => id24("toolu_"), fc: () => id24("fc_"), msg: () => id24("msg_"), rs: () => id24("rs_"), resp: () => id24("resp_") };

// Reasoning carried through a client that only stores an opaque string (Anthropic's thinking
// signature, Responses' encrypted_content): "pooled1." + base64url(utf-8). Not encryption: the same
// user holds both ends. decode -> the text, or null for anything else (a foreign signature).
export const blob = {
  encode: (text) => "pooled1." + Buffer.from(String(text ?? ""), "utf8").toString("base64url"),
  decode: (s) => {
    if (typeof s !== "string" || !s.startsWith("pooled1.")) return null;
    const b = s.slice(8);
    if (!/^[A-Za-z0-9_-]*$/.test(b)) return null;
    try { return Buffer.from(b, "base64url").toString("utf8"); } catch { return null; }
  },
};

// a tool definition, checked and with defaults: { name, description, parameters }
export function normTool({ name, description, parameters } = {}, param = "tools") {
  if (typeof name !== "string" || !LIMITS.toolName.test(name)) throw bad(`${param}: tool names must be 1 to 128 letters, digits, _ . : or - (got ${JSON.stringify(String(name ?? "")).slice(0, 60)})`, param);
  if (description != null && typeof description !== "string") throw bad(`${param}: the description of ${name} must be a string`, param);
  const p = parameters == null ? { type: "object", properties: {} } : parameters;
  if (!p || typeof p !== "object" || Array.isArray(p)) throw bad(`${param}: the parameters of ${name} must be a JSON schema object`, param);
  const n = JSON.stringify(p).length;
  if (n > LIMITS.schemaChars) throw bad(`${param}: the schema of ${name} is ${n} characters; at most ${LIMITS.schemaChars}`, param);
  return { name, description: description || "", parameters: p };
}
export function checkTools(tools, param = "tools") {
  if (!tools || !tools.length) return null;
  if (tools.length > LIMITS.tools) throw bad(`at most ${LIMITS.tools} tools`, param);
  const seen = new Set();
  for (const t of tools) { if (seen.has(t.name)) throw bad(`${param}: tool ${t.name} is declared twice`, param); seen.add(t.name); }
  return tools;
}
// normalized shapes only (each adapter maps its own first)
export function checkToolChoice(toolChoice, tools, allowed, param = "tool_choice") {
  const names = new Set((tools || []).map((t) => t.name));
  if (toolChoice === "auto" || toolChoice === "none") { /* fine with or without tools */ }
  else if (toolChoice === "required") { if (!names.size) throw bad(`${param} "required" needs tools`, param); }
  else if (toolChoice && typeof toolChoice === "object" && typeof toolChoice.name === "string") {
    if (!names.size) throw bad(`${param} names a tool but there are no tools`, param);
    if (!names.has(toolChoice.name)) throw bad(`${param}: tool ${toolChoice.name} is not in tools`, param);
  } else throw bad(`${param} must be auto, none, required or a named tool`, param);
  if (allowed != null) {
    if (!Array.isArray(allowed) || !allowed.length) throw bad(`${param}: allowed tools must be a non-empty list`, param);
    for (const n of allowed) if (!names.has(n)) throw bad(`${param}: allowed tool ${n} is not in tools`, param);
  }
}
// { type: "json" } | { type: "schema", schema, name? } -> the same, checked
export function checkFormat(format, param = "response_format") {
  if (format == null) return null;
  if (format.type === "json") return { type: "json" };
  if (format.type === "schema") {
    if (!format.schema || typeof format.schema !== "object" || Array.isArray(format.schema)) throw bad(`${param}: a JSON schema object is required`, param);
    const n = JSON.stringify(format.schema).length;
    if (n > LIMITS.schemaChars) throw bad(`${param}: the schema is ${n} characters; at most ${LIMITS.schemaChars}`, param);
    return { type: "schema", schema: format.schema, ...(typeof format.name === "string" && format.name ? { name: format.name.slice(0, 64) } : {}) };
  }
  throw bad(`${param}: unsupported format`, param);
}

// a call's arguments as the client sent them: a JSON string (or already an object). Anything that
// is not a JSON object becomes {} (logged once per request), as vLLM does.
export function parseArgs(str, log = () => {}) {
  if (str && typeof str === "object" && !Array.isArray(str)) return str;
  if (str == null || str === "") return {};
  if (typeof str === "string") {
    try { const v = JSON.parse(str); if (v && typeof v === "object" && !Array.isArray(v)) return v; } catch { /* below */ }
  }
  log(`a tool call's arguments are not a JSON object; sent to the model as {}: ${String(typeof str === "string" ? str : JSON.stringify(str)).slice(0, 80)}`);
  return {};
}
// a tool result's content: a string, or parts; text is kept, images (and files) become a note
export function toolText(content, param = "content") {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) {
    if (typeof content === "object" && typeof content.text === "string") return content.text;
    throw bad(`${param} must be a string or a list of parts`, param);
  }
  return content.map((p) => {
    if (typeof p === "string") return p;
    const t = p?.type;
    if ((t === "text" || t === "input_text" || t === "output_text") && typeof p.text === "string") return p.text;
    if (t === "image" || t === "image_url" || t === "input_image") return IMAGE_PLACEHOLDER;
    if (t === "document" || t === "file" || t === "input_file") return "[file omitted: this model reads text only]";
    if (typeof p?.text === "string") return p.text;
    return "";
  }).join("");
}

// A request's messages as the host renders them (docs/design/serve.md 4.1):
//  1. system messages before anything else join `system`; a mid-conversation one folds into the
//     user-side turn before it (after a blank line; after tool results it becomes an aside), or,
//     with nothing before it, into the next one
//  2. consecutive user messages merge; consecutive assistant messages merge (text joined, calls
//     concatenated, the first non-empty reasoning kept)
//  3. each run of tool results is put in the order of the calls before it (matched by id; the
//     templates match results by position), unmatched ones last
//  4. the last message must be the user's or a tool result
// -> { system, messages }
export function normalizeMessages(messages, { system = "", log = () => {} } = {}) {
  void log;
  const out = [];
  let sys = system, pending = [];
  let i = 0;
  for (; i < messages.length && messages[i].role === "system"; i++) sys = sys ? sys + "\n\n" + messages[i].text : messages[i].text;
  const userSide = (m) => m && (m.role === "user" || m.role === "tool");
  for (; i < messages.length; i++) {
    const m = messages[i];
    if (m.role === "system") {
      const prev = out[out.length - 1];
      if (userSide(prev)) {
        if (prev.role === "user") prev.text = prev.text ? prev.text + "\n\n" + m.text : m.text;
        else out.push({ role: "user", text: m.text, aside: true });
      } else pending.push(m.text);
      continue;
    }
    let cur = { ...m };
    if (cur.calls) cur.calls = cur.calls.slice();
    if (pending.length && (cur.role === "user" || cur.role === "tool")) {
      if (cur.role === "user") cur.text = pending.join("\n\n") + (cur.text ? "\n\n" + cur.text : "");
      else { out.push({ role: "user", text: pending.join("\n\n"), aside: true }); }
      pending = [];
    }
    const prev = out[out.length - 1];
    if (prev && cur.role === "user" && prev.role === "user" && !!prev.aside === !!cur.aside) { prev.text = prev.text ? prev.text + "\n\n" + cur.text : cur.text; continue; }
    if (prev && cur.role === "assistant" && prev.role === "assistant") {
      prev.text = prev.text && cur.text ? prev.text + "\n\n" + cur.text : prev.text || cur.text;
      if (cur.calls?.length) prev.calls = [...(prev.calls || []), ...cur.calls];
      if (!prev.reasoning && cur.reasoning) prev.reasoning = cur.reasoning;
      continue;
    }
    out.push(cur);
  }
  if (pending.length) out.push({ role: "user", text: pending.join("\n\n") });
  // tool results in call order
  for (let j = 0; j < out.length; j++) {
    if (out[j].role !== "tool") continue;
    let k = j;
    while (k < out.length && out[k].role === "tool") k++;
    const prev = j > 0 ? out[j - 1] : null;
    const order = new Map((prev?.role === "assistant" ? prev.calls || [] : []).map((c, n) => [c.id, n]));
    if (order.size) {
      const run = out.slice(j, k).map((m, n) => ({ m, n, at: order.has(m.id) ? order.get(m.id) : Infinity }));
      run.sort((a, b) => a.at - b.at || a.n - b.n);
      out.splice(j, k - j, ...run.map((x) => x.m));
    }
    j = k - 1;
  }
  return { system: sys, messages: out };
}

// the ask needs the v2 protocol (a host that answers tools / formats)
export function needsV2(req) {
  return !!(req.tools?.length || req.format || (req.toolChoice && req.toolChoice !== "auto" && req.toolChoice !== "none")
    || req.messages.some((m) => m.role === "tool" || m.calls?.length || m.aside));
}

// characters of text the host will render
export function textChars(req) {
  let n = (req.system || "").length;
  for (const m of req.messages) { n += (m.text || "").length + (m.reasoning || "").length; for (const c of m.calls || []) n += c.name.length + JSON.stringify(c.args || {}).length; }
  for (const t of req.tools || []) n += t.name.length + t.description.length + JSON.stringify(t.parameters).length;
  return n;
}

// the body of the ai-ask this request becomes (docs/protocol.md "API clients"): v2 when the host
// answers it, else v1 (plain conversations only)
export function askBody(req, v2) {
  const params = { maxTokens: req.maxTokens, temperature: req.temperature ?? undefined, topK: req.topK ?? undefined, stop: req.stop, thinking: req.thinking,
    thinkBudget: req.thinkBudget ?? undefined, client: req.client };
  if (!v2) return { system: req.system, messages: req.messages.map((m) => ({ role: m.role, text: m.text })), params };
  return {
    api: 2, system: req.system,
    messages: req.messages.map((m) => m.role === "assistant"
      ? { role: "assistant", text: m.text, ...(m.calls?.length ? { calls: m.calls.map((c) => ({ name: c.name, args: c.args })) } : {}), ...(m.reasoning ? { reasoning: m.reasoning } : {}) }
      : m.role === "tool" ? { role: "tool", text: m.text } : { role: "user", text: m.text, ...(m.aside ? { aside: true } : {}) }),
    ...(req.tools?.length ? { tools: req.tools } : {}),
    params: { ...params, toolChoice: req.toolChoice, ...(req.allowed ? { allowed: req.allowed } : {}), parallel: req.parallel,
      ...(req.maxCalls != null ? { maxCalls: req.maxCalls } : {}), ...(req.format ? { format: req.format } : {}), ...(req.effort ? { effort: req.effort } : {}) },
  };
}

// Normalize and check an adapter's request before it is queued: messages (normalizeMessages), the
// limits, the tool choice, the size of the message that will go to the room, and an early context
// check against the host's context size (8 characters per token is far more than any text has).
// hostMeta: the host's hello meta ({ api, ctx }). -> the request, ready (throws ApiError)
export function finishRequest(req, { hostMeta = null, log = () => {} } = {}) {
  req = withDefaults(req);
  const { system, messages } = normalizeMessages(req.messages, { system: req.system || "", log });
  req.system = system; req.messages = messages;
  if (!messages.length) throw bad("messages must contain at least one user message", "messages");
  if (messages.length > LIMITS.totalMessages) throw bad(`at most ${LIMITS.totalMessages} messages`, "messages");
  const last = messages[messages.length - 1];
  if (last.role !== "user" && last.role !== "tool") throw bad("the last message must be from the user or a tool result (assistant prefill is not supported)", "messages");
  for (const m of messages) if (m.calls && m.calls.length > LIMITS.calls) throw bad(`at most ${LIMITS.calls} tool calls in one assistant message`, "messages");
  req.tools = checkTools(req.tools);
  checkToolChoice(req.toolChoice, req.tools, req.allowed);
  req.format = checkFormat(req.format);
  if (req.effort != null && !EFFORTS.includes(req.effort)) throw bad(`reasoning effort must be one of ${EFFORTS.join(", ")}`, "reasoning_effort");
  if (req.maxCalls != null && !(Number.isInteger(req.maxCalls) && req.maxCalls >= 1 && req.maxCalls <= LIMITS.maxCalls)) throw bad(`max_tool_calls must be an integer from 1 to ${LIMITS.maxCalls}`, "max_tool_calls");
  const chars = textChars(req);
  if (chars > LIMITS.totalChars) throw bad(`the conversation is ${chars} characters; at most ${LIMITS.totalChars}`, "messages");
  const v2 = (hostMeta?.api ?? 1) >= 2;
  if (!v2 && needsV2(req)) throw bad(OLD_HOST_MSG);
  const bytes = Buffer.byteLength(JSON.stringify(askBody(req, v2)));
  if (bytes > LIMITS.askBytes) throw new ApiError("toolarge", `the request is ${bytes} bytes for the room; at most ${LIMITS.askBytes}`);
  const ctx = +hostMeta?.ctx;
  if (ctx > 0 && chars > 8 * ctx) {
    const n = Math.ceil(chars / 8);
    throw new ApiError("ctx", req.api === "anthropic" ? `prompt is too long: ${n} tokens > ${ctx} maximum`
      : `This model's maximum context length is ${ctx} tokens. However, your messages resulted in more than ${n} tokens.`, { param: "messages" });
  }
  return req;
}

// How an answer ended, for each API's finish / stop reason:
//   tool (complete calls, ended normally, the choice not a named tool) | stop | stop_seq | length | ctx | error
export function outcome(answer, req) {
  if (answer.reason === "error") return "error";
  if (answer.reason === "ctx") return "ctx";
  if (answer.reason === "max") return "length";
  if (answer.reason === "stop_seq") return "stop_seq";
  if (answer.calls?.length && !(req.toolChoice && typeof req.toolChoice === "object")) return "tool";
  return "stop";
}
