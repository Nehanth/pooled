// Shared by the OpenAI and Anthropic mappings: the internal request both turn into, the error
// kinds and their statuses, limits, and the client label the room shows.

// Limits (docs/design/serve.md section 7). The host checks the same ones again.
export const LIMITS = {
  body: 1 << 20,          // bytes
  chars: 400000,          // total text
  messages: 200,
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
