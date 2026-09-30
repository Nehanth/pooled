// OpenClaw's model context (system prompt, user / assistant / toolResult messages, tools) -> the
// internal request `pooled serve` builds from an OpenAI or Anthropic body (cli/lib/common.js), then
// the same normalization, checks and ask body (finishRequest, askBody). From there the room's host
// renders the chat template, constrains tool calls with the grammar and parses them (room/api.js
// apiRun2) whether it is this process (@pooled/room-node) or a browser tab: one tool-call path.
import { compileSchema } from "../../../harness/jsonschema.js";
import { withDefaults, finishRequest, askBody, needsV2, parseArgs, IMAGE_PLACEHOLDER, LIMITS } from "../../../cli/lib/common.js";

// the start of a notice turn (stream.js: the room talking, not the model): dropped on replay
export const NOTICE = "⚠️ Pooled: ";
export const MAX_TOKENS = 8192;   // one answer's cap (OpenClaw asks for its model's output limit)
const partsText = (c) => typeof c === "string" ? c
  : Array.isArray(c) ? c.map((p) => p?.type === "text" ? p.text : p?.type === "image" ? IMAGE_PLACEHOLDER : "").join("") : "";

// a tool's JSON schema as plain JSON; one the room's call grammar cannot compile falls back to a
// free object (the call still parses, its arguments are just not constrained)
export function toolSchema(t, log = () => {}) {
  let p = t.parameters ? JSON.parse(JSON.stringify(t.parameters)) : { type: "object", properties: {} };
  if (!p || typeof p !== "object" || Array.isArray(p)) p = { type: "object", properties: {} };
  try { compileSchema(p); } catch (e) { log(`tool ${t.name}: schema not usable by the call grammar (${e.message}); arguments unconstrained`); p = { type: "object" }; }
  return p;
}

// -> the internal request (withDefaults' shape), not yet checked
export function toRequest(context, options = {}, model = {}, log = () => {}) {
  const messages = [];
  if (context.systemPrompt) messages.push({ role: "system", text: String(context.systemPrompt) });
  for (const m of context.messages || []) {
    if (m.role === "user") messages.push({ role: "user", text: partsText(m.content) });
    else if (m.role === "assistant") {
      const parts = Array.isArray(m.content) ? m.content : [{ type: "text", text: String(m.content ?? "") }];
      const text = parts.filter((p) => p.type === "text").map((p) => p.text).join("");
      const reasoning = parts.filter((p) => p.type === "thinking" && !p.redacted).map((p) => p.thinking).join("");
      const calls = parts.filter((p) => p.type === "toolCall").map((p) => ({ id: p.id, name: p.name, args: parseArgs(p.arguments, log) }));
      if (m.stopReason === "error" && !text && !calls.length) continue;   // a failed turn: nothing to replay
      if (text.startsWith(NOTICE) && !calls.length) continue;   // a notice from the room, not an answer
      messages.push({ role: "assistant", text, ...(calls.length ? { calls } : {}), ...(reasoning ? { reasoning } : {}) });
    } else if (m.role === "toolResult") {
      messages.push({ role: "tool", id: m.toolCallId, text: (m.isError ? "Error: " : "") + partsText(m.content) });
    }
  }
  // the room answers after a user message or tool results
  const last = messages[messages.length - 1];
  if (!last || (last.role !== "user" && last.role !== "tool")) messages.push({ role: "user", text: "Continue." });
  const tools = (context.tools || []).filter((t) => LIMITS.toolName.test(t.name))
    .map((t) => ({ name: t.name, description: String(t.description || "").slice(0, 4000), parameters: toolSchema(t, log) }));
  const reasoning = options.reasoning && options.reasoning !== "off" && options.reasoning !== "none";
  return withDefaults({
    client: "OpenClaw", messages,
    maxTokens: Math.max(1, Math.min(options.maxTokens || model.maxTokens || 4096, MAX_TOKENS)),
    temperature: options.temperature ?? null,
    thinking: !!(reasoning && model.reasoning),
    tools: tools.length ? tools : null, toolChoice: "auto",
  });
}

// -> { req (normalized and checked against the host), v2, body (the ai-ask body) }; throws ApiError
// (an older host and tools, too large, past the host's context)
export function toAsk(context, options = {}, model = {}, { hostMeta = null, log = () => {} } = {}) {
  const req = finishRequest(toRequest(context, options, model, log), { hostMeta, log });
  const v2 = needsV2(req);
  return { req, v2, body: askBody(req, v2) };
}
