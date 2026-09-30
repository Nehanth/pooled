// The OpenAI Responses API (docs/design/serve.md section 7): POST /v1/responses mapped onto the
// internal request (input items, instructions, function tools, tool_choice, text.format, reasoning,
// previous_response_id), the response object and its typed event stream, and the stored responses
// behind GET / DELETE /v1/responses/{id} and GET /v1/responses/{id}/input_items (store.js).
//
// Ids are minted once per response and item, so a streamed response and the object it ends with
// (and the stored one) carry the same ids. Reasoning goes out as reasoning items with the text in
// content[] (reasoning_text); with include: ["reasoning.encrypted_content"] they also carry
// encrypted_content = "pooled1." + base64url(text), which, sent back, restores the reasoning exactly
// (it is not encryption: the same user holds both ends).
import { ApiError, bad, LIMITS, TEXT_MSG, checkInt, checkNum, capTokens, checkFormat, withDefaults, ids, id24, normTool, parseArgs, toolText, nonTextPart, blob, outcome, clientFromUA, cleanText, EFFORTS, warnOnce } from "./common.js";
import { openaiError } from "./openai.js";
import { sseSequence, sseComment } from "./sse.js";
import { ResponseStore } from "./store.js";

// max_output_tokens when the client sends none (Codex never does): the room's context bounds it anyway
export const DEFAULT_MAX_OUTPUT = LIMITS.defaultMaxTokens;

// items a session resumed from OpenAI's own service may hold: calls of tools that ran there. The
// model here cannot see them; they are dropped with a warning, never a 400
const HOSTED_ITEM = /^(web_search_call|file_search_call|local_shell_call(_output)?|shell_call(_output)?|apply_patch_call(_output)?|code_interpreter_call|image_generation_call|mcp_[a-z_]+|computer_call(_output)?|tool_search_call|tool_search_output)$/;

// the hosted tool types (they run on OpenAI's side; none runs here)
const HOSTED_TOOL = /^(web_search|file_search|code_interpreter|image_generation|mcp|computer(_use)?|local_shell|shell|apply_patch|tool_search)(_[a-z0-9_]+)?$/;

// errors in OpenAI's shape; a code of our own where the API has one (previous_response_not_found)
export function responsesError(e) {
  const r = openaiError(e);
  if (e?.code) r.body.error.code = e.code;
  return r;
}
const notFound = (id) => new ApiError("notfound", `Response with id '${cleanText(id, 80)}' not found.`);

// Custom (free-form) tools: the model writes one raw string (Codex's apply_patch, with a GPT-5 model
// name). Each becomes a function tool with a single string parameter, `input`; its calls go back out as
// custom_tool_call items with that string. The grammar a custom tool may name (lark, regex) is shown
// to the model in the description, not enforced.
const CUSTOM_PARAMS = { type: "object", properties: { input: { type: "string", description: "the tool's raw input" } }, required: ["input"], additionalProperties: false };
function customTool(t, p) {
  if (typeof t.name !== "string") throw bad(`${p}.name is required`, `${p}.name`);
  if (t.description != null && typeof t.description !== "string") throw bad(`${p}.description must be a string`, `${p}.description`);
  const f = t.format;
  let desc = t.description || "";
  if (f != null && f.type === "grammar" && typeof f.definition === "string") desc += `${desc ? "\n\n" : ""}The input follows this ${cleanText(f.syntax, 20) || ""} grammar:\n${f.definition}`;
  desc += `${desc ? "\n\n" : ""}Put the whole raw input in the input argument.`;
  return normTool({ name: t.name, description: desc, parameters: CUSTOM_PARAMS }, p);
}
// a custom call's input from the function arguments the room wrote ({"input": "..."})
function customInput(args) {
  try { const v = JSON.parse(args); if (v && typeof v.input === "string") return v.input; } catch { /* below */ }
  return typeof args === "string" ? args : "";
}

// a message's content: a string or parts; text kept, images and files a short note (common.nonTextPart)
function contentText(content, p, log = () => {}) {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) throw bad(`${p}.content must be a string or a list of parts`, `${p}.content`);
  return content.map((c, j) => {
    const t = c?.type;
    if ((t === "input_text" || t === "output_text" || t === "text") && typeof c.text === "string") return c.text;
    if (t === "refusal") return typeof c.refusal === "string" ? c.refusal : "";
    const note = nonTextPart(t, log, "responses");
    if (note != null) return note;
    throw bad(`${TEXT_MSG} (${p}.content[${j}] is ${JSON.stringify(t ?? typeof c)})`, `${p}.content`);
  }).join("");
}

// a reasoning item's text: our own blob wins; else content[] (reasoning_text), else summary[]; null if none
function reasoningText(it) {
  const own = blob.decode(it.encrypted_content);
  if (own != null) return own;
  const parts = (list) => (Array.isArray(list) ? list.filter((c) => typeof c?.text === "string").map((c) => c.text) : []);
  const content = parts(it.content).join("");
  if (content) return content;
  const summary = parts(it.summary).join("\n\n");
  return summary || null;
}

// an input item as the input_items listing shows it (an id for each)
function listed(it) {
  if (typeof it.id === "string" && it.id) return it;
  const t = it.type ?? "message";
  const id = t === "message" ? ids.msg() : t === "function_call" ? ids.fc() : t === "custom_tool_call" ? id24("ctc_") : t === "reasoning" ? ids.rs()
    : id24(t === "function_call_output" ? "fco_" : t === "custom_tool_call_output" ? "ctco_" : "item_");
  return { id, ...(it.type ? {} : { type: "message" }), ...it };
}

// input: a string or a list of items -> { messages: Msg[], items: [the items, with ids] }
function parseInput(input, { store, log, client }) {
  if (typeof input === "string") return { messages: [{ role: "user", text: input }], items: [{ id: ids.msg(), type: "message", role: "user", content: [{ type: "input_text", text: input }] }] };
  if (!Array.isArray(input)) throw bad("input must be a string or a list of items", "input");
  const msgs = [], items = [];
  let reasoning = null;
  // a reasoning item belongs to the assistant turn after it
  const assistant = (m) => { if (reasoning != null) { m.reasoning = reasoning; reasoning = null; } msgs.push(m); };
  input.forEach((raw, i) => {
    const p = `input[${i}]`;
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw bad(`${p} must be an object`, p);
    let it = raw;
    if (it.type === "item_reference") {
      const got = typeof it.id === "string" ? store.item(it.id) : null;
      if (!got) throw bad(`${p}: item ${cleanText(it.id, 80)} is not stored here (pooled serve keeps responses in memory for an hour)`, `${p}.id`);
      it = got;
    }
    items.push(listed(it));
    const type = it.type ?? (it.role != null ? "message" : undefined);
    switch (type) {
      case "message": {
        const text = contentText(it.content, p, log);
        if (it.role === "user") msgs.push({ role: "user", text });
        else if (it.role === "system" || it.role === "developer") msgs.push({ role: "system", text });
        else if (it.role === "assistant") assistant({ role: "assistant", text });
        else throw bad(`${p}.role must be user, assistant, system or developer`, `${p}.role`);
        return;
      }
      case "function_call": {
        if (typeof it.name !== "string" || !LIMITS.toolName.test(it.name)) throw bad(`${p}.name must be a tool name`, `${p}.name`);
        if (typeof it.call_id !== "string" || !it.call_id) throw bad(`${p}.call_id is required`, `${p}.call_id`);
        assistant({ role: "assistant", text: "", calls: [{ id: it.call_id, name: it.name, args: parseArgs(it.arguments, log) }] });
        return;
      }
      case "custom_tool_call": {
        if (typeof it.name !== "string" || !LIMITS.toolName.test(it.name)) throw bad(`${p}.name must be a tool name`, `${p}.name`);
        if (typeof it.call_id !== "string" || !it.call_id) throw bad(`${p}.call_id is required`, `${p}.call_id`);
        if (it.input != null && typeof it.input !== "string") throw bad(`${p}.input must be a string`, `${p}.input`);
        assistant({ role: "assistant", text: "", calls: [{ id: it.call_id, name: it.name, args: { input: it.input ?? "" } }] });
        return;
      }
      case "custom_tool_call_output":
      case "function_call_output": {
        if (typeof it.call_id !== "string" || !it.call_id) throw bad(`${p}.call_id is required`, `${p}.call_id`);
        msgs.push({ role: "tool", text: toolText(it.output, `${p}.output`), id: it.call_id });
        return;
      }
      case "reasoning": {
        const r = reasoningText(it);
        if (r != null) reasoning = reasoning ? reasoning + "\n\n" + r : r;
        return;
      }
      default:
        if (typeof type === "string" && HOSTED_ITEM.test(type)) { warnOnce(log, `${client}:item:${type}`, `responses (${client}): dropped ${cleanText(type, 40)} items from the input (a tool that ran elsewhere; the model here cannot see them)`); return; }
        throw bad(`${p}: input items of type ${JSON.stringify(cleanText(type ?? typeof it, 40))} are not supported`, `${p}.type`);
    }
  });
  return { messages: msgs, items };
}

// a stored response's output items -> the assistant turn they were (incomplete calls left out)
export function outputMessages(output) {
  let text = "", reasoning = "";
  const calls = [];
  for (const it of output || []) {
    if (it.type === "reasoning") { const r = reasoningText(it); if (r) reasoning = reasoning ? reasoning + "\n\n" + r : r; }
    else if (it.type === "message") text += contentText(it.content, "output");
    else if (it.type === "function_call" && it.status !== "incomplete") calls.push({ id: it.call_id, name: it.name, args: parseArgs(it.arguments) });
    else if (it.type === "custom_tool_call" && it.status !== "incomplete") calls.push({ id: it.call_id, name: it.name, args: { input: it.input ?? "" } });
  }
  if (!text && !calls.length && !reasoning) return [];
  return [{ role: "assistant", text, ...(calls.length ? { calls } : {}), ...(reasoning ? { reasoning } : {}) }];
}

// body (parsed JSON) -> internal request
export function parseResponses(b, headers = {}, { log = () => {}, store }) {
  if (!b || typeof b !== "object" || Array.isArray(b)) throw bad("the body must be a JSON object");
  if (b.background) throw bad("background responses are not supported by pooled serve", "background");
  if (b.conversation != null) throw bad("conversations are not supported by pooled serve: use previous_response_id", "conversation");
  if (b.prompt != null) throw bad("prompt templates are not supported by pooled serve", "prompt");
  if (b.top_logprobs) throw bad("logprobs are not supported", "top_logprobs");
  if (b.include != null && !Array.isArray(b.include)) throw bad("include must be a list", "include");
  const include = new Set(b.include || []);
  if (include.has("message.output_text.logprobs")) throw bad("logprobs are not supported", "include");
  if (b.instructions != null && typeof b.instructions !== "string") throw bad("instructions must be a string", "instructions");
  if (b.input == null) throw bad("input is required", "input");
  if (b.metadata != null && (typeof b.metadata !== "object" || Array.isArray(b.metadata))) throw bad("metadata must be an object", "metadata");
  if (b.store != null && typeof b.store !== "boolean") throw bad("store must be a boolean", "store");
  if (b.parallel_tool_calls != null && typeof b.parallel_tool_calls !== "boolean") throw bad("parallel_tool_calls must be a boolean", "parallel_tool_calls");
  const client = clientFromUA(headers["user-agent"]) || "API";

  // the conversation so far: the previous response's (its input chain and its output), then this input
  let history = [];
  const previousId = b.previous_response_id ?? null;
  if (previousId != null) {
    if (typeof previousId !== "string") throw bad("previous_response_id must be a string", "previous_response_id");
    const prev = store.get(previousId);
    if (!prev) {
      const e = new ApiError("bad", `Previous response with id '${cleanText(previousId, 80)}' not found (pooled serve keeps stored responses in memory for an hour).`, { param: "previous_response_id" });
      e.code = "previous_response_not_found";
      throw e;
    }
    history = [...prev.history, ...prev.output];
  }
  const { messages: input, items: inputItems } = parseInput(b.input, { store, log, client });
  const messages = [...history, ...input];
  if (!input.length) throw bad("input must contain at least one item", "input");
  if (messages.at(-1).role === "assistant") throw bad("the last input item must be a user message or a function_call_output (assistant prefill is not supported)", "input");

  // tools: function tools; the hosted ones (web_search, file_search, …) cannot run here and are skipped
  let tools = null;
  const echoTools = [], custom = new Set();
  if (b.tools != null) {
    if (!Array.isArray(b.tools)) throw bad("tools must be a list", "tools");
    tools = [];
    b.tools.forEach((t, i) => {
      if (!t || typeof t !== "object") throw bad(`tools[${i}] must be an object`, `tools[${i}]`);
      if (t.type === "function") {
        tools.push(normTool({ name: t.name, description: t.description, parameters: t.parameters }, `tools[${i}]`));
        echoTools.push({ type: "function", name: t.name, description: t.description ?? null, parameters: t.parameters ?? null, strict: t.strict ?? true });
      } else if (t.type === "custom") {
        tools.push(customTool(t, `tools[${i}]`));
        custom.add(t.name);
        echoTools.push({ type: "custom", name: t.name, description: t.description ?? null, ...(t.format != null ? { format: t.format } : {}) });
      } else warnOnce(log, `${client}:tool:${t.type}`, `responses (${client}): skipped the hosted tool ${cleanText(t.type, 40)} (pooled serve runs function tools only)`);
    });
  }

  let toolChoice = "auto", allowed = null;
  const tc = b.tool_choice;
  if (tc == null || tc === "auto") { /* default */ }
  else if (tc === "none" || tc === "required") toolChoice = tc;
  else if (tc && typeof tc === "object" && (tc.type === "function" || tc.type === "custom")) {
    if (typeof tc.name !== "string") throw bad("tool_choice.name is required", "tool_choice");
    if (tc.type === "custom" && !custom.has(tc.name)) throw bad(`tool_choice: ${cleanText(tc.name, 60)} is not a custom tool in tools`, "tool_choice");
    toolChoice = { name: tc.name };
  } else if (tc && typeof tc === "object" && tc.type === "allowed_tools") {
    const at = tc.allowed_tools && typeof tc.allowed_tools === "object" ? tc.allowed_tools : tc;
    const mode = at.mode ?? "auto";
    if (mode !== "auto" && mode !== "required") throw bad("tool_choice.mode must be auto or required", "tool_choice");
    if (!Array.isArray(at.tools)) throw bad("tool_choice.tools must be a list", "tool_choice");
    // each entry: a function or custom tool by name, or a hosted tool (none of which runs here). A
    // function entry without a flat name (the Chat shape, {function: {name}}) or an unknown type is a
    // 400, never silently "no tools"
    const names = [];
    at.tools.forEach((t, j) => {
      const q = `tool_choice.tools[${j}]`;
      if (!t || typeof t !== "object" || typeof t.type !== "string") throw bad(`${q} must be an object with a type`, "tool_choice");
      if (t.type === "function" || t.type === "custom") {
        if (typeof t.name !== "string" || !t.name) throw bad(`${q}.name is required (the Responses shape: {"type": "${t.type}", "name": "..."})`, "tool_choice");
        names.push(t.name);
      } else if (!HOSTED_TOOL.test(t.type)) throw bad(`${q}: unknown tool type ${JSON.stringify(cleanText(t.type, 40))}`, "tool_choice");
    });
    if (names.length) { toolChoice = mode; allowed = names; }
    else if (mode === "required") throw bad("tool_choice: no function tool is allowed, so none can be required", "tool_choice");
    else toolChoice = "none";   // only hosted tools allowed: none of them runs here
  } else if (tc && typeof tc === "object" && typeof tc.type === "string") throw bad(`tool_choice ${JSON.stringify(cleanText(tc.type, 40))} is not supported by pooled serve (function tools only)`, "tool_choice");
  else throw bad("tool_choice must be auto, none, required or a function", "tool_choice");

  // text.format -> the answer's format
  const f = b.text?.format;
  let format = null;
  if (f != null && f.type !== "text") {
    if (f.type === "json_object") format = { type: "json" };
    else if (f.type === "json_schema") format = checkFormat({ type: "schema", schema: f.schema, name: f.name }, "text.format");
    else throw bad(`text.format.type must be text, json_object or json_schema`, "text.format");
  }

  // reasoning.effort: none / minimal off; low … max on; absent: off (the serve default)
  let thinking = false, effort = null;
  if (b.reasoning != null) {
    if (typeof b.reasoning !== "object" || Array.isArray(b.reasoning)) throw bad("reasoning must be an object", "reasoning");
    const e = b.reasoning.effort;
    if (e == null || e === "none" || e === "minimal") thinking = false;
    else if (EFFORTS.includes(e)) { thinking = true; effort = e; }
    else throw bad(`reasoning.effort must be one of none, minimal, ${EFFORTS.join(", ")}`, "reasoning.effort");
  }

  // above LIMITS.maxTokens: capped, as on the Messages side (the echo keeps the client's value)
  const mt = capTokens(b.max_output_tokens, "max_output_tokens");
  const temperature = checkNum(b.temperature, 0, 2, "temperature");
  checkNum(b.top_p, 0, 1, "top_p");   // accepted, ignored: the sampler has no nucleus cut
  const tk = b.top_k == null ? null : checkInt(b.top_k, 1, 1e9, "top_k");
  const maxCalls = checkInt(b.max_tool_calls, 1, LIMITS.maxCalls, "max_tool_calls");

  return withDefaults({
    api: "responses",
    stream: !!b.stream,
    system: b.instructions || "",
    messages,
    tools: tools && tools.length ? tools : null,
    toolChoice, allowed,
    parallel: b.parallel_tool_calls !== false,
    maxCalls,
    format,
    thinking, effort,
    maxTokens: mt ?? DEFAULT_MAX_OUTPUT,
    temperature, topK: tk == null ? null : Math.min(tk, LIMITS.topK),
    stop: [],
    extra: {
      respId: ids.resp(),
      store: b.store !== false,
      previousId,
      encrypted: include.has("reasoning.encrypted_content"),
      history: messages,     // as parsed (before normalization): what a later previous_response_id continues
      inputItems,
      fc: new Map(),         // call_id -> fc_ / ctc_ item id
      custom,                // names of the custom tools (their calls go out as custom_tool_call)
      echo: {
        instructions: b.instructions ?? null,
        max_output_tokens: b.max_output_tokens ?? null,
        max_tool_calls: maxCalls ?? null,
        reasoning: { effort: b.reasoning?.effort ?? null, summary: b.reasoning?.summary ?? null },
        temperature: temperature ?? 1,
        text: { format: f == null ? { type: "text" } : f, ...(b.text?.verbosity ? { verbosity: b.text.verbosity } : {}) },
        tool_choice: tc ?? "auto",
        tools: echoTools,
        top_p: b.top_p ?? 1,
        truncation: b.truncation ?? "disabled",
        user: b.user ?? null,
        metadata: b.metadata ?? {},
      },
    },
  });
}

// ---- output ----
const isCustom = (req, name) => !!req.extra.custom?.has(name);
// a call's item id, minted once per call_id (so stream, final and store agree): fc_ for a function
// call, ctc_ for a custom tool's
const fcId = (req, callId, name) => {
  let id = req.extra.fc.get(callId);
  if (!id) { id = isCustom(req, name) ? id24("ctc_") : ids.fc(); req.extra.fc.set(callId, id); }
  return id;
};
const outputText = (text) => ({ type: "output_text", text, annotations: [], logprobs: [] });
const reasoningItem = (req, id, text) => ({ id, type: "reasoning", summary: [], content: [{ type: "reasoning_text", text }], ...(req.extra.encrypted ? { encrypted_content: blob.encode(text) } : {}) });
const messageItem = (id, text, status) => ({ id, type: "message", status, role: "assistant", content: [outputText(text)] });
// a call as its output item: function_call { arguments }, or for a custom tool custom_tool_call { input }
const callItem = (req, callId, name, args, status) => (isCustom(req, name)
  ? { id: fcId(req, callId, name), type: "custom_tool_call", status, call_id: callId, name, input: customInput(args) }
  : { id: fcId(req, callId, name), type: "function_call", status, arguments: args, call_id: callId, name });
const usageOf = (a) => ({ input_tokens: a.usage.in, input_tokens_details: { cached_tokens: a.reused || 0 }, output_tokens: a.usage.out,
  output_tokens_details: { reasoning_tokens: a.usage.think || 0 }, total_tokens: a.usage.in + a.usage.out });
const isIncomplete = (a, req) => { const o = outcome(a, req); return o === "length" || o === "ctx"; };

// the response object: { status, output, usage, … } over the request's echoed fields
export function responseObject(req, meta, { status = "in_progress", output = [], usage = null, incomplete = false, error = null } = {}) {
  const x = req.extra, e = x.echo;
  return {
    id: x.respId, object: "response", created_at: meta.created, status, background: false, error,
    incomplete_details: incomplete ? { reason: "max_output_tokens" } : null,
    instructions: e.instructions, max_output_tokens: e.max_output_tokens, max_tool_calls: e.max_tool_calls, model: meta.model,
    output, parallel_tool_calls: req.parallel, previous_response_id: x.previousId, reasoning: e.reasoning, service_tier: "default",
    store: x.store, temperature: e.temperature, text: e.text, tool_choice: e.tool_choice, tools: e.tools, top_p: e.top_p,
    truncation: e.truncation, usage, user: e.user, metadata: e.metadata,
  };
}

// the whole response, non-stream: reasoning, then the message (left out when empty), then the calls
export function responsesFinal(a, req) {
  const inc = isIncomplete(a, req);
  const out = [];
  if (a.think) out.push(reasoningItem(req, ids.rs(), a.think));
  if (a.text.trim()) out.push(messageItem(ids.msg(), a.text, inc && !a.calls.length && !a.open ? "incomplete" : "completed"));
  for (const c of a.calls) out.push(callItem(req, c.id, c.name, c.args, "completed"));
  if (a.open) out.push(callItem(req, a.open.id, a.open.name, a.open.args, "incomplete"));
  const r = responseObject(req, { created: a.created, model: a.model }, { status: inc ? "incomplete" : "completed", output: out, usage: usageOf(a), incomplete: inc });
  req.extra.final = r;
  return r;
}

// The stream: response.created, response.in_progress, then each item in the order it is generated
// (opened on its first event, closed before the next one opens), then response.completed or
// response.incomplete with the whole response; every event with a strictly increasing
// sequence_number. After an error: response.failed.
export class ResponsesEncoder {
  constructor(req, sse, meta) {
    this.req = req; this.sse = sse; this.meta = meta;
    this.ev = sseSequence();
    this.started = false;
    this.items = [];   // { kind: "reasoning" | "message" | "call", item, text, i?, closed }
    this.cur = null;
    this.ws = "";      // whitespace-only text held until a message has something to say
  }
  emit(type, obj) { this.sse.write(this.ev(type, obj)); }
  start() {
    if (this.started) return;
    this.started = true;
    this.emit("response.created", { response: responseObject(this.req, this.meta) });
    this.emit("response.in_progress", { response: responseObject(this.req, this.meta) });
  }
  open(kind, item, extra = {}) {
    this.close();
    this.ws = "";
    const x = { kind, item, text: "", closed: false, index: this.items.length, ...extra };
    this.items.push(x);
    this.cur = x;
    this.emit("response.output_item.added", { output_index: x.index, item });
    if (kind === "reasoning") this.emit("response.content_part.added", { item_id: item.id, output_index: x.index, content_index: 0, part: { type: "reasoning_text", text: "" } });
    if (kind === "message") this.emit("response.content_part.added", { item_id: item.id, output_index: x.index, content_index: 0, part: outputText("") });
    return x;
  }
  // close the open reasoning or message item (a call closes itself at callEnd)
  close(status = "completed") {
    const x = this.cur;
    if (!x || x.closed || x.kind === "call") return;
    x.closed = true;
    const base = { item_id: x.item.id, output_index: x.index, content_index: 0 };
    if (x.kind === "reasoning") {
      x.final = reasoningItem(this.req, x.item.id, x.text);
      this.emit("response.reasoning_text.done", { ...base, text: x.text });
      this.emit("response.content_part.done", { ...base, part: { type: "reasoning_text", text: x.text } });
    } else {
      x.final = messageItem(x.item.id, x.text, status);
      this.emit("response.output_text.done", { ...base, text: x.text, logprobs: [] });
      this.emit("response.content_part.done", { ...base, part: outputText(x.text) });
    }
    this.emit("response.output_item.done", { output_index: x.index, item: x.final });
    this.cur = null;
  }
  think(t) {
    if (!t) return;
    if (this.cur?.kind !== "reasoning") this.open("reasoning", { id: ids.rs(), type: "reasoning", summary: [], content: [] });
    this.cur.text += t;
    this.emit("response.reasoning_text.delta", { item_id: this.cur.item.id, output_index: this.cur.index, content_index: 0, delta: t });
  }
  text(t) {
    if (!t) return;
    if (this.cur?.kind !== "message") {
      if (!(this.ws + t).trim()) { this.ws += t; return; }
      const held = this.ws;
      this.open("message", { id: ids.msg(), type: "message", status: "in_progress", role: "assistant", content: [] });
      t = held + t;
    }
    this.cur.text += t;
    this.emit("response.output_text.delta", { item_id: this.cur.item.id, output_index: this.cur.index, content_index: 0, delta: t, logprobs: [] });
  }
  callStart(i, id, name) {
    this.open("call", callItem(this.req, id, name, isCustom(this.req, name) ? '{"input":""}' : "", "in_progress"), { i, callId: id, name, custom: isCustom(this.req, name) });
  }
  // a custom call's input is a string inside the JSON arguments: it goes out whole when the call ends
  callArgs(i, frag) {
    const x = this.items.find((y) => y.kind === "call" && y.i === i);
    if (!x || x.closed) return;
    x.text += frag;
    if (!x.custom) this.emit("response.function_call_arguments.delta", { item_id: x.item.id, output_index: x.index, delta: frag });
  }
  callEnd(i, args) {
    const x = this.items.find((y) => y.kind === "call" && y.i === i);
    if (!x || x.closed) return;
    this.endCall(x, typeof args === "string" ? args : x.text);
  }
  // status "incomplete": the call the answer was cut inside (max_output_tokens); its item still gets
  // its output_item.done, as every added item does, but no arguments / input done event
  endCall(x, args, status = "completed") {
    x.closed = true;
    x.text = args;
    x.final = callItem(this.req, x.callId, x.name, args, status);
    if (status === "completed") {
      if (x.custom) {
        const input = customInput(args);
        this.emit("response.custom_tool_call_input.delta", { item_id: x.item.id, output_index: x.index, delta: input });
        this.emit("response.custom_tool_call_input.done", { item_id: x.item.id, output_index: x.index, input });
      } else this.emit("response.function_call_arguments.done", { item_id: x.item.id, output_index: x.index, name: x.name, arguments: args });
    }
    this.emit("response.output_item.done", { output_index: x.index, item: x.final });
    if (this.cur === x) this.cur = null;
  }
  done(a) {
    this.start();
    const inc = isIncomplete(a, this.req);
    const final = new Map(a.calls.map((c) => [c.id, c]));
    const openId = a.open?.id ?? null;
    this.ws = "";
    // the open reasoning / message ends here; a call the room ended without its end message ends with
    // the final arguments; the call the answer was cut inside ends as incomplete
    if (this.cur && this.cur.kind !== "call") this.close(inc ? "incomplete" : "completed");
    for (const x of this.items) if (x.kind === "call" && !x.closed && x.callId !== openId && final.has(x.callId)) this.endCall(x, final.get(x.callId).args);
    for (const x of this.items) if (x.kind === "call" && !x.closed && x.callId === openId) this.endCall(x, x.text, "incomplete");
    // calls only the final message had (never streamed)
    const streamed = new Set(this.items.filter((x) => x.kind === "call").map((x) => x.callId));
    for (const c of a.calls) {
      if (streamed.has(c.id)) continue;
      const custom = isCustom(this.req, c.name);
      const x = this.open("call", callItem(this.req, c.id, c.name, custom ? '{"input":""}' : "", "in_progress"), { i: -1, callId: c.id, name: c.name, custom });
      if (!custom) this.emit("response.function_call_arguments.delta", { item_id: x.item.id, output_index: x.index, delta: c.args });
      this.endCall(x, c.args);
    }
    // the final output: what was streamed, with the final arguments; calls the answer does not have are left out
    const output = [];
    for (const x of this.items) {
      if (x.kind !== "call") { output.push(x.final); continue; }
      if (final.has(x.callId)) output.push(callItem(this.req, x.callId, x.name, final.get(x.callId).args, "completed"));
      else if (x.callId === openId) output.push(callItem(this.req, x.callId, x.name, x.text, "incomplete"));
    }
    const r = responseObject(this.req, this.meta, { status: inc ? "incomplete" : "completed", output, usage: usageOf(a), incomplete: inc });
    this.req.extra.final = r;
    this.emit(inc ? "response.incomplete" : "response.completed", { response: r });
  }
  error(e) {
    this.start();
    const { body } = responsesError(e);
    const output = this.items.filter((x) => x.closed && x.final).map((x) => x.final);
    this.emit("response.failed", { response: responseObject(this.req, this.meta, { status: "failed", output, error: { code: body.error.code || "server_error", message: body.error.message } }) });
  }
  // Real events, not SSE comments: Codex's stream_idle_timeout_ms (300 s) restarts only on an event,
  // and a queue wait or a long prefill can be longer. Before the room starts: response.created (the
  // status is committed from here on; errors become response.failed); after: response.in_progress.
  keepAlive() {
    if (!this.started) { this.start(); return; }
    this.emit("response.in_progress", { response: responseObject(this.req, this.meta) });
  }
}

// ---- stored responses ----
function storeIt(store, a, req) {
  const x = req.extra;
  if (!x.store || !x.final) return;
  store.put({ id: x.respId, response: x.final, history: x.history, output: outputMessages(x.final.output), inputItems: x.inputItems });
}
const idOf = (path) => { try { return decodeURIComponent(path.split("/")[3] || ""); } catch { return ""; } };

export function makeAdapter({ store = new ResponseStore() } = {}) {
  const get = (req, res, { json, path, url }) => {
    const id = idOf(path);
    if (url.searchParams.get("stream") === "true") throw bad("streaming a stored response again is not supported by pooled serve", "stream");
    const e = store.get(id);
    if (!e) throw notFound(id);
    json(res, 200, e.response);
  };
  const del = (req, res, { json, path }) => {
    const id = idOf(path);
    if (!store.delete(id)) throw notFound(id);
    json(res, 200, { id, object: "response.deleted", deleted: true });
  };
  const inputItems = (req, res, { json, path, url }) => {
    const id = idOf(path);
    const e = store.get(id);
    if (!e) throw notFound(id);
    const q = url.searchParams;
    const order = q.get("order") || "desc";
    if (order !== "asc" && order !== "desc") throw bad("order must be asc or desc", "order");
    const limit = q.get("limit") == null ? 20 : Number(q.get("limit"));
    if (!Number.isInteger(limit) || limit < 1 || limit > 100) throw bad("limit must be an integer from 1 to 100", "limit");
    let list = order === "asc" ? e.inputItems : [...e.inputItems].reverse();
    const after = q.get("after");
    if (after) { const k = list.findIndex((it) => it.id === after); list = k >= 0 ? list.slice(k + 1) : []; }
    const data = list.slice(0, limit);
    json(res, 200, { object: "list", data, first_id: data[0]?.id ?? null, last_id: data[data.length - 1]?.id ?? null, has_more: list.length > limit });
  };
  const notServed = (req, res, { path }) => { throw new ApiError("notfound", `${req.method} ${path} is not available in pooled serve`); };
  return {
    api: "responses",
    label: "responses",
    store,
    routes: [
      { method: "POST", path: "/v1/responses" },
      { method: "POST", path: /^\/v1\/responses\/(input_tokens|compact)$/, handler: notServed },
      { method: "GET", path: /^\/v1\/responses\/[^/]+$/, handler: get },
      { method: "DELETE", path: /^\/v1\/responses\/[^/]+$/, handler: del },
      { method: "GET", path: /^\/v1\/responses\/[^/]+\/input_items$/, handler: inputItems },
    ],
    parse: (body, headers, { log } = {}) => parseResponses(body, headers, { log, store }),
    // call i's call_id; its item id (fc_ / ctc_) is minted on first use and kept, so stream, final
    // and store agree
    idFor: () => () => ids.call(),
    encoder: (req, sse, meta) => new ResponsesEncoder(req, sse, meta),
    final: (a, req) => responsesFinal(a, req),
    after: (a, req) => storeIt(store, a, req),
    error: responsesError,
    streamError: (e) => sseComment(cleanText(e.message, 300)),
  };
}

export const adapter = makeAdapter();
