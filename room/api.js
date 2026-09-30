// API clients on the host: `pooled serve` (cli/) joins the room as an ask-only guest and sends
// whole OpenAI / Anthropic style conversations as `ai-ask {api: 1, rid, system, messages, params}`.
// This module is the DOM-free part of answering one: validating the ask, rendering it to ids with
// the chat's own template, stop strings, the think block, and the exact-id cache that keeps a
// client's multi-turn conversation cheap. room.js wires it to the lock, the queue and the screens.
// docs/design/serve.md section 5; docs/protocol.md "API clients".
//
// An API request never reads or writes the room's chat conversation (ai.conv): the client owns its
// history and resends all of it every time, so the host renders it and runs roomGenerate on the
// ids, exactly as Code mode does.
import { buildIds, specials, renderApi, headerTail } from "./conversation.js";
import { makeSampler } from "./sampling.js";
import { CallStream } from "../harness/tools.js";
import { compileSchema, SchemaError, SCHEMA_CAPS, schemaWork } from "../harness/jsonschema.js";
import { constrainedSampler, tokenTexts } from "../harness/model-common.js";
import { hash64, grammarNodeCount } from "../harness/constrain.js";

export const API_LIMITS = {
  messages: 200,        // per request
  chars: 400000,        // total text in system + messages
  stops: 4, stopLen: 64,
  rid: 32,              // request id, bridge-chosen
  client: 40,           // the attribution label the room shows
  shown: 2000,          // the question as the screens show it
  maxTokens: 65536,
  reserve: 32,          // a prompt must leave this many tokens of context (room/models.js MIN_ROOM)
};

const clean = (s, n) => String(s ?? "").replace(/[\u0000-\u001f\u007f<>"'`&]/g, "").trim().slice(0, n);

// -> { req } (normalized) or { err, code: "bad" }. Everything that comes off the wire is checked here;
// the bridge validates the HTTP request more precisely, this is the host's own guard.
export function validateApiAsk(d, opts = {}) {
  const bad = (err) => ({ err, code: "bad" });
  if (!d || typeof d !== "object") return bad("empty request");
  if (d.api === 2) {
    // nothing a request holds may throw past here (the caller has no catch): a schema that breaks
    // the compiler in a way it does not name is still just a bad request
    try { return validateApiAskV2(d, opts); } catch (e) { return bad(`the request could not be checked: ${String(e?.message || e).slice(0, 200)}`); }
  }
  const rid = typeof d.rid === "string" ? d.rid : "";
  if (!/^[A-Za-z0-9_-]{1,32}$/.test(rid)) return bad("bad request id");
  const system = d.system == null ? "" : d.system;
  if (typeof system !== "string") return bad("system must be text");
  if (!Array.isArray(d.messages) || !d.messages.length) return bad("messages must be a non-empty list");
  if (d.messages.length > API_LIMITS.messages) return bad(`at most ${API_LIMITS.messages} messages`);
  let chars = system.length;
  const messages = [];
  for (const m of d.messages) {
    if (!m || (m.role !== "user" && m.role !== "assistant") || typeof m.text !== "string") return bad("each message needs role user|assistant and text");
    chars += m.text.length;
    messages.push({ role: m.role, text: m.text });
  }
  if (chars > API_LIMITS.chars) return bad(`the request is ${chars} characters; at most ${API_LIMITS.chars}`);
  if (messages[messages.length - 1].role !== "user") return bad("the last message must be from the user");
  const p = d.params && typeof d.params === "object" ? d.params : {};
  const maxTokens = p.maxTokens == null ? 1024 : Math.floor(+p.maxTokens);
  if (!(maxTokens >= 1 && maxTokens <= API_LIMITS.maxTokens)) return bad("maxTokens out of range");
  let temperature = null, topK = null;
  if (p.temperature != null) { temperature = +p.temperature; if (!(temperature >= 0 && temperature <= 2)) return bad("temperature out of range"); }
  if (p.topK != null) { topK = Math.floor(+p.topK); if (!(topK >= 1)) return bad("topK out of range"); }
  let thinkBudget = null;
  if (p.thinkBudget != null) { thinkBudget = Math.floor(+p.thinkBudget); if (!(thinkBudget >= 1 && thinkBudget <= API_LIMITS.maxTokens)) return bad("thinkBudget out of range"); }
  const stop = p.stop == null ? [] : p.stop;
  if (!Array.isArray(stop) || stop.length > API_LIMITS.stops || stop.some((s) => typeof s !== "string" || !s || s.length > API_LIMITS.stopLen))
    return bad(`stop: at most ${API_LIMITS.stops} non-empty strings of at most ${API_LIMITS.stopLen} characters`);
  return { req: { rid, system, messages, params: { maxTokens, temperature, topK, stop, thinking: !!p.thinking, thinkBudget, client: clean(p.client, API_LIMITS.client) || "API" } } };
}

// text -> the exact ids that were sampled for it, for the last few answers (host memory only). A
// client's next turn resends the answer as text; re-tokenizing can split it differently from what the
// caches hold, which would cost a full re-prefill every turn. A miss is only slower, never wrong.
export class AnswerCache {
  constructor(n = 8) { this.n = n; this.m = new Map(); }
  put(text, ids) {
    if (!text || !ids?.length) return;
    this.m.delete(text); this.m.set(text, ids.slice());
    while (this.m.size > this.n) this.m.delete(this.m.keys().next().value);
  }
  get(text) {
    const ids = this.m.get(text);
    if (ids) { this.m.delete(text); this.m.set(text, ids); }
    return ids || null;
  }
}

// consecutive messages of one role are merged (blank line between); assistant text is swapped for
// the exact sampled ids when the cache knows it byte for byte
export function apiTurns(tok, messages, cache) {
  const merged = [];
  for (const m of messages) {
    const last = merged[merged.length - 1];
    if (last && last.role === m.role) last.text += "\n\n" + m.text;
    else merged.push({ role: m.role, text: m.text });
  }
  let exact = 0;
  const turns = merged.map((m) => {
    if (m.role === "user") return { role: "user", text: m.text };
    const hit = cache?.get(m.text);
    if (hit) exact++;
    return { role: "assistant", ids: hit || tok.encode(m.text) };
  });
  return { turns, exact };
}

// Stop strings, streaming: emit text as it comes, except the longest tail that could still be the
// start of a stop string; on a full match emit what came before it and report the match.
export class StopMatcher {
  constructor(stops = []) { this.stops = stops.filter(Boolean); this.buf = ""; this.hit = null; this.max = Math.max(0, ...this.stops.map((s) => s.length)); }
  push(text) {
    if (this.hit) return "";
    if (!this.stops.length) return text;
    this.buf += text;
    let at = -1, which = null;
    for (const s of this.stops) {
      const i = this.buf.indexOf(s);
      if (i >= 0 && (at < 0 || i < at)) { at = i; which = s; }
    }
    if (at >= 0) { const out = this.buf.slice(0, at); this.buf = ""; this.hit = which; return out; }
    let hold = 0;
    for (let L = Math.min(this.buf.length, this.max - 1); L > 0; L--) {
      const tail = this.buf.slice(-L);
      if (this.stops.some((s) => s.startsWith(tail))) { hold = L; break; }
    }
    const out = this.buf.slice(0, this.buf.length - hold);
    this.buf = this.buf.slice(this.buf.length - hold);
    return out;
  }
  flush() { const out = this.hit ? "" : this.buf; this.buf = ""; return out; }
}

// The think block, streaming: with thinking on, Qwen3 answers "<think>\n…\n</think>\n\n answer".
// push(piece) -> [{ th, text }]: th true for the reasoning, false for the answer; the tags and the
// whitespace around them are dropped. With thinking off everything is answer.
// inPrompt: the prompt already opened the block (Qwen3.5+ templates end the generation header with
// "<think>\n"), so the answer starts inside it. callCloses: a "<tool_call>" inside the block ends it
// (XML-style models open calls there); the tag stays in the answer.
const OPEN = "<think>", CLOSE = "</think>", CALL = "<tool_call>";
export class ThinkSplit {
  constructor(on, { inPrompt = false, callCloses = false } = {}) {
    this.state = !on ? "answer" : inPrompt ? "think" : "start"; this.buf = ""; this.fresh = true; this.callCloses = callCloses;
    this.ended = false;   // the block closed (by </think> or a call)
  }
  push(piece) {
    const out = [];
    const emit = (th, text) => { if (text) out.push({ th, text }); };
    this.buf += piece;
    for (;;) {
      if (this.state === "answer") { emit(false, this.buf); this.buf = ""; return out; }
      if (this.state === "start") {
        const b = this.buf.replace(/^\s+/, "");
        if (b.startsWith(OPEN)) { this.state = "think"; this.buf = b.slice(OPEN.length).replace(/^\s+/, ""); this.fresh = !this.buf; continue; }
        if (OPEN.startsWith(b)) return out;          // still could be the tag (or only whitespace so far)
        this.state = "answer"; continue;              // no think block: all answer
      }
      if (this.state === "think") {
        if (this.fresh) { this.buf = this.buf.replace(/^\s+/, ""); if (!this.buf) return out; this.fresh = false; }
        const i = this.buf.indexOf(CLOSE), c = this.callCloses ? this.buf.indexOf(CALL) : -1;
        if (c >= 0 && (i < 0 || c < i)) { emit(true, this.buf.slice(0, c).replace(/\s+$/, "")); this.buf = this.buf.slice(c); this.state = "answer"; this.ended = true; continue; }
        if (i >= 0) { emit(true, this.buf.slice(0, i).replace(/\s+$/, "")); this.buf = this.buf.slice(i + CLOSE.length); this.state = "gap"; this.ended = true; continue; }
        let hold = 0;
        for (const tag of this.callCloses ? [CLOSE, CALL] : [CLOSE]) for (let L = Math.min(this.buf.length, tag.length - 1); L > hold; L--) if (tag.startsWith(this.buf.slice(-L))) { hold = L; break; }
        // trailing whitespace is held too: the think text ends right before "</think>"
        const keep = this.buf.length - hold, body = this.buf.slice(0, keep), ws = /\s*$/.exec(body)[0].length;
        emit(true, body.slice(0, body.length - ws));
        this.buf = this.buf.slice(keep - ws);
        return out;
      }
      if (this.state === "gap") {
        const b = this.buf.replace(/^\s+/, "");
        if (!b) { this.buf = ""; return out; }
        this.buf = b; this.state = "answer"; continue;
      }
    }
  }
  flush() {
    const out = [];
    if (this.state === "think") { const t = this.buf.replace(/\s+$/, ""); if (t) out.push({ th: true, text: t }); }
    else if (this.state === "start" || this.state === "answer") { if (this.buf) out.push({ th: false, text: this.buf }); }
    this.buf = "";
    return out;
  }
}

// Token ids -> text as they are sampled. One character can span tokens (an emoji, most non-Latin
// scripts): decoding each id on its own gives U+FFFD for each half. Ids are held until their text
// no longer ends in a partial character (at most 16, after which the text goes out as it is).
// push(id) -> the text that is complete now ("" while holding); flush() -> whatever is held.
export function pieceDecoder(tok) {
  let held = [];
  return {
    push(id) {
      held.push(id);
      const s = tok.decode(held);
      if (s.endsWith("\uFFFD") && held.length < 16) return "";
      held = [];
      return s;
    },
    flush() { const s = held.length ? tok.decode(held) : ""; held = []; return s; },
  };
}

// the sampler an API ask gets: temperature 0 greedy, > 0 top-k at that temperature (top-k 40 unless
// asked); no temperature -> the room's preset (fallback), or creative's 0.8 when only top_k was given
export function apiSampler(params, fallback) {
  if (params.temperature == null && params.topK == null) return fallback;
  return makeSampler({ temp: params.temperature ?? 0.8, topK: params.topK ?? 40 });
}

// Render an ask to ids and check the context. -> { ids, thinking, exact } or { err, code: "ctx", n, max }
export function apiPrompt(tok, req, ctxMax, cache) {
  const S = specials(tok);
  const thinking = !!req.params.thinking && S.think !== undefined;
  const { turns, exact } = apiTurns(tok, req.messages, cache);
  const ids = buildIds(tok, { system: req.system, turns, thinking });
  const max = ctxMax - API_LIMITS.reserve;
  if (ids.length > max) return { err: `prompt is too long: ${ids.length} tokens > ${max} maximum`, code: "ctx", n: ids.length, max };
  return { ids, thinking, exact, S };
}

// Answer one ask: generate(ids, opts) is roomGenerate. send(msg) goes to the asker only (it always
// gets the whole stream, whatever the room's visibility); onPiece(raw) gets the raw text for the
// screens. signal: the asker's stop. -> { reason, stopSeq, usage, text, think, reused, stats, err? }
//   reason: "stop" (end token) | "stop_seq" | "max" | "ctx" (context full) | "abort" | "error"
export async function apiRun({ tok, req, prompt, generate, send, onPiece = () => {}, cache, fallback, ctxMax, signal }) {
  const { ids, thinking, S } = prompt;
  const rid = req.rid;
  const split = new ThinkSplit(thinking), stopper = new StopMatcher(req.params.stop);
  let ac = new AbortController();
  const onAbort = () => ac.abort();
  signal?.addEventListener?.("abort", onAbort);
  const answerIds = [];
  const pieces = pieceDecoder(tok);
  let text = "", think = "", count = 0;
  const out = (parts, d = 0) => {
    for (const p of parts) {
      if (p.th) { think += p.text; send({ t: "ai-token", rid, text: p.text, d, th: 1 }); continue; }
      const e = stopper.push(p.text);
      if (e) { text += e; send({ t: "ai-token", rid, text: e, d }); }
      if (stopper.hit) { ac.abort(); return; }
    }
  };
  // thinking budget (Anthropic thinking.budget_tokens): once the reasoning has used it, this pass is
  // stopped and a second one continues from the same ids with the think block closed, so the rest
  // of max_tokens goes to the answer
  const budget = thinking && S.thinkEnd !== undefined && req.params.thinkBudget ? req.params.thinkBudget : 0;
  let overBudget = false;
  const stopIds = new Set([S.imEnd, S.eot].filter((x) => x !== undefined));
  const sample = apiSampler(req.params, fallback);
  const onToken = (id, drafted) => {
    if (stopper.hit || overBudget) return;
    count++;
    answerIds.push(id);
    const piece = pieces.push(id);
    if (piece) { onPiece(piece, drafted); out(split.push(piece), drafted || 0); }
    if (budget && count >= budget && count < req.params.maxTokens && split.state !== "answer" && split.state !== "gap") { overBudget = true; ac.abort(); }
  };
  let r = null, err = null;
  try {
    r = await generate(ids, { stop: stopIds, maxNew: Math.max(1, Math.min(req.params.maxTokens, ctxMax - ids.length)), sample, signal: ac.signal, onToken });
    if (overBudget && r.reason === "abort" && !signal?.aborted) {
      const close = [...tok.encode("\n"), S.thinkEnd, ...tok.encode("\n\n")];
      const ids2 = [...ids, ...answerIds, ...close];
      const held = pieces.flush();
      if (held) { onPiece(held, 0); out(split.push(held)); }
      const closeText = tok.decode(close);
      onPiece(closeText, 0); out(split.push(closeText));
      const left = Math.min(req.params.maxTokens - count, ctxMax - ids2.length);
      if (left > 0) {
        overBudget = false;
        ac = new AbortController();
        if (signal?.aborted) ac.abort();
        const r2 = await generate(ids2, { stop: stopIds, maxNew: left, sample, signal: ac.signal, onToken });
        r = { ...r2, reused: r.reused || 0 };
      } else r = { ...r, reason: ctxMax - ids2.length <= 0 ? "ctx" : "max" };
    }
  } catch (e) { err = e; }
  signal?.removeEventListener?.("abort", onAbort);
  if (!stopper.hit) {
    const last = pieces.flush();
    if (last) { onPiece(last, 0); out(split.push(last)); }
  }
  if (!stopper.hit) {
    out(split.flush());
    const rest = stopper.flush();
    if (rest) { text += rest; send({ t: "ai-token", rid, text: rest, d: 0 }); }
  }
  let reason;
  if (err) reason = "error";
  else if (stopper.hit) reason = "stop_seq";
  else if (r.reason === "abort") reason = "abort";
  else reason = r.reason === "ctx" ? "ctx" : r.reason === "max" ? "max" : "stop";
  // exact-id reuse for the client's next turn: only when the ids are exactly the text it will send
  // back (no think block, nothing cut off by a stop string, nothing held back)
  if (!thinking && (reason === "stop" || reason === "max")) cache?.put(text, answerIds);
  return { reason, stopSeq: stopper.hit, usage: { in: ids.length, out: count }, text, think,
    reused: r?.reused || 0, stats: r?.stats || "", err: err ? String(err.message || err) : null };
}

// The meta a device keeps for a peer's hello. API clients connect to the host only, so on the host
// meta.api marks an API client (kept to a fixed, cleaned shape), while on a guest it can only be
// the host saying it serves them: dropped there, or the guest would show the host as an API client.
export function helloMeta(meta, isHost) {
  if (!meta?.api) return meta;
  if (isHost) return { api: 1, webgpu: false, ua: "API", client: String(meta.client ?? "").replace(/[\u0000-\u001f\u007f<>"'`&]/g, "").slice(0, 40) };
  const { api, ctx, ...rest } = meta;   // (ctx: the context size the host tells API clients, v2)
  return rest;
}

// The room's settings after the host changes an answer style: the style fields come from the
// controls, everything else (apiAllow and any later setting) is kept.
export function withStyle(settings, style) {
  return { ...settings, persona: style.persona, sampling: style.sampling, thinking: !!style.thinking, length: style.length };
}

// ============================================================================================
// v2 asks: tools, tool calls, structured output, reasoning in history (docs/design/serve.md
// sections 4-5; docs/protocol.md "API clients"). `ai-ask {api: 2, rid, system, messages, tools,
// params}`; the host answers with ai-token (content and reasoning), ai-call (tool calls, streamed)
// and ai-gendone {calls, open, usage.think}.
// ============================================================================================
export const API2_LIMITS = {
  // tools: agents send every tool they have (Claude Code with MCP servers: hundreds); the real cost is
  // bounded by chars, the ask's size on the wire and the grammar's node cap
  messages: 1000, chars: 1500000, tools: 1024, schemaChars: 32000, calls: 64, maxCalls: 128,
  name: /^[A-Za-z0-9_.:-]{1,128}$/, xmlParam: /^[^<>\n\r]{1,128}$/,
};
export const EFFORTS = ["low", "medium", "high", "xhigh", "max"];

// -> { req } or { err, code: "bad" }. profile: the loaded model's template profile (room/conversation.js
// templateProfile); XML-style models need parameter names that fit in <parameter=NAME>.
function validateApiAskV2(d, { profile = null } = {}) {
  const bad = (err) => ({ err, code: "bad" });
  const rid = typeof d.rid === "string" ? d.rid : "";
  if (!/^[A-Za-z0-9_-]{1,32}$/.test(rid)) return bad("bad request id");
  const system = d.system == null ? "" : d.system;
  if (typeof system !== "string") return bad("system must be text");
  if (!Array.isArray(d.messages) || !d.messages.length) return bad("messages must be a non-empty list");
  if (d.messages.length > API2_LIMITS.messages) return bad(`at most ${API2_LIMITS.messages} messages`);
  let chars = system.length;
  const work = schemaWork();   // one budget for every schema in the request
  // tools
  let tools = null;
  if (d.tools != null) {
    if (!Array.isArray(d.tools)) return bad("tools must be a list");
    if (d.tools.length > API2_LIMITS.tools) return bad(`at most ${API2_LIMITS.tools} tools`);
    const seen = new Set();
    tools = [];
    for (const t of d.tools) {
      if (!t || typeof t.name !== "string" || !API2_LIMITS.name.test(t.name)) return bad(`tool names must match ${API2_LIMITS.name.source}`);
      if (seen.has(t.name)) return bad(`tool ${t.name} is declared twice`);
      seen.add(t.name);
      if (t.description != null && typeof t.description !== "string") return bad(`tool ${t.name}: description must be text`);
      const params = t.parameters == null ? { type: "object", properties: {} } : t.parameters;
      if (!params || typeof params !== "object" || Array.isArray(params)) return bad(`tool ${t.name}: parameters must be a JSON schema object`);
      const n = JSON.stringify(params).length;
      if (n > API2_LIMITS.schemaChars) return bad(`tool ${t.name}: its schema is ${n} characters; at most ${API2_LIMITS.schemaChars}`);
      if (profile?.style === "xml" && params.properties && typeof params.properties === "object") {
        for (const k of Object.keys(params.properties)) if (!API2_LIMITS.xmlParam.test(k)) return bad(`tool ${t.name}: parameter name ${JSON.stringify(k.slice(0, 40))} cannot be written by this model (no <, >, line breaks; 1 to 128 characters)`);
      }
      try { compileSchema(params, { work }); } catch (e) { if (e instanceof SchemaError) return bad(`tool ${t.name}: ${e.message}`); throw e; }
      chars += n + t.name.length + (t.description || "").length;
      tools.push({ name: t.name, description: t.description || "", parameters: params });
    }
    if (!tools.length) tools = null;
  }
  const names = new Set((tools || []).map((t) => t.name));
  // messages
  const messages = [];
  for (const m of d.messages) {
    if (!m || typeof m !== "object" || typeof m.text !== "string") return bad("each message needs a role and text");
    chars += m.text.length;
    if (m.role === "user") messages.push(m.aside ? { role: "user", text: m.text, aside: true } : { role: "user", text: m.text });
    else if (m.role === "tool") messages.push({ role: "tool", text: m.text });
    else if (m.role === "assistant") {
      const a = { role: "assistant", text: m.text };
      if (m.reasoning != null) { if (typeof m.reasoning !== "string") return bad("reasoning must be text"); a.reasoning = m.reasoning; chars += m.reasoning.length; }
      if (m.calls != null) {
        if (!Array.isArray(m.calls) || m.calls.length > API2_LIMITS.calls) return bad(`calls: a list of at most ${API2_LIMITS.calls}`);
        a.calls = [];
        for (const c of m.calls) {
          if (!c || typeof c.name !== "string" || !c.name || c.name.length > 128) return bad("each call needs a name");
          const args = c.args == null ? {} : c.args;
          if (typeof args !== "object" || Array.isArray(args)) return bad("call arguments must be an object");
          chars += c.name.length + JSON.stringify(args).length;
          a.calls.push({ name: c.name, args });
        }
      }
      messages.push(a);
    } else return bad("each message needs role user, assistant or tool");
  }
  if (chars > API2_LIMITS.chars) return bad(`the request is ${chars} characters; at most ${API2_LIMITS.chars}`);
  const last = messages[messages.length - 1];
  if (last.role !== "user" && last.role !== "tool") return bad("the last message must be from the user or a tool result");
  const usesTools = !!tools || messages.some((m) => m.role === "tool" || m.calls?.length);
  if (usesTools && profile && !profile.tools) return bad("the room's model has no tool-call format");
  // params: v1's, plus the tool and format ones
  const p = d.params && typeof d.params === "object" ? d.params : {};
  const v1 = validateApiAsk({ api: 1, rid, system: "", messages: [{ role: "user", text: "" }], params: { ...p, stop: p.stop } });
  if (v1.err) return v1;
  const params = { ...v1.req.params };
  let toolChoice = p.toolChoice == null ? "auto" : p.toolChoice;
  if (typeof toolChoice === "object") {
    if (!toolChoice || typeof toolChoice.name !== "string") return bad("toolChoice must be auto, none, required or {name}");
    if (!names.has(toolChoice.name)) return bad(`toolChoice names ${toolChoice.name}, which is not a declared tool`);
    toolChoice = { name: toolChoice.name };
  } else if (!["auto", "none", "required"].includes(toolChoice)) return bad("toolChoice must be auto, none, required or {name}");
  if ((toolChoice === "required" || typeof toolChoice === "object") && !tools) return bad("toolChoice needs tools");
  let allowed = null;
  if (p.allowed != null) {
    if (!Array.isArray(p.allowed) || p.allowed.some((n) => typeof n !== "string" || !names.has(n))) return bad("allowed must list declared tool names");
    allowed = [...new Set(p.allowed)];
  }
  const parallel = p.parallel !== false;
  let maxCalls = null;
  if (p.maxCalls != null) { maxCalls = Math.floor(+p.maxCalls); if (!(maxCalls >= 1 && maxCalls <= API2_LIMITS.maxCalls)) return bad("maxCalls out of range"); }
  let format = null;
  if (p.format != null) {
    const f = p.format;
    if (f?.type === "json") format = { type: "json" };
    else if (f?.type === "schema" && f.schema && typeof f.schema === "object" && !Array.isArray(f.schema)) {
      const n = JSON.stringify(f.schema).length;
      if (n > API2_LIMITS.schemaChars) return bad(`the response format schema is ${n} characters; at most ${API2_LIMITS.schemaChars}`);
      try { compileSchema(f.schema, { work }); } catch (e) { if (e instanceof SchemaError) return bad(`response format: ${e.message}`); throw e; }
      format = { type: "schema", schema: f.schema, ...(typeof f.name === "string" ? { name: f.name.slice(0, 64) } : {}) };
    } else return bad("format must be {type: json} or {type: schema, schema}");
  }
  let effort = null;
  if (p.effort != null) { if (!EFFORTS.includes(p.effort)) return bad(`effort must be one of ${EFFORTS.join(", ")}`); effort = p.effort; }
  // the whole grammar, as the answer will build it: each schema fits on its own, but together they
  // must too (GrammarConstraint refuses more than SCHEMA_CAPS.nodes)
  if (tools || format) {
    let n;
    try { n = grammarNodeCount(tools || [], { style: profile?.style ?? "json", format }); }
    catch (e) { if (e instanceof SchemaError) return bad(`tools: ${e.message}`); throw e; }
    if (n > SCHEMA_CAPS.nodes) return bad(`the tool schemas are too large together (${n} grammar nodes; at most ${SCHEMA_CAPS.nodes})`);
  }
  return { req: { api: 2, rid, system, messages, tools, params: { ...params, toolChoice, allowed, parallel, maxCalls, format, effort } } };
}

// tok.encode in front of an LRU by bytes: an agent's every step resends the whole history, and the
// host tab would otherwise run JS BPE over all of it each time. Keyed by the text itself.
export class EncodeCache {
  constructor(budget = 32 << 20, min = 64) { this.budget = budget; this.min = min; this.bytes = 0; this.m = new Map(); }
  encode(tok, text) {
    if (text.length < this.min) return tok.encode(text);
    if (this.tok !== tok) { this.m.clear(); this.bytes = 0; this.tok = tok; }
    const hit = this.m.get(text);
    if (hit) { this.m.delete(text); this.m.set(text, hit); return hit; }
    const ids = tok.encode(text);
    const cost = text.length * 2 + ids.length * 4;
    if (cost > this.budget / 4) return ids;
    this.m.set(text, ids); this.bytes += cost;
    while (this.bytes > this.budget && this.m.size) { const [k, v] = this.m.entries().next().value; this.m.delete(k); this.bytes -= k.length * 2 + v.length * 4; }
    return ids;
  }
  clear() { this.m.clear(); this.bytes = 0; }
}

// The canonical form of an answer, as the client will send it back: its content (trimmed) and each
// call's name and arguments (keys sorted); call ids and reasoning are left out.
const sortKeys = (v) => (Array.isArray(v) ? v.map(sortKeys) : v && typeof v === "object" ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, sortKeys(v[k])])) : v);
export function canonAnswer(text, calls) {
  return String(text || "").trim() + "\u0000" + (calls || []).map((c) => c.name + "\u0001" + JSON.stringify(sortKeys(typeof c.args === "string" ? safeParse(c.args) : c.args ?? {}))).join("\u0002");
}
const safeParse = (s) => { try { return JSON.parse(s); } catch { return {}; } };
// hash of the history before each message: h[j] covers system, the tool set and messages[0..j)
export function historyHashes(req) {
  const h = [hash64("s\u0000" + (req.system || "") + "\u0000" + JSON.stringify(req.tools || null))];
  for (const m of req.messages) {
    const c = m.role === "assistant" ? "a\u0000" + canonAnswer(m.text, m.calls) : (m.role === "tool" ? "t" : m.aside ? "x" : "u") + "\u0000" + m.text;
    h.push(hash64(h[h.length - 1] + "\u0003" + c));
  }
  return h;
}

// The exact ids each recent API answer was sampled as, so a client's next step replays them instead
// of re-tokenizing its text (which would change the prompt and cost a full re-prefill). Keyed by
// (model, hash of the history before the answer, canonical answer): the same answer text in two
// conversations never crosses over. Value: { ids (after "assistant\n", the header's think part
// included), thinkEnd (where the content starts), reasoningHash, reasoned }. A client that sends
// reasoning back hits only when it is the same reasoning. Bounded by entries and total ids;
// cleared when the room loads another model.
export class TurnCache {
  constructor({ entries = 512, ids = 2000000 } = {}) { this.max = entries; this.maxIds = ids; this.n = 0; this.m = new Map(); }
  put(key, v) {
    if (!v?.ids?.length) return;
    const old = this.m.get(key);
    if (old) { this.n -= old.ids.length; this.m.delete(key); }
    const val = { ...v, ids: Int32Array.from(v.ids) };
    this.m.set(key, val); this.n += val.ids.length;
    while ((this.m.size > this.max || this.n > this.maxIds) && this.m.size) { const [k, x] = this.m.entries().next().value; this.m.delete(k); this.n -= x.ids.length; }
  }
  get(key, reasoning = "") {
    const v = this.m.get(key);
    if (!v) return null;
    const r = String(reasoning || "").trim();
    if (r && v.reasoningHash !== hash64(r)) return null;
    this.m.delete(key); this.m.set(key, v);
    return v;
  }
  clear() { this.m.clear(); this.n = 0; }
  get size() { return this.m.size; }
}

// Render a v2 ask to ids and check the context.
// turnIds(j, m) (optional): the exact ids of assistant message j from the caller's own store (Code
// mode keeps them on its turns), in place of the cache lookup.
// -> { ids, thinking, exact, S, profile, systemLen, histKey } or { err, code: "ctx", n, max }
export function apiPrompt2(tok, req, ctxMax, { profile, cache = null, encoder = null, model = "", turnIds: own = null }) {
  const S = specials(tok);
  const thinking = !!req.params.thinking && S.think !== undefined;
  const h = historyHashes(req);
  const turnIds = own || (cache ? (j, m) => cache.get(model + "|" + h[j] + "|" + canonAnswer(m.text, m.calls), m.reasoning) : () => null);
  const encode = encoder ? (s) => encoder.encode(tok, s) : (s) => tok.encode(s);
  const { ids, systemLen, exact } = renderApi(tok, req, profile, { encode, turnIds, thinking });
  const max = ctxMax - API_LIMITS.reserve;
  if (ids.length > max) return { err: `prompt is too long: ${ids.length} tokens > ${max} maximum`, code: "ctx", n: ids.length, max };
  return { ids, thinking, exact, S, profile, systemLen, histKey: model + "|" + h[h.length - 1] + "|" };
}

export const GARBAGE_MSG = "the room's output did not follow the tool-call format (the engine looks unhealthy)";

// Answer one v2 ask. As apiRun, plus: the tool-call grammar on the sampler (harness/constrain.js),
// call parsing and streaming (harness/tools.js CallStream) with ai-call messages, stop strings on
// content only, a default think budget in the forcing modes, and the exact-id cache.
//   tt / vocabSize: token texts and vocabulary size for the grammar (harness/model-common.js tokenTexts)
//   garbage: the grammar's garbage rule ("count", or "mass" for Code mode: harness/model-common.js)
// -> { reason, stopSeq, usage: {in, out, think}, text, think, calls: [{name, args}], open, reused, stats, err }
// plus, for in-process callers (Code mode, harness/core-model.js; the wire never carries them):
//   garbage (reason is then "error" with GARBAGE_MSG), forced / forcedFree (the grammar's forced
//   positions: all of them / the model's own choices only), openArgs (the open call's arguments so
//   far), raw (the decoded answer), gen: { prefilled, tps, tDecode } (from generate), and ids /
//   thinkEnd whenever the answer is exact-id reusable (what the cache would get).
export async function apiRun2({ tok, req, prompt, generate, send, onPiece = () => {}, cache, fallback, ctxMax, signal, tt = null, vocabSize = 0, log = () => {}, garbage: garbageRule = "count" }) {
  const { ids, thinking, S, profile } = prompt;
  const P = req.params, rid = req.rid;
  const tools = req.tools || [];
  const mode = P.toolChoice ?? "auto";
  const style = profile.style;
  const V = tok.vocab || {};
  const respId = V["<tool_response>"];
  const stopIds = new Set([S.imEnd, S.eot, respId].filter(Number.isInteger));
  const base = apiSampler(P, fallback);
  // the grammar: whenever there are tools (any mode) or a format
  const grammar = tools.length || P.format;
  if (grammar && !vocabSize) {
    if (!tok.__vocabSize) { let n = 0; for (const v of Object.values(V)) if (v >= n) n = v + 1; tok.__vocabSize = n; }
    vocabSize = tok.__vocabSize;
  }
  // building the grammar can still fail (the host checked its size, but a failure here must end this
  // answer as an error, never leave the room busy)
  let cs;
  try {
    cs = grammar
      ? constrainedSampler(base, tools, { tokenText: tt || tokenTexts(tok), vocabSize, style, stops: [...stopIds], thinking, thinkInPrompt: !!profile.thinkInPrompt,
        mode, allowed: P.allowed, maxCalls: P.maxCalls, parallel: P.parallel, format: P.format, garbage: garbageRule,
        tags: Object.fromEntries(["<tool_call>", "</tool_call>", "<think>", "</think>", "<tool_response>", "</tool_response>"].map((t) => [t, V[t]])) })
      : constrainedSampler(base, null, {});
  } catch (e) {
    return { reason: "error", stopSeq: null, usage: { in: ids.length, out: 0, think: 0 }, text: "", think: "", calls: [], open: null, reused: 0, stats: "",
      err: `could not build the tool-call grammar: ${String(e?.message || e).slice(0, 200)}`,
      garbage: false, forced: 0, forcedFree: 0, openArgs: null, raw: "", gen: null, ids: null, thinkEnd: 0 };
  }
  const C = cs.constraint;
  const forcing = !!C?.forcing;
  const names = typeof mode === "object" ? [mode.name] : P.allowed || tools.map((t) => t.name);
  const split = new ThinkSplit(thinking, { inPrompt: !!profile.thinkInPrompt, callCloses: style === "xml" && mode !== "none" && tools.length > 0 });
  const calls = new CallStream({ style, tools, allowed: mode === "none" ? [] : names, constrained: !!C });
  const stopper = new StopMatcher(P.stop);
  let ac = new AbortController();
  const onAbort = () => ac.abort();
  signal?.addEventListener?.("abort", onAbort);
  const seq = [];   // every id after the prompt, as the caches hold them (the budget's injected close included)
  const pieces = pieceDecoder(tok);
  // gv: the grammar's view of the answer so far (each token's C.tt: a tag token is one symbol)
  let raw = "", gv = "", text = "", think = "", count = 0, thinkCount = 0, garbage = false, mismatch = false;
  const onEvents = (evs, d) => {
    for (const e of evs) {
      if (stopper.hit) return;
      if (e.t === "text") {
        const out = stopper.push(e.text);
        if (out) { text += out; send({ t: "ai-token", rid, text: out, d }); }
        if (stopper.hit) { ac.abort(); return; }
      } else if (e.t === "call") send({ t: "ai-call", rid, i: e.i, name: e.name });
      else if (e.t === "args") send({ t: "ai-call", rid, i: e.i, a: e.a });
      else if (e.t === "end") { if (e.mismatch) { mismatch = true; log(`call ${e.i} (${calls.calls[e.i]?.name}): the streamed arguments differ from the parsed ones`); } send({ t: "ai-call", rid, i: e.i, end: 1 }); }
    }
  };
  const feed = (piece, d) => {
    raw += piece;
    for (const p of split.push(piece)) {
      if (p.th) { think += p.text; send({ t: "ai-token", rid, text: p.text, d, th: 1 }); continue; }
      onEvents(calls.push(p.text), d);
      if (stopper.hit) return;
    }
  };
  const budget = thinking && S.thinkEnd !== undefined ? (P.thinkBudget || (forcing ? Math.min(Math.floor(P.maxTokens / 2), 4096) : 0)) : 0;
  let overBudget = false;
  const inThink = () => thinking && !split.ended && split.state !== "answer" && split.state !== "gap";
  const onToken = (id, drafted) => {
    if (stopper.hit || overBudget || garbage) return;
    count++;
    if (inThink()) thinkCount++;
    seq.push(id);
    cs.keep(1);
    if (C) { gv += C.tt(id); cs.setText(gv); }
    const piece = pieces.push(id);
    if (piece) { onPiece(piece, drafted); feed(piece, drafted || 0); }
    if (cs.garbage) { garbage = true; ac.abort(); return; }
    if (budget && count >= budget && count < P.maxTokens && inThink()) { overBudget = true; ac.abort(); }
  };
  let r = null, err = null;
  try {
    r = await generate(ids, { stop: stopIds, maxNew: Math.max(1, Math.min(P.maxTokens, ctxMax - ids.length)), sample: cs.sample, signal: ac.signal, onToken });
    if (overBudget && r.reason === "abort" && !signal?.aborted) {
      // the reasoning used its budget: close the block and go on from the same ids (the grammar
      // reads the close like sampled text, so a forcing mode engages right after it)
      const close = [...tok.encode("\n"), S.thinkEnd, ...tok.encode("\n\n")];
      const held = pieces.flush();
      if (held) { onPiece(held, 0); feed(held, 0); }
      const closeText = tok.decode(close);
      onPiece(closeText, 0); feed(closeText, 0);
      if (C) { for (const x of close) gv += C.tt(x); cs.setText(gv); }
      seq.push(...close);
      const ids2 = [...ids, ...seq];
      const left = Math.min(P.maxTokens - count, ctxMax - ids2.length);
      if (left > 0) {
        overBudget = false;
        ac = new AbortController();
        if (signal?.aborted) ac.abort();
        const r2 = await generate(ids2, { stop: stopIds, maxNew: left, sample: cs.sample, signal: ac.signal, onToken });
        r = { ...r2, reused: r.reused || 0 };
      } else r = { ...r, reason: ctxMax - ids2.length <= 0 ? "ctx" : "max" };
    }
  } catch (e) { err = e; }
  signal?.removeEventListener?.("abort", onAbort);
  if (!stopper.hit && !garbage) {
    const last = pieces.flush();
    if (last) { onPiece(last, 0); feed(last, 0); }
  }
  if (!stopper.hit) {
    for (const p of split.flush()) {
      if (p.th) { think += p.text; send({ t: "ai-token", rid, text: p.text, d: 0, th: 1 }); }
      else onEvents(calls.push(p.text), 0);
    }
    if (!stopper.hit) onEvents(calls.end(), 0);
    const rest = stopper.flush();
    if (rest) { text += rest; send({ t: "ai-token", rid, text: rest, d: 0 }); }
  }
  let reason;
  if (err || garbage) reason = "error";
  else if (stopper.hit) reason = "stop_seq";
  else if (r.reason === "abort") reason = "abort";
  else reason = r.reason === "ctx" ? "ctx" : r.reason === "max" ? "max" : "stop";
  const done = calls.calls.map((c) => ({ name: c.name, args: c.args }));
  const open = !stopper.hit && calls.open ? { i: calls.open.i, name: calls.open.name } : null;
  if (C?.broken) log("the answer left the tool-call grammar (a bug): parsed as it came");
  // exact-id reuse: an answer that ended on an end token (or the grammar's end), nothing cut or held
  let exactIds = null, exactEnd = 0;
  if (reason === "stop" && !open && !mismatch && (text.trim() || done.length)) {
    const tail = headerTail(tok, S, profile, thinking);
    let thinkEnd = tail.length;
    if (thinking) {
      let k = seq.lastIndexOf(S.thinkEnd);
      if (k >= 0) { k++; while (k < seq.length && !tok.decode([seq[k]]).trim()) k++; }
      else if (!profile.thinkInPrompt && !think.trim()) k = 0;   // the model did not open a block at all
      else { k = seq.indexOf(V["<tool_call>"]); if (k < 0) k = seq.length; }   // closed by a call (XML) 
      thinkEnd = tail.length + k;
    }
    exactIds = [...tail, ...seq]; exactEnd = thinkEnd;
    cache?.put(prompt.histKey + canonAnswer(text, done), { ids: exactIds, thinkEnd, reasoned: !!think.trim(), reasoningHash: think.trim() ? hash64(think.trim()) : "" });
  }
  return { reason, stopSeq: stopper.hit, usage: { in: ids.length, out: count, think: thinkCount }, text, think, calls: done, open,
    reused: r?.reused || 0, stats: r?.stats || "", err: garbage ? GARBAGE_MSG : err ? String(err.message || err) : null,
    garbage, forced: cs.forced || 0, forcedFree: cs.forcedFree || 0, openArgs: open ? calls.open.args ?? "" : null, raw,
    gen: r ? { prefilled: r.prefilled ?? null, tps: r.tps || 0, tDecode: r.tDecode ?? null } : null, ids: exactIds, thinkEnd: exactEnd };
}
