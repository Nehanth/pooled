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
import { buildIds, specials } from "./conversation.js";
import { makeSampler } from "./sampling.js";

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
export function validateApiAsk(d) {
  const bad = (err) => ({ err, code: "bad" });
  if (!d || typeof d !== "object") return bad("empty request");
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
const OPEN = "<think>", CLOSE = "</think>";
export class ThinkSplit {
  constructor(on) { this.state = on ? "start" : "answer"; this.buf = ""; this.fresh = true; }
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
        const i = this.buf.indexOf(CLOSE);
        if (i >= 0) { emit(true, this.buf.slice(0, i).replace(/\s+$/, "")); this.buf = this.buf.slice(i + CLOSE.length); this.state = "gap"; continue; }
        let hold = 0;
        for (let L = Math.min(this.buf.length, CLOSE.length - 1); L > 0; L--) if (CLOSE.startsWith(this.buf.slice(-L))) { hold = L; break; }
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
  const { api, ...rest } = meta;
  return rest;
}

// The room's settings after the host changes an answer style: the style fields come from the
// controls, everything else (apiAllow and any later setting) is kept.
export function withStyle(settings, style) {
  return { ...settings, persona: style.persona, sampling: style.sampling, thinking: !!style.thinking, length: style.length };
}
