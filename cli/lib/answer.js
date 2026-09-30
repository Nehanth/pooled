// One request's answer from the room, on the bridge (docs/design/serve.md 4.3): the room's
// messages for it (ai-genstart, ai-token, ai-call, ai-gendone) are checked and turned into calls on
// Encoders: a Collector that builds the whole Answer (for non-stream responses, logs and the
// Responses store) and, when streaming, the API's own encoder writing SSE as things arrive.
//
//   Answer = { id, created, model, think, text, calls: [{ id, name, args /* JSON string */ }],
//              open: { id, name, args } | null, reason, stopSeq, usage: { in, out, think }, reused }
//   Encoder = { start(promptTokens), think(text), text(text), callStart(i, id, name), callArgs(i, frag),
//               callEnd(i, argsJson), done(answer), error(e), keepAlive(ahead) }
//   reason: "stop" | "stop_seq" | "max" | "ctx" (as the room says; common.outcome maps it per API)
//
// The host is another person's browser tab: everything it sends is checked, never trusted to be
// well formed or to stop. A violation ends the request (500 "the room sent more than was asked for"
// or "... did not follow the protocol").
import { ApiError, cleanText } from "./common.js";

export const TOKEN_CHARS = 4096;           // one ai-token's text
export const ANSWER_CHARS = 8 << 20;       // an answer's text, its reasoning, and each call's arguments
export const TOKEN_SLACK = 16;             // messages allowed past max_tokens before the bridge ends it
export const MAX_CALLS = 64;               // calls accepted in one answer
const REASONS = new Set(["stop", "stop_seq", "max", "ctx", "abort", "error"]);
const count = (x) => Math.max(0, Math.floor(+x) || 0);
// a finished call's arguments: a JSON object (clients JSON.parse them)
const isArgs = (s) => { try { const v = JSON.parse(s); return !!v && typeof v === "object" && !Array.isArray(v); } catch { return false; } };

// Builds the Answer; the non-stream path renders it with adapter.final(answer, req)
export class Collector {
  constructor(meta = {}) {
    this.answer = { id: meta.id ?? null, created: meta.created ?? 0, model: meta.model ?? null, think: "", text: "", calls: [], open: null,
      reason: null, stopSeq: null, usage: { in: 0, out: 0, think: 0 }, reused: 0 };
    this.parts = [];   // per call index: { id, name, args, done }
  }
  start(promptTokens) { this.answer.usage.in = promptTokens; }
  think(t) { this.answer.think += t; }
  text(t) { this.answer.text += t; }
  callStart(i, id, name) { this.parts[i] = { id, name, args: "", done: false }; }
  callArgs(i, frag) { if (this.parts[i]) this.parts[i].args += frag; }
  callEnd(i, args) { const p = this.parts[i]; if (p) { p.done = true; if (typeof args === "string") p.args = args; } }
  done() {}
  error() {}
  keepAlive() {}
}

// The state of one ask on the bridge. feed(d) takes a room message for it and returns null (go on),
// { answer } (complete: finish the response) or { error: ApiError }. encoders: the Collector first,
// then the stream encoder if any. idFor(i) -> the id call i gets in this API.
export class Ask {
  constructor({ req, meta, v2 = false, encoders, idFor = (i) => `call_${i}`, log = () => {}, label = "" }) {
    this.req = req; this.meta = meta; this.v2 = v2; this.enc = encoders; this.idFor = idFor; this.log = log; this.label = label;
    this.collector = encoders[0];
    this.state = "asked";
    this.promptTokens = 0; this.tokens = 0; this.thinkChars = 0; this.textChars = 0;
    this.calls = [];   // [{ id, name, args, done }]
    // the tools a call may name: none under tool_choice none, the named one, else the allowed ones
    const tc = req.toolChoice;
    this.names = new Set(tc === "none" ? [] : tc && typeof tc === "object" ? [tc.name]
      : (req.tools || []).map((t) => t.name).filter((n) => !req.allowed || req.allowed.includes(n)));
  }
  each(f) { for (const e of this.enc) f(e); }
  // the room went past what was asked: stop there, as a finished answer cut at max_tokens (v1's rule)
  over() {
    this.log(`${this.label}: the room sent more than was asked for; ended it`);
    return this.finish({ reason: "max", usage: { in: this.promptTokens, out: Math.max(0, this.tokens - 1), think: 0 }, reused: 0, calls: null });
  }
  bad(why) { this.log(`${this.label}: ${why}`); return { error: new ApiError("server", `the room did not follow the protocol (${why})`) }; }
  feed(d) {
    switch (d.t) {
      case "ai-genstart":
        if (this.state === "streaming") return null;
        if (this.v2 && d.api !== 2) {
          this.log(`${this.label}: the host answered a v2 request as an older Pooled`);
          return { error: new ApiError("server", "the room's host changed to an older Pooled; retry") };
        }
        this.state = "streaming";
        this.promptTokens = count(d.promptTokens);
        this.each((e) => e.start(this.promptTokens));
        return null;
      case "ai-token": {
        if (this.state !== "streaming" || typeof d.text !== "string" || d.text.length > TOKEN_CHARS) return null;
        this.tokens++;
        const th = !!d.th;
        if (th) this.thinkChars += d.text.length; else this.textChars += d.text.length;
        if (this.tokens > this.req.maxTokens + TOKEN_SLACK || this.thinkChars > ANSWER_CHARS || this.textChars > ANSWER_CHARS) return this.over();
        if (th) this.each((e) => e.think(d.text)); else this.each((e) => e.text(d.text));
        return null;
      }
      case "ai-call": {
        if (this.state !== "streaming" || !this.v2) return null;
        const i = d.i;
        if (!Number.isInteger(i) || i < 0 || i >= MAX_CALLS) return this.bad("a call index out of range");
        if (d.name != null) {
          if (i !== this.calls.length || typeof d.name !== "string") return this.bad("calls out of order");
          if (!this.names.has(d.name)) return this.bad(`a call to ${cleanText(d.name, 60)}, which is not a tool this request allows`);
          const id = this.idFor(i);
          this.calls.push({ id, name: d.name, args: "", done: false });
          this.each((e) => e.callStart(i, id, d.name));
          return null;
        }
        const c = this.calls[i];
        if (!c || c.done) return this.bad("arguments for a call that is not open");
        if (d.a != null) {
          if (typeof d.a !== "string" || d.a.length > TOKEN_CHARS) return this.bad("bad call arguments");
          c.args += d.a;
          this.tokens++;
          if (this.tokens > this.req.maxTokens + TOKEN_SLACK || c.args.length > ANSWER_CHARS) return this.over();
          this.each((e) => e.callArgs(i, d.a));
          return null;
        }
        if (d.end) {
          if (!isArgs(c.args)) return this.bad(`call ${i}'s arguments are not a JSON object`);
          c.done = true; this.each((e) => e.callEnd(i, c.args));
        }
        return null;
      }
      case "ai-gendone": {
        const usage = { in: d.usage?.in != null ? count(d.usage.in) : this.promptTokens || 0, out: count(d.usage?.out), think: count(d.usage?.think) };
        const reason = REASONS.has(d.reason) ? d.reason : d.failed ? "error" : "stop";
        if (reason === "error") return { error: new ApiError("server", `generation failed in the room: ${cleanText(d.err, 300) || "unknown error"}`) };
        // the host pressed Stop (the only abort a live client sees): a cut-off answer must not read as a
        // finished one, so it ends as an error (an error event, or 503 / 529), never with stop / end_turn
        if (reason === "abort") return { error: new ApiError("unavailable", `the room's host stopped this answer after ${usage.out} tokens`) };
        const stopSeq = reason === "stop_seq" && this.req.stop.includes(d.stopSeq) ? d.stopSeq : null;
        return this.finish({ reason, stopSeq, usage, reused: Math.min(count(d.reused), usage.in), calls: this.v2 ? d.calls : null, open: this.v2 ? d.open : null });
      }
    }
    return null;
  }
  // gendone's calls are the truth for the final answer (the streamed fragments are checked against them)
  finish({ reason, stopSeq = null, usage, reused, calls, open = null }) {
    const a = this.collector.answer;
    a.reason = reason; a.stopSeq = stopSeq; a.usage = usage; a.reused = reused;
    let final = this.calls.filter((c) => c.done).map((c) => ({ id: c.id, name: c.name, args: c.args }));
    if (Array.isArray(calls)) {
      const got = [];
      for (let i = 0; i < Math.min(calls.length, MAX_CALLS); i++) {
        const c = calls[i];
        if (!c || typeof c.name !== "string" || typeof c.args !== "string" || c.args.length > ANSWER_CHARS || !this.names.has(c.name) || !isArgs(c.args)) return this.bad("bad calls in the answer");
        const s = this.calls[i];
        if (s && s.done && s.name === c.name && s.args !== c.args) this.log(`${this.label}: call ${i} (${c.name}): the streamed arguments differ from the final ones; the final ones count`);
        got.push({ id: s?.name === c.name ? s.id : this.idFor(i), name: c.name, args: c.args });
      }
      final = got;
    }
    // belt and braces: never more calls than allowed
    const cap = this.req.parallel === false ? 1 : this.req.maxCalls || MAX_CALLS;
    if (final.length > cap) final = final.slice(0, cap);
    a.calls = final;
    if (open && Number.isInteger(open.i) && this.calls[open.i] && !this.calls[open.i].done) { const c = this.calls[open.i]; a.open = { id: c.id, name: c.name, args: c.args }; }
    else a.open = null;
    this.state = "done";
    return { answer: a };
  }
}
