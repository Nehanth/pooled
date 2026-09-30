// Pieces the agent's model adapters share (harness/engine-model.js over one engine,
// harness/room-model.js over the room): the tool-call constraint as a sampler wrapper, streaming
// decode with UTF-8 holdback, an async queue from callbacks to an async iterator, and the map
// from an assistant turn's text to the exact ids it was sampled as. DOM-free.
import { ToolCallConstraint } from "./constrain.js";

// The conversation no longer fits the context: both adapters throw it before asking the model, and
// the agent ends the request with reason "context" (harness/agent.js).
export class ContextFull extends Error {
  constructor(n, max) {
    super(`the conversation is ${n} tokens and the context is ${max}: start a new task (the files are kept)`);
    this.name = "ContextFull"; this.tokens = n; this.max = max;
  }
}

// id -> decoded text of that one token, cached (the constraint scans the vocabulary with it)
export function tokenTexts(tok) {
  const texts = [];
  return (id) => (texts[id] ??= tok.decode([id]));
}

// Wrap a sampler with the tool-call constraint (harness/constrain.js). setText(t) tells it the
// answer so far (call it after each emitted token); within one speculative step every sampled
// column is appended to it in order, so each verified position is masked against the text before
// it, and accepted tokens always satisfy the constraint.
// keep(n): the caller emitted the next n sampled tokens (in order; the columns of a speculative
// step that were rejected are dropped at the next step's first sample). `forced` counts kept
// positions where the model's own top token was not allowed (reset by setText("")): a healthy
// model forces ~0 per call, garbage logits (a misbehaving engine) force most tokens. `garbage`
// turns on when one call forces GARBAGE.abs tokens, or GARBAGE.min and more than GARBAGE.ratio of
// its tokens: the caller should end the answer there and run none of its calls.
// Without tools it is the base sampler.
export const GARBAGE = { abs: 16, min: 6, ratio: 0.2, lowMass: 1e-4, lowRun: 8 };
export function constrainedSampler(base, tools, opts = {}) {
  const { tokenText, vocabSize, style = "xml", stops = [], thinking = false } = opts;
  // API asks (mode / format given): the strict grammar over the whole answer
  if (opts.mode !== undefined || opts.format != null) return strictSampler(base, tools || [], opts);
  if (!tools?.length) return { sample: base, setText() {}, keep() {}, constraint: null, forced: 0, garbage: false };
  const C = new ToolCallConstraint(tools, { vocabSize, tokenText, style, stops, thinking });
  let cols = [], kept = false, callF = 0, callN = 0, lastIn = false;
  const w = {
    forced: 0, garbage: false,
    sample(lg) {
      if (kept) { cols = []; kept = false; }   // a new step
      const inCall = C.inCall;
      C.mask(lg);
      cols.push({ f: C.forced, inCall });
      const t = base(lg);
      C.push(tokenText(t));
      return t;
    },
    keep(n = 1) {
      kept = true;
      for (let i = 0; i < n && cols.length; i++) {
        const { f, inCall } = cols.shift();
        if (f) w.forced++;
        if (!inCall) { lastIn = false; continue; }
        if (!lastIn) { callF = 0; callN = 0; lastIn = true; }
        callN++; if (f) callF++;
        if (callF >= GARBAGE.abs || (callF >= GARBAGE.min && callF > GARBAGE.ratio * callN)) w.garbage = true;
      }
    },
    setText(t) {
      if (!t) { w.forced = 0; w.garbage = false; cols = []; kept = false; callF = callN = 0; lastIn = false; }
      C.setText(t);
    },
    constraint: C,
  };
  return w;
}

// The strict grammar (harness/constrain.js GrammarConstraint) as a sampler wrapper, for API asks.
// The garbage guard counts forced positions only where the text is the model's own choice (a
// value's contents, number digits: forced literals, names, keys and closers are expected), per call
// (or per format value): GARBAGE.abs, or GARBAGE.min and more than GARBAGE.ratio of those positions.
// Also garbage: NaN / +Infinity logits, or the allowed share of the probability under
// GARBAGE.lowMass for GARBAGE.lowRun forced positions in a row there. The mask sees the sampler's
// top-k (k = 1 when greedy), so the candidate fast path stays exact. setText() takes the grammar's
// view of the answer so far: each emitted token's C.tt(id) (a tag token's symbol, not its text).
// forcedFree: the forced positions among the model's own choices only (what `forced` counts minus
// the grammar's literals: Code mode's "the call format forced N tokens" note reads this).
// garbage: "mass" (Code mode) turns only on for an engine fault (NaN / +Infinity logits, or the low
// allowed-mass run), never on a model that keeps preferring something the grammar forbids inside a
// long value; the default also counts GARBAGE.abs / GARBAGE.ratio forced positions.
function strictSampler(base, tools, opts) {
  const { tokenText, vocabSize, style = "xml", stops = [], thinking = false, thinkInPrompt = false, mode = "auto", allowed = null, maxCalls = null, parallel = true, format = null, tags = null, garbage: rule = "count" } = opts;
  const C = new ToolCallConstraint(tools, { vocabSize, tokenText, style, stops, thinking, thinkInPrompt, mode, allowed, maxCalls, parallel, format, tags });
  const k = base.gpu ? (base.gpu.kind === "greedy" ? 1 : base.gpu.k || 64) : 64;
  let cols = [], kept = false, ended = false, runF = 0, runN = 0, low = 0, lastIn = false;
  const w = {
    forced: 0, forcedFree: 0, garbage: false,
    sample(lg) {
      if (kept) { cols = []; kept = false; ended = false; }
      // past an end token in one speculative verify: those columns are never emitted, only written
      // to the caches, so they take the model's own choice (the template's "\n<|im_start|>" that the
      // next prompt holds), not the grammar's view from before the end (which bans <|im_start|> and
      // lets a drafted <|endoftext|> through: the next prompt then misses the cached prefix)
      if (ended) { cols.push({ f: false, free: false, bad: false, mass: 1, inValue: false }); return base(lg); }
      const inValue = C.inCall || (C.state.k === "J");
      C.mask(lg, k);
      cols.push({ f: C.forced, free: C.free, bad: C.bad, mass: C.mass, inValue });
      const t = base(lg);
      if (C.stops.has(t)) ended = true;   // an end token is not text
      else C.push(C.tt(t));               // a tag token is its symbol
      return t;
    },
    keep(n = 1) {
      kept = true;
      for (let i = 0; i < n && cols.length; i++) {
        const c = cols.shift();
        if (c.f) w.forced++;
        if (c.bad) w.garbage = true;
        if (!c.inValue) { lastIn = false; continue; }
        if (!lastIn) { runF = 0; runN = 0; low = 0; lastIn = true; }
        if (!c.free) continue;
        runN++;
        if (c.f) { runF++; w.forcedFree++; low = c.mass < GARBAGE.lowMass ? low + 1 : 0; } else low = 0;
        if (low >= GARBAGE.lowRun || (rule !== "mass" && (runF >= GARBAGE.abs || (runF >= GARBAGE.min && runF > GARBAGE.ratio * runN)))) w.garbage = true;
      }
    },
    setText(t) {
      if (!t) { w.forced = 0; w.forcedFree = 0; w.garbage = false; cols = []; kept = false; ended = false; runF = runN = low = 0; lastIn = false; }
      C.setText(t);
    },
    constraint: C,
  };
  return w;
}

// Streaming decode: push(id) -> the new text, holding back while the tail is an incomplete UTF-8
// sequence (U+FFFD). Decodes only the held-back ids each time, so a long answer stays O(n).
export function deltaDecoder(tok) {
  let held = [];
  const d = {
    ids: [], text: "",
    push(id) {
      d.ids.push(id); held.push(id);
      const s = tok.decode(held);
      if (s.endsWith("�") && held.length < 8) return "";
      held = [];
      d.text += s;
      return s;
    },
    end() {
      if (!held.length) return "";
      const s = tok.decode(held);
      held = [];
      d.text += s;
      return s;
    },
  };
  return d;
}

// Callbacks in, async iterator out: push(x) queues, end(err?) finishes (or fails) the iterator.
export function asyncQueue() {
  const items = [];
  let done = false, error = null, wake = null;
  const poke = () => { const w = wake; wake = null; w?.(); };
  return {
    push(x) { items.push(x); poke(); },
    end(err) { done = true; error = err || null; poke(); },
    async *[Symbol.asyncIterator]() {
      for (;;) {
        if (items.length) { yield items.shift(); continue; }
        if (done) { if (error) throw error; return; }
        await new Promise((r) => (wake = r));
      }
    },
  };
}

// assistant text -> the ids it was sampled as. Re-tokenizing the text can split it differently,
// and then the prompt no longer extends what the caches hold. prune(texts) keeps only the turns
// still in the conversation, so the map does not grow without bound.
// An assistant turn's text as ids, when its sampled ids are not at hand (a new tokenizer, a restored
// session): the tags the model writes as single added tokens (<tool_call>, <think>, ...) become those
// tokens, as they were sampled, instead of being spelled out in text pieces. Spelled out, the model
// sees its own earlier calls in a form it was never trained on and copies it on the next request
// ("<tool_tool_calls>..."), which no longer parses as a call. Only these tags: the rest is plain text.
const TURN_TAGS = ["<tool_call>", "</tool_call>", "<think>", "</think>", "<tool_response>", "</tool_response>"];
export function encodeTurn(tok, text) {
  const tags = TURN_TAGS.filter((t) => Number.isInteger(tok.vocab?.[t]));
  if (!tags.length) return tok.encode(text);
  const ids = [];
  const re = new RegExp(tags.map((t) => t.replace(/[/]/g, "\\/")).join("|"), "g");
  let at = 0;
  for (const m of text.matchAll(re)) {
    if (m.index > at) for (const x of tok.encode(text.slice(at, m.index))) ids.push(x);
    ids.push(tok.vocab[m[0]]);
    at = m.index + m[0].length;
  }
  if (at < text.length) for (const x of tok.encode(text.slice(at))) ids.push(x);
  return ids;
}

export class OwnIds {
  constructor() { this.map = new Map(); }
  get(text) { return this.map.get(text); }
  set(text, ids) { this.map.set(text, Array.from(ids)); }
  prune(texts) { const keep = new Set(texts); for (const k of this.map.keys()) if (!keep.has(k)) this.map.delete(k); }
  clear() { this.map.clear(); }
  get size() { return this.map.size; }
}
