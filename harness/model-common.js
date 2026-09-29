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
export const GARBAGE = { abs: 16, min: 6, ratio: 0.2 };
export function constrainedSampler(base, tools, { tokenText, vocabSize, style = "xml", stops = [], thinking = false }) {
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
    if (m.index > at) ids.push(...tok.encode(text.slice(at, m.index)));
    ids.push(tok.vocab[m[0]]);
    at = m.index + m[0].length;
  }
  if (at < text.length) ids.push(...tok.encode(text.slice(at)));
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
