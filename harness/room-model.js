// The agent's model interface (harness/agent.js: generate({ system, turns, signal }) -> async
// iterable of text deltas) over the room: the model spread over everyone's GPUs, driven through
// roomApi.generate (room.js roomGenerate). The room decides prefix reuse by comparing exact ids
// with what its caches and checkpoints hold, so this adapter only has to render the conversation
// the same way every step: assistant turns replay the exact ids they were sampled as, and step
// N+1 prefills just the tool response and the next assistant header.
//
// api: { tok(), maxSeq(), generate(ids, { onToken, stop, maxNew, sample, signal }) -> result }.
// The caller holds the room's lock (api.lock) for the whole agent run.
import { buildIds, specials, splitThink } from "../room/conversation.js";
import { pickSampler } from "../room/sampling.js";
import { tokenTexts, constrainedSampler, deltaDecoder, asyncQueue, OwnIds, encodeTurn } from "./model-common.js";

export class ContextFull extends Error {
  constructor(n, max) {
    super(`the conversation is ${n} tokens and the context is ${max}: start a new task (the files are kept)`);
    this.name = "ContextFull"; this.tokens = n; this.max = max;
  }
}

const TAG = "<tool_response>";   // the model starting to invent a tool's result: end the answer there
const MARGIN = 16;               // positions kept free past the answer (the template's end tokens)
// how much of the text's end could still grow into TAG (held back from the stream until it can't)
function tagHold(s) {
  for (let k = Math.min(TAG.length - 1, s.length); k > 0; k--) if (TAG.startsWith(s.slice(-k))) return k;
  return 0;
}

export function roomModel(api, {
  thinking = false,       // Code mode default off: a think block eats the answer budget
  maxNew = 4096,          // answer cap; per call also min(maxNew, maxSeq - prompt - 16)
  tools = null,           // for the tool-call constraint (harness/constrain.js)
  style = "xml",
  sampling = "focused",   // room/sampling.js preset; code wants a low temperature
  sample = null,          // (logits) -> id, overrides `sampling` (tests)
  onThink = null,         // (delta) => void: the think block's text, when thinking
} = {}) {
  const own = new OwnIds();
  const counts = new Map();   // text -> tokens, LRU
  const stats = { calls: 0, reused: 0, prefilled: 0, generated: 0, tps: 0, last: null };
  let lastTok = null, tt = null, vocabSize = 0;
  // a re-deal can load a different model: its tokenizer invalidates every cached id and count
  const tok = () => {
    const t = api.tok();
    if (!t) throw new Error("the model is not loaded");
    if (t !== lastTok) {
      lastTok = t; tt = tokenTexts(t); own.clear(); counts.clear();
      vocabSize = 0;
      for (const v of Object.values(t.vocab || {})) if (v >= vocabSize) vocabSize = v + 1;
    }
    return t;
  };
  const reserve = () => Math.min(maxNew, Math.floor(api.maxSeq() / 4)) + 64;

  async function* generate({ system = "", turns, signal } = {}) {
    const T = tok(), S = specials(T), maxSeq = api.maxSeq();
    const think = thinking && S.think !== undefined;
    own.prune(turns.filter((t) => t.role === "assistant").map((t) => t.text));
    const R = turns.map((t) => (t.role === "assistant" ? { role: "assistant", ids: own.get(t.text) || encodeTurn(T, t.text) } : { role: "user", text: t.text }));
    const ids = buildIds(T, { system, turns: R, thinking: think });
    if (ids.length > maxSeq - MARGIN) throw new ContextFull(ids.length, maxSeq);
    stats.calls++;

    // a single-token <tool_response> (Qwen's added token) is a stop token: it is then never written
    // into the caches, and the next step still extends them. Split across tokens it is caught as text.
    const stop = new Set([S.imEnd, S.eot, T.vocab?.[TAG]].filter(Number.isInteger));
    const cs = constrainedSampler(sample || pickSampler(sampling), tools, { tokenText: tt, vocabSize, style, stops: [...stop], thinking: think });
    const ctrl = new AbortController();
    const onAbort = () => ctrl.abort();
    if (signal?.aborted) ctrl.abort(); else signal?.addEventListener("abort", onAbort, { once: true });

    const q = asyncQueue();
    const dec = deltaDecoder(T);
    let raw = "", sent = 0, cutAt = -1, thinkSent = 0, text = "", garbage = false;
    // the visible text so far: the answer part when thinking, cut at an invented tool response
    const visible = () => {
      let s = raw;
      if (think) {
        const sp = splitThink(raw);
        if (onThink && sp.think != null && sp.think.length > thinkSent) { onThink(sp.think.slice(thinkSent)); thinkSent = sp.think.length; }
        s = sp.answer;
      }
      return s;
    };
    const flush = (final) => {
      let s = visible();
      const i = s.indexOf(TAG);
      if (i >= 0) { s = s.slice(0, i); if (cutAt < 0) { cutAt = i; ctrl.abort(); } }
      const upto = final || cutAt >= 0 ? s.length : s.length - tagHold(s);
      if (upto > sent) { q.push(s.slice(sent, upto)); sent = upto; }
      return s;
    };
    const onToken = (id) => {
      if (cutAt >= 0) return;   // tokens of the step that was in flight when the tag appeared
      cs.keep(1);
      const d = dec.push(id);
      raw += d;
      cs.setText(raw);
      if (d) flush(false);
      // the call grammar forced most of a call's tokens: the logits are not the model's (a
      // misbehaving engine), so stop decoding garbage instead of running to the cap
      if (cs.garbage && cutAt < 0) { garbage = true; cutAt = visible().length; ctrl.abort(); flush(true); }
    };
    const run = api.generate(ids, { onToken, stop, maxNew: Math.min(maxNew, maxSeq - ids.length - MARGIN), sample: cs.sample, signal: ctrl.signal })
      .then((r) => { if (cutAt < 0) raw += dec.end(); text = flush(true); q.end(); return r; }, (err) => { q.end(err); throw err; });
    run.catch(() => {});
    let finished = false;
    try {
      for await (const d of q) yield d;
      const r = await run;
      finished = true;
      // the ids this answer was sampled as, for the next step's prompt. When the text was cut, keep
      // only the ids that decode to exactly the cut text (else the next render re-tokenizes it).
      let mine = dec.ids;
      if (cutAt >= 0 || think) mine = idsFor(T, dec.ids, text, think);
      if (mine) own.set(text, mine);
      stats.reused += r.reused || 0; stats.prefilled += r.prefilled || 0; stats.generated += (r.tokens?.length ?? r.count ?? 0);
      stats.tps = r.tps || 0;
      stats.last = { reason: garbage ? "garbage" : cutAt >= 0 ? "tool_response" : r.reason, prompt: ids.length, reused: r.reused || 0, prefilled: r.prefilled || 0, generated: r.tokens?.length ?? r.count ?? 0, tps: r.tps || 0, stats: r.stats || "", forced: cs.forced || 0 };
    } finally {
      signal?.removeEventListener("abort", onAbort);
      if (!finished) { ctrl.abort(); await run.catch(() => {}); }   // consumer left early: stop the room's step
    }
  }

  // the shortest run of ids whose decode is exactly `text` (with thinking, the whole think block
  // plus the answer: the prompt must replay what was sampled, think block included)
  function idsFor(T, ids, text, think) {
    if (think) return ids;
    for (let k = ids.length; k >= 0; k--) {
      const s = T.decode(ids.slice(0, k));
      if (s === text) return ids.slice(0, k);
      if (s.length < text.length) return null;
    }
    return null;
  }

  return {
    generate,
    // tokens the conversation may take: the context minus room for an answer
    budget: () => api.maxSeq() - reserve(),
    // exact token count, cached per string (256 most recent)
    count(text) {
      const T = tok();
      let n = counts.get(text);
      if (n !== undefined) { counts.delete(text); counts.set(text, n); return n; }
      n = T.encode(text).length;
      counts.set(text, n);
      if (counts.size > 256) counts.delete(counts.keys().next().value);
      return n;
    },
    idsFor: (text) => own.get(text),       // for Agent.toJSON
    // which tokenizer those ids belong to: the vocabulary size and how a fixed probe encodes
    idsTag: () => { const T = tok(); return `${vocabSize}:` + T.encode("Tabby ids · fn(x) => 1024 ✓").join(","); },
    adopt: (text, ids) => { tok(); own.set(text, ids); },   // Agent.from
    stats,
  };
}
