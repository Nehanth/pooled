// The agent's model interface (harness/agent.js: generate({ system, turns }) -> async iterable of
// text) over one Qwen35Engine and its tokenizer. Every request re-renders the whole conversation
// with the chat template, but only the tokens after what the engine already holds are prefilled:
// an agent resends ~96% of its input every step (docs/long-context-and-sessions.md).
//
// The model's own turns are kept as the exact ids it sampled (keyed by their text), so rendering
// never re-tokenizes them differently and the prefix stays reusable. Decoding is greedy by
// default; with the draft head, speculative steps give the same tokens faster.
//
// The system prompt + tools is checkpointed in a GPU slot on the way through the prefill (issue
// #73): when the rest of the prompt changes in the middle (Agent compaction), the next request
// loads that slot and prefills only what follows it. `pin: false` turns it off.
import { buildIds, specials } from "../room/conversation.js";
import { tokenTexts, constrainedSampler, OwnIds, encodeTurn, ContextFull, asyncQueue } from "./model-common.js";
import { engineGenerate } from "./engine-gen.js";

// tools (optional): the agent's tool list; inside a tool call the whole XML call (function and
// parameter names included) is then constrained to the grammar (harness/constrain.js), on every sampled position including
// the ones a speculative step checks, so accepted tokens always satisfy it.
// The prefill, the pinned slot and the decode loop are harness/engine-gen.js (shared with Code
// mode's core path, harness/core-model.js).
export function engineModel(engine, tok, { thinking = false, maxNew = 1024, K = 3, spec = true, sample = null, tools = null, style = "xml", pin = true } = {}) {
  const S = specials(tok);
  const pick0 = sample || ((lg) => { let b = 0; for (let i = 1; i < lg.length; i++) if (lg[i] > lg[b]) b = i; return b; });
  const stop = new Set([S.imEnd, S.eot].filter(Number.isInteger));
  const cs = constrainedSampler(pick0, tools, { tokenText: tokenTexts(tok), vocabSize: engine.dims?.vocab ?? Object.keys(tok.vocab).length, style, stops: [...stop], thinking });
  const own = new OwnIds();   // assistant text -> the ids it was sampled as
  const gen = engineGenerate(engine, { spec, K, pin });
  const stats = { calls: 0, reused: 0, prefilled: 0, generated: 0, pins: 0, last: null };
  // tokens of the system prompt alone (the conversation's ids start with exactly these)
  let sysKey = null, sysN = 0;
  const systemLen = (system) => {
    if (system !== sysKey) { sysKey = system; sysN = system ? buildIds(tok, { system, turns: [], thinking }).length : 0; }
    return sysN;
  };

  async function* generate({ system = "", turns, signal } = {}) {
    stats.calls++;
    own.prune(turns.filter((t) => t.role === "assistant").map((t) => t.text));
    const T = turns.map((t) => (t.role === "assistant" ? { role: "assistant", ids: own.get(t.text) || encodeTurn(tok, t.text) } : { role: "user", text: t.text }));
    const ids = buildIds(tok, { system, turns: T, thinking });
    if (ids.length + 2 > engine.maxSeq) throw new ContextFull(ids.length, engine.maxSeq);   // the agent ends the request (reason "context")
    const ctrl = new AbortController();
    const onAbort = () => ctrl.abort();
    if (signal?.aborted) ctrl.abort(); else signal?.addEventListener?.("abort", onAbort, { once: true });
    const q = asyncQueue();
    const out = [];
    let text = "", garbage = false;
    cs.setText("");
    const onToken = (t) => {
      out.push(t); cs.keep(1);
      const now = tok.decode(out);
      cs.setText(now);   // the constraint sees exactly the answer so far
      // the call grammar forced most tokens: the logits are not the model's (stops after this step)
      if (cs.garbage && !garbage) { garbage = true; ctrl.abort(); }
      if (now.length > text.length && !now.endsWith("\uFFFD")) { q.push(now.slice(text.length)); text = now; }
    };
    const run = gen.generate(ids, { onToken, stop, maxNew, sample: cs.sample, signal: ctrl.signal, pin: slotsPin(system) })
      .then((r) => { q.end(); return r; }, (err) => { q.end(err); throw err; });
    run.catch(() => {});
    let finished = false;
    try {
      for await (const d of q) yield d;
      const r = await run;
      finished = true;
      const all = tok.decode(out);
      if (all.length > text.length) yield all.slice(text.length);
      own.set(all, out.slice());
      stats.reused += r.reused; stats.prefilled += r.prefilled; stats.generated += out.length; stats.pins = gen.stats.pins;
      stats.last = { reason: garbage ? "garbage" : r.reason === "stop" ? "stop" : "max", prompt: ids.length, reused: r.reused, prefilled: r.prefilled, generated: out.length, forced: cs.forced || 0, tps: r.tps, tDecode: r.tDecode };
    } finally {
      signal?.removeEventListener?.("abort", onAbort);
      if (!finished) { ctrl.abort(); await run.catch(() => {}); }
    }
  }
  const slotsPin = (system) => (pin ? systemLen(system) : 0);
  return { generate, stats, get fed() { return gen.fed; } };
}
