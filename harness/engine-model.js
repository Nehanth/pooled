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
import { buildIds, reusablePrefix, specials } from "../room/conversation.js";
import { tokenTexts, constrainedSampler, OwnIds, encodeTurn } from "./model-common.js";
import { isPrefix, pinSplit } from "./prefix.js";

const PIN_SLOT = "engine-model:pin";

// tools (optional): the agent's tool list; inside a tool call the whole XML call (function and
// parameter names included) is then constrained to the grammar (harness/constrain.js), on every sampled position including
// the ones a speculative step checks, so accepted tokens always satisfy it.
export function engineModel(engine, tok, { thinking = false, maxNew = 1024, K = 3, spec = true, sample = null, tools = null, style = "xml", pin = true } = {}) {
  const S = specials(tok);
  const pick0 = sample || ((lg) => { let b = 0; for (let i = 1; i < lg.length; i++) if (lg[i] > lg[b]) b = i; return b; });
  const stop = new Set([S.imEnd, S.eot].filter(Number.isInteger));
  const cs = constrainedSampler(pick0, tools, { tokenText: tokenTexts(tok), vocabSize: engine.dims?.vocab ?? Object.keys(tok.vocab).length, style, stops: [...stop], thinking });
  const pick = cs.sample;
  const own = new OwnIds();   // assistant text -> the ids it was sampled as
  let fed = [];            // exactly the tokens the engine's caches hold
  let pinned = null;       // the tokens saved in PIN_SLOT (the system prompt + tools), or null
  const slots = pin && typeof engine.saveSlot === "function" && typeof engine.loadSlot === "function";
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
    if (ids.length + 2 > engine.maxSeq) throw new Error(`conversation is ${ids.length} tokens; the context is ${engine.maxSeq}`);
    let reused = reusablePrefix(fed, ids);
    // the middle changed: resume after the system prompt + tools if that is still the start
    if (!reused && pinned && pinned.length < ids.length && isPrefix(pinned, ids)) {
      try { engine.loadSlot(PIN_SLOT); fed = pinned.slice(); reused = pinned.length; }
      catch { pinned = null; }   // the slot is gone (dropped by someone else): prefill it again
    }
    if (!reused) { engine.reset(); fed = []; }
    stats.reused += reused; stats.prefilled += ids.length - reused;
    // prefill up to the end of the system prompt, save it, then the rest (same tokens, same
    // positions: the answer does not change, only where the prefill pauses)
    const cut = slots ? pinSplit(reused, systemLen(system), ids.length) : 0;
    if (cut) {
      await engine.prefillTokens(ids.slice(reused, cut));
      engine.saveSlot(PIN_SLOT);
      pinned = ids.slice(0, cut);
      stats.pins++;
    }
    const rest = ids.slice(cut || reused);
    if (rest.length > 1) await engine.prefillTokens(rest.slice(0, -1));
    fed = ids.slice(0, -1);   // (prefillTokens wrote all but the last prompt token)
    cs.setText("");
    let next = pick(await engine.forwardToken(ids[ids.length - 1]));
    fed.push(ids[ids.length - 1]);
    // `next` is sampled, not yet written. A plain step writes it; a speculative step writes it
    // and the drafts it accepts, and returns the tokens sampled after it (the last one is the
    // new `next`), exactly as the room does.
    const out = [];
    let text = "", done = stop.has(next), garbage = false;
    if (!done) { out.push(next); cs.keep(1); }
    const room = () => Math.min(maxNew - out.length, engine.maxSeq - engine.pos - 2);
    cs.setText(tok.decode(out));
    while (!done && !garbage && room() > 0 && !signal?.aborted) {
      let toks;
      if (spec && engine.mtp && room() > K + 1) { toks = await engine.specStep(next, pick, K); fed.push(next, ...toks.slice(0, -1)); }
      else { toks = [pick(await engine.forwardToken(next))]; fed.push(next); }
      for (const t of toks) { if (stop.has(t)) { done = true; break; } out.push(t); cs.keep(1); }
      garbage = cs.garbage;   // the call grammar forced most tokens: the logits are not the model's
      next = toks[toks.length - 1];
      const now = tok.decode(out);
      cs.setText(now);   // the constraint sees exactly the answer so far
      if (now.length > text.length && !now.endsWith("\uFFFD")) { yield now.slice(text.length); text = now; }
    }
    const all = tok.decode(out);
    if (all.length > text.length) yield all.slice(text.length);
    own.set(all, out.slice());
    stats.generated += out.length;
    stats.last = { reason: garbage ? "garbage" : done ? "stop" : "max", prompt: ids.length, reused, prefilled: ids.length - reused, generated: out.length, forced: cs.forced || 0 };
  }
  return { generate, stats, get fed() { return fed; } };
}
