// One engine (a Qwen35Engine or DenseEngine over a single device) as a generate function with the
// room's contract (room.js roomGenerate), so the serve v2 core (room/api.js apiRun2) and Code mode's
// adapter (harness/core-model.js) run on it exactly as they run on the room:
//   generate(ids, { onToken(id, drafted), stop, maxNew, sample, signal, pin })
//     -> { reason: "stop"|"max"|"ctx"|"abort", reused, prefilled, count, tps, tPre, tDecode }
// Only the tokens after what the engine already holds are prefilled (an agent resends ~96% of its
// prompt every step). `pin`: the length of the prompt's fixed start (the system prompt + tools):
// when the caches do not hold it yet, the prefill pauses there and saves it in a GPU slot (issue
// #73), so a prompt that changes after it (a compacted agent conversation) resumes there.
// onToken gets each sampled token in order, never a stop id (as the room: an end token is not
// emitted); with the draft head, a speculative step gives the same tokens faster.
//
// `fed` is exactly what the engine's caches hold. A speculative step can accept a stop id as a
// draft and write tokens after it: fed says so, and the next prompt (which renders the turn's end
// its own way) then resumes from the pinned slot. Cutting fed back instead would be wrong on a
// hybrid model, whose DeltaNet state cannot be rolled back.
import { reusablePrefix } from "../room/conversation.js";
import { isPrefix, pinSplit } from "./prefix.js";

const PIN_SLOT = "engine-model:pin";
const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

export function engineGenerate(engine, { spec = true, K = 3, pin: usePin = true } = {}) {
  let fed = [];            // exactly the tokens the engine's caches hold
  let pinned = null;       // the tokens saved in PIN_SLOT, or null
  const slots = usePin && typeof engine.saveSlot === "function" && typeof engine.loadSlot === "function";
  const stats = { calls: 0, reused: 0, prefilled: 0, generated: 0, pins: 0 };

  async function generate(ids, { onToken = () => {}, stop = new Set(), maxNew = 1024, sample, signal = null, pin = 0 } = {}) {
    if (ids.length + 2 > engine.maxSeq) throw new Error(`the prompt is ${ids.length} tokens and the context is ${engine.maxSeq}`);
    stats.calls++;
    let reused = reusablePrefix(fed, ids);
    // the middle changed: resume after the pinned start if that is still the start
    if (!reused && pinned && pinned.length < ids.length && isPrefix(pinned, ids)) {
      try { engine.loadSlot(PIN_SLOT); fed = pinned.slice(); reused = pinned.length; }
      catch { pinned = null; }   // the slot is gone: prefill it again
    }
    if (!reused) { engine.reset(); fed = []; }
    stats.reused += reused; stats.prefilled += ids.length - reused;
    const t0 = now();
    // prefill up to the end of the pinned start, save it, then the rest (same tokens, same
    // positions: the answer does not change, only where the prefill pauses)
    const cut = slots ? pinSplit(reused, pin, ids.length) : 0;
    if (cut) {
      await engine.prefillTokens(ids.slice(reused, cut));
      engine.saveSlot(PIN_SLOT);
      pinned = ids.slice(0, cut);
      stats.pins++;
    }
    const rest = ids.slice(cut || reused);
    if (rest.length > 1) await engine.prefillTokens(rest.slice(0, -1));
    fed = ids.slice(0, -1);   // (prefillTokens wrote all but the last prompt token)
    let next = sample(await engine.forwardToken(ids[ids.length - 1]));
    fed.push(ids[ids.length - 1]);
    const t1 = now();
    // `next` is sampled, not yet written. A plain step writes it; a speculative step writes it and
    // the drafts it accepts, and returns the tokens sampled after it (the last one is the new `next`)
    let count = 0, reason = null;
    if (stop.has(next)) reason = "stop";
    else { count++; onToken(next, 0); }
    const room = () => Math.min(maxNew - count, engine.maxSeq - engine.pos - 2);
    while (!reason) {
      if (signal?.aborted) { reason = "abort"; break; }
      if (room() <= 0) { reason = count >= maxNew ? "max" : "ctx"; break; }
      let toks;
      if (spec && engine.mtp && room() > K + 1) { toks = await engine.specStep(next, sample, K); for (const x of [next, ...toks.slice(0, -1)]) fed.push(x); }
      else { toks = [sample(await engine.forwardToken(next))]; fed.push(next); }
      for (let k = 0; k < toks.length; k++) {
        if (stop.has(toks[k])) { reason = "stop"; break; }
        count++;
        onToken(toks[k], k < toks.length - 1 ? 1 : 0);
      }
      next = toks[toks.length - 1];
    }
    const t2 = now();
    stats.generated += count;
    const tDecode = t2 - t1;
    return { reason, reused, prefilled: ids.length - reused, count, tps: tDecode > 0 ? count / (tDecode / 1000) : 0, tPre: t1 - t0, tDecode, stats: "" };
  }
  return { generate, stats, get fed() { return fed; } };
}

// What harness/core-model.js needs from where the model runs, over one engine and its tokenizer.
// chatTemplate: the GGUF's tokenizer.chat_template (or tokenizer.json's); the vocabulary size is the
// larger of the tokenizer's and the engine's (padded) logits, so the grammar's masks cover every id.
export function engineHost(engine, tok, { chatTemplate = "", spec = true, K = 3, pin = true } = {}) {
  const g = engineGenerate(engine, { spec, K, pin });
  let vs = 0;
  for (const v of Object.values(tok.vocab || {})) if (v >= vs) vs = v + 1;
  vs = Math.max(vs, engine.dims?.vocab || 0);
  return {
    tok: () => tok,
    chatTemplate: () => chatTemplate || tok.chatTemplate || engine.chatTemplate || "",
    maxSeq: () => engine.maxSeq,
    modelKey: () => "",
    vocabSize: () => vs,
    generate: g.generate,
    stats: g.stats,
    get fed() { return g.fed; },
  };
}
