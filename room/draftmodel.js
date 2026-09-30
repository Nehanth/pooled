// A small draft model on the host (Qwen3 0.6B for the 1.7B / 4B: same tokenizer): it proposes the
// next K tokens greedily, and the room verifies them in one lap exactly like prompt-lookup drafts
// (DenseEngine.specStepDrafts), so the output is the big model's whatever the drafts are. It only
// changes how many tokens a lap carries.
//
// The drafter keeps its own KV cache in step with the context by token ids: before drafting it
// compares what its caches hold (`held`) with the context and prefills only the difference, so a
// new chat, a checkpoint restore, a plain step or a rejected draft need no bookkeeping elsewhere.
//   engine: { pos, prefillTokens(ids), forwardToken(id) -> logits } (a DenseEngine holding the
//   whole draft model, embed and head included)
export class DraftModel {
  constructor(engine, { argmax }) {
    this.e = engine;
    this.argmax = argmax;
    this.held = [];          // token ids at positions 0 .. held.length - 1 of the draft caches
    this.stats = { calls: 0, drafted: 0, syncTok: 0, ms: 0 };
  }
  // ctx: the context so far, ending with the token about to be verified (not yet written anywhere).
  // Returns up to k greedy guesses for the tokens after it.
  async propose(ctx, k) {
    if (k < 1 || !ctx.length) return [];
    const t0 = performance.now();
    const E = this.e, maxSeq = E.maxSeq ?? Infinity;
    // longest common prefix of what the caches hold and ctx without its last token
    const want = ctx.length - 1;
    let c = 0;
    const lim = Math.min(this.held.length, want);
    while (c < lim && this.held[c] === ctx[c]) c++;
    this.held.length = c;
    E.pos = c;
    if (c < want) {
      await E.prefillTokens(ctx.slice(c, want));
      this.stats.syncTok += want - c;
      for (let i = c; i < want; i++) this.held.push(ctx[i]);
    }
    k = Math.min(k, maxSeq - want - 1);
    const out = [];
    let t = ctx[want];
    for (let j = 0; j < k; j++) {
      const lg = await E.forwardToken(t);
      this.held.push(t);
      t = this.argmax(lg);
      out.push(t);
    }
    this.stats.calls++; this.stats.drafted += out.length; this.stats.ms += performance.now() - t0;
    return out;
  }
}
