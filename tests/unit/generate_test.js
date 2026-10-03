import { createGenerator } from "../../engine/generate.js";

const eq = (a, b, message = "mismatch") => {
  if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(`${message}: ${JSON.stringify(a)} != ${JSON.stringify(b)}`);
};
const assert = (v, message) => { if (!v) throw new Error(message); };
async function rejects(fn, message) {
  try { await fn(); } catch (err) { eq(err.message, message); return; }
  throw new Error(`expected rejection: ${message}`);
}

// Symbolic logits are token ids: the fixture records the actual cache writes and checkpoint
// calls separately from emitted tokens, which catches continuation and stop-token regressions.
function fixture({ outputs = [3, 4, 9], maxSeq = 64, state = {}, options = {}, hooks = {} } = {}) {
  const ai = { engine: { maxSeq }, settings: { sampling: "exact" }, chain: [], fed: null,
    pos: 0, pending: null, pendingCtl: {}, abort: false, frames: 0, ...state };
  const calls = { resets: 0, prefill: [], pipe: [], saved: 0, clear: [], resume: [], sent: [], laps: [] };
  const queue = [...outputs];
  const pipeline = {
    ckptEngine() { return !!ai.engine?.saveSlot && ai.engine.hostCkpt !== false; },
    ckptResume(ids, prefix) { calls.resume.push(prefix); return prefix; },
    resetState() { calls.resets++; ai.fed = []; ai.pos = 0; },
    async aiPrefill(ids, opts) {
      calls.prefill.push([...ids]); calls.prefillOptions = opts;
      if (opts.aborted()) return null;
      ai.fed.push(...ids); ai.pos += ids.length; ai.frames += ids.length;
      return queue.shift();
    },
    async aiPipeToken(id, logits, fillNext, desc) {
      calls.pipe.push(id); calls.pipeDesc = desc; ai.fed.push(id); ai.pos++; return queue.shift();
    },
    ckptClear(all) { calls.clear.push(all); },
    ckptSave() { calls.saved++; },
    lapWait(key, ms, why) { calls.laps.push([key, ms, why]); return Promise.resolve(new Float32Array([1, 2])); },
    sendChain(msg) { calls.sent.push(msg); },
    noteLap() {},
  };
  const generate = createGenerator({ state: ai, pipeline, options, hooks });
  const run = (ids = [1, 2], opts = {}) => generate(ids, { stop: new Set([9]), sample: (x) => x, ...opts });
  return { ai, calls, pipeline, generate, run };
}

Deno.test("generate: stop tokens are neither emitted nor piped, and observers see prefill and tokens", async () => {
  const passes = [], emitted = [], statuses = [];
  const f = fixture({ hooks: { computePass: (n) => passes.push(n) } });
  const r = await f.run([1, 2], { onToken: (...args) => emitted.push(args), onStatus: (s) => statuses.push(s) });
  eq(r.tokens, [3, 4]); eq(r.reason, "stop"); eq(r.count, 2); eq(r.prefilled, 2); eq(r.preFrames, 2);
  eq(f.calls.pipe, [3, 4]); eq(f.ai.fed, [1, 2, 3, 4]); eq(f.calls.saved, 1);
  eq(emitted, [[3, false], [4, false]]); eq(passes, [2, 1, 1]);
  assert(statuses[0].startsWith("prefill:"), "prefill status");
});

Deno.test("generate: a length cap preserves the sampled pending token for Continue without prefill", async () => {
  const f = fixture();
  const first = await f.run([1, 2], { maxNew: 1 });
  eq(first.tokens, [3]); eq(first.reason, "max"); eq(f.ai.pending, { next: 4, at: 3 });
  const next = await f.run([...f.ai.fed], { maxNew: 2 });
  eq(next.tokens, [4]); eq(next.reason, "stop"); eq(next.reused, 3); eq(next.prefilled, 0);
  eq(f.calls.prefill, [[1, 2]]); eq(f.calls.resets, 1); eq(f.ai.pending, null);
});

Deno.test("generate: context cap emits the last token without writing beyond capacity", async () => {
  const f = fixture({ maxSeq: 4 });
  const r = await f.run();
  eq(r.tokens, [3, 4]); eq(r.reason, "ctx"); eq(r.capped, true);
  eq(f.calls.pipe, [3]); eq(f.ai.pos, 3); eq(f.ai.pending, null);
});

Deno.test("generate: zero maxNew emits nothing and preserves the first sampled token", async () => {
  const f = fixture();
  const r = await f.run([1, 2], { maxNew: 0 });
  eq(r.tokens, []); eq(r.reason, "max"); eq(f.ai.pending, { next: 3, at: 2 }); eq(f.calls.pipe, []);
});

Deno.test("generate: signal abort during prefill and state abort after a token retain existing semantics", async () => {
  const controller = new AbortController(); controller.abort();
  const before = fixture();
  const r = await before.run([1, 2], { signal: controller.signal });
  eq(r.reason, "abort"); eq(r.tokens, []); eq(before.calls.pipe, []);
  const after = fixture();
  const r2 = await after.run([1, 2], { onToken() { after.ai.abort = true; } });
  eq(r2.reason, "abort"); eq(r2.tokens, [3]); eq(after.calls.pipe, [3]); eq(after.ai.pending, null);
});

Deno.test("generate: strict-prefix reuse fills only the suffix and a different prompt resets", async () => {
  const f = fixture({ state: { fed: [1, 2], pos: 2 }, outputs: [9, 9] });
  const reused = await f.run([1, 2, 7]);
  eq(reused.reused, 2); eq(reused.prefilled, 1); eq(f.calls.prefill, [[7]]); eq(f.calls.resets, 0);
  await f.run([8]);
  eq(f.calls.resets, 1); eq(f.calls.prefill, [[7], [8]]); eq(f.calls.resume, [2, 0]);
});

Deno.test("generate: checkpoints can restore a prefix and fill the draft-cache row", async () => {
  const f = fixture({ outputs: [9] });
  const draftRows = [];
  f.ai.engine.mtp = {};
  f.ai.engine.mtpRun = (...args) => draftRows.push(args);
  // Rebuild after replacing the injected method: the factory captures its dependencies.
  f.pipeline.ckptResume = () => { f.ai.fed = [1]; f.ai.pos = 1; f.ai.xAt = 1; return 1; };
  const generate = createGenerator({ state: f.ai, pipeline: f.pipeline });
  const r = await generate([1, 2], { stop: new Set([9]), sample: (x) => x });
  eq(r.reused, 1); eq(draftRows, [[null, 2, 1, false]]); eq(f.ai.xAt, null); eq(f.calls.resets, 0);
});

Deno.test("generate: pipeline or callback failure clears reusable state and checkpoints, then rethrows", async () => {
  for (const where of ["prefill", "callback"]) {
    const f = fixture();
    f.ai.pendingCtl = { rb: 3 };
    const error = new Error(where);
    if (where === "prefill") f.pipeline.aiPrefill = () => { throw error; };
    const generate = createGenerator({ state: f.ai, pipeline: f.pipeline });
    await rejects(() => generate([1, 2], { stop: new Set([9]), sample: (x) => x,
      onToken() { if (where === "callback") throw error; } }), where);
    eq(f.ai.fed, null); eq(f.ai.pendingCtl, {}); eq(f.calls.clear, [true]); eq(f.calls.saved, 0);
  }
});

Deno.test("generate: readiness errors happen before any cache mutation", async () => {
  const unloaded = fixture({ state: { engine: null } });
  await rejects(() => unloaded.run(), "the model is not loaded");
  const degraded = fixture({ state: { degraded: true } });
  await rejects(() => degraded.run(), "a device left: re-deal the layers first");
  eq(unloaded.calls.resets, 0); eq(degraded.calls.clear, []);
});

Deno.test("generate: default sampler is selected per call and its GPU descriptor reaches the pipeline", async () => {
  const f = fixture({ outputs: [new Float32Array([0, 1, 0, 9]), new Float32Array([0, 0, 9])] });
  const desc = { kind: "greedy" };
  f.ai.engine.gpuDescFor = (sample) => { eq(sample.gpu, desc); return desc; };
  const r = await f.generate([1], { stop: new Set([2]) });
  eq(r.tokens, [3]); eq(f.calls.prefillOptions.desc, desc); eq(f.calls.pipeDesc, desc);
});

function specFixture({ chain = false, maxSeq = 64 } = {}) {
  const f = fixture({ outputs: [3], maxSeq });
  f.ai.chain = chain ? ["worker"] : [];
  f.ai.engine.mtp = { stats: { drafts: 0, accepted: 0 } };
  f.ai.engine.specStep = async (next, sample, K) => {
    eq(next, 3); eq(K, 2);
    f.ai.engine.pos += 3;
    f.ai.engine.mtp.stats = { drafts: 2, accepted: 2 };
    return [4, 5, 6];
  };
  return f;
}

Deno.test("generate: speculative cap tracks accepted cache writes, draft flags and pending next token", async () => {
  const f = specFixture(); const emitted = [];
  const r = await f.run([1, 2], { maxNew: 3, onToken: (...args) => emitted.push(args) });
  eq(r.tokens, [3, 4, 5]); eq(r.reason, "max"); eq(r.acc, 1);
  eq(emitted, [[3, false], [4, 1], [5, 1]]);
  eq(f.ai.fed, [1, 2, 3, 4, 5]); eq(f.ai.pos, 5); eq(f.ai.xAt, 5); eq(f.ai.pending, { next: 6, at: 5 });
});

Deno.test("generate: speculative context guard stops before a verify pass overflows", async () => {
  const f = specFixture({ maxSeq: 4 });
  f.ai.engine.specStep = () => { throw new Error("must not verify"); };
  const r = await f.run();
  eq(r.tokens, [3]); eq(r.reason, "ctx"); eq(f.ai.pending, null);
});

Deno.test("generate: distributed speculative verify sends existing wire frames and defers rollback", async () => {
  const f = specFixture({ chain: true }); const hidden = [];
  f.ai.engine.dims = { dim: 1 }; f.ai.engine.NC = 1;
  f.ai.lastHidden = new Float32Array([8]);
  f.ai.engine.setHidden = (h) => hidden.push([...h]);
  f.ai.engine.embedRunBatch = async (tokens) => new Float32Array(tokens);
  f.ai.engine.specStep = async (_next, _sample, _K, spec) => {
    const h = await spec.runTrunk([3, 4], 2); eq([...h], [1, 2]);
    await spec.onReject(1); f.ai.engine.pos += 1; return [9];
  };
  const r = await f.run();
  eq(r.tokens, [3]); eq(r.reason, "stop"); eq(hidden, [[8]]);
  eq(f.calls.laps, [["b2", 90000, "verify"]]); eq(f.ai.pendingCtl, { rb: 1 });
  const msg = f.calls.sent[0];
  eq([msg.t, msg.basePos, msg.n, msg.spec, msg.enc], ["ai-hidden-b", 2, 2, 1, "f16"]);
  assert(msg.data instanceof Uint16Array || msg.data instanceof ArrayBuffer, "packed wire payload");
});

Deno.test("generate: invalid speculative hidden states fail and invalidate checkpoints", async () => {
  for (const badHost of [true, false]) {
    const f = specFixture({ chain: true });
    f.ai.engine.dims = { dim: 1 };
    f.ai.engine.embedRunBatch = async () => new Float32Array([badHost ? NaN : 1]);
    f.pipeline.lapWait = () => Promise.resolve(new Float32Array([Infinity]));
    f.ai.engine.specStep = async (_next, _sample, _K, spec) => await spec.runTrunk([3], 2);
    const generate = createGenerator({ state: f.ai, pipeline: f.pipeline });
    await rejects(() => generate([1, 2], { stop: new Set([9]), sample: (x) => x }),
      badHost ? "NaN after HOST layers (pos 2)" : "NaN in hidden returned by peers (pos 2)");
    eq(f.ai.fed, null); eq(f.calls.clear, [true]); eq(f.calls.saved, 0);
  }
});

Deno.test("generate: stale pending token is discarded when the prompt changes", async () => {
  const f = fixture({ state: { fed: [1, 2], pos: 2, pending: { next: 8, at: 2 } }, outputs: [9] });
  const r = await f.run([7, 2]);
  eq(r.tokens, []); eq(r.reused, 0); eq(f.ai.pending, null); eq(f.calls.prefill, [[7, 2]]);
});

Deno.test("generate: lookup drafts preserve copied-token flags and the lookup opt-out", async () => {
  for (const lookup of [true, false]) {
    const f = fixture({ outputs: [2], options: { lookup } });
    const emitted = [];
    f.ai.engine.mtp = { stats: { drafts: 0, accepted: 0 } };
    f.ai.engine.specStep = async () => {
      assert(!lookup, "lookup should handle this repetition"); f.ai.engine.pos++; return [9];
    };
    f.ai.engine.specStepDrafts = async (next, _sample, drafts) => {
      assert(lookup, "lookup is disabled"); eq(next, 2); eq(drafts.slice(0, 2), [7, 8]);
      f.ai.engine.pos += 3; return [7, 8, 9];
    };
    const r = await f.run([1, 2, 7, 8, 1], { onToken: (...args) => emitted.push(args) });
    eq(r.reason, "stop"); eq(r.tokens, lookup ? [2, 7, 8] : [2]); eq(r.copied, lookup ? 2 : 0);
    eq(emitted, lookup ? [[2, false], [7, 2], [8, 2]] : [[2, false]]);
  }
});

Deno.test("generate: request defaults and context follow the currently loaded engine", async () => {
  const f = fixture({ outputs: [3, 4, 9], options: { maxNew: 1, maxSeq: 10 } });
  const r = await f.run(); eq(r.reason, "max"); eq(r.tokens, [3]);
  f.ai.engine = { maxSeq: 4 };
  const r2 = await f.run([...f.ai.fed]);
  eq(r2.tokens, [4]); eq(r2.reason, "ctx");
});

Deno.test("generate: pinned prefixes split prefill and a recent different tag keeps its slot", async () => {
  const f = fixture({ state: { engine: { maxSeq: 64, saveSlot() {} } } });
  const saved = [];
  // Factory dependencies are captured, so install the observer before creating this generator.
  f.pipeline.ckptSave = (pin = false) => saved.push(pin);
  const generate = createGenerator({ state: f.ai, pipeline: f.pipeline });
  const request = { stop: new Set([9]), sample: (x) => x, pin: 2, pinTag: "first", maxNew: 1 };
  await generate([1, 2, 3], request);
  eq(f.calls.prefill, [[1, 2], [3]]); eq(saved, [true, false]);
  eq(f.ai.pinInfo.tag, "first");
  f.ai.ckpt = { pinned: () => [{ ids: [1, 2] }] };
  f.ai.fed = null;
  f.calls.prefill.length = 0; saved.length = 0;
  await generate([5, 6, 7], { ...request, pinTag: "second", maxNew: 0 });
  eq(f.calls.prefill, [[5, 6, 7]]); eq(saved, [false]);
  eq(f.ai.pinInfo.tag, "first");
});

Deno.test("generate: fused head sends before emission, preserves Continue and drops unused work", async () => {
  const events = [];
  const ai = { engine: { maxSeq: 64, gpuDescFor: () => ({ kind: "greedy" }),
    canHeadAhead: () => true, dropAhead: () => events.push("drop"), keepAhead: () => events.push("keep"),
    headAhead: async () => ({ cands: { ids: [4], bad: false }, h: Float32Array.of(1) }),
    headFromHiddenIds: async () => ({ ids: [5], bad: false }) }, chain: ["worker"] };
  const f = fixture({ outputs: [{ ids: [3], bad: false }], state: ai });
  f.pipeline.aiPipeToken = async (id, needLogits, next, desc, ahead) => {
    events.push("send" + id); ahead.onSent(); f.ai.pos++; f.ai.fed.push(id); return null;
  };
  const generate = createGenerator({ state: f.ai, pipeline: f.pipeline });
  const result = await generate([1, 2], { stop: new Set(), sample: (c) => c.ids[0], maxNew: 2,
    onToken: (id) => events.push("emit" + id) });
  eq(result.tokens, [3, 4]); eq(f.ai.pending, { next: 5, at: 4 });
  eq(events, ["drop", "send3", "emit3", "keep", "send4", "emit4", "drop"]);
});

Deno.test("generate: dense lookup verifies only with capable peers and preserves draft flags", async () => {
  for (const capable of [true, false]) {
    const warnings = [], emitted = [], verifies = [];
    const f = fixture({ outputs: [2, 9], state: { chain: ["worker"] },
      hooks: { getPeerMeta: () => capable ? { dspec: 1 } : {}, log: (...args) => warnings.push(args) } });
    f.ai.engine.specStats = { drafts: 0, accepted: 0 };
    f.ai.engine.specStepDrafts = async (next, _sample, drafts) => {
      verifies.push([next, drafts]);
      f.ai.engine.pos += 3;
      f.ai.engine.specStats = { drafts: 2, accepted: 2 };
      return [7, 8, 9];
    };
    const r = await f.run([1, 2, 7, 8, 1], { onToken: (...args) => emitted.push(args) });
    eq(r.reason, "stop"); eq(r.tokens, capable ? [2, 7, 8] : [2]);
    eq(r.copied, capable ? 2 : 0); eq(verifies.length, capable ? 1 : 0);
    eq(emitted, capable ? [[2, 0], [7, 2], [8, 2]] : [[2, 0]]);
    eq(f.calls.pipe, capable ? [] : [2]); eq(warnings.length, capable ? 0 : 1);
    eq(f.ai.fed, capable ? [1, 2, 7, 8, 1, 2, 7, 8] : [1, 2, 7, 8, 1, 2]);
  }
});

Deno.test("generate: dense speculation rechecks returning peer capabilities before each lap", async () => {
  let capable = true;
  const f = fixture({ outputs: [2, 9], state: { chain: ["worker"] },
    hooks: { getPeerMeta: () => capable ? { dspec: 1 } : {} } });
  let verified = 0;
  f.ai.engine.specStepDrafts = async () => {
    verified++; f.ai.engine.pos += 2; capable = false; return [7, 1];
  };
  const r = await f.run([1, 2, 7, 8, 1]);
  eq(r.tokens, [2, 7, 1]); eq(verified, 1); eq(f.calls.pipe, [1]);
  eq(f.ai.lkFullD, false); eq(f.ai.dspecWarned, true);
});

Deno.test("generate: engines that disable host checkpoints do not split pinned prefill", async () => {
  const f = fixture({ outputs: [9], state: { engine: { maxSeq: 64, saveSlot() {}, hostCkpt: false } } });
  await f.run([1, 2, 3], { pin: 2, pinTag: "dense" });
  eq(f.calls.prefill, [[1, 2, 3]]); eq(f.ai.pinInfo, undefined);
});


Deno.test("generate: draft model requires capable peers and rechecks after a verify", async () => {
  for (const initiallyCapable of [false, true]) {
    let capable = initiallyCapable, proposed = 0, verified = 0;
    const f = fixture({ outputs: [2, 9], options: { lookup: false }, state: { chain: ["worker"] },
      hooks: { getPeerMeta: () => capable ? { dspec: 1 } : {} } });
    f.ai.draft = { pickK: () => 2, propose: async () => { proposed++; return [7, 8]; }, note() {}, stats: {} };
    f.ai.engine.specStats = { drafts: 0, accepted: 0 };
    f.ai.engine.specStepDrafts = async () => {
      verified++; capable = false; f.ai.engine.pos += 3;
      return [7, 8, 1];
    };
    const result = await f.run([1, 2], { maxNew: 8 });
    eq([proposed, verified], initiallyCapable ? [1, 1] : [0, 0]);
    eq(result.tokens, initiallyCapable ? [2, 7, 8, 1] : [2]);
    eq(f.calls.pipe, initiallyCapable ? [1] : [2]);
  }
});

// Expert offload (room/plan.js specWithOffload): a deal with a device offloading experts decodes plainly by default,
// a forced override speculates again, and a room without offload speculates as before
function offloadFixture({ offloadBy = null, experts = false, options = {} } = {}) {
  const f = fixture({ outputs: [3, 4, 9], options });
  f.ai.engine.mtp = { stats: { drafts: 0, accepted: 0 } };
  f.verifies = 0;
  f.ai.engine.specStep = async () => { f.verifies++; f.ai.engine.pos += 2; return [4, 9]; };
  if (offloadBy) f.ai.offloadBy = offloadBy;
  if (experts) f.ai.engine.experts = { layers: [1] };
  return f;
}
Deno.test("generate: a deal with expert offload decodes plainly unless forced; without offload it speculates", async () => {
  const pc = { lo: 15, hi: 48, vramBytes: 1, ramBytes: 1 };
  for (const [what, f, spec] of [
    ["no offload", offloadFixture(), true],
    ["an empty offload map (the deal offloads nothing)", offloadFixture({ offloadBy: {} }), true],
    ["a worker offloads", offloadFixture({ offloadBy: { pc } }), false],
    ["this device's own engine offloads", offloadFixture({ experts: true }), false],
    ["forced on (POOLED_OFFLOAD_SPEC=1, ?offspec=1)", offloadFixture({ offloadBy: { pc }, options: { offloadSpec: true } }), true],
    ["forced off", offloadFixture({ options: { offloadSpec: false } }), false],
  ]) {
    const r = await f.run([1, 2]);
    eq(r.tokens, [3, 4], what);
    eq(f.verifies > 0, spec, what + ": speculated");
    eq(f.calls.pipe, spec ? [] : [3, 4], what + ": plain tokens piped");
  }
});
