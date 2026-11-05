import { createPipeline } from "../../room/pipeline.js";
import { createGenerator } from "../../engine/generate.js";
import { createTimingCollector } from "../../room/timing.js";
import { correlateTiming, operationSummary, generationSample } from "../../benchmarks/bench_room_report.js";
import { wireSnapshot, wireDifference } from "../../benchmarks/bench_room_wire.js";
import { makeLink } from "../../room/transport.js";

const eq = (a, b) => { if (JSON.stringify(a) !== JSON.stringify(b)) throw Error(`${JSON.stringify(a)} != ${JSON.stringify(b)}`); };
const ok = (v) => { if (!v) throw Error("assertion failed"); };
const scores = (id) => Float32Array.from({ length: 10 }, (_, i) => i === id ? 1 : 0);

async function run({ observer = null, chain = false, fused = false, speculative = false, profile = "browser" } = {}) {
  const calls = [], frames = [], outputs = [3, 4, 9];
  const ai = { chain: chain ? ["worker"] : [], role: "host", settings: { sampling: "exact" },
    waiters: new Map(), pendingCtl: {}, dropQ: [], fed: [], pos: 0, q: Promise.resolve() };
  const candidates = () => ({ ids: Uint32Array.of(outputs.shift()), vals: Float32Array.of(1), bad: false });
  ai.engine = { maxSeq: 64, dims: { dim: 1 }, NC: 4, pos: 0,
    reset() { calls.push("reset"); this.pos = 0; },
    async prefillTokens(ids) { calls.push(["prefill", ...ids]); this.pos += ids.length; },
    async prefillToken(id) { calls.push(["prefill-one", id]); },
    async embedRun(id, pos) { calls.push(["host", id, pos]); return Float32Array.of(id); },
    async embedRunBatch(ids, pos) { calls.push(["batch", ...ids, pos]); return Float32Array.from(ids); },
    async headFromHidden() { calls.push("head"); return scores(outputs.shift()); },
    gpuDescFor: fused ? (sample) => sample.gpu : () => null,
    async headFromHiddenIds() { calls.push("head-ids"); return candidates(); },
    canHeadAhead: () => fused,
    async headAhead() { calls.push("ahead"); const cands = candidates(); return { cands, h: Float32Array.of(cands.ids[0]) }; },
    keepAhead() { calls.push("keep"); }, dropAhead() { calls.push("drop"); },
  };
  ai.device = { queue: { onSubmittedWorkDone: async () => calls.push("sync") } };
  if (speculative) {
    ai.engine.mtp = { stats: { drafts: 0, accepted: 0 } };
    ai.engine.specStep = async function (next, sample) {
      eq(sample.gpu, { kind: "greedy" });
      calls.push(["verify", next]); this.pos += 3; this.mtp.stats = { drafts: 2, accepted: 2 };
      return [sample(scores(4)), 5, 9];
    };
  }
  let pipeline;
  pipeline = createPipeline({ state: ai, options: { profile, fillDrafts: false }, hooks: { onTiming: observer },
    transport: { sendTo() {}, sendHidden(_, msg) {
      frames.push(msg);
      queueMicrotask(() => pipeline.handleFrame("worker", { ...msg, t: msg.t === "ai-hidden-b" ? "ai-hiddenret-b" : "ai-hiddenret" }));
    } } });
  const generate = createGenerator({ state: ai, pipeline, options: { profile, fillDrafts: false, denseSpec: false } });
  const sample = (logits) => logits.ids ? logits.ids[0] : logits.indexOf(Math.max(...logits));
  Object.defineProperty(sample, "gpu", { value: { kind: "greedy" } });
  const result = await generate([1, 2, 1, 2, 1, 2, 1, 2], { stop: new Set([9]), maxNew: 4, sample });
  return { calls, frames, tokens: result.tokens, reason: result.reason, fed: ai.fed };
}

for (const profile of ["browser", "node"]) {
  for (const mode of [{}, { chain: true }, { chain: true, fused: true }, { speculative: true }]) {
    Deno.test(`host timing ${profile}: output/frame parity ${JSON.stringify(mode)}`, async () => {
      const baseline = await run({ ...mode, profile });
      const collector = createTimingCollector(); collector.begin("test");
      eq(await run({ ...mode, profile, observer: collector.record }), baseline);
      eq(await run({ ...mode, profile, observer() { throw Error("observer failed"); } }), baseline);
      const capture = { ...collector.end(), epoch: 1 };
      const summary = operationSummary(capture);
      const generation = generationSample(capture);
      eq(generation.observed, baseline.tokens); eq(generation.result.reason, baseline.reason);
      eq(generation.inputIds, [1, 2, 1, 2, 1, 2, 1, 2]);
      ok(summary.sample.n > 0); ok(summary.head.n > 0); ok(summary.host.n > 0);
      if (mode.fused) ok(summary["head-host"].n > 0);
      if (mode.speculative) ok(summary.verify.n > 0);
      if (!mode.chain) eq(correlateTiming({ R: capture }, "R", []), []);
      ok(capture.records.filter((r) => r.phase === "operation").every((r) => r.durationMs >= 0));
    });
  }
}

Deno.test("host timing: corrupt operation intervals cannot produce a valid trace", () => {
  for (const record of [{ operation: "unknown", durationMs: 1 }, { operation: "head", durationMs: -1 },
    { operation: "host", durationMs: 11 }, { operation: "sample", durationMs: 1, n: 0 }]) {
    const capture = { attempt: "test", epoch: 1, dropped: 0,
      records: [{ sequence: 0, phase: "operation", pos: 0, n: 1, at: 10, ...record }] };
    let error; try { correlateTiming({ R: capture }, "R", []); } catch (e) { error = e; }
    ok(error?.message.includes("invalid host operation"));
  }
});

Deno.test("generation evidence rejects partial, repeated, missing and reordered tokens", () => {
  const start = { phase: "generation", event: "start", ids: [1, 2], at: 1 };
  const token = { phase: "generation", event: "token", id: 7, drafted: false, at: 2 };
  const end = { phase: "generation", event: "end", result: { tokens: [7] }, at: 3 };
  for (const records of [[start, token], [start, start, token, end], [start, end],
    [start, { ...token, at: 0 }, end], [start, { ...token, id: -1 }, end],
    [start, token, { ...end, result: { tokens: [8] } }],
    [start, { phase: "generation", event: "error", error: "lost", at: 2 }]]) {
    let error; try { generationSample({ dropped: 0, records }); } catch (e) { error = e; }
    ok(error);
  }
  let error; try { generationSample({ dropped: 1, records: [start, token, end] }); } catch (e) { error = e; }
  ok(error);
});

Deno.test("wire snapshots: count units, detached values, link replacement and reset detection", () => {
  const link = makeLink(), connections = new Map([["private-id", { link }]]), read = wireSnapshot(connections, () => "S");
  const before = read(); link.sent += 2; link.recv++; link.dups += 4; link.kaSent += 3;
  const delta = wireDifference(before, read()); eq(before.S.framesSent, 0);
  eq(delta.links.S, { framesSent: 2, framesReceived: 1, duplicateSlices: 4, skippedFrames: 0,
    keepaliveMessages: 3, openChannelsBefore: 0, openChannelsAfter: 0 });
  eq(delta.wireBytes, null); eq(delta.status, "pass");
  const after = read(); link.sent = 0; eq(wireDifference(after, read()).status, "incomplete");
  connections.get("private-id").link = makeLink(); eq(wireDifference(before, read()).status, "incomplete");
  connections.clear(); eq(wireDifference(before, read()).status, "incomplete");
  eq(wireDifference(null, {}).status, "unavailable"); eq(wireDifference({}, {}).status, "pass");
});
