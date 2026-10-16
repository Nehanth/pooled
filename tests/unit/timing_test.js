import { createPipeline } from "../../room/pipeline.js";
import { createTimingCollector } from "../../room/timing.js";
import { correlateTiming, deliveryRate, summarize } from "../../benchmarks/bench_room_report.js";
import { packWire } from "../../room/wire.js";

const eq = (a, b) => { if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(`${JSON.stringify(a)} != ${JSON.stringify(b)}`); };
const ok = (v, m = "assertion failed") => { if (!v) throw new Error(m); };
const fails = (f, text) => { let error; try { f(); } catch (e) { error = e; } ok(error?.message.includes(text), `expected ${text}, got ${error}`); };
const state = () => ({ chain: [], waiters: new Map(), pendingCtl: {}, dropQ: [], pos: 0, fed: [], q: Promise.resolve() });
const frame = (pos, extra = {}) => ({ t: "ai-hidden", pos, ...packWire(Float32Array.of(1)), ...extra });

Deno.test("timing collector: inactive, bounded, detached snapshots and attempt lifecycle", () => {
  const c = createTimingCollector({ capacity: 2 });
  c.record({ phase: "ignored" });
  fails(() => c.end(), "no active");
  fails(() => c.begin(""), "needs an id");
  c.begin("a");
  fails(() => c.begin("b"), "already active");
  const event = { phase: "dispatch" };
  c.record(event); event.phase = "changed";
  c.record({ phase: "return" }); c.record({ phase: "extra" });
  const a = c.end();
  eq(a, { attempt: "a", records: [{ phase: "dispatch", sequence: 0 }, { phase: "return", sequence: 1 }], dropped: 1 });
  c.begin("b"); eq(c.end(), { attempt: "b", records: [], dropped: 0 });
  eq(a.records.length, 2);
  for (const capacity of [0, -1, Infinity, 1.5]) fails(() => createTimingCollector({ capacity }), "capacity");
});

for (const profile of ["browser", "node"]) {
  Deno.test(`timing ${profile}: observer preserves frames, control order and results`, async () => {
    async function run(onTiming) {
      const ai = { ...state(), role: "worker", hostId: "host", next: "host", range: [1, 2] }, sent = [], calls = [];
      ai.engine = { dims: { dim: 1 }, NC: 4,
        reset: () => calls.push("reset"), restoreDN: () => calls.push("rollback"), saveSlot: () => calls.push("save"),
        runHidden: async (h) => { calls.push("single"); return h.map((x) => x + 1); },
        runHiddenBatch: async (h) => { calls.push("batch"); return h.map((x) => x + 1); },
      };
      const telemetry = [];
      const p = createPipeline({ state: ai, options: { profile }, hooks: { onTiming,
        teleNote(kind, ms, n) { ok(Number.isFinite(ms) && ms >= 0); telemetry.push([kind, n]); } },
        transport: { sendTo() {}, sendHidden: (_, d) => sent.push(d) } });
      for (const d of [frame(0, { reset: 1 }),
        { t: "ai-hidden-b", basePos: 1, n: 5, spec: 1, rb: 0, sv: 1, ...packWire(new Float32Array(5)) }, frame(1)]) p.handleFrame("host", d);
      await ai.q;
      eq(telemetry, [["one", 1], ["spec", 5], ["one", 1]]);
      return { sent, calls, telemetry };
    }
    const baseline = await run(null), records = [];
    eq(await run((r) => records.push(r)), baseline);
    eq(await run(() => { throw Error("observer failed"); }), baseline);
    eq(await run(() => Promise.reject(Error("observer rejected"))), baseline);
    eq(records.map((r) => [r.phase, r.kind, r.pos, r.n, r.spec, r.payloadBytes, r.status]),
      [["worker", "token", 0, 1, false, 2, "forwarded"], ["worker", "batch", 1, 5, true, 10, "forwarded"], ["worker", "token", 1, 1, false, 2, "forwarded"]]);
    for (const r of records) ok(r.queueMs >= 0 && r.serviceMs >= r.engineMs && r.engineMs >= 0);
  });

  Deno.test(`timing ${profile}: queue wait includes prior frame and engine failure is incomplete`, async () => {
    let release;
    const gate = new Promise((r) => { release = r; }), records = [], errors = [];
    const ai = { ...state(), role: "worker", hostId: "host", next: "host", range: [1, 2] };
    ai.engine = { runHidden: async (h, pos) => { if (!pos) await gate; else throw Error("engine failed"); return h; } };
    const p = createPipeline({ state: ai, options: { profile }, hooks: { onTiming: (r) => records.push(r), onError: (e) => errors.push(e.message) },
      transport: { sendTo() {}, sendHidden() {} } });
    p.handleFrame("host", frame(0)); p.handleFrame("host", frame(1));
    await Promise.resolve();
    eq(records.length, 0);
    release(); await ai.q;
    eq(errors, ["engine failed"]);
    eq(records.map((r) => r.status), ["forwarded", "incomplete"]);
    eq([records[1].engineMs, records[1].serviceMs], [null, null]);
    ok(records[1].queueMs >= records[0].engineMs, "queue did not include the blocked engine operation");
  });

  Deno.test(`timing ${profile}: Flash residual width and token ids survive observation`, async () => {
    for (const next of ["worker2", "host"]) {
      const frames = [
        { t: "ai-hidden", pos: 0, ...packWire(new Float32Array(4).fill(1), [248044]) },
        { t: "ai-hidden-b", basePos: 1, n: 3, spec: 1, ...packWire(new Float32Array(12).fill(2), [65536, 248045, 248046]) },
        { t: "ai-hidden-b", basePos: 4, n: 4, ...packWire(new Float32Array(16).fill(3), [248047, 248048, 248049, 248050]) },
      ];
      async function run(onTiming) {
        const ai = { ...state(), role: "worker", hostId: "host", next, range: [1, 2] }, sent = [], calls = [];
        const result = (kind, h, ids) => { calls.push([kind, h.length, [...ids]]); return h; };
        ai.engine = { dims: { dim: 1, hDim: 4 }, NC: 2, prefillFrame: () => 4,
          runHidden: async (h, _pos, { ids }) => result("single", h, ids),
          runHiddenBatch: async (h, _pos, _spec, { ids }) => result("batch", h, ids),
          prefillHidden: async (h, _pos, { ids }) => result("prefill", h, ids),
        };
        const p = createPipeline({ state: ai, options: { profile }, hooks: { onTiming },
          transport: { sendTo() {}, sendHidden: (to, d) => sent.push({ to, ...d }) } });
        for (const d of frames) await p.workerFrame(d);
        return { sent, calls };
      }
      const baseline = await run(null), records = [];
      eq(await run((r) => records.push(r)), baseline);
      eq(baseline.calls, [["single", 4, [248044]], ["batch", 8, [65536, 248045]],
        ["batch", 4, [248046]], ["prefill", 16, [248047, 248048, 248049, 248050]]]);
      for (let i = 0; i < frames.length; i++) {
        const n = frames[i].n || 1, ids = i === 0 ? [248044] : i === 1 ? [65536, 248045, 248046] : [248047, 248048, 248049, 248050];
        const expected = packWire(new Float32Array(n * 4).fill(i + 1), next === "host" ? null : ids);
        eq(baseline.sent[i].data, expected.data);
        eq(baseline.sent[i].to, next);
      }
      eq(records.map((r) => r.status), ["forwarded", "forwarded", "forwarded"]);
      eq(records.map((r) => r.payloadBytes), frames.map((d) => d.data.byteLength));
    }
  });
}

Deno.test("timing pipeline: concurrent prefill, repeated speculative positions and traversal correlation", async () => {
  const host = { ...state(), role: "host", chain: ["worker"] };
  const worker = { ...state(), role: "worker", hostId: "host", next: "host", range: [1, 2] };
  worker.engine = { dims: { dim: 1 }, NC: 4, runHiddenBatch: async (h) => h };
  const h = createTimingCollector(), w = createTimingCollector(); h.begin("run"); w.begin("run");
  let hp, wp;
  hp = createPipeline({ state: host, hooks: { onTiming: h.record }, transport: { sendTo() {}, sendHidden: (_, d) => wp.handleFrame("host", d) } });
  wp = createPipeline({ state: worker, options: { profile: "node" }, hooks: { onTiming: w.record },
    transport: { sendTo() {}, sendHidden: (_, d) => hp.handleFrame("worker", d) } });
  const send = (pos, spec) => {
    const wait = hp.lapWait("b" + pos, 1000, "test");
    hp.sendChain({ t: "ai-hidden-b", basePos: pos, n: 2, spec, ...packWire(Float32Array.of(1, 2)) });
    return wait;
  };
  await Promise.all([send(0, 0), send(2, 0)]);
  await send(4, 1); await send(4, 1); await worker.q;
  const captures = { host: { ...h.end(), epoch: 1 }, worker: { ...w.end(), epoch: 1 } };
  const laps = correlateTiming(captures, "host", ["worker"]);
  eq(laps.map((r) => r.frame.pos), [0, 2, 4, 4]);
  ok(laps.every((r) => r.roundTripMs >= 0 && r.workers.length === 1));
  const damaged = structuredClone(captures); damaged.worker.records.pop();
  fails(() => correlateTiming(damaged, "host", ["worker"]), "count mismatch");
  damaged.worker = { ...captures.worker, dropped: 1 };
  fails(() => correlateTiming(damaged, "host", ["worker"]), "dropped");
  damaged.worker = { ...captures.worker, epoch: 2 };
  fails(() => correlateTiming(damaged, "host", ["worker"]), "identity mismatch");
  fails(() => correlateTiming({ host: { ...captures.host, epoch: undefined } }, "host", []), "chain epoch");
  damaged.worker = structuredClone(captures.worker); damaged.worker.records[0].pos = 99;
  fails(() => correlateTiming(damaged, "host", ["worker"]), "order or status");
});

Deno.test("timing report: rejects ambiguous, duplicate, missing and incomplete observations", () => {
  const event = { kind: "token", pos: 0, n: 1, spec: false };
  const capture = (records) => ({ attempt: "r", epoch: 1, dropped: 0, records: records.map((r, sequence) => ({ ...event, at: sequence, ...r, sequence })) });
  const worker = capture([{ phase: "worker", status: "forwarded", queueMs: 0, serviceMs: 1, engineMs: 1 }]);
  const check = (records) => correlateTiming({ host: capture(records), worker }, "host", ["worker"]);
  fails(() => check([{ phase: "dispatch" }, { phase: "dispatch" }]), "ambiguous");
  fails(() => check([{ phase: "return" }]), "unmatched");
  fails(() => check([{ phase: "dispatch" }]), "incomplete");
  fails(() => check([{ phase: "dispatch" }, { phase: "return" }, { phase: "return" }]), "unmatched");
  worker.records[0].status = "incomplete";
  fails(() => check([{ phase: "dispatch" }, { phase: "return" }]), "order or status");
});

Deno.test("timing report: delivery rate handles bursts, EOS and insufficient samples", () => {
  eq(deliveryRate([10, 10, 110]), 20);
  for (const times of [[], [10], [10, 10]]) eq(deliveryRate(times), null);
  fails(() => deliveryRate([20, 10]), "invalid");
  fails(() => deliveryRate([0, NaN]), "invalid");
  eq(summarize([9, 1, 4, 2, 3]), { n: 5, median: 3, min: 1, max: 9 });
  eq(summarize([2, 4]), { n: 2, median: 3, min: 2, max: 4 });
  eq(summarize([]), { n: 0, median: null, min: null, max: null });
});
