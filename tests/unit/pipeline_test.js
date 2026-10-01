import { createPipeline } from "../../room/pipeline.js";
import { packWire, unpackWire } from "../../room/wire.js";
import { DROP_ALL } from "../../room/transport.js";

const eq = (a, b) => { if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(`${JSON.stringify(a)} != ${JSON.stringify(b)}`); };
const ok = (v, m) => { if (!v) throw new Error(m || "assertion failed"); };
const state = () => ({ chain: [], waiters: new Map(), pendingCtl: {}, dropQ: [], pos: 0, fed: [], q: Promise.resolve() });

for (const profile of ["browser", "node"]) {
  Deno.test(`pipeline ${profile}: worker control precedes ordered batch columns and travels to the next worker`, async () => {
    const calls = [], sent = [], frames = [];
    const ai = { ...state(), role: "worker", next: "worker2", hostId: "host" };
    ai.engine = {
      dims: { dim: 1 }, NC: 4,
      restoreDN: (k) => calls.push(["rollback", k]), saveSlot: (k) => calls.push(["save", k]),
      dropSlot: (k) => calls.push(["drop", k]), reset: () => calls.push(["reset"]), loadSlot: (k) => calls.push(["load", k]),
      runHiddenBatch: async (xs, pos, spec) => { calls.push(["batch", pos, spec]); return xs.map((x) => x + 1); },
    };
    const p = createPipeline({ state: ai, options: { profile },
      transport: { sendTo() {}, sendHidden: (id, d) => sent.push([id, d]) },
      hooks: { onWorkerFrame: (ms) => frames.push(ms) } });
    const ctl = { rb: 2, sv: 3, dp: [4], reset: 1, ld: 5 };
    await p.workerFrame({ t: "ai-hidden-b", basePos: 10, n: 6, spec: 1, ...ctl, ...packWire(new Float32Array([1, 2, 3, 4, 5, 6])) });
    eq(calls, [["rollback", 2], ["save", 3], ["drop", 4], ["reset"], ["load", 5],
      ["batch", 10, { base: 0, total: 6 }], ["batch", 14, { base: 4, total: 6 }]]);
    eq(sent[0][0], "worker2");
    const d = sent[0][1];
    eq([d.t, d.basePos, d.n, d.spec, d.rb, d.sv, d.dp, d.reset, d.ld], ["ai-hidden-b", 10, 6, 1, 2, 3, [4], 1, 5]);
    eq([...unpackWire(d)], [2, 3, 4, 5, 6, 7]);
    if (profile === "node") ok(frames.length === 1 && frames[0] >= 0);
    else eq(frames, []);
    ai.next = "host";
    await p.workerFrame({ t: "ai-hidden-b", basePos: 16, n: 1, ...ctl, ...packWire(new Float32Array([8])) });
    eq([sent[1][0], sent[1][1].t, sent[1][1].sv], ["host", "ai-hiddenret-b", undefined]);
  });

  Deno.test(`pipeline ${profile}: head-ahead sends before token notification and defers the head`, async () => {
    const ai = state(), order = [];
    ai.chain = ["worker"];
    ai.engine = { headFromHidden() { throw new Error("head must be deferred"); } };
    let p;
    p = createPipeline({ state: ai, options: { profile }, transport: { sendTo() {}, sendHidden(id, d) {
      order.push(["send", id, d.pos]);
      queueMicrotask(() => p.lapDone(d.pos, new Float32Array([9])));
    } } });
    const r = await p.aiPipeToken(7, true, undefined, { kind: "greedy" }, {
      h: new Float32Array([8]), defer: true, onSent: () => order.push(["token", 7]),
    });
    eq(order, [["send", "worker", 0], ["token", 7]]);
    eq([r, ai.pos, ai.fed, [...ai.lastHidden], ai.waiters.size], [null, 1, [7], [9], 0]);
  });

  Deno.test(`pipeline ${profile}: abort awaits frames already issued and preserves their cached prefix`, async () => {
    const ai = state(), sent = [];
    let aborted = false, p;
    ai.chain = ["worker"];
    ai.engine = { dims: { dim: 1 }, NC: 4,
      embedRunBatch: async (ids) => new Float32Array(ids),
    };
    p = createPipeline({ state: ai, options: { profile }, transport: { sendTo() {}, sendHidden(id, d) {
      sent.push(d); aborted = true;
      queueMicrotask(() => p.lapDone("b" + d.basePos, unpackWire(d)));
    } } });
    const r = await p.aiPrefill(Array.from({ length: 21 }, (_, i) => i), { aborted: () => aborted });
    eq([r, sent.length, ai.pos, ai.fed.length, ai.waiters.size], [null, 1, 16, 16, 0]);
    eq(ai.fed, Array.from({ length: 16 }, (_, i) => i));
  });
}

Deno.test("pipeline node: checkpoint drops drain two per frame and DROP_ALL cancels queued drops", () => {
  const ai = state(), sent = [];
  ai.chain = ["worker"]; ai.dropQ = [1, 2, 3, 4, 5]; ai.pendingCtl = { sv: 8, rb: 2 };
  const p = createPipeline({ state: ai, options: { profile: "node" }, transport: { sendTo() {}, sendHidden: (id, d) => sent.push(d) } });
  p.sendChain({ t: "ai-hidden", pos: 0 });
  p.sendChain({ t: "ai-hidden", pos: 1 });
  eq(sent.map((d) => [d.dp, d.sv, d.rb]), [[[1, 2], 8, 2], [[3, 4], undefined, undefined]]);
  ai.pendingCtl = { dp: [DROP_ALL] };
  p.sendChain({ t: "ai-hidden", pos: 2 });
  eq([sent[2].dp, ai.dropQ, ai.frames, ai.hostAmax], [[DROP_ALL], [], 3, undefined]);
});

Deno.test("pipeline browser: dense engines with hostCkpt false do not start host checkpointing", () => {
  const ai = state(); ai.fed = [1];
  ai.engine = { hostCkpt: false, saveSlot() { throw new Error("disabled checkpoint"); } };
  const p = createPipeline({ state: ai, transport: { sendTo() {}, sendHidden() {} } });
  eq(p.ckptEngine(), false);
  p.ckptSave(true);
  eq(ai.ckpt, undefined);
});

Deno.test("pipeline profiles: keep current tail-prefill validation behavior", async () => {
  for (const profile of ["browser", "node"]) {
    const ai = state(); ai.chain = ["worker"];
    ai.engine = { dims: { dim: 1 }, specStep() {},
      embedRunBatch: async (ids) => new Float32Array(ids), headFromHiddenIds: async () => ({ bad: true }),
    };
    let p;
    p = createPipeline({ state: ai, options: { profile }, transport: { sendTo() {}, sendHidden(id, d) {
      queueMicrotask(() => p.lapDone("b" + d.basePos, unpackWire(d)));
    } } });
    let error, r;
    try { r = await p.aiPrefill([1, 2], { aborted: () => false, desc: { kind: "greedy" } }); } catch (e) { error = e; }
    if (profile === "browser") ok(error?.message.startsWith("NaN in logits"));
    else eq(r, { bad: true });
    eq(ai.waiters.size, 0);
  }
});

Deno.test("pipeline: a failed worker frame reports once and the next queued frame still runs", async () => {
  const ai = { ...state(), role: "worker", next: "host", hostId: "host" }, sent = [], errors = [];
  ai.engine = { runHidden: async (h, pos) => { if (!pos) throw new Error("broken frame"); return h; } };
  const p = createPipeline({ state: ai, options: { profile: "node" },
    transport: { sendTo: (id, d) => sent.push(d), sendHidden: (id, d) => sent.push(d) },
    hooks: { onError: (e) => errors.push(e.message) } });
  for (const pos of [0, 1]) p.handleFrame({ t: "ai-hidden", pos, ...packWire(new Float32Array([2])) });
  await ai.q;
  eq(errors, ["broken frame"]);
  eq(sent.map((d) => [d.t, d.message, d.pos]), [["ai-error", "broken frame", undefined], ["ai-hiddenret", undefined, 1]]);
  eq(p.handleFrame({ t: "ai-chat" }), false);
});

Deno.test("pipeline profiles: a replaced engine stays captured in node calls and remains live in browser calls", async () => {
  for (const profile of ["browser", "node"]) {
    const ai = { ...state(), next: "host", hostId: "host" }, calls = [];
    const makeEngine = (name) => ({
      dims: { dim: 1 }, NC: 4, specStep() {},
      runHiddenBatch: async (h) => { calls.push(name); ai.engine = replacement; return h; },
      embedRun: async () => { ai.engine = replacement; return new Float32Array([1]); },
      embedRunBatch: async (ids) => { calls.push(name); ai.engine = replacement; return new Float32Array(ids); },
      headFromHidden: async () => { calls.push(name); return new Float32Array([1]); },
    });
    const original = makeEngine("original"), replacement = makeEngine("replacement");
    ai.engine = original;
    let p;
    p = createPipeline({ state: ai, options: { profile }, transport: { sendTo() {}, sendHidden(id, d) {
      if (d.t === "ai-hidden-b") queueMicrotask(() => p.lapDone("b" + d.basePos, unpackWire(d)));
    } } });
    const expected = profile === "node" ? "original" : "replacement";
    await p.workerFrame({ t: "ai-hidden-b", basePos: 0, n: 5, ...packWire(new Float32Array(5)) });
    eq(calls.splice(0), ["original", expected]);
    ai.engine = original;
    await p.aiPipeToken(1);
    eq(calls.splice(0), [expected]);
    ai.engine = original; ai.chain = ["worker"];
    await p.aiPrefill([1, 2, 3, 4, 5, 6], { aborted: () => false });
    eq(calls, ["original", expected, expected]);
  }
});

Deno.test("pipeline node: browser abort and observer hooks do not affect node execution", async () => {
  const ai = { ...state(), abort: true, next: "host", hostId: "host" };
  const unexpected = () => { throw new Error("browser observer called"); };
  ai.engine = { dims: { dim: 1 }, runHidden: async (h) => h,
    embedRun: async () => new Float32Array([1]), headFromHidden: async () => new Float32Array([2]),
  };
  const p = createPipeline({ state: ai, options: { profile: "node" },
    transport: { sendTo() {}, sendHidden() {} },
    hooks: { onStatus: unexpected, teleNote: unexpected, computePass: unexpected, keepWarm: unexpected } });
  eq([...await p.aiPrefill([1])], [2]);
  await p.workerFrame({ t: "ai-hidden", pos: 0, ...packWire(new Float32Array([1])) });
});
