import { setImmediate } from "node:timers";
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
  for (const pos of [0, 1]) p.handleFrame("host", { t: "ai-hidden", pos, ...packWire(new Float32Array([2])) });
  await ai.q;
  eq(errors, ["broken frame"]);
  eq(sent.map((d) => [d.t, d.message, d.pos]), [["ai-error", "broken frame", undefined], ["ai-hiddenret", undefined, 1]]);
  eq(p.handleFrame("host", { t: "ai-chat" }), false);
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


Deno.test("pipeline: only the current chain tail can resolve returned frames", async () => {
  for (const type of ["ai-hiddenret", "ai-hiddenret-b"]) {
    const ai = { ...state(), role: "host", chain: ["first", "tail"] };
    const p = createPipeline({ state: ai, transport: { sendTo() {}, sendHidden() {} } });
    const key = type === "ai-hiddenret" ? 3 : "b3";
    const frame = { t: type, pos: 3, basePos: 3, ...packWire(Float32Array.of(7)) };
    const returned = p.lapWait(key, 1000, "test");
    returned.catch(() => {});
    try {
      for (const from of ["first", "stranger", undefined]) {
        eq(p.handleFrame(from, frame), false);
        ok(ai.waiters.has(key), "untrusted reply consumed waiter");
      }
      ai.chain = [];
      eq(p.handleFrame(undefined, frame), false);
      ok(ai.waiters.has(key), "empty chain consumed waiter");
      ai.chain = ["replacement"];
      eq(p.handleFrame("tail", frame), false);
      ok(ai.waiters.has(key), "old tail consumed waiter");
      eq(p.handleFrame("replacement", frame), true);
      eq([...await returned], [7]);
      eq(ai.waiters.size, 0);
    } finally { p.failWaiters(new Error("test finished")); }
  }
});

Deno.test("pipeline: activation frames require the worker role", async () => {
  for (const profile of ["browser", "node"]) for (const type of ["ai-hidden", "ai-hidden-b"]) {
    const calls = [], sent = [];
    const ai = { ...state(), role: "host", hostId: "host", next: "host", range: [0, 1], engine: {
      dims: { dim: 1 }, runHidden: async (h) => { calls.push("one"); return h; },
      runHiddenBatch: async (h) => { calls.push("batch"); return h; },
    } };
    const p = createPipeline({ state: ai, options: { profile }, transport: {
      sendTo() {}, sendHidden: (_, d) => sent.push(d),
    } });
    const frame = { t: type, n: 1, pos: 0, basePos: 0, ...packWire(Float32Array.of(1)) };
    for (const role of ["host", null]) {
      ai.role = role;
      const before = ai.q;
      p.handleFrame("host", frame);
      await ai.q;
      ok(before === ai.q, "non-worker queued activation work");
      eq([calls, sent], [[], []]);
    }
    ai.role = "worker";
    p.handleFrame("host", frame);
    await ai.q;
    eq(calls, [type === "ai-hidden" ? "one" : "batch"]);
    eq(sent.length, 1);
  }
});

// Drain queued promise work to an event-loop boundary, without a timing-based delay.
const nextTurn = () => new Promise((resolve) => setImmediate(resolve));
for (const [window, expected] of [[undefined, 6], [NaN, 6], [Infinity, 6], [-Infinity, 6], [0, 1], [-2, 1], [1, 1], [2.9, 2], [3, 3]]) {
  Deno.test(`pipeline: prefill window ${String(window)} bounds outstanding frames at ${expected}`, async () => {
    const ai = { ...state(), role: "host", chain: ["worker"] }, sent = [], pending = [];
    ai.engine = { dims: { dim: 1 }, NC: 4,
      embedRunBatch: async (ids) => Float32Array.from(ids),
      embedRun: async (id) => Float32Array.of(id), headFromHidden: async (h) => h,
    };
    const p = createPipeline({ state: ai, options: { prefillWindow: window }, transport: {
      sendTo() {}, sendHidden(_, d) { sent.push(d); pending.push(d); },
    } });
    const ids = Array.from({ length: 16 * (expected + 3) + 1 }, (_, i) => i);
    let done = false, error;
    const work = p.aiPrefill(ids).then(() => { done = true; }, (e) => { done = true; error = e; });
    const release = () => {
      const d = pending.shift();
      p.handleFrame("worker", { ...d, t: d.t === "ai-hidden-b" ? "ai-hiddenret-b" : "ai-hiddenret" });
    };
    try {
      await nextTurn();
      eq(sent.length, expected);
      eq(pending.length, expected);
      release();
      await nextTurn();
      eq(sent.length, expected + 1);
      eq(pending.length, expected);
      for (let turns = 0; !done && turns < ids.length; turns++) {
        ok(pending.length <= expected, "prefill exceeded its window");
        if (pending.length) release();
        await nextTurn();
      }
      ok(done && !error, error?.message || "prefill did not complete");
      eq(ai.fed, ids);
      eq(ai.waiters.size, 0);
    } finally { p.failWaiters(new Error("test finished")); await work; }
  });
}

for (const profile of ["browser", "node"]) {
  Deno.test(`pipeline ${profile}: wide worker prefill never replaces speculative passes`, async () => {
    const ai = { ...state(), role: "worker", hostId: "host", next: "host" }, calls = [];
    ai.engine = { dims: { dim: 1 }, NC: 4, prefillFrame: () => 8,
      prefillHidden: async (h) => { calls.push("wide"); return h; },
      runHiddenBatch: async (h, pos, spec) => { calls.push([pos, spec]); return h; },
    };
    const p = createPipeline({ state: ai, options: { profile }, transport: { sendTo() {}, sendHidden() {} } });
    const frame = { t: "ai-hidden-b", n: 8, basePos: 0, ...packWire(new Float32Array(8)) };
    await p.workerFrame(frame);
    eq(calls.splice(0), ["wide"]);
    await p.workerFrame({ ...frame, spec: 1 });
    eq(calls, [[0, { base: 0, total: 8 }], [4, { base: 4, total: 8 }]]);
  });

  Deno.test(`pipeline ${profile}: node wide-frame override preserves browser host framing`, async () => {
    for (const override of ["8", "0"]) {
      const ai = { ...state(), role: "host", chain: ["worker"] }, sent = [], wide = [];
      ai.engine = { dims: { dim: 1 }, NC: 4, prefillFrame: () => 32,
        prefillHidden: async (ids) => { wide.push(ids.length); return Float32Array.from(ids); },
        embedRunBatch: async (ids) => Float32Array.from(ids), embedRun: async (id) => Float32Array.of(id),
        headFromHidden: async (h) => h,
      };
      let p;
      p = createPipeline({ state: ai, options: { profile }, hooks: { prefillFrame: () => override }, transport: {
        sendTo() {}, sendHidden(_, d) {
          if (d.n) sent.push(d.n);
          queueMicrotask(() => p.handleFrame("worker", { ...d, t: d.n ? "ai-hiddenret-b" : "ai-hiddenret" }));
        },
      } });
      const ids = Array.from({ length: 33 }, (_, i) => i);
      await p.aiPrefill(ids);
      eq(sent, profile === "node" && override !== "0" ? [8, 8, 8, 8] : [16, 16]);
      eq(wide, profile === "node" && override !== "0" ? [8, 8, 8, 8] : []);
      eq(ai.fed, ids);
    }
  });

  Deno.test(`pipeline ${profile}: wide draft fills retain caller batch policy`, () => {
    const ai = state(), batches = [], singles = [];
    ai.engine = { mtp: {}, dims: { dim: 1 }, NC: 4, B: { x: { buf: {}, stride: 4 } },
      device: { queue: { writeBuffer() {} } },
      _mtpFillBatch: (_, at, pos, n) => batches.push([at, pos, n]), setHidden() {},
      mtpRun: (_, next, pos) => singles.push([next, pos]),
    };
    const p = createPipeline({ state: ai, options: { profile }, transport: { sendTo() {}, sendHidden() {} } });
    p.fillDrafts(new Float32Array(9), Array.from({ length: 10 }, (_, i) => i), 0, 100, 9);
    eq(batches, profile === "node" ? [[0, 100, 4], [4, 104, 4]] : []);
    eq(singles, profile === "node" ? [[9, 109]] : Array.from({ length: 9 }, (_, i) => [i + 1, 101 + i]));
  });
}
