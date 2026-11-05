import { browserBenchmark } from "../../benchmarks/bench_room_browser.js";
import { createTimingCollector } from "../../room/timing.js";

const eq = (a, b) => { if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(`${JSON.stringify(a)} != ${JSON.stringify(b)}`); };
const ok = (v) => { if (!v) throw new Error("assertion failed"); };
async function rejects(f, match) { let error; try { await f(); } catch (e) { error = e; } ok(error?.message.includes(match)); }
function fixture({ defaults = false, generate } = {}) {
  const order = [], ai = { role: "host", readyPeers: new Set(), chain: [], waiters: new Map(),
    q: Promise.resolve(), settings: { sampling: "focused" }, pos: 8, fed: [1, 2], engine: {},
    model: "qwen3-1.7b", range: [0, 28], layersByName: { R: "0–27" } };
  const tok = { vocab: { "<|im_start|>": 1, "<|im_end|>": 2, "<|endoftext|>": 3 }, encode: (s) => [...s].map((c) => c.charCodeAt(0)) };
  const api = { name: () => "R", nameOf: (s) => s, ready: () => true, maxSeq: () => 2048, tok: () => tok,
    lock: () => { if (ai.busy) return false; ai.busy = true; order.push("lock"); return true; },
    unlock: () => { ai.busy = false; order.push("unlock"); },
    generate: async (ids, request) => {
      order.push("generate"); eq([ai.pos, ai.fed.length], [0, 0]);
      if (generate) return generate(ids, request);
      request.onToken(7, false); request.onToken(8, false);
      return { tokens: [7, 8], reused: 0, prefilled: ids.length, reason: "stop", count: 2 };
    },
  };
  const pipeline = { ckptClear: (tell) => { order.push("checkpoints"); eq(tell, true); },
    resetState: () => { order.push("reset"); ai.pos = 0; ai.fed = []; } };
  const timing = createTimingCollector();
  return { ai, api, order, timing, bench: browserBenchmark({ state: ai, pipeline, api, timing,
    options: { lookup: defaults, denseSpec: defaults } }) };
}

Deno.test("browser benchmark: normal lock/generate path, fresh prefix, tokens and timings", async () => {
  const f = fixture();
  f.bench.begin("run", 1);
  const r = await f.bench.run({ prompt: "hello", maxNew: 20 });
  eq(f.order, ["lock", "checkpoints", "reset", "generate", "unlock"]);
  eq([r.status, r.mode, r.sampling, r.observed, r.result.tokens], ["pass", "plain", "exact", [7, 8], [7, 8]]);
  eq(r.inputIds.length, r.result.prefilled);
  ok(r.ttftMs >= 0 && r.elapsedMs >= r.tokenTimes.at(-1));
  eq(await f.bench.end(), { attempt: "run", records: [], dropped: 0, epoch: 1 });
});

Deno.test("browser benchmark: rejects falsely labelled plain/default runs and always unlocks", async () => {
  const plain = fixture(), defaults = fixture({ defaults: true });
  await rejects(() => defaults.bench.run({ prompt: "hello" }), "require ?lookup=0");
  await rejects(() => plain.bench.run({ prompt: "hello", mode: "defaults" }), "require default");
  eq(defaults.order, []); eq(plain.order, []);
  eq((await defaults.bench.run({ prompt: "hello", mode: "defaults" })).sampling, "focused");
  eq((await defaults.bench.run({ prompt: "hello", mode: "spec-greedy" })).sampling, "exact");
  const broken = fixture({ generate: () => { throw new Error("GPU lost"); } });
  const failure = await broken.bench.run({ prompt: "hello" });
  eq([failure.status, failure.error], ["failed", "GPU lost"]);
  eq(broken.ai.busy, false); eq(broken.order.at(-1), "unlock");
  const overflow = await plain.bench.run({ prompt: "hello", maxNew: 10000 });
  eq(overflow.status, "failed"); ok(overflow.error.includes("exceeds context"));
  eq(plain.ai.busy, false);
});

Deno.test("browser benchmark: unexpected recovery/reuse is retained as non-equivalent", async () => {
  for (const extra of [{ resumed: 1, reused: 0 }, { reused: 3 }]) {
    const f = fixture({ generate: () => ({ tokens: [], ...extra }) });
    const r = await f.bench.run({ prompt: "hello" });
    eq(r.status, "non-equivalent"); ok(r.invalid);
  }
  const f = fixture(); f.ai.waiters.set(1, {});
  await rejects(() => f.bench.end(), "frames still outstanding");
  await rejects(() => f.bench.begin("r", 1), "busy");
});

Deno.test("browser benchmark: aborted or context-limited runs cannot count as passed", async () => {
  for (const reason of ["abort", "ctx"]) {
    const f = fixture({ generate: () => ({ tokens: [], reused: 0, reason }) });
    eq((await f.bench.run({ prompt: "hello" })).status, "failed");
  }
  await rejects(() => fixture().bench.begin("r"), "chain epoch");
});
