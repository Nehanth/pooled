import { test } from "node:test";
import assert from "node:assert/strict";
import { RoomNode } from "../roomnode.js";
import { nodeBenchmark } from "../../../benchmarks/bench_room_node.js";
import { createTimingCollector } from "../../../room/timing.js";
import { validateApiAsk } from "../../../room/api.js";

test("node benchmark uses its room lock/reset and preserves partial failures", async () => {
  const room = new RoomNode({ name: "G1", pledgeGB: 6, ckptDisk: false }), order = [];
  room.isHost = true;
  Object.assign(room.ai, { role: "host", model: "qwen3-1.7b", range: [0, 28],
    engine: { maxSeq: 2048, reset() {} },
    tok: { vocab: { "<|im_start|>": 1, "<|im_end|>": 2, "<|endoftext|>": 3 }, encode: (s) => [...s].map((c) => c.charCodeAt(0)) } });
  room.ckptClear = (tell) => { assert.equal(tell, true); order.push("checkpoints"); };
  room.generate = async (ids, opts) => {
    assert.equal(room.ai.busy, true); assert.equal(room.ai.pos, 0); assert.deepEqual(room.ai.fed, []);
    assert.equal(opts.spec, false); order.push("generate"); opts.onToken(7, false);
    throw new Error("GPU lost after first token");
  };
  const bench = nodeBenchmark(room, { timing: createTimingCollector() });
  bench.begin("test", 1);
  const sample = await bench.run({ prompt: "hello" });
  assert.equal(sample.status, "failed"); assert.match(sample.error, /GPU lost/);
  assert.deepEqual(sample.observed, [7]); assert.equal(sample.tokenTimes.length, 1);
  assert.equal(room.ai.busy, false); assert.deepEqual(order, ["checkpoints", "generate"]);
  assert.deepEqual((await bench.end()).records, []);
  const before = bench.snapshot(); room.ai.range[1] = 14;
  assert.deepEqual(before.layers.G1, [0, 28]);
});

test("node client preparation uses normal API dispatch and restores the generation method", async () => {
  for (const mode of ["plain", "defaults", "spec-greedy"]) {
    const room = new RoomNode({ name: "G1", pledgeGB: 6, ckptDisk: false }); room.isHost = true;
    Object.assign(room.ai, { role: "host", online: true, model: "qwen3-1.7b", range: [0, 28],
      engine: { maxSeq: 2048, reset() {} },
      tok: { vocab: { "<|im_start|>": 1, "<|im_end|>": 2, "<|endoftext|>": 3 },
        encode: (s) => [...s].map((c) => c.charCodeAt(0)), decode: (ids) => String.fromCharCode(...ids) } });
    const requests = [], messages = [];
    room.generate = async (ids, options) => {
      assert.equal(room.ai.busy, true); requests.push(options); options.onToken(65, false);
      return { tokens: [65], count: 1, reused: 0, reason: "stop", stats: "fixture" };
    };
    const original = Object.getOwnPropertyDescriptor(room, "generate"), bench = nodeBenchmark(room, { timing: createTimingCollector() });
    bench.begin("client", 1);
    const prepared = await bench.prepareClient({ prompt: "hello", mode, maxNew: 400 });
    assert.ok(prepared.inputIds.length); assert.equal(room.ai.busy, false);
    const { req } = validateApiAsk({ rid: "one", messages: [{ role: "user", text: "hello" }], params: { maxTokens: 400, temperature: 0 } });
    await room.locked(() => room.runApi(req, "client", "C", (m) => messages.push(m)));
    assert.equal(requests.length, 1); assert.equal(requests[0].spec, mode === "plain" ? false : undefined);
    assert.equal(messages.at(-1).t, "ai-gendone"); assert.equal(messages.at(-1).failed, 0);
    await bench.end(); assert.deepEqual(Object.getOwnPropertyDescriptor(room, "generate"), original);
  }
});

test("node client cleanup restores an inherited generator even if the worker queue failed", async () => {
  const room = new RoomNode({ name: "G1", pledgeGB: 6, ckptDisk: false }); room.isHost = true;
  Object.assign(room.ai, { role: "host", online: true, engine: { maxSeq: 2048, reset() {} },
    tok: { vocab: { "<|im_start|>": 1, "<|im_end|>": 2 }, encode: () => [5] } });
  const generate = room.generate, bench = nodeBenchmark(room);
  await bench.prepareClient({ prompt: "hello", maxNew: 400 });
  assert.ok(Object.hasOwn(room, "generate"));
  room.ai.q = Promise.reject(new Error("worker failed"));
  await assert.rejects(bench.end(), /worker failed/);
  assert.equal(Object.hasOwn(room, "generate"), false); assert.equal(room.generate, generate);
});
