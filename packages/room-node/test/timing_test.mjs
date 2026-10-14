import { test } from "node:test";
import assert from "node:assert/strict";
import { RoomNode } from "../roomnode.js";
import { packWire } from "../../../room/wire.js";

test("room node passes its optional timing observer to the shared worker pipeline", async () => {
  const records = [], node = new RoomNode({ pledgeGB: 6, ckptDisk: false, onTiming: (r) => records.push(r) });
  Object.assign(node.ai, { role: "worker", hostId: "host", next: "host", range: [1, 2],
    engine: { runHidden: async (h) => h } });
  const sent = [], telemetry = []; node.sendHidden = (_, d) => sent.push(d);
  node.teleNote = (kind, ms, n) => { assert.ok(Number.isFinite(ms)); telemetry.push([kind, n]); };
  await node.aiOnData("host", { t: "ai-hidden", pos: 7, ...packWire(Float32Array.of(8)) });
  await node.ai.q;
  assert.equal(sent[0].t, "ai-hiddenret");
  assert.deepEqual(telemetry, [["one", 1]]);
  assert.equal(records.length, 1);
  assert.equal(records[0].phase, "worker");
  assert.equal(records[0].status, "forwarded");
  assert.ok(Number.isFinite(records[0].queueMs));
});

test("node ingress measures queue wait without moving disk work, and observes host returns", async () => {
  const records = [], order = [], node = new RoomNode({ ckptDisk: false, onTiming: (r) => records.push(r) });
  let release;
  node.ai.q = new Promise((resolve) => { release = resolve; });
  Object.assign(node.ai, { role: "worker", hostId: "host", next: "host", range: [1, 2],
    engine: { runHidden: async (h) => { order.push("engine"); return h; } } });
  node.sendHidden = () => order.push("send"); node.diskAfterFrame = async () => order.push("disk");
  await node.aiOnData("host", { t: "ai-hidden", pos: 8, ...packWire(Float32Array.of(9)) });
  assert.equal(records.length, 0);
  await new Promise((resolve) => setTimeout(resolve, 10)); release(); await node.ai.q;
  assert.ok(records[0].queueMs >= 5); assert.deepEqual(order, ["engine", "send", "disk"]);
  node.ai.role = "host"; node.ai.chain = ["tail"];
  const wait = node.pipeline.lapWait(9, 1000, "test");
  const reply = { t: "ai-hiddenret", pos: 9, ...packWire(Float32Array.of(10)) };
  await node.aiOnData("stranger", reply); assert.equal(records.length, 1);
  await node.aiOnData("tail", reply); assert.deepEqual([...await wait], [10]);
  assert.equal(records[1].phase, "return");
});
