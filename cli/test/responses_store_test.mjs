// cli/lib/store.js: the Responses store's bounds (count, bytes, idle TTL), LRU order, the item index
// behind item_reference, and outputMessages (a stored output as the assistant turn it was).
import { test } from "node:test";
import assert from "node:assert/strict";
import { ResponseStore, STORE_LIMITS } from "../lib/store.js";
import { outputMessages } from "../lib/responses.js";

const resp = (id, output = []) => ({ id, object: "response", output });

test("defaults: 256 responses, 64 MB, one hour idle", () => {
  assert.deepEqual(STORE_LIMITS, { responses: 256, bytes: 64 << 20, ttlMs: 3600000 });
});

test("over the count, the least recently used goes first", () => {
  const s = new ResponseStore({ max: 3 });
  for (const id of ["a", "b", "c"]) s.put({ id, response: resp(id) });
  assert.ok(s.get("a"), "a used: now the most recent");
  s.put({ id: "d", response: resp("d") });
  assert.equal(s.get("b"), null, "b was the least recently used");
  assert.deepEqual(["a", "c", "d"].map((id) => !!s.get(id)), [true, true, true]);
  assert.equal(s.size, 3);
});

test("over the byte budget, old ones go; one response bigger than the budget is not kept", () => {
  const s = new ResponseStore({ bytes: 3000 });
  const big = (id) => ({ id, response: resp(id), history: [{ role: "user", text: "x".repeat(1000) }] });
  s.put(big("a")); s.put(big("b"));
  assert.equal(s.size, 2);
  s.put(big("c"));
  assert.equal(s.get("a"), null);
  assert.ok(s.bytes <= 3000);
  assert.equal(s.put({ id: "huge", response: resp("huge"), history: [{ role: "user", text: "x".repeat(5000) }] }), false);
  assert.equal(s.get("huge"), null);
  s.delete("b"); s.delete("c");
  assert.equal(s.bytes, 0, "the byte count follows deletes exactly");
});

test("an hour without use expires a response; use keeps it", () => {
  let now = 0;
  const s = new ResponseStore({ ttlMs: 1000, now: () => now });
  s.put({ id: "a", response: resp("a") });
  s.put({ id: "b", response: resp("b") });
  now = 900; assert.ok(s.get("a"));
  now = 1500;
  assert.ok(s.get("a"), "used at 900: still fresh");
  assert.equal(s.get("b"), null, "idle since 0: gone");
  s.put({ id: "c", response: resp("c") });
  now = 3000; s.sweep();
  assert.equal(s.size, 0);
});

test("items are found by id while their response is stored", () => {
  const s = new ResponseStore();
  const out = [{ id: "msg_1", type: "message", role: "assistant", content: [{ type: "output_text", text: "hi" }] }];
  s.put({ id: "r1", response: resp("r1", out), inputItems: [{ id: "msg_0", type: "message", role: "user", content: "q" }] });
  assert.equal(s.item("msg_1").content[0].text, "hi");
  assert.equal(s.item("msg_0").content, "q");
  s.delete("r1");
  assert.equal(s.item("msg_1"), null);
  assert.equal(s.item("msg_0"), null);
});

test("outputMessages: reasoning, text and completed calls become one assistant turn", () => {
  const blob = "pooled1." + Buffer.from("exact reasoning").toString("base64url");
  assert.deepEqual(outputMessages([
    { type: "reasoning", summary: [], content: [{ type: "reasoning_text", text: "shown" }], encrypted_content: blob },
    { type: "message", role: "assistant", content: [{ type: "output_text", text: "Let me look." }] },
    { type: "function_call", call_id: "call_1", name: "f", arguments: '{"a":1}', status: "completed" },
    { type: "function_call", call_id: "call_2", name: "f", arguments: '{"a":', status: "incomplete" },
  ]), [{ role: "assistant", text: "Let me look.", calls: [{ id: "call_1", name: "f", args: { a: 1 } }], reasoning: "exact reasoning" }]);
  assert.deepEqual(outputMessages([]), []);
});
