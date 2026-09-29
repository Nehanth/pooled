// cli/lib/room.js without a network: what the Bridge keeps from the host's messages.
import { test } from "node:test";
import assert from "node:assert/strict";
import { Bridge, guardChunks } from "../lib/room.js";
import { cleanText } from "../lib/common.js";

const ESC = "\x1b]52;c;ZXZpbA==\x07\x1b[2Jpwned\r\n\x9b31m";

test("text from the host reaches the terminal with no control characters", () => {
  const logs = [];
  const b = new Bridge({ code: "ABCD", name: "t", client: "c", log: (m) => logs.push(m) });
  b.onData({ t: "ai-ready-all", model: "qwen3-1.7b" + ESC, label: "Qwen3 1.7B" + ESC });
  assert.match(b.model, /^qwen3-1\.7b[\w.:+\-/=]*$/);
  assert.ok(b.modelLabel.startsWith("Qwen3 1.7B") && !/[\u0000-\u001f\u007f-\u009f]/.test(b.modelLabel), JSON.stringify(b.modelLabel));
  b.onData({ t: "bye", reason: "go away" + ESC + "x".repeat(1000) });
  assert.ok(!/[\u0000-\u001f\u007f-\u009f]/.test(b.kicked + logs.join("")), JSON.stringify(logs));
  assert.ok(b.kicked.length <= 300);
  assert.equal(cleanText("a\x1b[2Jb\u0085c"), "a[2Jbc");
});

test("chunked messages from the host are rebuilt only within limits", () => {
  const seen = [];
  const conn = { _chunkedData: {}, _handleChunk(d) { seen.push(d.n); this._chunkedData[d.__peerData] = 1; } };
  guardChunks(conn);
  conn._handleChunk({ __peerData: 1, total: 3, n: 0, data: [] });
  conn._handleChunk({ __peerData: 2, total: 1e9, n: 0, data: [] });
  conn._handleChunk({ __peerData: 3, total: 3, n: 7, data: [] });
  conn._handleChunk({ __peerData: 4, total: "3", n: 0, data: [] });
  assert.deepEqual(seen, [0]);
  for (let id = 10; id < 30; id++) conn._handleChunk({ __peerData: id, total: 2, n: 0, data: [] });
  assert.equal(Object.keys(conn._chunkedData).length, 8, "at most 8 messages rebuilt at once");
  conn._handleChunk({ __peerData: 1, total: 3, n: 1, data: [] });
  assert.equal(seen.at(-1), 1, "one already started goes on");
});
