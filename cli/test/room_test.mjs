// cli/lib/room.js without a network: what the Bridge keeps from the host's messages.
import { test } from "node:test";
import assert from "node:assert/strict";
import { Bridge } from "../lib/room.js";
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
