// cli/lib/banner.js: the terminal banner and the agent settings it prints as the room's model becomes
// ready. Run with `node --test cli/test/`.
import { test } from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { showRoom, settingsFor } from "../lib/banner.js";
import { opencodeConfig } from "../lib/http.js";

class FakeBridge extends EventEmitter {
  constructor(over = {}) { super(); Object.assign(this, { ready: false, kicked: null, model: null, modelLabel: null, hostMeta: null }, over); }
}
function watch(bridge) {
  const printed = [], logs = [];
  showRoom(bridge, { code: "ABCD", port: 8081, token: null, print: (s) => printed.push(s), log: (m) => logs.push(m) });
  return { printed, logs, out: () => printed.join("\n") };
}

test("banner: no agent settings while the model is not ready, even with a ctx in the host's hello", () => {
  const b = new FakeBridge({ hostMeta: { api: 2, ctx: 2048 } });
  const w = watch(b);
  assert.match(w.out(), /room ABCD · model not ready yet/);
  assert.doesNotMatch(w.out(), /For this room|2048|model_context_window/);
  assert.equal(settingsFor(b, 8081), null);
});

test("banner: when the model becomes ready, the settings for its context are printed once", () => {
  const b = new FakeBridge({ hostMeta: { api: 2, ctx: 2048 } });
  const w = watch(b);
  w.printed.length = 0;
  Object.assign(b, { ready: true, model: "qwen36", modelLabel: "Qwen3.6 35B MoE" });
  b.hostMeta.ctx = 65536;
  b.emit("state");
  assert.deepEqual(w.logs, ["the room's model is ready: Qwen3.6 35B MoE · 65536 tokens of context"]);
  const s = w.out();
  assert.match(s, /For this room's 65536-token context:/);
  assert.match(s, /model_context_window = 65536, model_auto_compact_token_limit = 52428/);
  assert.match(s, /CLAUDE_CODE_MAX_CONTEXT_TOKENS=65536 CLAUDE_CODE_MAX_OUTPUT_TOKENS=16384/);
  assert.doesNotMatch(s, /2048/);
  const json = JSON.parse(s.split("\n").find((l) => l.trim().startsWith("{")));
  assert.deepEqual(json, opencodeConfig(65536, 8081, "pooled/qwen36", "Qwen3.6 35B MoE"));
  assert.deepEqual(json.provider.pooled.models["pooled/qwen36"], { name: "Qwen3.6 35B MoE", tool_call: true, limit: { context: 65536, output: 8192 } });
  assert.equal(json.provider.pooled.options.baseURL, "http://127.0.0.1:8081/v1");
  assert.equal(json.model, "pooled/pooled/qwen36");
  // another state change with the same context prints nothing new; a re-deal back to ready neither
  w.printed.length = 0;
  b.emit("state");
  b.ready = false; b.emit("state");
  b.ready = true; b.emit("state");
  assert.deepEqual(w.printed, []);
  // a new context: new settings
  b.hostMeta.ctx = 32768; b.emit("state");
  assert.match(w.out(), /For this room's 32768-token context:/);
});

test("banner: a room already ready at startup prints its settings in the banner", () => {
  const b = new FakeBridge({ ready: true, model: "qwen17", modelLabel: "Qwen3 1.7B", hostMeta: { ctx: 16384 } });
  const w = watch(b);
  assert.match(w.printed[0], /Qwen3 1.7B · 16384 tokens of context/);
  assert.match(w.out(), /For this room's 16384-token context:/);
  assert.match(w.out(), /"limit":\{"context":16384,"output":4096\}/);
  w.printed.length = 0;
  b.emit("state");
  assert.deepEqual(w.printed, []);
});
