// cli/lib/http.js against a stand-in room (no PeerJS, no GPU): run with `node --test cli/test/`.
import { test } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { EventEmitter } from "node:events";
import { createServer } from "../lib/http.js";

class FakeBridge extends EventEmitter {
  constructor() { super(); this.code = "ABCD"; this.connected = true; this.ready = true; this.model = "qwen17"; this.onAsk = null; this.stopped = []; }
  ask(rid, body, h) { this.onAsk?.(rid, h, body); return true; }
  stop(rid) { this.stopped.push(rid); }
}

export async function start(opts = {}) {
  const bridge = new FakeBridge();
  const logs = [];
  const api = createServer({ bridge, port: 0, log: (m) => logs.push(m), ...opts });
  const port = await api.listen();
  const req = (method, path, { headers = {}, body = null, timeout = 3000 } = {}) => new Promise((res) => {
    const r = http.request({ host: "127.0.0.1", port, method, path, headers: { "content-type": "application/json", ...headers } }, (s) => {
      let d = ""; s.on("data", (c) => (d += c)); s.on("end", () => res({ status: s.statusCode, headers: s.headers, body: d }));
    });
    r.setTimeout(timeout, () => { r.destroy(); res({ status: "timeout" }); });
    r.on("error", (e) => res({ status: "error " + e.message }));
    if (body != null) r.write(typeof body === "string" ? body : JSON.stringify(body));
    r.end();
  });
  const close = () => new Promise((r) => { api.closeAll("test over"); api.server.closeAllConnections?.(); api.server.close(r); });
  return { bridge, api, port, req, logs, close };
}
export const chatBody = (over = {}) => ({ model: "x", messages: [{ role: "user", content: "hi" }], ...over });
export const msgBody = (over = {}) => ({ model: "x", max_tokens: 50, messages: [{ role: "user", content: "hi" }], ...over });
// a room that answers "ok" at once
export const answer = (text = "ok", extra = {}) => (rid, h) => setImmediate(() => {
  h({ t: "ai-genstart", rid, promptTokens: 5 });
  h({ t: "ai-token", rid, text });
  h({ t: "ai-gendone", rid, reason: "stop", usage: { in: 5, out: 1 }, ...extra });
});

test("--max-queue 0 answers when idle and refuses only while busy", async () => {
  const t = await start({ maxQueue: 0 });
  let hold = null;
  t.bridge.onAsk = (rid, h) => { hold = () => answer()(rid, h); };
  const first = t.req("POST", "/v1/chat/completions", { body: chatBody() });
  await new Promise((r) => setTimeout(r, 50));
  const second = await t.req("POST", "/v1/chat/completions", { body: chatBody() });
  assert.equal(second.status, 429);
  hold();
  const r1 = await first;
  assert.equal(r1.status, 200);
  t.bridge.onAsk = answer();
  assert.equal((await t.req("POST", "/v1/chat/completions", { body: chatBody() })).status, 200, "idle again: accepted");
  await t.close();
});

test("the host's Stop ends the request as an error, never as a finished answer", async () => {
  const t = await start();
  const stopped = (rid, h) => setImmediate(() => {
    h({ t: "ai-genstart", rid, promptTokens: 5 });
    h({ t: "ai-token", rid, text: "par" });
    h({ t: "ai-gendone", rid, reason: "abort", usage: { in: 5, out: 1 } });
  });
  t.bridge.onAsk = stopped;
  const a = await t.req("POST", "/v1/chat/completions", { body: chatBody() });
  assert.equal(a.status, 503);
  assert.match(JSON.parse(a.body).error.message, /host stopped this answer/);
  const s = await t.req("POST", "/v1/chat/completions", { body: chatBody({ stream: true }) });
  assert.equal(s.status, 200);
  assert.match(s.body, /"content":"par"/);
  assert.match(s.body, /host stopped this answer/);
  assert.doesNotMatch(s.body, /\[DONE\]|"finish_reason":"stop"/);
  const m = await t.req("POST", "/v1/messages", { body: msgBody({ stream: true }) });
  assert.match(m.body, /event: error/);
  assert.doesNotMatch(m.body, /message_stop|end_turn/);
  await t.close();
});

test("a full context window is model_context_window_exceeded on Anthropic, length on OpenAI", async () => {
  const t = await start();
  t.bridge.onAsk = answer("x", { reason: "ctx" });
  assert.equal(JSON.parse((await t.req("POST", "/v1/messages", { body: msgBody() })).body).stop_reason, "model_context_window_exceeded");
  assert.equal(JSON.parse((await t.req("POST", "/v1/chat/completions", { body: chatBody() })).body).choices[0].finish_reason, "length");
  await t.close();
});
