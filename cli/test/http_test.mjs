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
