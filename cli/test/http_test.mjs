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

test("with a token, /health tells a caller without it only that it is up", async () => {
  const t = await start({ token: "s3cret" });
  assert.deepEqual(JSON.parse((await t.req("GET", "/health")).body), { ok: true });
  assert.deepEqual(JSON.parse((await t.req("GET", "/health", { headers: { authorization: "Bearer nope" } })).body), { ok: true });
  const full = JSON.parse((await t.req("GET", "/health", { headers: { authorization: "Bearer s3cret" } })).body);
  assert.equal(full.room, "ABCD");
  assert.equal((await t.req("GET", "/v1/models")).status, 401);
  assert.equal((await t.req("GET", "/v1/models", { headers: { "x-api-key": "s3cre" } })).status, 401);
  assert.equal((await t.req("GET", "/v1/models", { headers: { "x-api-key": "s3cret" } })).status, 200);
  await t.close();
  const open = await start();
  assert.equal(JSON.parse((await open.req("GET", "/health")).body).room, "ABCD", "no token: everything is open anyway");
  await open.close();
});

test("the room dropping the running request (ai-busy gone) answers the client instead of hanging", async () => {
  const t = await start();
  t.bridge.onAsk = (rid, h) => setImmediate(() => h({ t: "ai-busy", rid, code: "gone" }));
  const r = await t.req("POST", "/v1/chat/completions", { body: chatBody() });
  assert.equal(r.status, 503);
  assert.match(JSON.parse(r.body).error.message, /dropped the request/);
  t.bridge.onAsk = answer();
  assert.equal((await t.req("POST", "/v1/chat/completions", { body: chatBody() })).status, 200, "the next one runs");
  await t.close();
});

test("a request the room goes quiet on fails with 504 and frees the line; a host-queued one waits longer", async () => {
  const t = await start({ idleMs: 150, hostQueuedMs: 600 });
  t.bridge.onAsk = () => {};   // taken, never answered
  const t0 = Date.now();
  const r = await t.req("POST", "/v1/chat/completions", { body: chatBody() });
  assert.equal(r.status, 504);
  assert.ok(Date.now() - t0 < 1000);
  assert.equal(t.bridge.stopped.length, 1, "the room is told to stop it");
  const m = await t.req("POST", "/v1/messages", { body: msgBody({ stream: true }) });
  assert.equal(m.status, 504, "nothing streamed yet: a real status");
  assert.equal(JSON.parse(m.body).error.type, "timeout_error");
  // queued at the host: ai-queued arms the long timer, then the answer comes after the short one
  t.bridge.onAsk = (rid, h) => { setImmediate(() => h({ t: "ai-queued", rid, pos: 1 })); setTimeout(() => answer()(rid, h), 350); };
  assert.equal((await t.req("POST", "/v1/chat/completions", { body: chatBody() })).status, 200);
  await t.close();
});

test("a known path with the wrong method is 405 with Allow; an unknown one 404", async () => {
  const t = await start();
  const g = await t.req("GET", "/v1/chat/completions");
  assert.equal(g.status, 405);
  assert.equal(g.headers.allow, "POST");
  assert.equal(JSON.parse(g.body).error.code, "method_not_allowed");
  const m = await t.req("GET", "/v1/messages", { headers: { "anthropic-version": "2023-06-01" } });
  assert.equal(m.status, 405);
  assert.equal(JSON.parse(m.body).type, "error");
  assert.equal((await t.req("POST", "/v1/models")).status, 405);
  assert.equal((await t.req("GET", "/v1/nope")).status, 404);
  await t.close();
});

test("a room that floods tokens, sends non-text or odd usage cannot blow up the response", async () => {
  const t = await start();
  t.bridge.onAsk = (rid, h) => setImmediate(() => {
    h({ t: "ai-genstart", rid, promptTokens: "x" });
    h({ t: "ai-token", rid, text: { o: 1 } });
    h({ t: "ai-token", rid, text: "y".repeat(5000) });
    for (let i = 0; i < 200000; i++) h({ t: "ai-token", rid, text: "A" });
    h({ t: "ai-gendone", rid, reason: "stop", usage: { in: "1", out: "2" } });
  });
  const r = await t.req("POST", "/v1/chat/completions", { body: chatBody({ max_tokens: 3 }) });
  const j = JSON.parse(r.body);
  assert.equal(j.choices[0].message.content, "A".repeat(3 + 16), "max_tokens plus the slack, then cut");
  assert.equal(j.choices[0].finish_reason, "length");
  assert.deepEqual(j.usage, { prompt_tokens: 0, completion_tokens: 19, total_tokens: 19 });
  assert.ok(t.bridge.stopped.length === 1, "the room was told to stop");

  t.bridge.onAsk = (rid, h) => setImmediate(() => {
    h({ t: "ai-genstart", rid, promptTokens: 9 });
    h({ t: "ai-token", rid, text: "ok" });
    h({ t: "ai-gendone", rid, reason: "weird", stopSeq: "nope", usage: { in: "12", out: -4 }, reused: 1e9 });
  });
  const m = JSON.parse((await t.req("POST", "/v1/messages", { body: msgBody({ stop_sequences: ["END"] }) })).body);
  assert.equal(m.stop_reason, "end_turn");
  assert.equal(m.stop_sequence, null);
  assert.deepEqual(m.usage, { input_tokens: 0, output_tokens: 0, cache_read_input_tokens: 12 });
  await t.close();
});
