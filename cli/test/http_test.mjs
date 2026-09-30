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

// ---- v2 (docs/design/serve.md 4): negotiation, calls through an adapter, limits, keep-alives ----
import { ADAPTERS } from "../lib/http.js";
import { withDefaults } from "../lib/common.js";

// a test adapter whose requests carry tools (the real endpoints add them in their own change)
const calls = [];
ADAPTERS.push({
  api: "test", label: "test",
  routes: [{ method: "POST", path: "/test/tools" }],
  parse: (b) => withDefaults({ api: "test", stream: !!b.stream, maxTokens: 50, messages: [{ role: "user", text: String(b.q || "hi") }],
    tools: b.tools === false ? null : [{ name: "f", description: "", parameters: { type: "object" } }], toolChoice: b.required ? "required" : "auto" }),
  idFor: () => (i) => `call_${i}`,
  encoder: (req, sse) => ({ start: () => sse.write("start\n"), think() {}, text: (t) => sse.write(`text ${t}\n`), callStart: (i, id, n) => sse.write(`call ${i} ${id} ${n}\n`),
    callArgs: (i, a) => sse.write(`args ${i} ${a}\n`), callEnd: (i, a) => sse.write(`end ${i} ${a}\n`), done: (a) => sse.write(`done ${a.reason} ${a.calls.length}\n`),
    error: (e) => sse.write(`error ${e.message}\n`), keepAlive: (ahead) => sse.write(ahead == null ? "ka\n" : `queued ${ahead}\n`) }),
  final: (a) => ({ text: a.text, calls: a.calls, open: a.open, reason: a.reason }),
  after: (a) => calls.push(a.calls.length),
  error: (e) => ({ status: e.kind === "bad" ? 400 : e.kind === "toolarge" ? 413 : 500, body: { error: e.message } }),
  streamError: (e) => `error ${e.message}\n`,
});
const v2answer = (rid, h, body) => setImmediate(() => {
  h({ t: "ai-genstart", rid, promptTokens: 5, api: body.api === 2 ? 2 : undefined });
  h({ t: "ai-token", rid, text: "ok" });
  h({ t: "ai-call", rid, i: 0, name: "f" });
  h({ t: "ai-call", rid, i: 0, a: '{"x": 1}' });
  h({ t: "ai-call", rid, i: 0, end: 1 });
  h({ t: "ai-gendone", rid, api: 2, reason: "stop", usage: { in: 5, out: 3 }, calls: [{ name: "f", args: '{"x": 1}' }] });
});

test("a v2 host gets v2 asks; tools come back as calls, streamed and whole", async () => {
  const t = await start();
  t.bridge.hostMeta = { api: 2, ctx: 4096 };
  let body = null;
  t.bridge.onAsk = (rid, h, b) => { body = b; v2answer(rid, h, b); };
  const r = await t.req("POST", "/test/tools", { body: { q: "weather" } });
  assert.equal(r.status, 200, r.body);
  assert.deepEqual(JSON.parse(r.body), { text: "ok", calls: [{ id: "call_0", name: "f", args: '{"x": 1}' }], open: null, reason: "stop" });
  assert.equal(body.api, 2);
  assert.deepEqual(body.tools, [{ name: "f", description: "", parameters: { type: "object" } }]);
  const s = await t.req("POST", "/test/tools", { body: { stream: true } });
  assert.equal(s.body, 'start\ntext ok\ncall 0 call_0 f\nargs 0 {"x": 1}\nend 0 {"x": 1}\ndone stop 1\n');
  // plain chat to a v2 host is a v2 ask too
  t.bridge.onAsk = (rid, h, b) => { body = b; answer("ok", { api: 2 })(rid, (d) => h(d.t === "ai-genstart" ? { ...d, api: 2 } : d)); };
  assert.equal((await t.req("POST", "/v1/chat/completions", { body: chatBody() })).status, 200);
  assert.equal(body.api, 2);
  await t.close();
});

test("an old host: plain requests as v1, tools refused with a clear 400; a host that drops to v1 mid-way is a 500", async () => {
  const t = await start();
  t.bridge.hostMeta = { api: 1 };
  let body = null;
  t.bridge.onAsk = (rid, h, b) => { body = b; answer()(rid, h); };
  assert.equal((await t.req("POST", "/v1/chat/completions", { body: chatBody() })).status, 200);
  assert.equal(body.api, undefined, "a v1 body");
  const r = await t.req("POST", "/test/tools", { body: {} });
  assert.equal(r.status, 400);
  assert.match(JSON.parse(r.body).error, /older Pooled without tool calling/);
  t.bridge.hostMeta = { api: 2 };
  t.bridge.onAsk = (rid, h) => setImmediate(() => h({ t: "ai-genstart", rid, promptTokens: 1 }));   // no api: 2 in genstart
  const g = await t.req("POST", "/test/tools", { body: {} });
  assert.equal(g.status, 500);
  assert.match(JSON.parse(g.body).error, /changed to an older Pooled; retry/);
  assert.equal(t.bridge.stopped.length, 1, "and the room is told to stop");
  await t.close();
});

test("the legacy completions API is a 404 that points at chat completions; oversized bodies and asks are 413", async () => {
  const t = await start();
  const c = await t.req("POST", "/v1/completions", { body: { prompt: "x" } });
  assert.equal(c.status, 404);
  assert.match(JSON.parse(c.body).error.message, /legacy completions API is not served; use POST \/v1\/chat\/completions/);
  const r = await t.req("POST", "/v1/responses", { body: {} });
  assert.equal(r.status, 400, "Responses: served (cli/test/responses_test.mjs)");
  assert.match(JSON.parse(r.body).error.message, /input is required/);
  const huge = await t.req("POST", "/v1/chat/completions", { body: "x".repeat((4 << 20) + 10) });
  assert.equal(huge.status, 413);
  t.bridge.hostMeta = { api: 2 };
  const bytes = await t.req("POST", "/test/tools", { body: { q: "€".repeat(1250000) } });
  assert.equal(bytes.status, 413, "under the body cap, over what the room can take in one message");
  await t.close();
});

test("keep-alives while the room is silent mid-stream, not only while waiting", async () => {
  const t = await start({ keepAliveMs: 60 });
  t.bridge.hostMeta = { api: 2 };
  t.bridge.onAsk = (rid, h, b) => setImmediate(() => {
    h({ t: "ai-genstart", rid, promptTokens: 5, api: 2 });
    h({ t: "ai-call", rid, i: 0, name: "f" });
    setTimeout(() => { h({ t: "ai-call", rid, i: 0, a: "{}" }); h({ t: "ai-call", rid, i: 0, end: 1 }); h({ t: "ai-gendone", rid, reason: "stop", usage: { in: 5, out: 1 }, calls: [{ name: "f", args: "{}" }] }); }, 250);
  });
  const s = await t.req("POST", "/test/tools", { body: { stream: true } });
  assert.match(s.body, /call 0 call_0 f\n(ka\n)+args 0 \{\}/, s.body);
  await t.close();
});
