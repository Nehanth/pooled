import { test } from "node:test";
import assert from "node:assert/strict";
import { measureRequest, pairClient } from "../../../benchmarks/bench_room_client.js";
import { validateApiAsk, apiRun, apiPrompt } from "../../../room/api.js";

const request = { prompt: "Hello", mode: "plain", maxNew: 4 };
function bridge(send) {
  return { reqs: new Map(), connected: true, ready: true, stopped: [],
    stop(rid) { this.stopped.push(rid); this.reqs.delete(rid); },
    ask(rid, body, receive) { this.body = body; this.reqs.set(rid, receive); send(receive, rid); return true; } };
}
const start = (receive, rid) => receive({ t: "ai-genstart", rid, model: "fixture", promptTokens: 3 });
const token = (receive, rid, fields = {}) => receive({ t: "ai-token", rid, text: "answer", ...fields });
const done = (receive, rid, fields = {}) => receive({ t: "ai-gendone", rid, reason: "stop", ...fields });

test("client timing retains queue delay, ignores empty/thinking pieces and labels text chunks", async () => {
  const b = bridge((receive, rid) => {
    receive({ t: "ai-queued", rid, pos: 1 }); start(receive, rid);
    token(receive, rid, { text: "" }); token(receive, rid, { text: "thought", th: 1 });
    token(receive, rid, { text: "two tokens", d: 1 }); token(receive, rid, { text: "!" }); done(receive, rid);
  });
  const r = await measureRequest(b, request, { rid: "one" });
  assert.equal(r.status, "pass"); assert.equal(r.textChunks, 2);
  assert.equal(r.ttftMs, r.events[4].at); assert.equal(r.lastTextMs, r.events[5].at);
  assert.ok(r.elapsedMs >= r.lastTextMs); assert.equal(r.tokenIds, null);
  assert.equal(b.body.params.temperature, 0); assert.equal(b.body.params.thinking, false);
  assert.deepEqual(b.body.messages, [{ role: "user", text: "Hello" }]);
});

test("client settings preserve room defaults and greedy speculation comparison", async () => {
  for (const mode of ["defaults", "spec-greedy"]) {
    const b = bridge((receive, rid) => { start(receive, rid); token(receive, rid); done(receive, rid); });
    await measureRequest(b, { ...request, mode }, { rid: mode });
    assert.equal(b.body.params.temperature, mode === "defaults" ? undefined : 0);
  }
});

test("client uses a valid normal API request, preserving prompt and output text", async () => {
  const tok = { vocab: { "<|im_start|>": 1, "<|im_end|>": 2 },
    encode: (s) => [...s].map((c) => c.charCodeAt(0)), decode: (ids) => String.fromCharCode(...ids) };
  let task, generated;
  const b = bridge(() => {});
  b.ask = (rid, body, receive) => {
    const { req, err } = validateApiAsk({ rid, ...body }); assert.equal(err, undefined);
    const prompt = apiPrompt(tok, req, 1024);
    start(receive, rid);
    task = apiRun({ tok, req, prompt, ctxMax: 1024, send: receive,
      async generate(ids, opts) {
        generated = { ids, maxNew: opts.maxNew };
        opts.onToken(65, false); opts.onToken(66, true);
        return { tokens: [65, 66], count: 2, reason: "stop", reused: 0, stats: "fixture" };
      } }).then((r) => done(receive, rid, r));
    return true;
  };
  const result = await measureRequest(b, request, { rid: "api" }); await task;
  assert.equal(result.status, "pass"); assert.equal(generated.maxNew, 4);
  assert.equal(result.events.filter((e) => e.type === "ai-token").map((e) => e.text).join(""), "AB");
  assert.equal(result.done.reused, 0);
});

test("client retains partial evidence on disconnect, failure and timeout and never retries", async () => {
  for (const terminal of [{ t: "x-fail", why: "link lost" }, { t: "ai-busy", why: "not ready" },
    { t: "ai-gendone", reason: "error", failed: 1, err: "GPU lost" }, null]) {
    let sends = 0;
    const b = bridge((receive, rid) => { sends++; start(receive, rid); token(receive, rid); if (terminal) receive({ ...terminal, rid }); });
    const r = await measureRequest(b, request, { rid: "failed", timeoutMs: 5 });
    assert.equal(r.status, "failed"); assert.equal(sends, 1); assert.equal(r.textChunks, 1);
    if (!terminal) assert.deepEqual(b.stopped, ["failed"]);
  }
});

test("client rejects unrelated responses, recovered generation and dropped evidence", async () => {
  const cases = [
    { send: (receive, rid) => token(receive, rid), status: "incomplete" },
    { send: (receive) => token(receive, "other"), status: "incomplete" },
    { send: (receive, rid) => { start(receive, rid); start(receive, rid); }, status: "non-equivalent" },
    { send: (receive, rid) => { start(receive, rid); done(receive, rid); }, status: "incomplete" },
    { send: (receive, rid) => { start(receive, rid); token(receive, rid); done(receive, rid); }, status: "incomplete", capacity: 1 },
  ];
  for (const c of cases) {
    const r = await measureRequest(bridge(c.send), request, { rid: "bad", capacity: c.capacity });
    assert.equal(r.status, c.status);
  }
});

test("client refuses unready hosts and duplicate IDs without sending", async () => {
  const b = bridge(() => assert.fail("must not send")); b.ready = false;
  assert.equal((await measureRequest(b, request, { rid: "blocked" })).status, "blocked");
  b.reqs.set("duplicate", () => {});
  await assert.rejects(measureRequest(b, request, { rid: "duplicate" }), /already in use/);
});

test("client-host pairing requires exact input, output, usage, completion and no prefix reuse", () => {
  const client = { status: "pass", ttftMs: 15, elapsedMs: 35,
    done: { reason: "stop", usage: { in: 2, out: 1 } }, events: [{ type: "ai-token", text: "A", thinking: false, at: 15 }] };
  const capture = { dropped: 0, records: [
    { phase: "generation", event: "start", ids: [1, 2], at: 100 },
    { phase: "generation", event: "token", id: 65, drafted: false, at: 110 },
    { phase: "generation", event: "end", result: { tokens: [65], reason: "stop", reused: 0 }, at: 120 },
  ] };
  const decode = (ids) => String.fromCharCode(...ids);
  const r = pairClient(client, capture, [1, 2], decode);
  assert.equal(r.clientTtftMs, 15); assert.deepEqual(r.tokenTimes, [10]); assert.equal(r.text, "A");
  assert.throws(() => pairClient(client, capture, [1, 3], decode), /prompt differs/);
  assert.throws(() => pairClient(client, capture, [1, 2], () => "B"), /text differs/);
  for (const change of [
    (c) => { c.status = "failed"; }, (c) => { c.done.usage.out = 2; }, (c) => { c.done.reason = "max"; },
  ]) {
    const c = structuredClone(client); change(c);
    assert.throws(() => pairClient(c, capture, [1, 2], decode));
  }
  const reused = structuredClone(capture); reused.records[2].result.reused = 1;
  assert.throws(() => pairClient(client, reused, [1, 2], decode), /cached prefix/);
});

test("client delivery uses exact IDs and the API's Unicode holding rule, not chunk counts", () => {
  const bytes = { 1: [65], 2: [0xf0, 0x9f], 3: [0x98, 0x8a], 4: [66] };
  const decode = (ids) => new TextDecoder().decode(Uint8Array.from(ids.flatMap((id) => bytes[id])));
  const client = { status: "pass", ttftMs: 10, elapsedMs: 50,
    done: { reason: "stop", usage: { in: 1, out: 4 } },
    events: [ ["A", 10], ["😊", 20], ["B", 40] ].map(([text, at]) => ({ type: "ai-token", text, at })) };
  const capture = { dropped: 0, records: [ { phase: "generation", event: "start", ids: [5], at: 0 },
    ...[1, 2, 3, 4].map((id) => ({ phase: "generation", event: "token", id, at: id, drafted: false })),
    { phase: "generation", event: "end", at: 5, result: { tokens: [1, 2, 3, 4], reason: "stop", reused: 0 } } ] };
  const paired = pairClient(client, capture, [5], decode);
  assert.deepEqual(paired.clientDelivery.tokenTimes, [10, 20, 20, 40]);
  assert.equal(paired.clientDelivery.tps, 100);
  const merged = structuredClone(client); merged.events = [{ type: "ai-token", text: "A😊B", at: 40 }];
  const unavailable = pairClient(merged, capture, [5], decode).clientDelivery;
  assert.equal(unavailable.tokenTimes, null); assert.equal(unavailable.tps, null);
});
