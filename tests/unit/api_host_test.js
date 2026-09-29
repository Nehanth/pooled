// room/api.js: the host's side of `pooled serve` API asks (validation, stop strings, the think
// block, exact-id reuse, context rejection) and room/sampling.js makeSampler.
import { validateApiAsk, AnswerCache, apiTurns, StopMatcher, ThinkSplit, apiPrompt, apiRun, apiSampler, API_LIMITS, pieceDecoder } from "../../room/api.js";
import { makeSampler, pickSampler } from "../../room/sampling.js";
import { buildIds, reusablePrefix } from "../../room/conversation.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

// a toy tokenizer: one id per character (1000 + code), plus the chat specials
const SPECIAL = { 1: "<|im_start|>", 2: "<|im_end|>", 3: "<|endoftext|>", 4: "<think>", 5: "</think>" };
const tok = {
  vocab: { "<|im_start|>": 1, "<|im_end|>": 2, "<|endoftext|>": 3, "<think>": 4, "</think>": 5 },
  encode: (s) => [...s].map((c) => 1000 + c.codePointAt(0)),
  decode: (ids) => ids.map((i) => SPECIAL[i] ?? String.fromCodePoint(i - 1000)).join(""),
};
const ask = (over = {}) => ({ api: 1, rid: "r1", messages: [{ role: "user", text: "hi" }], params: { maxTokens: 50 }, ...over });

Deno.test("api: validation accepts a plain ask and fills defaults", () => {
  const { req, err } = validateApiAsk(ask({ params: {} }));
  ok(!err, err);
  eq(req.params, { maxTokens: 1024, temperature: null, topK: null, stop: [], thinking: false, client: "API" });
});
Deno.test("api: validation rejects bad shapes", () => {
  const bad = (d, m) => { const v = validateApiAsk(d); ok(v.err && v.code === "bad", m + ": " + JSON.stringify(v)); };
  bad(ask({ rid: "" }), "no rid");
  bad(ask({ rid: "x".repeat(33) }), "long rid");
  bad(ask({ rid: "a b" }), "rid chars");
  bad(ask({ messages: [] }), "empty");
  bad(ask({ messages: [{ role: "tool", text: "x" }] }), "role");
  bad(ask({ messages: [{ role: "user", text: 5 }] }), "text type");
  bad(ask({ messages: [{ role: "user", text: "a" }, { role: "assistant", text: "b" }] }), "prefill last");
  bad(ask({ messages: Array.from({ length: API_LIMITS.messages + 1 }, () => ({ role: "user", text: "a" })) }), "too many");
  bad(ask({ messages: [{ role: "user", text: "a".repeat(API_LIMITS.chars + 1) }] }), "too long");
  bad(ask({ params: { temperature: 3 } }), "temperature");
  bad(ask({ params: { maxTokens: 0 } }), "maxTokens");
  bad(ask({ params: { stop: ["a", "b", "c", "d", "e"] } }), "five stops");
  bad(ask({ params: { stop: ["x".repeat(65)] } }), "long stop");
  bad(ask({ params: { stop: [""] } }), "empty stop");
  bad(ask({ system: 3 }), "system type");
});
Deno.test("api: the client label is cleaned and capped", () => {
  const { req } = validateApiAsk(ask({ params: { client: "<b>Continue</b>\n" + "x".repeat(60) } }));
  ok(!/[<>\n]/.test(req.params.client) && req.params.client.length <= 40, req.params.client);
});

Deno.test("api: same-role messages merge; the assistant's exact ids come from the cache", () => {
  const cache = new AnswerCache(8);
  cache.put("Hello there", [7, 8, 9]);
  const { turns, exact } = apiTurns(tok, [{ role: "user", text: "a" }, { role: "user", text: "b" }, { role: "assistant", text: "Hello there" }, { role: "user", text: "c" }], cache);
  eq(turns, [{ role: "user", text: "a\n\nb" }, { role: "assistant", ids: [7, 8, 9] }, { role: "user", text: "c" }]);
  eq(exact, 1);
  const miss = apiTurns(tok, [{ role: "assistant", text: "Hello there!" }, { role: "user", text: "c" }], cache);
  eq(miss.turns[0].ids, tok.encode("Hello there!"), "a changed answer is re-tokenized");
});
Deno.test("api: the answer cache keeps the last n, most recently used", () => {
  const c = new AnswerCache(2);
  c.put("a", [1]); c.put("b", [2]); c.get("a"); c.put("c", [3]);
  eq([c.get("a"), c.get("b"), c.get("c")], [[1], null, [3]]);
});

Deno.test("api: stop strings are held back across token boundaries", () => {
  const m = new StopMatcher(["END", "\n\n"]);
  let out = "";
  for (const p of ["Hel", "lo E", "N", "x ", "and E", "N", "D more"]) out += m.push(p);
  eq(out, "Hello ENx and ");
  eq(m.hit, "END");
  eq(m.flush(), "");
  const n = new StopMatcher(["\n"]);
  eq(n.push("one"), "one"); eq(n.push(" two\nthree"), " two"); eq(n.hit, "\n");
  const none = new StopMatcher(["zz"]);
  eq(none.push("abz"), "ab"); eq(none.flush(), "z", "a held tail comes out at the end");
  const first = new StopMatcher(["b", "abc"]);
  eq(first.push("xabc"), "x"); eq(first.hit, "abc", "the match that starts first wins");
});

Deno.test("api: the think block splits into reasoning and answer, streaming", () => {
  const run = (pieces, on = true) => { const s = new ThinkSplit(on); const out = []; for (const p of pieces) out.push(...s.push(p)); out.push(...s.flush()); return out; };
  const join = (parts, th) => parts.filter((p) => p.th === th).map((p) => p.text).join("");
  const a = run(["<think>", "\n", "Let me", " see.", "\n", "</", "think>", "\n\n", "The answer", " is 4."]);
  eq(join(a, true), "Let me see.");
  eq(join(a, false), "The answer is 4.");
  ok(a.findIndex((p) => !p.th) > a.findIndex((p) => p.th), "reasoning first");
  eq(run(["Straight", " answer"]), [{ th: false, text: "Straight" }, { th: false, text: " answer" }], "no think block");
  eq(join(run(["<think>", "cut off"]), true), "cut off", "an unclosed think block is all reasoning");
  eq(run(["<think>x</think>y"], false), [{ th: false, text: "<think>x</think>y" }], "off: all answer");
});

Deno.test("api: sampling descriptors (GPU sampling follows them)", () => {
  eq(makeSampler({ temp: 0 }).gpu, { kind: "greedy" });
  eq(makeSampler({ temp: 0.7, topK: 500 }).gpu, { kind: "topk", k: 64, temp: 0.7 }, "top-k clamped to 64");
  eq(makeSampler({ temp: 1, topK: 0 }).gpu, { kind: "topk", k: 40, temp: 1 });
  eq(pickSampler("exact").gpu, { kind: "greedy" });
  eq(pickSampler("focused").gpu, { kind: "topk", k: 20, temp: 0.4 });
  eq(makeSampler({ temp: 0 })(new Float32Array([0.1, 3, 2])), 1);
  const fb = pickSampler("creative");
  ok(apiSampler({ temperature: null, topK: null }, fb) === fb, "no temperature: the room's preset");
  eq(apiSampler({ temperature: null, topK: 5 }, fb).gpu, { kind: "topk", k: 5, temp: 0.8 });
  eq(apiSampler({ temperature: 0, topK: null }, fb).gpu, { kind: "greedy" });
});

Deno.test("api: a prompt over the context is refused, not trimmed", () => {
  const { req } = validateApiAsk(ask({ messages: [{ role: "user", text: "x".repeat(100) }] }));
  const p = apiPrompt(tok, req, 120, null);
  eq(p.code, "ctx");
  ok(p.n > 88 && p.max === 88 && /prompt is too long: \d+ tokens > 88 maximum/.test(p.err), JSON.stringify(p));
  ok(apiPrompt(tok, req, 400, null).ids, "fits in a bigger room");
});

// a fake roomGenerate: emits the given ids one by one, honouring the stop set, maxNew and the signal
const fakeGen = (answer, reason = "stop") => async (ids, o) => {
  let n = 0;
  for (const t of answer) {
    if (o.signal?.aborted) return { reason: "abort", reused: 0, stats: "s" };
    if (o.stop.has(t)) return { reason: "stop", reused: 0, stats: "s" };
    if (n >= o.maxNew) return { reason: "max", reused: 0, stats: "s" };
    o.onToken(t, 0); n++;
  }
  return { reason, reused: 0, stats: "s" };
};
const runAsk = async (d, answer, extra = {}) => {
  const { req } = validateApiAsk(d);
  const cache = extra.cache || new AnswerCache();
  const prompt = apiPrompt(tok, req, 4096, cache);
  const sent = [];
  const res = await apiRun({ tok, req, prompt, cache, ctxMax: 4096, fallback: pickSampler("exact"), generate: extra.gen || fakeGen(answer), send: (m) => sent.push(m) });
  return { res, sent, prompt, cache, req };
};

Deno.test("api: a run streams tokens with the rid and records exact ids", async () => {
  const answer = [...tok.encode("Hi!"), 2];
  const { res, sent, cache } = await runAsk(ask(), answer);
  eq(sent.map((m) => m.text).join(""), "Hi!");
  ok(sent.every((m) => m.t === "ai-token" && m.rid === "r1"));
  eq(res.reason, "stop"); eq(res.usage.out, 3); eq(res.text, "Hi!");
  eq(cache.get("Hi!"), tok.encode("Hi!"));
});
Deno.test("api: turn 2 resends turn 1 and its prompt extends turn 1's fed ids exactly", async () => {
  const cache = new AnswerCache();
  // the model answers "a~b" where "~" re-tokenizes differently: stand in with a raw id sequence
  const odd = [1000 + 97, 1000 + 126, 1000 + 98];
  const t1 = await runAsk(ask(), [...odd, 2], { cache });
  const fed = [...t1.prompt.ids, ...odd];   // what the caches hold after turn 1 (spec path: no <|im_end|>)
  const d2 = ask({ rid: "r2", messages: [{ role: "user", text: "hi" }, { role: "assistant", text: t1.res.text }, { role: "user", text: "more" }] });
  const { req } = validateApiAsk(d2);
  const p2 = apiPrompt(tok, req, 4096, cache);
  eq(p2.exact, 1);
  eq(reusablePrefix(fed, p2.ids), fed.length, "only the new turn is prefilled");
  eq(p2.ids, buildIds(tok, { turns: [{ role: "user", text: "hi" }, { role: "assistant", ids: odd }, { role: "user", text: "more" }] }));
});
Deno.test("api: a stop string ends the run, reported with the match", async () => {
  const { res, sent } = await runAsk(ask({ params: { maxTokens: 50, stop: ["\n"] } }), [...tok.encode("line one\nline two"), 2]);
  eq(sent.map((m) => m.text).join(""), "line one");
  eq(res.reason, "stop_seq"); eq(res.stopSeq, "\n");
  eq(res.usage.out, 9, "the stop token counted, nothing after it");
});
Deno.test("api: max_tokens reports max; the host's Stop reports abort", async () => {
  const { res } = await runAsk(ask({ params: { maxTokens: 3 } }), tok.encode("abcdef"));
  eq(res.reason, "max"); eq(res.text, "abc");
  const ac = new AbortController();
  const gen = async (ids, o) => { o.onToken(1000 + 120, 0); ac.abort(); return { reason: "abort", reused: 0, stats: "" }; };
  const r2 = await runAsk(ask(), [], { gen });
  eq(r2.res.reason, "abort");
});
Deno.test("api: thinking tags the reasoning th: 1 and keeps it out of the answer (and the cache)", async () => {
  const answer = [4, ...tok.encode("\nhmm\n"), 5, ...tok.encode("\n\nfour"), 2];
  const { res, sent, cache } = await runAsk(ask({ params: { maxTokens: 50, thinking: true } }), answer);
  eq(sent.filter((m) => m.th).map((m) => m.text).join(""), "hmm");
  eq(sent.filter((m) => !m.th).map((m) => m.text).join(""), "four");
  eq(res.text, "four"); eq(res.think, "hmm");
  eq(cache.get("four"), null);
});
Deno.test("api: a generation failure is reason error with the message", async () => {
  const gen = async (ids, o) => { o.onToken(1000 + 120, 0); throw new Error("pipeline timeout (decode)"); };
  const { res, sent } = await runAsk(ask(), [], { gen });
  eq(res.reason, "error"); eq(res.err, "pipeline timeout (decode)");
  eq(sent.map((m) => m.text).join(""), "x");
});

// a byte-level tokenizer like the real one: one id per UTF-8 byte (2000 + byte), so emoji and most
// scripts take several ids per character
const byteTok = {
  vocab: { "<|im_start|>": 1, "<|im_end|>": 2, "<|endoftext|>": 3 },
  encode: (s) => [...new TextEncoder().encode(s)].map((b) => 2000 + b),
  decode: (ids) => new TextDecoder().decode(new Uint8Array(ids.filter((i) => i >= 2000).map((i) => i - 2000))),
};
Deno.test("api: characters split across tokens arrive whole (no U+FFFD), streamed and in the text", async () => {
  const text = "🧑‍💻 👍🏽 🦀 नमस्ते ∃y 𝔘𝔫𝔦 ok";
  const d = pieceDecoder(byteTok);
  const out = byteTok.encode(text).map((id) => d.push(id)).join("") + d.flush();
  eq(out, text);
  const { req } = validateApiAsk(ask({ params: { maxTokens: 500 } }));
  const prompt = apiPrompt(byteTok, req, 4096, null);
  const sent = [], screens = [];
  const res = await apiRun({ tok: byteTok, req, prompt, ctxMax: 4096, fallback: pickSampler("exact"), generate: fakeGen([...byteTok.encode(text), 2]),
    send: (m) => sent.push(m), onPiece: (p) => screens.push(p) });
  eq(res.text, text);
  eq(sent.map((m) => m.text).join(""), text);
  ok(sent.every((m) => !m.text.includes("\uFFFD")), "a token message with U+FFFD");
  eq(screens.join(""), text);
  eq(res.usage.out, byteTok.encode(text).length, "every id counted");
});
Deno.test("api: a partial character at the very end still goes out (flushed as it is)", () => {
  const d = pieceDecoder(byteTok);
  const ids = byteTok.encode("a🦀").slice(0, -1);
  eq(ids.map((id) => d.push(id)).join(""), "a");
  eq(d.flush(), "\uFFFD");
});
