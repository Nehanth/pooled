// room/conversation.js edge cases: the chat template on unusual tokenizers, the context budget at
// its limits, the reuse test the room's caches depend on, and splitting think blocks.
import { buildIds, fitContext, reusablePrefix, splitThink, specials } from "../../room/conversation.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const throws = (f, re, m) => {
  let err = null; try { f(); } catch (e) { err = e; }
  if (!err) throw new Error((m || "expected a throw") + ": nothing thrown");
  if (re && !re.test(err.message)) throw new Error((m || "wrong error") + ": " + err.message);
  return err;
};

// one id per character (1000 + code point), plus whichever specials a case gives it
const mkTok = (vocab) => ({ vocab, encode: (s) => [...s].map((c) => 1000 + c.codePointAt(0)) });
const FULL = { "<|im_start|>": 1, "<|im_end|>": 2, "<|endoftext|>": 3, "<think>": 4, "</think>": 5 };
const tok = mkTok(FULL);
const E = (s) => tok.encode(s);
const NL = E("\n");

// ---------------------------------------------------------------------------------------------
// specials / buildIds

Deno.test("specials: a tokenizer without chat tokens is refused with a readable message", () => {
  for (const vocab of [{}, { "<|im_start|>": 1 }, { "<|im_end|>": 2 }, { "<|im_start|>": "1", "<|im_end|>": 2 }]) {
    throws(() => specials(mkTok(vocab)), /no chat tokens/, JSON.stringify(vocab));
    throws(() => buildIds(mkTok(vocab), { turns: [{ role: "user", text: "hi" }] }), /no chat tokens/);
  }
  eq(specials(mkTok({ "<|im_start|>": 0, "<|im_end|>": 7 })).imStart, 0, "id 0 is a valid id");
});

Deno.test("buildIds: without think tokens there is no think block to pre-close", () => {
  const cases = [
    { vocab: { "<|im_start|>": 1, "<|im_end|>": 2 } },
    { vocab: { "<|im_start|>": 1, "<|im_end|>": 2, "<think>": 4 } },         // half a pair is no pair
    { vocab: { "<|im_start|>": 1, "<|im_end|>": 2, "</think>": 5 } },
  ];
  for (const c of cases) {
    const t = mkTok(c.vocab);
    for (const thinking of [false, true]) {
      const ids = buildIds(t, { turns: [{ role: "user", text: "hi" }], thinking });
      eq(ids, [1, ...E("user\nhi"), 2, ...NL, 1, ...E("assistant\n")], JSON.stringify(c.vocab) + " thinking=" + thinking);
    }
  }
});

Deno.test("buildIds: the think block is pre-closed on every user turn, not only the first", () => {
  const ids = buildIds(tok, { turns: [
    { role: "user", text: "a" }, { role: "assistant", ids: [50] }, { role: "user", text: "b" },
  ] });
  const closed = [4, ...E("\n\n"), 5, ...E("\n\n")];
  const count = (arr, sub) => { let n = 0; for (let i = 0; i + sub.length <= arr.length; i++) if (sub.every((x, j) => arr[i + j] === x)) n++; return n; };
  eq(count(ids, closed), 2);
  eq(count(buildIds(tok, { turns: [{ role: "user", text: "a" }, { role: "assistant", ids: [50] }, { role: "user", text: "b" }], thinking: true }), closed), 0);
});

Deno.test("buildIds: system prompt first, assistant ids verbatim, open turns get no end token", () => {
  const cases = [
    { name: "system + one question", system: "S", turns: [{ role: "user", text: "q" }],
      want: [1, ...E("system\nS"), 2, ...NL, 1, ...E("user\nq"), 2, ...NL, 1, ...E("assistant\n"), 4, ...E("\n\n"), 5, ...E("\n\n")] },
    { name: "an empty system prompt adds nothing", system: "", turns: [{ role: "user", text: "q" }],
      want: [1, ...E("user\nq"), 2, ...NL, 1, ...E("assistant\n"), 4, ...E("\n\n"), 5, ...E("\n\n")] },
    { name: "open assistant turn last (Continue)", turns: [{ role: "user", text: "q" }, { role: "assistant", ids: [7, 8], open: true }], thinking: true,
      want: [1, ...E("user\nq"), 2, ...NL, 1, ...E("assistant\n"), 7, 8] },
    { name: "assistant ids may hold specials (a think block the model wrote)", turns: [{ role: "user", text: "q" }, { role: "assistant", ids: [4, 9, 5, 7] }, { role: "user", text: "r" }], thinking: true,
      want: [1, ...E("user\nq"), 2, ...NL, 1, ...E("assistant\n"), 4, 9, 5, 7, 2, ...NL, 1, ...E("user\nr"), 2, ...NL, 1, ...E("assistant\n")] },
    { name: "an empty user message still gets its header", turns: [{ role: "user", text: "" }], thinking: true,
      want: [1, ...E("user\n"), 2, ...NL, 1, ...E("assistant\n")] },
  ];
  for (const c of cases) eq(buildIds(tok, c), c.want, c.name);
});

Deno.test("buildIds: an assistant turn's ids are copied, never re-tokenized or mutated", () => {
  const answer = [4, 5, 999999];
  const turns = [{ role: "user", text: "q" }, { role: "assistant", ids: answer }, { role: "user", text: "r" }];
  const ids = buildIds(tok, { turns });
  ok(ids.includes(999999));
  eq(answer, [4, 5, 999999]);
});

Deno.test("buildIds: an invalid id anywhere is refused", () => {
  const bad = { vocab: FULL, encode: (s) => (s.startsWith("user") ? [NaN] : E(s)) };
  throws(() => buildIds(bad, { turns: [{ role: "user", text: "q" }] }), /invalid token id/);
  throws(() => buildIds(tok, { turns: [{ role: "user", text: "q" }, { role: "assistant", ids: [1.5] }, { role: "user", text: "r" }] }), /invalid token id/);
  throws(() => buildIds(tok, { turns: [{ role: "user", text: "q" }, { role: "assistant", ids: [undefined] }, { role: "user", text: "r" }] }), /invalid token id/);
});

// ---------------------------------------------------------------------------------------------
// fitContext

const U = (text) => ({ role: "user", text }), A = (n) => ({ role: "assistant", ids: new Array(n).fill(9) });

Deno.test("fitContext: table of budgets", () => {
  const turns = [U("a".repeat(20)), A(20), U("b".repeat(20)), A(20), U("c".repeat(20)), A(20), U("q")];
  const len = (t, system = "") => buildIds(tok, { system, turns: t }).length;
  const L = [0, 2, 4, 6].map((i) => len(turns.slice(i)));   // after dropping 0..3 exchanges
  const cases = [
    { maxSeq: L[0] + 16, reserve: 16, dropped: 0 },         // exactly fits: ids.length == maxSeq - reserve
    { maxSeq: L[0] + 15, reserve: 16, dropped: 1 },         // one token over
    { maxSeq: L[1] + 16, reserve: 16, dropped: 1 },
    { maxSeq: L[2] + 16, reserve: 16, dropped: 2 },
    { maxSeq: L[3] + 16, reserve: 16, dropped: 3 },         // only the question is left
    { maxSeq: L[3] + 15, reserve: 16, throws: true },
    { maxSeq: 1 << 20, reserve: 1 << 20, throws: true },   // the reserve eats the whole context
  ];
  for (const c of cases) {
    const what = JSON.stringify(c);
    if (c.throws) { throws(() => fitContext(tok, { turns }, c.maxSeq, c.reserve), /Shorten/, what); continue; }
    const r = fitContext(tok, { turns }, c.maxSeq, c.reserve);
    eq(r.dropped, c.dropped, what);
    eq(r.turns, turns.slice(2 * c.dropped), what);
    eq(r.ids, buildIds(tok, { turns: r.turns }), what);
    ok(r.ids.length <= c.maxSeq - c.reserve, what);
    eq(r.turns[r.turns.length - 1], U("q"), "the question is never dropped");
  }
});

Deno.test("fitContext: a system prompt that alone overflows is refused, and is never dropped to make room", () => {
  const system = "s".repeat(200);
  const turns = [U("a"), A(3), U("q")];
  const err = throws(() => fitContext(tok, { system, turns }, 128, 16), /Shorten/);
  ok(/context is 128 tokens/.test(err.message), err.message);
  // it keeps the system prompt when it does fit: the exchanges go first
  const need = buildIds(tok, { system, turns: [U("q")] }).length;
  const r = fitContext(tok, { system, turns }, need + 16, 16);
  eq(r.dropped, 1);
  ok(r.ids.slice(0, 1 + 7).every((t, i) => t === [1, ...E("system\n")][i]), "system prompt still first");
});

Deno.test("fitContext: a Continue (open assistant turn last) keeps the open answer with its question", () => {
  const turns = [U("a".repeat(30)), A(30), U("q"), { role: "assistant", ids: [7, 8, 9], open: true }];
  const need = buildIds(tok, { turns: turns.slice(2) }).length;
  const r = fitContext(tok, { turns }, need + 4, 4);
  eq(r.dropped, 1);
  eq(r.turns, turns.slice(2));
  eq(r.ids.slice(-3), [7, 8, 9], "left open: no end token after it");
  throws(() => fitContext(tok, { turns }, need + 3, 4), /Shorten/, "the open answer is never cut off to make room");
});

Deno.test("fitContext: a history that starts with an answer (no question before it) drops it first", () => {
  const turns = [A(50), U("q")];
  const need = buildIds(tok, { turns: [U("q")] }).length;
  const r = fitContext(tok, { turns }, need + 1, 1);
  eq(r.dropped, 1); eq(r.turns, [U("q")]);
});

Deno.test("fitContext: dropping changes the prefix, so the caches are never reused across a drop", () => {
  const turns = [U("a"), A(3), U("b"), A(3), U("c")];
  const before = fitContext(tok, { system: "S", turns: turns.slice(0, 3) }, 1 << 20, 0).ids;   // what the caches held
  const all = buildIds(tok, { system: "S", turns }).length;
  const after = fitContext(tok, { system: "S", turns }, all - 1 + 8, 8);
  eq(after.dropped, 1);
  eq(reusablePrefix(before, after.ids), 0);
  // and without the drop the old prompt is a strict prefix of the new one... up to the answer
  const grown = [...before, ...new Array(3).fill(9), 2, ...NL];
  eq(reusablePrefix(grown, buildIds(tok, { system: "S", turns })), grown.length);
});

Deno.test("fitContext: does not mutate the caller's turns", () => {
  const turns = [U("a".repeat(40)), A(40), U("q")];
  const copy = JSON.stringify(turns);
  fitContext(tok, { turns }, 30, 4);
  eq(JSON.stringify(turns), copy);
});

// ---------------------------------------------------------------------------------------------
// reusablePrefix

Deno.test("reusablePrefix: table", () => {
  const cases = [
    { fed: [1, 2, 3], ids: [1, 2, 3], want: 0, why: "fed == ids: the last token must still run through the head" },
    { fed: [1, 2, 3], ids: [1, 2, 3, 4], want: 3 },
    { fed: [1], ids: [1, 2], want: 1 },
    { fed: [1, 2, 3, 4], ids: [1, 2, 3], want: 0, why: "fed longer than ids" },
    { fed: [2, 2, 3], ids: [1, 2, 3, 4], want: 0, why: "first token differs" },
    { fed: [1, 2, 9], ids: [1, 2, 3, 4], want: 0, why: "last fed token differs: never a partial reuse" },
    { fed: [], ids: [], want: 0 },
    { fed: undefined, ids: [1], want: 0 },
    { fed: [1], ids: [], want: 0 },
    { fed: Uint32Array.of(1, 2), ids: [1, 2, 3], want: 2, why: "typed arrays compare by value" },
    { fed: ["1", 2], ids: [1, 2, 3], want: 0, why: "strict equality" },
  ];
  for (const c of cases) eq(reusablePrefix(c.fed, c.ids), c.want, c.why || JSON.stringify(c));
});

// ---------------------------------------------------------------------------------------------
// splitThink

Deno.test("splitThink: table", () => {
  const cases = [
    { raw: "", want: { think: null, answer: "" } },
    { raw: "<think></think>yes", want: { think: "", answer: "yes" } },
    { raw: "<think>", want: { think: "", answer: "", open: true } },
    { raw: "pre<think>x</think>\n\npost", want: { think: "x", answer: "prepost" } },
    // only the first block is split out; a later one stays in the answer text as written
    { raw: "<think>a</think>x<think>b</think>y", want: { think: "a", answer: "x<think>b</think>y" } },
    { raw: "<think>a</think>x<think>b", want: { think: "a", answer: "x<think>b" } },
    // a stray close before any open is answer text, and the open block after it is still open
    { raw: "a</think>b<think>c", want: { think: "c", answer: "a</think>b", open: true } },
    { raw: "no think here </think>", want: { think: null, answer: "no think here </think>" } },
    // a nested open inside the block is part of the thinking
    { raw: "<think>x<think>y</think>z", want: { think: "x<think>y", answer: "z" } },
    { raw: "<think>  spaced  </think>   \n answer  ", want: { think: "spaced", answer: "answer  " } },
  ];
  for (const c of cases) eq(splitThink(c.raw), c.want, JSON.stringify(c.raw));
});
