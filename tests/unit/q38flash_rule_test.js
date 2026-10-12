// CPU check of the qwen4exp goldens' pass rule (tests/q38flash_rule.js) on the real golden file: what it excuses
// (a backend near tie: our token within 0.5 nats of llama.cpp's top, our own top-2 gap under 0.5, top-k from any
// llama.cpp run that agrees up to the step) and what it does not.
// No GPU.   deno test --allow-read tests/unit/q38flash_rule_test.js
import { judgeGolden, TIE_NATS } from "../q38flash_rule.js";

const gold = JSON.parse(Deno.readTextFileSync(new URL("../golden/q38flash_llama_greedy.json", import.meta.url)));
const assert = (c, m) => { if (!c) throw new Error(m); };
// the golden's ids with step d replaced by tok (and the rest left as llama.cpp's: the rule must stop at d)
const swap = (g, d, tok) => g.ids.map((x, i) => (i === d ? tok : x));
const gaps = (g, d, ours) => g.ids.map((_, i) => (i === d ? ours : 5));

Deno.test("every qwen4exp golden carries top-k per step", () => {
  for (const [k, g] of Object.entries(gold)) {
    if (k.startsWith("_")) continue;
    assert(g.top?.length === g.ids.length && g.top.every((t) => t.length >= 4), `${k}: top-k per step`);
    assert(g.top.every((t, i) => t[0][0] === g.ids[i]), `${k}: the greedy id is llama.cpp's top`);
  }
});

Deno.test("an exact match", () => {
  const g = gold["two-sum"], r = judgeGolden(g, g.ids.slice(), gaps(g, -1, 0));
  assert(r.ok && r.diff === -1, r.msg);
});

Deno.test("ctx2k_b step 6: 7276 (0.35 below llama.cpp's 513), our gap 0.04: a backend near tie", () => {
  const g = gold.ctx2k_b, r = judgeGolden(g, swap(g, 6, 7276), gaps(g, 6, 0.04));
  assert(r.excused && r.rule === "near-tie" && r.diff === 6, r.msg);
});

Deno.test("the same token with our own gap over 0.5 fails (the engine is confident about the other token)", () => {
  const g = gold.ctx2k_b, r = judgeGolden(g, swap(g, 6, 7276), gaps(g, 6, 0.8));
  assert(!r.excused && !r.ok, r.msg);
});

Deno.test("a token outside llama.cpp's top-k fails, even at a llama.cpp near tie", () => {
  const g = gold.ctx2k, r = judgeGolden(g, swap(g, 6, 12345), gaps(g, 6, 0.01));
  assert(!r.excused, r.msg);
});

Deno.test("a token in the top-k but more than 0.5 below the top fails", () => {
  const g = gold["two-sum"], alt = g.top[3][1][0];   // step 3: llama.cpp's runner-up, ~2.8 nats down
  assert(g.top[3][0][1] - g.top[3][1][1] > TIE_NATS, "fixture: a confident step");
  const r = judgeGolden(g, swap(g, 3, alt), gaps(g, 3, 0.1));
  assert(!r.excused, r.msg);
});

Deno.test("a difference after a matched tie still fails; only the first difference is judged", () => {
  const g = gold.ctx2k_b, d = g.top.findIndex((t, i) => i > 6 && t[0][1] - t[1][1] > 1);   // past the step-6 tie, confident
  assert(d > 6, "fixture: a confident step after step 6");
  const r = judgeGolden(g, swap(g, d, g.top[d][1][0]), gaps(g, d, 0.1));
  assert(!r.excused && r.diff === d, r.msg);
});

Deno.test("top-k from a llama.cpp rerun counts only when its ids agree up to the step", () => {
  const ids = [1, 2, 3], top = [[[1, 0]], [[2, 0]], [[3, -0.1], [9, -1.0]]];
  const g = { ids, top, gap: [5, 5, 0.9], runs: [{ ids: [1, 2, 9], top: [[[1, 0]], [[2, 0]], [[9, -0.2], [3, -0.3]]] }] };
  assert(judgeGolden(g, [1, 2, 9], [5, 5, 0.1]).excused, "the rerun, on the same prefix, had 9 on top");
  const h = { ...g, runs: [{ ids: [1, 7, 9], top: g.runs[0].top }] };
  assert(!judgeGolden(h, [1, 2, 9], [5, 5, 0.1]).excused, "a rerun that left the golden's ids before the step does not count");
});

Deno.test("fallback for a golden without top-k: llama.cpp's own gap under 0.25 at that step", () => {
  const g = { ids: [1, 2, 3], gap: [5, 0.1, 3] };
  assert(judgeGolden(g, [1, 9, 3], [5, 0.3, 5]).rule === "fallback", "near tie at 1");
  assert(!judgeGolden(g, [1, 2, 9], [5, 5, 0.1]).excused, "no tie at 2");
});
