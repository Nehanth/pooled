// Speculation for models without a draft head, on the CPU: the n-gram drafter (room/lookup.js) and
// DenseEngine.specStepDrafts' bookkeeping with the GPU replaced by a symbolic model. The model's
// next token is a fixed function of the context, so plain decoding is a known sequence and a
// lookup-speculative run must give exactly that sequence, whatever the drafts were.
import { lookupDrafts } from "../../room/lookup.js";
import { DenseEngine } from "../../engine/dense.js";
const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

Deno.test("lookup: the longest matching n-gram wins, then the most recent copy", () => {
  // n = 3 matches at 0 (-> 6 0 2) although the 2-gram "1 2" is also more recent nowhere else
  eq(lookupDrafts([1, 2, 6, 0, 2, 7, 0, 1, 2], 3), [6, 0, 2]);
  // two earlier copies of the 2-gram: the most recent one's continuation
  eq(lookupDrafts([4, 5, 1, 9, 4, 5, 2, 8, 4, 5], 2), [2, 8]);
  // maxN caps the n-gram: with maxN 2 the most recent 2-gram copy is used
  eq(lookupDrafts([7, 4, 5, 1, 3, 4, 5, 2, 7, 4, 5], 1, { maxN: 2 }), [2]);
  eq(lookupDrafts([7, 4, 5, 1, 3, 4, 5, 2, 7, 4, 5], 1, { maxN: 3 }), [1]);
});

Deno.test("lookup: no match, too-short context, k = 0 and minN", () => {
  eq(lookupDrafts([], 4), []);
  eq(lookupDrafts([3], 4), []);
  eq(lookupDrafts([3, 3], 4), []);          // a 2-gram needs an earlier copy
  eq(lookupDrafts([1, 2, 3, 1, 2], 0), []);
  // a single repeated token is not enough with minN 2, is with minN 1
  eq(lookupDrafts([5, 8, 9, 5], 2), []);
  eq(lookupDrafts([5, 8, 9, 5], 2, { minN: 1 }), [8, 9]);
});

Deno.test("lookup: the window limits how far back it looks", () => {
  const ctx = [1, 2, 3, ...Array(50).fill(0).map((_, i) => 100 + i), 1, 2];
  eq(lookupDrafts(ctx, 1), [3]);
  eq(lookupDrafts(ctx, 1, { window: 20 }), []);
});

Deno.test("lookup: a code edit copies the unchanged lines after the edited identifier", () => {
  // "for u in users :" ... answer so far "for u in" -> the rest of the line comes from the prompt
  const prompt = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
  const ans = [10, 11, 12];
  eq(lookupDrafts([...prompt, ...ans], 5), [13, 14, 15, 16, 17]);
  // after a renamed token (99 instead of 14) the 2-gram "99 15" is new; "15 16" is found again later
  eq(lookupDrafts([...prompt, 10, 11, 12, 13, 99], 3), []);
  eq(lookupDrafts([...prompt, 10, 11, 12, 13, 99, 15, 16], 3), [17, 18, 19]);
});

// ---- DenseEngine.specStepDrafts with a symbolic model ----
// A hidden is the position of the token it came from; headBatch turns hidden p into "logits" whose
// argmax is target(p + 1). The sampler below reads the argmax back.
function fakeDense(target, { maxSeq = 1e9, maxDrafts = 7 } = {}) {
  const e = Object.create(DenseEngine.prototype);
  const log = { batches: [], written: new Map() };
  Object.assign(e, { dims: { dim: 1, vocab: 1 }, pos: 0, maxSeq, maxDrafts, log });
  e.embedRunBatch = async function (ids, basePos, verify) {
    ok(ids.length >= 1 && ids.length <= 4, "1..4 columns per pass");
    log.batches.push({ n: ids.length, basePos, verify });
    ids.forEach((t, c) => log.written.set(basePos + c, t));
    this.pos = basePos + ids.length;
    if (verify) this._vBase = basePos - (typeof verify === "object" ? verify.base : 0);
    return Float32Array.from(ids, (_, c) => basePos + c);
  };
  e.headBatch = async (hs, n) => Array.from({ length: n }, (_, c) => ({ next: target(hs[c] + 1) }));
  return e;
}
const pick = (lg) => lg.next;

Deno.test("dense specStepDrafts: accepted drafts, rejection, rollback position", async () => {
  const seq = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
  const target = (p) => seq[p];
  const e = fakeDense(target);
  e.pos = 3;   // seq[3] = 8 is the token chosen for position 3
  // all drafts right: K + 1 tokens, the last is the next tNext (not written)
  let rej = null;
  eq(await e.specStepDrafts(8, pick, [9, 10, 11], { onReject: async (k) => { rej = k; } }), [9, 10, 11, 12]);
  eq(e.pos, 7); eq(rej, null);
  eq(e.specStats, { steps: 1, drafts: 3, accepted: 3 });
  // a wrong second draft: tNext and the first draft are kept, the model's own token follows
  eq(await e.specStepDrafts(12, pick, [13, 99, 5], { onReject: async (k) => { rej = k; } }), [13, 14]);
  eq(e.pos, 9); eq(rej, 1);
  eq(e.specStats, { steps: 2, drafts: 6, accepted: 4 });
  // the worker-side rollback lands on the same position
  const w = fakeDense(target);
  await w.embedRunBatch([12, 13, 99, 5], 7, { base: 0, total: 4 });
  eq(w.pos, 11);
  w.restoreDN(1);
  eq(w.pos, 9);
});

Deno.test("dense specStepDrafts: 8 columns go in two batched passes; K is capped by maxDrafts and the context", async () => {
  const target = (p) => 100 + p;
  const e = fakeDense(target, { maxDrafts: 7 });
  e.pos = 10;
  const drafts = Array.from({ length: 12 }, (_, i) => 111 + i);   // all right
  const out = await e.specStepDrafts(110, pick, drafts);
  eq(out, [111, 112, 113, 114, 115, 116, 117, 118]);
  eq(e.log.batches.map((b) => [b.n, b.basePos, b.verify]), [[4, 10, { base: 0, total: 8 }], [4, 14, { base: 4, total: 8 }]]);
  eq(e.pos, 18);
  // near the end of the context: positions pos .. pos + K must fit in maxSeq
  const f = fakeDense(target, { maxSeq: 20 });
  f.pos = 17;
  eq(await f.specStepDrafts(117, pick, [118, 119, 120, 121]), [118, 119, 120]);
  eq(f.pos, 20);
  // runTrunk (a room) replaces the local pass and gets all the tokens at once
  const g = fakeDense(target);
  g.pos = 2;
  let got = null;
  const r = await g.specStepDrafts(102, pick, [103, 7], { runTrunk: async (toks, pos) => { got = [toks, pos]; return Float32Array.from(toks, (_, c) => pos + c); } });
  eq(got, [[102, 103, 7], 2]); eq(r, [103, 104]); eq(g.log.batches.length, 0);
});

Deno.test("lookup + dense verify: speculative greedy output equals plain greedy output", async () => {
  // a "model" that repeats a code block with one identifier renamed, then some fresh text: its next
  // token is a function of the position only, so plain decoding is exactly `answer`
  const block = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33];
  const prompt = [1, 2, 3, ...block, 4, 5];
  const answer = [...block.map((t) => (t === 25 ? 99 : t)), 40, 41, 42, 20, 21, 50, 51, 52, 53];
  const all = [...prompt, ...answer];
  const target = (p) => all[p];
  const e = fakeDense(target);
  // prefill leaves pos at the last prompt token, whose logits pick answer[0]
  e.pos = prompt.length;
  const fed = [...prompt];
  const toks = [answer[0]];
  let laps = 0;
  while (toks.length < answer.length) {
    const next = toks[toks.length - 1];
    const lk = lookupDrafts([...fed, next], Math.min(7, answer.length - toks.length));
    laps++;
    let out;
    if (lk.length) out = await e.specStepDrafts(next, pick, lk);
    else { out = [target(e.pos + 1)]; e.pos++; }   // a plain step
    fed.push(next, ...out.slice(0, -1));
    toks.push(...out);
  }
  eq(toks.slice(0, answer.length), answer);
  ok(laps < answer.length * 0.6, `lookup should save laps on a copy: ${laps} laps for ${answer.length} tokens`);
});

// ---- room/draftmodel.js: the draft model keeps its caches in step with the context by token ids ----
import { DraftModel } from "../../room/draftmodel.js";
function fakeDrafter(guess) {
  const e = { pos: 0, maxSeq: 1e9, rows: [], prefilled: 0 };
  e.prefillTokens = async (ids) => { for (const t of ids) e.rows[e.pos++] = t; e.prefilled += ids.length; };
  e.forwardToken = async (t) => { e.rows[e.pos++] = t; return { g: guess(e.rows.slice(0, e.pos)) }; };
  return e;
}
Deno.test("draft model: syncs by common prefix, drafts greedily, never re-prefills what it holds", async () => {
  // the drafter guesses "last token + 1"
  const e = fakeDrafter((rows) => rows[rows.length - 1] + 1);
  const d = new DraftModel(e, { argmax: (lg) => lg.g });
  eq(await d.propose([5, 6, 7], 3), [8, 9, 10]);
  eq(e.prefilled, 2); eq(e.rows.slice(0, e.pos), [5, 6, 7, 8, 9]);
  // the verify accepted 8 and 9, then the model picked 42: ctx grows by 7 8 9, next is 42
  eq(await d.propose([5, 6, 7, 8, 9, 42], 2), [43, 44]);
  eq(e.prefilled, 2, "7 8 9 were already in the draft caches");
  eq(d.held, [5, 6, 7, 8, 9, 42, 43]);
  // a different conversation: back to the common prefix, prefill the rest
  eq(await d.propose([5, 1, 2], 1), [3]);
  eq(e.prefilled, 3); eq(d.held, [5, 1, 2]);
  eq(await d.propose([5], 0), []);
});

Deno.test("draft model: drafts only when the measured lap is long enough to pay for them", () => {
  const d = new DraftModel({ pos: 0 }, { argmax: () => 0 });
  eq(d.pickK(30, 4), 4, "nothing measured yet: draft (and measure)");
  d.msPerTok = 8; d.acc = 0.5;
  // 1 + 0.5 = 1.5 tokens for 30 + 8 ms beats 1 token per 30 ms? 1.5/38 > 1/30: yes
  ok(d.pickK(30, 4) >= 1);
  // a 10 ms lap: any draft costs more than the lap it would save
  eq(d.pickK(10, 4), 0);
  // a 100 ms lap (two 50 ms hops) with good acceptance: the longest run
  d.acc = 0.8;
  eq(d.pickK(100, 4), 4);
  // acceptance falls: fewer drafts
  d.acc = 0.2;
  ok(d.pickK(100, 4) < 4);
  d.note(4, 4); ok(d.acc > 0.2);
});
