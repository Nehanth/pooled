// The qwen4exp goldens' pass rule (tests/golden/q38flash_llama_greedy.json, made by tests/golden/make_q38flash_llama.js),
// shared by tests/test_q38flash.js and tests/test_q38flash_batch.js. docs/kernels.md ("Qwen3.8-Flash goldens") has
// the evidence.
//
// The engine's greedy ids are compared with llama.cpp's step by step. At the first step where they differ:
//   backend near tie  passes when the engine's token is in llama.cpp's top-k at that step within TIE_NATS (0.5) of
//                     llama.cpp's top logprob, AND the engine's own top-2 gap there is under TIE_NATS. Comparing
//                     stops there (the two continuations are different texts from then on). llama.cpp's top-k comes
//                     from the golden's run, or from any of its other runs ("runs": llama.cpp is not reproducible on
//                     the >2K-token prompts) whose ids agree with the golden's up to that step.
//   fallback          a golden without top-k data: llama.cpp's own top-2 gap at that step under LLAMA_GAP (0.25).
//   anything else     a mismatch.
// Why 0.5 nats: two correct backends of the same Q4_0 model move a step's top-2 gap by that much. On ctx2k_b step 6
// llama.cpp's CPU backend (f32 KV, no FA) moves the gap from the CUDA golden's 0.382 to 0.140, our engine by 0.42
// (the same spread as at 23 tokens); between llama.cpp backends |d gap| is mostly 0.25-0.6 (max 0.99); and the
// same CUDA llama-server request past ~2K tokens gives top-2 gaps that vary by up to 0.5 run to run (ctx4k step 6:
// 0.04-0.48, and one run picks another token).
// What it catches, and what not (measured on the full model, docs/kernels.md): one DeltaNet layer's output sign-
// flipped fails (bash at 12, our gap 0.56; eos-mid at 0, 1.14 nats down). Scaling one layer's output by 1.03, 1.25
// or 2, or every DeltaNet layer's by 1.1, still passes: the model absorbs it and the ids only move at near ties.
// Greedy ids are a coarse oracle under any tie rule; numeric errors are the per-layer check's job (LAYERS=48).
export const TIE_NATS = 0.5, LLAMA_GAP = 0.25;

// g: the golden case; gen: the engine's ids; gaps: the engine's top-2 gap per step (nats); n: ids compared.
// -> { diff (first differing step, -1: none), ok (exact match), excused, rule ("near-tie" | "fallback" | null), msg }
export function judgeGolden(g, gen, gaps, n = Math.min(gen.length, g.ids.length)) {
  const diff = gen.slice(0, n).findIndex((x, i) => x !== g.ids[i]);
  if (diff < 0) return { diff, ok: true, excused: false, rule: null, msg: `MATCH llama.cpp (${n} ids)` };
  const ours = gen[diff], ourGap = gaps?.[diff];
  if (g.top?.[diff]) {
    // llama.cpp's top-k at this step, from every run that reached it on the golden's ids
    const lists = [{ run: 0, top: g.top[diff] }];
    for (const [k, r] of (g.runs || []).entries())
      if (r.top?.[diff] && r.ids.slice(0, diff).every((x, i) => x === g.ids[i])) lists.push({ run: k + 1, top: r.top[diff] });
    let best = null;
    for (const { run, top } of lists) {
      const hit = top.find(([id]) => id === ours);
      if (hit) { const below = top[0][1] - hit[1]; if (!best || below < best.below) best = { run, below }; }
    }
    const tieOk = !!best && best.below < TIE_NATS && ourGap != null && ourGap < TIE_NATS;
    const where = best ? `${best.below.toFixed(3)} nats below llama.cpp's top${best.run ? ` (its rerun ${best.run})` : ""}` : `not in llama.cpp's top ${g.top[diff].length}`;
    const why = `our ${ours} vs llama.cpp's ${g.ids[diff]}: ${where}, our top-2 gap ${ourGap?.toFixed(3)}`;
    return tieOk ? { diff, ok: false, excused: true, rule: "near-tie", msg: `differs at ${diff} at a backend near tie (${why}; not a failure)` }
      : { diff, ok: false, excused: false, rule: null, msg: `MISMATCH at ${diff} (${why}; a near tie needs both under ${TIE_NATS})` };
  }
  const lg = g.gap?.[diff];
  if (lg != null && lg < LLAMA_GAP) return { diff, ok: false, excused: true, rule: "fallback", msg: `differs at ${diff}, llama.cpp's near tie there (gap ${lg} < ${LLAMA_GAP}; golden without top-k; not a failure)` };
  return { diff, ok: false, excused: false, rule: null, msg: `MISMATCH at ${diff} (llama.cpp gap ${lg}; golden without top-k)` };
}
