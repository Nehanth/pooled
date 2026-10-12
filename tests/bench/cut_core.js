// The offload cut benchmark's body, shared by Chrome (tests/bench/tab_cut.html, driver tab_cut.mjs) and Node / Deno
// (tests/bench/cut_prof.js): on a built engine, the bit-exactness fingerprint (tests/test_q38_bits.js's plain part:
// 300-token batched prefill + 13 greedy steps) and decode tok/s with each cut's time split by phase (engine cutProf).
// -> { bits: { plain, hidden, tokens }, decode: [{ tps, msPerTok, cuts, phases (ms per token), hit }] }
import { argmax } from "../../engine/engine.js";
const h = (arrs) => { let x = 0x811c9dc5; for (const a of arrs) { const u = new Uint32Array(a.buffer, a.byteOffset, a.length); for (let i = 0; i < u.length; i++) x = Math.imul(x ^ u[i], 0x01000193) >>> 0; } return x.toString(16); };
export async function cutBench({ eng, tok, prompt, N = 64, bits = true, log = console.log }) {
  const res = {};
  if (bits) {
    const ids = tok.encode(prompt).slice(0, 300);
    eng.reset();
    await eng.prefillTokens(ids.slice(0, -1));
    let lg = await eng.forwardToken(ids.at(-1));
    const L0 = [Float32Array.from(lg)], out = [argmax(lg)];
    for (let i = 0; i < 12; i++) { lg = await eng.forwardToken(out.at(-1)); L0.push(Float32Array.from(lg)); out.push(argmax(lg)); }
    const x = await eng._readback(eng.x, eng.stageX, eng.dims.dim);
    res.bits = { plain: h(L0), hidden: h([x]), tokens: out.join(",") };
    log(`BITS plain ${res.bits.plain} hidden ${res.bits.hidden} tokens ${res.bits.tokens}`);
  }
  const V = tok.vocab, chat = (s) => [V["<|im_start|>"], ...tok.encode("user\n" + s), V["<|im_end|>"], ...tok.encode("\n"), V["<|im_start|>"], ...tok.encode("assistant\n"), V["<think>"], ...tok.encode("\n\n"), V["</think>"], ...tok.encode("\n\n")];
  const G1 = Object.assign((x) => (x && x.ids instanceof Uint32Array ? x.ids[0] : argmax(x)), { gpu: { kind: "greedy" } });
  const GS = !!eng.gpuSample, fwd = (id) => (GS ? eng.forwardTokenIds(id) : eng.forwardToken(id)), pick = GS ? G1 : argmax;
  res.decode = [];
  for (const p of ["Write a Python function that returns the two indices of a list whose values sum to a target.", "Explain what a hash map is, in two paragraphs."]) {
    const pre = chat(p);
    eng.reset(); await eng.prefillTokens(pre.slice(0, -1));
    let next = pick(await fwd(pre.at(-1)));
    const s0 = eng.experts ? { ...eng.experts.stats } : null;
    if (eng.experts) eng.cutProf = { n: 0, enc: 0, submit: 0, wait: 0, plan: 0, last: 0 };
    const gen = [next], t0 = performance.now();
    for (let j = 1; j < N; j++) { next = pick(await fwd(next)); gen.push(next); }
    const ms = performance.now() - t0, n = N - 1, r = { tps: 1000 * n / ms, msPerTok: ms / n, text: tok.decode(gen).slice(0, 60) };
    if (eng.experts) {
      const P = eng.cutProf, d = (k) => eng.experts.stats[k] - s0[k];
      eng.cutProf = null;
      r.cutsPerTok = P.n / n;
      r.phases = Object.fromEntries(["enc", "submit", "wait", "plan"].map((k) => [k, P[k] / n]));
      r.phases.rest = r.msPerTok - r.phases.enc - r.phases.submit - r.phases.wait - r.phases.plan;
      r.hit = d("hits") / Math.max(1, d("lookups")); r.missPerTok = d("misses") / n;
      if (eng.cutSpec) { r.specPerTok = d("specs") / n; r.rollbackPerTok = d("specFails") / n; }
    }
    res.decode.push(r);
    log(`decode ${n} tokens: ${r.tps.toFixed(2)} tok/s (${r.msPerTok.toFixed(2)} ms/token)` + (r.phases ? `, ${r.cutsPerTok.toFixed(1)} cuts/token, per token: ` + Object.entries(r.phases).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(" ") + ` ms; hits ${(100 * r.hit).toFixed(1)}%, ${r.missPerTok.toFixed(1)} misses/token` + (r.specPerTok != null ? `, ${r.specPerTok.toFixed(1)} layers/token speculated, ${r.rollbackPerTok.toFixed(2)} rollbacks/token` : "") : "") + `  ${JSON.stringify(r.text)}`);
  }
  if (eng.experts) { res.summary = eng.experts.summary(); log(res.summary); }
  return res;
}
