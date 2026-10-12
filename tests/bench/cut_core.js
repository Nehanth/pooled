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

// A room's per-token readbacks one at a time, map kick variants interleaved call by call on one load (?hop=N in
// tab_cut.html, HOP=N in cut_prof.js): a worker's hop (runHidden: write the hidden, run the layers, read the hidden
// back) over all layers, 20, 13, 10, 5 and 2, the host's first hop (embedRun), its head (headFromHiddenIds,
// gpuSample's 16-byte readback) and the fused head + next hop (headAhead). Variants: off (plain mapAsync), on (kick at once), auto
// (the engine default: in a browser, kick only if this staging buffer's last readback took under map_kick.js AUTO.ms;
// never on Node / Deno). Each call is timed from its start to its result.
// -> { [call]: { off, on, auto } } medians in ms, plus runHidden's submit -> onSubmittedWorkDone (no map)
export async function hopBench({ eng, N = 40, log = console.log }) {
  const med = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };
  const dim = eng.dims.dim, pos = 64, desc = { kind: "greedy" };
  eng.reset();
  const x = Float32Array.from(await eng.embedRun(1000, pos));
  const calls = {
    runHidden: () => eng.runHidden(x, pos),
    embedRun: () => eng.embedRun(1000, pos),
    headFromHiddenIds: () => eng.headFromHiddenIds(x, desc),
    ...(eng.mtp && eng.headAhead && { headAhead: () => eng.headAhead(x, pos, desc) }),
  };
  // shorter hops (a device holding fewer layers): runHidden over the first n layers only
  const full = eng.layers;
  for (const n of [20, 13, 10, 5, 2]) if (n < full.length) calls[`runHidden_${n}layers`] = async () => { eng.layers = full.slice(0, n); try { return await eng.runHidden(x, pos); } finally { eng.layers = full; } };
  const res = {}, k0 = eng.mapKick;
  for (const [name, f] of Object.entries(calls)) {
    for (let i = 0; i < 4; i++) await f();   // warm
    const V = { off: false, on: true, auto: "auto" }, keys = Object.keys(V), t = { off: [], on: [], auto: [] };
    for (let i = 0; i < N; i++) for (let j = 0; j < keys.length; j++) {
      const k = keys[(i + j) % keys.length];
      eng.mapKick = V[k];
      const t0 = performance.now(); await f(); const ms = performance.now() - t0;
      t[k].push(ms);
    }
    res[name] = { off: med(t.off), on: med(t.on), auto: med(t.auto) };
    log(`hop ${name}: kick off ${res[name].off.toFixed(2)} ms, on ${res[name].on.toFixed(2)} ms, auto ${res[name].auto.toFixed(2)} ms`);
  }
  eng.mapKick = k0;
  // GPU floor of the full hop: the same submit, waited with onSubmittedWorkDone (no map)
  const g = [];
  for (let i = 0; i < N; i++) {
    eng.device.queue.writeBuffer(eng.x, 0, x);
    eng._setFrame(pos, pos + 1);
    const enc = eng.device.createCommandEncoder();
    for (let l = 0; l < eng.layers.length; l++) eng._encodeLayer(enc, l);
    const t0 = performance.now(); eng.device.queue.submit([enc.finish()]); await eng.device.queue.onSubmittedWorkDone(); g.push(performance.now() - t0);
  }
  res.runHidden.submitToDone = med(g);
  log(`hop runHidden: submit -> onSubmittedWorkDone ${res.runHidden.submitToDone.toFixed(2)} ms`);
  eng.reset();
  return res;
}
