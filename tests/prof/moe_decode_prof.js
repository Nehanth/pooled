// Decode profile of a Qwen 3.5/3.6 engine (built for the 35B-A3B MoE; works on the 27B too), shared
// by the Deno runner (tests/prof/prof_moe_decode.js) and the Chrome page (tests/bench/prof.html).
// Measures, on the two-sum chat prompt, greedy:
//   1. clean wall time (no hooks): plain ms/token, speculative ms/step and tok/s
//   2. submit mode: per plain token and per speculative step, submits / mapAsyncs / writeBuffers,
//      CPU encode ms (encoder created -> submitted), GPU span per submit, GPU busy fraction of the wall,
//      mapAsync wait, and a per-phase table of the speculative step (drafts, verify, refill, rollback)
//   3. kernel mode: GPU ms per category per plain token and per speculative step
import { installProf, bySum, r2, table } from "./gpuprof.js";

export async function profileDecode({ eng, device, tok, argmax, log, N = 24, STEPS = 12, K = 3, prompt: text, roomPath = false }) {
  const V = tok.vocab;
  const chat = (q) => [V["<|im_start|>"], ...tok.encode("user\n" + q), V["<|im_end|>"], ...tok.encode("\n"), V["<|im_start|>"], ...tok.encode("assistant\n"), V["<think>"], ...tok.encode("\n\n"), V["</think>"], ...tok.encode("\n\n")];
  const ids = chat(text || "Write the Python code for two sum. Code only.");
  const out = { plain: {}, spec: {}, kernels: {} };
  const hasMtp = !!(eng.mtp && eng.specStep);
  if (!eng.B) eng._initBatch();
  // eng.gpuSample (exp/gpu-sample, ?gpusample=1): greedy on the GPU, 16 bytes back per head instead of the logits
  const GS = !!eng.gpuSample;
  const pick = Object.assign((x) => (x && x.ids instanceof Uint32Array ? x.ids[0] : argmax(x)), GS ? { gpu: { kind: "greedy" } } : {});
  const fwd = GS ? (t) => eng.forwardTokenIds(t) : (t) => eng.forwardToken(t);
  out.gpuSample = GS; out.argmaxWide = !!eng.argmaxWide;
  const startPlain = async () => { eng.reset(); if (eng.mtp) eng.mtpFill = false; await eng.prefillTokens(ids.slice(0, -1)); return pick(await fwd(ids.at(-1))); };
  const startSpec = async () => { eng.reset(); eng.mtpFill = true; eng.mtp.stats = { drafts: 0, accepted: 0 }; await eng.prefillTokens(ids.slice(0, -1)); return pick(await fwd(ids.at(-1))); };
  const plainRun = async (n, next) => { const times = []; for (let i = 0; i < n; i++) { const t = performance.now(); next = pick(await fwd(next)); times.push(performance.now() - t); } return { next, times }; };
  const specRun = async (n, next) => { const steps = []; for (let i = 0; i < n; i++) { const t = performance.now(); const toks = await eng.specStep(next, pick, K); steps.push({ ms: performance.now() - t, toks: toks.length }); next = toks.at(-1); } return { next, steps }; };
  const med = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };

  // 1. clean wall
  let nx = await startPlain(); ({ next: nx } = await plainRun(4, nx));
  let t0 = performance.now(); const pr = await plainRun(N, nx); const plainWall = (performance.now() - t0) / N;
  out.plain.wallMs = r2(plainWall); out.plain.medianMs = r2(med(pr.times)); out.plain.tokPerS = r2(1000 / plainWall);
  log(`plain: ${out.plain.wallMs} ms/token (median ${out.plain.medianMs}), ${out.plain.tokPerS} tok/s`);
  if (hasMtp) {
    nx = await startSpec(); ({ next: nx } = await specRun(2, nx));
    t0 = performance.now(); const sr = await specRun(STEPS, nx); const w = performance.now() - t0; const nt = sr.steps.reduce((a, s) => a + s.toks, 0);
    out.spec = { stepMs: r2(w / STEPS), medianStepMs: r2(med(sr.steps.map((s) => s.ms))), tokPerStep: r2(nt / STEPS), tokPerS: r2(nt / (w / 1000)), acceptance: `${eng.mtp.stats.accepted}/${eng.mtp.stats.drafts}` };
    log(`spec K=${K}: ${out.spec.stepMs} ms/step, ${out.spec.tokPerStep} tok/step, ${out.spec.tokPerS} tok/s, acceptance ${out.spec.acceptance}`);
  }

  // 1b. roomPath: the room's solo decode (room.js aiPipeToken: embedRun, hidden read back, then headFromHidden[Ids],
  // two submits and two mapAsyncs per token) against forwardToken[Ids] (one of each), same tokens, interleaved
  if (roomPath) {
    const head = GS ? (h) => eng.headFromHiddenIds(h) : (h) => eng.headFromHidden(h);
    const roomRun = async (n, next) => { for (let i = 0; i < n; i++) { const h = await eng.embedRun(next, eng.pos); eng.pos++; next = pick(await head(h)); } return next; };
    const r = { fwd: [], room: [] };
    for (let rep = 0; rep < 3; rep++) {
      let x = await startPlain(); x = (await plainRun(2, x)).next;
      let t = performance.now(); await plainRun(N, x); r.fwd.push((performance.now() - t) / N);
      x = await startPlain(); x = await roomRun(2, x);
      t = performance.now(); await roomRun(N, x); r.room.push((performance.now() - t) / N);
    }
    out.roomPath = { forwardMs: r2(Math.min(...r.fwd)), roomMs: r2(Math.min(...r.room)) };
    log(`room path: forwardToken${GS ? "Ids" : ""} ${out.roomPath.forwardMs} ms/token, embedRun + headFromHidden${GS ? "Ids" : ""} ${out.roomPath.roomMs} ms/token (best of 3)`);
  }

  // 2. submit mode
  const P = installProf(device, eng, { mode: "submit" });
  out.timestamps = P.mode !== "none";
  const summarize = (units, wallMs) => {
    const subs = P.subs, maps = P.maps;
    const gpu = subs.reduce((a, s) => a + s.gpu, 0), enc = subs.reduce((a, s) => a + Math.max(0, s.tSubmit - s.tEnc), 0);
    const mapWait = maps.reduce((a, m) => a + (m.t1 - m.t0), 0);
    // GPU span from the first submit's start to the last one's end, per unit, is the "GPU window";
    // gaps between consecutive submits' GPU spans are GPU idle waiting on the CPU
    let idle = 0; const g = subs.filter((s) => s.g1 > 0).sort((a, b) => a.g0 - b.g0);
    for (let i = 1; i < g.length; i++) idle += Math.max(0, g[i].g0 - g[i - 1].g1);
    return { wallMs: r2(wallMs / units), gpuMs: r2(gpu / units), busyPct: r2(100 * gpu / wallMs), cpuEncodeMs: r2(enc / units), submits: r2(subs.length / units),
      mapAsyncs: r2(maps.length / units), mapWaitMs: r2(mapWait / units), writeBuffers: r2(P.writes / units), writeKB: r2(P.writeBytes / units / 1024), encoders: r2(P.encs / units),
      gpuIdleBetweenSubmitsMs: r2(idle / units), passes: r2(subs.reduce((a, s) => a + (s.passes || 0), 0) / units),
      dispatches: r2(subs.reduce((a, s) => a + (s.dispatches || 0), 0) / units), copies: r2(subs.reduce((a, s) => a + (s.copies || 0), 0) / units),
      copyKB: r2(subs.reduce((a, s) => a + (s.copyBytes || 0), 0) / units / 1024) };
  };
  const perLabel = (units) => {
    const L = {};
    for (const s of P.subs) { const o = (L[s.label] ||= { submits: 0, gpuMs: 0, encodeMs: 0, maps: 0, mapWaitMs: 0, passes: 0, dispatches: 0, copies: 0 }); o.submits++; o.gpuMs += s.gpu; o.encodeMs += Math.max(0, s.tSubmit - s.tEnc); o.passes += s.passes || 0; o.dispatches += s.dispatches || 0; o.copies += s.copies || 0; }
    for (const m of P.maps) { const o = (L[m.label] ||= { submits: 0, gpuMs: 0, encodeMs: 0, maps: 0, mapWaitMs: 0, passes: 0, dispatches: 0, copies: 0 }); o.maps++; o.mapWaitMs += m.t1 - m.t0; }
    return Object.entries(L).map(([label, o]) => ({ label, submits: r2(o.submits / units), gpuMs: r2(o.gpuMs / units), encodeMs: r2(o.encodeMs / units), maps: r2(o.maps / units), mapWaitMs: r2(o.mapWaitMs / units), passes: r2(o.passes / units), dispatches: r2(o.dispatches / units), copies: r2(o.copies / units) }))
      .sort((a, b) => b.gpuMs - a.gpuMs);
  };
  nx = await startPlain(); ({ next: nx } = await plainRun(3, nx));
  P.start(); t0 = performance.now(); ({ next: nx } = await plainRun(N, nx)); let w = performance.now() - t0; P.stop(); await P.resolve();
  out.plain.submit = summarize(N, w); out.plain.byLabel = perLabel(N);
  log("plain, per token (submit mode):\n" + table([out.plain.submit], Object.keys(out.plain.submit)));
  log(table(out.plain.byLabel, ["label", "submits", "gpuMs", "encodeMs", "maps", "mapWaitMs", "passes", "dispatches", "copies"]));
  let specToks = 0;
  if (hasMtp) {
    nx = await startSpec(); ({ next: nx } = await specRun(2, nx));
    P.start(); t0 = performance.now(); const sr = await specRun(STEPS, nx); w = performance.now() - t0; P.stop(); await P.resolve();
    specToks = sr.steps.reduce((a, s) => a + s.toks, 0);
    out.spec.submit = summarize(STEPS, w); out.spec.byLabel = perLabel(STEPS); out.spec.submit.tokPerStep = r2(specToks / STEPS);
    log(`spec K=${K}, per step (submit mode, ${out.spec.submit.tokPerStep} tok/step):\n` + table([out.spec.submit], Object.keys(out.spec.submit)));
    log(table(out.spec.byLabel, ["label", "submits", "gpuMs", "encodeMs", "maps", "mapWaitMs", "passes", "dispatches", "copies"]));
    // one step's timeline: CPU submit times and GPU spans relative to the step's first submit
    const s0 = P.subs.filter((s) => s.q >= 0);
    const firstStepEnd = s0.findIndex((s, i) => i > 0 && s.label === s0[0].label);
    const one = s0.slice(0, firstStepEnd > 0 ? firstStepEnd : 8);
    const base = one[0]?.tSubmit || 0, gbase = one[0]?.g0 || 0;
    out.spec.timeline = one.map((s) => ({ label: s.label, cpuSubmitAt: r2(s.tSubmit - base), encodeMs: r2(s.tSubmit - s.tEnc), gpuStartAt: r2(s.g0 - gbase), gpuMs: r2(s.gpu) }));
    log("one speculative step (ms from its first submit; GPU clock aligned at the first GPU start):\n" + table(out.spec.timeline, ["label", "cpuSubmitAt", "encodeMs", "gpuStartAt", "gpuMs"]));
  }

  // 3. kernel mode
  if (P.mode !== "none") {
    P.mode = "kernel";
    const cats = (units) => {
      const all = P.subs.flatMap((s) => s.kern);
      const c = bySum(all, (k) => k.cat, (k) => k.ms), total = all.reduce((a, k) => a + k.ms, 0);
      return { total: r2(total / units), dispatches: r2(all.length / units), rows: Object.entries(c).map(([cat, o]) => ({ category: cat, ms: r2(o.ms / units), dispatches: r2(o.n / units), pct: r2(100 * o.ms / total) })).sort((a, b) => b.ms - a.ms),
        pipes: Object.entries(bySum(all, (k) => k.name, (k) => k.ms)).map(([p, o]) => ({ pipe: p, ms: r2(o.ms / units), n: r2(o.n / units), usEach: r2(1000 * o.ms / o.n) })).sort((a, b) => b.ms - a.ms).slice(0, 25) };
    };
    nx = await startPlain(); ({ next: nx } = await plainRun(2, nx));
    const KN = 6;
    P.start(); ({ next: nx } = await plainRun(KN, nx)); P.stop(); await P.resolve();
    out.kernels.plain = cats(KN);
    log(`kernel mode, plain token: sum ${out.kernels.plain.total} ms over ${out.kernels.plain.dispatches} dispatches\n` + table(out.kernels.plain.rows, ["category", "ms", "dispatches", "pct"]));
    log(table(out.kernels.plain.pipes, ["pipe", "ms", "n", "usEach"]));
    if (hasMtp) {
      nx = await startSpec(); ({ next: nx } = await specRun(2, nx));
      const KS = 6;
      P.start(); await specRun(KS, nx); P.stop(); await P.resolve();
      out.kernels.spec = cats(KS);
      const byL = bySum(P.subs.flatMap((s) => s.kern), (k) => k.label, (k) => k.ms);
      out.kernels.specByLabel = Object.entries(byL).map(([l, o]) => ({ label: l, ms: r2(o.ms / KS), dispatches: r2(o.n / KS) })).sort((a, b) => b.ms - a.ms);
      log(`kernel mode, speculative step K=${K}: sum ${out.kernels.spec.total} ms over ${out.kernels.spec.dispatches} dispatches\n` + table(out.kernels.spec.rows, ["category", "ms", "dispatches", "pct"]));
      log(table(out.kernels.specByLabel, ["label", "ms", "dispatches"]));
    }
    P.mode = "submit";
  }
  // 4. sync floor: empty submit + 4-byte readback, and a 1 MB (logits-size) readback
  {
    const s4 = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const n = 30; let a = 0, b = 0;
    for (let i = 0; i < n; i++) { const t = performance.now(); const e = device.createCommandEncoder(); e.copyBufferToBuffer(eng.logits, 0, s4, 0, 4); device.queue.submit([e.finish()]); await s4.mapAsync(GPUMapMode.READ); s4.unmap(); a += performance.now() - t; }
    for (let i = 0; i < n; i++) { const t = performance.now(); await eng._readback(eng.logits, eng.stageLogits, eng.dims.vocab); b += performance.now() - t; }
    out.syncFloorMs = r2(a / n); out.logitsReadbackMs = r2(b / n);
    log(`sync floor (empty submit + 4-byte map): ${out.syncFloorMs} ms; logits readback (${(eng.dims.vocab * 4 / 2 ** 20).toFixed(2)} MB): ${out.logitsReadbackMs} ms`);
  }
  return out;
}
