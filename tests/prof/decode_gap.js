// Where a decode token's wall time goes, for Deno and the browser (same module): the GPU span of every
// submit (an empty timestamped pass at the start and end of each command encoder), the GPU idle between
// consecutive submits, the CPU encode cost of one token, and the back-to-back GPU throughput (N tokens
// pre-encoded and submitted without a readback in between, so the GPU never waits on the CPU).
//   wall - backToBack = what the round trip per token costs (readback, JS, submit latency, GPU idle)
//   backToBack - span = what the GPU loses between dispatches inside a token that it does not lose
//   between submits (should be ~0)
// Plain decode only runs through forwardTokenIds (GPU sampling, greedy), the room's solo path.
// The engine state is garbage after the back-to-back run: callers reset and prefill again.
export async function decodeGap({ eng, device, prefix, N = 32, K = 3, log = console.log, spec = true }) {
  const q = device.queue, desc = { kind: "greedy" }, pick = Object.assign((c) => c.ids[0], { gpu: desc });   // .gpu: specStep samples on the GPU too
  const ts = device.features.has("timestamp-query");
  const QN = 4096, qs = ts ? device.createQuerySet({ type: "timestamp", count: QN }) : null;
  // pair i (queries i, i + 1) resolves to byte (i / 2) * 256: resolveQuerySet's destination offset must be 256-aligned
  const res = ts ? device.createBuffer({ size: QN * 128, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }) : null;
  const rd = ts ? device.createBuffer({ size: QN * 128, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) : null;
  let on = false, nq = 0, encMs = 0;
  // the timestamps go on a one-workgroup no-op dispatch: empty timestamped compute passes hang the GB10 (Vulkan;
  // tests/prof/gpuprof.js's submit mode hangs there the same way)
  const nop = ts ? device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: "@compute @workgroup_size(1) fn main() {}" }), entryPoint: "main" } }) : null;
  // only queries of submitted command buffers are resolved: resolving a query no submitted pass wrote waits forever
  // on the GB10 (an encode-ahead buffer is often still unsubmitted when a run ends)
  const cbQ = new WeakMap(), done = new Set(), oSubmit = q.submit.bind(q);
  q.submit = (cbs) => { for (const cb of cbs) if (cbQ.has(cb)) done.add(cbQ.get(cb)); return oSubmit(cbs); };
  const stamp = (enc, tw) => { const p = enc.beginComputePass({ timestampWrites: tw }); p.setPipeline(nop); p.dispatchWorkgroups(1); p.end(); };
  const oc = device.createCommandEncoder.bind(device);
  device.createCommandEncoder = (d) => {
    const enc = oc(d);
    if (on) { const tE = performance.now(), f0 = enc.finish.bind(enc); enc.finish = (x) => { const cb = f0(x); encMs += performance.now() - tE; return cb; }; }
    if (!on || !ts || nq + 2 > QN) return enc;
    const i = nq; nq += 2;
    stamp(enc, { querySet: qs, beginningOfPassWriteIndex: i });
    const of = enc.finish.bind(enc);
    enc.finish = (x) => { stamp(enc, { querySet: qs, endOfPassWriteIndex: i + 1 }); const cb = of(x); cbQ.set(cb, i); return cb; };
    return enc;
  };
  const spans = async () => {
    if (!ts || !nq) return [];
    const B = nq * 128;
    const enc = oc(); for (const i of done) enc.resolveQuerySet(qs, i, 2, res, i * 128);
    enc.copyBufferToBuffer(res, 0, rd, 0, B); enc.clearBuffer(res, 0, B); oSubmit([enc.finish()]); done.clear();
    await rd.mapAsync(GPUMapMode.READ, 0, B); const t = new BigUint64Array(rd.getMappedRange(0, B).slice(0)); rd.unmap();
    const s = []; for (let i = 0; i < nq; i += 2) { const a = t[i * 16], b = t[i * 16 + 1]; if (a && b >= a) s.push([Number(a) / 1e6, Number(b) / 1e6]); }
    nq = 0; return s.sort((a, b) => a[0] - b[0]);
  };
  const stats = (s, units) => {
    let busy = 0, idle = 0; for (let i = 0; i < s.length; i++) { busy += s[i][1] - s[i][0]; if (i) idle += Math.max(0, s[i][0] - s[i - 1][1]); }
    const win = s.length ? s.at(-1)[1] - s[0][0] : 0;
    const e = encMs; encMs = 0;
    return { encodeMsCpu: +(e / units).toFixed(3), submits: +(s.length / units).toFixed(2), gpuMs: +(busy / units).toFixed(3), idleMs: +(idle / units).toFixed(3), windowMs: +(win / units).toFixed(3) };
  };
  const r2 = (x) => +x.toFixed(3);
  const out = {};
  const start = async (mtp) => { eng.reset(); if (eng.mtp) eng.mtpFill = mtp; if (eng.mtp?.stats) eng.mtp.stats = { drafts: 0, accepted: 0 }; await eng.prefillTokens(prefix.slice(0, -1)); return pick(await eng.forwardTokenIds(prefix.at(-1), desc)); };
  try {
    // 1. plain: wall, then GPU spans of the same loop
    const dbg = globalThis.Deno?.env.get("GAP_DEBUG") ? log : () => {};
    dbg("gap: start"); let nx = await start(false); dbg("gap: started");
    for (let i = 0; i < 4; i++) nx = pick(await eng.forwardTokenIds(nx, desc));
    let t0 = performance.now(); for (let i = 0; i < N; i++) nx = pick(await eng.forwardTokenIds(nx, desc));
    out.plainWallMs = r2((performance.now() - t0) / N); dbg("gap: wall done");
    on = true; t0 = performance.now(); for (let i = 0; i < N; i++) nx = pick(await eng.forwardTokenIds(nx, desc)); const w2 = (performance.now() - t0) / N; on = false; dbg("gap: timed loop done");
    out.plain = { wallMsTimed: r2(w2), ...stats(await spans(), N) };
    log("gap: plain done");
    // 2. CPU encode of one token, and back-to-back GPU throughput (pre-encoded, no readback in between)
    const pos0 = eng.pos, emb = eng._embedRowF32(nx);
    t0 = performance.now(); const jobs = []; for (let i = 0; i < N; i++) jobs.push(eng._encodeForward(pos0 + i, desc)); out.encodeMs = r2((performance.now() - t0) / N);
    await q.onSubmittedWorkDone();
    on = true;
    // re-encode with timestamps (the encoders above were made with on = false)
    const tj = []; for (let i = 0; i < N; i++) tj.push(eng._encodeForward(pos0 + i, desc));
    t0 = performance.now();
    for (let i = 0; i < N; i++) { eng._setFrame(pos0 + i, pos0 + i + 1); q.writeBuffer(eng.x, 0, emb); q.submit([tj[i].cb]); }
    await q.onSubmittedWorkDone(); const bb = (performance.now() - t0) / N; on = false;
    out.backToBack = { wallMs: r2(bb), ...stats(await spans(), N) };
    log("gap: back-to-back done");
    eng._fwdPre = null;
    // 3. speculative steps
    if (spec && eng.mtp && eng.specStep) {
      nx = await start(true);
      let toks = 0;
      for (let i = 0; i < 2; i++) nx = (await eng.specStep(nx, pick, K)).at(-1);
      const S = Math.max(8, N >> 1);
      t0 = performance.now(); for (let i = 0; i < S; i++) { const t = await eng.specStep(nx, pick, K); toks += t.length; nx = t.at(-1); }
      const sw = performance.now() - t0;
      on = true; let toks2 = 0; t0 = performance.now(); for (let i = 0; i < S; i++) { const t = await eng.specStep(nx, pick, K); toks2 += t.length; nx = t.at(-1); } const sw2 = performance.now() - t0; on = false;
      const sp = await spans();
      out.spec = { stepMs: r2(sw / S), tokPerStep: r2(toks / S), tokPerS: r2(toks / sw * 1000), stepMsTimed: r2(sw2 / S), tokPerStepTimed: r2(toks2 / S), ...stats(sp, S) };
      // a few consecutive submits: [GPU idle before it, GPU time] in ms
      out.spec.timeline = sp.slice(4, 16).map((x, i) => [r2(x[0] - sp[i + 3][1]), r2(x[1] - x[0])]);
    }
  } finally { device.createCommandEncoder = oc; q.submit = oSubmit; }
  log(JSON.stringify(out));
  return out;
}
