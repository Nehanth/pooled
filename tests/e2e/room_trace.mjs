// Trace marks and GPU hooks shared by the room profilers (tests/e2e/room_prof.mjs on one machine,
// tests/e2e/xroom.mjs across two). Nothing here runs by itself.
//
//   patchRoom(src), patchTransport(src): room.js and room/transport.js as served with trace marks
//     (__HP(event, kind, pos, a, b) -> window.__hp) and the dev-only plain-decode switch
//     window.__nospec. They throw when an anchor moved, so a stale patch fails loudly.
//   INIT: an init script for every tab: the mark recorder, a GPU timestamp pair around every
//     command buffer while tracing (window.__gp.subs) and the time of every mapAsync (__gp.maps).
//   clockOffset(page): (timeOrigin + now) in the page against this process's clock.
//
// Mark events. Host: h.lap0 / h.emb1 / h.pack1 / h.ret (one lap), h.head0 / h.head1 (plain head),
// h.step0 / h.step1 (one speculative step: a = drafts offered, then tokens returned), h.unp0 / h.unp1
// (returned frame unpacked). Worker: w.enq (frame handed to the queue), w.start, w.unp1, w.gpu1
// (its layers ran and the hidden is back on the CPU), w.pack1. Transport, every tab: send0 / send1
// (first / last slice handed to the channel), rx0 / rx1 (first / last slice received), dlv (frame
// handed to the room in send order).

// ---- serve-time trace marks ----
const HP = "const __HP = (...a) => globalThis.__hpMark?.(...a);\n";
function rep(src, a, b, file) { if (!src.includes(a)) throw new Error(`${file} changed: anchor not found: ${a.slice(0, 80)}`); return src.replace(a, b); }
export function patchRoom(s) {
  const f = "room.js";
  s = HP + s;
  // plain-decode switch (room_latency's)
  s = rep(s, "else if (ai.engine.mtp && ai.engine.specStep) {", "else if (ai.engine.mtp && ai.engine.specStep && !window.__nospec) {", f);
  // host, one token
  s = rep(s, "  let h = await ai.engine.embedRun(id, pos);", "  __HP('h.lap0', 'ai-hidden', pos); let h = await ai.engine.embedRun(id, pos); __HP('h.emb1', 'ai-hidden', pos);", f);
  s = rep(s, "    sendChain({ t: \"ai-hidden\", pos, ...packWire(h) });\n    h = await returned;",
    "    { const __w = packWire(h); __HP('h.pack1', 'ai-hidden', pos); sendChain({ t: \"ai-hidden\", pos, ...__w }); }\n    h = await returned; __HP('h.ret', 'ai-hidden', pos);", f);
  s = rep(s, "  const logits = await ai.engine.headFromHidden(h);", "  __HP('h.head0', 'ai-hidden', pos); const logits = await ai.engine.headFromHidden(h); __HP('h.head1', 'ai-hidden', pos);", f);
  // GPU sampling (exp/gpu-sample, --query gpusample=1): the same marks around the candidates head
  if (s.includes("    const c = await ai.engine.headFromHiddenIds(h, desc);"))
    s = rep(s, "    const c = await ai.engine.headFromHiddenIds(h, desc);", "    __HP('h.head0', 'ai-hidden', pos); const c = await ai.engine.headFromHiddenIds(h, desc); __HP('h.head1', 'ai-hidden', pos);", f);
  // host, speculative verify lap
  s = rep(s, "          const tLap = performance.now();", "          const tLap = performance.now(); __HP('h.lap0', 'ai-hidden-b', pos, tokens.length);", f);
  s = rep(s, "          const hostMs = performance.now() - tLap;", "          const hostMs = performance.now() - tLap; __HP('h.emb1', 'ai-hidden-b', pos, tokens.length);", f);
  s = rep(s, "          sendChain({ t: \"ai-hidden-b\", basePos: pos, n: tokens.length, spec: 1, ...packWire(hb) });\n          const h = await returned;",
    "          { const __w = packWire(hb); __HP('h.pack1', 'ai-hidden-b', pos, tokens.length); sendChain({ t: \"ai-hidden-b\", basePos: pos, n: tokens.length, spec: 1, ...__w }); }\n          const h = await returned; __HP('h.ret', 'ai-hidden-b', pos, tokens.length);", f);
  s = rep(s, "        const toks = viaLookup ? await ai.engine.specStepDrafts(next, sample, lk, spec) : await ai.engine.specStep(next, sample, K, spec);",
    "        __HP('h.step0', viaLookup ? 'lookup' : 'draft', ai.engine.pos, viaLookup ? lk.length : K); const toks = viaLookup ? await ai.engine.specStepDrafts(next, sample, lk, spec) : await ai.engine.specStep(next, sample, K, spec); __HP('h.step1', viaLookup ? 'lookup' : 'draft', ai.engine.pos, toks.length);", f);
  // host, returned frames
  s = rep(s, "    case \"ai-hiddenret-b\": lapDone(\"b\" + d.basePos, unpackWire(d)); break;",
    "    case \"ai-hiddenret-b\": { __HP('h.unp0', 'ai-hiddenret-b', d.basePos); const __u = unpackWire(d); __HP('h.unp1', 'ai-hiddenret-b', d.basePos); lapDone(\"b\" + d.basePos, __u); break; }", f);
  s = rep(s, "    case \"ai-hiddenret\": lapDone(d.pos, unpackWire(d)); break;",
    "    case \"ai-hiddenret\": { __HP('h.unp0', 'ai-hiddenret', d.pos); const __u = unpackWire(d); __HP('h.unp1', 'ai-hiddenret', d.pos); lapDone(d.pos, __u); break; }", f);
  // worker
  s = rep(s, "      ai.q = ai.q.then(() => workerFrame(d))", "      __HP('w.enq', d.t, d.t === 'ai-hidden' ? d.pos : d.basePos); ai.q = ai.q.then(() => workerFrame(d))", f);
  s = rep(s, "  const t0 = performance.now();\n  if (d.t === \"ai-hidden-b\") {", "  const t0 = performance.now(); __HP('w.start', d.t, d.t === 'ai-hidden' ? d.pos : d.basePos, d.n || 1);\n  if (d.t === \"ai-hidden-b\") {", f);
  s = rep(s, "    const xs = unpackWire(d);", "    const xs = unpackWire(d); __HP('w.unp1', d.t, d.basePos);", f);
  s = rep(s, "    teleNote(d.spec ? \"spec\" : \"pre\", performance.now() - t0);", "    __HP('w.gpu1', d.t, d.basePos); teleNote(d.spec ? \"spec\" : \"pre\", performance.now() - t0);", f);
  s = rep(s, "    const bmsg = { basePos: d.basePos, n: nTok, ...(d.spec ? { spec: 1 } : {}), ...packWire(hb) };",
    "    const bmsg = { basePos: d.basePos, n: nTok, ...(d.spec ? { spec: 1 } : {}), ...packWire(hb) }; __HP('w.pack1', d.t, d.basePos);", f);
  s = rep(s, "    const hin = unpackWire(d);", "    const hin = unpackWire(d); __HP('w.unp1', d.t, d.pos);", f);
  s = rep(s, "    teleNote(\"one\", performance.now() - t0);", "    __HP('w.gpu1', d.t, d.pos); teleNote(\"one\", performance.now() - t0);", f);
  s = rep(s, "    const msg = { pos: d.pos, ...packWire(h) };", "    const msg = { pos: d.pos, ...packWire(h) }; __HP('w.pack1', d.t, d.pos);", f);
  return s;
}
export function patchTransport(s) {
  const f = "room/transport.js";
  s = HP + s;
  s = rep(s, "  const flags = packFlags(msg);", "  const flags = packFlags(msg); __HP('send0', msg.t, pos, bytes.length, nSlices);", f);
  s = rep(s, "    ch.send(buf);\n  }\n  return true;", "    ch.send(buf);\n  }\n  __HP('send1', msg.t, pos);\n  return true;", f);
  s = rep(s, "  if (!r) { r = {", "  if (!r) { __HP('rx0', KINDS[kind], pos, nSlices); r = {", f);
  s = rep(s, "  link.recv++;", "  link.recv++; __HP('rx1', KINDS[kind], pos);", f);
  s = rep(s, "    link.expect++;\n    onFrame(m);", "    link.expect++;\n    __HP('dlv', m.t, m.pos ?? m.basePos);\n    onFrame(m);", f);
  return s;
}
// ---- per-tab trace + GPU hooks (runs before any page script) ----
export const INIT = `(() => {
  const T = () => performance.timeOrigin + performance.now();
  window.__hpOn = false; window.__hp = []; window.__gp = { subs: [], maps: [] };
  window.__hpMark = (ev, kind, pos, a, b) => { if (window.__hpOn) window.__hp.push([ev, kind, pos, T(), a, b]); };
  if (!self.GPUAdapter) return;
  const oReq = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function (desc = {}) {
    const ts = this.features.has("timestamp-query");
    const d = await oReq.call(this, ts ? { ...desc, requiredFeatures: [...(desc.requiredFeatures || []), "timestamp-query"] } : desc);
    if (!ts) return d;
    const MAXQ = 4096, qs = d.createQuerySet({ type: "timestamp", count: MAXQ });
    let nq = 0; const cbRec = new WeakMap(); const did = (window.__gpDevs ||= []).length;
    const oEnc = d.createCommandEncoder.bind(d);
    d.createCommandEncoder = (dd) => {
      const e = oEnc(dd);
      if (!window.__hpOn || nq + 2 > MAXQ) return e;
      const rec = { q: nq, d: did, tEnc: T(), t: 0, gpu: 0 }; nq += 2;
      e.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: rec.q } }).end();
      const fin = e.finish.bind(e);
      e.finish = (x) => { e.beginComputePass({ timestampWrites: { querySet: qs, endOfPassWriteIndex: rec.q + 1 } }).end(); const cb = fin(x); cbRec.set(cb, rec); return cb; };
      return e;
    };
    const q = d.queue, oSub = q.submit.bind(q);
    q.submit = (cbs) => { if (window.__hpOn) { const t = T(); for (const cb of cbs) { const r = cbRec.get(cb) || { q: -1, tEnc: t }; r.t = t; window.__gp.subs.push(r); } } return oSub(cbs); };
    // the room makes several devices (a probe, the engine's, a throwaway): each resolves its own queries
    window.__gpDevs.push(async () => {
      if (!nq) return;
      const res = d.createBuffer({ size: nq * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      const rd = d.createBuffer({ size: nq * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const e = oEnc(); e.resolveQuerySet(qs, 0, nq, res, 0); e.copyBufferToBuffer(res, 0, rd, 0, nq * 8); oSub([e.finish()]);
      await rd.mapAsync(GPUMapMode.READ); const t = new BigUint64Array(rd.getMappedRange().slice(0)); rd.unmap();
      // q = -2 once resolved: the slots are reused, so a later resolve (a long trace, resolved
      // between answers) must not read them again
      for (const s of window.__gp.subs) if (s.d === did && s.q >= 0) { if (t[s.q + 1] > t[s.q]) s.gpu = Number(t[s.q + 1] - t[s.q]) / 1e6; s.q = -2; }
      nq = 0;
    });
    window.__gpResolve = async () => { for (const f of window.__gpDevs) await f(); };
    return d;
  };
  const oMap = GPUBuffer.prototype.mapAsync;
  GPUBuffer.prototype.mapAsync = function (...a) {
    if (!window.__hpOn) return oMap.apply(this, a);
    const r = { t0: T(), t1: 0, bytes: a[2] ?? this.size }; window.__gp.maps.push(r);
    return oMap.apply(this, a).then((v) => { r.t1 = T(); return v; });
  };
})();`;

// each browser has its own clock origin: offset of (timeOrigin + now) against this process's clock,
// from the lowest-round-trip of 15 probes (error <= half that round trip)
export async function clockOffset(p) {
  let best = null;
  for (let i = 0; i < 15; i++) {
    const a = performance.timeOrigin + performance.now();
    const t = await p.evaluate(() => performance.timeOrigin + performance.now());
    const b = performance.timeOrigin + performance.now();
    if (!best || b - a < best.rtt) best = { rtt: b - a, off: t - (a + b) / 2 };
  }
  return best;
}
