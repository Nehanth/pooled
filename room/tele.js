// Worker -> host telemetry (ai-tele): what a device holding layers reports about its own part of each token, for the
// host's room map and the speed panel's device cards (room/speedo.js). The same code in a browser worker (room.js
// teleNote) and a room node worker (packages/room-node/roomnode.js), both fed by room/pipeline.js workerFrame.
// Pure: no clock of its own (callers pass performance.now()), no transport (report() returns the message).
//
//   { t: "ai-tele", k: { one | spec | pre: compute ms per frame, EMA }, amax: the largest |x| on the wire,
//     exp?: { hits, misses, tok, all: [hits, misses], reads?, readBytes? } }
//
// exp (optional; only from a device whose engine offloads experts, engine/expert_store.js): the expert cache's
// lookups since the last report (hits, misses: experts already in the GPU slot pool, or copied in from RAM), the
// tokens this device ran in that time (tok), the counts since the load (all), and, when the store reads experts
// from disk (a disk reader at store.disk with expert_disk.js's stats), the disk reads and bytes since the last
// report. A host that predates it ignores it; a worker that predates it sends none.

export const TELE_EVERY_MS = 700;

// an ExpertStore's counters now (cumulative since its load); null without a store that looks up experts
export function expertCounters(store) {
  const s = store?.stats;
  if (!s || !Number.isFinite(s.hits) || !Number.isFinite(s.misses)) return null;
  const out = { hits: s.hits, misses: s.misses };
  const d = store.disk?.stats;
  if (d && Number.isFinite(d.parts)) { out.reads = d.parts; out.readBytes = +d.bytes || 0; }
  return out;
}

// the exp field of a report: the counters' change since `prev` (null: since zero), over `tok` tokens
export function expertDelta(prev, cur, tok) {
  if (!cur) return null;
  const p = prev || { hits: 0, misses: 0, reads: 0, readBytes: 0 };
  const out = { hits: Math.max(0, cur.hits - p.hits), misses: Math.max(0, cur.misses - p.misses), tok: Math.max(0, tok | 0), all: [cur.hits, cur.misses] };
  if (cur.reads != null) { out.reads = Math.max(0, cur.reads - (p.reads || 0)); out.readBytes = Math.max(0, cur.readBytes - (p.readBytes || 0)); }
  return out;
}

// A worker's meter: note() every frame it computes, report() when due (at most every TELE_EVERY_MS).
export class TeleMeter {
  constructor(every = TELE_EVERY_MS) { this.every = every; this.k = {}; this.at = 0; this.amax = 0; this.tok = 0; this.xPrev = null; this.xStore = null; }
  // kind: "one" | "spec" | "pre"; ms: this frame's compute; n: its tokens; amax: the wire's largest |x| (wire.js)
  note(kind, ms, n = 1, amax = 0) {
    this.amax = Math.max(this.amax, +amax || 0);
    const k = this.k[kind] ||= { ema: ms, n: 0 };
    k.ema = k.n ? 0.7 * k.ema + 0.3 * ms : ms; k.n++;
    this.tok += n > 0 ? n : 1;
  }
  due(now) { return now - this.at > this.every; }
  // the ai-tele message, and the window starts again; store: the engine's ExpertStore (engine.experts), if any
  report(now, store = null) {
    this.at = now;
    const msg = { t: "ai-tele", k: Object.fromEntries(Object.entries(this.k).map(([a, b]) => [a, Math.round(b.ema * 10) / 10])), amax: Math.round(this.amax * 10) / 10 };
    const cur = expertCounters(store);
    if (store !== this.xStore) { this.xStore = store; this.xPrev = null; }   // a new load: its counters start at zero
    if (cur) { msg.exp = expertDelta(this.xPrev, cur, this.tok); this.xPrev = cur; }
    this.amax = 0; this.tok = 0;
    return msg;
  }
}

// Host side: a report's exp -> what the device card shows, { hit: % of lookups that hit the GPU cache, hits,
// misses, missPerTok, readsPerTok?, readMBPerTok?, all: % since the load }. A report with no lookups (an idle
// window, or prefill frames: they copy whole layers and look nothing up) keeps the previous one. Untrusted input:
// numbers only, else null.
const num = (v) => (Number.isFinite(+v) && +v >= 0 ? +v : 0);
export function expertCache(exp, prev = null) {
  if (!exp || typeof exp !== "object") return prev;
  const hits = num(exp.hits), misses = num(exp.misses), tok = num(exp.tok), look = hits + misses;
  const all = Array.isArray(exp.all) ? num(exp.all[0]) + num(exp.all[1]) : 0;
  const allPct = all ? 100 * num(exp.all[0]) / all : prev?.allPct ?? null;
  if (!look) return prev ? { ...prev, allPct } : null;
  const out = { hit: 100 * hits / look, hits, misses, missPerTok: tok ? misses / tok : null, allPct };
  if (exp.reads != null) { out.readsPerTok = tok ? num(exp.reads) / tok : null; out.readMBPerTok = tok ? num(exp.readBytes) / tok / 2 ** 20 : null; }
  return out;
}
// the host's own store (a node host, or a browser that offloads one day): since its load
export function expertCacheOf(store) {
  const c = expertCounters(store);
  return c ? expertCache({ hits: c.hits, misses: c.misses, tok: 0, all: [c.hits, c.misses] }) : null;
}
