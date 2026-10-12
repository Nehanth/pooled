// room/tele.js: a worker's ai-tele (compute per frame kind, the expert cache's hits and misses) and what the host
// makes of it for the speed panel's device cards. No DOM, times passed in.
import { TeleMeter, expertStoreOf, expertCounters, expertDelta, expertCache, expertCacheOf, TELE_EVERY_MS } from "../../room/tele.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const near = (a, b, m, e = 1e-6) => { if (!(Math.abs(a - b) <= e)) throw new Error((m || "mismatch") + `: ${a} != ${b}`); };
// engine/expert_store.js's shape: stats (with the disk tier's diskReads / diskBytes), diskSet: the disk layers
const store = (hits, misses, disk) => ({ stats: { cuts: 0, lookups: hits + misses, hits, misses, diskReads: disk?.reads || 0, diskBytes: disk?.bytes || 0 }, diskSet: new Set(disk ? [40, 41] : []) });

Deno.test("TeleMeter: an EMA per frame kind, due every 700 ms, the same message a browser worker always sent", () => {
  const T = new TeleMeter();
  T.note("one", 10, 1, 3.04); T.note("one", 20, 1, 1); T.note("spec", 40, 4);
  eq(T.due(100), true, "the first report at once (a process younger than 700 ms too)");
  const m = T.report(100);
  eq(m, { t: "ai-tele", k: { one: 13, spec: 40 }, amax: 3 }, "no exp without an expert store");
  eq(T.due(700), false, "then not before 700 ms"); eq(T.due(TELE_EVERY_MS + 101), true);
  T.note("one", 5); eq(T.report(1600).amax, 0, "amax is per report");
});

Deno.test("TeleMeter: with an expert store, the hits and misses since the last report, over the tokens run", () => {
  const T = new TeleMeter(), S = store(0, 0);
  T.note("one", 10); S.stats.hits = 30; S.stats.misses = 10;
  eq(T.report(800, S).exp, { hits: 30, misses: 10, tok: 1, all: [30, 10] });
  for (let i = 0; i < 4; i++) T.note("one", 10);
  S.stats.hits = 66; S.stats.misses = 14;
  eq(T.report(1600, S).exp, { hits: 36, misses: 4, tok: 4, all: [66, 14] });
  // a new load: a new store, counted from zero
  const S2 = store(5, 5); T.note("spec", 30, 8);
  eq(T.report(2400, S2).exp, { hits: 5, misses: 5, tok: 8, all: [5, 5] });
  // disk reads, when the store reads experts from disk
  const S3 = store(0, 2, { reads: 6, bytes: 6 * 2 ** 20 });
  T.note("one", 10);
  eq(T.report(3200, S3).exp, { hits: 0, misses: 2, tok: 1, all: [0, 2], reads: 6, readBytes: 6 * 2 ** 20 });
});

Deno.test("expertCounters / expertDelta: null without a store, never negative (lower counters: a reset, from zero)", () => {
  eq(expertCounters(null), null); eq(expertCounters({}), null); eq(expertCounters({ stats: { hits: "x" } }), null);
  eq(expertDelta(null, null, 3), null);
  eq(expertDelta({ hits: 10, misses: 10 }, { hits: 4, misses: 2 }, 2), { hits: 4, misses: 2, tok: 2, all: [4, 2] });
});

Deno.test("expertCache: the host's view of a report: hit rate, misses per token, disk reads; empty windows keep the last", () => {
  eq(expertCache(undefined), null, "a worker that predates exp");
  eq(expertCache({ hits: 0, misses: 0, tok: 3, all: [0, 0] }), null, "no lookups yet");
  const a = expertCache({ hits: 84, misses: 16, tok: 4, all: [84, 16] });
  eq([a.hit, a.missPerTok, a.allPct], [84, 4, 84]);
  eq(a.readsPerTok, undefined, "no disk fields without a disk reader");
  // a prefill window looks nothing up: the last rate stays, the since-load rate follows
  const b = expertCache({ hits: 0, misses: 0, tok: 512, all: [84, 16] }, a);
  eq(b.hit, 84);
  const c = expertCache({ hits: 9, misses: 1, tok: 2, all: [93, 17], reads: 3, readBytes: 3 * 2 ** 20 }, b);
  eq([c.hit, c.missPerTok, c.readsPerTok, c.readMBPerTok], [90, 0.5, 1.5, 1.5]);
  near(c.allPct, 100 * 93 / 110);
  // untrusted input: junk is zero, not NaN
  const d = expertCache({ hits: "lots", misses: -4, tok: null, all: "x" });
  eq(d, null);
  const e = expertCache({ hits: 1, misses: "2", tok: "z" });
  near(e.hit, 100 / 3); eq(e.missPerTok, null);
  eq(expertCache("nope", a), a);
});

Deno.test("expertCacheOf: a host's own store, since its load", () => {
  eq(expertCacheOf(null), null);
  eq(expertCacheOf(store(0, 0)), null);
  eq(expertCacheOf(store(3, 1)).hit, 75);
});

Deno.test("TeleMeter: a store's resetStats counts from zero; prompt frames add no tokens to exp.tok", () => {
  const T = new TeleMeter(), S = store(50, 50);
  T.note("one", 10); T.report(800, S);
  S.stats = { ...S.stats, hits: 3, misses: 1 };   // resetStats: a fresh stats object, lower counters
  T.note("one", 10); T.note("pre", 300, 512);
  eq(T.report(1600, S).exp, { hits: 3, misses: 1, tok: 1, all: [3, 1] });
  // a store without disk layers reports no reads, even with the fields at zero
  eq("reads" in T.report(2400, S).exp, false);
});

Deno.test("expertStoreOf: the engine's store, or its core's behind the Flash adapter (room/flash.js)", () => {
  const S = store(1, 1);
  eq(expertStoreOf(null), null); eq(expertStoreOf({}), null);
  eq(expertStoreOf({ experts: S }) === S, true);
  eq(expertStoreOf({ core: { experts: S } }) === S, true);
});

Deno.test("expertCounters on engine/expert_store.js's own ExpertStore: its stats and its disk layers, as the engines hold it", async () => {
  const { ExpertStore } = await import("../../engine/expert_store.js");
  const dev = { limits: { maxBufferSize: 2 ** 30 } };
  const ram = new ExpertStore(dev, { layers: [10, 11] });
  eq(expertCounters(ram), { hits: 0, misses: 0 }, "parked in RAM: no disk fields");
  const disk = new ExpertStore(dev, { layers: [10, 11, 12], disk: { layers: [12], reader: {} } });
  disk.stats.hits = 8; disk.stats.misses = 2; disk.stats.diskReads = 2; disk.stats.diskBytes = 2 * 2 ** 20;
  eq(expertCounters(disk), { hits: 8, misses: 2, reads: 2, readBytes: 2 * 2 ** 20 });
  eq(expertStoreOf({ core: { experts: disk } }) === disk, true);
});
