// The PC's share of the Flash demo's deal, loaded the way a room node loads it (shard.js loadShard), timed: what a
// re-deal costs on the device that joins. Defaults are the demo's deal (Mac tab 20 GB + RTX 5070 `--gb 10 --ram 43
// --disk`): layers 14-47, the experts of 17-47 parked in RAM, a 2.13 GiB GPU cache.
//   node packages/room-node/test/flash_load_bench.mjs
// env: MODELS (POOLED_MODELS: <dir>/q38flash/<files>), LO (14), HI (48), OFF_LO (17), VRAM_GIB (2.13), RAM_GB (43),
//      DISK_LO (none: layers [DISK_LO, HI) read their experts from disk), REPS (1)
import os from "node:os";
import path from "node:path";
import { setupNode } from "../env.js";
import { openModel } from "../source.js";
import { loadShard, flashExpertBytes } from "../shard.js";
import { ExpertStore } from "../../../engine/expert_store.js";

// the store's buffer allocations, timed apart from the copies into them (PARK_DETAIL=1)
const allocT = { ms: 0, n: 0 };
if (process.env.PARK_DETAIL === "1") {
  const a0 = ExpertStore.prototype._alloc;
  ExpertStore.prototype._alloc = function (...a) { const t = performance.now(); const n = this.park_.length; try { return a0.apply(this, a); } finally { if (this.park_.length > n) { const d = performance.now() - t; allocT.ms += d; allocT.n++; allocT.each = (allocT.each || []).concat(Math.round(d)); } } };
}

const env = (k, d) => process.env[k] ?? d;
await setupNode();
const MODELS = path.resolve(env("MODELS", env("POOLED_MODELS", path.join(os.homedir(), "llm/models"))));
const lo = +env("LO", 14), hi = +env("HI", 48), offLo = +env("OFF_LO", 17);
const diskLo = env("DISK_LO", "") === "" ? null : +env("DISK_LO");
for (let rep = 0; rep < +env("REPS", 1); rep++) {
  const src = openModel("qwen3.8-flash", { modelDir: MODELS });
  const G = await src.header(false);
  let park = 0;
  for (let i = offLo; i < (diskLo ?? hi); i++) park += flashExpertBytes(G, i);
  const offload = { lo: offLo, hi, vramBytes: Math.round(+env("VRAM_GIB", 2.13) * 2 ** 30), ramBytes: park, ramCap: +env("RAM_GB", 43) * 2 ** 30 - 2 * 2 ** 30,
    ...(diskLo != null ? { diskLo, diskRamBytes: 2 * 2 ** 30 } : {}) };
  const t0 = performance.now();
  let lastPct = -1;
  const r = await loadShard({ modelKey: "qwen3.8-flash", range: [lo, hi], hasEmbed: false, hasHead: false, src, offload,
    log: (s) => console.log(`${((performance.now() - t0) / 1000).toFixed(1)}s ${s}`),
    onProgress: (d, t) => { const p = Math.floor(100 * d / t / 10); if (p !== lastPct) { lastPct = p; console.log(`${((performance.now() - t0) / 1000).toFixed(1)}s ${p * 10}%`); } } });
  const s = (performance.now() - t0) / 1000;
  console.log(`LOADED layers ${lo}-${hi - 1} in ${s.toFixed(1)} s; parked ${(r.experts?.parkedBytes / 2 ** 30).toFixed(2)} GiB`);
  if (r.loadTimes) console.log("TIMES " + JSON.stringify(r.loadTimes));
  if (allocT.n) console.log(`park buffers: ${allocT.n} created in ${(allocT.ms / 1000).toFixed(1)} s: ${allocT.each.join(" ")} ms`);
  // decode steps through the share (STEPS: per mode): the same residual every step (after the first, every expert is in
  // the pools: compute and cuts only), each step's output fed back in (routing that moves), a fresh random residual each
  if (+env("STEPS", 0) > 0) {
    const fe = r.engine, W = fe.dims.hDim, st = r.experts;
    let h = new Float32Array(W).map(() => (Math.random() - 0.5) * 0.2), pos = 0;
    const rnd = () => new Float32Array(W).map(() => (Math.random() - 0.5) * 0.2);
    for (const mode of ["same", "feedback", "random"]) {
      const ts = [], s0 = st ? { ...st.stats } : null;
      for (let i = 0; i < +env("STEPS"); i++) {
        const x = mode === "random" ? rnd() : h, t = performance.now();
        const o = await fe.runHidden(x, pos++, { ids: [0] });
        ts.push(performance.now() - t);
        if (mode === "feedback") { let n = 0; for (const v of o) n += v * v; const k = 1 / Math.sqrt(n / o.length || 1) * 0.06; h = o.map((v) => v * k); }
        if (pos >= 1900) { fe.reset(); pos = 0; }
      }
      const slow = ts.filter((x) => x > 100).length;
      ts.sort((a, b) => a - b);
      const d = st ? { hits: st.stats.hits - s0.hits, misses: st.stats.misses - s0.misses, cuts: st.stats.cuts - s0.cuts } : null;
      console.log(`STEP ${mode}: median ${ts[ts.length >> 1].toFixed(1)} ms, p10 ${ts[ts.length / 10 | 0].toFixed(1)}, p90 ${ts[ts.length * 0.9 | 0].toFixed(1)}, max ${ts.at(-1).toFixed(0)}, ${slow} over 100 ms`
        + (d ? `; ${(d.misses / ts.length).toFixed(1)} misses a step, hits ${(100 * d.hits / Math.max(1, d.hits + d.misses)).toFixed(1)}%, ${(d.cuts / ts.length).toFixed(0)} cuts a step` : ""));
    }
  }
  await r.experts?.close().catch(() => {});
  try { await r.engine?.close?.(); } catch {}
  try { r.device.destroy(); } catch {}
  await src.close();
}
process.exit(0);
