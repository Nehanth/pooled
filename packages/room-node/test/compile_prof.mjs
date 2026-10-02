// Shader compile timing on this machine's WebGPU (Dawn): loads a shard like compile_check.mjs and
// prints one JSON line with the wall time of each phase (self-test, autotune, engine), every
// pipeline's compile time (request -> resolve), and the longest event-loop stalls (a timer that
// should fire every 20 ms; a synchronous compile shows up as one long gap).
//   node packages/room-node/test/compile_prof.mjs [model] [lo-hi] [head] [noselftest]
// DAWN_OPTS passes Dawn toggles as in compile_check.mjs.
import { setupNode } from "../env.js";
import { openModel } from "../source.js";
import { loadShard } from "../shard.js";
import os from "node:os";
import path from "node:path";

const model = process.argv[2] || "qwen3.6-35b-moe";
const [lo, hi] = (process.argv[3] || "0-2").split("-").map(Number);
const head = process.argv.includes("head");
const selfTest = !process.argv.includes("noselftest");
const T0 = performance.now();
await setupNode();
const pipes = [];
const P = globalThis.GPUDevice.prototype, orig = P.createComputePipelineAsync, origSync = P.createComputePipeline, origMod = P.createShaderModule;
let modMs = 0, nMod = 0, syncMs = 0, nSync = 0;
P.createComputePipelineAsync = function (desc) {
  const t = performance.now(), rec = { e: desc.compute?.entryPoint, t0: +(t - T0).toFixed(0) };
  pipes.push(rec);
  const p = orig.call(this, desc);
  rec.callMs = +(performance.now() - t).toFixed(1);   // time inside the call itself (blocks the loop)
  p.then(() => { rec.ms = +(performance.now() - t).toFixed(0); }, () => { rec.ms = -1; });
  return p;
};
P.createComputePipeline = function (desc) { const t = performance.now(); try { return origSync.call(this, desc); } finally { syncMs += performance.now() - t; nSync++; } };
const mods = [];
P.createShaderModule = function (desc) { mods.push(desc.code); const t = performance.now(); try { return origMod.call(this, desc); } finally { modMs += performance.now() - t; nMod++; } };
const stalls = [], bigStalls = [];
let last = performance.now();
const tick = setInterval(() => { const n = performance.now(); if (n - last > 100) { stalls.push(+(n - last).toFixed(0)); if (n - last > 1000) bigStalls.push(`${((last - T0) / 1000).toFixed(1)}s+${((n - last) / 1000).toFixed(1)}`); } last = n; }, 20);
const phases = {};
const modelDir = process.env.MODELS || process.env.POOLED_MODELS || path.join(os.homedir(), ".pooled", "models");
const src = openModel(model, { modelDir });
let error = null;
const tL = performance.now();
try {
  const r = await loadShard({ modelKey: model, range: [lo, hi], hasEmbed: head, hasHead: head, src, selfTest,
    log: (m) => { phases[m.split(" ")[0]] = +((performance.now() - tL) / 1000).toFixed(1); } });
  r.device.destroy();
} catch (e) { error = String(e?.message || e).split("\n")[0].slice(0, 400); }
phases.total = +((performance.now() - tL) / 1000).toFixed(1);
clearInterval(tick);
await src.close?.();
const slow = [...pipes].sort((a, b) => (b.ms || 0) - (a.ms || 0)).slice(0, 15).map((r) => `${r.e}:${r.ms}`);
const callBlock = pipes.reduce((s, r) => s + (r.callMs || 0), 0);
console.log(JSON.stringify({ model, range: [lo, hi], head, selfTest, dawn: process.env.DAWN_OPTS || "", error, phases,
  firstPipeS: +(pipes[0]?.t0 / 1000).toFixed(1), enginePipesFromS: +((pipes.find((r, i) => i && r.t0 - pipes[i - 1].t0 > 500)?.t0 || 0) / 1000).toFixed(1), pipelines: pipes.length, uniqueEntries: new Set(pipes.map((r) => r.e)).size, sumCompileS: +(pipes.reduce((s, r) => s + Math.max(0, r.ms || 0), 0) / 1000).toFixed(1),
  callBlockS: +(callBlock / 1000).toFixed(1), shaderModules: nMod, shaderModuleS: +(modMs / 1000).toFixed(1), syncPipes: nSync, syncS: +(syncMs / 1000).toFixed(1),
  maxStallMs: Math.max(0, ...stalls), stallsOver1s: bigStalls, stallSumS: +(stalls.reduce((a, b) => a + b, 0) / 1000).toFixed(1), slow,
  ...(process.env.PROF_ALL ? { all: pipes } : {}) }));
if (process.env.PROF_DUMP) { const fs = await import("node:fs"); mods.forEach((c, i) => fs.writeFileSync(`${process.env.PROF_DUMP}_${i}.wgsl`, c)); }
process.exit(error ? 1 : 0);
