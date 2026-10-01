// Shader compile check on this machine's WebGPU (Dawn): builds every compute pipeline a room device
// would (GPU self-test, kernel micro-tests, autotune, the engine), records each pipeline that fails
// instead of stopping at the first one, and prints one JSON line. For Windows, where Dawn's D3D12
// backend compiles HLSL with FXC when DXC is missing and rejects some kernels the other backends take.
//   node packages/room-node/test/compile_check.mjs [model] [lo-hi] [head]
//     model: a key from room/models.js (default qwen3.6-35b-moe), from MODELS or ~/.pooled/models
//     lo-hi: the layers to load (default 0-2); "head": also the embed + LM head (the room host's shard)
// Exit 0 when every pipeline compiled. DAWN_OPTS passes Dawn toggles, e.g.
// DAWN_OPTS=enable-dawn-features=dump_shaders prints the HLSL of each shader.
import { setupNode } from "../env.js";
import { openModel } from "../source.js";
import { loadShard } from "../shard.js";
import os from "node:os";
import path from "node:path";

const model = process.argv[2] || "qwen3.6-35b-moe";
const [lo, hi] = (process.argv[3] || "0-2").split("-").map(Number);
const head = process.argv[4] === "head";
await setupNode();
const fails = [], pending = [], names = [];
let n = 0;
const P = globalThis.GPUDevice.prototype, orig = P.createComputePipelineAsync, origSync = P.createComputePipeline;
P.createComputePipelineAsync = function (desc) {
  n++; names.push(desc.compute?.entryPoint);
  const p = orig.call(this, desc).catch((e) => {
    fails.push({ entryPoint: desc.compute?.entryPoint, error: String(e?.message || e).split("\n").slice(0, 3).join(" | ").slice(0, 400) });
    throw e;
  });
  pending.push(p);
  return p;
};
P.createComputePipeline = function (desc) { n++; return origSync.call(this, desc); };
const modelDir = process.env.MODELS || process.env.POOLED_MODELS || path.join(os.homedir(), ".pooled", "models");
const src = openModel(model, { modelDir });
let error = null;
try {
  const r = await loadShard({ modelKey: model, range: [lo, hi], hasEmbed: head, hasHead: head, src, selfTest: true, log: () => {} });
  r.device.destroy();
} catch (e) { error = String(e?.message || e).split("\n")[0].slice(0, 400); }
await Promise.allSettled(pending);   // a failed Promise.all leaves its other compiles running: count them too
await src.close?.();
console.log(JSON.stringify({ model, range: [lo, hi], head, pipelines: n, ok: !fails.length && !error, fails, error,
  ...(process.env.COMPILE_LIST ? { entryPoints: [...new Set(names)].sort() } : {}) }));
process.exit(fails.length || error ? 1 : 0);
