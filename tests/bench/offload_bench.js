// Expert offload: prefill and decode tok/s with and without it, the cache's hit rate and the bytes it copies.
// Per prompt length: reset, prefill n - 1 tokens + the last one (forwardTokenIds), then DECODE greedy tokens; the store's
// counters are read separately for the prefill and the decode.
//   MODEL=moe|122b [MOE=path] LENS=512,2048 DECODE=128   OFFLOAD / OFFLOAD_GB / OFFLOAD_SLOTS as tests/load_model.js offloadFromEnv
//   Deno:  deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights tests/bench/offload_bench.js
//   Dawn:  node --no-maglev tests/dawn_run.mjs tests/bench/offload_bench.js   (Deno's ~12 ms mapAsync dominates offloaded decode)
import { Qwen35Engine } from "../../engine/qwen35.js";
import { roomQwen35Options, applyRoomFlags } from "../../engine/preset.js";
import { openGGUF, gpuDevice, watchGpuErrors, trunkLayers, MOE_PATH, Q122_PATH, streamWeights, offloadFromEnv, roomFlags } from "../load_model.js";
import { gpuGreedy } from "../gpusample_check.js";
const env = (k, d) => Deno.env.get(k) ?? d;
const BIG = env("MODEL", "moe") === "122b", LENS = env("LENS", "512,2048").split(",").map(Number), DEC = +env("DECODE", 128);
const { device } = await gpuDevice();
const errors = watchGpuErrors(device);
const model = openGGUF(env("MOE", BIG ? Q122_PATH : MOE_PATH));   // MOE=path: another copy of the file
const G = model.G, L = trunkLayers(G), tok = model.tokenizer();
const experts = offloadFromEnv(device, L);
const t0 = performance.now();
const range = { lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: false, experts };
const weights = BIG ? await streamWeights(model, device, range) : await model.weights(range);
const maxSeq = Math.ceil((Math.max(...LENS) + DEC + 64) / 256) * 256;
// the room's engine settings (engine/preset.js; ROOM_FLAGS for its query switches)
const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq, ...roomQwen35Options(roomFlags()) });
applyRoomFlags(eng, roomFlags());
console.log(`${BIG ? "122B" : "35B"}: ${L} layers, loaded in ${((performance.now() - t0) / 1000).toFixed(0)} s; ${eng.experts ? eng.experts.summary() : "no offload"}`);
let src = [];
for (const f of ["../../engine/qwen35.js", "../../engine/gguf.js", "../../engine/wgsl/moe.js"]) src.push(...tok.encode(Deno.readTextFileSync(new URL(f, import.meta.url))));   // ~70K tokens of this repo's source
const st = () => ({ ...(eng.experts?.stats || {}) });
const d = (a, b, k) => (b[k] || 0) - (a[k] || 0);
for (const n of LENS) {
  const ids = src.slice(0, n);
  eng.reset();
  const s0 = st(), p0 = performance.now();
  await eng.prefillTokens(ids.slice(0, -1));
  let next = gpuGreedy(await eng.forwardTokenIds(ids[n - 1]));
  const tp = (performance.now() - p0) / 1000, s1 = st(), d0 = performance.now();
  for (let i = 0; i < DEC; i++) next = gpuGreedy(await eng.forwardTokenIds(next));
  const td = (performance.now() - d0) / 1000, s2 = st();
  let line = `n ${n}: prefill ${(n / tp).toFixed(1)} tok/s (${tp.toFixed(2)} s) · decode ${(DEC / td).toFixed(2)} tok/s`;
  if (eng.experts) {
    line += `\n  prefill: ${d(s0, s1, "layerLoads")} whole-layer loads (${(d(s0, s1, "layerBytes") / 2 ** 30).toFixed(1)} GiB), ${d(s0, s1, "regionCuts")} region cuts (${(d(s0, s1, "regionBytes") / 2 ** 20).toFixed(0)} MiB)`;
    const lk = d(s1, s2, "lookups"), h = d(s1, s2, "hits");
    line += `\n  decode: hits ${(100 * h / Math.max(1, lk)).toFixed(1)}% (${(lk - h) / DEC} misses / token), ${(d(s1, s2, "bytes") / 2 ** 20 / DEC).toFixed(1)} MiB to the pools / token, ${d(s1, s2, "regionCuts")} region cuts`;
  }
  console.log(line);
}
console.log(`gpu errors ${errors.count}`);
Deno.exit(errors.count ? 1 : 0);
