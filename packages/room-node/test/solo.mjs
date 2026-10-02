// Reference answer: the whole model on one engine (no room code), greedy, same chat template.
//   node packages/room-node/test/solo.mjs "<prompt>" [maxNew] [model]  -> one JSON line { ids, text }
// env: MODELS (model dir; default <checkout>/models); OFFLOAD=lo-hi OFFLOAD_GB=n: those layers' experts in RAM with an
// n GB GPU cache (expert offload, as a room deal gives it: shard.js offload)
import { setupNode } from "../env.js";
import { openModel } from "../source.js";
import { loadShard, modelLayers } from "../shard.js";
import { buildIds, specials } from "../../../room/conversation.js";
import { greedy } from "../../../room/sampling.js";
import path from "node:path";

const prompt = process.argv[2] || "Why is the sky blue? Answer in two sentences.";
const maxNew = +(process.argv[3] || 64), model = process.argv[4] || "qwen3-1.7b";
await setupNode();
const src = openModel(model, { modelDir: path.resolve(process.env.MODELS || new URL("../../../models", import.meta.url).pathname) });
const L = await modelLayers(src);
const t0 = performance.now();
const OFF = process.env.OFFLOAD ? process.env.OFFLOAD.split("-").map(Number) : null;
const offload = OFF ? { lo: OFF[0], hi: OFF[1] + 1, vramBytes: +(process.env.OFFLOAD_GB || 4) * 2 ** 30 } : null;
const r = await loadShard({ modelKey: model, range: [0, L], hasEmbed: true, hasHead: true, src, selfTest: false, offload, log: (m) => console.error(m) });
const tLoad = performance.now() - t0;
const S = specials(r.tok), E = r.engine;
const ids = buildIds(r.tok, { system: "", turns: [{ role: "user", text: prompt }], thinking: false });
let logits;
for (const id of ids) logits = await E.forwardToken(id);
const out = [];
const t1 = performance.now();
for (let i = 0; i < maxNew; i++) {
  const n = greedy(logits);
  if (n === S.imEnd || n === S.eot) break;
  out.push(n);
  logits = await E.forwardToken(n);
}
const dt = (performance.now() - t1) / 1000;
console.log(JSON.stringify({ model, prompt, promptTokens: ids.length, ids: out, text: r.tok.decode(out), tps: +(out.length / dt).toFixed(1), loadS: +(tLoad / 1000).toFixed(1), gpuErrors: r.gpuErrors,
  ...(r.experts ? { offload: r.experts.summary() } : {}) }));
r.device.destroy();
process.exit(0);
