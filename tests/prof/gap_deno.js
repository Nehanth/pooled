// tests/prof/decode_gap.js in Deno on the MoE (or MODEL=27b): where a token's wall time goes, plain and speculative.
//   cd tests && SET='{"verifyPass":false}' CTX=1024 deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights prof/gap_deno.js
//   SET: engine fields set after create (runtime switches); OPTS: create options; N tokens (default 32)
import { Qwen35Engine } from "../../engine/qwen35.js";
import { openGGUF, gpuDevice, trunkLayers, MOE_PATH, Q38_PATH } from "../load_model.js";
import { roomQwen35Options } from "../../engine/preset.js";
import { decodeGap } from "./decode_gap.js";
const env = (k, d) => Deno.env.get(k) ?? d;
const MODEL = env("MODEL", "moe"), CTX = +env("CTX", 0), N = +env("N", 32);
const model = openGGUF(MODEL === "moe" ? MOE_PATH : Q38_PATH);
const { device } = await gpuDevice();
const G = model.G, L = trunkLayers(G), tok = model.tokenizer();
const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true });
const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: Math.max(2048, CTX + 4 * N + 64),
  ...roomQwen35Options(""), ...JSON.parse(env("OPTS", "{}")) });
Object.assign(eng, JSON.parse(env("SET", "{}")));
let ids = tok.encode("<|im_start|>user\nWrite a long, detailed essay about the history of computing.<|im_end|>\n<|im_start|>assistant\n");
if (CTX > ids.length) {
  let src = [];
  for (const f of ["../../room.js", "../../engine/gguf.js", "../../engine/wgsl/base.js", "../../harness/agent.js"]) { src.push(...tok.encode(await Deno.readTextFile(new URL(f, import.meta.url)))); if (src.length > CTX) break; }
  while (src.length < CTX) src = src.concat(src);
  ids = [...src.slice(0, CTX - ids.length), ...ids];
}
console.log(`${MODEL}: set ${env("SET", "{}")}, context ${ids.length}`);
await decodeGap({ eng, device, prefix: ids, N, K: 3 });
