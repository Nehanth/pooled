// layerFuse A/B in one process: plain greedy decode (GPU sampling, the room's path), the decode layer fusions
// flipped at runtime (engine.layerFuse.<k>) in alternating blocks, so both arms share the same load, weights
// and any GPU contention. Reports the median ms/token of each arm and checks both arms give the same tokens.
//   cd tests && deno run --unstable-webgpu --allow-read --allow-env --allow-write=$HOME/.cache/swarmllm-weights bench/lf_ab.js
//   MODEL=moe|27b (default moe)  ROUNDS=8  BLOCK=24 (tokens per block)  FLAGS=comb,kv (flip only these; default all)
import { Qwen35Engine } from "../../engine/qwen35.js";
import { openGGUF, gpuDevice, trunkLayers, MOE_PATH, Q38_PATH } from "../load_model.js";
import { roomQwen35Options } from "../../engine/preset.js";
import { gpuGreedy } from "../gpusample_check.js";

const env = (k, d) => Deno.env.get(k) ?? d;
const MODEL = env("MODEL", "moe"), ROUNDS = +env("ROUNDS", 8), BLOCK = +env("BLOCK", 24), FLAGS = env("FLAGS", "");
const model = openGGUF(MODEL === "moe" ? MOE_PATH : Q38_PATH);
const { device } = await gpuDevice();
const G = model.G, L = trunkLayers(G), tok = model.tokenizer();
const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true });
const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: 2048,
  ...roomQwen35Options(""), layerFuse: true });
const KEYS = FLAGS ? FLAGS.split(",") : Object.keys(eng.layerFuse);
console.log(`attnDecode ${eng.attnDecode}, flipping ${KEYS.join(",")}, attn_dec_combine_g ${!!eng.pipes.attn_dec_combine_g}`);
const set = (on) => { for (const k of KEYS) eng.layerFuse[k] = on; };
const ids = tok.encode("<|im_start|>user\nWrite a long, detailed essay about the history of computing.<|im_end|>\n<|im_start|>assistant\n");

async function block(on) {
  set(on);
  eng.reset();
  await eng.prefillTokens(ids.slice(0, -1));
  let t = gpuGreedy(await eng.forwardTokenIds(ids.at(-1)));
  const out = [t];
  for (let i = 0; i < 4; i++) { t = gpuGreedy(await eng.forwardTokenIds(t)); out.push(t); }   // warm (encode-ahead primed)
  const t0 = performance.now();
  for (let i = 0; i < BLOCK; i++) { t = gpuGreedy(await eng.forwardTokenIds(t)); out.push(t); }
  return { ms: (performance.now() - t0) / BLOCK, out };
}
const res = { off: [], on: [] };
let ref = null, same = true;
for (let r = 0; r < ROUNDS; r++) {
  for (const on of r % 2 ? [true, false] : [false, true]) {
    const b = await block(on);
    res[on ? "on" : "off"].push(b.ms);
    if (!ref) ref = b.out; else same &&= b.out.every((x, i) => x === ref[i]);
  }
}
const med = (a) => { const s = [...a].sort((x, y) => x - y); return s.length % 2 ? s[s.length >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2; };
const mOff = med(res.off), mOn = med(res.on);
console.log(`${MODEL}: off ${res.off.map((x) => x.toFixed(2)).join(" ")} ms/token`);
console.log(`${MODEL}: on  ${res.on.map((x) => x.toFixed(2)).join(" ")} ms/token`);
console.log(`${MODEL}: median off ${mOff.toFixed(2)} ms (${(1000 / mOff).toFixed(2)} tok/s), on ${mOn.toFixed(2)} ms (${(1000 / mOn).toFixed(2)} tok/s), ${((mOff / mOn - 1) * 100).toFixed(1)}% tok/s; tokens ${same ? "identical" : "DIFFER"}`);
Deno.exit(same ? 0 : 1);
