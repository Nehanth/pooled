// Expert offload's per-layer cut on Dawn / Deno: tests/bench/cut_core.js on the 35B MoE (fingerprint + decode tok/s +
// the cut's phases). The same as tests/bench/tab_cut.html in Chrome.
//   node --no-maglev tests/dawn_run.mjs tests/bench/cut_prof.js
// env: OFFLOAD / OFFLOAD_SLOTS / OFFLOAD_GB (tests/load_model.js offloadFromEnv; unset: resident), DECODE (64 tokens),
//      KICK=0|1 (engine mapKick; unset: Chrome only), VARIANTS (a JSON list of engine flag sets, one run each), BITS=0 (skip the fingerprint), ROOM_FLAGS, plus any engine flag in ENG='{"k":v}'
import { Qwen35Engine } from "../../engine/qwen35.js";
import { roomQwen35Options, applyRoomFlags } from "../../engine/preset.js";
import { openGGUF, gpuDevice, MOE_PATH, offloadFromEnv, roomFlags, trunkLayers } from "../load_model.js";
import { cutBench } from "./cut_core.js";
const env = (k, d) => Deno.env.get(k) ?? d;
const { device } = await gpuDevice();
device.addEventListener?.("uncapturederror", (e) => console.log("GPU ERROR " + e.error?.message?.slice(0, 200)));
const model = openGGUF(env("MOE", MOE_PATH)), G = model.G, L = trunkLayers(G), tok = model.tokenizer();
const experts = offloadFromEnv(device, L);
const weights = await model.weights({ lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: false, experts });
const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: 2048, ...roomQwen35Options(roomFlags()) });
applyRoomFlags(eng, roomFlags());
if (env("KICK")) eng.mapKick = env("KICK") === "1";
Object.assign(eng, JSON.parse(env("ENG", "{}")));
console.log(`35B: ${L} layers, ${eng.experts ? eng.experts.P + " experts cached per offloaded layer" : "resident"}, kick ${eng._kick()}`);
const prompt = Deno.readTextFileSync(new URL("../golden/q38_bits_prompt.txt", import.meta.url));
// VARIANTS='[{"cutPipe":false,"mapKick":false},{}]': engine flags per run, in one process (one load)
const base = Object.fromEntries([...new Set(JSON.parse(env("VARIANTS", "[{}]")).flatMap(Object.keys))].map((k) => [k, eng[k]]));
for (const v of JSON.parse(env("VARIANTS", "[{}]"))) {
  Object.assign(eng, base, v);   // each variant from the same base (flags of earlier variants do not carry over)
  console.log(`--- ${JSON.stringify(v)}: cutPipe ${eng.cutPipe === true} kick ${eng._kick()}`);
  await cutBench({ eng, tok, prompt, N: +env("DECODE", 64), bits: env("BITS", "1") !== "0" });
}
Deno.exit(0);
