// Deno: decode profile of the MoE (or Q38=1 for the dense 27B), see tests/prof/moe_decode_prof.js.
//   cd tests && deno run --unstable-webgpu --allow-read --allow-env --allow-write prof/prof_moe_decode.js [out.json]
// env: TOKENS (plain tokens, 24), STEPS (speculative steps, 12), K (3), MOE=<gguf>, Q38=1
//      ROOM_FLAGS: the room's switches (engine/preset.js), e.g. ROOM_FLAGS="gpusample=0"; unset: the room's settings
import { Qwen35Engine } from "../../engine/qwen35.js";
import { makeTokenizer, argmax } from "../../engine/engine.js";
import { parseGGUFHeader, qwen35Weights, tokenizerFromGGUF } from "../../engine/gguf.js";
import { profileDecode } from "./moe_decode_prof.js";
import { roomQwen35Options, applyRoomFlags } from "../../engine/preset.js";
import { roomFlags } from "../load_model.js";
const here = new URL("../..", import.meta.url).pathname;
const PATH = Deno.env.get("Q38") ? here + "models/q38/model.gguf" : (Deno.env.get("MOE") || here + "models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf");
const fh = await Deno.open(PATH);
const readAt = async (off, len) => { await fh.seek(off, 0); const o = new Uint8Array(len); let g = 0; while (g < len) { const n = await fh.read(o.subarray(g)); if (n === null) break; g += n; } return o; };
const ad = await navigator.gpu.requestAdapter();
const feats = ad.features.has("timestamp-query") ? ["timestamp-query"] : [];
const device = await ad.requestDevice({ requiredFeatures: feats, requiredLimits: { maxBufferSize: ad.limits.maxBufferSize, maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize } });
device.addEventListener?.("uncapturederror", (e) => console.error("GPU ERROR:", e.error?.message));
const G = parseGGUFHeader((await readAt(0, 64 << 20)).buffer); const tok = makeTokenizer(tokenizerFromGGUF(G.meta));
const arch = G.meta["general.architecture"], nBlk = G.meta[arch + ".block_count"], L = nBlk - (G.meta[arch + ".nextn_predict_layers"] || 0);
const t0 = performance.now();
const weights = await qwen35Weights(G, (i) => readAt(i.byteOffset, i.byteLength), { lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true });
// the room's settings (engine/preset.js); GPU_SAMPLE=0 / ARGMAX_WIDE=0|1 still work, as ?gpusample / ?argmaxwide
const flags = roomFlags({ ...(Deno.env.get("GPU_SAMPLE") ? { gpusample: Deno.env.get("GPU_SAMPLE") } : {}), ...(Deno.env.get("ARGMAX_WIDE") ? { argmaxwide: Deno.env.get("ARGMAX_WIDE") } : {}) });
const eng = applyRoomFlags(await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: 512,
  vocab: G.tensors["token_embd.weight"]?.shape?.[0], ...roomQwen35Options(flags) }), flags);
console.log(`${arch}: loaded in ${((performance.now() - t0) / 1000).toFixed(0)}s, mtp ${!!eng.mtp}, moeFuse ${!!eng.moeFuse}, fuseProj ${eng.fuseProj}, draftChain ${!!eng.draftChain}, specFuse ${eng.specFuse}, batchCols ${eng.NC}, draftVocab ${eng.draftVocab || "off"}, timestamps ${feats.length > 0}`);
const out = await profileDecode({ eng, device, tok, argmax, log: (s) => console.log(s), N: +(Deno.env.get("TOKENS") || 24), STEPS: +(Deno.env.get("STEPS") || 12), K: +(Deno.env.get("K") || 3) });
if (Deno.args[0]) Deno.writeTextFileSync(Deno.args[0], JSON.stringify({ runtime: "deno", arch, ...out }, null, 1));
Deno.exit(0);
