// Off-GPU cost of one decode token: CPU encode time, and submit + readback of the logits, measured apart.
import { Qwen35Engine } from "../engine/qwen35.js";
import { parseGGUFHeader, qwen35Weights } from "../engine/gguf.js";
import { roomQwen35Options, applyRoomFlags } from "../engine/preset.js";
import { roomFlags } from "./load_model.js";
const fh = await Deno.open(Deno.env.get("MOE") || "../models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf");
const readAt = async (off, len) => { await fh.seek(off, 0); const o = new Uint8Array(len); let g = 0; while (g < len) { const n = await fh.read(o.subarray(g)); if (n === null) break; g += n; } return o; };
const ad = await navigator.gpu.requestAdapter(); const device = await ad.requestDevice({ requiredLimits: { maxBufferSize: ad.limits.maxBufferSize, maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize } });
const G = parseGGUFHeader((await readAt(0, 64 << 20)).buffer); const arch = G.meta["general.architecture"], L = G.meta[arch + ".block_count"] - 1;
// the room's settings (engine/preset.js); ROOM_FLAGS takes the room's ?flags, e.g. ROOM_FLAGS="gpusample=0"
const flags = roomFlags();
const eng = applyRoomFlags(await Qwen35Engine.create({ device, meta: G.meta, weights: await qwen35Weights(G, (i) => readAt(i.byteOffset, i.byteLength), { lo: 0, hi: L, hasEmbed: true, hasHead: true }), layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: 512,
  vocab: G.tensors["token_embd.weight"]?.shape?.[0], ...roomQwen35Options(flags) }), flags);
for (let i = 0; i < 5; i++) await eng.forwardToken(1);
const N = 30; let enc = 0, t0;
// 1) CPU encode only (never submitted)
for (let i = 0; i < N; i++) { t0 = performance.now(); const e = device.createCommandEncoder(); for (let l = 0; l < eng.layers.length; l++) eng._encodeLayer(e, l); e.finish(); enc += performance.now() - t0; }
// 2) submit an empty command buffer and read back the logits
let rb = 0; for (let i = 0; i < N; i++) { t0 = performance.now(); device.queue.submit([device.createCommandEncoder().finish()]); await eng._readback(eng.logits, eng.stageLogits, eng.dims.vocab); rb += performance.now() - t0; }
// 3) a tiny 4-byte readback, to separate "wait for GPU" from "copy 1 MB"
const s4 = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }); let tiny = 0;
for (let i = 0; i < N; i++) { t0 = performance.now(); const e = device.createCommandEncoder(); e.copyBufferToBuffer(eng.logits, 0, s4, 0, 4); device.queue.submit([e.finish()]); await s4.mapAsync(GPUMapMode.READ); s4.unmap(); tiny += performance.now() - t0; }
let full = 0; for (let i = 0; i < N; i++) { t0 = performance.now(); await eng.forwardToken(1); full += performance.now() - t0; }
console.log(`encode ${(enc / N).toFixed(2)} ms · submit+logits readback (${(eng.dims.vocab * 4 / 2 ** 20).toFixed(2)} MB) ${(rb / N).toFixed(2)} ms · 4-byte readback ${(tiny / N).toFixed(2)} ms · whole token ${(full / N).toFixed(2)} ms`);
