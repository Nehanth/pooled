// Probe: CPU encode time vs GPU time inside a real decode token.
import { Qwen35Engine } from "../engine/qwen35.js";
import { parseGGUFHeader, qwen35Weights } from "../engine/gguf.js";
import { roomQwen35Options, applyRoomFlags } from "../engine/preset.js";
import { roomFlags } from "../tests/load_model.js";
const openFile = async (path) => { const fh = await Deno.open(path); return async (off, len) => { await fh.seek(off, Deno.SeekMode.Start); const out = new Uint8Array(len); let got = 0; while (got < len) { const n = await fh.read(out.subarray(got)); if (n === null) break; got += n; } return out; }; };
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
const readAt = await openFile(new URL("../models/q38/model.gguf", import.meta.url).pathname);
const G = parseGGUFHeader((await readAt(0, 64 << 20)).buffer, { skipTokenizer: true });
const L = +(Deno.env.get("LAYERS") || 64);
const weights = await qwen35Weights(G, (i) => readAt(i.byteOffset, i.byteLength), { lo: 0, hi: L, hasEmbed: true, hasHead: true });
// the room's settings (engine/preset.js); ROOM_FLAGS takes the room's ?flags, e.g. ROOM_FLAGS="fuse=0"
const flags = roomFlags();
const eng = applyRoomFlags(await Qwen35Engine.create({ device, meta: G.meta, weights, vocab: 248320, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: 512, ...roomQwen35Options(flags) }), flags);
const dev = device, vocab = eng.dims.vocab;
async function probe(n = 25) {
  eng.reset(); eng.pos = 0;
  for (let i = 0; i < 5; i++) await eng.forwardToken(10);
  const E = [], S = [], W = [], T = [];
  for (let i = 0; i < n; i++) {
    const tA = performance.now();
    eng._setFrame(eng.pos, eng.pos + 1);
    dev.queue.writeBuffer(eng.x, 0, eng._embedRowF32(10));
    const tB = performance.now();
    const enc = dev.createCommandEncoder();
    for (let l = 0; l < eng.layers.length; l++) eng._encodeLayer(enc, l);
    { const p = enc.beginComputePass(); eng._d(p, "rmsnorm", eng.bgFinalNorm, 256, 256); eng._dop(p, eng.headOp); p.end(); }
    const cb = enc.finish();
    const tC = performance.now();
    dev.queue.submit([cb]);
    const tD = performance.now();
    await eng._readback(eng.logits, eng.stageLogits, vocab);
    const tE = performance.now();
    eng.pos++;
    W.push(tB - tA); E.push(tC - tB); S.push(tD - tC); T.push(tE - tA);
  }
  const med = a => { a = a.slice().sort((x, y) => x - y); return a[a.length >> 1]; };
  console.log(`writes(frame+embed): ${med(W).toFixed(2)} ms | encode: ${med(E).toFixed(2)} ms | submit: ${med(S).toFixed(2)} ms | total: ${med(T).toFixed(2)} ms (wait+readback is the rest)`);
  console.log(`encode min ${Math.min(...E).toFixed(2)} max ${Math.max(...E).toFixed(2)} ; total/token ${med(T).toFixed(2)} ms => ${(1000/med(T)).toFixed(2)} tok/s`);
}
await probe();
