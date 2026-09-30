// Qwen35Engine keeps each attention layer's K and V cache in one buffer, bound whole: at create it
// refuses a context whose buffer the device cannot bind (a readable error instead of invalid bind
// groups later), accepts one that fits exactly, and does not care when its layer range holds no
// attention layer. The room keeps its context within every device's limit (room/models.js
// ctxForBinding, tests/unit/models_test.js). CPU only: synthetic model + recording mock device.
//   deno test --no-check --allow-read tests/unit/kv_bind_limit_test.js
import { mockDevice } from "./mock_gpu.js";
import { buildSynthGGUF } from "../e2e/synth.mjs";
import { parseGGUFHeader, qwen35Weights, GGML_EMBED } from "../../engine/gguf.js";
import { Qwen35Engine } from "../../engine/qwen35.js";
import { kvLayerBufBytes } from "../../room/models.js";

const eq = (a, b, m) => { if (a !== b) throw new Error(`${m}: ${a} != ${b}`); };
const buf = buildSynthGGUF({ layers: 8 }).bytes;
const G = parseGGUFHeader(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const bytesOf = async (i) => buf.slice(i.byteOffset, i.byteOffset + i.byteLength);
const L = G.meta["qwen35.block_count"] - 1;

async function make({ maxSeq, bind, kvQ8 = false, lo = 0, hi = L }) {
  const device = mockDevice();
  device.limits.maxStorageBufferBindingSize = bind;
  const head = hi === L, emb = lo === 0;
  const weights = await qwen35Weights(G, bytesOf, { lo, hi, hasEmbed: emb, hasHead: head });
  const origWarn = console.warn; console.warn = () => {};
  try {
    return await Qwen35Engine.create({ device, meta: G.meta, layerRange: [lo, hi], hasEmbed: emb, hasHead: head, vocab: G.tensors[GGML_EMBED].shape[0],
      maxSeq, batchCols: 16, coopRowsB: 1, weights, kvQ8 });
  } finally { console.warn = origWarn; }
}

Deno.test("KV buffer exactly at the binding limit: created, one buffer per K and V", async () => {
  const maxSeq = 2048, need = kvLayerBufBytes(G.meta, maxSeq);
  const eng = await make({ maxSeq, bind: need });
  eq(eng.kvBufBytes, need, "engine's per-layer KV buffer");
  const full = eng.layers.filter((R) => R.isFull);
  eq(full.length > 0, true, "the synthetic model has attention layers");
  for (const R of full) { eq(R.kCache.size, need, "K cache size"); eq(R.vCache.size, need, "V cache size"); }
});

Deno.test("KV buffer one position over the binding limit: a readable error at create", async () => {
  const maxSeq = 2048, need = kvLayerBufBytes(G.meta, maxSeq);
  let err = null;
  try { await make({ maxSeq, bind: need - 1 }); } catch (e) { err = e; }
  eq(!!err, true, "create throws");
  eq(/context 2048 needs a .* MiB KV buffer per attention layer/.test(err.message), true, "message: " + err.message);
});

Deno.test("int8 KV halves the buffer: twice the context under the same limit", async () => {
  const maxSeq = 4096, bind = kvLayerBufBytes(G.meta, 2048);
  const eng = await make({ maxSeq, bind, kvQ8: true });
  eq(eng.kvQ8, true, "kvQ8 on");
  eq(eng.kvBufBytes, kvLayerBufBytes(G.meta, maxSeq, "q8"), "int8 buffer");
  eq(eng.kvBufBytes, bind, "fits exactly");
});

Deno.test("a layer range without attention layers ignores the limit", async () => {
  // full_attention_interval 4: layers 0..2 are DeltaNet only
  const eng = await make({ maxSeq: 65536, bind: kvLayerBufBytes(G.meta, 2048), lo: 0, hi: 3 });
  eq(eng.layers.some((R) => R.isFull), false, "no attention layer in 0..2");
});
