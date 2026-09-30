// Load this device's layers: room.js aiLoadShard without the DOM. Same GPU bring-up (a throwaway
// device for the self-test and kernel micro tests, then autotuneCoop), the same weight loaders
// (ggufWeights / qwen35Weights, each matrix uploaded as it is converted so RAM stays flat) and the
// same engine options (engine/preset.js roomQwen35Options + applyRoomFlags), so a node's layers
// compute what a browser's would.
import { autotuneCoop, makeTokenizer, DenseEngine, gpuSelfTest, kernelMicroTests } from "../../engine/engine.js";
import { Qwen35Engine } from "../../engine/qwen35.js";
import { ggufWeights, ggufShardBytes, qwen35Weights, qwen35ShardBytes, tokenizerFromGGUF, gpuUploadEntry, GGML_EMBED } from "../../engine/gguf.js";
import { roomQwen35Options, applyRoomFlags } from "../../engine/preset.js";
import { maxSeqFor, kvModeFor } from "../../room/models.js";

// -> { device, engine, tok, cfg, G, tune, gpuErrors }
export async function loadShard({ modelKey, range, hasEmbed, hasHead, ctx = maxSeqFor(modelKey), kv = kvModeFor(modelKey, null),
  src, flags = "", onProgress = () => {}, log = () => {}, selfTest = true, onGpuError = () => {} }) {
  const M = src.M;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter on this machine");
  const device = await adapter.requestDevice({ requiredLimits: {
    maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
  const out = { device, gpuErrors: 0 };
  device.addEventListener?.("uncapturederror", (ev) => { if (out.gpuErrors++ < 3) onGpuError(ev.error?.message || "GPU error"); });
  try {
    if (selfTest) {
      const tdev = await (await navigator.gpu.requestAdapter()).requestDevice();   // an adapter gives out one device only
      const st = await gpuSelfTest(tdev);
      if (!st.ok) throw new Error("GPU self-test FAILED on this device: " + st.detail);
      const mt = await kernelMicroTests(tdev);
      if (!mt.ok) throw new Error("GPU kernel FAILED on this device: " + mt.firstFail);
      try { tdev.destroy(); } catch {}
    }
    out.tune = await autotuneCoop(device).catch(() => ({ wg: 256, rows: 4 }));
    log(`autotune WG=${out.tune.wg} ROWS=${out.tune.rows}`);
    const upload = (e, name) => gpuUploadEntry(device, e, name === GGML_EMBED);   // straight to the GPU
    if (M.kind === "qwen35") {
      const needTok = hasEmbed || hasHead;
      const G = out.G = await src.header(needTok);
      out.cfg = { num_hidden_layers: G.meta["qwen35.block_count"] - (G.meta["qwen35.nextn_predict_layers"] || 0) };
      if (needTok) { out.tok = makeTokenizer(tokenizerFromGGUF(G.meta)); out.tok.chatTemplate = G.meta["tokenizer.chat_template"] || ""; }
      const opts = { lo: range[0], hi: range[1], hasEmbed, hasHead, mtp: hasHead };
      const total = qwen35ShardBytes(G, opts);
      const weights = await qwen35Weights(G, src.bytesOf, opts, (done) => onProgress(done, total), upload);
      out.engine = await Qwen35Engine.create({
        device, meta: G.meta, weights, vocab: G.tensors[GGML_EMBED]?.shape?.[0],
        layerRange: range, hasEmbed, hasHead, maxSeq: ctx, coopWG: out.tune.wg, coopRows: out.tune.rows,
        ...roomQwen35Options(flags), kvQ8: kv === "q8",
      });
    } else {
      out.cfg = await src.cfg();
      if (hasEmbed || hasHead) out.tok = makeTokenizer(await src.tokJson());
      const G = out.G = await src.header(false);   // vocab comes from tokenizer.json
      const opts = { lo: range[0], hi: range[1], hasEmbed, hasHead };
      const total = ggufShardBytes(G, opts);
      const weights = await ggufWeights(G, src.bytesOf, opts, (done) => onProgress(done, total), upload);
      out.engine = await DenseEngine.create({ coopWG: out.tune.wg, coopRows: out.tune.rows, device, cfg: out.cfg, weights, layerRange: range, hasEmbed, hasHead, maxSeq: ctx });
    }
    applyRoomFlags(out.engine, flags);
    return out;
  } catch (e) { try { device.destroy(); } catch {} throw e; }
}
