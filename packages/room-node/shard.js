// Load this device's layers: room.js aiLoadShard without the DOM. Same GPU bring-up (a throwaway
// device for the self-test and kernel micro tests, then autotuneCoop), the same weight loaders
// (ggufWeights / qwen35Weights, each matrix uploaded as it is converted so RAM stays flat) and the
// same engine options (engine/preset.js roomQwen35Options + applyRoomFlags), so a node's layers
// compute what a browser's would.
import { autotuneCoop, makeTokenizer, DenseEngine, gpuSelfTest, kernelMicroTests } from "../../engine/engine.js";
import { Qwen35Engine } from "../../engine/qwen35.js";
import { ggufWeights, ggufShardBytes, qwen35Weights, qwen35ShardBytes, tokenizerFromGGUF, gpuUploadEntry, GGML_EMBED, GGML_FINAL_NORM, GGML_OUTPUT,
  ggmlLayerNames, qwen35NamesFor } from "../../engine/gguf.js";
import { roomQwen35Options, applyRoomFlags } from "../../engine/preset.js";
import { maxSeqFor, kvModeFor } from "../../room/models.js";
import { convertPool } from "./convert.js";

// The tensors a shard's loader reads (engine/gguf.js ggufWeights / qwen35Weights), in about its order,
// as { byteOffset, byteLength }: what source.js fetches ahead of it when it streams
export function shardTensors(G, kind, { lo, hi, hasEmbed, hasHead, mtp = false }) {
  const names = [];
  const strings = (o) => Object.values(o).filter((v) => typeof v === "string");
  for (let i = lo; i < hi; i++) names.push(...strings(kind === "qwen35" ? qwen35NamesFor(G, i) : ggmlLayerNames(i)));
  if (hasEmbed || hasHead) names.push(GGML_EMBED);
  if (hasHead) names.push(GGML_FINAL_NORM, GGML_OUTPUT);
  if (kind === "qwen35" && mtp && hasHead) {
    const N = G.meta["qwen35.block_count"] - 1, p = `blk.${N}.nextn.`;
    if (G.tensors[p + "eh_proj.weight"]) names.push(...strings(qwen35NamesFor(G, N, true)), ...["eh_proj", "enorm", "hnorm", "shared_head_norm"].map((n) => p + n + ".weight"));
  }
  const seen = new Set(), out = [];
  for (const n of names) { const t = G.tensors[n]; if (t && !seen.has(n)) { seen.add(n); out.push({ byteOffset: t.byteOffset, byteLength: t.byteLength }); } }
  return out;
}

// A model's transformer layers: config.json's num_hidden_layers for a dense GGUF (qwen3-*), the GGUF
// index for qwen35 (block_count less the MTP layers). Keyed on the model's kind, not its name: reading
// qwen35.* keys from a dense model's index gives NaN, and a [0, NaN] range loads no layers at all.
export async function modelLayers(src) {
  if (src.M.kind === "qwen35") {
    const m = (await src.header(false)).meta;
    return m["qwen35.block_count"] - (m["qwen35.nextn_predict_layers"] || 0);
  }
  return (await src.cfg()).num_hidden_layers;
}

// [lo, hi) must be whole layer numbers, 0 <= lo <= hi. A NaN bound used to load an engine with no
// layers that still "worked": embed -> final norm -> tied head, the same token every step.
export function checkRange(range) {
  const [lo, hi] = range || [];
  if (!Number.isInteger(lo) || !Number.isInteger(hi) || lo < 0 || hi < lo) throw new Error(`bad layer range [${lo}, ${hi}]`);
}

// -> { device, engine, tok, cfg, G, tune, gpuErrors }
export async function loadShard({ modelKey, range, hasEmbed, hasHead, ctx = maxSeqFor(modelKey), kv = kvModeFor(modelKey, null),
  src, flags = "", onProgress = () => {}, log = () => {}, selfTest = true, onGpuError = () => {} }) {
  checkRange(range);
  const M = src.M;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter on this machine");
  const device = await adapter.requestDevice({ requiredLimits: {
    maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
  const out = { device, gpuErrors: 0 };
  let pool = null;
  device.addEventListener?.("uncapturederror", (ev) => { if (out.gpuErrors++ < 3) onGpuError(ev.error?.message || "GPU error"); });
  try {
    // the index first, so a streamed shard's weights download while the GPU tests and tunes below run
    const kind = M.kind === "qwen35" ? "qwen35" : "gguf";
    const needTok = kind === "qwen35" && (hasEmbed || hasHead);
    const opts = { lo: range[0], hi: range[1], hasEmbed, hasHead, ...(kind === "qwen35" ? { mtp: hasHead } : {}) };
    const G0 = await src.header(needTok);
    // requantizing the tensors the GPU has no kernel for (Q4_1 / K-quants -> Q8) on worker threads:
    // seconds per tensor on the event loop otherwise (convert.js)
    pool = convertPool({ log });
    G0.convert = pool.convert;
    src.plan?.(shardTensors(G0, kind, opts));
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
      const G = out.G = G0;
      out.cfg = { num_hidden_layers: G.meta["qwen35.block_count"] - (G.meta["qwen35.nextn_predict_layers"] || 0) };
      if (needTok) { out.tok = makeTokenizer(tokenizerFromGGUF(G.meta)); out.tok.chatTemplate = G.meta["tokenizer.chat_template"] || ""; }
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
      const G = out.G = G0;   // vocab comes from tokenizer.json
      const total = ggufShardBytes(G, opts);
      const weights = await ggufWeights(G, src.bytesOf, opts, (done) => onProgress(done, total), upload);
      out.engine = await DenseEngine.create({ coopWG: out.tune.wg, coopRows: out.tune.rows, device, cfg: out.cfg, weights, layerRange: range, hasEmbed, hasHead, maxSeq: ctx });
    }
    applyRoomFlags(out.engine, flags);
    return out;
  } catch (e) { try { device.destroy(); } catch {} throw e; }
  finally { if (out.G) delete out.G.convert; await pool?.close(); }
}
